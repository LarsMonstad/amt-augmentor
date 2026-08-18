"""Tests for atomic, synchronized integral pitch-grid materialization."""

import json
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from amt_augmentor.pitch_shift_grid import (
    CONSERVATIVE_PITCH_SHIFT_GRID_V1,
    DENSE_PITCH_SHIFT_GRID_V1,
    materialize_pitch_shift_grid_v1,
    validate_pitch_shift_grid_v1,
    verify_pitch_shift_grid_v1,
)


def _write_pair(root: Path, *, pitches=(60, 64)):
    root.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    duration = 1.0
    time = np.arange(round(sample_rate * duration)) / sample_rate
    envelope = np.minimum(1.0, time / 0.01) * np.minimum(1.0, (duration - time) / 0.02)
    audio = envelope * (
        0.18 * np.sin(2.0 * np.pi * 261.625565 * time)
        + 0.08 * np.sin(2.0 * np.pi * 523.251131 * time)
        + 0.03 * np.sin(2.0 * np.pi * 784.876696 * time)
    )
    audio_path = root / "source.wav"
    midi_path = root / "source.mid"
    sf.write(audio_path, audio, sample_rate, subtype="PCM_16")
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=9600)
    instrument = pretty_midi.Instrument(program=40, name="fiddle")
    for index, pitch in enumerate(pitches):
        instrument.notes.append(
            pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=0.1 + 0.35 * index,
                end=0.3 + 0.35 * index,
            )
        )
    midi.instruments.append(instrument)
    midi.write(str(midi_path))
    return audio_path, midi_path


def _notes(path: Path):
    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        (float(note.start), float(note.end), int(note.pitch), int(note.velocity))
        for instrument in midi.instruments
        for note in instrument.notes
    ]


def test_research_pitch_grids_are_exact_and_range_checked():
    assert validate_pitch_shift_grid_v1(CONSERVATIVE_PITCH_SHIFT_GRID_V1) == (
        -2,
        -1,
        1,
        2,
    )
    assert validate_pitch_shift_grid_v1(DENSE_PITCH_SHIFT_GRID_V1) == (
        -6,
        -5,
        -4,
        -3,
        -2,
        -1,
        1,
        2,
        3,
        4,
        5,
        6,
    )
    with pytest.raises(ValueError, match="below the model range"):
        validate_pitch_shift_grid_v1(
            DENSE_PITCH_SHIFT_GRID_V1,
            source_pitch_minimum=25,
            source_pitch_maximum=80,
        )
    with pytest.raises(ValueError, match="duplicates"):
        validate_pitch_shift_grid_v1((-1, -1, 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_pitch_shift_grid_v1((1, -1))
    with pytest.raises(ValueError, match="zero"):
        validate_pitch_shift_grid_v1((-1, 0, 1))


def test_dense_grid_materialization_is_complete_synchronized_and_verified(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path / "source")
    output_root = tmp_path / "dense-grid"
    report = materialize_pitch_shift_grid_v1(
        source_audio,
        source_midi,
        output_root,
        seed=424242,
        semitones=DENSE_PITCH_SHIFT_GRID_V1,
    )

    assert report["status"] == "pass"
    assert report["output_count"] == 12
    assert tuple(report["semitones"]) == DENSE_PITCH_SHIFT_GRID_V1
    manifest = json.loads(
        (output_root / "pitch-shift-grid-manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["outputs"]) == 12
    source_notes = _notes(source_midi)
    for record in manifest["outputs"]:
        shift = record["semitones"]
        output_notes = _notes(output_root / record["midi_relpath"])
        assert len(output_notes) == len(source_notes)
        for source, output in zip(source_notes, output_notes):
            assert output[0] == pytest.approx(source[0], abs=0.0001)
            assert output[1] == pytest.approx(source[1], abs=0.0001)
            assert output[2] == source[2] + shift
            assert output[3] == source[3]
        assert record["acoustic_qc"]["best_semitones"] == shift
        assert record["acoustic_qc"]["runner_up_margin"] > 0.0
    assert verify_pitch_shift_grid_v1(output_root) == report


def test_grid_preflight_failure_publishes_nothing(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path / "source", pitches=(25,))
    output_root = tmp_path / "invalid-grid"
    with pytest.raises(ValueError, match="below the model range"):
        materialize_pitch_shift_grid_v1(
            source_audio,
            source_midi,
            output_root,
            seed=1,
            semitones=DENSE_PITCH_SHIFT_GRID_V1,
        )
    assert not output_root.exists()
    assert not list(tmp_path.glob(".invalid-grid.staging-*"))


def test_grid_verifier_rejects_payload_tampering(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path / "source")
    output_root = tmp_path / "conservative-grid"
    materialize_pitch_shift_grid_v1(
        source_audio,
        source_midi,
        output_root,
        seed=7,
        semitones=CONSERVATIVE_PITCH_SHIFT_GRID_V1,
    )
    manifest = json.loads(
        (output_root / "pitch-shift-grid-manifest.json").read_text(encoding="utf-8")
    )
    first_audio = output_root / manifest["outputs"][0]["audio_relpath"]
    first_audio.write_bytes(first_audio.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="audio digest mismatch"):
        verify_pitch_shift_grid_v1(output_root)
