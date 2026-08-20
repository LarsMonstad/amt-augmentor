"""Tests for atomic, synchronized integral pitch-grid materialization."""

import json
import os
import stat
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

import amt_augmentor.pitch_shift_grid as pitch_grid
from amt_augmentor.conventional_augmentations import PitchShiftParameters
from amt_augmentor.pitch_shift_grid import (
    materialize_pitch_shift_grid_v1,
    validate_pitch_shift_grid_v1,
    verify_pitch_shift_grid_v1,
)

CONSERVATIVE_GRID = (-2, -1, 1, 2)
DENSE_GRID = (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)


def _write_pair(root: Path, *, pitches=(60, 64), audio_midi_pitch=None):
    root.mkdir(parents=True, exist_ok=True)
    sample_rate = 8000
    duration = 1.0
    time = np.arange(round(sample_rate * duration)) / sample_rate
    envelope = np.minimum(1.0, time / 0.01) * np.minimum(1.0, (duration - time) / 0.02)
    if audio_midi_pitch is None:
        audio = envelope * (
            0.18 * np.sin(2.0 * np.pi * 261.625565 * time)
            + 0.08 * np.sin(2.0 * np.pi * 523.251131 * time)
            + 0.03 * np.sin(2.0 * np.pi * 784.876696 * time)
        )
    else:
        frequency_hz = 440.0 * (2.0 ** ((audio_midi_pitch - 69) / 12.0))
        audio = envelope * 0.18 * np.sin(2.0 * np.pi * frequency_hz * time)
    audio_path = root / "source.wav"
    midi_path = root / "source.mid"
    sf.write(audio_path, audio, sample_rate, subtype="PCM_16")
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=9600)
    instrument = pretty_midi.Instrument(program=40, name="lead_strings")
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


def test_caller_pitch_grids_are_explicit_and_range_checked():
    assert validate_pitch_shift_grid_v1(CONSERVATIVE_GRID) == (
        -2,
        -1,
        1,
        2,
    )
    assert validate_pitch_shift_grid_v1(DENSE_GRID) == (
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
    assert validate_pitch_shift_grid_v1(
        DENSE_GRID,
        source_pitch_minimum=25,
        source_pitch_maximum=80,
    ) == DENSE_GRID
    with pytest.raises(ValueError, match="below the output MIDI range"):
        validate_pitch_shift_grid_v1(
            DENSE_GRID,
            source_pitch_minimum=25,
            source_pitch_maximum=80,
            minimum_midi_pitch=21,
            maximum_midi_pitch=108,
        )
    with pytest.raises(ValueError, match="duplicates"):
        validate_pitch_shift_grid_v1((-1, -1, 1))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_pitch_shift_grid_v1((1, -1))
    with pytest.raises(ValueError, match="zero"):
        validate_pitch_shift_grid_v1((-1, 0, 1))


def test_grid_qc_extent_is_source_aware_and_fails_closed_at_nyquist():
    low_pitch_plan = pitch_grid._pitch_grid_qc_plan(
        (-2, 3),
        source_pitch_minimum=5,
        source_pitch_maximum=9,
        sample_rate=8000,
    )
    assert low_pitch_plan["fmin_midi"] == 0
    assert low_pitch_plan["relevant_midi_pitch_extent"] == [2, 13]
    assert set(low_pitch_plan["candidates"]) >= {-3, -2, -1, 0, 2, 3, 4}

    with pytest.raises(ValueError, match="cannot acoustically verify"):
        pitch_grid._pitch_grid_qc_plan(
            (-1,),
            source_pitch_minimum=106,
            source_pitch_maximum=106,
            sample_rate=8000,
        )


def test_grid_qc_includes_highest_legal_boundary_bin(tmp_path):
    source_audio, source_midi = _write_pair(
        tmp_path / "source-boundary",
        pitches=(104,),
        audio_midi_pitch=104,
    )
    plan = pitch_grid._pitch_grid_qc_plan(
        (1,),
        source_pitch_minimum=104,
        source_pitch_maximum=104,
        sample_rate=8000,
    )

    assert pitch_grid._count_cqt_bins_below_frequency_limit(100.0, 200.0) == 12
    assert plan["candidates"] == (0, 1)
    assert plan["clipped_candidates"] == [2]
    assert plan["fmin_midi"] == 79
    assert plan["n_bins"] == 27
    assert plan["maximum_cqt_midi_pitch"] == 105
    assert plan["maximum_cqt_center_hz"] == pytest.approx(3520.0)
    assert plan["maximum_cqt_center_hz"] < plan["maximum_cqt_hz"]
    next_center_hz = plan["fmin_hz"] * (2.0 ** (plan["n_bins"] / 12.0))
    assert next_center_hz >= plan["maximum_cqt_hz"]

    source_cqt, source_cqt_info = pitch_grid._absolute_frequency_cqt(
        source_audio,
        n_bins=plan["n_bins"],
        fmin_hz=plan["fmin_hz"],
    )
    assert source_cqt.shape[0] == 27
    assert source_cqt_info["n_bins"] == 27

    output_root = tmp_path / "boundary-grid"
    report = materialize_pitch_shift_grid_v1(
        source_audio,
        source_midi,
        output_root,
        seed=23,
        semitones=(1,),
    )
    manifest = json.loads(
        (output_root / "pitch-shift-grid-manifest.json").read_text(encoding="utf-8")
    )
    acoustic_qc = manifest["outputs"][0]["acoustic_qc"]
    assert acoustic_qc["n_bins"] == 27
    assert acoustic_qc["cqt_maximum_midi_pitch"] == 105
    assert acoustic_qc["cqt_maximum_center_hz"] == pytest.approx(3520.0)
    assert verify_pitch_shift_grid_v1(output_root) == report


def test_dense_grid_materialization_is_complete_synchronized_and_verified(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path / "source")
    output_root = tmp_path / "dense-grid"
    report = materialize_pitch_shift_grid_v1(
        source_audio,
        source_midi,
        output_root,
        seed=424242,
        semitones=DENSE_GRID,
    )

    assert report["status"] == "pass"
    assert report["output_count"] == 12
    assert tuple(report["semitones"]) == DENSE_GRID
    manifest = json.loads(
        (output_root / "pitch-shift-grid-manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["outputs"]) == 12
    assert manifest["output_midi_pitch_bounds"] == {"minimum": 0, "maximum": 127}
    assert "model_pitch_bounds" not in manifest
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
        assert set(record["acoustic_qc"]["candidate_scores"]) == {
            str(value) for value in range(-7, 8)
        }
    assert verify_pitch_shift_grid_v1(output_root) == report


def test_grid_preflight_failure_publishes_nothing(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path / "source", pitches=(3,))
    output_root = tmp_path / "invalid-grid"
    with pytest.raises(ValueError, match="below the output MIDI range"):
        materialize_pitch_shift_grid_v1(
            source_audio,
            source_midi,
            output_root,
            seed=1,
            semitones=DENSE_GRID,
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
        semitones=CONSERVATIVE_GRID,
    )
    manifest = json.loads(
        (output_root / "pitch-shift-grid-manifest.json").read_text(encoding="utf-8")
    )
    first_audio = output_root / manifest["outputs"][0]["audio_relpath"]
    first_audio.write_bytes(first_audio.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="audio digest mismatch"):
        verify_pitch_shift_grid_v1(output_root)


def test_grid_qc_rejects_an_adjacent_but_wrong_realized_shift(
    tmp_path,
    monkeypatch,
):
    source_audio, source_midi = _write_pair(tmp_path / "source")
    output_root = tmp_path / "wrong-shift-grid"
    real_pitch_shift = pitch_grid.pitch_shift_v1

    def render_one_semitone_low(*args, parameters, **kwargs):
        return real_pitch_shift(
            *args,
            parameters=PitchShiftParameters(
                semitones=parameters.semitones - 1,
                minimum_midi_pitch=parameters.minimum_midi_pitch,
                maximum_midi_pitch=parameters.maximum_midi_pitch,
            ),
            **kwargs,
        )

    monkeypatch.setattr(pitch_grid, "pitch_shift_v1", render_one_semitone_low)
    with pytest.raises(ValueError, match="audio pitch shift.*best=1"):
        materialize_pitch_shift_grid_v1(
            source_audio,
            source_midi,
            output_root,
            seed=11,
            semitones=(2,),
        )
    assert not output_root.exists()
    assert not list(tmp_path.glob(".wrong-shift-grid.staging-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX umask/mode semantics")
def test_grid_publication_respects_caller_umask(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path / "source")
    output_root = tmp_path / "grid"
    previous_umask = os.umask(0o077)
    try:
        materialize_pitch_shift_grid_v1(
            source_audio,
            source_midi,
            output_root,
            seed=19,
            semitones=(-1, 1),
        )
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(output_root.stat().st_mode) == 0o700
    for path in output_root.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
