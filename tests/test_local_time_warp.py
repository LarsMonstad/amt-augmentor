"""Acceptance tests for the continuous paired local-time-warp backend."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from amt_augmentor.local_time_warp import (
    LocalTimeWarpParameters,
    local_time_warp_v1,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pair(root: Path, *, seconds: float = 10.0, sample_rate: int = 8000):
    root.mkdir(parents=True, exist_ok=True)
    sample_count = round(seconds * sample_rate)
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    audio = 0.24 * np.sin(2.0 * np.pi * 220.0 * time)
    # Distinct, non-periodic transients make chunk repetition and dropout
    # detectable without requiring sample-identical phase-vocoder output.
    for center, amplitude in ((1.25, 0.45), (3.75, 0.35), (6.5, 0.40), (8.8, 0.30)):
        start = int(center * sample_rate)
        width = int(0.025 * sample_rate)
        audio[start : start + width] += amplitude * np.hanning(width)
    audio_path = root / "source.wav"
    sf.write(audio_path, audio, sample_rate, subtype="PCM_16")

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=9600)
    instrument = pretty_midi.Instrument(program=0, name="piano")
    instrument.notes.extend(
        [
            pretty_midi.Note(velocity=90, pitch=57, start=0.75, end=2.0),
            pretty_midi.Note(velocity=90, pitch=60, start=3.2, end=5.1),
            pretty_midi.Note(velocity=90, pitch=64, start=6.1, end=9.5),
        ]
    )
    midi.instruments.append(instrument)
    midi_path = root / "source.mid"
    midi.write(str(midi_path))
    return audio_path, midi_path, sample_count, sample_rate


def _notes(path: Path):
    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        (note.start, note.end, note.pitch, note.velocity)
        for instrument in midi.instruments
        for note in instrument.notes
    ]


def _write_impulse_pair(root: Path, *, sample_rate: int = 8000):
    root.mkdir(parents=True, exist_ok=True)
    sample_count = 10 * sample_rate
    impulse_seconds = (1.25, 3.75, 6.5, 8.8)
    audio = np.zeros(sample_count, dtype=np.float64)
    for time in impulse_seconds:
        audio[round(time * sample_rate)] = 0.8
    audio_path = root / "impulses.wav"
    sf.write(audio_path, audio, sample_rate, subtype="PCM_16")
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=9600)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.1, end=9.9))
    midi.instruments.append(instrument)
    midi_path = root / "impulses.mid"
    midi.write(str(midi_path))
    return audio_path, midi_path, impulse_seconds, sample_count


def _render(tmp_path: Path, suffix: str, *, seed: int = 143):
    source_audio, source_midi, sample_count, sample_rate = _write_pair(tmp_path)
    output_audio = tmp_path / f"warped-{suffix}.wav"
    output_midi = tmp_path / f"warped-{suffix}.mid"
    provenance = local_time_warp_v1(
        source_audio,
        source_midi,
        output_audio,
        output_midi,
        seed=seed,
        parameters=LocalTimeWarpParameters(),
    )
    return (
        source_audio,
        source_midi,
        output_audio,
        output_midi,
        provenance,
        sample_count,
        sample_rate,
    )


def test_local_time_warp_is_seed_deterministic_and_has_complete_backend_provenance(
    tmp_path,
):
    first = _render(tmp_path / "first", "first")
    second = _render(tmp_path / "second", "second")
    _, _, first_audio, first_midi, first_provenance, _, _ = first
    _, _, second_audio, second_midi, second_provenance, _, _ = second

    assert first_audio.read_bytes() == second_audio.read_bytes()
    assert first_midi.read_bytes() == second_midi.read_bytes()
    assert first_provenance["sample_map"] == second_provenance["sample_map"]
    assert first_provenance["backend"]["name"] == "librosa_nonuniform_phase_vocoder_v1"
    assert first_provenance["backend"]["phase_recurrence"].startswith("adapted_")
    assert first_provenance["qc"]["no_chunking_or_looping"] is True
    assert first_provenance["invariants"]["chunking_looping_or_splicing_used"] is False
    marker = first_audio.with_suffix(".wav.provenance.json")
    assert json.loads(marker.read_text(encoding="utf-8")) == first_provenance
    assert first_provenance["output"]["audio_sha256"] == _sha256(first_audio)


def test_local_time_warp_map_is_monotonic_and_preserves_exact_sample_duration(tmp_path):
    _, _, source_audio, _, provenance, sample_count, sample_rate = _render(
        tmp_path, "map"
    )
    output_audio = tmp_path / "warped-map.wav"
    source_info = sf.info(source_audio)
    output_info = sf.info(output_audio)
    sample_map = provenance["sample_map"]
    source_knots = np.asarray(sample_map["source_sample_knots"], dtype=np.float64)
    target_knots = np.asarray(sample_map["target_sample_knots"], dtype=np.float64)
    rates = np.asarray(sample_map["segment_target_per_source_rates"], dtype=np.float64)

    assert source_info.frames == output_info.frames == sample_count
    assert output_info.samplerate == sample_rate
    assert (source_knots[0], target_knots[0]) == (0.0, 0.0)
    assert (source_knots[-1], target_knots[-1]) == (sample_count, sample_count)
    assert np.all(np.diff(source_knots) > 0.0)
    assert np.all(np.diff(target_knots) > 0.0)
    assert np.all((0.94 <= rates) & (rates <= 1.06))
    assert rates[0] == pytest.approx(1.0, abs=1e-12)
    assert rates[-1] == pytest.approx(1.0, abs=1e-12)


def test_local_time_warp_maps_every_midi_boundary_through_published_sample_map(
    tmp_path,
):
    source_audio, source_midi, _, output_midi, provenance, _, sample_rate = _render(
        tmp_path,
        "midi",
    )
    del source_audio
    source_knots = np.asarray(
        provenance["sample_map"]["source_sample_knots"], dtype=np.float64
    )
    target_knots = np.asarray(
        provenance["sample_map"]["target_sample_knots"], dtype=np.float64
    )
    for source_note, output_note in zip(_notes(source_midi), _notes(output_midi)):
        for source_time, output_time in zip(source_note[:2], output_note[:2]):
            expected = (
                np.interp(
                    source_time * sample_rate,
                    source_knots,
                    target_knots,
                )
                / sample_rate
            )
            assert output_time == pytest.approx(expected, abs=0.0001)
        assert output_note[2:] == source_note[2:]


def test_local_time_warp_has_no_repeated_chunks_or_silent_dropouts(tmp_path):
    _, _, output_audio, _, provenance, _, sample_rate = _render(tmp_path, "continuity")
    rendered, rendered_rate = sf.read(output_audio, dtype="float64")
    assert rendered_rate == sample_rate
    # A full-recording 220 Hz tone should retain energy in every half-second;
    # a chunk insertion/repetition implementation commonly creates gaps here.
    block = sample_rate // 2
    block_rms = np.array(
        [
            np.sqrt(np.mean(np.square(rendered[start : start + block])))
            for start in range(0, len(rendered) - block + 1, block)
        ]
    )
    assert np.min(block_rms) > 0.02
    assert provenance["invariants"]["single_continuous_phase_vocoder_render"] is True
    assert provenance["invariants"]["chunking_looping_or_splicing_used"] is False


def test_local_time_warp_preserves_tone_pitch_and_keeps_impulses_approximately_aligned(
    tmp_path,
):
    _, _, output_audio, _, _provenance, _, sample_rate = _render(tmp_path, "pitch")
    rendered, _ = sf.read(output_audio, dtype="float64")
    # Estimate pitch away from the deliberately added transients.
    segment = rendered[int(2.35 * sample_rate) : int(3.35 * sample_rate)]
    frequencies = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    magnitude = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
    dominant_hz = frequencies[np.argmax(magnitude[1:]) + 1]
    assert dominant_hz == pytest.approx(220.0, abs=3.0)

    impulse_audio, impulse_midi, impulse_seconds, sample_count = _write_impulse_pair(
        tmp_path / "impulses",
        sample_rate=sample_rate,
    )
    rendered_impulses = tmp_path / "warped-impulses.wav"
    rendered_impulse_midi = tmp_path / "warped-impulses.mid"
    impulse_provenance = local_time_warp_v1(
        impulse_audio,
        impulse_midi,
        rendered_impulses,
        rendered_impulse_midi,
        seed=143,
    )
    impulses, _ = sf.read(rendered_impulses, dtype="float64")
    assert impulses.size == sample_count
    source_knots = np.asarray(
        impulse_provenance["sample_map"]["source_sample_knots"], dtype=np.float64
    )
    target_knots = np.asarray(
        impulse_provenance["sample_map"]["target_sample_knots"], dtype=np.float64
    )
    # The transients may smear by the phase-vocoder window, but each remains
    # energetically centered within one n_fft window of its mapped location.
    for source_seconds in impulse_seconds:
        expected = round(
            np.interp(source_seconds * sample_rate, source_knots, target_knots)
        )
        radius = 2048
        start = max(0, expected - radius)
        stop = min(impulses.size, expected + radius)
        assert np.max(np.abs(impulses[start:stop])) > 1e-4
