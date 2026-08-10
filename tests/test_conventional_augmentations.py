"""Fail-closed synchronization tests for conventional research transforms."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

import amt_augmentor._paired_io as paired_io
from amt_augmentor.conventional_augmentations import (
    GainChorusParameters,
    NoiseSNRParameters,
    PitchShiftParameters,
    ReverbFiltersParameters,
    TimeStretchParameters,
    gain_chorus_v1,
    noise_snr_v1,
    pitch_shift_v1,
    reverb_filters_v1,
    time_stretch_v1,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pair(
    root: Path,
    *,
    seconds: float = 1.5,
    sample_rate: int = 8000,
    pitches=(60, 64),
):
    root.mkdir(parents=True, exist_ok=True)
    audio_path = root / "source.wav"
    midi_path = root / "source.mid"
    time = np.arange(round(seconds * sample_rate)) / sample_rate
    audio = 0.20 * np.sin(2 * np.pi * 220.0 * time) + 0.04 * np.sin(
        2 * np.pi * 440.0 * time
    )
    sf.write(audio_path, audio, sample_rate, subtype="PCM_16")
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=9600)
    instrument = pretty_midi.Instrument(program=40, name="fiddle")
    for index, pitch in enumerate(pitches):
        start = 0.15 + 0.50 * index
        instrument.notes.append(
            pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=start,
                end=start + 0.25,
            )
        )
    midi.instruments.append(instrument)
    midi.write(str(midi_path))
    return audio_path, midi_path


def _notes(path: Path):
    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        (note.start, note.end, note.pitch, note.velocity)
        for instrument in midi.instruments
        for note in instrument.notes
    ]


@pytest.mark.parametrize(
    "function,parameters,transform",
    [
        (
            gain_chorus_v1,
            GainChorusParameters(
                gain_db=-1.5,
                chorus_depth=0.2,
                chorus_rate_hz=0.8,
            ),
            "gain_chorus_v1",
        ),
        (
            noise_snr_v1,
            NoiseSNRParameters(target_snr_db=24.0),
            "noise_snr_v1",
        ),
        (
            reverb_filters_v1,
            ReverbFiltersParameters(
                room_size=0.25,
                wet_level=0.15,
                dry_level=0.85,
                highpass_hz=40.0,
                lowpass_hz=3500.0,
            ),
            "reverb_filters_v1",
        ),
    ],
)
def test_audio_only_transforms_preserve_midi_bytes_and_shape(
    tmp_path, function, parameters, transform
):
    source_audio, source_midi = _write_pair(tmp_path)
    output_audio = tmp_path / f"{transform}.wav"
    output_midi = tmp_path / f"{transform}.mid"
    provenance = function(
        source_audio,
        source_midi,
        output_audio,
        output_midi,
        seed=123,
        parameters=parameters,
    )

    assert output_midi.read_bytes() == source_midi.read_bytes()
    source_data, source_rate = sf.read(source_audio, always_2d=True)
    output_data, output_rate = sf.read(output_audio, always_2d=True)
    assert output_rate == source_rate
    assert output_data.shape == source_data.shape
    assert np.isfinite(output_data).all()
    assert np.max(np.abs(output_data)) <= 1.0
    assert provenance["transform"] == transform
    assert provenance["seed"] == 123
    assert provenance["invariants"]["midi_preserved_byte_for_byte"] is True
    assert provenance["output"]["midi_sha256"] == _sha256(source_midi)


def test_noise_has_seeded_measured_snr_without_opaque_normalization(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path)
    outputs = []
    provenances = []
    for suffix in ("a", "b"):
        output_audio = tmp_path / f"noise-{suffix}.wav"
        output_midi = tmp_path / f"noise-{suffix}.mid"
        provenance = noise_snr_v1(
            source_audio,
            source_midi,
            output_audio,
            output_midi,
            seed=987654,
            parameters=NoiseSNRParameters(target_snr_db=27.0),
        )
        outputs.append((output_audio.read_bytes(), output_midi.read_bytes()))
        provenances.append(provenance)
    assert outputs[0] == outputs[1]
    assert provenances[0]["qc"]["measured_float_snr_db"] == pytest.approx(
        27.0, abs=1e-10
    )
    assert provenances[0]["qc"]["opaque_peak_normalization_used"] is False
    assert provenances[0]["plan_config_sha256"] == provenances[1]["plan_config_sha256"]


def test_pitch_shift_is_integral_synchronized_and_never_drops_labels(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path)
    output_audio = tmp_path / "pitch.wav"
    output_midi = tmp_path / "pitch.mid"
    provenance = pitch_shift_v1(
        source_audio,
        source_midi,
        output_audio,
        output_midi,
        seed=11,
        parameters=PitchShiftParameters(semitones=2),
    )

    source_notes = _notes(source_midi)
    output_notes = _notes(output_midi)
    assert len(output_notes) == len(source_notes)
    for source, output in zip(source_notes, output_notes):
        assert output[0] == pytest.approx(source[0], abs=0.0001)
        assert output[1] == pytest.approx(source[1], abs=0.0001)
        assert output[2] == source[2] + 2
        assert output[3] == source[3]
    assert provenance["qc"]["out_of_range_labels_dropped"] == 0

    edge_audio, edge_midi = _write_pair(tmp_path / "edge", pitches=(107,))
    with pytest.raises(ValueError, match="refuses to drop labels"):
        pitch_shift_v1(
            edge_audio,
            edge_midi,
            tmp_path / "bad.wav",
            tmp_path / "bad.mid",
            seed=11,
            parameters=PitchShiftParameters(semitones=2),
        )
    assert not (tmp_path / "bad.wav").exists()


def test_time_stretch_uses_realized_sample_ratio_for_every_note_boundary(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path, seconds=1.333375)
    output_audio = tmp_path / "stretch.wav"
    output_midi = tmp_path / "stretch.mid"
    provenance = time_stretch_v1(
        source_audio,
        source_midi,
        output_audio,
        output_midi,
        seed=7,
        parameters=TimeStretchParameters(rate=1.07),
    )

    source_info = sf.info(source_audio)
    output_info = sf.info(output_audio)
    expected_samples = round(source_info.frames / 1.07)
    assert output_info.frames == expected_samples
    realized = output_info.frames / source_info.frames
    assert provenance["qc"]["realized_time_scale"] == realized
    assert provenance["qc"]["annotation_text_rounding_used"] is False
    for source, output in zip(_notes(source_midi), _notes(output_midi)):
        assert output[0] == pytest.approx(source[0] * realized, abs=0.0001)
        assert output[1] == pytest.approx(source[1] * realized, abs=0.0001)
        assert output[2:] == source[2:]


@pytest.mark.parametrize(
    "function,parameters",
    [
        (noise_snr_v1, NoiseSNRParameters(target_snr_db=float("nan"))),
        (pitch_shift_v1, PitchShiftParameters(semitones=0)),
        (pitch_shift_v1, PitchShiftParameters(semitones=1.5)),
        (time_stretch_v1, TimeStretchParameters(rate=0.0)),
        (
            reverb_filters_v1,
            ReverbFiltersParameters(
                room_size=0.2,
                wet_level=0.2,
                dry_level=0.8,
                highpass_hz=4000.0,
                lowpass_hz=2000.0,
            ),
        ),
    ],
)
def test_invalid_parameters_fail_before_publishing(tmp_path, function, parameters):
    source_audio, source_midi = _write_pair(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        function(
            source_audio,
            source_midi,
            tmp_path / "output.wav",
            tmp_path / "output.mid",
            seed=1,
            parameters=parameters,
        )
    assert not (tmp_path / "output.wav").exists()
    assert not (tmp_path / "output.mid").exists()


def test_malformed_campaign_midi_is_an_error_not_a_silent_drop(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path)
    midi = pretty_midi.PrettyMIDI(str(source_midi))
    midi.instruments[0].control_changes.append(
        pretty_midi.ControlChange(number=64, value=127, time=0.2)
    )
    midi.write(str(source_midi))
    with pytest.raises(ValueError, match="control changes"):
        noise_snr_v1(
            source_audio,
            source_midi,
            tmp_path / "output.wav",
            tmp_path / "output.mid",
            seed=1,
            parameters=NoiseSNRParameters(target_snr_db=20.0),
        )


def test_midi_note_beyond_audio_duration_fails_before_publishing(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path, seconds=1.0, pitches=(60,))
    midi = pretty_midi.PrettyMIDI(str(source_midi))
    midi.instruments[0].notes[0].end = 1.00025
    midi.write(str(source_midi))

    output_audio = tmp_path / "output.wav"
    output_midi = tmp_path / "output.mid"
    with pytest.raises(ValueError, match="beyond the audio duration"):
        noise_snr_v1(
            source_audio,
            source_midi,
            output_audio,
            output_midi,
            seed=1,
            parameters=NoiseSNRParameters(target_snr_db=20.0),
        )

    assert not output_audio.exists()
    assert not output_midi.exists()


def test_provenance_completion_marker_contains_output_hashes(tmp_path):
    source_audio, source_midi = _write_pair(tmp_path)
    output_audio = tmp_path / "output.wav"
    output_midi = tmp_path / "output.mid"
    provenance = gain_chorus_v1(
        source_audio,
        source_midi,
        output_audio,
        output_midi,
        seed=1,
        parameters=GainChorusParameters(
            gain_db=0.0,
            chorus_depth=0.1,
            chorus_rate_hz=0.5,
        ),
    )
    marker = output_audio.with_suffix(".wav.provenance.json")
    assert json.loads(marker.read_text(encoding="utf-8")) == provenance
    assert provenance["output"]["audio_sha256"] == _sha256(output_audio)
    assert provenance["output"]["midi_sha256"] == _sha256(output_midi)


@pytest.mark.parametrize("existing_kind", ["audio", "midi", "provenance"])
def test_existing_bundle_outputs_are_never_overwritten(tmp_path, existing_kind):
    source_audio, source_midi = _write_pair(tmp_path)
    output_audio = tmp_path / "output.wav"
    output_midi = tmp_path / "output.mid"
    provenance = output_audio.with_suffix(".wav.provenance.json")
    outputs = {
        "audio": output_audio,
        "midi": output_midi,
        "provenance": provenance,
    }
    sentinel = b"do-not-overwrite"
    outputs[existing_kind].write_bytes(sentinel)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        noise_snr_v1(
            source_audio,
            source_midi,
            output_audio,
            output_midi,
            seed=1,
            parameters=NoiseSNRParameters(target_snr_db=20.0),
        )

    assert outputs[existing_kind].read_bytes() == sentinel
    for kind, path in outputs.items():
        if kind != existing_kind:
            assert not path.exists()


def test_mid_publish_failure_rolls_back_payloads(tmp_path, monkeypatch):
    source_audio, source_midi = _write_pair(tmp_path)
    output_audio = tmp_path / "output.wav"
    output_midi = tmp_path / "output.mid"
    provenance = output_audio.with_suffix(".wav.provenance.json")
    real_publish = paired_io._publish_stage
    publish_count = 0

    def fail_second_publish(stage_path, target_path):
        nonlocal publish_count
        publish_count += 1
        if publish_count == 2:
            raise OSError("simulated publication failure")
        real_publish(stage_path, target_path)

    monkeypatch.setattr(paired_io, "_publish_stage", fail_second_publish)
    with pytest.raises(OSError, match="simulated publication failure"):
        noise_snr_v1(
            source_audio,
            source_midi,
            output_audio,
            output_midi,
            seed=1,
            parameters=NoiseSNRParameters(target_snr_db=20.0),
        )

    assert not output_audio.exists()
    assert not output_midi.exists()
    assert not provenance.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


@pytest.mark.parametrize(
    "bad_seed,expected_error",
    [
        (True, TypeError),
        (False, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (-1, ValueError),
    ],
)
def test_conventional_api_rejects_invalid_seeds(
    tmp_path,
    bad_seed,
    expected_error,
):
    source_audio, source_midi = _write_pair(tmp_path)
    with pytest.raises(expected_error):
        noise_snr_v1(
            source_audio,
            source_midi,
            tmp_path / "output.wav",
            tmp_path / "output.mid",
            seed=bad_seed,
            parameters=NoiseSNRParameters(target_snr_db=20.0),
        )

    assert not (tmp_path / "output.wav").exists()
    assert not (tmp_path / "output.mid").exists()
