"""Acoustic and fail-closed tests for detuning and archival noise."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

import amt_augmentor._paired_io as paired_io
import amt_augmentor.conventional_augmentations as conventional
from amt_augmentor.conventional_augmentations import (
    ArchivalNoiseParameters,
    FractionalDetuningParameters,
    archival_noise_v1,
    fractional_detuning_v1,
)


def _write_pair(
    root: Path,
    *,
    seconds: float = 2.0,
    sample_rate: int = 8000,
    stereo: bool = False,
    silent: bool = False,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    audio_path = root / "source.wav"
    midi_path = root / "source.mid"
    sample_count = round(seconds * sample_rate)
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    if silent:
        audio = np.zeros(sample_count, dtype=np.float64)
    else:
        audio = 0.18 * np.sin(2.0 * np.pi * 440.0 * time)
        audio += 0.04 * np.sin(2.0 * np.pi * 880.0 * time)
    if stereo:
        audio = np.column_stack((audio, 0.75 * audio))
    sf.write(audio_path, audio, sample_rate, subtype="PCM_16")

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=9600)
    instrument = pretty_midi.Instrument(program=40, name="lead_strings")
    instrument.notes.append(
        pretty_midi.Note(
            velocity=90,
            pitch=69,
            start=0.10,
            end=min(0.80, seconds),
        )
    )
    midi.instruments.append(instrument)
    midi.write(str(midi_path))
    return audio_path, midi_path


def _write_long_pair_bounded(
    root: Path,
    *,
    sample_count: int = 3_249_628,
    sample_rate: int = 44_100,
) -> tuple[Path, Path]:
    """Write a real-scale fixture without first allocating the complete signal."""

    root.mkdir(parents=True, exist_ok=True)
    audio_path = root / "source.wav"
    midi_path = root / "source.mid"
    with sf.SoundFile(
        audio_path,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        subtype="PCM_16",
    ) as output:
        for start in range(0, sample_count, conventional.ARCHIVAL_NOISE_BLOCK_SAMPLES):
            block_samples = min(
                conventional.ARCHIVAL_NOISE_BLOCK_SAMPLES,
                sample_count - start,
            )
            time = (start + np.arange(block_samples, dtype=np.float64)) / sample_rate
            block = 0.18 * np.sin(2.0 * np.pi * 440.0 * time)
            block += 0.04 * np.sin(2.0 * np.pi * 880.0 * time)
            output.write(block)

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=9600)
    instrument = pretty_midi.Instrument(program=40, name="lead_strings")
    instrument.notes.append(
        pretty_midi.Note(velocity=90, pitch=69, start=0.10, end=0.80)
    )
    midi.instruments.append(instrument)
    midi.write(str(midi_path))
    return audio_path, midi_path


def _dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    mono = np.asarray(audio).reshape(-1)
    trim = sample_rate // 4
    mono = mono[trim:-trim]
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
    frequencies = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)
    return float(frequencies[int(np.argmax(spectrum))])


def _archival_parameters(
    target_snr_db: float,
    *,
    hum_power_fraction: float = 0.20,
    mains_frequency_hz: float = 60.0,
    harmonic_count: int = 3,
) -> ArchivalNoiseParameters:
    return ArchivalNoiseParameters(
        target_snr_db=target_snr_db,
        hum_power_fraction=hum_power_fraction,
        mains_frequency_hz=mains_frequency_hz,
        harmonic_count=harmonic_count,
    )


@pytest.mark.parametrize(
    "function,parameters,transform",
    [
        (
            fractional_detuning_v1,
            FractionalDetuningParameters(cents=30.0),
            "fractional_detuning_v1",
        ),
        (
            archival_noise_v1,
            _archival_parameters(28.0),
            "archival_noise_v1",
        ),
    ],
)
def test_new_audio_only_transforms_preserve_stereo_shape_duration_and_midi_bytes(
    tmp_path: Path,
    function,
    parameters,
    transform: str,
) -> None:
    source_audio, source_midi = _write_pair(tmp_path, stereo=True)
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

    source, source_rate = sf.read(source_audio, always_2d=True)
    output, output_rate = sf.read(output_audio, always_2d=True)
    assert output_rate == source_rate
    assert output.shape == source.shape
    assert sf.info(output_audio).frames == sf.info(source_audio).frames
    assert sf.info(output_audio).channels == 2
    assert np.isfinite(output).all()
    assert np.max(np.abs(output)) <= 0.999 + 1.0 / 32768.0
    assert output_midi.read_bytes() == source_midi.read_bytes()
    marker = output_audio.with_suffix(".wav.provenance.json")
    assert json.loads(marker.read_text(encoding="utf-8")) == provenance
    assert provenance["transform"] == transform
    assert provenance["invariants"]["midi_preserved_byte_for_byte"] is True


@pytest.mark.parametrize("cents", [-30.0, 30.0])
def test_fractional_detuning_moves_audio_by_declared_cents_and_not_midi(
    tmp_path: Path,
    cents: float,
) -> None:
    source_audio, source_midi = _write_pair(
        tmp_path / str(cents),
        seconds=3.0,
        sample_rate=16000,
    )
    output_audio = tmp_path / f"detuned-{cents}.wav"
    output_midi = tmp_path / f"detuned-{cents}.mid"
    provenance = fractional_detuning_v1(
        source_audio,
        source_midi,
        output_audio,
        output_midi,
        seed=41,
        parameters=FractionalDetuningParameters(cents=cents),
    )

    output, sample_rate = sf.read(output_audio)
    expected_frequency = 440.0 * (2.0 ** (cents / 1200.0))
    assert _dominant_frequency(output, sample_rate) == pytest.approx(
        expected_frequency,
        abs=0.75,
    )
    assert output_midi.read_bytes() == source_midi.read_bytes()
    assert provenance["qc"]["requested_cents"] == cents
    assert provenance["qc"]["requested_frequency_ratio"] == pytest.approx(
        2.0 ** (cents / 1200.0),
        abs=1e-15,
    )
    assert provenance["qc"]["source_audio_samples"] == provenance["qc"][
        "output_audio_samples"
    ]


def test_fractional_detuning_is_byte_deterministic(tmp_path: Path) -> None:
    source_audio, source_midi = _write_pair(tmp_path / "source")
    audio_bytes = []
    for suffix in ("a", "b"):
        output_audio = tmp_path / f"detune-{suffix}.wav"
        output_midi = tmp_path / f"detune-{suffix}.mid"
        fractional_detuning_v1(
            source_audio,
            source_midi,
            output_audio,
            output_midi,
            seed=808,
            parameters=FractionalDetuningParameters(cents=15.0),
        )
        audio_bytes.append(output_audio.read_bytes())
    assert audio_bytes[0] == audio_bytes[1]


def test_archival_noise_is_seeded_and_has_exact_aggregate_float_snr(
    tmp_path: Path,
) -> None:
    source_audio, source_midi = _write_pair(tmp_path / "source", stereo=True)
    parameters = ArchivalNoiseParameters(
        target_snr_db=28.0,
        hum_power_fraction=0.20,
        mains_frequency_hz=60.0,
        harmonic_count=3,
    )
    results = []
    for suffix, seed in (("a", 5150), ("b", 5150), ("c", 5151)):
        output_audio = tmp_path / f"archival-{suffix}.wav"
        output_midi = tmp_path / f"archival-{suffix}.mid"
        provenance = archival_noise_v1(
            source_audio,
            source_midi,
            output_audio,
            output_midi,
            seed=seed,
            parameters=parameters,
        )
        results.append((output_audio.read_bytes(), provenance))

    assert results[0][0] == results[1][0]
    assert results[0][0] != results[2][0]
    assert (
        results[0][1]["qc"]["interference_float64_sha256"]
        == results[1][1]["qc"]["interference_float64_sha256"]
    )
    assert (
        results[0][1]["qc"]["interference_float64_sha256"]
        != results[2][1]["qc"]["interference_float64_sha256"]
    )
    qc = results[0][1]["qc"]
    assert qc["measured_float_snr_db"] == pytest.approx(28.0, abs=1e-12)
    assert qc["snr_absolute_error_db"] <= 1e-12
    assert qc["measured_hum_power_fraction"] == pytest.approx(0.20, abs=1e-12)
    assert abs(qc["component_cross_correlation"]) <= 1e-12
    assert qc["pink_stream_spawn_key"] == [0]
    assert qc["hum_stream_spawn_key"] == [1]
    assert qc["pink_generation_passes"] == 2
    assert qc["whole_record_fft_used"] is False


def test_archival_noise_real_scale_uses_bounded_blocks_and_preserves_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_audio, source_midi = _write_long_pair_bounded(tmp_path / "source")
    output_audio = tmp_path / "archival-long.wav"
    output_midi = tmp_path / "archival-long.mid"
    observed_block_samples = []
    real_generator = conventional._pink_noise_blocks

    def observed_generator(sample_count, channel_count, seed_sequence):
        for start, block in real_generator(
            sample_count,
            channel_count,
            seed_sequence,
        ):
            observed_block_samples.append(block.shape[0])
            yield start, block

    monkeypatch.setattr(conventional, "_pink_noise_blocks", observed_generator)
    provenance = archival_noise_v1(
        source_audio,
        source_midi,
        output_audio,
        output_midi,
        seed=424242,
        parameters=_archival_parameters(24.0),
    )

    source_info = sf.info(source_audio)
    output_info = sf.info(output_audio)
    assert source_info.frames == 3_249_628
    assert output_info.frames == source_info.frames
    assert output_info.samplerate == source_info.samplerate == 44_100
    assert output_info.channels == source_info.channels == 1
    assert output_midi.read_bytes() == source_midi.read_bytes()
    assert observed_block_samples
    assert max(observed_block_samples) <= conventional.ARCHIVAL_NOISE_BLOCK_SAMPLES
    assert len(observed_block_samples) > 2
    qc = provenance["qc"]
    assert qc["working_block_samples"] == conventional.ARCHIVAL_NOISE_BLOCK_SAMPLES
    assert qc["temporary_memory_model"] == (
        "O(working_block_samples * channels); one full-size output buffer"
    )
    assert qc["whole_record_fft_used"] is False
    assert qc["measured_float_snr_db"] == pytest.approx(24.0, abs=1e-12)
    assert qc["measured_hum_power_fraction"] == pytest.approx(0.20, abs=1e-12)


def test_archival_noise_publishes_pink_spectrum_and_seeded_harmonic_hum(
    tmp_path: Path,
) -> None:
    sample_rate = 8000
    source_audio, source_midi = _write_pair(
        tmp_path / "source",
        seconds=4.0,
        sample_rate=sample_rate,
    )
    output_audio = tmp_path / "archival.wav"
    output_midi = tmp_path / "archival.mid"
    parameters = ArchivalNoiseParameters(
        target_snr_db=24.0,
        hum_power_fraction=0.20,
        mains_frequency_hz=60.0,
        harmonic_count=3,
    )
    provenance = archival_noise_v1(
        source_audio,
        source_midi,
        output_audio,
        output_midi,
        seed=123456,
        parameters=parameters,
    )

    source, _ = sf.read(source_audio, always_2d=True)
    output, _ = sf.read(output_audio, always_2d=True)
    guard = float(provenance["qc"]["peak_guard_linear_gain"])
    interference = output / guard - source
    signal_rms = math.sqrt(float(np.mean(np.square(source), dtype=np.float64)))
    interference_rms = math.sqrt(
        float(np.mean(np.square(interference), dtype=np.float64))
    )
    decoded_snr = 20.0 * math.log10(signal_rms / interference_rms)
    assert decoded_snr == pytest.approx(parameters.target_snr_db, abs=0.02)

    phases = np.asarray(
        provenance["qc"]["hum_phases_radians_by_channel"],
        dtype=np.float64,
    )
    relative_amplitudes = np.asarray(
        provenance["qc"]["hum_harmonic_relative_amplitudes"],
        dtype=np.float64,
    )
    time = np.arange(source.shape[0], dtype=np.float64) / sample_rate
    hum = np.zeros_like(source)
    for harmonic_index, relative_amplitude in enumerate(
        relative_amplitudes,
        start=1,
    ):
        hum += relative_amplitude * np.sin(
            2.0 * np.pi * parameters.mains_frequency_hz * harmonic_index
            * time[:, np.newaxis]
            + phases[:, harmonic_index - 1][np.newaxis, :]
        )
    hum -= np.mean(hum, axis=0, keepdims=True)
    hum /= math.sqrt(float(np.mean(np.square(hum), dtype=np.float64)))
    measured_hum_coefficient = float(
        np.mean(interference * hum, dtype=np.float64)
    )
    measured_hum_fraction = measured_hum_coefficient**2 / (
        interference_rms**2
    )
    assert measured_hum_fraction == pytest.approx(
        parameters.hum_power_fraction,
        abs=0.005,
    )

    pink_residual = interference - measured_hum_coefficient * hum
    power = np.square(np.abs(np.fft.rfft(pink_residual[:, 0])))
    frequencies = np.fft.rfftfreq(pink_residual.shape[0], 1.0 / sample_rate)
    mask = (frequencies >= 20.0) & (frequencies <= 2000.0) & (power > 0.0)
    for harmonic_index in range(1, parameters.harmonic_count + 1):
        mask &= np.abs(
            frequencies - parameters.mains_frequency_hz * harmonic_index
        ) > 1.0
    spectral_slope = float(
        np.polyfit(np.log10(frequencies[mask]), np.log10(power[mask]), 1)[0]
    )
    assert -1.20 <= spectral_slope <= -0.80


@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100])
def test_recursive_pink_spectrum_contract_across_supported_sample_rates(
    sample_rate: int,
) -> None:
    sample_count = 4 * sample_rate
    blocks = conventional._pink_noise_blocks(
        sample_count,
        1,
        np.random.SeedSequence(9321),
    )
    pink = np.concatenate([block[:, 0] for _, block in blocks])
    pink -= np.mean(pink)
    power = np.square(np.abs(np.fft.rfft(pink)))
    frequencies = np.fft.rfftfreq(sample_count, 1.0 / sample_rate)
    upper_hz = min(10_000.0, 0.40 * sample_rate)
    mask = (frequencies >= 20.0) & (frequencies <= upper_hz) & (power > 0.0)
    spectral_slope = float(
        np.polyfit(np.log10(frequencies[mask]), np.log10(power[mask]), 1)[0]
    )
    assert -1.20 <= spectral_slope <= -0.80


def test_archival_noise_uses_explicit_peak_guard_instead_of_clipping(
    tmp_path: Path,
) -> None:
    sample_rate = 8000
    source_audio, source_midi = _write_pair(tmp_path / "source")
    time = np.arange(2 * sample_rate, dtype=np.float64) / sample_rate
    sf.write(
        source_audio,
        0.95 * np.sin(2.0 * np.pi * 440.0 * time),
        sample_rate,
        subtype="PCM_16",
    )
    output_audio = tmp_path / "guarded.wav"
    output_midi = tmp_path / "guarded.mid"
    provenance = archival_noise_v1(
        source_audio,
        source_midi,
        output_audio,
        output_midi,
        seed=77,
        parameters=_archival_parameters(0.0),
    )

    output, _ = sf.read(output_audio)
    qc = provenance["qc"]
    assert qc["peak_before_guard"] > qc["peak_limit"]
    assert 0.0 < qc["peak_guard_linear_gain"] < 1.0
    assert np.max(np.abs(output)) <= 0.999 + 1.0 / 32768.0
    assert qc["hard_clipping_used"] is False
    assert qc["opaque_peak_normalization_used"] is False


@pytest.mark.parametrize(
    "parameters",
    [
        FractionalDetuningParameters(cents=0.0),
        FractionalDetuningParameters(cents=50.0),
        FractionalDetuningParameters(cents=-50.0),
        FractionalDetuningParameters(cents=float("nan")),
        FractionalDetuningParameters(cents=float("inf")),
        FractionalDetuningParameters(cents=True),
        FractionalDetuningParameters(cents=np.float64(15.0)),
    ],
)
def test_fractional_detuning_invalid_parameters_publish_nothing(
    tmp_path: Path,
    parameters: FractionalDetuningParameters,
) -> None:
    source_audio, source_midi = _write_pair(tmp_path / "source")
    output_audio = tmp_path / "output.wav"
    output_midi = tmp_path / "output.mid"
    with pytest.raises((TypeError, ValueError)):
        fractional_detuning_v1(
            source_audio,
            source_midi,
            output_audio,
            output_midi,
            seed=1,
            parameters=parameters,
        )
    assert not output_audio.exists()
    assert not output_midi.exists()
    assert not output_audio.with_suffix(".wav.provenance.json").exists()


@pytest.mark.parametrize(
    "parameters",
    [
        _archival_parameters(float("nan")),
        _archival_parameters(24.0, hum_power_fraction=0.0),
        _archival_parameters(24.0, hum_power_fraction=1.0),
        _archival_parameters(24.0, hum_power_fraction=True),
        _archival_parameters(24.0, mains_frequency_hz=0.0),
        _archival_parameters(24.0, mains_frequency_hz=float("inf")),
        _archival_parameters(24.0, harmonic_count=0),
        _archival_parameters(24.0, harmonic_count=17),
        _archival_parameters(24.0, harmonic_count=1.5),
        _archival_parameters(24.0, harmonic_count=True),
        ArchivalNoiseParameters(
            target_snr_db=24.0,
            hum_power_fraction=0.20,
            mains_frequency_hz=2000.0,
            harmonic_count=2,
        ),
    ],
)
def test_archival_noise_invalid_parameters_publish_nothing(
    tmp_path: Path,
    parameters: ArchivalNoiseParameters,
) -> None:
    source_audio, source_midi = _write_pair(tmp_path / "source")
    output_audio = tmp_path / "output.wav"
    output_midi = tmp_path / "output.mid"
    with pytest.raises((TypeError, ValueError)):
        archival_noise_v1(
            source_audio,
            source_midi,
            output_audio,
            output_midi,
            seed=1,
            parameters=parameters,
        )
    assert not output_audio.exists()
    assert not output_midi.exists()
    assert not output_audio.with_suffix(".wav.provenance.json").exists()


def test_archival_noise_requires_an_explicit_corpus_profile() -> None:
    with pytest.raises(TypeError):
        ArchivalNoiseParameters(target_snr_db=24.0)


def test_silent_source_and_wrong_detuning_shape_fail_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    silent_audio, silent_midi = _write_pair(tmp_path / "silent", silent=True)
    with pytest.raises(RuntimeError, match="input audio RMS"):
        archival_noise_v1(
            silent_audio,
            silent_midi,
            tmp_path / "silent-output.wav",
            tmp_path / "silent-output.mid",
            seed=1,
            parameters=_archival_parameters(24.0),
        )
    assert not (tmp_path / "silent-output.wav").exists()

    source_audio, source_midi = _write_pair(tmp_path / "wrong-shape")

    def wrong_shape(audio, **kwargs):
        del kwargs
        return audio[:, :-1]

    monkeypatch.setattr(conventional.librosa.effects, "pitch_shift", wrong_shape)
    with pytest.raises(RuntimeError, match="changed audio shape"):
        fractional_detuning_v1(
            source_audio,
            source_midi,
            tmp_path / "shape-output.wav",
            tmp_path / "shape-output.mid",
            seed=1,
            parameters=FractionalDetuningParameters(cents=15.0),
        )
    assert not (tmp_path / "shape-output.wav").exists()
    assert not (tmp_path / "shape-output.mid").exists()


@pytest.mark.parametrize(
    "function,parameters",
    [
        (fractional_detuning_v1, FractionalDetuningParameters(cents=15.0)),
        (archival_noise_v1, _archival_parameters(24.0)),
    ],
)
def test_new_transforms_refuse_overwrite_and_roll_back_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    function,
    parameters,
) -> None:
    source_audio, source_midi = _write_pair(tmp_path / "source")
    occupied_audio = tmp_path / "occupied.wav"
    occupied_midi = tmp_path / "occupied.mid"
    sentinel = b"do-not-overwrite"
    occupied_audio.write_bytes(sentinel)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        function(
            source_audio,
            source_midi,
            occupied_audio,
            occupied_midi,
            seed=9,
            parameters=parameters,
        )
    assert occupied_audio.read_bytes() == sentinel
    assert not occupied_midi.exists()

    output_audio = tmp_path / "rollback.wav"
    output_midi = tmp_path / "rollback.mid"
    output_provenance = output_audio.with_suffix(".wav.provenance.json")
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
        function(
            source_audio,
            source_midi,
            output_audio,
            output_midi,
            seed=9,
            parameters=parameters,
        )
    assert not output_audio.exists()
    assert not output_midi.exists()
    assert not output_provenance.exists()
    assert not list(tmp_path.rglob(".*.staging-*"))
