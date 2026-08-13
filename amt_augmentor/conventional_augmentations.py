"""Fail-closed conventional augmentations for controlled AMT experiments.

The original AMT-Augmentor functions remain available for backward
compatibility, but they are not suitable for reproducible comparisons:
some use process-global randomness, some silently discard malformed labels,
and none publishes source-bound provenance. This module provides corrected,
versioned forms of the toolbox's established conventional transforms.

All public functions:

* require explicit, finite parameters and a nonnegative integer seed;
* accept paired audio/MIDI files and refuse to overwrite any output;
* preserve audio sample rate/channel count and publish PCM audio;
* preserve MIDI bytes for audio-only transforms;
* publish audio, MIDI, and a hash-bearing provenance sidecar as one logical
  bundle (the sidecar is the completion marker);
* record deterministic DSP and synchronization checks.

``time_stretch_v1`` maps all MIDI times by the *realized* integer-sample
duration ratio, rather than rounding annotation text or assuming that the
nominal playback rate can be represented exactly in samples.

The opt-in ``fractional_detuning_v1`` and ``archival_noise_v1`` APIs are
audio-only successor transforms. They intentionally are not wired into the
frozen Galdr conventional campaign adapter.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import librosa
import numpy as np
import pretty_midi
from pedalboard import (
    Chorus,
    Gain,
    HighpassFilter,
    LowpassFilter,
    Pedalboard,
    Reverb,
)
from scipy.signal import lfilter

from amt_augmentor._paired_io import (
    _attach_plan_config,
    _copy_midi_with_time_map,
    _load_pair,
    _provenance_base,
    _stage_and_publish_bundle,
    _validate_output_paths,
    _validate_seed,
)

PEAK_LIMIT = 0.999
MODEL_MINIMUM_MIDI_PITCH = 21
MODEL_MAXIMUM_MIDI_PITCH = 108
MAXIMUM_HUM_HARMONICS = 16
ARCHIVAL_NOISE_BLOCK_SAMPLES = 65536
PINK_FILTER_WARMUP_SAMPLES = 16384
PINK_FILTER_POLES = (0.99886, 0.99332, 0.96900, 0.86650, 0.55000, -0.7616)
PINK_FILTER_GAINS = (
    0.0555179,
    0.0750759,
    0.1538520,
    0.3104856,
    0.5329522,
    -0.0168980,
)
PINK_DIRECT_GAIN = 0.5362
PINK_DELAYED_WHITE_GAIN = 0.115926
PINK_FILTER_REFERENCE_SAMPLE_RATE_HZ = 44100
PINK_SPECTRUM_VALIDATION_SAMPLE_RATES_HZ = (8000, 16000, 44100)


@dataclass(frozen=True)
class GainChorusParameters:
    """Parameters for :func:`gain_chorus_v1`."""

    gain_db: float
    chorus_depth: float
    chorus_rate_hz: float
    chorus_centre_delay_ms: float = 7.0
    chorus_feedback: float = 0.2
    chorus_mix: float = 0.25


@dataclass(frozen=True)
class FractionalDetuningParameters:
    """A finite, nonzero detuning strictly between -50 and 50 cents."""

    cents: float


@dataclass(frozen=True)
class NoiseSNRParameters:
    """Parameters for :func:`noise_snr_v1`."""

    target_snr_db: float


@dataclass(frozen=True)
class ArchivalNoiseParameters:
    """Target SNR and finite-record pink-noise/harmonic-hum power mix."""

    target_snr_db: float
    hum_power_fraction: float = 0.20
    mains_frequency_hz: float = 50.0
    harmonic_count: int = 3


@dataclass(frozen=True)
class ReverbFiltersParameters:
    """Parameters for :func:`reverb_filters_v1`."""

    room_size: float
    wet_level: float
    dry_level: float
    highpass_hz: float
    lowpass_hz: float


@dataclass(frozen=True)
class PitchShiftParameters:
    """Parameters for :func:`pitch_shift_v1`."""

    semitones: int
    minimum_midi_pitch: int = MODEL_MINIMUM_MIDI_PITCH
    maximum_midi_pitch: int = MODEL_MAXIMUM_MIDI_PITCH


@dataclass(frozen=True)
class TimeStretchParameters:
    """Playback-rate parameters for :func:`time_stretch_v1`."""

    rate: float


def _require_builtin_number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite built-in int or float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_range(
    value: Any,
    name: str,
    minimum: float,
    maximum: float,
    *,
    inclusive_minimum: bool = True,
) -> float:
    result = _require_builtin_number(value, name)
    minimum_ok = result >= minimum if inclusive_minimum else result > minimum
    if not minimum_ok or result > maximum:
        left = "[" if inclusive_minimum else "("
        raise ValueError(f"{name} must be in {left}{minimum}, {maximum}]")
    return result


def _validate_gain_chorus(parameters: GainChorusParameters) -> None:
    _require_range(parameters.gain_db, "gain_db", -24.0, 24.0)
    _require_range(parameters.chorus_depth, "chorus_depth", 0.0, 1.0)
    _require_range(parameters.chorus_rate_hz, "chorus_rate_hz", 0.01, 20.0)
    _require_range(
        parameters.chorus_centre_delay_ms,
        "chorus_centre_delay_ms",
        0.1,
        100.0,
    )
    _require_range(parameters.chorus_feedback, "chorus_feedback", 0.0, 0.95)
    _require_range(parameters.chorus_mix, "chorus_mix", 0.0, 1.0)


def _validate_noise(parameters: NoiseSNRParameters) -> None:
    _require_range(parameters.target_snr_db, "target_snr_db", -20.0, 100.0)


def _validate_fractional_detuning(
    parameters: FractionalDetuningParameters,
) -> None:
    cents = _require_builtin_number(parameters.cents, "cents")
    if cents == 0.0 or abs(cents) >= 50.0:
        raise ValueError("cents must satisfy 0 < abs(cents) < 50")


def _validate_archival_noise(
    parameters: ArchivalNoiseParameters,
    *,
    sample_rate: Optional[int] = None,
) -> None:
    _require_range(parameters.target_snr_db, "target_snr_db", -20.0, 100.0)
    hum_power_fraction = _require_builtin_number(
        parameters.hum_power_fraction,
        "hum_power_fraction",
    )
    if not 0.0 < hum_power_fraction < 1.0:
        raise ValueError("hum_power_fraction must be strictly between 0 and 1")
    mains_frequency_hz = _require_builtin_number(
        parameters.mains_frequency_hz,
        "mains_frequency_hz",
    )
    if mains_frequency_hz <= 0.0:
        raise ValueError("mains_frequency_hz must be positive")
    if type(parameters.harmonic_count) is not int:
        raise TypeError("harmonic_count must be a built-in int")
    if not 1 <= parameters.harmonic_count <= MAXIMUM_HUM_HARMONICS:
        raise ValueError(
            f"harmonic_count must be in [1, {MAXIMUM_HUM_HARMONICS}]"
        )
    if sample_rate is not None:
        highest_harmonic_hz = mains_frequency_hz * parameters.harmonic_count
        if highest_harmonic_hz >= sample_rate / 2.0:
            raise ValueError(
                "highest mains harmonic must be strictly below the audio Nyquist "
                "frequency"
            )


def _validate_reverb(parameters: ReverbFiltersParameters, sample_rate: int) -> None:
    _require_range(parameters.room_size, "room_size", 0.0, 1.0)
    _require_range(parameters.wet_level, "wet_level", 0.0, 1.0)
    _require_range(parameters.dry_level, "dry_level", 0.0, 1.0)
    highpass = _require_range(
        parameters.highpass_hz,
        "highpass_hz",
        20.0,
        sample_rate / 2.0,
    )
    lowpass = _require_range(
        parameters.lowpass_hz,
        "lowpass_hz",
        20.0,
        sample_rate / 2.0,
    )
    if highpass >= lowpass:
        raise ValueError("highpass_hz must be lower than lowpass_hz")


def _validate_pitch(parameters: PitchShiftParameters) -> None:
    if type(parameters.semitones) is not int:
        raise TypeError("semitones must be a nonzero built-in int")
    if parameters.semitones == 0:
        raise ValueError("semitones must be nonzero")
    if type(parameters.minimum_midi_pitch) is not int:
        raise TypeError("minimum_midi_pitch must be a built-in int")
    if type(parameters.maximum_midi_pitch) is not int:
        raise TypeError("maximum_midi_pitch must be a built-in int")
    if not 0 <= parameters.minimum_midi_pitch <= parameters.maximum_midi_pitch <= 127:
        raise ValueError(
            "MIDI pitch bounds must satisfy 0 <= minimum <= maximum <= 127"
        )


def _validate_time_stretch(parameters: TimeStretchParameters) -> None:
    _require_range(
        parameters.rate,
        "rate",
        0.25,
        4.0,
        inclusive_minimum=True,
    )


def _validate_loaded_pair(
    audio: np.ndarray,
    sample_rate: int,
    midi: pretty_midi.PrettyMIDI,
) -> None:
    if audio.ndim != 2 or audio.shape[0] <= 0 or audio.shape[1] <= 0:
        raise ValueError("audio must contain a non-empty samples-by-channels array")
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    if not np.isfinite(audio).all():
        raise ValueError("input audio contains NaN or infinite samples")
    if len(midi.instruments) != 1:
        raise ValueError("MIDI must contain exactly one instrument")
    instrument = midi.instruments[0]
    if instrument.is_drum:
        raise ValueError("MIDI instrument must not be a drum track")
    if instrument.control_changes:
        raise ValueError("MIDI control changes are not supported by these transforms")
    if instrument.pitch_bends:
        raise ValueError("MIDI pitch bends are not supported by these transforms")
    if not instrument.notes:
        raise ValueError("MIDI must contain at least one note")
    audio_duration = audio.shape[0] / sample_rate
    for index, note in enumerate(instrument.notes):
        if (
            not math.isfinite(float(note.start))
            or not math.isfinite(float(note.end))
            or note.start < 0
            or note.end <= note.start
        ):
            raise ValueError(f"MIDI note {index} has an invalid time interval")
        if not 0 <= int(note.pitch) <= 127:
            raise ValueError(f"MIDI note {index} has an invalid pitch")
        if not 1 <= int(note.velocity) <= 127:
            raise ValueError(f"MIDI note {index} has an invalid velocity")
        if float(note.end) > audio_duration:
            raise ValueError(
                f"MIDI note {index} ends at {float(note.end):.9f} s, beyond "
                f"the audio duration of {audio_duration:.9f} s"
            )


def _output_paths(
    audio_path: os.PathLike,
    midi_path: os.PathLike,
    output_audio_path: os.PathLike,
    output_midi_path: os.PathLike,
    provenance_path: Optional[os.PathLike],
) -> Tuple[Path, Path, Path, Path, Path]:
    source_audio = Path(audio_path)
    source_midi = Path(midi_path)
    target_audio = Path(output_audio_path)
    target_midi = Path(output_midi_path)
    target_provenance = (
        Path(provenance_path)
        if provenance_path is not None
        else target_audio.with_suffix(target_audio.suffix + ".provenance.json")
    )
    _validate_output_paths(
        source_audio,
        source_midi,
        target_audio,
        target_midi,
        target_provenance,
    )
    return (
        source_audio,
        source_midi,
        target_audio,
        target_midi,
        target_provenance,
    )


def _peak_guard(audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    if not np.isfinite(audio).all():
        raise RuntimeError("rendered audio contains NaN or infinite samples")
    peak_before = float(np.max(np.abs(audio), initial=0.0))
    if peak_before == 0.0:
        raise RuntimeError("rendered audio is completely silent")
    applied_gain = min(1.0, PEAK_LIMIT / peak_before)
    guarded = audio * applied_gain
    peak_after = float(np.max(np.abs(guarded), initial=0.0))
    return guarded, {
        "peak_before_guard": peak_before,
        "peak_limit": PEAK_LIMIT,
        "peak_guard_linear_gain": applied_gain,
        "peak_after_guard": peak_after,
    }


def _peak_guard_in_place(audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply the explicit peak guard without another full-length allocation."""

    peak_before = 0.0
    for start in range(0, audio.shape[0], ARCHIVAL_NOISE_BLOCK_SAMPLES):
        block = audio[start : start + ARCHIVAL_NOISE_BLOCK_SAMPLES]
        if not np.isfinite(block).all():
            raise RuntimeError("rendered audio contains NaN or infinite samples")
        peak_before = max(
            peak_before,
            float(np.max(np.abs(block), initial=0.0)),
        )
    if peak_before == 0.0:
        raise RuntimeError("rendered audio is completely silent")
    applied_gain = min(1.0, PEAK_LIMIT / peak_before)
    np.multiply(audio, applied_gain, out=audio)
    peak_after = 0.0
    for start in range(0, audio.shape[0], ARCHIVAL_NOISE_BLOCK_SAMPLES):
        block = audio[start : start + ARCHIVAL_NOISE_BLOCK_SAMPLES]
        peak_after = max(
            peak_after,
            float(np.max(np.abs(block), initial=0.0)),
        )
    return audio, {
        "peak_before_guard": peak_before,
        "peak_limit": PEAK_LIMIT,
        "peak_guard_linear_gain": applied_gain,
        "peak_after_guard": peak_after,
    }


def _array_rms(audio: np.ndarray, label: str) -> float:
    square_sum = 0.0
    for start in range(0, audio.shape[0], ARCHIVAL_NOISE_BLOCK_SAMPLES):
        block = audio[start : start + ARCHIVAL_NOISE_BLOCK_SAMPLES]
        square_sum += float(np.sum(np.square(block), dtype=np.float64))
    value = math.sqrt(square_sum / audio.size)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{label} RMS must be finite and positive")
    return value


def _float64_array_sha256(audio: np.ndarray) -> str:
    digest = hashlib.sha256()
    for start in range(0, audio.shape[0], ARCHIVAL_NOISE_BLOCK_SAMPLES):
        canonical = np.ascontiguousarray(
            audio[start : start + ARCHIVAL_NOISE_BLOCK_SAMPLES],
            dtype=np.dtype("<f8"),
        )
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _pink_noise_blocks(
    sample_count: int,
    channel_count: int,
    seed_sequence: np.random.SeedSequence,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield deterministic pink-noise blocks with constant-size DSP state."""

    generator = np.random.default_rng(seed_sequence)
    filter_states = np.zeros(
        (channel_count, len(PINK_FILTER_POLES)),
        dtype=np.float64,
    )
    previous_white = np.zeros(channel_count, dtype=np.float64)

    def render(white: np.ndarray) -> np.ndarray:
        nonlocal previous_white
        pink = white * PINK_DIRECT_GAIN
        for channel in range(channel_count):
            for filter_index, (pole, gain) in enumerate(
                zip(PINK_FILTER_POLES, PINK_FILTER_GAINS)
            ):
                filtered, final_state = lfilter(
                    (gain,),
                    (1.0, -pole),
                    white[:, channel],
                    zi=(filter_states[channel, filter_index],),
                )
                filter_states[channel, filter_index] = final_state[0]
                pink[:, channel] += filtered
        pink[0, :] += PINK_DELAYED_WHITE_GAIN * previous_white
        if white.shape[0] > 1:
            pink[1:, :] += PINK_DELAYED_WHITE_GAIN * white[:-1, :]
        previous_white = white[-1, :].copy()
        return pink

    remaining_warmup = PINK_FILTER_WARMUP_SAMPLES
    while remaining_warmup > 0:
        block_samples = min(remaining_warmup, ARCHIVAL_NOISE_BLOCK_SAMPLES)
        render(
            generator.standard_normal(
                (block_samples, channel_count),
                dtype=np.float64,
            )
        )
        remaining_warmup -= block_samples

    for start in range(0, sample_count, ARCHIVAL_NOISE_BLOCK_SAMPLES):
        block_samples = min(
            ARCHIVAL_NOISE_BLOCK_SAMPLES,
            sample_count - start,
        )
        white = generator.standard_normal(
            (block_samples, channel_count),
            dtype=np.float64,
        )
        yield start, render(white)


def _harmonic_hum_block(
    start: int,
    block_samples: int,
    channel_count: int,
    sample_rate: int,
    mains_frequency_hz: float,
    phases: np.ndarray,
    relative_amplitudes: np.ndarray,
) -> np.ndarray:
    """Render one bounded block of seeded-phase harmonic mains hum."""

    time = (start + np.arange(block_samples, dtype=np.float64)) / sample_rate
    hum = np.zeros((block_samples, channel_count), dtype=np.float64)
    for harmonic_index, relative_amplitude in enumerate(
        relative_amplitudes,
        start=1,
    ):
        angular_frequency = 2.0 * np.pi * mains_frequency_hz * harmonic_index
        hum += relative_amplitude * np.sin(
            angular_frequency * time[:, np.newaxis]
            + phases[:, harmonic_index - 1][np.newaxis, :]
        )
    return hum


def _raw_component_statistics(
    shape: Tuple[int, int],
    sample_rate: int,
    pink_seed: np.random.SeedSequence,
    mains_frequency_hz: float,
    phases: np.ndarray,
    relative_amplitudes: np.ndarray,
) -> Dict[str, Any]:
    """Measure finite-record moments without retaining either component."""

    sample_count, channel_count = shape
    pink_sum = np.zeros(channel_count, dtype=np.float64)
    hum_sum = np.zeros(channel_count, dtype=np.float64)
    pink_square_sum = 0.0
    hum_square_sum = 0.0
    cross_sum = 0.0
    for start, pink in _pink_noise_blocks(
        sample_count,
        channel_count,
        pink_seed,
    ):
        hum = _harmonic_hum_block(
            start,
            pink.shape[0],
            channel_count,
            sample_rate,
            mains_frequency_hz,
            phases,
            relative_amplitudes,
        )
        pink_sum += np.sum(pink, axis=0, dtype=np.float64)
        hum_sum += np.sum(hum, axis=0, dtype=np.float64)
        pink_square_sum += float(np.sum(np.square(pink), dtype=np.float64))
        hum_square_sum += float(np.sum(np.square(hum), dtype=np.float64))
        cross_sum += float(np.sum(pink * hum, dtype=np.float64))

    pink_mean = pink_sum / sample_count
    hum_mean = hum_sum / sample_count
    value_count = sample_count * channel_count
    pink_power = (
        pink_square_sum - sample_count * float(np.dot(pink_mean, pink_mean))
    ) / value_count
    hum_power = (
        hum_square_sum - sample_count * float(np.dot(hum_mean, hum_mean))
    ) / value_count
    centered_cross_power = (
        cross_sum - sample_count * float(np.dot(pink_mean, hum_mean))
    ) / value_count
    if (
        not math.isfinite(pink_power)
        or not math.isfinite(hum_power)
        or pink_power <= 0.0
        or hum_power <= 0.0
    ):
        raise RuntimeError("archival-noise component power is not finite and positive")
    pink_rms = math.sqrt(pink_power)
    hum_rms = math.sqrt(hum_power)
    unit_cross_power = centered_cross_power / (pink_rms * hum_rms)
    orthogonalized_pink_power = 1.0 - unit_cross_power**2
    if (
        not math.isfinite(orthogonalized_pink_power)
        or orthogonalized_pink_power <= 0.0
    ):
        raise RuntimeError("pink noise is degenerate after hum orthogonalization")
    return {
        "pink_mean_by_channel": pink_mean,
        "hum_mean_by_channel": hum_mean,
        "pink_rms_before_normalization": pink_rms,
        "hum_rms_before_normalization": hum_rms,
        "hum_projection_removed_from_pink": unit_cross_power,
        "orthogonalized_pink_rms_before_normalization": math.sqrt(
            orthogonalized_pink_power
        ),
    }


def _archival_interference(
    shape: Tuple[int, int],
    sample_rate: int,
    seed: int,
    parameters: ArchivalNoiseParameters,
    target_interference_rms: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build exact-SNR interference with bounded temporary working memory."""

    sample_count, channel_count = shape
    seed_sequence = np.random.SeedSequence(seed)
    pink_seed, hum_seed = seed_sequence.spawn(2)
    phases = np.random.default_rng(hum_seed).uniform(
        0.0,
        2.0 * np.pi,
        size=(channel_count, parameters.harmonic_count),
    )
    relative_amplitudes = 1.0 / np.arange(
        1,
        parameters.harmonic_count + 1,
        dtype=np.float64,
    )
    statistics = _raw_component_statistics(
        shape,
        sample_rate,
        pink_seed,
        float(parameters.mains_frequency_hz),
        phases,
        relative_amplitudes,
    )
    hum_fraction = float(parameters.hum_power_fraction)
    pink_weight = math.sqrt(1.0 - hum_fraction)
    hum_weight = math.sqrt(hum_fraction)
    interference = np.empty(shape, dtype=np.float64)
    pink_unit_square_sum = 0.0
    hum_unit_square_sum = 0.0
    unit_cross_sum = 0.0
    combined_square_sum = 0.0
    for start, pink in _pink_noise_blocks(
        sample_count,
        channel_count,
        pink_seed,
    ):
        hum = _harmonic_hum_block(
            start,
            pink.shape[0],
            channel_count,
            sample_rate,
            float(parameters.mains_frequency_hz),
            phases,
            relative_amplitudes,
        )
        pink -= statistics["pink_mean_by_channel"][np.newaxis, :]
        pink /= statistics["pink_rms_before_normalization"]
        hum -= statistics["hum_mean_by_channel"][np.newaxis, :]
        hum /= statistics["hum_rms_before_normalization"]
        pink -= statistics["hum_projection_removed_from_pink"] * hum
        pink /= statistics["orthogonalized_pink_rms_before_normalization"]

        pink_unit_square_sum += float(np.sum(np.square(pink), dtype=np.float64))
        hum_unit_square_sum += float(np.sum(np.square(hum), dtype=np.float64))
        unit_cross_sum += float(np.sum(pink * hum, dtype=np.float64))
        output_block = interference[start : start + pink.shape[0]]
        np.multiply(pink, pink_weight, out=output_block)
        output_block += hum_weight * hum
        combined_square_sum += float(
            np.sum(np.square(output_block), dtype=np.float64)
        )

    value_count = sample_count * channel_count
    pink_unit_power = pink_unit_square_sum / value_count
    hum_unit_power = hum_unit_square_sum / value_count
    unit_cross_power = unit_cross_sum / value_count
    combined_rms_before_scaling = math.sqrt(
        combined_square_sum / value_count
    )
    final_scale = target_interference_rms / combined_rms_before_scaling
    np.multiply(interference, final_scale, out=interference)
    interference_rms = _array_rms(interference, "scaled archival interference")
    pink_rms = abs(final_scale * pink_weight) * math.sqrt(pink_unit_power)
    hum_rms = abs(final_scale * hum_weight) * math.sqrt(hum_unit_power)
    component_cross_power = (
        final_scale**2 * pink_weight * hum_weight * unit_cross_power
    )
    component_cross_correlation = component_cross_power / (pink_rms * hum_rms)
    component_power_sum = pink_rms**2 + hum_rms**2

    return interference, {
        "rng_algorithm": (
            "numpy SeedSequence.spawn with independent default_rng streams"
        ),
        "pink_stream_spawn_key": list(pink_seed.spawn_key),
        "hum_stream_spawn_key": list(hum_seed.spawn_key),
        "pink_noise_algorithm": (
            "six-state recursive filter plus direct and delayed white terms v1"
        ),
        "pink_spectrum_contract": "deterministic 1/f-like power approximation",
        "pink_filter_poles": list(PINK_FILTER_POLES),
        "pink_filter_gains": list(PINK_FILTER_GAINS),
        "pink_direct_gain": PINK_DIRECT_GAIN,
        "pink_delayed_white_gain": PINK_DELAYED_WHITE_GAIN,
        "pink_filter_reference_sample_rate_hz": (
            PINK_FILTER_REFERENCE_SAMPLE_RATE_HZ
        ),
        "pink_spectrum_validation_sample_rates_hz": list(
            PINK_SPECTRUM_VALIDATION_SAMPLE_RATES_HZ
        ),
        "pink_filter_warmup_samples": PINK_FILTER_WARMUP_SAMPLES,
        "pink_centered_per_channel": True,
        "pink_generation_passes": 2,
        "whole_record_fft_used": False,
        "working_block_samples": ARCHIVAL_NOISE_BLOCK_SAMPLES,
        "temporary_memory_model": (
            "O(working_block_samples * channels); one full-size output buffer"
        ),
        "channel_generation": "independent pink stream and harmonic phases per channel",
        "hum_harmonic_relative_amplitudes": relative_amplitudes.tolist(),
        "hum_phases_radians_by_channel": phases.tolist(),
        "pink_mean_by_channel_before_centering": statistics[
            "pink_mean_by_channel"
        ].tolist(),
        "hum_mean_by_channel_before_centering": statistics[
            "hum_mean_by_channel"
        ].tolist(),
        "pink_rms_before_normalization": statistics[
            "pink_rms_before_normalization"
        ],
        "hum_rms_before_normalization": statistics[
            "hum_rms_before_normalization"
        ],
        "hum_projection_removed_from_pink": statistics[
            "hum_projection_removed_from_pink"
        ],
        "unit_component_cross_power_after_orthogonalization": unit_cross_power,
        "combined_unit_rms_before_final_scaling": combined_rms_before_scaling,
        "pink_component_rms": pink_rms,
        "hum_component_rms": hum_rms,
        "component_cross_power": component_cross_power,
        "component_cross_correlation": component_cross_correlation,
        "measured_hum_power_fraction": hum_rms**2 / component_power_sum,
        "interference_rms": interference_rms,
        "interference_float64_sha256": _float64_array_sha256(interference),
    }


def _pedalboard_audio(
    audio: np.ndarray,
    sample_rate: int,
    board: Pedalboard,
) -> np.ndarray:
    channel_first = np.ascontiguousarray(audio.T, dtype=np.float32)
    rendered = np.asarray(board(channel_first, sample_rate), dtype=np.float64)
    if rendered.ndim == 1:
        rendered = rendered[np.newaxis, :]
    output = rendered.T
    if output.shape != audio.shape:
        raise RuntimeError(
            "audio-only effect changed sample/channel shape: "
            f"{audio.shape} -> {output.shape}"
        )
    return output


def _pitch_shift_midi(
    source: pretty_midi.PrettyMIDI,
    semitones: int,
) -> pretty_midi.PrettyMIDI:
    output = _copy_midi_with_time_map(
        source,
        map_time=lambda value: value,
    )
    for instrument in output.instruments:
        for note in instrument.notes:
            note.pitch += semitones
    return output


def _audio_only_provenance(
    *,
    transform: str,
    seed: int,
    source_audio: Path,
    source_midi: Path,
    audio: np.ndarray,
    sample_rate: int,
    midi: pretty_midi.PrettyMIDI,
    parameters: Dict[str, Any],
    qc: Dict[str, Any],
) -> Dict[str, Any]:
    provenance = _provenance_base(
        transform,
        seed,
        source_audio,
        source_midi,
        audio,
        sample_rate,
        midi,
    )
    provenance.update(
        {
            "parameters": parameters,
            "qc": qc,
            "invariants": {
                "audio_sample_count_preserved": True,
                "audio_channel_count_preserved": True,
                "midi_preserved_byte_for_byte": True,
                "rendered_audio_finite": True,
                "peak_guard_is_explicit": True,
            },
        }
    )
    _attach_plan_config(provenance, parameters=parameters, operations=[])
    return provenance


def _publish_audio_only(
    *,
    transform: str,
    seed: int,
    parameters: Dict[str, Any],
    source_audio: Path,
    source_midi: Path,
    target_audio: Path,
    target_midi: Path,
    target_provenance: Path,
    output_audio: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
    audio_info,
    midi: pretty_midi.PrettyMIDI,
    qc: Dict[str, Any],
) -> Dict[str, Any]:
    provenance = _audio_only_provenance(
        transform=transform,
        seed=seed,
        source_audio=source_audio,
        source_midi=source_midi,
        audio=audio,
        sample_rate=sample_rate,
        midi=midi,
        parameters=parameters,
        qc=qc,
    )
    return _stage_and_publish_bundle(
        output_audio_path=target_audio,
        output_midi_path=target_midi,
        provenance_path=target_provenance,
        output_audio=output_audio,
        sample_rate=sample_rate,
        audio_info=audio_info,
        output_midi=midi,
        output_midi_bytes=source_midi.read_bytes(),
        provenance=provenance,
    )


def gain_chorus_v1(
    audio_path: os.PathLike,
    midi_path: os.PathLike,
    output_audio_path: os.PathLike,
    output_midi_path: os.PathLike,
    *,
    seed: int,
    parameters: GainChorusParameters,
    provenance_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Apply an explicit gain followed by a deterministic chorus effect."""

    _validate_seed(seed)
    _validate_gain_chorus(parameters)
    (
        source_audio,
        source_midi,
        target_audio,
        target_midi,
        target_provenance,
    ) = _output_paths(
        audio_path,
        midi_path,
        output_audio_path,
        output_midi_path,
        provenance_path,
    )
    audio, sample_rate, audio_info, midi = _load_pair(source_audio, source_midi)
    _validate_loaded_pair(audio, sample_rate, midi)
    board = Pedalboard(
        [
            Gain(gain_db=float(parameters.gain_db)),
            Chorus(
                depth=float(parameters.chorus_depth),
                rate_hz=float(parameters.chorus_rate_hz),
                centre_delay_ms=float(parameters.chorus_centre_delay_ms),
                feedback=float(parameters.chorus_feedback),
                mix=float(parameters.chorus_mix),
            ),
        ]
    )
    rendered = _pedalboard_audio(audio, sample_rate, board)
    rendered, peak_qc = _peak_guard(rendered)
    return _publish_audio_only(
        transform="gain_chorus_v1",
        seed=seed,
        parameters=asdict(parameters),
        source_audio=source_audio,
        source_midi=source_midi,
        target_audio=target_audio,
        target_midi=target_midi,
        target_provenance=target_provenance,
        output_audio=rendered,
        audio=audio,
        sample_rate=sample_rate,
        audio_info=audio_info,
        midi=midi,
        qc=peak_qc,
    )


def fractional_detuning_v1(
    audio_path: os.PathLike,
    midi_path: os.PathLike,
    output_audio_path: os.PathLike,
    output_midi_path: os.PathLike,
    *,
    seed: int,
    parameters: FractionalDetuningParameters,
    provenance_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Detune audio by less than half a semitone without changing MIDI labels."""

    _validate_seed(seed)
    _validate_fractional_detuning(parameters)
    (
        source_audio,
        source_midi,
        target_audio,
        target_midi,
        target_provenance,
    ) = _output_paths(
        audio_path,
        midi_path,
        output_audio_path,
        output_midi_path,
        provenance_path,
    )
    audio, sample_rate, audio_info, midi = _load_pair(source_audio, source_midi)
    _validate_loaded_pair(audio, sample_rate, midi)
    cents = float(parameters.cents)
    semitones = cents / 100.0
    channel_first = np.ascontiguousarray(audio.T, dtype=np.float64)
    rendered = np.asarray(
        librosa.effects.pitch_shift(
            channel_first,
            sr=sample_rate,
            n_steps=semitones,
            bins_per_octave=12,
            res_type="soxr_hq",
            scale=False,
        ),
        dtype=np.float64,
    ).T
    if rendered.shape != audio.shape:
        raise RuntimeError(
            "fractional detuning changed audio shape: "
            f"{audio.shape} -> {rendered.shape}"
        )
    rendered, peak_qc = _peak_guard(rendered)
    qc = {
        **peak_qc,
        "requested_cents": cents,
        "requested_semitones": semitones,
        "requested_frequency_ratio": 2.0 ** (cents / 1200.0),
        "pitch_shift_backend": "librosa.effects.pitch_shift",
        "bins_per_octave": 12,
        "resample_type": "soxr_hq",
        "energy_scaling_requested": False,
        "randomness_used": False,
        "source_audio_samples": int(audio.shape[0]),
        "output_audio_samples": int(rendered.shape[0]),
        "source_audio_channels": int(audio.shape[1]),
        "output_audio_channels": int(rendered.shape[1]),
        "symbolic_pitch_policy": "MIDI bytes retained because abs(cents) < 50",
    }
    return _publish_audio_only(
        transform="fractional_detuning_v1",
        seed=seed,
        parameters=asdict(parameters),
        source_audio=source_audio,
        source_midi=source_midi,
        target_audio=target_audio,
        target_midi=target_midi,
        target_provenance=target_provenance,
        output_audio=rendered,
        audio=audio,
        sample_rate=sample_rate,
        audio_info=audio_info,
        midi=midi,
        qc=qc,
    )


def noise_snr_v1(
    audio_path: os.PathLike,
    midi_path: os.PathLike,
    output_audio_path: os.PathLike,
    output_midi_path: os.PathLike,
    *,
    seed: int,
    parameters: NoiseSNRParameters,
    provenance_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Add seeded Gaussian noise at an explicit measured RMS SNR."""

    _validate_seed(seed)
    _validate_noise(parameters)
    (
        source_audio,
        source_midi,
        target_audio,
        target_midi,
        target_provenance,
    ) = _output_paths(
        audio_path,
        midi_path,
        output_audio_path,
        output_midi_path,
        provenance_path,
    )
    audio, sample_rate, audio_info, midi = _load_pair(source_audio, source_midi)
    _validate_loaded_pair(audio, sample_rate, midi)
    signal_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if not math.isfinite(signal_rms) or signal_rms <= 0:
        raise ValueError("input audio RMS must be finite and positive")
    generator = np.random.default_rng(seed)
    unit_noise = generator.standard_normal(audio.shape, dtype=np.float64)
    unit_noise_rms = float(np.sqrt(np.mean(np.square(unit_noise), dtype=np.float64)))
    target_noise_rms = signal_rms / (10.0 ** (parameters.target_snr_db / 20.0))
    noise = unit_noise * (target_noise_rms / unit_noise_rms)
    measured_noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    measured_snr = 20.0 * math.log10(signal_rms / measured_noise_rms)
    rendered, peak_qc = _peak_guard(audio + noise)
    qc = {
        **peak_qc,
        "signal_rms": signal_rms,
        "noise_rms": measured_noise_rms,
        "measured_float_snr_db": measured_snr,
        "snr_measurement_stage": "before PCM quantization and after noise scaling",
        "opaque_peak_normalization_used": False,
    }
    return _publish_audio_only(
        transform="noise_snr_v1",
        seed=seed,
        parameters=asdict(parameters),
        source_audio=source_audio,
        source_midi=source_midi,
        target_audio=target_audio,
        target_midi=target_midi,
        target_provenance=target_provenance,
        output_audio=rendered,
        audio=audio,
        sample_rate=sample_rate,
        audio_info=audio_info,
        midi=midi,
        qc=qc,
    )


def archival_noise_v1(
    audio_path: os.PathLike,
    midi_path: os.PathLike,
    output_audio_path: os.PathLike,
    output_midi_path: os.PathLike,
    *,
    seed: int,
    parameters: ArchivalNoiseParameters,
    provenance_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Add deterministic 1/f-like noise and seeded-phase hum at target SNR."""

    _validate_seed(seed)
    _validate_archival_noise(parameters)
    (
        source_audio,
        source_midi,
        target_audio,
        target_midi,
        target_provenance,
    ) = _output_paths(
        audio_path,
        midi_path,
        output_audio_path,
        output_midi_path,
        provenance_path,
    )
    audio, sample_rate, audio_info, midi = _load_pair(source_audio, source_midi)
    _validate_archival_noise(parameters, sample_rate=sample_rate)
    _validate_loaded_pair(audio, sample_rate, midi)
    if audio.shape[0] < 2:
        raise ValueError("archival noise requires at least two audio samples")
    signal_rms = _array_rms(audio, "input audio")
    target_interference_rms = signal_rms / (
        10.0 ** (float(parameters.target_snr_db) / 20.0)
    )
    if not math.isfinite(target_interference_rms) or target_interference_rms <= 0.0:
        raise ValueError("target archival-interference RMS must be finite and positive")
    interference, interference_qc = _archival_interference(
        audio.shape,
        sample_rate,
        seed,
        parameters,
        target_interference_rms,
    )
    measured_interference_rms = float(interference_qc["interference_rms"])
    measured_snr = 20.0 * math.log10(signal_rms / measured_interference_rms)
    interference += audio
    rendered, peak_qc = _peak_guard_in_place(interference)
    qc = {
        **peak_qc,
        **interference_qc,
        "signal_rms": signal_rms,
        "signal_rms_scope": "all samples and channels equally weighted",
        "target_interference_rms": target_interference_rms,
        "target_snr_db": float(parameters.target_snr_db),
        "measured_float_snr_db": measured_snr,
        "snr_absolute_error_db": abs(
            measured_snr - float(parameters.target_snr_db)
        ),
        "snr_measurement_stage": (
            "before peak guard and PCM quantization, after component "
            "orthogonalization and aggregate-RMS scaling"
        ),
        "component_power_scope": "all samples and channels equally weighted",
        "hard_clipping_used": False,
        "opaque_peak_normalization_used": False,
    }
    return _publish_audio_only(
        transform="archival_noise_v1",
        seed=seed,
        parameters=asdict(parameters),
        source_audio=source_audio,
        source_midi=source_midi,
        target_audio=target_audio,
        target_midi=target_midi,
        target_provenance=target_provenance,
        output_audio=rendered,
        audio=audio,
        sample_rate=sample_rate,
        audio_info=audio_info,
        midi=midi,
        qc=qc,
    )


def reverb_filters_v1(
    audio_path: os.PathLike,
    midi_path: os.PathLike,
    output_audio_path: os.PathLike,
    output_midi_path: os.PathLike,
    *,
    seed: int,
    parameters: ReverbFiltersParameters,
    provenance_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Apply deterministic reverb, high-pass, and low-pass filters."""

    _validate_seed(seed)
    (
        source_audio,
        source_midi,
        target_audio,
        target_midi,
        target_provenance,
    ) = _output_paths(
        audio_path,
        midi_path,
        output_audio_path,
        output_midi_path,
        provenance_path,
    )
    audio, sample_rate, audio_info, midi = _load_pair(source_audio, source_midi)
    _validate_reverb(parameters, sample_rate)
    _validate_loaded_pair(audio, sample_rate, midi)
    board = Pedalboard(
        [
            Reverb(
                room_size=float(parameters.room_size),
                wet_level=float(parameters.wet_level),
                dry_level=float(parameters.dry_level),
            ),
            HighpassFilter(cutoff_frequency_hz=float(parameters.highpass_hz)),
            LowpassFilter(cutoff_frequency_hz=float(parameters.lowpass_hz)),
        ]
    )
    rendered = _pedalboard_audio(audio, sample_rate, board)
    rendered, peak_qc = _peak_guard(rendered)
    return _publish_audio_only(
        transform="reverb_filters_v1",
        seed=seed,
        parameters=asdict(parameters),
        source_audio=source_audio,
        source_midi=source_midi,
        target_audio=target_audio,
        target_midi=target_midi,
        target_provenance=target_provenance,
        output_audio=rendered,
        audio=audio,
        sample_rate=sample_rate,
        audio_info=audio_info,
        midi=midi,
        qc=peak_qc,
    )


def pitch_shift_v1(
    audio_path: os.PathLike,
    midi_path: os.PathLike,
    output_audio_path: os.PathLike,
    output_midi_path: os.PathLike,
    *,
    seed: int,
    parameters: PitchShiftParameters,
    provenance_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Shift audio and MIDI together by an integral number of semitones."""

    _validate_seed(seed)
    _validate_pitch(parameters)
    (
        source_audio,
        source_midi,
        target_audio,
        target_midi,
        target_provenance,
    ) = _output_paths(
        audio_path,
        midi_path,
        output_audio_path,
        output_midi_path,
        provenance_path,
    )
    audio, sample_rate, audio_info, midi = _load_pair(source_audio, source_midi)
    _validate_loaded_pair(audio, sample_rate, midi)
    shifted_pitches = [
        note.pitch + parameters.semitones
        for instrument in midi.instruments
        for note in instrument.notes
    ]
    if not shifted_pitches:
        raise ValueError("MIDI must contain at least one note")
    if (
        min(shifted_pitches) < parameters.minimum_midi_pitch
        or max(shifted_pitches) > parameters.maximum_midi_pitch
    ):
        raise ValueError(
            "pitch shift would move a label outside the configured model range; "
            "the transform refuses to drop labels"
        )
    channel_first = np.ascontiguousarray(audio.T, dtype=np.float64)
    rendered = librosa.effects.pitch_shift(
        channel_first,
        sr=sample_rate,
        n_steps=parameters.semitones,
    ).T
    if rendered.shape != audio.shape:
        raise RuntimeError(
            f"pitch shift changed audio shape: {audio.shape} -> {rendered.shape}"
        )
    rendered, peak_qc = _peak_guard(rendered)
    output_midi = _pitch_shift_midi(midi, parameters.semitones)
    provenance = _provenance_base(
        "pitch_shift_v1",
        seed,
        source_audio,
        source_midi,
        audio,
        sample_rate,
        midi,
    )
    parameter_dict = asdict(parameters)
    provenance.update(
        {
            "parameters": parameter_dict,
            "qc": {
                **peak_qc,
                "source_pitch_min": min(
                    note.pitch
                    for instrument in midi.instruments
                    for note in instrument.notes
                ),
                "source_pitch_max": max(
                    note.pitch
                    for instrument in midi.instruments
                    for note in instrument.notes
                ),
                "output_pitch_min": min(shifted_pitches),
                "output_pitch_max": max(shifted_pitches),
                "out_of_range_labels_dropped": 0,
            },
            "invariants": {
                "audio_sample_count_preserved": True,
                "audio_channel_count_preserved": True,
                "midi_note_count_preserved": True,
                "midi_note_times_preserved": True,
                "integral_semitone_shift_applied_to_audio_and_midi": True,
                "rendered_audio_finite": True,
                "peak_guard_is_explicit": True,
            },
        }
    )
    _attach_plan_config(provenance, parameters=parameter_dict, operations=[])
    return _stage_and_publish_bundle(
        output_audio_path=target_audio,
        output_midi_path=target_midi,
        provenance_path=target_provenance,
        output_audio=rendered,
        sample_rate=sample_rate,
        audio_info=audio_info,
        output_midi=output_midi,
        provenance=provenance,
    )


def time_stretch_v1(
    audio_path: os.PathLike,
    midi_path: os.PathLike,
    output_audio_path: os.PathLike,
    output_midi_path: os.PathLike,
    *,
    seed: int,
    parameters: TimeStretchParameters,
    provenance_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Time-stretch audio and map MIDI by the realized sample-duration ratio."""

    _validate_seed(seed)
    _validate_time_stretch(parameters)
    (
        source_audio,
        source_midi,
        target_audio,
        target_midi,
        target_provenance,
    ) = _output_paths(
        audio_path,
        midi_path,
        output_audio_path,
        output_midi_path,
        provenance_path,
    )
    audio, sample_rate, audio_info, midi = _load_pair(source_audio, source_midi)
    _validate_loaded_pair(audio, sample_rate, midi)
    source_samples = int(audio.shape[0])
    requested_output_samples = int(round(source_samples / parameters.rate))
    if requested_output_samples <= 0:
        raise ValueError("time-stretch rate yields an empty output")
    channel_first = np.ascontiguousarray(audio.T, dtype=np.float64)
    rendered = librosa.effects.time_stretch(
        channel_first,
        rate=float(parameters.rate),
        n_fft=2048,
        hop_length=512,
    ).T
    if rendered.shape[0] != requested_output_samples:
        raise RuntimeError(
            "time-stretch implementation returned an unexpected sample count: "
            f"expected {requested_output_samples}, found {rendered.shape[0]}"
        )
    if rendered.shape[1] != audio.shape[1]:
        raise RuntimeError("time stretch changed the audio channel count")
    rendered, peak_qc = _peak_guard(rendered)
    realized_time_scale = rendered.shape[0] / source_samples
    output_midi = _copy_midi_with_time_map(
        midi,
        map_time=lambda value: float(value) * realized_time_scale,
    )
    parameter_dict = asdict(parameters)
    provenance = _provenance_base(
        "time_stretch_v1",
        seed,
        source_audio,
        source_midi,
        audio,
        sample_rate,
        midi,
    )
    provenance.update(
        {
            "parameters": parameter_dict,
            "qc": {
                **peak_qc,
                "source_audio_samples": source_samples,
                "output_audio_samples": int(rendered.shape[0]),
                "nominal_time_scale": 1.0 / parameters.rate,
                "realized_time_scale": realized_time_scale,
                "realized_duration_seconds": rendered.shape[0] / sample_rate,
                "annotation_time_map": "source_seconds * realized_time_scale",
                "annotation_text_rounding_used": False,
            },
            "invariants": {
                "audio_channel_count_preserved": True,
                "midi_note_count_preserved": True,
                "every_midi_boundary_uses_one_realized_sample_domain_time_map": True,
                "rendered_audio_finite": True,
                "peak_guard_is_explicit": True,
            },
        }
    )
    _attach_plan_config(provenance, parameters=parameter_dict, operations=[])
    return _stage_and_publish_bundle(
        output_audio_path=target_audio,
        output_midi_path=target_midi,
        provenance_path=target_provenance,
        output_audio=rendered,
        sample_rate=sample_rate,
        audio_info=audio_info,
        output_midi=output_midi,
        provenance=provenance,
    )


__all__ = [
    "ArchivalNoiseParameters",
    "FractionalDetuningParameters",
    "GainChorusParameters",
    "NoiseSNRParameters",
    "PitchShiftParameters",
    "ReverbFiltersParameters",
    "TimeStretchParameters",
    "archival_noise_v1",
    "fractional_detuning_v1",
    "gain_chorus_v1",
    "noise_snr_v1",
    "pitch_shift_v1",
    "reverb_filters_v1",
    "time_stretch_v1",
]
