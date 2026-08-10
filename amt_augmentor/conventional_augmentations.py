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
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
class NoiseSNRParameters:
    """Parameters for :func:`noise_snr_v1`."""

    target_snr_db: float


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
    "GainChorusParameters",
    "NoiseSNRParameters",
    "PitchShiftParameters",
    "ReverbFiltersParameters",
    "TimeStretchParameters",
    "gain_chorus_v1",
    "noise_snr_v1",
    "pitch_shift_v1",
    "reverb_filters_v1",
    "time_stretch_v1",
]
