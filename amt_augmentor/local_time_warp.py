"""Continuous, deterministic local time warping for paired AMT examples.

``local_time_warp_v1`` is deliberately an opt-in transform.  It renders one
full-recording, nonuniform phase-vocoder pass; it does not cut, repeat, loop,
or splice audio chunks.  The exact source-sample to target-sample map used for
the audio and every MIDI time boundary is included in its provenance.
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
LOCAL_TIME_WARP_MINIMUM_RATE = 0.94
LOCAL_TIME_WARP_MAXIMUM_RATE = 1.06
LOCAL_TIME_WARP_ANCHOR_SECONDS = 2.0
LOCAL_TIME_WARP_CORRELATION_SECONDS = 12.0
LOCAL_TIME_WARP_N_FFT = 2048
LOCAL_TIME_WARP_HOP_LENGTH = 512


@dataclass(frozen=True)
class LocalTimeWarpParameters:
    """Fixed safe operating envelope for :func:`local_time_warp_v1`.

    These defaults are intentionally explicit in the API and provenance.  The
    transform accepts no wider rate envelope: a local warp is useful only when
    it remains a subtle, label-preserving perturbation.
    """

    minimum_rate: float = LOCAL_TIME_WARP_MINIMUM_RATE
    maximum_rate: float = LOCAL_TIME_WARP_MAXIMUM_RATE
    anchor_seconds: float = LOCAL_TIME_WARP_ANCHOR_SECONDS
    correlation_seconds: float = LOCAL_TIME_WARP_CORRELATION_SECONDS
    n_fft: int = LOCAL_TIME_WARP_N_FFT
    hop_length: int = LOCAL_TIME_WARP_HOP_LENGTH


def _require_builtin_number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite built-in int or float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_parameters(parameters: LocalTimeWarpParameters) -> None:
    minimum_rate = _require_builtin_number(parameters.minimum_rate, "minimum_rate")
    maximum_rate = _require_builtin_number(parameters.maximum_rate, "maximum_rate")
    anchor_seconds = _require_builtin_number(
        parameters.anchor_seconds, "anchor_seconds"
    )
    correlation_seconds = _require_builtin_number(
        parameters.correlation_seconds,
        "correlation_seconds",
    )
    if type(parameters.n_fft) is not int or type(parameters.hop_length) is not int:
        raise TypeError("n_fft and hop_length must be built-in ints")
    if (
        minimum_rate != LOCAL_TIME_WARP_MINIMUM_RATE
        or maximum_rate != LOCAL_TIME_WARP_MAXIMUM_RATE
        or anchor_seconds != LOCAL_TIME_WARP_ANCHOR_SECONDS
        or correlation_seconds != LOCAL_TIME_WARP_CORRELATION_SECONDS
        or parameters.n_fft != LOCAL_TIME_WARP_N_FFT
        or parameters.hop_length != LOCAL_TIME_WARP_HOP_LENGTH
    ):
        raise ValueError(
            "local_time_warp_v1 has a fixed 0.94--1.06 rate envelope, "
            "2 s anchors, 12 s correlation, n_fft=2048, and hop_length=512"
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
    if instrument.control_changes or instrument.pitch_bends:
        raise ValueError("MIDI controls and pitch bends are not supported")
    if not instrument.notes:
        raise ValueError("MIDI must contain at least one note")
    audio_duration = audio.shape[0] / sample_rate
    for index, note in enumerate(instrument.notes):
        if (
            not math.isfinite(float(note.start))
            or not math.isfinite(float(note.end))
            or note.start < 0.0
            or note.end <= note.start
        ):
            raise ValueError(f"MIDI note {index} has an invalid time interval")
        if not 0 <= int(note.pitch) <= 127 or not 1 <= int(note.velocity) <= 127:
            raise ValueError(f"MIDI note {index} has invalid MIDI values")
        if float(note.end) > audio_duration:
            raise ValueError(
                f"MIDI note {index} ends beyond the audio duration "
                f"({float(note.end):.9f} > {audio_duration:.9f} seconds)"
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
    return source_audio, source_midi, target_audio, target_midi, target_provenance


def _peak_guard(audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    if not np.isfinite(audio).all():
        raise RuntimeError("rendered audio contains NaN or infinite samples")
    peak_before = float(np.max(np.abs(audio), initial=0.0))
    if peak_before == 0.0:
        raise RuntimeError("rendered audio is completely silent")
    applied_gain = min(1.0, PEAK_LIMIT / peak_before)
    guarded = audio * applied_gain
    return guarded, {
        "peak_before_guard": peak_before,
        "peak_limit": PEAK_LIMIT,
        "peak_guard_linear_gain": applied_gain,
        "peak_after_guard": float(np.max(np.abs(guarded), initial=0.0)),
    }


def _source_anchor_samples(sample_count: int, sample_rate: int) -> np.ndarray:
    """Return exact source-domain anchors, including 0 and the duration N."""

    stride = round(LOCAL_TIME_WARP_ANCHOR_SECONDS * sample_rate)
    if stride <= 0:
        raise RuntimeError("local time-warp anchor stride is not positive")
    anchors = np.arange(0, sample_count, stride, dtype=np.int64)
    if anchors.size == 0 or anchors[0] != 0:
        anchors = np.insert(anchors, 0, 0)
    if anchors[-1] != sample_count:
        anchors = np.append(anchors, sample_count)
    return anchors


def _gaussian_smooth_seeded_values(
    values: np.ndarray, sigma_anchors: float
) -> np.ndarray:
    """Smooth one finite vector with a deterministic reflected Gaussian kernel."""

    radius = max(1, math.ceil(3.0 * sigma_anchors))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(positions / sigma_anchors))
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _build_sample_map(
    sample_count: int,
    sample_rate: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build an exact monotonic source-sample -> target-sample knot map.

    The source anchors are two seconds apart.  Segment rates are seeded,
    Gaussian-smoothed at a six-anchor sigma (approximately 12 seconds), then
    constrained to have a source-duration weighted mean of exactly one.  Thus
    both sample-map endpoints are exactly ``(0, 0)`` and ``(N, N)``.
    """

    source = _source_anchor_samples(sample_count, sample_rate).astype(np.float64)
    source_lengths = np.diff(source)
    segment_count = int(source_lengths.size)
    rates = np.ones(segment_count, dtype=np.float64)
    sigma_anchors = LOCAL_TIME_WARP_CORRELATION_SECONDS / LOCAL_TIME_WARP_ANCHOR_SECONDS

    # Two endpoint segments are held at exactly one.  At least two interior
    # degrees of freedom are needed to add a nonzero, endpoint-fixed, zero-mean
    # local warp; shorter records therefore correctly reduce to the identity.
    if segment_count >= 4:
        rng = np.random.default_rng(seed)
        variation = _gaussian_smooth_seeded_values(
            rng.standard_normal(segment_count),
            sigma_anchors,
        )
        endpoint_window = np.sin(
            np.pi * np.arange(segment_count, dtype=np.float64) / (segment_count - 1)
        )
        endpoint_window[0] = 0.0
        endpoint_window[-1] = 0.0
        variation *= endpoint_window
        correction_denominator = float(np.dot(endpoint_window, source_lengths))
        if correction_denominator <= 0.0:
            raise RuntimeError("local time-warp cannot construct a zero-mean rate")
        variation -= (
            float(np.dot(variation, source_lengths))
            / correction_denominator
            * endpoint_window
        )
        variation[0] = 0.0
        variation[-1] = 0.0
        maximum_deviation = float(np.max(np.abs(variation), initial=0.0))
        if maximum_deviation > 0.0:
            variation *= (LOCAL_TIME_WARP_MAXIMUM_RATE - 1.0) / maximum_deviation
            rates += variation

    if (
        not np.isfinite(rates).all()
        or rates[0] != 1.0
        or rates[-1] != 1.0
        or np.min(rates) < LOCAL_TIME_WARP_MINIMUM_RATE - 1e-12
        or np.max(rates) > LOCAL_TIME_WARP_MAXIMUM_RATE + 1e-12
    ):
        raise RuntimeError("local time-warp constructed an invalid local-rate map")

    target = np.empty_like(source)
    target[0] = 0.0
    target[1:] = np.cumsum(source_lengths * rates, dtype=np.float64)
    # The zero-mean construction establishes N mathematically; set the final
    # floating representation explicitly so the published map has exact,
    # auditable integer-sample duration endpoints.  Pinning the preceding knot
    # as well keeps the final piecewise-linear segment's rate exactly one, not
    # merely one within cumulative floating-point rounding.
    target[-1] = float(sample_count)
    if target.size > 2:
        target[-2] = float(sample_count - source_lengths[-1])
    if (
        not np.all(np.diff(source) > 0.0)
        or not np.all(np.diff(target) > 0.0)
        or source[0] != 0.0
        or source[-1] != float(sample_count)
        or target[0] != 0.0
        or target[-1] != float(sample_count)
    ):
        raise RuntimeError("local time-warp sample map is not strictly monotonic")

    realized_rates = np.diff(target) / source_lengths
    if (
        np.min(realized_rates) < LOCAL_TIME_WARP_MINIMUM_RATE - 1e-12
        or np.max(realized_rates) > LOCAL_TIME_WARP_MAXIMUM_RATE + 1e-12
        or abs(realized_rates[0] - 1.0) > 1e-12
        or abs(realized_rates[-1] - 1.0) > 1e-12
    ):
        raise RuntimeError(
            "published local time-warp sample map violates its rate bounds"
        )
    return (
        source,
        target,
        realized_rates,
        {
            "direction": "source_sample_to_target_sample",
            "interpolation": "piecewise_linear",
            "rate_definition": (
                "target_sample_delta / source_sample_delta; values above one "
                "locally lengthen duration and values below one locally shorten it"
            ),
            "source_sample_knots": [int(value) for value in source],
            "target_sample_knots": [float(value) for value in target],
            "segment_target_per_source_rates": [
                float(value) for value in realized_rates
            ],
            "minimum_rate": float(np.min(realized_rates)),
            "maximum_rate": float(np.max(realized_rates)),
            "endpoint_rates": [float(realized_rates[0]), float(realized_rates[-1])],
            "anchor_seconds": LOCAL_TIME_WARP_ANCHOR_SECONDS,
            "correlation_seconds": LOCAL_TIME_WARP_CORRELATION_SECONDS,
            "gaussian_sigma_anchors": sigma_anchors,
            "endpoint_pairs": [[0, 0], [sample_count, sample_count]],
        },
    )


def _phase_vocoder_at_time_steps(
    spectrum: np.ndarray,
    time_steps: np.ndarray,
    *,
    hop_length: int,
) -> np.ndarray:
    """Adapt librosa's phase-vocoder recurrence to explicit frame positions."""

    if spectrum.ndim < 2:
        raise ValueError("phase-vocoder spectrum must have frequency and time axes")
    if time_steps.ndim != 1 or time_steps.size == 0:
        raise ValueError("phase-vocoder time steps must be a non-empty vector")
    if not np.isfinite(time_steps).all() or not np.all(np.diff(time_steps) > 0.0):
        raise ValueError(
            "phase-vocoder time steps must be finite and strictly increasing"
        )
    source_frames = spectrum.shape[-1]
    if source_frames < 2:
        raise ValueError("phase-vocoder input must contain at least two STFT frames")
    if time_steps[0] != 0.0 or time_steps[-1] > source_frames - 1:
        raise ValueError("phase-vocoder time steps are outside the source frame map")

    phase_advance = np.linspace(
        0.0,
        np.pi * hop_length,
        spectrum.shape[-2],
        dtype=np.float64,
    )
    phase_accumulator = np.angle(spectrum[..., 0])
    stretched = np.zeros(
        spectrum.shape[:-1] + (time_steps.size,),
        dtype=np.complex128,
    )
    padded = np.pad(
        spectrum,
        [(0, 0)] * (spectrum.ndim - 1) + [(0, 2)],
        mode="constant",
    )

    # This is one continuous phase accumulator across all requested output
    # frames.  It deliberately has no per-segment reset, repetition, or splice.
    for output_frame, source_frame in enumerate(time_steps):
        frame_index = int(source_frame)
        columns = padded[..., frame_index : frame_index + 2]
        alpha = source_frame - frame_index
        magnitude = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(
            columns[..., 1]
        )
        stretched[..., output_frame] = magnitude * np.exp(1.0j * phase_accumulator)
        phase_delta = np.angle(columns[..., 1]) - np.angle(columns[..., 0])
        phase_delta -= phase_advance
        phase_delta -= 2.0 * np.pi * np.round(phase_delta / (2.0 * np.pi))
        phase_accumulator += phase_advance + phase_delta
    return stretched


def _render_continuous_nonuniform_phase_vocoder(
    audio: np.ndarray,
    source_sample_knots: np.ndarray,
    target_sample_knots: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render all channels together from the inverse of one sample map."""

    sample_count = int(audio.shape[0])
    if sample_count < n_fft:
        raise ValueError(
            f"local_time_warp_v1 requires at least n_fft={n_fft} input samples"
        )
    channel_first = np.ascontiguousarray(audio.T, dtype=np.float64)
    spectrum = librosa.stft(
        channel_first,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
    )
    target_frame_samples = np.arange(spectrum.shape[-1], dtype=np.float64) * hop_length
    source_frame_samples = np.interp(
        target_frame_samples,
        target_sample_knots,
        source_sample_knots,
    )
    time_steps = source_frame_samples / hop_length
    # Centered STFT analysis has frames only through floor(N / hop_length).
    # A legal sample map can move the final target-frame centre slightly beyond
    # that last source-frame centre, so use the terminal analysis frame rather
    # than reading outside the one full-recording spectrogram.
    time_steps = np.minimum(time_steps, float(spectrum.shape[-1] - 1))
    time_steps[0] = 0.0
    if not np.all(np.diff(time_steps) > 0.0):
        raise RuntimeError(
            "inverse local time-warp frame map is not strictly monotonic"
        )
    stretched = _phase_vocoder_at_time_steps(
        spectrum,
        time_steps,
        hop_length=hop_length,
    )
    rendered = librosa.istft(
        stretched,
        hop_length=hop_length,
        length=sample_count,
        center=True,
    ).T
    if rendered.shape != audio.shape:
        raise RuntimeError(
            "nonuniform phase-vocoder changed audio shape: "
            f"expected {audio.shape}, found {rendered.shape}"
        )
    return np.ascontiguousarray(rendered, dtype=np.float64), time_steps


def local_time_warp_v1(
    audio_path: os.PathLike,
    midi_path: os.PathLike,
    output_audio_path: os.PathLike,
    output_midi_path: os.PathLike,
    *,
    seed: int,
    parameters: Optional[LocalTimeWarpParameters] = None,
    provenance_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Apply one exact-duration continuous local audio/MIDI time warp.

    ``parameters`` intentionally defaults to the versioned safe envelope, but
    callers must opt in by selecting this transform explicitly.  The same
    source-sample to target-sample piecewise-linear map drives the nonuniform
    phase-vocoder frame positions and MIDI note onset/offset mapping.
    """

    _validate_seed(seed)
    if parameters is None:
        parameters = LocalTimeWarpParameters()
    _validate_parameters(parameters)
    source_audio, source_midi, target_audio, target_midi, target_provenance = (
        _output_paths(
            audio_path,
            midi_path,
            output_audio_path,
            output_midi_path,
            provenance_path,
        )
    )
    audio, sample_rate, audio_info, midi = _load_pair(source_audio, source_midi)
    _validate_loaded_pair(audio, sample_rate, midi)
    source_knots, target_knots, realized_rates, map_provenance = _build_sample_map(
        int(audio.shape[0]),
        sample_rate,
        seed,
    )
    rendered, time_steps = _render_continuous_nonuniform_phase_vocoder(
        audio,
        source_knots,
        target_knots,
        n_fft=parameters.n_fft,
        hop_length=parameters.hop_length,
    )
    rendered, peak_qc = _peak_guard(rendered)

    def map_midi_time(source_seconds: float) -> float:
        source_sample = float(source_seconds) * sample_rate
        if source_sample < 0.0 or source_sample > source_knots[-1]:
            raise ValueError("MIDI time is outside the source sample map")
        return float(np.interp(source_sample, source_knots, target_knots) / sample_rate)

    output_midi = _copy_midi_with_time_map(midi, map_time=map_midi_time)
    parameter_dict = asdict(parameters)
    provenance = _provenance_base(
        "local_time_warp_v1",
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
            "backend": {
                "name": "librosa_nonuniform_phase_vocoder_v1",
                "librosa_version": librosa.__version__,
                "phase_recurrence": "adapted_librosa_phase_vocoder_explicit_time_steps",
                "n_fft": parameters.n_fft,
                "hop_length": parameters.hop_length,
                "center": True,
                "channel_rendering": "joint_multichannel_stft_axes",
                "transient_handling": (
                    "none; librosa reference phase-vocoder recurrence"
                ),
                "render_topology": "one full-recording STFT/ISTFT pass",
            },
            "sample_map": map_provenance,
            "qc": {
                **peak_qc,
                "source_audio_samples": int(audio.shape[0]),
                "output_audio_samples": int(rendered.shape[0]),
                "source_duration_seconds": audio.shape[0] / sample_rate,
                "output_duration_seconds": rendered.shape[0] / sample_rate,
                "frame_map_source_positions": [float(value) for value in time_steps],
                "frame_map_target_frame_count": int(time_steps.size),
                "exact_duration_preserved": rendered.shape[0] == audio.shape[0],
                "no_chunking_or_looping": True,
                "no_segment_repetition_or_splicing": True,
                "annotation_time_map": (
                    "piecewise-linear source_sample_to_target_sample"
                ),
            },
            "invariants": {
                "audio_sample_count_preserved_exactly": rendered.shape[0]
                == audio.shape[0],
                "audio_channel_count_preserved": rendered.shape[1] == audio.shape[1],
                "sample_map_strictly_monotonic": True,
                "sample_map_exact_endpoints": True,
                "sample_map_endpoint_rates_are_one": bool(
                    abs(realized_rates[0] - 1.0) <= 1e-12
                    and abs(realized_rates[-1] - 1.0) <= 1e-12
                ),
                "every_midi_boundary_uses_exact_piecewise_linear_sample_map": True,
                "midi_note_count_preserved": True,
                "rendered_audio_finite": True,
                "single_continuous_phase_vocoder_render": True,
                "chunking_looping_or_splicing_used": False,
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


__all__ = ["LocalTimeWarpParameters", "local_time_warp_v1"]
