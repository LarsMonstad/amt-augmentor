"""Private I/O, provenance, and publication helpers for paired transforms.

This module contains no augmentation method.  It centralizes the mechanical
parts shared by the supported conventional audio/MIDI transforms: validation,
annotation copying, deterministic provenance, and fail-closed bundle
publication.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pretty_midi
import soundfile as sf

PROVENANCE_SCHEMA_VERSION = 1
ANNOTATION_MIDI_RESOLUTION = 9600
ANNOTATION_MIDI_TEMPO = 120.0
PUBLISHED_FILE_MODE = 0o640
SUPPORTED_OUTPUT_AUDIO_SUFFIXES = {".flac", ".wav"}
SUPPORTED_OUTPUT_MIDI_SUFFIXES = {".mid", ".midi"}


def _validate_seed(seed: int) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be a nonnegative built-in int")
    if seed < 0:
        raise ValueError("seed must be nonnegative")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version() -> str:
    from amt_augmentor import __version__

    return __version__


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _audio_format_for_output(source_info: sf.SoundFile, output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix == ".wav":
        return "WAV"
    if suffix == ".flac":
        return "FLAC"
    raise ValueError(
        "Unsupported output audio suffix "
        f"{output_path.suffix!r}; expected one of .wav or .flac"
    )


def _validate_output_paths(
    source_audio_path: Path,
    source_midi_path: Path,
    target_audio_path: Path,
    target_midi_path: Path,
    provenance_path: Path,
) -> None:
    if target_audio_path.suffix.lower() not in SUPPORTED_OUTPUT_AUDIO_SUFFIXES:
        raise ValueError(
            "Unsupported output audio suffix "
            f"{target_audio_path.suffix!r}; expected one of .wav or .flac"
        )
    if target_midi_path.suffix.lower() not in SUPPORTED_OUTPUT_MIDI_SUFFIXES:
        raise ValueError(
            "Unsupported output MIDI suffix "
            f"{target_midi_path.suffix!r}; expected one of .mid or .midi"
        )
    sources = {
        source_audio_path.resolve(strict=False),
        source_midi_path.resolve(strict=False),
    }
    targets = {
        target_audio_path.resolve(strict=False),
        target_midi_path.resolve(strict=False),
        provenance_path.resolve(strict=False),
    }
    if sources & targets:
        raise ValueError("Output paths must not overwrite either source file")
    if len(targets) != 3:
        raise ValueError("Audio, MIDI, and provenance output paths must be distinct")
    for target in (target_audio_path, target_midi_path, provenance_path):
        if os.path.lexists(str(target)):
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")


def _write_audio(
    output_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    source_info: sf.SoundFile,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = _audio_format_for_output(source_info, output_path)
    subtype = source_info.subtype
    if output_format == "FLAC" and subtype in {"FLOAT", "DOUBLE"}:
        subtype = "PCM_24"
    sf.write(
        str(output_path),
        audio,
        sample_rate,
        format=output_format,
        subtype=subtype,
    )


def _copy_midi_with_time_map(
    source: pretty_midi.PrettyMIDI,
    *,
    map_time,
) -> pretty_midi.PrettyMIDI:
    """Copy an AMT annotation MIDI through one time mapping."""

    output = pretty_midi.PrettyMIDI(
        initial_tempo=ANNOTATION_MIDI_TEMPO,
        resolution=ANNOTATION_MIDI_RESOLUTION,
    )

    for instrument in source.instruments:
        copied_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name,
        )
        for note in instrument.notes:
            copied_note = copy.deepcopy(note)
            copied_note.start = map_time(float(note.start))
            copied_note.end = map_time(float(note.end))
            copied_instrument.notes.append(copied_note)
        for pitch_bend in instrument.pitch_bends:
            copied_event = copy.deepcopy(pitch_bend)
            copied_event.time = map_time(float(pitch_bend.time))
            copied_instrument.pitch_bends.append(copied_event)
        for control_change in instrument.control_changes:
            copied_event = copy.deepcopy(control_change)
            copied_event.time = map_time(float(control_change.time))
            copied_instrument.control_changes.append(copied_event)
        output.instruments.append(copied_instrument)

    for attribute in (
        "key_signature_changes",
        "time_signature_changes",
        "lyrics",
        "text_events",
    ):
        copied_events = []
        for event in getattr(source, attribute, []):
            copied_event = copy.deepcopy(event)
            copied_event.time = map_time(float(event.time))
            copied_events.append(copied_event)
        setattr(output, attribute, copied_events)

    return output


def _write_midi(output_path: Path, midi: pretty_midi.PrettyMIDI) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))


def _midi_timing_quantization_report(
    serialized_midi_path: Path,
    expected_midi: pretty_midi.PrettyMIDI,
) -> Dict[str, Any]:
    serialized_midi = pretty_midi.PrettyMIDI(str(serialized_midi_path))
    expected_notes = [
        (instrument.program, instrument.is_drum, note)
        for instrument in expected_midi.instruments
        for note in instrument.notes
    ]
    serialized_notes = [
        (instrument.program, instrument.is_drum, note)
        for instrument in serialized_midi.instruments
        for note in instrument.notes
    ]
    if len(serialized_notes) != len(expected_notes):
        raise RuntimeError("Serialized MIDI changed the retained note count")

    errors: List[float] = []
    for expected, serialized in zip(expected_notes, serialized_notes):
        expected_program, expected_is_drum, expected_note = expected
        serialized_program, serialized_is_drum, serialized_note = serialized
        if (
            serialized_program != expected_program
            or serialized_is_drum != expected_is_drum
            or serialized_note.pitch != expected_note.pitch
            or serialized_note.velocity != expected_note.velocity
        ):
            raise RuntimeError("Serialized MIDI changed retained note attributes")
        errors.extend(
            (
                abs(float(serialized_note.start) - float(expected_note.start)),
                abs(float(serialized_note.end) - float(expected_note.end)),
            )
        )

    tick_seconds = 60.0 / ANNOTATION_MIDI_TEMPO / ANNOTATION_MIDI_RESOLUTION
    half_tick_seconds = tick_seconds / 2.0
    numeric_tolerance_seconds = 1e-9
    maximum_error_seconds = max(errors, default=0.0)
    maximum_allowed_seconds = half_tick_seconds + numeric_tolerance_seconds
    if maximum_error_seconds > maximum_allowed_seconds:
        raise RuntimeError(
            "Serialized retained-note endpoint error exceeds half an output "
            f"MIDI tick: {maximum_error_seconds} > {maximum_allowed_seconds}"
        )
    return {
        "comparison_basis": (
            "expected transformed note endpoints before MIDI serialization"
        ),
        "retained_note_endpoint_count": len(errors),
        "output_midi_tick_seconds": tick_seconds,
        "bound_ticks": 0.5,
        "numeric_tolerance_seconds": numeric_tolerance_seconds,
        "maximum_allowed_error_seconds": maximum_allowed_seconds,
        "maximum_retained_note_endpoint_error_seconds": maximum_error_seconds,
        "maximum_retained_note_endpoint_error_ticks": (
            maximum_error_seconds / tick_seconds
        ),
    }


def _load_pair(
    audio_path: Path,
    midi_path: Path,
) -> Tuple[np.ndarray, int, sf.SoundFile, pretty_midi.PrettyMIDI]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    audio, sample_rate = sf.read(str(audio_path), dtype="float64", always_2d=True)
    info = sf.info(str(audio_path))
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    return audio, int(sample_rate), info, midi


def _provenance_base(
    transform: str,
    seed: int,
    audio_path: Path,
    midi_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    midi: pretty_midi.PrettyMIDI,
) -> Dict[str, Any]:
    note_count = sum(len(instrument.notes) for instrument in midi.instruments)
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "tool_version": _tool_version(),
        "transform": transform,
        "seed": seed,
        "source": {
            "audio_name": audio_path.name,
            "midi_name": midi_path.name,
            "audio_sha256": _sha256_file(audio_path),
            "midi_sha256": _sha256_file(midi_path),
            "sample_rate": sample_rate,
            "audio_samples": int(audio.shape[0]),
            "channels": int(audio.shape[1]),
            "midi_note_count": note_count,
        },
        "midi_serialization": {
            "purpose": "time-aligned AMT annotation",
            "constant_tempo_bpm": ANNOTATION_MIDI_TEMPO,
            "ticks_per_beat": ANNOTATION_MIDI_RESOLUTION,
        },
    }


def _attach_plan_config(
    provenance: Dict[str, Any],
    *,
    parameters: Dict[str, Any],
    operations: Sequence[Dict[str, Any]],
) -> None:
    plan_config = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "tool_version": provenance["tool_version"],
        "transform": provenance["transform"],
        "seed": provenance["seed"],
        "source_audio_sha256": provenance["source"]["audio_sha256"],
        "source_midi_sha256": provenance["source"]["midi_sha256"],
        "parameters": parameters,
        "operations": list(operations),
    }
    provenance["plan_config"] = plan_config
    provenance["plan_config_sha256"] = _canonical_sha256(plan_config)


def _new_stage_path(target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, path = tempfile.mkstemp(
        dir=str(target_path.parent),
        prefix=f".{target_path.name}.staging-",
        suffix=target_path.suffix,
    )
    os.close(descriptor)
    return Path(path)


def _write_provenance(path: Path, provenance: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_path(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(directory), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directories(directories: Sequence[Path]) -> None:
    seen: Set[Path] = set()
    for directory in directories:
        resolved = directory.resolve(strict=False)
        if resolved in seen:
            continue
        _fsync_directory(resolved)
        seen.add(resolved)


def _set_published_mode(path: Path) -> None:
    os.chmod(str(path), PUBLISHED_FILE_MODE)


def _publish_stage(stage_path: Path, target_path: Path) -> None:
    """Publish without overwriting an output created by another process."""

    os.link(str(stage_path), str(target_path))


def _remove_published_if_ours(stage_path: Path, target_path: Path) -> None:
    try:
        if target_path.exists() and os.path.samefile(stage_path, target_path):
            target_path.unlink()
    except FileNotFoundError:
        pass


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _stage_and_publish_bundle(
    *,
    output_audio_path: Path,
    output_midi_path: Path,
    provenance_path: Path,
    output_audio: np.ndarray,
    sample_rate: int,
    audio_info: sf.SoundFile,
    output_midi: pretty_midi.PrettyMIDI,
    output_midi_bytes: Optional[bytes] = None,
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    """Stage a paired bundle and publish its provenance sidecar last."""

    stages: Dict[Path, Path] = {}
    published: List[Tuple[Path, Path]] = []
    try:
        for target in (output_audio_path, output_midi_path, provenance_path):
            stages[target] = _new_stage_path(target)

        _write_audio(stages[output_audio_path], output_audio, sample_rate, audio_info)
        if output_midi_bytes is None:
            _write_midi(stages[output_midi_path], output_midi)
        else:
            with stages[output_midi_path].open("wb") as handle:
                handle.write(output_midi_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        provenance["midi_timing_quantization"] = _midi_timing_quantization_report(
            stages[output_midi_path],
            output_midi,
        )

        output_note_count = sum(
            len(instrument.notes) for instrument in output_midi.instruments
        )
        provenance["output"] = {
            "audio_name": output_audio_path.name,
            "midi_name": output_midi_path.name,
            "audio_sha256": _sha256_file(stages[output_audio_path]),
            "midi_sha256": _sha256_file(stages[output_midi_path]),
            "audio_samples": int(output_audio.shape[0]),
            "midi_note_count": int(output_note_count),
        }
        provenance["publication"] = {
            "completion_marker": provenance_path.name,
            "completion_rule": (
                "The audio/MIDI pair is complete only when this provenance "
                "sidecar exists and its hashes match."
            ),
            "publish_order": ["audio", "midi", "provenance"],
            "file_mode_octal": "0640",
            "file_mode_rationale": (
                "Owner read/write and group read access; no world access."
            ),
        }
        _write_provenance(stages[provenance_path], provenance)

        for stage in stages.values():
            _set_published_mode(stage)
            _fsync_path(stage)

        for target in (output_audio_path, output_midi_path):
            _publish_stage(stages[target], target)
            published.append((stages[target], target))
        _fsync_directories([output_audio_path.parent, output_midi_path.parent])

        _publish_stage(stages[provenance_path], provenance_path)
        published.append((stages[provenance_path], provenance_path))
        _fsync_directories([provenance_path.parent])
        return provenance
    except Exception:
        for stage, target in reversed(published):
            _remove_published_if_ours(stage, target)
        raise
    finally:
        for stage in stages.values():
            _safe_unlink(stage)
