"""Deterministic conventional augmentation materialization for Galdr.

The adapter consumes Galdr's hash-pinned clean-dataset provenance, selects
training sources only, and produces two conditions:

``O`` contains one unchanged audio view paired with a derivative annotation
whose same-pitch overlaps have been normalized; ``C`` contains that same view
plus one view from each supported conventional transform family.

Planning and rendering are separate.  A plan contains every seed and parameter
value, so materialization performs no random choice.  Canonical media are read
and hash-checked but never modified.  A materialization is usable only after
its top-level completion report has been published.
"""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pretty_midi
import soundfile as sf

from amt_augmentor import __version__
from amt_augmentor._paired_io import (
    ANNOTATION_MIDI_RESOLUTION,
    ANNOTATION_MIDI_TEMPO,
)
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

PLAN_SCHEMA_VERSION = 1
PLAN_KIND = "galdr_conventional_campaign_plan"
PLAN_ALGORITHM = "galdr-conventional/plan/v1"
REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "galdr_conventional_campaign_materialization"
LINEAGE_SCHEMA_VERSION = 1
LINEAGE_KIND = "galdr_materialized_view_lineage"
CONDITION_IDS = ("O", "C")
ORIGINAL_VIEW_SLOT = "00-original"
VIEW_SLOTS = (
    ("10-gain-chorus", "gain_chorus_v1"),
    ("20-noise-snr", "noise_snr_v1"),
    ("30-reverb-filters", "reverb_filters_v1"),
    ("40-pitch-shift", "pitch_shift_v1"),
    ("50-time-stretch", "time_stretch_v1"),
)
GLOBAL_SEED_MAXIMUM = 2**63 - 1
MAXIMUM_WORKERS = 16
PUBLISHED_FILE_MODE = 0o640
PUBLISHED_DIRECTORY_MODE = 0o750
RANDOM_STREAM_ALGORITHM = "galdr-sigma2/conventional-stream/v1"
NORMALIZATION_POLICY = "same_pitch_overlap_truncate_previous_v1"
SUPPORTED_AUDIO_SUFFIXES = {".wav"}

PARAMETER_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "gain_chorus_v1": {
        "gain_db": [-6.0, -3.0, 0.0, 3.0, 6.0],
        "chorus_depth": [0.10, 0.15, 0.20, 0.25, 0.30],
        "chorus_rate_hz": [0.4, 0.6, 0.8, 1.0, 1.2],
        "chorus_centre_delay_ms": [7.0],
        "chorus_feedback": [0.2],
        "chorus_mix": [0.25],
    },
    "noise_snr_v1": {
        "target_snr_db": [20.0, 24.0, 28.0, 32.0],
    },
    "reverb_filters_v1": {
        "room_size": [0.15, 0.25, 0.35, 0.45],
        "wet_level": [0.10, 0.15, 0.20, 0.25],
        "dry_level": [0.75, 0.80, 0.85, 0.90],
        "highpass_hz": [30.0, 40.0, 60.0, 80.0],
        "lowpass_hz": [9000.0, 11000.0, 13000.0, 15000.0],
    },
    "pitch_shift_v1": {
        "semitones": [-2, -1, 1, 2],
        "minimum_midi_pitch": [21],
        "maximum_midi_pitch": [108],
    },
    "time_stretch_v1": {
        "rate": [0.90, 0.95, 1.05, 1.10],
    },
}


class CampaignError(ValueError):
    """An input, plan, or artifact violates the campaign contract."""


def _canonical_json_bytes(document: Any) -> bytes:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(document: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(document)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _stable_file_sha256(
    path: Path,
    expected: str,
    label: str,
    *,
    expected_bytes: Optional[int] = None,
) -> None:
    expected = _require_digest(expected, f"{label} expected digest")
    before = path.stat()
    if expected_bytes is not None and before.st_size != expected_bytes:
        raise CampaignError(
            f"{label} size mismatch: expected {expected_bytes}, found {before.st_size}"
        )
    observed = _sha256_file(path)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise CampaignError(f"{label} changed while it was being hashed")
    if observed != expected:
        raise CampaignError(
            f"{label} SHA-256 mismatch: expected {expected}, found {observed}"
        )


def _source_tree_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_versions() -> Dict[str, str]:
    packages = {}
    for distribution in (
        "librosa",
        "numpy",
        "pedalboard",
        "pretty-midi-bfm",
        "soundfile",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation().lower(),
        "python_executable_basename": Path(sys.executable).name,
        "libsndfile": str(sf.__libsndfile_version__),
        **packages,
    }


def _require_global_seed(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= GLOBAL_SEED_MAXIMUM:
        raise CampaignError(
            f"global_seed must be a built-in int from 0 to {GLOBAL_SEED_MAXIMUM}"
        )
    return value


def _require_workers(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAXIMUM_WORKERS:
        raise CampaignError(
            f"workers must be a built-in int from 1 to {MAXIMUM_WORKERS}"
        )
    return value


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CampaignError(f"{label} must be a non-empty trimmed string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise CampaignError(f"{label} must be a normalized relative POSIX path")
    return value


def _require_identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in ("\n", "\r", "\0"))
    ):
        raise CampaignError(f"{label} must be a non-empty, trimmed identifier")
    return value


def _absolute_lexical(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise CampaignError(f"{label} must not contain symbolic links")


def _existing_regular_file(path: Path, label: str) -> Path:
    _reject_symlink_components(path, label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CampaignError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise CampaignError(f"{label} must be a regular file")
    return resolved


def _existing_directory(path: Path, label: str) -> Path:
    _reject_symlink_components(path, label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CampaignError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise CampaignError(f"{label} must be a directory")
    return resolved


def _prepare_new_path(path: Path, label: str) -> Path:
    _reject_symlink_components(path.parent, f"{label} parent")
    if os.path.lexists(str(path)):
        raise CampaignError(f"refusing to overwrite {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent, f"{label} parent")
    return path.absolute()


def _resolve_input_media(root: Path, relative: str, label: str) -> Path:
    relative = _safe_relative_path(relative, label)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    _reject_symlink_components(candidate, label)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise CampaignError(f"{label} is not a contained regular file") from exc
    if not resolved.is_file():
        raise CampaignError(f"{label} is not a regular file")
    return resolved


def _load_json_file(path: Path, label: str) -> Dict[str, Any]:
    path = _existing_regular_file(path, label)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise CampaignError(f"{label} must contain a JSON object")
    return document


def _load_clean_provenance(path: Path, expected_sha256: str) -> Dict[str, Any]:
    resolved = _existing_regular_file(path, "clean dataset provenance")
    _stable_file_sha256(
        resolved,
        expected_sha256,
        "clean dataset provenance",
    )
    document = _load_json_file(resolved, "clean dataset provenance")
    required = {
        "schema_version": 2,
        "kind": "galdr_clean_original_index_bundle",
        "original_only": True,
        "augmentation_views": 0,
        "canonical_media_mutated": False,
    }
    for field, expected in required.items():
        if document.get(field) != expected:
            raise CampaignError(
                f"clean dataset provenance {field!r} must equal {expected!r}"
            )
    _require_digest(document.get("dataset_identity_sha256"), "dataset identity")
    normalization = document.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("policy") != NORMALIZATION_POLICY:
        raise CampaignError("clean dataset normalization policy is unsupported")
    if normalization.get("canonical_raw_midi_unchanged") is not True:
        raise CampaignError("clean provenance does not protect canonical MIDI")
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CampaignError("clean dataset provenance has no source records")
    return document


def _normalization_values(
    midi_path: Path,
    *,
    duration_seconds: float,
) -> Tuple[pretty_midi.PrettyMIDI, List[Dict[str, Any]], Dict[str, Any]]:
    try:
        raw = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        raise CampaignError(f"MIDI is unreadable: {midi_path}") from exc
    if len(raw.instruments) != 1:
        raise CampaignError("campaign MIDI must contain exactly one instrument")
    instrument = raw.instruments[0]
    if instrument.is_drum or instrument.control_changes or instrument.pitch_bends:
        raise CampaignError(
            "campaign MIDI must be a non-drum track without controls or pitch bends"
        )
    if not instrument.notes:
        raise CampaignError("campaign MIDI must contain at least one note")
    raw_notes = sorted(
        instrument.notes,
        key=lambda note: (note.start, note.end, note.pitch),
    )
    working: List[Dict[str, Any]] = []
    latest_by_pitch: Dict[int, int] = {}
    adjusted = 0
    for index, note in enumerate(raw_notes):
        if (
            not math.isfinite(float(note.start))
            or not math.isfinite(float(note.end))
            or note.start < 0
            or note.end <= note.start
        ):
            raise CampaignError(f"raw MIDI note {index} has an invalid interval")
        if note.end > duration_seconds + 1e-9:
            raise CampaignError(f"raw MIDI note {index} ends after source audio")
        if not 0 <= int(note.pitch) <= 127 or not 1 <= int(note.velocity) <= 127:
            raise CampaignError(f"raw MIDI note {index} has invalid MIDI values")
        previous_index = latest_by_pitch.get(int(note.pitch))
        if previous_index is not None:
            previous = working[previous_index]
            if float(previous["end_seconds"]) > float(note.start):
                previous["end_seconds"] = float(note.start)
                adjusted += 1
        latest_by_pitch[int(note.pitch)] = len(working)
        working.append(
            {
                "start_seconds": float(note.start),
                "end_seconds": float(note.end),
                "pitch": int(note.pitch),
                "velocity": int(note.velocity),
            }
        )
    values = [
        note
        for note in working
        if float(note["start_seconds"]) < float(note["end_seconds"])
    ]
    observed = {
        "raw_note_count": len(raw_notes),
        "adjusted_previous_offsets": adjusted,
        "removed_nonpositive_after_truncation": len(working) - len(values),
        "normalized_note_count": len(values),
        "annotation_semantic_sha256": _canonical_sha256(values),
    }
    normalized = pretty_midi.PrettyMIDI(
        initial_tempo=ANNOTATION_MIDI_TEMPO,
        resolution=ANNOTATION_MIDI_RESOLUTION,
    )
    output_instrument = pretty_midi.Instrument(
        program=instrument.program,
        is_drum=False,
        name=instrument.name,
    )
    for note in values:
        output_instrument.notes.append(
            pretty_midi.Note(
                velocity=int(note["velocity"]),
                pitch=int(note["pitch"]),
                start=float(note["start_seconds"]),
                end=float(note["end_seconds"]),
            )
        )
    normalized.instruments.append(output_instrument)
    return normalized, values, observed


def _validate_normalization_report(source: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    for field, value in observed.items():
        if source.get(field) != value:
            raise CampaignError(
                f"source {source.get('source_id')!r} normalization mismatch for "
                f"{field}: provenance={source.get(field)!r}, observed={value!r}"
            )


def _digest_choice(parts: Sequence[Any], choices: Sequence[Any]) -> Any:
    if not choices:
        raise CampaignError("no valid parameter value remains for this source")
    digest = hashlib.sha256(_canonical_json_bytes(list(parts))).digest()
    return choices[int.from_bytes(digest[:8], "big") % len(choices)]


def _planned_parameters(
    global_seed: int,
    source_id: str,
    view_slot: str,
    transform: str,
    *,
    sample_rate: int,
    minimum_pitch: int,
    maximum_pitch: int,
) -> Dict[str, Any]:
    # These two values are validated elsewhere.  They are intentionally not
    # part of this established stream: retaining the accepted stream makes a
    # conventional projection of an older plan byte-replayable.
    del sample_rate, minimum_pitch, maximum_pitch
    relevant_configuration_sha256 = _canonical_sha256(
        {
            "stream_algorithm": RANDOM_STREAM_ALGORITHM,
            "transform": transform,
            "parameter_grid": PARAMETER_GRIDS[transform],
        }
    )
    output: Dict[str, Any] = {}
    for field in sorted(PARAMETER_GRIDS[transform]):
        choices = list(PARAMETER_GRIDS[transform][field])
        output[field] = _digest_choice(
            (
                RANDOM_STREAM_ALGORITHM,
                global_seed,
                source_id,
                view_slot,
                transform,
                field,
                relevant_configuration_sha256,
            ),
            choices,
        )
    return output


def _item_seed(
    global_seed: int,
    source_id: str,
    view_slot: str,
    transform: str,
) -> int:
    relevant_configuration_sha256 = _canonical_sha256(
        {
            "stream_algorithm": RANDOM_STREAM_ALGORITHM,
            "transform": transform,
            "parameter_grid": PARAMETER_GRIDS[transform],
        }
    )
    digest = hashlib.sha256(
        _canonical_json_bytes(
            [
                RANDOM_STREAM_ALGORITHM,
                global_seed,
                source_id,
                view_slot,
                transform,
                relevant_configuration_sha256,
            ]
        )
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _configuration_document() -> Dict[str, Any]:
    return {
        "conditions": list(CONDITION_IDS),
        "original_view_slot": ORIGINAL_VIEW_SLOT,
        "view_slots": [
            {"view_slot": slot, "transform": transform}
            for slot, transform in VIEW_SLOTS
        ],
        "parameter_grids": PARAMETER_GRIDS,
        "random_stream_algorithm": RANDOM_STREAM_ALGORITHM,
        "midi_normalization": NORMALIZATION_POLICY,
        "selection_split": "train",
    }


def _validate_audio_source(path: Path, source: Mapping[str, Any]) -> sf.SoundFile:
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise CampaignError(f"unsupported source audio suffix: {path.suffix}")
    try:
        info = sf.info(str(path))
        audio, sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception as exc:
        raise CampaignError(f"source audio is unreadable: {path}") from exc
    expected = {
        "sample_rate_hz": source.get("sample_rate_hz"),
        "sample_count": source.get("sample_count"),
        "channel_count": source.get("channel_count"),
    }
    observed = {
        "sample_rate_hz": int(info.samplerate),
        "sample_count": int(info.frames),
        "channel_count": int(info.channels),
    }
    if expected != observed:
        raise CampaignError(
            f"source {source.get('source_id')!r} audio metadata mismatch: {observed}"
        )
    if (
        int(info.samplerate) != 44100
        or int(info.channels) != 1
        or info.format != "WAV"
        or info.subtype != "PCM_16"
    ):
        raise CampaignError(
            "Galdr conventional replay requires mono 44.1-kHz PCM-16 WAV input"
        )
    if int(sample_rate) != int(info.samplerate) or audio.shape != (
        int(info.frames),
        int(info.channels),
    ):
        raise CampaignError("decoded audio shape does not match its header")
    if audio.size == 0 or not np.isfinite(audio).all():
        raise CampaignError("source audio must be nonempty and finite")
    expected_duration = int(info.frames) / int(info.samplerate)
    provenance_duration = source.get("duration_seconds")
    if type(provenance_duration) not in (int, float) or not math.isfinite(
        float(provenance_duration)
    ):
        raise CampaignError("source duration_seconds must be finite")
    if abs(float(provenance_duration) - expected_duration) > 1.0 / int(info.samplerate):
        raise CampaignError("source duration does not match its audio frame count")
    return info


def _source_records(clean: Mapping[str, Any]) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, int]]:
    records: Dict[str, Mapping[str, Any]] = {}
    split_counts: Dict[str, int] = {}
    for index, source in enumerate(clean["sources"]):
        if not isinstance(source, dict):
            raise CampaignError(f"clean provenance source {index} is not an object")
        source_id = _require_digest(source.get("source_id"), f"source {index} ID")
        if source_id in records:
            raise CampaignError(f"clean provenance repeats source ID {source_id}")
        split = source.get("split")
        if split not in {"train", "validation", "test"}:
            raise CampaignError(f"source {source_id} has unsupported split {split!r}")
        split_counts[split] = split_counts.get(split, 0) + 1
        records[source_id] = source
    return records, split_counts


def build_plan(
    *,
    clean_provenance: Path,
    expected_clean_provenance_sha256: str,
    dataset_root: Path,
    global_seed: int,
    source_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a deterministic training-only plan without rendering views."""

    global_seed = _require_global_seed(global_seed)
    expected_clean_provenance_sha256 = _require_digest(
        expected_clean_provenance_sha256,
        "expected clean provenance digest",
    )
    root = _existing_directory(dataset_root, "dataset root")
    clean = _load_clean_provenance(
        clean_provenance,
        expected_clean_provenance_sha256,
    )
    all_records, split_counts = _source_records(clean)
    training = {
        source_id: source
        for source_id, source in all_records.items()
        if source["split"] == "train"
    }
    if source_ids is not None:
        requested = [
            _require_digest(value, "requested source ID") for value in source_ids
        ]
        if not requested or len(requested) != len(set(requested)):
            raise CampaignError("source_ids must be a non-empty unique sequence")
        unavailable = sorted(set(requested) - set(training))
        if unavailable:
            raise CampaignError(f"requested non-training source IDs: {unavailable}")
        training = {source_id: training[source_id] for source_id in sorted(requested)}
    if not training:
        raise CampaignError("plan has no training sources")

    planned_sources = []
    for source_id in sorted(training):
        source = training[source_id]
        required_integer_fields = (
            "audio_bytes",
            "raw_midi_bytes",
            "sample_rate_hz",
            "sample_count",
            "channel_count",
            "raw_note_count",
            "adjusted_previous_offsets",
            "removed_nonpositive_after_truncation",
            "normalized_note_count",
        )
        for field in required_integer_fields:
            if type(source.get(field)) is not int or int(source[field]) < 0:
                raise CampaignError(f"source {source_id} has invalid {field}")
        if source["sample_rate_hz"] <= 0 or source["sample_count"] <= 0 or source["channel_count"] <= 0:
            raise CampaignError(f"source {source_id} has invalid audio dimensions")
        audio_relative = _safe_relative_path(
            source.get("audio_relpath"),
            f"source {source_id} audio path",
        )
        midi_relative = _safe_relative_path(
            source.get("raw_midi_relpath"),
            f"source {source_id} MIDI path",
        )
        audio_path = _resolve_input_media(root, audio_relative, f"source {source_id} audio")
        midi_path = _resolve_input_media(root, midi_relative, f"source {source_id} MIDI")
        _stable_file_sha256(
            audio_path,
            source.get("audio_sha256"),
            f"source {source_id} audio",
            expected_bytes=source["audio_bytes"],
        )
        _stable_file_sha256(
            midi_path,
            source.get("raw_midi_sha256"),
            f"source {source_id} MIDI",
            expected_bytes=source["raw_midi_bytes"],
        )
        info = _validate_audio_source(audio_path, source)
        normalized, values, normalization_report = _normalization_values(
            midi_path,
            duration_seconds=int(info.frames) / int(info.samplerate),
        )
        del normalized
        _validate_normalization_report(source, normalization_report)
        minimum_pitch = min(int(note["pitch"]) for note in values)
        maximum_pitch = max(int(note["pitch"]) for note in values)
        slots = []
        for view_slot, transform in VIEW_SLOTS:
            slots.append(
                {
                    "view_slot": view_slot,
                    "transform": transform,
                    "seed": _item_seed(global_seed, source_id, view_slot, transform),
                    "parameters": _planned_parameters(
                        global_seed,
                        source_id,
                        view_slot,
                        transform,
                        sample_rate=int(info.samplerate),
                        minimum_pitch=minimum_pitch,
                        maximum_pitch=maximum_pitch,
                    ),
                }
            )
        planned_sources.append(
            {
                "source_id": source_id,
                "tune_key": _require_identifier(
                    source.get("tune_key"),
                    f"source {source_id} tune_key",
                ),
                "split": "train",
                "source_audio_relpath": audio_relative,
                "source_midi_relpath": midi_relative,
                "source_audio_sha256": source["audio_sha256"],
                "source_midi_sha256": source["raw_midi_sha256"],
                "source_audio_bytes": source["audio_bytes"],
                "source_midi_bytes": source["raw_midi_bytes"],
                "source_duration_seconds": float(source["duration_seconds"]),
                "sample_rate_hz": int(source["sample_rate_hz"]),
                "sample_count": int(source["sample_count"]),
                "channel_count": int(source["channel_count"]),
                "normalization": normalization_report,
                "normalized_pitch_minimum": minimum_pitch,
                "normalized_pitch_maximum": maximum_pitch,
                "view_slots": slots,
            }
        )
        _stable_file_sha256(
            audio_path,
            source["audio_sha256"],
            f"source {source_id} audio after planning",
            expected_bytes=source["audio_bytes"],
        )
        _stable_file_sha256(
            midi_path,
            source["raw_midi_sha256"],
            f"source {source_id} MIDI after planning",
            expected_bytes=source["raw_midi_bytes"],
        )

    configuration = _configuration_document()
    plan: Dict[str, Any] = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "algorithm": PLAN_ALGORITHM,
        "global_seed": global_seed,
        "conditions": list(CONDITION_IDS),
        "configuration": configuration,
        "configuration_sha256": _canonical_sha256(configuration),
        "clean_dataset": {
            "provenance_sha256": expected_clean_provenance_sha256,
            "dataset_identity_sha256": clean["dataset_identity_sha256"],
            "normalization_policy": NORMALIZATION_POLICY,
        },
        "source_selection": {
            "included_split": "train",
            "included_source_count": len(planned_sources),
            "available_split_counts": {
                key: split_counts.get(key, 0) for key in ("train", "validation", "test")
            },
            "excluded_split_counts": {
                "validation": split_counts.get("validation", 0),
                "test": split_counts.get("test", 0),
            },
        },
        "tool": {
            "name": "amt-augmentor",
            "version": __version__,
            "source_tree_sha256": _source_tree_sha256(),
            "runtime": _runtime_versions(),
        },
        "sources": planned_sources,
    }
    plan["plan_semantic_sha256"] = _canonical_sha256(plan)
    return validate_plan(plan)


def _require_exact_keys(document: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    if set(document) != set(expected):
        raise CampaignError(f"{label} has an invalid schema")


def validate_plan(document: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a plan and return it unchanged."""

    _require_exact_keys(
        document,
        (
            "plan_schema_version",
            "kind",
            "algorithm",
            "global_seed",
            "conditions",
            "configuration",
            "configuration_sha256",
            "clean_dataset",
            "source_selection",
            "tool",
            "sources",
            "plan_semantic_sha256",
        ),
        "plan",
    )
    if document["plan_schema_version"] != PLAN_SCHEMA_VERSION:
        raise CampaignError("unsupported plan schema version")
    if document["kind"] != PLAN_KIND or document["algorithm"] != PLAN_ALGORITHM:
        raise CampaignError("unexpected plan identity")
    _require_global_seed(document["global_seed"])
    if document["conditions"] != list(CONDITION_IDS):
        raise CampaignError("plan conditions must be exactly O and C")
    configuration = _configuration_document()
    if document["configuration"] != configuration:
        raise CampaignError("plan configuration does not match this adapter version")
    if document["configuration_sha256"] != _canonical_sha256(configuration):
        raise CampaignError("plan configuration digest mismatch")
    clean = document["clean_dataset"]
    if not isinstance(clean, dict):
        raise CampaignError("plan clean_dataset must be an object")
    _require_exact_keys(
        clean,
        ("provenance_sha256", "dataset_identity_sha256", "normalization_policy"),
        "plan clean_dataset",
    )
    _require_digest(clean["provenance_sha256"], "plan provenance digest")
    _require_digest(clean["dataset_identity_sha256"], "plan dataset identity")
    if clean["normalization_policy"] != NORMALIZATION_POLICY:
        raise CampaignError("plan normalization policy mismatch")
    tool = document["tool"]
    if not isinstance(tool, dict) or set(tool) != {
        "name",
        "version",
        "source_tree_sha256",
        "runtime",
    }:
        raise CampaignError("plan tool record is invalid")
    _require_digest(tool["source_tree_sha256"], "plan source-tree digest")
    if tool["name"] != "amt-augmentor" or tool["version"] != __version__:
        raise CampaignError("plan tool identity is invalid")
    sources = document["sources"]
    if not isinstance(sources, list) or not sources:
        raise CampaignError("plan must contain training sources")
    source_ids = [
        _require_digest(source.get("source_id"), f"plan source {index} ID")
        for index, source in enumerate(sources)
        if isinstance(source, dict)
    ]
    if len(source_ids) != len(sources):
        raise CampaignError("plan source must be an object")
    if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
        raise CampaignError("plan sources must be unique and sorted")
    for source in sources:
        _require_exact_keys(
            source,
            (
                "source_id",
                "tune_key",
                "split",
                "source_audio_relpath",
                "source_midi_relpath",
                "source_audio_sha256",
                "source_midi_sha256",
                "source_audio_bytes",
                "source_midi_bytes",
                "source_duration_seconds",
                "sample_rate_hz",
                "sample_count",
                "channel_count",
                "normalization",
                "normalized_pitch_minimum",
                "normalized_pitch_maximum",
                "view_slots",
            ),
            "plan source",
        )
        if source.get("split") != "train":
            raise CampaignError("plan contains a non-training source")
        _require_identifier(source.get("tune_key"), "plan tune_key")
        for field in ("source_audio_relpath", "source_midi_relpath"):
            _safe_relative_path(source.get(field), f"plan {field}")
        for field in ("source_audio_sha256", "source_midi_sha256"):
            _require_digest(source.get(field), f"plan {field}")
        normalization = source.get("normalization")
        if not isinstance(normalization, dict):
            raise CampaignError("plan source normalization must be an object")
        _require_exact_keys(
            normalization,
            (
                "raw_note_count",
                "adjusted_previous_offsets",
                "removed_nonpositive_after_truncation",
                "normalized_note_count",
                "annotation_semantic_sha256",
            ),
            "plan source normalization",
        )
        _require_digest(
            normalization.get("annotation_semantic_sha256"),
            "normalized annotation digest",
        )
        slots = source.get("view_slots")
        if not isinstance(slots, list) or len(slots) != len(VIEW_SLOTS):
            raise CampaignError("plan source has an invalid view-slot count")
        expected_slots = [slot for slot, _ in VIEW_SLOTS]
        if [slot.get("view_slot") for slot in slots] != expected_slots:
            raise CampaignError("plan source view slots are not canonical")
        for slot, (_, transform) in zip(slots, VIEW_SLOTS):
            _require_exact_keys(
                slot,
                ("view_slot", "transform", "seed", "parameters"),
                "plan view slot",
            )
            if slot.get("transform") != transform:
                raise CampaignError("plan source transform order is not canonical")
            if type(slot.get("seed")) is not int or not 0 <= slot["seed"] < 2**32:
                raise CampaignError("plan view seed is invalid")
            expected_seed = _item_seed(
                document["global_seed"],
                source["source_id"],
                slot["view_slot"],
                transform,
            )
            expected_parameters = _planned_parameters(
                document["global_seed"],
                source["source_id"],
                slot["view_slot"],
                transform,
                sample_rate=int(source["sample_rate_hz"]),
                minimum_pitch=int(source["normalized_pitch_minimum"]),
                maximum_pitch=int(source["normalized_pitch_maximum"]),
            )
            if slot["seed"] != expected_seed or slot.get("parameters") != expected_parameters:
                raise CampaignError("plan view seed or parameters are not reproducible")
    selection = document["source_selection"]
    if not isinstance(selection, dict) or selection.get("included_split") != "train":
        raise CampaignError("plan source selection is invalid")
    if selection.get("included_source_count") != len(sources):
        raise CampaignError("plan included-source count mismatch")
    digest = document["plan_semantic_sha256"]
    _require_digest(digest, "plan semantic digest")
    without_digest = dict(document)
    without_digest.pop("plan_semantic_sha256")
    if digest != _canonical_sha256(without_digest):
        raise CampaignError("plan semantic digest mismatch")
    return document


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, PUBLISHED_FILE_MODE)


def write_plan(path: Path, document: Mapping[str, Any]) -> str:
    validate_plan(dict(document))
    path = _prepare_new_path(path, "plan")
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.staging-",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, PUBLISHED_FILE_MODE)
        os.link(str(temporary), str(path))
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise CampaignError(f"refusing to overwrite plan: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def load_plan(path: Path) -> Tuple[Dict[str, Any], str]:
    resolved = _existing_regular_file(path, "plan")
    document = _load_json_file(resolved, "plan")
    return validate_plan(document), _sha256_file(resolved)


def _write_normalized_midi(path: Path, midi: pretty_midi.PrettyMIDI) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(str(path)):
        raise CampaignError(f"refusing to overwrite derivative MIDI: {path}")
    midi.write(str(path))
    os.chmod(path, PUBLISHED_FILE_MODE)
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as read_handle, target.open("xb") as write_handle:
        shutil.copyfileobj(read_handle, write_handle, length=1024 * 1024)
        write_handle.flush()
        os.fsync(write_handle.fileno())
    os.chmod(target, PUBLISHED_FILE_MODE)


def _view_paths(root: Path, category: str, source_id: str, slot: str, suffix: str) -> Tuple[Path, Path, Path]:
    directory = root / "media" / category / source_id
    audio = directory / f"{slot}{suffix}"
    midi = directory / f"{slot}.mid"
    provenance = audio.with_suffix(audio.suffix + ".provenance.json")
    return audio, midi, provenance


def _render_conventional(
    slot: Mapping[str, Any],
    source_audio: Path,
    source_midi: Path,
    output_audio: Path,
    output_midi: Path,
    output_provenance: Path,
) -> Dict[str, Any]:
    common = {
        "audio_path": source_audio,
        "midi_path": source_midi,
        "output_audio_path": output_audio,
        "output_midi_path": output_midi,
        "provenance_path": output_provenance,
        "seed": slot["seed"],
    }
    parameters = slot["parameters"]
    transform = slot["transform"]
    if transform == "gain_chorus_v1":
        return gain_chorus_v1(**common, parameters=GainChorusParameters(**parameters))
    if transform == "noise_snr_v1":
        return noise_snr_v1(**common, parameters=NoiseSNRParameters(**parameters))
    if transform == "reverb_filters_v1":
        return reverb_filters_v1(**common, parameters=ReverbFiltersParameters(**parameters))
    if transform == "pitch_shift_v1":
        return pitch_shift_v1(**common, parameters=PitchShiftParameters(**parameters))
    if transform == "time_stretch_v1":
        return time_stretch_v1(**common, parameters=TimeStretchParameters(**parameters))
    raise CampaignError(f"unsupported transform {transform!r}")


def _midi_notes(path: Path) -> Tuple[pretty_midi.PrettyMIDI, List[pretty_midi.Note]]:
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        raise CampaignError(f"output MIDI is unreadable: {path}") from exc
    if len(midi.instruments) != 1:
        raise CampaignError("output MIDI must contain exactly one instrument")
    instrument = midi.instruments[0]
    if instrument.is_drum or instrument.control_changes or instrument.pitch_bends:
        raise CampaignError("output MIDI contains unsupported events")
    notes = list(instrument.notes)
    if not notes:
        raise CampaignError("output MIDI contains no notes")
    return midi, notes


def _validate_view(
    *,
    audio_path: Path,
    midi_path: Path,
    provenance_path: Path,
    source_midi_path: Path,
    source: Mapping[str, Any],
    transform: str,
    seed: Optional[int],
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    for path, label in (
        (audio_path, "view audio"),
        (midi_path, "view MIDI"),
        (provenance_path, "view provenance"),
    ):
        if not path.is_file() or path.is_symlink():
            raise CampaignError(f"{label} is missing or unsafe: {path}")
    try:
        audio, sample_rate = sf.read(str(audio_path), dtype="float64", always_2d=True)
        info = sf.info(str(audio_path))
    except Exception as exc:
        raise CampaignError(f"view audio is unreadable: {audio_path}") from exc
    if audio.size == 0 or not np.isfinite(audio).all():
        raise CampaignError("view audio must be nonempty and finite")
    if int(sample_rate) != source["sample_rate_hz"] or int(info.channels) != source["channel_count"]:
        raise CampaignError("view changed sample rate or channel count")
    source_midi, source_notes = _midi_notes(source_midi_path)
    del source_midi
    output_midi, output_notes = _midi_notes(midi_path)
    del output_midi
    if len(output_notes) != len(source_notes):
        raise CampaignError("view changed the normalized note count")
    duration = int(info.frames) / int(info.samplerate)
    half_tick = 60.0 / ANNOTATION_MIDI_TEMPO / ANNOTATION_MIDI_RESOLUTION / 2.0
    for note in output_notes:
        if note.start < 0 or note.end <= note.start or note.end > duration + half_tick + 1e-9:
            raise CampaignError("view MIDI has invalid timing for its audio")
    if transform == "identity_v1":
        if _sha256_file(audio_path) != source["source_audio_sha256"]:
            raise CampaignError("original view audio is not byte-identical to its source")
    elif transform in {"gain_chorus_v1", "noise_snr_v1", "reverb_filters_v1"}:
        if _sha256_file(midi_path) != _sha256_file(source_midi_path):
            raise CampaignError("audio-only view changed derivative MIDI bytes")
        if int(info.frames) != source["sample_count"]:
            raise CampaignError("audio-only view changed sample count")
    elif transform == "pitch_shift_v1":
        delta = int(parameters["semitones"])
        for input_note, output_note in zip(source_notes, output_notes):
            if (
                output_note.pitch != input_note.pitch + delta
                or output_note.velocity != input_note.velocity
                or abs(output_note.start - input_note.start) > half_tick + 1e-9
                or abs(output_note.end - input_note.end) > half_tick + 1e-9
            ):
                raise CampaignError("pitch-shifted MIDI is not synchronized")
        if int(info.frames) != source["sample_count"]:
            raise CampaignError("pitch shift changed sample count")
    elif transform == "time_stretch_v1":
        provenance = _load_json_file(provenance_path, "view provenance")
        scale = provenance.get("qc", {}).get("realized_time_scale")
        if type(scale) not in (int, float) or not math.isfinite(float(scale)):
            raise CampaignError("time-stretch provenance lacks realized time scale")
        for input_note, output_note in zip(source_notes, output_notes):
            if (
                output_note.pitch != input_note.pitch
                or output_note.velocity != input_note.velocity
                or abs(output_note.start - input_note.start * float(scale)) > half_tick + 1e-9
                or abs(output_note.end - input_note.end * float(scale)) > half_tick + 1e-9
            ):
                raise CampaignError("time-stretched MIDI is not synchronized")
    else:
        raise CampaignError(f"unsupported rendered transform {transform!r}")

    provenance = _load_json_file(provenance_path, "view provenance")
    if transform == "identity_v1":
        if provenance.get("kind") != "galdr_conventional_original_view_provenance":
            raise CampaignError("original view provenance identity mismatch")
    else:
        if (
            provenance.get("transform") != transform
            or provenance.get("seed") != seed
            or provenance.get("parameters") != dict(parameters)
            or provenance.get("output", {}).get("audio_sha256") != _sha256_file(audio_path)
            or provenance.get("output", {}).get("midi_sha256") != _sha256_file(midi_path)
        ):
            raise CampaignError("conventional view provenance mismatch")
    return {
        "sample_rate_hz": int(info.samplerate),
        "sample_count": int(info.frames),
        "channel_count": int(info.channels),
        "duration_seconds": duration,
        "note_count": len(output_notes),
        "audio_sha256": _sha256_file(audio_path),
        "midi_sha256": _sha256_file(midi_path),
        "provenance_sha256": _sha256_file(provenance_path),
    }


def _record_for_view(
    root: Path,
    source: Mapping[str, Any],
    view_slot: str,
    transform: str,
    audio: Path,
    midi: Path,
    provenance: Path,
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "source_id": source["source_id"],
        "tune_key": source["tune_key"],
        "split": "train",
        "view_slot": view_slot,
        "view_kind": "original" if view_slot == ORIGINAL_VIEW_SLOT else "augmented",
        "transform": transform,
        "source_duration_seconds": source["source_duration_seconds"],
        "view_duration_seconds": validation["duration_seconds"],
        "sample_rate_hz": validation["sample_rate_hz"],
        "sample_count": validation["sample_count"],
        "channel_count": validation["channel_count"],
        "note_count": validation["note_count"],
        "audio_filename": audio.relative_to(root).as_posix(),
        "midi_filename": midi.relative_to(root).as_posix(),
        "provenance_filename": provenance.relative_to(root).as_posix(),
        "audio_sha256": validation["audio_sha256"],
        "midi_sha256": validation["midi_sha256"],
        "provenance_sha256": validation["provenance_sha256"],
    }


def _write_condition_files(
    root: Path,
    condition: str,
    records: Sequence[Mapping[str, Any]],
    plan_sha256: str,
) -> Dict[str, Any]:
    condition_root = root / "conditions" / condition
    condition_root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda row: (row["source_id"], row["view_slot"]))
    metadata_path = condition_root / "metadata.csv"
    metadata_fields = [
        "source_id",
        "tune_key",
        "view_slot",
        "split",
        "audio_filename",
        "midi_filename",
        "duration",
    ]
    with metadata_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields, lineterminator="\n")
        writer.writeheader()
        for record in ordered:
            writer.writerow(
                {
                    "source_id": record["source_id"],
                    "tune_key": record["tune_key"],
                    "view_slot": record["view_slot"],
                    "split": "train",
                    "audio_filename": record["audio_filename"],
                    "midi_filename": record["midi_filename"],
                    "duration": format(float(record["view_duration_seconds"]), ".17g"),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(metadata_path, PUBLISHED_FILE_MODE)

    lineage = {
        "lineage_schema_version": LINEAGE_SCHEMA_VERSION,
        "kind": LINEAGE_KIND,
        "records": [
            {
                "audio_filename": record["audio_filename"],
                "source_id": record["source_id"],
                "view_id": record["view_slot"],
                "view_kind": record["view_kind"],
                "source_duration_seconds": record["source_duration_seconds"],
                "view_duration_seconds": record["view_duration_seconds"],
            }
            for record in ordered
        ],
    }
    lineage_path = condition_root / "training-lineage.json"
    _write_json(lineage_path, lineage)
    identity = {
        "schema_version": 1,
        "kind": "galdr_conventional_condition_identity",
        "condition": condition,
        "materialization_plan_sha256": plan_sha256,
        "source_count": len({record["source_id"] for record in ordered}),
        "recording_count": len(ordered),
        "view_slots_per_source": 1 if condition == "O" else 1 + len(VIEW_SLOTS),
        "media": [dict(record) for record in ordered],
    }
    identity_path = condition_root / "condition-identity.json"
    _write_json(identity_path, identity)
    return {
        "condition": condition,
        "source_count": identity["source_count"],
        "recording_count": identity["recording_count"],
        "metadata_path": metadata_path.relative_to(root).as_posix(),
        "metadata_sha256": _sha256_file(metadata_path),
        "lineage_path": lineage_path.relative_to(root).as_posix(),
        "lineage_sha256": _sha256_file(lineage_path),
        "identity_path": identity_path.relative_to(root).as_posix(),
        "identity_sha256": _sha256_file(identity_path),
    }


def _write_derivatives(
    root: Path,
    records: Sequence[Mapping[str, Any]],
    plan_semantic_sha256: str,
) -> Dict[str, Any]:
    """Write the compact manifest consumed by Galdr's split/lineage gate."""

    path = root / "derivatives.csv"
    fieldnames = [
        "example_id",
        "source_id",
        "tune_key",
        "split",
        "audio_sha256",
        "midi_sha256",
    ]
    ordered = sorted(records, key=lambda row: (row["source_id"], row["view_slot"]))
    rows = []
    for record in ordered:
        example_id = _canonical_sha256(
            {
                "identity": "galdr-conventional/example/v1",
                "plan_semantic_sha256": plan_semantic_sha256,
                "source_id": record["source_id"],
                "view_slot": record["view_slot"],
                "transform": record["transform"],
                "audio_sha256": record["audio_sha256"],
                "midi_sha256": record["midi_sha256"],
            }
        )
        rows.append(
            {
                "example_id": example_id,
                "source_id": record["source_id"],
                "tune_key": record["tune_key"],
                "split": "train",
                "audio_sha256": record["audio_sha256"],
                "midi_sha256": record["midi_sha256"],
            }
        )
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, PUBLISHED_FILE_MODE)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "recording_count": len(rows),
        "fieldnames": fieldnames,
    }


def _inventory(root: Path) -> List[Dict[str, Any]]:
    entries = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise CampaignError(f"artifact contains symbolic link: {path}")
        if path.is_file() and path.name != "materialization-report.json":
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return entries


def _publish_directory(source: Path, destination: Path) -> None:
    try:
        destination.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise CampaignError(f"refusing to replace output root: {destination}") from exc
    completion = source / "materialization-report.json"
    if not completion.is_file():
        raise CampaignError("staging directory has no completion report")
    try:
        for entry in sorted(
            (item for item in source.iterdir() if item.name != completion.name),
            key=lambda item: item.name,
        ):
            os.rename(str(entry), str(destination / entry.name))
        _fsync_directory(destination)
        os.rename(str(completion), str(destination / completion.name))
        os.chmod(destination, PUBLISHED_DIRECTORY_MODE)
        _fsync_directory(destination)
        source.rmdir()
    except Exception:
        # The reserved directory intentionally remains incomplete.  The absent
        # completion report makes it unusable, and a rerun cannot overwrite it.
        raise


def _clean_record_matches_plan(clean: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    if clean.get("split") != "train":
        raise CampaignError("materialization source is no longer training data")
    comparisons = {
        "audio_relpath": source["source_audio_relpath"],
        "raw_midi_relpath": source["source_midi_relpath"],
        "audio_sha256": source["source_audio_sha256"],
        "raw_midi_sha256": source["source_midi_sha256"],
        "audio_bytes": source["source_audio_bytes"],
        "raw_midi_bytes": source["source_midi_bytes"],
        "sample_rate_hz": source["sample_rate_hz"],
        "sample_count": source["sample_count"],
        "channel_count": source["channel_count"],
    }
    for field, expected in comparisons.items():
        if clean.get(field) != expected:
            raise CampaignError(f"clean provenance changed source field {field}")


def materialize_plan(
    *,
    plan_path: Path,
    clean_provenance: Path,
    dataset_root: Path,
    output_root: Path,
    workers: int = 1,
) -> Dict[str, Any]:
    """Render a plan into a no-overwrite, completion-marked artifact."""

    workers = _require_workers(workers)
    plan, plan_sha256 = load_plan(plan_path)
    if plan["tool"]["source_tree_sha256"] != _source_tree_sha256():
        raise CampaignError("plan source-tree digest does not match this checkout")
    if plan["tool"]["runtime"] != _runtime_versions():
        raise CampaignError("plan runtime versions do not match the materializer")
    root = _existing_directory(dataset_root, "dataset root")
    clean = _load_clean_provenance(
        clean_provenance,
        plan["clean_dataset"]["provenance_sha256"],
    )
    if clean["dataset_identity_sha256"] != plan["clean_dataset"]["dataset_identity_sha256"]:
        raise CampaignError("clean dataset identity does not match the plan")
    clean_records, _ = _source_records(clean)
    output = _prepare_new_path(output_root, "output root")
    staging = Path(
        tempfile.mkdtemp(dir=str(output.parent), prefix=f".{output.name}.staging-")
    )
    condition_records: Dict[str, List[Dict[str, Any]]] = {"O": [], "C": []}
    source_reports = []
    try:
        for source in plan["sources"]:
            source_id = source["source_id"]
            clean_source = clean_records.get(source_id)
            if clean_source is None:
                raise CampaignError(f"source {source_id} is absent from clean provenance")
            _clean_record_matches_plan(clean_source, source)
            canonical_audio = _resolve_input_media(
                root,
                source["source_audio_relpath"],
                f"source {source_id} audio",
            )
            canonical_midi = _resolve_input_media(
                root,
                source["source_midi_relpath"],
                f"source {source_id} MIDI",
            )
            _stable_file_sha256(
                canonical_audio,
                source["source_audio_sha256"],
                f"source {source_id} audio",
                expected_bytes=source["source_audio_bytes"],
            )
            _stable_file_sha256(
                canonical_midi,
                source["source_midi_sha256"],
                f"source {source_id} MIDI",
                expected_bytes=source["source_midi_bytes"],
            )
            info = _validate_audio_source(canonical_audio, clean_source)
            normalized, values, normalization_report = _normalization_values(
                canonical_midi,
                duration_seconds=int(info.frames) / int(info.samplerate),
            )
            if normalization_report != source["normalization"]:
                raise CampaignError("normalization result differs from the plan")

            original_audio, original_midi, original_provenance = _view_paths(
                staging,
                "original",
                source_id,
                ORIGINAL_VIEW_SLOT,
                canonical_audio.suffix.lower(),
            )
            _copy_file(canonical_audio, original_audio)
            _write_normalized_midi(original_midi, normalized)
            normalized_serialized_sha256 = _sha256_file(original_midi)
            original_document = {
                "schema_version": 1,
                "kind": "galdr_conventional_original_view_provenance",
                "transform": "identity_v1",
                "source": {
                    "source_id": source_id,
                    "audio_sha256": source["source_audio_sha256"],
                    "raw_midi_sha256": source["source_midi_sha256"],
                },
                "normalization": {
                    "policy": NORMALIZATION_POLICY,
                    **normalization_report,
                    "serialized_derivative_midi_sha256": normalized_serialized_sha256,
                },
                "output": {
                    "audio_name": original_audio.name,
                    "midi_name": original_midi.name,
                    "audio_sha256": _sha256_file(original_audio),
                    "midi_sha256": normalized_serialized_sha256,
                    "audio_samples": int(info.frames),
                    "channels": int(info.channels),
                    "midi_note_count": len(values),
                },
                "publication": {
                    "completion_marker": original_provenance.name,
                    "canonical_media_modified": False,
                },
            }
            _write_json(original_provenance, original_document)
            original_validation = _validate_view(
                audio_path=original_audio,
                midi_path=original_midi,
                provenance_path=original_provenance,
                source_midi_path=original_midi,
                source=source,
                transform="identity_v1",
                seed=None,
                parameters={},
            )
            original_record = _record_for_view(
                staging,
                source,
                ORIGINAL_VIEW_SLOT,
                "identity_v1",
                original_audio,
                original_midi,
                original_provenance,
                original_validation,
            )
            condition_records["O"].append(original_record)
            condition_records["C"].append(original_record)

            def render_slot(slot: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                output_audio, output_midi, output_provenance = _view_paths(
                    staging,
                    "C",
                    source_id,
                    slot["view_slot"],
                    canonical_audio.suffix.lower(),
                )
                rendered_provenance = _render_conventional(
                    slot,
                    canonical_audio,
                    original_midi,
                    output_audio,
                    output_midi,
                    output_provenance,
                )
                if rendered_provenance.get("plan_config_sha256") != _canonical_sha256(
                    rendered_provenance.get("plan_config")
                ):
                    raise CampaignError("rendered provenance plan digest mismatch")
                validation = _validate_view(
                    audio_path=output_audio,
                    midi_path=output_midi,
                    provenance_path=output_provenance,
                    source_midi_path=original_midi,
                    source=source,
                    transform=slot["transform"],
                    seed=slot["seed"],
                    parameters=slot["parameters"],
                )
                record = _record_for_view(
                    staging,
                    source,
                    slot["view_slot"],
                    slot["transform"],
                    output_audio,
                    output_midi,
                    output_provenance,
                    validation,
                )
                slot_report = {
                    "view_slot": slot["view_slot"],
                    "transform": slot["transform"],
                    "seed": slot["seed"],
                    "parameters": slot["parameters"],
                    "audio_sha256": validation["audio_sha256"],
                    "midi_sha256": validation["midi_sha256"],
                    "provenance_sha256": validation["provenance_sha256"],
                }
                return record, slot_report

            if workers == 1:
                rendered_slots = [render_slot(slot) for slot in source["view_slots"]]
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(workers, len(VIEW_SLOTS))
                ) as executor:
                    rendered_slots = list(executor.map(render_slot, source["view_slots"]))
            condition_records["C"].extend(record for record, _ in rendered_slots)
            slot_reports = [slot_report for _, slot_report in rendered_slots]
            _stable_file_sha256(
                canonical_audio,
                source["source_audio_sha256"],
                f"source {source_id} audio after materialization",
                expected_bytes=source["source_audio_bytes"],
            )
            _stable_file_sha256(
                canonical_midi,
                source["source_midi_sha256"],
                f"source {source_id} MIDI after materialization",
                expected_bytes=source["source_midi_bytes"],
            )
            source_reports.append(
                {
                    "source_id": source_id,
                    "canonical_sources_unchanged": True,
                    "normalization": normalization_report,
                    "serialized_derivative_midi_sha256": normalized_serialized_sha256,
                    "views": slot_reports,
                }
            )

        condition_reports = [
            _write_condition_files(
                staging,
                condition,
                condition_records[condition],
                plan_sha256,
            )
            for condition in CONDITION_IDS
        ]
        derivatives_report = _write_derivatives(
            staging,
            condition_records["C"],
            plan["plan_semantic_sha256"],
        )
        report: Dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "status": "complete",
            "algorithm": PLAN_ALGORITHM,
            "plan_sha256": plan_sha256,
            "plan_semantic_sha256": plan["plan_semantic_sha256"],
            "configuration_sha256": plan["configuration_sha256"],
            "clean_dataset": plan["clean_dataset"],
            "tool": plan["tool"],
            "global_seed": plan["global_seed"],
            "source_count": len(plan["sources"]),
            "conditions": condition_reports,
            "derivatives": derivatives_report,
            "sources": source_reports,
            "canonical_sources_unchanged": True,
            "payload_inventory": _inventory(staging),
            "completion_rule": (
                "The artifact is complete only when this report exists and all "
                "recorded hashes verify."
            ),
        }
        report["report_semantic_sha256"] = _canonical_sha256(report)
        _write_json(staging / "materialization-report.json", report)
        verify_materialization(staging)
        for directory in sorted(
            [path for path in staging.rglob("*") if path.is_dir()],
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.chmod(directory, PUBLISHED_DIRECTORY_MODE)
            _fsync_directory(directory)
        os.chmod(staging, PUBLISHED_DIRECTORY_MODE)
        _fsync_directory(staging)
        _publish_directory(staging, output)
        _fsync_directory(output.parent)
        verify_materialization(output)
        return report
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_materialization(output_root: Path) -> Dict[str, Any]:
    """Verify hashes, structure, channels, and timing for a completed artifact."""

    root = _existing_directory(output_root, "materialization root")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise CampaignError(f"materialization contains symbolic link: {path}")
    report_path = root / "materialization-report.json"
    report = _load_json_file(report_path, "materialization report")
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("kind") != REPORT_KIND
        or report.get("status") != "complete"
    ):
        raise CampaignError("materialization report identity is invalid")
    semantic = report.get("report_semantic_sha256")
    _require_digest(semantic, "report semantic digest")
    without_digest = dict(report)
    without_digest.pop("report_semantic_sha256")
    if semantic != _canonical_sha256(without_digest):
        raise CampaignError("materialization report semantic digest mismatch")
    expected_inventory = report.get("payload_inventory")
    if expected_inventory != _inventory(root):
        raise CampaignError("materialization payload inventory mismatch")
    for entry in expected_inventory:
        path = _resolve_input_media(root, entry["path"], "inventory path")
        if path.stat().st_size != entry["bytes"] or _sha256_file(path) != entry["sha256"]:
            raise CampaignError(f"materialization payload mismatch: {entry['path']}")
    conditions = report.get("conditions")
    if not isinstance(conditions, list) or [item.get("condition") for item in conditions] != list(CONDITION_IDS):
        raise CampaignError("materialization conditions are not exactly O and C")
    expected_slots = {
        "O": {ORIGINAL_VIEW_SLOT},
        "C": {ORIGINAL_VIEW_SLOT, *(slot for slot, _ in VIEW_SLOTS)},
    }
    derivatives = report.get("derivatives")
    if not isinstance(derivatives, dict):
        raise CampaignError("materialization lacks derivatives manifest")
    derivatives_path = _resolve_input_media(root, derivatives.get("path"), "derivatives path")
    if _sha256_file(derivatives_path) != derivatives.get("sha256"):
        raise CampaignError("derivatives manifest digest mismatch")
    with derivatives_path.open(newline="", encoding="utf-8") as handle:
        derivative_reader = csv.DictReader(handle)
        derivative_rows = list(derivative_reader)
        if derivative_reader.fieldnames != derivatives.get("fieldnames"):
            raise CampaignError("derivatives manifest fields are incompatible")
    if len(derivative_rows) != derivatives.get("recording_count"):
        raise CampaignError("derivatives manifest row count mismatch")
    for row in derivative_rows:
        if row.get("split") != "train":
            raise CampaignError("derivatives manifest contains non-training data")
        _require_digest(row.get("example_id"), "derivative example ID")
        _require_digest(row.get("source_id"), "derivative source ID")
        _require_digest(row.get("audio_sha256"), "derivative audio digest")
        _require_digest(row.get("midi_sha256"), "derivative MIDI digest")
        _require_identifier(row.get("tune_key"), "derivative tune_key")
    if len({row["example_id"] for row in derivative_rows}) != len(derivative_rows):
        raise CampaignError("derivatives manifest repeats an example ID")
    for condition in conditions:
        for path_field, digest_field in (
            ("metadata_path", "metadata_sha256"),
            ("lineage_path", "lineage_sha256"),
            ("identity_path", "identity_sha256"),
        ):
            path = _resolve_input_media(root, condition[path_field], path_field)
            if _sha256_file(path) != condition[digest_field]:
                raise CampaignError(f"condition {path_field} digest mismatch")
        identity_path = _resolve_input_media(root, condition["identity_path"], "condition identity")
        identity = _load_json_file(identity_path, "condition identity")
        if (
            identity.get("kind") != "galdr_conventional_condition_identity"
            or identity.get("condition") != condition["condition"]
            or identity.get("materialization_plan_sha256") != report["plan_sha256"]
        ):
            raise CampaignError("condition identity mismatch")
        media = identity.get("media")
        if not isinstance(media, list) or len(media) != condition["recording_count"]:
            raise CampaignError("condition media count mismatch")
        by_source: Dict[str, set] = {}
        for record in media:
            if record.get("split") != "train":
                raise CampaignError("condition contains non-training data")
            by_source.setdefault(record["source_id"], set()).add(record["view_slot"])
            for path_field, digest_field in (
                ("audio_filename", "audio_sha256"),
                ("midi_filename", "midi_sha256"),
                ("provenance_filename", "provenance_sha256"),
            ):
                media_path = _resolve_input_media(root, record[path_field], path_field)
                if _sha256_file(media_path) != record[digest_field]:
                    raise CampaignError(f"condition media digest mismatch: {record[path_field]}")
            info = sf.info(str(root / record["audio_filename"]))
            if (
                int(info.samplerate) != record["sample_rate_hz"]
                or int(info.frames) != record["sample_count"]
                or int(info.channels) != record["channel_count"]
            ):
                raise CampaignError("condition audio metadata mismatch")
            _, notes = _midi_notes(root / record["midi_filename"])
            if len(notes) != record["note_count"]:
                raise CampaignError("condition MIDI note-count mismatch")
            duration = int(info.frames) / int(info.samplerate)
            half_tick = 60.0 / ANNOTATION_MIDI_TEMPO / ANNOTATION_MIDI_RESOLUTION / 2.0
            if any(note.end > duration + half_tick + 1e-9 for note in notes):
                raise CampaignError("condition MIDI extends beyond its audio")
        if any(slots != expected_slots[condition["condition"]] for slots in by_source.values()):
            raise CampaignError("condition source has an incomplete or unexpected view set")
        if len(by_source) != condition["source_count"]:
            raise CampaignError("condition source count mismatch")
        metadata_path = root / condition["metadata_path"]
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != condition["recording_count"] or any(row.get("split") != "train" for row in rows):
            raise CampaignError("condition metadata rows are invalid")
    return report


def estimate_plan(document: Mapping[str, Any]) -> Dict[str, Any]:
    plan = validate_plan(dict(document))
    source_pcm_bytes = sum(
        int(source["sample_count"]) * int(source["channel_count"]) * 2
        for source in plan["sources"]
    )
    source_seconds = sum(float(source["source_duration_seconds"]) for source in plan["sources"])
    estimated_bytes = int(source_pcm_bytes * (1 + len(VIEW_SLOTS) * 1.12))
    return {
        "sources": len(plan["sources"]),
        "source_duration_hours": source_seconds / 3600.0,
        "source_pcm_bytes": source_pcm_bytes,
        "persistent_view_multiplier_before_duration_allowance": 1 + len(VIEW_SLOTS),
        "estimated_persistent_pcm_bytes_with_12pct_transform_allowance": estimated_bytes,
        "estimated_persistent_gib": estimated_bytes / 1024**3,
        "conventional_dsp_input_hours": source_seconds / 3600.0 * len(VIEW_SLOTS),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amt-augmentor-galdr-conventional",
        description="Plan and materialize deterministic Galdr O/C augmentations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="create a deterministic plan")
    plan.add_argument("--dataset-provenance", type=Path, required=True)
    plan.add_argument("--expected-provenance-sha256", required=True)
    plan.add_argument("--dataset-root", type=Path, required=True)
    plan.add_argument("--global-seed", type=int, required=True)
    plan.add_argument("--source-id", action="append", dest="source_ids")
    plan.add_argument("--output-plan", type=Path, required=True)
    estimate = subparsers.add_parser("estimate", help="estimate storage and DSP work")
    estimate.add_argument("--plan", type=Path, required=True)
    materialize = subparsers.add_parser("materialize", help="render a complete artifact")
    materialize.add_argument("--plan", type=Path, required=True)
    materialize.add_argument("--dataset-provenance", type=Path, required=True)
    materialize.add_argument("--dataset-root", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    materialize.add_argument("--workers", type=int, default=1)
    verify = subparsers.add_parser("verify", help="verify a completed artifact")
    verify.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "plan":
            document = build_plan(
                clean_provenance=arguments.dataset_provenance,
                expected_clean_provenance_sha256=arguments.expected_provenance_sha256,
                dataset_root=arguments.dataset_root,
                global_seed=arguments.global_seed,
                source_ids=arguments.source_ids,
            )
            file_sha256 = write_plan(arguments.output_plan, document)
            result = {
                "plan": str(arguments.output_plan),
                "plan_sha256": file_sha256,
                "plan_semantic_sha256": document["plan_semantic_sha256"],
                "source_count": len(document["sources"]),
            }
        elif arguments.command == "estimate":
            document, _ = load_plan(arguments.plan)
            result = estimate_plan(document)
        elif arguments.command == "materialize":
            report = materialize_plan(
                plan_path=arguments.plan,
                clean_provenance=arguments.dataset_provenance,
                dataset_root=arguments.dataset_root,
                output_root=arguments.output_root,
                workers=arguments.workers,
            )
            result = {
                "output_root": str(arguments.output_root),
                "report_semantic_sha256": report["report_semantic_sha256"],
                "source_count": report["source_count"],
            }
        else:
            report = verify_materialization(arguments.output_root)
            result = {
                "output_root": str(arguments.output_root),
                "report_semantic_sha256": report["report_semantic_sha256"],
                "status": "verified",
            }
    except (CampaignError, FileExistsError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
