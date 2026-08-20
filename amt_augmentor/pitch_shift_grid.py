"""Atomic materialization and acoustic QC for integral pitch-shift grids."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from amt_augmentor._paired_io import _validate_seed
from amt_augmentor.conventional_augmentations import (
    MIDI_MAXIMUM_PITCH,
    MIDI_MINIMUM_PITCH,
    PitchShiftParameters,
    pitch_shift_v1,
)

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "amt_augmentor_pitch_shift_grid_v1"
TIME_QUANTIZATION_DECIMALS = 4


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_pitch_shift_grid_v1(
    semitones: Sequence[int],
    *,
    source_pitch_minimum: Optional[int] = None,
    source_pitch_maximum: Optional[int] = None,
    minimum_midi_pitch: int = MIDI_MINIMUM_PITCH,
    maximum_midi_pitch: int = MIDI_MAXIMUM_PITCH,
) -> Tuple[int, ...]:
    """Validate one explicit, ordered, nonzero integral shift grid."""

    if isinstance(semitones, (str, bytes)) or not isinstance(semitones, Sequence):
        raise TypeError("semitones must be a non-empty sequence of built-in ints")
    values = tuple(semitones)
    if not values:
        raise ValueError("semitones must not be empty")
    if any(type(value) is not int for value in values):
        raise TypeError("every pitch shift must be a built-in int")
    if 0 in values:
        raise ValueError("the pitch-shift grid must not contain zero")
    if len(values) != len(set(values)):
        raise ValueError("the pitch-shift grid must not contain duplicates")
    if values != tuple(sorted(values)):
        raise ValueError("the pitch-shift grid must be strictly increasing")
    if type(minimum_midi_pitch) is not int or type(maximum_midi_pitch) is not int:
        raise TypeError("MIDI pitch bounds must be built-in ints")
    if not 0 <= minimum_midi_pitch <= maximum_midi_pitch <= 127:
        raise ValueError(
            "MIDI pitch bounds must satisfy 0 <= minimum <= maximum <= 127"
        )
    if (source_pitch_minimum is None) != (source_pitch_maximum is None):
        raise ValueError("source pitch minimum and maximum must be supplied together")
    if source_pitch_minimum is not None:
        if (
            type(source_pitch_minimum) is not int
            or type(source_pitch_maximum) is not int
        ):
            raise TypeError("source MIDI pitch bounds must be built-in ints")
        if not 0 <= source_pitch_minimum <= source_pitch_maximum <= 127:
            raise ValueError("source MIDI pitch bounds are invalid")
        if source_pitch_minimum + min(values) < minimum_midi_pitch:
            raise ValueError(
                "pitch grid would move a label below the output MIDI range"
            )
        if source_pitch_maximum + max(values) > maximum_midi_pitch:
            raise ValueError(
                "pitch grid would move a label above the output MIDI range"
            )
    return values


def _note_semantics(midi_path: Path, *, undo_shift: int = 0) -> List[List[Any]]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for instrument_index, instrument in enumerate(midi.instruments):
        for note in instrument.notes:
            notes.append(
                [
                    instrument_index,
                    round(float(note.start), TIME_QUANTIZATION_DECIMALS),
                    round(float(note.end), TIME_QUANTIZATION_DECIMALS),
                    int(note.pitch) - undo_shift,
                    int(note.velocity),
                ]
            )
    notes.sort()
    if not notes:
        raise ValueError(f"MIDI contains no notes: {midi_path}")
    return notes


def _shift_token(semitones: int) -> str:
    direction = "plus" if semitones > 0 else "minus"
    return f"{direction}-{abs(semitones):02d}"


def _derived_seed(seed: int, semitones: int) -> int:
    digest = hashlib.sha256(
        _canonical_json_bytes(["pitch-shift-grid/v1", seed, semitones])
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _aligned_frequency_score(
    source: np.ndarray,
    output: np.ndarray,
    semitones: int,
) -> float:
    if semitones > 0:
        source_aligned = source[:-semitones]
        output_aligned = output[semitones:]
    elif semitones < 0:
        source_aligned = source[-semitones:]
        output_aligned = output[:semitones]
    else:
        source_aligned = source
        output_aligned = output
    numerator = float(np.sum(source_aligned * output_aligned, dtype=np.float64))
    denominator = math.sqrt(
        float(np.sum(np.square(source_aligned), dtype=np.float64))
        * float(np.sum(np.square(output_aligned), dtype=np.float64))
    )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("pitch-shift QC has no finite spectral energy")
    return numerator / denominator


def _count_cqt_bins_below_frequency_limit(
    fmin_hz: float,
    frequency_limit_hz: float,
) -> int:
    """Count semitone-spaced centers strictly below a frequency limit."""

    if (
        not math.isfinite(fmin_hz)
        or not math.isfinite(frequency_limit_hz)
        or fmin_hz <= 0.0
        or frequency_limit_hz <= 0.0
    ):
        raise ValueError("CQT frequency bounds must be finite and positive")
    if fmin_hz >= frequency_limit_hz:
        return 0

    highest_index = math.floor(
        12.0 * math.log2(frequency_limit_hz / fmin_hz)
    )

    def center(index: int) -> float:
        return fmin_hz * (2.0 ** (index / 12.0))

    # Floating log2 can round an exact power-of-two ratio upward to the strict
    # boundary.  Correct both directions against the actual bin-center formula
    # used by the CQT instead of relying on floor/ceil alone.
    while highest_index >= 0 and center(highest_index) >= frequency_limit_hz:
        highest_index -= 1
    while center(highest_index + 1) < frequency_limit_hz:
        highest_index += 1
    return highest_index + 1


def _absolute_frequency_cqt(
    audio_path: os.PathLike,
    *,
    n_bins: Optional[int] = None,
    hop_length: int = 4096,
    fmin_hz: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    audio, sample_rate = sf.read(str(audio_path), always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    fmin = (
        float(librosa.note_to_hz("C1"))
        if fmin_hz is None
        else float(fmin_hz)
    )
    if not math.isfinite(fmin) or fmin <= 0.0:
        raise ValueError("pitch-shift QC fmin_hz must be finite and positive")
    maximum_bins = _count_cqt_bins_below_frequency_limit(
        fmin,
        sample_rate * 0.45,
    )
    if n_bins is not None and type(n_bins) is not int:
        raise TypeError("pitch-shift QC n_bins must be a built-in int")
    selected_bins = min(96, maximum_bins) if n_bins is None else n_bins
    if selected_bins <= 0 or selected_bins > maximum_bins:
        raise ValueError("invalid CQT frequency extent for pitch-shift QC")
    representation = np.log1p(
        np.abs(
            librosa.cqt(
                mono,
                sr=sample_rate,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=selected_bins,
                bins_per_octave=12,
            )
        )
    )
    return representation, {
        "sample_rate_hz": int(sample_rate),
        "sample_count": int(audio.shape[0]),
        "channel_count": int(audio.shape[1]),
        "n_bins": int(selected_bins),
        "hop_length": int(hop_length),
        "fmin_hz": fmin,
    }


def _pitch_grid_qc_plan(
    shifts: Tuple[int, ...],
    *,
    source_pitch_minimum: int,
    source_pitch_maximum: int,
    sample_rate: int,
) -> Dict[str, Any]:
    """Derive directed QC alternatives and a source-aware CQT extent."""

    requested_candidates = {0}
    for shift in shifts:
        requested_candidates.update((shift - 1, shift, shift + 1))

    maximum_cqt_hz = sample_rate * 0.45
    candidates = []
    clipped_candidates = []
    for candidate in sorted(requested_candidates):
        output_minimum = source_pitch_minimum + candidate
        output_maximum = source_pitch_maximum + candidate
        scoreable = (
            output_minimum >= MIDI_MINIMUM_PITCH
            and output_maximum <= MIDI_MAXIMUM_PITCH
            and float(librosa.midi_to_hz(output_maximum)) < maximum_cqt_hz
        )
        (candidates if scoreable else clipped_candidates).append(candidate)

    missing_required = sorted(set(shifts + (0,)) - set(candidates))
    if missing_required:
        raise ValueError(
            "sample rate and MIDI pitch extent cannot acoustically verify "
            f"required pitch shifts {missing_required} below 0.45 * sample_rate"
        )

    candidate_tuple = tuple(candidates)
    largest_shift = max(abs(value) for value in candidate_tuple)
    relevant_minimum = min(
        source_pitch_minimum,
        source_pitch_minimum + min(candidate_tuple),
    )
    relevant_maximum = max(
        source_pitch_maximum,
        source_pitch_maximum + max(candidate_tuple),
    )
    padding_semitones = 24 + largest_shift
    fmin_midi = max(
        MIDI_MINIMUM_PITCH,
        math.floor(relevant_minimum - padding_semitones),
    )
    desired_maximum_midi = min(
        MIDI_MAXIMUM_PITCH,
        math.ceil(relevant_maximum + padding_semitones),
    )
    fmin_hz = float(librosa.midi_to_hz(fmin_midi))
    maximum_bins = _count_cqt_bins_below_frequency_limit(
        fmin_hz,
        maximum_cqt_hz,
    )
    desired_bins = desired_maximum_midi - fmin_midi + 1
    n_bins = min(desired_bins, maximum_bins)
    maximum_cqt_midi_pitch = fmin_midi + n_bins - 1
    maximum_cqt_center_hz = fmin_hz * (2.0 ** ((n_bins - 1) / 12.0))
    if maximum_cqt_midi_pitch < relevant_maximum:
        raise ValueError(
            "derived CQT extent does not include the highest required MIDI "
            f"pitch {relevant_maximum}"
        )
    if n_bins <= 24 + largest_shift:
        raise ValueError(
            "sample rate and source pitch extent leave too few CQT bins for "
            "directed pitch-shift QC"
        )
    return {
        "candidates": candidate_tuple,
        "clipped_candidates": clipped_candidates,
        "candidate_policy": (
            "zero, every requested shift, and each requested shift +/- 1; "
            "alternatives outside MIDI 0..127 or above 0.45 * sample_rate "
            "are clipped, while requested shifts fail closed"
        ),
        "fmin_hz": fmin_hz,
        "fmin_midi": fmin_midi,
        "n_bins": n_bins,
        "maximum_cqt_hz": maximum_cqt_hz,
        "maximum_cqt_midi_pitch": maximum_cqt_midi_pitch,
        "maximum_cqt_center_hz": maximum_cqt_center_hz,
        "relevant_midi_pitch_extent": [relevant_minimum, relevant_maximum],
    }


def _pitch_shift_measurement_from_cqt(
    source_cqt: np.ndarray,
    output_cqt: np.ndarray,
    *,
    expected_semitones: int,
    candidates: Tuple[int, ...],
    n_bins: int,
    hop_length: int,
) -> Dict[str, Any]:
    largest_shift = max(abs(value) for value in candidates)
    if n_bins <= 24 + largest_shift:
        raise ValueError("sample rate is too low for absolute-frequency pitch QC")
    frame_count = min(source_cqt.shape[1], output_cqt.shape[1])
    if frame_count <= 0:
        raise ValueError("pitch-shift QC produced no CQT frames")
    source_cqt = source_cqt[:, :frame_count]
    output_cqt = output_cqt[:, :frame_count]
    scores = {
        value: _aligned_frequency_score(source_cqt, output_cqt, value)
        for value in candidates
    }
    best = max(scores, key=scores.__getitem__)
    alternatives = [
        score for value, score in scores.items() if value != expected_semitones
    ]
    margin = scores[expected_semitones] - max(alternatives)
    if best != expected_semitones or margin <= 1e-6:
        raise ValueError(
            "audio pitch shift is not uniquely identified as "
            f"{expected_semitones}: best={best}, margin={margin}"
        )
    return {
        "method": "nonwrapping_absolute_frequency_cqt_v1",
        "bins_per_octave": 12,
        "n_bins": n_bins,
        "hop_length": hop_length,
        "expected_semitones": expected_semitones,
        "best_semitones": best,
        "expected_score": scores[expected_semitones],
        "runner_up_margin": margin,
        "candidate_scores": {str(value): scores[value] for value in candidates},
    }


def measure_absolute_pitch_shift_v1(
    source_audio_path: os.PathLike,
    output_audio_path: os.PathLike,
    *,
    expected_semitones: int,
    candidate_semitones: Iterable[int],
    fmin_hz: Optional[float] = None,
    n_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure a directed shift on a non-wrapping absolute-frequency CQT.

    The standalone default starts at C1 and uses up to 96 bins.  Callers whose
    source or shifted content falls outside that span must provide an explicit
    ``fmin_hz`` and ``n_bins``; the grid materializer derives both from its
    source MIDI pitch extent and requested shifts.
    """

    if type(expected_semitones) is not int:
        raise TypeError("expected_semitones must be a built-in int")
    raw_candidates = tuple(candidate_semitones)
    if any(type(value) is not int for value in raw_candidates):
        raise TypeError("candidate semitone shifts must be built-in ints")
    candidates = tuple(sorted(set(raw_candidates) | {0, expected_semitones}))
    source_cqt, source_info = _absolute_frequency_cqt(
        source_audio_path,
        fmin_hz=fmin_hz,
        n_bins=n_bins,
    )
    output_cqt, output_info = _absolute_frequency_cqt(
        output_audio_path,
        n_bins=source_info["n_bins"],
        hop_length=source_info["hop_length"],
        fmin_hz=source_info["fmin_hz"],
    )
    if any(
        output_info[field] != source_info[field]
        for field in ("sample_rate_hz", "sample_count", "channel_count")
    ):
        raise ValueError("pitch-shift QC requires matching audio shape and sample rate")
    measurement = _pitch_shift_measurement_from_cqt(
        source_cqt,
        output_cqt,
        expected_semitones=expected_semitones,
        candidates=candidates,
        n_bins=source_info["n_bins"],
        hop_length=source_info["hop_length"],
    )
    measurement["fmin_hz"] = source_info["fmin_hz"]
    measurement["maximum_cqt_center_hz"] = source_info["fmin_hz"] * (
        2.0 ** ((source_info["n_bins"] - 1) / 12.0)
    )
    return measurement


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    with path.open("wb") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _new_staging_directory(target: Path) -> Path:
    """Create a sibling directory with ordinary umask-filtered permissions."""

    for _ in range(100):
        stage = target.parent / (
            f".{target.name}.staging-{secrets.token_hex(8)}"
        )
        try:
            os.mkdir(stage, 0o777)
        except FileExistsError:
            continue
        return stage
    raise FileExistsError(
        f"Unable to reserve a unique staging directory beside {target}"
    )


def _expected_inventory(manifest: Dict[str, Any]) -> set[str]:
    inventory = {"pitch-shift-grid-manifest.json"}
    for output in manifest["outputs"]:
        inventory.update(
            {
                output["audio_relpath"],
                output["midi_relpath"],
                output["provenance_relpath"],
            }
        )
    return inventory


def verify_pitch_shift_grid_v1(output_root: os.PathLike) -> Dict[str, Any]:
    """Verify a completed grid without needing to trust its filenames."""

    root = Path(output_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("pitch-shift grid root must be a non-symlink directory")
    manifest_path = root / "pitch-shift-grid-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pitch-shift grid manifest is missing or invalid") from exc
    expected_keys = {
        "schema_version",
        "kind",
        "seed",
        "semitones",
        "output_midi_pitch_bounds",
        "source",
        "outputs",
        "manifest_semantic_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("pitch-shift grid manifest has an invalid schema")
    if (
        manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
        or manifest["kind"] != MANIFEST_KIND
    ):
        raise ValueError("pitch-shift grid manifest identity is invalid")
    semantic_document = dict(manifest)
    semantic_digest = semantic_document.pop("manifest_semantic_sha256")
    if _require_sha256(
        semantic_digest, "manifest semantic digest"
    ) != _canonical_sha256(semantic_document):
        raise ValueError("pitch-shift grid manifest semantic digest mismatch")
    output_midi_bounds = manifest["output_midi_pitch_bounds"]
    if not isinstance(output_midi_bounds, dict) or set(output_midi_bounds) != {
        "minimum",
        "maximum",
    }:
        raise ValueError("pitch-shift grid output MIDI bounds are invalid")
    source = manifest["source"]
    expected_source_keys = {
        "audio_name",
        "midi_name",
        "audio_sha256",
        "midi_sha256",
        "sample_rate_hz",
        "sample_count",
        "channel_count",
        "note_count",
        "pitch_minimum",
        "pitch_maximum",
        "note_semantic_sha256",
    }
    if not isinstance(source, dict) or set(source) != expected_source_keys:
        raise ValueError("pitch-shift grid source record is invalid")
    _require_sha256(source["audio_sha256"], "source audio digest")
    _require_sha256(source["midi_sha256"], "source MIDI digest")
    source_semantic_sha = _require_sha256(
        source["note_semantic_sha256"], "source note semantic digest"
    )
    shifts = validate_pitch_shift_grid_v1(
        manifest["semitones"],
        source_pitch_minimum=source["pitch_minimum"],
        source_pitch_maximum=source["pitch_maximum"],
        minimum_midi_pitch=output_midi_bounds["minimum"],
        maximum_midi_pitch=output_midi_bounds["maximum"],
    )
    qc_plan = _pitch_grid_qc_plan(
        shifts,
        source_pitch_minimum=source["pitch_minimum"],
        source_pitch_maximum=source["pitch_maximum"],
        sample_rate=source["sample_rate_hz"],
    )
    expected_candidate_score_keys = {
        str(value) for value in qc_plan["candidates"]
    }
    outputs = manifest["outputs"]
    if not isinstance(outputs, list) or len(outputs) != len(shifts):
        raise ValueError("pitch-shift grid output count is invalid")
    actual_inventory = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_inventory != _expected_inventory(manifest):
        raise ValueError("pitch-shift grid file inventory mismatch")
    for expected_shift, output in zip(shifts, outputs):
        expected_output_keys = {
            "semitones",
            "seed",
            "audio_relpath",
            "midi_relpath",
            "provenance_relpath",
            "audio_sha256",
            "midi_sha256",
            "provenance_sha256",
            "acoustic_qc",
        }
        if not isinstance(output, dict) or set(output) != expected_output_keys:
            raise ValueError("pitch-shift grid output record is invalid")
        if output["semitones"] != expected_shift:
            raise ValueError("pitch-shift grid output order differs")
        paths = {
            label: root / output[f"{label}_relpath"]
            for label in ("audio", "midi", "provenance")
        }
        for label, path in paths.items():
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"pitch-shift grid {label} payload is invalid")
            if _sha256_file(path) != _require_sha256(
                output[f"{label}_sha256"], f"output {label} digest"
            ):
                raise ValueError(f"pitch-shift grid {label} digest mismatch")
        info = sf.info(str(paths["audio"]))
        if (
            info.samplerate != source["sample_rate_hz"]
            or info.frames != source["sample_count"]
            or info.channels != source["channel_count"]
        ):
            raise ValueError("pitch-shift grid audio dimensions differ from source")
        shifted_semantics = _note_semantics(paths["midi"], undo_shift=expected_shift)
        if _canonical_sha256(shifted_semantics) != source_semantic_sha:
            raise ValueError("pitch-shift grid MIDI is not synchronized")
        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        if (
            provenance.get("transform") != "pitch_shift_v1"
            or provenance.get("seed") != output["seed"]
            or provenance.get("parameters", {}).get("semitones") != expected_shift
            or provenance.get("parameters", {}).get("minimum_midi_pitch")
            != output_midi_bounds["minimum"]
            or provenance.get("parameters", {}).get("maximum_midi_pitch")
            != output_midi_bounds["maximum"]
            or provenance.get("output", {}).get("audio_sha256")
            != output["audio_sha256"]
            or provenance.get("output", {}).get("midi_sha256") != output["midi_sha256"]
        ):
            raise ValueError("pitch-shift grid provenance does not bind its payload")
        acoustic_qc = output["acoustic_qc"]
        if (
            not isinstance(acoustic_qc, dict)
            or acoustic_qc.get("expected_semitones") != expected_shift
            or acoustic_qc.get("best_semitones") != expected_shift
            or not float(acoustic_qc.get("runner_up_margin", 0.0)) > 0.0
            or not isinstance(acoustic_qc.get("candidate_scores"), dict)
            or set(acoustic_qc.get("candidate_scores", {}))
            != expected_candidate_score_keys
            or acoustic_qc.get("candidate_policy") != qc_plan["candidate_policy"]
            or acoustic_qc.get("clipped_candidate_semitones")
            != qc_plan["clipped_candidates"]
            or acoustic_qc.get("n_bins") != qc_plan["n_bins"]
            or acoustic_qc.get("cqt_fmin_midi") != qc_plan["fmin_midi"]
            or acoustic_qc.get("cqt_fmin_hz") != qc_plan["fmin_hz"]
            or acoustic_qc.get("cqt_frequency_limit_hz")
            != qc_plan["maximum_cqt_hz"]
            or acoustic_qc.get("cqt_maximum_midi_pitch")
            != qc_plan["maximum_cqt_midi_pitch"]
            or acoustic_qc.get("cqt_maximum_center_hz")
            != qc_plan["maximum_cqt_center_hz"]
            or acoustic_qc.get("relevant_midi_pitch_extent")
            != qc_plan["relevant_midi_pitch_extent"]
        ):
            raise ValueError("pitch-shift grid acoustic QC is invalid")
    return {
        "status": "pass",
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_semantic_sha256": semantic_digest,
        "output_count": len(outputs),
        "semitones": list(shifts),
    }


def materialize_pitch_shift_grid_v1(
    audio_path: os.PathLike,
    midi_path: os.PathLike,
    output_root: os.PathLike,
    *,
    seed: int,
    semitones: Sequence[int],
    minimum_midi_pitch: int = MIDI_MINIMUM_PITCH,
    maximum_midi_pitch: int = MIDI_MAXIMUM_PITCH,
) -> Dict[str, Any]:
    """Atomically render and verify every requested whole-recording shift."""

    _validate_seed(seed)
    source_audio = Path(audio_path)
    source_midi = Path(midi_path)
    if not source_audio.is_file() or source_audio.is_symlink():
        raise ValueError("source audio must be a non-symlink regular file")
    if not source_midi.is_file() or source_midi.is_symlink():
        raise ValueError("source MIDI must be a non-symlink regular file")
    source_semantics = _note_semantics(source_midi)
    source_pitches = [int(note[3]) for note in source_semantics]
    shifts = validate_pitch_shift_grid_v1(
        semitones,
        source_pitch_minimum=min(source_pitches),
        source_pitch_maximum=max(source_pitches),
        minimum_midi_pitch=minimum_midi_pitch,
        maximum_midi_pitch=maximum_midi_pitch,
    )
    target = Path(output_root)
    if os.path.lexists(str(target)):
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = _new_staging_directory(target)
    try:
        source_info = sf.info(str(source_audio))
        qc_plan = _pitch_grid_qc_plan(
            shifts,
            source_pitch_minimum=min(source_pitches),
            source_pitch_maximum=max(source_pitches),
            sample_rate=int(source_info.samplerate),
        )
        qc_candidates = qc_plan["candidates"]
        source_cqt, source_cqt_info = _absolute_frequency_cqt(
            source_audio,
            n_bins=qc_plan["n_bins"],
            fmin_hz=qc_plan["fmin_hz"],
        )
        output_records = []
        for shift in shifts:
            token = _shift_token(shift)
            output_audio = stage / f"pitch-{token}.wav"
            output_midi = stage / f"pitch-{token}.mid"
            output_provenance = stage / f"pitch-{token}.provenance.json"
            item_seed = _derived_seed(seed, shift)
            provenance = pitch_shift_v1(
                source_audio,
                source_midi,
                output_audio,
                output_midi,
                seed=item_seed,
                parameters=PitchShiftParameters(
                    semitones=shift,
                    minimum_midi_pitch=minimum_midi_pitch,
                    maximum_midi_pitch=maximum_midi_pitch,
                ),
                provenance_path=output_provenance,
            )
            output_cqt, output_cqt_info = _absolute_frequency_cqt(
                output_audio,
                n_bins=source_cqt_info["n_bins"],
                hop_length=source_cqt_info["hop_length"],
                fmin_hz=source_cqt_info["fmin_hz"],
            )
            if any(
                output_cqt_info[field] != source_cqt_info[field]
                for field in ("sample_rate_hz", "sample_count", "channel_count")
            ):
                raise ValueError(
                    "pitch-shift QC requires matching audio shape and sample rate"
                )
            acoustic_qc = _pitch_shift_measurement_from_cqt(
                source_cqt,
                output_cqt,
                expected_semitones=shift,
                candidates=qc_candidates,
                n_bins=source_cqt_info["n_bins"],
                hop_length=source_cqt_info["hop_length"],
            )
            acoustic_qc.update(
                {
                    "candidate_policy": qc_plan["candidate_policy"],
                    "clipped_candidate_semitones": qc_plan["clipped_candidates"],
                    "cqt_fmin_hz": qc_plan["fmin_hz"],
                    "cqt_fmin_midi": qc_plan["fmin_midi"],
                    "cqt_frequency_limit_hz": qc_plan["maximum_cqt_hz"],
                    "cqt_maximum_midi_pitch": qc_plan[
                        "maximum_cqt_midi_pitch"
                    ],
                    "cqt_maximum_center_hz": qc_plan[
                        "maximum_cqt_center_hz"
                    ],
                    "relevant_midi_pitch_extent": qc_plan[
                        "relevant_midi_pitch_extent"
                    ],
                }
            )
            output_records.append(
                {
                    "semitones": shift,
                    "seed": item_seed,
                    "audio_relpath": output_audio.name,
                    "midi_relpath": output_midi.name,
                    "provenance_relpath": output_provenance.name,
                    "audio_sha256": provenance["output"]["audio_sha256"],
                    "midi_sha256": provenance["output"]["midi_sha256"],
                    "provenance_sha256": _sha256_file(output_provenance),
                    "acoustic_qc": acoustic_qc,
                }
            )
        manifest: Dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "seed": seed,
            "semitones": list(shifts),
            "output_midi_pitch_bounds": {
                "minimum": minimum_midi_pitch,
                "maximum": maximum_midi_pitch,
            },
            "source": {
                "audio_name": source_audio.name,
                "midi_name": source_midi.name,
                "audio_sha256": _sha256_file(source_audio),
                "midi_sha256": _sha256_file(source_midi),
                "sample_rate_hz": int(source_info.samplerate),
                "sample_count": int(source_info.frames),
                "channel_count": int(source_info.channels),
                "note_count": len(source_semantics),
                "pitch_minimum": min(source_pitches),
                "pitch_maximum": max(source_pitches),
                "note_semantic_sha256": _canonical_sha256(source_semantics),
            },
            "outputs": output_records,
        }
        manifest["manifest_semantic_sha256"] = _canonical_sha256(manifest)
        _write_manifest(stage / "pitch-shift-grid-manifest.json", manifest)
        verify_pitch_shift_grid_v1(stage)
        if os.path.lexists(str(target)):
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
        os.rename(stage, target)
        return verify_pitch_shift_grid_v1(target)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
