"""Fixture tests for the deterministic Galdr conventional adapter."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from amt_augmentor.galdr_conventional_campaign import (
    CampaignError,
    VIEW_SLOTS,
    _item_seed,
    _planned_parameters,
    build_plan,
    main,
    materialize_plan,
    verify_materialization,
    write_plan,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path):
    root = tmp_path / "dataset"
    audio_path = root / "data" / "raw" / "audio" / "train.wav"
    midi_path = root / "data" / "raw" / "midi" / "train.mid"
    audio_path.parent.mkdir(parents=True)
    midi_path.parent.mkdir(parents=True)
    sample_rate = 44100
    sample_count = sample_rate
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    audio = 0.15 * np.sin(2.0 * np.pi * 220.0 * time)
    sf.write(audio_path, audio, sample_rate, subtype="PCM_16")

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=9600)
    instrument = pretty_midi.Instrument(program=40, name="fixture")
    instrument.notes.extend(
        [
            pretty_midi.Note(velocity=80, pitch=60, start=0.10, end=0.60),
            pretty_midi.Note(velocity=82, pitch=64, start=0.20, end=0.70),
            pretty_midi.Note(velocity=84, pitch=60, start=0.40, end=0.80),
        ]
    )
    midi.instruments.append(instrument)
    midi.write(str(midi_path))
    serialized = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted(
        serialized.instruments[0].notes,
        key=lambda note: (note.start, note.end, note.pitch),
    )
    normalized_values = [
        {
            "start_seconds": float(note.start),
            "end_seconds": float(note.end),
            "pitch": int(note.pitch),
            "velocity": int(note.velocity),
        }
        for note in notes
    ]
    normalized_values[0]["end_seconds"] = normalized_values[2]["start_seconds"]
    training_id = hashlib.sha256(b"training-fixture").hexdigest()
    validation_id = hashlib.sha256(b"validation-fixture").hexdigest()
    source = {
        "adjusted_previous_offsets": 1,
        "annotation_semantic_sha256": _canonical_sha256(normalized_values),
        "audio_bytes": audio_path.stat().st_size,
        "audio_relpath": audio_path.relative_to(root).as_posix(),
        "audio_sha256": _sha256(audio_path),
        "channel_count": 1,
        "duration_seconds": 1.0,
        "normalized_note_count": 3,
        "raw_midi_bytes": midi_path.stat().st_size,
        "raw_midi_relpath": midi_path.relative_to(root).as_posix(),
        "raw_midi_sha256": _sha256(midi_path),
        "raw_note_count": 3,
        "removed_nonpositive_after_truncation": 0,
        "sample_count": sample_count,
        "sample_rate_hz": sample_rate,
        "source_id": training_id,
        "split": "train",
        "tune_key": "fixture-tune",
    }
    provenance = {
        "schema_version": 2,
        "kind": "galdr_clean_original_index_bundle",
        "original_only": True,
        "augmentation_views": 0,
        "canonical_media_mutated": False,
        "dataset_identity_sha256": hashlib.sha256(b"fixture-dataset").hexdigest(),
        "normalization": {
            "policy": "same_pitch_overlap_truncate_previous_v1",
            "canonical_raw_midi_unchanged": True,
        },
        "sources": [
            source,
            {
                "source_id": validation_id,
                "split": "validation",
                "audio_relpath": "deliberately/missing.wav",
                "raw_midi_relpath": "deliberately/missing.mid",
            },
        ],
    }
    provenance_path = tmp_path / "dataset-provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "root": root,
        "audio": audio_path,
        "midi": midi_path,
        "source": source,
        "training_id": training_id,
        "validation_id": validation_id,
        "provenance": provenance_path,
        "provenance_sha256": _sha256(provenance_path),
    }


def _plan(fixture, *, source_ids=None):
    return build_plan(
        clean_provenance=fixture["provenance"],
        expected_clean_provenance_sha256=fixture["provenance_sha256"],
        dataset_root=fixture["root"],
        global_seed=50131,
        source_ids=source_ids,
    )


def test_plan_is_training_only_deterministic_and_explicit(tmp_path):
    fixture = _fixture(tmp_path)
    first = _plan(fixture)
    second = _plan(fixture)
    assert first == second
    assert first["conditions"] == ["O", "C"]
    assert first["source_selection"]["excluded_split_counts"] == {
        "validation": 1,
        "test": 0,
    }
    assert [source["source_id"] for source in first["sources"]] == [
        fixture["training_id"]
    ]
    slots = first["sources"][0]["view_slots"]
    assert [slot["transform"] for slot in slots] == [
        "gain_chorus_v1",
        "noise_snr_v1",
        "reverb_filters_v1",
        "pitch_shift_v1",
        "time_stretch_v1",
    ]
    assert all(type(slot["seed"]) is int and slot["parameters"] for slot in slots)
    first_path = tmp_path / "first-plan.json"
    second_path = tmp_path / "second-plan.json"
    assert write_plan(first_path, first) == write_plan(second_path, second)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_conventional_stream_replays_the_accepted_projection():
    source_id = "00d511eb969abd4a5b6a6689a3a59ff2b91f38217e3943ad8d4150375a9af7d4"
    expected = [
        (
            3047662574,
            {
                "chorus_centre_delay_ms": 7.0,
                "chorus_depth": 0.3,
                "chorus_feedback": 0.2,
                "chorus_mix": 0.25,
                "chorus_rate_hz": 0.6,
                "gain_db": 6.0,
            },
        ),
        (2678720089, {"target_snr_db": 20.0}),
        (
            3671442444,
            {
                "dry_level": 0.9,
                "highpass_hz": 30.0,
                "lowpass_hz": 15000.0,
                "room_size": 0.35,
                "wet_level": 0.2,
            },
        ),
        (
            514717432,
            {
                "maximum_midi_pitch": 108,
                "minimum_midi_pitch": 21,
                "semitones": -2,
            },
        ),
        (1140868471, {"rate": 0.9}),
    ]
    observed = []
    for view_slot, transform in VIEW_SLOTS:
        observed.append(
            (
                _item_seed(424242, source_id, view_slot, transform),
                _planned_parameters(
                    424242,
                    source_id,
                    view_slot,
                    transform,
                    sample_rate=44100,
                    minimum_pitch=40,
                    maximum_pitch=90,
                ),
            )
        )
    assert observed == expected


def test_explicit_nontraining_source_is_rejected_before_media_access(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(CampaignError, match="non-training"):
        _plan(fixture, source_ids=[fixture["validation_id"]])


def test_materialization_is_complete_galdr_compatible_and_worker_invariant(tmp_path):
    fixture = _fixture(tmp_path)
    raw_audio = fixture["audio"].read_bytes()
    raw_midi = fixture["midi"].read_bytes()
    plan = _plan(fixture)
    plan_path = tmp_path / "plan.json"
    write_plan(plan_path, plan)
    cli_plan_path = tmp_path / "cli-plan.json"
    assert (
        main(
            [
                "plan",
                "--dataset-provenance",
                str(fixture["provenance"]),
                "--expected-provenance-sha256",
                fixture["provenance_sha256"],
                "--dataset-root",
                str(fixture["root"]),
                "--global-seed",
                "50131",
                "--output-plan",
                str(cli_plan_path),
            ]
        )
        == 0
    )
    assert cli_plan_path.read_bytes() == plan_path.read_bytes()

    first_root = tmp_path / "artifact-one"
    second_root = tmp_path / "artifact-two"
    materialize_plan(
        plan_path=plan_path,
        clean_provenance=fixture["provenance"],
        dataset_root=fixture["root"],
        output_root=first_root,
        workers=1,
    )
    assert (
        main(
            [
                "materialize",
                "--plan",
                str(cli_plan_path),
                "--dataset-provenance",
                str(fixture["provenance"]),
                "--dataset-root",
                str(fixture["root"]),
                "--output-root",
                str(second_root),
                "--workers",
                "3",
            ]
        )
        == 0
    )
    assert main(["verify", "--output-root", str(second_root)]) == 0
    first_report = verify_materialization(first_root)
    second_report = verify_materialization(second_root)
    assert first_report == second_report
    assert fixture["audio"].read_bytes() == raw_audio
    assert fixture["midi"].read_bytes() == raw_midi

    first_inventory = {
        path.relative_to(first_root).as_posix(): _sha256(path)
        for path in first_root.rglob("*")
        if path.is_file()
    }
    second_inventory = {
        path.relative_to(second_root).as_posix(): _sha256(path)
        for path in second_root.rglob("*")
        if path.is_file()
    }
    assert first_inventory == second_inventory
    conditions = {row["condition"]: row for row in first_report["conditions"]}
    assert conditions["O"]["recording_count"] == 1
    assert conditions["C"]["recording_count"] == 6
    for condition in ("O", "C"):
        lineage = json.loads(
            (first_root / "conditions" / condition / "training-lineage.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(lineage) == {"lineage_schema_version", "kind", "records"}
        assert {record["view_kind"] for record in lineage["records"]} <= {
            "original",
            "augmented",
        }
        for record in lineage["records"]:
            assert set(record) == {
                "audio_filename",
                "source_id",
                "view_id",
                "view_kind",
                "source_duration_seconds",
                "view_duration_seconds",
            }

    normalized_midi = pretty_midi.PrettyMIDI(
        str(
            first_root
            / "media"
            / "original"
            / fixture["training_id"]
            / "00-original.mid"
        )
    )
    same_pitch = sorted(
        (note for note in normalized_midi.instruments[0].notes if note.pitch == 60),
        key=lambda note: note.start,
    )
    assert same_pitch[0].end == pytest.approx(same_pitch[1].start, abs=3e-5)

    with (first_root / "derivatives.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames == [
        "example_id",
        "source_id",
        "tune_key",
        "split",
        "audio_sha256",
        "midi_sha256",
    ]
    assert len(rows) == 6
    assert {row["split"] for row in rows} == {"train"}
    assert {row["source_id"] for row in rows} == {fixture["training_id"]}


def test_refuses_overwrite_symlinks_and_tampered_inputs(tmp_path):
    fixture = _fixture(tmp_path)
    plan = _plan(fixture)
    plan_path = tmp_path / "plan.json"
    write_plan(plan_path, plan)
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(CampaignError, match="overwrite"):
        materialize_plan(
            plan_path=plan_path,
            clean_provenance=fixture["provenance"],
            dataset_root=fixture["root"],
            output_root=existing,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    root_link = tmp_path / "dataset-link"
    root_link.symlink_to(fixture["root"], target_is_directory=True)
    with pytest.raises(CampaignError, match="symbolic"):
        build_plan(
            clean_provenance=fixture["provenance"],
            expected_clean_provenance_sha256=fixture["provenance_sha256"],
            dataset_root=root_link,
            global_seed=1,
        )

    fixture["audio"].write_bytes(raw := fixture["audio"].read_bytes() + b"tamper")
    with pytest.raises(CampaignError, match="size mismatch"):
        build_plan(
            clean_provenance=fixture["provenance"],
            expected_clean_provenance_sha256=fixture["provenance_sha256"],
            dataset_root=fixture["root"],
            global_seed=1,
        )
    assert raw.endswith(b"tamper")
