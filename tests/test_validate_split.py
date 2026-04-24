"""Regression tests for amt_augmentor.validate_split.

These guard against the false-negative contamination bug: filenames containing
embedded extension-like fragments (e.g. ``foo.mid_cleaned``) previously bypassed
the cross-split check because originals and augmented rows were canonicalized
differently.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from amt_augmentor.validate_split import (
    canonical_stem,
    find_contamination,
    is_augmented_version,
    original_from_aug,
    validate_dataset_split,
)


FIXTURES = Path(__file__).parent / "fixtures" / "contamination"


# ---------- canonical_stem unit tests ----------

@pytest.mark.parametrize("name,expected", [
    ("foo.wav", "foo"),
    ("foo.mid", "foo"),
    ("foo.MIDI", "foo"),
    ("path/to/foo.mid", "foo"),
    ("foo.mid_cleaned.mid", "foo.mid_cleaned"),  # embedded .mid must survive
    ("foo.mid_cleaned", "foo.mid_cleaned"),      # no trailing known ext
    ("foo_augmented_timestretch_1.2.mid", "foo_augmented_timestretch_1.2"),
    ("foo_augmented_timestretch_1.2", "foo_augmented_timestretch_1.2"),
    ("no_extension", "no_extension"),
])
def test_canonical_stem(name, expected):
    assert canonical_stem(name) == expected


def test_is_augmented_version():
    assert is_augmented_version("foo_augmented_timestretch_1.2.mid")
    assert is_augmented_version("Foo_AUGMENTED_pitchshift_-3")
    assert not is_augmented_version("foo_augment_1.mid")  # partial match
    assert not is_augmented_version("original_song.mid")


def test_original_from_aug_embedded_mid_cleaned():
    aug = "Peisestugu_..._13-43-39.mid_cleaned_augmented_timestretch_1.2.mid"
    # The source original's canonical stem has the embedded .mid_cleaned intact.
    assert original_from_aug(aug) == "Peisestugu_..._13-43-39.mid_cleaned"


# ---------- fixture-driven integration tests ----------

def _load_expected(stem: str) -> dict:
    with open(FIXTURES / f"{stem}.expected.json") as f:
        return json.load(f)


def test_clean_fixture():
    report = find_contamination(str(FIXTURES / "clean.csv"))
    expected = _load_expected("clean")
    assert report["clean"] is expected["clean"]
    assert report["cross_split"] == []
    assert report["aug_in_eval"] == []
    assert report["orphan_aug"] == []


def test_simple_leak_fixture():
    report = find_contamination(str(FIXTURES / "simple_leak.csv"))
    assert not report["clean"]
    assert len(report["cross_split"]) == 1
    item = report["cross_split"][0]
    assert item["original"] == "FooBar"
    assert item["original_split"] == "test"
    assert item["split"] == "train"


def test_embedded_mid_cleaned_fixture():
    """The heart of the bug report: filenames with embedded .mid_cleaned."""
    report = find_contamination(str(FIXTURES / "embedded_mid_cleaned.csv"))
    assert not report["clean"]
    assert len(report["cross_split"]) == 2
    for item in report["cross_split"]:
        assert item["original_split"] == "test"
        assert item["split"] == "train"
        assert item["original"].endswith(".mid_cleaned")


def test_unicode_case_fixture():
    report = find_contamination(str(FIXTURES / "unicode_case.csv"))
    assert not report["clean"]
    assert len(report["cross_split"]) == 1
    item = report["cross_split"][0]
    assert "Godværsdagen_Øst" in item["original"]


def test_orphan_aug_fixture():
    report = find_contamination(str(FIXTURES / "orphan_aug.csv"))
    assert not report["clean"]
    assert len(report["orphan_aug"]) == 1
    assert report["orphan_aug"][0]["missing_original"] == "Ghost"
    assert report["cross_split"] == []


def test_reverse_leak_fixture():
    """Augmented in validation, original in train — should be flagged."""
    report = find_contamination(str(FIXTURES / "reverse_leak.csv"))
    assert not report["clean"]
    assert len(report["cross_split"]) == 1
    assert len(report["aug_in_eval"]) == 1
    assert report["aug_in_eval"][0]["split"] == "validation"


def test_subfolder_layout_fixture():
    """Validator must handle paths like ds/original/... and ds/augmented/..."""
    report = find_contamination(str(FIXTURES / "subfolder_layout.csv"))
    # CleanSong rows are aligned (train/train). FooBar has a cross-split leak.
    assert not report["clean"]
    assert len(report["cross_split"]) == 1
    item = report["cross_split"][0]
    assert item["original"] == "FooBar"
    assert item["original_split"] == "test"
    assert item["split"] == "train"


def test_validate_dataset_split_returns_bool(capsys):
    assert validate_dataset_split(str(FIXTURES / "clean.csv")) is True
    assert validate_dataset_split(str(FIXTURES / "simple_leak.csv")) is False


def test_validate_dataset_split_strict_raises():
    with pytest.raises(SystemExit) as exc:
        validate_dataset_split(str(FIXTURES / "simple_leak.csv"), strict=True)
    assert exc.value.code == 1


# ---------- CLI exit code test ----------

def _run_module(*args):
    return subprocess.run(
        [sys.executable, "-m", "amt_augmentor.validate_split", *args],
        capture_output=True,
        text=True,
    )


def test_cli_exit_code_clean():
    result = _run_module(str(FIXTURES / "clean.csv"), "--strict")
    assert result.returncode == 0


def test_cli_exit_code_strict_fail():
    result = _run_module(str(FIXTURES / "simple_leak.csv"), "--strict")
    assert result.returncode == 1


def test_cli_exit_code_non_strict_does_not_fail():
    # Without --strict we return 0 even when contamination is found.
    result = _run_module(str(FIXTURES / "simple_leak.csv"))
    assert result.returncode == 0


def test_cli_json_output():
    result = _run_module(str(FIXTURES / "embedded_mid_cleaned.csv"), "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["clean"] is False
    assert len(payload["cross_split"]) == 2
