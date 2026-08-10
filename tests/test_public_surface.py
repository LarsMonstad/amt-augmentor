"""The current package must expose only the supported augmentation surface."""

import sys
from pathlib import Path
from unittest.mock import patch

import amt_augmentor

from amt_augmentor.config import Config, save_default_config
from amt_augmentor.main import main


REJECTED_PUBLIC_NAMES = {
    "DropNoteIsolatedPolicy",
    "PauseInsertPolicy",
    "PauseInsertPolicyV2",
    "drop_note_isolated_v1",
    "pause_insert_v1",
    "pause_insert_v2",
}


def test_package_exports_only_supported_transform_families():
    assert amt_augmentor.__version__ == "2.0.0a5"
    assert REJECTED_PUBLIC_NAMES.isdisjoint(amt_augmentor.__all__)
    for name in REJECTED_PUBLIC_NAMES:
        assert not hasattr(amt_augmentor, name)


def test_rejected_modules_and_console_entry_are_absent():
    package_directory = Path(amt_augmentor.__file__).parent
    assert not (package_directory / "research_augmentations.py").exists()
    assert not (package_directory / "galdr_campaign.py").exists()
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "amt-augmentor-galdr-campaign" not in project


def test_legacy_range_masker_is_disabled_and_not_advertised(capsys, tmp_path):
    assert Config().add_pause.enabled is False
    generated_config = tmp_path / "config.yaml"
    save_default_config(str(generated_config))
    configuration = generated_config.read_text(encoding="utf-8")
    assert "\nadd_pause:" not in configuration
    with patch.object(sys, "argv", ["amt-augmentor", "--list-effects"]):
        main()
    output = capsys.readouterr().out.lower()
    assert "pause" not in output
    assert "dropnote" not in output
