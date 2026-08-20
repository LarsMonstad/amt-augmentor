"""Cross-platform checks for publication durability barriers."""

from __future__ import annotations

import os

import amt_augmentor._paired_io as paired_io
import amt_augmentor.galdr_conventional_campaign as campaign


def test_file_fsync_uses_windows_compatible_descriptor(monkeypatch):
    opened = []
    synced = []
    closed = []
    binary_flag = 0x8000

    monkeypatch.setattr(paired_io.os, "O_BINARY", binary_flag, raising=False)
    monkeypatch.setattr(
        paired_io.os,
        "open",
        lambda path, flags: opened.append((path, flags)) or 123,
    )
    monkeypatch.setattr(paired_io.os, "fsync", synced.append)
    monkeypatch.setattr(paired_io.os, "close", closed.append)

    paired_io._fsync_path("payload.bin")

    assert opened == [("payload.bin", os.O_RDWR | binary_flag)]
    assert synced == [123]
    assert closed == [123]


def test_directory_fsync_is_skipped_on_windows(tmp_path, monkeypatch):
    def unexpected_open(*args, **kwargs):
        raise AssertionError("Windows directory fsync must not call os.open")

    monkeypatch.setattr(paired_io.os, "name", "nt")
    monkeypatch.setattr(paired_io.os, "open", unexpected_open)

    paired_io._fsync_directory(tmp_path)


def test_campaign_uses_shared_portable_fsync_helpers():
    assert campaign._fsync_path is paired_io._fsync_path
    assert campaign._fsync_directory is paired_io._fsync_directory
