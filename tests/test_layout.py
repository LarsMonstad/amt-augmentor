"""Tests for amt_augmentor.layout — dataset directory staging."""
import os

from amt_augmentor.layout import (
    AUGMENTED_SUBDIR,
    ORIGINAL_SUBDIR,
    ensure_subdirs,
    stage_originals,
    subdirs,
)


def _touch(path: str, content: bytes = b"\x00") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def test_ensure_subdirs_creates_both(tmp_path):
    ds = tmp_path / "ds"
    orig, aug = ensure_subdirs(str(ds))
    assert os.path.isdir(orig)
    assert os.path.isdir(aug)
    assert os.path.basename(orig) == ORIGINAL_SUBDIR
    assert os.path.basename(aug) == AUGMENTED_SUBDIR


def test_subdirs_returns_paths_without_creating(tmp_path):
    ds = tmp_path / "ds"
    orig, aug = subdirs(str(ds))
    # Nothing on disk yet.
    assert not os.path.exists(orig)
    assert not os.path.exists(aug)


def test_stage_originals_moves_when_input_equals_output(tmp_path):
    """Default pipeline case: input_dir == output_dir. Files should move, not copy."""
    ds = tmp_path / "ds"
    ds.mkdir()
    _touch(str(ds / "song1.wav"))
    _touch(str(ds / "song1.mid"))
    _touch(str(ds / "song2.wav"))
    _touch(str(ds / "song2.mid"))
    # Augmented-looking leftover from an old run — must NOT be treated as original.
    _touch(str(ds / "song1_augmented_timestretch_1.2.wav"))

    originals_dir, _ = ensure_subdirs(str(ds))
    staged = stage_originals(str(ds), originals_dir)

    assert sorted(staged) == ["song1.wav", "song2.wav"]
    # Files should have moved (no longer at top level).
    assert not (ds / "song1.wav").exists()
    assert not (ds / "song1.mid").exists()
    assert (ds / ORIGINAL_SUBDIR / "song1.wav").exists()
    assert (ds / ORIGINAL_SUBDIR / "song1.mid").exists()
    # The augmented leftover must stay exactly where it was.
    assert (ds / "song1_augmented_timestretch_1.2.wav").exists()
    assert not (ds / ORIGINAL_SUBDIR / "song1_augmented_timestretch_1.2.wav").exists()


def test_stage_originals_copies_when_dirs_differ(tmp_path):
    """When input_dir and output_dir are distinct, originals are COPIED."""
    inp = tmp_path / "in"
    out = tmp_path / "out"
    inp.mkdir()
    out.mkdir()
    _touch(str(inp / "s.wav"))
    _touch(str(inp / "s.mid"))

    originals_dir, _ = ensure_subdirs(str(out))
    stage_originals(str(inp), originals_dir)

    # User's input dir must be untouched.
    assert (inp / "s.wav").exists()
    assert (inp / "s.mid").exists()
    assert (out / ORIGINAL_SUBDIR / "s.wav").exists()
    assert (out / ORIGINAL_SUBDIR / "s.mid").exists()


def test_stage_originals_ignores_unpaired_files(tmp_path):
    """Audio without a matching .mid (or vice-versa) is not staged."""
    ds = tmp_path / "ds"
    ds.mkdir()
    _touch(str(ds / "paired.wav"))
    _touch(str(ds / "paired.mid"))
    _touch(str(ds / "orphan_audio.wav"))  # no .mid
    _touch(str(ds / "orphan_midi.mid"))  # no audio

    originals_dir, _ = ensure_subdirs(str(ds))
    staged = stage_originals(str(ds), originals_dir)

    assert staged == ["paired.wav"]
    assert (ds / "orphan_audio.wav").exists()
    # The unmatched files stay at the top level; only the paired ones move.
    assert (ds / "orphan_midi.mid").exists()


def test_stage_originals_is_idempotent(tmp_path):
    """Running twice must not lose or duplicate files."""
    ds = tmp_path / "ds"
    ds.mkdir()
    _touch(str(ds / "a.wav"))
    _touch(str(ds / "a.mid"))

    originals_dir, _ = ensure_subdirs(str(ds))
    stage_originals(str(ds), originals_dir)
    stage_originals(str(ds), originals_dir)

    assert sorted(os.listdir(originals_dir)) == ["a.mid", "a.wav"]
