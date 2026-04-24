"""Dataset layout helpers.

Augmented output is kept strictly separate from original files. Every dataset
directory produced by the toolkit looks like:

    <dataset_dir>/
        original/   # pristine audio/MIDI pairs the user supplied
        augmented/  # every output of the augmentation pipeline

Keeping the originals in their own subfolder means downstream users can delete
``augmented/`` to reset the dataset without losing any source material.
"""
import os
import shutil
from typing import Iterable, List, Tuple


ORIGINAL_SUBDIR = "original"
AUGMENTED_SUBDIR = "augmented"

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".aiff")

_AUG_MARKER = "_augmented_"


def subdirs(dataset_dir: str) -> Tuple[str, str]:
    """Return ``(originals_dir, augmented_dir)`` for a dataset directory."""
    return (
        os.path.join(dataset_dir, ORIGINAL_SUBDIR),
        os.path.join(dataset_dir, AUGMENTED_SUBDIR),
    )


def ensure_subdirs(dataset_dir: str) -> Tuple[str, str]:
    """Create the standard subdirs if missing and return their paths."""
    originals_dir, augmented_dir = subdirs(dataset_dir)
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(augmented_dir, exist_ok=True)
    return originals_dir, augmented_dir


def _iter_original_pairs(source_dir: str) -> Iterable[Tuple[str, str]]:
    """Yield ``(audio_filename, midi_filename)`` pairs of non-augmented originals.

    A pair is yielded only when both files exist at the top level of ``source_dir``.
    """
    try:
        entries = os.listdir(source_dir)
    except FileNotFoundError:
        return
    names = set(entries)
    for name in entries:
        lower = name.lower()
        if _AUG_MARKER in lower:
            continue
        if not lower.endswith(AUDIO_EXTS):
            continue
        stem, _ = os.path.splitext(name)
        midi = f"{stem}.mid"
        if midi in names:
            yield name, midi


def stage_originals(input_dir: str, originals_dir: str) -> List[str]:
    """Place every original audio/MIDI pair from ``input_dir`` into ``originals_dir``.

    If ``input_dir`` and ``originals_dir`` resolve to the same path (which is
    impossible: ``originals_dir`` is always a child), or if ``input_dir`` is
    the parent of ``originals_dir`` (the default same-dir pipeline layout),
    pairs are **moved**. Otherwise they are **copied** so the user's input
    directory is left untouched.

    Pairs already present at the top level of ``originals_dir`` (e.g. from a
    previous run) are left alone. Returns the list of original audio
    basenames now in ``originals_dir``.
    """
    os.makedirs(originals_dir, exist_ok=True)

    # Move if originals_dir sits directly inside input_dir (the default
    # same-directory case: `--output-directory` not set). Otherwise copy.
    input_abs = os.path.abspath(input_dir)
    originals_abs = os.path.abspath(originals_dir)
    same_tree = os.path.dirname(originals_abs) == input_abs

    transfer = shutil.move if same_tree else shutil.copy2

    existing = set(os.listdir(originals_dir))

    for audio_name, midi_name in _iter_original_pairs(input_dir):
        src_audio = os.path.join(input_dir, audio_name)
        src_midi = os.path.join(input_dir, midi_name)
        dst_audio = os.path.join(originals_dir, audio_name)
        dst_midi = os.path.join(originals_dir, midi_name)

        if audio_name not in existing and os.path.exists(src_audio):
            transfer(src_audio, dst_audio)
        if midi_name not in existing and os.path.exists(src_midi):
            transfer(src_midi, dst_midi)

    return [
        f
        for f in os.listdir(originals_dir)
        if f.lower().endswith(AUDIO_EXTS) and _AUG_MARKER not in f.lower()
    ]
