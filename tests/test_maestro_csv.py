import os
import csv
import numpy as np
import soundfile as sf
import pytest
from collections import defaultdict

# Import the functions we wanna test
from amt_augmentor.create_maestro_csv import (
    is_augmented_version,
    get_original_song_name,
    get_split_status,
    get_wav_duration,
    create_song_list,
)

# -------------------------------
# Tests for helper functions
# -------------------------------

# Test if a filename is marked as augmented
@pytest.mark.parametrize("filename, expected", [
    ("song.wav", False),
    ("song.mid", False),
    ("song_augmented_timestretch_aug.mid", True),
    ("song_augmented_pitchshift_extra.mid", True),
    ("song_augmented_reverb_filters_test.mid", True),
    ("song_augmented_gain_chorus_final.mid", True),
    ("song_augmented_addpauses_demo.mid", True),
    ("SONG_AUGMENTED_PITCHSHIFT_EXTRA.MID", True),  # checking case insensitivity
    # Also test old format (should not be recognized as augmented anymore)
    ("song_timestretch_aug.mid", False),
    ("song_pitchshift_extra.mid", False),
])
def test_is_augmented_version(filename, expected):
    # Let's see if this works as expected...
    assert is_augmented_version(filename) == expected


# Test extracting the original song name from filenames with augment markers.
@pytest.mark.parametrize("filename, expected", [
    ("song_augmented_timestretch_aug.mid", "song"),
    ("song_augmented_pitchshift_extra.mid", "song"),
    ("song_augmented_reverb_filters_test.mid", "song"),
    ("song_augmented_gain_chorus_final.mid", "song"),
    ("song_augmented_addpauses_demo.mid", "song"),
    ("song.mid", "song"),
    ("my.song.mid", "my.song"),
    # Test that old format returns the full name (no extraction)
    ("song_timestretch_aug.mid", "song_timestretch_aug"),
    ("song_pitchshift_extra.mid", "song_pitchshift_extra"),
])
def test_get_original_song_name(filename, expected):
    # Hopefully this returns the right base name...
    assert get_original_song_name(filename) == expected


# Test the split assignment when there are no songs yet.
def test_get_split_status_empty():
    songs = {}
    ratios = {'train': 0.7, 'test': 0.15, 'validation': 0.15}
    new_split = get_split_status(songs, "new_song", ratios)
    # By default, should go to 'train'
    assert new_split == "train"


# Test the split assignment when we already have some songs.
def test_get_split_status_nonempty():
    # Dummy existing song splits
    songs = {"song1": "train", "song2": "test", "song3": "train"}
    ratios = {'train': 0.5, 'test': 0.3, 'validation': 0.2}
    # With 3 songs, adding a new one makes total 4.
    # Expected targets: train = 2, test = 1, validation = 0.
    # So the new song should get 'train' (ties break to first key).
    new_split = get_split_status(songs, "song4", ratios)
    assert new_split == "train"


# -------------------------------
# Test for WAV duration calculation
# -------------------------------

def test_get_wav_duration(tmp_path):
    """
    Create a dummy WAV file with a sine wave (should be about 1 sec long).
    """
    dur = 1.0  # seconds
    sr = 22050
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    wav_file = tmp_path / "dummy.wav"
    sf.write(str(wav_file), y, sr)
    
    measured = get_wav_duration(str(wav_file))
    # Allow a wee bit of tolerance
    assert pytest.approx(measured, rel=0.05) == dur


# -------------------------------
# Integration Test for CSV Creation
# -------------------------------

def test_create_song_list(tmp_path, monkeypatch):
    """
    Create a fake dataset with original/ and augmented/ subfolders, run the CSV
    creation, and check the CSV uses the subfolder-prefixed paths.
    """
    data_dir = tmp_path / "test_data"
    originals_dir = data_dir / "original"
    augmented_dir = data_dir / "augmented"
    originals_dir.mkdir(parents=True)
    augmented_dir.mkdir(parents=True)

    # Create original song files under original/
    orig_title = "song1"
    orig_mid = originals_dir / f"{orig_title}.mid"
    orig_wav = originals_dir / f"{orig_title}.wav"
    orig_mid.write_text("dummy midi content")

    sr = 22050
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    sf.write(str(orig_wav), y, sr)

    # Create an augmented version under augmented/
    aug_stem = f"{orig_title}_augmented_timestretch_aug"
    aug_mid = augmented_dir / f"{aug_stem}.mid"
    aug_wav = augmented_dir / f"{aug_stem}.wav"
    aug_mid.write_text("dummy augmented midi content")

    dur_aug = 1.2  # slightly different duration
    t_aug = np.linspace(0, dur_aug, int(sr * dur_aug), endpoint=False)
    y_aug = 0.5 * np.sin(2 * np.pi * 330 * t_aug)
    sf.write(str(aug_wav), y_aug, sr)

    # Save the current directory to restore later
    original_cwd = os.getcwd()

    try:
        # Change directory so the CSV gets written to tmp_path
        monkeypatch.chdir(tmp_path)

        # Run our CSV creation function
        create_song_list(str(data_dir), {'train': 0.7, 'test': 0.15, 'validation': 0.15})
    finally:
        # Make sure we restore the original directory
        os.chdir(original_cwd)

    csv_filename = f"{data_dir.name}.csv"
    csv_file = tmp_path / csv_filename
    assert csv_file.exists(), "CSV file wasn't created, somethin' went wrong"

    # Read in the CSV and check its contents
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # First row is header; expect 2 data rows (one for orig, one for aug)
    data_rows = rows[1:]
    assert len(data_rows) == 2, f"Expected 2 rows in CSV, but got {len(data_rows)}"

    # Find the original and augmented rows by title (order-independent)
    orig_row = next(r for r in data_rows if r[1] == orig_title)
    aug_row = next(r for r in data_rows if r[1] == aug_stem)

    # Paths must be prefixed with the subfolder name.
    assert orig_row[4] == f"test_data/original/{orig_title}.mid"
    assert orig_row[5] == f"test_data/original/{orig_title}.wav"
    assert aug_row[4] == f"test_data/augmented/{aug_stem}.mid"
    assert aug_row[5] == f"test_data/augmented/{aug_stem}.wav"

    assert pytest.approx(float(orig_row[6]), rel=0.05) == dur, "Original duration seems off"
    assert pytest.approx(float(aug_row[6]), rel=0.05) == dur_aug, "Augmented duration seems off"


def _build_dataset(data_dir, songs):
    """Create the original/ subfolder layout with `songs` originals and no augmented files."""
    originals_dir = data_dir / "original"
    augmented_dir = data_dir / "augmented"
    originals_dir.mkdir(parents=True)
    augmented_dir.mkdir(parents=True)
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    for s in songs:
        (originals_dir / f"{s}.mid").write_text("dummy")
        sf.write(str(originals_dir / f"{s}.wav"), y, sr)


def test_custom_validation_songs(tmp_path, monkeypatch):
    """Songs matching --custom-validation-songs land in validation, not augmented."""
    data_dir = tmp_path / "ds"
    _build_dataset(data_dir, ["alpha", "beta", "gamma"])

    monkeypatch.chdir(tmp_path)
    create_song_list(
        str(data_dir),
        {'train': 0.7, 'test': 0.15, 'validation': 0.15},
        custom_test_songs=[],
        custom_validation_songs=["beta"],
    )

    with open(tmp_path / f"{data_dir.name}.csv") as f:
        rows = list(csv.DictReader(f))

    by_title = {r['canonical_title']: r for r in rows}
    assert by_title['beta']['split'] == 'validation'
    # Beta must not have augmented versions
    assert not any('beta_augmented_' in r['canonical_title'] for r in rows)


def test_split_seed_is_deterministic(tmp_path, monkeypatch):
    """Same split_seed must produce the same split assignment across runs."""
    songs = [f"song{i}" for i in range(20)]
    runs = []
    for i in range(2):
        ds = tmp_path / f"ds{i}"
        _build_dataset(ds, songs)
        monkeypatch.chdir(tmp_path)
        path = create_song_list(str(ds), split_seed=42)
        with open(path) as f:
            assignments = {r['canonical_title']: r['split'] for r in csv.DictReader(f)}
        runs.append(assignments)
    assert runs[0] == runs[1]


def test_split_seed_changes_assignment(tmp_path, monkeypatch):
    """Different seeds should produce different assignments (with high probability)."""
    songs = [f"song{i}" for i in range(20)]
    a = tmp_path / "a"
    b = tmp_path / "b"
    _build_dataset(a, songs)
    _build_dataset(b, songs)
    monkeypatch.chdir(tmp_path)
    pa = create_song_list(str(a), split_seed=1)
    pb = create_song_list(str(b), split_seed=2)
    with open(pa) as f:
        sa = {r['canonical_title']: r['split'] for r in csv.DictReader(f)}
    with open(pb) as f:
        sb = {r['canonical_title']: r['split'] for r in csv.DictReader(f)}
    # With 20 songs and 70/15/15 splits, two different seeds should differ on
    # at least one song almost always.
    assert sa != sb


def test_custom_test_takes_precedence_over_validation(tmp_path, monkeypatch, capsys):
    """When a title matches both lists, test wins and a warning is printed."""
    data_dir = tmp_path / "ds"
    _build_dataset(data_dir, ["sharedname"])

    monkeypatch.chdir(tmp_path)
    create_song_list(
        str(data_dir),
        custom_test_songs=["sharedname"],
        custom_validation_songs=["sharedname"],
    )

    with open(tmp_path / f"{data_dir.name}.csv") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]['split'] == 'test'
    assert "WARNING" in capsys.readouterr().out

