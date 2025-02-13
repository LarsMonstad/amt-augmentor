import os
import csv
import numpy as np
import soundfile as sf
import pytest
from collections import defaultdict

# Import the functions we wanna test
from amt_augpy.create_maestro_csv import (
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
    ("song_timestretch_aug.mid", True),
    ("song_pitchshift_extra.mid", True),
    ("song_reverb_filters_test.mid", True),
    ("song_gain_chorus_final.mid", True),
    ("song_addpauses_demo.mid", True),
    ("SONG_PITCHSHIFT_EXTRA.MID", True),  # checking case insensitivity
])
def test_is_augmented_version(filename, expected):
    # Let's see if this works as expected...
    assert is_augmented_version(filename) == expected


# Test extracting the original song name from filenames with augment markers.
@pytest.mark.parametrize("filename, expected", [
    ("song_timestretch_aug.mid", "song"),
    ("song_pitchshift_extra.mid", "song"),
    ("song_reverb_filters_test.mid", "song"),
    ("song_gain_chorus_final.mid", "song"),
    ("song_addpauses_demo.mid", "song"),
    ("song.mid", "song"),
    ("my.song.mid", "my.song"),
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
    We'll create a fake directory with one original song and one augmented version,
    run the CSV creation, and check if the CSV looks right.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create original song files
    orig_title = "song1"
    orig_mid = data_dir / f"{orig_title}.mid"
    orig_wav = data_dir / f"{orig_title}.wav"
    orig_mid.write_text("dummy midi content")
    
    sr = 22050
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    sf.write(str(orig_wav), y, sr)
    
    # Create an augmented version
    aug_mid = data_dir / f"{orig_title}_timestretch_aug.mid"
    aug_wav = data_dir / f"{orig_title}_timestretch_aug.wav"
    aug_mid.write_text("dummy augmented midi content")
    
    dur_aug = 1.2  # slightly different duration
    t_aug = np.linspace(0, dur_aug, int(sr * dur_aug), endpoint=False)
    y_aug = 0.5 * np.sin(2 * np.pi * 330 * t_aug)
    sf.write(str(aug_wav), y_aug, sr)
    
    # Change directory so the CSV gets written to tmp_path
    monkeypatch.chdir(tmp_path)
    
    # Run our CSV creation function
    create_song_list(str(data_dir), {'train': 0.7, 'test': 0.15, 'validation': 0.15})
    
    csv_filename = f"{data_dir.name}.csv"
    csv_file = tmp_path / csv_filename
    assert csv_file.exists(), "CSV file wasn't created, somethin' went wrong"
    
    # Read in the CSV and check its contents
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # First row is header; expect 2 data rows (one for orig, one for aug)
    header = rows[0]
    data_rows = rows[1:]
    assert len(data_rows) == 2, f"Expected 2 rows in CSV, but got {len(data_rows)}"
    
    # Check original song row
    orig_row = data_rows[0]
    assert orig_row[1] == orig_title, "Original song title doesn't match"
    assert pytest.approx(float(orig_row[6]), rel=0.05) == dur, "Original duration seems off"
    
    # Check augmented song row
    aug_row = data_rows[1]
    expected_aug_title = os.path.splitext(aug_mid.name)[0]
    assert aug_row[1] == expected_aug_title, "Augmented title doesn't match"
    assert pytest.approx(float(aug_row[6]), rel=0.05) == dur_aug, "Augmented duration seems off"

