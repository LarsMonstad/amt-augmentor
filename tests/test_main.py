# tests/test_main.py

import os
import random
import string
import pytest
import pretty_midi

# Import the functions to test from your main module
from amt_augpy.main import (
    random_word,
    generate_output_filename,
    delete_file,
    midi_to_ann,
    ann_to_midi,
    check_matching_files,
)


def test_random_word():
    """Test that random_word returns a lowercase string of the correct length."""
    length = 5
    word = random_word(length)
    assert isinstance(word, str)
    assert len(word) == length
    # Ensure all characters are lowercase letters
    assert all(ch in string.ascii_lowercase for ch in word)


def test_generate_output_filename():
    """Test that generate_output_filename returns the expected string."""
    base_name = "audio"
    effect_name = "pitchshift"
    measure = 1.2
    random_suffix = "abcde"
    extension = ".wav"
    result = generate_output_filename(base_name, effect_name, measure, random_suffix, extension)
    expected = "audio_pitchshift_1.2_abcde.wav"
    assert result == expected


def test_delete_file(tmp_path):
    """Test that delete_file successfully removes a file."""
    # Create a temporary file
    temp_file = tmp_path / "temp.txt"
    temp_file.write_text("dummy content")
    # Confirm the file exists
    assert temp_file.exists()
    # Call delete_file (convert tmp_path object to string)
    delete_file(str(temp_file))
    # After deletion, the file should no longer exist
    assert not temp_file.exists()


def test_midi_to_ann_and_ann_to_midi(tmp_path):
    """
    Test that a MIDI file can be converted to an annotation file and back to MIDI.
    This test creates a simple MIDI file with one note.
    """
    # Paths for temporary files
    midi_path = tmp_path / "test_input.mid"
    ann_path = tmp_path / "test_output.ann"
    # ann_to_midi will generate a MIDI file by replacing ".ann" with ".mid"
    converted_midi_path = tmp_path / "test_output.mid"

    # Create a dummy MIDI file using pretty_midi
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    # Create a note: pitch 60, velocity 100, start at 0.5, end at 1.0 seconds
    note = pretty_midi.Note(velocity=100, pitch=60, start=0.5, end=1.0)
    instrument.notes.append(note)
    pm.instruments.append(instrument)
    pm.write(str(midi_path))

    # Convert the MIDI file to an annotation file
    midi_to_ann(str(midi_path), str(ann_path))
    # Read and verify the contents of the annotation file
    ann_lines = ann_path.read_text().strip().splitlines()
    # Expect one line per note (we have one note)
    assert len(ann_lines) == 1
    parts = ann_lines[0].split('\t')
    assert len(parts) == 4
    onset, offset, pitch, velocity = parts
    # Check that values match what we wrote (formatted to 6 decimals)
    assert float(onset) == pytest.approx(0.5, rel=1e-5)
    assert float(offset) == pytest.approx(1.0, rel=1e-5)
    assert int(pitch) == 60
    assert int(velocity) == 100

    # Convert the annotation file back to MIDI
    ann_to_midi(str(ann_path))
    # The converted MIDI file is created by replacing ".ann" with ".mid" in the filename
    assert converted_midi_path.exists()
    # Load the newly created MIDI file and verify its contents
    pm_converted = pretty_midi.PrettyMIDI(str(converted_midi_path))
    # There should be at least one instrument
    assert len(pm_converted.instruments) > 0
    converted_instrument = pm_converted.instruments[0]
    # Expect one note in the instrument
    assert len(converted_instrument.notes) == 1
    converted_note = converted_instrument.notes[0]
    assert converted_note.pitch == 60
    assert converted_note.start == pytest.approx(0.5, rel=1e-5)
    assert converted_note.end == pytest.approx(1.0, rel=1e-5)


def test_check_matching_files(tmp_path, capsys):
    """
    Test the check_matching_files function by creating a temporary directory
    with matching and non-matching files.
    """
    # Create dummy files:
    # - A matching pair: song1.wav and song1.mid
    # - A WAV file with no matching MIDI: song2.wav
    # - A MIDI file with no matching WAV: song3.mid
    (tmp_path / "song1.wav").write_text("dummy")
    (tmp_path / "song1.mid").write_text("dummy")
    (tmp_path / "song2.wav").write_text("dummy")
    (tmp_path / "song3.mid").write_text("dummy")

    # Run the check_matching_files function on the temporary directory
    check_matching_files(str(tmp_path))

    # Capture printed output
    captured = capsys.readouterr().out
    # Verify that the expected messages are present
    assert "No matching MIDI file for: song2.wav" in captured
    assert "No matching WAV file for: song3.mid" in captured
    # Also check that the summary shows one complete match
    assert "Complete matches found: 1" in captured

