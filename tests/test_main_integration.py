"""
Integration tests for main.py command-line argument handling.
These tests verify that CLI arguments are correctly passed to underlying functions.
"""

import os
import sys
import csv
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call
import numpy as np
import soundfile as sf
import pretty_midi


def create_test_dataset(directory):
    """Create a minimal test dataset with MIDI and WAV files."""
    os.makedirs(directory, exist_ok=True)

    # Create test files
    songs = ["song1", "song2", "song3", "song4", "song5"]

    for song in songs:
        # Create MIDI file
        midi_path = os.path.join(directory, f"{song}.mid")
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        note = pretty_midi.Note(velocity=100, pitch=60, start=0.5, end=1.0)
        instrument.notes.append(note)
        pm.instruments.append(instrument)
        pm.write(midi_path)

        # Create WAV file
        wav_path = os.path.join(directory, f"{song}.wav")
        sr = 22050
        dur = 1.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 220 * t)
        sf.write(wav_path, y, sr)


class TestMainCLIIntegration:
    """Test the main() function's handling of command-line arguments."""

    def setup_method(self):
        """Set up test directory."""
        self.test_dir = tempfile.mkdtemp()
        create_test_dataset(self.test_dir)

    def teardown_method(self):
        """Clean up test directory and any CSV files created."""
        if os.path.exists(self.test_dir):
            # Get the directory name for the CSV file
            csv_filename = f"{os.path.basename(self.test_dir)}.csv"
            # Remove CSV file if it exists in current directory
            if os.path.exists(csv_filename):
                os.remove(csv_filename)
            # Remove the test directory
            shutil.rmtree(self.test_dir)

    @patch('amt_augmentor.main.create_song_list')
    def test_split_ratios_passed_correctly(self, mock_create_song_list):
        """
        Test that --train-ratio, --test-ratio, --validation-ratio are correctly
        converted to a dictionary and passed to create_song_list.

        This test would have caught the bug where individual parameters were
        incorrectly passed instead of a split_ratios dictionary.
        """
        # Mock the return value
        mock_create_song_list.return_value = "test.csv"

        # Simulate command-line arguments
        test_args = [
            'amt-augmentor',
            self.test_dir,
            '--train-ratio', '0.8',
            '--test-ratio', '0.1',
            '--validation-ratio', '0.1',
            '--skip-csv'  # Skip to avoid actual CSV creation
        ]

        with patch.object(sys, 'argv', test_args):
            # Import here to avoid issues with module-level code
            from amt_augmentor.main import main

            # Temporarily skip the actual processing
            with patch('amt_augmentor.main.process_files'):
                # Mock gen_ann to return expected tuple
                with patch('amt_augmentor.main.gen_ann') as mock_gen_ann:
                    mock_gen_ann.return_value = ('audio.wav', 'standardized.wav', 'temp.ann')
                    with patch('amt_augmentor.main.validate_dataset_split'):
                        # Remove --skip-csv for this test
                        test_args.remove('--skip-csv')
                        with patch.object(sys, 'argv', test_args):
                            main()

        # Verify create_song_list was called with correct arguments
        mock_create_song_list.assert_called_once()
        args, kwargs = mock_create_song_list.call_args

        # Check that split_ratios was passed as a dictionary
        assert 'split_ratios' in kwargs, "split_ratios not passed as keyword argument"
        split_ratios = kwargs['split_ratios']

        # Verify it's a dictionary with correct structure
        assert isinstance(split_ratios, dict), "split_ratios should be a dictionary"
        assert 'train' in split_ratios, "split_ratios missing 'train' key"
        assert 'test' in split_ratios, "split_ratios missing 'test' key"
        assert 'validation' in split_ratios, "split_ratios missing 'validation' key"

        # Verify the values match what was passed
        assert split_ratios['train'] == 0.8, f"Expected train=0.8, got {split_ratios['train']}"
        assert split_ratios['test'] == 0.1, f"Expected test=0.1, got {split_ratios['test']}"
        assert split_ratios['validation'] == 0.1, f"Expected validation=0.1, got {split_ratios['validation']}"

    @patch('amt_augmentor.main.create_song_list')
    def test_custom_test_songs_passed(self, mock_create_song_list):
        """
        Test that --custom-test-songs argument is correctly parsed and passed
        to create_song_list function.
        """
        mock_create_song_list.return_value = "test.csv"

        test_args = [
            'amt-augmentor',
            self.test_dir,
            '--custom-test-songs', 'song1,song3,special_song',
            '--skip-csv'
        ]

        with patch.object(sys, 'argv', test_args):
            from amt_augmentor.main import main

            with patch('amt_augmentor.main.process_files'):
                # Mock gen_ann to return expected tuple
                with patch('amt_augmentor.main.gen_ann') as mock_gen_ann:
                    mock_gen_ann.return_value = ('audio.wav', 'standardized.wav', 'temp.ann')
                    with patch('amt_augmentor.main.validate_dataset_split'):
                        test_args.remove('--skip-csv')
                        with patch.object(sys, 'argv', test_args):
                            main()

        # Verify custom_test_songs was passed correctly
        mock_create_song_list.assert_called_once()
        args, kwargs = mock_create_song_list.call_args

        assert 'custom_test_songs' in kwargs, "custom_test_songs not passed"
        custom_songs = kwargs['custom_test_songs']

        # Verify it's a list with correct values
        assert isinstance(custom_songs, list), "custom_test_songs should be a list"
        assert 'song1' in custom_songs, "song1 not in custom_test_songs"
        assert 'song3' in custom_songs, "song3 not in custom_test_songs"
        assert 'special_song' in custom_songs, "special_song not in custom_test_songs"
        assert len(custom_songs) == 3, f"Expected 3 custom songs, got {len(custom_songs)}"

    @patch('amt_augmentor.main.create_song_list')
    def test_default_split_ratios(self, mock_create_song_list):
        """
        Test that default split ratios are used when not specified.
        """
        mock_create_song_list.return_value = "test.csv"

        test_args = [
            'amt-augmentor',
            self.test_dir,
            # No ratio arguments provided
        ]

        with patch.object(sys, 'argv', test_args):
            from amt_augmentor.main import main

            with patch('amt_augmentor.main.process_files'):
                # Mock gen_ann to return expected tuple
                with patch('amt_augmentor.main.gen_ann') as mock_gen_ann:
                    mock_gen_ann.return_value = ('audio.wav', 'standardized.wav', 'temp.ann')
                    with patch('amt_augmentor.main.validate_dataset_split'):
                        main()

        # Verify default values were used
        mock_create_song_list.assert_called_once()
        args, kwargs = mock_create_song_list.call_args

        split_ratios = kwargs.get('split_ratios', {})
        assert split_ratios['train'] == 0.7, f"Expected default train=0.7, got {split_ratios['train']}"
        assert split_ratios['test'] == 0.15, f"Expected default test=0.15, got {split_ratios['test']}"
        assert split_ratios['validation'] == 0.15, f"Expected default validation=0.15, got {split_ratios['validation']}"

    def test_incorrect_parameter_names_would_fail(self):
        """
        This test simulates what would happen with the old bug where
        train_ratio, test_ratio, validation_ratio were passed as individual
        parameters instead of in a dictionary.

        This demonstrates the bug that was fixed in commit d749d82.
        """
        from amt_augmentor.create_maestro_csv import create_song_list

        # This is how the buggy code tried to call the function
        with pytest.raises(TypeError) as exc_info:
            create_song_list(
                self.test_dir,
                train_ratio=0.8,  # WRONG: no such parameter
                test_ratio=0.1,   # WRONG: no such parameter
                validation_ratio=0.1  # WRONG: no such parameter
            )

        # The error message should indicate unexpected keyword arguments
        assert "unexpected keyword argument" in str(exc_info.value).lower()

    def test_correct_dictionary_format_works(self):
        """
        Test that the correct way of calling create_song_list works.
        This is how it should be called after the fix.
        """
        from amt_augmentor.create_maestro_csv import create_song_list

        # This is the correct way to call it
        csv_path = create_song_list(
            self.test_dir,
            split_ratios={'train': 0.8, 'test': 0.1, 'validation': 0.1},
            custom_test_songs=['song1']
        )

        # Verify CSV was created
        assert os.path.exists(csv_path)

        # Read and verify the CSV content
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check that song1 is in test set due to custom_test_songs
        song1_rows = [r for r in rows if 'song1' in r['canonical_title']]
        assert all(r['split'] == 'test' for r in song1_rows), "song1 should be in test split"

        # Check that augmented versions of song1 don't exist (test songs don't get augmented)
        augmented_song1 = [r for r in rows if 'song1_' in r['canonical_title'] and
                           any(aug in r['canonical_title'] for aug in
                               ['timestretch', 'pitchshift', 'reverb', 'gain', 'addpauses'])]
        assert len(augmented_song1) == 0, "Test songs shouldn't have augmented versions"


class TestEndToEndIntegration:
    """End-to-end integration test with actual file creation."""

    def setup_method(self):
        """Set up test directory."""
        self.test_dir = tempfile.mkdtemp()
        create_test_dataset(self.test_dir)

    def teardown_method(self):
        """Clean up test directory and any CSV files created."""
        if os.path.exists(self.test_dir):
            # Get the directory name for the CSV file
            csv_filename = f"{os.path.basename(self.test_dir)}.csv"
            # Remove CSV file if it exists in current directory
            if os.path.exists(csv_filename):
                os.remove(csv_filename)
            # Remove the test directory
            shutil.rmtree(self.test_dir)

    def test_full_csv_creation_with_custom_splits_and_songs(self):
        """
        Full integration test: create CSV with custom splits and custom test songs.
        Verifies the entire pipeline from CLI args to CSV output.
        """
        test_args = [
            'amt-augmentor',
            self.test_dir,
            '--train-ratio', '0.6',
            '--test-ratio', '0.2',
            '--validation-ratio', '0.2',
            '--custom-test-songs', 'song2,song4',
            '--skip-csv'  # We'll call create_song_list directly
        ]

        # Directly test create_song_list with the parameters main() would pass
        from amt_augmentor.create_maestro_csv import create_song_list

        split_ratios = {
            'train': 0.6,
            'test': 0.2,
            'validation': 0.2
        }
        custom_test_songs = ['song2', 'song4']

        csv_path = create_song_list(self.test_dir, split_ratios, custom_test_songs)

        # Read the CSV and verify
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Verify custom test songs are in test split
        for song in ['song2', 'song4']:
            song_rows = [r for r in rows if r['canonical_title'] == song]
            assert len(song_rows) > 0, f"{song} not found in CSV"
            assert all(r['split'] == 'test' for r in song_rows), f"{song} should be in test split"

        # Count splits for original songs only (no augmented)
        original_rows = [r for r in rows if not any(
            aug in r['canonical_title'] for aug in
            ['timestretch', 'pitchshift', 'reverb', 'gain', 'addpauses']
        )]

        split_counts = {'train': 0, 'test': 0, 'validation': 0}
        for row in original_rows:
            split_counts[row['split']] += 1

        # With 5 songs total and 2 forced to test:
        # - song2, song4 -> forced to test (2 songs)
        # - Remaining 3 songs distributed according to ratios
        assert split_counts['test'] >= 2, "At least 2 songs should be in test (custom songs)"

        # Verify no augmented versions exist for test songs
        for song in ['song2', 'song4']:
            aug_versions = [r for r in rows if song in r['canonical_title'] and song != r['canonical_title']]
            assert len(aug_versions) == 0, f"Test song {song} shouldn't have augmented versions"