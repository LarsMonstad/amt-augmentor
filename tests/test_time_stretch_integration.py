"""
Integration tests for time_stretch module.
These tests verify that audio and MIDI stay synchronized after stretching.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest
import pretty_midi

from amt_augmentor.time_stretch import apply_time_stretch


class TestTimeStretchIntegration:
    """Integration tests to verify audio and MIDI synchronization."""

    def setup_method(self):
        """Set up test files with known durations."""
        self.test_dir = tempfile.mkdtemp()

        # Create test audio file with exact 2 second duration
        self.audio_file = os.path.join(self.test_dir, "test.wav")
        sr = 44100  # Sample rate
        duration = 2.0  # Exactly 2 seconds
        samples = int(sr * duration)
        t = np.linspace(0, duration, samples, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(self.audio_file, audio, sr)

        # Create test MIDI that matches audio duration
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        # Add notes that span the full 2 seconds
        # Note 1: 0.0 - 0.5 seconds
        note1 = pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5)
        instrument.notes.append(note1)

        # Note 2: 0.5 - 1.0 seconds
        note2 = pretty_midi.Note(velocity=100, pitch=62, start=0.5, end=1.0)
        instrument.notes.append(note2)

        # Note 3: 1.0 - 1.5 seconds
        note3 = pretty_midi.Note(velocity=100, pitch=64, start=1.0, end=1.5)
        instrument.notes.append(note3)

        # Note 4: 1.5 - 2.0 seconds (end of audio)
        note4 = pretty_midi.Note(velocity=100, pitch=65, start=1.5, end=2.0)
        instrument.notes.append(note4)

        midi.instruments.append(instrument)

        # Save MIDI file
        self.midi_file = os.path.join(self.test_dir, "test.mid")
        midi.write(self.midi_file)

        # Convert MIDI to annotation format
        self.ann_file = os.path.join(self.test_dir, "test.ann")
        with open(self.ann_file, 'w', encoding='utf-8') as f:
            for note in instrument.notes:
                f.write(f"{note.start:.3f}\t{note.end:.3f}\t{note.pitch}\t{note.velocity}\n")

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_time_stretch_synchronization(self):
        """Test that audio and MIDI remain synchronized after time stretching."""

        test_factors = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

        for factor in test_factors:
            output_file = os.path.join(self.test_dir, f"stretched_{factor}.wav")

            # Apply time stretch
            output_ann = apply_time_stretch(
                self.audio_file,
                self.ann_file,
                output_file,
                factor
            )

            # Load the stretched audio and measure duration
            stretched_audio, sr = sf.read(output_file)
            audio_duration = len(stretched_audio) / sr

            # Expected audio duration (2 seconds / factor)
            expected_audio_duration = 2.0 / factor

            # Audio duration should match expected (within 5% tolerance for algorithm artifacts)
            assert abs(audio_duration - expected_audio_duration) < expected_audio_duration * 0.05, \
                f"Factor {factor}: Audio duration {audio_duration:.3f}s != expected {expected_audio_duration:.3f}s"

            # Read the updated annotations
            with open(output_ann, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Check that MIDI events are correctly scaled
            for i, line in enumerate(lines):
                parts = line.strip().split('\t')
                start = float(parts[0])
                end = float(parts[1])

                # Original notes were at 0.0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0
                original_starts = [0.0, 0.5, 1.0, 1.5]
                original_ends = [0.5, 1.0, 1.5, 2.0]

                expected_start = original_starts[i] / factor
                expected_end = original_ends[i] / factor

                assert abs(start - expected_start) < 0.001, \
                    f"Factor {factor}, Note {i}: Start {start:.3f} != expected {expected_start:.3f}"
                assert abs(end - expected_end) < 0.001, \
                    f"Factor {factor}, Note {i}: End {end:.3f} != expected {expected_end:.3f}"

            # Verify last MIDI event ends at the same time as audio
            last_note_end = float(lines[-1].strip().split('\t')[1])
            assert abs(last_note_end - audio_duration) < 0.1, \
                f"Factor {factor}: Last MIDI note ends at {last_note_end:.3f}s but audio is {audio_duration:.3f}s"

    def test_time_stretch_with_silence_at_end(self):
        """Test time stretching when audio has silence at the end."""
        # Create audio with 1 second of sound and 1 second of silence
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sound = 0.5 * np.sin(2 * np.pi * 440 * t)
        silence = np.zeros(sr)
        audio = np.concatenate([sound, silence])

        audio_file = os.path.join(self.test_dir, "with_silence.wav")
        sf.write(audio_file, audio, sr)

        # Create annotation that only covers the first second
        ann_file = os.path.join(self.test_dir, "with_silence.ann")
        with open(ann_file, 'w', encoding='utf-8') as f:
            f.write("0.0\t0.5\t60\t100\n")
            f.write("0.5\t1.0\t62\t100\n")

        # Stretch by 2x (faster)
        output_file = os.path.join(self.test_dir, "stretched_silence.wav")
        output_ann = apply_time_stretch(audio_file, ann_file, output_file, 2.0)

        # Audio should be 1 second (2 seconds / 2)
        stretched_audio, sr = sf.read(output_file)
        audio_duration = len(stretched_audio) / sr
        assert abs(audio_duration - 1.0) < 0.05

        # MIDI should be scaled correctly
        with open(output_ann, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # First note: 0.0-0.5 becomes 0.0-0.25
        first = lines[0].strip().split('\t')
        assert abs(float(first[0]) - 0.0) < 0.001
        assert abs(float(first[1]) - 0.25) < 0.001

        # Second note: 0.5-1.0 becomes 0.25-0.5
        second = lines[1].strip().split('\t')
        assert abs(float(second[0]) - 0.25) < 0.001
        assert abs(float(second[1]) - 0.5) < 0.001