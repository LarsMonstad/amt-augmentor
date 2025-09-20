"""
Tests for pitch_shift module.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest

from amt_augmentor.pitch_shift import (
    apply_pitch_shift,
    load_ann_file,
    save_ann_file,
    update_ann_file,
)


class TestPitchShift:
    """Test pitch shifting functionality."""

    def setup_method(self):
        """Set up test files."""
        self.test_dir = tempfile.mkdtemp()

        # Create test audio file
        self.audio_file = os.path.join(self.test_dir, "test.wav")
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(self.audio_file, audio, sr)

        # Create test annotation file
        self.ann_file = os.path.join(self.test_dir, "test.ann")
        with open(self.ann_file, 'w', encoding='utf-8') as f:
            f.write("0.1\t0.5\t60\t100\n")
            f.write("0.6\t1.0\t62\t100\n")
            f.write("1.1\t1.5\t64\t100\n")

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_ann_file(self):
        """Test loading annotation file."""
        lines = load_ann_file(self.ann_file)
        assert len(lines) == 3
        assert lines[0] == "0.1\t0.5\t60\t100"
        assert lines[1] == "0.6\t1.0\t62\t100"
        assert lines[2] == "1.1\t1.5\t64\t100"

    def test_save_ann_file(self):
        """Test saving annotation file."""
        test_content = ["1.0\t2.0\t60\t100", "2.0\t3.0\t62\t100"]
        output_file = os.path.join(self.test_dir, "output.ann")
        save_ann_file(output_file, test_content)

        # Verify file was saved correctly
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == "1.0\t2.0\t60\t100\n2.0\t3.0\t62\t100"

    def test_update_ann_file_positive_shift(self):
        """Test updating annotation with positive pitch shift."""
        ann_content = ["0.5\t1.0\t60\t100", "1.5\t2.0\t62\t100"]

        # Test pitch shift up by 2 semitones
        updated = update_ann_file(ann_content, 2)
        assert len(updated) == 2
        assert updated[0] == "0.5\t1.0\t62\t100"
        assert updated[1] == "1.5\t2.0\t64\t100"

    def test_update_ann_file_negative_shift(self):
        """Test updating annotation with negative pitch shift."""
        ann_content = ["0.5\t1.0\t60\t100", "1.5\t2.0\t62\t100"]

        # Test pitch shift down by 3 semitones
        updated = update_ann_file(ann_content, -3)
        assert len(updated) == 2
        assert updated[0] == "0.5\t1.0\t57\t100"
        assert updated[1] == "1.5\t2.0\t59\t100"

    def test_apply_pitch_shift_wav(self):
        """Test the main pitch shift function with WAV output."""
        output_file = os.path.join(self.test_dir, "pitched.wav")
        pitch_shift = 3

        # Apply pitch shift
        output_ann = apply_pitch_shift(
            self.audio_file,
            self.ann_file,
            output_file,
            pitch_shift
        )

        # Check output files exist
        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)
        assert output_ann.endswith(".ann")

        # Check output audio was created
        original_audio, sr1 = sf.read(self.audio_file)
        pitched_audio, sr2 = sf.read(output_file)
        assert sr1 == sr2  # Sample rate should be preserved
        assert len(pitched_audio) == len(original_audio)  # Length preserved

        # Check annotation was updated
        with open(output_ann, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # First note should have pitch 60 + 3 = 63
        first_note = lines[0].strip().split('\t')
        assert int(first_note[2]) == 63
        # Second note should have pitch 62 + 3 = 65
        second_note = lines[1].strip().split('\t')
        assert int(second_note[2]) == 65

    def test_apply_pitch_shift_flac(self):
        """Test pitch shift with FLAC output format."""
        # Create FLAC input file
        flac_file = os.path.join(self.test_dir, "test.flac")
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(flac_file, audio, sr, format='FLAC')

        output_file = os.path.join(self.test_dir, "pitched.flac")
        pitch_shift = -2

        # Apply pitch shift
        output_ann = apply_pitch_shift(
            flac_file,
            self.ann_file,
            output_file,
            pitch_shift
        )

        # Check output files exist
        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)

        # Verify it's a FLAC file
        info = sf.info(output_file)
        assert info.format == 'FLAC'

    def test_apply_pitch_shift_extreme_values(self):
        """Test pitch shift with extreme shift values."""
        # Test very high pitch shift (12 semitones = 1 octave up)
        output_high = os.path.join(self.test_dir, "very_high.wav")
        result = apply_pitch_shift(self.audio_file, self.ann_file, output_high, 12)
        assert os.path.exists(output_high)
        assert os.path.exists(result)

        # Test very low pitch shift (-12 semitones = 1 octave down)
        output_low = os.path.join(self.test_dir, "very_low.wav")
        result = apply_pitch_shift(self.audio_file, self.ann_file, output_low, -12)
        assert os.path.exists(output_low)
        assert os.path.exists(result)

    def test_pitch_shift_maintains_timing(self):
        """Test that pitch shift doesn't affect note timing."""
        ann_content = ["0.5\t1.0\t60\t100", "1.5\t2.0\t62\t100"]
        updated = update_ann_file(ann_content, 5)

        # Check that timing values remain unchanged
        assert updated[0].startswith("0.5\t1.0")
        assert updated[1].startswith("1.5\t2.0")

    def test_mono_audio_handling(self):
        """Test pitch shift with mono audio input."""
        # Create mono audio file
        mono_file = os.path.join(self.test_dir, "mono.wav")
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(mono_file, audio, sr)

        output_file = os.path.join(self.test_dir, "mono_pitched.wav")
        result = apply_pitch_shift(mono_file, self.ann_file, output_file, 4)
        assert os.path.exists(output_file)
        assert os.path.exists(result)

    def test_stereo_audio_handling(self):
        """Test pitch shift with stereo audio input."""
        # Create stereo audio file
        stereo_file = os.path.join(self.test_dir, "stereo.wav")
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 550 * t)
        stereo = np.column_stack((left, right))
        sf.write(stereo_file, stereo, sr)

        output_file = os.path.join(self.test_dir, "stereo_pitched.wav")
        result = apply_pitch_shift(stereo_file, self.ann_file, output_file, -3)
        assert os.path.exists(output_file)
        assert os.path.exists(result)

        # Verify stereo is preserved
        output_audio, _ = sf.read(output_file)
        assert output_audio.ndim == 2  # Should be stereo