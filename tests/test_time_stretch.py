"""
Tests for time_stretch module.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest

from amt_augmentor.time_stretch import apply_time_stretch, load_ann_file, save_ann_file, update_ann_file


class TestTimeStretch:
    """Test time stretching functionality."""

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

    def test_save_ann_file(self):
        """Test saving annotation file."""
        test_content = ["1.0\t2.0\t60\t100", "2.0\t3.0\t62\t100"]
        output_file = os.path.join(self.test_dir, "output.ann")
        save_ann_file(output_file, test_content)

        # Verify file was saved correctly
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == "1.0\t2.0\t60\t100\n2.0\t3.0\t62\t100"

    def test_update_ann_file(self):
        """Test updating annotation timings with stretch factor."""
        ann_content = ["0.5\t1.0\t60\t100", "1.5\t2.0\t62\t100"]

        # Test stretch factor 2.0 (double the time)
        updated = update_ann_file(ann_content, 2.0)
        assert len(updated) == 2
        assert updated[0] == "1.000\t2.000\t60\t100"
        assert updated[1] == "3.000\t4.000\t62\t100"

        # Test stretch factor 0.5 (half the time)
        updated = update_ann_file(ann_content, 0.5)
        assert updated[0] == "0.250\t0.500\t60\t100"
        assert updated[1] == "0.750\t1.000\t62\t100"

    def test_apply_time_stretch(self):
        """Test the main time stretch function."""
        output_file = os.path.join(self.test_dir, "stretched.wav")
        stretch_factor = 1.5

        # Apply time stretch
        output_ann = apply_time_stretch(
            self.audio_file,
            self.ann_file,
            output_file,
            stretch_factor
        )

        # Check output files exist
        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)
        assert output_ann.endswith(".ann")

        # Check output audio was created (different size due to stretch)
        original_audio, sr1 = sf.read(self.audio_file)
        stretched_audio, sr2 = sf.read(output_file)
        assert sr1 == sr2  # Sample rate should be preserved
        # Length should be approximately stretched (within 10% tolerance)
        expected_length = len(original_audio) * stretch_factor
        assert abs(len(stretched_audio) - expected_length) < expected_length * 0.1

        # Check annotation was updated
        with open(output_ann, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # First note should start at 0.1 * 1.5 = 0.15
        first_note = lines[0].strip().split('\t')
        assert float(first_note[0]) == pytest.approx(0.15, rel=0.01)
        assert float(first_note[1]) == pytest.approx(0.75, rel=0.01)

    def test_apply_time_stretch_extreme_values(self):
        """Test time stretch with extreme stretch factors."""
        # Test very slow (3x slower)
        output_slow = os.path.join(self.test_dir, "very_slow.wav")
        result = apply_time_stretch(self.audio_file, self.ann_file, output_slow, 3.0)
        assert os.path.exists(output_slow)
        assert os.path.exists(result)

        # Test very fast (0.3x = 3x faster)
        output_fast = os.path.join(self.test_dir, "very_fast.wav")
        result = apply_time_stretch(self.audio_file, self.ann_file, output_fast, 0.3)
        assert os.path.exists(output_fast)
        assert os.path.exists(result)

    def test_malformed_annotation(self):
        """Test handling of malformed annotation file."""
        bad_ann_file = os.path.join(self.test_dir, "bad.ann")
        with open(bad_ann_file, 'w', encoding='utf-8') as f:
            f.write("0.1\t0.5\t60\t100\n")  # Good line
            f.write("bad line\n")  # Bad line
            f.write("0.6\t1.0\t62\n")  # Missing velocity

        lines = load_ann_file(bad_ann_file)
        assert len(lines) == 3  # Should load all lines

        # update_ann_file should handle bad lines gracefully
        updated = update_ann_file(lines, 2.0)
        assert len(updated) == 1  # Should only process valid lines (only first line has 4 fields)