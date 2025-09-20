"""
Tests for add_pauses module.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest

from amt_augmentor.add_pauses import (
    insert_silence,
    remove_silence_ranges,
    calculate_time_distance,
)


class TestAddPauses:
    """Test pause detection and silence insertion functionality."""

    def setup_method(self):
        """Set up test files."""
        self.test_dir = tempfile.mkdtemp()

        # Create test audio file
        self.audio_file = os.path.join(self.test_dir, "test.wav")
        self.sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        self.original_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(self.audio_file, self.original_audio, self.sr)

        # Create test annotation file with pauses
        self.ann_file = os.path.join(self.test_dir, "test.ann")
        # Notes with gaps between them
        self.ann_lines = [
            "0.1\t0.5\t60\t100\n",  # First note ends at 0.5
            "0.6\t1.0\t62\t100\n",  # Second note starts at 0.6 (0.1s gap)
            "2.2\t2.6\t64\t100\n",  # Third note starts at 2.2 (1.2s gap - suitable for silence)
        ]
        with open(self.ann_file, 'w', encoding='utf-8') as f:
            f.writelines(self.ann_lines)

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_insert_silence_basic(self):
        """Test basic silence insertion (replacement)."""
        output_file = os.path.join(self.test_dir, "with_silence.wav")
        silence_ranges = [(0.5, 0.8), (1.5, 2.0)]

        insert_silence(self.audio_file, silence_ranges, output_file)

        # Check output file exists
        assert os.path.exists(output_file)

        # Check that audio was modified
        original_audio, sr1 = sf.read(self.audio_file)
        modified_audio, sr2 = sf.read(output_file)
        assert sr1 == sr2
        # Audio length should remain the same (ranges are replaced with silence, not added)
        assert len(modified_audio) == len(original_audio)

    def test_insert_silence_empty_ranges(self):
        """Test silence insertion with empty ranges."""
        output_file = os.path.join(self.test_dir, "no_silence.wav")
        silence_ranges = []

        insert_silence(self.audio_file, silence_ranges, output_file)

        # Output should be identical to input
        original_audio, _ = sf.read(self.audio_file)
        modified_audio, _ = sf.read(output_file)
        assert np.allclose(original_audio, modified_audio)

    def test_insert_silence_multiple_ranges(self):
        """Test silence insertion with multiple ranges."""
        output_file = os.path.join(self.test_dir, "multiple_silence.wav")
        silence_ranges = [(0.3, 0.4), (0.8, 1.0), (1.5, 1.8)]

        insert_silence(self.audio_file, silence_ranges, output_file)

        assert os.path.exists(output_file)
        # Ranges are replaced, not added, so length should remain approximately the same
        original_audio, sr = sf.read(self.audio_file)
        modified_audio, _ = sf.read(output_file)
        # Allow small tolerance for rounding
        assert abs(len(modified_audio) - len(original_audio)) <= 1

    def test_remove_silence_ranges_basic(self):
        """Test removal of annotation lines in silence ranges."""
        lines = [
            "0.1\t0.5\t60\t100\n",
            "0.6\t0.9\t62\t100\n",
            "1.0\t1.3\t64\t100\n",
            "1.5\t1.8\t66\t100\n",
        ]
        silence_ranges = [(0.6, 1.0)]  # Should remove second note

        result = remove_silence_ranges(lines, silence_ranges)

        assert len(result) == 3
        assert "0.6\t0.9\t62\t100\n" not in result
        assert "0.1\t0.5\t60\t100\n" in result
        assert "1.0\t1.3\t64\t100\n" in result

    def test_remove_silence_ranges_multiple(self):
        """Test removal with multiple silence ranges."""
        lines = [
            "0.1\t0.5\t60\t100\n",
            "0.6\t0.9\t62\t100\n",
            "1.0\t1.3\t64\t100\n",
            "1.5\t1.8\t66\t100\n",
        ]
        silence_ranges = [(0.6, 0.9), (1.5, 1.8)]  # Should remove 2nd and 4th notes

        result = remove_silence_ranges(lines, silence_ranges)

        assert len(result) == 2
        assert "0.1\t0.5\t60\t100\n" in result
        assert "1.0\t1.3\t64\t100\n" in result

    def test_remove_silence_ranges_malformed_lines(self):
        """Test handling of malformed annotation lines."""
        lines = [
            "0.1\t0.5\t60\t100\n",
            "malformed line\n",  # Should be skipped
            "0.6\t0.9\t62\t100\n",
            "1.0",  # Incomplete line
        ]
        silence_ranges = [(0.5, 0.7)]

        result = remove_silence_ranges(lines, silence_ranges)

        # Should handle malformed lines gracefully
        assert "0.1\t0.5\t60\t100\n" in result
        assert "malformed line\n" not in result

    def test_calculate_time_distance_basic(self):
        """Test basic pause detection and modification."""
        output_file = os.path.join(self.test_dir, "paused.wav")

        # Use default parameters
        output_ann = calculate_time_distance(
            self.audio_file,
            self.ann_file,
            output_file
        )

        # Should detect the 1.2s pause between notes 2 and 3
        assert output_ann is not None
        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)

        # Check annotation was modified
        with open(output_ann, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # No notes fall within the silence range (1.0-2.2), so all notes should remain
        assert len(lines) == len(self.ann_lines)

    def test_calculate_time_distance_custom_thresholds(self):
        """Test with custom pause thresholds."""
        output_file = os.path.join(self.test_dir, "custom_paused.wav")

        # Use very small min_pause_duration to catch smaller gaps
        output_ann = calculate_time_distance(
            self.audio_file,
            self.ann_file,
            output_file,
            pause_threshold=0.001,
            min_pause_duration=0.05,
            max_pause_duration=2.0
        )

        if output_ann:  # May or may not find pauses depending on test data
            assert os.path.exists(output_file)
            assert os.path.exists(output_ann)

    def test_calculate_time_distance_no_pauses(self):
        """Test when no suitable pauses are found."""
        # Create annotation with no suitable pauses
        ann_file = os.path.join(self.test_dir, "no_pauses.ann")
        with open(ann_file, 'w', encoding='utf-8') as f:
            # Continuous notes with no gaps
            f.write("0.0\t0.5\t60\t100\n")
            f.write("0.5\t1.0\t62\t100\n")
            f.write("1.0\t1.5\t64\t100\n")

        output_file = os.path.join(self.test_dir, "no_pauses_out.wav")

        output_ann = calculate_time_distance(
            self.audio_file,
            ann_file,
            output_file
        )

        # Should return None when no pauses found
        assert output_ann is None

    def test_calculate_time_distance_empty_annotation(self):
        """Test with empty annotation file."""
        empty_ann = os.path.join(self.test_dir, "empty.ann")
        with open(empty_ann, 'w', encoding='utf-8') as f:
            f.write("")

        output_file = os.path.join(self.test_dir, "empty_out.wav")

        output_ann = calculate_time_distance(
            self.audio_file,
            empty_ann,
            output_file
        )

        assert output_ann is None

    def test_calculate_time_distance_overlapping_notes(self):
        """Test with overlapping notes (polyphonic)."""
        # Create annotation with overlapping notes
        ann_file = os.path.join(self.test_dir, "overlap.ann")
        with open(ann_file, 'w', encoding='utf-8') as f:
            f.write("0.0\t1.0\t60\t100\n")  # Long note
            f.write("0.3\t0.7\t64\t100\n")  # Overlaps with first
            f.write("1.5\t2.0\t62\t100\n")  # After a pause

        output_file = os.path.join(self.test_dir, "overlap_out.wav")

        output_ann = calculate_time_distance(
            self.audio_file,
            ann_file,
            output_file
        )

        # May or may not find pauses depending on overlap handling
        if output_ann:
            assert os.path.exists(output_file)
            assert os.path.exists(output_ann)

    def test_calculate_time_distance_malformed_annotation(self):
        """Test with malformed annotation lines."""
        # Create annotation with some malformed lines
        ann_file = os.path.join(self.test_dir, "malformed.ann")
        with open(ann_file, 'w', encoding='utf-8') as f:
            f.write("0.1\t0.5\t60\t100\n")
            f.write("bad line\n")  # Malformed
            f.write("1.1\t1.5\t62\t100\n")
            f.write("2.1\t2.5\t64\n")  # Missing velocity

        output_file = os.path.join(self.test_dir, "malformed_out.wav")

        # Should handle malformed lines gracefully
        output_ann = calculate_time_distance(
            self.audio_file,
            ann_file,
            output_file
        )

        # Function should complete without crashing

    def test_calculate_time_distance_nonexistent_ann_file(self):
        """Test error handling for non-existent annotation file."""
        output_file = os.path.join(self.test_dir, "output.wav")

        with pytest.raises(FileNotFoundError):
            calculate_time_distance(
                self.audio_file,
                "nonexistent.ann",
                output_file
            )

    def test_stereo_audio_handling(self):
        """Test pause insertion with stereo audio."""
        # Create stereo audio file
        stereo_file = os.path.join(self.test_dir, "stereo.wav")
        t = np.linspace(0, 3, 3 * self.sr, endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 550 * t)
        stereo = np.column_stack((left, right))
        sf.write(stereo_file, stereo, self.sr)

        output_file = os.path.join(self.test_dir, "stereo_paused.wav")

        output_ann = calculate_time_distance(
            stereo_file,
            self.ann_file,
            output_file
        )

        if output_ann:
            assert os.path.exists(output_file)
            # Verify stereo is preserved
            output_audio, _ = sf.read(output_file)
            assert output_audio.ndim == 2  # Should be stereo

    def test_extreme_pause_durations(self):
        """Test with very long pauses."""
        # Create annotation with a very long pause
        ann_file = os.path.join(self.test_dir, "long_pause.ann")
        with open(ann_file, 'w', encoding='utf-8') as f:
            f.write("0.1\t0.5\t60\t100\n")
            f.write("0.6\t1.0\t62\t100\n")
            f.write("4.0\t4.5\t64\t100\n")  # 3 second pause

        output_file = os.path.join(self.test_dir, "long_pause_out.wav")

        output_ann = calculate_time_distance(
            self.audio_file,
            ann_file,
            output_file,
            min_pause_duration=2.0,
            max_pause_duration=4.0
        )

        if output_ann:
            assert os.path.exists(output_file)
            assert os.path.exists(output_ann)