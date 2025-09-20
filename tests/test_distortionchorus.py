"""
Tests for distortionchorus module.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest

from amt_augmentor.distortionchorus import apply_gain_and_chorus, generate_output_filename


class TestDistortionChorus:
    """Test distortion and chorus functionality."""

    def setup_method(self):
        """Set up test files."""
        self.test_dir = tempfile.mkdtemp()

        # Create test audio file
        self.audio_file = os.path.join(self.test_dir, "test.wav")
        self.sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        self.original_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(self.audio_file, self.original_audio, self.sr)

        # Create test annotation file
        self.ann_file = os.path.join(self.test_dir, "test.ann")
        self.ann_content = "0.1\t0.5\t60\t100\n0.6\t1.0\t62\t100\n1.1\t1.5\t64\t100\n"
        with open(self.ann_file, 'w', encoding='utf-8') as f:
            f.write(self.ann_content)

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_generate_output_filename(self):
        """Test output filename generation."""
        input_name = "test.wav"
        suffix = "abc123"
        output = generate_output_filename(input_name, suffix)
        assert output == "test_gain_chorus_abc123.wav"

        # Test with different extension
        input_name = "audio.flac"
        output = generate_output_filename(input_name, suffix)
        assert output == "audio_gain_chorus_abc123.flac"

    def test_apply_gain_and_chorus_basic(self):
        """Test basic gain and chorus application."""
        output_file = os.path.join(self.test_dir, "processed.wav")
        gain = 10.0
        chorus_depth = 0.5
        chorus_rate = 2.0

        # Apply gain and chorus
        output_ann = apply_gain_and_chorus(
            self.audio_file,
            self.ann_file,
            output_file,
            gain,
            chorus_depth,
            chorus_rate
        )

        # Check output files exist
        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)
        assert output_ann.endswith(".ann")

        # Check output audio was created and is different from original
        processed_audio, sr = sf.read(output_file)
        assert sr == self.sr
        # Audio length should be similar (chorus might add slight delay)
        assert abs(len(processed_audio) - len(self.original_audio)) < self.sr * 0.1
        # Audio should be different after processing
        assert not np.allclose(processed_audio[:len(self.original_audio)], self.original_audio)

        # Check annotation file is unchanged
        with open(output_ann, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == self.ann_content

    def test_apply_gain_only(self):
        """Test with gain only (no chorus)."""
        output_file = os.path.join(self.test_dir, "gain_only.wav")

        output_ann = apply_gain_and_chorus(
            self.audio_file,
            self.ann_file,
            output_file,
            gain=20.0,
            chorus_depth=0.0,  # No chorus
            chorus_rate=0.0
        )

        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)

        # Audio should be distorted but not chorused
        processed_audio, _ = sf.read(output_file)
        assert not np.allclose(processed_audio[:len(self.original_audio)], self.original_audio)

    def test_apply_chorus_only(self):
        """Test with chorus only (minimal gain)."""
        output_file = os.path.join(self.test_dir, "chorus_only.wav")

        output_ann = apply_gain_and_chorus(
            self.audio_file,
            self.ann_file,
            output_file,
            gain=0.0,  # Minimal gain
            chorus_depth=0.8,
            chorus_rate=3.0
        )

        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)

        # Audio should have chorus effect
        processed_audio, _ = sf.read(output_file)
        assert not np.allclose(processed_audio[:len(self.original_audio)], self.original_audio)

    def test_extreme_gain_values(self):
        """Test with extreme gain values."""
        # High gain (heavy distortion)
        output_high = os.path.join(self.test_dir, "high_gain.wav")
        result = apply_gain_and_chorus(
            self.audio_file,
            self.ann_file,
            output_high,
            gain=50.0,
            chorus_depth=0.5,
            chorus_rate=2.0
        )
        assert os.path.exists(output_high)
        assert os.path.exists(result)

        # Negative gain (should still work)
        output_neg = os.path.join(self.test_dir, "neg_gain.wav")
        result = apply_gain_and_chorus(
            self.audio_file,
            self.ann_file,
            output_neg,
            gain=-10.0,
            chorus_depth=0.5,
            chorus_rate=2.0
        )
        assert os.path.exists(output_neg)
        assert os.path.exists(result)

    def test_extreme_chorus_values(self):
        """Test with extreme chorus values."""
        # Maximum chorus depth
        output_max_depth = os.path.join(self.test_dir, "max_depth.wav")
        result = apply_gain_and_chorus(
            self.audio_file,
            self.ann_file,
            output_max_depth,
            gain=5.0,
            chorus_depth=1.0,  # Max depth
            chorus_rate=2.0
        )
        assert os.path.exists(output_max_depth)
        assert os.path.exists(result)

        # High chorus rate
        output_high_rate = os.path.join(self.test_dir, "high_rate.wav")
        result = apply_gain_and_chorus(
            self.audio_file,
            self.ann_file,
            output_high_rate,
            gain=5.0,
            chorus_depth=0.5,
            chorus_rate=10.0  # Fast chorus
        )
        assert os.path.exists(output_high_rate)
        assert os.path.exists(result)

    def test_stereo_audio_handling(self):
        """Test gain and chorus with stereo audio."""
        # Create stereo audio file
        stereo_file = os.path.join(self.test_dir, "stereo.wav")
        t = np.linspace(0, 1, self.sr, endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 550 * t)
        stereo = np.column_stack((left, right))
        sf.write(stereo_file, stereo, self.sr)

        output_file = os.path.join(self.test_dir, "stereo_processed.wav")
        result = apply_gain_and_chorus(
            stereo_file,
            self.ann_file,
            output_file,
            gain=15.0,
            chorus_depth=0.6,
            chorus_rate=2.5
        )

        assert os.path.exists(output_file)
        assert os.path.exists(result)

        # Verify stereo is preserved
        output_audio, _ = sf.read(output_file)
        assert output_audio.ndim == 2  # Should be stereo

    def test_flac_format(self):
        """Test with FLAC input/output format."""
        # Create FLAC input file
        flac_file = os.path.join(self.test_dir, "test.flac")
        sf.write(flac_file, self.original_audio, self.sr, format='FLAC')

        output_file = os.path.join(self.test_dir, "processed.flac")

        output_ann = apply_gain_and_chorus(
            flac_file,
            self.ann_file,
            output_file,
            gain=12.0,
            chorus_depth=0.4,
            chorus_rate=1.5
        )

        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)

        # Verify it's a FLAC file
        info = sf.info(output_file)
        assert info.format == 'FLAC'

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        output_file = os.path.join(self.test_dir, "new_dir", "output.wav")

        output_ann = apply_gain_and_chorus(
            self.audio_file,
            self.ann_file,
            output_file,
            gain=8.0,
            chorus_depth=0.3,
            chorus_rate=2.0
        )

        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)
        assert os.path.exists(os.path.join(self.test_dir, "new_dir"))

    def test_annotation_preservation(self):
        """Test that annotation files are preserved exactly."""
        # Create annotation with various formats
        complex_ann = os.path.join(self.test_dir, "complex.ann")
        complex_content = "0.1\t0.5\t60\t100\n# Comment line\n0.6\t1.0\t62\t100\n\n1.1\t1.5\t64\t100\n"
        with open(complex_ann, 'w', encoding='utf-8') as f:
            f.write(complex_content)

        output_file = os.path.join(self.test_dir, "output.wav")
        output_ann = apply_gain_and_chorus(
            self.audio_file,
            complex_ann,
            output_file,
            10.0, 0.5, 2.0
        )

        # Check annotation is preserved exactly
        with open(output_ann, 'r', encoding='utf-8') as f:
            output_content = f.read()
        assert output_content == complex_content

    def test_various_parameter_combinations(self):
        """Test various combinations of gain and chorus parameters."""
        test_cases = [
            (0.0, 0.0, 0.0),    # No effects
            (5.0, 0.2, 1.0),     # Light effects
            (20.0, 0.7, 4.0),    # Moderate effects
            (40.0, 1.0, 8.0),    # Heavy effects
        ]

        for i, (gain, depth, rate) in enumerate(test_cases):
            output_file = os.path.join(self.test_dir, f"combo_{i}.wav")
            result = apply_gain_and_chorus(
                self.audio_file,
                self.ann_file,
                output_file,
                gain, depth, rate
            )
            assert os.path.exists(output_file)
            assert os.path.exists(result)

            # Verify audio was processed
            processed, _ = sf.read(output_file)
            assert processed.size > 0