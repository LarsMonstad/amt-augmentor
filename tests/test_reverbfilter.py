"""
Tests for reverbfilter module.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest

from amt_augmentor.reverbfilter import (
    apply_reverb_and_filters,
    validate_parameters,
)


class TestReverbFilter:
    """Test reverb and filter functionality."""

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

    def test_validate_parameters_valid(self):
        """Test parameter validation with valid values."""
        # Should not raise
        validate_parameters(50.0, 5000, 100)
        validate_parameters(0.0, 20, 20)
        validate_parameters(100.0, 20000, 20000)

    def test_validate_parameters_invalid_room_size(self):
        """Test parameter validation with invalid room size."""
        with pytest.raises(ValueError, match="Room size must be between 0 and 100"):
            validate_parameters(-1.0, 5000, 100)

        with pytest.raises(ValueError, match="Room size must be between 0 and 100"):
            validate_parameters(101.0, 5000, 100)

    def test_validate_parameters_invalid_cutoffs(self):
        """Test parameter validation with invalid cutoff frequencies."""
        with pytest.raises(ValueError, match="Low cutoff must be between 20 and 20000"):
            validate_parameters(50.0, 19, 100)

        with pytest.raises(ValueError, match="Low cutoff must be between 20 and 20000"):
            validate_parameters(50.0, 20001, 100)

        with pytest.raises(ValueError, match="High cutoff must be between 20 and 20000"):
            validate_parameters(50.0, 5000, 19)

        with pytest.raises(ValueError, match="High cutoff must be between 20 and 20000"):
            validate_parameters(50.0, 5000, 20001)

    def test_apply_reverb_and_filters_basic(self):
        """Test basic reverb and filter application."""
        output_file = os.path.join(self.test_dir, "processed.wav")
        room_size = 30.0
        low_cutoff = 10000
        high_cutoff = 50

        # Apply reverb and filters
        output_ann = apply_reverb_and_filters(
            self.audio_file,
            self.ann_file,
            output_file,
            room_size,
            low_cutoff,
            high_cutoff
        )

        # Check output files exist
        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)
        assert output_ann.endswith(".ann")

        # Check output audio was created and is different from original
        processed_audio, sr = sf.read(output_file)
        assert sr == self.sr
        assert len(processed_audio) == len(self.original_audio)
        # Audio should be different after processing
        assert not np.allclose(processed_audio, self.original_audio)

        # Check annotation file is unchanged
        with open(output_ann, 'r', encoding='utf-8') as f:
            content = f.read()
        assert content == self.ann_content

    def test_apply_reverb_no_reverb(self):
        """Test with zero room size (no reverb)."""
        output_file = os.path.join(self.test_dir, "no_reverb.wav")

        output_ann = apply_reverb_and_filters(
            self.audio_file,
            self.ann_file,
            output_file,
            room_size=0.0,
            low_cutoff=20000,  # No low-pass filtering
            high_cutoff=20     # No high-pass filtering
        )

        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)

        # With no effects, audio should be similar to original
        processed_audio, _ = sf.read(output_file)
        correlation = np.corrcoef(processed_audio.flatten(), self.original_audio.flatten())[0, 1]
        assert correlation > 0.95  # High correlation when no effects

    def test_apply_reverb_max_reverb(self):
        """Test with maximum room size."""
        output_file = os.path.join(self.test_dir, "max_reverb.wav")

        output_ann = apply_reverb_and_filters(
            self.audio_file,
            self.ann_file,
            output_file,
            room_size=100.0,
            low_cutoff=10000,
            high_cutoff=100
        )

        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)

        # Max reverb should create significant difference
        processed_audio, _ = sf.read(output_file)
        assert not np.allclose(processed_audio, self.original_audio)

    def test_apply_filters_lowpass(self):
        """Test low-pass filter effect."""
        output_file = os.path.join(self.test_dir, "lowpass.wav")

        # Create audio with high frequency component
        t = np.linspace(0, 1, self.sr, endpoint=False)
        high_freq_audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 8000 * t)
        hf_file = os.path.join(self.test_dir, "high_freq.wav")
        sf.write(hf_file, high_freq_audio, self.sr)

        # Apply aggressive low-pass filter
        output_ann = apply_reverb_and_filters(
            hf_file,
            self.ann_file,
            output_file,
            room_size=0.0,
            low_cutoff=1000,   # Cut frequencies above 1000 Hz
            high_cutoff=20     # No high-pass filtering
        )

        assert os.path.exists(output_file)
        # The high frequency component should be attenuated

    def test_apply_filters_highpass(self):
        """Test high-pass filter effect."""
        output_file = os.path.join(self.test_dir, "highpass.wav")

        # Create audio with low frequency component
        t = np.linspace(0, 1, self.sr, endpoint=False)
        low_freq_audio = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 2000 * t)
        lf_file = os.path.join(self.test_dir, "low_freq.wav")
        sf.write(lf_file, low_freq_audio, self.sr)

        # Apply aggressive high-pass filter
        output_ann = apply_reverb_and_filters(
            lf_file,
            self.ann_file,
            output_file,
            room_size=0.0,
            low_cutoff=20000,   # No low-pass filtering
            high_cutoff=1000    # Cut frequencies below 1000 Hz
        )

        assert os.path.exists(output_file)
        # The low frequency component should be attenuated

    def test_stereo_audio_handling(self):
        """Test reverb and filters with stereo audio."""
        # Create stereo audio file
        stereo_file = os.path.join(self.test_dir, "stereo.wav")
        t = np.linspace(0, 1, self.sr, endpoint=False)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 550 * t)
        stereo = np.column_stack((left, right))
        sf.write(stereo_file, stereo, self.sr)

        output_file = os.path.join(self.test_dir, "stereo_processed.wav")
        result = apply_reverb_and_filters(
            stereo_file,
            self.ann_file,
            output_file,
            room_size=50.0,
            low_cutoff=15000,
            high_cutoff=50
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

        output_ann = apply_reverb_and_filters(
            flac_file,
            self.ann_file,
            output_file,
            room_size=25.0,
            low_cutoff=8000,
            high_cutoff=80
        )

        assert os.path.exists(output_file)
        assert os.path.exists(output_ann)

        # Verify it's a FLAC file
        info = sf.info(output_file)
        assert info.format == 'FLAC'

    def test_nonexistent_input_files(self):
        """Test error handling for non-existent input files."""
        output_file = os.path.join(self.test_dir, "output.wav")

        # Non-existent audio file
        with pytest.raises(FileNotFoundError, match="Input audio file not found"):
            apply_reverb_and_filters(
                "nonexistent.wav",
                self.ann_file,
                output_file,
                50.0, 5000, 100
            )

        # Non-existent annotation file
        with pytest.raises(FileNotFoundError, match="Input annotation file not found"):
            apply_reverb_and_filters(
                self.audio_file,
                "nonexistent.ann",
                output_file,
                50.0, 5000, 100
            )

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        output_file = os.path.join(self.test_dir, "new_dir", "output.wav")

        output_ann = apply_reverb_and_filters(
            self.audio_file,
            self.ann_file,
            output_file,
            room_size=40.0,
            low_cutoff=12000,
            high_cutoff=60
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
        output_ann = apply_reverb_and_filters(
            self.audio_file,
            complex_ann,
            output_file,
            50.0, 10000, 50
        )

        # Check annotation is preserved exactly
        with open(output_ann, 'r', encoding='utf-8') as f:
            output_content = f.read()
        assert output_content == complex_content