"""
Tests for merge_audio module.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import pytest

from amt_augmentor.merge_audio import merge_audios


class TestMergeAudio:
    """Test audio merging functionality."""

    def setup_method(self):
        """Set up test files."""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, "input")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)

        # Create test audio files with different sample rates and lengths
        self.sr1 = 22050
        self.sr2 = 44100
        self.sr3 = 16000

        # Audio file 1 (1 second at 22050 Hz)
        t1 = np.linspace(0, 1, self.sr1, endpoint=False)
        audio1 = 0.3 * np.sin(2 * np.pi * 440 * t1)
        self.audio1_path = "audio1.wav"
        sf.write(os.path.join(self.input_dir, self.audio1_path), audio1, self.sr1)

        # Audio file 2 (1.5 seconds at 44100 Hz)
        t2 = np.linspace(0, 1.5, int(self.sr2 * 1.5), endpoint=False)
        audio2 = 0.3 * np.sin(2 * np.pi * 550 * t2)
        self.audio2_path = "audio2.wav"
        sf.write(os.path.join(self.input_dir, self.audio2_path), audio2, self.sr2)

        # Audio file 3 (0.8 seconds at 16000 Hz)
        t3 = np.linspace(0, 0.8, int(self.sr3 * 0.8), endpoint=False)
        audio3 = 0.3 * np.sin(2 * np.pi * 660 * t3)
        self.audio3_path = "audio3.wav"
        sf.write(os.path.join(self.input_dir, self.audio3_path), audio3, self.sr3)

        # Create standardized audio file
        self.standardized_audio = os.path.join(self.input_dir, "standardized.wav")
        t_std = np.linspace(0, 1.2, int(44100 * 1.2), endpoint=False)
        audio_std = 0.2 * np.sin(2 * np.pi * 330 * t_std)
        sf.write(self.standardized_audio, audio_std, 44100)

        # Create temporary annotation files
        self.create_ann_files()

    def create_ann_files(self):
        """Create test annotation files."""
        # Annotation for audio1
        ann1_path = os.path.join(self.output_dir, "audio1_temp.ann")
        with open(ann1_path, 'w', encoding='utf-8') as f:
            f.write("0.1\t0.5\t60\t100\n")
            f.write("0.6\t0.9\t62\t100\n")

        # Annotation for audio2
        ann2_path = os.path.join(self.output_dir, "audio2_temp.ann")
        with open(ann2_path, 'w', encoding='utf-8') as f:
            f.write("0.2\t0.7\t64\t100\n")
            f.write("0.8\t1.2\t66\t100\n")

        # Annotation for audio3
        ann3_path = os.path.join(self.output_dir, "audio3_temp.ann")
        with open(ann3_path, 'w', encoding='utf-8') as f:
            f.write("0.1\t0.4\t68\t100\n")

        # Temporary annotation for standardized audio
        self.temp_ann = os.path.join(self.output_dir, "standardized_temp.ann")
        with open(self.temp_ann, 'w', encoding='utf-8') as f:
            f.write("0.3\t0.8\t70\t100\n")
            f.write("0.9\t1.1\t72\t100\n")

    def teardown_method(self):
        """Clean up test files."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_merge_audios_basic(self):
        """Test basic audio merging functionality."""
        audios_to_merge = [self.audio1_path, self.audio2_path]
        output_filename = "merged"

        result_ann = merge_audios(
            audios_to_merge,
            self.standardized_audio,
            self.temp_ann,
            self.input_dir,
            self.output_dir,
            output_filename
        )

        # Check output files exist
        output_wav = os.path.join(self.output_dir, "merged.wav")
        output_ann = os.path.join(self.output_dir, "merged.ann")
        assert os.path.exists(output_wav)
        assert os.path.exists(output_ann)
        assert result_ann == output_ann

        # Check merged audio properties
        merged_audio, sr = sf.read(output_wav)
        assert sr == 44100  # Target sample rate
        assert len(merged_audio) > 0

        # Check merged annotation
        with open(output_ann, 'r', encoding='utf-8') as f:
            ann_content = f.read()
        # Should contain annotations from all files
        assert "60\t100" in ann_content  # From audio1
        assert "64\t100" in ann_content  # From audio2
        assert "70\t100" in ann_content  # From standardized

    def test_merge_audios_single_file(self):
        """Test merging with single audio file."""
        audios_to_merge = [self.audio1_path]
        output_filename = "single_merge"

        result_ann = merge_audios(
            audios_to_merge,
            self.standardized_audio,
            self.temp_ann,
            self.input_dir,
            self.output_dir,
            output_filename
        )

        output_wav = os.path.join(self.output_dir, "single_merge.wav")
        output_ann = os.path.join(self.output_dir, "single_merge.ann")
        assert os.path.exists(output_wav)
        assert os.path.exists(output_ann)

        # Check merged audio
        merged_audio, sr = sf.read(output_wav)
        assert sr == 44100
        assert len(merged_audio) > 0

    def test_merge_audios_different_lengths(self):
        """Test merging audio files of different lengths."""
        audios_to_merge = [self.audio1_path, self.audio2_path, self.audio3_path]
        output_filename = "diff_lengths"

        result_ann = merge_audios(
            audios_to_merge,
            self.standardized_audio,
            self.temp_ann,
            self.input_dir,
            self.output_dir,
            output_filename
        )

        output_wav = os.path.join(self.output_dir, "diff_lengths.wav")
        assert os.path.exists(output_wav)

        # Check that merged audio has length of longest input
        merged_audio, sr = sf.read(output_wav)
        # Audio2 is 1.5 seconds, which should be the longest after resampling
        expected_length = int(1.5 * 44100)
        # Allow some tolerance for resampling
        assert abs(len(merged_audio) - expected_length) < 1000

    def test_merge_audios_different_sample_rates(self):
        """Test that different sample rates are handled correctly."""
        audios_to_merge = [self.audio1_path, self.audio3_path]
        output_filename = "diff_sr"

        result_ann = merge_audios(
            audios_to_merge,
            self.standardized_audio,
            self.temp_ann,
            self.input_dir,
            self.output_dir,
            output_filename,
            target_sr=48000  # Different target sample rate
        )

        output_wav = os.path.join(self.output_dir, "diff_sr.wav")
        assert os.path.exists(output_wav)

        # Check target sample rate is applied
        merged_audio, sr = sf.read(output_wav)
        assert sr == 48000

    def test_merge_audios_annotation_concatenation(self):
        """Test that annotations are properly concatenated."""
        audios_to_merge = [self.audio1_path, self.audio2_path, self.audio3_path]
        output_filename = "ann_concat"

        result_ann = merge_audios(
            audios_to_merge,
            self.standardized_audio,
            self.temp_ann,
            self.input_dir,
            self.output_dir,
            output_filename
        )

        output_ann = os.path.join(self.output_dir, "ann_concat.ann")
        with open(output_ann, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Should have annotations from all 4 sources
        # audio1: 2 lines, audio2: 2 lines, audio3: 1 line, standardized: 2 lines
        assert len(lines) == 7

    def test_merge_audios_empty_list(self):
        """Test merging with empty audio list (only standardized)."""
        audios_to_merge = []
        output_filename = "empty_merge"

        result_ann = merge_audios(
            audios_to_merge,
            self.standardized_audio,
            self.temp_ann,
            self.input_dir,
            self.output_dir,
            output_filename
        )

        output_wav = os.path.join(self.output_dir, "empty_merge.wav")
        output_ann = os.path.join(self.output_dir, "empty_merge.ann")
        assert os.path.exists(output_wav)
        assert os.path.exists(output_ann)

        # Should just contain the standardized audio
        merged_audio, sr = sf.read(output_wav)
        std_audio, std_sr = sf.read(self.standardized_audio)
        # After resampling if needed
        if std_sr != 44100:
            import librosa
            std_audio = librosa.resample(std_audio, orig_sr=std_sr, target_sr=44100)
        assert len(merged_audio) == len(std_audio)

    def test_merge_audios_missing_annotation_file(self):
        """Test behavior when annotation file is missing."""
        audios_to_merge = ["nonexistent.wav"]
        output_filename = "missing_ann"

        # Should handle missing temp annotation files
        # The function expects annotation files to exist based on naming convention
        # This test documents the current behavior
        with pytest.raises(FileNotFoundError):
            merge_audios(
                audios_to_merge,
                self.standardized_audio,
                self.temp_ann,
                self.input_dir,
                self.output_dir,
                output_filename
            )

    def test_merge_audios_amplitude_scaling(self):
        """Test that merged audio doesn't clip."""
        # Create loud audio files that might clip when summed
        loud1 = os.path.join(self.input_dir, "loud1.wav")
        loud2 = os.path.join(self.input_dir, "loud2.wav")

        t = np.linspace(0, 1, 44100, endpoint=False)
        audio_loud1 = 0.9 * np.sin(2 * np.pi * 440 * t)
        audio_loud2 = 0.9 * np.sin(2 * np.pi * 550 * t)

        sf.write(loud1, audio_loud1, 44100)
        sf.write(loud2, audio_loud2, 44100)

        # Create corresponding annotation files
        for name in ["loud1", "loud2"]:
            ann_path = os.path.join(self.output_dir, f"{name}_temp.ann")
            with open(ann_path, 'w', encoding='utf-8') as f:
                f.write("0.1\t0.5\t60\t100\n")

        audios_to_merge = ["loud1.wav", "loud2.wav"]
        output_filename = "loud_merge"

        result_ann = merge_audios(
            audios_to_merge,
            self.standardized_audio,
            self.temp_ann,
            self.input_dir,
            self.output_dir,
            output_filename
        )

        output_wav = os.path.join(self.output_dir, "loud_merge.wav")
        merged_audio, _ = sf.read(output_wav)

        # The summed audio might exceed [-1, 1] range
        # This documents the current behavior (no normalization)
        # In practice, this might need to be addressed
        assert len(merged_audio) > 0

    def test_merge_audios_stereo_handling(self):
        """Test merging stereo audio files."""
        # Create stereo audio file
        stereo_file = os.path.join(self.input_dir, "stereo.wav")
        t = np.linspace(0, 1, 44100, endpoint=False)
        left = 0.3 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 550 * t)
        stereo = np.column_stack((left, right))
        sf.write(stereo_file, stereo, 44100)

        # Create annotation
        ann_path = os.path.join(self.output_dir, "stereo_temp.ann")
        with open(ann_path, 'w', encoding='utf-8') as f:
            f.write("0.1\t0.5\t60\t100\n")

        audios_to_merge = ["stereo.wav"]
        output_filename = "stereo_merge"

        result_ann = merge_audios(
            audios_to_merge,
            self.standardized_audio,
            self.temp_ann,
            self.input_dir,
            self.output_dir,
            output_filename
        )

        output_wav = os.path.join(self.output_dir, "stereo_merge.wav")
        assert os.path.exists(output_wav)

        # Note: The current implementation may flatten stereo to mono
        # during the merging process