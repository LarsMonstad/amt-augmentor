"""Regression tests for non-destructive, channel-safe standardization."""

import hashlib

import numpy as np
import pytest
import soundfile as sf

from amt_augmentor.convertfiles import standardize_audio


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_standardize_audio_preserves_channels_and_source_bytes(tmp_path):
    source = tmp_path / "stereo.wav"
    target = tmp_path / "work" / "stereo.wav"
    sample_rate = 8000
    samples = np.zeros((8000, 2), dtype=np.float64)
    samples[800, 0] = 0.75
    samples[2400, 1] = -0.50
    sf.write(source, samples, sample_rate, subtype="PCM_16")
    source_hash = _sha256(source)

    output, converted = standardize_audio(
        source,
        target_sr=16000,
        output_file=target,
    )

    assert converted is True
    assert output == str(target)
    assert source.exists()
    assert _sha256(source) == source_hash
    output_samples, output_rate = sf.read(target, always_2d=True)
    assert output_rate == 16000
    assert output_samples.shape == (16000, 2)
    assert np.argmax(np.abs(output_samples[:, 0])) != np.argmax(
        np.abs(output_samples[:, 1])
    )
    assert np.max(np.abs(output_samples[:, 0])) > 0
    assert np.max(np.abs(output_samples[:, 1])) > 0


def test_non_wav_default_conversion_retains_source(tmp_path):
    source = tmp_path / "recording.flac"
    samples = np.column_stack(
        [np.linspace(-0.2, 0.2, 4000), np.linspace(0.2, -0.2, 4000)]
    )
    sf.write(source, samples, 8000)
    source_hash = _sha256(source)

    output, converted = standardize_audio(source, target_sr=8000)

    target = tmp_path / "recording.wav"
    assert converted is True
    assert output == str(target)
    assert source.exists()
    assert _sha256(source) == source_hash
    assert sf.info(target).channels == 2


def test_standardize_audio_refuses_to_overwrite_destination(tmp_path):
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    sf.write(source, np.zeros(4000), 8000)
    target.write_bytes(b"do-not-overwrite")
    source_hash = _sha256(source)
    target_hash = _sha256(target)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        standardize_audio(source, target_sr=16000, output_file=target)

    assert _sha256(source) == source_hash
    assert _sha256(target) == target_hash


def test_already_standard_wav_is_returned_without_copy(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((4000, 2)), 8000)
    source_hash = _sha256(source)

    output, converted = standardize_audio(source, target_sr=8000)

    assert output == str(source)
    assert converted is False
    assert _sha256(source) == source_hash
    assert sorted(tmp_path.iterdir()) == [source]
