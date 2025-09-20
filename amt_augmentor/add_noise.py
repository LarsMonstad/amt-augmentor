import os
import librosa
import soundfile as sf
import numpy as np


def apply_noise(audio_file, ann_file, output_file_path, intensity):
    samples, sample_rate = librosa.load(audio_file, sr=None, mono=False)

    noise = np.random.normal(0, 1, samples.shape)

    # Determine output format based on file extension
    output_format = "WAV" if output_file_path.lower().endswith(".wav") else "FLAC"

    # Add noise with intensity scaling and normalize
    noise_audio = samples + noise * intensity
    noise_audio = librosa.util.normalize(noise_audio)

    # Transpose stereo audio for soundfile (expects [samples, channels])
    if noise_audio.ndim > 1:
        noise_audio = noise_audio.T

    sf.write(
        output_file_path, noise_audio, sample_rate, format=output_format
    )

    output_ann_file = (
        os.path.splitext(output_file_path)[0] + os.path.splitext(ann_file)[1]
    )

    # Copy paste ANN : it is not changed at all
    with open(ann_file, 'r', encoding='utf-8') as ref:
        ref_data = ref.read()
        with open(output_ann_file, 'w', encoding='utf-8') as clone:
            clone.write(ref_data)

    return output_ann_file

