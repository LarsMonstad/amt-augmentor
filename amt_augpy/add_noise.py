import os
import librosa
import soundfile as sf
import numpy as np


def apply_noise(audio_file, ann_file, output_file_path, intensity):
    samples, sample_rate = librosa.load(audio_file, sr=None, mono=False)

    noise = np.random.normal(0, 1, samples.size)

    # Determine output format based on file extension
    output_format = "WAV" if output_file_path.lower().endswith(".wav") else "FLAC"
    
    noise_audio = noise + samples
    noise_audio = noise_audio / np.max(np.abs(noise_audio))

    sf.write(
        output_file_path, librosa.util.normalize(samples+noise*intensity), sample_rate, format=output_format
    )

    output_ann_file = (
        os.path.splitext(output_file_path)[0] + os.path.splitext(ann_file)[1]
    )

    # Copy paste ANN : it is not changed at all
    with open(ann_file, 'r') as ref:
        ref_data = ref.read()
        with open(output_ann_file, 'w') as clone:
            clone.write(ref_data)

    return output_ann_file

