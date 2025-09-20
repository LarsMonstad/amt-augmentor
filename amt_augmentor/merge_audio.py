from typing_extensions import Union
import librosa
import numpy as np
import soundfile as sf
import os
from typing import Any, List, Tuple


def merge_audios(
    audios4merge: List[str],
    standardized_audio: str,
    temp_ann: str,
    input_directory: str,
    output_directory: str,
    output_filename: str,
    target_sr: int = 44100,
) -> str:
    audios: List[Tuple[np.ndarray, Union[int, float]]] = [
        librosa.load(os.path.join(input_directory, x), sr=None) for x in audios4merge
    ] + [librosa.load(standardized_audio, sr=None)]
    audios_resampled: List[np.ndarray] = [
        librosa.resample(x[0], orig_sr=x[1], target_sr=target_sr) for x in audios
    ]
    audios_max_length: int = max(len(x) for x in audios_resampled)
    aduis_padded: np.ndarray = np.array(
        [np.pad(x, (0, audios_max_length - len(x))) for x in audios_resampled]
    )
    audios_merged: np.ndarray = np.sum(aduis_padded, axis=0)

    print(len(audios_merged))
    ann_files: List[str] = [
        os.path.join(output_directory, x.rsplit(".", 1)[0] + "_temp.ann")
        for x in audios4merge
    ] + [temp_ann]
    ann_merged: str = ""
    for ann_file in ann_files:
        with open(ann_file) as f:
            ann_merged += f.read()

    with open(os.path.join(output_directory, output_filename + ".ann"), "w") as f:
        f.write(ann_merged)

    sf.write(
        os.path.join(output_directory, output_filename + ".wav"),
        audios_merged,
        target_sr,
    )
    return os.path.join(output_directory, output_filename + ".ann")
