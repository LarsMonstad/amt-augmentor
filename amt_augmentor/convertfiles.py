"""Audio-format conversion helpers.

Conversion is deliberately non-destructive: the source recording is never
replaced or deleted.  Callers receive the path of a separate standardized WAV
and are responsible for removing that working copy when it is no longer
needed.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


def _default_output_path(source: Path) -> Path:
    """Return a distinct, predictable WAV path for a converted source."""

    if source.suffix.lower() == ".wav":
        return source.with_name(f"{source.stem}.standardized.wav")
    return source.with_suffix(".wav")


def standardize_audio(
    input_file: os.PathLike,
    target_sr: int = 44100,
    output_file: Optional[os.PathLike] = None,
) -> Tuple[str, bool]:
    """Return audio in WAV format at ``target_sr`` without mutating the source.

    The input is returned unchanged when it is already a WAV at the requested
    sample rate.  Otherwise, a separate WAV is written to ``output_file``.  If
    no destination is supplied, non-WAV inputs use the same stem with a
    ``.wav`` suffix and WAV inputs use ``<stem>.standardized.wav``.

    Multichannel inputs remain multichannel.  Existing destinations and a
    destination resolving to the source path are rejected.

    Returns:
        ``(file_path, was_converted)``
    """

    if type(target_sr) is not int or target_sr <= 0:
        raise ValueError("target_sr must be a positive built-in int")

    source = Path(input_file)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")

    # ``mono=False`` is essential here: librosa otherwise folds every input
    # down to one channel before we have a chance to inspect or resample it.
    audio, source_sr = librosa.load(str(source), sr=None, mono=False)
    audio = np.asarray(audio)
    if audio.ndim not in (1, 2) or audio.size == 0:
        raise ValueError(f"Audio must contain one or more non-empty channels: {source}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio contains NaN or infinite samples: {source}")

    needs_conversion = source_sr != target_sr or source.suffix.lower() != ".wav"
    if not needs_conversion:
        return str(source), False

    target = (
        Path(output_file)
        if output_file is not None
        else _default_output_path(source)
    )
    if source.resolve(strict=False) == target.resolve(strict=False):
        raise ValueError("Standardized output must be distinct from the source audio")
    if os.path.lexists(target):
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")

    if source_sr != target_sr:
        # librosa represents multichannel audio as channels-by-samples and
        # resamples along the final (sample) axis.
        audio = librosa.resample(
            audio,
            orig_sr=source_sr,
            target_sr=target_sr,
            axis=-1,
        )

    channels = 1 if audio.ndim == 1 else int(audio.shape[0])
    samples_by_channels = audio if audio.ndim == 1 else audio.T
    target.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        sf.write(
            str(temporary),
            samples_by_channels,
            target_sr,
            format="WAV",
            subtype="PCM_16",
        )
        info = sf.info(str(temporary))
        if (
            temporary.stat().st_size <= 0
            or info.frames <= 0
            or info.samplerate != target_sr
            or info.channels != channels
        ):
            raise IOError(
                f"Failed to verify standardized audio for {source}; "
                "source audio is unchanged."
            )

        # Preserve the source's permission bits, then publish with a hard link.
        # The hard-link operation is atomic and fails if another process created
        # the destination after the existence check above.
        os.chmod(temporary, source.stat().st_mode & 0o777)
        os.link(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    print(
        f"Converted {source.name} to {target_sr} Hz WAV at {target}; "
        "source retained"
    )
    return str(target), True


def process_audio_directory(input_directory, target_sr=44100):
    """Create non-destructive standardized copies for audio in a directory."""

    audio_files = [
        f
        for f in os.listdir(input_directory)
        if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".aiff"))
    ]
    standardized_directory = Path(input_directory) / "standardized"

    for audio_file in audio_files:
        input_path = Path(input_directory) / audio_file
        output_path = standardized_directory / f"{input_path.stem}.wav"
        try:
            standardized_path, was_converted = standardize_audio(
                input_path,
                target_sr,
                output_path,
            )
            if was_converted:
                print(f"Standardized copy: {standardized_path}")
            else:
                print(f"{audio_file} already in correct format")

        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create non-destructive 44.1 kHz WAV copies of audio files"
    )
    parser.add_argument("input_directory", help="Directory containing audio files")
    parser.add_argument(
        "--sr", type=int, default=44100, help="Target sample rate (default: 44100)"
    )

    args = parser.parse_args()

    process_audio_directory(args.input_directory, args.sr)
    print("\nProcessing complete.")
