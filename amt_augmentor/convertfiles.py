import librosa
import soundfile as sf
import os


def standardize_audio(input_file, target_sr=44100):
    """
    Standardize audio file to 44.1kHz WAV format by overwriting the original if needed.
    Returns:
        tuple: (file_path, was_converted)
    """
    # Load audio with original sample rate
    y, sr = librosa.load(input_file, sr=None)

    # Check if conversion is needed
    base, ext = os.path.splitext(input_file)
    needs_conversion = sr != target_sr or ext.lower() not in [".wav"]

    if needs_conversion:
        # Resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Create temporary filename and specify the format explicitly
        temp_file = input_file + ".temp"

        # Save as WAV with explicit format specification
        sf.write(temp_file, y, target_sr, format="WAV", subtype="PCM_16")

        # Verify the temp file was written before removing the original — if the
        # write was interrupted or sf.write produced an empty file we must NOT
        # delete the user's source audio.
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
            try:
                os.remove(temp_file)
            except OSError:
                pass
            raise IOError(
                f"Failed to write standardized audio for {input_file}; "
                "original file is unchanged."
            )

        new_path = base + ".wav"
        # If the original is .wav we'd be replacing it with itself; rename then
        # the temp into place. Otherwise rename the temp first, *then* delete
        # the original — so an interrupted run never loses the source.
        if new_path == input_file:
            os.replace(temp_file, new_path)
        else:
            os.rename(temp_file, new_path)
            os.remove(input_file)
        print(f"Converted {os.path.basename(input_file)} to 44.1kHz WAV")
        return new_path, True

    return input_file, False


def process_audio_directory(input_directory, target_sr=44100):
    """
    Convert all audio files in directory to 44.1kHz WAV.
    """
    audio_files = [
        f
        for f in os.listdir(input_directory)
        if f.endswith((".wav", ".flac", ".mp3", ".m4a", ".aiff"))
    ]

    for audio_file in audio_files:
        input_path = os.path.join(input_directory, audio_file)
        try:
            standardized_path, was_converted = standardize_audio(input_path, target_sr)
            if was_converted:
                print(f"Converted {audio_file} to 44.1kHz WAV")
            else:
                print(f"{audio_file} already in correct format")

        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standardize audio files to 44.1kHz WAV"
    )
    parser.add_argument("input_directory", help="Directory containing audio files")
    parser.add_argument(
        "--sr", type=int, default=44100, help="Target sample rate (default: 44100)"
    )

    args = parser.parse_args()

    process_audio_directory(args.input_directory, args.sr)
    print("\nProcessing complete.")
