import os
import sys
import argparse
import random
import librosa
import numpy as np
import soundfile as sf


def load_ann_file(file_path):
    with open(file_path, "r") as f:
        content = f.readlines()
    return [line.strip() for line in content]


def save_ann_file(file_path, content):
    with open(file_path, "w") as f:
        f.write("\n".join(content))


def update_ann_file(ann_content, stretch_factor):
    updated_content = []
    for line in ann_content:
        parts = line.strip().split("\t")
        if len(parts) != 4:
            # Skip malformed lines
            continue
        onset, offset, pitch, channel = parts
        # When audio is stretched by a factor (e.g., 1.2x = 20% faster),
        # the timestamps need to be divided by that factor to stay in sync
        onset_new = float(onset) / stretch_factor
        offset_new = float(offset) / stretch_factor
        updated_line = f"{onset_new:.3f}\t{offset_new:.3f}\t{pitch}\t{channel}"
        updated_content.append(updated_line)
    return updated_content


def apply_time_stretch(audio_file, ann_file, output_file_path, stretch_factor):
    # Determine input and output formats
    input_format = os.path.splitext(audio_file)[1].lower()
    output_format = os.path.splitext(output_file_path)[1].lower()

    if input_format not in [".wav", ".flac"]:
        raise ValueError("Input audio file must be WAV or FLAC format")

    if output_format not in [".wav", ".flac"]:
        raise ValueError("Output audio file must be WAV or FLAC format")

    # Load audio file
    try:
        y, sr = librosa.load(audio_file, sr=None, mono=False)
    except Exception as e:
        raise Exception(f"Error loading audio file: {str(e)}")

    # Handle both mono and stereo files
    if y.ndim == 1:
        # Mono file - time stretch directly
        time_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
    else:
        # Stereo file - time stretch each channel separately
        time_stretched = np.vstack(
            [
                librosa.effects.time_stretch(channel, rate=stretch_factor)
                for channel in y
            ]
        )

    # Save the time-stretched audio
    try:
        # Transpose if stereo to match soundfile's expected format
        if time_stretched.ndim > 1:
            time_stretched = time_stretched.T

        # Determine format for soundfile
        save_format = "FLAC" if output_format == ".flac" else "WAV"
        sf.write(output_file_path, time_stretched, sr, format=save_format)
    except Exception as e:
        raise Exception(f"Error saving audio file: {str(e)}")

    # Update and save the time-stretched annotations
    try:
        ann_content = load_ann_file(ann_file)
        updated_ann_content = update_ann_file(ann_content, stretch_factor)
        output_ann_file = (
            os.path.splitext(output_file_path)[0] + os.path.splitext(ann_file)[1]
        )
        save_ann_file(output_ann_file, updated_ann_content)
        return output_ann_file
    except Exception as e:
        raise Exception(f"Error processing annotation file: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Time-stretch audio and annotation files"
    )
    parser.add_argument(
        "input_audio_file", help="Path to the input audio file (FLAC or WAV)"
    )
    parser.add_argument(
        "input_ann_file", help="Path to the input annotation file (.ann)"
    )
    parser.add_argument("output_directory", help="Path to the output directory")
    parser.add_argument(
        "stretch_factor",
        type=float,
        help="Time stretch factor (e.g., 1.4 for 40% faster, 0.8 for 20% slower)",
    )

    args = parser.parse_args()

    # Make sure the output directory exists
    os.makedirs(args.output_directory, exist_ok=True)

    # Generate output file path for the audio file
    output_file_name = os.path.basename(args.input_audio_file)
    output_file_path = os.path.join(args.output_directory, output_file_name)

    try:
        apply_time_stretch(
            args.input_audio_file,
            args.input_ann_file,
            output_file_path,
            args.stretch_factor,
        )
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
