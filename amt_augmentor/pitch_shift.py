import os
import argparse
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


def update_ann_file(ann_content, pitch_shift):
    updated_content = []
    for line in ann_content:
        parts = line.strip().split("\t")
        onset, offset, pitch, channel = parts
        pitch_new = int(pitch) + pitch_shift
        updated_line = f"{onset}\t{offset}\t{pitch_new}\t{channel}"
        updated_content.append(updated_line)
    return updated_content


def apply_pitch_shift(audio_file, ann_file, output_file_path, pitch_shift):
    samples, sample_rate = librosa.load(audio_file, sr=None, mono=False)
    pitch_shifted_samples = librosa.effects.pitch_shift(
        samples, sr=sample_rate, n_steps=pitch_shift
    )

    # Determine output format based on file extension
    output_format = "WAV" if output_file_path.lower().endswith(".wav") else "FLAC"

    sf.write(
        output_file_path, pitch_shifted_samples.T, sample_rate, format=output_format
    )

    ann_content = load_ann_file(ann_file)
    updated_ann_content = update_ann_file(ann_content, pitch_shift)
    output_ann_file = (
        os.path.splitext(output_file_path)[0] + os.path.splitext(ann_file)[1]
    )
    save_ann_file(output_ann_file, updated_ann_content)
    return output_ann_file


def main():
    parser = argparse.ArgumentParser(
        description="Pitch-shift audio and update annotation files"
    )
    parser.add_argument(
        "input_audio_file", help="Path to the input audio file (FLAC or WAV)"
    )
    parser.add_argument(
        "input_ann_file", help="Path to the input annotation file (.ann)"
    )
    parser.add_argument("output_directory", help="Path to the output directory")
    parser.add_argument(
        "pitch_shift",
        type=int,
        help="Pitch shift in semitones (e.g., 2 for two semitones up, -3 for three semitones down)",
    )

    args = parser.parse_args()

    output_file_name = os.path.basename(args.input_audio_file)
    output_file_path = os.path.join(args.output_directory, output_file_name)

    apply_pitch_shift(
        args.input_audio_file, args.input_ann_file, output_file_path, args.pitch_shift
    )


if __name__ == "__main__":
    main()
