# used to create csv files for the data, with default training splits
import os
import csv
import argparse
import librosa
from collections import defaultdict


def is_augmented_version(filename):
    """Check if the file is an augmented version based on the '_augmented_' identifier."""
    return '_augmented_' in filename.lower()


def get_original_song_name(filename):
    """Extract original song name from augmented filename."""
    base_name = os.path.splitext(filename)[0]
    # If the file has the _augmented_ marker, extract the original name
    if '_augmented_' in base_name:
        return base_name.split('_augmented_')[0]
    return base_name


def get_split_status(songs, title, split_ratios):
    """
    Determine split status based on ratios and existing assignments for non-custom songs.
    (This function is used only when the song isn't forced to test.)
    """
    if title in songs:
        return songs[title]

    current_counts = defaultdict(int)
    for split in songs.values():
        current_counts[split] += 1
    total_songs = len(songs) + 1
    targets = {
        "train": int(total_songs * split_ratios["train"]),
        "test": int(total_songs * split_ratios["test"]),
        "validation": int(total_songs * split_ratios["validation"]),
    }

    differences = {
        split: targets[split] - current_counts[split]
        for split in ['train', 'test', 'validation']
    }
    return max(differences, key=differences.get)


def get_wav_duration(file_path):
    """Calculate the duration of a WAV file using librosa."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return round(duration, 2)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0


def create_song_list(directory, split_ratios=None, custom_test_songs=None):
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'test': 0.15, 'validation': 0.15}
    """
    Creates a CSV file with a list of original MIDI-WAV pairs.
    For each original song (non-augmented):
      - If its title (case-insensitive) contains one of the custom test song substrings,
        its split is forced to "test" and no augmented versions are added.
      - Otherwise, its split is determined based on the provided ratios, and augmented
        versions are added if the song is in the training split.
    """
    if custom_test_songs is None:
        custom_test_songs = []
    # Convert custom song names to lower case for comparison.
    custom_test_songs = [s.lower() for s in custom_test_songs]

    directory = os.path.abspath(directory)
    folder_name = os.path.basename(directory)
    csv_filename = f"{folder_name}.csv"

    # Get all files
    all_files = os.listdir(directory)
    song_splits = {}
    split_counts = defaultdict(int)
    original_pairs = []

    # Loop over all non-augmented MIDI files.
    for f in all_files:
        if f.endswith(".mid") and not is_augmented_version(f):
            title = os.path.splitext(f)[0]
            # Find matching WAV file (case-insensitive).
            wav_file = None
            for w in all_files:
                if w.lower() == (title + '.wav').lower():
                    wav_file = w
                    break
            if wav_file:
                original_pairs.append((f, wav_file, title))
                # If the title contains any custom test song substring, force split to "test".
                if any(custom in title.lower() for custom in custom_test_songs):
                    split = 'test'
                else:
                    split = get_split_status(song_splits, title, split_ratios)
                song_splits[title] = split
                split_counts[split] += 1

    rows = []
    headers = [
        'canonical_composer',
        'canonical_title',
        'split',
        'year',
        'midi_filename',
        'audio_filename',
        'duration',
    ]

    # Process each original song.
    for midi_file, wav_file, title in original_pairs:
        split = song_splits[title]
        wav_path = os.path.join(directory, wav_file)
        duration = get_wav_duration(wav_path)

        # Add original song
        rows.append(
            [
                'Standard composer',
                title,
                split,
                2022,
                f"{folder_name}/{midi_file}",
                f"{folder_name}/{wav_file}",
                duration,
            ]
        )

        # Only add augmented versions if the song is in train and is NOT forced to test.
        if split == 'train':
            for f in all_files:
                if f.endswith(".mid") and is_augmented_version(f):
                    aug_base = get_original_song_name(f)
                    if aug_base == title:
                        aug_midi = f
                        aug_wav = os.path.splitext(f)[0] + '.wav'
                        # Find the matching WAV file (case-insensitive).
                        for w in all_files:
                            if w.lower() == aug_wav.lower():
                                aug_duration = get_wav_duration(os.path.join(directory, w))
                                rows.append(
                                    [
                                        'Standard composer',
                                        os.path.splitext(aug_midi)[0],
                                        'train',
                                        2022,
                                        f"{folder_name}/{aug_midi}",
                                        f"{folder_name}/{w}",
                                        aug_duration,
                                    ]
                                )
                                break

    # Write CSV
    if rows:
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        total_orig = sum(split_counts.values())
        print(f"\nSuccessfully wrote {len(rows)} entries to {csv_filename}")
        print("\nOriginal songs split distribution:")
        for split, count in split_counts.items():
            print(f"{split}: {count} songs ({count/total_orig*100:.1f}%)")
    else:
        print("No valid MIDI-WAV pairs found")
    return csv_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create CSV list of MIDI and WAV files from a directory'
    )
    parser.add_argument(
        'directory',
        type=str,
        help='Path to the directory containing MIDI and WAV files',
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.7, help='Ratio for training set'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.15, help='Ratio for test set'
    )
    parser.add_argument(
        '--validation-ratio', type=float, default=0.15, help='Ratio for validation set'
    )
    parser.add_argument(
        '--custom-test-songs', type=str, default="", help="Comma-separated list of song names to force as test (originals only)"
    )

    args = parser.parse_args()
    ratios = {
        "train": args.train_ratio,
        "test": args.test_ratio,
        "validation": args.validation_ratio,
    }

    if sum(ratios.values()) != 1.0:
        print("Error: Split ratios must sum to 1.0")
        exit(1)

    custom_test_songs = [s.strip() for s in args.custom_test_songs.split(",") if s.strip()]
    create_song_list(args.directory, ratios, custom_test_songs)
