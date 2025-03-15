#used to create csv files for the data, with default trainign splits
import os
import csv
from pathlib import Path
import argparse
import librosa 
from collections import defaultdict

def is_augmented_version(filename):
    """Check if the file is an augmented version based on common augmentation keywords"""
    aug_keywords = ['timestretch', 'pitchshift', 'reverb_filters', 
                   'gain_chorus', 'addpauses']
    return any(keyword in filename.lower() for keyword in aug_keywords)

def get_original_song_name(filename):
    """Extract original song name from augmented filename"""
    base_name = os.path.splitext(filename)[0]
    aug_markers = ['_timestretch_', '_pitchshift_', '_reverb_filters_', 
                  '_gain_chorus_', '_addpauses_']
    
    for marker in aug_markers:
        if marker in base_name:
            return base_name.split(marker)[0]
    return base_name

def get_split_status(songs, title, split_ratios):
    """Determine split status based on ratios and existing assignments"""
    if title in songs:
        return songs[title]
    
    current_counts = defaultdict(int)
    for split in songs.values():
        current_counts[split] += 1
    
    total_songs = len(songs) + 1
    targets = {
        'train': int(total_songs * split_ratios['train']),
        'test': int(total_songs * split_ratios['test']),
        'validation': int(total_songs * split_ratios['validation'])
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
        return round(duration,2)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0

def create_song_list(directory, split_ratios={'train': 0.7, 'test': 0.15, 'validation': 0.15}):
    directory = os.path.abspath(directory)
    folder_name = os.path.basename(directory)
    csv_filename = f"{folder_name}.csv"
    
    # Get all files
    all_files = os.listdir(directory)
    
    # First identify original songs and assign splits
    original_songs = {}
    song_splits = {}
    split_counts = defaultdict(int)

    # Get all original songs first (non-augmented)
    original_pairs = []
    for f in all_files:
        if f.endswith('.mid') and not is_augmented_version(f):
            title = os.path.splitext(f)[0]
            wav_file = title + '.wav'
            if wav_file in all_files:
                original_pairs.append((f, wav_file, title))
                split = get_split_status(song_splits, title, split_ratios)
                song_splits[title] = split
                split_counts[split] += 1

    # Prepare CSV data
    rows = []
    headers = ['canonical_composer', 'canonical_title', 'split', 'year', 
              'midi_filename', 'audio_filename', 'duration']

    # Process original files and their augmentations
    for midi_file, wav_file, title in original_pairs:
        # Get split for original song
        split = song_splits[title]
        
        # Add original song
        midi_path = os.path.join(directory, midi_file)
        wav_path = os.path.join(directory, wav_file)
        duration = get_wav_duration(wav_path)
        
        rows.append([
            'Standard composer',
            title,
            split,
            2022,
            f"{folder_name}/{midi_file}",
            f"{folder_name}/{wav_file}",   
            duration
        ])

        if split == 'train':
            for f in all_files:
                if f.endswith('.mid') and is_augmented_version(f):
                    aug_base = get_original_song_name(f)
                    if aug_base == title:
                        aug_midi = f
                        aug_wav = os.path.splitext(f)[0] + '.wav'
                        if aug_wav in all_files:
                            aug_duration = get_wav_duration(os.path.join(directory, aug_wav))
                            rows.append([
                                'Standard composer',
                                os.path.splitext(aug_midi)[0],
                                'train',
                                2022,
                                f"{folder_name}/{aug_midi}",
                                f"{folder_name}/{aug_wav}",
                                aug_duration
                            ])
                            
    # Write CSV
    if rows:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"\nSuccessfully wrote {len(rows)} entries to {csv_filename}")
        print("\nOriginal songs split distribution:")
        total_orig = sum(split_counts.values())
        for split, count in split_counts.items():
            print(f"{split}: {count} songs ({count/total_orig*100:.1f}%)")
    else:
        print("No valid MIDI-WAV pairs found")
    
    # Return the CSV file path so it can be used by other functions
    return csv_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CSV list of MIDI and WAV files from a directory')
    parser.add_argument('directory', type=str, help='Path to the directory containing MIDI and WAV files')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio for training set')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Ratio for test set')
    parser.add_argument('--validation-ratio', type=float, default=0.15, help='Ratio for validation set')
    
    args = parser.parse_args()
    
    ratios = {
        'train': args.train_ratio,
        'test': args.test_ratio,
        'validation': args.validation_ratio
    }
    
    if sum(ratios.values()) != 1.0:
        print("Error: Split ratios must sum to 1.0")
        exit(1)
    
    create_song_list(args.directory, ratios)
