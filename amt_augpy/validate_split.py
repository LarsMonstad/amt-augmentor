import csv
import os
import argparse
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

def validate_dataset_split(csv_file):
    # Store songs by split
    split_songs = defaultdict(list)
    original_songs = defaultdict(set)  # Keep track of original songs in each split
    augmented_songs = defaultdict(list)  # Track augmented versions
    issues_found = False

    print(f"\nValidating dataset split in {csv_file}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            midi_file = os.path.basename(row['midi_filename'])
            split = row['split']
            is_augmented = is_augmented_version(midi_file)
            
            if is_augmented:
                original_name = get_original_song_name(midi_file)
                augmented_songs[split].append((midi_file, original_name))
            else:
                original_songs[split].add(midi_file.replace('.mid', ''))
            
            split_songs[split].append(midi_file)

    print("\n1. Basic Split Statistics:")
    print("-" * 50)
    for split, songs in split_songs.items():
        print(f"{split}: {len(songs)} total entries")
        aug_count = len([s for s in songs if is_augmented_version(s)])
        orig_count = len([s for s in songs if not is_augmented_version(s)])
        print(f"  - Original songs: {orig_count}")
        print(f"  - Augmented versions: {aug_count}")

    print("\n2. Checking for Augmented Songs in Test/Validation:")
    print("-" * 50)
    for split in ['test', 'validation']:
        aug_in_split = [s for s in split_songs[split] if is_augmented_version(s)]
        if aug_in_split:
            issues_found = True
            print(f"ERROR: Found {len(aug_in_split)} augmented songs in {split} split:")
            for song in aug_in_split:
                print(f"  - {song}")
        else:
            print(f"✓ No augmented songs found in {split} split")

    print("\n3. Checking for Cross-Split Contamination:")
    print("-" * 50)
    # Check each augmented song to ensure its original isn't in test/validation
    for split, aug_list in augmented_songs.items():
        for aug_file, original_name in aug_list:
            for check_split in ['test', 'validation']:
                if original_name in original_songs[check_split]:
                    issues_found = True
                    print(f"ERROR: Song contamination detected!")
                    print(f"  - Original song '{original_name}' is in {check_split}")
                    print(f"  - But augmented version '{aug_file}' is in {split}")

    if not issues_found:
        print("✓ No cross-split contamination found")
        print("✓ All augmented versions are properly segregated")
        print("\nValidation PASSED: Dataset split appears to be clean!")
    else:
        print("\nValidation FAILED: Issues were found in the dataset split!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate dataset split in CSV file')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    
    args = parser.parse_args()
    validate_dataset_split(args.csv_file)
