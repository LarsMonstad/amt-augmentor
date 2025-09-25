#!/usr/bin/env python3
"""
Dataset Modifier - Tool for modifying existing dataset CSV files
Allows listing, moving, and removing songs from train/test/validation splits
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
import shutil
from tabulate import tabulate


def load_dataset(csv_path: str) -> Tuple[List[Dict], List[str]]:
    """Load dataset from CSV file"""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found!")
        sys.exit(1)

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            rows.append(row)

    return rows, headers


def save_dataset(csv_path: str, rows: List[Dict], headers: List[str], backup: bool = False):
    """Save dataset to CSV file with optional backup"""
    if backup and os.path.exists(csv_path):
        backup_path = csv_path.replace('.csv', '_backup.csv')
        shutil.copy2(csv_path, backup_path)
        print(f"Created backup: {backup_path}")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset saved: {csv_path}")


def get_original_songs(rows: List[Dict]) -> Dict[str, Dict]:
    """Extract original (non-augmented) songs with their info"""
    original_songs = {}
    for row in rows:
        # Check if it's an original song (not augmented)
        if '_augmented_' not in row['midi_filename']:
            title = row['canonical_title']
            original_songs[title] = {
                'split': row['split'],
                'midi_filename': row['midi_filename'],
                'audio_filename': row['audio_filename']
            }
    return original_songs


def list_songs(args):
    """List songs in specified split or all splits"""
    rows, _ = load_dataset(args.csv_path)
    original_songs = get_original_songs(rows)

    # Group songs by split
    splits = defaultdict(list)
    for title, info in original_songs.items():
        splits[info['split']].append(title)

    if args.split:
        # Show specific split
        if args.split not in splits:
            print(f"No songs in '{args.split}' split")
            return

        print(f"\n{args.split.upper()} Split ({len(splits[args.split])} songs):")
        print("-" * 50)
        for i, song in enumerate(sorted(splits[args.split]), 1):
            print(f"{i:3d}. {song}")
    else:
        # Show all splits with statistics
        total = len(original_songs)
        print(f"\nDataset Overview (Total: {total} original songs)")
        print("=" * 60)

        # Create summary table
        summary_data = []
        for split in ['train', 'validation', 'test']:
            if split in splits:
                count = len(splits[split])
                percentage = (count / total * 100) if total > 0 else 0
                summary_data.append([split.upper(), count, f"{percentage:.1f}%"])

        print(tabulate(summary_data, headers=['Split', 'Count', 'Percentage'], tablefmt='grid'))

        # List songs in each split if verbose
        if args.verbose:
            for split in ['train', 'validation', 'test']:
                if split in splits:
                    print(f"\n{split.upper()} Split:")
                    print("-" * 50)
                    for i, song in enumerate(sorted(splits[split]), 1):
                        print(f"{i:3d}. {song}")


def move_songs(args):
    """Move songs to a different split"""
    rows, headers = load_dataset(args.csv_path)

    # Parse song patterns
    patterns = [p.strip() for p in args.songs.split(',')]
    target_split = args.to_split.lower()

    if target_split not in ['train', 'test', 'validation']:
        print(f"Error: Invalid split '{target_split}'. Must be train, test, or validation.")
        sys.exit(1)

    # Track changes
    moved_songs = []
    moved_augmented = []

    for row in rows:
        title = row['canonical_title']
        is_augmented = '_augmented_' in row['midi_filename']

        # Check if this song matches any pattern
        matches = any(pattern.lower() in title.lower() for pattern in patterns)

        if matches:
            if not is_augmented:
                # Original song
                if row['split'] != target_split:
                    moved_songs.append((title, row['split'], target_split))

                    # Move original to target split
                    if target_split == 'train':
                        row['split'] = 'train'
                    else:
                        # For test/validation, only move the original
                        row['split'] = target_split
            else:
                # Augmented version
                base_name = title.split('_augmented_')[0]
                base_matches = any(pattern.lower() in base_name.lower() for pattern in patterns)

                if base_matches:
                    if target_split == 'train':
                        # Move augmented to train
                        if row['split'] != 'train':
                            moved_augmented.append((title, row['split'], 'train'))
                            row['split'] = 'train'
                    else:
                        # Remove augmented from test/validation (they shouldn't be there)
                        if row['split'] == target_split:
                            # This shouldn't happen but if it does, move to train
                            moved_augmented.append((title, row['split'], 'train'))
                            row['split'] = 'train'

    # Report changes
    if moved_songs or moved_augmented:
        print(f"\nMoved {len(moved_songs)} original songs:")
        for song, from_split, to_split in moved_songs:
            print(f"  {song}: {from_split} -> {to_split}")

        if moved_augmented:
            print(f"\nAdjusted {len(moved_augmented)} augmented versions")

        # Save changes
        save_dataset(args.csv_path, rows, headers, backup=not args.no_backup)

        # Show new distribution
        print("\nNew distribution:")
        args.split = None
        args.verbose = False
        list_songs(args)
    else:
        print("No songs matched the specified patterns.")


def remove_songs(args):
    """Remove songs from the dataset"""
    rows, headers = load_dataset(args.csv_path)

    # Parse song patterns
    patterns = [p.strip() for p in args.songs.split(',')]

    # Track removals
    removed_songs = set()

    # First pass: identify songs to remove
    for row in rows:
        title = row['canonical_title']
        base_title = title.split('_augmented_')[0] if '_augmented_' in title else title

        # Check if this song matches any pattern
        if any(pattern.lower() in base_title.lower() for pattern in patterns):
            removed_songs.add(base_title)

    if not removed_songs:
        print("No songs matched the specified patterns.")
        return

    # Second pass: remove all versions (original + augmented)
    filtered_rows = []
    removed_count = 0

    for row in rows:
        title = row['canonical_title']
        base_title = title.split('_augmented_')[0] if '_augmented_' in title else title

        if base_title not in removed_songs:
            filtered_rows.append(row)
        else:
            removed_count += 1

    # Report changes
    print(f"\nRemoved {len(removed_songs)} songs (and their augmented versions):")
    for song in sorted(removed_songs):
        print(f"  - {song}")
    print(f"Total rows removed: {removed_count}")

    # Save changes
    save_dataset(args.csv_path, filtered_rows, headers, backup=not args.no_backup)

    # Show new distribution
    print("\nNew distribution:")
    args.split = None
    args.verbose = False
    args.csv_path = args.csv_path  # Ensure the path is set
    list_songs(args)


def add_to_split(args):
    """Force specific songs to a split (useful for custom test/validation sets)"""
    rows, headers = load_dataset(args.csv_path)

    # Parse song patterns
    patterns = [p.strip() for p in args.songs.split(',')]
    target_split = args.split.lower()

    if target_split not in ['train', 'test', 'validation']:
        print(f"Error: Invalid split '{target_split}'. Must be train, test, or validation.")
        sys.exit(1)

    # Track changes
    forced_songs = []

    for row in rows:
        title = row['canonical_title']
        is_augmented = '_augmented_' in row['midi_filename']

        # For originals only
        if not is_augmented:
            # Check exact match or substring match
            matches = any(
                pattern.lower() == title.lower() or
                pattern.lower() in title.lower()
                for pattern in patterns
            )

            if matches and row['split'] != target_split:
                forced_songs.append((title, row['split'], target_split))
                row['split'] = target_split
        else:
            # Handle augmented versions
            base_name = title.split('_augmented_')[0]
            matches = any(
                pattern.lower() == base_name.lower() or
                pattern.lower() in base_name.lower()
                for pattern in patterns
            )

            if matches:
                # Augmented versions should only be in train
                if target_split != 'train' and row['split'] != 'train':
                    row['split'] = 'train'  # Move augmented to train
                elif target_split == 'train':
                    row['split'] = 'train'

    # Report changes
    if forced_songs:
        print(f"\nForced {len(forced_songs)} songs to {target_split}:")
        for song, from_split, to_split in forced_songs:
            print(f"  {song}: {from_split} -> {to_split}")

        # Save changes
        save_dataset(args.csv_path, rows, headers, backup=not args.no_backup)

        # Show new distribution
        print("\nNew distribution:")
        args.split = None
        args.verbose = False
        list_songs(args)
    else:
        print(f"No changes made. Songs might already be in {target_split} or no matches found.")


def main():
    parser = argparse.ArgumentParser(
        description='Modify dataset CSV files - manage train/test/validation splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all songs and their splits
  python -m amt_augmentor.dataset_modifier list dataset.csv

  # List only test songs
  python -m amt_augmentor.dataset_modifier list dataset.csv --split test

  # Move songs containing "Mozart" or "Chopin" to test split
  python -m amt_augmentor.dataset_modifier move dataset.csv "Mozart,Chopin" --to-split test

  # Force specific songs to validation
  python -m amt_augmentor.dataset_modifier add-to-split dataset.csv "Beethoven_Op27,Liszt" --split validation

  # Remove songs from dataset
  python -m amt_augmentor.dataset_modifier remove dataset.csv "BadRecording1,BadRecording2"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List songs in dataset')
    list_parser.add_argument('csv_path', help='Path to dataset CSV file')
    list_parser.add_argument('--split', choices=['train', 'test', 'validation'],
                            help='Show only songs in specified split')
    list_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Show detailed listing')
    list_parser.set_defaults(func=list_songs)

    # Move command
    move_parser = subparsers.add_parser('move', help='Move songs to different split')
    move_parser.add_argument('csv_path', help='Path to dataset CSV file')
    move_parser.add_argument('songs', help='Comma-separated song names/patterns to move')
    move_parser.add_argument('--to-split', required=True,
                            choices=['train', 'test', 'validation'],
                            help='Target split to move songs to')
    move_parser.add_argument('--no-backup', action='store_true',
                            help='Do not create backup of original CSV')
    move_parser.set_defaults(func=move_songs)

    # Add to split command (force assignment)
    add_parser = subparsers.add_parser('add-to-split',
                                       help='Force songs to specific split (like custom test songs)')
    add_parser.add_argument('csv_path', help='Path to dataset CSV file')
    add_parser.add_argument('songs', help='Comma-separated song names/patterns')
    add_parser.add_argument('--split', required=True,
                           choices=['train', 'test', 'validation'],
                           help='Split to assign songs to')
    add_parser.add_argument('--no-backup', action='store_true',
                           help='Do not create backup of original CSV')
    add_parser.set_defaults(func=add_to_split)

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove songs from dataset')
    remove_parser.add_argument('csv_path', help='Path to dataset CSV file')
    remove_parser.add_argument('songs', help='Comma-separated song names/patterns to remove')
    remove_parser.add_argument('--no-backup', action='store_true',
                              help='Do not create backup of original CSV')
    remove_parser.set_defaults(func=remove_songs)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == '__main__':
    main()