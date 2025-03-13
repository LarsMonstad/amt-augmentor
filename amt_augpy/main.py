"""
Main module for the AMT-AugPy package.

This module provides the main entry point for the package and coordinates
the various audio transformations to create an augmented dataset.
"""

import os
import sys
import argparse
import random
import string
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Optional, Set, Union

import pretty_midi
from tqdm import tqdm

from amt_augpy.time_stretch import apply_time_stretch
from amt_augpy.pitch_shift import apply_pitch_shift
from amt_augpy.reverbfilter import apply_reverb_and_filters
from amt_augpy.distortionchorus import apply_gain_and_chorus
from amt_augpy.add_pauses import calculate_time_distance
from amt_augpy.merge_audio import merge_audios
from amt_augpy.convertfiles import standardize_audio
from amt_augpy.create_maestro_csv import create_song_list
from amt_augpy.validate_split import validate_dataset_split
from amt_augpy.config import load_config, save_default_config, Config

# Configure logger
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def grab_audios(input_directory: str) -> List[str]:
    return [
        os.path.basename(f)
        for f in os.listdir(input_directory)
        if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".aiff"))
    ]


def midi_to_ann(input_midi: str, output_ann: str) -> str:
    """
    Convert a MIDI file to an annotation file.

    Args:
        input_midi: Path to the input MIDI file
        output_ann: Path to save the annotation file

    Returns:
        Path to the created annotation file

    Raises:
        FileNotFoundError: If the MIDI file doesn't exist
        Exception: For other processing errors
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_midi):
            raise FileNotFoundError(f"MIDI file not found: {input_midi}")

        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(input_midi)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_ann)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Get note onsets, offsets, pitch and velocity
        with open(output_ann, "w") as f_out:
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    onset = note.start
                    offset = note.end
                    pitch = note.pitch
                    velocity = note.velocity
                    f_out.write(f"{onset:.6f}\t{offset:.6f}\t{pitch}\t{velocity}\n")

        logger.debug(f"Annotation file created: {output_ann}")
        return output_ann

    except FileNotFoundError:
        logger.error(f"MIDI file not found: {input_midi}")
        raise
    except Exception as e:
        logger.error(f"Error converting MIDI to annotation: {e}")
        raise


def ann_to_midi(ann_file: str) -> str:
    """
    Convert an annotation file to a MIDI file.

    Args:
        ann_file: Path to the annotation file

    Returns:
        Path to the created MIDI file

    Raises:
        FileNotFoundError: If the annotation file doesn't exist
        ValueError: If the annotation file is malformed
    """
    midi_file = ann_file.replace(".ann", ".mid")

    try:
        with open(ann_file, "r") as f:
            lines = f.readlines()

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Default to piano

        for i, line in enumerate(lines):
            try:
                parts = line.strip().split("\t")
                if len(parts) != 4:
                    logger.warning(
                        f"Skipping malformed line {i+1} in {ann_file}: {line}"
                    )
                    continue

                onset_str, offset_str, pitch_str, velocity_str = parts

                # Convert strings to appropriate types
                onset = float(onset_str)
                offset = float(offset_str)
                pitch = int(pitch_str)
                velocity = int(velocity_str)

                # Create note with the correct velocity from the annotation
                note = pretty_midi.Note(
                    velocity=velocity, pitch=pitch, start=onset, end=offset
                )
                instrument.notes.append(note)

            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing line {i+1} in {ann_file}: {e}")
                continue

        midi.instruments.append(instrument)
        midi.write(midi_file)
        return midi_file

    except FileNotFoundError:
        logger.error(f"Annotation file not found: {ann_file}")
        raise
    except Exception as e:
        logger.error(f"Error converting annotation to MIDI: {e}")
        raise


def delete_file(file_path: str) -> bool:
    """
    Delete a file from the filesystem.

    Args:
        file_path: Path to the file to delete

    Returns:
        True if the file was deleted, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File to delete does not exist: {file_path}")
            return False

        os.remove(file_path)
        logger.debug(f"Deleted file: {file_path}")
        return True
    except OSError as e:
        logger.error(f"Error deleting file {file_path}: {e.strerror}")
        return False


def random_word(length: int) -> str:
    """
    Generate a random lowercase string of specified length.

    Args:
        length: Length of the random string

    Returns:
        A random string of lowercase letters
    """
    if length <= 0:
        return ""
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def generate_output_filename(
    base_name: str, effect_name: str, measure: float, random_suffix: str, extension: str
) -> str:
    """
    Generate an output filename with a specific format.

    Args:
        base_name: Base name of the file
        effect_name: Name of the effect applied
        measure: Effect parameter value
        random_suffix: Random suffix to ensure uniqueness
        extension: File extension with period (e.g., ".wav")

    Returns:
        A formatted output filename
    """
    return f"{base_name}_{effect_name}_{measure}_{random_suffix}{extension}"


# def process_directory(input_directory):
#     # Find audio and MIDI files in the directory, excluding ones that have been processed
#     effect_keywords = ['timestretch', 'pitchshift', 'reverb_filters', 'gain_chorus', 'addpauses']

#     audio_files = [f for f in os.listdir(input_directory)
#                    if f.endswith(('.flac', '.wav'))
#                    and not any(keyword in f for keyword in effect_keywords)]

#     midi_files = [f for f in os.listdir(input_directory)
#                   if f.endswith('.mid')
#                   and not any(keyword in f for keyword in effect_keywords)]

#     if not audio_files or not midi_files:
#         print("Error: Directory must contain at least one audio file (.flac or .wav) and one MIDI file (.mid)")
#         return

#     # Process each pair of audio and MIDI files
#     for audio_file in audio_files:
#         for midi_file in midi_files:
#             print(f"\nProcessing {audio_file} with {midi_file}")
#             process_files(
#                 os.path.join(input_directory, audio_file),
#                 os.path.join(input_directory, midi_file),
#                 input_directory
#             )


def process_effect(
    input_directory: str,
    effect_type: str,
    audio_base: str,
    audio_ext: str,
    standardized_audio: str,
    temp_ann_file: str,
    output_directory: str,
    config: Config,
) -> List[str]:
    """
    Process a specific effect type and return the list of created annotation files.

    Args:
        input_directory: Path to the input dataset
        effect_type: Type of effect to apply ('pauses', 'timestretch', 'pitchshift', 'reverb', 'chorus', 'merge')
        audio_base: Base name of the audio file
        audio_ext: Extension of the audio file
        standardized_audio: Path to the standardized audio file
        temp_ann_file: Path to the temporary annotation file
        output_directory: Directory to save output files
        config: Configuration object

    Returns:
        List of created annotation files
    """
    new_ann_files = []

    try:
        if effect_type == "pauses" and config.add_pause.enabled:
            # Apply pauses
            logger.info("Applying pause manipulation")
            random_suffix = random_word(5)
            output_filename = generate_output_filename(
                audio_base, "addpauses", 1, random_suffix, audio_ext
            )
            output_file_path = os.path.join(output_directory, output_filename)

            output_ann_file = calculate_time_distance(
                standardized_audio,
                temp_ann_file,
                output_file_path,
                pause_threshold=config.add_pause.pause_threshold,
                min_pause_duration=config.add_pause.min_pause_duration,
                max_pause_duration=config.add_pause.max_pause_duration,
            )

            if output_ann_file is not None:
                new_ann_files.append(output_ann_file)

        elif effect_type == "timestretch" and config.time_stretch.enabled:
            # Time stretch variations
            variations = config.time_stretch.variations
            min_factor = config.time_stretch.min_factor
            max_factor = config.time_stretch.max_factor

            generated_factors = set()
            for i in range(variations):
                stretch_factor = 1.0
                max_attempts = 10  # Prevent infinite loops
                attempts = 0

                while (
                    stretch_factor == 1.0 or stretch_factor in generated_factors
                ) and attempts < max_attempts:
                    stretch_factor = round(random.uniform(min_factor, max_factor), 1)
                    attempts += 1

                if attempts == max_attempts:
                    logger.warning(
                        f"Could not find unique stretch factor after {max_attempts} attempts"
                    )
                    if i > 0:  # Skip if we already have some variations
                        continue
                    stretch_factor = round(
                        random.uniform(min_factor, max_factor), 1
                    )  # Use anyway

                generated_factors.add(stretch_factor)
                random_suffix = random_word(5)
                output_filename = generate_output_filename(
                    audio_base, "timestretch", stretch_factor, random_suffix, audio_ext
                )
                output_file_path = os.path.join(output_directory, output_filename)

                logger.info(f"Applying time stretch: {stretch_factor}x")
                try:
                    output_ann_file = apply_time_stretch(
                        standardized_audio,
                        temp_ann_file,
                        output_file_path,
                        stretch_factor,
                    )
                    if output_ann_file:
                        new_ann_files.append(output_ann_file)
                except Exception as e:
                    logger.error(
                        f"Error applying time stretch ({stretch_factor}x): {e}"
                    )

        elif effect_type == "pitchshift" and config.pitch_shift.enabled:
            # Pitch shift variations
            variations = config.pitch_shift.variations
            min_semitones = config.pitch_shift.min_semitones
            max_semitones = config.pitch_shift.max_semitones

            generated_semitones = set()
            for i in range(variations):
                semitones = 0
                max_attempts = 10
                attempts = 0

                while (
                    semitones == 0 or semitones in generated_semitones
                ) and attempts < max_attempts:
                    semitones = random.randint(min_semitones, max_semitones)
                    attempts += 1

                if attempts == max_attempts:
                    logger.warning(
                        f"Could not find unique semitones value after {max_attempts} attempts"
                    )
                    if i > 0:
                        continue
                    semitones = random.randint(min_semitones, max_semitones)

                generated_semitones.add(semitones)
                random_suffix = random_word(5)
                output_filename = generate_output_filename(
                    audio_base, "pitchshift", semitones, random_suffix, audio_ext
                )
                output_file_path = os.path.join(output_directory, output_filename)

                logger.info(f"Applying pitch shift: {semitones} semitones")
                try:
                    output_ann_file = apply_pitch_shift(
                        standardized_audio, temp_ann_file, output_file_path, semitones
                    )
                    if output_ann_file:
                        new_ann_files.append(output_ann_file)
                except Exception as e:
                    logger.error(
                        f"Error applying pitch shift ({semitones} semitones): {e}"
                    )

        elif effect_type == "reverb" and config.reverb_filter.enabled:
            # Reverb and filter variations
            variations = config.reverb_filter.variations
            min_room_scale = config.reverb_filter.min_room_scale
            max_room_scale = config.reverb_filter.max_room_scale
            cutoff_pairs = config.reverb_filter.cutoff_pairs

            generated_room_scales = set()
            for i in range(variations):
                room_scale = 0
                max_attempts = 10
                attempts = 0

                while (
                    room_scale == 0 or room_scale in generated_room_scales
                ) and attempts < max_attempts:
                    room_scale = random.randint(min_room_scale, max_room_scale)
                    attempts += 1

                if attempts == max_attempts:
                    logger.warning(
                        f"Could not find unique room scale after {max_attempts} attempts"
                    )
                    if i > 0:
                        continue
                    room_scale = random.randint(min_room_scale, max_room_scale)

                generated_room_scales.add(room_scale)

                # Note: We're fixing the variable name confusion by always using consistent names
                low_cutoff, high_cutoff = random.choice(cutoff_pairs)

                random_suffix = random_word(5)
                output_filename = generate_output_filename(
                    audio_base, "reverb_filters", room_scale, random_suffix, audio_ext
                )
                output_file_path = os.path.join(output_directory, output_filename)

                logger.info(
                    f"Applying reverb (room_scale={room_scale}) and filters (LP={low_cutoff}Hz, HP={high_cutoff}Hz)"
                )
                try:
                    output_ann_file = apply_reverb_and_filters(
                        standardized_audio,
                        temp_ann_file,
                        output_file_path,
                        room_scale,
                        low_cutoff,
                        high_cutoff,
                    )
                    if output_ann_file:
                        new_ann_files.append(output_ann_file)
                except Exception as e:
                    logger.error(f"Error applying reverb and filters: {e}")

        elif effect_type == "chorus" and config.gain_chorus.enabled:
            # Gain and chorus variations
            variations = config.gain_chorus.variations
            min_gain = config.gain_chorus.min_gain
            max_gain = config.gain_chorus.max_gain
            min_depth = config.gain_chorus.min_depth
            max_depth = config.gain_chorus.max_depth
            chorus_rates = config.gain_chorus.rates

            generated_depths: Set[float] = set()
            generated_gains: Set[int] = set()

            for i in range(variations):
                depth: float = 0.0
                gain: int = 0
                max_attempts = 10
                depth_attempts = 0
                gain_attempts = 0

                while (
                    depth == 0.0 or depth in generated_depths
                ) and depth_attempts < max_attempts:
                    depth = round(random.uniform(min_depth, max_depth), 1)
                    depth_attempts += 1

                while (
                    gain == 0 or gain in generated_gains
                ) and gain_attempts < max_attempts:
                    gain = random.randint(min_gain, max_gain)
                    gain_attempts += 1

                if depth_attempts == max_attempts or gain_attempts == max_attempts:
                    logger.warning(
                        f"Could not find unique depth/gain after {max_attempts} attempts"
                    )
                    if i > 0:
                        continue
                    depth = round(random.uniform(min_depth, max_depth), 1)
                    gain = random.randint(min_gain, max_gain)

                generated_depths.add(depth)
                generated_gains.add(gain)

                chorus_rate = random.choice(chorus_rates)
                random_suffix = random_word(5)
                output_filename = generate_output_filename(
                    audio_base, "gain_chorus", gain, random_suffix, audio_ext
                )
                output_file_path = os.path.join(output_directory, output_filename)

                logger.info(
                    f"Applying gain ({gain}) and chorus (depth={depth}, rate={chorus_rate})"
                )
                try:
                    output_ann_file = apply_gain_and_chorus(
                        standardized_audio,
                        temp_ann_file,
                        output_file_path,
                        gain,
                        depth,
                        chorus_rate,
                    )
                    if output_ann_file:
                        new_ann_files.append(output_ann_file)
                except Exception as e:
                    logger.error(f"Error applying gain and chorus: {e}")

        elif effect_type == "merge" and config.merge_audio.enabled:
            target_audio_files: list[str] = [
                x
                for x in grab_audios(input_directory)
                if os.path.basename(standardized_audio) not in x
            ]
            if len(target_audio_files) >= config.merge_audio.merge_num:
                audios4merge: list[str] = list()
                while len(audios4merge) < config.merge_audio.merge_num:
                    audios4merge.append(
                        target_audio_files.pop(
                            random.randrange(len(target_audio_files))
                        )
                    )

                output_filename: str = (
                    "_".join(
                        [
                            x.rsplit(".", 1)[0]
                            for x in audios4merge
                            + [os.path.basename(standardized_audio)]
                        ]
                    )
                    + "_merged"
                )
                try:
                    ann_file = merge_audios(
                        audios4merge,
                        standardized_audio,
                        temp_ann_file,
                        input_directory,
                        output_directory,
                        output_filename,
                    )
                    new_ann_files.append(ann_file)
                    logger.info(
                        f"{audios4merge=} have been merged to {os.path.join(output_directory, output_filename)}"
                    )
                except Exception as e:
                    logger.error(f"Error merging {audios4merge=}, {e}")
            else:
                logger.error(
                    f"No merging is possible since {config.merge_audio.merge_num=} and {len(target_audio_files)=}"
                )

        return new_ann_files

    except Exception as e:
        logger.error(f"Error processing effect {effect_type}: {e}")
        return []


def gen_ann(
    input_directory: str,
    input_audio_file: str,
    input_midi_file: str,
    output_directory: str,
    config: Config,
) -> Tuple[str, str, str]:

    # Set output directory from config if specified
    if config.processing.output_dir:
        output_directory = config.processing.output_dir
        logger.info(f"Using output directory from config: {output_directory}")
        os.makedirs(output_directory, exist_ok=True)

    # First standardize the audio file
    logger.info(f"Standardizing audio: {input_audio_file}")
    standardized_audio, was_converted = standardize_audio(input_audio_file)
    if was_converted:
        logger.info(f"Converted audio format to: {standardized_audio}")

    # Get base name of the audio file without extension
    audio_base = os.path.splitext(os.path.basename(standardized_audio))[0]

    # Convert input MIDI to ANN
    temp_ann_file = os.path.join(output_directory, f"{audio_base}_temp.ann")
    logger.info(f"Converting MIDI to annotation: {input_midi_file}")
    midi_to_ann(input_midi_file, temp_ann_file)

    return (input_audio_file, standardized_audio, temp_ann_file)


def process_files(
    input_directory: str,
    input_audio_file: str,
    input_midi_file: str,
    output_directory: str,
    standardized_audio: str,
    temp_ann_file: str,
    config: Config,
) -> None:
    """
    Process a pair of audio and MIDI files applying various augmentations.

    Args:
        input_directory: Path to the input dataset
        input_audio_file: Path to the input audio file
        input_midi_file: Path to the input MIDI file
        output_directory: Directory to save output files
        config: Loaded config

    Raises:
        FileNotFoundError: If input files don't exist
        Exception: For other processing errors
    """

    try:
        # Set output directory from config if specified
        if config.processing.output_dir:
            output_directory = config.processing.output_dir
            logger.info(f"Using output directory from config: {output_directory}")
            os.makedirs(output_directory, exist_ok=True)

        # Get base name of the audio file without extension
        audio_base = os.path.splitext(os.path.basename(standardized_audio))[0]
        audio_ext = os.path.splitext(standardized_audio)[1]

        # List to store all created annotation files
        all_ann_files = []

        # Define effect types to process
        effect_types = [
            "pauses",
            "timestretch",
            "pitchshift",
            "reverb",
            "chorus",
            "merge",
        ]

        # Process effects in parallel if multiple workers are specified
        if config.processing.num_workers > 1:
            logger.info(
                f"Processing effects in parallel with {config.processing.num_workers} workers"
            )
            with ProcessPoolExecutor(
                max_workers=config.processing.num_workers
            ) as executor:
                futures = []

                for effect_type in effect_types:
                    future = executor.submit(
                        process_effect,
                        input_directory,
                        effect_type,
                        audio_base,
                        audio_ext,
                        standardized_audio,
                        temp_ann_file,
                        output_directory,
                        config,
                    )
                    futures.append(future)

                for future in tqdm(futures, desc="Processing effects"):
                    ann_files = future.result()
                    all_ann_files.extend(ann_files)
        else:
            # Process sequentially
            logger.info("Processing effects sequentially")
            for effect_type in tqdm(effect_types, desc="Processing effects"):
                ann_files = process_effect(
                    input_directory,
                    effect_type,
                    audio_base,
                    audio_ext,
                    standardized_audio,
                    temp_ann_file,
                    output_directory,
                    config,
                )
                all_ann_files.extend(ann_files)

        # Convert all ann files to midi
        logger.info(f"Converting {len(all_ann_files)} annotation files to MIDI")
        for ann_file in tqdm(all_ann_files, desc="Converting to MIDI"):
            try:
                ann_to_midi(ann_file)
                delete_file(ann_file)
            except Exception as e:
                logger.error(f"Error converting {ann_file} to MIDI: {e}")

        logger.info(
            f"Successfully processed files and created {len(all_ann_files)} augmented versions"
        )

    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise e


def check_matching_files(directory: str) -> Tuple[int, int, int]:
    """
    Check for matching WAV and MIDI files in the specified directory.

    Args:
        directory: Directory to check for matching files

    Returns:
        Tuple containing (matches, wav_missing, mid_missing) counts

    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Initialize counters
    matches = 0
    wav_missing = 0
    mid_missing = 0
    total_wav = 0
    total_mid = 0

    # Get list of all files
    try:
        files = os.listdir(directory)
        wav_files = [f for f in files if f.lower().endswith(".wav")]
        mid_files = [f for f in files if f.lower().endswith(".mid")]
    except Exception as e:
        logger.error(f"Error listing directory {directory}: {e}")
        raise

    # Check WAV files for matching MIDI files
    logger.info(f"Checking WAV files for matching MIDI files in {directory}...")
    for wav in wav_files:
        total_wav += 1
        base_name = os.path.splitext(wav)[0]
        midi_name = f"{base_name}.mid"
        if midi_name not in mid_files:
            logger.warning(f"No matching MIDI file for: {wav}")
            wav_missing += 1
        else:
            matches += 1

    # Check MIDI files for matching WAV files
    logger.info("Checking MIDI files for matching WAV files...")
    for mid in mid_files:
        total_mid += 1
        base_name = os.path.splitext(mid)[0]
        wav_name = f"{base_name}.wav"
        if wav_name not in wav_files:
            logger.warning(f"No matching WAV file for: {mid}")
            mid_missing += 1

    # Print summary
    logger.info("\nMatching Files Summary:")
    logger.info(f"Total WAV files: {total_wav}")
    logger.info(f"Total MIDI files: {total_mid}")
    logger.info(f"Complete matches found: {matches}")
    logger.info(f"WAV files without MIDI: {wav_missing}")
    logger.info(f"MIDI files without WAV: {mid_missing}")

    return matches, wav_missing, mid_missing


def main() -> None:
    """Main entry point for the AMT-AugPy package."""
    parser = argparse.ArgumentParser(
        description="Apply audio effects to audio and MIDI files"
    )

    # Input/output arguments
    parser.add_argument(
        "input_directory",
        nargs="?",
        help="Directory containing input audio and MIDI files",
    )
    parser.add_argument(
        "--output-directory",
        "-o",
        help="Directory to save output files (default: input directory)",
    )

    # Configuration arguments
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument(
        "--generate-config",
        "-g",
        help="Generate default configuration file at the specified path",
    )

    # Processing options
    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=0,
        help="Number of parallel workers (default: use config value)",
    )
    parser.add_argument(
        "--disable-effect",
        "-d",
        action="append",
        choices=["pauses", "timestretch", "pitchshift", "reverb", "chorus", "merge"],
        help="Disable specific effect (can be used multiple times)",
    )

    # CSV options
    parser.add_argument(
        "--skip-csv", action="store_true", help="Skip creating dataset CSV file"
    )
    parser.add_argument(
        "--train-ratio", type=float, help="Train split ratio (default: 0.7)"
    )
    parser.add_argument(
        "--test-ratio", type=float, help="Test split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--validation-ratio", type=float, help="Validation split ratio (default: 0.15)"
    )

    args = parser.parse_args()

    # Generate default config if requested
    if args.generate_config:
        try:
            save_default_config(args.generate_config)
            logger.info(
                f"Default configuration file generated at: {args.generate_config}"
            )
            if not args.input_directory:
                return  # Exit if only generating config
        except Exception as e:
            logger.error(f"Failed to generate configuration file: {e}")
            sys.exit(1)

    # Check that input directory exists
    if not os.path.isdir(args.input_directory):
        logger.error(f"Input directory not found: {args.input_directory}")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.output_directory:
        config.processing.output_dir = args.output_directory

    if args.num_workers > 0:
        config.processing.num_workers = args.num_workers

    # Disable specified effects
    if args.disable_effect:
        for effect in args.disable_effect:
            if effect == "pauses":
                config.add_pause.enabled = False
            elif effect == "timestretch":
                config.time_stretch.enabled = False
            elif effect == "pitchshift":
                config.pitch_shift.enabled = False
            elif effect == "reverb":
                config.reverb_filter.enabled = False
            elif effect == "chorus":
                config.gain_chorus.enabled = False
            elif effect == "merge":
                config.merge_audio.enabled = False

    # Setup output directory
    output_directory = config.processing.output_dir or args.input_directory
    os.makedirs(output_directory, exist_ok=True)

    # Get all audio files with matching MIDI files
    audio_files = grab_audios(args.input_directory)

    # Filter out files that have already been processed based on naming pattern
    effect_keywords = [
        "timestretch",
        "pitchshift",
        "reverb_filters",
        "gain_chorus",
        "addpauses",
        "merge",
    ]
    audio_files = [
        f for f in audio_files if not any(keyword in f for keyword in effect_keywords)
    ]

    if not audio_files:
        logger.error("No unprocessed audio files found in the input directory")
        sys.exit(1)

    # Count files with matching MIDI
    matched_count = 0
    for audio in audio_files:
        matching_midi = os.path.splitext(audio)[0] + ".mid"
        if os.path.exists(os.path.join(args.input_directory, matching_midi)):
            matched_count += 1

    if matched_count == 0:
        logger.error("No matching audio/MIDI pairs found in the input directory")
        sys.exit(1)

    logger.info(f"Found {matched_count} audio files with matching MIDI files")

    # Process each audio/MIDI pair
    processed_count = 0

    audio_files_described: List[Tuple[str, str, str]] = list()
    # Generate the ANN's beforehand to allow merging of audio files
    for audio in tqdm(audio_files, desc="Generating MIDI annotations"):
        matching_midi = os.path.splitext(audio)[0] + ".mid"
        midi_path = os.path.join(args.input_directory, matching_midi)
        logger.info(f"Generating ANN for {audio} with {matching_midi}")

        audio_files_described.append(
            gen_ann(
                args.input_directory,
                os.path.join(args.input_directory, audio),
                midi_path,
                output_directory,
                config,
            )
        )

    logger.info(f"{audio_files_described}")
    for audio, standardized_audio, temp_ann_file in audio_files_described:
        matching_midi = os.path.splitext(audio)[0] + ".mid"
        midi_path = matching_midi

        print(midi_path)
        if os.path.exists(midi_path):
            logger.info(f"Processing {audio} with {matching_midi}")
            try:
                process_files(
                    args.input_directory,
                    audio,
                    midi_path,
                    output_directory,
                    standardized_audio,
                    temp_ann_file,
                    config,
                )
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {audio}: {e}")

    logger.info(
        f"Successfully processed {processed_count} out of {matched_count} audio/MIDI pairs"
    )

    # Delete the previously generated audio files
    for _, _, temp_ann_file in tqdm(
        audio_files_described, desc="Deleting generated annotations"
    ):
        # Delete temporary input ann file
        delete_file(temp_ann_file)

    # After all processing is done, check for matching files
    logger.info("Checking final results...")
    check_matching_files(output_directory)

    # Create and validate dataset CSV if not skipped
    if not args.skip_csv:
        logger.info("Creating dataset CSV file...")
        csv_kwargs = {}
        if args.train_ratio:
            csv_kwargs["train_ratio"] = args.train_ratio
        if args.test_ratio:
            csv_kwargs["test_ratio"] = args.test_ratio
        if args.validation_ratio:
            csv_kwargs["validation_ratio"] = args.validation_ratio

        csv_path = create_song_list(output_directory, **csv_kwargs)

        logger.info("Validating dataset split...")
        validate_dataset_split(csv_path)

    logger.info("Processing complete!")

    return


if __name__ == "__main__":
    main()
