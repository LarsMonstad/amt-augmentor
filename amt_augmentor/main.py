"""
Main module for the AMT-Augmentor package.

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

from amt_augmentor.time_stretch import apply_time_stretch
from amt_augmentor.pitch_shift import apply_pitch_shift
from amt_augmentor.reverbfilter import apply_reverb_and_filters
from amt_augmentor.distortionchorus import apply_gain_and_chorus
from amt_augmentor.add_noise import apply_noise
from amt_augmentor.add_pauses import calculate_time_distance
from amt_augmentor.merge_audio import merge_audios
from amt_augmentor.convertfiles import standardize_audio
from amt_augmentor.create_maestro_csv import create_song_list
from amt_augmentor.validate_split import validate_dataset_split
from amt_augmentor.config import load_config, save_default_config, Config

import numpy as np

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

        logger.debug("Annotation file created: %s", output_ann)
        return output_ann

    except FileNotFoundError:
        logger.error("MIDI file not found: %s", input_midi)
        raise
    except Exception as e:
        logger.error("Error converting MIDI to annotation: %s", e)
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
                logger.warning("Error parsing line %s in %s: %s", i+1, ann_file, e)
                continue

        midi.instruments.append(instrument)
        midi.write(midi_file)
        return midi_file

    except FileNotFoundError:
        logger.error("Annotation file not found: %s", ann_file)
        raise
    except Exception as e:
        logger.error("Error converting annotation to MIDI: %s", e)
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
            logger.warning("File to delete does not exist: %s", file_path)
            return False

        os.remove(file_path)
        logger.debug("Deleted file: %s", file_path)
        return True
    except OSError as e:
        logger.error("Error deleting file %s: %s", file_path, e.strerror)
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
        A formatted output filename with '_augmented_' prefix for identification
    """
    if not random_suffix:
        return f"{base_name}_augmented_{effect_name}_{measure}{extension}"

    return f"{base_name}_augmented_{effect_name}_{measure}_{random_suffix}{extension}"

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
            random_suffix: str = random_word(5) if config.enable_random_suffix else ''
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
            variations: int = config.time_stretch.variations
            min_factor: int = config.time_stretch.min_factor
            max_factor: int = config.time_stretch.max_factor

            generated_factors: set[int] = set()
            if config.time_stretch.randomized:
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
                    
            else:
                generated_factors = set(list(np.linspace(min_factor,
                                                         max_factor,
                                                         variations+1,
                                                         dtype=float)))
                try:
                    generated_factors.remove(1.0)
                except KeyError:
                    pass
                    
                while len(generated_factors) > variations:
                    generated_factors.pop()
                    
            for stretch_factor in generated_factors:
                random_suffix: str = random_word(5) if config.enable_random_suffix else ''
                output_filename = generate_output_filename(
                    audio_base, "timestretch", stretch_factor, random_suffix, audio_ext
                )
                output_file_path = os.path.join(output_directory, output_filename)

                logger.info("Applying time stretch: %sx", stretch_factor)
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
            variations: int = config.pitch_shift.variations
            min_semitones: int = config.pitch_shift.min_semitones
            max_semitones: int = config.pitch_shift.max_semitones

            generated_semitones: set[int] = set()
            if config.pitch_shift.randomized:
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
            else:
                generated_semitones = set(list(np.linspace(min_semitones,
                                                            max_semitones,
                                                            variations+1,
                                                            dtype=int)))
                try:
                    generated_semitones.remove(0)
                except KeyError:
                    pass
                    
                if len(generated_semitones) < variations:
                    logger.warning(
                        f"Impossible to have {variations} unique semitones values"
                    )
                    
                while len(generated_semitones) > variations:
                    generated_semitones.pop()
                        
                for semitones in generated_semitones:
                    random_suffix: str = random_word(5) if config.enable_random_suffix else ''
                    output_filename = generate_output_filename(
                        audio_base, "pitchshift", semitones, random_suffix, audio_ext
                    )
                    output_file_path = os.path.join(output_directory, output_filename)

                    logger.info("Applying pitch shift: %s semitones", semitones)
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

                random_suffix: str = random_word(5) if config.enable_random_suffix else ''
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
                    logger.error("Error applying reverb and filters: %s", e)

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
                    logger.error("Error applying gain and chorus: %s", e)

        elif effect_type == "merge" and config.merge_audio.enabled:
            target_audio_files: List[str] = [
                x
                for x in grab_audios(input_directory)
                if os.path.basename(standardized_audio) not in x
            ]
            if len(target_audio_files) >= config.merge_audio.merge_num:
                audios4merge: List[str] = list()
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
                    logger.error("Error merging %s, %s", audios4merge, e)
            else:
                logger.error(
                    f"No merging is possible since {config.merge_audio.merge_num=} and {len(target_audio_files)=}"
                )
                
        elif effect_type == "noise" and config.add_noise.enabled:
            # Noise intensity variations

            variations: int = config.add_noise.variations
            min_intensity: int = config.add_noise.min_intensity
            max_intensity: int = config.add_noise.max_intensity

            generated_intensities: set[int] = set()
            if config.add_noise.randomized:
                for i in range(variations):
                    intensity = 1.0
                    max_attempts = 10  # Prevent infinite loops
                    attempts = 0

                    while (
                        intensity == 1.0 or intensity in generated_intensities
                    ) and attempts < max_attempts:
                        intensity = round(random.uniform(min_intensity, max_intensity), 1)
                        attempts += 1

                    if attempts == max_attempts:
                        logger.warning(
                            f"Could not find unique noise intensity factor after {max_attempts} attempts"
                        )
                        if i > 0:  # Skip if we already have some variations
                            continue
                        intensity = round(
                            random.uniform(min_intensity, max_intensity), 1
                        )  # Use anyway

                    generated_intensities.add(intensity)

            else:
                generated_intensities = set(list(np.logspace(min_intensity,
                                                         max_intensity,
                                                         variations+1,
                                                         dtype=float)))
                try:
                    generated_intensities.remove(1.0)
                except KeyError:
                    pass

                while len(generated_intensities) > variations:
                    generated_intensities.pop()

            for intensity in generated_intensities:
                random_suffix: str = random_word(5) if config.enable_random_suffix else ''
                output_filename = generate_output_filename(
                    audio_base, "noise", intensity, random_suffix, audio_ext
                )
                output_file_path = os.path.join(output_directory, output_filename)

                logger.info("Applying noise with intensity %s", intensity)
                try:
                    output_ann_file = apply_noise(
                        standardized_audio,
                        temp_ann_file,
                        output_file_path,
                        intensity
                    )
                    if output_ann_file:
                        new_ann_files.append(output_ann_file)
                except Exception as e:
                    logger.error("Error applying gain and chorus: %s", e)

        return new_ann_files

    except Exception as e:
        logger.error("Error processing effect %s: %s", effect_type, e)
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
        logger.info("Using output directory from config: %s", output_directory)
        os.makedirs(output_directory, exist_ok=True)

    # First standardize the audio file
    logger.info("Standardizing audio: %s", input_audio_file)
    standardized_audio, was_converted = standardize_audio(input_audio_file)
    if was_converted:
        logger.info("Converted audio format to: %s", standardized_audio)

    # Get base name of the audio file without extension
    audio_base = os.path.splitext(os.path.basename(standardized_audio))[0]

    # Convert input MIDI to ANN
    temp_ann_file = os.path.join(output_directory, f"{audio_base}_temp.ann")
    logger.info("Converting MIDI to annotation: %s", input_midi_file)
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
            logger.info("Using output directory from config: %s", output_directory)
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
            "noise"
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
        logger.info("Converting %s annotation files to MIDI", len(all_ann_files))
        for ann_file in tqdm(all_ann_files, desc="Converting to MIDI"):
            try:
                ann_to_midi(ann_file)
                delete_file(ann_file)
            except Exception as e:
                logger.error("Error converting %s to MIDI: %s", ann_file, e)

        logger.info(
            f"Successfully processed files and created {len(all_ann_files)} augmented versions"
        )

    except FileNotFoundError as e:
        logger.error("Input file not found: %s", e)
        raise e
    except Exception as e:
        logger.error("Error processing files: %s", e)
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
        logger.error("Error listing directory %s: %s", directory, e)
        raise

    # Check WAV files for matching MIDI files
    logger.info("Checking WAV files for matching MIDI files in %s...", directory)
    for wav in wav_files:
        total_wav += 1
        base_name = os.path.splitext(wav)[0]
        midi_name = f"{base_name}.mid"
        if midi_name not in mid_files:
            logger.warning("No matching MIDI file for: %s", wav)
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
            logger.warning("No matching WAV file for: %s", mid)
            mid_missing += 1

    # Print summary
    logger.info("\nMatching Files Summary:")
    logger.info("Total WAV files: %s", total_wav)
    logger.info("Total MIDI files: %s", total_mid)
    logger.info("Complete matches found: %s", matches)
    logger.info("WAV files without MIDI: %s", wav_missing)
    logger.info("MIDI files without WAV: %s", mid_missing)

    return matches, wav_missing, mid_missing


def main() -> None:
    """Main entry point for the AMT-Augmentor package."""
    parser = argparse.ArgumentParser(
        description="Apply audio effects to audio and MIDI files",
        prog="amt-augmentor"
    )

    # Version
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {__import__('amt_augmentor').__version__}",
        help="Show program version and exit"
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
        choices=["pauses", "timestretch", "pitchshift", "reverb", "chorus", "merge", "noise"],
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
    parser.add_argument(
        "--custom-test-songs", type=str, default="",
        help="Comma-separated list of song names to force as test (originals only, no augmented versions)"
    )

    # Additional utility options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing files"
    )
    parser.add_argument(
        "--list-effects",
        action="store_true",
        help="List all available audio effects and exit"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--check-pairs",
        action="store_true",
        help="Check for matching audio/MIDI pairs and exit without processing"
    )

    args = parser.parse_args()

    # Handle list-effects command
    if args.list_effects:
        print("Available audio effects:")
        print("  - timestretch    : Stretch or compress audio in time")
        print("  - pitchshift     : Shift pitch up or down by semitones")
        print("  - reverb         : Add reverb with room simulation")
        print("  - chorus         : Add chorus effect with adjustable depth")
        print("  - pauses         : Manipulate pauses in audio")
        print("  - merge          : Merge multiple audio files")
        print("  - noise          : Add noise to audio files")
        print("\nUse --disable-effect <effect_name> to disable specific effects")
        return

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

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
            logger.error("Failed to generate configuration file: %s", e)
            sys.exit(1)

    # Check that input directory exists
    if not os.path.isdir(args.input_directory):
        logger.error("Input directory not found: %s", args.input_directory)
        sys.exit(1)

    # Handle check-pairs command
    if args.check_pairs:
        logger.info("Checking for matching audio/MIDI pairs in %s", args.input_directory)
        matches, wav_missing, mid_missing = check_matching_files(args.input_directory)
        print(f"\nTotal matching pairs: {matches}")
        print(f"Audio files without MIDI: {wav_missing}")
        print(f"MIDI files without audio: {mid_missing}")
        return

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
            elif effect == "noise":
                config.add_noise.enabled = False

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
        "noise"
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

    logger.info("Found %s audio files with matching MIDI files", matched_count)

    # Handle dry-run mode
    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"Would process {matched_count} audio/MIDI pairs")
        print(f"Output directory: {output_directory}")
        print("\nEnabled effects:")
        if config.time_stretch.enabled:
            print(f"  - Time stretch: {config.time_stretch.variations} variations")
        if config.pitch_shift.enabled:
            print(f"  - Pitch shift: {config.pitch_shift.variations} variations")
        if config.reverb_filter.enabled:
            print(f"  - Reverb/Filter: {config.reverb_filter.variations} variations")
        if config.gain_chorus.enabled:
            print(f"  - Gain/Chorus: {config.gain_chorus.variations} variations")
        if config.add_pause.enabled:
            print(f"  - Pause manipulation: enabled")
        if config.merge_audio.enabled:
            print(f"  - Audio merge: merge {config.merge_audio.merge_num} files")
        if config.add_noise.enabled:
            print(f"  - Add noise: {config.add_noise.variations} variations")

        print(f"\nProcessing with {config.processing.num_workers} workers")
        print("\nFiles to be processed:")
        for audio in audio_files[:10]:  # Show first 10 files
            print(f"  - {audio}")
        if len(audio_files) > 10:
            print(f"  ... and {len(audio_files) - 10} more files")
        print("\n=== DRY RUN COMPLETE (no files were modified) ===")
        return

    # Process each audio/MIDI pair
    processed_count = 0

    audio_files_described: List[Tuple[str, str, str]] = list()
    # Generate the ANN's beforehand to allow merging of audio files
    for audio in tqdm(audio_files, desc="Generating MIDI annotations"):
        matching_midi = os.path.splitext(audio)[0] + ".mid"
        midi_path = os.path.join(args.input_directory, matching_midi)
        logger.info("Generating ANN for %s with %s", audio, matching_midi)

        audio_files_described.append(
            gen_ann(
                args.input_directory,
                os.path.join(args.input_directory, audio),
                midi_path,
                output_directory,
                config,
            )
        )

    logger.info("%s", audio_files_described)
    for audio, standardized_audio, temp_ann_file in audio_files_described:
        matching_midi = os.path.splitext(audio)[0] + ".mid"
        midi_path = matching_midi

        print(midi_path)
        if os.path.exists(midi_path):
            logger.info("Processing %s with %s", audio, matching_midi)
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
                logger.error("Failed to process %s: %s", audio, e)

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
        # Build split_ratios dictionary
        split_ratios = {
            'train': args.train_ratio if args.train_ratio else 0.7,
            'test': args.test_ratio if args.test_ratio else 0.15,
            'validation': args.validation_ratio if args.validation_ratio else 0.15
        }

        # Parse custom test songs if provided
        custom_test_songs = [s.strip() for s in args.custom_test_songs.split(",") if s.strip()]

        csv_path = create_song_list(output_directory, split_ratios=split_ratios, custom_test_songs=custom_test_songs)

        logger.info("Validating dataset split...")
        validate_dataset_split(csv_path)

    logger.info("Processing complete!")

    return


if __name__ == "__main__":
    main()
