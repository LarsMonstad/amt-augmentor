"""
Module for detecting and manipulating musical pauses in audio files.

This module identifies pauses between notes in audio files using annotation data
and can insert silence at appropriate points without cutting off notes. It is based on
research from "Composer Classification with Cross-Modal Transfer Learning and
Musically-Informed Augmentation" (ISMIR).
"""

import sys
import os
import logging
from typing import List, Tuple, Optional

import librosa
import numpy as np
import soundfile as sf

# Configure logger
logger = logging.getLogger(__name__)


def insert_silence(
    audio_file: str, silence_ranges: List[Tuple[float, float]], output_file: str
) -> None:
    """
    Insert silence at specified ranges in an audio file.

    Args:
        audio_file: Path to the input audio file
        silence_ranges: List of (start, end) time ranges where silence should be inserted
        output_file: Path to save the modified audio file

    Raises:
        Exception: If there's an error loading or writing the audio file
    """
    try:
        audio, sr = librosa.load(audio_file, sr=None)
        segments = []

        if not silence_ranges:
            logger.warning("No silence ranges provided, audio will be unchanged")
            sf.write(output_file, audio, sr)
            return

        # Add the start of the audio to the segments list
        segments.append(audio[: int(silence_ranges[0][0] * sr)])

        # Iterate over the silence_ranges and replace them with silence
        for i, (start, end) in enumerate(silence_ranges):
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            silence_duration = end - start
            silence_samples = int(silence_duration * sr)
            silence = np.zeros(silence_samples)

            segments.append(silence)

            # Get the next segment until the next silence range or the end of the audio
            if i < len(silence_ranges) - 1:
                next_start_sample = int(silence_ranges[i + 1][0] * sr)
                segments.append(audio[end_sample:next_start_sample])
            else:
                segments.append(audio[end_sample:])

        # Concatenate all segments
        output_audio = np.concatenate(segments)
        sf.write(output_file, output_audio, sr)
        logger.debug(f"Modified audio saved to {output_file}")
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise


def remove_silence_ranges(
    lines: List[str], silence_ranges: List[Tuple[float, float]]
) -> List[str]:
    """
    Remove note entries that fall within the specified silence ranges.

    Args:
        lines: List of annotation file lines
        silence_ranges: List of (start, end) time ranges to remove

    Returns:
        List of annotation lines with entries in silence ranges removed
    """
    lines_to_keep = []
    for line in lines:
        try:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                logger.warning(f"Skipping malformed line: {line}")
                continue

            onset = float(parts[0])
            offset = float(parts[1])

            # Only keep the line if it does not fall within any of the silence_ranges
            if not any(
                start <= onset and offset <= end for start, end in silence_ranges
            ):
                lines_to_keep.append(line)
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing line '{line}': {e}")
            continue

    return lines_to_keep


def calculate_time_distance(
    audio_filename: str,
    ann_filename: str,
    output_audio_file_path: str,
    pause_threshold: float = 0.0033,
    min_pause_duration: float = 1.0,
    max_pause_duration: float = 5.0,
) -> Optional[str]:
    """
    Detect pauses between notes and insert silence at appropriate points.

    Args:
        audio_filename: Path to the input audio file
        ann_filename: Path to the annotation file
        output_audio_file_path: Path to save the output audio file
        pause_threshold: Minimum time between notes to consider as a pause (default: 0.0033s)
        min_pause_duration: Minimum pause duration to modify (default: 1.0s)
        max_pause_duration: Maximum pause duration to modify (default: 5.0s)

    Returns:
        Path to the output annotation file, or None if no pauses were found

    Raises:
        FileNotFoundError: If the annotation file doesn't exist
        Exception: For other processing errors
    """
    try:
        with open(ann_filename, "r") as file:
            lines = file.readlines()

        if not lines:
            logger.warning(f"Annotation file {ann_filename} is empty")
            return None

        # Detect pauses between consecutive notes
        pauses = []
        for i in range(len(lines) - 1):
            try:
                current_line = lines[i].strip().split("\t")
                next_line = lines[i + 1].strip().split("\t")

                offset_current = float(current_line[1])
                onset_next = float(next_line[0])

                distance = onset_next - offset_current

                # Check if this is a real pause (no overlapping notes)
                if distance > pause_threshold and all(
                    onset_next > float(line.strip().split("\t")[1])
                    for line in lines[:i]
                ):
                    pauses.append(lines[i + 1])
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing line {i}: {e}")
                continue

        # Identify silence ranges based on pause durations
        silence_ranges = []
        for i in range(1, len(pauses)):
            try:
                end_prev_note = float(pauses[i - 1].strip().split("\t")[1])
                start_next_note = float(pauses[i].strip().split("\t")[0])
                duration = start_next_note - end_prev_note

                if min_pause_duration < duration < max_pause_duration:
                    silence_ranges.append((end_prev_note, start_next_note))
            except (ValueError, IndexError) as e:
                logger.warning(f"Error calculating pause duration: {e}")
                continue

        logger.info(f"Detected {len(silence_ranges)} silence ranges to modify")

        if not silence_ranges:
            logger.info("No suitable pauses detected for modification")
            return None

        # Modify audio file with silence
        insert_silence(audio_filename, silence_ranges, output_audio_file_path)

        # Update annotation file
        filtered_lines = remove_silence_ranges(lines, silence_ranges)
        output_ann_file_path = (
            os.path.splitext(output_audio_file_path)[0]
            + os.path.splitext(ann_filename)[1]
        )

        with open(output_ann_file_path, "w") as file:
            file.writelines(filtered_lines)

        logger.info(f"Modified annotation saved to {output_ann_file_path}")
        return output_ann_file_path

    except FileNotFoundError:
        logger.error(f"Annotation file not found: {ann_filename}")
        raise
    except Exception as e:
        logger.error(f"Error in calculate_time_distance: {str(e)}")
        raise


if __name__ == "__main__":
    # Set up command-line logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) != 4:
        logger.error(f"Usage: {sys.argv[0]} audio_file ann_file output_audio_file")
        sys.exit(1)

    audio_filename = sys.argv[1]
    ann_filename = sys.argv[2]
    output_audio_file_path = sys.argv[3]

    try:
        result = calculate_time_distance(
            audio_filename, ann_filename, output_audio_file_path
        )
        if result:
            logger.info(
                f"Successfully processed file and saved to {output_audio_file_path}"
            )
            sys.exit(0)
        else:
            logger.info("No modifications made to the file")
            sys.exit(0)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)
