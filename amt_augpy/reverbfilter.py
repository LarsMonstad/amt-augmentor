"""
Module for applying reverb and filter effects to audio files.

This module applies room acoustics and frequency filtering effects to audio files
while preserving the synchronization with annotation files.
"""

import os
import sys
import argparse
import random
import string
import shutil
import logging
from typing import Tuple, Optional

import numpy as np
from pedalboard import Pedalboard, Reverb, LowpassFilter, HighpassFilter
from pedalboard.io import AudioFile

# Configure logger
logger = logging.getLogger(__name__)


def validate_parameters(room_size: float, low_cutoff: int, high_cutoff: int) -> None:
    """
    Validate reverb and filter parameters.
    
    Args:
        room_size: Room size parameter (0-100)
        low_cutoff: Low pass filter cutoff frequency (Hz)
        high_cutoff: High pass filter cutoff frequency (Hz)
        
    Raises:
        ValueError: If parameters are outside valid ranges
    """
    if not 0 <= room_size <= 100:
        raise ValueError(f"Room size must be between 0 and 100, got {room_size}")
    
    if not 20 <= low_cutoff <= 20000:
        raise ValueError(f"Low cutoff must be between 20 and 20000 Hz, got {low_cutoff}")
    
    if not 20 <= high_cutoff <= 20000:
        raise ValueError(f"High cutoff must be between 20 and 20000 Hz, got {high_cutoff}")


def apply_reverb_and_filters(
    input_audio_file: str, 
    input_ann_file: str, 
    output_path: str, 
    room_size: float, 
    low_cutoff: int, 
    high_cutoff: int
) -> Optional[str]:
    """
    Apply reverb and filter effects to an audio file and copy its annotation file.
    
    Args:
        input_audio_file: Path to the input audio file
        input_ann_file: Path to the input annotation file
        output_path: Path to save the output audio file
        room_size: Room size parameter (0-100)
        low_cutoff: Low pass filter cutoff frequency (Hz)
        high_cutoff: High pass filter cutoff frequency (Hz)
        
    Returns:
        Path to the output annotation file
        
    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If input files don't exist
        Exception: For other processing errors
    """
    try:
        # Validate parameters
        validate_parameters(room_size, low_cutoff, high_cutoff)
        
        # Check if input files exist
        if not os.path.exists(input_audio_file):
            raise FileNotFoundError(f"Input audio file not found: {input_audio_file}")
        
        if not os.path.exists(input_ann_file):
            raise FileNotFoundError(f"Input annotation file not found: {input_ann_file}")
        
        # Create output directory if it doesn't exist
        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        
        # Process audio
        logger.info(f"Applying reverb (room_size={room_size}) and filters (LP={low_cutoff}Hz, HP={high_cutoff}Hz)")
        
        try:
            with AudioFile(input_audio_file) as input_file:
                audio = input_file.read(input_file.frames)
                samplerate = input_file.samplerate

            # Create effects chain
            reverb_effect = Reverb(room_size=room_size / 100.0, wet_level=room_size / 100.0)
            low_pass_filter = LowpassFilter(cutoff_frequency_hz=low_cutoff)
            high_pass_filter = HighpassFilter(cutoff_frequency_hz=high_cutoff)
            pedalboard = Pedalboard([reverb_effect, low_pass_filter, high_pass_filter])
            
            # Apply effects
            processed_audio = pedalboard(audio, samplerate)

            # Write processed audio
            with AudioFile(output_path, 'w', samplerate, audio.shape[0]) as output_file:
                output_file.write(processed_audio)
                
            logger.debug(f"Processed audio saved to {output_path}")
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise Exception(f"Failed to process audio: {str(e)}")

        # Generate the ann file path in the same directory as the output audio
        output_ann_file_path = os.path.splitext(output_path)[0] + '.ann'
        
        # Copy the annotation file (using safe file operations)
        try:
            shutil.copy2(input_ann_file, output_ann_file_path)
            logger.debug(f"Annotation file copied to {output_ann_file_path}")
        except Exception as e:
            logger.error(f"Error copying annotation file: {str(e)}")
            raise Exception(f"Failed to copy annotation file: {str(e)}")
        
        return output_ann_file_path
    
    except Exception as e:
        logger.error(f"Error in apply_reverb_and_filters: {str(e)}")
        raise


def main() -> None:
    """Command-line interface for the module."""
    # Set up command-line logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="Apply reverb and filters to audio files")
    parser.add_argument("input_audio_file", help="Path to the input audio file (FLAC or WAV)")
    parser.add_argument("input_ann_file", help="Path to the input annotation file (.ann)")
    parser.add_argument("output_directory", help="Path to the output directory")
    parser.add_argument("room_size", type=float, help="Room size and reverberance (0 to 100)")
    parser.add_argument("low_cutoff", type=int, default=20000, help="LowPassFilter cutoff frequency (20 to 20000)")
    parser.add_argument("high_cutoff", type=int, default=20, help="HighPassFilter cutoff frequency (20 to 20000)")
    
    args = parser.parse_args()
    
    try:
        os.makedirs(args.output_directory, exist_ok=True)
        
        # Generate random suffix
        random_suffix = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
        
        # Generate output filename
        output_filename = f"{os.path.splitext(os.path.basename(args.input_audio_file))[0]}_reverb_filters_{args.room_size}_{random_suffix}{os.path.splitext(args.input_audio_file)[1]}"
        output_path = os.path.join(args.output_directory, output_filename)
        
        # Apply effects
        output_ann_file = apply_reverb_and_filters(
            args.input_audio_file, 
            args.input_ann_file, 
            output_path, 
            args.room_size, 
            args.low_cutoff, 
            args.high_cutoff
        )
        
        logger.info(f"Successfully processed file: {output_path}")
        logger.info(f"Annotation file: {output_ann_file}")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()