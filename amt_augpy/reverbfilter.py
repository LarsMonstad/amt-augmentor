import os
import argparse
import random
import string
import numpy as np
from pedalboard import Pedalboard, Reverb, LowpassFilter, HighpassFilter
from pedalboard.io import AudioFile

def random_word(length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_output_filename(input_filename, random_suffix):
    return os.path.splitext(input_filename)[0] + "_reverb_filters_" + random_suffix + os.path.splitext(input_filename)[1]

def apply_reverb_and_filters(input_audio_file, input_ann_file, output_path, room_size, low_cutoff, high_cutoff):
    # Generate random suffix for both files
    random_suffix = random_word(5)
    
    # Get the directory part of the output path
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    
    # Generate output audio file path based on the provided output path
    output_audio_file_path = output_path
    
    # Process audio
    with AudioFile(input_audio_file) as input_file:
        audio = input_file.read(input_file.frames)
        samplerate = input_file.samplerate

    reverb_effect = Reverb(room_size=room_size / 100.0, wet_level=room_size / 100.0)
    low_pass_filter = LowpassFilter(cutoff_frequency_hz=low_cutoff)
    high_pass_filter = HighpassFilter(cutoff_frequency_hz=high_cutoff)
    pedalboard = Pedalboard([reverb_effect, low_pass_filter, high_pass_filter])
    processed_audio = pedalboard(audio, samplerate)

    with AudioFile(output_audio_file_path, 'w', samplerate, audio.shape[0]) as output_file:
        output_file.write(processed_audio)

    # Generate the ann file path in the same directory as the output audio
    output_ann_file_path = os.path.splitext(output_audio_file_path)[0] + '.ann'
    
    # Copy the annotation file
    os.system(f'cp "{input_ann_file}" "{output_ann_file_path}"')
    
    return output_ann_file_path

def main():
    parser = argparse.ArgumentParser(description="Apply reverb and filters to audio files")
    parser.add_argument("input_audio_file", help="Path to the input audio file (FLAC or WAV)")
    parser.add_argument("input_ann_file", help="Path to the input annotation file (.ann)")
    parser.add_argument("output_directory", help="Path to the output directory")
    parser.add_argument("room_scale", type=float, help="Room scale and reverberance (0 to 100)")
    parser.add_argument("low_cutoff", type=int, default=20000, help="LowPassFilter cutoff frequency (20 to 20000)")
    parser.add_argument("high_cutoff", type=float, default=20, help="HighPassFilter cutoff frequency (20 to 20000)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_directory, exist_ok=True)
    
    # Generate output filename
    random_suffix = random_word(5)
    output_filename = generate_output_filename(os.path.basename(args.input_audio_file), random_suffix)
    output_path = os.path.join(args.output_directory, output_filename)
    
    apply_reverb_and_filters(args.input_audio_file, args.input_ann_file, output_path, 
                            args.room_scale, args.low_cutoff, args.high_cutoff)

if __name__ == "__main__":
    main()