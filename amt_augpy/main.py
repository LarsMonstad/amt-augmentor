import os
import argparse
import random
import string
import pretty_midi
from amt_augpy.time_stretch import apply_time_stretch
from amt_augpy.pitch_shift import apply_pitch_shift
from amt_augpy.reverbfilter import apply_reverb_and_filters
from amt_augpy.distortionchorus import apply_gain_and_chorus
from amt_augpy.add_pauses import calculate_time_distance
from amt_augpy.convertfiles import standardize_audio
from tqdm import tqdm
from amt_augpy.create_maestro_csv import create_song_list
from amt_augpy.validate_split import validate_dataset_split

def midi_to_ann(input_midi, output_ann):
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(input_midi)
    
    # Get note onsets, offsets, pitch and velocity
    with open(output_ann, 'w') as f_out:
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                onset = note.start
                offset = note.end
                pitch = note.pitch
                velocity = note.velocity
                f_out.write(f"{onset:.6f}\t{offset:.6f}\t{pitch}\t{velocity}\n")

def ann_to_midi(ann_file):
    midi_file = ann_file.replace(".ann", ".mid")
    with open(ann_file, 'r') as f:
        lines = f.readlines()
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for line in lines:
        onset, offset, pitch, channel = line.strip().split('\t')
        onset = float(onset)
        offset = float(offset)
        pitch = int(pitch)
        channel = int(channel)
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=onset, end=offset)
        instrument.notes.append(note)
    midi.instruments.append(instrument)
    midi.write(midi_file)

def delete_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"Error: {file_path} : {e.strerror}")

def random_word(length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_output_filename(base_name, effect_name, measure, random_suffix, extension):
    return f"{base_name}_{effect_name}_{measure}_{random_suffix}{extension}"

def process_directory(input_directory):
    # Find audio and MIDI files in the directory, excluding ones that have been processed
    effect_keywords = ['timestretch', 'pitchshift', 'reverb_filters', 'gain_chorus', 'addpauses']
    
    audio_files = [f for f in os.listdir(input_directory) 
                   if f.endswith(('.flac', '.wav')) 
                   and not any(keyword in f for keyword in effect_keywords)]
    
    midi_files = [f for f in os.listdir(input_directory) 
                  if f.endswith('.mid')
                  and not any(keyword in f for keyword in effect_keywords)]
    
    if not audio_files or not midi_files:
        print("Error: Directory must contain at least one audio file (.flac or .wav) and one MIDI file (.mid)")
        return
    
    # Process each pair of audio and MIDI files
    for audio_file in audio_files:
        for midi_file in midi_files:
            print(f"\nProcessing {audio_file} with {midi_file}")
            process_files(
                os.path.join(input_directory, audio_file),
                os.path.join(input_directory, midi_file),
                input_directory
            )

def process_files(input_audio_file, input_midi_file, output_directory):
    # First standardize the audio file
    standardized_audio, was_converted = standardize_audio(input_audio_file)

    # Get base name of the audio file without extension
    audio_base = os.path.splitext(os.path.basename(standardized_audio))[0]
    audio_ext = os.path.splitext(standardized_audio)[1]
    
    # Convert input MIDI to ANN
    temp_ann_file = os.path.join(output_directory, f"{audio_base}_temp.ann")
    midi_to_ann(input_midi_file, temp_ann_file)

    # Create a list of ann files 
    new_ann_files = []
    
    # Apply pauses
    pausesfactor = 1
    random_suffix = random_word(5)
    output_filename = generate_output_filename(audio_base, "addpauses", pausesfactor, random_suffix, audio_ext)
    output_file_path = os.path.join(output_directory, output_filename)
    output_ann_filep = calculate_time_distance(standardized_audio, temp_ann_file, output_file_path)
    if output_ann_filep != "Silent":
        new_ann_files.append(output_ann_filep)

    # Time stretch variations
    generated_stretch_factors = set()
    for _ in tqdm(range(3), desc="Time Stretch"):
        stretch_factor = 1
        while stretch_factor == 1 or stretch_factor in generated_stretch_factors:
            stretch_factor = round(random.uniform(0.6, 1.6), 1)
        generated_stretch_factors.add(stretch_factor)
        random_suffix = random_word(5)
        output_filename = generate_output_filename(audio_base, "timestretch", stretch_factor, random_suffix, audio_ext)
        output_file_path = os.path.join(output_directory, output_filename)
        output_ann_file = apply_time_stretch(standardized_audio, temp_ann_file, output_file_path, stretch_factor)
        new_ann_files.append(output_ann_file)

    # Pitch shift variations
    generated_semitones = set()
    for _ in tqdm(range(3), desc="Pitch Shift"):
        semitones = 0
        while semitones == 0 or semitones in generated_semitones:
            semitones = random.choice([n for n in range(-5, 5) if n != 0])
        generated_semitones.add(semitones)
        random_suffix = random_word(5)
        output_filename = generate_output_filename(audio_base, "pitchshift", semitones, random_suffix, audio_ext)
        output_file_path = os.path.join(output_directory, output_filename)
        output_ann_file = apply_pitch_shift(standardized_audio, temp_ann_file, output_file_path, semitones)
        new_ann_files.append(output_ann_file)

    # Reverb and filter variations
    generated_roomscales = set()
    cutoff_pairs = [(20, 20000), (300, 20000), (3000, 20000), (20, 16300), (20, 17500), (20, 18000)]
    for _ in tqdm(range(3), desc="Reverb and filter"):
        room_scale = 0
        while room_scale == 0 or room_scale in generated_roomscales:
            room_scale = random.choice([n for n in range(10, 100)])
        generated_roomscales.add(room_scale)
        high_cutoff, low_cutoff = random.choice(cutoff_pairs)
        random_suffix = random_word(5)
        output_filename = generate_output_filename(audio_base, "reverb_filters", room_scale, random_suffix, audio_ext)
        output_file_path = os.path.join(output_directory, output_filename)
        output_ann_file = apply_reverb_and_filters(standardized_audio, temp_ann_file, output_file_path, room_scale, low_cutoff, high_cutoff)
        new_ann_files.append(output_ann_file)

    # Gain and chorus variations
    chorusrates = [1, 1, 1]
    generated_depths = set()
    generated_gains = set()
    for _ in tqdm(range(3), desc="gain and chorusrate"):
        depth = 0
        gain = 0 
        while depth == 0 or depth in generated_depths:
            depth = round(random.uniform(0.1, 0.6), 1)
        generated_depths.add(depth)
        while gain == 0 or gain in generated_gains:
            gain = random.choice([n for n in range(2, 11)])
        generated_gains.add(gain)
        chorusrate = random.choice(chorusrates)
        random_suffix = random_word(5)
        output_filename = generate_output_filename(audio_base, "gain_chorus", gain, random_suffix, audio_ext)
        output_file_path = os.path.join(output_directory, output_filename)
        output_ann_file = apply_gain_and_chorus(standardized_audio, temp_ann_file, output_file_path, gain, depth, chorusrate)
        new_ann_files.append(output_ann_file)

    # Delete temporary input ann file
    delete_file(temp_ann_file)

    # Convert all ann files to midi
    for ann_file in tqdm(new_ann_files, desc="Converting ann files to midi"):
        ann_to_midi(ann_file)
        delete_file(ann_file)

def check_matching_files(directory):
    """
    Check for matching WAV and MIDI files in the specified directory.
    """
    # Initialize counters
    matches = 0
    wav_missing = 0
    mid_missing = 0
    total_wav = 0
    total_mid = 0

    # Get list of all files
    files = os.listdir(directory)
    wav_files = [f for f in files if f.endswith('.wav')]
    mid_files = [f for f in files if f.endswith('.mid')]

    # Check WAV files for matching MIDI files
    print(f"\nChecking WAV files for matching MIDI files in {directory}...")
    for wav in wav_files:
        total_wav += 1
        base_name = os.path.splitext(wav)[0]
        midi_name = f"{base_name}.mid"
        if midi_name not in mid_files:
            print(f"No matching MIDI file for: {wav}")
            wav_missing += 1
        else:
            matches += 1

    # Check MIDI files for matching WAV files
    print(f"\nChecking MIDI files for matching WAV files in {directory}...")
    for mid in mid_files:
        total_mid += 1
        base_name = os.path.splitext(mid)[0]
        wav_name = f"{base_name}.wav"
        if wav_name not in wav_files:
            print(f"No matching WAV file for: {mid}")
            mid_missing += 1

    # Print summary
    print("\nSummary:")
    print(f"Total WAV files: {total_wav}")
    print(f"Total MIDI files: {total_mid}")
    print(f"Complete matches found: {matches}")
    print(f"WAV files without MIDI: {wav_missing}")
    print(f"MIDI files without WAV: {mid_missing}")

def main():
    parser = argparse.ArgumentParser(description="Apply audio effects to audio and MIDI files")
    parser.add_argument("input_directory", help="Directory containing input audio (FLAC, mp3 or WAV) and MIDI files")
    
    args = parser.parse_args()
    
    # Simply process all wav/flac files with their corresponding mid files
    audio_files = [f for f in os.listdir(args.input_directory) if f.endswith(('.wav', '.flac', '.mp3'))]
    
    for audio in audio_files:
        matching_midi = os.path.splitext(audio)[0] + '.mid'
        if os.path.exists(os.path.join(args.input_directory, matching_midi)):
            print(f"Processing {audio}")
            process_files(
                os.path.join(args.input_directory, audio),
                os.path.join(args.input_directory, matching_midi),
                args.input_directory
            )
    
    # After all processing is done, check for matching files
    print("\nChecking final results...")
    check_matching_files(args.input_directory)

    print("\nCreating dataset CSV file...")
    create_song_list(args.input_directory)
    
    print("\nValidating dataset split...")
    csv_filename = f"{os.path.basename(args.input_directory)}.csv"
    validate_dataset_split(csv_filename)

if __name__ == "__main__":
    main()
