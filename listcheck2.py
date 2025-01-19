import os
import csv
from pathlib import Path
import argparse
import librosa 

def get_split_status(title):
   """Determine split status based on filename"""
   test_prefixes = ['Spretten', 'Godvaersdagen']
   validation_prefixes = ['Baggen', 'Peisestugu']
   
   for prefix in test_prefixes:
       if title.lower().startswith(prefix.lower()):
           return 'test'
           
   for prefix in validation_prefixes:
       if title.lower().startswith(prefix.lower()):
           return 'validation'
           
   return 'train'
def get_wav_duration(file_path):
    """
    Calculate the duration of a WAV file using librosa.
    
    Args:
        file_path (str): Path to the WAV file
    
    Returns:
        float: Duration of the audio file in seconds
    """
    # Load the audio file
    # We use sr=None to prevent resampling and get the original duration
    y, sr = librosa.load(file_path, sr=None)
    
    # Get the duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    return duration   

def create_song_list(directory):
   # Get absolute path and folder name
   #directory = os.path.abspath(directory)
   folder_name = os.path.basename(directory)
   csv_filename = f"{folder_name}.csv"
   
   # Get all midi files and wav files with full paths
   midi_files = set(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mid'))
   wav_files = set(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav'))
   
   rows = []
   errors = []
   headers = ['canonical_composer', 'canonical_title', 'split', 'year', 
              'midi_filename', 'audio_filename', 'duration']
   
   split_counts = {'train': 0, 'test': 0, 'validation': 0}
   
   for midi_path in midi_files:
       midi_file = os.path.basename(midi_path)
       title = os.path.splitext(midi_file)[0]
       wav_path = os.path.join(directory, title + '.wav')
       
       if wav_path not in wav_files:
           errors.append(f"Error: No matching WAV file for {midi_file}")
           continue
       
       duration = get_wav_duration(wav_path)
       split = get_split_status(title)
       split_counts[split] += 1
       
       row = [
           'Standard composer',
           title,
           split,
           2022,
           midi_path,
           wav_path,
           duration,
       ]
       rows.append(row)
       print(f"Processed: {title} -> {split}")
   
   for wav_path in wav_files:
       wav_file = os.path.basename(wav_path)
       title = os.path.splitext(wav_file)[0]
       midi_path = os.path.join(directory, title + '.mid')
       if midi_path not in midi_files:
           errors.append(f"Error: No matching MIDI file for {wav_file}")
   
   if rows:
       with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
           writer = csv.writer(f)
           writer.writerow(headers)
           writer.writerows(rows)
       print(f"\nSuccessfully wrote {len(rows)} songs to {csv_filename}")
       print("\nSplit distribution:")
       for split, count in split_counts.items():
           print(f"{split}: {count} songs")
   else:
       print("No valid MIDI-WAV pairs found")
   
   if errors:
       print("\nErrors found:")
       for error in errors:
           print(error)

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Create CSV list of MIDI and WAV files from a directory')
   parser.add_argument('directory', type=str, help='Path to the directory containing MIDI and WAV files')
   args = parser.parse_args()
   
   create_song_list(args.directory)