import sys
import librosa 
import numpy as np
import soundfile as sf
import os
#based on the COMPOSER CLASSIFICATION WITH CROSS-MODAL TRANSFER LEARNING AND MUSICALLY-INFORMED AUGMENTATION article at ismir 
#removes colomuns/phrases of notes with a pause threshold to avoid cutting off. Uses the ANN file to detect which notes to remove and mutes the same part from the audio 
#from plot import create_comparison_plot

def insert_silence(audio_file, silence_ranges, output_file):
    audio, sr = librosa.load(audio_file, sr=None)
    segments = []

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
            next_start_sample = int(silence_ranges[i+1][0] * sr)
            segments.append(audio[end_sample:next_start_sample])
        else:
            segments.append(audio[end_sample:])

    # Concatenate all segments
    output_audio = np.concatenate(segments)
    sf.write(output_file, output_audio, sr)


def remove_silence_ranges(lines, silence_ranges):
    lines_to_keep = []
    for line in lines:
        onset = float(line.strip().split('\t')[0])
        offset = float(line.strip().split('\t')[1])

        # Only keep the line if it does not fall within any of the silence_ranges
        if not any(start <= onset and offset <= end for start, end in silence_ranges):
            lines_to_keep.append(line)

    return lines_to_keep

def calculate_time_distance(audio_filename,ann_filename, output_audio_file_path):
    with open(ann_filename, "r") as file:
        lines = file.readlines()
    pauses = []
    for i in range(len(lines) - 1):
        current_line = lines[i].strip().split('\t')
        next_line = lines[i + 1].strip().split('\t')
        
        offset_current = float(current_line[1])
        onset_next = float(next_line[0])
        
        distance = onset_next - offset_current
        threshold = 0.0033
        #print(threshold)
        if distance > threshold and all(onset_next > float(line.strip().split('\t')[1]) for line in lines[:i]):
            #print(f"Pause between line {i + 1} and line {i + 2}")
            #print(lines[i+1])
            pauses.append(lines[i+1])
    silence_ranges = []
    for i in range(1, len(pauses)):
        first_value_1 = float(pauses[i-1].strip().split('\t')[1])
        first_value_2 = float(pauses[i].strip().split('\t')[0])
        distance = first_value_2 - first_value_1
        #print(f"Distance between value {i} and value {i+1}: {distance:.3f}")
        if distance > 1 and distance < 5:
            #print("threshold met")
            #print("start line to remove from and including", pauses[i-1]) 
            #print("start line to remove to from and not including", pauses[i])
            if 1 < distance < 5:
                start_time = float(pauses[i-1].strip().split('\t')[1])
                end_time = float(pauses[i].strip().split('\t')[0])
                silence_ranges.append((start_time, end_time))
    lines = remove_silence_ranges(lines, silence_ranges)

    print("silence ranges",len(silence_ranges))
    if len(silence_ranges)>0:
        print("Creating modified song file")
        insert_silence(audio_filename, silence_ranges, output_audio_file_path)
    
    if len(silence_ranges)==0:
        print("No silence detected ")
        return "Silent"

    # Save the new lines to the output_ann_file_path
    output_ann_file_path = os.path.splitext(output_audio_file_path)[0] + os.path.splitext(ann_filename)[1]
    if len(silence_ranges)>0:
        with open(output_ann_file_path, "w") as file:
            file.writelines(lines)
        return output_ann_file_path
        #ann_to_midi(output_ann_file_path)
    #return output_ann_file_path
    #create_comparison_plot(audio_filename, output_audio_file_path,ann_filename, output_ann_file_path)
    
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} ann_file audio_file")
        sys.exit(1)

    
    audio_filename = sys.argv[1]
    ann_filename = sys.argv[2]
    output_audio_file_path = sys.argv[3]

    calculate_time_distance(ann_filename, audio_filename, output_audio_file_path)
