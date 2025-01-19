import pretty_midi
import sys

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
                velocity = note.velocity  # Get the note velocity
                # Format with 6 decimal places
                f_out.write(f"{onset:.6f}\t{offset:.6f}\t{pitch}\t{velocity}\n")

if __name__ == '__main__':
    input_midi = 'Spretten_tender.mid'  # Replace with your input MIDI file name
    output_ann = 'Spretten_tendertest.ann'  # Replace with your desired output annotation file name
    midi_to_ann(input_midi, output_ann)