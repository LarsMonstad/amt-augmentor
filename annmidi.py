import pretty_midi
import argparse

def ann_to_midi(ann_file):
    # Set the output MIDI file path based on the input ANN file path
    midi_file = ann_file.replace(".ann", ".mid")

    # Read the input ANN file
    with open(ann_file, 'r') as f:
        lines = f.readlines()

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()

    # Create an instrument instance for a piano (instrument number 0)
    instrument = pretty_midi.Instrument(program=0)

    # Iterate through the lines in the ANN file
    for line in lines:
        onset, offset, pitch, channel = line.strip().split('\t')
        onset = float(onset)
        offset = float(offset)
        pitch = int(pitch)
        channel = int(channel)

        # Create a new Note instance for this note
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=onset, end=offset)

        # Add the note to the instrument
        instrument.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(instrument)

    # Write the PrettyMIDI object to the output MIDI file
    midi.write(midi_file)

def main():
    parser = argparse.ArgumentParser(description="Convert an ANN file to a MIDI file")
    parser.add_argument("ann_file", help="Path to the input ANN file")
    #parser.add_argument("output_midi_file", help="Path to the output MIDI file")

    args = parser.parse_args()

    ann_to_midi(args.ann_file)

if __name__ == "__main__":
    main()
