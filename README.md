# amt-augpy 1.0

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Librosa](https://img.shields.io/badge/librosa-0.10.1-green.svg)](https://librosa.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24.0-blue.svg)](https://numpy.org)
[![SoundFile](https://img.shields.io/badge/soundfile-0.12.1-red.svg)](https://python-soundfile.readthedocs.io/)

A comprehensive Python toolkit for augmenting Automatic Music Transcription (AMT) datasets through various audio transformations while maintaining synchronization between audio and MIDI files.

## Features

- **Time Stretching**: Modify the tempo of audio files while maintaining pitch
- **Pitch Shifting**: Transpose audio files up or down while preserving timing
- **Reverb & Filtering**: Apply room acoustics and frequency filtering effects
- **Gain & Chorus**: Add depth and richness through gain and chorus effects
- **Smart Pause Detection**: Identify and manipulate musical pauses based on note timing
- **Audio Standardization**: Convert various audio formats to 44.1kHz WAV

## Installation

You can install amt-augpy either via pip or by cloning the repository:

### Using pip

    pip install amt-augpy

### From source

    git clone https://github.com/LarsMonstad/amt-augpy1.0.git
    cd amt-augpy1.0
    pip install -r requirements.txt

### Dependencies
- librosa
- soundfile
- numpy
- pedalboard
- pretty_midi
- tqdm

## Usage

### Basic Usage

    python main.py /path/to/dataset/directory

This will process all compatible audio files in the directory and their corresponding MIDI files. The script automatically selects random parameters within predefined ranges (specified in main.py) for each augmentation type.

### Parameter Ranges (defined in main.py)

- Time stretch: 0.6 to 1.6x
- Pitch shift: -5 to +5 semitones
- Reverb room size: 10 to 100
- Gain: 2 to 11 dB
- Chorus depth: 0.1 to 0.6
- Filter cutoff pairs: Various predefined frequency ranges

Each input file will generate multiple augmented versions using randomly selected parameters within these ranges.

## File Format Support

### Audio
- Input: WAV, FLAC, MP3, M4A, AIFF 
- Output: WAV (44.1kHz)

### Annotations
- MIDI (.mid)

## Output Structure

For each input file pair (audio + MIDI), the toolkit generates multiple augmented versions with the following naming convention:

    original_name_effect_parameter_randomsuffix.extension

Example:

    piano_timestretch_1.2_abc123.wav
    piano_timestretch_1.2_abc123.mid

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

    @software{amt_augpy,
      title = {amt-augpy: Audio augmentation toolkit for AMT datasets},
      version = {1.0},
      year = {2024}
    }
