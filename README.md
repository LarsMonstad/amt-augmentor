# AMT-AugPy

## Python Data Augmentation Toolkit for Automatic Music Transcription (AMT)

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Librosa](https://img.shields.io/badge/librosa-0.9-green.svg)](https://librosa.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.23-blue.svg)](https://numpy.org)
[![SoundFile](https://img.shields.io/badge/soundfile-0.12%2B-red.svg)](https://python-soundfile.readthedocs.io/)

A comprehensive Python toolkit for augmenting Automatic Music Transcription (AMT) datasets through various audio transformations while maintaining synchronization between audio and MIDI files. The dataset follows the same format as [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro), which is commonly used for Automatic Music Transcription (AMT) tasks. 

The toolkit expects a folder containing paired audio and MIDI files with matching names. The audio file and MIDI file must be ground truth data, as this toolkit is only for augmenting existing datasets - a common technique in Machine Learning.

```
Folder/
├── song1.wav        # Audio file
├── song1.mid        # Ground truth annotated midi file
```

## Features

- **Time Stretching**: Modify the tempo of audio files while maintaining pitch
- **Pitch Shifting**: Transpose audio files up or down while preserving timing
- **Reverb & Filtering**: Apply room acoustics and frequency filtering effects
- **Gain & Chorus**: Add depth and richness through gain and chorus effects
- **Smart Pause Detection**: Identify and manipulate musical pauses based on note timing
- **Audio Standardization**: Convert various audio formats to 44.1kHz WAV
- **Parallel Processing**: Utilize multi-core processing for faster augmentation
- **Configurable Parameters**: Easily customize all augmentation parameters

## What's New in 1.0.3

- **Configuration System**: Use YAML configuration files to customize all parameters
- **Parallel Processing**: Process multiple effects concurrently for faster performance
- **Better Error Handling**: Improved error detection and reporting
- **Extended Format Support**: Added support for M4A and AIFF audio formats
- **Type Annotations**: Full Python type hints for better code quality
- **Expanded Documentation**: Improved documentation and examples

## Installation

You can install amt-augpy either via pip or by cloning the repository:

### Using pip

```bash
pip install amt-augpy
```

### From source

```bash
git clone https://github.com/LarsMonstad/amt-augpy.git
cd amt-augpy
pip install -e .
```

For development, install with additional development dependencies:

```bash
pip install -e ".[dev]"
```

### Dependencies
- librosa
- soundfile
- numpy
- pedalboard
- pretty_midi
- pyyaml
- tqdm

## Usage

### Basic Usage

```bash
amt-augpy /path/to/dataset/directory
```

This will process all compatible audio files in the directory and their corresponding MIDI files. The script automatically selects random parameters within predefined ranges for each augmentation type.

### Advanced Usage

```bash
# Use a custom configuration file
amt-augpy /path/to/dataset/directory --config my_config.yaml

# Specify an output directory
amt-augpy /path/to/dataset/directory --output-directory /path/to/output

# Generate a default configuration file
amt-augpy --generate-config my_config.yaml

# Disable specific effects
amt-augpy /path/to/dataset/directory --disable-effect timestretch --disable-effect chorus

# Parallel processing with 8 workers
amt-augpy /path/to/dataset/directory --num-workers 8

# Custom train/test/validation split
amt-augpy /path/to/dataset/directory --train-ratio 0.8 --test-ratio 0.1 --validation-ratio 0.1
```

### Help and options

```bash
amt-augpy --help
```

## Configuration

All augmentation parameters can be customized using a YAML configuration file. See `config.sample.yaml` for a complete example with documentation.

### Sample Configuration

```yaml
# Time stretching configuration
time_stretch:
  enabled: true
  variations: 3
  min_factor: 0.6
  max_factor: 1.6

# Pitch shifting configuration
pitch_shift:
  enabled: true
  variations: 3
  min_semitones: -5
  max_semitones: 5

# Processing configuration
processing:
  num_workers: 4
  output_dir: null
```

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

## Dataset Creation & Validation

The dataset follows the same format as [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro). Songs assigned to test or validation splits will have their augmented versions excluded to prevent data leakage.

### Creating the Dataset CSV

```bash
# Create dataset with default split ratios (70% train, 15% test, 15% validation)
amt-augpy /path/to/directory

# Create dataset with custom split ratios
amt-augpy /path/to/directory --train-ratio 0.8 --test-ratio 0.1 --validation-ratio 0.1
```

### Validating the Dataset Split

Dataset split validation is automatically performed after CSV creation to ensure:
- Augmented songs are not included in test/validation splits
- No cross-split contamination occurs
- Original and augmented songs are properly distributed

### CSV Format

The generated CSV follows the MAESTRO format with the following columns:
- canonical_composer
- canonical_title 
- split
- year
- midi_filename
- audio_filename
- duration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For development:
1. Install development dependencies: `pip install -e ".[dev]"`
2. Run tests: `pytest tests/`
3. Check typing: `mypy amt_augpy`
4. Format code: `black amt_augpy`

## License

MIT License - see LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```
@software{amt_augpy,
  author    = {Lars Monstad},
  title     = {amt-augpy: Audio augmentation toolkit for AMT datasets},
  version   = {1.0},
  year      = {2025}
}
```
