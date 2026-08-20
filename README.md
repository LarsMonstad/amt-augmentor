<div align="center">
  <img src="https://raw.githubusercontent.com/LarsMonstad/amt-augmentor/refs/heads/main/images/BotsForMusic_Logo_Black_2.png" alt="Bots for Music Logo" width="300">

  # AMT-Augmentor

  ## Python Data Augmentation Toolkit for Automatic Music Transcription (AMT)

  **Developed by [Bots for Music](https://botsformusic.com), maintained by Lars Monstad**

  [![PyPI version](https://badge.fury.io/py/amt-augmentor.svg)](https://badge.fury.io/py/amt-augmentor)
  [![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![CI](https://github.com/LarsMonstad/amt-augmentor/actions/workflows/ci.yml/badge.svg)](https://github.com/LarsMonstad/amt-augmentor/actions/workflows/ci.yml)
  [![Downloads](https://pepy.tech/badge/amt-augmentor)](https://pepy.tech/project/amt-augmentor)
</div>

> **📦 [View on PyPI](https://pypi.org/project/amt-augmentor/)** 
>
> **Note:** Formerly known as `amt-augpy`. Starting with v1.0.9, the package is **`amt-augmentor`**.

A Python toolkit for augmenting Automatic Music Transcription (AMT) datasets
through conventional audio transformations while maintaining synchronization
between audio and MIDI files. Its optional CSV helpers follow the same metadata
concepts as [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro).

The toolkit expects a folder containing paired audio and MIDI files with
matching names. The versioned paired Python APIs accept MIDI containing
multiple annotated instruments. The legacy batch CLI converts MIDI through a
note-only intermediate representation, so it does not retain instrument or
non-note event structure. The pair must already be aligned ground truth: this
toolkit augments existing datasets; it does not create annotations from
unlabelled audio.

```
dataset/
├── song1.wav        # Audio file
├── song1.mid        # Ground truth annotated midi file
```

## Features

### Audio Transformations
- **Time Stretching**: Tempo modification while maintaining pitch
- **Pitch Shifting**: Transposition while preserving timing
- **Reverb & Filtering**: Room acoustics and frequency filtering effects
- **Gain & Chorus**: Depth and richness enhancement
- **Noise Augmentation**: Controlled noise addition for robustness training
- **Fractional Detuning**: Audio detuning by less than 50 cents with unchanged
  symbolic note labels
- **Archival Noise (opt-in)**: Configurable seeded 1/f-like noise and harmonic
  mains hum for datasets whose target recordings have those characteristics
- **Legacy Audio Merging**: Disabled by default because sources must be assigned
  to splits before any merge can be shown to be leakage-safe

### Processing & Dataset Handling
- **Audio Standardization**: Channel-preserving, non-destructive conversion to
  44.1 kHz WAV working copies
- **Parallel Processing**: Multi-core processing for faster augmentation
- **Configuration System**: YAML-based parameter customization
- **Dataset Validation**: Automatic validation of train/test/validation splits
- **Dataset Modification**: Built-in tools to modify existing dataset splits
- **MAESTRO Compatibility**: Dataset format compatible with MAESTRO v3.0.0

## Why AMT-Augmentor?

Built for AMT, not just audio. Unlike general audio augmenters, the versioned
paired APIs keep audio and MIDI aligned through transform-consistent label
updates: pitch shift transposes pitched notes, while time-changing methods map
note boundaries, control changes, pitch bends, and supported global timed
events. The toolkit also provides MAESTRO-style CSV construction and split
validation.

## Deterministic paired transforms

The versioned Python API provides deterministic paired implementations of
gain/chorus, target-SNR noise, reverb and filtering, integral and fractional
pitch changes, global time stretch, and local time variation. Each transform
validates the audio--MIDI pair, records its parameters and synchronization
checks, and publishes source-bound provenance. Useful strengths and parameter
ranges depend on the instrument, recording conditions, annotation policy, and
downstream model; no research recipe is enabled implicitly.

Several methods were refined while studying historical Hardanger fiddle
recordings, but they are exposed as configurable AMT primitives rather than as
a fiddle-specific pipeline:

- `reverb_only_v1` separates room simulation from high-pass and low-pass
  filtering so their effects can be evaluated independently.
- `materialize_pitch_shift_grid_v1` renders and verifies an explicit,
  caller-supplied integral pitch grid. Audio and MIDI are transposed together,
  and callers may set narrower output pitch bounds for a particular model.
  Drum tracks are rejected because General MIDI drum numbers identify kit
  pieces rather than transposable pitches.
- `fractional_detuning_v1` changes audio by less than half a semitone while
  leaving symbolic MIDI pitches unchanged.
- `archival_noise_v1` models configurable coloured noise and mains hum. It is
  useful only when those conditions resemble the intended target domain.
- `local_time_warp_v1` renders a complete recording once through a continuous,
  monotonic time map and applies that exact map to MIDI event boundaries. It
  does not cut, repeat, or splice chunks.

These methods are opt-in. Their effect on transcription accuracy should be
measured on held-out source groups for each dataset rather than assumed from
another instrument or corpus.

```python
from amt_augmentor import NoiseSNRParameters, noise_snr_v1

noise_snr_v1(
    "tune.wav",
    "tune.mid",
    "tune_augmented_noise.wav",
    "tune_augmented_noise.mid",
    seed=42,
    parameters=NoiseSNRParameters(target_snr_db=24.0),
)
```

```python
from amt_augmentor import (
    ArchivalNoiseParameters,
    FractionalDetuningParameters,
    archival_noise_v1,
    fractional_detuning_v1,
)

fractional_detuning_v1(
    "tune.wav",
    "tune.mid",
    "tune_detuned.wav",
    "tune_detuned.mid",
    seed=42,
    parameters=FractionalDetuningParameters(cents=30.0),
)

archival_noise_v1(
    "tune.wav",
    "tune.mid",
    "tune_archival.wav",
    "tune_archival.mid",
    seed=42,
    parameters=ArchivalNoiseParameters(
        target_snr_db=28.0,
        hum_power_fraction=0.20,
        mains_frequency_hz=50.0,  # Use the value appropriate for the corpus.
        harmonic_count=3,
    ),
)
```

For an integral pitch study, pass the tested shifts explicitly rather than
relying on a package-wide grid:

```python
from amt_augmentor import materialize_pitch_shift_grid_v1

materialize_pitch_shift_grid_v1(
    "tune.wav",
    "tune.mid",
    "pitch_grid",
    seed=42,
    semitones=(-2, -1, 1, 2),
)
```

The research-facing functions require a nonnegative integer seed, validate all
selected parameters, and record them in provenance. They refuse to overwrite
source or output files, validate the audio/MIDI pair (including relevant note
bounds), preserve channel count, and write a JSON sidecar containing source and
output hashes plus the exact parameter plan.
Time-stretched and pitch-shifted labels are serialized as high-resolution,
constant-tempo AMT annotations; they are not intended as symbolic scores with
an inherited tempo map.

The audio, MIDI, and provenance files form one logical bundle. Payloads are
staged first, the provenance sidecar is published last as the completion
marker, and caught failures are rolled back. Consumers should only accept a
bundle when the sidecar exists and its hashes match.

Mixed-audio note removal, experimental synthetic pause insertion, and an
experiment-specific composite materializer were withdrawn after review and
are not part of the package API. Their prototypes remain available through Git
history only. The historical `add_pause` implementation is retained solely to
reproduce old runs because the legacy CLI depends on it. It is disabled by
default, omitted from the advertised effects, emits a visible warning when
explicitly enabled, and must not be used for new training data. See
[`docs/AUGMENTATION_SCOPE.md`](https://github.com/LarsMonstad/amt-augmentor/blob/main/docs/AUGMENTATION_SCOPE.md)
for the decision and current evaluation scope.

## Research case study

A configuration evaluated on historical Hardanger fiddle recordings is
described separately in the
[Hardanger fiddle case study](https://github.com/LarsMonstad/amt-augmentor/blob/main/docs/HARDANGER_FIDDLE_CASE_STUDY.md).
It supports
the manuscript *Automatic Music Transcription for Oral Traditions: Methodology
and a Hardanger Fiddle Case Study* (in preparation). The case study documents
one corpus-specific use of the toolbox; it is neither a package default nor a
claim that the same settings are optimal for other AMT datasets. The exact
Galdr experiment manifests and training adapters belong with that research
artifact, not in the general-purpose package API.

## Requirements

- Python 3.9, 3.10, 3.11, 3.12, or 3.13
- System dependencies: `libsndfile` and `ffmpeg` (for audio processing)

## Installation

You can install AMT-Augmentor either via pip or by cloning the repository:

### Using pip

```bash
pip install amt-augmentor
```

### From source

```bash
git clone https://github.com/LarsMonstad/amt-augmentor.git
cd amt-augmentor
pip install -e .
```


## Usage

### Basic Usage

```bash
amt-augmentor /path/to/dataset/directory
# Or running directly
python -m amt_augmentor.main /path/to/dataset/directory
```



This processes compatible audio files and their corresponding MIDI files using
the legacy batch configuration. That workflow samples parameters from the YAML
ranges; the deterministic paired Python APIs shown above instead use explicit,
provenance-recorded settings. The legacy workflow flattens notes into a simple
event annotation during processing, so use the paired Python APIs when MIDI
instrument tracks, controls, or pitch bends must be retained.

### Advanced Usage

```bash
# Use a custom configuration file
amt-augmentor /path/to/dataset/directory --config my_config.yaml

# Set random seed for reproducible augmentation
# (forces num_workers=1 — worker subprocesses don't inherit RNG state)
amt-augmentor /path/to/dataset/directory --seed 42

# Reproducible train/test/validation split (independent of --seed)
amt-augmentor /path/to/dataset/directory --split-seed 7

# Specify an output directory
amt-augmentor /path/to/dataset/directory --output-directory /path/to/output

# Generate a default configuration file
amt-augmentor --generate-config my_config.yaml

# Disable specific effects
amt-augmentor /path/to/dataset/directory --disable-effect timestretch --disable-effect chorus

# Control legacy merge behavior via the YAML config (merge_audio.merge_num)
# (no CLI flag — see config.sample.yaml for the merge_audio.merge_num key)

# Modify existing dataset CSV files
amt-augmentor --modify-csv dataset.csv --list-split all  # List all songs
amt-augmentor --modify-csv dataset.csv --move-to-split test --song-patterns "Mozart"  # Move songs
amt-augmentor --modify-csv dataset.csv --remove-songs --song-patterns "BadRecording"  # Remove songs

# Parallel processing with 8 workers
amt-augmentor /path/to/dataset/directory --num-workers 8

# Custom train/test/validation split
amt-augmentor /path/to/dataset/directory --train-ratio 0.8 --test-ratio 0.1 --validation-ratio 0.1

# Force specific songs to test set (prevents augmentation)
amt-augmentor /path/to/dataset/directory --custom-test-songs "song1,song3,song5"

# Force specific songs to validation set (prevents augmentation)
amt-augmentor /path/to/dataset/directory --custom-validation-songs "song2,song4"

# Dry run to preview what will be processed
amt-augmentor /path/to/dataset/directory --dry-run

# Verbose output for debugging
amt-augmentor /path/to/dataset/directory --verbose

# Check for valid MIDI-WAV pairs before processing
amt-augmentor /path/to/dataset/directory --check-pairs

# List available effects
amt-augmentor --list-effects

# Check version
amt-augmentor --version
```

### Help and options

```bash
amt-augmentor --help
```

## Configuration

All augmentation parameters can be customized using a YAML configuration file. See `config.sample.yaml` for a complete example with documentation.


## File Format Support

### Audio
- Input: WAV, FLAC, MP3, M4A, AIFF 
- Output: WAV (44.1kHz)

### Annotations
- MIDI (.mid)

## Output Structure

On first run, the toolkit reorganizes your dataset into two subfolders:

    <dataset>/
        original/    # your pristine audio + MIDI pairs (moved from the input dir)
        augmented/   # every augmented file produced by the pipeline

Keeping augmented output in its own folder means you can delete `augmented/` to
reset the dataset without touching any source material.

Augmented files follow the naming convention:

    original_name_augmented_effect_parameter_randomsuffix.extension

The `_augmented_` identifier ensures all augmented files are properly recognized
and handled during dataset creation. Example of `augmented/` contents:

    piano_augmented_timestretch_1.2_abc123.wav
    piano_augmented_timestretch_1.2_abc123.mid
    piano_augmented_noise_1.5_def456.wav
    piano_augmented_noise_1.5_def456.mid

The generated CSV references files with their subfolder prefix
(`<dataset>/original/...` and `<dataset>/augmented/...`), so the physical
layout mirrors the CSV exactly.

## Dataset Creation & Validation

The dataset follows the same format as [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro). Songs assigned to test or validation splits will have their augmented versions excluded to prevent data leakage.

### Creating the Dataset CSV

```bash
# Create dataset with default split ratios (70% train, 15% test, 15% validation)
amt-augmentor /path/to/directory

# Create dataset with custom split ratios
amt-augmentor /path/to/directory --train-ratio 0.8 --test-ratio 0.1 --validation-ratio 0.1

# Force specific songs to test set (they won't be augmented)
amt-augmentor /path/to/directory --custom-test-songs "song1,song3,song5"

# Force specific songs to validation set (they won't be augmented)
amt-augmentor /path/to/directory --custom-validation-songs "song2,song4"
```

`--custom-test-songs` and `--custom-validation-songs` use case-insensitive
substring matching against the song stem. If the same title matches both lists,
test wins and a warning is printed. Pinned songs are also **skipped at
augmentation time** — no augmented WAVs/MIDIs are written for them — so
held-out evaluation data stays untouched on disk and in the CSV. Substring
matching means `--custom-test-songs "piece_17"` will pin every variant
(`piece_17_take1`, `piece_17_take2`, `piece_17_studio`, ...) to the same
split.

### Validating the Dataset Split

Dataset split validation is automatically performed after CSV creation to ensure:
- Augmented songs are not included in test/validation splits
- No cross-split contamination occurs (an augmented row's source original must
  live in the same split)
- Every augmented row has a matching original (no "orphan aug" rows)

You can also run validation as a standalone, side-effect-free check — useful
after hand-editing a CSV, merging datasets, or receiving a split from someone
else:

```bash
# Human-readable report (exits 0 regardless)
amt-augmentor --validate-csv dataset.csv

# CI-friendly: non-zero exit when contamination is found
amt-augmentor --validate-csv dataset.csv --strict

# Machine-readable JSON (for piping into other tooling)
amt-augmentor --validate-csv dataset.csv --json

# Equivalent direct invocation of the validator module
python -m amt_augmentor.validate_split dataset.csv --strict
```

### CSV Format

The generated CSV follows the MAESTRO format with the following columns:
- canonical_composer
- canonical_title
- split
- year
- midi_filename
- audio_filename
- duration

### Modifying Existing Datasets

After creating a dataset CSV, you can easily modify it to adjust train/test/validation splits:

```bash
# List all songs and their distribution
amt-augmentor --modify-csv dataset.csv --list-split all

# List only test songs
amt-augmentor --modify-csv dataset.csv --list-split test

# List all songs with detailed view
amt-augmentor --modify-csv dataset.csv --list-split all --verbose

# Move songs to a different split (substring matching)
amt-augmentor --modify-csv dataset.csv --move-to-split test --song-patterns "Mozart,Chopin"

# Remove songs from dataset
amt-augmentor --modify-csv dataset.csv --remove-songs --song-patterns "BadRecording1,BadRecording2"

# Create backup before modifications (off by default)
amt-augmentor --modify-csv dataset.csv --move-to-split validation --song-patterns "Bach" --backup
```

**Features:**
- **Substring matching**: Patterns like "Mozart" match any song containing that substring
- **Smart augmented handling**: Augmented versions automatically stay in train split only
- **Backup option**: Use `--backup` to create a backup before modifications

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For development:
1. Install development dependencies: `pip install -e ".[dev]"`
2. Run tests: `pytest tests/`
3. Check typing: `mypy amt_augmentor`
4. Format code: `black amt_augmentor`

## Contributors

- **Lars Monstad (@LarsMonstad)** – Original author and maintainer
- **@monoamine11231** – Noise augmentation, custom test songs feature, and various improvements

## Contact

For questions or collaboration:
- Email: lars@botsformusic.com
- Organization: https://botsformusic.com
- GitHub: https://github.com/LarsMonstad/amt-augmentor

## License

MIT License - see LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{amt_augmentor,
  author       = {Lars Monstad and contributors},
  title        = {AMT-Augmentor: Audio + MIDI augmentation toolkit for AMT datasets},
  version      = {2.0.0},
  year         = {2026},
  publisher    = {Bots for Music},
  url          = {https://github.com/LarsMonstad/amt-augmentor}
}
```
