# Changelog

All notable changes to AMT-Augmentor will be documented in this file.

## [1.1.0] - 2025-09-25

### Added
- Dataset modification commands integrated into main CLI:
  - `--modify-csv` to modify existing dataset CSV files
  - `--list-split` to list songs in specific splits
  - `--move-to-split` to move songs between train/test/validation
  - `--remove-songs` to remove songs from dataset
  - `--song-patterns` for pattern matching when modifying datasets
  - `--backup` flag to create backups (off by default)
- Audio merge feature for combining multiple audio files
- `--seed` flag for reproducible augmentation parameters
- `tabulate` dependency for formatted table output
- PyPI and CI status badges to README
- Multi-OS CI testing (Linux, macOS, Windows)

### Fixed
- Merge audio filenames now include `_augmented_` marker to prevent test/validation contamination
- `gain_chorus` effect now respects `config.enable_random_suffix` setting (was always adding random suffix)
- Consistent file naming across all augmentation effects
- PyPI metadata now correctly shows MIT license

### Changed
- Backup creation for dataset modification is now opt-in (use `--backup` flag)
- Updated Python requirement to >=3.9 (dropped 3.8 support)
- Added Python 3.12 and 3.13 support (with setuptools dependency for pretty_midi compatibility)
- Added upper bounds to all dependencies for stability
- Improved CI/CD workflow with matrix testing across Python 3.9-3.13
- Enhanced package metadata with proper classifiers
- Fixed Windows test failures with proper temporary file handling

### Removed
- Legacy `amt_augpy` directory and references
- `amt-augpy` console script alias

## [1.0.9] - 2025

### Changed
- Renamed package from `amt-augpy` to `amt-augmentor`
- Import path changed from `amt_augpy` to `amt_augmentor`
- All augmented files now use `_augmented_` prefix for consistent identification

### Added
- Noise augmentation effect (contributed by @monoamine11231)
- Custom test songs feature with `--custom-test-songs` flag (contributed by @monoamine11231)
- Enhanced CLI commands: `--version`, `--list-effects`, `--dry-run`, `--verbose`, `--check-pairs`
- Comprehensive test suite with 76% coverage

### Fixed
- Context manager issues in distortionchorus module
- Python 3.8 compatibility issues
- Stereo audio handling in add_noise and add_pauses modules
- Time stretching annotation scaling

## [1.0.5] - 2025

### Added
- YAML configuration system for parameter customization
- Parallel processing for multiple effects
- Support for M4A and AIFF audio formats
- Python type annotations throughout codebase
- Expanded documentation and examples

### Improved
- Error detection and reporting
- Processing performance

## [1.0.0] - 2025

### Initial Release
- Time stretching augmentation
- Pitch shifting augmentation
- Reverb and filtering effects
- Gain and chorus effects
- Pause detection and manipulation
- MAESTRO v3.0.0 dataset compatibility
- Audio standardization to 44.1kHz WAV
- Train/test/validation split generation