# Changelog

All notable changes to AMT-Augmentor will be documented in this file.

## [1.0.9] - 2024

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

## [1.0.5] - 2024

### Added
- YAML configuration system for parameter customization
- Parallel processing for multiple effects
- Support for M4A and AIFF audio formats
- Python type annotations throughout codebase
- Expanded documentation and examples

### Improved
- Error detection and reporting
- Processing performance

## [1.0.0] - 2024

### Initial Release
- Time stretching augmentation
- Pitch shifting augmentation
- Reverb and filtering effects
- Gain and chorus effects
- Pause detection and manipulation
- MAESTRO v3.0.0 dataset compatibility
- Audio standardization to 44.1kHz WAV
- Train/test/validation split generation