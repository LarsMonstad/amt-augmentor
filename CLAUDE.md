# AMT-AugPy Development Notes

## Common Development Commands

### Testing
```bash
pytest tests/
```

### Linting and Type Checking
```bash
# Install dev dependencies
pip install mypy pylint black

# Run type checker
mypy amt_augpy

# Run linter
pylint amt_augpy

# Auto-format code
black amt_augpy
```

## Codebase Structure

AMT-AugPy is an audio augmentation toolkit for Automatic Music Transcription datasets that processes paired audio/MIDI files with various transformations:

- `time_stretch.py` - Time stretching (0.6-1.6x)
- `pitch_shift.py` - Pitch shifting (-5 to +5 semitones)
- `reverbfilter.py` - Reverb and filtering effects
- `distortionchorus.py` - Gain and chorus effects
- `add_pauses.py` - Manipulates musical pauses
- `convertfiles.py` - Standardizes audio formats
- `create_maestro_csv.py` - Creates dataset CSV
- `validate_split.py` - Validates dataset integrity

## Style Guidelines

- Use Python type hints throughout
- Follow PEP 8 naming conventions
- Document functions with docstrings
- Use structured exception handling
- Centralize configuration parameters

## Improvement Roadmap

- Add configuration system for augmentation parameters
- Implement parallelized processing for better performance
- Support additional audio augmentations
- Improve error handling and logging
- Expand test coverage