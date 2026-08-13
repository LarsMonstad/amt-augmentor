# amt_augmentor/__init__.py
__version__ = "2.0.0a7"

from amt_augmentor.conventional_augmentations import (  # noqa: E402
    ArchivalNoiseParameters,
    FractionalDetuningParameters,
    GainChorusParameters,
    NoiseSNRParameters,
    PitchShiftParameters,
    ReverbFiltersParameters,
    TimeStretchParameters,
    archival_noise_v1,
    fractional_detuning_v1,
    gain_chorus_v1,
    noise_snr_v1,
    pitch_shift_v1,
    reverb_filters_v1,
    time_stretch_v1,
)

__all__ = [
    "ArchivalNoiseParameters",
    "FractionalDetuningParameters",
    "GainChorusParameters",
    "NoiseSNRParameters",
    "PitchShiftParameters",
    "ReverbFiltersParameters",
    "TimeStretchParameters",
    "archival_noise_v1",
    "fractional_detuning_v1",
    "gain_chorus_v1",
    "noise_snr_v1",
    "pitch_shift_v1",
    "reverb_filters_v1",
    "time_stretch_v1",
]
