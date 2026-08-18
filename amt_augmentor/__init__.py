# amt_augmentor/__init__.py
__version__ = "2.0.0a8"

from amt_augmentor.conventional_augmentations import (  # noqa: E402
    AGGRESSIVE_REVERB_ONLY_PRESETS_V1,
    MILD_REVERB_ONLY_PRESETS_V1,
    ArchivalNoiseParameters,
    FractionalDetuningParameters,
    GainChorusParameters,
    NoiseSNRParameters,
    PitchShiftParameters,
    ReverbFiltersParameters,
    ReverbOnlyParameters,
    TimeStretchParameters,
    archival_noise_v1,
    fractional_detuning_v1,
    gain_chorus_v1,
    noise_snr_v1,
    pitch_shift_v1,
    reverb_filters_v1,
    reverb_only_v1,
    time_stretch_v1,
)
from amt_augmentor.local_time_warp import (  # noqa: E402
    LocalTimeWarpParameters,
    local_time_warp_v1,
)
from amt_augmentor.pitch_shift_grid import (  # noqa: E402
    CONSERVATIVE_PITCH_SHIFT_GRID_V1,
    DENSE_PITCH_SHIFT_GRID_V1,
    materialize_pitch_shift_grid_v1,
    measure_absolute_pitch_shift_v1,
    validate_pitch_shift_grid_v1,
    verify_pitch_shift_grid_v1,
)

__all__ = [
    "AGGRESSIVE_REVERB_ONLY_PRESETS_V1",
    "CONSERVATIVE_PITCH_SHIFT_GRID_V1",
    "DENSE_PITCH_SHIFT_GRID_V1",
    "MILD_REVERB_ONLY_PRESETS_V1",
    "ArchivalNoiseParameters",
    "FractionalDetuningParameters",
    "GainChorusParameters",
    "LocalTimeWarpParameters",
    "NoiseSNRParameters",
    "PitchShiftParameters",
    "ReverbFiltersParameters",
    "ReverbOnlyParameters",
    "TimeStretchParameters",
    "archival_noise_v1",
    "fractional_detuning_v1",
    "gain_chorus_v1",
    "local_time_warp_v1",
    "materialize_pitch_shift_grid_v1",
    "measure_absolute_pitch_shift_v1",
    "noise_snr_v1",
    "pitch_shift_v1",
    "reverb_filters_v1",
    "reverb_only_v1",
    "time_stretch_v1",
    "validate_pitch_shift_grid_v1",
    "verify_pitch_shift_grid_v1",
]
