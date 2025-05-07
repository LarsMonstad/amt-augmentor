"""
Configuration module for AMT-AugPy.

This module provides default configuration parameters and utilities for loading
custom configurations from YAML files.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import os
import yaml
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Big thanks to https://github.com/Xpra-org/xpra/commit/04108790f05f0894e520c2d3e62dd16e136b9da2
class TupleLoader(yaml.SafeLoader):
    def construct_tuple(self, node):
        return tuple(yaml.Loader.construct_sequence(self, node))

TupleLoader.add_constructor("tag:yaml.org,2002:python/tuple", TupleLoader.construct_tuple)

@dataclass
class TimeStretchConfig:
    """Configuration for time stretching augmentation."""

    enabled: bool = True
    variations: int = 3
    min_factor: float = 0.6
    max_factor: float = 1.6
    randomized: bool = True


@dataclass
class PitchShiftConfig:
    """Configuration for pitch shifting augmentation."""

    enabled: bool = True
    variations: int = 3
    min_semitones: int = -5
    max_semitones: int = 5
    randomized: bool = True


@dataclass
class ReverbFilterConfig:
    """Configuration for reverb and filter augmentation."""

    enabled: bool = True
    variations: int = 3
    min_room_scale: int = 10
    max_room_scale: int = 100
    cutoff_pairs: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (20, 20000),
            (300, 20000),
            (3000, 20000),
            (20, 16300),
            (20, 17500),
            (20, 18000),
        ]
    )


@dataclass
class GainChorusConfig:
    """Configuration for gain and chorus augmentation."""

    enabled: bool = True
    variations: int = 3
    min_gain: int = 2
    max_gain: int = 11
    min_depth: float = 0.1
    max_depth: float = 0.6
    rates: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])


@dataclass
class AddPauseConfig:
    """Configuration for pause manipulation augmentation."""

    enabled: bool = True
    pause_threshold: float = 0.0033
    min_pause_duration: int = 1
    max_pause_duration: int = 5


@dataclass
class AudioMergeConfig:
    """Configuration for merging of multiple audio files"""

    enabled: bool = True
    merge_num: int = 1

@dataclass
class AddNoiseConfig:
    """Configuration for adding noise to audio files"""

    enabled: bool = true

@dataclass
class ProcessingConfig:
    """Configuration for processing settings."""

    num_workers: int = 4
    cache_dir: Optional[str] = None
    output_dir: Optional[str] = None


@dataclass
class Config:
    """Main configuration container."""

    enable_random_suffix: bool = False

    time_stretch: TimeStretchConfig = field(default_factory=TimeStretchConfig)
    pitch_shift: PitchShiftConfig = field(default_factory=PitchShiftConfig)
    reverb_filter: ReverbFilterConfig = field(default_factory=ReverbFilterConfig)
    gain_chorus: GainChorusConfig = field(default_factory=GainChorusConfig)
    add_pause: AddPauseConfig = field(default_factory=AddPauseConfig)
    merge_audio: AudioMergeConfig = field(default_factory=AudioMergeConfig)
    add_noise: AddNoiseConfig = field(default_factory=AddNoiseConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. If None, returns default config.

    Returns:
        Config object with parameters from the YAML file or defaults
    """
    config = Config()

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                yaml_config = yaml.load(f, Loader=TupleLoader)

            # Update configurations from YAML
            if yaml_config:
                if 'enable_random_suffix' in yaml_config:
                    config.enable_random_suffix = yaml_config['enable_random_suffix']
            
                # Time stretch config
                if "time_stretch" in yaml_config:
                    for key, value in yaml_config["time_stretch"].items():
                        if hasattr(config.time_stretch, key):
                            setattr(config.time_stretch, key, value)

                # Pitch shift config
                if "pitch_shift" in yaml_config:
                    for key, value in yaml_config["pitch_shift"].items():
                        if hasattr(config.pitch_shift, key):
                            setattr(config.pitch_shift, key, value)

                # Reverb filter config
                if "reverb_filter" in yaml_config:
                    for key, value in yaml_config["reverb_filter"].items():
                        if hasattr(config.reverb_filter, key):
                            setattr(config.reverb_filter, key, value)

                # Gain chorus config
                if "gain_chorus" in yaml_config:
                    for key, value in yaml_config["gain_chorus"].items():
                        if hasattr(config.gain_chorus, key):
                            setattr(config.gain_chorus, key, value)

                # Add pause config
                if "add_pause" in yaml_config:
                    for key, value in yaml_config["add_pause"].items():
                        if hasattr(config.add_pause, key):
                            setattr(config.add_pause, key, value)

                # Audio merge config
                if "merge_audio" in yaml_config:
                    for key, value in yaml_config["merge_audio"].items():
                        if hasattr(config.add_pause, key):
                            setattr(config.merge_audio, key, value)
                
                if "add_noise" in yaml_config:
                    for key, value in yaml_config["add_noise"].items():
                        if hasattr(config.add_noise, key):
                            setattr(config.add_noise, key, value)

                # Processing config
                if "processing" in yaml_config:
                    for key, value in yaml_config["processing"].items():
                        if hasattr(config.processing, key):
                            setattr(config.processing, key, value)

            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        if config_path:
            logger.warning(
                f"Configuration file {config_path} not found. Using default configuration."
            )
        else:
            logger.info("No configuration file specified. Using default configuration.")

    return config


def save_default_config(output_path: str) -> None:
    """
    Save the default configuration to a YAML file.

    Args:
        output_path: Path where to save the config file
    """
    config = Config()

    # Convert dataclasses to dictionaries
    config_dict = {
        "time_stretch": {k: v for k, v in vars(config.time_stretch).items()},
        "pitch_shift": {k: v for k, v in vars(config.pitch_shift).items()},
        "reverb_filter": {k: v for k, v in vars(config.reverb_filter).items()},
        "gain_chorus": {k: v for k, v in vars(config.gain_chorus).items()},
        "add_pause": {k: v for k, v in vars(config.add_pause).items()},
        "merge_audio": {k: v for k, v in vars(config.merge_audio).items()},
        "processing": {k: v for k, v in vars(config.processing).items()},
    }

    try:
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Default configuration saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving default configuration to {output_path}: {e}")
