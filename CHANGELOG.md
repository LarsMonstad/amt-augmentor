# Changelog

All notable changes to AMT-Augmentor will be documented in this file.

## [1.2.2]

### Fixed
- **Hotfix for v1.2.1 silent no-op.** The processing loop's existence check
  (`os.path.exists(midi_path)`) ran against a double-prefixed path because
  `audio` from `audio_files_described` is already the originals_dir-prefixed
  path returned by `gen_ann`, and v1.2.1 prepended `originals_dir` again.
  Result: `os.path.exists` failed for every file and `process_files` was never
  called — pipelines silently produced 0 augmented files. Loop now derives
  paths from `os.path.basename(audio)` so it works regardless of caller
  convention. Added a regression test that fails on v1.2.1's code.

## [1.2.1]

### Fixed
- Songs pinned to test/validation via `--custom-test-songs` /
  `--custom-validation-songs` are now also skipped at augmentation time,
  not just excluded from the CSV. Previously the pipeline still wrote
  augmented WAVs/MIDIs for those songs to `augmented/` (orphaned files
  that did not appear in the CSV — wasted compute and disk).

## [1.2.0]

### Fixed
- **Cross-split contamination detection** — `validate_dataset_split` previously
  missed contamination on filenames containing embedded extension-like fragments
  (e.g. `foo.mid_cleaned.mid`). The original-stem key was built with
  `str.replace(".mid", "")` which stripped interior `.mid` fragments, while the
  augmented-stem key used `os.path.splitext`, so the two drifted and lookups
  silently failed. Replaced with a single `canonical_stem()` helper that only
  strips a known trailing extension. Also now flags "orphan aug" rows.
- **Showstopper after subfolder migration**: the main processing loop checked
  `os.path.exists(midi_path)` against a bare filename, so after originals moved
  into `<dataset>/original/` the existence check failed and the entire
  augmentation loop silently no-op'd unless cwd happened to be `originals_dir`.
- Stray `print(midi_path)` removed from the processing loop.
- Wrong logger message ("Error applying gain and chorus") emitted on noise
  failures.
- `pitch_shift.update_ann_file` now skips malformed lines and drops notes that
  would transpose outside MIDI 0–127 instead of writing invalid annotations.
- `distortionchorus.apply_gain_and_chorus` no longer shells out via
  `os.system("cp …")` — replaced with `shutil.copy2`, removing a shell-quoting
  hazard on filenames with spaces or special characters.
- `convertfiles.standardize_audio` now writes the converted file, *verifies*
  it, then deletes the original — previously a crash between `os.remove` and
  `os.rename` could lose the source audio.
- Effect-parameter generation (time-stretch, pitch-shift, noise) used
  `set.pop()` to drop excess linspace values; this is unordered and broke
  `--seed` reproducibility. Now sorted-then-truncated.
- `create_song_list` now `sorted()`s the originals listing before greedy
  split assignment, so split membership no longer depends on filesystem
  enumeration order.

### Added
- `--split-seed` flag — deterministic random shuffle of originals before
  greedy split assignment. When `--seed` is set and `--split-seed` is not,
  `--split-seed` defaults to `--seed` so a single seed makes the whole run
  reproducible.
- Auto-fallback to `num_workers=1` when `--seed` is set, with a warning. Worker
  subprocesses spawn fresh interpreters and would otherwise silently ignore
  the seed.

### Changed
- **`merge_audio` is now disabled by default.** The current implementation can
  introduce cross-split contamination — at augmentation time, splits are not
  yet assigned, so a merged "augmented" output (later added to train) may
  contain audio from songs that end up in test or validation. A loud warning
  is emitted whenever the effect is re-enabled. Proper fix (split-aware
  augmentation) is tracked for a follow-up release.
- README: removed reference to a `--merge-num` CLI flag that does not exist;
  merge count is configured via the YAML config (`merge_audio.merge_num`).
- **Dataset layout is now segregated into `original/` and `augmented/` subfolders**
  under the dataset directory. On first run the pipeline moves existing audio +
  MIDI pairs into `<dataset>/original/` and writes all augmented output to
  `<dataset>/augmented/`. The generated CSV references files via those
  subfolder-prefixed paths. Removing an augmented-only dataset is now as simple
  as `rm -rf augmented/` — source material is never mixed in.

### Added
- `amt-augmentor --validate-csv <path>` — standalone, side-effect-free CSV
  contamination check. `--strict` sets a non-zero exit code on any finding,
  `--json` emits a structured report.
- `python -m amt_augmentor.validate_split` grows `--strict` and `--json` flags
  and a non-zero exit code under `--strict`.
- Regression fixtures under `tests/fixtures/contamination/` (clean, simple leak,
  embedded `.mid_cleaned`, Unicode/case, orphan aug, reverse leak) and
  `tests/test_validate_split.py`.

## [?]

### Fixed
- Fixed critical time stretch bug where MIDI annotations were scaled incorrectly (multiplied instead of divided by stretch factor)
- Fixed double file extension bug in merge audio effect (files were incorrectly named as `.wav.wav` and `.wav.mid`)
- Fixed merge effect to only use original files for merging, not augmented versions (was causing "file not found" errors)
- Fixed CRITICAL reverb filter bug where high-pass and low-pass filters were swapped, causing near-silent output
- Fixed noise effect being way too loud by reducing intensity range from 0.1-3.0 to 0.005-0.05
- Fixed noise effect type casting bug that was converting all float intensities to 0 (int cast issue)
- Fixed noise effect using np.logspace incorrectly for non-randomized mode (changed to np.linspace)
- Fixed pitch shift effect not running due to incorrect indentation (for loop was inside else block)

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
