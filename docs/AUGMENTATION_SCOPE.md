# Augmentation scope

## Supported work

The active research surface is limited to conventional AMT augmentations:
gain/chorus, target-SNR noise, reverb/filtering, integral pitch shift, and time
stretch. AMT-Augmentor's contribution in this scope is engineering control:
deterministic parameters, synchronized MIDI changes, strict pair validation,
no-overwrite publication, and source-bound provenance. Similar acoustics to
the original toolbox are expected.

Evaluation should first compare unaugmented training with the complete
conventional set while holding source exposure, split membership, training
budget, seed policy, and evaluation code fixed. Follow-up ablations can then
separate transformations such as gain from chorus or reverb from filtering if
the combined set produces a measurable effect.

## Opt-in successor transforms

The package also exposes two controlled audio-only transforms for new,
separately planned studies. They are not retroactively part of the frozen
Galdr O/C conventional campaign:

- `fractional_detuning_v1` accepts a finite, nonzero detuning strictly inside
  `(-50, 50)` cents. It preserves sample rate, sample count, channel count, and
  MIDI bytes. A conservative first grid is `[-30, -15, 15, 30]` cents.
- `archival_noise_v1` generates independent seeded 1/f-like recursive-noise
  and seeded-phase harmonic-hum streams. It orthogonalizes and normalizes the
  two finite-record components, allocates their requested power fractions,
  and scales the combined interference to the requested RMS SNR over all
  samples and channels before the explicit peak guard and PCM quantization.
  The generator replays two deterministic 65,536-sample-block passes and uses
  one full-length output buffer; it does not allocate a whole-record FFT. A
  fixed recursive-filter coefficient set is identified in provenance with its
  44.1 kHz reference rate. The 1/f-like (not exact 1/f) spectrum contract is
  regression-tested at 8, 16, and 44.1 kHz, including the current dataset
  rate. A conservative first grid is `[24, 28, 32]` dB SNR with
  `hum_power_fraction=0.20`, `mains_frequency_hz=50.0`, and
  `harmonic_count=3`.

Both transforms retain MIDI byte for byte, publish source/output hashes and
DSP measurements in the completion sidecar, refuse all overwrites, and fail
before publication if an input, parameter, shape, or numeric invariant is
invalid. Their acoustic validity and model effect still require a new
training-only plan, independent PCM-level QC, and a matched evaluation.

The bounded archival-noise path was also exercised on an actual 3,249,628-
sample (73.688-second), 44.1 kHz mono dataset record. On the validation host it
completed in 6.99 seconds with 152,888 KiB maximum resident memory, measured
24.0 dB float SNR exactly, and measured a `0.2000000000000001` hum-power
fraction for a requested `0.20`. Runtime and resident memory are host-specific
integration-smoke observations, not portable performance guarantees.

## Withdrawn methods

The following experimental methods are not supported and must not be used for
new datasets or paper claims:

- mixed-audio note removal: deleting a MIDI label cannot isolate and remove
  one note from a polyphonic recording;
- synthetic pause insertion based on short repeated donor audio: review found
  audible periodic artifacts rather than a convincing pause;
- the former five-condition materializer that composed those two methods;
- historical `add_pauses` range masking: it zeros existing musical spans and
  can leave partially intersecting MIDI notes over the modified audio.

The rejected prototypes were removed from the current package, console entry
points, tools, and active documentation. Their implementation remains in Git
history for auditability. The historical range masker alone remains in the
source tree as `legacy_addpauses_unsafe` because the old CLI imports it; it is
disabled by default and retained only for exact forensic reproduction.
