# Augmentation scope

AMT-Augmentor is a general-purpose toolbox for paired audio and symbolic
augmentation in automatic music transcription (AMT). It provides reusable
transform mechanisms and the controls needed to reproduce them; it does not
prescribe one augmentation recipe for every instrument, repertoire, recording
condition, or transcription model.

## Supported transforms

The versioned API provides the following transform families:

| Transform | Audio--MIDI relationship | Intended use |
| --- | --- | --- |
| `gain_chorus_v1` | Changes audio only; preserves MIDI | Level and modulation variation |
| `noise_snr_v1` | Adds seeded noise at an explicit target SNR; preserves MIDI | Generic background-noise robustness |
| `reverb_filters_v1` | Changes audio only; preserves MIDI | Combined room and frequency-response variation |
| `reverb_only_v1` | Changes audio only; preserves MIDI | Isolating room simulation from filtering |
| `pitch_shift_v1` | Shifts audio and pitched MIDI notes together | Integral transposition with synchronized labels |
| `time_stretch_v1` | Stretches audio and maps MIDI event times by the realized sample ratio | Global tempo variation |
| `fractional_detuning_v1` | Detunes audio by less than half a semitone; preserves MIDI | Tuning-reference or recording-speed variation while retaining symbolic pitch labels |
| `archival_noise_v1` | Adds configurable coloured noise and harmonic hum; preserves MIDI | Optional simulation of a measured historical-recording domain |
| `local_time_warp_v1` | Applies one continuous, monotonic time map to audio and MIDI | Local tempo variation without cutting or repeating audio chunks |
| `materialize_pitch_shift_grid_v1` | Materializes and verifies an explicit caller-supplied set of integral shifts | Reproducible pitch-coverage experiments |

The mechanisms are general AMT primitives. Their parameters and relative
frequency of use are experiment choices. In particular, archival noise,
fractional detuning, local time variation, dense pitch grids, and stronger
room simulation are opt-in methods rather than universal defaults. Archival
noise should be configured from the target corpus, including its noise level,
hum contribution, mains frequency, and harmonic structure; those properties
are not the same across archives or countries.

## Pair integrity and reproducibility

The deterministic APIs are designed around the audio--MIDI pair rather than
independent file processing. Depending on the transform, MIDI is either
preserved byte for byte or changed by the same pitch or time mapping applied
to the audio. Time-changing methods map note boundaries and supported timed
MIDI events consistently. Pitch-changing methods validate the requested shift
and the available MIDI range before publishing labels. They reject drum tracks
because General MIDI drum numbers identify kit pieces rather than transposable
pitches. The generic default accepts the legal MIDI range 0--127; a caller can
provide narrower bounds when the downstream model has a smaller vocabulary.

Each transform validates its inputs, parameters, output shape, and finite
numeric invariants. Publication is no-overwrite and transactional: incomplete
output bundles are rolled back rather than presented as successful results.
Completion sidecars bind the result to the source and output hashes and record
the selected parameters, random seed, synchronization checks, and relevant DSP
measurements. These controls make an experiment auditable; they do not by
themselves establish that an augmentation improves a model.

## Choosing and evaluating a recipe

Augmentation strength must be selected for the instrument, corpus, label
representation, and model. A defensible AMT experiment should:

1. assign original recordings to train, validation, and test groups before
   generating derivatives;
2. inherit the source recording's group for every derivative and materialize
   augmentations for training sources only;
3. keep closely related takes, tune variants, and all derivatives of one
   source in the same group to prevent leakage;
4. record explicit parameters and deterministic seeds rather than relying on
   hidden presets;
5. compare recipes with matched source exposure, training budget, checkpoint
   rule, and evaluation code;
6. select augmentation choices on validation data and keep the test set sealed
   until the recipe is fixed; and
7. use focused ablations to distinguish the effects of transformations that
   were previously combined, such as reverb and filtering.

Acoustic quality control and pair-integrity tests are necessary before
training. Model benefit still has to be measured on held-out source groups;
results from one repertoire should not be presented as proof of a universal
AMT augmentation rule.

## Hardanger fiddle case study

Several of the configurable methods were motivated or refined during the
forthcoming Galdr/Hardanger fiddle study. That work is a case study of how the
general APIs can be assembled and evaluated for one historical fiddle corpus,
not the package's default configuration. Exact grids, augmentation mixtures,
seeds, and exposure schedules belong to the experiment artifact and article
rather than the reusable toolbox API. See
[Hardanger fiddle case study](HARDANGER_FIDDLE_CASE_STUDY.md) for the current
scope and appropriately limited interpretation of those findings.

## Withdrawn methods

The following experimental methods are not supported and must not be used for
new datasets or research claims:

- mixed-audio note removal: deleting a MIDI label cannot isolate and remove
  one note from a polyphonic recording;
- synthetic pause insertion based on short repeated donor audio: review found
  audible periodic artifacts rather than a convincing pause;
- the experimental composite materializer that combined those rejected
  methods; and
- historical `add_pauses` range masking: it zeros existing musical spans and
  can leave partially intersecting MIDI notes over modified audio.

The rejected prototypes were removed from the current package, console entry
points, tools, and active documentation. Their implementation remains in Git
history for auditability. The historical range masker alone remains in the
source tree as `legacy_addpauses_unsafe` because the old CLI imports it; it is
disabled by default and retained only for exact forensic reproduction.
