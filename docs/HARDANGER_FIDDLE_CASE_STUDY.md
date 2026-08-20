# Hardanger fiddle research context

Some of AMT-Augmentor's controlled APIs were refined while developing Galdr,
an automatic music transcription system for historical and contemporary
Hardanger fiddle recordings. Galdr is the case study in the manuscript
*Automatic Music Transcription for Oral Traditions: Methodology and a
Hardanger Fiddle Case Study* (manuscript in preparation).

The study motivated reusable engineering changes in this package, but its
augmentation recipe is not an AMT-Augmentor default. Exact training manifests,
model settings, seeds, and dataset-specific parameter grids belong with the
Galdr research artifact and the eventual article.

## Findings that informed the toolbox

The current observations are interim, single-seed validation evidence from one
small target-domain dataset. They motivate features and follow-up experiments;
they do not establish a universally optimal augmentation stack.

- Removing integral pitch shift from an otherwise matched conventional stack
  reduced both onset and onset-plus-offset F1 in the Hardanger fiddle study.
  This motivated explicit pitch-grid materialization, label-range preflight,
  and acoustic verification. The useful transposition range remains dependent
  on the instrument, corpus, and output vocabulary.
- Replacing generic white-noise augmentation with a coloured-noise and
  mains-hum simulation was promising for historical archive recordings. This
  motivated `archival_noise_v1`, but archive degradation is target-domain
  simulation and is therefore opt-in. Its SNR, hum share, mains frequency, and
  harmonic count must be selected for the intended collection.
- Fractional detuning was not clearly better than the selected pitch-plus-noise
  recipe. It remains available for datasets with tuning drift, non-standard
  reference pitch, or relevant recording-speed variation, but is not enabled
  implicitly.
- The study did not isolate a reliable general benefit for every conventional
  family. Gain/chorus, reverb/filtering, and time variation remain selectable
  hypotheses that should be ablated on held-out source groups.

## General experimental lessons

The following controls are useful beyond the Hardanger fiddle case:

1. Split by source group before generating any derivative, and augment only
   training sources.
2. Apply exactly the same pitch or time map to audio and MIDI, then verify the
   serialized result rather than trusting nominal transform parameters.
3. Keep augmentation-family exposure and optimizer updates matched across
   comparisons. Adding more physical variants must not silently give one arm
   more training draws.
4. Record parameters, seeds, source hashes, output hashes, and synchronization
   measurements so a generated pair can be audited independently.
5. Treat plausible audio as a prerequisite, not evidence that an augmentation
   improves transcription. Select methods on untouched validation data and
   reserve test data for the frozen final system.

These principles shape AMT-Augmentor's general APIs. They do not make the
toolbox a Hardanger fiddle transcriber, and no Galdr-specific dataset adapter or
training recipe is installed with the package.
