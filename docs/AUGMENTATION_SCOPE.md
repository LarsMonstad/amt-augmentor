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
