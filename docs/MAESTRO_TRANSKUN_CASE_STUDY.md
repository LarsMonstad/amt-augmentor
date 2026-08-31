# MAESTRO / Transkun augmentation case study

This completed reference asks a deliberately practical question: when a mature
Transkun piano-transcription checkpoint is fine-tuned on MAESTRO, how do clean
training, Transkun's built-in augmentation, three controlled single-family
AMT-Augmentor treatments, and two physical dataset-expansion bundles compare?

The answer in this one-seed validation study is clear enough to guide the next
experiment, but not broad enough to define a package recipe. The three
single-family AMT-Augmentor arms stayed close to clean fine-tuning and all
scored above the built-in bundle. A follow-up RT expansion was effectively
tied with clean, while NRT was slightly lower. None established an improvement
over the clean control.

## What was compared

The downloaded official Transkun paper checkpoint was first evaluated
untouched. That **paper-baseline row is a contextual reference**, not a member
of the 50,000-update fine-tuning experiment.

The five controlled fine-tuning arms then started from that same checkpoint,
used fresh optimizers, seed 424242, two GPUs, an effective batch size of 12,
and a fixed final checkpoint at 50,000 optimizer updates:

- **Clean:** 1,924 original-only metadata entries, formed by duplicating each
  of 962 training recordings.
- **Built-in:** the same original-only metadata, with Transkun's on-the-fly
  bundle enabled: random stereo downmix, small detuning, seven-band EQ, and
  Gaussian noise.
- **Noise:** 50% original and 50% deterministic `noise_snr_v1` views at 20,
  25, 30, or 35 dB SNR.
- **Reverb:** 50% original and 50% deterministic `reverb_only_v1` views using
  three mild-to-moderately-strong room profiles.
- **Time stretch:** 50% original and 50% deterministic `time_stretch_v1`
  views at rates 0.90, 0.95, 1.05, or 1.10, with MIDI timing transformed with
  the audio.

Training and validation used pedal-extended labels. Evaluation covered all
137 MAESTRO validation recordings, aggregated as a macro mean by recording,
using 16-second segments with an 8-second hop. Test audio and labels were not
opened. There was no checkpoint search.

This is mature-checkpoint fine-tuning, not a reproduction of Transkun's
500,000-update paper training schedule.

## Exact completed results

All values below are fractions copied at full available precision from the
canonical `validation.json` artifacts. Deltas are arm minus reference; the
clean row is the primary fine-tuning control.

| Arm | Onset F1 | Onset + offset F1 | Δ onset vs clean | Δ onset + offset vs clean |
| --- | ---: | ---: | ---: | ---: |
| Untouched official checkpoint | 0.9816826530904008 | 0.932441981955764 | +0.00002120527567051056 | +0.00015655673830972106 |
| Clean, 50k | 0.9816614478147303 | 0.9322854252174543 | 0 | 0 |
| Transkun built-in, 50k | 0.9792357943657011 | 0.923509606997237 | -0.0024256534490292303 | -0.008775818220217224 |
| AMT-Augmentor noise, 50k | 0.981678700714919 | 0.9320669956940891 | +0.000017252900188724496 | -0.00021842952336514898 |
| AMT-Augmentor reverb, 50k | 0.9817032120322566 | 0.9322743458926179 | +0.00004176421752632997 | -0.000011079324836349436 |
| AMT-Augmentor time stretch, 50k | 0.9815850579806292 | 0.932239119077584 | -0.00007638983410107603 | -0.000046306139870222474 |

The same trained arms relative to Transkun's built-in augmentation are:

| Arm | Δ onset vs built-in | Δ onset + offset vs built-in |
| --- | ---: | ---: |
| Clean | +0.0024256534490292303 | +0.008775818220217224 |
| AMT-Augmentor noise | +0.002442906349217955 | +0.008557388696852075 |
| AMT-Augmentor reverb | +0.0024674176665555603 | +0.008764738895380875 |
| AMT-Augmentor time stretch | +0.0023492636149281543 | +0.008729512080347002 |

The machine-readable card binds every row to its validation result and final
checkpoint SHA-256:
[`maestro-transkun-v1.json`](../amt_augmentor/research_references/maestro-transkun-v1.json).

## Physical dataset-expansion follow-up

The follow-up tested AMT-Augmentor in the ordinary dataset-expansion mode: it
kept every original recording and added deterministic transformed song
versions as distinct training entries. This was not a stochastic replacement
scheme and it did not mean that 50% of batches were deliberately kept clean.

- **RT expanded:** each of 962 sources contributed its original, one reverb
  audio version, and one time-stretched audio version with synchronized MIDI,
  for 2,886 entries.
- **NRT expanded:** each source contributed its original plus one noise, one
  reverb, and one synchronized time-stretch version, for 3,848 entries.

The views reused the already verified single-family materializations; dataset
composition performed no new DSP. Both arms disabled Transkun's built-in
augmentation and used the same official starting checkpoint, fresh optimizer,
seed, effective batch, and fixed 50,000-update budget as the earlier controls.
Consequently each arm consumed exactly 600,000 scheduled examples even though
its number of dataset entries differed. RT consumed 199,332 original, 199,483
reverb, and 201,185 time-stretch draws. NRT consumed 149,632 original, 149,766
noise, 149,641 reverb, and 150,961 time-stretch draws. These counts exclude
worker prefetch beyond the fixed-update boundary because those samples did not
contribute gradients.

| Expanded arm | Onset F1 | Onset + offset F1 | Δ onset vs clean | Δ onset + offset vs clean | Δ onset vs built-in | Δ onset + offset vs built-in |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reverb + time (RT) | 0.9816605804254718 | 0.9322877381110631 | -0.0000008673892585298049 | +0.000002312893608857891 | +0.0024247860597707005 | +0.008778131113826082 |
| Noise + reverb + time (NRT) | 0.9816149843649851 | 0.9320641948953048 | -0.00004646344974523764 | -0.00022123032214949045 | +0.0023791899992839927 | +0.008554587898067734 |

RT is, at this precision, an essentially exact tie with clean: its onset F1 is
lower by less than one millionth and its onset-plus-offset F1 is higher by
about two millionths. NRT is slightly below clean on both metrics. Both remain
above the built-in arm, but neither result supports a claim that dataset
expansion improved MAESTRO transcription over clean fine-tuning.

NRT minus RT is -0.000045596060486707835 onset F1 and
-0.00022354321575834835 onset-plus-offset F1. That difference is a direct
bundle contrast, **not an isolated noise effect**: adding the noise view also
reduces the original, reverb, and time-stretch fractions from roughly one
third each to roughly one quarter each. RT was also chosen post hoc from the
earlier single-family results, and no pitch-expanded bundle was tested because
no pitch materialization existed.

## Interpretation

The untouched checkpoint and clean fine-tuning control are extremely close.
That is useful operational evidence: the evaluation and fine-tuning pipeline
did not produce a broad regression before augmentation was introduced.

Against clean, the toolkit differences are tiny and mixed. Noise slightly
increased onset F1 but reduced onset-plus-offset F1. Reverb produced the
highest onset F1, while its onset-plus-offset result was just below clean.
Time stretch was slightly below clean on both. With one seed and no uncertainty
estimate, these small differences do not establish a ranking.

The expanded-bundle follow-up gives the same practical answer under a different
augmentation delivery mode. Adding transformed song versions on top of the
dataset is reproducible and auditable, but in this fixed-compute run RT merely
matched clean and NRT did not improve it. This supports AMT-Augmentor as a way
to construct controlled paired dataset expansions; it does not justify a
universal RT or NRT recipe.

The built-in bundle was lower than clean by 0.0024256534490292303 onset F1 and
0.008775818220217224 onset-plus-offset F1. All three toolkit arms exceeded it
on both metrics. This does **not** show that a toolkit implementation is
intrinsically better: the built-in arm combines four stochastic effects,
whereas each toolkit arm is one deterministic transform family occupying 50%
of metadata. The experiment intentionally tests each approach as configured,
not component-matched implementations.

## What this adds to AMT research

AMT-Augmentor's research contribution is not a claim that every model should
enable a sparse fixed recipe. It provides controllable, audio-and-MIDI-paired
interventions that can be separated by family, assigned explicit exposure,
reproduced from deterministic parameters, and compared with a model's bundled
augmentation under linked protocols that share the same evaluation procedure.

This MAESTRO result and the
[Galdr / Hardanger fiddle reference](HARDANGER_FIDDLE_CASE_STUDY.md) together
show why that separation matters. Selected augmentation helped substantially
in the small archive-domain Galdr study, while on mature MAESTRO/Transkun
fine-tuning the same broad idea did not improve both metrics over clean. The
portable conclusion is therefore methodological: keep a clean control,
measure model-native augmentation, test a small number of explicit families,
and report the exposure and provenance. The winning recipe, if any, remains a
dataset-and-model result.

## Limits of the claim

- There is one seed, 424242.
- Fine-tuning began from a mature official checkpoint and ran for 50,000
  updates; this does not answer what augmentation does during full training
  from scratch.
- The built-in bundle and fixed 50/50 single-family arms are not
  component-matched.
- RT was selected post hoc; the bundle follow-up is descriptive rather than a
  prespecified confirmatory test.
- RT and NRT change both view diversity and original/shared-family exposure,
  so NRT minus RT cannot isolate noise.
- Fixed compute means equal optimizer updates and effective batch, not equal
  epochs or equal component exposure.
- No significance test, multi-seed interval, or paired bootstrap interval was
  computed.
- The evidence is validation-only over 137 recordings; no test data were
  accessed.
- Canonical manifests, checkpoints, predictions, and evaluations are
  hash-bound but are not redistributed with the package.

Accordingly, this card records completed evidence and does not change
AMT-Augmentor defaults.
