# Hardanger fiddle augmentation case study

AMT-Augmentor's controlled APIs were refined while developing Galdr, an
automatic music transcription system for historical and contemporary
Hardanger fiddle recordings. Galdr is the case study in the manuscript
*Automatic Music Transcription for Oral Traditions: Methodology and a
Hardanger Fiddle Case Study* (in preparation).

This page reports the completed engineering evidence available on 2026-08-30.
It is deliberately narrower than a paper claim: every model result uses seed
`424242`, the current `hf2-group-v2` dataset, and six tune-disjoint validation
tunes. Test inference remained disabled. The full-precision results and
artifact hashes are in
[`galdr-hardanger-v1.json`](../amt_augmentor/research_references/galdr-hardanger-v1.json).

## What this contributes to AMT research

Many transcription models provide a small, model-local augmentation option.
AMT-Augmentor addresses a different layer: it makes paired audio and symbolic
transformations model-independent, materializable, and auditable. A rendered
dataset can be reused across architectures, and its MIDI changes, realized
time map, seed, parameters, source hashes, output hashes, and synchronization
checks can be inspected without trusting a training loader's hidden random
state.

The case study adds evidence and experimental patterns beyond merely offering
more effects:

- corpus-matched degradation (`archival_noise_v1`) can be compared with generic
  noise rather than treating all noise as interchangeable;
- pitch grids, reverb strength, global versus local time variation, and
  fractional detuning can be tested as explicit hypotheses;
- logical-slot replacement keeps source/window exposure and optimizer budget
  controlled in ablations; and
- versioned reference cards preserve results, protocol scope, and upstream
  artifact hashes without turning a winning arm into a package default.

This complements built-in augmentation; it does not claim that every model's
built-in implementation lacks these properties. The completed
[MAESTRO / Transkun study](MAESTRO_TRANSKUN_CASE_STUDY.md) provides the direct
built-in-versus-toolkit comparison in a separate card.

## What a targeted, exposure-controlled mixture means

These experiments do not establish that a sparse mixture, or simply using
fewer families, is intrinsically better. They support selecting families for
an explicit reason and assigning each a controlled share of optimizer
exposure. Augmentation itself need not be rare: the current working candidate
uses augmented views for two thirds of its logical slots. The preference
screen keeps the draw budget fixed while each treatment changes 12 of 72
logical slots relative to its control.

The target is not maximum acoustic variety. It is improved transcription on
untouched, source-disjoint data while preserving correct labels and avoiding
families that consume exposure without helping the primary onset-plus-offset
metric. The current Galdr evidence supports retaining pitch shift and
corpus-matched archival noise, preferring global over local time variation,
and treating strong reverb or the combined reverb/filter family as
context-sensitive candidates. It argues against fractional detuning,
moderate reverb, gain/chorus, and an unnecessarily wide four-point pitch grid
for this corpus. This is enough to choose a Galdr engineering recipe; it is not
a universal AMT recipe and is not encoded as an AMT-Augmentor default.

The working full-schedule candidate for later confirmation has 72 logical
slots: 24 clean and 12 each for archival noise, integral pitch shift, the
combined reverb/filter family, and global time stretch. This is an
evidence-informed candidate, not an estimate of the universally optimal
number of families or clean/augmented fraction.

## Common evaluation scope

Unless a table says otherwise, scores are tune-macro F1 against untouched
canonical expert MIDI. Epoch 100 is step 24,000 and epoch 150 is step 36,000.
The ablation, successor, preference, AP/APD, and interaction tables report
protocol-specified fixed checkpoints without checkpoint search. The O/C
campaign instead evaluated 30 prespecified candidates at five-epoch intervals
with a fixed winner rule; its table here reports the epoch-150 rows, not the
winner-selected estimates. Values below are displayed to nine decimal places;
the reference card stores the full available precision.

### Original-only versus conventional augmentation

| Training arm | Epoch | Onset F1 | Onset+offset F1 |
| --- | ---: | ---: | ---: |
| Original-only `O` | 150 | 0.906037072 | 0.846708649 |
| Conventional package `C` | 150 | 0.923736273 | 0.862926445 |
| `C - O` | -- | +0.017699202 | +0.016217797 |

Both arms have the same 150-epoch/36,000-update budget. `O` repeatedly samples
originals; `C` samples the original plus gain/chorus, generic Gaussian noise,
combined reverb/filter, pitch-shift, and global-stretch views. The campaign's
30 candidate epochs and winner rule were prespecified: highest full-precision
onset-plus-offset F1, then onset F1, then earliest update for exact ties. The
rows above answer whether augmentation helps at the fixed epoch-150 horizon;
they are not winner-selected scores and do not identify which family is
responsible.

### Conventional leave-one-family-out

Each removed family is replaced with the original view. Source, window,
logical-slot draws, initialization seed, and update count are controlled.
Delta is `(C-X) - C`; positive means removal improved the score.

| Arm at epoch 150 | Onset F1 | Onset+offset F1 | Onset delta | Onset+offset delta |
| --- | ---: | ---: | ---: | ---: |
| Full `C` | 0.923736273 | 0.862926445 | -- | -- |
| Minus gain/chorus | 0.924114269 | 0.866725412 | +0.000377995 | +0.003798966 |
| Minus Gaussian noise | 0.919540674 | 0.864573956 | -0.004195600 | +0.001647510 |
| Minus reverb/filters | 0.926075373 | 0.868676638 | +0.002339100 | +0.005750193 |
| Minus pitch shift | 0.908132698 | 0.852915820 | **-0.015603576** | **-0.010010626** |
| Minus global stretch | 0.923411334 | 0.867768211 | -0.000324939 | +0.004841766 |

Pitch is the only conventional family whose removal clearly hurts both
metrics. Other rows are conditional effects inside an already dense package;
they are not claims that those transforms can never help in a simpler recipe.

### Targeted archival noise and fractional detuning

Each successor replaces the duplicated-original slot in its paired omitted-
family control.

| Candidate | Epoch 100 onset / onset+offset | Epoch 150 onset / onset+offset | Epoch 150 delta vs paired control |
| --- | ---: | ---: | ---: |
| Archival noise/hum | 0.922709796 / 0.862835770 | **0.928012876 / 0.874963099** | **+0.008472202 / +0.010389143** |
| Fractional detuning | 0.918608692 / 0.861618281 | 0.923315059 / 0.868091496 | -0.000799210 / +0.001366084 |

Archival noise is a clear target-domain contribution in this experiment.
Detuning is mixed here, and the more direct recipe comparison below resolves
the decision against adding it.

### Matched preference screen

All arms use 72 logical slots and 2,880,000 draws. The `p1` control contains
48 original, 12 archival-noise, and 12 legacy-pitch slots. Pitch arms replace
the 12 pitch slots. Reverb and time arms replace 12 of the 48 original slots,
giving 36 original, 12 archival-noise, 12 legacy-pitch, and 12 treatment slots.
Every arm therefore changes 12 logical slots while preserving total draws and
updates. Pitch-grid substitution and adding a new family still answer
different questions.

| Arm | Epoch 100 onset / onset+offset | Epoch 150 onset / onset+offset | Epoch 150 delta vs `p1` |
| --- | ---: | ---: | ---: |
| `p1`: archival + legacy pitch | 0.912034231 / 0.846627059 | 0.918027038 / 0.859133418 | -- |
| `p4`: pitch +/-1, +/-2 | 0.916787097 / 0.858558532 | 0.921450818 / 0.866944432 | +0.003423780 / +0.007811015 |
| `p4-wide`: pitch +/-1, +/-6 | 0.912930070 / 0.848828338 | 0.920087567 / 0.860643573 | +0.002060529 / +0.001510155 |
| `p12`: every semitone +/-1 through +/-6 | 0.917823399 / 0.857061154 | 0.922062593 / 0.865728597 | +0.004035554 / +0.006595179 |
| RF: combined reverb + filters | 0.918273584 / 0.856944805 | 0.923297783 / 0.864555096 | +0.005270745 / +0.005421679 |
| Moderate reverb | 0.912073996 / 0.846076349 | 0.916183632 / 0.858529990 | -0.001843406 / -0.000603428 |
| Strong reverb | 0.920493705 / 0.861401863 | 0.925767074 / 0.868965409 | +0.007740035 / +0.009831992 |
| Global stretch | **0.921160539 / 0.863179524** | **0.927350332 / 0.871910152** | **+0.009323294 / +0.012776734** |
| Local time warp | 0.920922968 / 0.855642208 | 0.924894303 / 0.864745710 | +0.006867265 / +0.005612293 |

Global stretch is the strongest preference arm at both checkpoints. Strong
reverb beats moderate reverb. `p4` is preferable to `p12` on onset-plus-offset
F1, while `p12` is slightly higher on onset F1; `p4-wide` offers little.

RF means the legacy combined **reverb plus high/low-pass filtering** treatment,
not random forest. Because the components were rendered together, its score
cannot be attributed separately to reverb and filtering.

### Archival + pitch with and without detuning

AP and APD use the same source/window/logical-slot stream. APD changes exactly
one original slot to fractional detuning.

| Recipe | Epoch 100 onset / onset+offset | Epoch 150 onset / onset+offset |
| --- | ---: | ---: |
| AP: archival + pitch | 0.921313168 / 0.861548088 | 0.924811708 / 0.868226643 |
| APD: AP + detuning | 0.917235185 / 0.851587302 | 0.921921518 / 0.861704999 |
| `APD - AP` | **-0.004077983 / -0.009960786** | **-0.002890189 / -0.006521644** |

The exact paired comparison is negative at both checkpoints, so fractional
detuning is not part of the current Galdr recipe.

### Late-training RF by global-stretch interaction

These four cells share `p1` training through epoch 100. Treatments are applied
on separate draws during epochs 101--150.

| Cell at epoch 150 | Onset F1 | Onset+offset F1 | Delta vs control |
| --- | ---: | ---: | ---: |
| Control | 0.918027038 | 0.859133418 | -- |
| RF exposure | 0.918505510 | 0.858447744 | +0.000478472 / -0.000685674 |
| Global-stretch exposure | 0.918741114 | 0.859634880 | +0.000714076 / +0.000501463 |
| Both exposures | **0.919857834** | **0.862003359** | **+0.001830796 / +0.002869941** |
| Difference-in-differences | -- | -- | +0.000638248 / **+0.003054152** |

The onset-plus-offset interaction is positive on all six tunes. This supports
non-redundancy in that late, simplified context, but it is a post-hoc,
late-training result—not a full-course factorial estimate or a composition of
both effects on the same audio draw.

## Package boundary

The evidence motivates general mechanisms and controlled study design, not a
hidden preset. AMT-Augmentor therefore keeps archival noise, detuning, local
time warp, explicit pitch grids, and reverb-only processing opt-in. No Galdr
dataset adapter, training schedule, exposure mixture, or default recipe is
installed with the package. See [Research references](RESEARCH_REFERENCES.md)
for the reference-card policy and cross-dataset index.
