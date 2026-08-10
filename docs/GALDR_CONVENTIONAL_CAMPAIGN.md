# Galdr conventional O/C campaign

This adapter creates a controlled training comparison from the clean Galdr
dataset:

- `O`: one unchanged audio view per training source, paired with a derivative
  MIDI in which earlier same-pitch overlaps are truncated at the later onset.
- `C`: the same normalized original plus one deterministic gain/chorus, noise,
  reverb/filter, pitch-shift, and time-stretch view per source.

Validation and test records are never materialized. Canonical audio and MIDI
are hash-checked before and after work and are never edited. Every parameter
and seed is fixed in a plan before rendering. The conventional stream is the
established `galdr-sigma2/conventional-stream/v1`; with global seed `424242`,
it reproduces the accepted earlier C seed/parameter projection. Byte identity
also depends on identical source bytes, normalized MIDI, package versions, and
runtime, so it must be established by output hashes rather than assumed.

## Commands for the current clean dataset

Use canonical paths. The adapter intentionally refuses paths containing a
symbolic-link component; if a convenience path is a link, pass the result of
`readlink -f` instead.

```bash
PROVENANCE=/cluster/work/users/lamonsta/galdr-sigma2/artifacts/hf2-clean-original-v3-02455e3/dataset-provenance.json
DATASET_ROOT=/cluster/work/users/lamonsta/galdr-sigma2/artifacts/hf2-train-only-v3-02455e3
PLAN=/cluster/work/users/lamonsta/galdr-sigma2/augmentation-conventional-oc-v1-plan.json
OUTPUT=/cluster/work/users/lamonsta/galdr-sigma2/augmentation-conventional-oc-v1

amt-augmentor-galdr-conventional plan \
  --dataset-provenance "$PROVENANCE" \
  --expected-provenance-sha256 933dd135483898da36a363bf1e968d06b53dcdf77a86cd6cd2f72772d1e3eb3a \
  --dataset-root "$DATASET_ROOT" \
  --global-seed 424242 \
  --output-plan "$PLAN"

amt-augmentor-galdr-conventional estimate --plan "$PLAN"

amt-augmentor-galdr-conventional materialize \
  --plan "$PLAN" \
  --dataset-provenance "$PROVENANCE" \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUTPUT" \
  --workers 5

amt-augmentor-galdr-conventional verify --output-root "$OUTPUT"
```

`--workers` accepts 1--16. At most five independent transforms exist per
source, so values above five do not add campaign-level parallelism. Output
ordering and identity are independent of worker count. Run a one-source plan
first to measure wall time on the intended CPU node before submitting the full
Slurm job.

## Outputs

The completion-marked output contains:

- shared normalized-original and conventional media with a provenance sidecar
  for every audio/MIDI pair;
- `conditions/O/metadata.csv` and `conditions/C/metadata.csv`;
- Galdr-compatible `training-lineage.json` and condition identities;
- top-level `derivatives.csv` with the fields required by Galdr's dataset
  split/lineage gate;
- `materialization-report.json`, published last, with a complete payload hash
  inventory.

The checks establish deterministic selection, source integrity, channel and
sample-rate preservation, valid audio, MIDI timing synchronization, and
lineage. They do not establish that an effect is perceptually useful, that its
strength is optimal, or that it improves transcription. Those are separate
technical review and O-versus-C model questions.
