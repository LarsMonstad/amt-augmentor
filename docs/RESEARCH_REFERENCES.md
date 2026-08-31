# Research references

AMT-Augmentor publishes completed augmentation studies as human-readable case
studies and machine-readable reference cards. The cards make model, dataset,
split, seed, checkpoint, metric, exposure design, limitations, and upstream
artifact hashes explicit. They are evidence records, not package presets: a
card cannot change runtime defaults, and a result from one corpus is not a
recommended recipe for every AMT system.

## Available references

| Study | Status and scope | Narrative | Reference card |
| --- | --- | --- | --- |
| Galdr / Hardanger fiddle | Complete single-seed validation evidence on six tune-disjoint validation tunes; no test inference | [Case study](HARDANGER_FIDDLE_CASE_STUDY.md) | [`galdr-hardanger-v1.json`](../amt_augmentor/research_references/galdr-hardanger-v1.json) |
| MAESTRO / Transkun | Complete single-seed validation evidence on 137 recordings; fixed 50,000-update single-family and physical dataset-expansion checkpoints, with no test inference | [Case study](MAESTRO_TRANSKUN_CASE_STUDY.md) | [`maestro-transkun-v1.json`](../amt_augmentor/research_references/maestro-transkun-v1.json) |

The MAESTRO reference compares an untouched official checkpoint, clean
fine-tuning, Transkun's built-in augmentation, and selected AMT-Augmentor
treatments under two linked evaluation protocols. The follow-up physically
adds deterministic transformed song versions beside every original rather
than replacing originals stochastically. It records the completed audited
results and keeps the clean control distinct from the untouched baseline.

## Machine-readable format

[`schema-v1.json`](../amt_augmentor/research_references/schema-v1.json) defines the version 1
card format. Result rows retain full available metric precision and identify
their canonical upstream artifact by logical locator and SHA-256. The upstream
training and evaluation artifacts are not bundled with this package; the hash
allows a holder of the research workspace to verify the cited file.

Installed cards are available without a repository checkout:

```python
from importlib.resources import files

card = files("amt_augmentor").joinpath(
    "research_references", "maestro-transkun-v1.json"
)
text = card.read_text(encoding="utf-8")
```

When adding a card:

1. include only completed, audited results;
2. identify the evaluation partition and whether test data were accessed;
3. record every seed, checkpoint-selection protocol, metric aggregation, and relevant
   exposure control;
4. separate direct observations from derived contrasts; and
5. state dataset-specific limitations and keep
   `package_policy.changes_runtime_defaults` set to `false`.
