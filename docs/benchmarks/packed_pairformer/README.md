# Packed Pairformer measurement evidence

`accuracy-summary.json`, `benchmark.json`, `training.json`, `branch.json`, and `protocol.json` are the original validated experiment's summaries/raw timing samples. The historical `file` values in the accuracy summary identify the original run records. Its numerical results were reproduced after consolidation.

`cleanup/summary.json` summarizes the cleaned source's four full-stack runs. The corresponding `cleanup/*.json.gz` files contain every output, input-gradient and all 3,392 parameter-gradient metrics, plus per-record/max-absolute errors. Gzip keeps this evidence compact without losing values. Read it with `gzip -dc cleanup/random-b4.json.gz` or Python's `gzip.open`/`json.load`.

The cleanup tests, environment/source hashes, and real-batch timing check are saved alongside those results. All backend names in the historical timing files describe measured baselines, not additional selectable implementations. See [the guide](../../packed_pairformer.md) for the interpretation and limitations.
