# Grouped packed MSA evidence — 23 September 2026

See the [implementation and full results](../../grouped_msa.md).

- [summary.json](summary.json): settings, timings, accuracy deltas and uncertainty.
- [per-target.csv](per-target.csv): all 116 protein lDDT/backbone lDDT scores and 56 protein-interface DockQ comparisons.
- [per-batch.csv](per-batch.csv): matched full-model batch timings.
- [full_source_hashes.json](full_source_hashes.json): model source hashes for control `7d8233a` and the grouped candidate, relative to `src/`.
- [full_ledger.json](full_ledger.json): process clocks and sampled device-memory peaks.
- [grouped_profile4.json](grouped_profile4.json): a separate four-target trunk profile confirming grouped PWA/OPM dispatch. This is not the full-model timing measurement.
- [preflight_representation_comparison.json](preflight_representation_comparison.json): sampled single/pair numerical differences on the eight-target preflight; these are not structure-accuracy scores.
- [large](micro-large-v5.json), [mixed](micro-mixed-v5.json), [small](micro-small-v5.json), [shallow](micro-shallow-v5.json): warm complete-operation measurements (two warm-ups, seven timed repetitions).
- [validation.json](validation.json): 142 CUDA regression tests, including 21 grouped MSA tests.
- [archive.json](archive.json): SHA-256 of the separately retained raw archive containing predictions, receipts, profiles, logs and candidate source. The raw archive is not included in this repository.

Both full-prediction arms use packed batch size four, the same grouped triangle backend, seed 42, default three recycles, one diffusion sample and Euler 200 steps. This isolates the change from the previous packed implementation; it is not a fresh original-Boltz-2 baseline. Inputs/MSAs are cached. One run per arm and one seed do not establish statistical runtime significance or accuracy equivalence.

Verify these compact files with `sha256sum -c SHA256SUMS` in this directory.
