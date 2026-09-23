# Packed MSA evidence — 23 September 2026

See the [profiling and validation report](../../packed_msa_profiling.md).

- [summary.json](summary.json): complete timings, settings, warm operator measurements, validation and limits.
- [per-batch.csv](per-batch.csv): all 29 batches covering the same 116 targets in both arms.
- [representation_comparison.json](representation_comparison.json): all 58 sampled single/pair comparisons.
- [full-smoke-summary.json](full-smoke-summary.json): identical saved coordinates and confidence outputs for four full default-profile predictions per arm.
- [validation.json](validation.json): 62 exact-equality CUDA tests.
- [candidate source hashes](candidate-source-sha256.json) and [control source hashes](baseline-source-sha256.json): paths relative to `src/`. The control is commit `e331765`.
- [PWA before](large_pwa_sequential.txt), [PWA after](large_pwa_triton.txt), [OPM before](large_opm_sequential.txt), [OPM after](large_opm_triton.txt): operator profiles at 16,224 MSA rows × 1,062 tokens. Profiler tables include both operator and kernel entries; do not sum the overlapping rows.
- [archive.json](archive.json): checksum of the retained raw evidence archive, including full smoke outputs and receipts. The archive is retained separately, not a download included in this repository.

Verify these compact files with `sha256sum -c SHA256SUMS` from this directory.
The 116-target experiment stops before diffusion and confidence. Its 2.10% process-time reduction is not a full-model speedup. The full-prediction smoke check covers four targets with one sample each; it is not a new 116-target structural-accuracy evaluation.
