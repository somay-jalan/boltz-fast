# Structure116 evidence — 23 September 2026

See the [full benchmark report](../../full_structure116.md) for settings, interpretation and reproduction.

- [summary.json](summary.json): process times, aggregate scores, paired bootstrap intervals, confidence-ranking audit and outlier diagnostics.
- [per-target.csv](per-target.csv): all 116 target scores and differences.
- [input-integrity.json](input-integrity.json): verification receipt for 833 cached input, feature and MSA files.
- [candidate-source-sha256.json](candidate-source-sha256.json): hashes of the 111 benchmarked Python source files; paths are relative to the repository's `src/` directory.
- [archive.json](archive.json): size and SHA-256 of the retained prediction archive.
- [SHA256SUMS](SHA256SUMS): checksums of these five evidence files; verify with `sha256sum -c SHA256SUMS` in this directory.

The original comparison is historical and uses five recycles and five samples, which override CLI defaults. The new packed run completed all 580 structures. The raw prediction archive, model checkpoint and cached dataset are retained separately and are not included in this Git repository. The archive checksum identifies that retained artifact; it is not a download link. The input-integrity receipt records the completed check rather than redistributing the dataset.
