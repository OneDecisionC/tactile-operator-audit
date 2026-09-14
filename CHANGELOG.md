# Changes

## 0.2.0 — 2026-09-14

- Recovered survivor-v4/retention-v3 kernels and their existing tests.
- Restored the historical Conv1D initialization ID.
- Added matched and CF+matched training arms with original paired RNG schedules.
- Connected iPBJ, survivor and retention controls to evaluation.
- Added validated OOF aggregation, pooled paired bootstrap and new-run reports.
- Added bin-width recomputation and independent PNG/SVG plotting.
- Added clean core-install and synthetic CPU training/resume checks, source provenance and file manifests.
- Corrected transfer labels and separated archived display from computation.

- Added the author-confirmed MIT license and software citation metadata for Zhiqi Cai.

The old directory and historical aggregate CSVs remain unchanged. This is an MIT-licensed public-release candidate, not a GitHub upload. Historical severity-specific bootstrap intervals remain archived-only; the new analysis reports pooled contrasts.
