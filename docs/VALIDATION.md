# Release validation — 0.2.0

Checked on 2026-09-14, Windows x64, Python 3.11.14. These are software checks, not a rerun of the paper.

## Clean core installation

A new venv without system-site packages installed `.[test]` using normal dependency resolution and isolated build dependencies. NumPy 1.26.4, SciPy 1.13.1, pytest 8.4.2; `pip check` passed. The host's existing SOCKS configuration first blocked installation. Removing proxy variables/user pip configuration only for that child process allowed installation from official PyPI; machine settings were not changed.

## Training smoke environment

A separate existing environment supplies Torch 2.11.0+cu128 and snnTorch 0.9.4. Tests explicitly use the new release source, synthetic data and CPU execution. They exercise tiny Braille Stage-B and STEMNIST Stage-A TCN training, checkpoint save/load, evaluation, completed-job resume and tamper rejection. This is not a clean installation of the entire GPU stack.

## Checks

- Recovered iPBJ, survivor-v4 and retention-v3 test suites.
- Six-arm configuration, historical Conv1D RNG identity, all evaluation conditions and occupancy invariants.
- Bootstrap agreement with synthetic golden outputs from the original local historical functions.
- Distinct micro/macro estimands, paired shared draws and deterministic replay.
- Complete synthetic five-fold OOF analysis; rejection of missing units, duplicates and modified artifacts.
- Bin endpoints, the partial last bin and integer STEMNIST ticks.
- Tiny CPU training/evaluation/resume when training dependencies exist.

Final counts and packaging checks are in `RELEASE_CHECKS.json`. `python scripts/verify_release.py` checks the delivered manifest.

## Limits

After author confirmation, the MIT release wheel was rebuilt with the existing validation environment (no build isolation). Its author, SPDX license expression and bundled LICENSE bytes were checked. CITATION.cff was parsed as YAML and its author and license checked; formal CFF schema validation was not performed. No scientific code changed during this licensing step.

No full training, real-data inference, historical checkpoint replay or full historical experiment rerun was performed. Linux/macOS, clean GPU-stack installation, GPU numerical equivalence and distributed execution were not validated. Historical CSV bytes are unchanged. Release-check figures display archived aggregates, not new experimental outcomes. Sole author Zhiqi Cai confirmed the MIT license. Public repository URL and exact commit/tag remain to be added when published.
