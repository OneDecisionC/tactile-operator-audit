# Environment and dependency scope

## Minimal operator environment

The release candidate targets Python 3.11. Core dependencies are pinned to NumPy 1.26.4 and SciPy 1.13.1. NumPy 1.26.4 appears in the original STEMNIST run metadata; SciPy 1.13.1 is the recorded assignment-solver version in the implementation audit.

```bash
python -m pip install -e .
```

No Torch import is required by the `tactile_audit` public facade. The archived-table renderer uses only the Python standard library and can be used without installing this package.

## Training and cache dependencies

The optional `training` extra records Torch 2.11.0 and snnTorch 0.9.4, with h5py for reading the STEMNIST source data. An exact historical h5py version was not established, so its requirement is a compatibility range, not a claimed historical pin.

```bash
python -m pip install -e ".[training]"
```

The original Stage-B environment assertions specify Python 3.11.14, Torch `2.11.0+cu128`, snnTorch 0.9.4, and an RTX 5060 Ti. These are recorded historical constraints, not a requirement that readers buy that GPU. Platform-specific Torch wheels, CUDA runtime, GPU driver, deterministic-kernel availability, and installation sources must be selected for the reader's system.

The portable launcher is a new-run pathway. It does not turn another machine into the original execution environment and does not certify bitwise identity. The original formal runner retains its historical assertions rather than silently weakening them.

## Not a complete lockfile

This package does not reconstruct the entire original conda environment. The clean core environment was installed and tested; the full GPU dependency stack, wheel hashes, drivers and platform libraries are not a historical lockfile. Actual checks and limits are recorded in [VALIDATION.md](VALIDATION.md).

Before a public release, record the actually checked operating system, Python, package versions, CPU/GPU, driver, and CUDA runtime in a release note. Keep environment validation separate from scientific reproduction and from any comparison of new results with the archived paper values.

## Determinism is more than setting a seed

Keep the original per-sample random namespaces, severity coupling, split rules, initialization and epoch seeds, final-checkpoint rule, and solver version. Preserve dataset-specific training lineage: the paper's STEMNIST Stage-A path uses Adam without gradient clipping, whereas Braille Stage B uses AdamW with clipping. Equal seed integers do not imply equivalent experiments if these rules change.
