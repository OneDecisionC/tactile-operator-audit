# Tactile Operator Audit — 0.2.0

Research code for **Occupancy-Matched Auditing of Temporal Jitter in Binary-Binned Tactile Event Representations**, a manuscript prepared for ICASSP 2027. Conference acceptance is not claimed.

[Experiments](docs/RUNNING_EXPERIMENTS.md) | [Protocol](docs/EXPERIMENT_PROTOCOL.md) | [Operators](docs/OPERATORS.md) | [Data](docs/DATA.md) | [Validation](docs/VALIDATION.md) | [Changes](CHANGELOG.md)

## Where to start

- **Checking the paper's numbers?** Start with `results/main_tables/` and [the results map](docs/RESULTS_MAP.md). `scripts/render_tables.py` puts the archived values into readable tables; you don't need to train a model for this.
- **Reading the method?** [The operator notes](docs/OPERATORS.md) explain how the code maps to the paper. In particular, equal active-cell counts do not mean two perturbations move or delete the same cells.
- **Running a new experiment?** Start with `examples/run_portable.py` and the commands below. Run `plan` first: the full configurations are much larger than a quick test.

## Included

- Five paper probes with distinct Braille Stage-B and STEMNIST Stage-A training profiles.
- Six Braille TCN training arms: clean, PBJ, CF, pre-bin, matched deletion M, and CF+M.
- Occupancy-matched evaluation, primary iPBJ controls, all four survivor rules, and exact 95/90/85% retention controls.
- Validated out-of-fold aggregation, paired bootstrap, seed summaries and generated analysis reports.
- Operator-only bin-width recomputation and PNG/SVG plotting.
- Historical aggregate CSVs, clearly separated from new computations.

Datasets, checkpoints, participant-level predictions, credentials and private registries are **not distributed**. These synthetic perturbations are not calibrated physical noise or identified causal mechanisms. New runs do not certify historical checkpoint replay.

## Install and test

Use Python 3.11 in an isolated environment without system-site packages:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest tests -q
.venv/bin/python examples/operator_demo.py
```

On Windows, use `py -3.11 -m venv .venv` and `.\.venv\Scripts\python.exe`. Subsequent `python` commands refer to your environment's interpreter.

Training additionally requires `python -m pip install -e ".[training]"`. See [environment notes](docs/ENVIRONMENT.md) for CUDA limitations. Clean core installation and synthetic CPU integration tests are checked; the entire GPU dependency stack has not been clean-installed or experimentally reproduced.

## Plan before computing

```bash
python examples/run_portable.py --config examples/paper_probes.json plan
python examples/run_portable.py --config examples/braille_transfer.json plan
```

The paper plan has **75 training jobs and 10,200 evaluation units** including sensitivity controls. Six-arm transfer has **90 jobs and 4,140 units**. These are substantial experiments, not smoke tests. Planning does not open datasets or import Torch.

Copy a configuration to `*.local.json`, set data/output paths, then:

```bash
python examples/run_portable.py --config examples/paper_probes.local.json cache --execute
python examples/run_portable.py --config examples/paper_probes.local.json train --execute --allow-training
python examples/run_portable.py --config examples/paper_probes.local.json eval --execute
python scripts/analyze_run.py --run outputs/portable/CONFIG_HASH --output outputs/analysis_NEW --execute
```

Replace `CONFIG_HASH` with the directory printed by `plan`. Analysis verifies committed hashes, complete five-fold sample coverage, participant grouping and matched occupancy before generating `condition_summary.csv`, `paired_ci.csv`, `report.md` and provenance. Bootstrap intervals are pointwise and conditional on frozen OOF models, not refitting uncertainty. [Full instructions](docs/RUNNING_EXPERIMENTS.md) explain selectors, resume, RNG identities and limitations.

## Archived display versus recomputation

```bash
python scripts/render_tables.py --output outputs/archived_tables
python -m pip install -e ".[figures]"
python scripts/plot_binwidth.py --output outputs/archived_binwidth_NEW
```

These commands display historical measurements; they do not reproduce them. Actual bin-width computation uses `scripts/audit_binwidth.py`. See [results mapping](docs/RESULTS_MAP.md).

## Release status

Run `python scripts/verify_release.py` to verify the file manifest. Distribute only files in `RELEASE_FILES.json`; do not include local environments, outputs or build products.

This is a **public-release candidate**, not a completed historical replication or a GitHub upload. [Validation](docs/VALIDATION.md), [source provenance](docs/SOURCE_PROVENANCE.json) and [licensing](docs/LICENSING.md) define its scope.

## Author, citation and license

Author: **Zhiqi Cai**. This software is released under the [MIT License](LICENSE), Copyright (c) 2026 Zhiqi Cai. Dataset and dependency licenses remain separate.

Please cite **Zhiqi Cai. Tactile Operator Audit, version 0.2.0.** Machine-readable software citation metadata is in [CITATION.cff](CITATION.cff). Add the exact public repository URL and commit/tag to citations once published; no repository URL or DOI has been assigned here.
