# Protocol summary

`paper_protocol.json` records the settings reported in the v3 manuscript and the historical checkpoint audit. It is a research-protocol reference, not an executable training configuration.

The paper's STEMNIST models are the Stage-A SCNN with 6,507 parameters and TCN-h128-l3 with 514,851 parameters. Their optimizer is Adam, with no gradient clipping. Braille uses the Stage-B checkpoints, AdamW, and global-norm clipping at 1.0. Do not substitute a smaller Stage-B STEMNIST screening model for either paper probe.

Historical hashes and software versions are copied from prior records; they were not recomputed or verified while preparing this candidate. The file contains no machine-specific filesystem paths, dataset contents, fold-membership lists, or checkpoint weights.

For current portable execution, use the separate configuration and command documented in the repository README: `examples/paper_probes.json` and `examples/run_portable.py`. A portable new run is not an exact replay of the historical paper experiments.

The experiment definition is in [EXPERIMENT_PROTOCOL.md](../docs/EXPERIMENT_PROTOCOL.md). The archived numbers and their provenance are described in [RESULTS_MAP.md](../docs/RESULTS_MAP.md).

