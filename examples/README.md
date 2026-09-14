# Example configurations

All paths inside these files are relative to the repository root. No dataset,
checkpoint, historical result, private registry, or private workstation path is
included here.

- `paper_probes.json`: five correctly sized paper probes, clean training,
  three split seeds, five folds, and 18 evaluation conditions including sensitivity controls.
- `braille_transfer.json`: the Braille small TCN with clean/PBJ/CF/pre-bin
  plus matched/CF+matched training arms (six total). STEMNIST Stage-A profiles are not used for augmentation.
- `run_portable.py`: explicit new-run orchestration. Start with
  `python examples/run_portable.py --config examples/paper_probes.json plan`.

The configuration records the optimizer separately for each training lineage.
Braille Stage-B uses AdamW with gradient clipping; STEMNIST Stage-A uses Adam
without gradient clipping. Retain that distinction when editing examples.

`--probe`, `--arm`, `--seed`, and `--fold` select existing jobs.
They do not alter the original fold construction. To intentionally change
epochs, model sizes or other settings, edit a separately named configuration.
Its complete configuration hash will place outputs in a separate new-run
directory. Expected parameter counts must match the selected model architecture.

Plans, demos and synthetic software checks are recorded in `docs/VALIDATION.md`; a full real-data experiment has not been rerun for this release.
See `docs/RUNNING_EXPERIMENTS.md` for provenance, limitations, data layout,
training authorization flags, resume semantics and output schemas.
