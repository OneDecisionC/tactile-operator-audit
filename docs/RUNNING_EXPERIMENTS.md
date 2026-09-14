# New-run workflow — 0.2.0

The portable workflow reuses recovered historical kernels but creates independent new artifacts. It does not bypass or monkey-patch the frozen formal Stage-B registry gate. Dataset acquisition is documented in [DATA.md](DATA.md).

## Configuration

Copy a JSON example to a `*.local.json` and edit `data`, `device` and `output_root`. Configuration paths resolve from the repository root. Output roots must be children of `outputs/`.

Expected inputs:

```text
data/braille_letters_dataset/data/data_braille_letters_th1
data/STEMNIST Dataset/ProcessedSpikes/<class>/*.h5
data/cache/stemnist_icassp/
```

`cache --execute` builds the STEMNIST raw-event/packed-binary cache. Braille uses its original trusted serialized input. Incomplete caches are not deleted automatically.

## Profiles and RNG identity

| Dataset | Probe parameters | Optimizer | Epochs / batch | Clipping |
|---|---|---|---|---|
| Braille | RateSNN 47,067; TCN 45,595; Conv1D 47,171 | AdamW, lr .002, wd .0001 | 80 / 128 | norm 1 |
| STEMNIST | Conv-SNN 6,507; large TCN 514,851 | Adam, lr .002, wd .0001 | 50 / 32 | none |

Retain the final epoch; no validation-based checkpoint selection. STEMNIST supports clean training only in this workflow. Its large TCN is not the historical Stage-B tiny TCN.

Do not shorten `braille_conv1d_h83_l2_k5`: the ID participates in initialization hashing. Braille uses the `icassp_stage_b_formal_v1` RNG root; STEMNIST training uses `binary80_groupcv_v1`. Six Braille arms share arm-independent initialization, epoch permutation, augmentation gate, severity and operator roots. M/CF+M use the recovered matched-augmentation dataset logic.

For jitter/survivor evaluation, Braille uses the Stage-B root and STEMNIST the aligned Stage-A root. The recovered retention-v3 implementation intentionally uses the Stage-B sample root for both datasets. It is preserved rather than silently harmonized. Priorities exclude retention target, model, fold and split seed.

## Execution and selection

```bash
python examples/run_portable.py --config examples/paper_probes.local.json plan
python examples/run_portable.py --config examples/paper_probes.local.json cache --execute
python examples/run_portable.py --config examples/paper_probes.local.json train --probe braille_tcn_small_h64_b2_k3 --seed 42 --fold 1 --execute --allow-training
python examples/run_portable.py --config examples/paper_probes.local.json eval --probe braille_tcn_small_h64_b2_k3 --seed 42 --fold 1 --execute
```

Use `--arm matched_aug` to select a transfer arm. Omit selectors to run all jobs sequentially. No data download or training starts on import/installation. Training requires both execution flags.

The paper configuration has 75 jobs and 136 units per job: clean once, 14 jitter-related conditions over nine severity-realization cells, and three retention targets over three realizations. Total: 10,200 units. Six-arm transfer has 90 jobs and 4,140 units. Retention is not repeated for each jitter severity.

## Artifacts and resume

```text
outputs/portable/<configuration-hash>/
  run_manifest.json
  checkpoints/<job-hash>/{model.pt,history.csv,commit.json}
  evaluation/<job-hash>/<condition-hash>/{predictions.csv,commit.json}
```

Short hashes keep Windows paths manageable; manifests/CSVs retain readable identities. The run manifest includes data hashes/counts, source hashes and NumPy/SciPy/Torch/CUDA versions. Configuration changes, including evaluation changes, create a new identity: select conditions before training.

Identical commands skip verified committed artifacts. Tampering and interrupted files cause errors, not overwrites. Resume skips completed jobs/units; it does not restore a partially trained optimizer. Use one launcher per run root. Distributed scheduling/cross-device merging are unsupported.

## Recompute OOF statistics

```bash
python scripts/analyze_run.py --run outputs/portable/CONFIG_HASH --output outputs/analysis_NEW --execute
```

Complete all five folds and configured seeds/arms/conditions first. Analysis validates checksum/identity, unique sample coverage, consistent labels/folds, participant-disjoint STEMNIST folds and per-sample total occupancy equality for matched controls. Kernel tests separately check per-channel equality.

- `condition_summary.csv`: pooled accuracy, per-seed values, mean/sample SD; sample micro and STEMNIST participant macro. Retention targets are also included.
- `paired_ci.csv`: pooled PBJ-minus-CF+M, clean-minus-identity-loss, four-rule interactions/residuals and available augmentation comparisons.
- `report.md`: automatically formatted new values, never mixed with archived manuscript tables.
- `provenance.json`: prediction hashes, manifest hash and analysis settings.

The recovered bootstrap kernel uses 10,000 shared multinomial draws, seed 20260907. Braille resamples within classes; STEMNIST resamples participants and reports both cluster-ratio micro and macro estimates. Correctness is first averaged over corruption cells and split seeds. Do not treat repeated seeds/corruptions as independent samples. Intervals are pointwise, exclude refitting uncertainty and are not equivalence tests. One seed has no sample SD (blank).

This new report currently pools severities. Historical severity-specific intervals remain archived-only; exact historical checkpoint replay and original prediction-file formats are not supported here.

## Recompute and plot bin-width statistics

```bash
python scripts/audit_binwidth.py --dataset braille --data data --output outputs/binwidth_braille_NEW --execute
python scripts/audit_binwidth.py --dataset stemnist --data data/cache/stemnist_icassp --output outputs/binwidth_stemnist_NEW --execute
python scripts/plot_binwidth.py --input outputs/binwidth_braille_NEW/summary.csv --output outputs/figure_braille_NEW
```

The audit preserves 12.5/25/50-ms grids, J=50 ms, three realizations, integer STEMNIST ticks and the half-width final Braille bin. The 25-ms reconstruction must exactly match the clean recognition input. Ratios pool counts. Error bars are realization min/max, not CIs. No model is evaluated on a new grid. Plotting requires the `figures` extra.

Run `python -m pytest tests -q` for software tests. Training-path tests explicitly skip when training dependencies are absent. [VALIDATION.md](VALIDATION.md) records what was actually checked.
