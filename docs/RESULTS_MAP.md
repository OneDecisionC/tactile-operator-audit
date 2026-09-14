# Archived results and source mapping

The files under `results/` are curated aggregate results from completed historical analyses. No model was evaluated and no statistic was re-estimated while preparing this candidate. `results/PROVENANCE.json` records the source identifier, extraction method, precision, and units for each file.

## Main tables

| Paper item | Release input | What is stored |
|---|---|---|
| Table 1: occupancy-matched recognition | `results/main_tables/table1.csv` | Five rows transcribed at the v3 draft's published two-decimal precision. |
| Table 2: primary iPBJ controls | `results/main_tables/table2.csv` | Five rows transcribed at the v3 draft's published two-decimal precision. |
| Table 3: six-arm transfer | `results/main_tables/table3.csv` | Existing full-precision means and sample SDs selected from the archived transfer snapshot. |
| Transfer seed values | `results/transfer/by_seed.csv` | Existing per-seed aggregate accuracies, flattened from the same snapshot. |

The term "published-rounded" in provenance means values printed in the manuscript draft, not a claim that the manuscript has been accepted or published. Tables 1 and 2 are a convenience transcription, not independently recomputed results. Rounding can make the difference of displayed means differ by 0.01 from the displayed paired difference.

The table-rendering command is:

```console
python scripts/render_tables.py --output outputs/tables
```

It reads the three main-table CSVs and writes `tables.md`. It only formats archived values. It does not compute predictions, estimate means from raw trials, regenerate bootstrap intervals, or establish a new replication. The renderer was run successfully during release validation; see `VALIDATION.md`.

## Supporting aggregates

| Release file | Paper connection | Units and scope |
|---|---|---|
| `bootstrap/paired_ci.csv` | Table 1 CIs; Section 4.4 pooled and severity-specific contrasts | Accuracy columns are percent; difference and CI columns are percentage points. Read `estimand` and `primary_ci` per row. |
| `bootstrap/condition_accuracy_by_seed_pooled.csv` | Pooled per-seed condition accuracies | Percent; only historical `severity_ms=pooled` rows are included. |
| `bootstrap/paired_differences_by_seed_pooled.csv` | Pooled per-seed paired contrasts | Percentage points; only pooled rows are included. |
| `binwidth/summary.csv` | Figure 2 and Section 4.4 | Six dataset/bin-width aggregates, with named count, percentage, displacement, and min/max columns. |
| `survivor/factorial_sample_micro.csv` | Table 2 and the four-rule sensitivity | Raw accuracy mean/SD columns are fractions; drops, interactions, and residuals are percentage points. |
| `survivor/ranges_sample_micro.csv` | Section 4.2 cross-rule ranges | Accuracy min/max columns are fractions; accuracy range_pp, drop, interaction, and residual columns are percentage points. |
| `survivor/factorial_participant_macro.csv` | STEMNIST participant-macro sensitivity | Same mixed units as the sample-micro factorial file. |
| `survivor/ranges_participant_macro.csv` | STEMNIST participant-macro rule ranges | Same mixed units as the sample-micro range file. |
| `retention/sample_micro.csv` | Section 4.2 matched-retention calibration | Accuracy columns are fractions; drop_pp columns are percentage points. |
| `retention/participant_macro.csv` | STEMNIST matched-retention sensitivity | Participant-macro accuracy columns are fractions; drop_pp columns are percentage points. |

All paths in this table are relative to `results/`. Participant-macro files contain dataset/probe aggregates, not participant identifiers or per-person records. Seed files contain split seed numbers, not sample IDs.

Multiplying a fractional accuracy by 100 converts it to percent. Do not multiply a column already expressed in percentage points by 100 again. An across-seed SD and a bootstrap interval describe different sources of uncertainty.

## Code-to-paper mapping

| Paper component | Distributed code | Historical source or remaining boundary |
|---|---|---|
| PBJ, CF, M, and CF+M | `src/icassp_jitter_operators.py` | Count and movement operators; upstream sample-key lineage is required for historical draw replay. |
| iPBJ and the primary proposal-coupled control | `src/injective_pbj_operator.py` | Historical version string is `injective_pbj_controls_v2`; original directory used the name `injective_pbj_v1`. |
| Braille models and input handling | `src/run_braille_ablation_baselines.py`, `src/run_icassp_jitter_screening.py` | Match the documented th1 representation, crop, and historical checkpoint lineage. |
| STEMNIST models and input handling | `src/run_stemnist_robustness.py`, `src/run_icassp_jitter_screening.py` | The paper uses Stage-A SCNN/large-TCN checkpoints, not a tiny screening TCN. |
| Stage-B training and cache construction | `src/icassp_stage_b_runner.py`, `src/icassp_stage_b_operator_cache.py` | Portable runs do not recreate missing historical folds, sample keys, or weights. |
| Archived table display | `scripts/render_tables.py` | Formatting only; no statistical or model recomputation. |
| Portable command entry | `examples/run_portable.py` and `examples/paper_probes.json` | Current portable-new-run workflow; see the repository README for execution controls. |

The table renderer remains formatting-only. Version 0.2.0 separately adds survivor/retention evaluation in the portable launcher, recovered paired bootstrap in `tactile_audit.analysis`, bin-width recomputation in `scripts/audit_binwidth.py`, and plotting in `scripts/plot_binwidth.py`. `scripts/analyze_run.py` recomputes pooled new-run statistics from verified predictions. Historical severity-specific CIs and exact historical publication layout remain archived-only; no old CSV was replaced with synthetic test values.

## Historical provenance identifiers

The confidence intervals and bin-width rows were selected from `no_retrain_followup_20260907/integration/source_snapshot.json`. Their original source labels are `paired_ci/paired_ci_all.csv` and `binwidth/summary.csv`. The transfer snapshot records `matched_augmentation_v1/results/six_arm_transfer_by_seed.csv` as its numerical source.

The survivor aggregates come from `survivor_rule_all_probe_extension_v1`, result identifier `78ae45ab29ee1cf505147a0f`. The retention aggregates come from `retention_matched_loss_v3`, result identifier `304147266d436e2a280d9d09`. These are historical identifiers, not public URLs or independently reverified hashes.

The complete historical snapshot is intentionally absent: it also embeds old manuscript and bibliography text. Only the specified numerical sections were selected. No third-party paper PDF, original manuscript PDF, source-prediction inventory, checkpoint, corruption cache, per-sample prediction file, or participant-level difference table is included in these curated results.

## Figure reconstruction boundary

`results/binwidth/summary.csv` contains the aggregate coordinates and realization min/max values needed to redraw Figure 2. The historical plotting source was `no_retrain_followup_20260907/build_figures.py`, whose bin-width plot uses a base-2 logarithmic x-axis. The original script also reads an embedded manuscript snapshot, so it is not copied as a portable plotting entry point.

`scripts/plot_binwidth.py` renders the aggregate coordinates with a base-2 log axis and realization min/max bars without loading the old embedded manuscript. This is an independent layout, not a claim of pixel-identical paper reproduction. Figure 1 is a schematic, not an additional experiment.

