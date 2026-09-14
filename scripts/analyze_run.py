"""Recompute aggregates and paired confidence intervals from committed new-run predictions."""
import argparse
import csv
import json
from pathlib import Path

from tactile_audit.analysis import analyze_run, file_sha
import tactile_audit.analysis as analysis_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.resamples < 2:
        parser.error("--resamples must be at least 2")
    if not args.execute:
        print("Plan only: validate complete five-fold OOF predictions, recompute pooled seed summaries and paired bootstrap. Add --execute.")
        return
    if args.output.exists():
        raise FileExistsError("Use a new output directory; analysis never overwrites archived or previous results")
    # Validate the predictions first, so a failed check doesn't leave an empty report folder.
    conditions, contrasts, sources = analyze_run(args.run.resolve(), resamples=args.resamples, bootstrap_seed=args.bootstrap_seed)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("condition_summary.csv", conditions), ("paired_ci.csv", contrasts)):
        if rows:
            with (args.output / name).open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    # Save the input hashes alongside the tables so each report can be traced back to its run.
    metadata = {"scope": "new-run pooled analysis; not historical checkpoint replay or refitting uncertainty",
                "run_manifest_sha256": file_sha(args.run / "run_manifest.json"),
                "analysis_source_sha256": file_sha(Path(analysis_module.__file__)),
                "source_predictions": sources, "resamples": args.resamples, "bootstrap_seed": args.bootstrap_seed,
                "intervals": "pointwise; conditional on frozen OOF models; no seed/realization pseudo-replication"}
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    lines = ["# Recomputed new-run results", "", "Generated from verified OOF predictions, not archived manuscript values.",
             "Pointwise intervals condition on frozen models; they exclude refitting uncertainty. Seed SD is blank for a single seed.",
             "", "## Condition accuracy", "", "| Probe | Arm | Condition | Estimand | Mean (%) | Seed SD (%) |",
             "|---|---|---|---|---:|---:|"]
    for row in conditions:
        sd = "" if row["sample_sd_pct"] is None else f'{row["sample_sd_pct"]:.4f}'
        lines.append(f'| {row["probe_id"]} | {row["train_arm"]} | {row["condition"]} | {row["estimand"]} | {row["mean_pct"]:.4f} | {sd} |')
    lines.extend(["", "## Paired contrasts", "", "| Probe | Arm | Contrast | Estimand | Difference (pp) | 95% CI (pp) |", "|---|---|---|---|---:|---|"])
    for row in contrasts:
        lines.append(f'| {row["probe_id"]} | {row["train_arm"]} | {row["contrast"]} | {row["estimand"]} | {row["difference_pp"]:.4f} | [{row["ci95_low_pp"]:.4f}, {row["ci95_high_pp"]:.4f}] |')
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
