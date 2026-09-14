"""Render archived aggregate results; no recomputation, training, or inference.

Tables 1 and 2 are published-rounded transcriptions of the manuscript draft.
Table 3 contains existing archived means and sample standard deviations.
This command formats those values; it is not an independent replication.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "braille_ratesnn_h192_l3": "RateSNN-h192",
    "braille_tcn_small_h64_b2_k3": "TCN-h64",
    "braille_conv1d_h83_l2_k5": "Conv1D-h83",
    "stemnist_paper_scnn_clean": "paper-topology SCNN",
    "stemnist_tcn_large_clean": "TCN-h128",
}
ARMS = {
    "clean": "Clean-trained",
    "pbj_aug": "PBJ augmentation",
    "cf_aug": "CF augmentation",
    "prebin_aug": "Pre-bin augmentation",
    "matched_aug": "M augmentation",
    "cf_matched_aug": "CF+M augmentation",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pair(row: dict[str, str], mean: str, sd: str) -> str:
    return f"{float(row[mean]):.2f} +/- {float(row[sd]):.2f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render(results: Path) -> str:
    table1 = read_rows(results / "table1.csv")
    table2 = read_rows(results / "table2.csv")
    table3 = read_rows(results / "table3.csv")
    # Keep the stored paired differences: subtracting rounded means can be off by 0.01 pp.
    # The CIs here are already selected for Table 1, including STEMNIST sample-micro CIs.
    first = []
    for row in table1:
        first.append([
            row["dataset"].capitalize(),
            LABELS.get(row["probe_id"], row["probe_id"]),
            pair(row, "clean_mean_pct", "clean_sd_pct"),
            pair(row, "pbj_mean_pct", "pbj_sd_pct"),
            pair(row, "cfm_mean_pct", "cfm_sd_pct"),
            pair(row, "difference_mean_pp", "difference_sd_pp"),
            f'[{float(row["ci95_low_pp"]):.2f}, {float(row["ci95_high_pp"]):.2f}]',
        ])
    second = []
    for row in table2:
        second.append([
            row["dataset"].capitalize(),
            LABELS.get(row["probe_id"], row["probe_id"]),
            pair(row, "clean_mean_pct", "clean_sd_pct"),
            pair(row, "move_mean_pct", "move_sd_pct"),
            pair(row, "deletion_mean_pct", "deletion_sd_pct"),
            pair(row, "combined_mean_pct", "combined_sd_pct"),
            pair(row, "pbj_mean_pct", "pbj_sd_pct"),
            pair(row, "interaction_mean_pp", "interaction_sd_pp"),
            pair(row, "residual_mean_pp", "residual_sd_pp"),
        ])
    third = []
    for row in table3:
        third.append([
            ARMS.get(row["train_arm"], row["train_arm"]),
            *[
                pair(row, condition + "_mean_pct", condition + "_sd_pct")
                for condition in ("clean", "prebin", "pbj", "cf", "matched", "cf_matched")
            ],
        ])
    return "\n\n".join([
        "# Archived paper tables",
        "Rendered from curated aggregate CSVs. No model, metric, or confidence "
        "interval was recomputed. This report is not an independent replication.",
        "## Table 1. Occupancy-matched recognition",
        "Published-rounded transcription. Accuracy and seed SD are percent; "
        "paired differences and pointwise 95% CIs are percentage points. "
        "Table 1 uses sample-micro estimates, participant-clustered for STEMNIST.",
        markdown_table(
            ["Dataset", "Probe", "Clean", "PBJ", "CF+M", "PBJ - CF+M (pp)", "95% CI (pp)"],
            first,
        ),
        "## Table 2. Primary operational iPBJ controls",
        "Published-rounded transcription. Accuracy and seed SD are percent; "
        "interaction and PBJ-minus-combined residual are percentage points. "
        "The survivor rule is assignment_preserving.",
        markdown_table(
            ["Dataset", "Probe", "Clean", "Move only", "Source deletion",
             "Move + loss", "PBJ", "Interaction (pp)", "Residual (pp)"],
            second,
        ),
        "## Table 3. Braille TCN augmentation transfer",
        "Existing archived means and seed sample SDs, formatted to two decimals "
        "in percent. Clean-trained and all five augmentation arms are shown.",
        markdown_table(
            ["Train / Test", "Clean", "Pre-bin", "PBJ", "CF", "M", "CF+M"],
            third,
        ),
        "Provenance and units: results/PROVENANCE.json and docs/RESULTS_MAP.md.",
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render archived aggregate results; no recomputation, training, "
                    "inference, or bootstrap.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "main_tables",
        help="Directory containing the archived table1.csv, table2.csv, and table3.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "tables",
        help="Output directory for tables.md.",
    )
    args = parser.parse_args()
    document = render(args.results_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "tables.md"
    target.write_text(document, encoding="utf-8")
    print(f"Archived table report written to {target}")


if __name__ == "__main__":
    main()
