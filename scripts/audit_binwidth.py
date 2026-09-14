"""Recompute bin-width occupancy statistics from separately acquired datasets."""
import argparse
import csv
import json
from pathlib import Path

import run_icassp_jitter_screening as screen
import tactile_audit.binwidth as implementation
from tactile_audit.analysis import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["braille", "stemnist"], required=True)
    parser.add_argument("--data", type=Path, required=True, help="Braille project root containing braille_letters_dataset, or verified STEMNIST cache directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Plan only: operator audit at 12.5/25/50 ms, J=50 ms, three realizations. No training or inference. Add --execute.")
        return
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    bundle = screen.load_braille_bundle(args.data) if args.dataset == "braille" else screen.load_stemnist_bundle(args.data, verify_hash=True)
    rows = implementation.audit_bundle(bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"scope": "new operator-only audit; no cross-grid recognition experiment", "data_metadata": bundle.metadata,
                "source_sha256": file_sha(implementation.__file__), "summary_sha256": file_sha(args.output / "summary.csv"),
                "uncertainty": "min/max across three realizations, not confidence intervals"}
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
