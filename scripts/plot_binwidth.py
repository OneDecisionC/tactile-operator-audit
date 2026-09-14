"""Plot archived or newly computed bin-width aggregates; no statistical recomputation."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parents[1] / "results/binwidth/summary.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new figure output directory")
    # This also runs on training servers without a display.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.3), layout="constrained")
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = sorted((row for row in rows if row["dataset"] == dataset), key=lambda row: float(row["bin_width_ms"]))
        widths = np.asarray([float(row["bin_width_ms"]) for row in selected])
        occupancy = [float(row["clean_occupancy_per_event_pct"]) for row in selected]
        retention = np.asarray([float(row["retention_pct"]) for row in selected])
        low = np.asarray([float(row["retention_pct_realization_min"]) for row in selected])
        high = np.asarray([float(row["retention_pct_realization_max"]) for row in selected])
        axes[0].plot(widths, occupancy, "o-", label=dataset)
        # These bars span the three realizations; they are not confidence intervals.
        axes[1].errorbar(widths, retention, yerr=[retention - low, high - retention], fmt="o-", capsize=3, label=dataset)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks([12.5, 25, 50], ["12.5", "25", "50"])
        axis.set_xlabel("Bin width (ms)")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Clean occupancy / raw events (%)")
    axes[1].set_ylabel("PBJ active retention (%)")
    axes[0].set_title("Representation occupancy")
    axes[1].set_title("J = 50 ms; realization min/max")
    args.output.mkdir(parents=True, exist_ok=False)
    figure.savefig(args.output / "binwidth.png", dpi=180)
    figure.savefig(args.output / "binwidth.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
