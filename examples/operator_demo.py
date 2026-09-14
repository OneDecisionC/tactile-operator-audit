"""Display the operators on one artificial binary array; no data or training."""

import numpy as np

from icassp_jitter_operators import make_deconfounded_bundle
from injective_pbj_operator import OPERATOR_VERSION, make_injective_pbj_controls


def show_counts(name, array):
    """Print observed counts instead of assuming a conservation result."""
    counts = np.count_nonzero(array, axis=0)
    print(f"{name:28s} total={int(counts.sum()):2d} per_channel={counts.tolist()}")


def main():
    clean = np.zeros((12, 3), dtype=np.uint8)
    clean[[0, 1, 2, 5, 6, 10, 11], 0] = 1
    clean[[1, 3, 4, 7, 8, 9], 1] = 1
    clean[[0, 2, 5, 8, 11], 2] = 1

    radius_bins = 3
    seed = 20260909
    realization = 0
    bundle = make_deconfounded_bundle(
        clean, radius_bins, seed, realization=realization
    )
    injective = make_injective_pbj_controls(
        clean, radius_bins, seed, realization=realization
    )

    print(f"shape={clean.shape}, radius_bins={radius_bins}, seed={seed}, realization={realization}")
    print(f"injective implementation: {OPERATOR_VERSION}")
    show_counts("clean", clean)
    for name in ("pbj", "cf", "matched", "cf_matched"):
        show_counts(name, bundle[name])
    for name in ("ipbj_movement", "pbj_identity_loss", "ipbj_plus_identity_loss"):
        show_counts(name, injective[name])

    print("PBJ collision loss:", bundle["audit"]["pbj"]["binary_collision_loss"])
    print("Selected CF gates per pass:", bundle["audit"]["cf"]["selected_swaps_by_pass"])
    print("Repaired injective assignments:", injective["audit"]["repaired_assignments"])
    print("Uncovered PBJ targets:", injective["audit"]["uncovered_pbj_targets_after_bounded_matching"])
    print(
        "Combined identity control vs PBJ, differing cells:",
        injective["audit"]["movement_plus_identity_loss_hamming_cells_vs_pbj"],
    )


if __name__ == "__main__":
    main()
