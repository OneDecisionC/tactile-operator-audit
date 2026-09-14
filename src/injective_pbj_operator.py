"""Proposal-coupled injective controls for the frozen Stage-B PBJ operator.

The operator forms a four-cell input-level factorial control:

* clean: neither movement nor PBJ collision loss;
* ``ipbj_movement``: the exact PBJ proposals repaired to a bounded injection;
* ``pbj_identity_loss``: collision-lost source identities removed at source;
* ``ipbj_plus_identity_loss``: injective movement followed by deletion of
  those same identities.

The existing logical-OR PBJ is retained as a separate reference.  A bounded
injection cannot always both cover every PBJ target and keep the original
maximum displacement in dense boundary cases, so the implementation preserves
the bound and reports the remaining PBJ/control geometry explicitly.  No model
training is performed by this module.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

import icassp_jitter_operators as parent


OPERATOR_VERSION = "injective_pbj_controls_v2"

# Integer weights encode aggregate lexicographic priorities.  With T,N <= 80,
# the sum of every lower-priority term is strictly below one unit of the next.
_W_ACTUAL_DISPLACEMENT = np.int64(100)
_W_DISPLACEMENT_MISMATCH = np.int64(10_000)
_W_PROPOSAL_DISTANCE = np.int64(10_000_000)
_W_MOVED_STATUS = np.int64(100_000_000_000)
_W_NONEXACT_PROPOSAL = np.int64(100_000_000_000_000)
_FORBIDDEN = np.int64(8_000_000_000_000_000_000)


def _assignment_digest(records: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(OPERATOR_VERSION.encode("ascii"))
    for record in records:
        contiguous = np.ascontiguousarray(record, dtype="<i8")
        digest.update(struct.pack("<q", int(contiguous.shape[0])))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _bounded_assignment(
    source: np.ndarray,
    proposal: np.ndarray,
    radius: int,
    time_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return destination and exact-proposal masks for one channel.

    Channels without proposal collisions take the proposals directly.  For a
    collision channel, rectangular linear assignment chooses one distinct time
    bin per source while keeping all assignments inside the original PBJ radius.
    """
    source = np.asarray(source, dtype=np.int64)
    proposal = np.asarray(proposal, dtype=np.int64)
    if source.ndim != 1 or proposal.shape != source.shape:
        raise ValueError("source and proposal must be equal-length vectors")
    if source.size == 0:
        return source.copy(), np.ones(0, dtype=bool)
    if np.unique(proposal).size == proposal.size:
        return proposal.copy(), np.ones(source.size, dtype=bool)

    destinations = np.arange(time_steps, dtype=np.int64)[None, :]
    source_column = source[:, None]
    proposal_column = proposal[:, None]
    allowed = (
        (destinations >= np.maximum(0, source_column - radius))
        & (destinations <= np.minimum(time_steps - 1, source_column + radius))
    )
    nonexact = destinations != proposal_column
    pbj_moved = proposal_column != source_column
    assigned_moved = destinations != source_column
    moved_status_mismatch = assigned_moved != pbj_moved
    proposal_distance = np.abs(destinations - proposal_column)
    displacement_mismatch = np.abs(
        np.abs(destinations - source_column) - np.abs(proposal_column - source_column)
    )
    actual_displacement = np.abs(destinations - source_column)

    source_rank = np.arange(source.size, dtype=np.int64)[:, None]
    edge_tie = (
        (source_rank + 1) * np.int64(1009)
        + (destinations + 1) * np.int64(9176)
        + (source_rank + 1) * (destinations + 1) * np.int64(37)
    ) % np.int64(97)
    cost = (
        nonexact.astype(np.int64) * _W_NONEXACT_PROPOSAL
        + moved_status_mismatch.astype(np.int64) * _W_MOVED_STATUS
        + proposal_distance * _W_PROPOSAL_DISTANCE
        + displacement_mismatch * _W_DISPLACEMENT_MISMATCH
        + actual_displacement * _W_ACTUAL_DISPLACEMENT
        + edge_tie
    )
    cost[~allowed] = _FORBIDDEN
    rows, columns = linear_sum_assignment(cost)
    if rows.size != source.size or np.any(cost[rows, columns] == _FORBIDDEN):
        raise RuntimeError("bounded injective assignment is infeasible")
    assigned = np.empty(source.size, dtype=np.int64)
    assigned[rows] = columns.astype(np.int64, copy=False)
    exact = assigned == proposal

    return assigned, exact


def make_injective_pbj_controls(
    clean: np.ndarray,
    radius_bins: int,
    seed: int,
    *,
    realization: int = 0,
) -> dict[str, Any]:
    """Generate proposal-coupled movement-only and source-loss-only controls."""
    clean = parent._validate_binary_sample(clean)
    radius = parent._validate_radius(radius_bins)
    time_steps, channels = clean.shape
    normalized = parent._normalized_displacement(
        clean.shape, seed, realization, "pbj-displacement"
    )
    displacement = parent._integer_displacement(normalized, radius)

    movement = np.zeros_like(clean)
    identity_loss = clean.copy()
    movement_plus_loss = np.zeros_like(clean)
    proposed_pbj = np.zeros_like(clean)
    records: list[np.ndarray] = []
    lost_by_channel: list[list[int]] = []

    input_active = proposal_moved = injective_moved = boundary_clipped = 0
    proposal_abs_sum = injective_abs_sum = repair_abs_sum = 0
    proposal_abs_max = injective_abs_max = 0
    exact_proposal_count = collision_channels = uncovered_pbj_targets = 0

    for channel in range(channels):
        source = np.flatnonzero(clean[:, channel] > 0).astype(np.int64)
        if source.size == 0:
            lost_by_channel.append([])
            records.append(np.empty((0, 5), dtype=np.int64))
            continue
        raw_proposal = source + displacement[source, channel]
        proposal = np.clip(raw_proposal, 0, time_steps - 1).astype(np.int64)
        if np.unique(proposal).size != proposal.size:
            collision_channels += 1
        assigned, exact = _bounded_assignment(source, proposal, radius, time_steps)
        lost_mask = np.zeros(source.size, dtype=bool)
        for target in np.unique(proposal):
            group = np.flatnonzero(proposal == target)
            if group.size <= 1:
                continue
            # OR has no observable source identity.  This frozen operational
            # convention retains the proposer best preserved by the bounded
            # injection, then breaks ties by actual shift and source time.
            order = np.lexsort(
                (
                    source[group],
                    np.abs(assigned[group] - source[group]),
                    np.abs(assigned[group] - proposal[group]),
                    (assigned[group] != proposal[group]).astype(np.int64),
                )
            )
            survivor = int(group[order[0]])
            lost_mask[group] = True
            lost_mask[survivor] = False
        lost_source = source[lost_mask]

        movement[assigned, channel] = np.asarray(1, dtype=clean.dtype)
        identity_loss[lost_source, channel] = np.asarray(0, dtype=clean.dtype)
        movement_plus_loss[assigned[~lost_mask], channel] = np.asarray(1, dtype=clean.dtype)
        proposed_pbj[proposal, channel] = np.asarray(1, dtype=clean.dtype)
        lost_by_channel.append(lost_source.tolist())
        records.append(
            np.column_stack(
                (
                    np.full(source.size, channel, dtype=np.int64),
                    source,
                    raw_proposal,
                    proposal,
                    assigned,
                )
            )
        )

        clipped_shift = proposal - source
        injective_shift = assigned - source
        input_active += int(source.size)
        proposal_moved += int(np.count_nonzero(clipped_shift))
        injective_moved += int(np.count_nonzero(injective_shift))
        boundary_clipped += int(np.count_nonzero(raw_proposal != proposal))
        proposal_abs_sum += int(np.abs(clipped_shift).sum())
        injective_abs_sum += int(np.abs(injective_shift).sum())
        repair_abs_sum += int(np.abs(assigned - proposal).sum())
        proposal_abs_max = max(proposal_abs_max, int(np.abs(clipped_shift).max()))
        injective_abs_max = max(injective_abs_max, int(np.abs(injective_shift).max()))
        exact_proposal_count += int(np.count_nonzero(exact))
        uncovered_pbj_targets += int(
            np.unique(proposal).size - np.unique(proposal[exact]).size
        )

    reference_pbj, reference_audit = parent.nested_pbj(
        clean, radius, seed, realization=realization, return_audit=True
    )
    if not np.array_equal(proposed_pbj, reference_pbj):
        raise AssertionError("proposal reconstruction differs from parent PBJ")
    clean_counts = np.count_nonzero(clean > 0, axis=0)
    pbj_counts = np.count_nonzero(reference_pbj > 0, axis=0)
    movement_counts = np.count_nonzero(movement > 0, axis=0)
    loss_counts = np.count_nonzero(identity_loss > 0, axis=0)
    combined_counts = np.count_nonzero(movement_plus_loss > 0, axis=0)
    if not np.array_equal(movement_counts, clean_counts):
        raise AssertionError("injective movement failed per-channel count conservation")
    if not np.array_equal(loss_counts, pbj_counts):
        raise AssertionError("identity loss failed per-channel PBJ count matching")
    if not np.array_equal(combined_counts, pbj_counts):
        raise AssertionError("combined control failed per-channel PBJ count matching")
    if injective_abs_max > radius:
        raise AssertionError("injective assignment exceeded the PBJ radius")

    lost = input_active - int(pbj_counts.sum())
    if lost != int(reference_audit["binary_collision_loss"]):
        raise AssertionError("identity-loss count differs from PBJ collision loss")
    denominator = max(input_active, 1)
    audit = {
        "operator": OPERATOR_VERSION,
        "seed": int(seed),
        "realization": int(realization),
        "radius_bins": radius,
        "normalized_displacement_digest": parent._array_digest(normalized),
        "assignment_digest": _assignment_digest(records),
        "input_active_cells": input_active,
        "pbj_output_active_cells": int(np.count_nonzero(reference_pbj)),
        "pbj_collision_lost_identities": lost,
        "collision_channels": collision_channels,
        "exact_proposal_assignments": exact_proposal_count,
        "repaired_assignments": input_active - exact_proposal_count,
        "uncovered_pbj_targets_after_bounded_matching": uncovered_pbj_targets,
        "proposal_moved_cells": proposal_moved,
        "injective_moved_cells": injective_moved,
        "proposal_moved_fraction": float(proposal_moved / denominator),
        "injective_moved_fraction": float(injective_moved / denominator),
        "boundary_clipped_active_cells": boundary_clipped,
        "proposal_mean_abs_displacement_bins": float(proposal_abs_sum / denominator),
        "injective_mean_abs_displacement_bins": float(injective_abs_sum / denominator),
        "mean_abs_repair_from_proposal_bins": float(repair_abs_sum / denominator),
        "proposal_abs_displacement_sum_bins": proposal_abs_sum,
        "injective_abs_displacement_sum_bins": injective_abs_sum,
        "repair_abs_displacement_sum_bins": repair_abs_sum,
        "proposal_max_abs_displacement_bins": proposal_abs_max,
        "injective_max_abs_displacement_bins": injective_abs_max,
        "lost_source_time_indices_by_channel": lost_by_channel,
        "exact_per_channel_movement_count_conservation": True,
        "exact_per_channel_identity_loss_pbj_count_match": True,
        "exact_per_channel_combined_pbj_count_match": True,
        "movement_plus_identity_loss_equals_pbj": bool(
            np.array_equal(movement_plus_loss, reference_pbj)
        ),
        "movement_plus_identity_loss_hamming_cells_vs_pbj": int(
            np.count_nonzero(movement_plus_loss != reference_pbj)
        ),
        "parent_pbj_audit": reference_audit,
    }
    return {
        "ipbj_movement": movement,
        "pbj_identity_loss": identity_loss,
        "ipbj_plus_identity_loss": movement_plus_loss,
        "pbj": reference_pbj,
        "audit": audit,
    }
