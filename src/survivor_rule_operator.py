"""Survivor-rule sensitivity controls for proposal-coupled injective PBJ.

This module leaves the frozen 2026-08-19 implementation untouched.  It calls
that implementation's bounded assignment exactly once per channel, reuses the
resulting movement, and changes only which source identity is retained inside
each many-to-one PBJ proposal group.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np

import icassp_jitter_operators as parent
import injective_pbj_operator as frozen


OPERATOR_VERSION = "survivor_rule_sensitivity_v4"
RANDOM_PRIORITY_NAMESPACE = "icassp-survivor-priority-v4"

CURRENT_RULE = "assignment_preserving"
ALTERNATIVE_RULES = (
    "fixed_index",
    "minimum_proposed_displacement",
    "fixed_random_priority",
)
ALL_RULES = (CURRENT_RULE, *ALTERNATIVE_RULES)

CONDITIONS_BY_RULE = {
    "fixed_index": ("fixed_index_loss", "fixed_index_combined"),
    "minimum_proposed_displacement": (
        "minimum_proposed_displacement_loss",
        "minimum_proposed_displacement_combined",
    ),
    "fixed_random_priority": (
        "fixed_random_priority_loss",
        "fixed_random_priority_combined",
    ),
}
ALTERNATIVE_CONDITIONS = tuple(
    condition
    for rule in ALTERNATIVE_RULES
    for condition in CONDITIONS_BY_RULE[rule]
)


def _digest_int_records(namespace: str, records: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii"))
    for record in records:
        contiguous = np.ascontiguousarray(record, dtype="<i8")
        digest.update(struct.pack("<q", int(contiguous.shape[0])))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _fixed_priority(priority_seed: int, channel: int, source_time: int) -> int:
    payload = (
        f"{RANDOM_PRIORITY_NAMESPACE}|{int(priority_seed)}|"
        f"{int(channel)}|{int(source_time)}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _survivor_local_index(
    rule: str,
    source: np.ndarray,
    proposal: np.ndarray,
    assigned: np.ndarray,
    group: np.ndarray,
    *,
    priority_seed: int,
    channel: int,
) -> int:
    """Return an index into ``source`` for one colliding proposal group."""
    if group.size < 2:
        raise ValueError("survivor selection requires a collision group")
    if rule == CURRENT_RULE:
        # Bit-for-bit equivalent to the frozen 2026-08-19 convention.
        order = np.lexsort(
            (
                source[group],
                np.abs(assigned[group] - source[group]),
                np.abs(assigned[group] - proposal[group]),
                (assigned[group] != proposal[group]).astype(np.int64),
            )
        )
    elif rule == "fixed_index":
        order = np.argsort(source[group], kind="stable")
    elif rule == "minimum_proposed_displacement":
        # Primary key: |proposal-source|.  Source time is the deterministic tie.
        order = np.lexsort(
            (
                source[group],
                np.abs(proposal[group] - source[group]),
            )
        )
    elif rule == "fixed_random_priority":
        # This priority excludes model, severity, realization, proposal and
        # assignment.  Therefore it is fixed for a source identity.
        priority = np.asarray(
            [
                _fixed_priority(priority_seed, channel, int(source[index]))
                for index in group
            ],
            dtype=np.uint64,
        )
        order = np.lexsort((source[group], priority))
    else:
        raise ValueError(f"unknown survivor rule: {rule}")
    return int(group[int(order[0])])


def make_survivor_rule_controls(
    clean: np.ndarray,
    radius_bins: int,
    seed: int,
    *,
    realization: int = 0,
    priority_seed: int | None = None,
) -> dict[str, Any]:
    """Generate current and alternative loss/combined controls.

    ``priority_seed`` identifies the immutable sample for the fixed-random
    rule.  It defaults to the model-independent parent PBJ sample seed.
    """
    clean = parent._validate_binary_sample(clean)
    radius = parent._validate_radius(radius_bins)
    time_steps, channels = clean.shape
    priority_seed = int(seed if priority_seed is None else priority_seed)

    normalized = parent._normalized_displacement(
        clean.shape, seed, realization, "pbj-displacement"
    )
    displacement = parent._integer_displacement(normalized, radius)

    movement = np.zeros_like(clean)
    proposed_pbj = np.zeros_like(clean)
    loss_by_rule = {rule: clean.copy() for rule in ALL_RULES}
    combined_by_rule = {rule: np.zeros_like(clean) for rule in ALL_RULES}
    assignment_records: list[np.ndarray] = []
    lost_records: dict[str, list[np.ndarray]] = {rule: [] for rule in ALL_RULES}

    input_active = 0
    proposal_moved = 0
    injective_moved = 0
    boundary_clipped = 0
    exact_proposal_count = 0
    collision_channels = 0
    collision_groups = 0

    for channel in range(channels):
        source = np.flatnonzero(clean[:, channel] > 0).astype(np.int64)
        if source.size == 0:
            assignment_records.append(np.empty((0, 5), dtype=np.int64))
            for rule in ALL_RULES:
                lost_records[rule].append(np.empty((0, 2), dtype=np.int64))
            continue

        raw_proposal = source + displacement[source, channel]
        proposal = np.clip(raw_proposal, 0, time_steps - 1).astype(np.int64)
        assigned, exact = frozen._bounded_assignment(
            source, proposal, radius, time_steps
        )
        if np.unique(proposal).size != proposal.size:
            collision_channels += 1

        movement[assigned, channel] = np.asarray(1, dtype=clean.dtype)
        proposed_pbj[proposal, channel] = np.asarray(1, dtype=clean.dtype)

        masks: dict[str, np.ndarray] = {
            rule: np.zeros(source.size, dtype=bool) for rule in ALL_RULES
        }
        for target in np.unique(proposal):
            group = np.flatnonzero(proposal == target)
            if group.size <= 1:
                continue
            collision_groups += 1
            for rule in ALL_RULES:
                survivor = _survivor_local_index(
                    rule,
                    source,
                    proposal,
                    assigned,
                    group,
                    priority_seed=priority_seed,
                    channel=channel,
                )
                masks[rule][group] = True
                masks[rule][survivor] = False

        for rule in ALL_RULES:
            lost_source = source[masks[rule]]
            loss_by_rule[rule][lost_source, channel] = np.asarray(
                0, dtype=clean.dtype
            )
            combined_by_rule[rule][assigned[~masks[rule]], channel] = np.asarray(
                1, dtype=clean.dtype
            )
            lost_records[rule].append(
                np.column_stack(
                    (
                        np.full(lost_source.size, channel, dtype=np.int64),
                        lost_source,
                    )
                )
            )

        assignment_records.append(
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
        input_active += int(source.size)
        proposal_moved += int(np.count_nonzero(clipped_shift))
        injective_moved += int(np.count_nonzero(assigned - source))
        boundary_clipped += int(np.count_nonzero(raw_proposal != proposal))
        exact_proposal_count += int(np.count_nonzero(exact))

    reference_pbj, parent_audit = parent.nested_pbj(
        clean, radius, seed, realization=realization, return_audit=True
    )
    if not np.array_equal(reference_pbj, proposed_pbj):
        raise AssertionError("proposal reconstruction differs from parent PBJ")

    clean_counts = np.count_nonzero(clean > 0, axis=0)
    pbj_counts = np.count_nonzero(reference_pbj > 0, axis=0)
    movement_counts = np.count_nonzero(movement > 0, axis=0)
    if not np.array_equal(movement_counts, clean_counts):
        raise AssertionError("movement failed per-channel count conservation")

    lost_expected = int(clean_counts.sum() - pbj_counts.sum())
    rule_audit: dict[str, Any] = {}
    for rule in ALL_RULES:
        loss_counts = np.count_nonzero(loss_by_rule[rule] > 0, axis=0)
        combined_counts = np.count_nonzero(combined_by_rule[rule] > 0, axis=0)
        if not np.array_equal(loss_counts, pbj_counts):
            raise AssertionError(f"{rule} loss does not match PBJ per-channel counts")
        if not np.array_equal(combined_counts, pbj_counts):
            raise AssertionError(
                f"{rule} combined does not match PBJ per-channel counts"
            )
        lost_actual = int(clean_counts.sum() - loss_counts.sum())
        if lost_actual != lost_expected:
            raise AssertionError(f"{rule} lost-identity count mismatch")
        rule_audit[rule] = {
            "lost_identities": lost_actual,
            "lost_identity_digest": _digest_int_records(
                f"{OPERATOR_VERSION}:{rule}", lost_records[rule]
            ),
            "combined_equals_pbj": bool(
                np.array_equal(combined_by_rule[rule], reference_pbj)
            ),
            "combined_hamming_cells_vs_pbj": int(
                np.count_nonzero(combined_by_rule[rule] != reference_pbj)
            ),
        }

    denominator = max(input_active, 1)
    audit = {
        "operator": OPERATOR_VERSION,
        "frozen_assignment_operator": frozen.OPERATOR_VERSION,
        "seed": int(seed),
        "priority_seed": priority_seed,
        "realization": int(realization),
        "radius_bins": radius,
        "normalized_displacement_digest": parent._array_digest(normalized),
        "assignment_digest": frozen._assignment_digest(assignment_records),
        "input_active_cells": input_active,
        "pbj_output_active_cells": int(pbj_counts.sum()),
        "pbj_collision_lost_identities": lost_expected,
        "collision_channels": collision_channels,
        "collision_groups": collision_groups,
        "proposal_moved_cells": proposal_moved,
        "injective_moved_cells": injective_moved,
        "proposal_moved_fraction": float(proposal_moved / denominator),
        "injective_moved_fraction": float(injective_moved / denominator),
        "boundary_clipped_active_cells": boundary_clipped,
        "exact_proposal_assignments": exact_proposal_count,
        "rule_audit": rule_audit,
        "parent_pbj_audit": parent_audit,
    }

    output: dict[str, Any] = {
        "ipbj_movement": movement,
        "pbj": reference_pbj,
        "assignment_preserving_loss": loss_by_rule[CURRENT_RULE],
        "assignment_preserving_combined": combined_by_rule[CURRENT_RULE],
        "audit": audit,
    }
    for rule in ALTERNATIVE_RULES:
        loss_name, combined_name = CONDITIONS_BY_RULE[rule]
        output[loss_name] = loss_by_rule[rule]
        output[combined_name] = combined_by_rule[rule]
    return output

