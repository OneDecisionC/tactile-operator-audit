"""Model-independent active-retention calibration operator.

The operator only deletes clean active entries.  It first fixes the requested
global retained count exactly, apportions deletions across channels with a
capped D'Hondt/Jefferson seat prefix, and then deletes a frozen SHA-256
priority prefix within every channel.  The priority deliberately excludes the
retention target, model, checkpoint and fold, so the 95/90/85 percent deletion
sets are nested for a fixed dataset/sample/realization.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


PRIORITY_NAMESPACE = "icassp-retention-matched-loss-v2"
DEFAULT_RETENTION_PERCENTAGES = (95, 90, 85)


class RetentionOperatorError(ValueError):
    """Raised when an operator input violates the frozen protocol domain."""


@dataclass(frozen=True)
class _DhondtCandidate:
    """One exact rational D'Hondt quotient in a max-priority min-heap."""

    numerator: int
    divisor: int
    channel: int

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _DhondtCandidate):
            return NotImplemented
        left = self.numerator * other.divisor
        right = other.numerator * self.divisor
        if left != right:
            # heapq pops the "smallest" item, so larger exact quotient wins.
            return left > right
        # Frozen tie rule: smaller channel index wins.
        return self.channel < other.channel


def _as_nonnegative_counts(channel_counts: Sequence[int] | np.ndarray) -> list[int]:
    values = np.asarray(channel_counts)
    if values.ndim != 1:
        raise RetentionOperatorError("channel_counts must be one-dimensional")
    if values.dtype.kind not in "biu":
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise RetentionOperatorError("channel_counts must contain integers")
    result = [int(value) for value in values.tolist()]
    if any(value < 0 for value in result):
        raise RetentionOperatorError("channel_counts cannot be negative")
    return result


def retained_target_count(active_count: int, retention_percent: int) -> int:
    """Return K=floor(retention_percent*N/100 + 0.5), exactly in integers."""

    n = int(active_count)
    percent = int(retention_percent)
    if n < 0:
        raise RetentionOperatorError("active_count cannot be negative")
    if percent != retention_percent or not 0 <= percent <= 100:
        raise RetentionOperatorError("retention_percent must be an integer in [0, 100]")
    return (percent * n + 50) // 100


def capped_dhondt_deletion_allocation(
    channel_counts: Sequence[int] | np.ndarray,
    deletion_total: int,
) -> np.ndarray:
    """Allocate exactly ``deletion_total`` capped deletion seats.

    Starting from d_c=0, each seat goes to the non-full channel maximizing
    N_c/(d_c+1).  Quotients are compared with integer cross-products, and an
    exact tie is resolved by the smaller channel index.  A channel is removed
    from consideration after d_c reaches its clean capacity N_c.

    Because every requested D is a prefix of the same deterministic seat
    sequence, the allocation is house-monotone: increasing D never decreases
    any d_c.
    """

    counts = _as_nonnegative_counts(channel_counts)
    total = sum(counts)
    requested = int(deletion_total)
    if requested != deletion_total or not 0 <= requested <= total:
        raise RetentionOperatorError(
            f"deletion_total must be an integer in [0, {total}]"
        )

    deleted = [0] * len(counts)
    heap = [
        _DhondtCandidate(numerator=count, divisor=1, channel=channel)
        for channel, count in enumerate(counts)
        if count > 0
    ]
    heapq.heapify(heap)
    for _ in range(requested):
        if not heap:
            raise AssertionError("capacity exhausted before exact deletion target")
        candidate = heapq.heappop(heap)
        channel = candidate.channel
        deleted[channel] += 1
        if deleted[channel] < counts[channel]:
            heapq.heappush(
                heap,
                _DhondtCandidate(
                    numerator=counts[channel],
                    divisor=deleted[channel] + 1,
                    channel=channel,
                ),
            )

    result = np.asarray(deleted, dtype=np.int64)
    capacities = np.asarray(counts, dtype=np.int64)
    if int(result.sum()) != requested:
        raise AssertionError("D'Hondt allocation did not meet the exact deletion target")
    if np.any(result < 0) or np.any(result > capacities):
        raise AssertionError("D'Hondt allocation exceeded a channel capacity")
    return result


def source_priority_digest(
    *,
    dataset: str,
    sample_operator_seed: int,
    realization: int,
    channel: int,
    source_bin: int,
    namespace: str = PRIORITY_NAMESPACE,
) -> bytes:
    """Return the frozen full SHA-256 source priority digest.

    Included fields are exactly namespace, dataset, model-independent sample
    operator seed, realization, channel, and clean source-bin identity.
    """

    if not isinstance(namespace, str) or not namespace:
        raise RetentionOperatorError("namespace must be a non-empty string")
    if not isinstance(dataset, str) or not dataset:
        raise RetentionOperatorError("dataset must be a non-empty string")
    fields = {
        "channel": int(channel),
        "dataset": dataset,
        "namespace": namespace,
        "realization": int(realization),
        "sample_operator_seed": int(sample_operator_seed),
        "source_bin": int(source_bin),
    }
    if any(fields[key] < 0 for key in ("realization", "channel", "source_bin")):
        raise RetentionOperatorError("realization, channel and source_bin must be nonnegative")
    encoded = json.dumps(
        fields,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _validate_clean(clean: np.ndarray) -> np.ndarray:
    value = np.asarray(clean)
    if value.ndim != 2:
        raise RetentionOperatorError("clean sample must have shape [time, channel]")
    if value.dtype.kind not in "buif":
        raise RetentionOperatorError("clean sample must be numeric or boolean")
    if not np.all(np.logical_or(value == 0, value == 1)):
        raise RetentionOperatorError("clean sample must be binary")
    return value.astype(np.uint8, copy=False)


def _priority_orders(
    clean: np.ndarray,
    *,
    dataset: str,
    sample_operator_seed: int,
    realization: int,
    namespace: str,
) -> list[np.ndarray]:
    orders: list[np.ndarray] = []
    for channel in range(clean.shape[1]):
        source_bins = np.flatnonzero(clean[:, channel]).astype(np.int64, copy=False)
        ranked = sorted(
            (int(source_bin) for source_bin in source_bins),
            key=lambda source_bin: (
                source_priority_digest(
                    dataset=dataset,
                    sample_operator_seed=sample_operator_seed,
                    realization=realization,
                    channel=channel,
                    source_bin=source_bin,
                    namespace=namespace,
                ),
                source_bin,
            ),
        )
        orders.append(np.asarray(ranked, dtype=np.int64))
    return orders


def make_retention_matched_controls(
    clean: np.ndarray,
    *,
    dataset: str,
    sample_operator_seed: int,
    realization: int,
    retention_percentages: Iterable[int] = DEFAULT_RETENTION_PERCENTAGES,
    namespace: str = PRIORITY_NAMESPACE,
) -> Mapping[str, object]:
    """Create exact, nested deletion-only controls for one clean sample.

    Returns ``outputs`` and ``audits`` dictionaries keyed by the integer target
    percentage.  The input is never modified.
    """

    clean_u8 = _validate_clean(clean)
    percentages = tuple(int(value) for value in retention_percentages)
    if not percentages or len(set(percentages)) != len(percentages):
        raise RetentionOperatorError("retention percentages must be non-empty and unique")
    if any(value < 0 or value > 100 for value in percentages):
        raise RetentionOperatorError("retention percentages must lie in [0, 100]")
    if int(realization) != realization or int(realization) < 0:
        raise RetentionOperatorError("realization must be a nonnegative integer")

    channel_counts = np.count_nonzero(clean_u8, axis=0).astype(np.int64, copy=False)
    active_count = int(channel_counts.sum())
    orders = _priority_orders(
        clean_u8,
        dataset=dataset,
        sample_operator_seed=int(sample_operator_seed),
        realization=int(realization),
        namespace=namespace,
    )

    outputs: dict[int, np.ndarray] = {}
    audits: dict[int, dict[str, object]] = {}
    for percent in percentages:
        retained = retained_target_count(active_count, percent)
        deletion_total = active_count - retained
        deleted_by_channel = capped_dhondt_deletion_allocation(
            channel_counts, deletion_total
        )
        output = clean_u8.copy()
        for channel, count in enumerate(deleted_by_channel.tolist()):
            if count:
                output[orders[channel][:count], channel] = 0

        after_by_channel = np.count_nonzero(output, axis=0).astype(np.int64, copy=False)
        if int(after_by_channel.sum()) != retained:
            raise AssertionError("operator output missed the exact retained target")
        if np.any(after_by_channel + deleted_by_channel != channel_counts):
            raise AssertionError("operator channel accounting mismatch")
        if np.any(output > clean_u8):
            raise AssertionError("deletion-only operator introduced a new active entry")

        outputs[percent] = output
        audits[percent] = {
            "target_retention_percent": percent,
            "active_before": active_count,
            "retained_target": retained,
            "deleted_target": deletion_total,
            "active_after": int(after_by_channel.sum()),
            "active_retention": (
                float(retained / active_count) if active_count else 1.0
            ),
            "empty_sample": active_count == 0,
            "clean_active_by_channel": channel_counts.tolist(),
            "deleted_by_channel": deleted_by_channel.tolist(),
            "retained_by_channel": after_by_channel.tolist(),
        }

    # Fail closed if an implementation change ever breaks the promised nesting.
    descending = sorted(percentages, reverse=True)
    for looser, stricter in zip(descending, descending[1:]):
        # Lower retention (stricter) must be a subset of higher retention.
        if np.any(outputs[stricter] > outputs[looser]):
            raise AssertionError(
                f"deletion sets are not nested for {looser}% -> {stricter}%"
            )

    return {
        "outputs": outputs,
        "audits": audits,
        "priority_namespace": namespace,
        "dataset": dataset,
        "sample_operator_seed": int(sample_operator_seed),
        "realization": int(realization),
    }


__all__ = [
    "DEFAULT_RETENTION_PERCENTAGES",
    "PRIORITY_NAMESPACE",
    "RetentionOperatorError",
    "capped_dhondt_deletion_allocation",
    "make_retention_matched_controls",
    "retained_target_count",
    "source_priority_digest",
]
