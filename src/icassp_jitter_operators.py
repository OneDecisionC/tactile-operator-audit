"""Deterministic, deconfounded jitter operators for the ICASSP study.

All post-bin functions operate on one binary NumPy sample shaped ``[T, C]``.
They accept ``uint8`` or ``float32`` and preserve the input dtype.  A realization
is keyed only by ``(seed, realization, shape, operator namespace)``: severity is
deliberately absent from the random seed so severity curves reuse nested random
primitives and can be replayed across models.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np


SUPPORTED_DTYPES = (np.dtype(np.uint8), np.dtype(np.float32))


def _validate_binary_sample(clean: np.ndarray) -> np.ndarray:
    if not isinstance(clean, np.ndarray):
        raise TypeError("clean must be a NumPy array")
    if clean.ndim != 2:
        raise ValueError("a single sample shaped [T, C] is required")
    if clean.dtype not in SUPPORTED_DTYPES:
        raise TypeError("clean dtype must be numpy.uint8 or numpy.float32")
    if clean.shape[0] <= 0 or clean.shape[1] <= 0:
        raise ValueError("T and C must both be positive")
    if not np.all((clean == 0) | (clean == 1)):
        raise ValueError("post-bin input must be binary (0/1)")
    return clean


def _validate_radius(radius_bins: int) -> int:
    if isinstance(radius_bins, (bool, np.bool_)) or not isinstance(
        radius_bins, (int, np.integer)
    ):
        raise TypeError("radius_bins must be an integer")
    radius = int(radius_bins)
    if radius < 0:
        raise ValueError("radius_bins must be non-negative")
    return radius


def _stable_seed(seed: int, realization: int, namespace: str, shape: tuple[int, ...]) -> int:
    """Derive a process-independent uint64 seed without Python's salted hash."""
    digest = hashlib.blake2b(digest_size=16, person=b"icassp-jit-v1")
    for value in (int(seed), int(realization)):
        digest.update(struct.pack("<q", value))
    encoded = namespace.encode("utf-8")
    digest.update(struct.pack("<I", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<I", len(shape)))
    for dimension in shape:
        digest.update(struct.pack("<q", int(dimension)))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def _rng(seed: int, realization: int, namespace: str, shape: tuple[int, ...]):
    return np.random.Generator(
        np.random.PCG64(_stable_seed(seed, realization, namespace, shape))
    )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(struct.pack("<I", contiguous.ndim))
    for dimension in contiguous.shape:
        digest.update(struct.pack("<q", int(dimension)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _normalized_displacement(
    shape: tuple[int, ...], seed: int, realization: int, namespace: str
) -> np.ndarray:
    rng = _rng(seed, realization, namespace, shape)
    # Separate sign and magnitude makes the signed primitive explicit.  Values
    # are in [-1, 1] and are generated once, before severity scaling.
    signs = np.where(rng.integers(0, 2, size=shape, dtype=np.uint8) == 0, -1.0, 1.0)
    magnitudes = rng.random(shape, dtype=np.float64)
    return signs * magnitudes


def _uniform_quantile(
    shape: tuple[int, ...], seed: int, realization: int, namespace: str
) -> np.ndarray:
    """Return replayable U(0,1) quantiles with neither endpoint represented."""
    rng = _rng(seed, realization, namespace, shape)
    quantile = rng.random(shape, dtype=np.float64)
    # PCG64's half-open output can in principle contain exactly zero.  Moving
    # either floating endpoint inward prevents deterministic point mass at a
    # legal interval boundary without changing the practical distribution.
    return np.clip(
        quantile,
        np.nextafter(np.float64(0.0), np.float64(1.0)),
        np.nextafter(np.float64(1.0), np.float64(0.0)),
    )


def _integer_displacement(normalized: np.ndarray, radius: int) -> np.ndarray:
    # Round half away from zero (np.rint uses banker's rounding).
    magnitude = np.floor(np.abs(normalized) * radius + 0.5).astype(np.int64)
    return np.where(normalized < 0, -magnitude, magnitude)


def _cf_plan(
    shape: tuple[int, int], passes: int, seed: int, realization: int
) -> list[dict[str, Any]]:
    t_steps, n_channels = shape
    rng = _rng(seed, realization, "cf-prefix-plan", shape)
    plan: list[dict[str, Any]] = []
    for pass_index in range(passes):
        phase = int(rng.integers(0, 2))
        left = np.arange(phase, t_steps - 1, 2, dtype=np.int64)
        swaps = rng.integers(
            0, 2, size=(left.size, n_channels), dtype=np.uint8
        ).astype(bool)
        packed = np.concatenate(
            (
                np.asarray([pass_index, phase], dtype=np.int64).view(np.uint8),
                left.view(np.uint8),
                swaps.astype(np.uint8, copy=False).ravel(),
            )
        )
        plan.append(
            {
                "phase": phase,
                "left": left,
                "swaps": swaps,
                "digest": hashlib.sha256(packed.tobytes()).hexdigest(),
            }
        )
    return plan


def make_jitter_primitives(
    shape: tuple[int, int], max_cf_passes: int, seed: int, *, realization: int = 0
) -> dict[str, Any]:
    """Expose the replayable primitives used by all post-bin severities.

    Callers normally need only the high-level operators.  This helper is useful
    for manifests and tests: the first ``r`` CF entries are exactly the plan for
    severity ``r``, while PBJ displacement and matched-drop ranks are shared by
    every severity.
    """
    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError("shape must be a positive (T, C) pair")
    max_passes = _validate_radius(max_cf_passes)
    normalized = _normalized_displacement(shape, seed, realization, "pbj-displacement")
    ranks = _rng(seed, realization, "matched-drop-rank", shape).random(
        shape, dtype=np.float64
    )
    cf_plan = _cf_plan(shape, max_passes, seed, realization)
    return {
        "normalized_displacement": normalized,
        "normalized_displacement_digest": _array_digest(normalized),
        "matched_rank": ranks,
        "matched_rank_digest": _array_digest(ranks),
        "cf_passes": cf_plan,
        "cf_pass_digests": [entry["digest"] for entry in cf_plan],
        "seed": int(seed),
        "realization": int(realization),
    }


def active_cell_audit(source: np.ndarray, corrupted: np.ndarray) -> dict[str, Any]:
    source_counts = np.count_nonzero(source > 0, axis=0).astype(np.int64)
    output_counts = np.count_nonzero(corrupted > 0, axis=0).astype(np.int64)
    input_active = int(source_counts.sum())
    output_active = int(output_counts.sum())
    return {
        "input_active_cells": input_active,
        "output_active_cells": output_active,
        "lost_active_cells": input_active - output_active,
        "input_active_by_channel": source_counts.tolist(),
        "output_active_by_channel": output_counts.tolist(),
        "lost_active_by_channel": (source_counts - output_counts).tolist(),
        "active_cell_retention": (
            1.0 if input_active == 0 else float(output_active / input_active)
        ),
        "exact_active_cell_conservation": bool(input_active == output_active),
        "exact_per_channel_active_cell_conservation": bool(
            np.array_equal(source_counts, output_counts)
        ),
    }


def nested_pbj(
    clean: np.ndarray,
    radius_bins: int,
    seed: int,
    *,
    realization: int = 0,
    return_audit: bool = False,
):
    """Post-bin jitter with shift, boundary clipping and binary collisions."""
    clean = _validate_binary_sample(clean)
    radius = _validate_radius(radius_bins)
    normalized = _normalized_displacement(
        clean.shape, seed, realization, "pbj-displacement"
    )
    displacement = _integer_displacement(normalized, radius)
    active_t, active_c = np.nonzero(clean > 0)
    proposed_t = active_t + displacement[active_t, active_c]
    target_t = np.clip(proposed_t, 0, clean.shape[0] - 1)
    output = np.zeros_like(clean)
    output[target_t, active_c] = np.asarray(1, dtype=clean.dtype)

    audit = {
        "operator": "postbin_binary_jitter_collision_v1",
        "seed": int(seed),
        "realization": int(realization),
        "radius_bins": radius,
        "normalized_displacement_digest": _array_digest(normalized),
        "active_cells_shifted": int(active_t.size),
        "active_cells_moved": int(np.count_nonzero(target_t != active_t)),
        "active_cells_with_zero_quantized_shift": int(
            np.count_nonzero(displacement[active_t, active_c] == 0)
        ),
        "boundary_clipped_active_cells": int(np.count_nonzero(proposed_t != target_t)),
        **active_cell_audit(clean, output),
    }
    audit["binary_collision_loss"] = audit["lost_active_cells"]
    return (output, audit) if return_audit else output


def nested_cf(
    clean: np.ndarray,
    radius_bins: int,
    seed: int,
    *,
    realization: int = 0,
    return_audit: bool = False,
):
    """Apply the first ``radius_bins`` collision-free adjacent-swap passes."""
    clean = _validate_binary_sample(clean)
    radius = _validate_radius(radius_bins)
    output = clean.copy()
    plan = _cf_plan(clean.shape, radius, seed, realization)
    selected_swaps: list[int] = []
    for entry in plan:
        left = entry["left"]
        swaps = entry["swaps"]
        if left.size:
            left_values = output[left, :].copy()
            right_values = output[left + 1, :].copy()
            output[left, :] = np.where(swaps, right_values, left_values)
            output[left + 1, :] = np.where(swaps, left_values, right_values)
        selected_swaps.append(int(np.count_nonzero(swaps)))

    audit = {
        "operator": "collision_free_adjacent_swap_prefix_v1",
        "seed": int(seed),
        "realization": int(realization),
        "radius_bins": radius,
        "cf_pass_phases": [entry["phase"] for entry in plan],
        "cf_pass_digests": [entry["digest"] for entry in plan],
        "selected_swaps_by_pass": selected_swaps,
        **active_cell_audit(clean, output),
    }
    return (output, audit) if return_audit else output


def exact_loss_matched_dropout(
    clean: np.ndarray,
    pbj: np.ndarray,
    seed: int,
    *,
    realization: int = 0,
    return_audit: bool = False,
):
    """Delete exactly PBJ's active-cell loss independently in every channel."""
    clean = _validate_binary_sample(clean)
    pbj = _validate_binary_sample(pbj)
    if pbj.shape != clean.shape:
        raise ValueError("clean and pbj must have identical [T, C] shapes")

    clean_counts = np.count_nonzero(clean > 0, axis=0).astype(np.int64)
    pbj_counts = np.count_nonzero(pbj > 0, axis=0).astype(np.int64)
    loss_by_channel = clean_counts - pbj_counts
    if np.any(loss_by_channel < 0):
        raise ValueError("pbj cannot contain more active cells than clean in any channel")

    ranks = _rng(seed, realization, "matched-drop-rank", clean.shape).random(
        clean.shape, dtype=np.float64
    )
    output = clean.copy()
    deleted_indices: list[list[int]] = []
    for channel, loss in enumerate(loss_by_channel):
        active = np.flatnonzero(clean[:, channel] > 0)
        # The random score is primary and time is a stable, deterministic tie-break.
        order = np.lexsort((active, ranks[active, channel]))
        deleted = active[order[: int(loss)]]
        output[deleted, channel] = np.asarray(0, dtype=clean.dtype)
        deleted_indices.append(deleted.astype(np.int64).tolist())

    audit = {
        "operator": "exact_pbj_loss_matched_dropout_per_channel_v1",
        "seed": int(seed),
        "realization": int(realization),
        "matched_rank_digest": _array_digest(ranks),
        "target_pbj_output_active_by_channel": pbj_counts.tolist(),
        "target_loss_by_channel": loss_by_channel.tolist(),
        "deleted_time_indices_by_channel": deleted_indices,
        **active_cell_audit(clean, output),
    }
    audit["exact_pbj_count_match_per_channel"] = bool(
        np.array_equal(np.count_nonzero(output > 0, axis=0), pbj_counts)
    )
    if not audit["exact_pbj_count_match_per_channel"]:
        raise AssertionError("internal error: matched dropout did not match PBJ counts")
    return (output, audit) if return_audit else output


def cf_matched(
    clean: np.ndarray,
    pbj: np.ndarray,
    radius_bins: int,
    seed: int,
    *,
    realization: int = 0,
    return_audit: bool = False,
):
    """Apply fixed-rank PBJ-loss deletion, then the same CF prefix plan."""
    matched, matched_audit = exact_loss_matched_dropout(
        clean, pbj, seed, realization=realization, return_audit=True
    )
    output, cf_audit = nested_cf(
        matched, radius_bins, seed, realization=realization, return_audit=True
    )
    pbj_counts = np.count_nonzero(pbj > 0, axis=0)
    output_counts = np.count_nonzero(output > 0, axis=0)
    audit = {
        "operator": "collision_free_jitter_plus_exact_matched_dropout_v1",
        "seed": int(seed),
        "realization": int(realization),
        "radius_bins": int(radius_bins),
        "matched_dropout": matched_audit,
        "collision_free_jitter": cf_audit,
        **active_cell_audit(clean, output),
        "exact_pbj_count_match_per_channel": bool(
            np.array_equal(output_counts, pbj_counts)
        ),
    }
    if not audit["exact_pbj_count_match_per_channel"]:
        raise AssertionError("internal error: CF+matched did not match PBJ counts")
    return (output, audit) if return_audit else output


def make_deconfounded_bundle(
    clean: np.ndarray,
    radius_bins: int,
    seed: int,
    *,
    realization: int = 0,
) -> dict[str, Any]:
    """Return PBJ, CF, matched-dropout and CF+matched for one realization."""
    pbj, pbj_audit = nested_pbj(
        clean, radius_bins, seed, realization=realization, return_audit=True
    )
    cf, cf_audit = nested_cf(
        clean, radius_bins, seed, realization=realization, return_audit=True
    )
    matched, matched_audit = exact_loss_matched_dropout(
        clean, pbj, seed, realization=realization, return_audit=True
    )
    combined, combined_audit = cf_matched(
        clean,
        pbj,
        radius_bins,
        seed,
        realization=realization,
        return_audit=True,
    )
    target_counts = np.count_nonzero(pbj > 0, axis=0)
    if not (
        np.array_equal(np.count_nonzero(matched > 0, axis=0), target_counts)
        and np.array_equal(np.count_nonzero(combined > 0, axis=0), target_counts)
    ):
        raise AssertionError("internal error: matched controls differ from PBJ counts")
    return {
        "pbj": pbj,
        "cf": cf,
        "matched": matched,
        "cf_matched": combined,
        "audit": {
            "pbj": pbj_audit,
            "cf": cf_audit,
            "matched": matched_audit,
            "cf_matched": combined_audit,
        },
    }


def prebin_jitter(
    timestamps: np.ndarray,
    channels: np.ndarray,
    severity_seconds: float,
    duration_seconds: float,
    seed: int,
    *,
    n_steps: int,
    n_channels: int | None = None,
    realization: int = 0,
    return_timestamps: bool = False,
):
    """Uniformly jitter raw timestamps over each event's legal interval.

    Each event receives one fixed quantile ``u_i in (0, 1)``.  At severity
    ``J``, its legal displacement interval is
    ``[max(-J, -t_i), min(J, duration-t_i)]`` and inverse-CDF sampling maps the
    shared quantile into that interval.  This conditional-uniform construction
    keeps every event in range without clipping out-of-range proposals onto the
    temporal endpoints (which would create artificial boundary point masses).
    Binary occupancy may still merge events, and that is audited separately.
    """
    timestamps = np.asarray(timestamps)
    channels = np.asarray(channels)
    if timestamps.ndim != 1 or channels.ndim != 1 or timestamps.shape != channels.shape:
        raise ValueError("timestamps and channels must be same-length 1-D arrays")
    if not np.issubdtype(timestamps.dtype, np.number):
        raise TypeError("timestamps must be numeric")
    if not np.issubdtype(channels.dtype, np.integer):
        raise TypeError("channels must contain integer indices")
    severity = float(severity_seconds)
    duration = float(duration_seconds)
    if not np.isfinite(severity) or severity < 0:
        raise ValueError("severity_seconds must be finite and non-negative")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    if isinstance(n_steps, (bool, np.bool_)) or int(n_steps) != n_steps or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    n_steps = int(n_steps)
    timestamps64 = timestamps.astype(np.float64, copy=False)
    if not np.all(np.isfinite(timestamps64)):
        raise ValueError("timestamps must be finite")
    if np.any(timestamps64 < 0) or np.any(timestamps64 > duration):
        raise ValueError("timestamps must lie in [0, duration_seconds]")
    channel64 = channels.astype(np.int64, copy=False)
    if np.any(channel64 < 0):
        raise ValueError("channel indices must be non-negative")
    if n_channels is None:
        if channel64.size == 0:
            raise ValueError("n_channels is required for an empty event stream")
        n_channels = int(channel64.max()) + 1
    if isinstance(n_channels, (bool, np.bool_)) or int(n_channels) != n_channels:
        raise ValueError("n_channels must be a positive integer")
    n_channels = int(n_channels)
    if n_channels <= 0 or np.any(channel64 >= n_channels):
        raise ValueError("channel index is outside [0, n_channels)")

    event_shape = (int(timestamps64.size),)
    # Bind random quantiles to a canonical event order, not to caller array
    # positions.  Exact duplicate (timestamp, channel) records are
    # indistinguishable; permuting quantiles within such a duplicate group
    # preserves the jittered event multiset and therefore the binary output.
    canonical_order = np.lexsort((channel64, timestamps64))
    canonical_quantile = _uniform_quantile(
        event_shape, seed, realization, "prebin-quantile"
    )
    quantile = np.empty_like(canonical_quantile)
    quantile[canonical_order] = canonical_quantile
    lower = np.maximum(-severity, -timestamps64)
    upper = np.minimum(severity, duration - timestamps64)
    effective = lower + quantile * (upper - lower)
    jittered = timestamps64 + effective
    # Preserve the open-interval property against the rare case where floating
    # addition rounds an interior inverse-CDF draw onto a global endpoint.
    nondegenerate = upper > lower
    lower_roundoff = nondegenerate & (jittered == 0.0)
    upper_roundoff = nondegenerate & (jittered == duration)
    if np.any(lower_roundoff):
        jittered[lower_roundoff] = np.nextafter(0.0, duration)
    if np.any(upper_roundoff):
        jittered[upper_roundoff] = np.nextafter(duration, 0.0)
    effective = jittered - timestamps64
    if np.any(jittered < 0.0) or np.any(jittered > duration):
        raise AssertionError("internal error: conditional jitter left the legal interval")
    # An event exactly at the right endpoint belongs to the final bin.
    bin_index = np.floor(jittered * (n_steps / duration)).astype(np.int64)
    bin_index = np.clip(bin_index, 0, n_steps - 1)
    output = np.zeros((n_steps, n_channels), dtype=np.uint8)
    output[bin_index, channel64] = 1

    event_count = int(timestamps64.size)
    active_cells = int(np.count_nonzero(output))
    audit = {
        "operator": "prebin_timestamp_jitter_conditional_uniform_v2",
        "seed": int(seed),
        "realization": int(realization),
        "severity_seconds": severity,
        "duration_seconds": duration,
        "n_steps": n_steps,
        "n_channels": n_channels,
        # Digest the canonical primitive so it is invariant to input record
        # ordering.  ``quantile`` itself is restored to caller order above.
        "uniform_quantile_digest": _array_digest(canonical_quantile),
        # Backward-compatible manifest field; the primitive is now an unsigned
        # inverse-CDF quantile rather than a signed displacement.
        "normalized_displacement_digest": _array_digest(canonical_quantile),
        "canonical_event_order": "lexsort(timestamp,channel); duplicate multiset",
        "input_event_count": event_count,
        "output_event_count": event_count,
        "events_dropped": 0,
        "exact_event_count_conservation": True,
        "boundary_constrained_events": int(
            np.count_nonzero((upper - lower) < (2.0 * severity))
        ),
        "events_exactly_at_boundary_after_jitter": int(
            np.count_nonzero((jittered == 0.0) | (jittered == duration))
        ),
        "binary_active_cells": active_cells,
        "binary_collision_count": event_count - active_cells,
        "jittered_timestamp_min": None if event_count == 0 else float(jittered.min()),
        "jittered_timestamp_max": None if event_count == 0 else float(jittered.max()),
        "all_events_in_bounds": bool(
            np.all((jittered >= 0.0) & (jittered <= duration))
        ),
        "shape": [n_steps, n_channels],
    }
    if return_timestamps:
        return output, jittered, audit
    return output, audit


# Readable aliases for callers that do not need the nesting terminology.
postbin_jitter = nested_pbj
collision_free_jitter = nested_cf
loss_matched_dropout = exact_loss_matched_dropout
prebin_timestamp_jitter = prebin_jitter


__all__ = [
    "active_cell_audit",
    "cf_matched",
    "collision_free_jitter",
    "exact_loss_matched_dropout",
    "loss_matched_dropout",
    "make_deconfounded_bundle",
    "make_jitter_primitives",
    "nested_cf",
    "nested_pbj",
    "postbin_jitter",
    "prebin_jitter",
    "prebin_timestamp_jitter",
]
