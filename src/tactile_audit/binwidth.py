"""Operator-only bin-width audit adapted from the completed historical audit."""
import math

import numpy as np

import icassp_jitter_operators as operators
import run_icassp_jitter_screening as screen
from tactile_audit.workflow import operator_seed

WIDTHS = (12.5, 25.0, 50.0)


def rebin(raw, dataset, width, duration, channels):
    if width not in WIDTHS:
        raise ValueError("The audited bin widths are 12.5, 25 and 50 ms")
    steps = math.ceil(duration * 1000 / width)
    times = np.asarray(raw.times_seconds, dtype=np.float64)
    channel_ids = np.asarray(raw.channels, dtype=np.int64)
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(times > duration):
        raise ValueError("Raw timestamps are outside the observation window")
    if times.shape != channel_ids.shape or np.any(channel_ids < 0) or np.any(channel_ids >= channels):
        raise ValueError("Invalid raw channel indices")
    if dataset == "stemnist":
        ticks = np.rint(times * 120).astype(np.int64)
        if np.any(np.abs(times * 120 - ticks) > 1e-3):
            raise ValueError("STEMNIST timestamps are not on the 120-Hz tick grid")
        bins = ticks * 2 // int(round(width * 120 / 1000 * 2))
    elif dataset == "braille":
        bins = np.floor(times / (width / 1000)).astype(np.int64)
    else:
        raise ValueError("Unknown dataset")
    clean = np.zeros((steps, channels), dtype=np.uint8)
    clean[np.minimum(bins, steps - 1), channel_ids] = 1
    return clean


def audit_bundle(bundle):
    totals = {(width, realization): np.zeros(6, dtype=np.int64) for width in WIDTHS for realization in range(3)}
    registry = {"protocol_id": "icassp_stage_b_formal_v1"}
    for index in range(len(bundle.labels)):
        raw = bundle.raw_getter(index)
        seed = operator_seed(bundle, index, registry)
        for width in WIDTHS:
            clean = rebin(raw, bundle.name, width, bundle.duration_seconds, bundle.channels)
            if width == 25.0 and not np.array_equal(clean, screen.unpack_clean_sample(bundle, index)):
                raise ValueError(f"25-ms reconstruction mismatch at sample {index}")
            for realization in range(3):
                output, audit = operators.nested_pbj(clean, int(50 / width), seed, realization=realization, return_audit=True)
                source_time, source_channel = np.nonzero(clean)
                normalized = operators._normalized_displacement(clean.shape, seed, realization, "pbj-displacement")
                shifts = operators._integer_displacement(normalized, int(50 / width))
                destination = np.clip(source_time + shifts[source_time, source_channel], 0, len(clean) - 1)
                displacement = int(np.abs(destination - source_time).sum())
                totals[width, realization] += [len(raw.times_seconds), int(clean.sum()), int(output.sum()),
                                                audit["active_cells_moved"], displacement, audit["boundary_clipped_active_cells"]]
    result = []
    for width in WIDTHS:
        raw_count, clean_count = totals[width, 0][:2]
        pooled = sum((totals[width, realization] for realization in range(3)))
        denominator = int(pooled[1])
        retention = [100 * totals[width, realization][2] / clean_count if clean_count else 100.0 for realization in range(3)]
        steps = math.ceil(bundle.duration_seconds * 1000 / width)
        result.append({"dataset": bundle.name, "bin_width_ms": width, "physical_jitter_ms": 50.0,
                       "radius_bins": int(50 / width), "time_bins": steps,
                       "last_bin_width_ms": bundle.duration_seconds * 1000 - (steps - 1) * width,
                       "samples": len(bundle.labels), "raw_events": int(raw_count), "clean_active_cells": int(clean_count),
                       "clean_occupancy_per_event_pct": float(100 * clean_count / raw_count) if raw_count else 100.0,
                       "retention_pct": float(100 * pooled[2] / denominator) if denominator else 100.0,
                       "moved_fraction_pct": float(100 * pooled[3] / denominator) if denominator else 0.0,
                       "mean_absolute_displacement_bins": float(pooled[4] / denominator) if denominator else 0.0,
                       "mean_absolute_displacement_ms": float(pooled[4] * width / denominator) if denominator else 0.0,
                       "boundary_clipping_fraction_pct": float(100 * pooled[5] / denominator) if denominator else 0.0,
                       "realization_count": 3, "retention_pct_realization_min": float(min(retention)),
                       "retention_pct_realization_max": float(max(retention))})
    return result
