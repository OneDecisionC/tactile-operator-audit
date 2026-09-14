from __future__ import annotations

import hashlib

import numpy as np
import pytest

from retention_matched_operator import (
    RetentionOperatorError,
    capped_dhondt_deletion_allocation,
    make_retention_matched_controls,
    retained_target_count,
    source_priority_digest,
)


TARGETS = (95, 90, 85)


def _hamilton_retained(counts: tuple[int, ...], retained_total: int) -> tuple[int, ...]:
    """Exact largest-remainder comparator used only to document its paradox."""

    total = sum(counts)
    numerators = [retained_total * count for count in counts]
    allocated = [value // total for value in numerators]
    remainder_seats = retained_total - sum(allocated)
    order = sorted(
        range(len(counts)),
        key=lambda channel: (-(numerators[channel] % total), channel),
    )
    for channel in order[:remainder_seats]:
        allocated[channel] += 1
    return tuple(allocated)


def test_round_half_up_is_exact_integer_arithmetic() -> None:
    assert [retained_target_count(10, target) for target in TARGETS] == [10, 9, 9]
    assert [retained_target_count(30, target) for target in TARGETS] == [29, 27, 26]
    assert retained_target_count(0, 85) == 0
    assert retained_target_count(1, 50) == 1
    assert retained_target_count(3, 50) == 2


def test_hamilton_counterexample_explains_house_monotone_choice() -> None:
    # Independent Hamilton allocations exhibit the Alabama paradox here:
    # decreasing retention from 90% to 85% increases channel 0's retained count.
    counts = (4, 11, 11)
    totals = tuple(retained_target_count(sum(counts), target) for target in TARGETS)
    assert totals == (25, 23, 22)
    allocations = tuple(_hamilton_retained(counts, total) for total in totals)
    assert allocations == ((4, 11, 10), (3, 10, 10), (4, 9, 9))
    assert allocations[2][0] > allocations[1][0]


def test_dhondt_exact_tie_capacity_and_empty_channels() -> None:
    np.testing.assert_array_equal(
        capped_dhondt_deletion_allocation([1, 1], 1), [1, 0]
    )
    np.testing.assert_array_equal(
        capped_dhondt_deletion_allocation([0, 2, 0, 1], 3), [0, 2, 0, 1]
    )
    np.testing.assert_array_equal(
        capped_dhondt_deletion_allocation([0, 0], 0), [0, 0]
    )


def test_dhondt_is_exact_capacity_safe_and_house_monotone() -> None:
    rng = np.random.default_rng(20260827)
    for _ in range(120):
        counts = rng.integers(0, 13, size=int(rng.integers(1, 14)), dtype=np.int64)
        total = int(counts.sum())
        previous = np.zeros_like(counts)
        for deletion_total in range(total + 1):
            current = capped_dhondt_deletion_allocation(counts, deletion_total)
            assert int(current.sum()) == deletion_total
            assert np.all(current >= 0)
            assert np.all(current <= counts)
            assert np.all(current >= previous)
            previous = current


def test_operator_exact_targets_capacity_and_global_nesting() -> None:
    rng = np.random.default_rng(31801271)
    for sample_number in range(80):
        clean = (rng.random((23, 9)) < rng.uniform(0.03, 0.8)).astype(np.uint8)
        generated = make_retention_matched_controls(
            clean,
            dataset="synthetic",
            sample_operator_seed=1000 + sample_number,
            realization=sample_number % 3,
        )
        outputs = generated["outputs"]
        audits = generated["audits"]
        active = int(clean.sum())
        for target in TARGETS:
            output = outputs[target]
            audit = audits[target]
            expected = retained_target_count(active, target)
            assert output.dtype == np.uint8
            assert int(output.sum()) == expected
            assert int(audit["retained_target"]) == expected
            assert int(audit["active_after"]) == expected
            assert sum(audit["deleted_by_channel"]) == active - expected
            assert all(
                deleted <= capacity
                for deleted, capacity in zip(
                    audit["deleted_by_channel"], audit["clean_active_by_channel"]
                )
            )
            assert np.all(output <= clean)
        assert np.all(outputs[85] <= outputs[90])
        assert np.all(outputs[90] <= outputs[95])


def test_priority_and_operator_are_reproducible_and_model_independent() -> None:
    clean = np.ones((40, 5), dtype=np.uint8)
    kwargs = {
        "dataset": "braille",
        "sample_operator_seed": 987654321,
        "realization": 2,
    }
    # Hypothetical model labels do not enter the API or the hash key.
    outputs_by_model = {}
    for model in ("ratesnn", "tcn", "conv1d"):
        result = make_retention_matched_controls(clean, **kwargs)
        outputs_by_model[model] = result["outputs"]
    for target in TARGETS:
        expected = outputs_by_model["ratesnn"][target]
        assert np.array_equal(expected, outputs_by_model["tcn"][target])
        assert np.array_equal(expected, outputs_by_model["conv1d"][target])

    first = make_retention_matched_controls(clean, **kwargs)
    second = make_retention_matched_controls(clean, **kwargs)
    for target in TARGETS:
        assert np.array_equal(first["outputs"][target], second["outputs"][target])


def test_three_realizations_are_truly_distinct_but_each_is_nested() -> None:
    clean = np.ones((128, 4), dtype=np.uint8)
    fingerprints = []
    for realization in (0, 1, 2):
        result = make_retention_matched_controls(
            clean,
            dataset="stemnist",
            sample_operator_seed=42,
            realization=realization,
        )
        outputs = result["outputs"]
        assert np.all(outputs[85] <= outputs[90])
        assert np.all(outputs[90] <= outputs[95])
        fingerprints.append(hashlib.sha256(outputs[85].tobytes()).hexdigest())
    assert len(set(fingerprints)) == 3


def test_priority_key_changes_for_every_included_field() -> None:
    base = dict(
        dataset="braille",
        sample_operator_seed=7,
        realization=0,
        channel=2,
        source_bin=3,
    )
    reference = source_priority_digest(**base)
    assert reference == source_priority_digest(**base)
    variants = [
        {**base, "dataset": "stemnist"},
        {**base, "sample_operator_seed": 8},
        {**base, "realization": 1},
        {**base, "channel": 3},
        {**base, "source_bin": 4},
        {**base, "namespace": "another-frozen-namespace"},
    ]
    assert all(source_priority_digest(**variant) != reference for variant in variants)


def test_empty_sample_and_empty_channels_are_defined() -> None:
    clean = np.zeros((11, 6), dtype=np.uint8)
    result = make_retention_matched_controls(
        clean,
        dataset="empty",
        sample_operator_seed=0,
        realization=0,
    )
    for target in TARGETS:
        assert not np.any(result["outputs"][target])
        assert result["audits"][target]["empty_sample"] is True
        assert result["audits"][target]["active_retention"] == 1.0


@pytest.mark.parametrize(
    "clean",
    [
        np.zeros(4, dtype=np.uint8),
        np.asarray([[0, 2]], dtype=np.uint8),
        np.asarray([[0.0, np.nan]], dtype=np.float64),
    ],
)
def test_invalid_clean_inputs_fail_closed(clean: np.ndarray) -> None:
    with pytest.raises(RetentionOperatorError):
        make_retention_matched_controls(
            clean,
            dataset="bad",
            sample_operator_seed=1,
            realization=0,
        )


def test_invalid_allocation_targets_fail_closed() -> None:
    with pytest.raises(RetentionOperatorError):
        capped_dhondt_deletion_allocation([1, 2], 4)
    with pytest.raises(RetentionOperatorError):
        capped_dhondt_deletion_allocation([1, -1], 0)
    with pytest.raises(RetentionOperatorError):
        retained_target_count(3, 101)
