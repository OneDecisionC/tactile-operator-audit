from __future__ import annotations

import numpy as np
import pytest


import injective_pbj_operator as frozen  # noqa: E402
import survivor_rule_operator as candidate  # noqa: E402


def _samples() -> list[np.ndarray]:
    rng = np.random.default_rng(20260826)
    values = [
        np.zeros((7, 3), dtype=np.uint8),
        np.ones((7, 3), dtype=np.uint8),
        np.asarray(
            [
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 1],
                [1, 1, 1],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=np.uint8,
        ),
    ]
    for time_steps, channels, probability in ((12, 5, 0.2), (12, 5, 0.7), (80, 8, 0.1)):
        values.append(
            (rng.random((time_steps, channels)) < probability).astype(np.uint8)
        )
    return values


@pytest.mark.parametrize("realization", [0, 1, 2])
@pytest.mark.parametrize("radius", [1, 2, 3])
def test_current_rule_is_bit_exact_with_frozen_operator(radius: int, realization: int) -> None:
    for sample_index, clean in enumerate(_samples()):
        if radius >= clean.shape[0]:
            continue
        seed = 91_000 + sample_index
        expected = frozen.make_injective_pbj_controls(
            clean, radius, seed, realization=realization
        )
        actual = candidate.make_survivor_rule_controls(
            clean, radius, seed, realization=realization
        )
        assert np.array_equal(actual["ipbj_movement"], expected["ipbj_movement"])
        assert np.array_equal(actual["pbj"], expected["pbj"])
        assert np.array_equal(
            actual["assignment_preserving_loss"], expected["pbj_identity_loss"]
        )
        assert np.array_equal(
            actual["assignment_preserving_combined"],
            expected["ipbj_plus_identity_loss"],
        )
        assert actual["audit"]["assignment_digest"] == expected["audit"][
            "assignment_digest"
        ]


@pytest.mark.parametrize("realization", [0, 1, 2])
def test_every_rule_matches_pbj_per_channel_counts(realization: int) -> None:
    for sample_index, clean in enumerate(_samples()):
        result = candidate.make_survivor_rule_controls(
            clean, 2, 77_000 + sample_index, realization=realization
        )
        clean_counts = np.count_nonzero(clean, axis=0)
        pbj_counts = np.count_nonzero(result["pbj"], axis=0)
        assert np.array_equal(
            np.count_nonzero(result["ipbj_movement"], axis=0), clean_counts
        )
        for rule in candidate.ALTERNATIVE_RULES:
            loss_name, combined_name = candidate.CONDITIONS_BY_RULE[rule]
            assert np.array_equal(np.count_nonzero(result[loss_name], axis=0), pbj_counts)
            assert np.array_equal(
                np.count_nonzero(result[combined_name], axis=0), pbj_counts
            )
            assert set(np.unique(result[loss_name])).issubset({0, 1})
            assert set(np.unique(result[combined_name])).issubset({0, 1})


def test_fixed_index_selects_minimum_source_bin() -> None:
    source = np.asarray([6, 1, 4], dtype=np.int64)
    proposal = np.asarray([3, 3, 3], dtype=np.int64)
    assigned = np.asarray([5, 2, 3], dtype=np.int64)
    group = np.arange(3, dtype=np.int64)
    selected = candidate._survivor_local_index(
        "fixed_index",
        source,
        proposal,
        assigned,
        group,
        priority_seed=123,
        channel=0,
    )
    assert int(source[selected]) == 1


def test_minimum_proposed_displacement_and_tie_break() -> None:
    source = np.asarray([1, 3, 5, 7], dtype=np.int64)
    proposal = np.asarray([4, 4, 4, 4], dtype=np.int64)
    assigned = np.asarray([1, 2, 4, 7], dtype=np.int64)
    group = np.arange(4, dtype=np.int64)
    selected = candidate._survivor_local_index(
        "minimum_proposed_displacement",
        source,
        proposal,
        assigned,
        group,
        priority_seed=123,
        channel=0,
    )
    # |proposal-source| = [3,1,1,3], then the lower source time wins.
    assert int(source[selected]) == 3


def test_fixed_random_priority_ignores_proposal_assignment_and_model() -> None:
    source = np.asarray([1, 3, 5, 7], dtype=np.int64)
    group = np.arange(4, dtype=np.int64)
    first = candidate._survivor_local_index(
        "fixed_random_priority",
        source,
        np.asarray([4, 4, 4, 4]),
        np.asarray([1, 2, 4, 7]),
        group,
        priority_seed=987654,
        channel=11,
    )
    second = candidate._survivor_local_index(
        "fixed_random_priority",
        source,
        np.asarray([6, 6, 6, 6]),
        np.asarray([2, 3, 5, 6]),
        group,
        priority_seed=987654,
        channel=11,
    )
    assert first == second
    priorities = [candidate._fixed_priority(987654, 11, int(value)) for value in source]
    expected = min(range(len(source)), key=lambda index: (priorities[index], int(source[index])))
    assert first == expected


def test_repeated_calls_are_deterministic() -> None:
    clean = _samples()[-1]
    first = candidate.make_survivor_rule_controls(
        clean, 3, 123456789, realization=2
    )
    second = candidate.make_survivor_rule_controls(
        clean, 3, 123456789, realization=2
    )
    for key in (
        "ipbj_movement",
        "pbj",
        "assignment_preserving_loss",
        "assignment_preserving_combined",
        *candidate.ALTERNATIVE_CONDITIONS,
    ):
        assert np.array_equal(first[key], second[key])
    assert first["audit"] == second["audit"]
