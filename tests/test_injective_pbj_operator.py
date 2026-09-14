import numpy as np
import pytest

import icassp_jitter_operators as parent
import injective_pbj_operator as strict


def dense_sample(dtype=np.uint8):
    value = np.zeros((17, 4), dtype=dtype)
    value[[0, 2, 5, 8, 11, 15], 0] = 1
    value[[1, 3, 4, 9, 12, 16], 1] = 1
    value[[0, 1, 6, 7, 13, 14], 2] = 1
    value[[2, 4, 6, 10, 12, 15], 3] = 1
    return value


def test_four_cell_factorization_and_count_contracts():
    clean = dense_sample()
    selected = None
    for seed in range(200):
        bundle = strict.make_injective_pbj_controls(clean, 3, seed)
        if bundle["audit"]["pbj_collision_lost_identities"] > 0:
            selected = bundle
            break
    assert selected is not None
    movement = selected["ipbj_movement"]
    loss = selected["pbj_identity_loss"]
    pbj = selected["pbj"]
    audit = selected["audit"]
    assert np.array_equal(np.count_nonzero(movement, axis=0), np.count_nonzero(clean, axis=0))
    assert np.array_equal(np.count_nonzero(loss, axis=0), np.count_nonzero(pbj, axis=0))
    assert np.array_equal(
        np.count_nonzero(selected["ipbj_plus_identity_loss"], axis=0),
        np.count_nonzero(pbj, axis=0),
    )
    assert audit["repaired_assignments"] >= audit["pbj_collision_lost_identities"]
    assert audit["injective_max_abs_displacement_bins"] <= 3


def test_exact_parent_pbj_proposals_and_replay():
    clean = dense_sample(np.float32)
    first = strict.make_injective_pbj_controls(clean, 2, 991, realization=4)
    second = strict.make_injective_pbj_controls(clean, 2, 991, realization=4)
    parent_pbj, parent_audit = parent.nested_pbj(
        clean, 2, 991, realization=4, return_audit=True
    )
    for key in ("ipbj_movement", "pbj_identity_loss", "ipbj_plus_identity_loss", "pbj"):
        assert np.array_equal(first[key], second[key])
        assert first[key].dtype == clean.dtype
    assert first["audit"] == second["audit"]
    assert np.array_equal(first["pbj"], parent_pbj)
    assert (
        first["audit"]["normalized_displacement_digest"]
        == parent_audit["normalized_displacement_digest"]
    )


def test_no_collision_reduces_to_pbj_and_clean_loss_cell():
    clean = np.zeros((12, 2), dtype=np.uint8)
    clean[2, 0] = 1
    clean[9, 1] = 1
    bundle = strict.make_injective_pbj_controls(clean, 1, 7)
    assert bundle["audit"]["pbj_collision_lost_identities"] == 0
    assert np.array_equal(bundle["ipbj_movement"], bundle["pbj"])
    assert np.array_equal(bundle["pbj_identity_loss"], clean)


def test_randomized_factorization_property():
    rng = np.random.default_rng(20260819)
    for case in range(250):
        time_steps = int(rng.integers(4, 25))
        channels = int(rng.integers(1, 8))
        density = float(rng.uniform(0.05, 0.9))
        clean = (rng.random((time_steps, channels)) < density).astype(np.uint8)
        radius = int(rng.integers(1, min(3, time_steps - 1) + 1))
        bundle = strict.make_injective_pbj_controls(
            clean, radius, case * 31 + 5, realization=case % 5
        )
        assert np.array_equal(
            np.count_nonzero(bundle["ipbj_plus_identity_loss"], axis=0),
            np.count_nonzero(bundle["pbj"], axis=0),
        )
        assert bundle["audit"]["injective_max_abs_displacement_bins"] <= radius


def test_dense_counterexample_preserves_bound_and_reports_uncovered_target():
    source = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 16, 18, 20, 21])
    proposal = np.array([2, 3, 4, 4, 5, 7, 7, 8, 10, 12, 12, 16, 17, 20, 21])
    assigned, exact = strict._bounded_assignment(source, proposal, 1, 22)
    assert np.unique(assigned).size == source.size
    assert np.max(np.abs(assigned - source)) <= 1
    assert set(proposal[exact].tolist()) != set(proposal.tolist())


def test_nested_severities_reuse_parent_primitive():
    clean = dense_sample()
    audits = [
        strict.make_injective_pbj_controls(clean, radius, 123, realization=2)["audit"]
        for radius in (1, 2, 3)
    ]
    assert len({row["normalized_displacement_digest"] for row in audits}) == 1


def test_parent_validation_is_preserved():
    with pytest.raises(ValueError, match=r"\[T, C\]"):
        strict.make_injective_pbj_controls(np.zeros((2, 3, 4), dtype=np.uint8), 1, 0)
    with pytest.raises(TypeError, match="dtype"):
        strict.make_injective_pbj_controls(np.zeros((3, 2), dtype=np.float64), 1, 0)
