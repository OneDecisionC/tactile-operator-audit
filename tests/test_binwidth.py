import numpy as np
import pytest
from types import SimpleNamespace

from run_icassp_jitter_screening import RawEvents
from tactile_audit.binwidth import audit_bundle, rebin


@pytest.mark.parametrize("width", [12.5, 25.0, 50.0])
def test_partial_last_bin_and_endpoint(width):
    raw = RawEvents(np.array([0, width / 1000, 1.125]), np.array([0, 1, 2]))
    clean = rebin(raw, "braille", width, 1.125, 3)
    assert clean[0, 0] == clean[1, 1] == clean[-1, 2] == 1
    assert clean.sum() == 3
    if width == 50:
        assert clean.shape == (23, 3)


def test_stemnist_integer_tick_binning():
    ticks = np.arange(241)
    raw = RawEvents(ticks / 120, np.zeros(241, dtype=int))
    clean = rebin(raw, "stemnist", 25.0, 2.0, 1)
    assert clean.shape == (80, 1) and clean.sum() == 80
    with pytest.raises(ValueError, match="tick grid"):
        rebin(RawEvents(np.array([0.001]), np.array([0])), "stemnist", 25.0, 2.0, 1)


def test_streaming_binwidth_audit_counts_and_baseline():
    raw = RawEvents(np.array([0, 0.025, 0.050, 1.125]), np.array([0, 0, 0, 0]))
    clean = rebin(raw, "braille", 25.0, 1.125, 1)
    bundle = SimpleNamespace(name="braille", labels=[0], sample_ids=["synthetic_0"], raw_getter=lambda _: raw,
                             duration_seconds=1.125, channels=1, clean=clean[None, :], time_steps=45)
    rows = audit_bundle(bundle)
    assert len(rows) == 3
    for row in rows:
        assert row["raw_events"] == 4 and row["samples"] == 1
        assert row["realization_count"] == 3
        assert row["retention_pct_realization_min"] <= row["retention_pct"] <= row["retention_pct_realization_max"]
        assert row["clean_active_cells"] <= row["raw_events"]
    bundle.clean[:] = 0
    with pytest.raises(ValueError, match="reconstruction mismatch"):
        audit_bundle(bundle)
