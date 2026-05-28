"""Tests for validity-range rejection of Martinez+2025 ICF/logU surfaces.

Out-of-range inputs are unphysical for the calibration and must be
*rejected* (set to NaN) — not clipped to the boundary and not
extrapolated.  NaN propagates so the offending value/draw is excluded
from the result, while Monte-Carlo / posterior sample arrays keep their
full length (n_mc / n_posterior); the rejected draws simply drop out of
the nanmedian/nanstd statistics.
"""

import numpy as np

from jwspecabund.martinez25_icf import (
    _LOG_N43_VALID,
    _LOG_O32_VALID,
    _LOG_U_VALID,
    _Z_ZSUN_VALID,
    icf_NpOp,
    log_U_from_N43,
    log_U_from_O32,
)


def test_log_U_from_O32_rejects_out_of_range_O32():
    """log(O32) above the upper bound is rejected (NaN), not clipped."""
    hi = _LOG_O32_VALID[1]
    z = 0.2  # in range
    assert np.isnan(log_U_from_O32(3.02, z))        # user's reported value
    # The boundary value would be finite if clipped — rejection differs.
    assert np.isfinite(log_U_from_O32(hi, z))


def test_log_U_from_O32_rejects_out_of_range_Z():
    """Z/Zsun below the lower bound is rejected (NaN)."""
    log_o32 = 1.0  # in range
    assert np.isnan(log_U_from_O32(log_o32, 0.027))  # user's reported value


def test_log_U_from_N43_rejects_out_of_range():
    """N43 or Z out of range yields NaN."""
    assert np.isnan(log_U_from_N43(_LOG_N43_VALID[0] - 5.0, 0.2))
    assert np.isnan(log_U_from_N43(0.0, _Z_ZSUN_VALID[1] + 5.0))


def test_in_range_values_finite():
    """Inputs inside the validity range return a finite result."""
    log_o32, z = 1.0, 0.2
    assert _LOG_O32_VALID[0] <= log_o32 <= _LOG_O32_VALID[1]
    assert _Z_ZSUN_VALID[0] <= z <= _Z_ZSUN_VALID[1]
    assert np.isfinite(log_U_from_O32(log_o32, z))


def test_icf_rejects_out_of_range_logU_and_Z():
    """ICF surface returns NaN when logU or Z is out of range."""
    assert np.isnan(icf_NpOp(_LOG_U_VALID[0] - 3.0, 0.2))    # logU too low
    assert np.isnan(icf_NpOp(-2.0, _Z_ZSUN_VALID[0] - 0.5))  # Z too low
    assert np.isfinite(icf_NpOp(-2.0, 0.2))                  # both in range


def test_sample_count_preserved_rejected_excluded():
    """Array length is preserved; rejected draws are NaN and excluded.

    Mimics an MC/posterior loop: every draw produces an entry (so the
    array length matches the requested sample count), but unphysical
    draws are NaN and ignored by nanmedian/nanstd.
    """
    z = 0.2
    # Mix of in-range (<=2.5) and out-of-range (>2.5) log(O32) draws.
    draws = np.array([0.5, 1.0, 2.0, 3.02, 4.0, 1.5])
    logU = np.array([log_U_from_O32(o, z) for o in draws])

    assert len(logU) == len(draws)                  # count preserved
    assert np.count_nonzero(np.isnan(logU)) == 2     # the two > 2.5 dropped
    assert np.count_nonzero(np.isfinite(logU)) == 4
    assert np.isfinite(np.nanmedian(logU))           # stats ignore the NaNs


def test_warning_emitted_on_rejection(caplog):
    """A warning is logged when an input is rejected."""
    import logging

    with caplog.at_level(logging.WARNING, logger="jwspecabund.martinez25_icf"):
        log_U_from_O32(3.02, 0.027)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "log(O32)" in msgs and "rejected" in msgs
    assert "Z/Zsun" in msgs
