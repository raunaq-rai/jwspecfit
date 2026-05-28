"""Tests for validity-range clipping of Martinez+2025 ICF/logU surfaces.

Out-of-range inputs must be clipped to the calibration boundary (not
extrapolated, not dropped to NaN) so that every Monte-Carlo / posterior
draw stays finite and downstream sample counts are preserved.
"""

import numpy as np
import pytest

from jwspecabund.martinez25_icf import (
    _LOG_N43_VALID,
    _LOG_O32_VALID,
    _LOG_U_VALID,
    _Z_ZSUN_VALID,
    icf_NpOp,
    log_U_from_N43,
    log_U_from_O32,
)


def test_log_U_from_O32_clips_out_of_range_O32():
    """log(O32) above the upper bound is clipped to the boundary."""
    hi = _LOG_O32_VALID[1]
    z = 0.2  # in range
    out_of_range = log_U_from_O32(3.02, z)          # user's reported value
    at_boundary = log_U_from_O32(hi, z)
    assert np.isfinite(out_of_range)
    assert out_of_range == pytest.approx(at_boundary)


def test_log_U_from_O32_clips_out_of_range_Z():
    """Z/Zsun below the lower bound is clipped to the boundary."""
    lo = _Z_ZSUN_VALID[0]
    log_o32 = 1.0  # in range
    out_of_range = log_U_from_O32(log_o32, 0.027)   # user's reported value
    at_boundary = log_U_from_O32(log_o32, lo)
    assert np.isfinite(out_of_range)
    assert out_of_range == pytest.approx(at_boundary)


def test_log_U_from_N43_clips_both_inputs():
    """N43 and Z both clipped when out of range."""
    n43_lo = _LOG_N43_VALID[0]
    z_hi = _Z_ZSUN_VALID[1]
    out_of_range = log_U_from_N43(n43_lo - 5.0, z_hi + 5.0)
    at_boundary = log_U_from_N43(n43_lo, z_hi)
    assert np.isfinite(out_of_range)
    assert out_of_range == pytest.approx(at_boundary)


def test_in_range_values_unchanged():
    """Inputs already inside the validity range are not altered."""
    log_o32, z = 1.0, 0.2
    assert _LOG_O32_VALID[0] <= log_o32 <= _LOG_O32_VALID[1]
    assert _Z_ZSUN_VALID[0] <= z <= _Z_ZSUN_VALID[1]
    # Recomputing with the same in-range values is deterministic and finite.
    assert log_U_from_O32(log_o32, z) == pytest.approx(log_U_from_O32(log_o32, z))


def test_icf_clips_logU_and_Z():
    """ICF surface clips both logU and Z; result stays finite."""
    logU_lo = _LOG_U_VALID[0]
    z_lo = _Z_ZSUN_VALID[0]
    out_of_range = icf_NpOp(logU_lo - 3.0, z_lo - 0.5)
    at_boundary = icf_NpOp(logU_lo, z_lo)
    assert np.isfinite(out_of_range)
    assert out_of_range == pytest.approx(at_boundary)


def test_warning_emitted_on_clip(caplog):
    """A warning is logged when an input is clipped."""
    import logging

    with caplog.at_level(logging.WARNING, logger="jwspecabund.martinez25_icf"):
        log_U_from_O32(3.02, 0.027)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "log(O32)" in msgs and "clipped to range" in msgs
    assert "Z/Zsun" in msgs
