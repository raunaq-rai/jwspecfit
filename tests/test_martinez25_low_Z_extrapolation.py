"""Tests for the extended low-metallicity edge of the Martinez+2025 surfaces.

The published Cloudy grid spans 0.05 <= Z/Zsun <= 0.40.  This package
enforces 0.02 <= Z/Zsun <= 0.40, extending the *lower* edge only, so that
genuinely metal-poor objects are not thrown away.  These tests pin down
three things: that the strip is accepted, that it is flagged as
extrapolated rather than silently treated as calibrated, and that the
extrapolation is as small as the decision to allow it assumed.
"""

import numpy as np
import pytest

from jwspecabund.martinez25_icf import (
    _LOG_N43_VALID,
    _LOG_O32_VALID,
    _Z_ZSUN_PUBLISHED,
    _Z_ZSUN_VALID,
    LOG_OH_SOLAR,
    icf_NpNpp_OpOpp,
    icf_NpOp,
    icf_NppOpp,
    is_Z_extrapolated,
    log_U_from_N43,
    log_U_from_O32,
)

#: Object 131: 12+log(O/H) = 7.226 +/- 0.129, below the published floor
#: of 7.389 by 0.163 dex (1.26 sigma).
Z_131 = 10.0 ** (7.226 - LOG_OH_SOLAR)
Z_FLOOR = _Z_ZSUN_PUBLISHED[0]


def test_bounds_extended_downwards_only():
    """Only the lower metallicity edge moved; the upper edge is published."""
    assert _Z_ZSUN_VALID == (0.02, _Z_ZSUN_PUBLISHED[1])
    assert _Z_ZSUN_VALID[0] < _Z_ZSUN_PUBLISHED[0]


def test_object_131_is_accepted():
    """A metallicity 0.163 dex below the grid floor now returns a number."""
    assert _Z_ZSUN_VALID[0] <= Z_131 < Z_FLOOR
    assert np.isfinite(log_U_from_O32(1.0, Z_131))
    assert np.isfinite(log_U_from_N43(-1.0, Z_131))
    assert np.isfinite(icf_NpOp(-2.0, Z_131))
    assert np.isfinite(icf_NppOpp(-2.0, Z_131))


def test_still_rejected_below_the_new_floor():
    """0.02 is a bound, not a suggestion: below it, still NaN."""
    assert np.isnan(log_U_from_O32(1.0, 0.01))
    assert np.isnan(icf_NpOp(-2.0, 0.01))


def test_upper_bound_unchanged():
    """Extending the low edge must not have loosened the high edge."""
    assert np.isnan(log_U_from_O32(1.0, _Z_ZSUN_PUBLISHED[1] + 0.05))
    assert np.isnan(icf_NpOp(-2.0, _Z_ZSUN_PUBLISHED[1] + 0.05))


def test_other_axes_still_hard_rejected():
    """log(O32) and log(N43) are the cubic axes and keep their bounds.

    They must reject even when the metallicity sits in the newly allowed
    strip, so the relaxation cannot leak into the axes where the surfaces
    genuinely diverge.
    """
    assert np.isnan(log_U_from_O32(_LOG_O32_VALID[1] + 0.5, Z_131))
    assert np.isnan(log_U_from_O32(_LOG_O32_VALID[0] - 0.5, Z_131))
    assert np.isnan(log_U_from_N43(_LOG_N43_VALID[1] + 0.5, Z_131))


class TestExtrapolationFlag:
    """is_Z_extrapolated marks the strip, and only the strip."""

    def test_strip_is_flagged(self):
        assert is_Z_extrapolated(Z_131)
        assert is_Z_extrapolated(_Z_ZSUN_VALID[0])

    def test_published_range_is_not_flagged(self):
        assert not is_Z_extrapolated(Z_FLOOR)
        assert not is_Z_extrapolated(0.10)
        assert not is_Z_extrapolated(_Z_ZSUN_PUBLISHED[1])

    def test_rejected_values_are_not_flagged(self):
        """Below the enforced floor the value is not used at all."""
        assert not is_Z_extrapolated(0.005)

    def test_missing_values_are_not_flagged(self):
        assert not is_Z_extrapolated(None)
        assert not is_Z_extrapolated(np.nan)


class TestExtrapolationIsSmall:
    """The size of the extrapolation, which is why it is allowed."""

    @pytest.mark.parametrize("log_O32", [-0.5, 0.0, 0.5, 1.0, 1.5])
    @pytest.mark.parametrize("ne", [1e2, 1e3, 1e4, 1e5, 1e6])
    def test_logU_shift_under_0p06_dex(self, log_O32, ne):
        """log(U) at object 131 differs from the floor by <= 0.06 dex."""
        shift = log_U_from_O32(log_O32, Z_131, ne) - log_U_from_O32(log_O32, Z_FLOOR, ne)
        assert abs(shift) < 0.06

    @pytest.mark.parametrize("icf", [icf_NpOp, icf_NppOpp, icf_NpNpp_OpOpp])
    def test_icf_shift_under_0p01_dex(self, icf):
        """The N/O ICFs move by <= 0.01 dex across the same step."""
        a = icf(-2.0, Z_131)
        b = icf(-2.0, Z_FLOOR)
        assert abs(np.log10(a / b)) < 0.01

    def test_surface_saturates_rather_than_diverging(self):
        """Towards Z = 0 the O32 surface flattens; it does not run away.

        This is the property that makes the extrapolation safe: the fit is
        a polynomial in *linear* Z/Zsun, so the high-order terms vanish as
        y -> 0 instead of dominating.
        """
        zs = np.array([Z_FLOOR, Z_131, 0.02, 0.01, 0.0])
        vals = np.array([log_U_from_O32(0.5, z) if z >= _Z_ZSUN_VALID[0]
                         else _bare_surface(0.5, z) for z in zs])
        assert np.all(np.isfinite(vals))
        # Monotonic decline, and the whole run below the floor is small.
        assert np.all(np.diff(vals) < 0)
        assert abs(vals[-1] - vals[0]) < 0.10
        # Successive steps shrink — the surface is flattening out.
        steps = np.abs(np.diff(vals))
        assert steps[-1] < steps[0]


def _bare_surface(x: float, y: float) -> float:
    """Evaluate the ne=100 O32 surface with no validity gate at all."""
    from jwspecabund.martinez25_icf import _COEFF_LOG_U_O32, _bicubic

    return _bicubic(x, y, _COEFF_LOG_U_O32[2])
