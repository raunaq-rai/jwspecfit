"""Tests for the Martinez (in prep.) C/O ICF: C2+/O2+ -> C/O.

Reference values are computed directly from the coefficient table Zorayda
Martinez provided (``CO_FITS_TAB.csv``), evaluated with her published
bicubic form ``getCO_ICF_M25``.  The package's ``icf_CppOpp`` must
reproduce them.
"""
import numpy as np
import pytest

from jwspecabund.martinez25_icf import (
    LOG_OH_SOLAR,
    _LOG_U_VALID,
    _Z_ZSUN_VALID_CO,
    icf_CppOpp,
)


def _Z(LOH):
    return 10 ** (LOH - 12.0) / 10 ** (LOG_OH_SOLAR - 12.0)


class TestCOICFValues:
    """icf_CppOpp reproduces Zorayda's reference values at grid densities."""

    @pytest.mark.parametrize(
        "logU, LOH, log_ne, expected",
        [
            (-2.5, 8.0, 3.0, 0.8743604615),  # D3
            (-2.0, 7.7, 4.0, 1.0696889486),  # D4
            (-3.0, 8.3, 2.0, 0.5947842214),  # D2, wider-Z than N/O range
            (-2.5, 8.0, 4.0, 0.8700774446),  # D4
        ],
    )
    def test_matches_reference(self, logU, LOH, log_ne, expected):
        got = icf_CppOpp(logU, _Z(LOH), ne=10 ** log_ne)
        assert got == pytest.approx(expected, abs=1e-6)

    def test_density_interpolates_linearly_in_log_ne(self):
        # Halfway (in log n_e) between D3 and D4 at logU=-2.5, LOH=8.0.
        got = icf_CppOpp(-2.5, _Z(8.0), ne=10 ** 3.5)
        assert got == pytest.approx(0.8722189531, abs=1e-6)

    def test_linear_not_log_convention(self):
        # C2+/O2+ tracks the N/O NppOpp *linear* ICF (both ~0.85-1.0 at
        # typical conditions); if it were log(ICF) it would sit near 0.
        from jwspecabund.martinez25_icf import icf_NppOpp
        co = icf_CppOpp(-2.5, _Z(8.0), ne=1e3)
        no = icf_NppOpp(-2.5, _Z(8.0), ne=1e3)
        assert 0.5 < co < 1.5
        assert abs(co - no) < 0.3


class TestCOICFValidity:
    """Out-of-range inputs are rejected (NaN), matching the calibration."""

    def test_logU_out_of_range_rejected(self):
        assert np.isnan(icf_CppOpp(_LOG_U_VALID[0] - 2.0, _Z(8.0)))
        assert np.isnan(icf_CppOpp(_LOG_U_VALID[1] + 2.0, _Z(8.0)))

    def test_Z_out_of_range_rejected(self):
        # Below and above the C/O metallicity validity (wider than N/O).
        assert np.isnan(icf_CppOpp(-2.5, _Z_ZSUN_VALID_CO[0] * 0.3))
        assert np.isnan(icf_CppOpp(-2.5, _Z_ZSUN_VALID_CO[1] * 2.0))

    def test_wider_Z_than_NO(self):
        # LOH=8.5 (Z~0.65) is valid for C/O but outside the N/O Z clamp.
        from jwspecabund.martinez25_icf import _Z_ZSUN_VALID
        z = _Z(8.5)
        assert z > _Z_ZSUN_VALID[1]          # outside N/O range
        assert z < _Z_ZSUN_VALID_CO[1]       # inside C/O range
        assert np.isfinite(icf_CppOpp(-2.5, z))
