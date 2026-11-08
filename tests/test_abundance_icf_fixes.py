"""Regression tests for the 2026-11 abundance/ICF audit."""

import numpy as np
import pytest

from jwspecabund.direct import compute_total_abundances


def _ionic(**extra):
    base = {"O+/H+": 1.0e-5, "O++/H+": 9.0e-5}
    base.update(extra)
    return base


class TestSulfurNeedsBothIons:
    """Izotov+06 Eq. 20 is ICF(S+ + S2+); it has no S+-only form."""

    def test_s_plus_only_gives_no_s_over_o(self):
        tot = compute_total_abundances(_ionic(**{"S+/H+": 2.0e-7}), icf_method="izotov06")
        assert "S/O" not in tot
        assert "S++" in tot["_failures"]["S/O"]

    def test_s_pp_only_gives_no_s_over_o(self):
        tot = compute_total_abundances(_ionic(**{"S++/H+": 8.0e-7}), icf_method="izotov06")
        assert "S/O" not in tot
        assert "S+ " in tot["_failures"]["S/O"]

    def test_both_ions_unchanged(self):
        from jwspecabund.icf import icf_sulfur
        ionic = _ionic(**{"S+/H+": 2.0e-7, "S++/H+": 8.0e-7})
        tot = compute_total_abundances(ionic, icf_method="izotov06")
        expected = icf_sulfur(1e-5, 1e-4) * 1.0e-6 / 1.0e-4
        assert tot["S/O"] == pytest.approx(expected, rel=1e-12, abs=0)


class TestGarnettCarbonNotDoubleCounted:
    """C+/C = O+/O implies C/O = (C2+ + C3+)/O2+ exactly."""

    @pytest.mark.parametrize("w", [0.95, 0.8, 0.5])
    def test_recovers_true_c_over_o(self, w):
        O_tot, CO_true = 1.0e-4, 0.3
        O_pp, O_p = w * O_tot, (1 - w) * O_tot
        # Carbon ionised like oxygen; C3+ folded into the "upper" fraction.
        C_tot = CO_true * O_tot
        ionic = {"O+/H+": O_p, "O++/H+": O_pp,
                 "C++/H+": 0.8 * w * C_tot, "C+++/H+": 0.2 * w * C_tot}
        tot = compute_total_abundances(ionic, co_icf_method="garnett97")
        assert tot["CO_method"] == "garnett97_icf"
        assert tot["C/O"] == pytest.approx(CO_true, rel=1e-12, abs=0)


class TestNVEmissivity:
    """N V 1239/1243 collisional emissivity (forward._nv_emissivity)."""

    def test_low_density_limit_matches_analytic(self):
        # n_e -> 0: every excitation decays radiatively, eps = q_13 h nu.
        from jwspecabund.forward import _nv_emissivity
        T = 1.5e4
        q13 = 8.629e-6 / (2 * np.sqrt(T)) * 4.32 * np.exp(-80721.9 * 1.4387769 / T)
        hnu = 6.62607015e-27 * 2.99792458e10 / 1238.821e-8
        # Upsilon interpolates in log T between 1.25e4 and 2.5e4 K.
        assert _nv_emissivity(T, 1.0, 1239) == pytest.approx(q13 * hnu, rel=0.01, abs=0)

    def test_boltzmann_temperature_dependence(self):
        from jwspecabund.forward import _nv_emissivity
        r = _nv_emissivity(2.0e4, 1e2, 1239) / _nv_emissivity(1.0e4, 1e2, 1239)
        # exp(-dE/k (1/2e4 - 1/1e4)) / sqrt(2) ~ 230
        assert 150 < r < 350

    def test_n4plus_excluded_from_n_over_o(self):
        ionic = {"O+/H+": 1e-5, "O++/H+": 9e-5, "N+/H+": 1e-6,
                 "N++/H+": 2e-6, "N4+/H+": 5e-6}
        with_nv = compute_total_abundances(ionic, icf_method="direct_sum")
        ionic.pop("N4+/H+")
        without = compute_total_abundances(ionic, icf_method="direct_sum")
        assert with_nv["N/O"] == without["N/O"]
        assert with_nv["N/O_UV_raw"] == without["N/O_UV_raw"]
