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
