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
