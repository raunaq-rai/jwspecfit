"""Tests for extrapolating a little past the log(U) calibration edges.

log(U) is the *cubic* axis of the Martinez+2025 surfaces, unlike the
metallicity axis (linear in y), so it genuinely runs away -- but only
eventually.  A small margin is allowed, a warning is raised, and the cost
is reported surface by surface so the caller can see whether it mattered.
"""

import logging

import numpy as np
import pytest

from jwspecabund.martinez25_icf import (
    _LOG_U_VALID,
    LOG_U_EXTRAP_MARGIN,
    icf_CppOpp,
    icf_NpNpp_OpOpp,
    icf_NpOp,
    icf_NppOpp,
    is_logU_extrapolated,
    logU_extrapolation_report,
    reset_extrapolation_warnings,
)

LO, HI = _LOG_U_VALID
Z = 0.10
NE = 1e3
#: The case that prompted this: forced O32 on a low-ionisation object.
REAL = -3.507


@pytest.fixture(autouse=True)
def _rearm():
    """Warnings are once-per-process; re-arm so each test sees its own."""
    reset_extrapolation_warnings()
    yield
    reset_extrapolation_warnings()


class TestFlag:
    def test_band_is_flagged(self):
        assert is_logU_extrapolated(REAL)
        assert is_logU_extrapolated(LO - LOG_U_EXTRAP_MARGIN)
        assert is_logU_extrapolated(HI + LOG_U_EXTRAP_MARGIN / 2)

    def test_inside_calibration_is_not_flagged(self):
        assert not is_logU_extrapolated(-2.0)
        assert not is_logU_extrapolated(LO)
        assert not is_logU_extrapolated(HI)

    def test_beyond_margin_is_not_flagged(self):
        """Past the margin the value is rejected, not extrapolated."""
        assert not is_logU_extrapolated(LO - 2 * LOG_U_EXTRAP_MARGIN)

    def test_missing_is_not_flagged(self):
        assert not is_logU_extrapolated(None)
        assert not is_logU_extrapolated(np.nan)


class TestAcceptance:
    @pytest.mark.parametrize("icf", [icf_NpOp, icf_NppOpp, icf_NpNpp_OpOpp])
    def test_inside_margin_returns_a_number(self, icf):
        assert np.isfinite(icf(REAL, Z, NE))
        assert np.isfinite(icf(LO - LOG_U_EXTRAP_MARGIN, Z, NE))

    @pytest.mark.parametrize("icf", [icf_NpOp, icf_NppOpp, icf_NpNpp_OpOpp])
    def test_beyond_margin_still_rejected(self, icf):
        assert np.isnan(icf(LO - LOG_U_EXTRAP_MARGIN - 0.01, Z, NE))
        assert np.isnan(icf(HI + LOG_U_EXTRAP_MARGIN + 0.01, Z, NE))

    def test_both_edges(self):
        assert np.isfinite(icf_NpOp(HI + LOG_U_EXTRAP_MARGIN / 2, Z, NE))

    def test_extrapolated_value_differs_from_the_edge(self):
        """Extrapolating, not clamping."""
        assert icf_NpOp(REAL, Z, NE) != pytest.approx(icf_NpOp(LO, Z, NE))

    def test_margin_zero_restores_strict_rejection(self, monkeypatch):
        monkeypatch.setattr(
            "jwspecabund.martinez25_icf.LOG_U_EXTRAP_MARGIN", 0.0,
        )
        assert np.isnan(icf_NpOp(REAL, Z, NE))


class TestWarning:
    def test_warns_on_first_extrapolation(self, caplog):
        with caplog.at_level(logging.WARNING, logger="jwspecabund.martinez25_icf"):
            icf_NpOp(REAL, Z, NE)
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "extrapolating" in msg
        assert "0.007" in msg          # the overshoot is named
        assert "log(U)" in msg

    def test_repeat_warnings_are_suppressed(self, caplog):
        """An MC run evaluates several surfaces per draw; one warning only."""
        with caplog.at_level(logging.WARNING, logger="jwspecabund.martinez25_icf"):
            for _ in range(200):
                icf_NpOp(REAL, Z, NE)
                icf_NppOpp(REAL, Z, NE)
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1

    def test_reset_rearms(self, caplog):
        with caplog.at_level(logging.WARNING, logger="jwspecabund.martinez25_icf"):
            icf_NpOp(REAL, Z, NE)
            reset_extrapolation_warnings()
            icf_NpOp(REAL, Z, NE)
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 2

    def test_rejection_beyond_margin_still_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="jwspecabund.martinez25_icf"):
            icf_NpOp(LO - 1.0, Z, NE)
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "rejected" in msg


class TestReport:
    def test_reports_the_overshoot_and_edge(self):
        r = logU_extrapolation_report(REAL, Z, NE)
        assert r["edge"] == pytest.approx(LO)
        assert r["overshoot"] == pytest.approx(0.007, abs=1e-6)

    def test_covers_every_surface(self):
        r = logU_extrapolation_report(REAL, Z, NE)
        assert len(r["surfaces"]) == 6
        assert "C2+/O2+" in r["surfaces"] and "N+/O+" in r["surfaces"]

    def test_the_real_case_is_cheap(self):
        """0.007 dex of overshoot must not move anything meaningfully."""
        r = logU_extrapolation_report(REAL, Z, NE)
        assert r["worst"] < 0.02
        assert all(abs(v) < 0.02 for v in r["surfaces"].values())

    def test_worst_surface_is_identified(self):
        r = logU_extrapolation_report(REAL, Z, NE)
        worst = max(r["surfaces"], key=lambda k: abs(r["surfaces"][k]))
        assert r["worst_surface"] == worst
        assert r["worst"] == pytest.approx(abs(r["surfaces"][worst]))

    def test_cost_grows_with_overshoot(self):
        """The point of reporting it: it is not constant."""
        near = logU_extrapolation_report(LO - 0.005, Z, NE)["worst"]
        far = logU_extrapolation_report(LO - 0.05, Z, NE)["worst"]
        assert far > near * 3

    def test_upper_edge_reports_against_the_upper_edge(self):
        r = logU_extrapolation_report(HI + 0.02, Z, NE)
        assert r["edge"] == pytest.approx(HI)
        assert r["overshoot"] == pytest.approx(0.02)

    def test_flags_the_dangerous_corner(self):
        """C/O at high Z and high ne is where extrapolating actually hurts."""
        r = logU_extrapolation_report(LO - 0.05, 0.40, 1e5)
        assert abs(r["surfaces"]["C2+/O2+"]) > 0.2
        assert r["worst_surface"] == "C2+/O2+"


class TestNonPositiveGuard:
    """A negative ICF is a meaningless abundance, not a small error."""

    def test_default_margin_never_reaches_the_zero_crossing(self):
        """C/O crosses zero ~0.09 dex out; the 0.05 margin stops short."""
        for m in np.linspace(0.0, LOG_U_EXTRAP_MARGIN, 11):
            assert icf_CppOpp(LO - m, 0.40, 1e5) > 0

    def test_non_positive_extrapolation_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "jwspecabund.martinez25_icf.LOG_U_EXTRAP_MARGIN", 0.30,
        )
        val = icf_CppOpp(LO - 0.25, 0.40, 1e5)
        assert np.isnan(val)
