"""Tests for forcing the ionisation-parameter diagnostic.

``logU_diagnostic`` overrides the automatic N43 -> O32 priority tree.
The contract is that forcing means *no fallback*: if the named
diagnostic cannot be formed, log(U) is unavailable rather than quietly
switching to the other one, so a run cannot silently mix diagnostics
across a sample.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from jwspecabund import VALID_LOGU_DIAGNOSTICS, compute_abundances

from .test_niii_multiplet import niii_split


@dataclass
class _Line:
    flux: float
    flux_err: float


class _MockFitResult:
    def __init__(self, fluxes, errors):
        self.lines = {k: _Line(fluxes[k], errors[k]) for k in fluxes}


def _result(Te=18000.0, ne=1000.0, O_pp=3.0e-5, O_p=2.0e-6,
            with_n43=True, with_o32=True, frac=0.03):
    """A synthetic object carrying both log(U) diagnostics.

    Oxygen lines come from PyNEB so O/H and T_e are self-consistent; the
    nitrogen doublets are set directly to a chosen N43, which is all the
    log(U) diagnostic needs and keeps the test deterministic.
    """
    import pyneb as pn

    from jwspecabund.direct import _get_oiii_atom

    o3 = _get_oiii_atom()
    o2 = pn.Atom("O", 2)
    jHb = pn.RecAtom("H", 1).getEmissivity(Te, ne, lev_i=4, lev_j=2)

    def _f(ion, emis):
        return 100.0 * ion * emis / jHb

    fluxes = {
        "HBETA": 100.0,
        "OIII_4363": _f(O_pp, o3.getEmissivity(Te, ne, wave=4363)),
        "OIII_4959": _f(O_pp, o3.getEmissivity(Te, ne, wave=4959)),
        "OIII_5007": _f(O_pp, o3.getEmissivity(Te, ne, wave=5007)),
    }
    if with_o32:
        fluxes["OII_3726"] = _f(O_p, o2.getEmissivity(Te, ne, wave=3726))
        fluxes["OII_3729"] = _f(O_p, o2.getEmissivity(Te, ne, wave=3729))
    if with_n43:
        # N43 = (1483 + 1486) / NIII] 1750 = 0.25 -> log N43 = -0.61,
        # comfortably inside the calibrated [-3.0, 0.5].  The N IV] doublet
        # sits at its low-density ratio (1486/1483 = 1.48) so ne(high) stays
        # modest -- O32 is evaluated there, and a high ne(high) would push
        # log(U) below the -3.5 validity floor and make the comparison moot.
        fluxes["NIV_1483"] = 2.0
        fluxes["NIV_1486"] = 2.96
        fluxes.update(niii_split(20.0))
    return _MockFitResult(fluxes, {k: frac * v for k, v in fluxes.items()})


def _run(res=None, **kw):
    return compute_abundances(
        res if res is not None else _result(), z=6.0, method="direct",
        dust_correct=False, n_mc=60, progress=False, **kw,
    )


# Each call below is a full pipeline run, so cache the handful of distinct
# configurations for the module rather than re-running them per test.
@pytest.fixture(scope="module")
def auto():
    return _run()


@pytest.fixture(scope="module")
def forced_o32():
    return _run(logU_diagnostic="O32")


@pytest.fixture(scope="module")
def forced_n43():
    return _run(logU_diagnostic="N43")


@pytest.fixture(scope="module")
def no_oii():
    """No [O II], so O32 cannot be formed; N43 still can."""
    return {
        "auto": _run(_result(with_o32=False)),
        "forced": _run(_result(with_o32=False), logU_diagnostic="O32"),
    }


@pytest.fixture(scope="module")
def no_nitrogen():
    """No N IV]/N III], so N43 cannot be formed; O32 still can."""
    return {
        "auto": _run(_result(with_n43=False)),
        "forced": _run(_result(with_n43=False), logU_diagnostic="N43"),
    }


class TestValidation:
    def test_valid_options_exported(self):
        assert VALID_LOGU_DIAGNOSTICS == ("N43", "O32", "O32_ionic")

    @pytest.mark.parametrize("bad", ["o32", "O32 ", "n43", "OIII", ""])
    def test_invalid_string_raises(self, bad):
        with pytest.raises(ValueError, match="logU_diagnostic"):
            compute_abundances(None, z=6.0, logU_diagnostic=bad)

    def test_none_is_accepted(self, auto):
        assert auto.logU is not None


class TestForcing:
    def test_default_prefers_n43(self, auto):
        """The automatic tree picks the density-insensitive diagnostic."""
        assert "N43" in auto.diagnostics["log(U)"]

    def test_force_o32_switches_diagnostic(self, forced_o32):
        d = forced_o32.diagnostics["log(U)"]
        assert "O32" in d and "N43" not in d

    def test_force_o32_changes_the_value(self, auto, forced_o32):
        """Forcing is not cosmetic: the two diagnostics disagree here."""
        assert auto.logU is not None and forced_o32.logU is not None
        assert abs(auto.logU - forced_o32.logU) > 0.01

    def test_force_n43_matches_the_automatic_choice(self, auto, forced_n43):
        """N43 is what the tree would have picked anyway."""
        assert forced_n43.logU == pytest.approx(auto.logU)

    def test_forcing_is_recorded_in_diagnostics(self, forced_o32):
        d = forced_o32.diagnostics["log(U)"]
        assert "logU_diagnostic" in d and "no fallback" in d
        assert "density-sensitive" in d


class TestNoSilentFallback:
    """The whole point: a forced diagnostic never falls through."""

    def test_forced_o32_without_oii_gives_no_logU(self, no_oii):
        out = no_oii["forced"]
        assert out.logU is None
        assert "not available" in out.diagnostics["log(U)"]
        assert "no fallback" in out.diagnostics["log(U)"]

    def test_that_object_would_have_had_logU_automatically(self, no_oii):
        """Confirms the previous test failed for the right reason."""
        assert no_oii["auto"].logU is not None

    def test_forced_n43_without_nitrogen_gives_no_logU(self, no_nitrogen):
        assert no_nitrogen["forced"].logU is None
        assert no_nitrogen["auto"].logU is not None


class TestErrorBar:
    """A forced diagnostic must not silently widen the uncertainty.

    ``AbundanceResult`` carries no log(U) posterior array, so the
    observable is ``logU_err``: draws mixing N43 and O32 -- which differ
    by ~1 dex on this object -- would inflate it far past what 3% flux
    errors can justify.
    """

    def test_forced_error_bar_stays_tight(self, forced_o32):
        err = forced_o32.logU_err
        err = max(err) if isinstance(err, tuple) else err
        assert np.isfinite(err)
        assert 0.0 < err < 0.2

    def test_forced_value_is_inside_the_calibration_range(self, forced_o32):
        """Otherwise every downstream ICF would reject it."""
        from jwspecabund.martinez25_icf import _LOG_U_VALID

        assert _LOG_U_VALID[0] <= forced_o32.logU <= _LOG_U_VALID[1]

    def test_the_two_diagnostics_really_do_disagree_here(self, auto, forced_o32):
        """Confirms the tightness above is purity, not a degenerate fixture."""
        assert abs(auto.logU - forced_o32.logU) > 0.5
