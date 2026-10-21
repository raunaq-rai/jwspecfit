"""Tests for pinning the density at which log(U) is derived.

``logU_ne`` decides which of the Martinez+2025 log n_e = 2..6 calibration
grids the ratio -> log(U) conversion is read off, replacing the measured
high-ionisation-zone density.  The contract is narrow on purpose: it
moves log(U) and nothing else.  The ionic abundances, the zone
temperatures and densities, and the ICF surfaces all keep the measured
values, so a pinned run and an unpinned one differ only where log(U)
enters.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from jwspecabund import compute_abundances
from jwspecabund._core import _compute_logU

from .test_niii_multiplet import niii_split


@dataclass
class _Line:
    flux: float
    flux_err: float


class _MockFitResult:
    def __init__(self, fluxes, errors):
        self.lines = {k: _Line(fluxes[k], errors[k]) for k in fluxes}


def _result(Te=18000.0, ne=1000.0, O_pp=3.0e-5, O_p=2.0e-6,
            with_n43=True, frac=0.03):
    """A synthetic object carrying both log(U) diagnostics.

    Mirrors the fixture in ``test_logU_diagnostic.py``: oxygen from PyNEB
    so O/H and T_e are self-consistent, nitrogen doublets set directly to
    a chosen N43.
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
        "OII_3726": _f(O_p, o2.getEmissivity(Te, ne, wave=3726)),
        "OII_3729": _f(O_p, o2.getEmissivity(Te, ne, wave=3729)),
    }
    if with_n43:
        # N43 = 0.25 (log N43 = -0.61), inside the calibrated [-3.0, 0.5];
        # N IV] at its low-density ratio so ne(high) stays modest.
        fluxes["NIV_1483"] = 2.0
        fluxes["NIV_1486"] = 2.96
        fluxes.update(niii_split(20.0))
    return _MockFitResult(fluxes, {k: frac * v for k, v in fluxes.items()})


def _run(res=None, **kw):
    return compute_abundances(
        res if res is not None else _result(), z=6.0, method="direct",
        dust_correct=False, n_mc=60, progress=False, **kw,
    )


# Each fixture is a full pipeline run; cache them for the module.
@pytest.fixture(scope="module")
def o32_free():
    return _run(_result(with_n43=False), logU_diagnostic="O32")


@pytest.fixture(scope="module")
def o32_pin_lo():
    return _run(_result(with_n43=False), logU_diagnostic="O32", logU_ne=100.0)


@pytest.fixture(scope="module")
def o32_pin_hi():
    return _run(_result(with_n43=False), logU_diagnostic="O32", logU_ne=1.0e5)


@pytest.fixture(scope="module")
def n43_free():
    return _run(logU_diagnostic="N43")


@pytest.fixture(scope="module")
def n43_pin_hi():
    return _run(logU_diagnostic="N43", logU_ne=1.0e5)


# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("bad", [0.0, -1.0, -1e4, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_raises(self, bad):
        with pytest.raises(ValueError, match="logU_ne"):
            compute_abundances(None, z=6.0, logU_ne=bad)

    def test_error_names_the_offending_value(self):
        with pytest.raises(ValueError, match="positive, finite"):
            compute_abundances(None, z=6.0, logU_ne=-5.0)

    def test_none_is_the_default(self, o32_free):
        assert o32_free.logU is not None

    def test_a_positive_density_is_accepted(self, o32_pin_lo):
        assert o32_pin_lo.logU is not None

    @pytest.mark.parametrize("off_grid", [1.0, 50.0, 1.0e7])
    def test_off_grid_density_warns(self, off_grid, caplog):
        """_interp_ne clips silently; the user must be told."""
        with caplog.at_level("WARNING", logger="jwspecabund._core"):
            _run(_result(with_n43=False), logU_diagnostic="O32",
                 logU_ne=off_grid)
        assert any("outside the Martinez+2025 grid" in r.message
                   for r in caplog.records)

    @pytest.mark.parametrize("on_grid", [100.0, 1.0e4, 1.0e6])
    def test_on_grid_density_does_not_warn(self, on_grid, caplog):
        with caplog.at_level("WARNING", logger="jwspecabund._core"):
            _run(_result(with_n43=False), logU_diagnostic="O32",
                 logU_ne=on_grid)
        assert not any("outside the Martinez+2025 grid" in r.message
                       for r in caplog.records)


class TestUnitOverride:
    """``logU_ne`` must be exactly equivalent to passing that ne_high."""

    FLUXES = {
        "OIII_5007": 100.0, "OII_3726": 6.0, "OII_3729": 6.0,
        "NIV_1483": 2.0, "NIV_1486": 2.96,
        **niii_split(20.0),
    }

    @pytest.mark.parametrize("pin", [100.0, 1e3, 1e4, 1e5])
    @pytest.mark.parametrize("diag", ["O32", "N43"])
    def test_pin_equals_passing_that_density(self, pin, diag):
        pinned, _ = _compute_logU(
            self.FLUXES, 0.1, ne_high=1e6, lock_diag=diag, logU_ne=pin,
        )
        direct, _ = _compute_logU(
            self.FLUXES, 0.1, ne_high=pin, lock_diag=diag,
        )
        assert pinned == pytest.approx(direct)

    def test_none_leaves_ne_high_in_charge(self):
        a, _ = _compute_logU(self.FLUXES, 0.1, ne_high=1e4, lock_diag="O32")
        b, _ = _compute_logU(
            self.FLUXES, 0.1, ne_high=1e4, lock_diag="O32", logU_ne=None,
        )
        assert a == pytest.approx(b)

    def test_pin_ignores_ne_high_entirely(self):
        """Two very different ne_high, same pin -> same log(U)."""
        a, _ = _compute_logU(
            self.FLUXES, 0.1, ne_high=100.0, lock_diag="O32", logU_ne=1e4,
        )
        b, _ = _compute_logU(
            self.FLUXES, 0.1, ne_high=1e6, lock_diag="O32", logU_ne=1e4,
        )
        assert a == pytest.approx(b)

    def test_o32_is_density_sensitive(self):
        """The premise of the whole feature."""
        lo, _ = _compute_logU(
            self.FLUXES, 0.1, ne_high=1e3, lock_diag="O32", logU_ne=1e2,
        )
        hi, _ = _compute_logU(
            self.FLUXES, 0.1, ne_high=1e3, lock_diag="O32", logU_ne=1e5,
        )
        assert abs(lo - hi) > 0.05

    def test_n43_is_density_insensitive(self):
        """Martinez+2025's reason for preferring N43."""
        lo, _ = _compute_logU(
            self.FLUXES, 0.1, ne_high=1e3, lock_diag="N43", logU_ne=1e2,
        )
        hi, _ = _compute_logU(
            self.FLUXES, 0.1, ne_high=1e3, lock_diag="N43", logU_ne=1e5,
        )
        assert abs(lo - hi) < abs(
            _compute_logU(self.FLUXES, 0.1, 1e3, lock_diag="O32", logU_ne=1e2)[0]
            - _compute_logU(self.FLUXES, 0.1, 1e3, lock_diag="O32", logU_ne=1e5)[0]
        )

    def test_pin_does_not_change_which_diagnostic_is_used(self):
        """Pinning is orthogonal to the N43 -> O32 priority tree."""
        _, free = _compute_logU(self.FLUXES, 0.1, ne_high=1e3)
        _, pinned = _compute_logU(self.FLUXES, 0.1, ne_high=1e3, logU_ne=1e5)
        assert free == pinned == "N43"


class TestEndToEnd:
    def test_pinning_moves_o32_log_u(self, o32_pin_lo, o32_pin_hi):
        assert abs(o32_pin_lo.logU - o32_pin_hi.logU) > 0.05

    def test_pinning_barely_moves_n43_log_u(self, n43_free, n43_pin_hi):
        assert abs(n43_free.logU - n43_pin_hi.logU) < 0.05

    def test_measured_density_is_untouched(self, o32_free, o32_pin_hi):
        """The pin is for the conversion only, not the density solve."""
        assert o32_pin_hi.ne_high == pytest.approx(o32_free.ne_high)
        assert o32_pin_hi.ne_low == pytest.approx(o32_free.ne_low)

    def test_oxygen_abundance_is_untouched(self, o32_free, o32_pin_hi):
        """O/H does not depend on log(U); pinning must not perturb it."""
        assert o32_pin_hi.OH == pytest.approx(o32_free.OH, abs=1e-6)

    def test_temperature_is_untouched(self, o32_free, o32_pin_hi):
        assert o32_pin_hi.Te_high == pytest.approx(o32_free.Te_high)

    def test_log_u_still_gets_an_error_bar(self, o32_pin_hi):
        assert o32_pin_hi.logU_err is not None


class TestDiagnostics:
    KEY = "log(U) density"

    def test_absent_when_not_pinned(self, o32_free):
        assert self.KEY not in o32_free.diagnostics

    def test_present_when_pinned(self, o32_pin_hi):
        assert self.KEY in o32_pin_hi.diagnostics

    def test_names_the_pinned_density(self, o32_pin_hi):
        assert "100000 cm^-3" in o32_pin_hi.diagnostics[self.KEY]

    def test_names_the_argument(self, o32_pin_hi):
        assert "logU_ne" in o32_pin_hi.diagnostics[self.KEY]

    def test_names_the_measured_density_it_replaced(self, o32_free, o32_pin_hi):
        txt = o32_pin_hi.diagnostics[self.KEY]
        assert f"{o32_free.ne_high:.0f} cm^-3" in txt

    def test_reports_the_shift(self, o32_pin_hi):
        txt = o32_pin_hi.diagnostics[self.KEY]
        assert "shift of" in txt and "dex" in txt

    def test_the_reported_shift_is_the_real_one(self, o32_free, o32_pin_hi):
        """The quoted difference must match the unpinned run."""
        txt = o32_pin_hi.diagnostics[self.KEY]
        quoted = float(txt.split("shift of")[1].split("dex")[0].strip())
        assert quoted == pytest.approx(o32_pin_hi.logU - o32_free.logU, abs=0.01)

    def test_log_u_entry_still_names_the_diagnostic(self, o32_pin_hi):
        assert "O32" in o32_pin_hi.diagnostics["log(U)"]

    def test_log_u_entry_does_not_claim_ne_high_when_pinned(self, o32_pin_hi):
        """The O32 density-sensitivity note must point at the pin."""
        txt = o32_pin_hi.diagnostics["log(U)"]
        assert "pinned logU_ne" in txt and "at ne(high)" not in txt

    def test_log_u_entry_still_says_ne_high_when_not_pinned(self, o32_free):
        assert "at ne(high)" in o32_free.diagnostics["log(U)"]


class TestMonteCarlo:
    def test_results_are_finite(self, o32_pin_hi):
        """Pinning must not poison the draws."""
        assert np.isfinite(o32_pin_hi.logU)
        assert np.isfinite(o32_pin_hi.OH)

    def test_error_bar_is_finite_and_positive(self, o32_pin_hi):
        err = o32_pin_hi.logU_err
        err = max(err) if isinstance(err, tuple) else err
        assert np.isfinite(err) and err > 0

    def test_oxygen_posterior_is_untouched(self, o32_free, o32_pin_hi):
        """Pinning moves log(U) only -- the O/H draws are identical."""
        assert o32_pin_hi.OH_err == pytest.approx(o32_free.OH_err, abs=1e-9)
        assert np.allclose(o32_pin_hi.OH_posterior, o32_free.OH_posterior)
