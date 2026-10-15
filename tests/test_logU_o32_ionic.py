"""Tests for the ``"O32_ionic"`` log(U) diagnostic (Martinez+2025 App. C).

O32_ionic uses the same two lines as O32 but converts them to ionic
abundances first, each at the density of its own zone, and reads log(U)
off the O2+/O+ surface of Martinez+2025 Table 8.  It is opt-in only:
the automatic N43 -> O32 tree never selects it, and when forced there is
no fallback to O32.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from jwspecabund import compute_abundances, log_U_from_O32_ionic
from jwspecabund._core import _compute_logU
from jwspecabund.martinez25_icf import (
    _COEFF_LOG_U_O32_IONIC,
    _LOG_NE_GRID,
    LOG_OH_SOLAR,
)

# Martinez+2025 Table 8, transcribed independently of the module.
_TABLE_8 = {
    2: [-3.01893, 0.68142, 0.39134, 0.40444, 0.13516, 1.93922, -0.38687, 0.11450, 0.07717, -3.95339],
    3: [-3.02826, 0.67769, 0.47521, 0.36650, 0.08499, 1.24708, -0.37495, 0.10476, 0.02243, -2.87921],
    4: [-3.03228, 0.68678, 0.47450, 0.36517, 0.08376, 1.01559, -0.41327, 0.11398, 0.00769, -2.47978],
    5: [-3.03319, 0.68770, 0.41954, 0.34904, 0.08278, 1.14266, -0.39662, 0.11960, 0.00657, -2.61133],
    6: [-3.03090, 0.68578, 0.30140, 0.25356, 0.07391, 1.42633, -0.25902, 0.13938, 0.01181, -2.90696],
}


@dataclass
class _Line:
    flux: float
    flux_err: float


class _MockFitResult:
    def __init__(self, fluxes, errors):
        self.lines = {k: _Line(fluxes[k], errors[k]) for k in fluxes}


def _result(Te=18000.0, ne=1000.0, O_pp=3.0e-5, O_p=2.0e-6, N_p=1.0e-7,
            C_pp=4.0e-6, with_oii=True, frac=0.03):
    """A synthetic object with oxygen, [N II] and C III] from PyNEB.

    Carries everything the O32_ionic log(U), the N+/O+ N/O and the C/O
    ICFs need, and no N IV]/N III], so the automatic tree lands on O32.
    """
    import pyneb as pn

    from jwspecabund.direct import _get_oiii_atom

    o3 = _get_oiii_atom()
    o2 = pn.Atom("O", 2)
    n2 = pn.Atom("N", 2)
    c3 = pn.Atom("C", 3)
    jHb = pn.RecAtom("H", 1).getEmissivity(Te, ne, lev_i=4, lev_j=2)

    def _f(ion, emis):
        return 100.0 * ion * emis / jHb

    fluxes = {
        "HBETA": 100.0,
        "OIII_4363": _f(O_pp, o3.getEmissivity(Te, ne, wave=4363)),
        "OIII_4959": _f(O_pp, o3.getEmissivity(Te, ne, wave=4959)),
        "OIII_5007": _f(O_pp, o3.getEmissivity(Te, ne, wave=5007)),
        "OIII_1661": _f(O_pp, o3.getEmissivity(Te, ne, lev_i=6, lev_j=2)),
        "OIII_1666": _f(O_pp, o3.getEmissivity(Te, ne, lev_i=6, lev_j=3)),
        "NII_6585": _f(N_p, n2.getEmissivity(Te, ne, wave=6584)),
        "CIII]_1907": _f(C_pp, c3.getEmissivity(Te, ne, wave=1907)),
        "CIII]": _f(C_pp, c3.getEmissivity(Te, ne, wave=1909)),
    }
    if with_oii:
        fluxes["OII_3726"] = _f(O_p, o2.getEmissivity(Te, ne, wave=3726))
        fluxes["OII_3729"] = _f(O_p, o2.getEmissivity(Te, ne, wave=3729))
    return _MockFitResult(fluxes, {k: frac * v for k, v in fluxes.items()})


def _run(res=None, **kw):
    return compute_abundances(
        res if res is not None else _result(), z=6.0, method="direct",
        dust_correct=False, n_mc=60, progress=False, **kw,
    )


def _by_hand_logU(res, ne):
    """log(U) from ``log_U_from_O32_ionic`` using the result's own ionic.

    Z comes from the point-estimate ionic sum, as in the pipeline, not
    from ``res.OH`` (the MC median).
    """
    opp, op = res.ionic["O++/H+"], res.ionic["O+/H+"]
    Z = 10.0 ** (12.0 + np.log10(opp + op) - LOG_OH_SOLAR)
    x = np.log10(opp / op)
    return log_U_from_O32_ionic(x, Z, ne)


# Each fixture is a full pipeline run; cache them for the module.
@pytest.fixture(scope="module")
def auto():
    return _run()


@pytest.fixture(scope="module")
def forced():
    return _run(logU_diagnostic="O32_ionic", marginalize_logU=False)


@pytest.fixture(scope="module")
def forced_pinned():
    return _run(logU_diagnostic="O32_ionic", marginalize_logU=False,
                logU_ne=1.0e3)


# ---------------------------------------------------------------------------


class TestCoefficients:
    def test_table_8_transcription(self):
        assert set(_COEFF_LOG_U_O32_IONIC) == set(_TABLE_8)
        for ne, row in _TABLE_8.items():
            assert _COEFF_LOG_U_O32_IONIC[ne] == row

    @pytest.mark.parametrize("ne, expected", [
        (1e2, -2.182), (960.0, -2.264), (27234.0, -2.278),
        (59603.0, -2.281), (1e6, -2.302),
    ])
    def test_spot_values(self, ne, expected):
        assert log_U_from_O32_ionic(0.842, 0.127, ne) == pytest.approx(
            expected, abs=0.002)


class TestFig18:
    """The fit must reproduce the Martinez+2025 Fig. 18 curves."""

    @pytest.mark.parametrize("Z, ne, expected", [
        (0.05, 1e2, -2.995), (0.40, 1e2, -2.805),
        (0.05, 1e6, -3.013), (0.40, 1e6, -2.868),
    ])
    def test_value_at_zero(self, Z, ne, expected):
        assert log_U_from_O32_ionic(0.0, Z, ne) == pytest.approx(
            expected, abs=0.002)

    @pytest.mark.parametrize("Z, x_expected", [(0.05, 1.71), (0.40, 1.48)])
    def test_reaches_minus_one(self, Z, x_expected):
        from scipy.optimize import brentq

        x = brentq(lambda x: log_U_from_O32_ionic(x, Z, 1e2) + 1.0, 0.5, 2.2)
        assert x == pytest.approx(x_expected, abs=0.02)

    @pytest.mark.parametrize("x, Z", [
        (-0.8, 0.05), (-0.5, 0.40), (0.0, 0.1), (0.3, 0.2), (0.6, 0.4),
    ])
    def test_density_insensitive_full_grid(self, x, Z):
        vals = [log_U_from_O32_ionic(x, Z, 10.0 ** g) for g in _LOG_NE_GRID]
        assert max(vals) - min(vals) < 0.15

    @pytest.mark.parametrize("x, Z", [
        (0.8, 0.05), (1.2, 0.2), (1.5, 0.3), (1.8, 0.4),
    ])
    def test_density_insensitive_above_1e3(self, x, Z):
        """At high ionisation only the 1e2 cm^-3 surface peels away."""
        vals = [log_U_from_O32_ionic(x, Z, 10.0 ** g) for g in _LOG_NE_GRID[1:]]
        assert max(vals) - min(vals) < 0.15

    @pytest.mark.parametrize("Z", [0.05, 0.40])
    @pytest.mark.parametrize("log_ne", list(_LOG_NE_GRID))
    def test_monotonic(self, Z, log_ne):
        xs = np.linspace(-1.0, 2.2, 161)
        vals = [log_U_from_O32_ionic(x, Z, 10.0 ** log_ne) for x in xs]
        assert np.all(np.diff(vals) > 0)

    def test_outside_validity_is_rejected(self):
        assert np.isnan(log_U_from_O32_ionic(2.5, 0.1, 1e3))
        assert np.isnan(log_U_from_O32_ionic(-1.2, 0.1, 1e3))


class TestComputeLogU:
    def test_uses_ionic_ratio(self):
        ionic = {"O++/H+": 3.0e-5, "O+/H+": 3.0e-6}
        logU, diag = _compute_logU({}, 0.1, 1e3, lock_diag="O32_ionic",
                                   ionic=ionic)
        assert diag == "O32_ionic"
        assert logU == pytest.approx(log_U_from_O32_ionic(1.0, 0.1, 1e3))

    def test_missing_op_gives_none(self):
        """No fallback to O32 even when the O32 lines are present."""
        fluxes = {"OIII_5007": 500.0, "OII_3726": 20.0, "OII_3729": 30.0}
        out = _compute_logU(fluxes, 0.1, 1e3, lock_diag="O32_ionic",
                            ionic={"O++/H+": 3.0e-5})
        assert out == (None, None)

    def test_never_in_automatic_tree(self):
        fluxes = {"OIII_5007": 500.0, "OII_3726": 20.0, "OII_3729": 30.0}
        ionic = {"O++/H+": 3.0e-5, "O+/H+": 3.0e-6}
        _, diag = _compute_logU(fluxes, 0.1, 1e3, ionic=ionic)
        assert diag == "O32"


class TestEndToEnd:
    def test_locked_without_oii_gives_no_logU(self):
        out = _run(_result(with_oii=False), logU_diagnostic="O32_ionic")
        assert out.logU is None
        assert "no fallback" in out.diagnostics["log(U)"]

    def test_default_unchanged(self, auto):
        """The automatic tree still picks O32, not O32_ionic."""
        assert "O32 diagnostic" in auto.diagnostics["log(U)"]
        assert "O32_ionic" not in auto.diagnostics["log(U)"]

    def test_diagnostics_string(self, forced):
        d = forced.diagnostics["log(U)"]
        assert "ionic O2+/O+" in d and "Martinez+2025 App. C" in d

    def test_logU_matches_by_hand(self, forced):
        assert forced.logU == pytest.approx(
            _by_hand_logU(forced, forced.ne_high), abs=1e-10)

    def test_has_error_bar(self, forced):
        err = forced.logU_err
        err = max(err) if isinstance(err, tuple) else err
        assert np.isfinite(err) and err > 0

    def test_override_equivalence_NO(self, forced):
        """Same log(U) handed in by override gives the same N/O."""
        ref = _run(logU_override=_by_hand_logU(forced, forced.ne_high),
                   marginalize_logU=False)
        assert forced.NO is not None and np.isfinite(forced.NO)
        assert forced.NO == pytest.approx(ref.NO, abs=1e-10)

    def test_override_equivalence_pinned(self, forced_pinned):
        """With logU_ne pinned one log(U) serves both zones, as an override does."""
        ref = _run(logU_override=_by_hand_logU(forced_pinned, 1.0e3),
                   marginalize_logU=False)
        assert forced_pinned.NO == pytest.approx(ref.NO, abs=1e-10)
        assert forced_pinned.CO is not None and np.isfinite(forced_pinned.CO)
        assert forced_pinned.CO == pytest.approx(ref.CO, abs=1e-10)

    def test_co_zone_shift_is_tiny(self, forced, forced_pinned):
        """C/O re-reads log(U) at ne_mid; for O32_ionic that barely moves it."""
        ref = _run(logU_override=_by_hand_logU(forced, forced.ne_high),
                   marginalize_logU=False)
        assert forced.CO == pytest.approx(ref.CO, abs=0.01)
