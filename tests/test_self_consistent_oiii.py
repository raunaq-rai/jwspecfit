"""Tests for the self-consistent O++ T_e-n_e solve (Hsiao et al. 2026).

[O III] lambda5007 has a critical density of only ~7e5 cm^-3, so above
n_e ~ 1e5 cm^-3 the classical lambda4363/lambda5007 ratio depends on both
T_e and n_e.  Adding O III] lambda1666, whose critical density is far
higher, gives two independent ratios for the two unknowns.  These tests
check that the joint solve recovers the truth where the density is
constrained, reports an upper limit where it is not, and leaves results
untouched when lambda1666 is absent.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from jwspecabund import compute_abundances, compute_Te_ne_OIII
from jwspecabund.direct import (
    OIII_GRID_LOGNE,
    _get_oiii_atom,
    _oiii_te_ne_grid,
)


def _oiii_fluxes(Te, ne, coll_file=None):
    """Synthesise exact O III fluxes at a known (T_e, n_e)."""
    atom = _get_oiii_atom() if coll_file is None else _get_oiii_atom(coll_file)
    return (
        atom.getEmissivity(Te, ne, lev_i=6, lev_j=3),   # 1666
        atom.getEmissivity(Te, ne, wave=4363),
        atom.getEmissivity(Te, ne, wave=5007),
        atom.getEmissivity(Te, ne, wave=4959),
    )


# Points where the density is high enough for the diagnostic to have
# leverage, so the solve must return the truth rather than a limit.
_CONSTRAINED = [
    (15000, 1e5), (13000, 1e6), (20000, 3e5), (22000, 2e5), (12000, 5e5),
    (28000, 3e5), (11000, 2e6), (16000, 8e4), (19000, 6e5), (14000, 1.5e6),
]


class TestRoundTrip:
    """Exact fluxes in, exact (T_e, n_e) out."""

    @pytest.mark.parametrize(("Te", "ne"), _CONSTRAINED)
    def test_recovers_truth(self, Te, ne):
        sol = compute_Te_ne_OIII(*_oiii_fluxes(Te, ne))
        assert sol.converged
        assert sol.Te == pytest.approx(Te, rel=2e-3)
        assert np.log10(sol.ne) == pytest.approx(np.log10(ne), abs=0.02)
        assert not sol.at_grid_edge

    def test_lambda4959_optional(self):
        """The nebular denominator may be lambda5007 alone."""
        f1666, f4363, f5007, f4959 = _oiii_fluxes(15000, 1e5)
        with_4959 = compute_Te_ne_OIII(f1666, f4363, f5007, f4959)
        # Scale lambda5007 up to stand in for the doublet sum.
        without = compute_Te_ne_OIII(f1666, f4363, f5007 + f4959)
        assert without.Te == pytest.approx(with_4959.Te, rel=1e-6)

    def test_rejects_nonpositive_flux(self):
        f1666, f4363, f5007, f4959 = _oiii_fluxes(15000, 1e5)
        with pytest.raises(ValueError, match="1666"):
            compute_Te_ne_OIII(0.0, f4363, f5007, f4959)
        with pytest.raises(ValueError, match="4363"):
            compute_Te_ne_OIII(f1666, 0.0, f5007, f4959)
        # The nebular denominator is the doublet sum, so only losing both
        # members is fatal.
        with pytest.raises(ValueError, match="nebular"):
            compute_Te_ne_OIII(f1666, f4363, 0.0, 0.0)
        assert compute_Te_ne_OIII(f1666, f4363, 0.0, f5007 + f4959).converged


class TestUncertainties:
    """Errors and the upper-limit branch come from the flux posterior."""

    @staticmethod
    def _solve(Te, ne, frac=0.10, n_draws=400):
        f = _oiii_fluxes(Te, ne)
        return compute_Te_ne_OIII(
            *f, **{
                f"err_{k}": frac * v
                for k, v in zip(("1666", "4363", "5007", "4959"), f)
            },
            n_draws=n_draws,
        )

    def test_high_density_is_measured(self):
        sol = self._solve(15000, 1e5)
        assert not sol.ne_is_upper_limit
        assert sol.Te == pytest.approx(15000, rel=0.05)
        assert np.log10(sol.ne) == pytest.approx(5.0, abs=0.2)
        assert sol.Te_err is not None and all(e > 0 for e in sol.Te_err)
        assert sol.ne_err is not None and all(e > 0 for e in sol.ne_err)
        assert sol.converged_fraction > 0.8

    def test_low_density_gives_upper_limit(self):
        """Below the lambda5007 critical density the ratios go flat."""
        sol = self._solve(25000, 1e3)
        assert sol.ne_is_upper_limit
        assert sol.ne_upper_limit is not None and sol.ne_upper_limit > 1e3
        # T_e is still the low-density limit and remains recoverable.
        assert sol.Te == pytest.approx(25000, rel=0.15)

    def test_no_errors_means_no_posterior(self):
        sol = compute_Te_ne_OIII(*_oiii_fluxes(15000, 1e5))
        assert sol.Te_err is None and sol.ne_err is None
        assert sol.ne_posterior is None
        assert not sol.ne_is_upper_limit
        # Without a posterior the adopted values are the raw intersection.
        assert sol.Te == sol.Te_intersection
        assert sol.ne == sol.ne_intersection


class TestModelGrid:
    """Guards on the assumptions the curve inversion relies on."""

    def test_ratios_monotonic_in_Te(self):
        """Every ratio must increase with T_e at every density.

        The inversion in ``_te_curve`` assumes a single root per density
        column; a non-monotonic ratio would silently return the wrong one.
        This pins the T_e grid ceiling, above which
        lambda4363/(lambda5007+lambda4959) turns over at high density.
        """
        grid = _oiii_te_ne_grid()
        for key in ("log_R_4363", "log_R_1666_4363"):
            assert np.all(np.diff(grid[key], axis=0) > 0), key

    def test_interpolated_grid_matches_direct_build(self):
        """The 141-node build must reproduce a full 1,000-node one.

        The emissivities are evaluated at 141 densities and interpolated up
        to the 1,000-step grid Hsiao et al. specify, because calling PyNEB
        at all 1,000 costs ~7x more.  If that shortcut ever drifts, the
        density scale drifts with it.
        """
        fast = _oiii_te_ne_grid()
        exact = _oiii_te_ne_grid(n_nodes=OIII_GRID_LOGNE[2])
        for key in ("log_R_4363", "log_R_1666_4363", "log_R_5007_1666"):
            assert np.allclose(fast[key], exact[key], atol=1e-5, rtol=0), key

    def test_alternative_collision_data(self):
        """AK99 (Hsiao et al.'s choice) is usable and close to TZ17."""
        Te, ne = 15000, 1e5
        f = _oiii_fluxes(Te, ne)
        ak = compute_Te_ne_OIII(*f, coll_file="o_iii_coll_AK99.dat")
        assert ak.coll_file == "o_iii_coll_AK99.dat"
        # Different atomic data, same physical answer to a few per cent.
        assert ak.Te == pytest.approx(Te, rel=0.05)
        assert np.log10(ak.ne) == pytest.approx(np.log10(ne), abs=0.2)

    def test_five_level_collision_data_rejected(self):
        """PyNEB's default cannot represent lambda1666 and must not be used."""
        with pytest.raises(RuntimeError, match="needs at least 6"):
            compute_Te_ne_OIII(
                *_oiii_fluxes(15000, 1e5), coll_file="o_iii_coll_SSB14.dat",
            )


@dataclass
class _Line:
    flux: float
    flux_err: float


class _MockFitResult:
    def __init__(self, fluxes, errors):
        self.lines = {k: _Line(fluxes[k], errors[k]) for k in fluxes}


def _mock_result(Te, ne, OH_pp=1e-4, frac=0.05, with_1666=True):
    """A fit result whose O III lines encode a known T_e, n_e and O++/H+."""
    import pyneb as pn

    atom = _get_oiii_atom()
    jHb = pn.RecAtom("H", 1).getEmissivity(Te, ne, lev_i=4, lev_j=2)
    fluxes = {"HBETA": 100.0}
    lines = {
        "OIII_1666": {"lev_i": 6, "lev_j": 3},
        "OIII_4363": {"wave": 4363},
        "OIII_5007": {"wave": 5007},
        "OIII_4959": {"wave": 4959},
    }
    if not with_1666:
        lines.pop("OIII_1666")
    for name, kw in lines.items():
        fluxes[name] = 100.0 * OH_pp * atom.getEmissivity(Te, ne, **kw) / jHb
    errors = {k: frac * v for k, v in fluxes.items()}
    return _MockFitResult(fluxes, errors)


class TestComputeAbundancesIntegration:
    """The joint solve must reach O/H, not just T_e."""

    @staticmethod
    def _run(result, **kw):
        return compute_abundances(
            result, z=6.0, method="direct", dust_correct=False,
            n_mc=150, progress=False, **kw,
        )

    def test_recovers_metallicity_at_high_density(self):
        """The traditional route fails badly here; the joint solve does not."""
        truth = 12.0 + np.log10(1e-4)
        res = self._run(_mock_result(13000, 1e6))

        assert res.Te_diagnostic == "self_consistent"
        assert res.OH == pytest.approx(truth, abs=0.1)
        assert res.Te_high == pytest.approx(13000, rel=0.1)
        assert np.log10(res.ne_Opp) == pytest.approx(6.0, abs=0.2)

        # The lambda4363-only answer is reported alongside, and is the
        # ~1 dex underestimate the paper describes.
        alt = res.alt_results["direct_4363"]
        assert alt.Te_diagnostic == "4363"
        assert truth - alt.OH > 0.8

    def test_flag_off_restores_single_ratio(self):
        res_on = self._run(_mock_result(13000, 1e6))
        res_off = self._run(_mock_result(13000, 1e6), self_consistent_OIII=False)
        assert res_off.Te_diagnostic == "4363"
        assert res_off.Te_ne_selfconsistent is None
        assert res_on.OH - res_off.OH > 0.8

    def test_inert_without_1666(self):
        """No UV auroral line: nothing about the result may change."""
        result = _mock_result(15000, 1e4, with_1666=False)
        on = self._run(result)
        off = self._run(result, self_consistent_OIII=False)
        assert on.Te_diagnostic == off.Te_diagnostic == "4363"
        assert on.Te_ne_selfconsistent is None
        assert on.OH == pytest.approx(off.OH, abs=1e-12)
        assert on.Te_high == pytest.approx(off.Te_high, abs=1e-12)

    def test_low_density_falls_back_and_caps(self):
        """An unconstrained density still bounds the borrowed one."""
        res = self._run(_mock_result(25000, 1e3))
        assert res.Te_diagnostic == "4363"
        assert res.ne_Opp_is_upper_limit
        sc = res.Te_ne_selfconsistent
        assert sc is not None and sc.ne_is_upper_limit
        # The adopted density may not exceed the bound the O III lines set.
        assert res.ne_Opp <= sc.ne_upper_limit + 1.0

    def test_non_converged_solve_does_not_cap_density(self):
        """A non-crossing solve must not override a measured density.

        Regression: when the two O III ratios are not mutually consistent,
        the curves never cross and the reported "upper limit" is only where
        near-parallel curves come closest -- it moves with a fraction of the
        atomic-data systematic on lambda1666.  Capping the borrowed density
        on that let a non-detection of density silently shift O/H (0.05 dex
        on a real stacked spectrum).  It must be inert instead.
        """
        result = _mock_result(17000, 1e3)
        # Break the mutual consistency: lambda1666 20 % brighter than any
        # (T_e, n_e) pair can produce alongside the observed lambda4363.
        result.lines["OIII_1666"].flux *= 1.2

        on = self._run(result)
        off = self._run(result, self_consistent_OIII=False)
        sc = on.Te_ne_selfconsistent
        assert sc is not None and not sc.converged
        # The limit is still reported...
        assert on.ne_Opp_is_upper_limit
        # ...but nothing downstream may move because of it.
        assert on.ne_Opp == pytest.approx(off.ne_Opp, rel=1e-12)
        assert on.OH == pytest.approx(off.OH, abs=1e-12)
        assert on.Te_high == pytest.approx(off.Te_high, abs=1e-12)

    def test_rejects_bad_flag_value(self):
        with pytest.raises(ValueError, match="self_consistent_OIII"):
            self._run(_mock_result(15000, 1e5), self_consistent_OIII="yes")

    def test_summary_lists_each_diagnostic(self):
        res = self._run(_mock_result(15000, 1e5))
        text = res.summary()
        assert "T_e(O++) diagnostics:" in text
        assert "self-consistent (UV+opt)" in text
        assert "[OIII] 4363 only" in text
        assert "[adopted]" in text


def test_plot_te_ne_diagnostic_renders():
    import matplotlib
    matplotlib.use("Agg")
    from jwspecabund import plot_te_ne_diagnostic

    f = _oiii_fluxes(15000, 1e5)
    sol = compute_Te_ne_OIII(
        *f, **{
            f"err_{k}": 0.1 * v
            for k, v in zip(("1666", "4363", "5007", "4959"), f)
        },
        n_draws=200,
    )
    fig = plot_te_ne_diagnostic(sol)
    ax = fig.axes[0]
    # Three ratio curves plus the adopted-solution marker.
    labelled = [
        ln for ln in ax.get_lines()
        if ln.get_label().startswith(("[O", "O I", "adopted"))
    ]
    assert len(labelled) == 4
    assert ax.get_xscale() == "log"


def test_plot_requires_a_solution():
    from jwspecabund import plot_te_ne_diagnostic
    from jwspecabund.result import AbundanceResult

    with pytest.raises(ValueError, match="No self-consistent"):
        plot_te_ne_diagnostic(AbundanceResult(method="direct", OH=8.0, OH_err=0.1))
