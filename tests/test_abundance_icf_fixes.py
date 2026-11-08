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


class TestNarrowBalmerOnly:
    """Broad Balmer components never enter the abundance normalisation."""

    def test_posteriors_skip_broad(self):
        from types import SimpleNamespace
        from jwspecabund._core import _extract_posteriors
        n = np.full(100, 1.0)
        b = np.full(100, 0.5)
        res = SimpleNamespace(lines={
            "HBETA": SimpleNamespace(flux_posterior=n),
            "HBETA_BROAD": SimpleNamespace(flux_posterior=b),
        })
        post = _extract_posteriors(res)
        assert np.array_equal(post["HBETA"], n)
        assert "HBETA_BROAD" not in post


# ---------------------------------------------------------------------------
# Strong-line (Sanders+25)
# ---------------------------------------------------------------------------

def _sanders_fluxes(Z, err_frac=0.02, with_o2=True):
    """Noise-free fluxes on the Sanders+25 relations at 12+log(O/H) = Z."""
    from jwspecabund.strong_line import CALIBRATIONS, _poly_eval
    hb = 1.0
    o3 = 10 ** _poly_eval(Z, CALIBRATIONS["O3"]["coeffs"])
    f = {"HBETA": hb, "OIII_5007": o3}
    if with_o2:
        f["OII_doublet"] = 10 ** _poly_eval(Z, CALIBRATIONS["O2"]["coeffs"])
    return f, {k: err_frac * v for k, v in f.items()}


class TestStrongLine:

    def test_o3_alone_is_refused(self):
        from jwspecabund.strong_line import sanders25_metallicity
        f, e = _sanders_fluxes(8.3, with_o2=False)
        with pytest.raises(ValueError, match="double-valued"):
            sanders25_metallicity(f, e, n_mc=0, progress=False)

    @pytest.mark.parametrize("Z", [7.35, 7.6, 7.9, 8.2, 8.5])
    def test_global_minimum_on_double_valued_o3(self, Z):
        # Exact O3 plus a weak O2: the global chi-squared minimum is the truth.
        # A bounded local minimiser lands on the wrong O3 branch here.
        from jwspecabund.strong_line import CALIBRATIONS, _poly_eval, solve_metallicity
        ratios = {
            m: {"val": _poly_eval(Z, CALIBRATIONS[m]["coeffs"]), "err": err}
            for m, err in (("O3", 0.01), ("O2", 0.5))
        }
        assert solve_metallicity(ratios) == pytest.approx(Z, abs=2e-3)

    @pytest.mark.parametrize("Z", [7.7, 8.0, 8.3, 8.5])
    def test_end_to_end_recovers_truth(self, Z):
        from jwspecabund.strong_line import sanders25_metallicity
        f, e = _sanders_fluxes(Z)
        e["OII_doublet"] = 0.6 * f["OII_doublet"]
        Zb = sanders25_metallicity(f, e, n_mc=0, progress=False)[0]
        # The four Sanders+25 fits are not exactly mutually consistent, so a
        # mock built from O3 and O2 sits a few 0.01 dex off R23/O32.
        assert abs(Zb - Z) < 0.05

    def test_solution_confined_to_calibrated_range(self):
        from jwspecabund.strong_line import Z_MAX, Z_MIN, sanders25_metallicity
        f, e = _sanders_fluxes(8.0)
        f["OIII_5007"] *= 0.05  # far below the O3 relation anywhere
        Zb, lo, hi, *_ = sanders25_metallicity(f, e, n_mc=50, progress=False)
        assert Z_MIN <= lo <= Zb <= hi <= Z_MAX


class _L:
    def __init__(self, post):
        self.flux = float(np.median(post))
        s = float(np.std(post))
        self.flux_err = (s, s)
        self.flux_posterior = post


class _M:
    def __init__(self, lines):
        self.lines = lines
        self.spectrum = None


class TestStrongLineMCMC:

    def _result(self, fl, frac=0.03, seed=0):
        rng = np.random.default_rng(seed)
        return _M({k: _L(rng.normal(v, frac * v, 2000)) for k, v in fl.items()})

    def test_undetected_line_cannot_enter_draws(self):
        from jwspecabund import compute_abundances
        f, _ = _sanders_fluxes(8.0)
        res = self._result(f)
        rng = np.random.default_rng(1)
        # An undetected [N II]-free extra: [O III] 4959 posterior centred on 0.
        res.lines["OIII_4959"] = _L(rng.normal(0.0, 0.3, 2000))
        base = compute_abundances(self._result(f), z=0.0, method="strong_line",
                                  dust_correct=False, n_posterior=300,
                                  snr_line=0, progress=False)
        r = compute_abundances(res, z=0.0, method="strong_line", dust_correct=False,
                               n_posterior=300, snr_line=0, progress=False)
        assert abs(r.OH - base.OH) < 0.05

    def test_raw_posteriors_are_dust_corrected_per_draw(self):
        from jwspecabund import compute_abundances
        from jwspecabund.dust import salim_attenuation
        from jwspecfit.lines import REST_LINES_A
        f, _ = _sanders_fluxes(8.0)
        obs = {k: v * 10 ** (-0.4 * salim_attenuation(
            np.array([REST_LINES_A[k]]), 1.0, delta=-0.45, B=2.27)[0]) for k, v in f.items()}
        kw = dict(z=0.0, method="strong_line", Av=1.0, n_posterior=300, progress=False)
        a = compute_abundances(self._result(obs), Av_err=None, **kw)
        b = compute_abundances(self._result(obs), Av_err=0.02, **kw)
        assert abs(a.OH - 8.0) < 0.03
        assert abs(b.OH - a.OH) < 0.03


def test_compute_abundances_reports_unavailable_strong_line():
    """O3-only input: O/H is NaN with the reason, not a guessed branch."""
    from jwspecabund import compute_abundances
    f, _ = _sanders_fluxes(8.3, with_o2=False)
    res = compute_abundances(
        _M({k: SimpleLine(v, 0.02 * v) for k, v in f.items()}), z=2.0,
        method="strong_line", dust_correct=False, n_mc=20, progress=False,
    )
    assert np.isnan(res.OH)
    assert "double-valued" in res.failures["strong_line"]


class SimpleLine:
    def __init__(self, flux, err):
        self.flux, self.flux_err = flux, err
        self.snr = flux / err


def test_izotov_tier_row_matches_adopted_izotov_value():
    """The comparison-table Izotov row must equal the adopted Izotov N/O."""
    ionic = {"O+/H+": 1e-5, "O++/H+": 9e-5, "N+/H+": 1e-6}
    tot = compute_total_abundances(ionic, icf_method="izotov06")
    row = [v for k, v in tot["_NO_tiers"].items() if k.startswith("Izotov")]
    assert row and row[0] == pytest.approx(np.log10(tot["N/O"]), abs=1e-12)
