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


def test_oii_upper_limit_uses_both_doublet_emissivities():
    from jwspecabund._core import _compute_ionic_upper_limits
    from jwspecabund.direct import _ionic_abundance
    fluxes = {"HBETA": 100.0, "OIII_5007": 500.0}
    errors = {"HBETA": 1.0, "OIII_5007": 5.0, "OII_doublet": 2.0}
    ul, _ = _compute_ionic_upper_limits(
        {"O++/H+": 1e-4}, fluxes, errors, 1.5e4, 1.35e4, 300.0, 300.0, 300.0,
    )
    expect = _ionic_abundance("O", 2, 6.0, 100.0, 1.35e4, 300.0, [3726, 3729])
    assert ul["O+/H+"] == pytest.approx(expect, rel=1e-10, abs=0)


@pytest.mark.parametrize("Te", [2.6e4, 3.5e4])
def test_hei_deblend_ratio_above_table_limit(Te):
    """He I tables stop at 25 kK: hotter gas must still give a ratio."""
    from jwspecabund.direct import heI_4714_over_4472
    r = heI_4714_over_4472(Te, 1e4)
    assert r == heI_4714_over_4472(2.5e4, 1e4)
    assert 0.1 < r < 0.3


class TestMartinezEdgeCases:

    def test_forced_icf4_needs_every_ion(self):
        from jwspecabund.martinez25_icf import compute_NO_martinez25_locked
        ionic = {"O+/H+": 1e-5, "O++/H+": 9e-5, "N++/H+": 2e-6}  # no N+
        assert compute_NO_martinez25_locked(ionic, -2.0, 0.1, 1e3, "NpNpp_OpOpp") is None

    def test_nan_density_raises_cleanly(self):
        from jwspecabund.martinez25_icf import icf_NpOp
        with pytest.raises(ValueError, match="finite n_e"):
            icf_NpOp(-2.0, 0.1, float("nan"))


def test_lya_fesc_error_counts_shared_lya_once():
    """Three Balmer lines and one noisy Lya: error is set by the Lya error."""
    from jwspecabund.dust import LYA_CASE_B_RATIOS, compute_lya_escape_fraction
    hb = 1.0
    fl = {k: 26.071 / r * hb for k, (r, _) in LYA_CASE_B_RATIOS.items()}
    er = {k: v / 30 for k, v in fl.items()}  # SNR 30 each
    lya = 0.2 * 26.071 * hb
    out = compute_lya_escape_fraction(lya, lya / 5, fl, er)  # Lya SNR 5
    assert out["f_esc"] == pytest.approx(0.2, rel=1e-12, abs=0)
    # Must be at least the Lya-only fractional error (20%).
    assert out["f_esc_err"] >= 0.2 * 0.2


def test_niii_quintuplet_mc_error_is_correlated():
    """Members share one fitted flux: the MC must not average their noise."""
    import pyneb as pn
    from jwspecabund import compute_abundances
    from jwspecabund.direct import NIII_MULTIPLET, _hbeta_emissivity

    Te, ne = 1.5e4, 300.0
    Te_int = 0.83 * Te + 1700
    o3, n3 = pn.Atom("O", 3), pn.Atom("N", 3)
    jhb = _hbeta_emissivity(Te, ne)
    fl = {"HBETA": 100.0}
    for w in (5007, 4959, 4363):
        fl[f"OIII_{w}"] = 100 * 1e-4 * o3.getEmissivity(Te, ne, wave=w) / jhb
    jhb_i = _hbeta_emissivity(Te_int, ne)
    for name, w in NIII_MULTIPLET.items():
        fl[name] = 100 * 1e-5 * n3.getEmissivity(Te_int, ne, wave=w) / jhb_i
    tot = sum(fl[n] for n in NIII_MULTIPLET)
    # Only the quintuplet is noisy; each member's error is its share of 10%.
    err = {k: (0.1 * v if k in NIII_MULTIPLET else 1e-9 * v) for k, v in fl.items()}
    res = compute_abundances(
        _M({k: SimpleLine(fl[k], err[k]) for k in fl}), z=0.0, method="direct",
        dust_correct=False, icf_method="direct_sum", n_mc=600,
        self_consistent_OIII=False, snr_line=0, progress=False,
    )
    expect = 0.4343 * sum(err[n] for n in NIII_MULTIPLET) / tot
    sig = res.NO_err if np.isscalar(res.NO_err) else 0.5 * sum(res.NO_err)
    assert sig == pytest.approx(expect, rel=0.15)


def test_av_err_propagates_for_least_squares_input():
    """Least-squares fit: a positive Av_err must widen the O/H error."""
    import pyneb as pn
    from jwspecabund import compute_abundances
    from jwspecabund.direct import _hbeta_emissivity

    Te, ne = 1.5e4, 300.0
    Te_low = 0.70 * Te + 3000
    o3, o2 = pn.Atom("O", 3), pn.Atom("O", 2)
    fl = {"HBETA": 100.0}
    for w in (5007, 4959, 4363):
        fl[f"OIII_{w}"] = 100 * 1e-4 * o3.getEmissivity(Te, ne, wave=w) / _hbeta_emissivity(Te, ne)
    for w in (3726, 3729):
        fl[f"OII_{w}"] = 100 * 3e-5 * o2.getEmissivity(Te_low, ne, wave=w) / _hbeta_emissivity(Te_low, ne)
    mk = lambda: _M({k: SimpleLine(v, 0.01 * v) for k, v in fl.items()})
    kw = dict(z=0.0, method="direct", Av=0.5, n_mc=300, progress=False,
              self_consistent_OIII=False)
    a = compute_abundances(mk(), Av_err=None, **kw)
    b = compute_abundances(mk(), Av_err=0.5, **kw)
    sa = a.OH_err if np.isscalar(a.OH_err) else 0.5 * sum(a.OH_err)
    sb = b.OH_err if np.isscalar(b.OH_err) else 0.5 * sum(b.OH_err)
    assert sb > 1.5 * sa
    assert b.Av_posterior is not None and len(b.Av_posterior) > 0


def _direct_fluxes(Te=1.5e4, ne=300.0):
    """PyNeb fluxes for O/H + C III] (Hbeta = 100), 3-tier Garnett zones."""
    import pyneb as pn
    from jwspecabund.direct import _hbeta_emissivity
    Te_low, Te_int = 0.70 * Te + 3000, 0.83 * Te + 1700

    def f(el, ion, X, T, w):
        return 100 * X * pn.Atom(el, ion).getEmissivity(T, ne, wave=w) / _hbeta_emissivity(T, ne)

    fl = {"HBETA": 100.0}
    for w in (5007, 4959, 4363):
        fl[f"OIII_{w}"] = f("O", 3, 1e-4, Te, w)
    for w in (3726, 3729):
        fl[f"OII_{w}"] = f("O", 2, 2e-5, Te_low, w)
    for w in (1907, 1909):
        fl[f"CIII]_{w}"] = f("C", 3, 2.5e-5, Te_int, w)
    return fl


def test_co_logu_marginalised_without_ciii_density():
    """No C2+-zone density: C/O must still carry the log(U) scatter."""
    from jwspecabund import compute_abundances
    fl = _direct_fluxes()
    err = {k: 0.002 * v for k, v in fl.items()}
    err["OII_3726"], err["OII_3729"] = 0.15 * fl["OII_3726"], 0.15 * fl["OII_3729"]
    kw = dict(z=0.0, method="direct", dust_correct=False, n_mc=200, progress=False,
              self_consistent_OIII=False, logU_diagnostic="O32", snr_ne=1e9)

    def run(marg):
        return compute_abundances(_M({k: SimpleLine(fl[k], err[k]) for k in fl}),
                                  marginalize_logU=marg, **kw)

    on, off = run(True), run(False)
    assert on.ne_mid is None and on.CO is not None and np.isfinite(on.CO)
    s_on = on.CO_err if np.isscalar(on.CO_err) else 0.5 * sum(on.CO_err)
    s_off = off.CO_err if np.isscalar(off.CO_err) else 0.5 * sum(off.CO_err)
    assert s_on > 1.3 * s_off


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

class TestForwardModelFixes:

    def _result(self, logs):
        from jwspecabund.forward import _build_forward_result
        names = ["log_Te", "log_ne", *logs]
        samples = np.tile([4.2, 2.5, *logs.values()], (50, 1))
        return _build_forward_result(samples, names, [])

    def test_n_over_o_with_optical_nitrogen_only(self):
        r = self._result({"log_Op": -4.5, "log_Opp": -4.0, "log_Np": -5.7})
        assert r["NO"] == pytest.approx(-5.7 + 4.5, abs=1e-12)

    def test_n_over_o_all_zones_and_no_n_v(self):
        r = self._result({"log_Op": -4.5, "log_Opp": -4.0, "log_Np": -5.7,
                          "log_Npp": -5.0, "log_Npppp": -4.0})
        expect = np.log10((10**-5.7 + 10**-5.0) / (10**-4.5 + 10**-4.0))
        assert r["NO"] == pytest.approx(expect, abs=1e-12)

    def test_cii_lines_build_one_multiplet_observable(self):
        from jwspecabund.forward import (_PRIOR_BOUNDS, _build_observables,
                                         _determine_free_params)
        f = {"HBETA": 1.0, "OIII_5007": 5.0, "CII]_2324": 0.02, "CII]_2326": 0.03}
        e = {k: 0.05 * v for k, v in f.items()}
        names, obs, _, comps = _build_observables(f, e)
        assert "CII_multiplet/Hb" in names
        assert obs[names.index("CII_multiplet/Hb")] == pytest.approx(0.05, abs=0, rel=1e-12)
        assert all(p in _PRIOR_BOUNDS for p in _determine_free_params(comps))

    def test_fitter_merged_niv_key_used(self):
        from jwspecabund.forward import _build_observables
        names, *_ = _build_observables({"HBETA": 1.0, "NIV_doublet": 0.1},
                                       {"HBETA": 0.02, "NIV_doublet": 0.01})
        assert names == ["NIV_doublet/Hb"]

    def test_shared_hbeta_covariance(self):
        from jwspecabund.forward import _build_observables
        f = {"HBETA": 1.0, "OIII_5007": 5.0, "OIII_4363": 0.05}
        e = {"HBETA": 0.1, "OIII_5007": 0.1, "OIII_4363": 0.001}
        names, r, icov, _ = _build_observables(f, e)
        fr = np.array([e[n.split("/")[0]] / f[n.split("/")[0]] for n in names])
        cov = np.diag((r * fr) ** 2) + np.outer(r, r) * 0.1**2
        np.testing.assert_allclose(np.linalg.inv(icov), cov, rtol=1e-10, atol=0)

    def test_zero_hbeta_error_keeps_line_errors(self):
        from jwspecabund.forward import _build_observables
        names, r, icov, _ = _build_observables(
            {"HBETA": 1.0, "OIII_5007": 5.0}, {"HBETA": 0.0, "OIII_5007": 0.005})
        assert np.sqrt(np.linalg.inv(icov)[0, 0]) == pytest.approx(0.005, rel=1e-10, abs=0)

    def test_density_prior_reaches_1e6(self):
        from jwspecabund.forward import _PRIOR_BOUNDS
        assert _PRIOR_BOUNDS["log_ne"][1] >= 6.0


def test_hbeta_emissivity_continuous_at_30kK():
    from jwspecabund.direct import _hbeta_emissivity
    for ne in (1e2, 1e5):
        below = _hbeta_emissivity(30000.0, ne)
        above = _hbeta_emissivity(30000.001, ne)
        assert above == pytest.approx(below, rel=1e-6, abs=0)
        # Keeps falling with the SH95 slope (~T^-0.97).
        r = _hbeta_emissivity(40000.0, ne) / below
        assert r == pytest.approx((4 / 3) ** -0.97, rel=0.02)
