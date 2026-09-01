"""Tests for the Balmer-decrement A_V estimators.

Three properties are pinned here.

1.  Balmer decrements are *not* independent measurements.  In magnitudes
    they are exactly additive, ``D(Hg/Ha) = D(Hg/Hb) + D(Hb/Ha)``, so a
    ladder of ``N`` lines carries ``N - 1`` constraints no matter how many
    of the ``N(N-1)/2`` pairs are formed.  Any estimator that averages the
    pairs as though they were independent is wrong.

2.  ``compute_Av_multi_balmer`` must not clip each decrement at zero
    before averaging (that biases the mean upward near ``A_V = 0``) and
    must account for the anchor flux being shared between the decrements
    (ignoring it understates the error on the mean).

3.  ``compute_Av_joint_balmer`` fits all the fluxes at once with two free
    parameters.  It must be unbiased, quote an honest error, degenerate to
    the single decrement when only two lines are available, and flag a
    ladder that no single ``A_V`` can reconcile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from jwspecabund.dust import (
    _BALMER_LADDER,
    _kappa,
    compute_Av_from_balmer,
    compute_Av_joint_balmer,
    compute_Av_multi_balmer,
)

RV = 3.1
LAW = "cardelli"
LAW_KW = {"Rv": RV}

# A four-line ladder at the SNRs of a real JWST/NIRSpec object (OMEGA 77).
SNR = {"Ha": 31.5, "HBETA": 39.7, "HGAMMA": 18.3, "HDELTA": 10.5}
NAMES = ["HDELTA", "HGAMMA", "HBETA", "Ha"]


def make_ladder(Av: float, hbeta: float = 1.0e-18) -> tuple[dict, dict]:
    """Noiseless Balmer fluxes attenuated by *Av*, with realistic errors."""
    fluxes, errors = {}, {}
    for name in NAMES:
        wave, ratio = _BALMER_LADDER[name]
        k = float(_kappa(np.array([wave]), LAW, **LAW_KW)[0])
        f = hbeta * ratio * 10.0 ** (-0.4 * Av * k)
        fluxes[name] = f
        errors[name] = f / SNR[name]
    return fluxes, errors


def _D(fluxes: dict, num: str, den: str) -> float:
    """Decrement in magnitudes: -2.5 log10[(F_num/F_den) / (R_num/R_den)]."""
    r0 = _BALMER_LADDER[num][1] / _BALMER_LADDER[den][1]
    return -2.5 * np.log10((fluxes[num] / fluxes[den]) / r0)


class TestDecrementsAreNotIndependent:
    """Property 1: the pairwise decrements carry no extra information."""

    @pytest.mark.parametrize("Av", [0.0, 0.35, 1.2])
    def test_decrements_are_exactly_additive(self, Av):
        rng = np.random.default_rng(7)
        fluxes, errors = make_ladder(Av)
        # Additivity is a property of the fluxes, so it must survive noise.
        noisy = {k: v + rng.normal(0.0, errors[k]) for k, v in fluxes.items()}
        for a, m, b in [
            ("HGAMMA", "HBETA", "Ha"),
            ("HDELTA", "HGAMMA", "Ha"),
            ("HDELTA", "HBETA", "HGAMMA"),
        ]:
            assert _D(noisy, a, b) == pytest.approx(
                _D(noisy, a, m) + _D(noisy, m, b), abs=1e-10
            )

    def test_third_pair_adds_no_constraint(self):
        """Hg/Ha is determined by Hg/Hb and Hb/Ha, so its A_V is too."""
        fluxes, _ = make_ladder(0.4)
        kap = {n: float(_kappa(np.array([_BALMER_LADDER[n][0]]), LAW, **LAW_KW)[0])
               for n in NAMES}
        av = {}
        for a, b in [("HGAMMA", "HBETA"), ("HBETA", "Ha"), ("HGAMMA", "Ha")]:
            av[(a, b)] = _D(fluxes, a, b) / (kap[a] - kap[b])
        # All three are the same number; the "extra" pair is redundant.
        assert av[("HGAMMA", "Ha")] == pytest.approx(av[("HGAMMA", "HBETA")], abs=1e-8)
        assert av[("HGAMMA", "Ha")] == pytest.approx(av[("HBETA", "Ha")], abs=1e-8)


class TestAnchoredEstimator:
    """Property 2: no per-decrement clipping, and a covariance-aware error."""

    def test_single_decrement_clip_is_opt_out(self):
        # Blue excess -> negative A_V.  Clipped by default, exposed when asked.
        wave_g, r_g = _BALMER_LADDER["HGAMMA"]
        wave_b, r_b = _BALMER_LADDER["HBETA"]
        f_b = 1.0e-18
        f_g = 1.10 * f_b * (r_g / r_b)          # 10 % *bluer* than Case B
        kw = dict(law=LAW, intrinsic_ratio=r_g / r_b,
                  wave_num_A=wave_g, wave_den_A=wave_b, **LAW_KW)
        clipped, _ = compute_Av_from_balmer(f_g, f_b, 0.0, 0.0, **kw)
        raw, _ = compute_Av_from_balmer(f_g, f_b, 0.0, 0.0, clip=False, **kw)
        assert clipped == 0.0
        assert raw < 0.0

    def test_mean_is_unbiased_at_zero_Av(self):
        """Clipping each decrement first would push the mean above zero."""
        rng = np.random.default_rng(11)
        fluxes, errors = make_ladder(0.0)
        draws = []
        for _ in range(600):
            noisy = {k: v + rng.normal(0.0, errors[k]) for k, v in fluxes.items()}
            out = compute_Av_multi_balmer(
                noisy, errors, law=LAW, anchor="Ha", **LAW_KW
            )
            # Read the unclipped GLS mean back off the individual values.
            ind = out["individual"]
            a = np.array([r["Av"] for r in ind])
            w = 1.0 / np.array([r["Av_err"] for r in ind]) ** 2
            draws.append(float(np.average(a, weights=w)))
        # Individual decrements must be able to go negative for this to hold.
        assert min(draws) < 0.0
        assert abs(float(np.mean(draws))) < 0.02

    def test_error_accounts_for_the_shared_anchor(self):
        """Every decrement divides by Ha, so their errors are correlated."""
        fluxes, errors = make_ladder(0.3)
        out = compute_Av_multi_balmer(
            fluxes, errors, law=LAW, anchor="Ha", **LAW_KW
        )
        ind = out["individual"]
        assert len(ind) == 3
        naive = 1.0 / np.sqrt(
            np.sum(1.0 / np.array([r["Av_err"] for r in ind]) ** 2)
        )
        # The shared anchor term is real and strictly inflates the error.
        assert all(r["Av_err_shared"] > 0 for r in ind)
        assert out["Av_err"] > naive

    def test_error_is_honest_under_monte_carlo(self):
        rng = np.random.default_rng(3)
        fluxes, errors = make_ladder(0.6)
        draws = [
            compute_Av_multi_balmer(
                {k: v + rng.normal(0.0, errors[k]) for k, v in fluxes.items()},
                errors, law=LAW, anchor="Ha", **LAW_KW,
            )["Av"]
            for _ in range(800)
        ]
        quoted = compute_Av_multi_balmer(
            fluxes, errors, law=LAW, anchor="Ha", **LAW_KW
        )["Av_err"]
        assert np.std(draws) == pytest.approx(quoted, rel=0.15)


class TestJointEstimator:
    """Property 3: the anchor-free two-parameter fit."""

    @pytest.mark.parametrize("Av", [0.0, 0.25, 0.8, 2.0])
    def test_recovers_injected_Av_exactly(self, Av):
        fluxes, errors = make_ladder(Av)
        out = compute_Av_joint_balmer(fluxes, errors, law=LAW, **LAW_KW)
        assert out["n_lines"] == 4
        assert out["dof"] == 2
        assert out["Av_fit"] == pytest.approx(Av, abs=2e-3)
        assert out["chi2"] < 1e-6

    def test_is_independent_of_the_normalisation(self):
        a = compute_Av_joint_balmer(*make_ladder(0.5, hbeta=1e-18),
                                    law=LAW, **LAW_KW)
        b = compute_Av_joint_balmer(*make_ladder(0.5, hbeta=1e-15),
                                    law=LAW, **LAW_KW)
        assert a["Av_fit"] == pytest.approx(b["Av_fit"], abs=1e-9)

    def test_two_lines_reproduce_that_single_decrement(self):
        """With two lines the fit is exact: dof = 0 and it is the decrement."""
        fluxes, errors = make_ladder(0.45)
        pair = {k: fluxes[k] for k in ("HGAMMA", "Ha")}
        perr = {k: errors[k] for k in ("HGAMMA", "Ha")}
        out = compute_Av_joint_balmer(pair, perr, law=LAW, **LAW_KW)
        wave_g, r_g = _BALMER_LADDER["HGAMMA"]
        wave_a, r_a = _BALMER_LADDER["Ha"]
        direct, _ = compute_Av_from_balmer(
            pair["HGAMMA"], pair["Ha"], perr["HGAMMA"], perr["Ha"],
            law=LAW, intrinsic_ratio=r_g / r_a,
            wave_num_A=wave_g, wave_den_A=wave_a, **LAW_KW,
        )
        assert out["dof"] == 0
        assert out["chi2"] == pytest.approx(0.0, abs=1e-9)
        assert out["Av_fit"] == pytest.approx(direct, abs=2e-3)

    def test_error_is_honest_under_monte_carlo(self):
        rng = np.random.default_rng(5)
        fluxes, errors = make_ladder(0.6)
        draws = [
            compute_Av_joint_balmer(
                {k: v + rng.normal(0.0, errors[k]) for k, v in fluxes.items()},
                errors, law=LAW, **LAW_KW,
            )["Av_fit"]
            for _ in range(800)
        ]
        quoted = compute_Av_joint_balmer(fluxes, errors, law=LAW, **LAW_KW)["Av_err"]
        assert np.std(draws) == pytest.approx(quoted, rel=0.15)
        assert np.mean(draws) == pytest.approx(0.6, abs=0.02)

    def test_chi2_flags_an_inconsistent_ladder(self):
        """No single A_V can fit a ladder with one line pushed off it."""
        fluxes, errors = make_ladder(0.3)
        good = compute_Av_joint_balmer(fluxes, errors, law=LAW, **LAW_KW)
        assert good["chi2"] / max(good["dof"], 1) < 1.0
        # Stellar Balmer absorption eats into Hgamma but not Halpha.
        fluxes["HGAMMA"] *= 0.70
        bad = compute_Av_joint_balmer(fluxes, errors, law=LAW, **LAW_KW)
        assert bad["chi2"] / bad["dof"] > 10.0

    def test_needs_at_least_two_lines(self):
        fluxes, errors = make_ladder(0.3)
        one = {"Ha": fluxes["Ha"]}, {"Ha": errors["Ha"]}
        out = compute_Av_joint_balmer(*one, law=LAW, **LAW_KW)
        assert out["n_lines"] == 0
        assert out["Av"] == 0.0

    def test_skips_lines_below_the_snr_floor(self):
        fluxes, errors = make_ladder(0.3)
        errors["HDELTA"] = fluxes["HDELTA"] / 1.2      # SNR 1.2
        out = compute_Av_joint_balmer(fluxes, errors, law=LAW,
                                      snr_min=3.0, **LAW_KW)
        assert out["n_lines"] == 3
        assert "HDELTA" not in [r["line"] for r in out["individual"]]

    def test_negative_fit_is_reported_but_Av_is_clipped(self):
        fluxes, errors = make_ladder(-0.4)     # unphysically blue ladder
        out = compute_Av_joint_balmer(fluxes, errors, law=LAW, **LAW_KW)
        assert out["Av_fit"] < 0.0
        assert out["Av"] == 0.0

    @pytest.mark.parametrize("law,kw", [("cardelli", {"Rv": 3.1}), ("salim", {})])
    def test_works_for_both_laws(self, law, kw):
        fluxes, errors = {}, {}
        for name in NAMES:
            wave, ratio = _BALMER_LADDER[name]
            k = float(_kappa(np.array([wave]), law, **kw)[0])
            f = 1e-18 * ratio * 10.0 ** (-0.4 * 0.7 * k)
            fluxes[name], errors[name] = f, f / SNR[name]
        out = compute_Av_joint_balmer(fluxes, errors, law=law, **kw)
        assert out["Av_fit"] == pytest.approx(0.7, abs=2e-3)


# ---------------------------------------------------------------------------
# Integration: the estimator reached through compute_abundances.
# ---------------------------------------------------------------------------

@dataclass
class _MockLine:
    name: str
    flux: float
    flux_err: float
    snr: float


@dataclass
class _MockResult:
    lines: dict


def _synthetic_result(Av_true: float, law: str = "cardelli",
                      err_frac: float = 0.03, **dust_kw) -> _MockResult:
    """A full SF-galaxy line set attenuated by *Av_true*."""
    from jwspecabund._core import _LINE_WAVES
    from jwspecabund.dust import cardelli_extinction, salim_attenuation

    intrinsic = {
        "Ha": 2.86, "HBETA": 1.00, "HGAMMA": 0.468, "HDELTA": 0.259,
        "OIII_4363": 0.05, "OIII_4959": 1.65, "OIII_5007": 4.95,
        "OII_3726": 0.85, "OII_3729": 1.20, "NII_6585": 0.30,
    }
    fn = cardelli_extinction if law == "cardelli" else salim_attenuation
    lines = {}
    for name, f_int in intrinsic.items():
        A_lam = fn(np.array([_LINE_WAVES[name]]), Av_true, **dust_kw)[0]
        f = f_int * 10.0 ** (-0.4 * A_lam)
        lines[name] = _MockLine(name=name, flux=f, flux_err=err_frac * f,
                                snr=1.0 / err_frac)
    return _MockResult(lines=lines)


class TestComputeAbundancesIntegration:

    @pytest.mark.parametrize("Av_true", [0.0, 0.7])
    def test_joint_recovers_the_injected_Av(self, Av_true):
        from jwspecabund import compute_abundances

        res = compute_abundances(
            _synthetic_result(Av_true, **LAW_KW), z=6.0, method="direct",
            dust_law=LAW, balmer_method="joint", n_mc=1, progress=False,
            **LAW_KW,
        )
        assert res.Av == pytest.approx(Av_true, abs=0.01)

    def test_joint_and_anchored_agree_on_clean_data(self):
        from jwspecabund import compute_abundances

        kw = dict(z=6.0, method="direct", dust_law=LAW, n_mc=1,
                  progress=False, **LAW_KW)
        result = _synthetic_result(0.5, **LAW_KW)
        joint = compute_abundances(result, balmer_method="joint", **kw)
        anch = compute_abundances(result, balmer_method="anchored",
                                  balmer_anchor="Ha", **kw)
        assert joint.Av == pytest.approx(anch.Av, abs=0.02)
        assert joint.OH == pytest.approx(anch.OH, abs=0.02)

    def test_diagnostics_report_the_joint_fit(self):
        from jwspecabund import compute_abundances

        res = compute_abundances(
            _synthetic_result(0.4, **LAW_KW), z=6.0, method="direct",
            dust_law=LAW, balmer_method="joint", n_mc=1, progress=False,
            **LAW_KW,
        )
        assert "joint fit" in res.diagnostics["A_V"]
        assert "χ²" in res.diagnostics["A_V"]

    def test_invalid_method_is_rejected(self):
        from jwspecabund import compute_abundances

        with pytest.raises(ValueError, match="balmer_method"):
            compute_abundances(
                _synthetic_result(0.3, **LAW_KW), z=6.0,
                balmer_method="weighted", progress=False,
            )

    def test_explicit_Av_still_wins(self):
        from jwspecabund import compute_abundances

        res = compute_abundances(
            _synthetic_result(0.9, **LAW_KW), z=6.0, method="direct",
            dust_law=LAW, balmer_method="joint", Av=0.2, n_mc=1,
            progress=False, **LAW_KW,
        )
        assert res.Av == pytest.approx(0.2)
