"""Tests for JAX-accelerated NUTS sampler."""

import numpy as np
import pytest

from jwspecfit.constraints import ConstraintSet
from jwspecfit.models import build_model
from jwspecmcmc.jax_likelihood import _compile_tying_ops, make_jax_log_likelihood
from jwspecmcmc.likelihood import LikelihoodSpec, log_likelihood
from jwspecmcmc.priors import priors_from_bounds


def _make_synthetic_spec(line_names, true_params, nL, noise_level=5e-18):
    """Build a synthetic LikelihoodSpec for testing."""
    wave_edges = np.linspace(4800, 5100, 301)
    model_true = build_model(true_params, wave_edges, nL)
    rng = np.random.default_rng(42)
    flam = model_true + rng.normal(0, noise_level, len(model_true))
    flam_err = np.full_like(flam, noise_level)
    valid = np.ones(len(flam), dtype=bool)
    cs = ConstraintSet(line_names, tie_nii=False, tie_balmer_to_oiii=True)
    return LikelihoodSpec(
        flam=flam, flam_err=flam_err, valid=valid,
        edges=wave_edges, n_lines=nL, constraints=cs,
    ), cs


class TestCompileTyingOps:
    """Test constraint compilation into tying operations."""

    def test_empty_constraints(self):
        cs = ConstraintSet(
            ["OIII_5007"], tie_nii=False, tie_balmer_to_oiii=False,
            tie_uv_doublets=False,
        )
        ops = _compile_tying_ops(cs)
        assert ops == []

    def test_ciii_always_tied(self):
        cs = ConstraintSet(
            ["CIII]_1907", "CIII]_1909"], tie_nii=False,
            tie_balmer_to_oiii=False, tie_uv_doublets=False,
        )
        ops = _compile_tying_ops(cs)
        # Should have exactly 1 op: sigma tying.
        assert len(ops) == 1
        dst, src, ratio = ops[0]
        nL = 2
        assert dst == 2 * nL + 1  # CIII] sigma
        assert src == 2 * nL + 0  # CIII]_1907 sigma

    def test_balmer_oiii_ops(self):
        cs = ConstraintSet(
            ["OIII_5007", "HBETA"], tie_nii=False,
            tie_balmer_to_oiii=True, tie_uv_doublets=False,
        )
        ops = _compile_tying_ops(cs)
        # HBETA width + centroid tied to OIII = 2 ops.
        assert len(ops) == 2

    def test_nii_doublet_ops(self):
        cs = ConstraintSet(
            ["NII_6585", "NII_6549"], tie_nii=True,
            tie_balmer_to_oiii=False, tie_uv_doublets=False,
        )
        ops = _compile_tying_ops(cs)
        # Amplitude + centroid + sigma = 3 ops.
        assert len(ops) == 3


class TestJaxLikelihood:
    """Test JAX likelihood agrees with NumPy version."""

    def test_agreement(self):
        import jax.numpy as jnp

        nL = 2
        true_params = np.array([
            1e-16, 3e-17, 5007.0, 4861.0, 2.0, 2.0 * 4861.0 / 5007.0,
        ])
        spec, cs = _make_synthetic_spec(
            ["OIII_5007", "HBETA"], true_params, nL,
        )

        log_lik_jax, _ = make_jax_log_likelihood(spec)
        free_mask = cs.free_mask()
        p0_free = true_params[free_mask]

        ll_jax = float(log_lik_jax(jnp.array(p0_free)))
        ll_np = log_likelihood(p0_free, spec)

        np.testing.assert_allclose(ll_jax, ll_np, rtol=1e-6)

    def test_gradient_exists(self):
        """JAX likelihood should be differentiable."""
        import jax
        import jax.numpy as jnp

        nL = 2
        true_params = np.array([
            1e-16, 3e-17, 5007.0, 4861.0, 2.0, 2.0 * 4861.0 / 5007.0,
        ])
        spec, cs = _make_synthetic_spec(
            ["OIII_5007", "HBETA"], true_params, nL,
        )

        log_lik_jax, _ = make_jax_log_likelihood(spec)
        free_mask = cs.free_mask()
        p0_free = true_params[free_mask]

        grad_fn = jax.grad(log_lik_jax)
        grads = grad_fn(jnp.array(p0_free))
        assert grads.shape == p0_free.shape
        assert np.all(np.isfinite(grads))


class TestUntiedMLESeed:
    """Regression: untied MCMC fit must not be wrecked by the MLE seed.

    The MLE initialiser must run under the SAME tying configuration as
    NUTS.  When it ran tied while NUTS ran untied (tie_balmer_to_oiii=
    False), the tied parameter vector mapped onto the untied free layout
    produced a catastrophic seed (log-likelihood ~ -1e16) and every NUTS
    transition diverged.  This test fits a synthetic spectrum untied and
    asserts the fit is healthy.
    """

    def test_untied_fit_not_all_divergent(self):
        import jwspecfit
        import jwspecmcmc

        # Synthetic z=0 spectrum: narrow Hα + broader [OIII]/Hβ, so the
        # untie is meaningful (Hα width differs from [OIII]).
        wave_um = np.linspace(0.480, 0.660, 1800)
        wave_A = wave_um * 1e4
        flux = np.full_like(wave_A, 0.1)
        inject = {  # name: (rest_A, peak, sigma_A)
            "HBETA": (4862.69, 3.0, 1.9),
            "OIII_4959": (4960.29, 4.0, 2.0),
            "OIII_5007": (5008.24, 12.0, 2.0),
            "HALPHA": (6564.63, 15.0, 1.6),   # narrower than [OIII]
        }
        for _, (lam, amp, sig) in inject.items():
            flux += amp * np.exp(-0.5 * ((wave_A - lam) / sig) ** 2)
        rng = np.random.default_rng(0)
        err = np.full_like(flux, 0.05)
        flux += err * rng.standard_normal(len(flux))
        spec = jwspecfit.Spectrum(
            wave_um=wave_um, flux_ujy=flux, err_ujy=err, grating=None,
        )

        result = jwspecmcmc.fit_lines(
            spec, z=0.0, sampler="nuts",
            lines=["HBETA", "OIII_4959", "OIII_5007", "HALPHA"],
            tie_balmer_to_oiii=False,      # the path that used to explode
            init_from_mle=True,
            n_warmup=120, n_samples_nuts=120, n_chains=1,
        )
        n_div = result.sampler_meta.get("n_divergent", 0)
        # Before the fix this was 120/120 (100%).  Allow a small number of
        # genuine divergences but nothing close to all transitions.
        assert n_div < 60, f"too many divergences: {n_div}/120"


class TestRunNuts:
    """Integration test for the NUTS sampler."""

    def test_basic_run(self):
        from jwspecmcmc.samplers import run_nuts

        nL = 2
        true_params = np.array([
            1e-16, 3e-17, 5007.0, 4861.0, 2.0, 2.0 * 4861.0 / 5007.0,
        ])
        spec, cs = _make_synthetic_spec(
            ["OIII_5007", "HBETA"], true_params, nL,
        )

        free_mask = cs.free_mask()
        p0_free = true_params[free_mask]
        lb_free = p0_free * 0.1
        ub_free = p0_free * 10.0
        prior_set = priors_from_bounds(lb_free, ub_free)

        result = run_nuts(
            spec, prior_set, p0_free,
            n_warmup=100, n_samples=200, n_chains=1,
            progress=False, seed=42,
        )

        assert result["sampler_name"] == "nuts"
        assert result["flat_chains"].shape == (200, len(p0_free))
        assert result["flat_log_prob"].shape == (200,)
        assert result["sampler_meta"]["n_divergent"] == 0

        # Posterior median should be within 3x of truth.
        median = np.median(result["flat_chains"], axis=0)
        for i in range(len(p0_free)):
            assert 0.3 * p0_free[i] < median[i] < 3.0 * p0_free[i], (
                f"Param {i}: median={median[i]:.3e}, truth={p0_free[i]:.3e}"
            )


def test_nuts_honours_gaussian_and_loguniform_priors():
    """Non-uniform prior_overrides must change the NUTS posterior, as in emcee.

    run_nuts used to sample every parameter from Uniform(lo, hi) and silently
    ignore Gaussian (and log-uniform) prior densities.
    """
    from jwspecmcmc.priors import GaussianPrior, LogUniformPrior
    from jwspecmcmc.samplers import run_nuts

    true_params = np.array([1e-16, 3e-17, 5007.0, 4861.0, 2.0, 2.0 * 4861.0 / 5007.0])
    spec, cs = _make_synthetic_spec(["OIII_5007", "HBETA"], true_params, 2)
    p0 = true_params[cs.free_mask()]
    # free: A_5007, A_Hb, mu_5007, sigma_5007.  A tight prior at 2x truth on A_Hb
    # dominates the likelihood (width ~1e-17), so the posterior must sit on it.
    overrides = {1: GaussianPrior(mean=6e-17, std=3e-19),
                 3: LogUniformPrior(lo=0.5, hi=8.0)}
    prior_set = priors_from_bounds(p0 * 0.1, p0 * 10.0, overrides=overrides)
    res = run_nuts(spec, prior_set, p0 * np.array([1, 2, 1, 1]), n_warmup=300, n_samples=500,
                   n_chains=1, progress=False, seed=1)
    a_hb = res["flat_chains"][:, 1]
    assert abs(np.median(a_hb) - 6e-17) < 3 * 3e-19
    assert np.std(a_hb) < 1e-18


def test_nuts_seed_on_a_prior_bound_still_samples():
    """A least-squares seed that converged onto a bound must not stall NUTS.

    NUTS works in a logit-transformed space where a point on a bound sits
    at +/-infinity.  On the combined SPURS 437 spectrum several widths end
    on their caps; started there NUTS either could not initialise or
    diverged on every transition (R-hat NaN, ESS ~4).
    """
    from pathlib import Path

    import jwspecfit
    import jwspecmcmc

    z = 5.771060469361607
    path = Path(__file__).resolve().parent.parent / "data" / "spurs437_a2744_combined-medium_spec1d.fits"
    spec = jwspecfit.read_fits(path, z=z)
    lines = ["NIV_1483", "NIV_1486", "CIV_1", "CIV_2", "HEII_1640", "OIII_1661", "OIII_1666",
             "NIII_1750", "CIII]_1907", "CIII]_1909", "OII_doublet", "NeIII_3869", "HDELTA",
             "HGAMMA", "OIII_4363", "HBETA", "OIII_4959", "OIII_5007", "HALPHA", "NII_6585"]
    m = jwspecmcmc.fit_lines(spec, z=z, lines=lines, deg=3, tie_uv_widths=False,
                             fix_blended_ratios=False, n_chains=1, n_warmup=100,
                             n_samples_nuts=100, progress=False, seed=1)
    assert m.sampler_meta["n_divergent"] < 10
    assert m.lines["OIII_5007"].flux_err[0] > 0
