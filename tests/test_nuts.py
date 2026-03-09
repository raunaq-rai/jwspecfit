"""Tests for JAX-accelerated NUTS sampler."""

import numpy as np
import pytest

from jwspecfit.constraints import ConstraintSet
from jwspecfit.models import build_model, pixel_weight
from jwspecmcmc.jax_likelihood import _compile_tying_ops, make_jax_log_likelihood
from jwspecmcmc.likelihood import LikelihoodSpec, log_likelihood
from jwspecmcmc.priors import priors_from_bounds


def _make_synthetic_spec(line_names, true_params, nL, noise_level=5e-18):
    """Build a synthetic LikelihoodSpec for testing."""
    wave_edges = np.linspace(4800, 5100, 301)
    dlam = np.diff(wave_edges)
    w_pix = pixel_weight(dlam)
    model_true = build_model(true_params, wave_edges, nL)
    rng = np.random.default_rng(42)
    flam = model_true + rng.normal(0, noise_level, len(model_true))
    flam_err = np.full_like(flam, noise_level)
    valid = np.ones(len(flam), dtype=bool)
    cs = ConstraintSet(line_names, tie_nii=False, tie_balmer_to_oiii=True)
    return LikelihoodSpec(
        flam=flam, flam_err=flam_err, valid=valid,
        edges=wave_edges, n_lines=nL, constraints=cs, w_pix=w_pix,
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
            ["CIII]_1907", "CIII]"], tie_nii=False,
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
