"""JAX-accelerated likelihood for NUTS/HMC sampling.

Provides JIT-compiled versions of the emission-line model and
log-likelihood that are compatible with JAX's automatic
differentiation, enabling gradient-based samplers like NumPyro NUTS.

The constraint system is "compiled" into static index arrays so that
the entire likelihood — from free parameters to chi-squared — is a
single differentiable JAX computation graph.
"""

from __future__ import annotations

import logging
from math import sqrt

import numpy as np

from jwspecfit.constraints import ConstraintSet

from .likelihood import LikelihoodSpec

logger = logging.getLogger(__name__)

_SQRT2 = sqrt(2.0)


def _compile_tying_ops(cs: ConstraintSet) -> list[tuple[int, int, float]]:
    """Ordered ``(dst, src, ratio)`` tying operations, ``p[dst] = p[src] * ratio``.

    Delegates to :meth:`ConstraintSet.tying_ops` so the JAX likelihood uses
    exactly the same ties as the NumPy one.
    """
    return cs.tying_ops()


def make_jax_log_likelihood(
    spec: LikelihoodSpec,
) -> tuple:
    """Build a JIT-compiled JAX log-likelihood function.

    Parameters
    ----------
    spec : LikelihoodSpec
        Cached data for likelihood evaluation (NumPy arrays).

    Returns
    -------
    log_likelihood_jax : callable
        ``f(p_free) -> float``, JIT-compiled.
    static_data : dict
        Pre-computed JAX arrays and metadata for the sampler.
    """
    import jax
    import jax.numpy as jnp

    # Enable 64-bit precision (essential for wavelengths in Angstroms).
    jax.config.update("jax_enable_x64", True)

    constraints = spec.constraints
    nL = spec.n_lines
    free_mask = constraints.free_mask()
    free_idx = np.where(free_mask)[0]
    ops_full = constraints.tying_ops_full()

    # Convert to Python tuples for static tracing.
    ops_tuples = [(int(d), int(s), float(r)) for d, s, r, _, _ in ops_full]
    ops_jax = [(int(d), int(s), float(r), float(a_s), float(a_d))
               for d, s, r, a_s, a_d in ops_full]
    free_idx_jax = jnp.array(free_idx, dtype=jnp.int32)

    # Pre-compute static data as JAX arrays.
    # Sanitize non-finite values to 0 — invalid pixels are masked by
    # inv_err_w=0, but JAX evaluates all arithmetic (NaN * 0 = NaN in
    # IEEE 754), so we must ensure no NaN enters the computation.
    _flam_safe = np.where(spec.valid, spec.flam, 0.0)
    _err_safe = np.where(spec.valid, spec.flam_err, 1.0)

    left = jnp.array(spec.edges[:-1], dtype=jnp.float64)
    right = jnp.array(spec.edges[1:], dtype=jnp.float64)
    widths = right - left
    flam = jnp.array(_flam_safe, dtype=jnp.float64)
    flam_err = jnp.array(_err_safe, dtype=jnp.float64)
    valid = jnp.array(spec.valid, dtype=jnp.bool_)

    # Inverse error, zeroed for invalid pixels.
    inv_err_w = jnp.where(valid, 1.0 / flam_err, 0.0)

    n_full = 3 * nL
    _n_lya = spec.n_lya
    _has_lya = _n_lya > 0

    # Pre-compute bin centres for the skewed Gaussian (used only for Lyα).
    if _has_lya:
        centres = 0.5 * (left + right)

    @jax.jit
    def log_likelihood_jax(p_free: jnp.ndarray) -> float:
        # Split off Lyα parameters if present.
        if _has_lya:
            p_gauss = p_free[:-_n_lya]
            p_lya = p_free[-_n_lya:]
        else:
            p_gauss = p_free

        # Expand free → full parameter vector.
        full = jnp.zeros(n_full, dtype=jnp.float64)
        full = full.at[free_idx_jax].set(p_gauss)

        # Apply tying ops (loop unrolled at trace time).
        for d, s, r, a_s, a_d in ops_jax:
            x = full[s]
            if a_s > 0:
                # Shared intrinsic velocity, own instrumental width
                # (jwspecfit.constraints.tied_width).  The floor keeps the
                # unselected branch finite so its gradient is not NaN.
                quad = jnp.sqrt(jnp.maximum(r * r * (x * x - a_s) + a_d, 1e-300))
                x = jnp.where(x * x <= a_s, x * sqrt(a_d / a_s), quad)
            else:
                x = x * r
            full = full.at[d].set(x)

        # Unpack parameters.
        amps = full[:nL]
        mus = full[nL:2 * nL]
        sigs = full[2 * nL:]

        # Build model: bin-averaged Gaussians via erf.
        inv = 1.0 / (_SQRT2 * sigs)  # (nL,)
        cdf_r = 0.5 * (1.0 + jax.lax.erf(
            (right[:, None] - mus[None, :]) * inv[None, :]
        ))
        cdf_l = 0.5 * (1.0 + jax.lax.erf(
            (left[:, None] - mus[None, :]) * inv[None, :]
        ))
        area = cdf_r - cdf_l  # (n_pix, nL)
        profiles = area / widths[:, None]  # (n_pix, nL)
        model = profiles @ amps  # (n_pix,)

        # Lyα: single asymmetric Gaussian (Bolan+2025 parameterisation).
        # p_lya = [A_peak, mu, sigma, alpha]
        if _has_lya:
            t = (centres - p_lya[1]) / p_lya[2]
            gauss = p_lya[0] * jnp.exp(-0.5 * t**2)
            skew_term = 1.0 + jax.lax.erf(p_lya[3] * t / _SQRT2)
            model = model + gauss * skew_term

        # Weighted residual.
        resid = (flam - model) * inv_err_w
        return -0.5 * jnp.dot(resid, resid)

    # Bounds in free-parameter space.
    lb_free = np.zeros(len(free_idx))
    ub_free = np.zeros(len(free_idx))
    # These will be set by the caller from the engine.

    static_data = {
        "free_idx": free_idx,
        "ops": ops_tuples,
        "n_lines": nL,
        "n_free": len(free_idx) + _n_lya,
        "n_lya": _n_lya,
    }

    return log_likelihood_jax, static_data
