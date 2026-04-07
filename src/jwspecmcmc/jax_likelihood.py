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

from jwspecfit.constraints import (
    ConstraintSet,
    NII_RATIO,
    NV_RATIO,
    _INTERCOM_LOW_DENSITY_RATIOS,
)
from jwspecfit.lines import REST_LINES_A

from .likelihood import LikelihoodSpec

logger = logging.getLogger(__name__)

_SQRT2 = sqrt(2.0)


def _compile_tying_ops(cs: ConstraintSet) -> list[tuple[int, int, float]]:
    """Extract ordered (dst, src, ratio) tying operations from a ConstraintSet.

    Each operation means: ``p[dst] = p[src] * ratio``.
    The operations are returned in the same order as ``ConstraintSet.apply()``
    so that dependent constraints see their sources already updated.

    Parameters
    ----------
    cs : ConstraintSet
        The constraint set to compile.

    Returns
    -------
    list of (int, int, float)
        Ordered tying operations.
    """
    nL = len(cs.line_names)
    idx = {name: i for i, name in enumerate(cs.line_names)}
    ops: list[tuple[int, int, float]] = []

    # --- Balmer/OIII tying ---
    if cs.tie_balmer_to_oiii and "OIII_5007" in idx:
        i_o3 = idx["OIII_5007"]
        lam_o3 = REST_LINES_A["OIII_5007"]

        width_targets = [
            "Ha", "HBETA", "HGAMMA", "HDELTA",
            "HEPSILON", "H8", "H9", "H10",
        ]
        for name in width_targets:
            if name in idx:
                ratio = REST_LINES_A[name] / lam_o3
                ops.append((2 * nL + idx[name], 2 * nL + i_o3, ratio))

        centroid_targets = [
            "Ha", "HBETA", "HGAMMA", "HDELTA",
            "HEPSILON", "H8", "H9", "H10",
        ]
        for name in centroid_targets:
            if name in idx:
                ratio = REST_LINES_A[name] / lam_o3
                ops.append((nL + idx[name], nL + i_o3, ratio))

    # --- Broad centroid tying ---
    _BROAD_PAIRS = [
        ("Ha_BROAD", "Ha"), ("Ha_BROAD2", "Ha"),
        ("HBETA_BROAD", "HBETA"), ("HBETA_BROAD2", "HBETA"),
        ("HDELTA_BROAD", "HDELTA"), ("HDELTA_BROAD2", "HDELTA"),
        ("HGAMMA_BROAD", "HGAMMA"), ("HGAMMA_BROAD2", "HGAMMA"),
    ]
    for broad_name, narrow_name in _BROAD_PAIRS:
        if broad_name in idx and narrow_name in idx:
            ops.append((nL + idx[broad_name], nL + idx[narrow_name], 1.0))

    # --- NII doublet ---
    if cs.tie_nii and "NII_6549" in idx and "NII_6585" in idx:
        i49 = idx["NII_6549"]
        i85 = idx["NII_6585"]
        lam_ratio = REST_LINES_A["NII_6549"] / REST_LINES_A["NII_6585"]
        ops.append((i49, i85, NII_RATIO))               # amplitude
        ops.append((nL + i49, nL + i85, lam_ratio))      # centroid
        ops.append((2 * nL + i49, 2 * nL + i85, lam_ratio))  # sigma

    # --- UV doublet constraints ---
    if cs.tie_uv_doublets:
        # Amplitude-tied (NV)
        _AMPLITUDE_TIED = [
            ("NV_1", "NV_2", NV_RATIO),
        ]
        for pri, sec, ratio in _AMPLITUDE_TIED:
            if pri in idx and sec in idx:
                i_pri, i_sec = idx[pri], idx[sec]
                lam_ratio = REST_LINES_A[sec] / REST_LINES_A[pri]
                ops.append((i_sec, i_pri, ratio))                     # amplitude
                if cs.tie_uv_centroids:
                    ops.append((nL + i_sec, nL + i_pri, lam_ratio))   # centroid
                ops.append((2 * nL + i_sec, 2 * nL + i_pri, lam_ratio))  # sigma

        # Kinematic-tied intercombination doublets
        _KINEMATIC_TIED = [
            ("CIV_1", "CIV_2"),
            ("OIII_1666", "OIII_1661"),
            ("CIII]_1907", "CIII]"),
            ("NIV_1486", "NIV_1483"),
            ("NIII_1749", "NIII_1752"),
            ("SiIII_1", "SiIII_2"),
        ]
        _blended = cs.blended_doublets or set()
        for pri, sec in _KINEMATIC_TIED:
            if pri in idx and sec in idx:
                i_pri, i_sec = idx[pri], idx[sec]
                lam_ratio = REST_LINES_A[sec] / REST_LINES_A[pri]
                if cs.tie_uv_centroids:
                    ops.append((nL + i_sec, nL + i_pri, lam_ratio))
                ops.append((2 * nL + i_sec, 2 * nL + i_pri, lam_ratio))
                if sec in _blended:
                    ratio = _INTERCOM_LOW_DENSITY_RATIOS.get((pri, sec), 0.67)
                    ops.append((i_sec, i_pri, ratio))

        # UV intercombination width tying
        if cs.tie_uv_widths:
            _UV_INTERCOM = ["CIII]_1907", "NIV_1486", "NIII_1749"]
            _uv_present = [n for n in _UV_INTERCOM if n in idx]
            if len(_uv_present) >= 2:
                anchor = _uv_present[0]
                i_anchor = idx[anchor]
                lam_anchor = REST_LINES_A[anchor]
                for name in _uv_present[1:]:
                    ratio = REST_LINES_A[name] / lam_anchor
                    ops.append((2 * nL + idx[name], 2 * nL + i_anchor, ratio))

    # --- Always-tie-width doublets (unconditional) ---
    _ALWAYS_TIE_WIDTH = [
        ("CIII]_1907", "CIII]"),
    ]
    for pri, sec in _ALWAYS_TIE_WIDTH:
        if pri in idx and sec in idx:
            lam_ratio = REST_LINES_A[sec] / REST_LINES_A[pri]
            ops.append((2 * nL + idx[sec], 2 * nL + idx[pri], lam_ratio))

    return ops


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
    ops = _compile_tying_ops(constraints)

    # Convert to Python tuples for static tracing.
    ops_tuples = [(int(d), int(s), float(r)) for d, s, r in ops]
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
    w_pix = jnp.array(spec.w_pix, dtype=jnp.float64)
    valid = jnp.array(spec.valid, dtype=jnp.bool_)

    # Combined weight: 1/(err) * w_pix, zeroed for invalid pixels.
    inv_err_w = jnp.where(valid, w_pix / flam_err, 0.0)

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
        for d, s, r in ops_tuples:
            full = full.at[d].set(full[s] * r)

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

        # Add two-component Lyα if present:
        # p_lya = [A_narrow, mu_narrow, sig_narrow,
        #          A_broad, mu_broad, sig_broad, skew_broad]
        if _has_lya:
            # Narrow component: bin-averaged Gaussian via erf.
            inv_n = 1.0 / (_SQRT2 * p_lya[2])
            cdf_r_n = 0.5 * (1.0 + jax.lax.erf((right - p_lya[1]) * inv_n))
            cdf_l_n = 0.5 * (1.0 + jax.lax.erf((left - p_lya[1]) * inv_n))
            narrow = p_lya[0] * (cdf_r_n - cdf_l_n) / widths

            # Broad component: skewed Gaussian at bin centres.
            t = (centres - p_lya[4]) / p_lya[5]
            phi = jnp.exp(-0.5 * t**2) / (_SQRT2 * jnp.sqrt(jnp.pi) * p_lya[5])
            big_phi = 0.5 * (1.0 + jax.lax.erf(p_lya[6] * t / _SQRT2))
            broad = p_lya[3] * 2.0 * phi * big_phi

            # Zero broad component blueward of Lyα rest (1215.67 Å).
            # For de-redshifted stacks (z=0), this is just the rest wavelength.
            broad = jnp.where(centres >= 1215.67, broad, 0.0)
            model = model + narrow + broad

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
