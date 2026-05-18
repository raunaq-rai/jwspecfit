"""Core MCMC fitting orchestrator.

Mirrors the setup logic from :func:`jwspecfit.fitter.fit_lines` —
spectrum preparation, line detection, continuum subtraction, bounds
computation — then delegates to the chosen MCMC sampler and
post-processes the chains into an :class:`~jwspecmcmc.result.MCMCResult`.
"""

from __future__ import annotations

import logging
from math import pi, sqrt
from typing import Any, Callable

import numpy as np

from jwspecfit.constraints import ConstraintSet
from jwspecfit.continuum import fit_continuum
from jwspecfit.fitter import _grating_bounds
from jwspecfit.io import Spectrum, _flam_to_ujy, _ujy_to_flam
from jwspecfit.lines import REST_LINES_A, get_line_list, observable_lines
from jwspecfit.models import asymmetric_gaussian, build_model, pixel_weight
from jwspecfit.resolution import resolve_R, sigma_inst_A

from .diagnostics import summarise_convergence
from .likelihood import LikelihoodSpec
from .priors import GaussianPrior, PriorSet, UniformPrior, priors_from_bounds
from .result import MCMCBroadFitResult, MCMCLineResult, MCMCResult
from .samplers import run_emcee, run_nautilus, run_nuts

logger = logging.getLogger(__name__)

_SQRT2PI = sqrt(2.0 * pi)


def _fit_lines_mcmc(
    spectrum: Spectrum,
    z: float,
    *,
    sampler: str = "emcee",
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    wave_range_A: tuple[float, float] | None = None,
    deg: int = 2,
    clip_sigma: float = 2.5,
    init_from_mle: bool = True,
    prior_overrides: dict[str, Any] | None = None,
    # emcee options
    n_walkers: int | str = "auto",
    n_steps: int = 2000,
    n_burn: int | None = None,
    # nautilus options
    n_live: int = 2000,
    n_eff: int = 10000,
    # nuts options
    n_warmup: int = 500,
    n_samples_nuts: int = 2000,
    n_chains: int = 6,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    # common options
    progress: bool = True,
    seed: int = 42,
    sigma_factor: float = 1.0,
    centroid_vmax: float = 500.0,
    centroid_max_sigma: float = 1.0,
    moving_average: bool | int = False,
    tie_balmer_to_oiii: bool = True,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,
    sigma_overrides: dict[str, tuple[float, float]] | None = None,
    centroid_overrides: dict[str, tuple[float, float]] | None = None,
    niv_doublet_ratio: float | None = None,
    ciii_doublet_ratio: float | None = None,
) -> MCMCResult:
    """Fit emission lines using MCMC sampling.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum.
    z : float
        Source redshift.
    sampler : str
        Sampler backend: ``"emcee"`` or ``"nautilus"`` (default ``"emcee"``).
    grating : str, optional
        Grating name.
    R : float or callable, optional
        Resolving power.
    lines : list of str, optional
        Lines to fit.
    wave_range_A : tuple, optional
        Observed wavelength range (Angstrom).
    deg : int
        Continuum polynomial degree.
    clip_sigma : float
        Continuum sigma-clipping threshold.
    init_from_mle : bool
        If ``True`` (default), initialise walkers from a least-squares
        MLE fit via :func:`jwspecfit.fit_lines`.
    prior_overrides : dict, optional
        Per-parameter prior overrides.  Keys are parameter names like
        ``"A_OIII_5007"`` or ``"sigma_Ha"``, values are
        :class:`~jwspecmcmc.priors.Prior` instances.
    n_walkers : int or ``"auto"``
        Number of emcee walkers (ignored for nautilus).  ``"auto"``
        (default) picks a value based on ``n_dim`` and CPU cores.
    n_steps : int
        Number of emcee MCMC steps (ignored for nautilus).
    n_burn : int or None
        Emcee burn-in steps (auto-estimated if ``None``).
    n_live : int
        Nautilus live points (ignored for emcee).
    n_eff : int
        Nautilus target effective sample size (ignored for emcee).
    progress : bool
        Show a progress bar.
    seed : int
        Random seed.
    sigma_factor : float
        Multiplicative factor on the upper line-width bound.
        Use values > 1 for stacked spectra (default 1.0).
    moving_average : bool or int
        If ``False`` (default), use polynomial continuum.  If ``True``,
        use a median filter with a default window of 75 pixels.  If an
        ``int``, use that as the median-filter window size.
    tie_uv_doublets : bool
        Tie UV doublet kinematics and fix resonance-line flux ratios.
        Recommended for stacked spectra where doublets are poorly
        resolved (default ``False``).
    tie_uv_centroids : bool
        Tie UV doublet secondary centroids to their primaries in
        velocity space (default ``True``).  Set ``False`` for
        well-resolved spectra.
    tie_uv_widths : bool
        Tie UV intercombination line widths to a shared velocity
        dispersion (default ``True``).  Set ``False`` for
        well-resolved spectra.
    sigma_overrides : dict, optional
        Per-line sigma (width) bounds in Angstroms, keyed by line name.
        Each value is ``(sigma_lo, sigma_hi)``.  Overrides the automatic
        grating-based bounds for the specified lines.

    Returns
    -------
    MCMCResult
    """
    # ------------------------------------------------------------------
    # 1. Spectrum setup (mirrors fitter.py lines 202-260)
    # ------------------------------------------------------------------
    spec = spectrum
    grating = grating or spec.grating
    R = R or spec.R

    if grating is None and R is None:
        from jwspecfit.resolution import R_from_pixels
        logger.info("No grating or R specified; estimating R from pixel spacing.")
        R = R_from_pixels(spec.wave_um)

    if wave_range_A is not None:
        lo_A, hi_A = wave_range_A
        mask_win = (spec.wave_A >= lo_A) & (spec.wave_A <= hi_A)
        if np.sum(mask_win) < 10:
            raise ValueError(
                f"Fit window [{lo_A:.0f}, {hi_A:.0f}] Å contains only "
                f"{np.sum(mask_win)} pixels."
            )
        spec = Spectrum(
            wave_um=spec.wave_um[mask_win],
            flux_ujy=spec.flux_ujy[mask_win],
            err_ujy=spec.err_ujy[mask_win],
            grating=spec.grating,
            z=spec.z,
            R=spec.R,
            meta=spec.meta,
        )

    # ------------------------------------------------------------------
    # 2. Line detection
    # ------------------------------------------------------------------
    if lines is None:
        if grating is not None:
            candidate_lines = get_line_list(grating)
        else:
            # Infer line list from estimated resolving power.
            R_arr = resolve_R(spec.wave_um, R=R)
            R_med = float(np.median(R_arr))
            if R_med > 500:
                candidate_lines = get_line_list("grating")
                logger.info("Median R ≈ %.0f → using resolved line list.", R_med)
            else:
                candidate_lines = get_line_list("prism")
                logger.info("Median R ≈ %.0f → using prism line list.", R_med)
        line_names = observable_lines(
            candidate_lines, z, spec.wave_um.min(), spec.wave_um.max()
        )
    else:
        line_names = list(lines)

    if len(line_names) == 0:
        raise ValueError(f"No observable lines for z={z:.4f} in wavelength range.")

    nL = len(line_names)
    logger.info("MCMC fitting %d lines at z=%.4f: %s", nL, z, line_names)

    # ------------------------------------------------------------------
    # 3. Resolution and continuum
    # ------------------------------------------------------------------
    sig_inst = sigma_inst_A(spec.wave_um, grating=grating, R=R)
    R_arr = resolve_R(spec.wave_um, grating=grating, R=R)
    R_med = float(np.median(R_arr))
    R_lo, R_hi = float(np.min(R_arr)), float(np.max(R_arr))
    if abs(R_hi - R_lo) < 10:
        print(f"Resolving power: R = {R_med:.0f}")
    else:
        print(f"Resolving power: R ≈ {R_med:.0f} (range {R_lo:.0f}–{R_hi:.0f})")

    # Auto-enable lya_break when Lya is in the line list.
    _lya_in_lines = "Lya" in line_names
    _lya_break = _lya_in_lines

    continuum = fit_continuum(
        spec.wave_um, spec.flux_ujy, spec.err_ujy, z, line_names,
        grating=grating, R=R, deg=deg, clip_sigma=clip_sigma,
        moving_average=moving_average,
        lya_break=_lya_break,
    )
    flux_sub = spec.flux_ujy - continuum

    flam = _ujy_to_flam(flux_sub, spec.wave_um)
    flam_err = _ujy_to_flam(spec.err_ujy, spec.wave_um)

    valid = np.isfinite(flam) & np.isfinite(flam_err) & (flam_err > 0)
    if _lya_in_lines:
        _blue_cutoff_A = 1150.0 * (1.0 + z)
        valid &= spec.wave_A >= _blue_cutoff_A
    else:
        nv_obs_A = REST_LINES_A["NV_1"] * (1.0 + z)
        valid &= spec.wave_A >= nv_obs_A

    edges = spec.wave_edges_A
    dlam = spec.dlam_A
    w_pix = pixel_weight(dlam)

    # ------------------------------------------------------------------
    # 3b. Drop lines in detector gaps (< 50% valid pixels within ±5σ)
    # ------------------------------------------------------------------
    _kept: list[str] = []
    for name in line_names:
        lam_obs_A = REST_LINES_A[name] * (1.0 + z)
        _, sig_seed, _ = _grating_bounds(grating, sig_inst, dlam, lam_obs_A, sigma_factor)
        near_mask = np.abs(spec.wave_A - lam_obs_A) < 5 * sig_seed
        n_valid = int(np.sum(valid & near_mask))
        n_total = int(np.sum(near_mask))
        frac = n_valid / n_total if n_total > 0 else 0.0
        if frac >= 0.5:
            _kept.append(name)
        else:
            logger.info(
                "Dropping %s (obs %.0f A): %d/%d valid pixels (%.0f%%) in ±5sigma",
                name, lam_obs_A, n_valid, n_total, 100 * frac,
            )
    if len(_kept) < len(line_names):
        logger.info("Kept %d / %d lines after gap filtering.", len(_kept), len(line_names))
        line_names = _kept
        nL = len(line_names)

    if nL == 0:
        raise ValueError("All lines fall in detector gaps — nothing to fit.")

    # ------------------------------------------------------------------
    # 3c. Separate Lyα from regular Gaussian lines
    # ------------------------------------------------------------------
    _lya_params = None  # (p0, lb, ub) for 4-element vector
    _n_lya = 0
    if _lya_in_lines:
        line_names = [n for n in line_names if n != "Lya"]
        nL = len(line_names)
        lya_obs_A = REST_LINES_A["Lya"] * (1.0 + z)

        idx_lya = np.argmin(np.abs(spec.wave_A - lya_obs_A))
        local_sig = sig_inst[idx_lya]
        near_lya = np.abs(spec.wave_A - lya_obs_A) < 10 * local_sig
        # Find the actual peak pixel location.
        _near_idx = np.where(near_lya & valid)[0]
        if len(_near_idx) > 0:
            _peak_idx = _near_idx[np.argmax(flam[_near_idx])]
            peak_lya = flam[_peak_idx]
            peak_wave_A = spec.wave_A[_peak_idx]
        else:
            peak_lya = 1.0
            peak_wave_A = lya_obs_A + 1.0
        if not np.isfinite(peak_lya) or peak_lya <= 0:
            peak_lya = 1.0

        _C_KMS = 299792.458
        cent_margin = max(500.0 / _C_KMS * lya_obs_A, 5.0)

        # Single asymmetric Gaussian: [A_peak, mu, sigma, alpha].
        # σ controls blue-side sharpness; α handles the red tail.
        sig_seed = max(local_sig, 0.5)
        sig_lo = 0.2
        sig_hi = 3.0 * sigma_factor

        A_peak_seed = 0.5 * peak_lya
        alpha_seed = 5.0

        lya_p0 = np.array([A_peak_seed, peak_wave_A, sig_seed, alpha_seed])
        lya_lb = np.array([0.0, lya_obs_A - cent_margin, sig_lo, 0.0])
        lya_ub = np.array([150.0 * peak_lya, lya_obs_A + cent_margin, sig_hi, 30.0])
        _lya_params = (lya_p0, lya_lb, lya_ub)
        _n_lya = 4
        logger.info(
            "Lyα: asymmetric Gaussian (σ [%.1f, %.1f] Å, seed=%.2f, "
            "α [0, 30]), peak at %.1f Å",
            sig_lo, sig_hi, sig_seed, peak_wave_A,
        )

    # ------------------------------------------------------------------
    # 4. Constraints and bounds (mirrors fitter.py lines 300-398)
    # ------------------------------------------------------------------
    # Detect unresolved intercombination doublets (same logic as fitter.py).
    from jwspecfit.constraints import _INTERCOM_LOW_DENSITY_RATIOS

    _BLEND_THRESHOLD = 4.71  # 2 × FWHM in sigma units
    _blended: set[str] = set()
    if tie_uv_doublets:
        for (pri, sec) in _INTERCOM_LOW_DENSITY_RATIOS:
            if pri in line_names and sec in line_names:
                lam_pri = REST_LINES_A[pri] * (1.0 + z)
                lam_sec = REST_LINES_A[sec] * (1.0 + z)
                sep = abs(lam_pri - lam_sec)
                idx_near_pri = np.argmin(np.abs(spec.wave_A - lam_pri))
                sig_at_line = sig_inst[idx_near_pri]
                if sep < _BLEND_THRESHOLD * sig_at_line:
                    _blended.add(sec)
                    logger.info(
                        "Blended: %s–%s (sep=%.1f Å < %.1f×σ_inst=%.1f Å) → fixing ratio",
                        pri, sec, sep, _BLEND_THRESHOLD, _BLEND_THRESHOLD * sig_at_line,
                    )

    constraints = ConstraintSet(
        line_names, tie_balmer_to_oiii=tie_balmer_to_oiii,
        tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids, tie_uv_widths=tie_uv_widths,
        blended_doublets=_blended if _blended else None,
        niv_doublet_ratio=niv_doublet_ratio,
        ciii_doublet_ratio=ciii_doublet_ratio,
    )

    p0 = np.zeros(3 * nL)
    lb = np.zeros(3 * nL)
    ub = np.zeros(3 * nL)

    for i, name in enumerate(line_names):
        lam_obs_A = REST_LINES_A[name] * (1.0 + z)
        sig_lo, sig_seed, sig_hi = _grating_bounds(grating, sig_inst, dlam, lam_obs_A, sigma_factor)

        # Peak flux near the line for amplitude seeding.
        near = np.abs(spec.wave_A - lam_obs_A)
        idx_near = np.where(near < 5 * sig_seed)[0]
        is_abs = name.startswith("abs_")

        if is_abs:
            if len(idx_near) > 0:
                peak_flam = abs(np.nanmin(flam[idx_near]))
            else:
                peak_flam = 1.0
        else:
            if len(idx_near) > 0:
                peak_flam = np.nanmax(flam[idx_near])
            else:
                peak_flam = np.nanmax(flam[valid]) if np.any(valid) else 1.0
        if not np.isfinite(peak_flam) or peak_flam <= 0:
            peak_flam = 1.0

        A_seed = max(peak_flam * _SQRT2PI * sig_seed, 1e-30)

        if is_abs:
            p0[i] = -A_seed
            lb[i] = -150.0 * max(peak_flam, 1e-30) * _SQRT2PI * sig_hi
            ub[i] = 0.0
        else:
            p0[i] = A_seed
            lb[i] = 0.0
            ub[i] = 150.0 * max(peak_flam, 1e-30) * _SQRT2PI * sig_hi

        # Centroid bounds — match jwspecfit.fitter logic so the LS seed
        # and the MCMC priors agree.  Velocity-space cap for everyone,
        # plus a resolution-aware cap (``centroid_max_sigma × σ_inst``)
        # for narrow lines only; broad components keep velocity-only
        # so real outflow blueshifts aren't clipped.
        _C_KMS_CENT = 299792.458
        local_sig = sig_inst[np.argmin(np.abs(spec.wave_A - lam_obs_A))]
        cent_margin_v = centroid_vmax / _C_KMS_CENT * lam_obs_A
        if "BROAD" in name:
            cent_margin = cent_margin_v
        else:
            cent_margin_res = centroid_max_sigma * local_sig
            cent_margin = min(cent_margin_v, cent_margin_res)
        cent_margin = max(cent_margin, 2.0 * np.median(dlam))

        # Exclude same-rest-wavelength neighbours so broad components
        # (e.g. OIII_5007_BROAD) don't collapse to lb == ub on their
        # centroid against their own narrow counterpart.  Mirrors the
        # same fix in jwspecfit.fitter._fit_one_window.
        own_rest = REST_LINES_A[name]
        other_obs = [
            REST_LINES_A[n] * (1.0 + z)
            for j, n in enumerate(line_names)
            if j != i and "BROAD" not in n and REST_LINES_A[n] != own_rest
        ]
        if other_obs:
            min_sep = min(abs(lam_obs_A - o) for o in other_obs)
            cent_margin = min(cent_margin, 0.5 * min_sep)

        # Apply per-line centroid overrides if provided.
        if centroid_overrides and name in centroid_overrides:
            cent_lo, cent_hi = centroid_overrides[name]
            p0[nL + i] = 0.5 * (cent_lo + cent_hi)
            lb[nL + i] = cent_lo
            ub[nL + i] = cent_hi
        else:
            p0[nL + i] = lam_obs_A
            lb[nL + i] = lam_obs_A - cent_margin
            ub[nL + i] = lam_obs_A + cent_margin

        # Sigma bounds.
        _C_KMS = 299792.458
        if "_BROAD2" in name:
            from jwspecfit.broad import (
                BROAD2_SIGMA_V_HI, BROAD2_SIGMA_V_LO, BROAD2_SIGMA_V_SEED,
            )
            sig_lo = BROAD2_SIGMA_V_LO / _C_KMS * lam_obs_A
            sig_seed = BROAD2_SIGMA_V_SEED / _C_KMS * lam_obs_A
            sig_hi = BROAD2_SIGMA_V_HI / _C_KMS * lam_obs_A
        elif "_BROAD" in name:
            from jwspecfit.broad import (
                BROAD1_SIGMA_V_HI, BROAD1_SIGMA_V_LO, BROAD1_SIGMA_V_SEED,
            )
            sig_lo = BROAD1_SIGMA_V_LO / _C_KMS * lam_obs_A
            sig_seed = BROAD1_SIGMA_V_SEED / _C_KMS * lam_obs_A
            sig_hi = BROAD1_SIGMA_V_HI / _C_KMS * lam_obs_A

        _MAX_SIGMA_A = 500.0
        sig_hi = min(sig_hi, _MAX_SIGMA_A)
        sig_seed = min(sig_seed, 0.9 * sig_hi)

        # Apply per-line sigma overrides if provided.
        if sigma_overrides and name in sigma_overrides:
            sig_lo, sig_hi = sigma_overrides[name]
            sig_hi = min(sig_hi, _MAX_SIGMA_A)
            sig_seed = 0.5 * (sig_lo + sig_hi)

        p0[2 * nL + i] = sig_seed
        lb[2 * nL + i] = sig_lo
        ub[2 * nL + i] = sig_hi

    free_mask = constraints.free_mask()
    p0_free = p0[free_mask]
    lb_free = lb[free_mask]
    ub_free = ub[free_mask]

    # Append Lyα parameters if present.
    if _lya_params is not None:
        lya_p0, lya_lb, lya_ub = _lya_params
        p0_free = np.concatenate([p0_free, lya_p0])
        lb_free = np.concatenate([lb_free, lya_lb])
        ub_free = np.concatenate([ub_free, lya_ub])

    p0_free = np.clip(p0_free, lb_free + 1e-30, ub_free - 1e-30)

    # ------------------------------------------------------------------
    # 5. Optional MLE initialisation
    # ------------------------------------------------------------------
    # Include Lya in the MLE line list so it gets initialised too.
    _mle_lines = (["Lya"] + line_names) if _lya_in_lines else line_names

    if init_from_mle:
        logger.info("Running quick MLE fit for MCMC initialisation...")
        from jwspecfit.fitter import fit_lines as _fit_lines_mle

        mle_result = _fit_lines_mle(
            spec, z,
            grating=grating, R=R, lines=_mle_lines,
            deg=deg, n_boot=0, clip_sigma=clip_sigma,
            tie_uv_doublets=tie_uv_doublets,
            tie_uv_centroids=tie_uv_centroids,
            tie_uv_widths=tie_uv_widths,
            sigma_overrides=sigma_overrides,
            centroid_overrides=centroid_overrides,
            niv_doublet_ratio=niv_doublet_ratio,
            ciii_doublet_ratio=ciii_doublet_ratio,
            lya_break=_lya_break,
            moving_average=moving_average,
            sigma_factor=sigma_factor,
        )
        if mle_result.success:
            # Map MLE params back by line name — the MLE fit may have
            # fewer lines (e.g. detector-gap filtering drops some).
            # Lya is handled separately (not in the Gaussian params),
            # so exclude it from the Gaussian param index.
            _mle_gauss_names = [n for n in mle_result.line_names if n != "Lya"]
            nL_mle = len(_mle_gauss_names)
            mle_idx = {n: j for j, n in enumerate(_mle_gauss_names)}
            for i, name in enumerate(line_names):
                if name in mle_idx:
                    j = mle_idx[name]
                    p0[i] = mle_result.params[j]                       # amplitude
                    p0[nL + i] = mle_result.params[nL_mle + j]         # centroid
                    p0[2 * nL + i] = mle_result.params[2 * nL_mle + j] # sigma
            p0_free = p0[free_mask]
            # Re-append Lyα with MLE values if available.
            if _lya_params is not None:
                _mle_lya = getattr(mle_result, "lya_params", None)
                if _mle_lya is not None and len(_mle_lya) == 4:
                    # Use the 4-parameter asymmetric Gaussian from MLE.
                    lya_p0 = np.array(_mle_lya, dtype=np.float64)
                    # Tighten σ bounds around MLE value.
                    _mle_sig_n = _mle_lya[2]
                    lya_lb[2] = max(lya_lb[2], 0.8 * _mle_sig_n)
                    lya_ub[2] = min(lya_ub[2], 1.3 * _mle_sig_n)
                    # Also tighten centroid.
                    _mle_mu_n = _mle_lya[1]
                    lya_lb[1] = max(lya_lb[1], _mle_mu_n - 0.5)
                    lya_ub[1] = min(lya_ub[1], _mle_mu_n + 0.5)
                    # Update the master bounds arrays (Lyα is at the end).
                    _lya_start = len(lb_free) - _n_lya
                    lb_free[_lya_start:] = lya_lb
                    ub_free[_lya_start:] = lya_ub
                    lya_p0 = np.clip(lya_p0, lya_lb + 1e-30, lya_ub - 1e-30)
                    logger.info(
                        "Lyα MCMC bounds tightened: σ [%.2f, %.2f], μ [%.1f, %.1f]",
                        lya_lb[2], lya_ub[2], lya_lb[1], lya_ub[1],
                    )
                elif "Lya" in mle_result.lines:
                    lr = mle_result.lines["Lya"]
                    lya_p0 = np.array([
                        peak_lya, lr.centroid_A, lr.sigma_A, 3.0,
                    ])
                    lya_p0 = np.clip(lya_p0, lya_lb + 1e-30, lya_ub - 1e-30)
                p0_free = np.concatenate([p0_free, lya_p0])
            p0_free = np.clip(p0_free, lb_free + 1e-30, ub_free - 1e-30)
            logger.info(
                "MLE initialisation successful (chi2=%.2f, %d/%d lines matched).",
                mle_result.chi2, len(mle_idx), nL,
            )
        else:
            logger.warning("MLE fit did not converge; using default initialisation.")

    # ------------------------------------------------------------------
    # 6. Build priors
    # ------------------------------------------------------------------
    # Map named overrides to free-parameter indices.
    idx_map = {name: i for i, name in enumerate(line_names)}
    index_overrides: dict[int, Any] = {}

    if prior_overrides:
        for pname, prior_obj in prior_overrides.items():
            # Parse names like "A_OIII_5007", "mu_Ha", "sigma_OIII_5007".
            if pname.startswith("A_"):
                line_key = pname[2:]
                if line_key in idx_map:
                    full_idx = idx_map[line_key]
                else:
                    logger.warning("Unknown line in prior override: %s", pname)
                    continue
            elif pname.startswith("mu_"):
                line_key = pname[3:]
                if line_key in idx_map:
                    full_idx = nL + idx_map[line_key]
                else:
                    logger.warning("Unknown line in prior override: %s", pname)
                    continue
            elif pname.startswith("sigma_"):
                line_key = pname[6:]
                if line_key in idx_map:
                    full_idx = 2 * nL + idx_map[line_key]
                else:
                    logger.warning("Unknown line in prior override: %s", pname)
                    continue
            else:
                logger.warning("Cannot parse prior override key: %s", pname)
                continue

            # Map full-param index to free-param index.
            if free_mask[full_idx]:
                free_idx = int(np.sum(free_mask[:full_idx]))
                index_overrides[free_idx] = prior_obj
            else:
                logger.warning(
                    "Prior override '%s' targets a constrained parameter; skipping.", pname
                )

    prior_set = priors_from_bounds(lb_free, ub_free, overrides=index_overrides)

    # ------------------------------------------------------------------
    # 7. Build likelihood spec
    # ------------------------------------------------------------------
    # Build Lyα model function if needed.
    _lya_model_fn = None
    if _n_lya > 0:
        left = edges[:-1]
        right = edges[1:]
        _centres = 0.5 * (left + right)
        def _lya_model_fn(p_lya):
            return asymmetric_gaussian(
                _centres, p_lya[0], p_lya[1], p_lya[2], p_lya[3],
            )

    # Cap inflated errors near the Lyα peak (same logic as fitter.py).
    if _n_lya > 0:
        _lya_vicinity = np.abs(spec.wave_A - lya_obs_A) < 15.0
        _med_err = np.nanmedian(flam_err[_lya_vicinity & valid])
        if np.isfinite(_med_err) and _med_err > 0:
            _peak_region = np.abs(spec.wave_A - peak_wave_A) < 3.0
            flam_err = flam_err.copy()
            flam_err[_peak_region] = np.minimum(flam_err[_peak_region], _med_err)

    like_spec = LikelihoodSpec(
        flam=flam,
        flam_err=flam_err,
        valid=valid,
        edges=edges,
        n_lines=nL,
        constraints=constraints,
        w_pix=w_pix,
        n_lya=_n_lya,
        lya_model_fn=_lya_model_fn,
    )

    # ------------------------------------------------------------------
    # 8. Run sampler
    # ------------------------------------------------------------------
    if sampler == "emcee":
        sampler_result = run_emcee(
            like_spec, prior_set, p0_free,
            n_walkers=n_walkers, n_steps=n_steps, n_burn=n_burn,
            progress=progress, seed=seed,
        )
    elif sampler == "nautilus":
        sampler_result = run_nautilus(
            like_spec, prior_set,
            n_live=n_live, n_eff=n_eff,
            progress=progress, seed=seed,
        )
    elif sampler == "nuts":
        sampler_result = run_nuts(
            like_spec, prior_set, p0_free,
            n_warmup=n_warmup, n_samples=n_samples_nuts,
            n_chains=n_chains,
            progress=progress, seed=seed,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )
    else:
        raise ValueError(f"Unknown sampler: '{sampler}'. Use 'emcee', 'nautilus', or 'nuts'.")

    # ------------------------------------------------------------------
    # 9. Post-process: expand chains to full param space
    # ------------------------------------------------------------------
    flat_chains_free = sampler_result["flat_chains"]
    flat_log_prob = sampler_result["flat_log_prob"]
    n_samples = len(flat_chains_free)

    # Split off Lyα chains if present.
    _lya_chains = None
    if _n_lya > 0:
        _lya_chains = flat_chains_free[:, -_n_lya:]  # (n_samples, 4)
        flat_chains_free_gauss = flat_chains_free[:, :-_n_lya]
    else:
        flat_chains_free_gauss = flat_chains_free

    # Expand each sample to the full parameter space.
    flat_chains_full = np.zeros((n_samples, 3 * nL))
    for s in range(n_samples):
        flat_chains_full[s] = constraints.expand_free_to_full(flat_chains_free_gauss[s])

    # Median posterior.
    p_median = np.median(flat_chains_full, axis=0)
    model_flam_median = build_model(p_median, edges, nL)

    # Add Lyα model to the median model.
    _lya_best = None
    if _lya_chains is not None and _lya_model_fn is not None:
        _lya_best = np.median(_lya_chains, axis=0)
        model_flam_median = model_flam_median + _lya_model_fn(_lya_best)

    model_ujy_median = _flam_to_ujy(model_flam_median, spec.wave_um)

    # ------------------------------------------------------------------
    # 10. Per-line MCMCLineResult
    # ------------------------------------------------------------------
    cont_flam = _ujy_to_flam(continuum, spec.wave_um)
    line_results: dict[str, MCMCLineResult] = {}

    for i, name in enumerate(line_names):
        # Amplitude posteriors (flux = amplitude for area-normalised Gaussians).
        amp_samples = flat_chains_full[:, i]
        mu_samples = flat_chains_full[:, nL + i]
        sig_samples = flat_chains_full[:, 2 * nL + i]

        amp_med = float(np.median(amp_samples))
        amp_lo = amp_med - float(np.percentile(amp_samples, 16))
        amp_hi = float(np.percentile(amp_samples, 84)) - amp_med

        mu_med = float(np.median(mu_samples))
        mu_lo = mu_med - float(np.percentile(mu_samples, 16))
        mu_hi = float(np.percentile(mu_samples, 84)) - mu_med

        sig_med = float(np.median(sig_samples))
        sig_lo = sig_med - float(np.percentile(sig_samples, 16))
        sig_hi = float(np.percentile(sig_samples, 84)) - sig_med

        flux_med = amp_med
        flux_lo = amp_lo
        flux_hi = amp_hi

        # Equivalent width from median values.
        lam_rest_A = REST_LINES_A[name]
        idx_cont = np.argmin(np.abs(spec.wave_A - mu_med))
        cont_at_line = cont_flam[idx_cont]
        if cont_at_line <= 0:
            near_mask = np.abs(spec.wave_A - mu_med) < 5.0 * sig_med
            near_valid = near_mask & valid
            if np.any(near_valid):
                local_median_ujy = np.nanmedian(spec.flux_ujy[near_valid])
                cont_at_line = _ujy_to_flam(
                    np.array([max(local_median_ujy, 0.0)]),
                    np.array([spec.wave_um[idx_cont]]),
                )[0]
        ew_rest = flux_med / cont_at_line / (1.0 + z) if cont_at_line > 0 else np.nan

        # SNR from mean of asymmetric errors.
        mean_err = 0.5 * (flux_lo + flux_hi)
        snr = flux_med / mean_err if mean_err > 0 else 0.0

        line_results[name] = MCMCLineResult(
            name=name,
            rest_wave_A=lam_rest_A,
            amplitude=amp_med,
            amplitude_err=(amp_lo, amp_hi),
            centroid_A=mu_med,
            centroid_err=(mu_lo, mu_hi),
            sigma_A=sig_med,
            sigma_err=(sig_lo, sig_hi),
            flux=flux_med,
            flux_err=(flux_lo, flux_hi),
            flux_posterior=amp_samples,
            ew_A=ew_rest,
            snr=snr,
        )

    # ------------------------------------------------------------------
    # 10b. Lyα MCMCLineResult
    # ------------------------------------------------------------------
    if _lya_chains is not None:
        # 4 params: [A_peak, mu, sigma, alpha]
        A_peak_samples = _lya_chains[:, 0]
        mu_samples_lya = _lya_chains[:, 1]
        sig_samples_lya = _lya_chains[:, 2]
        alpha_samples = _lya_chains[:, 3]

        # Numerical flux integration for each sample via the median model.
        _lya_best = np.median(_lya_chains, axis=0)
        _lya_median_model = _lya_model_fn(_lya_best)
        _dwave = np.diff(edges)
        flux_med_numerical = float(np.sum(_lya_median_model * _dwave))

        # Per-sample flux posterior via analytical approximation:
        # flux ≈ A_peak × sigma × sqrt(2π).
        flux_samples = A_peak_samples * sig_samples_lya * _SQRT2PI

        flux_med = float(np.median(flux_samples))
        flux_lo = flux_med - float(np.percentile(flux_samples, 16))
        flux_hi = float(np.percentile(flux_samples, 84)) - flux_med

        mu_med = float(np.median(mu_samples_lya))
        mu_lo = mu_med - float(np.percentile(mu_samples_lya, 16))
        mu_hi = float(np.percentile(mu_samples_lya, 84)) - mu_med

        sig_med = float(np.median(sig_samples_lya))
        sig_lo = sig_med - float(np.percentile(sig_samples_lya, 16))
        sig_hi = float(np.percentile(sig_samples_lya, 84)) - sig_med

        mean_err = 0.5 * (flux_lo + flux_hi)
        snr_lya = flux_med / mean_err if mean_err > 0 else 0.0

        idx_cont_lya = np.argmin(np.abs(spec.wave_A - mu_med))
        cont_at_lya = cont_flam[idx_cont_lya]
        ew_lya = flux_med / cont_at_lya / (1.0 + z) if cont_at_lya > 0 else np.nan

        line_results["Lya"] = MCMCLineResult(
            name="Lya",
            rest_wave_A=REST_LINES_A["Lya"],
            amplitude=float(np.median(A_peak_samples)),
            amplitude_err=(
                float(np.median(A_peak_samples)) - float(np.percentile(A_peak_samples, 16)),
                float(np.percentile(A_peak_samples, 84)) - float(np.median(A_peak_samples)),
            ),
            centroid_A=mu_med,
            centroid_err=(mu_lo, mu_hi),
            sigma_A=sig_med,
            sigma_err=(sig_lo, sig_hi),
            flux=flux_med,
            flux_err=(flux_lo, flux_hi),
            flux_posterior=flux_samples,
            ew_A=ew_lya,
            snr=snr_lya,
        )
        line_names = ["Lya"] + line_names

        logger.info(
            "Lyα MCMC: A_peak=%.3e, μ=%.1f Å, σ=%.2f Å, α=%.2f, "
            "flux=%.3e (+%.3e/-%.3e)",
            float(np.median(A_peak_samples)), mu_med, sig_med,
            float(np.median(alpha_samples)),
            flux_med, flux_hi, flux_lo,
        )

    # ------------------------------------------------------------------
    # 11. Convergence diagnostics (emcee + NUTS; nautilus has no chains)
    # ------------------------------------------------------------------
    convergence: dict[str, Any] = {}
    chains_raw = sampler_result.get("chains")
    if chains_raw is not None:
        convergence = summarise_convergence(chains_raw)

    # ------------------------------------------------------------------
    # 12. Assemble MCMCResult
    # ------------------------------------------------------------------
    return MCMCResult(
        lines=line_results,
        flat_chains=flat_chains_full,
        flat_chains_free=flat_chains_free,
        flat_log_prob=flat_log_prob,
        chains=chains_raw,
        params=p_median,
        model_flux=model_ujy_median,
        continuum=continuum,
        spectrum=spec,
        line_names=line_names,
        constraints=constraints,
        convergence=convergence,
        sampler_name=sampler_result["sampler_name"],
        sampler_meta=sampler_result["sampler_meta"],
        lya_params=_lya_best,
    )


def _fit_with_broad_mcmc(
    spectrum: Spectrum,
    z: float,
    *,
    sampler: str = "emcee",
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    wave_range_A: tuple[float, float] | None = None,
    deg: int = 2,
    clip_sigma: float = 2.5,
    fit_balmer_broad: bool = False,
    fit_oiii_broad: bool = False,
    fit_hei_broad: bool = False,
    n_boot_bic: int = 100,
    n_jobs: int = -1,
    snr_threshold: float = 5.0,
    oiii_snr_threshold: float = 5.0,
    hei_snr_threshold: float = 5.0,
    bic_delta: float = 6.0,
    prior_overrides: dict[str, Any] | None = None,
    # emcee options
    n_walkers: int | str = "auto",
    n_steps: int = 2000,
    n_burn: int | None = None,
    # nautilus options
    n_live: int = 2000,
    n_eff: int = 10000,
    # nuts options
    n_warmup: int = 500,
    n_samples_nuts: int = 2000,
    n_chains: int = 6,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    # common options
    progress: bool = True,
    seed: int = 42,
    sigma_factor: float = 1.0,
    centroid_vmax: float = 500.0,
    centroid_max_sigma: float = 1.0,
    moving_average: bool | int = False,
    tie_balmer_to_oiii: bool = True,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,
    sigma_overrides: dict[str, tuple[float, float]] | None = None,
    centroid_overrides: dict[str, tuple[float, float]] | None = None,
    niv_doublet_ratio: float | None = None,
    ciii_doublet_ratio: float | None = None,
) -> MCMCBroadFitResult:
    """Fit emission lines with BIC-based broad selection, then MCMC.

    Phase 1 uses :func:`jwspecfit.fit_with_broad` (fast least-squares)
    for BIC model selection.  Phase 2 runs MCMC on the winning model's
    line list, seeded from the MLE parameters.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum.
    z : float
        Source redshift.
    sampler : str
        ``"emcee"`` (default) or ``"nautilus"``.
    grating : str, optional
        Grating name.
    R : float or callable, optional
        Resolving power.
    lines : list of str, optional
        Narrow line list (broad entries are added automatically).
    wave_range_A : tuple, optional
        Observed wavelength range (Angstrom).
    deg : int
        Continuum polynomial degree.
    clip_sigma : float
        Continuum sigma-clipping threshold.
    fit_balmer_broad : bool
        If ``True`` (default), run BIC selection for a broad Balmer
        component (narrow vs. intermediate vs. very-broad vs. both).
    fit_oiii_broad : bool
        If ``True`` (default), run an independent BIC test for a broad
        component on [OIII] 5007/4959 (outflow signature).
    n_boot_bic : int
        Bootstrap iterations for BIC model selection (default 100).
        Set to 0 for single-point BIC.
    n_jobs : int
        Parallel jobs for BIC bootstrap (default ``-1`` = all cores).
    snr_threshold : float
        Minimum Ha SNR to attempt Balmer broad fitting (default 5.0).
    oiii_snr_threshold : float
        Minimum [OIII] 5007/4959 SNR to attempt OIII broad fitting
        (default 5.0).
    bic_delta : float
        ΔBIC threshold for accepting a more complex model (default 6.0).
    prior_overrides : dict, optional
        Per-parameter prior overrides for MCMC.
    n_walkers : int
        Emcee walkers (default 64).
    n_steps : int
        Emcee steps (default 2000).
    n_burn : int or None
        Emcee burn-in (auto if ``None``).
    n_live : int
        Nautilus live points (default 2000).
    n_eff : int
        Nautilus target effective samples (default 10000).
    progress : bool
        Show progress bar.
    seed : int
        Random seed.
    moving_average : bool or int
        If ``False`` (default), use polynomial continuum.  If ``True``,
        use a median filter with a default window of 75 pixels.  If an
        ``int``, use that as the median-filter window size.

    Returns
    -------
    MCMCBroadFitResult
    """
    from jwspecfit.broad import fit_with_broad as _fit_with_broad_mle

    # ------------------------------------------------------------------
    # Phase 1: BIC model selection via fast least-squares
    # ------------------------------------------------------------------
    logger.info(
        "Phase 1: BIC selection via jwspecfit.fit_with_broad "
        "(balmer=%s, oiii=%s, hei=%s)",
        fit_balmer_broad, fit_oiii_broad, fit_hei_broad,
    )

    bic_result = _fit_with_broad_mle(
        spectrum, z,
        grating=grating, R=R, lines=lines,
        deg=deg,
        fit_balmer_broad=fit_balmer_broad,
        fit_oiii_broad=fit_oiii_broad,
        fit_hei_broad=fit_hei_broad,
        n_boot=0,
        n_boot_bic=n_boot_bic,
        n_jobs=n_jobs,
        snr_threshold=snr_threshold,
        oiii_snr_threshold=oiii_snr_threshold,
        hei_snr_threshold=hei_snr_threshold,
        bic_delta=bic_delta,
        sigma_factor=sigma_factor,
        centroid_vmax=centroid_vmax,
        centroid_max_sigma=centroid_max_sigma,
        moving_average=moving_average,
        tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids,
        tie_uv_widths=tie_uv_widths,
        sigma_overrides=sigma_overrides,
        centroid_overrides=centroid_overrides,
        niv_doublet_ratio=niv_doublet_ratio,
        ciii_doublet_ratio=ciii_doublet_ratio,
    )

    selected_model = bic_result.selected_model
    best_mle = bic_result.best_fit
    winning_lines = best_mle.line_names

    logger.info(
        "BIC selected model: %s (%d lines: %s)",
        selected_model, len(winning_lines), winning_lines,
    )

    # Build _p0_hint from the MLE best-fit parameters.
    nL_mle = len(best_mle.line_names)
    p0_hint: dict[str, tuple[float, float, float]] = {}
    for i, name in enumerate(best_mle.line_names):
        p0_hint[name] = (
            best_mle.params[i],                # amplitude
            best_mle.params[nL_mle + i],       # centroid
            best_mle.params[2 * nL_mle + i],   # sigma
        )

    # ------------------------------------------------------------------
    # Phase 2: MCMC on the winning model
    # ------------------------------------------------------------------
    logger.info("Phase 2: MCMC sampling on %s model", selected_model)

    mcmc_result = _fit_lines_mcmc(
        spectrum, z,
        sampler=sampler,
        grating=grating, R=R,
        lines=winning_lines,
        wave_range_A=wave_range_A,
        deg=deg,
        clip_sigma=clip_sigma,
        init_from_mle=True,
        prior_overrides=prior_overrides,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        n_live=n_live,
        n_eff=n_eff,
        n_warmup=n_warmup,
        n_samples_nuts=n_samples_nuts,
        n_chains=n_chains,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        progress=progress,
        seed=seed,
        sigma_factor=sigma_factor,
        centroid_vmax=centroid_vmax,
        centroid_max_sigma=centroid_max_sigma,
        moving_average=moving_average,
        tie_balmer_to_oiii=tie_balmer_to_oiii,
        tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids,
        tie_uv_widths=tie_uv_widths,
        sigma_overrides=sigma_overrides,
        centroid_overrides=centroid_overrides,
        niv_doublet_ratio=niv_doublet_ratio,
        ciii_doublet_ratio=ciii_doublet_ratio,
    )

    return MCMCBroadFitResult(
        mcmc_result=mcmc_result,
        selected_model=selected_model,
        bic_narrow=bic_result.bic_narrow,
        bic_broad1=bic_result.bic_broad1,
        bic_broad2=bic_result.bic_broad2,
        bic_both=bic_result.bic_both,
        oiii_selected=bic_result.oiii_selected,
        bic_oiii_off=bic_result.bic_oiii_off,
        bic_oiii_broad1=bic_result.bic_oiii_broad1,
        bic_oiii_broad2=bic_result.bic_oiii_broad2,
        bic_oiii_both=bic_result.bic_oiii_both,
        hei_selected=bic_result.hei_selected,
        bic_hei_off=bic_result.bic_hei_off,
        bic_hei_broad1=bic_result.bic_hei_broad1,
        bic_hei_broad2=bic_result.bic_hei_broad2,
        bic_hei_both=bic_result.bic_hei_both,
    )
