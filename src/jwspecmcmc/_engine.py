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
from jwspecfit.models import build_model, pixel_weight
from jwspecfit.resolution import resolve_R, sigma_inst_A

from .diagnostics import summarise_convergence
from .likelihood import LikelihoodSpec
from .priors import GaussianPrior, PriorSet, UniformPrior, priors_from_bounds
from .result import MCMCBroadFitResult, MCMCLineResult, MCMCResult
from .samplers import run_emcee, run_nautilus

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
    # common options
    progress: bool = True,
    seed: int = 42,
    sigma_factor: float = 1.0,
    moving_average: bool | int = False,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,

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

    continuum = fit_continuum(
        spec.wave_um, spec.flux_ujy, spec.err_ujy, z, line_names,
        grating=grating, R=R, deg=deg, clip_sigma=clip_sigma,
        moving_average=moving_average,
    )
    flux_sub = spec.flux_ujy - continuum

    flam = _ujy_to_flam(flux_sub, spec.wave_um)
    flam_err = _ujy_to_flam(spec.err_ujy, spec.wave_um)

    valid = np.isfinite(flam) & np.isfinite(flam_err) & (flam_err > 0)
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
    # 4. Constraints and bounds (mirrors fitter.py lines 300-398)
    # ------------------------------------------------------------------
    constraints = ConstraintSet(
        line_names, tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids, tie_uv_widths=tie_uv_widths,

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
        if len(idx_near) > 0:
            peak_flam = np.nanmax(flam[idx_near])
        else:
            peak_flam = np.nanmax(flam[valid]) if np.any(valid) else 1.0
        if not np.isfinite(peak_flam) or peak_flam <= 0:
            peak_flam = 1.0

        A_seed = max(peak_flam * _SQRT2PI * sig_seed, 1e-30)

        p0[i] = A_seed
        lb[i] = 0.0
        ub[i] = 150.0 * max(peak_flam, 1e-30) * _SQRT2PI * sig_hi

        # Centroid bounds.
        _C_KMS_CENT = 299792.458
        _CENT_V_MAX = 500.0
        cent_margin_v = _CENT_V_MAX / _C_KMS_CENT * lam_obs_A
        cent_margin = max(cent_margin_v, 2.0 * np.median(dlam))

        other_obs = [
            REST_LINES_A[n] * (1.0 + z)
            for j, n in enumerate(line_names)
            if j != i and "BROAD" not in n
        ]
        if other_obs:
            min_sep = min(abs(lam_obs_A - o) for o in other_obs)
            cent_margin = min(cent_margin, 0.5 * min_sep)

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

        p0[2 * nL + i] = sig_seed
        lb[2 * nL + i] = sig_lo
        ub[2 * nL + i] = sig_hi

    free_mask = constraints.free_mask()
    p0_free = p0[free_mask]
    lb_free = lb[free_mask]
    ub_free = ub[free_mask]
    p0_free = np.clip(p0_free, lb_free + 1e-30, ub_free - 1e-30)

    # ------------------------------------------------------------------
    # 5. Optional MLE initialisation
    # ------------------------------------------------------------------
    if init_from_mle:
        logger.info("Running quick MLE fit for MCMC initialisation...")
        from jwspecfit.fitter import fit_lines as _fit_lines_mle

        mle_result = _fit_lines_mle(
            spec, z,
            grating=grating, R=R, lines=line_names,
            deg=deg, n_boot=0, clip_sigma=clip_sigma,
            tie_uv_centroids=tie_uv_centroids,
            tie_uv_widths=tie_uv_widths,
    
        )
        if mle_result.success:
            # Map MLE params back by line name — the MLE fit may have
            # fewer lines (e.g. detector-gap filtering drops some).
            nL_mle = len(mle_result.line_names)
            mle_idx = {n: j for j, n in enumerate(mle_result.line_names)}
            for i, name in enumerate(line_names):
                if name in mle_idx:
                    j = mle_idx[name]
                    p0[i] = mle_result.params[j]                       # amplitude
                    p0[nL + i] = mle_result.params[nL_mle + j]         # centroid
                    p0[2 * nL + i] = mle_result.params[2 * nL_mle + j] # sigma
            p0_free = p0[free_mask]
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
    like_spec = LikelihoodSpec(
        flam=flam,
        flam_err=flam_err,
        valid=valid,
        edges=edges,
        n_lines=nL,
        constraints=constraints,
        w_pix=w_pix,
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
    else:
        raise ValueError(f"Unknown sampler: '{sampler}'. Use 'emcee' or 'nautilus'.")

    # ------------------------------------------------------------------
    # 9. Post-process: expand chains to full param space
    # ------------------------------------------------------------------
    flat_chains_free = sampler_result["flat_chains"]
    flat_log_prob = sampler_result["flat_log_prob"]
    n_samples = len(flat_chains_free)

    # Expand each sample to the full parameter space.
    flat_chains_full = np.zeros((n_samples, 3 * nL))
    for s in range(n_samples):
        flat_chains_full[s] = constraints.expand_free_to_full(flat_chains_free[s])

    # Median posterior.
    p_median = np.median(flat_chains_full, axis=0)
    model_flam_median = build_model(p_median, edges, nL)
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
    # 11. Convergence diagnostics (emcee only)
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
    mode: str = "auto",
    n_boot_bic: int = 100,
    n_jobs: int = -1,
    snr_threshold: float = 5.0,
    bic_delta: float = 6.0,
    prior_overrides: dict[str, Any] | None = None,
    # emcee options
    n_walkers: int | str = "auto",
    n_steps: int = 2000,
    n_burn: int | None = None,
    # nautilus options
    n_live: int = 2000,
    n_eff: int = 10000,
    # common options
    progress: bool = True,
    seed: int = 42,
    sigma_factor: float = 1.0,
    moving_average: bool | int = False,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,

) -> MCMCBroadFitResult:
    """Fit emission lines with BIC-based broad Balmer selection, then MCMC.

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
    mode : str
        Broad component mode: ``"auto"`` (BIC selection, default),
        ``"off"`` (narrow only), ``"broad1"``, ``"broad2"``, ``"both"``.
    n_boot_bic : int
        Bootstrap iterations for BIC model selection (default 100).
        Set to 0 for single-point BIC.
    n_jobs : int
        Parallel jobs for BIC bootstrap (default ``-1`` = all cores).
    snr_threshold : float
        Minimum Ha SNR to attempt broad fitting (default 5.0).
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
    logger.info("Phase 1: BIC model selection via jwspecfit.fit_with_broad(mode=%r)", mode)

    bic_result = _fit_with_broad_mle(
        spectrum, z,
        grating=grating, R=R, lines=lines,
        deg=deg, mode=mode,
        n_boot=0,
        n_boot_bic=n_boot_bic,
        n_jobs=n_jobs,
        snr_threshold=snr_threshold,
        bic_delta=bic_delta,
        sigma_factor=sigma_factor,
        moving_average=moving_average,
        tie_uv_centroids=tie_uv_centroids,
        tie_uv_widths=tie_uv_widths,

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
        progress=progress,
        seed=seed,
        sigma_factor=sigma_factor,
        moving_average=moving_average,
        tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids,
        tie_uv_widths=tie_uv_widths,

    )

    return MCMCBroadFitResult(
        mcmc_result=mcmc_result,
        selected_model=selected_model,
        bic_narrow=bic_result.bic_narrow,
        bic_broad1=bic_result.bic_broad1,
        bic_broad2=bic_result.bic_broad2,
        bic_both=bic_result.bic_both,
    )
