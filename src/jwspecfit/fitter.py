"""Core emission-line fitting engine.

The main entry point is :func:`fit_lines`, which:
1. Subtracts a polynomial continuum.
2. Sets up Gaussian line models with resolution-aware widths.
3. Optimises via ``scipy.optimize.least_squares`` with bounds.
4. Returns a :class:`FitResult` with per-line fluxes, EWs, and SNR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import sqrt, pi
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .constraints import ConstraintSet
from .continuum import fit_continuum
from .io import Spectrum
from .lines import REST_LINES_A, get_line_list, observable_lines
from .models import build_model, gaussian_binned, pixel_weight
from .resolution import resolve_R, sigma_inst_A

logger = logging.getLogger(__name__)

_SQRT2PI = sqrt(2.0 * pi)


@dataclass
class LineResult:
    """Fit result for a single emission line.

    Attributes
    ----------
    name : str
        Line name.
    rest_wave_A : float
        Rest-frame wavelength (Å).
    amplitude : float
        Best-fit amplitude (flux × Å, in flux-density units × Å).
    centroid_A : float
        Best-fit observed centroid (Å).
    sigma_A : float
        Best-fit observed Gaussian σ (Å).
    flux : float
        Integrated line flux (amplitude, since profile is area-normalised).
    flux_err : float
        Bootstrap uncertainty on flux.
    ew_A : float
        Rest-frame equivalent width (Å).
    snr : float
        Signal-to-noise ratio (flux / flux_err).
    """

    name: str
    rest_wave_A: float
    amplitude: float
    centroid_A: float
    sigma_A: float
    flux: float
    flux_err: float
    ew_A: float
    snr: float


@dataclass
class FitResult:
    """Container for a complete line-fit result.

    Attributes
    ----------
    lines : dict of LineResult
        Per-line results, keyed by line name.
    params : np.ndarray
        Full best-fit parameter vector.
    model_flux : np.ndarray
        Best-fit emission-line model (continuum-subtracted flux density).
    continuum : np.ndarray
        Best-fit continuum (µJy).
    residuals : np.ndarray
        Fit residuals (data − continuum − model), in µJy.
    chi2 : float
        Reduced χ² of the fit.
    spectrum : Spectrum
        Input spectrum.
    line_names : list of str
        Ordered line names.
    constraints : ConstraintSet
        Applied constraints.
    success : bool
        Whether the optimiser converged.
    """

    lines: dict[str, LineResult]
    params: np.ndarray
    model_flux: np.ndarray
    continuum: np.ndarray
    residuals: np.ndarray
    chi2: float
    spectrum: Spectrum
    line_names: list[str] = field(default_factory=list)
    constraints: ConstraintSet | None = None
    success: bool = True


def _grating_bounds(
    grating: str | None,
    sigma_inst: np.ndarray,
    dlam_A: np.ndarray,
    line_wave_obs_A: float,
) -> tuple[float, float, float]:
    """Return (sigma_lo, sigma_seed, sigma_hi) in Å for a given grating.

    Parameters
    ----------
    grating : str or None
        Grating name.
    sigma_inst : np.ndarray
        Instrumental σ array (Å).
    dlam_A : np.ndarray
        Pixel widths (Å).
    line_wave_obs_A : float
        Observed wavelength of the line (Å).

    Returns
    -------
    tuple of float
        ``(sigma_lo, sigma_seed, sigma_hi)`` in Å.
    """
    # Find σ_inst at the line wavelength.
    idx = np.argmin(np.abs(dlam_A.cumsum() - line_wave_obs_A + dlam_A.cumsum()[0]))
    sig = sigma_inst[min(idx, len(sigma_inst) - 1)]
    pix = np.median(dlam_A)

    g = (grating or "").upper()
    if "PRISM" in g:
        return (0.40 * pix, 0.90 * sig, 1.70 * sig)
    if any(k in g for k in ("G140M", "G235M", "G395M")):
        return (max(0.12 * pix, 0.22 * sig), 0.60 * sig, 1.05 * sig)
    if any(k in g for k in ("G140H", "G235H", "G395H")):
        return (0.10 * pix, 0.50 * sig, 0.95 * sig)
    # Default (e.g. stacked spectra): generous bounds.
    return (0.20 * pix, 0.70 * sig, 1.50 * sig)


def fit_lines(
    spectrum: Spectrum,
    z: float,
    *,
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    deg: int = 2,
    n_boot: int = 200,
    clip_sigma: float = 2.5,
) -> FitResult:
    """Fit emission lines in a spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum.
    z : float
        Source redshift.
    grating : str, optional
        Grating name.  If ``None``, uses ``spectrum.grating``.
    R : float or callable, optional
        Resolving power.  If ``None``, uses ``spectrum.R`` or derives from *grating*.
    lines : list of str, optional
        Lines to fit.  If ``None``, auto-detected from grating and wavelength coverage.
    deg : int
        Continuum polynomial degree (default 2).
    n_boot : int
        Number of bootstrap iterations for uncertainty estimation (default 200).
    clip_sigma : float
        Continuum sigma-clipping threshold.

    Returns
    -------
    FitResult
    """
    spec = spectrum
    grating = grating or spec.grating
    R = R or spec.R

    if grating is None and R is None:
        from .resolution import R_from_pixels
        logger.info("No grating or R specified; estimating R from pixel spacing.")
        R = R_from_pixels(spec.wave_um)

    # Determine which lines to fit.
    if lines is None:
        if grating is not None:
            candidate_lines = get_line_list(grating)
        else:
            candidate_lines = get_line_list("prism")
        line_names = observable_lines(
            candidate_lines, z, spec.wave_um.min(), spec.wave_um.max()
        )
    else:
        line_names = list(lines)

    if len(line_names) == 0:
        logger.warning("No observable lines for z=%.4f in wavelength range.", z)
        return FitResult(
            lines={},
            params=np.array([]),
            model_flux=np.zeros(spec.n_pix),
            continuum=np.zeros(spec.n_pix),
            residuals=spec.flux_ujy.copy(),
            chi2=np.nan,
            spectrum=spec,
            line_names=[],
            success=False,
        )

    logger.info("Fitting %d lines at z=%.4f: %s", len(line_names), z, line_names)

    # Resolution.
    sig_inst = sigma_inst_A(spec.wave_um, grating=grating, R=R)

    # Continuum subtraction.
    continuum = fit_continuum(
        spec.wave_um,
        spec.flux_ujy,
        spec.err_ujy,
        z,
        line_names,
        grating=grating,
        R=R,
        deg=deg,
        clip_sigma=clip_sigma,
    )
    flux_sub = spec.flux_ujy - continuum

    # Convert to F_λ for fitting.
    from .io import _ujy_to_flam, _flam_to_ujy

    flam = _ujy_to_flam(flux_sub, spec.wave_um)
    flam_err = _ujy_to_flam(spec.err_ujy, spec.wave_um)

    valid = np.isfinite(flam) & np.isfinite(flam_err) & (flam_err > 0)

    # Pixel info.
    edges = spec.wave_edges_A
    dlam = spec.dlam_A
    left = edges[:-1]
    right = edges[1:]

    # Pixel weights.
    w_pix = pixel_weight(dlam)

    # Setup constraints.
    constraints = ConstraintSet(line_names)

    # Initial parameters: [amplitudes, centroids, sigmas].
    nL = len(line_names)
    p0 = np.zeros(3 * nL)
    lb = np.zeros(3 * nL)
    ub = np.zeros(3 * nL)

    for i, name in enumerate(line_names):
        lam_obs_A = REST_LINES_A[name] * (1.0 + z)
        sig_lo, sig_seed, sig_hi = _grating_bounds(grating, sig_inst, dlam, lam_obs_A)

        # Find peak flux near the line for amplitude seeding.
        near = np.abs(spec.wave_A - lam_obs_A)
        idx_near = np.where(near < 5 * sig_seed)[0]
        if len(idx_near) > 0:
            peak_flam = np.nanmax(flam[idx_near])
        else:
            peak_flam = np.nanmax(flam[valid]) if np.any(valid) else 1.0

        # Amplitude seed = peak × sqrt(2π) × σ (area under Gaussian).
        A_seed = max(peak_flam * _SQRT2PI * sig_seed, 1e-30)

        # Amplitude bounds.
        p0[i] = A_seed
        lb[i] = 0.0
        ub[i] = 150.0 * max(peak_flam, 1e-30) * _SQRT2PI * sig_hi

        # Centroid bounds.
        cent_margin = max(12.0 * np.median(sig_inst), 4.0 * np.median(dlam))
        if "PRISM" in (grating or "").upper():
            cent_margin = 12.0 * sig_inst[np.argmin(np.abs(spec.wave_A - lam_obs_A))]

        p0[nL + i] = lam_obs_A
        lb[nL + i] = lam_obs_A - cent_margin
        ub[nL + i] = lam_obs_A + cent_margin

        # Sigma bounds.
        p0[2 * nL + i] = sig_seed
        lb[2 * nL + i] = sig_lo
        ub[2 * nL + i] = sig_hi

    # Mask constrained parameters: only optimise free ones.
    free_mask = constraints.free_mask()

    def residual_fn(p_free: np.ndarray) -> np.ndarray:
        """Weighted residuals for least_squares."""
        p_full = constraints.expand_free_to_full(p_free)
        model = build_model(p_full, edges, nL)
        resid = (flam - model) / flam_err
        resid *= w_pix
        resid[~valid] = 0.0
        return resid

    p0_free = p0[free_mask]
    lb_free = lb[free_mask]
    ub_free = ub[free_mask]

    # Clip seeds to bounds.
    p0_free = np.clip(p0_free, lb_free + 1e-30, ub_free - 1e-30)

    result = least_squares(
        residual_fn,
        p0_free,
        bounds=(lb_free, ub_free),
        max_nfev=80000,
        xtol=1e-8,
        ftol=1e-8,
    )

    p_best = constraints.expand_free_to_full(result.x)
    model_flam = build_model(p_best, edges, nL)
    model_ujy = _flam_to_ujy(model_flam, spec.wave_um)
    resid_ujy = flux_sub - model_ujy

    # Chi-squared.
    r = (flam - model_flam) / flam_err
    r[~valid] = np.nan
    n_data = np.sum(valid)
    n_free = np.sum(free_mask)
    dof = max(n_data - n_free, 1)
    chi2_red = float(np.nansum(r**2)) / dof

    # Bootstrap uncertainties.
    flux_errs = _bootstrap_uncertainties(
        flam, flam_err, valid, edges, nL, constraints, free_mask,
        lb_free, ub_free, p0_free, w_pix, n_boot,
    )

    # Build per-line results.
    line_results = {}
    cont_flam = _ujy_to_flam(continuum, spec.wave_um)

    for i, name in enumerate(line_names):
        A = p_best[i]
        mu = p_best[nL + i]
        sig = p_best[2 * nL + i]
        flux_line = A  # Area-normalised Gaussian: integral = amplitude.
        f_err = flux_errs[i] if flux_errs is not None else _analytic_flux_err(
            A, sig, flam_err, spec.wave_A, mu, valid
        )
        snr = flux_line / f_err if f_err > 0 else 0.0

        # Equivalent width (rest-frame).
        lam_rest_A = REST_LINES_A[name]
        idx_cont = np.argmin(np.abs(spec.wave_A - mu))
        cont_at_line = cont_flam[idx_cont] if cont_flam[idx_cont] > 0 else 1e-30
        ew_rest = flux_line / cont_at_line / (1.0 + z)

        line_results[name] = LineResult(
            name=name,
            rest_wave_A=lam_rest_A,
            amplitude=A,
            centroid_A=mu,
            sigma_A=sig,
            flux=flux_line,
            flux_err=f_err,
            ew_A=ew_rest,
            snr=snr,
        )

    return FitResult(
        lines=line_results,
        params=p_best,
        model_flux=model_ujy,
        continuum=continuum,
        residuals=resid_ujy,
        chi2=chi2_red,
        spectrum=spec,
        line_names=line_names,
        constraints=constraints,
        success=bool(result.success),
    )


def _analytic_flux_err(
    A: float,
    sigma_A: float,
    err_flam: np.ndarray,
    wave_A: np.ndarray,
    mu_A: float,
    valid: np.ndarray,
) -> float:
    """Estimate flux uncertainty from local noise near the line.

    Parameters
    ----------
    A : float
        Best-fit amplitude.
    sigma_A : float
        Best-fit line width (Å).
    err_flam : np.ndarray
        Flux error array (erg/s/cm²/Å).
    wave_A : np.ndarray
        Wavelength array (Å).
    mu_A : float
        Line centroid (Å).
    valid : np.ndarray
        Validity mask.

    Returns
    -------
    float
        Estimated flux uncertainty.
    """
    near = np.abs(wave_A - mu_A) < 3.0 * sigma_A
    sel = near & valid
    if np.sum(sel) < 2:
        return np.abs(A) * 0.5  # fallback
    # RMS noise in the line region, scaled by effective width.
    dlam = np.median(np.diff(wave_A[sel]))
    n_eff = _SQRT2PI * sigma_A / dlam if dlam > 0 else 1.0
    rms = np.sqrt(np.nanmean(err_flam[sel] ** 2))
    return rms * dlam * sqrt(n_eff)


def _bootstrap_uncertainties(
    flam: np.ndarray,
    flam_err: np.ndarray,
    valid: np.ndarray,
    edges: np.ndarray,
    nL: int,
    constraints: ConstraintSet,
    free_mask: np.ndarray,
    lb_free: np.ndarray,
    ub_free: np.ndarray,
    p0_free: np.ndarray,
    w_pix: np.ndarray,
    n_boot: int,
) -> np.ndarray | None:
    """Run bootstrap resampling for flux uncertainties.

    Returns
    -------
    np.ndarray or None
        Standard deviation of flux for each line, or None if n_boot == 0.
    """
    if n_boot <= 0:
        return None

    rng = np.random.default_rng(42)
    flux_samples = np.zeros((n_boot, nL))

    for b in range(n_boot):
        # Perturb flux by Gaussian noise.
        noise = rng.standard_normal(len(flam)) * flam_err
        flam_b = flam + noise

        def residual_b(p_free: np.ndarray) -> np.ndarray:
            p_full = constraints.expand_free_to_full(p_free)
            model = build_model(p_full, edges, nL)
            resid = (flam_b - model) / flam_err
            resid *= w_pix
            resid[~valid] = 0.0
            return resid

        try:
            res_b = least_squares(
                residual_b, p0_free, bounds=(lb_free, ub_free),
                max_nfev=20000, xtol=1e-6, ftol=1e-6,
            )
            p_full_b = constraints.expand_free_to_full(res_b.x)
            flux_samples[b, :] = p_full_b[:nL]  # Amplitudes = fluxes.
        except Exception:
            flux_samples[b, :] = np.nan

    return np.nanstd(flux_samples, axis=0)
