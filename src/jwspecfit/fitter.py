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
from pathlib import Path
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
    snr_int_err : float
        Integrated SNR: flux / bootstrap flux error.
    snr_int_cont : float
        Integrated SNR: flux / (local continuum RMS × √N_pix).
    snr_peak_err : float
        Peak SNR: model peak / uncertainty array at peak pixel.
    snr_peak_cont : float
        Peak SNR: model peak / local continuum RMS.
    """

    name: str
    rest_wave_A: float
    amplitude: float
    centroid_A: float
    sigma_A: float
    flux: float
    flux_err: float
    ew_A: float
    snr_int_err: float
    snr_int_cont: float
    snr_peak_err: float
    snr_peak_cont: float

    def upper_limit(self, n_sigma: float = 3.0) -> float:
        """Return n-sigma flux upper limit."""
        return n_sigma * self.flux_err

    @property
    def snr(self) -> float:
        """Backward-compatible alias for snr_int_err."""
        return self.snr_int_err


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
    sigma_factor: float = 1.0,
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
    sigma_factor : float
        Multiplicative factor applied to the upper width bound.
        Use values > 1 for stacked spectra where redshift scatter
        broadens lines beyond the instrumental resolution (default 1.0).

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
        return (0.40 * pix, 0.90 * sig, 1.70 * sig * sigma_factor)
    if any(k in g for k in ("G140M", "G235M", "G395M")):
        return (max(0.12 * pix, 0.22 * sig), 0.60 * sig, 2.0 * sig * sigma_factor)
    if any(k in g for k in ("G140H", "G235H", "G395H")):
        return (0.10 * pix, 0.50 * sig, 1.8 * sig * sigma_factor)
    # Default (e.g. stacked spectra): generous bounds — lines are broadened
    # by galaxy velocity dispersions and the stacking process.
    return (0.20 * pix, 1.0 * sig, 5.0 * sig * sigma_factor)


def fit_lines(
    spectrum: Spectrum,
    z: float,
    *,
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    wave_range_A: tuple[float, float] | None = None,
    deg: int = 2,
    n_boot: int = 1000,
    clip_sigma: float = 2.5,
    n_jobs: int = -1,
    save_path: str | Path | None = None,
    sigma_factor: float = 1.0,
    centroid_vmax: float = 500.0,
    moving_average: bool | int = False,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,
    _label: str = "",
    _p0_hint: dict[str, tuple[float, float, float]] | None = None,
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
    wave_range_A : tuple of float, optional
        ``(lo, hi)`` observed wavelength range in Angstroms.  If given, only
        pixels within this range are used for continuum fitting and line
        fitting.  Lines outside the range are excluded.
    deg : int
        Continuum polynomial degree (default 2).
    n_boot : int
        Number of bootstrap iterations for uncertainty estimation (default 200).
    clip_sigma : float
        Continuum sigma-clipping threshold.
    n_jobs : int
        Number of parallel jobs for bootstrap. ``-1`` uses all cores,
        ``1`` runs sequentially. Default ``-1``.
    save_path : str or Path, optional
        If given, export per-line measurements to this text file after fitting.
    sigma_factor : float
        Multiplicative factor applied to the upper line-width bound.
        Use values > 1 for stacked spectra where redshift scatter
        broadens lines beyond the instrumental resolution (default 1.0).
    centroid_vmax : float
        Maximum centroid offset in km/s (default 500).  Increase for
        stacked spectra with larger velocity offsets between lines.
    moving_average : bool or int
        If ``False`` (default), use polynomial continuum.  If ``True``,
        use a median filter with a default window of 75 pixels.  If an
        ``int``, use that as the median-filter window size.
    tie_uv_doublets : bool
        Tie UV doublet kinematics and fix resonance-line flux ratios.
        Recommended for stacked spectra where doublets are poorly
        resolved (default ``False``).

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

    # Apply fit window: crop spectrum to wavelength range.
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
        logger.info(
            "Fit window: %.0f–%.0f Å (%d pixels)", lo_A, hi_A, spec.n_pix,
        )

    # Determine which lines to fit.
    if lines is None:
        if grating is not None:
            candidate_lines = get_line_list(grating)
        else:
            # Infer line list from estimated resolving power.
            # Prism: R ~ 30–300; gratings/stacks: R ≥ 500.
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
    R_arr = resolve_R(spec.wave_um, grating=grating, R=R)
    R_med = float(np.median(R_arr))
    R_lo, R_hi = float(np.min(R_arr)), float(np.max(R_arr))
    if not _label:
        if abs(R_hi - R_lo) < 10:
            print(f"Resolving power: R = {R_med:.0f}")
        else:
            print(f"Resolving power: R ≈ {R_med:.0f} (range {R_lo:.0f}–{R_hi:.0f})")

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
        moving_average=moving_average,
    )
    flux_sub = spec.flux_ujy - continuum

    # Convert to F_λ for fitting.
    from .io import _ujy_to_flam, _flam_to_ujy

    flam = _ujy_to_flam(flux_sub, spec.wave_um)
    flam_err = _ujy_to_flam(spec.err_ujy, spec.wave_um)

    valid = np.isfinite(flam) & np.isfinite(flam_err) & (flam_err > 0)

    # Exclude pixels blueward of NV (IGM-absorbed / Lyman-break region).
    nv_obs_A = REST_LINES_A["NV_1"] * (1.0 + z)
    valid &= spec.wave_A >= nv_obs_A

    # Drop lines that fall in detector gaps (< 50% valid pixels within ±5σ).
    _kept: list[str] = []
    for name in line_names:
        lam_obs_A = REST_LINES_A[name] * (1.0 + z)
        _, sig_seed, _ = _grating_bounds(grating, sig_inst, spec.dlam_A, lam_obs_A, sigma_factor)
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
        logger.info(
            "Kept %d / %d lines after gap filtering.", len(_kept), len(line_names),
        )
        line_names = _kept

    if len(line_names) == 0:
        logger.warning("All lines fall in detector gaps.")
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

    # Pixel info.
    edges = spec.wave_edges_A
    dlam = spec.dlam_A
    left = edges[:-1]
    right = edges[1:]

    # Pixel weights.
    w_pix = pixel_weight(dlam)

    # Detect unresolved intercombination doublets: if the doublet
    # separation is < 2×FWHM (≈ 4.71 × sigma_inst), the components
    # cannot be reliably decomposed.  Fix the amplitude ratio to the
    # low-density limit for fitting stability.
    from .constraints import _INTERCOM_LOW_DENSITY_RATIOS

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
                    print(
                        f"  Blended: {pri}–{sec} "
                        f"(sep={sep:.1f} Å < {_BLEND_THRESHOLD:.1f}×σ_inst="
                        f"{_BLEND_THRESHOLD * sig_at_line:.1f} Å) "
                        f"→ fixing ratio"
                    )

    # Setup constraints.
    constraints = ConstraintSet(
        line_names,
        tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids,
        tie_uv_widths=tie_uv_widths,
        blended_doublets=_blended if _blended else None,
    )

    # Initial parameters: [amplitudes, centroids, sigmas].
    nL = len(line_names)
    p0 = np.zeros(3 * nL)
    lb = np.zeros(3 * nL)
    ub = np.zeros(3 * nL)

    for i, name in enumerate(line_names):
        lam_obs_A = REST_LINES_A[name] * (1.0 + z)
        sig_lo, sig_seed, sig_hi = _grating_bounds(grating, sig_inst, dlam, lam_obs_A, sigma_factor)

        # Find peak/trough flux near the line for amplitude seeding.
        near = np.abs(spec.wave_A - lam_obs_A)
        idx_near = np.where(near < 5 * sig_seed)[0]
        is_abs = name.startswith("abs_")

        if is_abs:
            # Absorption line: seed from the deepest trough.
            if len(idx_near) > 0:
                peak_flam = abs(np.nanmin(flam[idx_near]))
            else:
                peak_flam = 1.0
        else:
            # Emission line: seed from the peak.
            if len(idx_near) > 0:
                peak_flam = np.nanmax(flam[idx_near])
            else:
                peak_flam = np.nanmax(flam[valid]) if np.any(valid) else 1.0
        if not np.isfinite(peak_flam) or peak_flam <= 0:
            peak_flam = 1.0

        # Amplitude seed = peak × sqrt(2π) × σ (area under Gaussian).
        A_seed = max(peak_flam * _SQRT2PI * sig_seed, 1e-30)

        # Amplitude bounds.
        if is_abs:
            # Absorption: amplitude must be ≤ 0 (negative Gaussian).
            p0[i] = -A_seed
            lb[i] = -150.0 * max(peak_flam, 1e-30) * _SQRT2PI * sig_hi
            ub[i] = 0.0
        else:
            p0[i] = A_seed
            lb[i] = 0.0
            ub[i] = 150.0 * max(peak_flam, 1e-30) * _SQRT2PI * sig_hi

        # Centroid bounds — expressed as a velocity offset (km/s) converted
        # to Å at the observed wavelength.  For stacked spectra the redshift
        # alignment can introduce ~100–300 km/s systematic offsets; individual
        # spectra typically need less.  The margin is capped at half the
        # separation to the nearest line so centroids cannot blend.
        local_sig = sig_inst[np.argmin(np.abs(spec.wave_A - lam_obs_A))]
        _C_KMS_CENT = 299792.458
        _CENT_V_MAX = centroid_vmax  # km/s — max centroid offset
        cent_margin_v = _CENT_V_MAX / _C_KMS_CENT * lam_obs_A
        # Floor: at least 2 pixel widths for coarsely sampled spectra.
        cent_margin = max(cent_margin_v, 2.0 * np.median(dlam))

        # Cap at half the distance to the nearest neighbour to prevent
        # lines from drifting into each other.
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

        # Sigma bounds — broad components use physically motivated velocity
        # widths (km/s) converted to σ_λ at the observed wavelength.
        _C_KMS = 299792.458  # speed of light (km/s)
        if "_BROAD2" in name:
            # Very broad (BLR): FWHM ~ 2000–5000 km/s.
            from .broad import (
                BROAD2_SIGMA_V_LO, BROAD2_SIGMA_V_SEED, BROAD2_SIGMA_V_HI,
            )
            sig_lo = BROAD2_SIGMA_V_LO / _C_KMS * lam_obs_A
            sig_seed = BROAD2_SIGMA_V_SEED / _C_KMS * lam_obs_A
            sig_hi = BROAD2_SIGMA_V_HI / _C_KMS * lam_obs_A
        elif "_BROAD" in name:
            # Intermediate broad (outflows / NLR): FWHM ~ 500–2000 km/s.
            from .broad import (
                BROAD1_SIGMA_V_LO, BROAD1_SIGMA_V_SEED, BROAD1_SIGMA_V_HI,
            )
            sig_lo = BROAD1_SIGMA_V_LO / _C_KMS * lam_obs_A
            sig_seed = BROAD1_SIGMA_V_SEED / _C_KMS * lam_obs_A
            sig_hi = BROAD1_SIGMA_V_HI / _C_KMS * lam_obs_A

        # Hard cap: no component wider than 500 Å (prevents fitting noise).
        _MAX_SIGMA_A = 500.0
        sig_hi = min(sig_hi, _MAX_SIGMA_A)
        sig_seed = min(sig_seed, 0.9 * sig_hi)

        p0[2 * nL + i] = sig_seed
        lb[2 * nL + i] = sig_lo
        ub[2 * nL + i] = sig_hi

    # Seed UV doublet secondaries at the expected flux ratio of the
    # primary.  Without this, both members pick up the same peak flux
    # as their seed, and the optimiser often dumps all flux into one
    # member.  The amplitude remains free — this just gives a
    # physically motivated starting point.
    from .constraints import _INTERCOM_LOW_DENSITY_RATIOS, NV_RATIO

    idx = {name: i for i, name in enumerate(line_names)}
    _ALL_DOUBLETS = [
        *[(*pair, ratio) for pair, ratio in _INTERCOM_LOW_DENSITY_RATIOS.items()],
        ("NV_1", "NV_2", NV_RATIO),
    ]
    for pri, sec, ratio in _ALL_DOUBLETS:
        if pri in idx and sec in idx:
            i_pri = idx[pri]
            i_sec = idx[sec]
            p0[i_sec] = np.clip(p0[i_pri] * ratio, lb[i_sec], ub[i_sec])
            # Cap the secondary's upper bound: physically the secondary
            # can never exceed ~2× the primary (even at extreme densities
            # the ratio stays < 1.5).  Without this cap, the optimizer
            # can push the secondary to 150× peak and fit noise.
            # Use 3× the primary seed as a generous ceiling.
            ub_cap = max(3.0 * p0[i_pri], p0[i_sec] * 5.0)
            ub[i_sec] = min(ub[i_sec], ub_cap)

    # Override seeds from a previous fit (e.g. narrow-only results).
    # For non-broad lines, also set an amplitude floor at 30% of the
    # narrow-only value to prevent the optimizer from zeroing out narrow
    # lines when a broad component can absorb their flux.
    if _p0_hint is not None:
        for hint_name, (hint_A, hint_mu, hint_sig) in _p0_hint.items():
            if hint_name in idx:
                j = idx[hint_name]
                p0[j] = np.clip(hint_A, lb[j], ub[j])
                p0[nL + j] = np.clip(hint_mu, lb[nL + j], ub[nL + j])
                p0[2 * nL + j] = np.clip(hint_sig, lb[2 * nL + j], ub[2 * nL + j])
                # Amplitude floor for narrow lines (not broad components).
                if "BROAD" not in hint_name and hint_A > 0:
                    lb[j] = max(lb[j], 0.3 * hint_A)

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
        x_scale="jac",
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
        lb_free, ub_free, p0_free, w_pix, n_boot, label=_label,
        n_jobs=n_jobs,
    )

    # Build per-line results.
    line_results = {}
    cont_flam = _ujy_to_flam(continuum, spec.wave_um)

    # Pre-compute the continuum-subtracted residual in f_lam space for
    # local continuum RMS estimates.
    resid_flam = flam - build_model(p_best, edges, nL)
    # Also need the f_lam error array for peak SNR from the uncertainty array.
    # (flam_err is already available from earlier.)

    for i, name in enumerate(line_names):
        A = p_best[i]
        mu = p_best[nL + i]
        sig = p_best[2 * nL + i]
        flux_line = A  # Area-normalised Gaussian: integral = amplitude.
        f_err = flux_errs[i] if flux_errs is not None else _analytic_flux_err(
            A, sig, flam_err, spec.wave_A, mu, valid
        )

        # --- SNR 1: integrated, bootstrap error ---
        snr_int_err = flux_line / f_err if f_err > 0 else 0.0

        # --- Peak of the Gaussian model (f_lam units) ---
        model_peak = A / (sig * _SQRT2PI) if sig > 0 else 0.0

        # --- SNR 3: peak, uncertainty array ---
        idx_peak = np.argmin(np.abs(spec.wave_A - mu))
        err_at_peak = flam_err[idx_peak] if valid[idx_peak] else 0.0
        snr_peak_err = model_peak / err_at_peak if err_at_peak > 0 else 0.0

        # --- Local continuum RMS (±15σ window, excluding ±3σ) ---
        outer_mask = np.abs(spec.wave_A - mu) < 15.0 * sig
        inner_mask = np.abs(spec.wave_A - mu) < 3.0 * sig
        cont_region = outer_mask & ~inner_mask & valid
        if np.sum(cont_region) >= 3:
            cont_rms = float(np.std(resid_flam[cont_region]))
        else:
            # Fallback: use the uncertainty array median near the line.
            fallback = outer_mask & valid
            cont_rms = float(np.median(flam_err[fallback])) if np.any(fallback) else 0.0

        # --- SNR 4: peak, continuum RMS ---
        snr_peak_cont = model_peak / cont_rms if cont_rms > 0 else 0.0

        # --- SNR 2: integrated, continuum RMS ---
        # Noise on integrated flux = cont_rms × √(N_pix in ±3σ window).
        n_pix_line = int(np.sum(inner_mask & valid))
        int_noise_cont = cont_rms * sqrt(n_pix_line) if n_pix_line > 0 else 0.0
        snr_int_cont = flux_line / int_noise_cont if int_noise_cont > 0 else 0.0

        # Equivalent width (rest-frame).
        # Use continuum at the line centroid.  If the polynomial continuum
        # is unreliable (near-zero or negative), fall back to the local
        # median flux in a ±5σ window around the line.  If that is also
        # non-positive, report EW as NaN.
        lam_rest_A = REST_LINES_A[name]
        idx_cont = np.argmin(np.abs(spec.wave_A - mu))
        cont_at_line = cont_flam[idx_cont]
        if cont_at_line <= 0:
            near_mask = np.abs(spec.wave_A - mu) < 5.0 * sig
            near_valid = near_mask & valid
            if np.any(near_valid):
                local_median_ujy = np.nanmedian(spec.flux_ujy[near_valid])
                cont_at_line = _ujy_to_flam(
                    np.array([max(local_median_ujy, 0.0)]),
                    np.array([spec.wave_um[idx_cont]]),
                )[0]
        ew_rest = flux_line / cont_at_line / (1.0 + z) if cont_at_line > 0 else np.nan

        line_results[name] = LineResult(
            name=name,
            rest_wave_A=lam_rest_A,
            amplitude=A,
            centroid_A=mu,
            sigma_A=sig,
            flux=flux_line,
            flux_err=f_err,
            ew_A=ew_rest,
            snr_int_err=snr_int_err,
            snr_int_cont=snr_int_cont,
            snr_peak_err=snr_peak_err,
            snr_peak_cont=snr_peak_cont,
        )

    fit_result = FitResult(
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

    if save_path is not None:
        from .io import export_lines_txt
        export_lines_txt(fit_result, save_path, z=z)

    return fit_result


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


def _run_single_bootstrap(
    noise_vec: np.ndarray,
    flam: np.ndarray,
    flam_err: np.ndarray,
    valid: np.ndarray,
    edges: np.ndarray,
    nL: int,
    constraints: ConstraintSet,
    lb_free: np.ndarray,
    ub_free: np.ndarray,
    p0_free: np.ndarray,
    w_pix: np.ndarray,
) -> np.ndarray:
    """Run a single bootstrap iteration.

    Parameters
    ----------
    noise_vec : np.ndarray
        Pre-generated standard-normal noise vector (length ``n_pix``).

    Returns
    -------
    np.ndarray
        Flux (amplitude) for each line, or NaN if the fit fails.
    """
    flam_b = flam + noise_vec * flam_err

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
            x_scale="jac",
        )
        p_full_b = constraints.expand_free_to_full(res_b.x)
        return p_full_b[:nL]
    except Exception:
        return np.full(nL, np.nan)


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
    label: str = "",
    n_jobs: int = -1,
) -> np.ndarray | None:
    """Run bootstrap resampling for flux uncertainties.

    Parameters
    ----------
    n_jobs : int
        Number of parallel jobs for bootstrap. ``-1`` uses all cores,
        ``1`` runs sequentially. Default ``-1``.

    Returns
    -------
    np.ndarray or None
        Standard deviation of flux for each line, or None if n_boot == 0.
    """
    if n_boot <= 0:
        return None

    # Pre-generate all noise vectors for reproducibility.
    rng = np.random.default_rng(42)
    noise_all = rng.standard_normal((n_boot, len(flam)))

    # Shared keyword arguments for _run_single_bootstrap.
    shared_kwargs = dict(
        flam=flam, flam_err=flam_err, valid=valid, edges=edges,
        nL=nL, constraints=constraints, lb_free=lb_free, ub_free=ub_free,
        p0_free=p0_free, w_pix=w_pix,
    )

    desc = f"Bootstrap ({label})" if label else "Bootstrap"

    try:
        from joblib import Parallel, delayed
        from tqdm import tqdm

        logger.info("%s: %d iterations, n_jobs=%d", desc, n_boot, n_jobs)

        # return_as="generator" yields results as they complete,
        # letting tqdm update incrementally.
        gen = Parallel(n_jobs=n_jobs, return_as="generator", prefer="processes")(
            delayed(_run_single_bootstrap)(noise_all[b], **shared_kwargs)
            for b in range(n_boot)
        )
        results = list(tqdm(gen, total=n_boot, desc=desc, unit="iter"))
        flux_samples = np.array(results)

    except ImportError:
        logger.warning("joblib not installed; falling back to sequential bootstrap.")
        from tqdm import tqdm

        flux_samples = np.zeros((n_boot, nL))

        for b in tqdm(range(n_boot), desc=desc, unit="iter"):
            flux_samples[b, :] = _run_single_bootstrap(
                noise_all[b], **shared_kwargs
            )

    # Use sigma-clipped std to reject pathological bootstrap iterations
    # where unconstrained parameters (e.g. UV secondaries) blow up.
    from astropy.stats import sigma_clipped_stats

    errs = np.zeros(flux_samples.shape[1])
    for i in range(flux_samples.shape[1]):
        col = flux_samples[:, i]
        finite = col[np.isfinite(col)]
        if len(finite) < 3:
            errs[i] = np.nanstd(col)
        else:
            _, _, std = sigma_clipped_stats(finite, sigma=3.0)
            # Fall back to nanstd if clipping removes all variance.
            errs[i] = std if std > 0 else np.nanstd(finite)
    return errs
