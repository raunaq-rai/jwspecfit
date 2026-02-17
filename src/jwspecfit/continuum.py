"""Continuum fitting via iterative σ-clipped polynomial."""

from __future__ import annotations

import logging

import numpy as np

from .lines import REST_LINES_A
from .resolution import sigma_inst_A

logger = logging.getLogger(__name__)


def fit_continuum(
    wave_um: np.ndarray,
    flux_ujy: np.ndarray,
    err_ujy: np.ndarray,
    z: float,
    line_names: list[str],
    *,
    grating: str | None = None,
    R: float | None = None,
    deg: int = 2,
    clip_sigma: float = 2.5,
    n_iter: int = 5,
    line_mask_nsigma: float = 6.0,
) -> np.ndarray:
    """Fit a polynomial continuum with emission-line masking.

    The algorithm:
    1. Mask pixels within ±``line_mask_nsigma`` × σ_inst of every known line.
    2. Fit a polynomial of degree *deg* to the unmasked, valid pixels.
    3. Iteratively σ-clip residuals above *clip_sigma* and refit.

    Parameters
    ----------
    wave_um : np.ndarray
        Observed wavelength in microns.
    flux_ujy : np.ndarray
        Flux density in µJy.
    err_ujy : np.ndarray
        Uncertainty in µJy.
    z : float
        Source redshift.
    line_names : list of str
        Emission lines to mask (keys of ``REST_LINES_A``).
    grating : str, optional
        Grating name for resolution model.
    R : float, optional
        Resolving power (overrides *grating*).
    deg : int
        Polynomial degree (default 2).
    clip_sigma : float
        Sigma-clipping threshold (default 2.5).
    n_iter : int
        Number of clipping iterations (default 5).
    line_mask_nsigma : float
        Number of instrumental σ to mask around each line (default 6).

    Returns
    -------
    np.ndarray
        Continuum evaluated at each pixel (µJy).
    """
    wave_A = wave_um * 1e4
    valid = np.isfinite(flux_ujy) & np.isfinite(err_ujy) & (err_ujy > 0)

    # Build line mask.
    line_mask = np.zeros(len(wave_A), dtype=bool)
    sig_inst = sigma_inst_A(wave_um, grating=grating, R=R)

    for name in line_names:
        if name not in REST_LINES_A:
            continue
        lam_obs_A = REST_LINES_A[name] * (1.0 + z)
        # Mask width = max(line_mask_nsigma * sigma_inst, a minimum of 20 Å)
        idx_near = np.argmin(np.abs(wave_A - lam_obs_A))
        mask_half = max(line_mask_nsigma * sig_inst[idx_near], 20.0)
        line_mask |= np.abs(wave_A - lam_obs_A) < mask_half

    use = valid & ~line_mask

    if np.sum(use) < deg + 1:
        logger.warning("Too few continuum pixels (%d); returning zeros.", np.sum(use))
        return np.zeros_like(flux_ujy)

    # Normalise wavelength for numerical stability.
    w_norm = (wave_um - wave_um[use].mean()) / wave_um[use].std()

    weights = np.where(use, 1.0 / err_ujy, 0.0)
    mask = use.copy()

    for _ in range(n_iter):
        if np.sum(mask) < deg + 1:
            break
        coeffs = np.polyfit(w_norm[mask], flux_ujy[mask], deg, w=weights[mask])
        cont = np.polyval(coeffs, w_norm)
        resid = flux_ujy - cont
        rms = np.sqrt(np.nanmedian(resid[mask] ** 2))
        if rms <= 0:
            break
        # Clip only positive outliers (emission above continuum).
        mask = use & (resid < clip_sigma * rms)

    # Final fit on clipped pixels.
    if np.sum(mask) >= deg + 1:
        coeffs = np.polyfit(w_norm[mask], flux_ujy[mask], deg, w=weights[mask])

    continuum = np.polyval(coeffs, w_norm)
    return continuum
