"""Continuum fitting via iterative σ-clipped polynomial or median filter."""

from __future__ import annotations

import logging

import numpy as np

from .lines import REST_LINES_A
from .resolution import sigma_inst_A

logger = logging.getLogger(__name__)

_DEFAULT_MOVING_AVERAGE_WINDOW = 75
_MA_LINE_MASK_NSIGMA = 3.0


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
    moving_average: bool | int = False,
) -> np.ndarray:
    """Fit a continuum with emission-line masking.

    Two modes are available:

    - **Polynomial** (default): fit a polynomial of degree *deg* to unmasked
      pixels with iterative σ-clipping.
    - **Median filter** (``moving_average``): apply a
      ``scipy.ndimage.median_filter`` to unmasked pixels and interpolate to
      the full wavelength grid.  Useful for stacked spectra where the
      continuum shape varies across the wavelength range.

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
        Polynomial degree (default 2).  Ignored when *moving_average* is
        active.
    clip_sigma : float
        Sigma-clipping threshold (default 2.5).
    n_iter : int
        Number of clipping iterations (default 5).
    line_mask_nsigma : float
        Number of instrumental σ to mask around each line (default 6).
    moving_average : bool or int
        If ``False`` (default), use the polynomial continuum.  If ``True``,
        use a median filter with a default window of
        {default_win} pixels.  If an ``int``, use that as the window size.

    Returns
    -------
    np.ndarray
        Continuum evaluated at each pixel (µJy).
    """.format(default_win=_DEFAULT_MOVING_AVERAGE_WINDOW)
    wave_A = wave_um * 1e4
    valid = np.isfinite(flux_ujy) & np.isfinite(err_ujy) & (err_ujy > 0)

    # Mask everything blueward of Lyman-alpha (1215.67 Å rest-frame).
    # The IGM absorbs flux shortward of Ly-α, making the continuum
    # unreliable there.
    _LYA_REST_A = 1215.670
    lya_obs_A = _LYA_REST_A * (1.0 + z)
    valid &= wave_A >= lya_obs_A

    # Resolve moving_average window size.
    if moving_average is True:
        _ma_window = _DEFAULT_MOVING_AVERAGE_WINDOW
    elif moving_average:
        _ma_window = int(moving_average)
    else:
        _ma_window = 0

    # Use a narrower line mask for the median-filter path (3σ vs 6σ):
    # the median filter is inherently robust to localised outliers, so
    # a tighter mask preserves more continuum pixels.
    _effective_nsigma = _MA_LINE_MASK_NSIGMA if _ma_window > 0 else line_mask_nsigma

    # Build line mask.
    line_mask = np.zeros(len(wave_A), dtype=bool)
    sig_inst = sigma_inst_A(wave_um, grating=grating, R=R)

    for name in line_names:
        if name not in REST_LINES_A:
            continue
        lam_obs_A = REST_LINES_A[name] * (1.0 + z)
        # Mask width = max(nsigma * sigma_inst, a minimum of 20 Å)
        idx_near = np.argmin(np.abs(wave_A - lam_obs_A))
        mask_half = max(_effective_nsigma * sig_inst[idx_near], 20.0)
        line_mask |= np.abs(wave_A - lam_obs_A) < mask_half

    use = valid & ~line_mask

    # ---- Moving-average (median filter) path ----
    if _ma_window > 0:
        from scipy.ndimage import median_filter

        n_use = int(np.sum(use))
        if n_use < 3:
            logger.warning(
                "Too few continuum pixels (%d) for median filter; returning zeros.",
                n_use,
            )
            return np.zeros_like(flux_ujy)

        # Ensure window is odd and does not exceed the number of usable pixels.
        win = min(_ma_window, n_use)
        if win % 2 == 0:
            win += 1

        mask = use.copy()

        for _ in range(n_iter):
            idx_mask = np.where(mask)[0]
            if len(idx_mask) < 3:
                break
            smoothed_masked = median_filter(flux_ujy[idx_mask], size=min(win, len(idx_mask)))
            # Interpolate smoothed values to all pixels.
            cont = np.interp(wave_um, wave_um[idx_mask], smoothed_masked)
            resid = flux_ujy - cont
            # Clip positive outliers (emission residuals) in error-normalised
            # space, same logic as the polynomial path.
            norm_resid = np.where(err_ujy > 0, resid / err_ujy, 0.0)
            mask = use & (norm_resid < clip_sigma)

        # Final smoothing on clipped pixels.
        idx_mask = np.where(mask)[0]
        if len(idx_mask) >= 3:
            smoothed_masked = median_filter(flux_ujy[idx_mask], size=min(win, len(idx_mask)))
            continuum = np.interp(wave_um, wave_um[idx_mask], smoothed_masked)
        else:
            continuum = cont

        return continuum

    # ---- Polynomial path (default) ----
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
        # Clip in error-normalised space so that noisy pixels (large err)
        # are not preferentially removed.  Only clip positive outliers
        # (emission above continuum).
        norm_resid = np.where(err_ujy > 0, resid / err_ujy, 0.0)
        mask = use & (norm_resid < clip_sigma)

    # Final fit on clipped pixels.
    if np.sum(mask) >= deg + 1:
        coeffs = np.polyfit(w_norm[mask], flux_ujy[mask], deg, w=weights[mask])

    continuum = np.polyval(coeffs, w_norm)
    return continuum
