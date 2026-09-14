"""Continuum fitting via iterative σ-clipped polynomial or median filter."""

from __future__ import annotations

import logging

import numpy as np

from .lines import REST_LINES_A
from .resolution import sigma_inst_A

logger = logging.getLogger(__name__)

_DEFAULT_MOVING_AVERAGE_WINDOW = 75
_MA_CLIP_NSIGMA = 3.0  # σ_inst radius around each line where clipping is applied


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
    lya_break: bool = False,
    return_err: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
    return_err : bool
        If ``True``, also return the 1σ uncertainty on the continuum.
        Needed for an equivalent-width error, where the continuum term
        is often larger than the line-flux term.  Default ``False``, so
        existing callers are unaffected.

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        Continuum evaluated at each pixel (µJy).  With *return_err*, the
        pair ``(continuum, continuum_err)``.

    Notes
    -----
    The uncertainty comes from the polynomial coefficient covariance,
    propagated as ``sigma(lambda)^2 = v C v.T`` with *v* the Vandermonde
    row.  It uses numpy's **scaled** covariance, which multiplies by the
    reduced chi-square: a low-order polynomial is an approximation to a
    real continuum rather than the truth, and the scaling lets a poor fit
    widen the error instead of hiding behind the formal one.

    The median-filter path (*moving_average*) has no parametric
    covariance, so its uncertainty is returned as NaN -- "unavailable",
    not zero.  The same applies when there are too few unmasked pixels to
    estimate a covariance at all (numpy needs more than ``deg + 2``).
    """.format(default_win=_DEFAULT_MOVING_AVERAGE_WINDOW)
    wave_A = wave_um * 1e4
    valid = np.isfinite(flux_ujy) & np.isfinite(err_ujy) & (err_ujy > 0)

    # Mask everything blueward of Lyman-alpha (1215.67 Å rest-frame).
    # The IGM absorbs flux shortward of Ly-α, making the continuum
    # unreliable there.
    _LYA_REST_A = 1215.670
    lya_obs_A = _LYA_REST_A * (1.0 + z)
    if not lya_break:
        valid &= wave_A >= lya_obs_A

    # Resolve moving_average window size.
    if moving_average is True:
        _ma_window = _DEFAULT_MOVING_AVERAGE_WINDOW
    elif moving_average:
        _ma_window = int(moving_average)
    else:
        _ma_window = 0

    # ---- Moving-average (median filter + boxcar) path ----
    if _ma_window > 0:
        from scipy.ndimage import median_filter, uniform_filter1d

        sig_inst = sigma_inst_A(wave_um, grating=grating, R=R)

        # Build a narrow "near-line" mask: pixels within ±_MA_CLIP_NSIGMA
        # of any known line.  Sigma-clipping is restricted to these pixels;
        # pixels far from lines are never removed.
        near_line = np.zeros(len(wave_A), dtype=bool)
        for name in line_names:
            if name not in REST_LINES_A:
                continue
            lam_obs_A = REST_LINES_A[name] * (1.0 + z)
            idx_near = np.argmin(np.abs(wave_A - lam_obs_A))
            clip_half = max(_MA_CLIP_NSIGMA * sig_inst[idx_near], 20.0)
            near_line |= np.abs(wave_A - lam_obs_A) < clip_half

        n_valid = int(np.sum(valid))
        if n_valid < 3:
            logger.warning(
                "Too few valid pixels (%d) for median filter; returning zeros.",
                n_valid,
            )
            zeros = np.zeros_like(flux_ujy)
            return (zeros, np.full_like(flux_ujy, np.nan)) if return_err else zeros

        # Ensure window is odd and does not exceed the number of valid pixels.
        win = min(_ma_window, n_valid)
        if win % 2 == 0:
            win += 1

        mask = valid.copy()

        for _ in range(n_iter):
            idx_mask = np.where(mask)[0]
            if len(idx_mask) < 3:
                break
            smoothed = median_filter(
                flux_ujy[idx_mask], size=min(win, len(idx_mask)),
            )
            # Boxcar pass to smooth out the median filter's staircase.
            smoothed = uniform_filter1d(smoothed, size=min(win, len(idx_mask)))
            cont = np.interp(wave_um, wave_um[idx_mask], smoothed)
            resid = flux_ujy - cont
            # Only clip positive outliers near known line positions.
            norm_resid = np.where(err_ujy > 0, resid / err_ujy, 0.0)
            clip_out = near_line & (norm_resid >= clip_sigma)
            mask = valid & ~clip_out

        # Final smoothing on surviving pixels.
        idx_mask = np.where(mask)[0]
        if len(idx_mask) >= 3:
            smoothed = median_filter(
                flux_ujy[idx_mask], size=min(win, len(idx_mask)),
            )
            smoothed = uniform_filter1d(smoothed, size=min(win, len(idx_mask)))
            continuum = np.interp(wave_um, wave_um[idx_mask], smoothed)
        else:
            continuum = cont

        if lya_break:
            continuum = _apply_lya_break(wave_A, continuum, lya_obs_A)
        if return_err:
            # A median filter has no parametric covariance, so there is no
            # honest uncertainty to hand back.  NaN says "unavailable"
            # rather than implying zero error.
            return continuum, np.full_like(continuum, np.nan)
        return continuum

    # ---- Polynomial path (default) ----
    # Build line mask.
    sig_inst = sigma_inst_A(wave_um, grating=grating, R=R)
    line_mask = np.zeros(len(wave_A), dtype=bool)

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
        zeros = np.zeros_like(flux_ujy)
        return (zeros, np.full_like(flux_ujy, np.nan)) if return_err else zeros

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

    # Final fit on clipped pixels.  ``cov=True`` also returns the
    # coefficient covariance, which is the only route to a continuum
    # uncertainty -- and hence to an equivalent-width uncertainty that
    # is not just the line-flux term.  numpy needs more than deg + 2
    # points to estimate it.
    cov = None
    if np.sum(mask) >= deg + 1:
        if return_err and np.sum(mask) > deg + 2:
            coeffs, cov = np.polyfit(
                w_norm[mask], flux_ujy[mask], deg, w=weights[mask], cov=True,
            )
        else:
            coeffs = np.polyfit(w_norm[mask], flux_ujy[mask], deg, w=weights[mask])

    continuum = np.polyval(coeffs, w_norm)
    if lya_break:
        continuum = _apply_lya_break(wave_A, continuum, lya_obs_A)
    if not return_err:
        return continuum

    if cov is None:
        logger.warning(
            "Continuum covariance needs more than deg + 2 = %d points, have "
            "%d; continuum uncertainty unavailable.", deg + 2, int(np.sum(mask)),
        )
        return continuum, np.full_like(continuum, np.nan)

    # sigma(lambda)^2 = v(lambda) C v(lambda)^T, with v the Vandermonde row
    # in the same descending-power order np.polyval expects.
    vander = np.vander(w_norm, deg + 1)
    var = np.einsum("ij,jk,ik->i", vander, cov, vander)
    cont_err = np.sqrt(np.clip(var, 0.0, None))
    if lya_break:
        # The continuum is forced to zero blueward of the break, so its
        # uncertainty goes with it.
        cont_err = np.where(wave_A < lya_obs_A, 0.0, cont_err)
    return continuum, cont_err


def _apply_lya_break(
    wave_A: np.ndarray,
    continuum: np.ndarray,
    lya_obs_A: float,
) -> np.ndarray:
    """Set continuum blueward of Lyα to zero.

    At z > 5, the mean IGM transmission blueward of Lyα is < 5%,
    so the intrinsic continuum is effectively extinguished.  For
    de-redshifted stacks of high-z galaxies, the blue-side flux is
    consistent with zero.

    Parameters
    ----------
    wave_A : np.ndarray
        Observed wavelength in Angstroms.
    continuum : np.ndarray
        Continuum array to modify.
    lya_obs_A : float
        Observed Lyα wavelength (Å).

    Returns
    -------
    np.ndarray
        Continuum with blue side set to zero.
    """
    out = continuum.copy()
    out[wave_A < lya_obs_A] = 0.0

    # Linear ramp from zero at Lyα to the moving-average value at
    # Lyα + ramp_width.  The moving average overestimates the continuum
    # near the break because it smooths over the Lyα emission; the ramp
    # corrects this by forcing the continuum to rise linearly from the
    # break rather than following the biased moving average.
    _ramp_width_A = 5.0  # Å — ramp from 1216 to 1221
    ramp_hi = lya_obs_A + _ramp_width_A
    ramp_mask = (wave_A >= lya_obs_A) & (wave_A < ramp_hi)
    if np.any(ramp_mask):
        # Value the continuum should reach at the end of the ramp.
        idx_ramp_end = np.argmin(np.abs(wave_A - ramp_hi))
        cont_at_ramp_end = continuum[idx_ramp_end]
        # Linear interpolation from 0 at lya_obs to cont_at_ramp_end.
        frac = (wave_A[ramp_mask] - lya_obs_A) / _ramp_width_A
        out[ramp_mask] = cont_at_ramp_end * frac

    return out
