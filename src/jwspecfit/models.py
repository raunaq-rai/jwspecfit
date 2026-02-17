"""Gaussian emission-line profiles with bin-averaged evaluation.

The core building block is :func:`gaussian_binned`, which computes the
mean value of a unit-area Gaussian over each pixel bin using the error
function.  This avoids the sampling bias that arises when the line is
narrower than the pixel width (as happens with the NIRSpec prism).
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.special import erf


def gaussian_binned(
    lam_left_A: np.ndarray,
    lam_right_A: np.ndarray,
    mu_A: float,
    sigma_A: float,
) -> np.ndarray:
    """Bin-averaged, area-normalised Gaussian profile.

    Returns the mean value (per Angstrom) of a Gaussian with unit area
    integrated over each ``[left, right]`` pixel bin.

    Parameters
    ----------
    lam_left_A : np.ndarray
        Left edges of wavelength bins (Å).
    lam_right_A : np.ndarray
        Right edges of wavelength bins (Å).
    mu_A : float
        Gaussian centroid (Å).
    sigma_A : float
        Gaussian standard deviation (Å).

    Returns
    -------
    np.ndarray
        Profile value in each bin (Å⁻¹).  Multiply by amplitude (in
        flux × Å units) to get flux density.
    """
    if not (np.isfinite(mu_A) and np.isfinite(sigma_A)) or sigma_A <= 0:
        return np.zeros_like(lam_left_A)

    inv = 1.0 / (sqrt(2.0) * sigma_A)
    cdf_r = 0.5 * (1.0 + erf((lam_right_A - mu_A) * inv))
    cdf_l = 0.5 * (1.0 + erf((lam_left_A - mu_A) * inv))
    area = cdf_r - cdf_l
    width = lam_right_A - lam_left_A
    width = np.where(width > 0, width, np.nan)
    mean = area / width
    mean[~np.isfinite(mean)] = 0.0
    return mean


def pixel_weight(dlam_A: np.ndarray, power: float = 0.35) -> np.ndarray:
    """Pixel-width weighting to down-weight overly narrow/wide pixels.

    Weight = (median_dλ / dλ)^power.  Pixels with atypical widths
    (edges of the detector, bad pixels) get reduced influence.

    Parameters
    ----------
    dlam_A : np.ndarray
        Pixel widths in Angstroms.
    power : float
        Weighting exponent (default 0.35).

    Returns
    -------
    np.ndarray
        Multiplicative weight for each pixel.
    """
    med = np.nanmedian(dlam_A)
    w = np.where(dlam_A > 0, (med / dlam_A) ** power, 0.0)
    return w


def build_model(
    params: np.ndarray,
    wave_edges_A: np.ndarray,
    n_lines: int,
) -> np.ndarray:
    """Build multi-line emission model from a flat parameter vector.

    Parameters are packed as::

        [A_0, A_1, ..., A_{n-1},   # amplitudes (flux × Å)
         mu_0, mu_1, ...,           # centroids (Å)
         sigma_0, sigma_1, ...]     # widths (Å)

    Uses vectorised numpy broadcasting to evaluate all lines simultaneously
    rather than looping in Python.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector of length ``3 * n_lines``.
    wave_edges_A : np.ndarray
        Pixel-edge wavelengths (length ``n_pix + 1``).
    n_lines : int
        Number of emission lines.

    Returns
    -------
    np.ndarray
        Model flux density per pixel (Å⁻¹ × amplitude units).
    """
    nL = n_lines
    left = wave_edges_A[:-1]
    right = wave_edges_A[1:]

    if nL == 0:
        return np.zeros(len(left))

    amplitudes = params[:nL]                    # (nL,)
    mus = params[nL : 2 * nL]                   # (nL,)
    sigmas = params[2 * nL : 3 * nL]            # (nL,)

    # Mask out invalid lines (non-finite or non-positive sigma).
    valid = np.isfinite(mus) & np.isfinite(sigmas) & (sigmas > 0)
    if not np.any(valid):
        return np.zeros(len(left))

    # Vectorised computation: shapes (n_pix, 1) op (nL,) -> (n_pix, nL)
    inv = 1.0 / (sqrt(2.0) * sigmas[valid])              # (nL_valid,)
    cdf_r = 0.5 * (1.0 + erf(
        (right[:, np.newaxis] - mus[np.newaxis, valid]) * inv  # (n_pix, nL_valid)
    ))
    cdf_l = 0.5 * (1.0 + erf(
        (left[:, np.newaxis] - mus[np.newaxis, valid]) * inv   # (n_pix, nL_valid)
    ))
    area = cdf_r - cdf_l                                  # (n_pix, nL_valid)

    width = right - left                                   # (n_pix,)
    width = np.where(width > 0, width, np.nan)
    profiles = area / width[:, np.newaxis]                 # (n_pix, nL_valid)
    profiles[~np.isfinite(profiles)] = 0.0

    # Matrix-vector multiply: profiles @ amplitudes -> (n_pix,)
    return profiles @ amplitudes[valid]
