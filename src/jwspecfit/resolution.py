"""Spectral resolution models for JWST NIRSpec gratings.

Provides resolving power R(λ) and instrumental line broadening σ(λ)
for prism, medium-resolution, and high-resolution gratings, as well
as user-supplied numeric or callable R values.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# FWHM → σ conversion factor.
_FWHM_TO_SIGMA = 1.0 / 2.3548200


def R_prism(lam_um: np.ndarray) -> np.ndarray:
    """Resolving power R(λ) for NIRSpec PRISM/CLEAR.

    Heuristic polynomial approximation calibrated against the
    NIRSpec instrument model.  Clipped to [30, 300].

    Parameters
    ----------
    lam_um : array_like
        Wavelength in microns.

    Returns
    -------
    np.ndarray
        Resolving power R at each wavelength.
    """
    lam_um = np.asarray(lam_um, dtype=float)
    x = lam_um - 1.0
    R = 50.0 + 50.0 * x + 15.0 * x**2
    return np.clip(R, 30.0, 300.0)


def R_grating(name: str) -> float:
    """Constant resolving power for a named grating.

    Parameters
    ----------
    name : str
        Grating name, e.g. ``"G395M"``, ``"G140H"``.

    Returns
    -------
    float
        Approximate resolving power.
    """
    g = name.upper()
    if g in ("G140M", "G235M", "G395M"):
        return 1000.0
    if g in ("G140H", "G235H", "G395H"):
        return 2700.0
    raise ValueError(f"Unknown grating: {name!r}")


def resolve_R(
    lam_um: np.ndarray,
    grating: str | None = None,
    R: float | Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Return R(λ) array from either a grating name or user-supplied R.

    Parameters
    ----------
    lam_um : array_like
        Wavelength in microns.
    grating : str, optional
        Grating name (``"PRISM"``, ``"G395M"``, etc.).
    R : float or callable, optional
        Numeric resolving power (constant) or callable ``R(lam_um)``.
        Takes precedence over *grating*.

    Returns
    -------
    np.ndarray
        Resolving power at each wavelength.

    Raises
    ------
    ValueError
        If neither *grating* nor *R* is specified.
    """
    lam_um = np.asarray(lam_um, dtype=float)

    if R is not None:
        if callable(R):
            return np.asarray(R(lam_um), dtype=float)
        return np.full_like(lam_um, float(R))

    if grating is not None:
        g = grating.upper()
        if "PRISM" in g:
            return R_prism(lam_um)
        return np.full_like(lam_um, R_grating(g))

    raise ValueError("Either grating or R must be specified.")


def sigma_inst_A(
    lam_um: np.ndarray,
    grating: str | None = None,
    R: float | Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Instrumental Gaussian σ in Angstroms.

    σ_inst = λ / (R × 2.3548)

    Parameters
    ----------
    lam_um : array_like
        Wavelength in microns.
    grating : str, optional
        Grating name.
    R : float or callable, optional
        Resolving power (overrides *grating*).

    Returns
    -------
    np.ndarray
        Instrumental σ in Angstroms at each wavelength.
    """
    lam_um = np.asarray(lam_um, dtype=float)
    R_arr = resolve_R(lam_um, grating=grating, R=R)
    lam_A = lam_um * 1e4
    return lam_A * _FWHM_TO_SIGMA / R_arr


def sigma_inst_kms(
    lam_um: np.ndarray,
    grating: str | None = None,
    R: float | Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Instrumental Gaussian σ in km/s.

    σ_v = c / (R × 2.3548)

    Parameters
    ----------
    lam_um : array_like
        Wavelength in microns.
    grating : str, optional
        Grating name.
    R : float or callable, optional
        Resolving power (overrides *grating*).

    Returns
    -------
    np.ndarray
        Instrumental σ in km/s at each wavelength.
    """
    lam_um = np.asarray(lam_um, dtype=float)
    R_arr = resolve_R(lam_um, grating=grating, R=R)
    c_kms = 299792.458
    return c_kms * _FWHM_TO_SIGMA / R_arr
