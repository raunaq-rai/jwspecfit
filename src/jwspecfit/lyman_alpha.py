"""Lyman-alpha fitting: skewed Gaussian profile × IGM absorption.

Uses Inoue et al. (2014) prescription for IGM transmission along the
line of sight, applied to a skewed Gaussian emission profile to capture
the characteristic asymmetry of Lyα.
"""

from __future__ import annotations

import logging
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.special import erf
from scipy.stats import skewnorm

logger = logging.getLogger(__name__)

# Path to Inoue+2014 Table 2 (shipped inside the package).
_DATA_DIR = Path(__file__).resolve().parent / "data"
_INOUE_TABLE = _DATA_DIR / "inoue2014_table2.txt"

# Lyman series rest wavelengths (Å) — first 40 transitions.
_LYA_REST = 1215.67  # Å

# Cache for loaded Inoue table.
_inoue_cache: dict | None = None


def _load_inoue_table() -> dict:
    """Load and cache the Inoue+2014 Table 2 coefficients.

    Returns
    -------
    dict
        Keys: ``j``, ``lam_j``, ``ALAF_j1``, ``ALAF_j2``, ``ALAF_j3``,
        ``ADLA_j1``, ``ADLA_j2``.
    """
    global _inoue_cache
    if _inoue_cache is not None:
        return _inoue_cache

    data = np.loadtxt(_INOUE_TABLE, comments="#")
    _inoue_cache = {
        "j": data[:, 0].astype(int),
        "lam_j": data[:, 1],
        "ALAF_j1": data[:, 2],
        "ALAF_j2": data[:, 3],
        "ALAF_j3": data[:, 4],
        "ADLA_j1": data[:, 5],
        "ADLA_j2": data[:, 6],
    }
    return _inoue_cache


def igm_transmission(wave_obs_A: np.ndarray, z_source: float) -> np.ndarray:
    """Mean IGM transmission using Inoue et al. (2014).

    Computes the average transmission through the intergalactic medium
    for photons observed at ``wave_obs_A`` from a source at ``z_source``,
    accounting for Lyman-series line absorption (LAF) and damped
    Lyman-alpha systems (DLA).

    Parameters
    ----------
    wave_obs_A : np.ndarray
        Observed wavelength in Angstroms.
    z_source : float
        Source redshift.

    Returns
    -------
    np.ndarray
        Transmission T(λ) ∈ [0, 1] at each wavelength.
    """
    wave_obs_A = np.asarray(wave_obs_A, dtype=float)
    tab = _load_inoue_table()

    tau_LAF = np.zeros_like(wave_obs_A)
    tau_DLA = np.zeros_like(wave_obs_A)

    for k in range(len(tab["j"])):
        lam_j = tab["lam_j"][k]
        lam_obs_j = lam_j * (1.0 + z_source)

        # Only absorb photons blueward of the Lyman line at the source redshift.
        mask = wave_obs_A < lam_obs_j

        if not np.any(mask):
            continue

        # LAF contribution (3-piece power law in z).
        z_abs = wave_obs_A[mask] / lam_j - 1.0
        z_abs = np.clip(z_abs, 0, None)

        A1 = tab["ALAF_j1"][k]
        A2 = tab["ALAF_j2"][k]
        A3 = tab["ALAF_j3"][k]

        tau_j = np.where(
            z_abs < 1.2,
            A1 * (1.0 + z_abs) ** 1.2,
            np.where(
                z_abs < 4.7,
                A2 * (1.0 + z_abs) ** 3.7,
                A3 * (1.0 + z_abs) ** 5.5,
            ),
        )
        tau_LAF[mask] += tau_j

        # DLA contribution (2-piece power law).
        D1 = tab["ADLA_j1"][k]
        D2 = tab["ADLA_j2"][k]

        tau_d = np.where(
            z_abs < 2.0,
            D1 * (1.0 + z_abs) ** 2.0,
            D2 * (1.0 + z_abs) ** 3.0,
        )
        tau_DLA[mask] += tau_d

    # Total optical depth → transmission.
    tau_total = tau_LAF + tau_DLA
    transmission = np.exp(-tau_total)

    # Ensure T=1 redward of Lyα at source redshift.
    lya_obs = _LYA_REST * (1.0 + z_source)
    transmission[wave_obs_A > lya_obs] = 1.0

    return np.clip(transmission, 0.0, 1.0)


def skewed_gaussian_binned(
    lam_left_A: np.ndarray,
    lam_right_A: np.ndarray,
    amplitude: float,
    mu_A: float,
    sigma_A: float,
    skew: float,
) -> np.ndarray:
    """Bin-averaged skewed Gaussian profile.

    Uses ``scipy.stats.skewnorm`` evaluated at bin centres, then
    normalised to unit area.  The *amplitude* parameter gives the
    integrated flux.

    Parameters
    ----------
    lam_left_A, lam_right_A : np.ndarray
        Bin edges (Å).
    amplitude : float
        Integrated flux (same units as flux × Å).
    mu_A : float
        Location parameter (Å).
    sigma_A : float
        Scale parameter (Å).
    skew : float
        Skewness parameter (0 = symmetric, positive = red tail).

    Returns
    -------
    np.ndarray
        Profile value in each bin (flux density, Å⁻¹ × amplitude units).
    """
    if sigma_A <= 0:
        return np.zeros_like(lam_left_A)

    centres = 0.5 * (lam_left_A + lam_right_A)
    widths = lam_right_A - lam_left_A

    # skewnorm.pdf normalised to unit area.
    profile = skewnorm.pdf(centres, skew, loc=mu_A, scale=sigma_A)

    return amplitude * profile


def lya_model(
    lam_left_A: np.ndarray,
    lam_right_A: np.ndarray,
    z: float,
    amplitude: float,
    mu_A: float,
    sigma_A: float,
    skew: float,
) -> np.ndarray:
    """Lyman-alpha emission model: skewed Gaussian × IGM transmission.

    Parameters
    ----------
    lam_left_A, lam_right_A : np.ndarray
        Bin edges (Å).
    z : float
        Source redshift.
    amplitude : float
        Integrated Lyα flux (before IGM absorption).
    mu_A : float
        Observed Lyα centroid (Å).
    sigma_A : float
        Observed Lyα width (Å).
    skew : float
        Skewness parameter (positive = red-asymmetric, typical for Lyα).

    Returns
    -------
    np.ndarray
        Model flux density in each bin (Å⁻¹ × flux units).
    """
    # Intrinsic skewed Gaussian profile.
    profile = skewed_gaussian_binned(
        lam_left_A, lam_right_A, amplitude, mu_A, sigma_A, skew,
    )

    # IGM absorption.
    wave_centres = 0.5 * (lam_left_A + lam_right_A)
    T_igm = igm_transmission(wave_centres, z)

    return profile * T_igm
