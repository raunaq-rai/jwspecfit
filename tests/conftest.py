"""Shared fixtures for jwspecfit tests."""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PRISM_FITS = DATA_DIR / "borg-v4_prism-clear_1747_732.spec.fits"
G395M_FITS = DATA_DIR / "excels-uds04-v4_g395m-f290lp_3543_63107.spec.fits"
G395M_FITS_2 = DATA_DIR / "stark-rxcj2248-v4_g395m-f290lp_2478_3.spec.fits"
STACK_NPZ = DATA_DIR / "stack_all_Muv19_21_DustCorrected.npz"
INOUE_TABLE = DATA_DIR / "inoue2014_table2.txt"


@pytest.fixture
def prism_spectrum():
    """Load prism spectrum."""
    from jwspecfit import read_fits
    return read_fits(PRISM_FITS, z=6.0)


@pytest.fixture
def grating_spectrum():
    """Load G395M grating spectrum."""
    from jwspecfit import read_fits
    return read_fits(G395M_FITS, z=5.0)


@pytest.fixture
def synthetic_spectrum():
    """Create a synthetic spectrum with known emission lines.

    Single Gaussian line at 5000 Å (rest) at z=1.0, observed at 10000 Å = 1.0 µm.
    """
    from jwspecfit import Spectrum

    rng = np.random.default_rng(123)
    wave_um = np.linspace(0.8, 1.2, 400)
    wave_A = wave_um * 1e4

    # Continuum: flat at 0.1 µJy
    continuum = np.full_like(wave_A, 0.1)

    # Add a Gaussian emission line at 1.0 µm (10000 Å).
    from scipy.special import erf
    sigma_A = 20.0  # Å
    mu_A = 10000.0
    amplitude = 5.0e-17  # erg/s/cm²
    # Convert to µJy for the spectrum.
    inv = 1.0 / (np.sqrt(2.0) * sigma_A)
    edges = np.zeros(len(wave_A) + 1)
    edges[1:-1] = 0.5 * (wave_A[:-1] + wave_A[1:])
    edges[0] = 2 * wave_A[0] - edges[1]
    edges[-1] = 2 * wave_A[-1] - edges[-2]
    left, right = edges[:-1], edges[1:]
    cdf_r = 0.5 * (1.0 + erf((right - mu_A) * inv))
    cdf_l = 0.5 * (1.0 + erf((left - mu_A) * inv))
    width = right - left
    profile = (cdf_r - cdf_l) / width
    # profile is in Å⁻¹; amplitude is in flux×Å
    line_flam = amplitude * profile  # erg/s/cm²/Å

    # Convert to µJy
    C_CGS = 2.99792458e10
    lam_cm = wave_um * 1e-4
    fnu_cgs = line_flam * 1e8 * lam_cm**2 / C_CGS
    line_ujy = fnu_cgs / 1e-29

    flux_ujy = continuum + line_ujy
    noise_level = 0.01
    err_ujy = np.full_like(flux_ujy, noise_level)
    flux_ujy += rng.normal(0, noise_level, len(flux_ujy))

    return Spectrum(
        wave_um=wave_um,
        flux_ujy=flux_ujy,
        err_ujy=err_ujy,
        grating=None,
        z=1.0,
        R=100.0,
    )
