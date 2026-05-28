"""Shared fixtures for jwspecfit tests."""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PRISM_FITS = DATA_DIR / "borg-v4_prism-clear_1747_732.spec.fits"
G395M_FITS = DATA_DIR / "excels-uds04-v4_g395m-f290lp_3543_63107.spec.fits"
G395M_FITS_2 = DATA_DIR / "stark-rxcj2248-v4_g395m-f290lp_2478_3.spec.fits"
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


# --- Cached fit results (session-scoped for speed) --------------------------

@pytest.fixture(scope="session")
def _prism_spec_session():
    from jwspecfit import read_fits
    return read_fits(PRISM_FITS, z=6.0)


@pytest.fixture(scope="session")
def _grating_spec_session():
    from jwspecfit import read_fits
    return read_fits(G395M_FITS, z=5.0)


@pytest.fixture(scope="session")
def prism_fit_result(_prism_spec_session):
    """Fit prism spectrum once per session (narrow-only, no bootstrap)."""
    from jwspecfit.fitter import fit_lines
    return fit_lines(
        _prism_spec_session, z=6.0, grating="PRISM", n_boot=0,
    )


@pytest.fixture(scope="session")
def grating_fit_result(_grating_spec_session):
    """Fit grating spectrum once per session (narrow-only, no bootstrap)."""
    from jwspecfit.fitter import fit_lines
    return fit_lines(
        _grating_spec_session, z=5.0, grating="G395M", n_boot=0,
    )


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


@pytest.fixture
def absorption_spectrum():
    """Create a synthetic spectrum with a known absorption line.

    Negative Gaussian (absorption trough) for abs_SiII1260 at z=1.0,
    observed at 2520.844 Å = 0.2521 µm.  We place the spectrum window
    around this wavelength with a flat continuum.
    """
    from jwspecfit import Spectrum
    from scipy.special import erf

    rng = np.random.default_rng(456)

    # abs_SiII1260 rest = 1260.422 Å, z = 1.0 → obs = 2520.844 Å = 0.2521 µm
    z = 1.0
    mu_A = 1260.422 * (1.0 + z)  # 2520.844 Å
    centre_um = mu_A * 1e-4

    wave_um = np.linspace(centre_um - 0.02, centre_um + 0.02, 400)
    wave_A = wave_um * 1e4

    # Flat continuum at 1.0 µJy.
    continuum_ujy = np.full_like(wave_A, 1.0)

    # Inject a negative Gaussian (absorption) in F_λ then convert to µJy.
    sigma_A = 3.0
    amplitude = -2.0e-17  # negative → absorption
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
    line_flam = amplitude * profile

    C_CGS = 2.99792458e10
    lam_cm = wave_um * 1e-4
    fnu_cgs = line_flam * 1e8 * lam_cm**2 / C_CGS
    line_ujy = fnu_cgs / 1e-29

    flux_ujy = continuum_ujy + line_ujy
    noise_level = 0.005
    err_ujy = np.full_like(flux_ujy, noise_level)
    flux_ujy += rng.normal(0, noise_level, len(flux_ujy))

    return Spectrum(
        wave_um=wave_um,
        flux_ujy=flux_ujy,
        err_ujy=err_ujy,
        grating=None,
        z=z,
        R=500.0,
    )
