"""Spectrum I/O: FITS and NPZ readers, Spectrum container."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# Physical constants for unit conversion.
_C_CGS = 2.99792458e10  # cm/s


@dataclass
class Spectrum:
    """Container for a 1-D spectrum.

    Attributes
    ----------
    wave_um : np.ndarray
        Observed wavelength in microns.
    flux_ujy : np.ndarray
        Flux density in micro-Jansky.
    err_ujy : np.ndarray
        1-sigma uncertainty in micro-Jansky.
    grating : str or None
        Grating name (e.g. ``"PRISM"``, ``"G395M"``).  ``None`` for stacked spectra.
    z : float or None
        Source redshift (set by user, not from header).
    R : float or callable or None
        Spectral resolving power.  Overrides ``grating`` when set (useful for stacks).
    meta : dict
        Arbitrary metadata from the FITS header or user.
    """

    wave_um: np.ndarray
    flux_ujy: np.ndarray
    err_ujy: np.ndarray
    grating: str | None = None
    z: float | None = None
    R: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # --- Derived properties ---------------------------------------------------

    @property
    def wave_A(self) -> np.ndarray:
        """Wavelength in Angstroms."""
        return self.wave_um * 1e4

    @property
    def n_pix(self) -> int:
        return len(self.wave_um)

    @property
    def wave_edges_A(self) -> np.ndarray:
        """Pixel-edge wavelengths in Angstroms (length n_pix + 1)."""
        w = self.wave_A
        mid = 0.5 * (w[:-1] + w[1:])
        left = 2.0 * w[0] - mid[0]
        right = 2.0 * w[-1] - mid[-1]
        return np.concatenate([[left], mid, [right]])

    @property
    def dlam_A(self) -> np.ndarray:
        """Pixel widths in Angstroms."""
        edges = self.wave_edges_A
        return edges[1:] - edges[:-1]

    @property
    def flux_flam(self) -> np.ndarray:
        """Flux density in erg/s/cm²/Å."""
        return _ujy_to_flam(self.flux_ujy, self.wave_um)

    @property
    def err_flam(self) -> np.ndarray:
        """Error in erg/s/cm²/Å."""
        return _ujy_to_flam(self.err_ujy, self.wave_um)

    def mask_valid(self) -> np.ndarray:
        """Boolean mask: True where flux and error are finite and err > 0."""
        return np.isfinite(self.flux_ujy) & np.isfinite(self.err_ujy) & (self.err_ujy > 0)

    def copy(self) -> "Spectrum":
        """Return a shallow copy with copied arrays."""
        return Spectrum(
            wave_um=self.wave_um.copy(),
            flux_ujy=self.flux_ujy.copy(),
            err_ujy=self.err_ujy.copy(),
            grating=self.grating,
            z=self.z,
            R=self.R,
            meta=dict(self.meta),
        )


def read_fits(path: str | Path, z: float | None = None) -> Spectrum:
    """Read a JWST NIRSpec 1-D extracted spectrum from FITS.

    Expects an HDU named ``SPEC1D`` with columns ``wave``, ``flux``, ``err``
    (wavelength in µm, flux and error in µJy).

    Parameters
    ----------
    path : str or Path
        Path to the FITS file.
    z : float, optional
        Source redshift to attach.

    Returns
    -------
    Spectrum
    """
    path = Path(path)
    with fits.open(path) as hdul:
        data = hdul["SPEC1D"].data
        header = hdul["SPEC1D"].header

        wave_um = np.asarray(data["wave"], dtype=float)
        flux_ujy = np.asarray(data["flux"], dtype=float)
        err_ujy = np.asarray(data["err"], dtype=float)

        grating = header.get("GRATING", None)
        filt = header.get("FILTER", None)

    meta = {"filename": path.name, "filter": filt}
    logger.info("Read %s: %d pixels, grating=%s", path.name, len(wave_um), grating)

    return Spectrum(
        wave_um=wave_um,
        flux_ujy=flux_ujy,
        err_ujy=err_ujy,
        grating=grating,
        z=z,
        meta=meta,
    )


def read_npz(
    path: str | Path,
    z: float | None = None,
    R: float | None = None,
) -> Spectrum:
    """Read a stacked spectrum from a NumPy .npz file.

    Expected keys: ``wave_angstrom``, ``flux``, ``err``.
    Optionally ``n_stacked``.

    Parameters
    ----------
    path : str or Path
        Path to the .npz file.
    z : float, optional
        Source redshift.
    R : float, optional
        Effective spectral resolving power of the stack.

    Returns
    -------
    Spectrum
    """
    path = Path(path)
    npz = np.load(path, allow_pickle=False)

    wave_A = np.asarray(npz["wave_angstrom"], dtype=float)
    flux = np.asarray(npz["flux"], dtype=float)
    err = np.asarray(npz["err"], dtype=float)

    meta = {"filename": path.name}
    if "n_stacked" in npz:
        meta["n_stacked"] = int(npz["n_stacked"])

    logger.info("Read %s: %d pixels, R=%s", path.name, len(wave_A), R)

    return Spectrum(
        wave_um=wave_A * 1e-4,
        flux_ujy=flux,
        err_ujy=err,
        grating=None,
        z=z,
        R=R,
        meta=meta,
    )


def read_dict(
    data: dict[str, np.ndarray],
    z: float | None = None,
    grating: str | None = None,
    R: float | None = None,
) -> Spectrum:
    """Create a Spectrum from a dict with keys ``wave``/``lam``, ``flux``, ``err``.

    Wavelength assumed in microns.

    Parameters
    ----------
    data : dict
        Must contain ``"wave"`` or ``"lam"`` (µm), ``"flux"`` (µJy), ``"err"`` (µJy).
    z : float, optional
        Source redshift.
    grating : str, optional
        Grating name.
    R : float, optional
        Resolving power.

    Returns
    -------
    Spectrum
    """
    wave = np.asarray(data.get("wave", data.get("lam")), dtype=float)
    flux = np.asarray(data["flux"], dtype=float)
    err = np.asarray(data["err"], dtype=float)

    return Spectrum(wave_um=wave, flux_ujy=flux, err_ujy=err, grating=grating, z=z, R=R)


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def _ujy_to_flam(flux_ujy: np.ndarray, wave_um: np.ndarray) -> np.ndarray:
    """Convert µJy → erg/s/cm²/Å."""
    lam_cm = wave_um * 1e-4
    fnu_cgs = flux_ujy * 1e-29  # µJy → erg/s/cm²/Hz
    # F_λ = F_ν · c / λ² (in CGS), then /1e8 to get per-Å instead of per-cm
    return fnu_cgs * _C_CGS / (lam_cm**2) / 1e8


def _flam_to_ujy(flux_flam: np.ndarray, wave_um: np.ndarray) -> np.ndarray:
    """Convert erg/s/cm²/Å → µJy."""
    lam_cm = wave_um * 1e-4
    fnu_cgs = flux_flam * 1e8 * lam_cm**2 / _C_CGS
    return fnu_cgs / 1e-29
