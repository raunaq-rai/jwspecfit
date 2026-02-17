"""Emission-line database and line-list helpers."""

from __future__ import annotations

import numpy as np

# Rest-frame vacuum wavelengths in Angstroms.
# Sources: NIST ASD, Morton (2003), Storey & Zeippen (2000).
REST_LINES_A: dict[str, float] = {
    # UV lines
    "Lya": 1215.670,
    "NV_1": 1238.821,
    "NV_2": 1242.804,
    "NV_doublet": 1240.81,
    "NIV_1": 1486.496,
    "CIV_1": 1548.187,
    "CIV_2": 1550.772,
    "CIV_doublet": 1549.48,
    "HEII_1640": 1640.42,
    "OIII_1663": 1663.48,
    "SiIII_1": 1882.71,
    "SiIII_2": 1892.03,
    "CIII]": 1908.734,
    # Semi-forbidden / weak
    "OIII]_2321": 2322.41306535397,
    "OIII]_2331": 2332.015190970458,
    "OII]_2471": 2471.7168502042923,
    "FeII*_2396": 2397.0897781493236,
    # Optical doublets
    "OII_3726": 3727.0917000220848,
    "OII_3729": 3729.8754212488416,
    "OII_doublet": 3728.4800597199715,
    # Balmer series
    "HDELTA": 4104.049910602101,
    "HGAMMA": 4342.904611871652,
    "HBETA": 4864.041335024339,
    # Oxygen
    "OIII_4363": 4364.436278914932,
    "OIII_4959": 4961.6792505239255,
    "OIII_5007": 5009.636990676784,
    # Nitrogen, Helium
    "NII_5756": 5757.786514027234,
    "HEI_5877": 5878.88092174501,
    "NII_6549": 6551.669402278604,
    "Ha": 6566.421366618156,
    "NII_6585": 6587.088921119292,
    # Sulphur
    "SII_6718": 6720.149693980653,
    "SII_6732": 6734.532561984591,
}

# Pre-defined line groups for different spectral resolutions.
# Prism: doublets are merged; grating: doublets resolved.
_PRISM_LINES = [
    "Lya",
    "CIV_doublet",
    "CIII]",
    "OII_doublet",
    "HDELTA",
    "HGAMMA",
    "OIII_4363",
    "HBETA",
    "OIII_4959",
    "OIII_5007",
    "NII_6549",
    "Ha",
    "NII_6585",
    "SII_6718",
    "SII_6732",
]

_MEDIUM_LINES = [
    "OII_3726",
    "OII_3729",
    "HDELTA",
    "HGAMMA",
    "OIII_4363",
    "HBETA",
    "OIII_4959",
    "OIII_5007",
    "NII_5756",
    "HEI_5877",
    "NII_6549",
    "Ha",
    "NII_6585",
    "SII_6718",
    "SII_6732",
]

_HIGH_LINES = _MEDIUM_LINES  # same set; resolution handles the splitting


def get_line_list(grating: str = "prism") -> list[str]:
    """Return default line names for a given grating.

    Parameters
    ----------
    grating : str
        One of ``"prism"``, ``"medium"`` / ``"g140m"`` / ``"g235m"`` / ``"g395m"``,
        or ``"high"`` / ``"g140h"`` / ``"g235h"`` / ``"g395h"``.

    Returns
    -------
    list of str
        Line names present in :data:`REST_LINES_A`.
    """
    g = grating.lower()
    if "prism" in g:
        return list(_PRISM_LINES)
    if any(k in g for k in ("medium", "g140m", "g235m", "g395m")):
        return list(_MEDIUM_LINES)
    if any(k in g for k in ("high", "g140h", "g235h", "g395h")):
        return list(_HIGH_LINES)
    return list(_PRISM_LINES)


def observable_lines(
    line_names: list[str],
    z: float,
    wave_min_um: float,
    wave_max_um: float,
    *,
    margin_sigma: float = 3.0,
    sigma_um: float = 0.005,
) -> list[str]:
    """Filter lines to those observable in the wavelength range.

    Parameters
    ----------
    line_names : list of str
        Candidate line names (keys of :data:`REST_LINES_A`).
    z : float
        Source redshift.
    wave_min_um, wave_max_um : float
        Observed wavelength range in microns.
    margin_sigma : float
        Number of sigma margin from the edges.
    sigma_um : float
        Approximate line width in microns (for margin calculation).

    Returns
    -------
    list of str
        Lines whose observed wavelength falls within the range.
    """
    margin = margin_sigma * sigma_um
    lo = wave_min_um + margin
    hi = wave_max_um - margin

    # Exclude lines at or blueward of Lyman-alpha (IGM-absorbed).
    # Lya itself needs special treatment (see lyman_alpha module) and
    # should never be fitted as a regular Gaussian.
    lya_obs_um = REST_LINES_A["Lya"] * (1 + z) * 1e-4

    out = []
    for name in line_names:
        lam_obs_um = REST_LINES_A[name] * (1 + z) * 1e-4
        if lam_obs_um <= lya_obs_um:
            continue
        if lo <= lam_obs_um <= hi:
            out.append(name)
    return out


def rest_wave_A(name: str) -> float:
    """Return rest wavelength in Angstroms for a line name.

    Parameters
    ----------
    name : str
        Line name (key of :data:`REST_LINES_A`).

    Returns
    -------
    float
        Rest wavelength in Angstroms.

    Raises
    ------
    KeyError
        If the line name is not found.
    """
    return REST_LINES_A[name]


def observed_wave_A(name: str, z: float) -> float:
    """Return observed wavelength in Angstroms for a line at redshift *z*."""
    return REST_LINES_A[name] * (1.0 + z)


def observed_wave_um(name: str, z: float) -> float:
    """Return observed wavelength in microns for a line at redshift *z*."""
    return REST_LINES_A[name] * (1.0 + z) * 1e-4
