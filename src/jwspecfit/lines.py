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
    "NIV_1483": 1483.321,
    "NIV_1": 1486.496,
    "NIV_doublet": 1484.91,
    "NIII_1749": 1748.646,
    "NIII_1752": 1752.160,
    "NIII_doublet": 1750.40,
    "CIV_1": 1548.187,
    "CIV_2": 1550.772,
    "CIV_doublet": 1549.48,
    "HEII_1640": 1640.42,
    "OIII_1661": 1660.809,
    "OIII_1663": 1663.48,
    "OIII_1666": 1666.15,
    "SiIII_1": 1882.71,
    "SiIII_2": 1892.03,
    "CIII]_1907": 1906.683,
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
    # Higher Balmer series + blue lines
    "H10": 3798.98234713,
    "H9": 3836.47908996,
    "NeIII_3869": 3869.85677179,
    "HEI_3889": 3889.74894995,
    "H8": 3890.16605856,
    "HEPSILON": 3971.20218326,
    # Balmer series
    "HDELTA": 4104.049910602101,
    "HGAMMA": 4342.904611871652,
    "HBETA": 4864.041335024339,
    # Oxygen / Helium
    "OIII_4363": 4364.436278914932,
    "HEI_4472": 4472.73381586,
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
    # Additional diagnostic lines (opt-in; not in default line lists)
    "HeII_4687": 4687.02,    # TODO: verify vacuum wavelength (air 4685.710)
    "ArIV_4713": 4712.58,    # TODO: verify vacuum wavelength (air 4711.260)
    "ArIV_4741": 4741.48,    # TODO: verify vacuum wavelength (air 4740.170)
    "SIII_6312": 6313.81,    # TODO: verify vacuum wavelength (air 6312.060)
    "OII_7320": 7322.01,     # TODO: verify vacuum wavelength (air 7319.990)
    "OII_7330": 7332.75,     # TODO: verify vacuum wavelength (air 7330.730)
    "ArIII_7136": 7137.77,   # TODO: verify vacuum wavelength (air 7135.790)
    "SIII_9069": 9071.10,    # TODO: verify vacuum wavelength (air 9068.600)
}

# ---------------------------------------------------------------------------
# Pre-defined line groups for different spectral resolutions.
#
# Prism (R ~ 30–300):  merged doublet entries for unresolvable pairs.
# Grating (R ~ 1000+): individual doublet components resolved.
#
# These are the *default* line lists used when the user does not supply
# their own ``lines=`` argument.  The ``observable_lines()`` filter then
# removes anything outside the observed wavelength range.
# ---------------------------------------------------------------------------

_PRISM_LINES = [
    # UV (merged doublets at prism resolution)
    "Lya",
    "NV_doublet",
    "NIV_doublet",
    "CIV_doublet",
    "HEII_1640",
    "NIII_doublet",
    "CIII]",
    # Semi-forbidden / weak UV
    "OIII]_2331",       # OIII]_2321/2331 blended at prism R; keep brighter 2331
    "OII]_2471",
    "FeII*_2396",
    # Optical (merged [OII] doublet)
    "OII_doublet",
    "H10",
    "H9",
    "NeIII_3869",
    "H8",               # HeI 3889 blended with H8 (Δ=0.4 Å); keep H8
    "HEPSILON",
    "HDELTA",
    "HGAMMA",
    "OIII_4363",
    "HEI_4472",
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

_GRATING_LINES = [
    # UV (individual doublet components)
    "Lya",
    "NV_1",
    "NV_2",
    "NIV_1483",
    "NIV_1",
    "CIV_1",
    "CIV_2",
    "HEII_1640",
    "OIII_1661",
    "OIII_1666",
    "NIII_1749",
    "NIII_1752",
    "SiIII_1",
    "SiIII_2",
    "CIII]_1907",
    "CIII]",
    # Semi-forbidden / weak UV
    "OIII]_2321",
    "OIII]_2331",
    "OII]_2471",
    "FeII*_2396",
    # Optical (resolved [OII] doublet)
    "OII_3726",
    "OII_3729",
    "H10",
    "H9",
    "NeIII_3869",
    "H8",               # HeI 3889 blended with H8 (Δ=0.4 Å); unresolvable at any NIRSpec R
    "HEPSILON",
    "HDELTA",
    "HGAMMA",
    "OIII_4363",
    "HEI_4472",
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


def get_line_list(grating: str = "prism") -> list[str]:
    """Return default line names for a given grating.

    For prism (R ~ 30–300), merged doublet entries are used for pairs
    that cannot be resolved.  For medium and high-resolution gratings
    (R ≥ 1000), individual doublet components are returned.

    The user can always override by passing an explicit ``lines=``
    argument to :func:`fit_lines`.

    Parameters
    ----------
    grating : str
        One of ``"prism"``, ``"medium"`` / ``"g140m"`` / ``"g235m"`` /
        ``"g395m"``, ``"high"`` / ``"g140h"`` / ``"g235h"`` / ``"g395h"``,
        or ``"grating"`` (generic resolved mode).

    Returns
    -------
    list of str
        Line names present in :data:`REST_LINES_A`.
    """
    g = grating.lower()
    if "prism" in g:
        return list(_PRISM_LINES)
    # All gratings (medium *and* high) use resolved individual components.
    if any(k in g for k in (
        "medium", "high", "grating", "stack",
        "g140m", "g235m", "g395m",
        "g140h", "g235h", "g395h",
    )):
        return list(_GRATING_LINES)
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

    # Exclude lines blueward of NV (IGM-absorbed region).
    # Lya needs special treatment (see lyman_alpha module) and should
    # never be fitted as a regular Gaussian.  NV is the bluest line
    # that can be reliably fitted.
    nv_obs_um = REST_LINES_A["NV_1"] * (1 + z) * 1e-4

    out = []
    for name in line_names:
        lam_obs_um = REST_LINES_A[name] * (1 + z) * 1e-4
        if lam_obs_um < nv_obs_um:
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
