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
    "NIV_1486": 1486.496,
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
    "CII]_2324": 2324.21346335,
    "CII]_2326": 2325.40372661,
    # UV absorption lines
    "abs_SiII1260": 1260.422,
    "abs_CII1334":  1334.532,
    "abs_SiIV1394": 1393.755,
    "abs_SiIV1403": 1402.770,
    "abs_AlII1672": 1670.787,
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
    "HEI_4027": 4027.327,
    # Balmer series
    "HDELTA": 4104.049910602101,
    "HEI_4145": 4144.928,
    "HGAMMA": 4342.904611871652,
    "HBETA": 4864.041335024339,
    # Oxygen / Helium
    "OIII_4363": 4364.436278914932,
    "HEI_4472": 4472.73381586,
    "OIII_4959": 4961.6792505239255,
    "OIII_5007": 5009.636990676784,
    # Nitrogen, Helium
    "NII_5756": 5757.786514027234,
    # CIV optical doublet (Wolf-Rayet / AGN feature):
    # 2p²P° → 3s²S transition, vacuum wavelengths.
    "CIV_5803": 5802.97,
    "CIV_5814": 5813.62,
    "HEI_5877": 5878.88092174501,
    "OI_6302": 6302.046,
    "NII_6549": 6551.669402278604,
    "Ha": 6566.421366618156,
    "NII_6585": 6587.088921119292,
    "HEI_6680": 6680.000,
    # Sulphur
    "SII_6718": 6720.149693980653,
    "SII_6732": 6734.532561984591,
    "HEI_7067": 7067.138,
    "ArIII_7138": 7137.770,
    # Blue diagnostic lines (4500–4750 Å region)
    "FeII_4584": 4584.11922116,   
    "NIII_4642": 4641.93950581,    
    "FeIII_4660": 4659.35411192,  
    "HeII_4687": 4687.02143223,   
    "ArIV_4713": 4712.57819671,   
    "FeII_4732": 4732.76354118,   
    "ArIV_4741": 4741.44584045,   

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
    # UV absorption lines
    "abs_SiII1260",
    "abs_CII1334",
    "abs_SiIV1394",
    "abs_SiIV1403",
    "abs_AlII1672",
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
    "NIV_1486",
    "CIV_1",
    "CIV_2",
    "HEII_1640",
    # UV absorption lines
    "abs_SiII1260",
    "abs_CII1334",
    "abs_SiIV1394",
    "abs_SiIV1403",
    "abs_AlII1672",
    "OIII_1661",
    "OIII_1666",
    "NIII_1749",
    "NIII_1752",
    "SiIII_1",
    "SiIII_2",
    "CIII]_1907",
    "CIII]",
    # CII] intercombination doublet
    "CII]_2324",
    "CII]_2326",
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
    "HEI_4027",
    "HDELTA",
    "HEI_4145",
    "HGAMMA",
    "OIII_4363",
    "HEI_4472",
    # Blue diagnostic lines (4500–4750 Å)
    "FeII_4584",
    "NIII_4642",
    "FeIII_4660",
    "HeII_4687",
    "ArIV_4713",        # blended with HeI 4713
    "FeII_4732",
    "ArIV_4741",
    "HBETA",
    "OIII_4959",
    "OIII_5007",
    "NII_5756",
    "CIV_5803",
    "CIV_5814",
    "HEI_5877",
    "OI_6302",
    "NII_6549",
    "Ha",
    "NII_6585",
    "HEI_6680",
    "SII_6718",
    "SII_6732",
    "HEI_7067",
    "ArIII_7138",
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


def show_lines(
    *,
    rest_min_A: float | None = None,
    rest_max_A: float | None = None,
    search: str | None = None,
) -> None:
    """Print the available emission/absorption lines in :data:`REST_LINES_A`.

    The printed names are the keys you can pass to the ``lines=`` or
    ``add_lines=`` arguments of
    :func:`~jwspecfit.plotting.plot_spectrum_interactive` (and to the
    fitter), grouped by wavelength region.

    Parameters
    ----------
    rest_min_A, rest_max_A : float, optional
        Restrict to lines with rest wavelengths in this range
        (Angstroms).
    search : str, optional
        Case-insensitive substring filter on the line name.

    Examples
    --------
    >>> import jwspecfit
    >>> jwspecfit.show_lines(rest_min_A=4000, rest_max_A=5100)
    >>> jwspecfit.show_lines(search="Fe")
    """
    rows = []
    for name, rest_A in REST_LINES_A.items():
        if rest_min_A is not None and rest_A < rest_min_A:
            continue
        if rest_max_A is not None and rest_A > rest_max_A:
            continue
        if search is not None and search.lower() not in name.lower():
            continue
        rows.append((name, rest_A))

    if not rows:
        print("No lines match the given filters.")
        return

    rows.sort(key=lambda r: r[1])

    regions = [
        ("UV (< 3000 Å)",          lambda w: w < 3000),
        ("Optical (3000–7500 Å)",  lambda w: 3000 <= w < 7500),
        ("Near-IR (≥ 7500 Å)",     lambda w: w >= 7500),
    ]
    name_w = max(len(n) for n, _ in rows)

    print(f"{'name':<{name_w}}   rest λ (Å)")
    print("-" * (name_w + 14))
    for region_name, pred in regions:
        sub = [r for r in rows if pred(r[1])]
        if not sub:
            continue
        print(f"\n# {region_name}")
        for name, rest_A in sub:
            print(f"{name:<{name_w}}   {rest_A:10.3f}")
    print(
        "\nUse with: plot_spectrum_interactive(..., z=..., "
        'add_lines=["NAME", ...])\n'
        "    or:   plot_spectrum_interactive(..., z=..., "
        "lines=[\"NAME\", ...])   # replaces defaults"
    )
