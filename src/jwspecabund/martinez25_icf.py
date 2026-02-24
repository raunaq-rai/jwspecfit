"""Density-dependent ICFs and logU diagnostics from Martinez+2025.

Bicubic surface fits to Cloudy photoionization models for five N/O
ionisation correction factors and two ionisation parameter diagnostics,
each evaluated on a grid of electron densities (10^2 to 10^6 cm^-3).

References
----------
Martinez, M. A., Berg, D. A., et al. 2025, arXiv:2510.21960.
    Table 3 (ionisation parameter fits) and Table 4 (ICF fits).
Berg, D. A., et al. 2025, arXiv:2511.13591.
    Recommended 6-step procedure for UV N/O abundance determinations.

The bicubic surface is:
    f(x, y) = A + Bx + Cy + Dxy + Ex^2 + Fy^2 + Gxy^2 + Hyx^2 + Ix^3 + Jy^3

where x and y depend on the diagnostic (see individual function docstrings).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Solar oxygen abundance: 12 + log(O/H)_sun (Asplund+2009).
LOG_OH_SOLAR = 8.69

# Density grid points (log10 ne in cm^-3).
_LOG_NE_GRID = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

# Validity ranges (Martinez+2025 Section 4).
_LOG_U_VALID = (-3.5, -1.0)
_Z_ZSUN_VALID = (0.05, 0.40)
_LOG_O32_VALID = (-1.0, 2.5)
_LOG_N43_VALID = (-3.0, 0.5)


# =========================================================================
# Coefficient tables
# =========================================================================
# Each dict maps log10(ne) -> [A, B, C, D, E, F, G, H, I, J].

# --- log(U) from O32 = [OIII]5007/[OII]3727 (Table 3, upper) ---
# x = log(O32), y = Z/Zsun, returns log(U_int)
_COEFF_LOG_U_O32: dict[float, list[float]] = {
    2: [-3.00199,  0.61875,  0.41277,  0.96910,  0.11489,  2.10518, -0.88265,  0.55524,  0.18323, -4.65728],
    3: [-3.06316,  0.65186,  0.45934,  0.60495,  0.04757,  1.34596, -0.61689,  0.25114,  0.04025, -3.24251],
    4: [-3.28872,  0.62685,  0.32662,  0.44956,  0.05904,  1.19054, -0.61832,  0.19987,  0.00771, -2.62744],
    5: [-3.71434,  0.53053,  0.07827,  0.16911,  0.05866,  1.65578, -0.62173,  0.18713,  0.00364, -2.54602],
    6: [-4.08448,  0.54480,  0.17599, -0.23221,  0.01351,  2.15922, -0.47220,  0.19535,  0.00841, -2.95869],
}

# --- log(U) from N43 = NIV]1486/NIII]1750 (Table 3, lower) ---
# x = log(N43), y = Z/Zsun, returns log(U_high)
_COEFF_LOG_U_N43: dict[float, list[float]] = {
    2: [-1.29754,  1.60610,  4.10271,  3.82481,  0.52310, 12.37774,  0.22144,  0.74401,  0.09083, -17.63486],
    3: [-1.61212,  1.03768,  2.59196,  2.04872,  0.18279,  9.37845, -0.27803,  0.32021,  0.02670, -15.11384],
    4: [-1.66787,  0.91085,  2.26720,  1.63593,  0.09245,  8.61368, -0.45877,  0.20331,  0.00723, -14.59398],
    5: [-1.67830,  0.89072,  2.20159,  1.56124,  0.08156,  8.23949, -0.55776,  0.17851,  0.00484, -14.35798],
    6: [-1.68390,  0.88034,  1.93562,  1.55433,  0.08954,  8.03766, -0.43756,  0.22411,  0.00874, -13.90218],
}

# --- ICF 1: N+/O+ -> N/O (Table 4) ---
# x = log(U), y = Z/Zsun
_COEFF_ICF_NpOp: dict[float, list[float]] = {
    2: [ 0.99684,  0.35805,  0.73829, -0.00147,  0.25599, -1.05333, -0.00247, -0.04340,  0.04517,  1.35162],
    3: [ 1.11259,  0.68396,  0.57962, -0.20911,  0.37927,  0.13139, -0.14117, -0.11589,  0.05690, -1.00523],
    4: [ 1.04256,  0.52349,  0.14444, -0.50730,  0.27256,  0.78050,  0.00501, -0.15029,  0.04052, -1.67611],
    5: [ 1.05359,  0.54522,  0.02129, -0.52622,  0.28401,  1.11581,  0.18414, -0.12303,  0.04511, -1.75450],
    6: [ 1.05845,  0.58814,  0.28991, -0.33298,  0.31784,  1.11112,  0.20679, -0.06881,  0.05389, -2.03291],
}

# --- ICF 2: N2+/O2+ -> N/O (Table 4) ---
_COEFF_ICF_NppOpp: dict[float, list[float]] = {
    2: [ 2.71718,  1.40827, -3.80233, -1.98828,  0.41454,  2.86050,  0.89889, -0.26525,  0.04878,  0.13269],
    3: [ 4.18005,  2.73917, -7.42258, -4.08756,  0.81576,  5.91850,  1.77628, -0.56759,  0.08888, -0.43755],
    4: [ 4.96600,  3.48636, -9.32333, -5.25762,  1.04937,  7.28429,  2.22999, -0.74483,  0.11297, -0.42475],
    5: [ 5.20832,  3.71419, -9.83354, -5.59671,  1.12023,  7.46491,  2.33000, -0.79911,  0.12032, -0.22878],
    6: [ 5.32595,  3.82306, -9.73777, -5.65519,  1.15115,  6.66459,  2.20890, -0.82169,  0.12301,  0.41586],
}

# --- ICF 3: N3+/O2+ -> N/O (Table 4) ---
# NOTE: this surface returns log(ICF), not ICF directly.
_COEFF_ICF_NpppOpp: dict[float, list[float]] = {
    2: [ 0.33721,  0.24224, -0.65061, -1.33811,  0.27971,  5.83889,  0.31924, -0.26487,  0.00662, -9.33063],
    3: [ 0.32989,  0.53349, -0.68007, -1.56364,  0.45572,  5.36918,  0.52243, -0.30772,  0.03347, -8.05623],
    4: [ 0.27897,  0.56120, -0.57998, -1.54907,  0.48623,  5.11709,  0.54710, -0.30843,  0.03900, -7.62178],
    5: [ 0.27431,  0.58497, -0.51353, -1.53730,  0.49791,  4.79966,  0.52024, -0.31010,  0.04044, -7.17482],
    6: [ 0.24361,  0.58882, -0.17741, -1.30763,  0.50260,  3.91905,  0.30299, -0.28039,  0.04061, -6.28983],
}

# --- ICF 4: (N+ + N2+)/(O+ + O2+) -> N/O (Table 4) ---
_COEFF_ICF_NpNpp_OpOpp: dict[float, list[float]] = {
    2: [ 2.80532,  1.51853, -4.17309, -2.26441,  0.42943,  3.55046,  1.09105, -0.29179,  0.04290, -0.39350],
    3: [ 4.28328,  2.84543, -7.91458, -4.44062,  0.81553,  6.75686,  2.04896, -0.59753,  0.08034, -0.97013],
    4: [ 5.00990,  3.50850, -9.76639, -5.51060,  1.01895,  8.30719,  2.52596, -0.74177,  0.10332, -1.20540],
    5: [ 5.23403,  3.73697,-10.16471, -5.67998,  1.11019,  8.65627,  2.76166, -0.73561,  0.11846, -0.92040],
    6: [ 5.38109,  3.91911, -9.97970, -5.52784,  1.19788,  8.25633,  2.83279, -0.67993,  0.13442, -0.33026],
}

# --- ICF 5: (N2+ + N3+)/O2+ -> N/O (Table 4) ---
_COEFF_ICF_NppNppp_Opp: dict[float, list[float]] = {
    2: [ 0.93486, -0.03453,  0.28011,  0.24291,  0.02786, -0.15188, -0.13845,  0.02356,  0.01429, -0.16515],
    3: [ 0.93562, -0.03369,  0.27477,  0.22211,  0.02993, -0.18937, -0.14771,  0.01550,  0.01478, -0.12369],
    4: [ 0.94769, -0.01451,  0.25282,  0.16544,  0.03802, -0.30544, -0.14786, -0.00045,  0.01579,  0.04361],
    5: [ 0.93356, -0.01832,  0.32434,  0.17688,  0.04019, -0.47628, -0.17401, -0.00391,  0.01642,  0.18150],
    6: [ 0.84213, -0.08984,  0.67947,  0.32963,  0.02188, -1.01073, -0.29574,  0.00886,  0.01486,  0.42906],
}


# =========================================================================
# Core evaluation functions
# =========================================================================

def _bicubic(x: float, y: float, coeffs: list[float]) -> float:
    """Evaluate a bicubic surface f(x, y) from 10 coefficients.

    Parameters
    ----------
    x : float
        First independent variable.
    y : float
        Second independent variable.
    coeffs : list of float
        [A, B, C, D, E, F, G, H, I, J].

    Returns
    -------
    float
        Surface value f(x, y).
    """
    A, B, C, D, E, F, G, H, I, J = coeffs
    return (A + B*x + C*y + D*x*y + E*x**2 + F*y**2
            + G*x*y**2 + H*y*x**2 + I*x**3 + J*y**3)


def _interp_ne(
    ne: float,
    coeff_table: dict[float, list[float]],
    x: float,
    y: float,
) -> float:
    """Evaluate the bicubic surface interpolated in log(ne) space.

    Linearly interpolates between the two bounding density grids.
    Clamps to [10^2, 10^6] cm^-3.

    Parameters
    ----------
    ne : float
        Electron density in cm^-3.
    coeff_table : dict
        Coefficient table mapping log10(ne) -> [A..J].
    x : float
        First surface variable (e.g. log(O32), log(U)).
    y : float
        Second surface variable (e.g. Z/Zsun).

    Returns
    -------
    float
        Interpolated surface value.
    """
    log_ne = np.clip(np.log10(max(ne, 1.0)), _LOG_NE_GRID[0], _LOG_NE_GRID[-1])

    # Exact grid match — no interpolation needed.
    for grid_pt in _LOG_NE_GRID:
        if abs(log_ne - grid_pt) < 1e-6:
            return _bicubic(x, y, coeff_table[grid_pt])

    # Find bounding grid points.
    idx_hi = int(np.searchsorted(_LOG_NE_GRID, log_ne))
    log_ne_lo = _LOG_NE_GRID[idx_hi - 1]
    log_ne_hi = _LOG_NE_GRID[idx_hi]

    val_lo = _bicubic(x, y, coeff_table[log_ne_lo])
    val_hi = _bicubic(x, y, coeff_table[log_ne_hi])

    # Linear interpolation in log(ne).
    frac = (log_ne - log_ne_lo) / (log_ne_hi - log_ne_lo)
    return val_lo + frac * (val_hi - val_lo)


# =========================================================================
# Ionisation parameter diagnostics
# =========================================================================

def log_U_from_O32(
    log_O32: float,
    Z_Zsun: float,
    ne: float = 100.0,
) -> float:
    """Compute log(U_int) from the O32 diagnostic.

    Parameters
    ----------
    log_O32 : float
        log10([OIII] 5007 / [OII] 3727).
    Z_Zsun : float
        Gas-phase metallicity in solar units.
    ne : float
        Electron density in cm^-3 (default 100).

    Returns
    -------
    float
        log(U_int).

    Notes
    -----
    Valid for 0.05 <= Z/Zsun <= 0.4, -1.0 <= log(O32) <= 2.5.
    O32 is density-sensitive; prefer N43 when available.
    Martinez+2025 Table 3 (upper).
    """
    if not (_LOG_O32_VALID[0] <= log_O32 <= _LOG_O32_VALID[1]):
        logger.warning(
            "log(O32)=%.2f outside validity range [%.1f, %.1f].",
            log_O32, *_LOG_O32_VALID,
        )
    if not (_Z_ZSUN_VALID[0] <= Z_Zsun <= _Z_ZSUN_VALID[1]):
        logger.warning(
            "Z/Zsun=%.3f outside validity range [%.2f, %.2f].",
            Z_Zsun, *_Z_ZSUN_VALID,
        )
    return _interp_ne(ne, _COEFF_LOG_U_O32, log_O32, Z_Zsun)


def log_U_from_N43(
    log_N43: float,
    Z_Zsun: float,
    ne: float = 100.0,
) -> float:
    """Compute log(U_high) from the N43 diagnostic.

    Parameters
    ----------
    log_N43 : float
        log10(NIV] 1486 / NIII] 1750).
    Z_Zsun : float
        Gas-phase metallicity in solar units.
    ne : float
        Electron density in cm^-3 (default 100).

    Returns
    -------
    float
        log(U_high).

    Notes
    -----
    Valid for 0.05 <= Z/Zsun <= 0.4, -3.0 <= log(N43) <= 0.5.
    N43 is density-insensitive — **recommended** over O32 when
    NIV] and NIII] are available (Martinez+2025 Section 4.2).
    Martinez+2025 Table 3 (lower).
    """
    if not (_LOG_N43_VALID[0] <= log_N43 <= _LOG_N43_VALID[1]):
        logger.warning(
            "log(N43)=%.2f outside validity range [%.1f, %.1f].",
            log_N43, *_LOG_N43_VALID,
        )
    if not (_Z_ZSUN_VALID[0] <= Z_Zsun <= _Z_ZSUN_VALID[1]):
        logger.warning(
            "Z/Zsun=%.3f outside validity range [%.2f, %.2f].",
            Z_Zsun, *_Z_ZSUN_VALID,
        )
    return _interp_ne(ne, _COEFF_LOG_U_N43, log_N43, Z_Zsun)


# =========================================================================
# ICF evaluation functions
# =========================================================================

def _check_logU_range(logU: float) -> None:
    """Warn if logU is outside the validity range."""
    if not (_LOG_U_VALID[0] <= logU <= _LOG_U_VALID[1]):
        logger.warning(
            "log(U)=%.2f outside validity range [%.1f, %.1f].",
            logU, *_LOG_U_VALID,
        )


def icf_NpOp(logU: float, Z_Zsun: float, ne: float = 100.0) -> float:
    """ICF 1: N/O = (N+/O+) x ICF.

    Parameters
    ----------
    logU : float
        Ionisation parameter log(U).
    Z_Zsun : float
        Gas-phase metallicity in solar units.
    ne : float
        Electron density in cm^-3 (default 100).

    Returns
    -------
    float
        ICF value (multiplicative correction to N+/O+).

    Notes
    -----
    At low density (ne ~ 100), the correction is small (< 10%);
    at high density (ne ~ 10^6), can overestimate N/O by up to 40%.
    Martinez+2025 Table 4, row 1.
    """
    _check_logU_range(logU)
    return _interp_ne(ne, _COEFF_ICF_NpOp, logU, Z_Zsun)


def icf_NppOpp(logU: float, Z_Zsun: float, ne: float = 100.0) -> float:
    """ICF 2: N/O = (N2+/O2+) x ICF.

    Parameters
    ----------
    logU : float
        Ionisation parameter log(U).
    Z_Zsun : float
        Gas-phase metallicity in solar units.
    ne : float
        Electron density in cm^-3 (default 100).

    Returns
    -------
    float
        ICF value (multiplicative correction to N2+/O2+).

    Notes
    -----
    Overestimates N/O at low logU, underestimates at high logU.
    Martinez+2025 Table 4, row 2.
    """
    _check_logU_range(logU)
    return _interp_ne(ne, _COEFF_ICF_NppOpp, logU, Z_Zsun)


def icf_NpppOpp(logU: float, Z_Zsun: float, ne: float = 100.0) -> float:
    """ICF 3: N/O = (N3+/O2+) x ICF.

    Parameters
    ----------
    logU : float
        Ionisation parameter log(U).
    Z_Zsun : float
        Gas-phase metallicity in solar units.
    ne : float
        Electron density in cm^-3 (default 100).

    Returns
    -------
    float
        ICF value (multiplicative correction to N3+/O2+).

    Notes
    -----
    The bicubic surface returns log(ICF); this function exponentiates
    to return the ICF itself.  N3+/O2+ alone underestimates N/O by
    100--15000%.  Martinez+2025 Table 4, row 3.
    """
    _check_logU_range(logU)
    log_icf = _interp_ne(ne, _COEFF_ICF_NpppOpp, logU, Z_Zsun)
    return 10.0 ** log_icf


def icf_NpNpp_OpOpp(logU: float, Z_Zsun: float, ne: float = 100.0) -> float:
    """ICF 4: N/O = ((N+ + N2+)/(O+ + O2+)) x ICF.

    Parameters
    ----------
    logU : float
        Ionisation parameter log(U).
    Z_Zsun : float
        Gas-phase metallicity in solar units.
    ne : float
        Electron density in cm^-3 (default 100).

    Returns
    -------
    float
        ICF value (multiplicative correction to (N++N2+)/(O++O2+)).

    Notes
    -----
    Combines optical (N+, O+) and UV (N2+, O2+) ionic abundances.
    Martinez+2025 Table 4, row 4.
    """
    _check_logU_range(logU)
    return _interp_ne(ne, _COEFF_ICF_NpNpp_OpOpp, logU, Z_Zsun)


def icf_NppNppp_Opp(logU: float, Z_Zsun: float, ne: float = 100.0) -> float:
    """ICF 5: N/O = ((N2+ + N3+)/O2+) x ICF.

    Parameters
    ----------
    logU : float
        Ionisation parameter log(U).
    Z_Zsun : float
        Gas-phase metallicity in solar units.
    ne : float
        Electron density in cm^-3 (default 100).

    Returns
    -------
    float
        ICF value (multiplicative correction to (N2++N3+)/O2+).

    Notes
    -----
    Pure UV diagnostic.  Most robust at log(U) > -2.75 with
    corrections < 5%.  **Recommended** when both NIII] and NIV]
    are detected.  Martinez+2025 Table 4, row 5.
    """
    _check_logU_range(logU)
    return _interp_ne(ne, _COEFF_ICF_NppNppp_Opp, logU, Z_Zsun)


# =========================================================================
# Orchestrator: compute N/O using the best available ICF
# =========================================================================

def compute_NO_martinez25(
    ionic: dict[str, float],
    logU: float,
    Z_Zsun: float,
    ne: float = 100.0,
) -> tuple[float, str] | tuple[None, None]:
    """Compute total N/O using the best available Martinez+2025 ICF.

    Selects the ICF based on which ionic abundances are present,
    with the following priority (most to least preferred):

    1. ICF 5: (N2+ + N3+)/O2+  — pure UV, best at high logU
    2. ICF 4: (N+ + N2+)/(O+ + O2+) — mixed UV+optical
    3. ICF 2: N2+/O2+ — UV only, single ion
    4. ICF 1: N+/O+ — classical optical
    5. ICF 3: N3+/O2+ — large corrections, least preferred

    Parameters
    ----------
    ionic : dict
        Ionic abundance dict, e.g. ``{"O+/H+": val, "O++/H+": val,
        "N+/H+": val, "N++/H+": val, "N+++/H+": val}``.
    logU : float
        Ionisation parameter log(U).
    Z_Zsun : float
        Gas-phase metallicity in solar units.
    ne : float
        Electron density in cm^-3 (default 100).

    Returns
    -------
    tuple
        ``(N/O, icf_name)`` where icf_name identifies the ICF used
        (e.g. ``"NppNppp_Opp"``).  Returns ``(None, None)`` if no
        suitable ionic abundances are available.
    """
    N_plus = ionic.get("N+/H+", 0.0)
    N_pp = ionic.get("N++/H+", 0.0)
    N_ppp = ionic.get("N+++/H+", 0.0)
    O_plus = ionic.get("O+/H+", 0.0)
    O_pp = ionic.get("O++/H+", 0.0)

    # ICF 5: (N2+ + N3+) / O2+
    if N_pp > 0 and N_ppp > 0 and O_pp > 0:
        ratio = (N_pp + N_ppp) / O_pp
        icf = icf_NppNppp_Opp(logU, Z_Zsun, ne)
        return ratio * icf, "NppNppp_Opp"

    # ICF 4: (N+ + N2+) / (O+ + O2+)
    if N_plus > 0 and N_pp > 0 and O_plus > 0 and O_pp > 0:
        ratio = (N_plus + N_pp) / (O_plus + O_pp)
        icf = icf_NpNpp_OpOpp(logU, Z_Zsun, ne)
        return ratio * icf, "NpNpp_OpOpp"

    # ICF 2: N2+ / O2+
    if N_pp > 0 and O_pp > 0:
        ratio = N_pp / O_pp
        icf = icf_NppOpp(logU, Z_Zsun, ne)
        return ratio * icf, "NppOpp"

    # ICF 1: N+ / O+
    if N_plus > 0 and O_plus > 0:
        ratio = N_plus / O_plus
        icf = icf_NpOp(logU, Z_Zsun, ne)
        return ratio * icf, "NpOp"

    # ICF 3: N3+ / O2+ (large corrections, last resort)
    if N_ppp > 0 and O_pp > 0:
        ratio = N_ppp / O_pp
        icf = icf_NpppOpp(logU, Z_Zsun, ne)
        return ratio * icf, "NpppOpp"

    return None, None
