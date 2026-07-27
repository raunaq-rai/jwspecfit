"""Strong-line metallicity calibrations from Sanders et al. (2025).

Simultaneous polynomial fit across available diagnostic ratios
(O3, O2, R23, O32) with Monte Carlo error propagation.

References
----------
Sanders, R. L. et al. 2025, ApJ (submitted).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Reference metallicity for the polynomial expansion.
Z_REF = 8.0
Z_MIN = 6.5
Z_MAX = 9.0

# Sanders+25 calibration coefficients and scatter (Table 3).
#   sigma_R_fit  — rms residual of the polynomial fit (σ_{R,fit})
#   sigma_R_int  — intrinsic scatter in the ratio at fixed Z (σ_{R,int})
#   sigma_OH_int — intrinsic scatter in 12+log(O/H) at fixed R (σ_{O/H,int})
#   sigma_cal    — total ratio-space scatter = sqrt(σ_{R,fit}² + σ_{R,int}²),
#                  used in the chi-squared denominator
CALIBRATIONS: dict[str, dict] = {
    "O3": {
        "coeffs": [0.852, -0.162, -1.149, -0.553],
        "sigma_R_fit": 0.04,
        "sigma_R_int": 0.13,
        "sigma_OH_int": 0.14,
        "sigma_cal": float(np.sqrt(0.04**2 + 0.13**2)),
        "lines": ["OIII_5007", "HBETA"],
    },
    "O2": {
        "coeffs": [0.172, 0.954, -0.832],
        "sigma_R_fit": 0.03,
        "sigma_R_int": 0.25,
        "sigma_OH_int": 0.22,
        "sigma_cal": float(np.sqrt(0.03**2 + 0.25**2)),
        "lines": ["OII_doublet", "HBETA"],
    },
    "R23": {
        "coeffs": [0.998, 0.053, -0.141, -0.493, -0.774],
        "sigma_R_fit": 0.03,
        "sigma_R_int": 0.07,
        "sigma_OH_int": 0.13,
        "sigma_cal": float(np.sqrt(0.03**2 + 0.07**2)),
        "lines": ["OIII_5007", "OIII_4959", "OII_doublet", "HBETA"],
    },
    "O32": {
        "coeffs": [0.697, -1.245, -0.869],
        "sigma_R_fit": 0.09,
        "sigma_R_int": 0.29,
        "sigma_OH_int": 0.25,
        "sigma_cal": float(np.sqrt(0.09**2 + 0.29**2)),
        "lines": ["OIII_5007", "OII_doublet"],
    },
}

# Minimum SNR for a ratio to be used.
SNR_THRESH = 1.5

# Theoretical [O III] lambda4960 / lambda5008 flux ratio, set by the transition
# probabilities alone.  Constant to <0.01% over Te = 1-2.5e4 K and
# ne = 1e2-1e5 cm^-3, so it is safe to use as a fixed scaling when lambda4960
# is not separately measured.
OIII_4959_5007_RATIO = 0.33512


def _poly_eval(z: float, coeffs: list[float]) -> float:
    """Evaluate a polynomial in (z - Z_REF).

    Parameters
    ----------
    z : float
        12 + log(O/H) value.
    coeffs : list of float
        Polynomial coefficients [c0, c1, c2, ...].

    Returns
    -------
    float
        Polynomial value.
    """
    x = z - Z_REF
    return sum(c * x**i for i, c in enumerate(coeffs))


def _log_ratio_err(
    num: float,
    num_err: float,
    den: float,
    den_err: float,
    num2: float = 0.0,
    num2_err: float = 0.0,
) -> tuple[float, float]:
    """Compute log10(ratio) and its error via propagation.

    Parameters
    ----------
    num : float
        Numerator flux.
    num_err : float
        Numerator flux error.
    den : float
        Denominator flux.
    den_err : float
        Denominator flux error.
    num2 : float
        Optional second numerator flux (for sums like O3+O2).
    num2_err : float
        Error on second numerator.

    Returns
    -------
    tuple of float
        ``(log10_ratio, log10_ratio_err)``. Returns ``(nan, nan)`` if
        the ratio is undefined.
    """
    numerator = num + num2
    num_err_tot = np.sqrt(num_err**2 + num2_err**2)

    if numerator <= 0 or den <= 0:
        return np.nan, np.nan

    R = numerator / den
    frac_err = np.sqrt((num_err_tot / numerator) ** 2 + (den_err / den) ** 2)
    log_err = 0.4343 * frac_err  # d(log10)/dx = 1 / (x ln 10)
    return float(np.log10(R)), float(log_err)


def _chi2_simultaneous(
    z: float,
    observed_ratios: dict[str, dict[str, float]],
) -> float:
    """Chi-squared across all available diagnostics.

    Parameters
    ----------
    z : float
        Trial 12+log(O/H).
    observed_ratios : dict
        ``{diag_name: {"val": log_ratio, "err": log_ratio_err}}``.

    Returns
    -------
    float
        Chi-squared value.
    """
    chi2 = 0.0
    for method, data in observed_ratios.items():
        R_obs = data["val"]
        sig_obs = data["err"]
        coeffs = CALIBRATIONS[method]["coeffs"]
        sig_cal = CALIBRATIONS[method]["sigma_cal"]

        R_model = _poly_eval(z, coeffs)
        total_var = sig_obs**2 + sig_cal**2
        chi2 += (R_obs - R_model) ** 2 / total_var
    return chi2


def compute_line_ratios(
    line_fluxes: dict[str, float],
    line_errors: dict[str, float],
    snr_thresh: float = SNR_THRESH,
) -> dict[str, dict[str, float]]:
    """Compute available strong-line diagnostic ratios.

    Parameters
    ----------
    line_fluxes : dict
        ``{line_name: flux}``.
    line_errors : dict
        ``{line_name: flux_err}``.
    snr_thresh : float
        Minimum SNR for a line to be used (default 1.5).

    Returns
    -------
    dict
        ``{ratio_name: {"val": log10_ratio, "err": error}}``.

    Notes
    -----
    Ratio definitions follow \\citet{sanders25} exactly:
    ``O3 = [OIII]5008/Hbeta``, ``O2 = [OII]3727,3730/Hbeta``,
    ``R23 = ([OIII]4960,5008 + [OII])/Hbeta`` and
    ``O32 = [OIII]5008/[OII]``.  Note that **R23 alone uses the [O III]
    doublet sum**; O3 and O32 use lambda5008 by itself.  When ``OIII_4959``
    is absent or below *snr_thresh*, the lambda4960 contribution is inferred
    from lambda5008 via :data:`OIII_4959_5007_RATIO`.
    """

    def _snr(name: str) -> float:
        f = line_fluxes.get(name, 0.0)
        e = line_errors.get(name, np.inf)
        return f / e if e > 0 else 0.0

    def _f(name: str) -> float:
        return line_fluxes.get(name, 0.0)

    def _e(name: str) -> float:
        return line_errors.get(name, 0.0)

    ratios: dict[str, dict[str, float]] = {}

    F_Hb = _f("HBETA")
    e_Hb = _e("HBETA")
    F_O3 = _f("OIII_5007")
    e_O3 = _e("OIII_5007")

    # [O III] doublet sum, used by R23 only.  Sanders+25 define R23 with
    # lambdalambda4960,5008 (their Eq. 3) whereas O3 and O32 use lambda5008
    # alone (Eqs. 1 and 4), so the two must not share a numerator.  Prefer the
    # measured lambda4960 (the narrow-line amplitudes are fitted freely, so it
    # is an independent constraint); fall back to the fixed transition-
    # probability ratio when lambda4960 is missing or below the SNR cut.
    if _snr("OIII_4959") > snr_thresh:
        F_O3sum = F_O3 + _f("OIII_4959")
        e_O3sum = np.sqrt(e_O3**2 + _e("OIII_4959") ** 2)
    else:
        F_O3sum = F_O3 * (1.0 + OIII_4959_5007_RATIO)
        e_O3sum = e_O3 * (1.0 + OIII_4959_5007_RATIO)

    # Also accept resolved or unresolved [OII]
    if "OII_doublet" in line_fluxes:
        F_O2 = _f("OII_doublet")
        e_O2 = _e("OII_doublet")
    elif "OII_3726" in line_fluxes and "OII_3729" in line_fluxes:
        F_O2 = _f("OII_3726") + _f("OII_3729")
        e_O2 = np.sqrt(_e("OII_3726") ** 2 + _e("OII_3729") ** 2)
    else:
        F_O2 = 0.0
        e_O2 = 0.0

    snr_Hb = F_Hb / e_Hb if e_Hb > 0 else 0.0
    snr_O3 = F_O3 / e_O3 if e_O3 > 0 else 0.0
    snr_O2 = F_O2 / e_O2 if e_O2 > 0 else 0.0
    has_o2 = snr_O2 > snr_thresh

    # O3 = log10([OIII]5007 / Hbeta)
    if snr_Hb > snr_thresh and snr_O3 > snr_thresh:
        val, err = _log_ratio_err(F_O3, e_O3, F_Hb, e_Hb)
        if np.isfinite(val):
            ratios["O3"] = {"val": val, "err": err}

    # O2 = log10([OII] / Hbeta)
    if snr_Hb > snr_thresh and has_o2:
        val, err = _log_ratio_err(F_O2, e_O2, F_Hb, e_Hb)
        if np.isfinite(val):
            ratios["O2"] = {"val": val, "err": err}

    # R23 = log10(([OIII]4959,5007 + [OII]) / Hbeta)
    if snr_Hb > snr_thresh and snr_O3 > snr_thresh and has_o2:
        val, err = _log_ratio_err(F_O3sum, e_O3sum, F_Hb, e_Hb, num2=F_O2, num2_err=e_O2)
        if np.isfinite(val):
            ratios["R23"] = {"val": val, "err": err}

    # O32 = log10([OIII]5007 / [OII])
    if snr_O3 > snr_thresh and has_o2:
        val, err = _log_ratio_err(F_O3, e_O3, F_O2, e_O2)
        if np.isfinite(val):
            ratios["O32"] = {"val": val, "err": err}

    return ratios


def sanders25_metallicity(
    line_fluxes: dict[str, float],
    line_errors: dict[str, float],
    n_mc: int = 1000,
    snr_thresh: float = SNR_THRESH,
    seed: int = 42,
    progress: bool = True,
) -> tuple[float, float, float, float, list[str], np.ndarray]:
    """Derive 12+log(O/H) via simultaneous Sanders+25 calibrations.

    Parameters
    ----------
    line_fluxes : dict
        ``{line_name: flux}``.
    line_errors : dict
        ``{line_name: flux_err}``.
    n_mc : int
        Monte Carlo iterations for error propagation (default 1000).
    snr_thresh : float
        Minimum SNR for a ratio to be used (default 1.5).
    seed : int
        Random seed for reproducibility.
    progress : bool
        Show a ``tqdm`` progress bar (default ``True``).

    Returns
    -------
    tuple
        ``(Z_best, Z_lo, Z_hi, chi2, ratios_used, Z_mc_samples)``
        where Z_best is 12+log(O/H), Z_lo/Z_hi are 16th/84th
        percentiles, chi2 is the best-fit chi-squared, ratios_used
        lists the diagnostics, and Z_mc_samples is the full MC array.

    Raises
    ------
    ValueError
        If no valid diagnostic ratios are available.
    """
    ratios = compute_line_ratios(line_fluxes, line_errors, snr_thresh=snr_thresh)

    if not ratios:
        raise ValueError(
            "No valid strong-line ratios. Need at minimum OIII_5007 + HBETA."
        )

    # Best-fit metallicity via bounded minimisation.
    res = minimize_scalar(
        _chi2_simultaneous,
        bounds=(Z_MIN, Z_MAX),
        args=(ratios,),
        method="bounded",
    )
    Z_best = float(res.x)
    chi2 = float(res.fun)

    # Monte Carlo error propagation.
    rng = np.random.default_rng(seed)
    mc_samples = np.empty(n_mc)
    for i in tqdm(range(n_mc), desc="Strong-line (MC)", disable=not progress):
        perturbed = {}
        for m, dat in ratios.items():
            sig_tot = np.sqrt(dat["err"] ** 2 + CALIBRATIONS[m]["sigma_cal"] ** 2)
            perturbed[m] = {
                "val": rng.normal(dat["val"], sig_tot),
                "err": dat["err"],
            }
        mc_res = minimize_scalar(
            _chi2_simultaneous,
            bounds=(Z_MIN, Z_MAX),
            args=(perturbed,),
            method="bounded",
        )
        mc_samples[i] = mc_res.x

    if n_mc > 0:
        Z_lo = float(np.percentile(mc_samples, 16))
        Z_hi = float(np.percentile(mc_samples, 84))
    else:
        Z_lo = np.nan
        Z_hi = np.nan

    return Z_best, Z_lo, Z_hi, chi2, list(ratios.keys()), mc_samples
