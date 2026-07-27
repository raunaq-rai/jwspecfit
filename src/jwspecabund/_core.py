"""Main orchestrator for abundance calculations.

:func:`compute_abundances` accepts a fit result from ``jwspecfit``
or ``jwspecmcmc`` and returns an :class:`AbundanceResult` with
chemical abundances derived via the direct T_e method or strong-line
calibrations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from tqdm import tqdm

from jwspecfit.fitter import FitResult
from jwspecfit.lines import REST_LINES_A

from .dust import (
    compute_Av_balmer_pair,
    compute_Av_from_balmer,
    compute_Av_multi_balmer,
    compute_lya_escape_fraction,
    compute_lya_escape_fraction_mc,
    dust_correct_fluxes,
)
from .result import AbundanceResult

logger = logging.getLogger(__name__)

# Rejection resampling: when an MC/posterior draw falls outside the
# Martinez+2025 calibration bounds it is rejected and re-drawn so that the
# number of *valid* draws still reaches n_mc / n_posterior.  This caps the
# total attempts at FACTOR x target to guarantee termination when an object
# is centred outside the bounds (acceptance ~0); the shortfall is then
# filled with NaN and a warning is emitted.
_RESAMPLE_MAX_ATTEMPT_FACTOR = 20


def _log10_or_nan(v: float | None) -> float:
    """Return log10(*v*) for a positive finite *v*, else NaN."""
    return float(np.log10(v)) if (v is not None and np.isfinite(v) and v > 0) else np.nan


# Line name to rest wavelength mapping for dust correction.
_LINE_WAVES: dict[str, float] = {
    name: wave for name, wave in REST_LINES_A.items()
}

# Balmer lines whose broad components should be summed with the narrow.
_BALMER_LINES = {"Ha", "HBETA", "HGAMMA", "HDELTA"}

# Lines that must never be SNR-filtered (required for Te computation
# and flux normalisation).
# UV doublet members are also protected: their individual SNR may be low,
# but the summed doublet flux is still useful for ionic abundances.
# Ratio diagnostics (logU, density) have their own completeness/SNR
# guards and do not rely on the per-line SNR filter.
_SNR_PROTECTED = {
    "OIII_4363", "OIII_5007", "OIII_4959", "HBETA",
    # Nitrogen lines used for ionic abundances — gated separately
    # by _gate_nitrogen_ions() when icf_method="direct_sum".
    "NII_6585",
    "NIII_1749", "NIII_1752",
    "NIV_1483", "NIV_1486",
    "NV_1", "NV_2",
    "CIV_1", "CIV_2",
    "CIII]_1907", "CIII]",
    "CII]_2324", "CII]_2326",
}


def _extract_fluxes(result: Any) -> tuple[dict[str, float], dict[str, float], bool]:
    """Extract line fluxes and errors from any result type.

    For Balmer lines (Ha, Hb, Hg, Hd), the broad component flux is
    summed with the narrow component so that the total hydrogen flux
    is used for the Balmer decrement and abundance calculations.

    Parameters
    ----------
    result : FitResult | BroadFitResult | MCMCResult | MCMCBroadFitResult
        A fitting result object.

    Returns
    -------
    tuple
        ``(fluxes, errors, is_mcmc)`` where fluxes and errors are
        dicts keyed by line name.
    """
    fluxes = {}
    errors = {}
    is_mcmc = False

    # First pass: extract narrow components.
    for name, lr in result.lines.items():
        if "_BROAD" in name:
            continue
        fluxes[name] = lr.flux
        if isinstance(lr.flux_err, tuple):
            errors[name] = 0.5 * (lr.flux_err[0] + lr.flux_err[1])
            is_mcmc = True
        else:
            errors[name] = lr.flux_err

    # Second pass: add broad Balmer components to narrow totals.
    for name, lr in result.lines.items():
        if "_BROAD" not in name:
            continue
        base_name = name.replace("_BROAD", "")
        if base_name not in _BALMER_LINES or base_name not in fluxes:
            continue

        broad_flux = lr.flux
        if isinstance(lr.flux_err, tuple):
            broad_err = 0.5 * (lr.flux_err[0] + lr.flux_err[1])
        else:
            broad_err = lr.flux_err

        fluxes[base_name] += broad_flux
        errors[base_name] = np.sqrt(errors[base_name] ** 2 + broad_err ** 2)
        logger.info(
            "Added broad component to %s: narrow+broad total = %.4e",
            base_name, fluxes[base_name],
        )

    return fluxes, errors, is_mcmc


def _extract_posteriors(result: Any) -> dict[str, np.ndarray]:
    """Extract flux posteriors from an MCMC result.

    For Balmer lines, broad component posteriors are summed with
    narrow posteriors sample-by-sample.

    Parameters
    ----------
    result : MCMCResult | MCMCBroadFitResult
        An MCMC fitting result.

    Returns
    -------
    dict
        ``{line_name: flux_posterior_array}``.
    """
    posteriors = {}
    for name, lr in result.lines.items():
        if "_BROAD" in name:
            continue
        if hasattr(lr, "flux_posterior") and lr.flux_posterior is not None:
            posteriors[name] = lr.flux_posterior

    # Add broad Balmer posteriors sample-by-sample.
    for name, lr in result.lines.items():
        if "_BROAD" not in name:
            continue
        base_name = name.replace("_BROAD", "")
        if base_name not in _BALMER_LINES or base_name not in posteriors:
            continue
        if hasattr(lr, "flux_posterior") and lr.flux_posterior is not None:
            posteriors[base_name] = posteriors[base_name] + lr.flux_posterior

    return posteriors


def _filter_low_snr(
    fluxes: dict[str, float],
    errors: dict[str, float],
    snr_thresh: float,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Remove emission lines below an SNR threshold.

    Lines in ``_SNR_PROTECTED`` are never removed (they are gated
    elsewhere by ``snr_auroral`` or are required for normalisation).

    Parameters
    ----------
    fluxes : dict
        ``{line_name: flux}`` (dust-corrected).
    errors : dict
        ``{line_name: flux_err}`` (dust-corrected).
    snr_thresh : float
        Minimum signal-to-noise ratio.  Lines with
        ``flux / error < snr_thresh`` are removed.

    Returns
    -------
    tuple
        ``(filtered_fluxes, filtered_errors, excluded_lines)`` where
        *excluded_lines* lists the names of removed lines.
    """
    filtered_fluxes: dict[str, float] = {}
    filtered_errors: dict[str, float] = {}
    excluded: list[str] = []

    for name in fluxes:
        if name in _SNR_PROTECTED:
            filtered_fluxes[name] = fluxes[name]
            filtered_errors[name] = errors.get(name, 0.0)
            continue

        err = errors.get(name, 0.0)
        snr = fluxes[name] / err if err > 0 else np.inf
        if snr >= snr_thresh:
            filtered_fluxes[name] = fluxes[name]
            filtered_errors[name] = err
        else:
            excluded.append(name)
            logger.info(
                "Excluding %s: SNR=%.1f < %.1f.",
                name, snr, snr_thresh,
            )

    return filtered_fluxes, filtered_errors, excluded


def _compute_continuum_rms_limits(
    result: Any,
    z: float,
    Av: float | None,
    dust_law: str,
    n_sigma: float = 3.0,
    **dust_kwargs,
) -> dict[str, float]:
    """Compute flux upper limits from continuum RMS for all fitted lines.

    For each line, measures the RMS of the fit residuals in a window
    around the expected line position, then converts to an integrated
    flux upper limit assuming a Gaussian profile with width set by the
    instrumental resolution.

    The limits are returned in dust-corrected units if ``Av`` is given,
    matching the dust-corrected fluxes used downstream.

    Parameters
    ----------
    result : FitResult or MCMCResult
        Fitting result with ``spectrum``, ``residuals``, and ``continuum``.
    z : float
        Source redshift.
    Av : float or None
        V-band attenuation for dust correction.
    dust_law : str
        Dust law name (``"salim"`` or ``"cardelli"``).
    n_sigma : float
        Number of sigma for the upper limit (default 3).

    Returns
    -------
    dict[str, float]
        ``{line_name: flux_upper_limit}`` in dust-corrected f_lam units.
    """
    from jwspecfit.io import _ujy_to_flam
    from jwspecfit.lines import REST_LINES_A
    from jwspecfit.resolution import R_from_pixels, sigma_inst_A

    if not hasattr(result, "spectrum") or result.spectrum is None:
        return {}
    spec = result.spectrum
    if not hasattr(result, "residuals") or result.residuals is None:
        return {}

    wave_A = spec.wave_A
    # Residuals are in µJy; convert to f_lam.
    resid_flam = _ujy_to_flam(result.residuals, spec.wave_um)
    valid = np.isfinite(resid_flam)

    # Instrumental sigma at each wavelength.
    grating = getattr(spec, "grating", None)
    R = getattr(spec, "R", None)
    if grating is None and R is None:
        R = R_from_pixels(spec.wave_um)
    sig_inst = sigma_inst_A(spec.wave_um, grating=grating, R=R)

    limits: dict[str, float] = {}
    for name in result.line_names:
        if "_BROAD" in name:
            continue
        lam_rest = REST_LINES_A.get(name)
        if lam_rest is None:
            continue

        # Use instrumental sigma at the observed wavelength.
        lam_obs = lam_rest * (1.0 + z)
        idx = np.argmin(np.abs(wave_A - lam_obs))
        sig_A = float(sig_inst[idx])

        flux_ul = _continuum_flux_upper_limit(
            wave_A, resid_flam, valid, lam_rest, z, sig_A,
            n_sigma=n_sigma,
        )
        if flux_ul is not None and flux_ul > 0:
            # Apply dust correction to match the corrected flux scale.
            if Av is not None and Av > 0:
                from .dust import cardelli_extinction, salim_attenuation

                wave_arr = np.array([lam_rest])
                if dust_law == "salim":
                    A_lam = salim_attenuation(wave_arr, Av, **dust_kwargs)[0]
                else:
                    A_lam = cardelli_extinction(wave_arr, Av)[0]
                flux_ul *= 10.0 ** (0.4 * A_lam)

            limits[name] = flux_ul

    return limits


def _apply_dust_correction(
    fluxes: dict[str, float],
    errors: dict[str, float],
    Av: float,
    law: str,
    **dust_kwargs,
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply dust correction to fluxes and errors.

    Parameters
    ----------
    fluxes : dict
        ``{line_name: flux}``.
    errors : dict
        ``{line_name: flux_err}``.
    Av : float
        V-band attenuation.
    law : str
        Dust law name.
    **dust_kwargs
        Extra keyword arguments for the dust law.

    Returns
    -------
    tuple
        ``(corrected_fluxes, corrected_errors)``.
    """
    # Build input dict for dust_correct_fluxes: {name: (flux, err, wave)}
    line_data = {}
    for name in fluxes:
        wave = _LINE_WAVES.get(name)
        if wave is None:
            logger.warning("No rest wavelength for %s; skipping dust correction.", name)
            continue
        line_data[name] = (fluxes[name], errors[name], wave)

    corrected = dust_correct_fluxes(line_data, Av, law=law, **dust_kwargs)

    corr_fluxes = {}
    corr_errors = {}
    for name in fluxes:
        if name in corrected:
            corr_fluxes[name] = corrected[name][0]
            corr_errors[name] = corrected[name][1]
        else:
            corr_fluxes[name] = fluxes[name]
            corr_errors[name] = errors[name]

    return corr_fluxes, corr_errors


def _doublet_snr_ok(
    line1: str,
    line2: str,
    fluxes: dict[str, float],
    errors: dict[str, float],
    snr_ne: float,
    *,
    combined: bool = False,
) -> bool:
    """Return True if a doublet passes the SNR gate.

    Parameters
    ----------
    combined : bool
        If ``True``, use the summed doublet flux and quadrature-
        propagated error.  This correctly handles blended doublets
        where individual member SNRs are low due to MCMC amplitude
        degeneracy (e.g. CIII] 1907/1909).  If ``False`` (default),
        require both members individually above *snr_ne*.
    """
    for name in (line1, line2):
        if name not in fluxes or name not in errors:
            return False

    if combined:
        f1, f2 = fluxes[line1], fluxes[line2]
        e1, e2 = errors.get(line1, 0.0), errors.get(line2, 0.0)
        flux_tot = f1 + f2
        err_tot = np.sqrt(e1**2 + e2**2) if (e1 > 0 and e2 > 0) else 0.0
        snr = flux_tot / err_tot if err_tot > 0 else np.inf
        return snr >= snr_ne

    for name in (line1, line2):
        err = errors.get(name, 0.0)
        snr = fluxes[name] / err if err > 0 else np.inf
        if snr < snr_ne:
            return False
    return True


def _compute_ne_arIV(
    fluxes: dict[str, float],
    errors: dict[str, float],
    snr_ne: float,
    te_high: float,
    ne_failures: dict[str, str],
) -> float | None:
    """Solve the high-ionisation density from [Ar IV] 4711/4740.

    Deblends the He I 4714 line out of the fitted ``ArIV_4713`` feature
    using the He I λ4472 anchor and a PyNEB He I emissivity ratio
    (Berg's method).  Requires the He I anchor to be present so the
    contamination can be quantified; otherwise the [Ar IV] density is
    skipped (``None``) and the caller falls back to CIII].

    Returns the [Ar IV] density in cm^-3, or ``None`` if unavailable.
    """
    from .direct import compute_ne_ArIV, heI_4714_over_4472

    if "ArIV_4713" not in fluxes or "ArIV_4741" not in fluxes:
        return None
    if not _doublet_snr_ok("ArIV_4713", "ArIV_4741", fluxes, errors, snr_ne, combined=True):
        logger.warning(
            "n_e(Ar IV) doublet below SNR threshold (%.1f); skipping.", snr_ne,
        )
        return None

    f_hei = fluxes.get("HEI_4472", 0.0)
    if f_hei <= 0:
        # Cannot quantify the He I 4714 contamination without the anchor.
        ne_failures["n_e(Ar IV)"] = (
            "He I 4472 anchor absent; cannot deblend [Ar IV] 4711 — using CIII]"
        )
        return None

    try:
        # He I 4714/4472 ratio is a recombination ratio, only weakly
        # density-dependent; evaluate at a fiducial n_e = 1e4 cm^-3.
        ratio = heI_4714_over_4472(te_high, 1e4)
        f4711 = fluxes["ArIV_4713"] - ratio * f_hei
        if f4711 <= 0:
            ne_failures["n_e(Ar IV)"] = (
                "deblended [Ar IV] 4711 <= 0 (He I-dominated); using CIII]"
            )
            return None
        ne_arIV = compute_ne_ArIV(f4711, fluxes["ArIV_4741"], Te_guess=te_high)
        logger.info(
            "n_e(O++ zone) from [Ar IV] = %.0f cm^-3 "
            "(He I 4714/4472 deblend ratio = %.3f).", ne_arIV, ratio,
        )
        return ne_arIV
    except Exception as exc:
        logger.warning("n_e(Ar IV) failed; using CIII] fallback.")
        ne_failures["n_e(Ar IV)"] = f"PyNEB solve failed: {exc}"
        return None


def _compute_multi_ne(
    fluxes: dict[str, float],
    errors: dict[str, float] | None = None,
    snr_ne: float = 3.0,
    ne_high_max: float = 5e5,
    niv_density_invalid: bool = False,
    z: float = 0.0,
    te_low: float = 1.5e4,
    te_int: float = 1.5e4,
    te_high: float = 1.5e4,
) -> tuple[float, float | None, float, float, dict[str, str]]:
    """Compute multi-zone electron densities (Martinez+2025 / Berg+2025).

    Parameters
    ----------
    fluxes : dict
        Dust-corrected emission-line fluxes.
    errors : dict, optional
        Dust-corrected flux errors.  Required for SNR gating.
    snr_ne : float
        Minimum SNR for both members of a density-sensitive doublet
        (default 3.0).  If either member falls below this, the doublet is
        skipped and the code falls back to the z-dependent default.
    ne_high_max : float
        Maximum allowed high-ionisation electron density in cm^-3.
    niv_density_invalid : bool
        If ``True``, skip the NIV] density solve (ratio out of range).
    z : float
        Source redshift, for the z-dependent density fallbacks
        (:func:`jwspecabund.direct.ne_zone_fallback`).
    te_low, te_int, te_high : float
        Per-zone electron temperatures (K) at which to solve each zone's
        density.  Defaults to the Martinez+2025 fiducial 1.5e4 K; callers
        pass the refined zone temperatures on the second pass.

    Returns
    -------
    tuple
        ``(ne_low, ne_mid, ne_high, ne_Opp, ne_failures)`` in cm^-3.

        - ``ne_low``: from [SII] 6718/6732 or [OII] 3726/3729 (~14 eV);
          z-fallback ``54*(1+z)^1.2``.  Used for O⁺, N⁺, S⁺, C⁺.
        - ``ne_mid``: from CIII] 1907/1909 (~24 eV); ``None`` if
          unavailable.  Used for C²⁺, N²⁺, S²⁺, Ar²⁺.
        - ``ne_high``: from NIV] 1483/1486 (~47 eV), then [Ar IV];
          z-fallback ``5400*(1+z)^1.62``.  Used for N³⁺, N⁴⁺, C³⁺.
        - ``ne_Opp``: O²⁺/Ne²⁺-zone density — [Ar IV] 4711/4740
          (preferred, He I-deblended) → CIII] → z-fallback
          ``1100*(1+z)^1.93``.
        - ``ne_failures``: dict of density-solve failure reasons.
    """
    from .direct import (
        compute_ne,
        compute_ne_CIII,
        compute_ne_NIV,
        compute_ne_SiIII,
        ne_zone_fallback,
    )

    if errors is None:
        errors = {}

    ne_failures: dict[str, str] = {}

    # Low-ionisation zone: [SII] 6718/6732 or [OII] 3726/3729.
    ne_low = ne_zone_fallback("low", z)
    _fb_low = ne_low
    if "SII_6718" in fluxes and "SII_6732" in fluxes:
        if _doublet_snr_ok("SII_6718", "SII_6732", fluxes, errors, snr_ne):
            try:
                ne_low = compute_ne(
                    fluxes["SII_6718"], fluxes["SII_6732"],
                    doublet="SII", Te_guess=te_low,
                )
            except Exception as exc:
                logger.warning("n_e(SII) failed; using %.0f cm^-3.", _fb_low)
                ne_failures["n_e(SII)"] = f"PyNEB solve failed: {exc}"
        else:
            logger.warning(
                "n_e(SII) doublet below SNR threshold (%.1f); "
                "using z-fallback %.0f cm^-3.", snr_ne, _fb_low,
            )
    elif "OII_3726" in fluxes and "OII_3729" in fluxes:
        if _doublet_snr_ok("OII_3726", "OII_3729", fluxes, errors, snr_ne):
            try:
                ne_low = compute_ne(
                    fluxes["OII_3726"], fluxes["OII_3729"],
                    doublet="OII", Te_guess=te_low,
                )
            except Exception as exc:
                logger.warning("n_e(OII) failed; using %.0f cm^-3.", _fb_low)
                ne_failures["n_e(OII)"] = f"PyNEB solve failed: {exc}"
        else:
            logger.warning(
                "n_e(OII) doublet below SNR threshold (%.1f); "
                "using z-fallback %.0f cm^-3.", snr_ne, _fb_low,
            )
    elif "SiIII_1" in fluxes and "SiIII_2" in fluxes:
        # UV fallback: optical [SII]/[OII] not covered (e.g. high-z UV
        # stacks).  [Si III] 1883/1892 is the bluest low-ionisation
        # density diagnostic (Martinez+2025 Table 2 n_e(Si²⁺)).
        if _doublet_snr_ok("SiIII_1", "SiIII_2", fluxes, errors, snr_ne, combined=True):
            try:
                ne_low = compute_ne_SiIII(
                    fluxes["SiIII_1"], fluxes["SiIII_2"], Te_guess=te_low,
                )
                logger.info(
                    "n_e(low) from [Si III] (UV fallback) = %.0f cm^-3.", ne_low,
                )
            except Exception as exc:
                logger.warning("n_e([Si III]) failed; using %.0f cm^-3.", _fb_low)
                ne_failures["n_e([Si III])"] = f"PyNEB solve failed: {exc}"
        else:
            logger.warning(
                "n_e([Si III]) doublet below SNR threshold (%.1f); "
                "using z-fallback %.0f cm^-3.", snr_ne, _fb_low,
            )

    # Mid-ionisation zone: CIII] 1907/1909 (~24 eV).
    ne_mid = None
    if "CIII]_1907" in fluxes and "CIII]" in fluxes:
        if _doublet_snr_ok("CIII]_1907", "CIII]", fluxes, errors, snr_ne, combined=True):
            try:
                ne_mid = compute_ne_CIII(
                    fluxes["CIII]_1907"], fluxes["CIII]"], Te_guess=te_int,
                )
                logger.info("n_e(mid) from CIII] = %.0f cm^-3.", ne_mid)
            except Exception as exc:
                logger.warning("n_e(CIII]) failed.")
                ne_failures["n_e(CIII])"] = f"PyNEB solve failed: {exc}"
        else:
            logger.warning(
                "n_e(CIII]) doublet below SNR threshold (%.1f); skipping.",
                snr_ne,
            )

    # High-ionisation density from [Ar IV] (preferred for the O²⁺ zone).
    ne_arIV = _compute_ne_arIV(fluxes, errors, snr_ne, te_high, ne_failures)

    # High-ionisation zone: NIV] 1483/1486 (~47 eV).
    ne_high_raw = None
    if niv_density_invalid:
        # The NIV] doublet ratio is outside the physical range, so it cannot
        # yield a density — but its summed flux is retained upstream for the
        # N³⁺/H⁺ abundance.  Skip the density solve and fall back.
        ne_failures["n_e(NIV])"] = (
            "NIV] ratio outside physical range; density from fallback zone"
        )
    elif "NIV_1483" in fluxes and "NIV_1486" in fluxes:
        if _doublet_snr_ok("NIV_1483", "NIV_1486", fluxes, errors, snr_ne, combined=True):
            try:
                ne_high_raw = compute_ne_NIV(
                    fluxes["NIV_1483"], fluxes["NIV_1486"], Te_guess=te_high,
                )
                logger.info("n_e(high) from NIV] = %.0f cm^-3.", ne_high_raw)
            except Exception as exc:
                logger.warning("n_e(NIV]) failed.")
                ne_failures["n_e(NIV])"] = f"PyNEB solve failed: {exc}"
        else:
            logger.warning(
                "n_e(NIV]) doublet below SNR threshold (%.1f); skipping.",
                snr_ne,
            )

    # High-zone density: NIV] -> [Ar IV] -> z-fallback("high").
    if ne_high_raw is not None:
        ne_high = ne_high_raw
    elif ne_arIV is not None:
        ne_high = ne_arIV
    else:
        ne_high = ne_zone_fallback("high", z)

    # Clamp ne_high if it exceeds the maximum (noisy doublet ratio).
    if ne_high > ne_high_max:
        _clamp = ne_mid if ne_mid is not None else ne_low
        logger.warning(
            "n_e(high) = %.0f cm^-3 exceeds ne_high_max=%.0f; "
            "falling back to %.0f cm^-3.", ne_high, ne_high_max, _clamp,
        )
        ne_high = _clamp

    # O²⁺/Ne²⁺-zone density: [Ar IV] (preferred) -> CIII] -> z-fallback("mid").
    if ne_arIV is not None:
        ne_Opp = ne_arIV
    elif ne_mid is not None:
        ne_Opp = ne_mid
    else:
        ne_Opp = ne_zone_fallback("mid", z)

    return ne_low, ne_mid, ne_high, ne_Opp, ne_failures


def _solve_Te_high(
    fluxes: dict[str, float], ne: float,
) -> tuple[float | None, str | None]:
    """Solve T_e(O++) from [OIII] 4363 (preferred) or O III] 1666.

    Returns ``(Te_high, diagnostic)`` where *diagnostic* is ``"4363"`` or
    ``"1666"``.  Returns ``(None, None)`` when neither auroral line is
    present.  Propagates ``ValueError`` from PyNEB if a present line ratio
    cannot be solved.
    """
    from .direct import compute_Te_OIII, compute_Te_OIII_1666

    f4363 = fluxes.get("OIII_4363", 0.0)
    f5007 = fluxes.get("OIII_5007", 0.0)
    f4959 = fluxes.get("OIII_4959", 0.0)
    f1666 = fluxes.get("OIII_1666", 0.0)
    if f4363 > 0 and f5007 > 0:
        return compute_Te_OIII(f4363, f5007, f4959, ne), "4363"
    if f1666 > 0 and f5007 > 0:
        return compute_Te_OIII_1666(f1666, f5007, f4959, ne), "1666"
    return None, None


def _solve_densities_refined(
    fluxes: dict[str, float],
    errors: dict[str, float],
    z: float,
    Te_relation: str,
    snr_ne: float,
    ne_high_max: float,
    niv_rejected: bool,
    ne_low_override: float | None = None,
    ne_mid_override: float | None = None,
    ne_high_override: float | None = None,
) -> tuple[
    float, float | None, float, float,
    float | None, float | None, float | None, str | None,
    dict[str, str],
]:
    """Solve zone densities and temperatures with one refinement iteration.

    Density depends only weakly on T_e (n_e ∝ T_e^0.5), so the densities
    are first solved at the Martinez+2025 fiducial 1.5e4 K, the zone
    temperatures are derived from T_e(O++), and the densities are then
    re-solved once at those zone temperatures.  T_e(O++) is finally
    recomputed at the refined O²⁺-zone density.

    Returns ``(ne_low, ne_mid, ne_high, ne_Opp, Te_high, Te_int, Te_low,
    Te_diagnostic, ne_failures)``.  The temperatures are ``None`` when
    neither [OIII] 4363 nor O III] 1666 is available.  Propagates
    ``ValueError`` from PyNEB on a failed (present) line solve.
    """
    from .direct import Te_int_from_high, Te_low_from_high

    def _apply_overrides(nl, nm, nh, nopp):
        if ne_low_override is not None:
            nl = ne_low_override
        if ne_mid_override is not None:
            # The mid override also sets the O²⁺-zone density (the previous
            # behaviour where O²⁺ tracked n_e(mid)).
            nm = ne_mid_override
            nopp = ne_mid_override
        if ne_high_override is not None:
            nh = ne_high_override
        return nl, nm, nh, nopp

    # First pass at the fiducial temperature.
    nl, nm, nh, nopp, fail = _compute_multi_ne(
        fluxes, errors=errors, snr_ne=snr_ne, ne_high_max=ne_high_max,
        niv_density_invalid=niv_rejected, z=z,
    )
    nl, nm, nh, nopp = _apply_overrides(nl, nm, nh, nopp)

    Te_high, diag = _solve_Te_high(fluxes, nopp)
    if Te_high is None:
        return nl, nm, nh, nopp, None, None, None, None, fail

    Te_int = Te_int_from_high(Te_high, Te_relation)
    Te_low = Te_low_from_high(Te_high, Te_relation)

    # Second pass: re-solve densities at the refined zone temperatures.
    nl, nm, nh, nopp, fail = _compute_multi_ne(
        fluxes, errors=errors, snr_ne=snr_ne, ne_high_max=ne_high_max,
        niv_density_invalid=niv_rejected, z=z,
        te_low=Te_low, te_int=(Te_int if Te_int is not None else 1.5e4),
        te_high=Te_high,
    )
    nl, nm, nh, nopp = _apply_overrides(nl, nm, nh, nopp)

    # Recompute T_e(O++) at the refined O²⁺-zone density.
    Te_high2, diag2 = _solve_Te_high(fluxes, nopp)
    if Te_high2 is not None:
        Te_high, diag = Te_high2, diag2
        Te_int = Te_int_from_high(Te_high, Te_relation)
        Te_low = Te_low_from_high(Te_high, Te_relation)

    return nl, nm, nh, nopp, Te_high, Te_int, Te_low, diag, fail


def _ions_from_incomplete_doublets(fluxes: dict[str, float]) -> set[str]:
    """Return ionic species derived from incomplete UV doublets.

    When a UV doublet has only one member present (e.g. after SNR
    filtering removes the other), the single-member ionic abundance
    is still physically valid, but should *not* be fed into ICF
    computations.  The ICF corrections (especially Martinez+25) are
    calibrated assuming reliable doublet measurements; using a
    single-member abundance with logU from a fallback diagnostic
    (O32 instead of N43) can produce severely biased N/O or C/O.

    These ions are kept in the result's ``ionic`` dict for display,
    but excluded from the dict passed to ``compute_total_abundances``.

    Parameters
    ----------
    fluxes : dict
        Emission-line flux dict (dust-corrected).

    Returns
    -------
    set of str
        Ionic species keys to exclude from ICF computation,
        e.g. ``{"N++/H+", "N+++/H+"}``.
    """
    exclude: set[str] = set()

    _uv_doublets = [
        (("NIII_1749", "NIII_1752"), "N++/H+"),
        (("NIV_1483", "NIV_1486"), "N+++/H+"),
        (("NV_1", "NV_2"), "N4+/H+"),
        (("CIV_1", "CIV_2"), "C+++/H+"),
        (("CII]_2324", "CII]_2326"), "C+/H+"),
    ]
    for (name_a, name_b), ion_key in _uv_doublets:
        has_a = fluxes.get(name_a, 0.0) > 0
        has_b = fluxes.get(name_b, 0.0) > 0
        if has_a != has_b:  # one but not both
            exclude.add(ion_key)
            logger.info(
                "Incomplete doublet (%s/%s): excluding %s from ICF.",
                name_a, name_b, ion_key,
            )

    return exclude


def _compute_logU(
    fluxes: dict[str, float],
    Z_Zsun: float,
    ne_high: float,
    errors: dict[str, float] | None = None,
    snr_logU: float = 3.0,
    lock_diag: str | None = None,
) -> tuple[float | None, str | None]:
    """Compute ionisation parameter (Berg+2025 step 5).

    Diagnostic priority and density (Martinez+2025)
    -----------------------------------------------
    N43 (NIV]/NIII]) is the preferred log(U) diagnostic: it is
    density-insensitive and traces the high-ionisation zone, so it is
    evaluated at *ne_high*.  When the N43 ratio exceeds the upper
    calibration edge the object is simply more highly ionised than the
    grid (log U > -1.0; Martinez+2025 report this for their high-z
    sample, §6.2), so the ratio is clipped to the edge and the resulting
    high-U value (~ -1.0) is used rather than discarding N43.  O32
    ([OIII]/[OII]) is only a fallback: it is density-*sensitive* and, per
    Martinez+2025 (recommendation 5), is evaluated with the measured
    high-ionisation-zone density *ne_high*; it is unreliable when that
    density is high or unknown, which is why N43 is preferred.

    Parameters
    ----------
    fluxes : dict
        Dust-corrected emission-line fluxes.
    Z_Zsun : float
        Gas-phase metallicity in solar units.
    ne_high : float
        High-ionisation zone electron density in cm^-3.  Used for both
        the N43 and (fallback) O32 diagnostics.
    errors : dict, optional
        Flux errors.  When provided, the **total doublet** SNR
        (summed flux / quadrature-summed error) must be >= *snr_logU*
        for each doublet in N43 to be used.
    snr_logU : float
        Minimum total-doublet SNR for N43 (default 3.0).
    lock_diag : str, optional
        Force a specific diagnostic instead of the N43→O32 priority
        tree.  ``"O32"`` skips N43 entirely; ``"N43"`` uses N43 only and
        returns ``(None, None)`` rather than falling back to O32 when
        N43 is unavailable or out of range.  Used by the Monte-Carlo
        loops to keep every draw on the same diagnostic as the point
        estimate, so the log(U) posterior is not a silent mixture of
        N43- and O32-based draws.  ``None`` (default) uses the full
        priority tree.

    Returns
    -------
    tuple
        ``(logU, diagnostic)`` where diagnostic is ``"N43"`` or
        ``"O32"``.  Returns ``(None, None)`` if the (locked or
        priority-selected) diagnostic is not available.
    """
    from .martinez25_icf import (
        LOG_OH_SOLAR, _LOG_N43_VALID, _LOG_U_VALID,
        log_U_from_N43, log_U_from_O32,
    )

    def _doublet_ok(name_a: str, name_b: str) -> tuple[bool, float]:
        """Check both members present and total doublet SNR above cut.

        Returns ``(ok, total_flux)``.
        """
        fa = fluxes.get(name_a, 0.0)
        fb = fluxes.get(name_b, 0.0)
        # Both members must be detected (positive flux).
        if fa <= 0 or fb <= 0:
            return False, 0.0
        total = fa + fb
        if errors is not None:
            ea = errors.get(name_a, 0.0)
            eb = errors.get(name_b, 0.0)
            total_err = np.sqrt(ea**2 + eb**2)
            if total_err > 0 and total / total_err < snr_logU:
                return False, 0.0
        return True, total

    # N43 = NIV]1486 / NIII]1750 — density-insensitive, recommended.
    # Total doublet SNR must pass the cut for both NIV] and NIII].
    if lock_diag in (None, "N43"):
        niv_ok, niv_flux = _doublet_ok("NIV_1483", "NIV_1486")
        if not niv_ok and (fluxes.get("NIV_1483", 0) > 0 or fluxes.get("NIV_1486", 0) > 0):
            logger.info("N43: NIV] doublet below total SNR threshold (%.1f); skipping.", snr_logU)

        niii_ok, niii_flux = _doublet_ok("NIII_1749", "NIII_1752")
        if not niii_ok and (fluxes.get("NIII_1749", 0) > 0 or fluxes.get("NIII_1752", 0) > 0):
            logger.info("N43: NIII] doublet below total SNR threshold (%.1f); skipping.", snr_logU)

        if niv_flux > 0 and niii_flux > 0:
            N43 = niv_flux / niii_flux
            log_N43 = np.log10(N43)
            # A ratio past the UPPER calibration edge means the object is more
            # highly ionised than the Martinez+25 grid (their high-z sample
            # shows this too, indicating log U > -1.0; Martinez+25 §6.2).  Clip
            # to the edge and use the resulting high-ionisation log(U) (~ -1.0)
            # rather than discarding the preferred, density-insensitive N43
            # diagnostic for the density-sensitive O32 fallback.  The
            # (N2+ + N3+)/O2+ ICF is near unity and barely changes for
            # -2 < log U < -1 (Martinez+25 §6.3), so N/O is robust to the
            # exact clipped value.
            if log_N43 > _LOG_N43_VALID[1]:
                logger.info(
                    "log(N43)=%.2f exceeds calibration ceiling %.1f "
                    "(log U > -1.0); clipping to the high-ionisation edge.",
                    log_N43, _LOG_N43_VALID[1],
                )
                log_N43 = _LOG_N43_VALID[1]
            if _LOG_N43_VALID[0] <= log_N43 <= _LOG_N43_VALID[1]:
                logU = log_U_from_N43(log_N43, Z_Zsun, ne_high)
                if _LOG_U_VALID[0] <= logU <= _LOG_U_VALID[1]:
                    logger.info("log(U) from N43 = %.2f (N43=%.3f).", logU, N43)
                    return logU, "N43"
                else:
                    logger.debug(
                        "N43 gives log(U)=%.2f outside validity [%.1f, %.1f].",
                        logU, *_LOG_U_VALID,
                    )
            else:
                logger.debug(
                    "log(N43)=%.2f below validity [%.1f, %.1f].",
                    log_N43, *_LOG_N43_VALID,
                )

        if lock_diag == "N43":
            # Locked to N43 (matching the point estimate) but it was
            # unavailable or out of range for this draw — do NOT fall back
            # to O32.  The caller records this draw as NaN so the log(U)
            # posterior stays a pure-N43 distribution.
            return None, None

    # O32 = [OIII]5007 / [OII]3727 — density-sensitive fallback.
    if lock_diag in (None, "O32"):
        oiii = fluxes.get("OIII_5007", 0.0)
        oii = 0.0
        if "OII_3726" in fluxes and "OII_3729" in fluxes:
            oii = fluxes["OII_3726"] + fluxes["OII_3729"]
        elif "OII_doublet" in fluxes:
            oii = fluxes["OII_doublet"]

        if oiii > 0 and oii > 0:
            O32 = oiii / oii
            # O32 is density-sensitive; Martinez+25 (recommendation 5)
            # evaluate it with the measured high-ionisation-zone density
            # (ne_high).  It is only a fallback and is unreliable when
            # ne_high is high/unknown.
            logU = log_U_from_O32(np.log10(O32), Z_Zsun, ne_high)
            logger.info("log(U) from O32 = %.2f (O32=%.3f, ne=%.0f cm^-3).",
                        logU, O32, ne_high)
            return logU, "O32"

    return None, None


# ---------------------------------------------------------------------------
# Nitrogen-ion SNR gating for ``direct_sum``
# ---------------------------------------------------------------------------

# Mapping from ionic-abundance key to the flux line(s) that produce it.
_N_ION_LINES: dict[str, list[str]] = {
    "N+/H+":   ["NII_6585"],
    "N++/H+":  ["NIII_1749", "NIII_1752"],
    "N+++/H+": ["NIV_1483", "NIV_1486"],
    "N4+/H+":  ["NV_1", "NV_2"],
}


def _gate_nitrogen_ions(
    ionic: dict[str, float],
    fluxes: dict[str, float],
    errors: dict[str, float],
    snr_NO: float = 1.5,
) -> dict[str, float]:
    """Remove nitrogen ionic abundances whose source lines are too noisy.

    For **doublets** (NIII], NIV], NV): both members must have positive
    flux, and the total doublet SNR (sum(flux) / sqrt(sum(err²))) must
    be >= *snr_NO*.  If only one member is detected, the ion is excluded
    regardless of SNR.

    For **single lines** (NII 6585): the line SNR must be >= *snr_NO*.

    Parameters
    ----------
    ionic : dict
        Ionic abundances (modified **in-place** and returned).
    fluxes : dict
        Dust-corrected fluxes.
    errors : dict
        Dust-corrected errors.
    snr_NO : float
        Minimum total-line SNR for each nitrogen ion (default 1.5).

    Returns
    -------
    dict
        The (possibly modified) ionic dict.
    """
    if snr_NO <= 0:
        return ionic

    for ion_key, line_names in _N_ION_LINES.items():
        if ion_key not in ionic or ionic[ion_key] <= 0:
            continue

        is_doublet = len(line_names) > 1

        if is_doublet:
            # At least one member must be genuinely detected: positive
            # flux and individual SNR >= 1 (catches machine-zero fluxes
            # like 1e-46 that are technically positive).
            any_ok = False
            for n in line_names:
                f = fluxes.get(n, 0.0)
                e = errors.get(n, 0.0)
                snr_i = f / e if e > 0 else 0.0
                if f > 0 and snr_i >= 1.0:
                    any_ok = True
                    break
            if not any_ok:
                logger.info(
                    "direct_sum: %s no member above SNR=1 (%s); excluding.",
                    ion_key,
                    ", ".join(
                        f"{n} SNR={fluxes.get(n, 0) / errors.get(n, 1):.1f}"
                        for n in line_names
                    ),
                )
                ionic[ion_key] = 0.0
                continue

        # Total SNR check.
        total_flux = sum(fluxes.get(n, 0.0) for n in line_names)
        total_err2 = sum(errors.get(n, 0.0) ** 2 for n in line_names)
        total_err = np.sqrt(total_err2) if total_err2 > 0 else 0.0
        if total_flux <= 0 or total_err <= 0:
            continue
        snr = total_flux / total_err
        if snr < snr_NO:
            logger.info(
                "direct_sum: %s total SNR=%.1f < %.1f; excluding.",
                ion_key, snr, snr_NO,
            )
            ionic[ion_key] = 0.0
    return ionic


def _continuum_flux_upper_limit(
    wave_A: np.ndarray,
    residuals_flam: np.ndarray,
    valid: np.ndarray,
    line_wave_rest_A: float,
    z: float,
    sigma_line_A: float,
    n_sigma: float = 3.0,
) -> float | None:
    """Compute a flux upper limit from the local continuum RMS.

    Parameters
    ----------
    wave_A : np.ndarray
        Observed wavelength array (Angstrom).
    residuals_flam : np.ndarray
        Continuum-and-model-subtracted residuals in f_lam units.
    valid : np.ndarray
        Boolean mask of valid pixels.
    line_wave_rest_A : float
        Rest-frame wavelength of the line (Angstrom).
    z : float
        Source redshift.
    sigma_line_A : float
        Expected Gaussian sigma of the line in Angstrom (from
        instrumental resolution or a detected line).
    n_sigma : float
        Number of sigma for the upper limit (default 3).

    Returns
    -------
    float or None
        Integrated flux upper limit (f_lam × Angstrom), or None if
        there are insufficient pixels.
    """
    lam_obs = line_wave_rest_A * (1.0 + z)
    # Window of ±5σ around the expected line position, excluding the
    # central ±2σ where the line itself would sit.
    near = np.abs(wave_A - lam_obs)
    window = valid & (near < 5.0 * sigma_line_A) & (near > 2.0 * sigma_line_A)
    n_pix = int(np.sum(window))
    if n_pix < 3:
        # Fall back to wider window without central exclusion.
        window = valid & (near < 10.0 * sigma_line_A)
        n_pix = int(np.sum(window))
    if n_pix < 3:
        return None

    rms = float(np.sqrt(np.nanmean(residuals_flam[window] ** 2)))
    # Flux upper limit: n_sigma × RMS × line width (Gaussian integral).
    _SQRT2PI = np.sqrt(2.0 * np.pi)
    return n_sigma * rms * sigma_line_A * _SQRT2PI


def _compute_ionic_upper_limits(
    ionic: dict[str, float],
    fluxes: dict[str, float],
    errors: dict[str, float],
    Te_high: float,
    Te_low: float,
    ne_low: float,
    ne_mid: float,
    ne_high: float,
    n_sigma: float = 3.0,
    continuum_rms_limits: dict[str, float] | None = None,
    ne_Opp: float | None = None,
    Te_int: float | None = None,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Compute n-sigma upper limits for non-detected ionic abundances.

    Parameters
    ----------
    continuum_rms_limits : dict, optional
        Pre-computed continuum-RMS flux upper limits keyed by line name.
        If provided, these are used instead of the fit errors.

    Returns
    -------
    upper_limits : dict[str, float]
        Ion key -> ionic abundance upper limit.
    details : dict[str, dict]
        Ion key -> metadata dict with keys ``lines``, ``flux_ul``,
        ``n_sigma``, ``method``.
    """
    from .direct import _ionic_abundance

    Hb = fluxes.get("HBETA", 0.0)
    if Hb <= 0:
        return {}, {}

    _ul = continuum_rms_limits or {}

    # O²⁺/Ne²⁺ use the O²⁺-zone density ([Ar IV] preferred); fall back to
    # ne_mid for backward compatibility.
    ne_opp = ne_Opp if ne_Opp is not None else ne_mid
    # Intermediate (S²⁺/C²⁺/N²⁺) temperature; fall back to the midpoint.
    Te_mid = Te_int if Te_int is not None else 0.5 * (Te_high + Te_low)

    # Mapping: ion_key -> (element, ion_stage, line_names, wave_labels, Te, ne)
    # C²⁺/N²⁺ use the intermediate temperature Te_mid (Martinez+2025 §5.4).
    _ION_MAP = [
        ("O+/H+",   "O", 2, ["OII_doublet"], [3727],       Te_low,  ne_low),
        ("O++/H+",  "O", 3, ["OIII_5007"],   [5007],       Te_high, ne_opp),
        ("N+/H+",   "N", 2, ["NII_6585"],    [6584],       Te_low,  ne_low),
        ("N++/H+",  "N", 3, ["NIII_1749", "NIII_1752"], [1749, 1752], Te_mid, ne_mid),
        ("N+++/H+", "N", 4, ["NIV_1483", "NIV_1486"],   [1483, 1486], Te_high, ne_high),
        ("C+/H+",   "C", 2, ["CII]_2324", "CII]_2326"], [2323, 2325, 2326, 2327, 2328], Te_low, ne_low),
        ("C++/H+",  "C", 3, ["CIII]_1907", "CIII]"],    [1907, 1909], Te_mid, ne_mid),
        ("C+++/H+", "C", 4, ["CIV_1", "CIV_2"],         [1548, 1551], Te_high, ne_high),
        ("Ne++/H+", "Ne", 3, ["NeIII_3869"], [3869],     Te_high, ne_opp),
        ("S+/H+",   "S", 2, ["SII_6718", "SII_6732"], [6718, 6732], Te_low, ne_low),
    ]

    upper_limits: dict[str, float] = {}
    details: dict[str, dict] = {}
    for ion_key, elem, stage, line_names, waves, Te, ne in _ION_MAP:
        # Only compute upper limit if the ion is not detected.
        if ionic.get(ion_key, 0.0) > 0:
            continue

        # Prefer continuum-RMS flux limits; fall back to fit errors.
        flux_ul = None
        ul_method = "continuum_rms"
        if _ul:
            # Add continuum-RMS limits in quadrature (independent noise
            # in each doublet member: combined 3σ = √(Σ (3σ_i)²)).
            member_uls = [_ul[n] for n in line_names if n in _ul]
            if member_uls:
                flux_ul = np.sqrt(sum(u**2 for u in member_uls))

        if flux_ul is None or flux_ul <= 0:
            # Fall back to fit errors (quadrature sum × n_sigma).
            total_err2 = sum(errors.get(n, 0.0) ** 2 for n in line_names)
            if total_err2 <= 0:
                continue
            flux_ul = n_sigma * np.sqrt(total_err2)
            ul_method = "fit_error"

        wave_arg = waves if len(waves) > 1 else waves[0]
        try:
            abund_ul = _ionic_abundance(elem, stage, flux_ul, Hb, Te, ne, wave_arg)
            if abund_ul > 0 and np.isfinite(abund_ul):
                upper_limits[ion_key] = abund_ul
                details[ion_key] = {
                    "lines": line_names,
                    "flux_ul": flux_ul,
                    "n_sigma": n_sigma,
                    "method": ul_method,
                }
        except Exception:
            pass

    return upper_limits, details


# Human-readable descriptions for Martinez+25 and direct-sum ICF names.
_ICF_DESCRIPTIONS: dict[str, str] = {
    "NppNppp_Opp": "Martinez+25 ICF 5: (N2+ + N3+)/O2+ x ICF — preferred (pure UV, both ions detected)",
    "NpNpp_OpOpp": "Martinez+25 ICF 4: (N+ + N2+)/(O+ + O2+) x ICF — mixed UV+optical",
    "NppOpp": "Martinez+25 ICF 2: N2+/O2+ x ICF — UV only (single N ion)",
    "NpOp": "Martinez+25 ICF 1: N+/O+ x ICF — optical only",
    "NpppOpp": "Martinez+25 ICF 3: N3+/O2+ x ICF — large correction, last resort",
    "Np_Npp_Nppp": "direct sum: (N+ + N2+ + N3+) / (O+ + O2+) — all zones, no ICF needed",
    "Npp_Nppp_Opp": "direct sum: (N2+ + N3+) / O2+ — UV only, no ICF needed",
    "Nppp_Opp": "direct sum: N3+ / O2+ — UV only, no ICF needed",
    "Npp_Opp": "direct sum: N2+ / O2+ — UV only, no ICF needed",
    "izotov06_fallback": "Izotov+06: ICF(O+/O) x N+/O+ — optical fallback",
}

_TE_RELATION_LABELS: dict[str, str] = {
    "desi": "DESI DR2",
    "classical": "classical (Garnett 1992)",
    "garnett": "Garnett (1992)",
    "3_tier": "3-tier Garnett (1992)",
}


def _build_diagnostics(
    fluxes: dict[str, float],
    Te_high: float | None,
    Te_relation: str,
    ne_low: float,
    ne_mid: float | None,
    ne_high: float,
    logU: float | None,
    logU_diag: str | None,
    icf_method: str | None,
    NO_icf_name: str | None,
    ne_default: float,
    totals: dict[str, Any] | None = None,
    niv_rejected: bool = False,
    ne_Opp: float | None = None,
    Te_int: float | None = None,
    te_high_diag: str | None = None,
    z: float = 0.0,
) -> dict[str, str]:
    """Build a diagnostics dict explaining how each quantity was derived.

    Parameters
    ----------
    fluxes : dict
        Dust-corrected emission-line fluxes.
    Te_high : float or None
        High-ionisation electron temperature in K.
    Te_relation : str
        Te-Te relation used (``"desi"`` or ``"classical"``).
    ne_low : float
        Low-ionisation electron density in cm^-3.
    ne_mid : float or None
        Mid-ionisation electron density in cm^-3 (from CIII]).
    ne_high : float
        High-ionisation electron density in cm^-3.
    logU : float or None
        Ionisation parameter log(U).
    logU_diag : str or None
        Diagnostic used for logU (``"N43"`` or ``"O32"``).
    icf_method : str or None
        ICF scheme used.
    NO_icf_name : str or None
        Specific ICF name used for N/O.
    ne_default : float
        Default electron density in cm^-3.
    totals : dict, optional
        Total abundance dict from ``compute_total_abundances()``.

    Returns
    -------
    dict
        Human-readable explanations keyed by quantity name.
    """
    totals = totals or {}
    diag: dict[str, str] = {}

    # Te(high) — solved at the O²⁺-zone density ne_Opp.
    _ne_OIII = ne_Opp if ne_Opp is not None else (
        ne_mid if ne_mid is not None else ne_low
    )
    _has_arIV = "ArIV_4713" in fluxes and "ArIV_4741" in fluxes
    _opp_src = (
        "[Ar IV] 4711/4740 (He I-deblended)"
        if (_has_arIV and ne_Opp is not None and ne_mid is not None
            and abs(ne_Opp - ne_mid) > 1e-6)
        or (_has_arIV and ne_mid is None and ne_Opp is not None)
        else "CIII] 1907/1909 -> z-fallback"
    )
    if Te_high is not None:
        _te_line = (
            "O III] 1666/(5007+4959)" if te_high_diag == "1666"
            else "[OIII] 4363/(5007+4959)"
        )
        # Compact method label (line + temperature) for table display.
        diag["Te(high) method"] = (
            f"{_te_line} -> T_e = {Te_high:.0f} K (PyNEB)"
        )
        diag["Te(high)"] = (
            f"{_te_line} ratio with n_e(O++ zone) = {_ne_OIII:.0f} cm^-3 (PyNEB)"
        )
        diag["O++/H+ density"] = (
            f"O²⁺-zone n_e = {_ne_OIII:.0f} cm^-3 from {_opp_src}; "
            "[Ar IV] (40-60 eV) is the preferred high-ionisation density "
            "(Martinez+2025 Table 2), CIII] the fallback"
        )

    # Te(int) — intermediate (S²⁺) zone, 3-tier Garnett relation.
    if Te_high is not None and Te_int is not None:
        diag["Te(int)"] = (
            f"Garnett (1992) S III relation 0.83*Te(high)+1700 = {Te_int:.0f} K "
            "(S²⁺/Ar²⁺ intermediate zone)"
        )

    # Te(low)
    if Te_high is not None:
        rel_label = _TE_RELATION_LABELS.get(Te_relation, Te_relation)
        diag["Te(low)"] = (
            f"{rel_label} Te-Te relation from Te(high) = {Te_high:.0f} K"
        )

    # ne(low) — measured from a doublet iff ne_low differs from the
    # z-dependent fallback (on a failed/absent solve ne_low IS the fallback).
    from .direct import ne_zone_fallback
    _fb_low = ne_zone_fallback("low", z)
    _has_sii = "SII_6718" in fluxes and "SII_6732" in fluxes
    _has_oii = "OII_3726" in fluxes and "OII_3729" in fluxes
    _has_siiii = "SiIII_1" in fluxes and "SiIII_2" in fluxes
    _measured_low = abs(ne_low - _fb_low) > 1e-6
    if _measured_low:
        if _has_sii:
            diag["ne(low)"] = f"[SII] 6718/6732 doublet ratio -> {ne_low:.0f} cm^-3"
        elif _has_oii:
            diag["ne(low)"] = f"[OII] 3726/3729 doublet ratio -> {ne_low:.0f} cm^-3"
        elif _has_siiii:
            diag["ne(low)"] = (
                f"[Si III] 1883/1892 doublet ratio (UV) -> {ne_low:.0f} cm^-3"
            )
        else:
            diag["ne(low)"] = f"{ne_low:.0f} cm^-3"
    else:
        _why = (
            "[SII] doublet failed SNR cut or solve" if _has_sii else
            "[OII] doublet failed SNR cut or solve" if _has_oii else
            "[Si III] doublet failed SNR cut or solve" if _has_siiii else
            "no [SII]/[OII]/[Si III] doublet available"
        )
        diag["ne(low)"] = (
            f"z-fallback ({_fb_low:.0f} cm^-3, Topping+2025a) — {_why}"
        )

    # ne(mid)
    _has_ciii = "CIII]_1907" in fluxes and "CIII]" in fluxes
    if ne_mid is not None:
        diag["ne(mid)"] = f"CIII] 1907/1909 doublet ratio -> {ne_mid:.0f} cm^-3"
    elif _has_ciii:
        diag["ne(mid)"] = (
            f"fallback to ne(low) = {ne_low:.0f} cm^-3 — CIII] failed SNR cut or solve"
        )
    else:
        diag["ne(mid)"] = (
            f"fallback to ne(low) = {ne_low:.0f} cm^-3 — no CIII] doublet available"
        )

    # ne(high) — NIV] (preferred) -> [Ar IV] -> z-fallback (Martinez+2025).
    _fb_high = ne_zone_fallback("high", z)
    _has_niv = "NIV_1483" in fluxes and "NIV_1486" in fluxes
    _has_arIV = "ArIV_4713" in fluxes and "ArIV_4741" in fluxes
    _is_fb_high = abs(ne_high - _fb_high) <= 1e-6
    if _has_niv and not niv_rejected and not _is_fb_high:
        diag["ne(high)"] = f"NIV] 1483/1486 doublet ratio -> {ne_high:.0f} cm^-3"
    elif _has_arIV and not _is_fb_high and ne_Opp is not None \
            and abs(ne_high - ne_Opp) <= 1e-6:
        diag["ne(high)"] = (
            f"[Ar IV] 4711/4740 (He I-deblended) -> {ne_high:.0f} cm^-3"
        )
    elif _is_fb_high:
        _why = (
            "NIV] rejected (ratio out of range)" if niv_rejected else
            "NIV] failed SNR cut or solve" if _has_niv else
            "no NIV]/[Ar IV] high-ion doublet available"
        )
        diag["ne(high)"] = (
            f"z-fallback ({_fb_high:.0f} cm^-3, Martinez+2025) — {_why}"
        )
    else:
        # Clamped to ne(mid)/ne(low) because the solved value exceeded
        # ne_high_max.
        diag["ne(high)"] = f"clamped to {ne_high:.0f} cm^-3 (exceeded ne_high_max)"

    # log(U)
    if logU is not None:
        if logU_diag == "N43":
            diag["log(U)"] = (
                f"N43 diagnostic (NIV] 1486 / NIII] 1750) -> log(U) = {logU:.2f}"
            )
        elif logU_diag == "O32":
            diag["log(U)"] = (
                f"O32 diagnostic ([OIII] 5007 / [OII] 3727) -> log(U) = {logU:.2f}"
            )
    else:
        diag["log(U)"] = "not available (N43 and O32 diagnostics both unavailable)"

    # N/O ICF
    if NO_icf_name is not None:
        diag["N/O ICF"] = _ICF_DESCRIPTIONS.get(
            NO_icf_name, f"{icf_method}: {NO_icf_name}"
        )
    elif icf_method is not None:
        diag["N/O ICF"] = "N/O could not be computed (no eligible ions)"

    # C/O method
    co_method = totals.get("CO_method")
    if co_method == "direct_sum":
        diag["C/O"] = "direct sum (C⁺ + C²⁺ + C³⁺) / (O⁺ + O²⁺) — CII] detected"
    elif co_method == "garnett97_icf":
        icf_val = totals.get("CO_icf_value", 1.0)
        diag["C/O"] = (
            f"Garnett+1997 ICF × (C²⁺ + C³⁺) / O²⁺ — "
            f"ICF = O_total/O²⁺ = {icf_val:.3f}"
        )
    elif "C/O" not in (totals.get("_failures") or {}):
        pass  # C/O not attempted (no carbon lines)
    else:
        diag["C/O"] = totals.get("_failures", {}).get("C/O", "not computed")

    return diag


def _run_direct(
    fluxes: dict[str, float],
    errors: dict[str, float],
    Te_relation: str,
    n_mc: int,
    seed: int = 42,
    progress: bool = True,
    ne_high_max: float = 5e5,
    snr_ne: float = 3.0,
    snr_logU: float = 1.5,
    marginalize_logU: bool = True,
    icf_method: str = "auto",
    co_icf_method: str = "auto",
    snr_NO: float = 1.5,
    icf_tier: str | None = None,
    continuum_rms_limits: dict[str, float] | None = None,
    niv_rejected: bool = False,
    ne_low_override: float | None = None,
    ne_mid_override: float | None = None,
    ne_high_override: float | None = None,
    z: float = 0.0,
) -> dict[str, Any]:
    """Run the direct T_e method following Berg+2025's 6-step procedure.

    Steps: (1) multi-phase ne, (2) zone-appropriate Te, (3) ionic
    abundances, (4) O/H and Z/Zsun, (5) logU from N43 or O32,
    (6) Martinez+25 ICFs for N/O (fallback: Izotov+06).

    Parameters
    ----------
    fluxes : dict
        Dust-corrected fluxes.
    errors : dict
        Dust-corrected errors.
    Te_relation : str
        T_e-T_e relation (``"desi"`` or ``"classical"``).
    n_mc : int
        Number of MC iterations for error propagation.
    seed : int
        Random seed.
    progress : bool
        Show a ``tqdm`` progress bar (default ``True``).
    ne_high_max : float
        Maximum allowed n_e(high) in cm^-3 (default 5e5).
    snr_ne : float
        Minimum SNR for density-sensitive doublet members (default 3.0).
        Doublets failing this cut are skipped and the default density
        is used instead.
    snr_NO : float
        Minimum total-line SNR for each nitrogen ion when using
        ``icf_method="direct_sum"`` (default 1.5).  Ions whose
        contributing lines fall below this are excluded from the sum.

    Returns
    -------
    dict
        Keys: OH, OH_err, NO, NO_err, Te_high, Te_low, ne, ne_low,
        ne_high, logU, icf_method, ionic, posteriors, etc.
    """
    from .direct import (
        NE_DEFAULT,
        Te_int_from_high,
        Te_low_from_high,
        compute_ionic_abundances,
        compute_Te_OIII,
        compute_Te_OIII_1666,
        compute_total_abundances,
    )
    from .martinez25_icf import LOG_OH_SOLAR, _LOG_U_VALID

    # --- Steps 1-2: Multi-phase electron density + zone temperatures ---
    # Densities are solved at the fiducial 1.5e4 K, the zone temperatures
    # are derived from T_e(O++), and the densities are re-solved once at
    # those zone temperatures (n_e depends only weakly on T_e).  T_e(O++)
    # is evaluated at the O²⁺-zone density ne_Opp ([Ar IV] preferred, else
    # CIII] -> z-fallback).
    (
        ne_low, ne_mid, ne_high, ne_Opp,
        Te_high, Te_int, Te_low, _Te_diagnostic, ne_failures,
    ) = _solve_densities_refined(
        fluxes, errors, z, Te_relation, snr_ne, ne_high_max, niv_rejected,
        ne_low_override=ne_low_override,
        ne_mid_override=ne_mid_override,
        ne_high_override=ne_high_override,
    )
    ne_OIII = ne_Opp
    if Te_high is None:
        raise ValueError(
            "Neither [OIII] 4363 nor O III] 1666 available for T_e computation."
        )
    if _Te_diagnostic == "1666":
        logger.info(
            "[OIII] 4363 not available; using O III] 1666/(5007+4959) "
            "for T_e(high) = %.0f K.", Te_high,
        )
    f_4363 = fluxes.get("OIII_4363", 0.0)
    f_5007 = fluxes.get("OIII_5007", 0.0)
    f_4959 = fluxes.get("OIII_4959", 0.0)
    f_1666 = fluxes.get("OIII_1666", 0.0)

    # --- Step 3: Ionic abundances with zone-appropriate ne and Te ---
    ionic = compute_ionic_abundances(
        fluxes, Te_high, Te_low, ne_low, ne_mid=ne_mid, ne_high=ne_high,
        Te_int=Te_int, ne_Opp=ne_Opp,
    )

    # --- Step 4: O/H and Z/Zsun ---
    OH = (ionic.get("O+/H+", 0.0) + ionic.get("O++/H+", 0.0))
    if OH > 0:
        OH_12 = 12.0 + np.log10(OH)
        Z_Zsun = 10.0 ** (OH_12 - LOG_OH_SOLAR)
    else:
        OH_12 = np.nan
        Z_Zsun = None

    # --- Step 5: Ionisation parameter ---
    logU = None
    logU_diag = None
    logU_co = None
    if Z_Zsun is not None:
        logU, logU_diag = _compute_logU(
            fluxes, Z_Zsun, ne_high, errors=errors, snr_logU=snr_logU,
        )
        # C/O ICF: recompute log U at the intermediate (C2+) zone density.
        # Only the density-sensitive O32 diagnostic changes; N43 is
        # density-insensitive so this is a no-op there.
        logU_co = logU
        if logU_diag is not None and ne_mid is not None:
            logU_co, _ = _compute_logU(
                fluxes, Z_Zsun, ne_mid, errors=errors, snr_logU=snr_logU,
                lock_diag=logU_diag,
            )

    # --- Step 6: Total abundances with ICFs ---
    # SNR-gate nitrogen ions to avoid noise-dominated N/O.
    _gate_nitrogen_ions(ionic, fluxes, errors, snr_NO=snr_NO)

    # Compute 3σ upper limits for non-detected ions.
    ionic_upper_limits, ionic_ul_details = _compute_ionic_upper_limits(
        ionic, fluxes, errors, Te_high, Te_low, ne_low,
        ne_mid if ne_mid is not None else ne_low,
        ne_high if ne_high is not None else ne_low,
        continuum_rms_limits=continuum_rms_limits,
        ne_Opp=ne_Opp, Te_int=Te_int,
    )

    totals = compute_total_abundances(
        ionic, logU=logU, Z_Zsun=Z_Zsun, ne=ne_high,
        icf_method=icf_method,
        co_icf_method=co_icf_method,
        co_logU=logU_co, co_ne=ne_mid,
        ionic_upper_limits=ionic_upper_limits,
        _lock_NO_icf=icf_tier,
    )

    NO = totals.get("N/O")
    NO_log = np.log10(NO) if NO is not None and NO > 0 else None

    SO = totals.get("S/O")
    SO_log = np.log10(SO) if SO is not None and SO > 0 else None

    NeO = totals.get("Ne/O")
    NeO_log = np.log10(NeO) if NeO is not None and NeO > 0 else None

    ArO = totals.get("Ar/O")
    ArO_log = np.log10(ArO) if ArO is not None and ArO > 0 else None

    CO = totals.get("C/O")
    CO_log = np.log10(CO) if CO is not None and CO > 0 else None

    icf_method = totals.get("icf_method")
    NO_icf_name = totals.get("NO_icf_name")
    NO_is_upper_limit = totals.pop("NO_is_upper_limit", False)
    failures = totals.pop("_failures", {})
    failures.update(ne_failures)
    NO_tiers = totals.pop("_NO_tiers", None)
    icf_values = totals.pop("_icf_values", None)

    # --- Build diagnostics dict ---
    diagnostics = _build_diagnostics(
        fluxes, Te_high, Te_relation, ne_low, ne_mid, ne_high,
        logU, logU_diag, icf_method, NO_icf_name, NE_DEFAULT,
        totals=totals, niv_rejected=niv_rejected,
        ne_Opp=ne_Opp, Te_int=Te_int, te_high_diag=_Te_diagnostic, z=z,
    )

    # --- MC error propagation (all 6 steps per iteration) ---
    rng = np.random.default_rng(seed)
    OH_mc = []
    NO_mc = []
    CO_mc = []
    SO_mc = []
    NeO_mc = []
    ArO_mc = []
    Te_high_mc = []
    Te_low_mc = []
    logU_mc_arr = []
    # Collect per-tier N/O posteriors for uncertainty on each method.
    _tier_keys = [k for k in (NO_tiers or {}) if not k.startswith("_")]
    NO_tier_mc: dict[str, list[float]] = {k: [] for k in _tier_keys}

    # Resampling: keep drawing until n_mc in-bounds N/O draws are collected
    # (capped so an out-of-bounds object can't hang).  Only N/O is gated on
    # the Martinez+2025 bounds; O/H, Te and the other X/O ratios do not use
    # the Martinez ICF, so they are recorded for *every* drawn sample and
    # their posteriors therefore hold >= n_mc finite draws.
    max_attempts = n_mc * _RESAMPLE_MAX_ATTEMPT_FACTOR
    attempts = 0
    n_collected = 0  # draws counted toward n_mc (in-bounds N/O or solver fail)
    _pbar = tqdm(total=n_mc, desc="Direct Te (MC)", disable=not progress)
    while n_collected < n_mc and attempts < max_attempts:
        attempts += 1
        mc_fluxes = {}
        for name in fluxes:
            mc_fluxes[name] = rng.normal(fluxes[name], errors.get(name, 0.0))
            mc_fluxes[name] = max(mc_fluxes[name], 1e-50)

        try:
            # Use fixed ne (varying ne per MC iteration adds noise
            # without improving accuracy for the density diagnostics).
            # Use the same Te diagnostic as the point estimate.
            if _Te_diagnostic == "4363":
                Te_h = compute_Te_OIII(
                    mc_fluxes.get("OIII_4363", 0),
                    mc_fluxes.get("OIII_5007", 0),
                    mc_fluxes.get("OIII_4959", 0),
                    ne_OIII,
                )
            else:
                Te_h = compute_Te_OIII_1666(
                    mc_fluxes.get("OIII_1666", 0),
                    mc_fluxes.get("OIII_5007", 0),
                    mc_fluxes.get("OIII_4959", 0),
                    ne_OIII,
                )
            Te_l = Te_low_from_high(Te_h, relation=Te_relation)
            Te_i = Te_int_from_high(Te_h, relation=Te_relation)
            ionic_mc = compute_ionic_abundances(
                mc_fluxes, Te_h, Te_l, ne_low, ne_mid=ne_mid, ne_high=ne_high,
                Te_int=Te_i, ne_Opp=ne_Opp,
            )

            # Compute Z_Zsun for this MC iteration.
            oh_val = ionic_mc.get("O+/H+", 0.0) + ionic_mc.get("O++/H+", 0.0)
            if oh_val > 0:
                z_zsun_mc = 10.0 ** (12.0 + np.log10(oh_val) - LOG_OH_SOLAR)
            else:
                z_zsun_mc = Z_Zsun  # fallback to point estimate

            # Compute logU for this MC iteration.  If the O32/N43/Z inputs
            # or the resulting logU fall outside the Martinez+2025 bounds,
            # logU is set to NaN so the Martinez N/O ICF returns NaN for
            # this draw (it does not count toward n_mc).  O/H, Te and the
            # other ratios do not use logU and are still recorded below.
            # Pass original errors so the same SNR gating applies as
            # for the point estimate, and lock the diagnostic to the one
            # the point estimate selected so every draw uses N43 (or every
            # draw uses O32) — the posterior is never a silent mixture.
            # Resample log(U) for this draw so the reported logU posterior
            # carries its measurement uncertainty.  marginalize_logU sets
            # whether that scatter propagates into the ICF-based ratios: when
            # False the ICFs use the fixed point-estimate logU (the analogue
            # of a fixed A_V applied to every draw), so logU does not inflate
            # N/O — but logU still gets a reported error bar.
            logU_sample = logU
            no_in_bounds = True
            if z_zsun_mc is not None and logU_diag is not None:
                logU_mc_val, _ = _compute_logU(
                    mc_fluxes, z_zsun_mc, ne_high, errors=errors,
                    snr_logU=snr_logU, lock_diag=logU_diag,
                )
                if (logU_mc_val is not None and np.isfinite(logU_mc_val)
                        and _LOG_U_VALID[0] <= logU_mc_val <= _LOG_U_VALID[1]):
                    logU_sample = float(logU_mc_val)
                elif marginalize_logU:
                    # logU drives the ICF when marginalising: reject N/O for
                    # this out-of-range draw (O/H etc. still kept).
                    logU_sample = np.nan
                    no_in_bounds = False
            logU_icf = logU_sample if marginalize_logU else logU

            # C/O log U at the intermediate (C2+) zone density.
            logU_co_sample = logU_co
            if (marginalize_logU and z_zsun_mc is not None
                    and logU_diag is not None and ne_mid is not None):
                logU_co_mc, _ = _compute_logU(
                    mc_fluxes, z_zsun_mc, ne_mid, errors=errors,
                    snr_logU=snr_logU, lock_diag=logU_diag,
                )
                if logU_co_mc is not None and np.isfinite(logU_co_mc):
                    logU_co_sample = float(logU_co_mc)
            logU_co_icf = logU_co_sample if marginalize_logU else logU_co

            Te_high_mc.append(Te_h)
            Te_low_mc.append(Te_l)
            logU_mc_arr.append(logU_sample if logU_sample is not None else np.nan)

            # Gate nitrogen ions using the *original* errors so the
            # same ions are included/excluded as in the point estimate.
            _gate_nitrogen_ions(ionic_mc, mc_fluxes, errors, snr_NO=snr_NO)

            totals_mc = compute_total_abundances(
                ionic_mc, logU=logU_icf, Z_Zsun=z_zsun_mc, ne=ne_high,
                icf_method=icf_method,
                co_icf_method=co_icf_method,
                co_logU=logU_co_icf, co_ne=ne_mid,
                _lock_NO_icf=NO_icf_name,
            )

            oh_mc = totals_mc.get("O/H", np.nan)
            if np.isfinite(oh_mc) and oh_mc > 0:
                OH_mc.append(12.0 + np.log10(oh_mc))
            else:
                OH_mc.append(np.nan)

            no_mc = totals_mc.get("N/O", np.nan)
            if no_mc is not None and np.isfinite(no_mc) and no_mc > 0:
                NO_mc.append(np.log10(no_mc))
            else:
                NO_mc.append(np.nan)

            co_mc = totals_mc.get("C/O", np.nan)
            if co_mc is not None and np.isfinite(co_mc) and co_mc > 0:
                CO_mc.append(np.log10(co_mc))
            else:
                CO_mc.append(np.nan)

            so_mc = totals_mc.get("S/O", np.nan)
            if so_mc is not None and np.isfinite(so_mc) and so_mc > 0:
                SO_mc.append(np.log10(so_mc))
            else:
                SO_mc.append(np.nan)

            neo_mc = totals_mc.get("Ne/O", np.nan)
            if neo_mc is not None and np.isfinite(neo_mc) and neo_mc > 0:
                NeO_mc.append(np.log10(neo_mc))
            else:
                NeO_mc.append(np.nan)

            aro_mc = totals_mc.get("Ar/O", np.nan)
            if aro_mc is not None and np.isfinite(aro_mc) and aro_mc > 0:
                ArO_mc.append(np.log10(aro_mc))
            else:
                ArO_mc.append(np.nan)

            # Collect per-tier N/O values.
            mc_tiers = totals_mc.get("_NO_tiers", {})
            for k in _tier_keys:
                val = mc_tiers.get(k, np.nan)
                NO_tier_mc[k].append(val if np.isfinite(val) else np.nan)
            # Only in-bounds draws count toward the N/O target; out-of-bounds
            # draws keep their (finite) O/H etc. but trigger another draw.
            if no_in_bounds:
                n_collected += 1
                _pbar.update(1)
        except (ValueError, RuntimeError):
            # Non-rejection failure (e.g. Te could not be solved): record
            # NaN for this draw.  It counts toward n_mc (re-drawing can't fix
            # it) and is excluded by nanmedian/nanstd.
            OH_mc.append(np.nan)
            NO_mc.append(np.nan)
            CO_mc.append(np.nan)
            SO_mc.append(np.nan)
            NeO_mc.append(np.nan)
            ArO_mc.append(np.nan)
            Te_high_mc.append(np.nan)
            Te_low_mc.append(np.nan)
            logU_mc_arr.append(np.nan)
            for k in _tier_keys:
                NO_tier_mc[k].append(np.nan)
            n_collected += 1
            _pbar.update(1)
    _pbar.close()

    # Warn if the attempt cap was hit before n_mc in-bounds N/O draws were
    # collected (object centred outside the Martinez+2025 bounds).  O/H and
    # the other non-Martinez ratios are unaffected and remain fully sampled.
    if n_collected < n_mc:
        logger.warning(
            "Direct Te MC: collected only %d/%d in-bounds N/O draws after "
            "%d attempts; N/O inputs repeatedly outside Martinez+2025 "
            "bounds. N/O is under-sampled (O/H and other ratios unaffected).",
            n_collected, n_mc, attempts,
        )

    OH_mc = np.array(OH_mc)
    NO_mc = np.array(NO_mc)
    CO_mc = np.array(CO_mc)
    SO_mc = np.array(SO_mc)
    NeO_mc = np.array(NeO_mc)
    ArO_mc = np.array(ArO_mc)

    OH_med_mc = float(np.nanmedian(OH_mc)) if np.any(np.isfinite(OH_mc)) else OH_12
    OH_err = float(np.nanstd(OH_mc)) if np.any(np.isfinite(OH_mc)) else np.nan
    NO_med_mc = float(np.nanmedian(NO_mc)) if np.any(np.isfinite(NO_mc)) else NO_log
    NO_err = float(np.nanstd(NO_mc)) if np.any(np.isfinite(NO_mc)) else None
    CO_med_mc = float(np.nanmedian(CO_mc)) if np.any(np.isfinite(CO_mc)) else CO_log
    CO_err = float(np.nanstd(CO_mc)) if np.any(np.isfinite(CO_mc)) else None
    SO_med_mc = float(np.nanmedian(SO_mc)) if np.any(np.isfinite(SO_mc)) else SO_log
    SO_err = float(np.nanstd(SO_mc)) if np.any(np.isfinite(SO_mc)) else None
    NeO_med_mc = float(np.nanmedian(NeO_mc)) if np.any(np.isfinite(NeO_mc)) else NeO_log
    NeO_err = float(np.nanstd(NeO_mc)) if np.any(np.isfinite(NeO_mc)) else None
    ArO_med_mc = float(np.nanmedian(ArO_mc)) if np.any(np.isfinite(ArO_mc)) else ArO_log
    ArO_err = float(np.nanstd(ArO_mc)) if np.any(np.isfinite(ArO_mc)) else None
    Te_high_mc = np.array(Te_high_mc)
    Te_low_mc = np.array(Te_low_mc)
    logU_mc_arr = np.array(logU_mc_arr)
    Te_high_err = float(np.nanstd(Te_high_mc)) if np.any(np.isfinite(Te_high_mc)) else None
    Te_low_err = float(np.nanstd(Te_low_mc)) if np.any(np.isfinite(Te_low_mc)) else None
    logU_err = float(np.nanstd(logU_mc_arr)) if np.any(np.isfinite(logU_mc_arr)) else None

    # Replace per-tier point estimates with MC medians and attach errors.
    if NO_tiers:
        for k in _tier_keys:
            arr = np.array(NO_tier_mc[k])
            if np.any(np.isfinite(arr)):
                med = float(np.nanmedian(arr))
                NO_tiers[k] = med
                NO_tiers[f"_err_{k}"] = float(np.nanstd(arr))

    # --- Alternative Te from O III] 1666 (cross-check) ---
    _alt_1666 = None
    if _Te_diagnostic == "4363" and f_1666 > 0 and f_5007 > 0:
        try:
            Te_alt = compute_Te_OIII_1666(f_1666, f_5007, f_4959, ne_OIII)
            Te_alt_low = Te_low_from_high(Te_alt, relation=Te_relation)
            Te_alt_int = Te_int_from_high(Te_alt, relation=Te_relation)
            ionic_alt = compute_ionic_abundances(
                fluxes, Te_alt, Te_alt_low, ne_low, ne_mid=ne_mid, ne_high=ne_high,
                Te_int=Te_alt_int, ne_Opp=ne_Opp,
            )
            OH_alt = ionic_alt.get("O+/H+", 0.0) + ionic_alt.get("O++/H+", 0.0)
            OH_alt_12 = 12.0 + np.log10(OH_alt) if OH_alt > 0 else np.nan
            _alt_1666 = {
                "Te_high": Te_alt,
                "Te_low": Te_alt_low,
                "OH": OH_alt_12,
                "ionic": ionic_alt,
            }
            # Point estimate only — no MC.  This is a cross-check on the
            # adopted lambda4363 temperature; propagating it would add a
            # second MC pass as costly as the main loop.
            diagnostics["Te(high) from 1666"] = (
                f"O III] 1666/(5007+4959) → T_e = {Te_alt:.0f} K → "
                f"12+log(O/H) = {OH_alt_12:.3f} (point estimate, not adopted)"
            )
        except (ValueError, RuntimeError) as e:
            logger.info("Could not compute alternative Te from 1666: %s", e)

    return {
        "OH": OH_med_mc,
        "OH_err": OH_err,
        "NO": NO_med_mc,
        "NO_err": NO_err,
        "CO": CO_med_mc,
        "CO_err": CO_err,
        "Te_high": Te_high,
        "Te_high_err": Te_high_err,
        "Te_mid": Te_int,
        "Te_mid_err": (
            0.83 * Te_high_err
            if (Te_int is not None and Te_high_err is not None) else None
        ),
        "Te_low": Te_low,
        "Te_low_err": Te_low_err,
        "ne": ne_low,
        "ne_low": ne_low,
        "ne_mid": ne_mid,
        "ne_high": ne_high,
        "ne_Opp": ne_Opp,
        "logU": logU,
        "logU_err": logU_err,
        "icf_method": icf_method,
        "NO_icf_name": NO_icf_name,
        "ionic": ionic,
        "ionic_upper_limits": ionic_upper_limits if ionic_upper_limits else None,
        "ionic_ul_details": ionic_ul_details if ionic_ul_details else None,
        "OH_posterior": OH_mc,
        "NO_posterior": NO_mc,
        "CO_posterior": CO_mc,
        "SO": SO_med_mc,
        "SO_err": SO_err,
        "NeO": NeO_med_mc,
        "NeO_err": NeO_err,
        "ArO": ArO_med_mc,
        "ArO_err": ArO_err,
        "diagnostics": diagnostics,
        "failures": failures if failures else None,
        "NO_tiers": NO_tiers,
        "icf_values": icf_values,
        "NO_is_upper_limit": NO_is_upper_limit,
    }


def _dust_correct_sample(
    sample: dict[str, float],
    Av: float,
    dust_law: str,
    dust_kwargs: dict,
) -> dict[str, float]:
    """Dust-correct a single flux sample in-place.

    Parameters
    ----------
    sample : dict
        ``{line_name: flux}`` — modified in place and returned.
    Av : float
        V-band attenuation for this draw.
    dust_law : str
        ``"salim"`` or ``"cardelli"``.
    dust_kwargs : dict
        Extra keyword arguments for the dust law.

    Returns
    -------
    dict
        The same dict with fluxes multiplied by dust correction factors.
    """
    from .dust import salim_attenuation, cardelli_extinction

    if Av <= 0:
        return sample
    for name in list(sample.keys()):
        wave = _LINE_WAVES.get(name)
        if wave is None:
            continue
        wave_arr = np.array([wave])
        if dust_law == "salim":
            A_lam = salim_attenuation(wave_arr, Av, **dust_kwargs)[0]
        else:
            A_lam = cardelli_extinction(wave_arr, Av)[0]
        sample[name] = sample[name] * 10.0 ** (0.4 * A_lam)
    return sample


def _run_direct_mcmc(
    posteriors: dict[str, np.ndarray],
    Te_relation: str,
    n_posterior: int = 1000,
    progress: bool = True,
    seed: int = 42,
    ne_high_max: float = 5e5,
    snr_ne: float = 3.0,
    snr_logU: float = 1.5,
    marginalize_logU: bool = True,
    icf_method: str = "auto",
    co_icf_method: str = "auto",
    snr_NO: float = 1.5,
    icf_tier: str | None = None,
    continuum_rms_limits: dict[str, float] | None = None,
    niv_rejected: bool = False,
    ne_low_override: float | None = None,
    ne_mid_override: float | None = None,
    ne_high_override: float | None = None,
    z: float = 0.0,
    # Per-draw dust resampling (when Av_err is set).
    Av: float | None = None,
    Av_err: float | None = None,
    Av_prior: str = "gaussian",
    dust_law: str = "salim",
    dust_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run the direct T_e method on MCMC posterior samples.

    Follows Berg+2025's 6-step procedure for each posterior sample.

    Parameters
    ----------
    posteriors : dict
        ``{line_name: flux_posterior_array}``.  When *Av_err* is set
        these must be **observed** (un-dust-corrected) fluxes; the
        function draws A_V per iteration and applies dust correction
        internally.  When *Av_err* is ``None``, posteriors are assumed
        to be already dust-corrected (legacy behaviour).
    Te_relation : str
        T_e-T_e relation.
    n_posterior : int
        Maximum number of posterior samples to use (default 1000).
        If the posterior is longer, a random subsample is drawn.
    progress : bool
        Show a ``tqdm`` progress bar (default ``True``).
    seed : int
        Random seed for subsampling (default 42).
    ne_high_max : float
        Maximum allowed n_e(high) in cm^-3 (default 5e5).
    snr_ne : float
        Minimum SNR for density-sensitive doublet members (default 3.0).
        SNR is computed from the median/std of the posterior for each
        doublet member.
    icf_method : str
        ICF scheme: ``"auto"``, ``"izotov06"``, ``"martinez25"``, or
        ``"direct_sum"`` (sum detected N ions; Topping+2024).
    snr_NO : float
        Minimum total-line SNR for each nitrogen ion when using
        ``icf_method="direct_sum"`` (default 2.0).
    Av : float or None
        Central A_V value for per-draw resampling.
    Av_err : float or None
        A_V uncertainty; triggers per-draw dust resampling when set.
    Av_prior : str
        ``"gaussian"`` or ``"uniform"``.
    dust_law : str
        ``"salim"`` or ``"cardelli"``.
    dust_kwargs : dict or None
        Extra keyword arguments for the dust law.

    Returns
    -------
    dict
        Same keys as :func:`_run_direct`.
    """
    from .direct import (
        NE_DEFAULT,
        Te_int_from_high,
        Te_low_from_high,
        compute_ionic_abundances,
        compute_Te_OIII,
        compute_Te_OIII_1666,
        compute_total_abundances,
    )
    from .martinez25_icf import LOG_OH_SOLAR, _LOG_U_VALID

    from .dust import _draw_Av

    _resample_dust = (Av_err is not None and Av_err > 0 and Av is not None)
    _dk = dust_kwargs or {}

    # Resampling over the full posterior pool: scan members in a shuffled
    # order until `n_samples` in-bounds N/O draws are collected (or the pool
    # is exhausted).  Only N/O is gated on the Martinez+2025 bounds; O/H, Te
    # and the other X/O ratios do not use the Martinez ICF, so they are
    # recorded for *every* scanned sample.  Their posteriors therefore hold
    # >= n_samples finite draws while N/O holds n_samples in-bounds draws.
    n_total = min(len(v) for v in posteriors.values())
    rng = np.random.default_rng(seed)
    n_samples = n_posterior if (n_posterior > 0 and n_total > n_posterior) else n_total
    scan_order = rng.permutation(n_total)
    max_scan = n_samples * _RESAMPLE_MAX_ATTEMPT_FACTOR

    OH_post = []
    NO_post = []
    CO_post = []
    SO_post = []
    NeO_post = []
    ArO_post = []
    Te_high_post = []
    Te_low_post = []
    logU_post = []
    Av_post = [] if _resample_dust else None

    # Compute medians and errors for the point estimate and multi-phase ne.
    # When resampling dust, posteriors are raw (observed); dust-correct
    # the medians with the central A_V for the point estimate.
    med_fluxes = {name: float(np.median(post)) for name, post in posteriors.items()}
    med_errors = {name: float(np.std(post)) for name, post in posteriors.items()}
    if _resample_dust:
        med_fluxes = _dust_correct_sample(dict(med_fluxes), Av, dust_law, _dk)

    # Steps 1-2: zone densities + temperatures with one refinement pass.
    # T_e(O++) is solved at the O²⁺-zone density ne_Opp ([Ar IV] preferred,
    # else CIII] -> z-fallback).
    try:
        (
            ne_low, ne_mid, ne_high, ne_Opp,
            Te_high_pt, Te_int_pt, Te_low_pt, _Te_diagnostic, ne_failures,
        ) = _solve_densities_refined(
            med_fluxes, med_errors, z, Te_relation, snr_ne, ne_high_max,
            niv_rejected,
            ne_low_override=ne_low_override,
            ne_mid_override=ne_mid_override,
            ne_high_override=ne_high_override,
        )
        if Te_high_pt is None:
            Te_high_pt = np.nan
            Te_low_pt = np.nan
            Te_int_pt = None
        elif _Te_diagnostic == "1666":
            logger.info(
                "[OIII] 4363 not available; using O III] 1666 for T_e(high) = %.0f K.",
                Te_high_pt,
            )
    except ValueError:
        # PyNEB could not solve the (present) auroral ratio at the medians.
        ne_low, ne_mid, ne_high, ne_Opp, ne_failures = _compute_multi_ne(
            med_fluxes, errors=med_errors, snr_ne=snr_ne,
            ne_high_max=ne_high_max, niv_density_invalid=niv_rejected, z=z,
        )
        if ne_low_override is not None:
            ne_low = ne_low_override
        if ne_mid_override is not None:
            ne_mid = ne_mid_override
            ne_Opp = ne_mid_override
        if ne_high_override is not None:
            ne_high = ne_high_override
        Te_high_pt = np.nan
        Te_low_pt = np.nan
        Te_int_pt = None
        _Te_diagnostic = None

    ne_OIII = ne_Opp

    ionic_pt = compute_ionic_abundances(
        med_fluxes, Te_high_pt, Te_low_pt, ne_low, ne_mid=ne_mid,
        ne_high=ne_high, Te_int=Te_int_pt, ne_Opp=ne_Opp,
    ) if np.isfinite(Te_high_pt) else {}

    OH_pt = ionic_pt.get("O+/H+", 0.0) + ionic_pt.get("O++/H+", 0.0)
    Z_Zsun_pt = 10.0 ** (12.0 + np.log10(OH_pt) - LOG_OH_SOLAR) if OH_pt > 0 else None
    logU_pt = None
    logU_diag = None
    logU_co_pt = None
    if Z_Zsun_pt is not None:
        logU_pt, logU_diag = _compute_logU(
            med_fluxes, Z_Zsun_pt, ne_high, errors=med_errors,
            snr_logU=snr_logU,
        )
        # C/O ICF: log U at the intermediate (C2+) zone density.
        logU_co_pt = logU_pt
        if logU_diag is not None and ne_mid is not None:
            logU_co_pt, _ = _compute_logU(
                med_fluxes, Z_Zsun_pt, ne_mid, errors=med_errors,
                snr_logU=snr_logU, lock_diag=logU_diag,
            )

    if ionic_pt:
        _gate_nitrogen_ions(ionic_pt, med_fluxes, med_errors, snr_NO=snr_NO)

    # Compute 3σ upper limits for non-detected ions.
    ionic_upper_limits, ionic_ul_details = {}, {}
    if ionic_pt and np.isfinite(Te_high_pt) and np.isfinite(Te_low_pt):
        ionic_upper_limits, ionic_ul_details = _compute_ionic_upper_limits(
            ionic_pt, med_fluxes, med_errors, Te_high_pt, Te_low_pt, ne_low,
            ne_mid if ne_mid is not None else ne_low,
            ne_high if ne_high is not None else ne_low,
            continuum_rms_limits=continuum_rms_limits,
            ne_Opp=ne_Opp, Te_int=Te_int_pt,
        )

    totals_pt = compute_total_abundances(
        ionic_pt, logU=logU_pt, Z_Zsun=Z_Zsun_pt, ne=ne_high,
        icf_method=icf_method,
        co_icf_method=co_icf_method,
        co_logU=logU_co_pt, co_ne=ne_mid,
        ionic_upper_limits=ionic_upper_limits,
        _lock_NO_icf=icf_tier,
    ) if ionic_pt else {}
    icf_method = totals_pt.get("icf_method")
    NO_icf_name = totals_pt.get("NO_icf_name")
    NO_is_upper_limit = totals_pt.pop("NO_is_upper_limit", False)
    failures = totals_pt.pop("_failures", {})
    failures.update(ne_failures)
    NO_tiers = totals_pt.pop("_NO_tiers", None)
    icf_values = totals_pt.pop("_icf_values", None)
    # Collect per-tier N/O posteriors for uncertainty on each method.
    _tier_keys = [k for k in (NO_tiers or {}) if not k.startswith("_")]
    NO_tier_post: dict[str, list[float]] = {k: [] for k in _tier_keys}

    n_collected = 0  # scanned draws counted toward n_samples (in-bounds N/O
    #                  or solver failure); out-of-bounds draws keep O/H etc.
    scanned = 0
    _pbar = tqdm(total=n_samples, desc="Direct Te (posterior)", disable=not progress)
    for src_i in scan_order:
        if n_collected >= n_samples or scanned >= max_scan:
            break
        scanned += 1
        sample = {name: max(float(post[src_i]), 1e-50) for name, post in posteriors.items()}

        # Per-draw dust correction: draw A_V and correct this sample.
        if _resample_dust:
            Av_draw = _draw_Av(rng, Av, Av_err, prior=Av_prior)
            Av_post.append(Av_draw)
            sample = _dust_correct_sample(sample, Av_draw, dust_law, _dk)

        try:
            if _Te_diagnostic == "4363":
                Te_h = compute_Te_OIII(
                    sample.get("OIII_4363", 0),
                    sample.get("OIII_5007", 0),
                    sample.get("OIII_4959", 0),
                    ne_OIII,
                )
            else:
                Te_h = compute_Te_OIII_1666(
                    sample.get("OIII_1666", 0),
                    sample.get("OIII_5007", 0),
                    sample.get("OIII_4959", 0),
                    ne_OIII,
                )
            Te_l = Te_low_from_high(Te_h, relation=Te_relation)
            Te_i = Te_int_from_high(Te_h, relation=Te_relation)
            ionic_i = compute_ionic_abundances(
                sample, Te_h, Te_l, ne_low, ne_mid=ne_mid, ne_high=ne_high,
                Te_int=Te_i, ne_Opp=ne_Opp,
            )

            # Z_Zsun for this sample.
            oh_val = ionic_i.get("O+/H+", 0.0) + ionic_i.get("O++/H+", 0.0)
            z_zsun_i = 10.0 ** (12.0 + np.log10(oh_val) - LOG_OH_SOLAR) if oh_val > 0 else Z_Zsun_pt

            # logU for this sample.  If the O32/N43/Z inputs or the resulting
            # logU fall outside the Martinez+2025 bounds, logU is set to NaN
            # so the Martinez N/O ICF returns NaN (this draw does not count
            # toward n_posterior); O/H, Te and the other ratios do not use
            # logU and are still recorded.
            # Pass med_errors so the same SNR gating applies as for the
            # point estimate, and lock the diagnostic to the one the point
            # estimate selected so every draw stays on N43 (or every draw
            # on O32) — the log(U) posterior is never a silent mixture.
            # Resample log(U) so the reported logU posterior carries its
            # measurement uncertainty.  marginalize_logU controls whether
            # that scatter propagates into the ICF ratios: when False the
            # ICFs use the fixed point-estimate logU (the analogue of a fixed
            # A_V applied to every draw), so logU does not inflate N/O — but
            # logU still gets a reported error bar.
            logU_sample = logU_pt
            no_in_bounds = True
            if z_zsun_i is not None and logU_diag is not None:
                logU_val, _ = _compute_logU(
                    sample, z_zsun_i, ne_high, errors=med_errors,
                    snr_logU=snr_logU, lock_diag=logU_diag,
                )
                if (logU_val is not None and np.isfinite(logU_val)
                        and _LOG_U_VALID[0] <= logU_val <= _LOG_U_VALID[1]):
                    logU_sample = float(logU_val)
                elif marginalize_logU:
                    logU_sample = np.nan
                    no_in_bounds = False
            logU_icf = logU_sample if marginalize_logU else logU_pt

            # C/O log U at the intermediate (C2+) zone density.
            logU_co_sample = logU_co_pt
            if (marginalize_logU and z_zsun_i is not None
                    and logU_diag is not None and ne_mid is not None):
                logU_co_val, _ = _compute_logU(
                    sample, z_zsun_i, ne_mid, errors=med_errors,
                    snr_logU=snr_logU, lock_diag=logU_diag,
                )
                if logU_co_val is not None and np.isfinite(logU_co_val):
                    logU_co_sample = float(logU_co_val)
            logU_co_icf = logU_co_sample if marginalize_logU else logU_co_pt

            Te_high_post.append(Te_h)
            Te_low_post.append(Te_l)
            logU_post.append(logU_sample if logU_sample is not None else np.nan)

            # Gate nitrogen ions using median errors (same ions as
            # point estimate to prevent tier-switching across samples).
            _gate_nitrogen_ions(ionic_i, sample, med_errors, snr_NO=snr_NO)

            totals_i = compute_total_abundances(
                ionic_i, logU=logU_icf, Z_Zsun=z_zsun_i, ne=ne_high,
                icf_method=icf_method,
                co_icf_method=co_icf_method,
                co_logU=logU_co_icf, co_ne=ne_mid,
                _lock_NO_icf=NO_icf_name,
            )

            oh = totals_i.get("O/H", np.nan)
            OH_post.append(12.0 + _log10_or_nan(oh))
            NO_post.append(_log10_or_nan(totals_i.get("N/O", np.nan)))
            CO_post.append(_log10_or_nan(totals_i.get("C/O", np.nan)))
            SO_post.append(_log10_or_nan(totals_i.get("S/O", np.nan)))
            NeO_post.append(_log10_or_nan(totals_i.get("Ne/O", np.nan)))
            ArO_post.append(_log10_or_nan(totals_i.get("Ar/O", np.nan)))

            # Collect per-tier N/O values.
            mc_tiers = totals_i.get("_NO_tiers", {})
            for k in _tier_keys:
                val = mc_tiers.get(k, np.nan)
                NO_tier_post[k].append(val if np.isfinite(val) else np.nan)

            # Only in-bounds draws count toward the N/O target; out-of-bounds
            # draws keep their (finite) O/H etc. but trigger another scan.
            if no_in_bounds:
                n_collected += 1
                _pbar.update(1)
        except (ValueError, RuntimeError):
            # Non-rejection failure: record NaN for this draw.  It counts
            # toward n_posterior (re-drawing can't fix it).
            OH_post.append(np.nan)
            NO_post.append(np.nan)
            CO_post.append(np.nan)
            SO_post.append(np.nan)
            NeO_post.append(np.nan)
            ArO_post.append(np.nan)
            Te_high_post.append(np.nan)
            Te_low_post.append(np.nan)
            logU_post.append(np.nan)
            for k in _tier_keys:
                NO_tier_post[k].append(np.nan)
            n_collected += 1
            _pbar.update(1)
    _pbar.close()

    # Convert per-quantity posteriors to arrays.  O/H, Te and the other
    # non-Martinez ratios hold one entry per scanned draw; N/O has the same
    # length with NaN on the out-of-bounds draws (n_collected are in-bounds).
    OH_post = np.array(OH_post)
    NO_post = np.array(NO_post)
    CO_post = np.array(CO_post)
    SO_post = np.array(SO_post)
    NeO_post = np.array(NeO_post)
    ArO_post = np.array(ArO_post)
    Te_high_post = np.array(Te_high_post)
    Te_low_post = np.array(Te_low_post)
    logU_post = np.array(logU_post)
    if Av_post is not None:
        Av_post = np.array(Av_post)

    # Warn if the scan ended before n_samples in-bounds N/O draws were
    # collected (object centred outside the Martinez+2025 bounds).  O/H and
    # the other non-Martinez ratios are unaffected and remain fully sampled.
    if n_collected < n_samples:
        logger.warning(
            "Direct Te posterior: collected only %d/%d in-bounds N/O draws "
            "from a pool of %d; N/O inputs outside Martinez+2025 bounds. "
            "N/O is under-sampled (O/H and other ratios unaffected).",
            n_collected, n_samples, n_total,
        )

    # Point estimates from posteriors.
    OH_med = float(np.nanmedian(OH_post))
    OH_lo = float(OH_med - np.nanpercentile(OH_post, 16))
    OH_hi = float(np.nanpercentile(OH_post, 84) - OH_med)
    NO_med = float(np.nanmedian(NO_post)) if np.any(np.isfinite(NO_post)) else None
    NO_lo = NO_hi = None
    if NO_med is not None:
        NO_lo = float(NO_med - np.nanpercentile(NO_post, 16))
        NO_hi = float(np.nanpercentile(NO_post, 84) - NO_med)
    CO_med = float(np.nanmedian(CO_post)) if np.any(np.isfinite(CO_post)) else None
    CO_lo = CO_hi = None
    if CO_med is not None:
        CO_lo = float(CO_med - np.nanpercentile(CO_post, 16))
        CO_hi = float(np.nanpercentile(CO_post, 84) - CO_med)
    SO_med = float(np.nanmedian(SO_post)) if np.any(np.isfinite(SO_post)) else None
    SO_lo = SO_hi = None
    if SO_med is not None:
        SO_lo = float(SO_med - np.nanpercentile(SO_post, 16))
        SO_hi = float(np.nanpercentile(SO_post, 84) - SO_med)
    NeO_med = float(np.nanmedian(NeO_post)) if np.any(np.isfinite(NeO_post)) else None
    NeO_lo = NeO_hi = None
    if NeO_med is not None:
        NeO_lo = float(NeO_med - np.nanpercentile(NeO_post, 16))
        NeO_hi = float(np.nanpercentile(NeO_post, 84) - NeO_med)
    ArO_med = float(np.nanmedian(ArO_post)) if np.any(np.isfinite(ArO_post)) else None
    ArO_lo = ArO_hi = None
    if ArO_med is not None:
        ArO_lo = float(ArO_med - np.nanpercentile(ArO_post, 16))
        ArO_hi = float(np.nanpercentile(ArO_post, 84) - ArO_med)
    Te_high_med = float(np.nanmedian(Te_high_post)) if np.any(np.isfinite(Te_high_post)) else None
    Te_high_lo = Te_high_hi = None
    if Te_high_med is not None:
        Te_high_lo = float(Te_high_med - np.nanpercentile(Te_high_post, 16))
        Te_high_hi = float(np.nanpercentile(Te_high_post, 84) - Te_high_med)
    Te_low_med = float(np.nanmedian(Te_low_post)) if np.any(np.isfinite(Te_low_post)) else None
    Te_low_lo = Te_low_hi = None
    if Te_low_med is not None:
        Te_low_lo = float(Te_low_med - np.nanpercentile(Te_low_post, 16))
        Te_low_hi = float(np.nanpercentile(Te_low_post, 84) - Te_low_med)
    logU_med = float(np.nanmedian(logU_post)) if np.any(np.isfinite(logU_post)) else None
    logU_lo = logU_hi = None
    if logU_med is not None:
        logU_lo = float(logU_med - np.nanpercentile(logU_post, 16))
        logU_hi = float(np.nanpercentile(logU_post, 84) - logU_med)

    # Replace per-tier point estimates with posterior medians and attach errors.
    if NO_tiers:
        for k in _tier_keys:
            arr = np.array(NO_tier_post[k])
            if np.any(np.isfinite(arr)):
                med = float(np.nanmedian(arr))
                lo = float(med - np.nanpercentile(arr, 16))
                hi = float(np.nanpercentile(arr, 84) - med)
                NO_tiers[k] = med
                NO_tiers[f"_err_{k}"] = (lo, hi)

    # --- Alternative Te from O III] 1666 (cross-check) ---
    _diag_extra = {}
    if _Te_diagnostic == "4363" and med_fluxes.get("OIII_1666", 0) > 0 and med_fluxes.get("OIII_5007", 0) > 0:
        try:
            Te_alt = compute_Te_OIII_1666(
                med_fluxes["OIII_1666"], med_fluxes["OIII_5007"],
                med_fluxes.get("OIII_4959", 0), ne_OIII,
            )
            Te_alt_low = Te_low_from_high(Te_alt, relation=Te_relation)
            Te_alt_int = Te_int_from_high(Te_alt, relation=Te_relation)
            ionic_alt = compute_ionic_abundances(
                med_fluxes, Te_alt, Te_alt_low, ne_low, ne_mid=ne_mid,
                ne_high=ne_high, Te_int=Te_alt_int, ne_Opp=ne_Opp,
            )
            OH_alt = ionic_alt.get("O+/H+", 0.0) + ionic_alt.get("O++/H+", 0.0)
            OH_alt_12 = 12.0 + np.log10(OH_alt) if OH_alt > 0 else np.nan
            # Point estimate only — no MC.  This is a cross-check on the
            # adopted lambda4363 temperature; propagating it through the
            # posterior would add a second pass as costly as the main loop.
            _diag_extra["Te(high) from 1666"] = (
                f"O III] 1666/(5007+4959) → T_e = {Te_alt:.0f} K → "
                f"12+log(O/H) = {OH_alt_12:.3f} (point estimate, not adopted)"
            )
        except (ValueError, RuntimeError) as e:
            logger.info("Could not compute alternative Te from 1666: %s", e)

    # Intermediate-zone temperature (deterministic function of T_e(high)).
    _Te_high_final = (
        Te_high_med if Te_high_med is not None
        else (Te_high_pt if np.isfinite(Te_high_pt) else None)
    )
    _Te_mid_final = (
        Te_int_from_high(_Te_high_final, Te_relation)
        if _Te_high_final is not None else None
    )
    _Te_mid_err = None
    if _Te_mid_final is not None and Te_high_lo is not None:
        _Te_mid_err = (0.83 * Te_high_lo, 0.83 * Te_high_hi)

    return {
        "OH": OH_med,
        "OH_err": (OH_lo, OH_hi),
        "NO": NO_med,
        "NO_err": (NO_lo, NO_hi) if NO_lo is not None else None,
        "CO": CO_med,
        "CO_err": (CO_lo, CO_hi) if CO_lo is not None else None,
        "Te_high": _Te_high_final,
        "Te_high_err": (Te_high_lo, Te_high_hi) if Te_high_lo is not None else None,
        "Te_mid": _Te_mid_final,
        "Te_mid_err": _Te_mid_err,
        "Te_low": Te_low_med if Te_low_med is not None else (Te_low_pt if np.isfinite(Te_low_pt) else None),
        "Te_low_err": (Te_low_lo, Te_low_hi) if Te_low_lo is not None else None,
        "ne": ne_low,
        "ne_low": ne_low,
        "ne_mid": ne_mid,
        "ne_high": ne_high,
        "ne_Opp": ne_Opp,
        "logU": logU_med if logU_med is not None else logU_pt,
        "logU_err": (logU_lo, logU_hi) if logU_lo is not None else None,
        "icf_method": icf_method,
        "NO_icf_name": NO_icf_name,
        "ionic": ionic_pt if ionic_pt else None,
        "ionic_upper_limits": ionic_upper_limits if ionic_upper_limits else None,
        "ionic_ul_details": ionic_ul_details if ionic_ul_details else None,
        "OH_posterior": OH_post,
        "NO_posterior": NO_post if np.any(np.isfinite(NO_post)) else None,
        "CO_posterior": CO_post if np.any(np.isfinite(CO_post)) else None,
        "SO": SO_med if SO_med is not None else (np.log10(totals_pt["S/O"]) if "S/O" in totals_pt and totals_pt["S/O"] > 0 else None),
        "SO_err": (SO_lo, SO_hi) if SO_lo is not None else None,
        "NeO": NeO_med if NeO_med is not None else (np.log10(totals_pt["Ne/O"]) if "Ne/O" in totals_pt and totals_pt["Ne/O"] > 0 else None),
        "NeO_err": (NeO_lo, NeO_hi) if NeO_lo is not None else None,
        "ArO": ArO_med if ArO_med is not None else (np.log10(totals_pt["Ar/O"]) if "Ar/O" in totals_pt and totals_pt["Ar/O"] > 0 else None),
        "ArO_err": (ArO_lo, ArO_hi) if ArO_lo is not None else None,
        "diagnostics": {
            **_build_diagnostics(
                med_fluxes, Te_high_pt if np.isfinite(Te_high_pt) else None,
                Te_relation, ne_low, ne_mid, ne_high, logU_pt, logU_diag,
                icf_method, NO_icf_name, NE_DEFAULT,
                totals=totals_pt, niv_rejected=niv_rejected,
                ne_Opp=ne_Opp, Te_int=_Te_mid_final,
                te_high_diag=_Te_diagnostic, z=z,
            ),
            **_diag_extra,
        },
        "failures": failures if failures else None,
        "NO_tiers": NO_tiers,
        "icf_values": icf_values,
        "NO_is_upper_limit": NO_is_upper_limit,
        "Av_posterior": Av_post,
    }


def _run_strong_line(
    fluxes: dict[str, float],
    errors: dict[str, float],
    is_mcmc: bool,
    posteriors: dict[str, np.ndarray],
    n_mc: int,
    n_posterior: int,
    progress: bool,
    Av_derived: float | None,
    Av_err_derived: float | None,
    excluded_lines: list[str],
) -> AbundanceResult:
    """Run the strong-line method and return an AbundanceResult."""
    from .strong_line import sanders25_metallicity

    if is_mcmc and posteriors:
        from .strong_line import (
            CALIBRATIONS,
            _chi2_simultaneous,
            Z_MAX,
            Z_MIN,
            compute_line_ratios,
        )
        from scipy.optimize import minimize_scalar

        n_total = min(len(v) for v in posteriors.values())
        if n_posterior > 0 and n_total > n_posterior:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_total, size=n_posterior, replace=False)
            idx.sort()
            thinned = {name: posteriors[name][idx] for name in posteriors}
            n_samples = n_posterior
        else:
            thinned = posteriors
            n_samples = n_total
            rng = np.random.default_rng(42)

        OH_post = np.full(n_samples, np.nan)
        sample_fluxes = {name: thinned[name] for name in thinned}
        dummy_errors = {name: 0.0 for name in posteriors}

        for i in tqdm(range(n_samples), desc="Strong-line (posterior)", disable=not progress):
            samp = {name: max(float(arr[i]), 1e-50) for name, arr in sample_fluxes.items()}
            try:
                ratios_i = compute_line_ratios(samp, dummy_errors, snr_thresh=-np.inf)
                if not ratios_i:
                    continue
                perturbed = {}
                for m, dat in ratios_i.items():
                    sig_cal = CALIBRATIONS[m]["sigma_cal"]
                    perturbed[m] = {
                        "val": rng.normal(dat["val"], sig_cal),
                        "err": dat["err"],
                    }
                res_i = minimize_scalar(
                    _chi2_simultaneous,
                    bounds=(Z_MIN, Z_MAX),
                    args=(perturbed,),
                    method="bounded",
                )
                OH_post[i] = res_i.x
            except (ValueError, RuntimeError):
                continue

        OH_med = float(np.nanmedian(OH_post))
        OH_lo = float(OH_med - np.nanpercentile(OH_post, 16))
        OH_hi = float(np.nanpercentile(OH_post, 84) - OH_med)

        med_fluxes = {name: float(np.median(arr)) for name, arr in posteriors.items()}
        med_errors = {name: float(np.std(arr)) for name, arr in posteriors.items()}
        ratios = compute_line_ratios(med_fluxes, med_errors)

        return AbundanceResult(
            method="strong_line",
            OH=OH_med,
            OH_err=(OH_lo, OH_hi),
            Av=Av_derived,
            Av_err=Av_err_derived,
            OH_posterior=OH_post,
            ratios_used=list(ratios.keys()),
            excluded_lines=excluded_lines if excluded_lines else None,
        )

    Z_best, Z_lo, Z_hi, chi2, ratios_used, Z_mc = sanders25_metallicity(
        fluxes, errors, n_mc=n_mc, progress=progress,
    )

    return AbundanceResult(
        method="strong_line",
        OH=Z_best,
        OH_err=(Z_best - Z_lo, Z_hi - Z_best),
        Av=Av_derived,
        Av_err=Av_err_derived,
        chi2=chi2,
        ratios_used=ratios_used,
        OH_posterior=Z_mc,
        excluded_lines=excluded_lines if excluded_lines else None,
    )


def compute_abundances(
    result: Any,
    z: float,
    *,
    dust_correct: bool = True,
    dust_law: str = "salim",
    Av: float | None = None,
    Av_err: float | None = None,
    Av_prior: str = "gaussian",
    method: str = "auto",
    snr_auroral: float = 3.0,
    snr_line: float = 2.0,
    ne_high_max: float = 5e5,
    snr_ne: float = 3.0,
    snr_logU: float = 1.5,
    marginalize_logU: bool = True,
    n_mc: int = 1000,
    Te_relation: str = "3_tier",
    Rv: float = 3.15,
    delta: float = -0.35,
    B_bump: float = 2.27,
    icf_method: str = "auto",
    co_icf_method: str = "auto",
    snr_NO: float = 1.5,
    icf_tier: str | None = None,
    # Electron density overrides (bypass diagnostic computation)
    ne_low_override: float | None = None,
    ne_mid_override: float | None = None,
    ne_high_override: float | None = None,
    # Physical redshift for the z-dependent n_e fallbacks (decoupled from
    # the geometric *z*; set for rest-frame stacks fit at z=0).
    z_ne: float | None = None,
    # Balmer decrement SNR floor for A_V derivation
    snr_balmer: float = 3.0,
    # Balmer decrement anchor line for A_V derivation
    balmer_anchor: str = "HBETA",
    balmer_pair: tuple[str, str] | None = None,
    # Forward model kwargs (method="forward")
    forward_sampler: str = "emcee",
    forward_n_walkers: int = 32,
    forward_n_steps: int = 5000,
    forward_n_burn: int = 1000,
    forward_n_live: int = 500,
    forward_seed: int = 42,
    forward_progress: bool = True,
    progress: bool = True,
    n_posterior: int = 1000,
) -> AbundanceResult:
    """Compute chemical abundances from a fitting result.

    Dust handling (default)
    -----------------------
    By default ``A_V`` is computed **once** from the weighted-mean
    Balmer decrement of the *median* posterior fluxes and applied as a
    deterministic correction to every posterior draw — i.e. ``A_V`` is
    held *constant* across draws.  The Balmer-propagated uncertainty
    on ``A_V`` is reported on the returned :class:`AbundanceResult`
    (``Av``, ``Av_err``) but is **not** marginalised into the
    abundance posteriors.

    To marginalise over ``A_V`` instead (each draw dust-corrected with
    its own ``A_V`` sample), opt in by passing ``Av_err > 0``
    explicitly — see *Av_err* below.

    Parameters
    ----------
    result : FitResult | BroadFitResult | MCMCResult | MCMCBroadFitResult
        Emission-line fitting result from ``jwspecfit`` or ``jwspecmcmc``.
    z : float
        Source redshift.
    dust_correct : bool
        Whether to apply dust correction (default ``True``).
    dust_law : str
        ``"salim"`` (default) or ``"cardelli"``.
    Av : float or None
        V-band attenuation. If ``None``, derived from Balmer decrement
        (see *balmer_anchor*).
    balmer_anchor : str
        Denominator line for the multi-Balmer A_V derivation.  Only
        Balmer lines **bluer than** the anchor (and above *snr_balmer*)
        are ratioed against it.  ``"HBETA"`` (default) therefore uses
        Hγ/Hβ, Hδ/Hβ, H9/Hβ, H10/Hβ — **Hα is excluded**.  ``"Ha"`` uses
        every other Balmer line (Hβ/Hα, Hγ/Hα, Hδ/Hα, H9/Hα, H10/Hα).
        So anchor on Hα when you want Hα included, on Hβ when you don't.
        Ignored when *Av* is supplied directly, or when *balmer_pair* is
        set.
    balmer_pair : tuple of str or None
        Force the A_V derivation onto a single Balmer decrement instead
        of the multi-line fit, given as ``(numerator, denominator)`` line
        names from the ladder (``"Ha"``, ``"HBETA"``, ``"HGAMMA"``,
        ``"HDELTA"``, ``"H9"``, ``"H10"``).  For example
        ``balmer_pair=("Ha", "HBETA")`` uses only Hα/Hβ — useful to avoid
        low-SNR Balmer lines.  Overrides *balmer_anchor* / *snr_balmer*
        for the A_V step.  Ignored when *Av* is supplied directly.
        ``None`` (default) keeps the multi-Balmer fit.
    Av_err : float or None
        Controls whether A_V is marginalised over.  ``None`` (default)
        keeps A_V fixed at its central value — derived once from the
        median Balmer decrement, or taken from *Av* if supplied — and
        applies a single deterministic dust correction to all draws.
        Passing a positive value here switches on per-draw resampling:
        each posterior draw is dust-corrected with an A_V drawn from
        the chosen *Av_prior* (centred on *Av*, or on the Balmer-derived
        value when *Av* is ``None``).  The Balmer-derived error is
        *always* reported on the result; it is used for resampling
        only when you pass it back in explicitly here.
    Av_prior : str
        Prior shape for A_V sampling: ``"gaussian"`` (default) draws
        from N(*Av*, *Av_err*) clipped at 0; ``"uniform"`` draws from
        U(max(*Av* − *Av_err*, 0), *Av* + *Av_err*).  Only matters
        when *Av_err* is set.
    method : str
        ``"auto"`` (default), ``"direct"``, ``"forward"``, or
        ``"strong_line"``.  ``"auto"`` uses direct if [OIII] 4363
        SNR >= *snr_auroral*.  ``"forward"`` runs the Bayesian
        forward model (Cullen+25) — see :func:`forward_model`.
    snr_auroral : float
        Minimum SNR for [OIII] 4363 to use the direct method (default 3.0).
    snr_line : float
        Minimum per-line SNR for inclusion in the abundance calculation
        (default 2.0).  Lines below this threshold are removed from
        the flux dict after dust correction.  Does not affect the
        auroral-line SNR check (controlled by *snr_auroral*) or lines
        essential for T_e computation (OIII 4363/5007/4959, Hbeta).
        Set to 0 to disable filtering.
    ne_high_max : float
        Maximum allowed high-ionisation electron density in cm^-3
        (default 5e5).  If n_e(high) from a UV doublet exceeds this,
        the code falls back to n_e(low).  Prevents unphysical density
        estimates from noisy doublet ratios.
    snr_ne : float
        Minimum SNR for both members of a density-sensitive doublet
        (default 3.0).  Doublets where either member has
        ``flux / error < snr_ne`` are skipped, and the default
        density (300 cm^-3) is used.  Set to 0 to disable.
    snr_logU : float
        Minimum **total-doublet** SNR for NIV] and NIII] when
        computing log(U) from N43 (default 1.5).  The summed doublet
        flux is divided by the quadrature-summed error; if this is
        below the threshold, N43 is skipped and O32 is used instead.
    marginalize_logU : bool
        Whether to **marginalise over** log(U) — i.e. let it vary per draw
        and propagate its uncertainty into the ICF-based ratios (default
        ``True``).  Note ``True`` means log(U) *varies* (it is integrated
        out), **not** that it is held constant.  When ``True`` the per-draw
        log(U) feeds each draw's ICFs, so its scatter inflates N/O (and the
        other ICF ratios).  When ``False`` the ICFs instead use the fixed
        point-estimate log(U) from the median fluxes — the analogue of
        applying a fixed ``A_V`` to every draw — so log(U) scatter does not
        inflate N/O.  In **both** cases log(U) is still resampled for
        reporting, so ``logU`` and ``logU_err`` always carry the
        measurement uncertainty (it is not forced to zero when fixed).
    n_mc : int
        Monte Carlo iterations for error propagation (default 1000).
    Te_relation : str
        T_e-T_e relation used to derive the intermediate- and low-zone
        electron temperatures from T_e(O++).  ``"3_tier"`` (default)
        follows Martinez+2025 (arXiv:2510.21960) with Garnett (1992)
        throughout: T_int = 0.83*T_high + 1700 (S²⁺ zone) and
        T_low = 0.70*T_high + 3000 (O⁺/N⁺ zone), giving monotone zones
        (T_high >= T_int >= T_low for T_high >~ 1e4 K).  ``"classical"`` /
        ``"garnett"`` are aliases.  ``"desi"`` uses the DESI DR2 low
        relation (T_low = 0.648*T_high + 3270) with no intermediate zone
        (S²⁺/Ar²⁺ then fall back to the T_high/T_low midpoint).
    Rv : float
        Total-to-selective ratio for Salim law (default 3.15).
    delta : float
        Slope deviation for Salim law (default -0.35).
    B_bump : float
        UV bump strength for Salim law (default 2.27).
    icf_method : str
        ICF scheme for the direct method.
        ``"auto"`` (default): use Martinez+25 when logU is available,
        fall back to Izotov+06 otherwise.
        ``"izotov06"``: always use Izotov+06 ICFs (N/O = ICF × N⁺/O⁺,
        independent of logU).
        ``"martinez25"``: force Martinez+25 ICFs (requires logU).
        ``"direct_sum"``: sum all detected nitrogen ions directly
        (Topping+2024, Yanagisawa+2025, Cameron+2023).  No ICF or logU
        needed; falls back to Izotov+06 if only N⁺ is available.

        This controls **N/O only**; use ``co_icf_method`` for C/O.
    co_icf_method : str
        ICF scheme for C/O in the direct method (independent of
        ``icf_method``).
        ``"auto"`` (default) and ``"martinez25"``: apply the Martinez
        (in prep.) C/O ICF to C²⁺/O²⁺ (from C III] and [O III]) when
        logU and Z are available — a Cloudy-calibrated correction that
        uses **only C²⁺** (no C III/λ1548 dependence).
        ``"garnett97"``: the legacy tiered approach — direct sum
        (C⁺ + C²⁺ + C³⁺)/(O⁺ + O²⁺) when C⁺ is detected, else the
        Garnett+1997 ICF applied to (C²⁺ + C³⁺)/O²⁺.
        ``"martinez25"`` falls back to ``"garnett97"`` (with a warning)
        when logU/Z or C²⁺/O²⁺ are unavailable.
    snr_NO : float
        Minimum total-line SNR for each nitrogen ion when using
        ``icf_method="direct_sum"`` (default 2.0).  For doublets
        (NIII], NIV], NV), the summed flux is divided by the
        quadrature-summed error.  Ions below this threshold are
        excluded from the direct sum, causing the code to fall
        through to a lower tier (or Izotov+06).
    icf_tier : str or None
        Lock the Martinez+25 N/O ICF to a specific tier instead of
        the automatic cascade.  One of ``"NpOp"`` (ICF 1: N⁺/O⁺),
        ``"NppOpp"`` (ICF 2: N²⁺/O²⁺), ``"NpppOpp"`` (ICF 3:
        N³⁺/O²⁺), ``"NpNpp_OpOpp"`` (ICF 4: (N⁺+N²⁺)/(O⁺+O²⁺)) or
        ``"NppNppp_Opp"`` (ICF 5: (N²⁺+N³⁺)/O²⁺).  When locked, no
        fallback to other tiers occurs; if the required ions are
        missing, N/O is reported as failed.  Any other string raises
        ``ValueError``.  Default ``None`` (automatic cascade).
    ne_low_override : float or None
        If set, use this value (cm^-3) for the low-ionisation zone
        density instead of deriving it from [SII] or [OII].
    ne_mid_override : float or None
        If set, use this value (cm^-3) for the mid-ionisation zone
        density instead of deriving it from CIII].
    ne_high_override : float or None
        If set, use this value (cm^-3) for the high-ionisation zone
        density instead of deriving it from NIV] (or the fallback
        chain).  Useful when the CIII]-derived fallback is suspect.
    z_ne : float or None
        Physical redshift used **only** for the z-dependent electron-
        density fallbacks (:func:`jwspecabund.direct.ne_zone_fallback`),
        decoupled from the geometric *z*.  Defaults to *z* when ``None``.
        Set this for rest-frame stacks fit at ``z=0``: pass ``z=0`` (so
        the continuum-RMS line windows stay rest-frame) and
        ``z_ne=<sample median redshift>`` so a zone whose doublet is
        missing/failed still falls back to the correct high-z density.
        Only matters when a density zone has no usable doublet.
    forward_sampler : str
        Sampler for forward model: ``"emcee"`` or ``"dynesty"`` (default ``"emcee"``).
    forward_n_walkers : int
        Number of walkers for emcee forward model (default 32).
    forward_n_steps : int
        MCMC steps for emcee forward model (default 5000).
    forward_n_burn : int
        Burn-in steps for emcee forward model (default 1000).
    forward_n_live : int
        Live points for dynesty forward model (default 500).
    forward_seed : int
        Random seed for the forward model (default 42).
    forward_progress : bool
        Show progress bar for the forward model (default ``True``).
        Deprecated — use *progress* instead.
    progress : bool
        Show progress bars for MC / posterior loops (default ``True``).
    n_posterior : int
        Maximum number of posterior samples to propagate through
        PyNEB / strong-line calculations (default 1000).  If the
        MCMC posterior has more samples, a random subsample is drawn.

    Returns
    -------
    AbundanceResult
        Chemical abundance measurement.
    """
    # --- Fail-fast validation of string-choice inputs ---
    # Every choice parameter raises here with the valid options rather
    # than falling back silently or failing deep inside the pipeline.
    from .dust import _BALMER_LADDER, _VALID_BALMER_ANCHORS
    from .martinez25_icf import VALID_NO_ICF_TIERS

    _choices = {
        "method": (method, ("auto", "direct", "forward", "strong_line")),
        "dust_law": (dust_law, ("salim", "cardelli")),
        "Av_prior": (Av_prior, ("gaussian", "uniform")),
        "Te_relation": (Te_relation, ("3_tier", "classical", "garnett", "desi")),
        "icf_method": (icf_method, ("auto", "martinez25", "izotov06", "direct_sum")),
        "co_icf_method": (co_icf_method, ("auto", "martinez25", "garnett97")),
        "forward_sampler": (forward_sampler, ("emcee", "dynesty")),
        "balmer_anchor": (balmer_anchor, _VALID_BALMER_ANCHORS),
    }
    if icf_tier is not None:
        _choices["icf_tier"] = (icf_tier, VALID_NO_ICF_TIERS)
    for _name, (_value, _valid) in _choices.items():
        if _value not in _valid:
            raise ValueError(
                f"Invalid {_name} {_value!r}; "
                f"valid options are {', '.join(_valid)}."
            )
    if balmer_pair is not None:
        if len(balmer_pair) != 2:
            raise ValueError(
                f"balmer_pair must be a (numerator, denominator) pair of "
                f"Balmer line names, got {balmer_pair!r}."
            )
        for _nm in balmer_pair:
            if _nm not in _BALMER_LADDER:
                raise ValueError(
                    f"Invalid balmer_pair line {_nm!r}; "
                    f"valid options are {', '.join(_BALMER_LADDER)}."
                )

    # Physical redshift for the z-dependent n_e fallbacks (decoupled from
    # the geometric z used for continuum-RMS line windows).
    _z_ne = z_ne if z_ne is not None else z

    # --- Extract fluxes ---
    fluxes, errors, is_mcmc = _extract_fluxes(result)
    posteriors = _extract_posteriors(result) if is_mcmc else {}

    # --- Save observed Lyα flux before dust correction ---
    # f_esc(Lyα) uses the observed flux (not dust-corrected).
    _lya_obs_flux = fluxes.get("Lya", 0.0)
    _lya_obs_err = errors.get("Lya", 0.0)

    # --- Dust correction ---
    dust_kwargs = {}
    if dust_law == "salim":
        dust_kwargs = {"Rv": Rv, "delta": delta, "B": B_bump}

    Av_derived = None
    Av_err_derived = None
    _balmer_info: dict | None = None
    if dust_correct:
        if Av is None:
            if balmer_pair is not None:
                # Force A_V onto a single chosen decrement (e.g. Hα/Hβ),
                # ignoring lower-SNR Balmer lines.
                balmer_out = compute_Av_balmer_pair(
                    fluxes, errors, balmer_pair, law=dust_law, **dust_kwargs,
                )
            else:
                # Derive A_V from all available Balmer decrements anchored
                # on either Hβ (default) or Hα via `balmer_anchor`.
                balmer_out = compute_Av_multi_balmer(
                    fluxes, errors, law=dust_law, snr_min=snr_balmer,
                    anchor=balmer_anchor, **dust_kwargs,
                )
            _ANCHOR_LABELS = {
                "Ha": "Hα", "HBETA": "Hβ", "HGAMMA": "Hγ",
                "HDELTA": "Hδ", "H9": "H9", "H10": "H10",
            }
            anchor_label = _ANCHOR_LABELS.get(
                balmer_out["anchor"], balmer_out["anchor"]
            )
            if balmer_out["n_lines"] > 0:
                Av_derived = balmer_out["Av"]
                Av_err_derived = balmer_out["Av_err"]
                _balmer_info = balmer_out
                for r in balmer_out["individual"]:
                    logger.info(
                        "A_V from %s/%s = %.3f +/- %.3f  (obs ratio = %.4f, "
                        "intrinsic = %.4f)",
                        r["line"], anchor_label,
                        r["Av"], r["Av_err"],
                        r["observed_ratio"], r["intrinsic_ratio"],
                    )
                logger.info(
                    "A_V weighted mean (anchor=%s) = %.3f +/- %.3f (%d lines).",
                    anchor_label,
                    balmer_out["Av"], balmer_out["Av_err"],
                    balmer_out["n_lines"],
                )
            else:
                Av_derived = 0.0
                logger.info(
                    "No Balmer pair available for A_V (anchor=%s); assuming A_V=0.",
                    anchor_label,
                )
        else:
            Av_derived = Av
            if Av_err is not None:
                Av_err_derived = Av_err

        if Av_derived > 0:
            fluxes, errors = _apply_dust_correction(
                fluxes, errors, Av_derived, dust_law, **dust_kwargs
            )
            # Also correct posteriors if available.
            # When Av_err is set, posteriors are kept raw (observed) so
            # that _run_direct_mcmc can resample A_V per draw.
            if posteriors and not (Av_err is not None and Av_err > 0):
                for name in list(posteriors.keys()):
                    wave = _LINE_WAVES.get(name)
                    if wave is None:
                        continue
                    from .dust import salim_attenuation, cardelli_extinction
                    wave_arr = np.array([wave])
                    if dust_law == "salim":
                        A_lam = salim_attenuation(wave_arr, Av_derived, **dust_kwargs)[0]
                    else:
                        A_lam = cardelli_extinction(wave_arr, Av_derived)[0]
                    posteriors[name] = posteriors[name] * 10.0 ** (0.4 * A_lam)
    else:
        Av_derived = Av  # store for the result even if not applied

    # --- SNR gating on individual lines ---
    excluded_lines: list[str] = []
    if snr_line > 0:
        fluxes, errors, excluded_lines = _filter_low_snr(
            fluxes, errors, snr_line,
        )
        if excluded_lines:
            logger.info(
                "Excluded %d lines below SNR=%.1f: %s",
                len(excluded_lines), snr_line, excluded_lines,
            )
        # Also filter posteriors to match.
        if posteriors:
            for name in excluded_lines:
                posteriors.pop(name, None)

    # --- NIV] doublet ratio validity check ---
    # NIV] 1483 (³P₂→¹S₀, M2) and 1486 (³P₁→¹S₀, E1 intercombination).
    # The ratio F(1483)/F(1486) is density-sensitive:
    #   - Low density (ne < 1e4):  ratio ≈ 1.50 (1483 brighter, g=5 vs 3)
    #   - High density (ne > 5e4): ratio < 1.0 (1483 collisionally de-excited)
    # Physical range: ~0 to ~1.53.  Reject if outside this with margin.
    _niv_rejected = False
    _niv1483 = fluxes.get("NIV_1483", 0.0)
    _niv1486 = fluxes.get("NIV_1486", 0.0)
    if _niv1483 > 0 and _niv1486 > 0:
        niv_ratio = _niv1483 / _niv1486
        if niv_ratio > 1.7:
            logger.warning(
                "NIV] ratio F(1483)/F(1486) = %.2f > 1.7 — exceeds physical "
                "low-density limit (~1.5); rejecting NIV] as a density "
                "diagnostic but KEEPING the summed flux for N³⁺/H⁺ "
                "(evaluated at the fallback intermediate/low density, to "
                "which N³⁺/H⁺ is only weakly sensitive).",
                niv_ratio,
            )
            # Only the *ratio* (density) is unphysical — usually noise in the
            # individual members.  The summed flux F(1483)+F(1486) is still a
            # valid measurement of the N³⁺ emission, so we keep both lines and
            # let the N³⁺/H⁺ abundance be computed with a fallback density
            # rather than throwing the ion (and ICF 5) away entirely.
            _niv_rejected = True
        else:
            logger.info(
                "NIV] ratio F(1483)/F(1486) = %.2f (physical range 0–1.5).",
                niv_ratio,
            )

    # --- Continuum-RMS flux limits for upper limits ---
    continuum_rms_limits = _compute_continuum_rms_limits(
        result, z, Av_derived, dust_law, **dust_kwargs,
    )

    # --- Method selection ---
    use_direct = False
    use_forward = False
    if method == "direct":
        use_direct = True
    elif method == "forward":
        use_forward = True
    elif method == "auto":
        # Check if [OIII] 4363 or O III] 1666 has sufficient SNR for direct method.
        has_auroral = False
        if "OIII_4363" in fluxes and "OIII_4363" in errors:
            snr_4363 = fluxes["OIII_4363"] / errors["OIII_4363"] if errors["OIII_4363"] > 0 else 0.0
            if snr_4363 >= snr_auroral:
                has_auroral = True
                logger.info("[OIII] 4363 SNR=%.1f >= %.1f; using direct method.", snr_4363, snr_auroral)
            else:
                logger.info("[OIII] 4363 SNR=%.1f < %.1f.", snr_4363, snr_auroral)
        if not has_auroral and "OIII_1666" in fluxes and "OIII_1666" in errors:
            snr_1666 = fluxes["OIII_1666"] / errors["OIII_1666"] if errors["OIII_1666"] > 0 else 0.0
            if snr_1666 >= snr_auroral:
                has_auroral = True
                logger.info("O III] 1666 SNR=%.1f >= %.1f; using direct method (UV auroral).", snr_1666, snr_auroral)
            else:
                logger.info("O III] 1666 SNR=%.1f < %.1f.", snr_1666, snr_auroral)
        if has_auroral:
            use_direct = True
        else:
            logger.info("No auroral line available; using strong-line method.")
    elif method != "strong_line":
        raise ValueError(
            f"Unknown method: {method!r}. "
            "Use 'auto', 'direct', 'forward', or 'strong_line'."
        )

    # --- Forward model ---
    if use_forward:
        from .forward import forward_model

        fwd_out = forward_model(
            fluxes, errors,
            sampler=forward_sampler,
            n_walkers=forward_n_walkers,
            n_steps=forward_n_steps,
            n_burn=forward_n_burn,
            n_live=forward_n_live,
            seed=forward_seed,
            progress=progress and forward_progress,
        )

        return AbundanceResult(
            method="forward",
            OH=fwd_out["OH"],
            OH_err=fwd_out.get("OH_err", np.nan),
            NO=fwd_out.get("NO"),
            NO_err=fwd_out.get("NO_err"),
            CO=fwd_out.get("CO"),
            CO_err=fwd_out.get("CO_err"),
            Te_high=fwd_out.get("Te"),
            Te_low=None,
            ne=fwd_out.get("ne"),
            Av=Av_derived,
            Av_err=Av_err_derived,
            ionic=fwd_out.get("ionic"),
            OH_posterior=fwd_out.get("OH_posterior"),
            NO_posterior=fwd_out.get("NO_posterior"),
            CO_posterior=fwd_out.get("CO_posterior"),
            NeO=fwd_out.get("NeO"),
            excluded_lines=excluded_lines if excluded_lines else None,
            _forward_result=fwd_out,
        )

    # --- Direct method ---
    primary_result = None

    if use_direct:
        if is_mcmc and posteriors and "OIII_4363" in posteriors:
            direct_out = _run_direct_mcmc(
                posteriors, Te_relation, n_posterior=n_posterior,
                progress=progress, ne_high_max=ne_high_max,
                snr_ne=snr_ne, snr_logU=snr_logU, marginalize_logU=marginalize_logU,
                icf_method=icf_method, co_icf_method=co_icf_method, snr_NO=snr_NO, icf_tier=icf_tier,
                continuum_rms_limits=continuum_rms_limits,
                niv_rejected=_niv_rejected,
                ne_low_override=ne_low_override,
                ne_mid_override=ne_mid_override,
                ne_high_override=ne_high_override,
                z=_z_ne,
                Av=Av_derived, Av_err=Av_err,
                Av_prior=Av_prior, dust_law=dust_law,
                dust_kwargs=dust_kwargs,
            )
        else:
            direct_out = _run_direct(
                fluxes, errors, Te_relation, n_mc,
                progress=progress, ne_high_max=ne_high_max,
                snr_ne=snr_ne, snr_logU=snr_logU, marginalize_logU=marginalize_logU,
                icf_method=icf_method, co_icf_method=co_icf_method, snr_NO=snr_NO, icf_tier=icf_tier,
                continuum_rms_limits=continuum_rms_limits,
                niv_rejected=_niv_rejected,
                ne_low_override=ne_low_override,
                ne_mid_override=ne_mid_override,
                ne_high_override=ne_high_override,
                z=_z_ne,
            )

        primary_result = AbundanceResult(
            method="direct",
            OH=direct_out["OH"],
            OH_err=direct_out["OH_err"],
            NO=direct_out.get("NO"),
            NO_err=direct_out.get("NO_err"),
            CO=direct_out.get("CO"),
            CO_err=direct_out.get("CO_err"),
            Te_high=direct_out.get("Te_high"),
            Te_high_err=direct_out.get("Te_high_err"),
            Te_mid=direct_out.get("Te_mid"),
            Te_mid_err=direct_out.get("Te_mid_err"),
            Te_low=direct_out.get("Te_low"),
            Te_low_err=direct_out.get("Te_low_err"),
            ne=direct_out.get("ne"),
            Av=Av_derived,
            Av_err=Av_err_derived,
            Av_posterior=direct_out.get("Av_posterior"),
            ionic=direct_out.get("ionic"),
            ionic_upper_limits=direct_out.get("ionic_upper_limits"),
            ionic_ul_details=direct_out.get("ionic_ul_details"),
            OH_posterior=direct_out.get("OH_posterior"),
            NO_posterior=direct_out.get("NO_posterior"),
            CO_posterior=direct_out.get("CO_posterior"),
            SO=direct_out.get("SO"),
            SO_err=direct_out.get("SO_err"),
            NeO=direct_out.get("NeO"),
            NeO_err=direct_out.get("NeO_err"),
            ArO=direct_out.get("ArO"),
            ArO_err=direct_out.get("ArO_err"),
            logU=direct_out.get("logU"),
            logU_err=direct_out.get("logU_err"),
            ne_low=direct_out.get("ne_low"),
            ne_mid=direct_out.get("ne_mid"),
            ne_high=direct_out.get("ne_high"),
            ne_Opp=direct_out.get("ne_Opp"),
            icf_method=direct_out.get("icf_method"),
            NO_icf_name=direct_out.get("NO_icf_name"),
            excluded_lines=excluded_lines if excluded_lines else None,
            NO_tiers=direct_out.get("NO_tiers"),
            icf_values=direct_out.get("icf_values"),
            failures=direct_out.get("failures"),
            diagnostics=direct_out.get("diagnostics"),
        )

    # Inject per-line Balmer decrement details into diagnostics.
    if _balmer_info and primary_result is not None and primary_result.diagnostics is not None:
        anchor_label = "Hα" if _balmer_info.get("anchor") == "Ha" else "Hβ"
        parts = []
        for r in _balmer_info["individual"]:
            parts.append(
                f"{r['line']}/{anchor_label} → A_V={r['Av']:.3f}±{r['Av_err']:.3f}"
            )
        primary_result.diagnostics["A_V"] = (
            f"weighted mean of {_balmer_info['n_lines']} decrements "
            f"(anchor={anchor_label}): " + "; ".join(parts)
        )

    if primary_result is None:
        # --- Strong-line method ---
        primary_result = _run_strong_line(
            fluxes, errors, is_mcmc, posteriors, n_mc, n_posterior,
            progress, Av_derived, Av_err_derived, excluded_lines,
        )

    # --- Auto mode: run the alternative method for comparison ---
    if method == "auto" and primary_result.alt_results is None:
        alt = {}
        if primary_result.method == "direct":
            # Also run strong-line for comparison.
            try:
                alt["strong_line"] = _run_strong_line(
                    fluxes, errors, is_mcmc, posteriors, n_mc, n_posterior,
                    progress, Av_derived, Av_err_derived, excluded_lines,
                )
            except Exception:
                logger.info("Alternative strong-line method failed; skipping.")
            # If primary used 4363, also compute Te from 1666 as an alternative.
            f_1666_alt = fluxes.get("OIII_1666", 0.0)
            f_5007_alt = fluxes.get("OIII_5007", 0.0)
            _has_1666_flux = f_1666_alt > 0 and f_5007_alt > 0
            if _has_1666_flux and fluxes.get("OIII_4363", 0.0) > 0:
                # Point estimate only.  This is a cross-check on the adopted
                # lambda4363 temperature, so it does not warrant a second
                # full MC/posterior pass over the direct method — that would
                # roughly double the runtime of compute_abundances.  Solve
                # T_e once from the dust-corrected fluxes and propagate it
                # deterministically through the ionic abundances; no error
                # bar is reported, by design.
                from .direct import (
                    NE_DEFAULT,
                    Te_int_from_high,
                    Te_low_from_high,
                    compute_ionic_abundances,
                    compute_Te_OIII_1666,
                )
                try:
                    _ne_lo = primary_result.ne_low or NE_DEFAULT
                    Te_1666_pt = compute_Te_OIII_1666(
                        f_1666_alt, f_5007_alt,
                        fluxes.get("OIII_4959", 0.0),
                        primary_result.ne_mid or _ne_lo,
                    )
                    Te_1666_low = Te_low_from_high(Te_1666_pt, relation=Te_relation)
                    Te_1666_int = Te_int_from_high(Te_1666_pt, relation=Te_relation)
                    ionic_1666 = compute_ionic_abundances(
                        fluxes, Te_1666_pt, Te_1666_low, _ne_lo,
                        ne_mid=primary_result.ne_mid,
                        ne_high=primary_result.ne_high,
                        Te_int=Te_1666_int,
                    )
                    OH_1666 = ionic_1666.get("O+/H+", 0.0) + ionic_1666.get("O++/H+", 0.0)
                    alt["direct_1666"] = AbundanceResult(
                        method="direct (O III] 1666, point estimate)",
                        OH=12.0 + np.log10(OH_1666) if OH_1666 > 0 else np.nan,
                        Te_high=Te_1666_pt,
                        Te_low=Te_1666_low,
                        Te_mid=Te_1666_int,
                        Av=Av_derived,
                        Av_err=Av_err_derived,
                        ionic=ionic_1666,
                    )
                except (ValueError, RuntimeError) as e:
                    logger.info("Point-estimate T_e from 1666 failed: %s", e)
        elif primary_result.method == "strong_line":
            # Also try direct if 4363 is present (even if SNR was below threshold).
            has_auroral_alt = (
                ("OIII_4363" in fluxes and fluxes.get("OIII_4363", 0) > 0)
                or ("OIII_1666" in fluxes and fluxes.get("OIII_1666", 0) > 0)
            )
            if has_auroral_alt:
                try:
                    if is_mcmc and posteriors and ("OIII_4363" in posteriors or "OIII_1666" in posteriors):
                        d_out = _run_direct_mcmc(
                            posteriors, Te_relation, n_posterior=n_posterior,
                            progress=progress, ne_high_max=ne_high_max,
                            snr_ne=snr_ne, snr_logU=snr_logU, marginalize_logU=marginalize_logU,
                            icf_method=icf_method, co_icf_method=co_icf_method, snr_NO=snr_NO, icf_tier=icf_tier,
                            niv_rejected=_niv_rejected,
                            ne_low_override=ne_low_override,
                            ne_mid_override=ne_mid_override,
                            ne_high_override=ne_high_override,
                            z=_z_ne,
                            Av=Av_derived, Av_err=Av_err,
                            Av_prior=Av_prior, dust_law=dust_law,
                            dust_kwargs=dust_kwargs,
                        )
                    else:
                        d_out = _run_direct(
                            fluxes, errors, Te_relation, n_mc,
                            progress=progress, ne_high_max=ne_high_max,
                            snr_ne=snr_ne, snr_logU=snr_logU, marginalize_logU=marginalize_logU,
                            icf_method=icf_method, co_icf_method=co_icf_method, snr_NO=snr_NO, icf_tier=icf_tier,
                            niv_rejected=_niv_rejected,
                            ne_low_override=ne_low_override,
                            ne_mid_override=ne_mid_override,
                            ne_high_override=ne_high_override,
                            z=_z_ne,
                        )
                    alt["direct"] = AbundanceResult(
                        method="direct",
                        OH=d_out["OH"],
                        OH_err=d_out["OH_err"],
                        NO=d_out.get("NO"),
                        NO_err=d_out.get("NO_err"),
                        CO=d_out.get("CO"),
                        CO_err=d_out.get("CO_err"),
                        Te_high=d_out.get("Te_high"),
                        Te_low=d_out.get("Te_low"),
                        ne=d_out.get("ne"),
                        Av=Av_derived,
                        Av_err=Av_err_derived,
                        ionic=d_out.get("ionic"),
                        failures=d_out.get("failures"),
                    )
                except Exception:
                    logger.info("Alternative direct method failed; skipping.")
        if alt:
            primary_result.alt_results = alt

    # --- Lyα escape fraction ---
    if _lya_obs_flux > 0 and _lya_obs_err > 0:
        _Av_for_esc = Av_derived if Av_derived is not None else 0.0
        _Av_err_for_esc = Av_err_derived if Av_err_derived is not None and np.isfinite(Av_err_derived) else 0.0

        if is_mcmc:
            lya_esc_out = compute_lya_escape_fraction_mc(
                _lya_obs_flux, _lya_obs_err,
                fluxes, errors,
                Av=_Av_for_esc, Av_err=_Av_err_for_esc,
                Av_prior=Av_prior,
                dust_law=dust_law, n_mc=n_mc,
                **dust_kwargs,
            )
            if lya_esc_out["n_lines"] > 0:
                primary_result.lya_f_esc = lya_esc_out["f_esc"]
                primary_result.lya_f_esc_err = lya_esc_out["f_esc_err"]
                primary_result.lya_f_esc_posterior = lya_esc_out["f_esc_posterior"]
                primary_result.lya_f_esc_details = lya_esc_out
                logger.info(
                    "Lyα f_esc = %.3f (+%.3f/-%.3f) from %d Balmer lines (MC).",
                    lya_esc_out["f_esc"],
                    lya_esc_out["f_esc_err"][1],
                    lya_esc_out["f_esc_err"][0],
                    lya_esc_out["n_lines"],
                )
        else:
            lya_esc_out = compute_lya_escape_fraction(
                _lya_obs_flux, _lya_obs_err,
                fluxes, errors,
            )
            if lya_esc_out["n_lines"] > 0:
                primary_result.lya_f_esc = lya_esc_out["f_esc"]
                primary_result.lya_f_esc_err = lya_esc_out["f_esc_err"]
                primary_result.lya_f_esc_details = lya_esc_out
                logger.info(
                    "Lyα f_esc = %.3f +/- %.3f from %d Balmer lines.",
                    lya_esc_out["f_esc"],
                    lya_esc_out["f_esc_err"],
                    lya_esc_out["n_lines"],
                )

    return primary_result
