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

from .dust import compute_Av_from_balmer, dust_correct_fluxes
from .result import AbundanceResult

logger = logging.getLogger(__name__)

# Line name to rest wavelength mapping for dust correction.
_LINE_WAVES: dict[str, float] = {
    name: wave for name, wave in REST_LINES_A.items()
}

# Balmer lines whose broad components should be summed with the narrow.
_BALMER_LINES = {"Ha", "HBETA", "HGAMMA", "HDELTA"}

# Lines that must never be SNR-filtered (required for Te computation
# and flux normalisation).
_SNR_PROTECTED = {"OIII_4363", "OIII_5007", "OIII_4959", "HBETA"}


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


def _compute_multi_ne(
    fluxes: dict[str, float],
    ne_high_max: float = 1e5,
) -> tuple[float, float]:
    """Compute multi-phase electron densities (Berg+2025 step 1).

    Parameters
    ----------
    fluxes : dict
        Dust-corrected emission-line fluxes.
    ne_high_max : float
        Maximum allowed high-ionisation electron density in cm^-3
        (default 1e5).  If n_e(high) exceeds this, falls back to
        n_e(low).

    Returns
    -------
    tuple
        ``(ne_low, ne_high)`` in cm^-3.  ``ne_low`` is from [SII] or
        [OII]; ``ne_high`` is from NIV] or CIII].  Falls back to
        ne_low or 100 cm^-3 when UV diagnostics are unavailable.
    """
    from .direct import compute_ne, compute_ne_CIII, compute_ne_NIV

    # Low-ionisation zone: [SII] 6718/6732 or [OII] 3726/3729.
    ne_low = 100.0
    if "SII_6718" in fluxes and "SII_6732" in fluxes:
        try:
            ne_low = compute_ne(fluxes["SII_6718"], fluxes["SII_6732"], doublet="SII")
        except Exception:
            logger.warning("n_e(SII) failed; using 100 cm^-3.")
    elif "OII_3726" in fluxes and "OII_3729" in fluxes:
        try:
            ne_low = compute_ne(fluxes["OII_3726"], fluxes["OII_3729"], doublet="OII")
        except Exception:
            logger.warning("n_e(OII) failed; using 100 cm^-3.")

    # High-ionisation zone: NIV] 1483/1486 (preferred) or CIII] 1907/1909.
    ne_high = None
    if "NIV_1483" in fluxes and "NIV_1486" in fluxes:
        try:
            ne_high = compute_ne_NIV(fluxes["NIV_1483"], fluxes["NIV_1486"])
            logger.info("n_e(high) from NIV] = %.0f cm^-3.", ne_high)
        except Exception:
            logger.warning("n_e(NIV]) failed.")
    if ne_high is None and "CIII]_1907" in fluxes and "CIII]" in fluxes:
        try:
            ne_high = compute_ne_CIII(fluxes["CIII]_1907"], fluxes["CIII]"])
            logger.info("n_e(high) from CIII] = %.0f cm^-3.", ne_high)
        except Exception:
            logger.warning("n_e(CIII]) failed.")

    # Fall back to ne_low if no high-ionisation density available.
    if ne_high is None:
        ne_high = ne_low

    # Clamp ne_high if it exceeds the maximum (prevents unphysical
    # density from noisy doublet ratios).
    if ne_high > ne_high_max:
        logger.warning(
            "n_e(high) = %.0f cm^-3 exceeds ne_high_max=%.0f; "
            "falling back to n_e(low) = %.0f cm^-3.",
            ne_high, ne_high_max, ne_low,
        )
        ne_high = ne_low

    return ne_low, ne_high


def _compute_logU(
    fluxes: dict[str, float],
    Z_Zsun: float,
    ne_high: float,
) -> tuple[float | None, str | None]:
    """Compute ionisation parameter (Berg+2025 step 5).

    Parameters
    ----------
    fluxes : dict
        Dust-corrected emission-line fluxes.
    Z_Zsun : float
        Gas-phase metallicity in solar units.
    ne_high : float
        High-ionisation zone electron density in cm^-3.

    Returns
    -------
    tuple
        ``(logU, diagnostic)`` where diagnostic is ``"N43"`` or
        ``"O32"``.  Returns ``(None, None)`` if neither diagnostic
        is available.
    """
    from .martinez25_icf import LOG_OH_SOLAR, log_U_from_N43, log_U_from_O32

    # N43 = NIV]1486 / NIII]1750 — density-insensitive, recommended.
    niv_flux = 0.0
    for name in ("NIV_1483", "NIV_1486"):
        if name in fluxes and fluxes[name] > 0:
            niv_flux += fluxes[name]
    niii_flux = 0.0
    for name in ("NIII_1749", "NIII_1752"):
        if name in fluxes and fluxes[name] > 0:
            niii_flux += fluxes[name]

    if niv_flux > 0 and niii_flux > 0:
        N43 = niv_flux / niii_flux
        logU = log_U_from_N43(np.log10(N43), Z_Zsun, ne_high)
        logger.info("log(U) from N43 = %.2f (N43=%.3f).", logU, N43)
        return logU, "N43"

    # O32 = [OIII]5007 / [OII]3727 — density-sensitive fallback.
    oiii = fluxes.get("OIII_5007", 0.0)
    oii = 0.0
    if "OII_3726" in fluxes and "OII_3729" in fluxes:
        oii = fluxes["OII_3726"] + fluxes["OII_3729"]
    elif "OII_doublet" in fluxes:
        oii = fluxes["OII_doublet"]

    if oiii > 0 and oii > 0:
        O32 = oiii / oii
        logU = log_U_from_O32(np.log10(O32), Z_Zsun, ne_high)
        logger.info("log(U) from O32 = %.2f (O32=%.3f).", logU, O32)
        return logU, "O32"

    return None, None


def _run_direct(
    fluxes: dict[str, float],
    errors: dict[str, float],
    Te_relation: str,
    n_mc: int,
    seed: int = 42,
    progress: bool = True,
    ne_high_max: float = 1e5,
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
        Maximum allowed n_e(high) in cm^-3 (default 1e5).

    Returns
    -------
    dict
        Keys: OH, OH_err, NO, NO_err, Te_high, Te_low, ne, ne_low,
        ne_high, logU, icf_method, ionic, posteriors, etc.
    """
    from .direct import (
        Te_low_from_high,
        compute_ionic_abundances,
        compute_Te_OIII,
        compute_total_abundances,
    )
    from .martinez25_icf import LOG_OH_SOLAR, _LOG_U_VALID

    # --- Step 1: Multi-phase electron density ---
    ne_low, ne_high = _compute_multi_ne(fluxes, ne_high_max=ne_high_max)

    # --- Step 2: Electron temperature with zone-appropriate ne ---
    Te_high = compute_Te_OIII(
        fluxes["OIII_4363"], fluxes["OIII_5007"], fluxes["OIII_4959"], ne_high
    )
    Te_low = Te_low_from_high(Te_high, relation=Te_relation)

    # --- Step 3: Ionic abundances with zone-appropriate ne ---
    ionic = compute_ionic_abundances(fluxes, Te_high, Te_low, ne_low, ne_high=ne_high)

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
    if Z_Zsun is not None:
        logU, logU_diag = _compute_logU(fluxes, Z_Zsun, ne_high)

    # --- Step 6: Total abundances with ICFs ---
    totals = compute_total_abundances(
        ionic, logU=logU, Z_Zsun=Z_Zsun, ne=ne_high,
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

    # --- MC error propagation (all 6 steps per iteration) ---
    rng = np.random.default_rng(seed)
    OH_mc = []
    NO_mc = []
    CO_mc = []

    for _ in tqdm(range(n_mc), desc="Direct Te (MC)", disable=not progress):
        mc_fluxes = {}
        for name in fluxes:
            mc_fluxes[name] = rng.normal(fluxes[name], errors.get(name, 0.0))
            mc_fluxes[name] = max(mc_fluxes[name], 1e-50)

        try:
            # Use fixed ne (varying ne per MC iteration adds noise
            # without improving accuracy for the density diagnostics).
            Te_h = compute_Te_OIII(
                mc_fluxes.get("OIII_4363", 0),
                mc_fluxes.get("OIII_5007", 0),
                mc_fluxes.get("OIII_4959", 0),
                ne_high,
            )
            Te_l = Te_low_from_high(Te_h, relation=Te_relation)
            ionic_mc = compute_ionic_abundances(
                mc_fluxes, Te_h, Te_l, ne_low, ne_high=ne_high,
            )

            # Compute Z_Zsun for this MC iteration.
            oh_val = ionic_mc.get("O+/H+", 0.0) + ionic_mc.get("O++/H+", 0.0)
            if oh_val > 0:
                z_zsun_mc = 10.0 ** (12.0 + np.log10(oh_val) - LOG_OH_SOLAR)
            else:
                z_zsun_mc = Z_Zsun  # fallback to point estimate

            # Compute logU for this MC iteration (clamped to validity
            # range to prevent wild ICF extrapolation).
            logU_mc = logU  # default to point estimate
            if z_zsun_mc is not None and logU_diag is not None:
                logU_mc_val, _ = _compute_logU(mc_fluxes, z_zsun_mc, ne_high)
                if logU_mc_val is not None:
                    logU_mc = float(np.clip(logU_mc_val, *_LOG_U_VALID))

            totals_mc = compute_total_abundances(
                ionic_mc, logU=logU_mc, Z_Zsun=z_zsun_mc, ne=ne_high,
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
        except (ValueError, RuntimeError):
            OH_mc.append(np.nan)
            NO_mc.append(np.nan)
            CO_mc.append(np.nan)

    OH_mc = np.array(OH_mc)
    NO_mc = np.array(NO_mc)
    CO_mc = np.array(CO_mc)

    OH_err = float(np.nanstd(OH_mc)) if np.any(np.isfinite(OH_mc)) else np.nan
    NO_err = float(np.nanstd(NO_mc)) if np.any(np.isfinite(NO_mc)) else None
    CO_err = float(np.nanstd(CO_mc)) if np.any(np.isfinite(CO_mc)) else None

    return {
        "OH": OH_12,
        "OH_err": OH_err,
        "NO": NO_log,
        "NO_err": NO_err,
        "CO": CO_log,
        "CO_err": CO_err,
        "Te_high": Te_high,
        "Te_low": Te_low,
        "ne": ne_low,
        "ne_low": ne_low,
        "ne_high": ne_high,
        "logU": logU,
        "icf_method": icf_method,
        "NO_icf_name": NO_icf_name,
        "ionic": ionic,
        "OH_posterior": OH_mc,
        "NO_posterior": NO_mc,
        "CO_posterior": CO_mc,
        "SO": SO_log,
        "NeO": NeO_log,
        "ArO": ArO_log,
    }


def _run_direct_mcmc(
    posteriors: dict[str, np.ndarray],
    Te_relation: str,
    n_posterior: int = 1000,
    progress: bool = True,
    seed: int = 42,
    ne_high_max: float = 1e5,
) -> dict[str, Any]:
    """Run the direct T_e method on MCMC posterior samples.

    Follows Berg+2025's 6-step procedure for each posterior sample.

    Parameters
    ----------
    posteriors : dict
        ``{line_name: flux_posterior_array}``.
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
        Maximum allowed n_e(high) in cm^-3 (default 1e5).

    Returns
    -------
    dict
        Same keys as :func:`_run_direct`.
    """
    from .direct import (
        Te_low_from_high,
        compute_ionic_abundances,
        compute_Te_OIII,
        compute_total_abundances,
    )
    from .martinez25_icf import LOG_OH_SOLAR, _LOG_U_VALID

    # Determine number of samples; thin if larger than n_posterior.
    n_total = min(len(v) for v in posteriors.values())
    if n_posterior > 0 and n_total > n_posterior:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=n_posterior, replace=False)
        idx.sort()
        posteriors = {name: arr[idx] for name, arr in posteriors.items()}
        n_samples = n_posterior
    else:
        n_samples = n_total

    OH_post = np.full(n_samples, np.nan)
    NO_post = np.full(n_samples, np.nan)
    CO_post = np.full(n_samples, np.nan)

    # Compute medians for the point estimate and multi-phase ne.
    med_fluxes = {name: float(np.median(post)) for name, post in posteriors.items()}
    ne_low, ne_high = _compute_multi_ne(med_fluxes, ne_high_max=ne_high_max)

    # Point estimate: logU and Z_Zsun from medians.
    try:
        Te_high_pt = compute_Te_OIII(
            med_fluxes.get("OIII_4363", 0),
            med_fluxes.get("OIII_5007", 0),
            med_fluxes.get("OIII_4959", 0),
            ne_high,
        )
    except ValueError:
        Te_high_pt = np.nan
    Te_low_pt = Te_low_from_high(Te_high_pt, relation=Te_relation) if np.isfinite(Te_high_pt) else np.nan

    ionic_pt = compute_ionic_abundances(
        med_fluxes, Te_high_pt, Te_low_pt, ne_low, ne_high=ne_high
    ) if np.isfinite(Te_high_pt) else {}

    OH_pt = ionic_pt.get("O+/H+", 0.0) + ionic_pt.get("O++/H+", 0.0)
    Z_Zsun_pt = 10.0 ** (12.0 + np.log10(OH_pt) - LOG_OH_SOLAR) if OH_pt > 0 else None
    logU_pt = None
    logU_diag = None
    if Z_Zsun_pt is not None:
        logU_pt, logU_diag = _compute_logU(med_fluxes, Z_Zsun_pt, ne_high)

    totals_pt = compute_total_abundances(
        ionic_pt, logU=logU_pt, Z_Zsun=Z_Zsun_pt, ne=ne_high,
    ) if ionic_pt else {}
    icf_method = totals_pt.get("icf_method")
    NO_icf_name = totals_pt.get("NO_icf_name")

    for i in tqdm(range(n_samples), desc="Direct Te (posterior)", disable=not progress):
        sample = {name: max(float(post[i]), 1e-50) for name, post in posteriors.items()}
        try:
            Te_h = compute_Te_OIII(
                sample.get("OIII_4363", 0),
                sample.get("OIII_5007", 0),
                sample.get("OIII_4959", 0),
                ne_high,
            )
            Te_l = Te_low_from_high(Te_h, relation=Te_relation)
            ionic_i = compute_ionic_abundances(
                sample, Te_h, Te_l, ne_low, ne_high=ne_high,
            )

            # Z_Zsun for this sample.
            oh_val = ionic_i.get("O+/H+", 0.0) + ionic_i.get("O++/H+", 0.0)
            z_zsun_i = 10.0 ** (12.0 + np.log10(oh_val) - LOG_OH_SOLAR) if oh_val > 0 else Z_Zsun_pt

            # logU for this sample (clamped to validity range to
            # prevent wild ICF extrapolation).
            logU_i = logU_pt
            if z_zsun_i is not None and logU_diag is not None:
                logU_val, _ = _compute_logU(sample, z_zsun_i, ne_high)
                if logU_val is not None:
                    logU_i = float(np.clip(logU_val, *_LOG_U_VALID))

            totals_i = compute_total_abundances(
                ionic_i, logU=logU_i, Z_Zsun=z_zsun_i, ne=ne_high,
            )

            oh = totals_i.get("O/H", np.nan)
            if np.isfinite(oh) and oh > 0:
                OH_post[i] = 12.0 + np.log10(oh)

            no = totals_i.get("N/O", np.nan)
            if no is not None and np.isfinite(no) and no > 0:
                NO_post[i] = np.log10(no)

            co = totals_i.get("C/O", np.nan)
            if co is not None and np.isfinite(co) and co > 0:
                CO_post[i] = np.log10(co)
        except (ValueError, RuntimeError):
            continue

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

    return {
        "OH": OH_med,
        "OH_err": (OH_lo, OH_hi),
        "NO": NO_med,
        "NO_err": (NO_lo, NO_hi) if NO_lo is not None else None,
        "CO": CO_med,
        "CO_err": (CO_lo, CO_hi) if CO_lo is not None else None,
        "Te_high": Te_high_pt if np.isfinite(Te_high_pt) else None,
        "Te_low": Te_low_pt if np.isfinite(Te_low_pt) else None,
        "ne": ne_low,
        "ne_low": ne_low,
        "ne_high": ne_high,
        "logU": logU_pt,
        "icf_method": icf_method,
        "NO_icf_name": NO_icf_name,
        "ionic": ionic_pt if ionic_pt else None,
        "OH_posterior": OH_post,
        "NO_posterior": NO_post if np.any(np.isfinite(NO_post)) else None,
        "CO_posterior": CO_post if np.any(np.isfinite(CO_post)) else None,
        "SO": np.log10(totals_pt["S/O"]) if "S/O" in totals_pt and totals_pt["S/O"] > 0 else None,
        "NeO": np.log10(totals_pt["Ne/O"]) if "Ne/O" in totals_pt and totals_pt["Ne/O"] > 0 else None,
        "ArO": np.log10(totals_pt["Ar/O"]) if "Ar/O" in totals_pt and totals_pt["Ar/O"] > 0 else None,
    }


def compute_abundances(
    result: Any,
    z: float,
    *,
    dust_correct: bool = True,
    dust_law: str = "salim",
    Av: float | None = None,
    method: str = "auto",
    snr_auroral: float = 3.0,
    snr_line: float = 2.0,
    ne_high_max: float = 1e5,
    n_mc: int = 1000,
    Te_relation: str = "desi",
    Rv: float = 3.15,
    delta: float = -0.35,
    B_bump: float = 2.27,
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
        V-band attenuation. If ``None``, derived from Balmer decrement.
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
        (default 1e5).  If n_e(high) from a UV doublet exceeds this,
        the code falls back to n_e(low).  Prevents unphysical density
        estimates from noisy doublet ratios.
    n_mc : int
        Monte Carlo iterations for error propagation (default 1000).
    Te_relation : str
        T_e-T_e relation: ``"desi"`` (default) or ``"classical"``.
    Rv : float
        Total-to-selective ratio for Salim law (default 3.15).
    delta : float
        Slope deviation for Salim law (default -0.35).
    B_bump : float
        UV bump strength for Salim law (default 2.27).
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
    # --- Extract fluxes ---
    fluxes, errors, is_mcmc = _extract_fluxes(result)
    posteriors = _extract_posteriors(result) if is_mcmc else {}

    # --- Dust correction ---
    dust_kwargs = {}
    if dust_law == "salim":
        dust_kwargs = {"Rv": Rv, "delta": delta, "B": B_bump}

    Av_derived = None
    if dust_correct:
        if Av is None:
            # Derive A_V from Balmer decrement (Hgamma/Hbeta).
            if "HGAMMA" in fluxes and "HBETA" in fluxes and fluxes["HGAMMA"] > 0 and fluxes["HBETA"] > 0:
                Av_val, Av_err = compute_Av_from_balmer(
                    fluxes["HGAMMA"], fluxes["HBETA"],
                    errors["HGAMMA"], errors["HBETA"],
                    law=dust_law,
                    intrinsic_ratio=0.468,
                    wave_num_A=REST_LINES_A.get("HGAMMA", 4341.68),
                    wave_den_A=REST_LINES_A.get("HBETA", 4862.68),
                    **dust_kwargs,
                )
                Av_derived = Av_val
                logger.info("A_V from Hg/Hb = %.3f +/- %.3f", Av_val, Av_err)
            else:
                Av_derived = 0.0
                logger.info("No Balmer pair available for A_V; assuming A_V=0.")
        else:
            Av_derived = Av

        if Av_derived > 0:
            fluxes, errors = _apply_dust_correction(
                fluxes, errors, Av_derived, dust_law, **dust_kwargs
            )
            # Also correct posteriors if available.
            if posteriors:
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

    # --- Method selection ---
    use_direct = False
    use_forward = False
    if method == "direct":
        use_direct = True
    elif method == "forward":
        use_forward = True
    elif method == "auto":
        # Check if [OIII] 4363 has sufficient SNR.
        if "OIII_4363" in fluxes and "OIII_4363" in errors:
            snr_4363 = fluxes["OIII_4363"] / errors["OIII_4363"] if errors["OIII_4363"] > 0 else 0.0
            if snr_4363 >= snr_auroral:
                use_direct = True
                logger.info("[OIII] 4363 SNR=%.1f >= %.1f; using direct method.", snr_4363, snr_auroral)
            else:
                logger.info("[OIII] 4363 SNR=%.1f < %.1f; using strong-line method.", snr_4363, snr_auroral)
        else:
            logger.info("[OIII] 4363 not detected; using strong-line method.")
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
            ionic=fwd_out.get("ionic"),
            OH_posterior=fwd_out.get("OH_posterior"),
            NO_posterior=fwd_out.get("NO_posterior"),
            CO_posterior=fwd_out.get("CO_posterior"),
            NeO=fwd_out.get("NeO"),
            excluded_lines=excluded_lines if excluded_lines else None,
            _forward_result=fwd_out,
        )

    # --- Direct method ---
    if use_direct:
        if is_mcmc and posteriors and "OIII_4363" in posteriors:
            direct_out = _run_direct_mcmc(
                posteriors, Te_relation, n_posterior=n_posterior,
                progress=progress, ne_high_max=ne_high_max,
            )
        else:
            direct_out = _run_direct(
                fluxes, errors, Te_relation, n_mc,
                progress=progress, ne_high_max=ne_high_max,
            )

        return AbundanceResult(
            method="direct",
            OH=direct_out["OH"],
            OH_err=direct_out["OH_err"],
            NO=direct_out.get("NO"),
            NO_err=direct_out.get("NO_err"),
            CO=direct_out.get("CO"),
            CO_err=direct_out.get("CO_err"),
            Te_high=direct_out.get("Te_high"),
            Te_low=direct_out.get("Te_low"),
            ne=direct_out.get("ne"),
            Av=Av_derived,
            ionic=direct_out.get("ionic"),
            OH_posterior=direct_out.get("OH_posterior"),
            NO_posterior=direct_out.get("NO_posterior"),
            CO_posterior=direct_out.get("CO_posterior"),
            SO=direct_out.get("SO"),
            NeO=direct_out.get("NeO"),
            ArO=direct_out.get("ArO"),
            logU=direct_out.get("logU"),
            ne_low=direct_out.get("ne_low"),
            ne_high=direct_out.get("ne_high"),
            icf_method=direct_out.get("icf_method"),
            NO_icf_name=direct_out.get("NO_icf_name"),
            excluded_lines=excluded_lines if excluded_lines else None,
        )

    # --- Strong-line method ---
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

        # Thin posterior if needed.
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
                # Perturb each ratio by the calibration scatter.
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

        # Determine ratios used from median fluxes.
        from .strong_line import compute_line_ratios
        med_fluxes = {name: float(np.median(arr)) for name, arr in posteriors.items()}
        med_errors = {name: float(np.std(arr)) for name, arr in posteriors.items()}
        ratios = compute_line_ratios(med_fluxes, med_errors)

        return AbundanceResult(
            method="strong_line",
            OH=OH_med,
            OH_err=(OH_lo, OH_hi),
            Av=Av_derived,
            OH_posterior=OH_post,
            ratios_used=list(ratios.keys()),
            excluded_lines=excluded_lines if excluded_lines else None,
        )

    # LS result: use MC within sanders25_metallicity.
    Z_best, Z_lo, Z_hi, chi2, ratios_used, Z_mc = sanders25_metallicity(
        fluxes, errors, n_mc=n_mc, progress=progress,
    )

    return AbundanceResult(
        method="strong_line",
        OH=Z_best,
        OH_err=(Z_best - Z_lo, Z_hi - Z_best),
        Av=Av_derived,
        chi2=chi2,
        ratios_used=ratios_used,
        OH_posterior=Z_mc,
        excluded_lines=excluded_lines if excluded_lines else None,
    )
