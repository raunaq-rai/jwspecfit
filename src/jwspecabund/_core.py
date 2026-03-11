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


def _compute_multi_ne(
    fluxes: dict[str, float],
    errors: dict[str, float] | None = None,
    snr_ne: float = 3.0,
    ne_high_max: float = 1e5,
) -> tuple[float, float | None, float, dict[str, str]]:
    """Compute 3-zone electron densities (Berg+2025 step 1).

    Parameters
    ----------
    fluxes : dict
        Dust-corrected emission-line fluxes.
    errors : dict, optional
        Dust-corrected flux errors.  Required for SNR gating.
    snr_ne : float
        Minimum SNR for both members of a density-sensitive doublet
        (default 3.0).  If either member falls below this, the
        doublet is skipped and the code falls back to the default
        density.  Set to 0 to disable gating.
    ne_high_max : float
        Maximum allowed high-ionisation electron density in cm^-3
        (default 1e5).  If n_e(high) exceeds this, falls back to
        n_e(mid) (or n_e(low) if no mid measurement).

    Returns
    -------
    tuple
        ``(ne_low, ne_mid, ne_high, ne_failures)`` in cm^-3.

        - ``ne_low``: from [SII] 6718/6732 or [OII] 3726/3729 (~14 eV).
        - ``ne_mid``: from CIII] 1907/1909 (~24 eV); ``None`` if
          unavailable.  Used for C²⁺, N²⁺, S²⁺, Ar²⁺.
        - ``ne_high``: from NIV] 1483/1486 (~47 eV); falls back to
          ``ne_mid`` then ``ne_low``.  Used for O²⁺, Ne²⁺, N³⁺, C³⁺.
        - ``ne_failures``: dict of density solve failure reasons,
          e.g. ``{"n_e(SII)": "PyNEB solve failed (ratio out of range)"}``.
    """
    from .direct import NE_DEFAULT, compute_ne, compute_ne_CIII, compute_ne_NIV

    if errors is None:
        errors = {}

    ne_failures: dict[str, str] = {}

    # Low-ionisation zone: [SII] 6718/6732 or [OII] 3726/3729.
    ne_low = NE_DEFAULT
    if "SII_6718" in fluxes and "SII_6732" in fluxes:
        if _doublet_snr_ok("SII_6718", "SII_6732", fluxes, errors, snr_ne):
            try:
                ne_low = compute_ne(fluxes["SII_6718"], fluxes["SII_6732"], doublet="SII")
            except Exception as exc:
                logger.warning("n_e(SII) failed; using %.0f cm^-3.", NE_DEFAULT)
                ne_failures["n_e(SII)"] = f"PyNEB solve failed: {exc}"
        else:
            logger.warning(
                "n_e(SII) doublet below SNR threshold (%.1f); "
                "using %.0f cm^-3.", snr_ne, NE_DEFAULT,
            )
    elif "OII_3726" in fluxes and "OII_3729" in fluxes:
        if _doublet_snr_ok("OII_3726", "OII_3729", fluxes, errors, snr_ne):
            try:
                ne_low = compute_ne(fluxes["OII_3726"], fluxes["OII_3729"], doublet="OII")
            except Exception as exc:
                logger.warning("n_e(OII) failed; using %.0f cm^-3.", NE_DEFAULT)
                ne_failures["n_e(OII)"] = f"PyNEB solve failed: {exc}"
        else:
            logger.warning(
                "n_e(OII) doublet below SNR threshold (%.1f); "
                "using %.0f cm^-3.", snr_ne, NE_DEFAULT,
            )

    # Mid-ionisation zone: CIII] 1907/1909 (~24 eV).
    ne_mid = None
    if "CIII]_1907" in fluxes and "CIII]" in fluxes:
        if _doublet_snr_ok("CIII]_1907", "CIII]", fluxes, errors, snr_ne, combined=True):
            try:
                ne_mid = compute_ne_CIII(fluxes["CIII]_1907"], fluxes["CIII]"])
                logger.info("n_e(mid) from CIII] = %.0f cm^-3.", ne_mid)
            except Exception as exc:
                logger.warning("n_e(CIII]) failed.")
                ne_failures["n_e(CIII])"] = f"PyNEB solve failed: {exc}"
        else:
            logger.warning(
                "n_e(CIII]) doublet below SNR threshold (%.1f); skipping.",
                snr_ne,
            )

    # High-ionisation zone: NIV] 1483/1486 (~47 eV).
    ne_high_raw = None
    if "NIV_1483" in fluxes and "NIV_1486" in fluxes:
        if _doublet_snr_ok("NIV_1483", "NIV_1486", fluxes, errors, snr_ne, combined=True):
            try:
                ne_high_raw = compute_ne_NIV(fluxes["NIV_1483"], fluxes["NIV_1486"])
                logger.info("n_e(high) from NIV] = %.0f cm^-3.", ne_high_raw)
            except Exception as exc:
                logger.warning("n_e(NIV]) failed.")
                ne_failures["n_e(NIV])"] = f"PyNEB solve failed: {exc}"
        else:
            logger.warning(
                "n_e(NIV]) doublet below SNR threshold (%.1f); skipping.",
                snr_ne,
            )

    # Fallback chain: ne_high → ne_mid → ne_low.
    if ne_high_raw is not None:
        ne_high = ne_high_raw
    elif ne_mid is not None:
        ne_high = ne_mid
    else:
        ne_high = ne_low

    # Clamp ne_high if it exceeds the maximum (prevents unphysical
    # density from noisy doublet ratios).
    if ne_high > ne_high_max:
        ne_fallback = ne_mid if ne_mid is not None else ne_low
        logger.warning(
            "n_e(high) = %.0f cm^-3 exceeds ne_high_max=%.0f; "
            "falling back to %.0f cm^-3.",
            ne_high, ne_high_max, ne_fallback,
        )
        ne_high = ne_fallback

    return ne_low, ne_mid, ne_high, ne_failures


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
    errors : dict, optional
        Flux errors.  When provided, the **total doublet** SNR
        (summed flux / quadrature-summed error) must be >= *snr_logU*
        for each doublet in N43 to be used.
    snr_logU : float
        Minimum total-doublet SNR for N43 (default 3.0).

    Returns
    -------
    tuple
        ``(logU, diagnostic)`` where diagnostic is ``"N43"`` or
        ``"O32"``.  Returns ``(None, None)`` if neither diagnostic
        is available.
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
    niv_ok, niv_flux = _doublet_ok("NIV_1483", "NIV_1486")
    if not niv_ok and (fluxes.get("NIV_1483", 0) > 0 or fluxes.get("NIV_1486", 0) > 0):
        logger.info("N43: NIV] doublet below total SNR threshold (%.1f); skipping.", snr_logU)

    niii_ok, niii_flux = _doublet_ok("NIII_1749", "NIII_1752")
    if not niii_ok and (fluxes.get("NIII_1749", 0) > 0 or fluxes.get("NIII_1752", 0) > 0):
        logger.info("N43: NIII] doublet below total SNR threshold (%.1f); skipping.", snr_logU)

    if niv_flux > 0 and niii_flux > 0:
        N43 = niv_flux / niii_flux
        log_N43 = np.log10(N43)
        # Reject N43 if the ratio is outside the calibration range.
        if _LOG_N43_VALID[0] <= log_N43 <= _LOG_N43_VALID[1]:
            logU = log_U_from_N43(log_N43, Z_Zsun, ne_high)
            if _LOG_U_VALID[0] <= logU <= _LOG_U_VALID[1]:
                logger.info("log(U) from N43 = %.2f (N43=%.3f).", logU, N43)
                return logU, "N43"
            else:
                logger.debug(
                    "N43 gives log(U)=%.2f outside validity [%.1f, %.1f]; "
                    "falling back to O32.", logU, *_LOG_U_VALID,
                )
        else:
            logger.debug(
                "log(N43)=%.2f outside validity [%.1f, %.1f]; "
                "falling back to O32.", log_N43, *_LOG_N43_VALID,
            )

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

    # Mapping: ion_key -> (element, ion_stage, line_names, wave_labels, Te, ne)
    _ION_MAP = [
        ("O+/H+",   "O", 2, ["OII_doublet"], [3727],       Te_low,  ne_low),
        ("O++/H+",  "O", 3, ["OIII_5007"],   [5007],       Te_high, ne_high),
        ("N+/H+",   "N", 2, ["NII_6585"],    [6584],       Te_low,  ne_low),
        ("N++/H+",  "N", 3, ["NIII_1749", "NIII_1752"], [1749, 1752], Te_high, ne_mid),
        ("N+++/H+", "N", 4, ["NIV_1483", "NIV_1486"],   [1483, 1486], Te_high, ne_high),
        ("C+/H+",   "C", 2, ["CII]_2324", "CII]_2326"], [2323, 2325, 2326, 2327, 2328], Te_low, ne_low),
        ("C++/H+",  "C", 3, ["CIII]_1907", "CIII]"],    [1907, 1909], Te_high, ne_mid),
        ("C+++/H+", "C", 4, ["CIV_1", "CIV_2"],         [1548, 1551], Te_high, ne_high),
        ("Ne++/H+", "Ne", 3, ["NeIII_3869"], [3869],     Te_high, ne_high),
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

    # Te(high)
    if Te_high is not None:
        diag["Te(high)"] = (
            f"[OIII] 4363/(5007+4959) ratio with n_e(high) = {ne_high:.0f} cm^-3 (PyNEB)"
        )

    # Te(low)
    if Te_high is not None:
        rel_label = _TE_RELATION_LABELS.get(Te_relation, Te_relation)
        diag["Te(low)"] = (
            f"{rel_label} Te-Te relation from Te(high) = {Te_high:.0f} K"
        )

    # ne(low)
    _has_sii = "SII_6718" in fluxes and "SII_6732" in fluxes
    _has_oii = "OII_3726" in fluxes and "OII_3729" in fluxes
    if ne_low != ne_default:
        if _has_sii:
            diag["ne(low)"] = f"[SII] 6718/6732 doublet ratio -> {ne_low:.0f} cm^-3"
        elif _has_oii:
            diag["ne(low)"] = f"[OII] 3726/3729 doublet ratio -> {ne_low:.0f} cm^-3"
    else:
        if _has_sii:
            diag["ne(low)"] = (
                f"default ({ne_default:.0f} cm^-3) — [SII] doublet failed SNR cut or solve"
            )
        elif _has_oii:
            diag["ne(low)"] = (
                f"default ({ne_default:.0f} cm^-3) — [OII] doublet failed SNR cut or solve"
            )
        else:
            diag["ne(low)"] = (
                f"default ({ne_default:.0f} cm^-3) — no [SII] or [OII] doublet available"
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

    # ne(high)
    _has_niv = "NIV_1483" in fluxes and "NIV_1486" in fluxes
    ne_mid_or_low = ne_mid if ne_mid is not None else ne_low
    if _has_niv and ne_high != ne_mid_or_low:
        diag["ne(high)"] = f"NIV] 1483/1486 doublet ratio -> {ne_high:.0f} cm^-3"
    else:
        fallback_label = "ne(mid)" if ne_mid is not None else "ne(low)"
        if niv_rejected:
            diag["ne(high)"] = (
                f"fallback to {fallback_label} = {ne_mid_or_low:.0f} cm^-3 "
                f"— NIV] rejected (doublet ratio F(1483)/F(1486) > 1.6, "
                f"exceeds low-density limit ~1.5)"
            )
        elif _has_niv:
            diag["ne(high)"] = (
                f"fallback to {fallback_label} = {ne_mid_or_low:.0f} cm^-3 "
                f"— NIV] failed SNR cut or solve"
            )
        else:
            diag["ne(high)"] = (
                f"fallback to {fallback_label} = {ne_mid_or_low:.0f} cm^-3 "
                f"— no NIV] doublet available"
            )

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
    ne_high_max: float = 1e5,
    snr_ne: float = 3.0,
    snr_logU: float = 1.5,
    icf_method: str = "auto",
    snr_NO: float = 1.5,
    continuum_rms_limits: dict[str, float] | None = None,
    niv_rejected: bool = False,
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
        Te_low_from_high,
        compute_ionic_abundances,
        compute_Te_OIII,
        compute_total_abundances,
    )
    from .martinez25_icf import LOG_OH_SOLAR, _LOG_U_VALID

    # --- Step 1: Multi-phase electron density ---
    ne_low, ne_mid, ne_high, ne_failures = _compute_multi_ne(
        fluxes, errors=errors, snr_ne=snr_ne, ne_high_max=ne_high_max,
    )

    # --- Step 2: Electron temperature with zone-appropriate ne ---
    Te_high = compute_Te_OIII(
        fluxes["OIII_4363"], fluxes["OIII_5007"], fluxes["OIII_4959"], ne_high
    )
    Te_low = Te_low_from_high(Te_high, relation=Te_relation)

    # --- Step 3: Ionic abundances with zone-appropriate ne ---
    ionic = compute_ionic_abundances(fluxes, Te_high, Te_low, ne_low, ne_mid=ne_mid, ne_high=ne_high)

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
        logU, logU_diag = _compute_logU(
            fluxes, Z_Zsun, ne_high, errors=errors, snr_logU=snr_logU,
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
    )

    totals = compute_total_abundances(
        ionic, logU=logU, Z_Zsun=Z_Zsun, ne=ne_high,
        icf_method=icf_method,
        ionic_upper_limits=ionic_upper_limits,
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
    failures = totals.pop("_failures", {})
    failures.update(ne_failures)
    NO_tiers = totals.pop("_NO_tiers", None)

    # --- Build diagnostics dict ---
    diagnostics = _build_diagnostics(
        fluxes, Te_high, Te_relation, ne_low, ne_mid, ne_high,
        logU, logU_diag, icf_method, NO_icf_name, NE_DEFAULT,
        totals=totals, niv_rejected=niv_rejected,
    )

    # --- MC error propagation (all 6 steps per iteration) ---
    rng = np.random.default_rng(seed)
    OH_mc = []
    NO_mc = []
    CO_mc = []
    # Collect per-tier N/O posteriors for uncertainty on each method.
    _tier_keys = [k for k in (NO_tiers or {}) if not k.startswith("_")]
    NO_tier_mc: dict[str, list[float]] = {k: [] for k in _tier_keys}

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
                mc_fluxes, Te_h, Te_l, ne_low, ne_mid=ne_mid, ne_high=ne_high,
            )

            # Compute Z_Zsun for this MC iteration.
            oh_val = ionic_mc.get("O+/H+", 0.0) + ionic_mc.get("O++/H+", 0.0)
            if oh_val > 0:
                z_zsun_mc = 10.0 ** (12.0 + np.log10(oh_val) - LOG_OH_SOLAR)
            else:
                z_zsun_mc = Z_Zsun  # fallback to point estimate

            # Compute logU for this MC iteration (clamped to validity
            # range to prevent wild ICF extrapolation).
            # Pass original errors so the same SNR gating applies as
            # for the point estimate (prevents switching between N43
            # and O32 across MC iterations).
            logU_mc = logU  # default to point estimate
            if z_zsun_mc is not None and logU_diag is not None:
                logU_mc_val, _ = _compute_logU(
                    mc_fluxes, z_zsun_mc, ne_high, errors=errors,
                    snr_logU=snr_logU,
                )
                if logU_mc_val is not None:
                    logU_mc = float(np.clip(logU_mc_val, *_LOG_U_VALID))

            # Gate nitrogen ions using the *original* errors so the
            # same ions are included/excluded as in the point estimate.
            _gate_nitrogen_ions(ionic_mc, mc_fluxes, errors, snr_NO=snr_NO)

            totals_mc = compute_total_abundances(
                ionic_mc, logU=logU_mc, Z_Zsun=z_zsun_mc, ne=ne_high,
                icf_method=icf_method,
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

            # Collect per-tier N/O values.
            mc_tiers = totals_mc.get("_NO_tiers", {})
            for k in _tier_keys:
                val = mc_tiers.get(k, np.nan)
                NO_tier_mc[k].append(val if np.isfinite(val) else np.nan)
        except (ValueError, RuntimeError):
            OH_mc.append(np.nan)
            NO_mc.append(np.nan)
            CO_mc.append(np.nan)
            for k in _tier_keys:
                NO_tier_mc[k].append(np.nan)

    OH_mc = np.array(OH_mc)
    NO_mc = np.array(NO_mc)
    CO_mc = np.array(CO_mc)

    OH_err = float(np.nanstd(OH_mc)) if np.any(np.isfinite(OH_mc)) else np.nan
    NO_err = float(np.nanstd(NO_mc)) if np.any(np.isfinite(NO_mc)) else None
    CO_err = float(np.nanstd(CO_mc)) if np.any(np.isfinite(CO_mc)) else None

    # Attach per-tier uncertainties (symmetric std) to NO_tiers.
    if NO_tiers:
        for k in _tier_keys:
            arr = np.array(NO_tier_mc[k])
            if np.any(np.isfinite(arr)):
                NO_tiers[f"_err_{k}"] = float(np.nanstd(arr))

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
        "ne_mid": ne_mid,
        "ne_high": ne_high,
        "logU": logU,
        "icf_method": icf_method,
        "NO_icf_name": NO_icf_name,
        "ionic": ionic,
        "ionic_upper_limits": ionic_upper_limits if ionic_upper_limits else None,
        "ionic_ul_details": ionic_ul_details if ionic_ul_details else None,
        "OH_posterior": OH_mc,
        "NO_posterior": NO_mc,
        "CO_posterior": CO_mc,
        "SO": SO_log,
        "NeO": NeO_log,
        "ArO": ArO_log,
        "diagnostics": diagnostics,
        "failures": failures if failures else None,
        "NO_tiers": NO_tiers,
    }


def _run_direct_mcmc(
    posteriors: dict[str, np.ndarray],
    Te_relation: str,
    n_posterior: int = 1000,
    progress: bool = True,
    seed: int = 42,
    ne_high_max: float = 1e5,
    snr_ne: float = 3.0,
    snr_logU: float = 1.5,
    icf_method: str = "auto",
    snr_NO: float = 1.5,
    continuum_rms_limits: dict[str, float] | None = None,
    niv_rejected: bool = False,
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

    Returns
    -------
    dict
        Same keys as :func:`_run_direct`.
    """
    from .direct import (
        NE_DEFAULT,
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

    # Compute medians and errors for the point estimate and multi-phase ne.
    med_fluxes = {name: float(np.median(post)) for name, post in posteriors.items()}
    med_errors = {name: float(np.std(post)) for name, post in posteriors.items()}
    ne_low, ne_mid, ne_high, ne_failures = _compute_multi_ne(
        med_fluxes, errors=med_errors, snr_ne=snr_ne, ne_high_max=ne_high_max,
    )

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
        med_fluxes, Te_high_pt, Te_low_pt, ne_low, ne_mid=ne_mid, ne_high=ne_high
    ) if np.isfinite(Te_high_pt) else {}

    OH_pt = ionic_pt.get("O+/H+", 0.0) + ionic_pt.get("O++/H+", 0.0)
    Z_Zsun_pt = 10.0 ** (12.0 + np.log10(OH_pt) - LOG_OH_SOLAR) if OH_pt > 0 else None
    logU_pt = None
    logU_diag = None
    if Z_Zsun_pt is not None:
        logU_pt, logU_diag = _compute_logU(
            med_fluxes, Z_Zsun_pt, ne_high, errors=med_errors,
            snr_logU=snr_logU,
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
        )

    totals_pt = compute_total_abundances(
        ionic_pt, logU=logU_pt, Z_Zsun=Z_Zsun_pt, ne=ne_high,
        icf_method=icf_method,
        ionic_upper_limits=ionic_upper_limits,
    ) if ionic_pt else {}
    icf_method = totals_pt.get("icf_method")
    NO_icf_name = totals_pt.get("NO_icf_name")
    failures = totals_pt.pop("_failures", {})
    failures.update(ne_failures)
    NO_tiers = totals_pt.pop("_NO_tiers", None)
    # Collect per-tier N/O posteriors for uncertainty on each method.
    _tier_keys = [k for k in (NO_tiers or {}) if not k.startswith("_")]
    NO_tier_post: dict[str, list[float]] = {k: [] for k in _tier_keys}

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
                sample, Te_h, Te_l, ne_low, ne_mid=ne_mid, ne_high=ne_high,
            )

            # Z_Zsun for this sample.
            oh_val = ionic_i.get("O+/H+", 0.0) + ionic_i.get("O++/H+", 0.0)
            z_zsun_i = 10.0 ** (12.0 + np.log10(oh_val) - LOG_OH_SOLAR) if oh_val > 0 else Z_Zsun_pt

            # logU for this sample (clamped to validity range to
            # prevent wild ICF extrapolation).
            # Pass med_errors so the same SNR gating applies as for
            # the point estimate.
            logU_i = logU_pt
            if z_zsun_i is not None and logU_diag is not None:
                logU_val, _ = _compute_logU(
                    sample, z_zsun_i, ne_high, errors=med_errors,
                    snr_logU=snr_logU,
                )
                if logU_val is not None:
                    logU_i = float(np.clip(logU_val, *_LOG_U_VALID))

            # Gate nitrogen ions using median errors (same ions as
            # point estimate to prevent tier-switching across samples).
            _gate_nitrogen_ions(ionic_i, sample, med_errors, snr_NO=snr_NO)

            totals_i = compute_total_abundances(
                ionic_i, logU=logU_i, Z_Zsun=z_zsun_i, ne=ne_high,
                icf_method=icf_method,
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

            # Collect per-tier N/O values.
            mc_tiers = totals_i.get("_NO_tiers", {})
            for k in _tier_keys:
                val = mc_tiers.get(k, np.nan)
                NO_tier_post[k].append(val if np.isfinite(val) else np.nan)
        except (ValueError, RuntimeError):
            for k in _tier_keys:
                NO_tier_post[k].append(np.nan)
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

    # Attach per-tier asymmetric uncertainties to NO_tiers.
    if NO_tiers:
        for k in _tier_keys:
            arr = np.array(NO_tier_post[k])
            if np.any(np.isfinite(arr)):
                med = float(np.nanmedian(arr))
                lo = float(med - np.nanpercentile(arr, 16))
                hi = float(np.nanpercentile(arr, 84) - med)
                NO_tiers[f"_err_{k}"] = (lo, hi)

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
        "ne_mid": ne_mid,
        "ne_high": ne_high,
        "logU": logU_pt,
        "icf_method": icf_method,
        "NO_icf_name": NO_icf_name,
        "ionic": ionic_pt if ionic_pt else None,
        "ionic_upper_limits": ionic_upper_limits if ionic_upper_limits else None,
        "ionic_ul_details": ionic_ul_details if ionic_ul_details else None,
        "OH_posterior": OH_post,
        "NO_posterior": NO_post if np.any(np.isfinite(NO_post)) else None,
        "CO_posterior": CO_post if np.any(np.isfinite(CO_post)) else None,
        "SO": np.log10(totals_pt["S/O"]) if "S/O" in totals_pt and totals_pt["S/O"] > 0 else None,
        "NeO": np.log10(totals_pt["Ne/O"]) if "Ne/O" in totals_pt and totals_pt["Ne/O"] > 0 else None,
        "ArO": np.log10(totals_pt["Ar/O"]) if "Ar/O" in totals_pt and totals_pt["Ar/O"] > 0 else None,
        "diagnostics": _build_diagnostics(
            med_fluxes, Te_high_pt if np.isfinite(Te_high_pt) else None,
            Te_relation, ne_low, ne_mid, ne_high, logU_pt, logU_diag,
            icf_method, NO_icf_name, NE_DEFAULT,
            totals=totals_pt, niv_rejected=niv_rejected,
        ),
        "failures": failures if failures else None,
        "NO_tiers": NO_tiers,
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
    method: str = "auto",
    snr_auroral: float = 3.0,
    snr_line: float = 2.0,
    ne_high_max: float = 1e5,
    snr_ne: float = 3.0,
    snr_logU: float = 1.5,
    n_mc: int = 1000,
    Te_relation: str = "desi",
    Rv: float = 3.15,
    delta: float = -0.35,
    B_bump: float = 2.27,
    icf_method: str = "auto",
    snr_NO: float = 1.5,
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
    snr_NO : float
        Minimum total-line SNR for each nitrogen ion when using
        ``icf_method="direct_sum"`` (default 2.0).  For doublets
        (NIII], NIV], NV), the summed flux is divided by the
        quadrature-summed error.  Ions below this threshold are
        excluded from the direct sum, causing the code to fall
        through to a lower tier (or Izotov+06).
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

    # --- NIV] doublet ratio validity check ---
    # NIV] 1483 (³P₂→¹S₀, M2) and 1486 (³P₁→¹S₀, E1 intercombination).
    # At low density F(1483)/F(1486) ≈ 1.50 (1483 is the brighter line).
    # The ratio decreases monotonically with density (1483 gets collisionally
    # de-excited first due to its tiny A-value).  Physical range: ~0 to ~1.53.
    # Reject if the ratio exceeds 1.6 (generous margin above low-density limit).
    _niv_rejected = False
    _niv1483 = fluxes.get("NIV_1483", 0.0)
    _niv1486 = fluxes.get("NIV_1486", 0.0)
    if _niv1483 > 0 and _niv1486 > 0:
        niv_ratio = _niv1483 / _niv1486
        if niv_ratio > 1.6:
            logger.warning(
                "NIV] ratio F(1483)/F(1486) = %.2f > 1.6 — exceeds "
                "low-density limit (~1.5); excluding NIV] doublet.",
                niv_ratio,
            )
            fluxes.pop("NIV_1483", None)
            fluxes.pop("NIV_1486", None)
            errors.pop("NIV_1483", None)
            errors.pop("NIV_1486", None)
            posteriors.pop("NIV_1483", None)
            posteriors.pop("NIV_1486", None)
            excluded_lines.extend(["NIV_1483", "NIV_1486"])
            _niv_rejected = True

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
    primary_result = None

    if use_direct:
        if is_mcmc and posteriors and "OIII_4363" in posteriors:
            direct_out = _run_direct_mcmc(
                posteriors, Te_relation, n_posterior=n_posterior,
                progress=progress, ne_high_max=ne_high_max,
                snr_ne=snr_ne, snr_logU=snr_logU,
                icf_method=icf_method, snr_NO=snr_NO,
                continuum_rms_limits=continuum_rms_limits,
                niv_rejected=_niv_rejected,
            )
        else:
            direct_out = _run_direct(
                fluxes, errors, Te_relation, n_mc,
                progress=progress, ne_high_max=ne_high_max,
                snr_ne=snr_ne, snr_logU=snr_logU,
                icf_method=icf_method, snr_NO=snr_NO,
                continuum_rms_limits=continuum_rms_limits,
                niv_rejected=_niv_rejected,
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
            Te_low=direct_out.get("Te_low"),
            ne=direct_out.get("ne"),
            Av=Av_derived,
            ionic=direct_out.get("ionic"),
            ionic_upper_limits=direct_out.get("ionic_upper_limits"),
            ionic_ul_details=direct_out.get("ionic_ul_details"),
            OH_posterior=direct_out.get("OH_posterior"),
            NO_posterior=direct_out.get("NO_posterior"),
            CO_posterior=direct_out.get("CO_posterior"),
            SO=direct_out.get("SO"),
            NeO=direct_out.get("NeO"),
            ArO=direct_out.get("ArO"),
            logU=direct_out.get("logU"),
            ne_low=direct_out.get("ne_low"),
            ne_mid=direct_out.get("ne_mid"),
            ne_high=direct_out.get("ne_high"),
            icf_method=direct_out.get("icf_method"),
            NO_icf_name=direct_out.get("NO_icf_name"),
            excluded_lines=excluded_lines if excluded_lines else None,
            NO_tiers=direct_out.get("NO_tiers"),
            failures=direct_out.get("failures"),
            diagnostics=direct_out.get("diagnostics"),
        )

    if primary_result is None:
        # --- Strong-line method ---
        primary_result = _run_strong_line(
            fluxes, errors, is_mcmc, posteriors, n_mc, n_posterior,
            progress, Av_derived, excluded_lines,
        )

    # --- Auto mode: run the alternative method for comparison ---
    if method == "auto" and primary_result.alt_results is None:
        alt = {}
        if primary_result.method == "direct":
            # Also run strong-line for comparison.
            try:
                alt["strong_line"] = _run_strong_line(
                    fluxes, errors, is_mcmc, posteriors, n_mc, n_posterior,
                    progress, Av_derived, excluded_lines,
                )
            except Exception:
                logger.info("Alternative strong-line method failed; skipping.")
        elif primary_result.method == "strong_line":
            # Also try direct if 4363 is present (even if SNR was below threshold).
            has_4363 = "OIII_4363" in fluxes and fluxes.get("OIII_4363", 0) > 0
            if has_4363:
                try:
                    if is_mcmc and posteriors and "OIII_4363" in posteriors:
                        d_out = _run_direct_mcmc(
                            posteriors, Te_relation, n_posterior=n_posterior,
                            progress=progress, ne_high_max=ne_high_max,
                            snr_ne=snr_ne, snr_logU=snr_logU,
                            icf_method=icf_method, snr_NO=snr_NO,
                            niv_rejected=_niv_rejected,
                        )
                    else:
                        d_out = _run_direct(
                            fluxes, errors, Te_relation, n_mc,
                            progress=progress, ne_high_max=ne_high_max,
                            snr_ne=snr_ne, snr_logU=snr_logU,
                            icf_method=icf_method, snr_NO=snr_NO,
                            niv_rejected=_niv_rejected,
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
                        ionic=d_out.get("ionic"),
                        failures=d_out.get("failures"),
                    )
                except Exception:
                    logger.info("Alternative direct method failed; skipping.")
        if alt:
            primary_result.alt_results = alt

    return primary_result
