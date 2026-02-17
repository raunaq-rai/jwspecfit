"""Broad Balmer component fitting with BIC model selection.

Detects broad Hα and Hβ components by comparing narrow-only vs
narrow+broad models using the Bayesian Information Criterion (BIC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .constraints import ConstraintSet
from .continuum import fit_continuum
from .fitter import FitResult, LineResult, _grating_bounds, fit_lines
from .io import Spectrum, _flam_to_ujy, _ujy_to_flam
from .lines import REST_LINES_A, get_line_list, observable_lines
from .models import build_model, pixel_weight
from .resolution import sigma_inst_A

logger = logging.getLogger(__name__)

# Broad component width factors relative to narrow σ_v.
BROAD1_FACTOR = 3.0   # Seed: 3× narrow σ_v
BROAD1_LO = 1.5       # Lower bound
BROAD1_HI = 5.0       # Upper bound

BROAD2_FACTOR = 7.0   # Seed: 7× narrow σ_v
BROAD2_LO = 4.0
BROAD2_HI = 12.0

# BIC threshold for accepting a more complex model.
BIC_DELTA_THRESHOLD = 6.0

# Broad line definitions — lines that get a broad component.
# NII_6549 and NII_6585 are included in the Ha complex so that the
# BIC comparison accounts for the full blend and prevents broad Ha
# from absorbing narrow NII flux.
BROAD_BALMER = ["Ha", "HBETA", "HDELTA", "HGAMMA"]

# Lines whose narrow kinematics are tied to Ha during broad fitting
# to guard against the broad component leaking into NII.
HA_NII_COMPLEX = ["Ha", "NII_6549", "NII_6585"]


@dataclass
class BroadFitResult:
    """Result of broad component model selection.

    Attributes
    ----------
    best_fit : FitResult
        The selected best-fit result.
    selected_model : str
        Model name: ``"narrow"``, ``"broad1"``, ``"broad2"``, or ``"both"``.
    bic_narrow : float
        BIC for narrow-only model.
    bic_broad1 : float
        BIC for narrow + BROAD1 model (NaN if not attempted).
    bic_broad2 : float
        BIC for narrow + BROAD2 model.
    bic_both : float
        BIC for narrow + BROAD1 + BROAD2 model.
    all_fits : dict[str, FitResult]
        All fitted models for inspection.
    """

    best_fit: FitResult
    selected_model: str
    bic_narrow: float
    bic_broad1: float
    bic_broad2: float
    bic_both: float
    all_fits: dict[str, FitResult]


def _compute_bic(
    residuals: np.ndarray,
    errors: np.ndarray,
    valid: np.ndarray,
    n_free: int,
) -> float:
    """Compute BIC = χ² + k·ln(N) on whitened residuals."""
    r = residuals[valid] / errors[valid]
    chi2 = float(np.sum(r**2))
    N = int(np.sum(valid))
    return chi2 + n_free * np.log(N)


def _add_broad_lines(
    line_names: list[str],
    broad_type: str,
) -> list[str]:
    """Add broad Balmer entries to line list.

    Parameters
    ----------
    line_names : list of str
        Base narrow line names.
    broad_type : str
        ``"broad1"``, ``"broad2"``, or ``"both"``.

    Returns
    -------
    list of str
        Extended line list with broad entries.
    """
    extended = list(line_names)
    for base in BROAD_BALMER:
        if base not in line_names:
            continue
        if broad_type in ("broad1", "both"):
            bname = f"{base}_BROAD"
            if bname not in extended:
                extended.append(bname)
                REST_LINES_A[bname] = REST_LINES_A[base]
        if broad_type in ("broad2", "both"):
            bname = f"{base}_BROAD2"
            if bname not in extended:
                extended.append(bname)
                REST_LINES_A[bname] = REST_LINES_A[base]
    return extended


def _make_broad_constraints(line_names: list[str]) -> ConstraintSet:
    """Build constraints that do NOT tie broad Balmer widths to OIII."""
    cs = ConstraintSet(line_names, tie_nii=True, tie_balmer_to_oiii=True)

    # Override: broad lines should NOT have their widths tied.
    # We handle this by customising the free_mask and apply methods.
    # For now, the base ConstraintSet only ties narrow Balmer widths.
    # Broad lines have "_BROAD" suffix so they won't match the tie targets.
    return cs


def _fit_model_variant(
    spec: Spectrum,
    z: float,
    line_names: list[str],
    grating: str | None,
    R: float | Callable | None,
    continuum: np.ndarray,
    deg: int,
    broad_type: str | None = None,
    n_boot: int = 200,
) -> tuple[FitResult, float]:
    """Fit a specific model variant and return (FitResult, BIC).

    Parameters
    ----------
    broad_type : str or None
        ``None`` for narrow-only, or ``"broad1"``, ``"broad2"``, ``"both"``.

    Returns
    -------
    tuple
        ``(FitResult, BIC)``.
    """
    if broad_type is not None:
        fit_lines_list = _add_broad_lines(line_names, broad_type)
    else:
        fit_lines_list = list(line_names)

    variant_label = broad_type or "narrow"
    result = fit_lines(
        spec, z, grating=grating, R=R, lines=fit_lines_list, deg=deg, n_boot=n_boot,
        _label=variant_label,
    )

    # Compute BIC.
    flam_err = _ujy_to_flam(spec.err_ujy, spec.wave_um)
    flam_data = _ujy_to_flam(spec.flux_ujy - result.continuum, spec.wave_um)
    flam_model = _ujy_to_flam(result.model_flux, spec.wave_um)
    valid = spec.mask_valid()
    resid = flam_data - flam_model

    if result.constraints is not None:
        n_free = int(np.sum(result.constraints.free_mask()))
    else:
        n_free = len(result.params)

    bic = _compute_bic(resid, flam_err, valid, n_free)

    return result, bic


def fit_with_broad(
    spectrum: Spectrum,
    z: float,
    *,
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    deg: int = 2,
    mode: str = "auto",
    n_boot: int = 200,
    snr_threshold: float = 5.0,
    bic_delta: float = BIC_DELTA_THRESHOLD,
) -> BroadFitResult:
    """Fit emission lines with optional broad Balmer components.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum.
    z : float
        Source redshift.
    grating : str, optional
        Grating name.
    R : float or callable, optional
        Resolving power.
    lines : list of str, optional
        Narrow line list. If ``None``, auto-detected.
    deg : int
        Continuum polynomial degree.
    mode : str
        Broad component mode:
        - ``"auto"``: BIC-based selection (default).
        - ``"off"``: Narrow-only.
        - ``"broad1"``: Force intermediate broad.
        - ``"broad2"``: Force very broad.
        - ``"both"``: Force both broad components.
    n_boot : int
        Number of bootstrap iterations for uncertainties (default 200).
    snr_threshold : float
        Minimum Hα SNR to attempt broad fitting (default 5.0).
    bic_delta : float
        ΔBIC threshold for model selection (default 6.0).

    Returns
    -------
    BroadFitResult
    """
    grating = grating or spectrum.grating
    R = R or spectrum.R

    # Determine narrow line list.
    if lines is None:
        if grating is not None:
            candidate = get_line_list(grating)
        else:
            candidate = get_line_list("prism")
        narrow_lines = observable_lines(
            candidate, z, spectrum.wave_um.min(), spectrum.wave_um.max()
        )
    else:
        narrow_lines = list(lines)

    # Continuum (shared across all variants).
    continuum = fit_continuum(
        spectrum.wave_um, spectrum.flux_ujy, spectrum.err_ujy,
        z, narrow_lines, grating=grating, R=R, deg=deg,
    )

    # --- Phase 1: fast fits without bootstrap for BIC comparison ---
    fit_narrow, bic_narrow = _fit_model_variant(
        spectrum, z, narrow_lines, grating, R, continuum, deg,
        broad_type=None, n_boot=0,
    )

    bic_b1 = np.nan
    bic_b2 = np.nan
    bic_both = np.nan
    all_fits: dict[str, FitResult] = {"narrow": fit_narrow}

    if mode == "off":
        # Re-fit with bootstrap for the narrow model.
        if n_boot > 0:
            fit_narrow, bic_narrow = _fit_model_variant(
                spectrum, z, narrow_lines, grating, R, continuum, deg,
                broad_type=None, n_boot=n_boot,
            )
            all_fits["narrow"] = fit_narrow
        return BroadFitResult(
            best_fit=fit_narrow,
            selected_model="narrow",
            bic_narrow=bic_narrow,
            bic_broad1=bic_b1,
            bic_broad2=bic_b2,
            bic_both=bic_both,
            all_fits=all_fits,
        )

    # Check Hα SNR before attempting broad fits (analytic errors are fine here).
    ha_snr = 0.0
    if "Ha" in fit_narrow.lines:
        ha_snr = fit_narrow.lines["Ha"].snr

    attempt_broad = (mode != "off") and (mode != "auto" or ha_snr >= snr_threshold)

    if not attempt_broad and mode == "auto":
        logger.info("Hα SNR=%.1f < %.1f; skipping broad fitting.", ha_snr, snr_threshold)
        # Re-fit with bootstrap for the narrow model.
        if n_boot > 0:
            fit_narrow, bic_narrow = _fit_model_variant(
                spectrum, z, narrow_lines, grating, R, continuum, deg,
                broad_type=None, n_boot=n_boot,
            )
            all_fits["narrow"] = fit_narrow
        return BroadFitResult(
            best_fit=fit_narrow,
            selected_model="narrow",
            bic_narrow=bic_narrow,
            bic_broad1=bic_b1,
            bic_broad2=bic_b2,
            bic_both=bic_both,
            all_fits=all_fits,
        )

    # Fast BIC fits (no bootstrap).
    if mode in ("auto", "broad1", "both"):
        fit_b1, bic_b1 = _fit_model_variant(
            spectrum, z, narrow_lines, grating, R, continuum, deg,
            "broad1", n_boot=0,
        )
        all_fits["broad1"] = fit_b1

    if mode in ("auto", "broad2", "both"):
        fit_b2, bic_b2 = _fit_model_variant(
            spectrum, z, narrow_lines, grating, R, continuum, deg,
            "broad2", n_boot=0,
        )
        all_fits["broad2"] = fit_b2

    if mode in ("auto", "both"):
        fit_both, bic_both = _fit_model_variant(
            spectrum, z, narrow_lines, grating, R, continuum, deg,
            "both", n_boot=0,
        )
        all_fits["both"] = fit_both

    # --- Phase 2: select best model by BIC ---
    if mode == "auto":
        candidates = {
            "narrow": bic_narrow,
            "broad1": bic_b1,
            "broad2": bic_b2,
            "both": bic_both,
        }
        candidates = {k: v for k, v in candidates.items() if np.isfinite(v)}
        best_name = min(candidates, key=candidates.get)

        if best_name != "narrow":
            delta = bic_narrow - candidates[best_name]
            if delta < bic_delta:
                best_name = "narrow"
                logger.info("ΔBIC=%.1f < %.1f; keeping narrow model.", delta, bic_delta)

        logger.info(
            "BIC selection: narrow=%.1f, broad1=%.1f, broad2=%.1f, both=%.1f → %s",
            bic_narrow, bic_b1, bic_b2, bic_both, best_name,
        )
    elif mode == "broad1":
        best_name = "broad1"
    elif mode == "broad2":
        best_name = "broad2"
    elif mode == "both":
        best_name = "both"
    else:
        best_name = "narrow"

    # --- Phase 3: re-fit only the selected model with bootstrap ---
    if n_boot > 0:
        broad_type = None if best_name == "narrow" else best_name
        best_fit, _ = _fit_model_variant(
            spectrum, z, narrow_lines, grating, R, continuum, deg,
            broad_type=broad_type, n_boot=n_boot,
        )
        all_fits[best_name] = best_fit

    return BroadFitResult(
        best_fit=all_fits[best_name],
        selected_model=best_name,
        bic_narrow=bic_narrow,
        bic_broad1=bic_b1,
        bic_broad2=bic_b2,
        bic_both=bic_both,
        all_fits=all_fits,
    )
