"""Broad Balmer component fitting with BIC model selection.

Detects broad Hα and Hβ components by comparing narrow-only vs
narrow+broad models using the Bayesian Information Criterion (BIC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .constraints import ConstraintSet
from .continuum import fit_continuum
from .fitter import FitResult, LineResult, _grating_bounds, fit_lines
from .io import Spectrum, _flam_to_ujy, _ujy_to_flam
from .lines import REST_LINES_A, get_line_list, observable_lines
from .models import build_model, pixel_weight
from .resolution import resolve_R, sigma_inst_A

logger = logging.getLogger(__name__)

# Intermediate broad component velocity bounds (km/s).
# FWHM ~ 500–2000 km/s corresponds to σ_v ~ 210–850 km/s.
# Physical origin: AGN narrow-line region outflows, turbulent ISM.
# References: Heckman et al. (1981), Zakamska & Greene (2014),
#             Förster Schreiber et al. (2019) for star-forming outflows.
BROAD1_SIGMA_V_LO = 210.0    # km/s — lower bound (FWHM ~ 500 km/s)
BROAD1_SIGMA_V_SEED = 420.0  # km/s — seed (FWHM ~ 1000 km/s)
BROAD1_SIGMA_V_HI = 850.0    # km/s — upper bound (FWHM ~ 2000 km/s)

# Very broad (BLR) component velocity bounds (km/s).
# FWHM ~ 2000–5000 km/s corresponds to σ_v ~ 850–2120 km/s.
# Physical origin: virial motions in the broad-line region.
# References: Greene & Ho (2005), Shen et al. (2011).
BROAD2_SIGMA_V_LO = 850.0    # km/s — lower bound (FWHM ~ 2000 km/s)
BROAD2_SIGMA_V_SEED = 1270.0  # km/s — seed (FWHM ~ 3000 km/s)
BROAD2_SIGMA_V_HI = 2120.0   # km/s — upper bound (FWHM ~ 5000 km/s)

# BIC threshold for accepting a more complex model.
BIC_DELTA_THRESHOLD = 6.0

# Broad line definitions — lines that get a broad component.
# NII_6549 and NII_6585 are included in the Ha complex so that the
# BIC comparison accounts for the full blend and prevents broad Ha
# from absorbing narrow NII flux.
BROAD_CANDIDATES = ["Ha", "HBETA", "HDELTA", "HGAMMA"]

# Lines whose narrow kinematics are tied to Ha during broad fitting
# to guard against the broad component leaking into NII.
HA_NII_COMPLEX = ["Ha", "NII_6549", "NII_6585"]

# Forbidden [OIII] lines eligible for an independent broad (outflow)
# component.  Independent of the Balmer BIC test: an outflow can be
# present in [OIII] without showing in Hα and vice versa.
OIII_BROAD_CANDIDATES = ["OIII_5007", "OIII_4959"]

# Permitted He I lines eligible for an independent broad component.
# All trace the He+ recombination zone (IP ~24.6 eV), so within each
# broad tier they share kinematics (anchored on first present).  Order
# preference: bright optical lines first so HEI_5877 / HEI_6680 tend
# to anchor when available.
HEI_BROAD_CANDIDATES = [
    "HEI_5877", "HEI_6680", "HEI_4472", "HEI_7067",
    "HEI_3889", "HEI_4027", "HEI_4145",
]


# Speed of light in km/s for velocity calculations.
_C_KMS = 2.99792458e5

# Velocity half-window for Balmer pixel mask (km/s).
_BALMER_MASK_DV = 3000.0

# Velocity half-window for [OIII] pixel mask (km/s).
_OIII_MASK_DV = 3000.0

# Velocity half-window for He I pixel mask (km/s).
_HEI_MASK_DV = 3000.0


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
        Median BIC for narrow-only model (from bootstrap when used).
    bic_broad1 : float
        Median BIC for narrow + BROAD1 model (NaN if not attempted).
    bic_broad2 : float
        Median BIC for narrow + BROAD2 model (NaN if not attempted).
    bic_both : float
        Median BIC for narrow + BROAD1 + BROAD2 model (NaN if not attempted).
    all_fits : dict[str, FitResult]
        All fitted models for inspection.
    bic_bootstrap : dict[str, np.ndarray]
        Full BIC distributions from bootstrap iterations.  Empty if
        bootstrap BIC was not run.
    """

    best_fit: FitResult
    selected_model: str
    bic_narrow: float
    bic_broad1: float
    bic_broad2: float
    bic_both: float
    all_fits: dict[str, FitResult]
    bic_bootstrap: dict[str, np.ndarray] = field(default_factory=dict)
    # --- Independent [OIII] outflow component selection ---
    # selected variant: one of "off" | "broad1" | "broad2" | "both"
    oiii_selected: str = "off"
    bic_oiii_off: float = float("nan")
    bic_oiii_broad1: float = float("nan")
    bic_oiii_broad2: float = float("nan")
    bic_oiii_both: float = float("nan")
    bic_oiii_bootstrap: dict[str, np.ndarray] = field(default_factory=dict)
    # --- Independent He I broad component selection ---
    hei_selected: str = "off"
    bic_hei_off: float = float("nan")
    bic_hei_broad1: float = float("nan")
    bic_hei_broad2: float = float("nan")
    bic_hei_both: float = float("nan")
    bic_hei_bootstrap: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def oiii_broad_selected(self) -> bool:
        """Convenience: True if any OIII broad component was selected."""
        return self.oiii_selected != "off"

    @property
    def hei_broad_selected(self) -> bool:
        """Convenience: True if any HeI broad component was selected."""
        return self.hei_selected != "off"

    # Delegate common FitResult attributes to best_fit for API compatibility.

    @property
    def lines(self) -> dict:
        """Per-line results (delegates to ``best_fit.lines``)."""
        return self.best_fit.lines

    @property
    def params(self) -> np.ndarray:
        """Best-fit parameter vector."""
        return self.best_fit.params

    @property
    def model_flux(self) -> np.ndarray:
        """Best-fit emission-line model."""
        return self.best_fit.model_flux

    @property
    def continuum(self) -> np.ndarray:
        """Best-fit continuum."""
        return self.best_fit.continuum

    @property
    def residuals(self) -> np.ndarray:
        """Fit residuals."""
        return self.best_fit.residuals

    @property
    def chi2(self) -> float:
        """Reduced chi-squared."""
        return self.best_fit.chi2

    @property
    def spectrum(self):
        """Input spectrum."""
        return self.best_fit.spectrum

    @property
    def line_names(self) -> list[str]:
        """Ordered line names."""
        return self.best_fit.line_names

    @property
    def constraints(self):
        """Applied constraints."""
        return self.best_fit.constraints

    @property
    def success(self) -> bool:
        """Whether the optimiser converged."""
        return self.best_fit.success


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


def _balmer_pixel_mask(
    wave_um: np.ndarray,
    z: float,
    dv: float = _BALMER_MASK_DV,
) -> np.ndarray:
    """Boolean mask selecting pixels within ±dv km/s of Hα and Hβ.

    Parameters
    ----------
    wave_um : np.ndarray
        Observed wavelength in microns.
    z : float
        Source redshift.
    dv : float
        Velocity half-window in km/s (default 3000).

    Returns
    -------
    np.ndarray
        Boolean mask (True for Balmer pixels).
    """
    mask = np.zeros(len(wave_um), dtype=bool)
    for line_name in ("Ha", "HBETA"):
        lam_rest_A = REST_LINES_A[line_name]
        lam_obs_um = lam_rest_A * (1.0 + z) * 1e-4  # Å → µm
        # ±dv km/s → wavelength window
        lam_lo = lam_obs_um * (1.0 - dv / _C_KMS)
        lam_hi = lam_obs_um * (1.0 + dv / _C_KMS)
        mask |= (wave_um >= lam_lo) & (wave_um <= lam_hi)
    return mask


def _oiii_pixel_mask(
    wave_um: np.ndarray,
    z: float,
    dv: float = _OIII_MASK_DV,
) -> np.ndarray:
    """Boolean mask selecting pixels within ±dv km/s of [OIII] 5007 and 4959."""
    mask = np.zeros(len(wave_um), dtype=bool)
    for line_name in OIII_BROAD_CANDIDATES:
        if line_name not in REST_LINES_A:
            continue
        lam_obs_um = REST_LINES_A[line_name] * (1.0 + z) * 1e-4
        lam_lo = lam_obs_um * (1.0 - dv / _C_KMS)
        lam_hi = lam_obs_um * (1.0 + dv / _C_KMS)
        mask |= (wave_um >= lam_lo) & (wave_um <= lam_hi)
    return mask


def _hei_pixel_mask(
    wave_um: np.ndarray,
    z: float,
    candidates: list[str],
    dv: float = _HEI_MASK_DV,
) -> np.ndarray:
    """Boolean mask selecting pixels within ±dv km/s of supplied He I lines."""
    mask = np.zeros(len(wave_um), dtype=bool)
    for line_name in candidates:
        if line_name not in REST_LINES_A:
            continue
        lam_obs_um = REST_LINES_A[line_name] * (1.0 + z) * 1e-4
        lam_lo = lam_obs_um * (1.0 - dv / _C_KMS)
        lam_hi = lam_obs_um * (1.0 + dv / _C_KMS)
        mask |= (wave_um >= lam_lo) & (wave_um <= lam_hi)
    return mask


def _bic_bootstrap_single(
    noise: np.ndarray,
    spec: Spectrum,
    z: float,
    narrow_lines: list[str],
    grating: str | None,
    R: float | Callable | None,
    continuum: np.ndarray,
    deg: int,
    narrow_fit: FitResult,
    balmer_mask: np.ndarray,
    variants: list[str],
    sigma_factor: float = 1.0,
    centroid_vmax: float = 500.0,
    moving_average: bool | int = False,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,
    sigma_overrides: dict[str, tuple[float, float]] | None = None,
    centroid_overrides: dict[str, tuple[float, float]] | None = None,
    niv_doublet_ratio: float | None = None,
    ciii_doublet_ratio: float | None = None,
    oiii_variants: list[str] | None = None,
    oiii_baseline_broad: str | None = None,
    oiii_mask: np.ndarray | None = None,
    hei_variants: list[str] | None = None,
    hei_baseline_broad: str | None = None,
    hei_baseline_oiii: str | None = None,
    hei_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Run one BIC bootstrap iteration for all model variants.

    Module-level function for joblib pickling.

    Parameters
    ----------
    noise : np.ndarray
        Standard-normal noise vector (length = n_pix).
    spec : Spectrum
        Original spectrum.
    z : float
        Source redshift.
    narrow_lines : list of str
        Narrow line list.
    grating, R, continuum, deg
        Fitting configuration.
    narrow_fit : FitResult
        Narrow-only fit on real data (used to seed broad variants).
    balmer_mask : np.ndarray
        Boolean mask for Balmer pixels.
    variants : list of str
        Model variant names to fit (includes ``"narrow"``).
    sigma_factor : float
        Width upper-bound multiplier (default 1.0).
    tie_uv_doublets : bool
        Tie UV doublet kinematics (default ``False``).
    sigma_overrides : dict, optional
        Per-line sigma bounds in Angstroms ``{name: (lo, hi)}``.
    niv_doublet_ratio : float, optional
        Fixed F(1483)/F(1486) ratio for the NIV] doublet.

    Returns
    -------
    dict[str, float]
        ``{variant_name: bic_value}`` for each variant.
    """
    # Create perturbed spectrum.
    perturbed = spec.copy()
    perturbed.flux_ujy = spec.flux_ujy + noise * spec.err_ujy

    results: dict[str, float] = {}
    for variant in variants:
        broad_type = None if variant == "narrow" else variant
        try:
            fit_v, _ = _fit_model_variant(
                perturbed, z, narrow_lines, grating, R, continuum, deg,
                broad_type=broad_type, n_boot=0,
                narrow_fit=narrow_fit if broad_type is not None else None,
                n_jobs=1, sigma_factor=sigma_factor,
                centroid_vmax=centroid_vmax,
                moving_average=moving_average,
                tie_uv_doublets=tie_uv_doublets,
                tie_uv_centroids=tie_uv_centroids,
                tie_uv_widths=tie_uv_widths,
                sigma_overrides=sigma_overrides,
                centroid_overrides=centroid_overrides,
                niv_doublet_ratio=niv_doublet_ratio,
                ciii_doublet_ratio=ciii_doublet_ratio,
            )
        except (ValueError, RuntimeError):
            results[variant] = np.nan
            continue

        # Compute BIC on Balmer pixels only.
        flam_err = _ujy_to_flam(perturbed.err_ujy, perturbed.wave_um)
        flam_data = _ujy_to_flam(
            perturbed.flux_ujy - fit_v.continuum, perturbed.wave_um
        )
        flam_model = _ujy_to_flam(fit_v.model_flux, perturbed.wave_um)
        valid = perturbed.mask_valid() & balmer_mask
        resid = flam_data - flam_model

        if fit_v.constraints is not None:
            n_free = int(np.sum(fit_v.constraints.free_mask()))
        else:
            n_free = len(fit_v.params)

        results[variant] = _compute_bic(resid, flam_err, valid, n_free)

    # ------------------------------------------------------------------
    # Optional second-stage [OIII] broad BIC test.  Independent of the
    # Balmer comparison above: variants here are ``"oiii_off"`` and
    # ``"oiii_on"``, both built on top of ``oiii_baseline_broad`` (the
    # Balmer model picked in stage 1).  BIC is evaluated on OIII pixels
    # only so the test is sensitive to outflow signatures regardless of
    # what happens in the Balmer complex.
    # ------------------------------------------------------------------
    if oiii_variants and oiii_mask is not None:
        flam_err = _ujy_to_flam(perturbed.err_ujy, perturbed.wave_um)
        for variant in oiii_variants:
            # variant is one of "off"|"broad1"|"broad2"|"both"; map to
            # the oiii_broad_type kwarg accepted by _fit_model_variant.
            oiii_type = None if variant == "off" else variant
            try:
                fit_v, _ = _fit_model_variant(
                    perturbed, z, narrow_lines, grating, R, continuum, deg,
                    broad_type=oiii_baseline_broad, n_boot=0,
                    narrow_fit=narrow_fit,
                    n_jobs=1, sigma_factor=sigma_factor,
                    centroid_vmax=centroid_vmax,
                    moving_average=moving_average,
                    tie_uv_doublets=tie_uv_doublets,
                    tie_uv_centroids=tie_uv_centroids,
                    tie_uv_widths=tie_uv_widths,
                    sigma_overrides=sigma_overrides,
                    centroid_overrides=centroid_overrides,
                    niv_doublet_ratio=niv_doublet_ratio,
                    ciii_doublet_ratio=ciii_doublet_ratio,
                    oiii_broad_type=oiii_type,
                )
            except (ValueError, RuntimeError):
                results[variant] = np.nan
                continue

            flam_data = _ujy_to_flam(
                perturbed.flux_ujy - fit_v.continuum, perturbed.wave_um
            )
            flam_model = _ujy_to_flam(fit_v.model_flux, perturbed.wave_um)
            valid = perturbed.mask_valid() & oiii_mask
            resid = flam_data - flam_model

            if fit_v.constraints is not None:
                n_free = int(np.sum(fit_v.constraints.free_mask()))
            else:
                n_free = len(fit_v.params)

            results[variant] = _compute_bic(resid, flam_err, valid, n_free)

    # ------------------------------------------------------------------
    # Optional third-stage He I broad BIC test.  Variants are
    # ``"hei_off"`` / ``"hei_broad1"`` / ``"hei_broad2"`` / ``"hei_both"``,
    # all built on top of (hei_baseline_broad, hei_baseline_oiii) — the
    # winners of stages 1 and 2 — so the test is sensitive specifically
    # to a broad HeI signature.
    # ------------------------------------------------------------------
    if hei_variants and hei_mask is not None:
        flam_err = _ujy_to_flam(perturbed.err_ujy, perturbed.wave_um)
        for variant in hei_variants:
            hei_type = None if variant == "off" else variant
            try:
                fit_v, _ = _fit_model_variant(
                    perturbed, z, narrow_lines, grating, R, continuum, deg,
                    broad_type=hei_baseline_broad, n_boot=0,
                    narrow_fit=narrow_fit,
                    n_jobs=1, sigma_factor=sigma_factor,
                    centroid_vmax=centroid_vmax,
                    moving_average=moving_average,
                    tie_uv_doublets=tie_uv_doublets,
                    tie_uv_centroids=tie_uv_centroids,
                    tie_uv_widths=tie_uv_widths,
                    sigma_overrides=sigma_overrides,
                    centroid_overrides=centroid_overrides,
                    niv_doublet_ratio=niv_doublet_ratio,
                    ciii_doublet_ratio=ciii_doublet_ratio,
                    oiii_broad_type=hei_baseline_oiii,
                    hei_broad_type=hei_type,
                )
            except (ValueError, RuntimeError):
                results[variant] = np.nan
                continue
            flam_data = _ujy_to_flam(
                perturbed.flux_ujy - fit_v.continuum, perturbed.wave_um
            )
            flam_model = _ujy_to_flam(fit_v.model_flux, perturbed.wave_um)
            valid = perturbed.mask_valid() & hei_mask
            resid = flam_data - flam_model
            if fit_v.constraints is not None:
                n_free = int(np.sum(fit_v.constraints.free_mask()))
            else:
                n_free = len(fit_v.params)
            results[variant] = _compute_bic(resid, flam_err, valid, n_free)

    return results


def _add_broad_lines(
    line_names: list[str],
    broad_type: str | None,
    *,
    oiii_broad_type: str | None = None,
    hei_broad_type: str | None = None,
) -> list[str]:
    """Add broad Balmer, [OIII], and/or He I entries to a line list.

    Parameters
    ----------
    line_names : list of str
        Base narrow line names.
    broad_type : {"broad1", "broad2", "both", None}
        Balmer broad variant.  ``None`` skips Balmer broad lines.
    oiii_broad_type : {"broad1", "broad2", "both", None}
        [OIII] broad variant.  ``None`` skips OIII broad lines.
        Mirrors the Balmer mapping: BROAD1 = intermediate (FWHM
        500–2000 km/s, outflow), BROAD2 = very broad (FWHM
        2000–5000 km/s, fast wind).  The ``_BROAD`` / ``_BROAD2``
        suffixes trigger the matching velocity bounds in
        ``jwspecfit.fitter`` automatically.

    Returns
    -------
    list of str
        Extended line list with broad entries.
    """
    extended = list(line_names)
    if broad_type is not None:
        for base in BROAD_CANDIDATES:
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
    if oiii_broad_type is not None:
        for base in OIII_BROAD_CANDIDATES:
            if base not in line_names:
                continue
            if oiii_broad_type in ("broad1", "both"):
                bname = f"{base}_BROAD"
                if bname not in extended:
                    extended.append(bname)
                    REST_LINES_A[bname] = REST_LINES_A[base]
            if oiii_broad_type in ("broad2", "both"):
                bname = f"{base}_BROAD2"
                if bname not in extended:
                    extended.append(bname)
                    REST_LINES_A[bname] = REST_LINES_A[base]
    if hei_broad_type is not None:
        for base in HEI_BROAD_CANDIDATES:
            if base not in line_names:
                continue
            if hei_broad_type in ("broad1", "both"):
                bname = f"{base}_BROAD"
                if bname not in extended:
                    extended.append(bname)
                    REST_LINES_A[bname] = REST_LINES_A[base]
            if hei_broad_type in ("broad2", "both"):
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
    narrow_fit: FitResult | None = None,
    n_jobs: int = -1,
    sigma_factor: float = 1.0,
    centroid_vmax: float = 500.0,
    moving_average: bool | int = False,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,
    sigma_overrides: dict[str, tuple[float, float]] | None = None,
    centroid_overrides: dict[str, tuple[float, float]] | None = None,
    niv_doublet_ratio: float | None = None,
    ciii_doublet_ratio: float | None = None,
    oiii_broad_type: str | None = None,
    hei_broad_type: str | None = None,
) -> tuple[FitResult, float]:
    """Fit a specific model variant and return (FitResult, BIC).

    Parameters
    ----------
    broad_type : str or None
        ``None`` for narrow-only, or ``"broad1"``, ``"broad2"``, ``"both"``.
    narrow_fit : FitResult or None
        Narrow-only fit results used to seed narrow line parameters
        when fitting broad variants.

    Returns
    -------
    tuple
        ``(FitResult, BIC)``.
    """
    any_broad = (
        broad_type is not None
        or oiii_broad_type is not None
        or hei_broad_type is not None
    )
    if any_broad:
        fit_lines_list = _add_broad_lines(
            line_names, broad_type,
            oiii_broad_type=oiii_broad_type,
            hei_broad_type=hei_broad_type,
        )
    else:
        fit_lines_list = list(line_names)

    # Build parameter hints from narrow-only fit to seed narrow lines.
    p0_hint = None
    if narrow_fit is not None and any_broad:
        p0_hint = {}
        nL_narrow = len(narrow_fit.line_names)
        for i, name in enumerate(narrow_fit.line_names):
            p0_hint[name] = (
                narrow_fit.params[i],                    # amplitude
                narrow_fit.params[nL_narrow + i],        # centroid
                narrow_fit.params[2 * nL_narrow + i],    # sigma
            )

    parts = []
    if broad_type is not None:
        parts.append(broad_type)
    if oiii_broad_type is not None:
        parts.append(f"oiii_{oiii_broad_type}")
    if hei_broad_type is not None:
        parts.append(f"hei_{hei_broad_type}")
    variant_label = "+".join(parts) if parts else "narrow"
    result = fit_lines(
        spec, z, grating=grating, R=R, lines=fit_lines_list, deg=deg, n_boot=n_boot,
        n_jobs=n_jobs, sigma_factor=sigma_factor, centroid_vmax=centroid_vmax,
        moving_average=moving_average, tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids,
        tie_uv_widths=tie_uv_widths,
        sigma_overrides=sigma_overrides,
        centroid_overrides=centroid_overrides,
        niv_doublet_ratio=niv_doublet_ratio,
        ciii_doublet_ratio=ciii_doublet_ratio,
        _label=variant_label, _p0_hint=p0_hint,
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
    fit_balmer_broad: bool = False,
    fit_oiii_broad: bool = False,
    fit_hei_broad: bool = False,
    n_boot: int = 1000,
    n_boot_bic: int = 100,
    n_jobs: int = -1,
    snr_threshold: float = 5.0,
    oiii_snr_threshold: float = 5.0,
    hei_snr_threshold: float = 5.0,
    bic_delta: float = BIC_DELTA_THRESHOLD,
    sigma_factor: float = 1.0,
    centroid_vmax: float = 500.0,
    moving_average: bool | int = False,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,
    sigma_overrides: dict[str, tuple[float, float]] | None = None,
    centroid_overrides: dict[str, tuple[float, float]] | None = None,
    niv_doublet_ratio: float | None = None,
    ciii_doublet_ratio: float | None = None,
    _print_R: bool = True,
) -> BroadFitResult:
    """Fit emission lines with optional BIC-selected broad components.

    Two independent BIC tests can be enabled:

    - **Balmer broad** (``fit_balmer_broad``): tests narrow vs.
      narrow + intermediate broad (FWHM 500–2000 km/s) vs.
      narrow + very broad (FWHM 2000–5000 km/s) vs. narrow + both.
      Gated by ``snr_threshold`` on Hα.
    - **[OIII] broad** (``fit_oiii_broad``): tests baseline vs.
      baseline + broad [OIII] 5007/4959 (outflow signature).  Gated
      by ``oiii_snr_threshold`` on [OIII] 5007 / 4959.

    The two tests are independent — both can be selected, only one,
    or neither.

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
    fit_balmer_broad : bool
        If ``True``, run BIC model selection for a broad Balmer
        component (intermediate / very-broad / both).  Default
        ``False`` — opt in only when you want to test for a BLR or
        outflow signature on Hα/Hβ.
    fit_oiii_broad : bool
        If ``True``, run an independent BIC test for a broad component
        on [OIII] 5007/4959.  Decoupled from the Balmer test.  Default
        ``False`` — opt in only when looking for outflow signatures.
    fit_hei_broad : bool
        If ``True``, run an independent BIC test for a broad component
        on all observable He I lines (5877/6680/4472/...).  Within
        each tier all HeI broads share kinematics (anchored on first
        present).  Default ``False``.
    n_boot : int
        Number of bootstrap iterations for flux uncertainties (default 1000).
    n_boot_bic : int
        Number of bootstrap iterations for BIC model selection (default 100).
        Set to 0 for single-point BIC (legacy behaviour).
    n_jobs : int
        Number of parallel jobs for bootstrap. ``-1`` uses all cores,
        ``1`` runs sequentially. Default ``-1``.
    snr_threshold : float
        Minimum Hα SNR to attempt the Balmer broad fit (default 5.0).
    oiii_snr_threshold : float
        Minimum [OIII] 5007 / 4959 SNR to attempt the OIII broad fit
        (default 5.0).
    bic_delta : float
        ΔBIC threshold for accepting a more complex model (default 6.0).

    Returns
    -------
    BroadFitResult
    """
    grating = grating or spectrum.grating
    R = R or spectrum.R

    # Print resolving power once for the top-level broad-fitting call.
    if _print_R:
        R_arr = resolve_R(spectrum.wave_um, grating=grating, R=R)
        R_med = float(np.median(R_arr))
        R_lo, R_hi = float(np.min(R_arr)), float(np.max(R_arr))
        if abs(R_hi - R_lo) < 10:
            print(f"Resolving power: R = {R_med:.0f}")
        else:
            print(f"Resolving power: R \u2248 {R_med:.0f} (range {R_lo:.0f}\u2013{R_hi:.0f})")

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
        moving_average=moving_average,
    )

    # --- Phase 1: fast fits without bootstrap for BIC comparison ---
    fit_narrow, bic_narrow = _fit_model_variant(
        spectrum, z, narrow_lines, grating, R, continuum, deg,
        broad_type=None, n_boot=0, n_jobs=n_jobs,
        sigma_factor=sigma_factor, centroid_vmax=centroid_vmax,
        moving_average=moving_average, tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids,
        tie_uv_widths=tie_uv_widths,
        sigma_overrides=sigma_overrides,
        centroid_overrides=centroid_overrides,
        niv_doublet_ratio=niv_doublet_ratio,
        ciii_doublet_ratio=ciii_doublet_ratio,
    )

    bic_b1 = np.nan
    bic_b2 = np.nan
    bic_both = np.nan
    all_fits: dict[str, FitResult] = {"narrow": fit_narrow}

    bic_boot_dist: dict[str, np.ndarray] = {}

    if not fit_balmer_broad and not fit_oiii_broad:
        # Narrow-only: skip every BIC stage and just return the (optionally
        # bootstrapped) narrow fit.
        if n_boot > 0:
            fit_narrow, bic_narrow = _fit_model_variant(
                spectrum, z, narrow_lines, grating, R, continuum, deg,
                broad_type=None, n_boot=n_boot, n_jobs=n_jobs,
                sigma_factor=sigma_factor, centroid_vmax=centroid_vmax,
                moving_average=moving_average, tie_uv_doublets=tie_uv_doublets,
                tie_uv_centroids=tie_uv_centroids,
                tie_uv_widths=tie_uv_widths,
                sigma_overrides=sigma_overrides,
                centroid_overrides=centroid_overrides,
                niv_doublet_ratio=niv_doublet_ratio,
                ciii_doublet_ratio=ciii_doublet_ratio,
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

    # Check Hα SNR before attempting Balmer broad fits.
    ha_snr = 0.0
    if "Ha" in fit_narrow.lines:
        ha_snr = fit_narrow.lines["Ha"].snr

    skip_balmer_bic = (not fit_balmer_broad) or (ha_snr < snr_threshold)
    if skip_balmer_bic:
        if fit_balmer_broad:
            logger.info(
                "Hα SNR=%.1f < %.1f; skipping Balmer broad fitting "
                "(OIII broad still tested if requested).",
                ha_snr, snr_threshold,
            )
        best_name = "narrow"

    # Determine which Balmer variants to fit.
    variants_to_fit: list[str] = ["narrow"]
    if not skip_balmer_bic:
        variants_to_fit.extend(["broad1", "broad2", "both"])

    # --- Phase 2: BIC model selection ---
    if not skip_balmer_bic and n_boot_bic > 0:
        # Bootstrap BIC on Balmer pixels only.
        from joblib import Parallel, delayed
        from tqdm import tqdm

        balmer_mask = _balmer_pixel_mask(spectrum.wave_um, z)
        rng = np.random.default_rng()
        noise_vectors = rng.standard_normal((n_boot_bic, len(spectrum.wave_um)))

        bic_records: list[dict[str, float]] = Parallel(
            n_jobs=n_jobs, return_as="generator"
        )(
            delayed(_bic_bootstrap_single)(
                noise_vectors[i],
                spectrum, z, narrow_lines, grating, R, continuum, deg,
                fit_narrow, balmer_mask, variants_to_fit, sigma_factor,
                centroid_vmax, moving_average, tie_uv_doublets,
                tie_uv_centroids, tie_uv_widths, sigma_overrides,
                centroid_overrides, niv_doublet_ratio, ciii_doublet_ratio,
            )
            for i in range(n_boot_bic)
        )
        bic_records = list(tqdm(
            bic_records, total=n_boot_bic, desc="BIC bootstrap",
        ))

        # Aggregate into arrays and compute medians.
        for variant in variants_to_fit:
            bic_boot_dist[variant] = np.array(
                [rec[variant] for rec in bic_records]
            )

        bic_medians = {v: float(np.nanmedian(bic_boot_dist[v])) for v in variants_to_fit}
        bic_narrow = bic_medians.get("narrow", bic_narrow)
        bic_b1 = bic_medians.get("broad1", bic_b1)
        bic_b2 = bic_medians.get("broad2", bic_b2)
        bic_both = bic_medians.get("both", bic_both)

        # Also do a single real-data fit for each variant (for all_fits dict).
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(variants_to_fit)) as pool:
            futures = {}
            for variant in variants_to_fit:
                if variant == "narrow":
                    continue  # already have fit_narrow
                broad_type = variant
                futures[variant] = pool.submit(
                    _fit_model_variant,
                    spectrum, z, narrow_lines, grating, R, continuum, deg,
                    broad_type, 0, fit_narrow, 1, sigma_factor, centroid_vmax,
                    moving_average, tie_uv_doublets, tie_uv_centroids, tie_uv_widths,
                    sigma_overrides, centroid_overrides, niv_doublet_ratio, ciii_doublet_ratio,
                )
            for variant, fut in futures.items():
                fit_v, _ = fut.result()
                all_fits[variant] = fit_v

    elif not skip_balmer_bic:
        # Single-point BIC (n_boot_bic=0).
        from concurrent.futures import ThreadPoolExecutor

        broad_variants = [v for v in variants_to_fit if v != "narrow"]
        with ThreadPoolExecutor(max_workers=len(broad_variants) or 1) as pool:
            bic_futures: dict[str, any] = {}
            for variant in broad_variants:
                fut = pool.submit(
                    _fit_model_variant,
                    spectrum, z, narrow_lines, grating, R, continuum, deg,
                    variant, 0, fit_narrow, n_jobs, sigma_factor, centroid_vmax,
                    moving_average, tie_uv_doublets, tie_uv_centroids, tie_uv_widths,
                    sigma_overrides, centroid_overrides, niv_doublet_ratio, ciii_doublet_ratio,
                )
                bic_futures[variant] = fut

            for variant, fut in bic_futures.items():
                fit_v, bic_v = fut.result()
                all_fits[variant] = fit_v
                if variant == "broad1":
                    bic_b1 = bic_v
                elif variant == "broad2":
                    bic_b2 = bic_v
                elif variant == "both":
                    bic_both = bic_v

    # --- Phase 3: select best Balmer model by BIC ---
    if not skip_balmer_bic:
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

        bic_src = "median" if n_boot_bic > 0 else "single-point"
        logger.info(
            "BIC selection (%s): narrow=%.1f, broad1=%.1f, broad2=%.1f, both=%.1f → %s",
            bic_src, bic_narrow, bic_b1, bic_b2, bic_both, best_name,
        )

    # --- Phase 3.5: independent [OIII] outflow BIC test ---
    # Mirrors the Balmer Phase 3 logic but on OIII pixels.  Variants:
    # "off" / "broad1" / "broad2" / "both", with the same BROAD1 /
    # BROAD2 velocity bounds as Balmer (outflow / fast wind).
    oiii_selected = "off"
    bic_oiii = {
        "off": float("nan"),
        "broad1": float("nan"),
        "broad2": float("nan"),
        "both": float("nan"),
    }
    bic_oiii_boot: dict[str, np.ndarray] = {}

    baseline_broad = None if best_name == "narrow" else best_name
    oiii_candidates_present = [
        n for n in OIII_BROAD_CANDIDATES if n in narrow_lines
    ]

    if fit_oiii_broad and oiii_candidates_present:
        # SNR gate: use the best narrow OIII SNR from the Stage-1 baseline.
        baseline_fit = all_fits.get(best_name, fit_narrow)
        oiii_snr = 0.0
        for cand in oiii_candidates_present:
            lr = baseline_fit.lines.get(cand)
            if lr is not None and lr.snr > oiii_snr:
                oiii_snr = lr.snr

        if oiii_snr < oiii_snr_threshold:
            logger.info(
                "[OIII] SNR=%.1f < %.1f; skipping OIII broad test.",
                oiii_snr, oiii_snr_threshold,
            )
        else:
            oiii_mask = _oiii_pixel_mask(spectrum.wave_um, z)
            balmer_mask_for_oiii = _balmer_pixel_mask(spectrum.wave_um, z)
            oiii_variants = ["off", "broad1", "broad2", "both"]

            if n_boot_bic > 0:
                from joblib import Parallel, delayed
                from tqdm import tqdm

                rng = np.random.default_rng()
                noise_vectors = rng.standard_normal(
                    (n_boot_bic, len(spectrum.wave_um))
                )
                oiii_records: list[dict[str, float]] = Parallel(
                    n_jobs=n_jobs, return_as="generator"
                )(
                    delayed(_bic_bootstrap_single)(
                        noise_vectors[i],
                        spectrum, z, narrow_lines, grating, R, continuum, deg,
                        fit_narrow, balmer_mask_for_oiii,
                        [],  # skip Balmer variants in this pass
                        sigma_factor, centroid_vmax, moving_average,
                        tie_uv_doublets, tie_uv_centroids, tie_uv_widths,
                        sigma_overrides, centroid_overrides,
                        niv_doublet_ratio, ciii_doublet_ratio,
                        oiii_variants, baseline_broad, oiii_mask,
                    )
                    for i in range(n_boot_bic)
                )
                oiii_records = list(tqdm(
                    oiii_records, total=n_boot_bic, desc="OIII BIC bootstrap",
                ))
                for v in oiii_variants:
                    bic_oiii_boot[v] = np.array(
                        [rec.get(v, np.nan) for rec in oiii_records]
                    )
                    bic_oiii[v] = float(np.nanmedian(bic_oiii_boot[v]))
            else:
                # Single-point BIC on real data.
                flam_err = _ujy_to_flam(spectrum.err_ujy, spectrum.wave_um)
                for v in oiii_variants:
                    oiii_type = None if v == "off" else v
                    try:
                        fit_v, _ = _fit_model_variant(
                            spectrum, z, narrow_lines, grating, R, continuum, deg,
                            broad_type=baseline_broad, n_boot=0,
                            narrow_fit=fit_narrow,
                            n_jobs=n_jobs, sigma_factor=sigma_factor,
                            centroid_vmax=centroid_vmax,
                            moving_average=moving_average,
                            tie_uv_doublets=tie_uv_doublets,
                            tie_uv_centroids=tie_uv_centroids,
                            tie_uv_widths=tie_uv_widths,
                            sigma_overrides=sigma_overrides,
                            centroid_overrides=centroid_overrides,
                            niv_doublet_ratio=niv_doublet_ratio,
                            ciii_doublet_ratio=ciii_doublet_ratio,
                            oiii_broad_type=oiii_type,
                        )
                    except (ValueError, RuntimeError):
                        bic_oiii[v] = float("nan")
                        continue
                    flam_data = _ujy_to_flam(
                        spectrum.flux_ujy - fit_v.continuum, spectrum.wave_um
                    )
                    flam_model = _ujy_to_flam(fit_v.model_flux, spectrum.wave_um)
                    valid = spectrum.mask_valid() & oiii_mask
                    resid = flam_data - flam_model
                    nf = (
                        int(np.sum(fit_v.constraints.free_mask()))
                        if fit_v.constraints is not None
                        else len(fit_v.params)
                    )
                    bic_oiii[v] = _compute_bic(resid, flam_err, valid, nf)

            # Selection: pick lowest BIC; require ΔBIC ≥ bic_delta vs "off".
            finite = {k: v for k, v in bic_oiii.items() if np.isfinite(v)}
            if finite:
                best_oiii = min(finite, key=finite.get)
                if best_oiii != "off":
                    delta = bic_oiii["off"] - finite[best_oiii]
                    if delta < bic_delta:
                        best_oiii = "off"
                        logger.info(
                            "[OIII] ΔBIC=%.1f < %.1f; keeping no OIII broad.",
                            delta, bic_delta,
                        )
                oiii_selected = best_oiii
                bic_src = "median" if n_boot_bic > 0 else "single-point"
                logger.info(
                    "[OIII] BIC (%s): off=%.1f, broad1=%.1f, broad2=%.1f, "
                    "both=%.1f → %s",
                    bic_src, bic_oiii["off"], bic_oiii["broad1"],
                    bic_oiii["broad2"], bic_oiii["both"], oiii_selected,
                )

    oiii_broad_type_sel = None if oiii_selected == "off" else oiii_selected

    # --- Phase 3.6: independent He I broad BIC test ---
    # Built on top of (best_name Balmer + selected OIII broad), so the
    # comparison isolates the He I broad signal.  Uses HeI pixels only.
    hei_selected = "off"
    bic_hei = {
        "off": float("nan"),
        "broad1": float("nan"),
        "broad2": float("nan"),
        "both": float("nan"),
    }
    bic_hei_boot: dict[str, np.ndarray] = {}

    hei_candidates_present = [
        n for n in HEI_BROAD_CANDIDATES if n in narrow_lines
    ]

    if fit_hei_broad and hei_candidates_present:
        baseline_fit = all_fits.get(best_name, fit_narrow)
        hei_snr = 0.0
        for cand in hei_candidates_present:
            lr = baseline_fit.lines.get(cand)
            if lr is not None and lr.snr > hei_snr:
                hei_snr = lr.snr

        if hei_snr < hei_snr_threshold:
            logger.info(
                "He I SNR=%.1f < %.1f; skipping HeI broad test.",
                hei_snr, hei_snr_threshold,
            )
        else:
            hei_mask = _hei_pixel_mask(
                spectrum.wave_um, z, hei_candidates_present,
            )
            balmer_mask_for_hei = _balmer_pixel_mask(spectrum.wave_um, z)
            hei_variants = ["off", "broad1", "broad2", "both"]

            if n_boot_bic > 0:
                from joblib import Parallel, delayed
                from tqdm import tqdm

                rng = np.random.default_rng()
                noise_vectors = rng.standard_normal(
                    (n_boot_bic, len(spectrum.wave_um))
                )
                hei_records: list[dict[str, float]] = Parallel(
                    n_jobs=n_jobs, return_as="generator"
                )(
                    delayed(_bic_bootstrap_single)(
                        noise_vectors[i],
                        spectrum, z, narrow_lines, grating, R, continuum, deg,
                        fit_narrow, balmer_mask_for_hei,
                        [],  # skip Balmer variants in this pass
                        sigma_factor, centroid_vmax, moving_average,
                        tie_uv_doublets, tie_uv_centroids, tie_uv_widths,
                        sigma_overrides, centroid_overrides,
                        niv_doublet_ratio, ciii_doublet_ratio,
                        None, None, None,
                        hei_variants, baseline_broad, oiii_broad_type_sel,
                        hei_mask,
                    )
                    for i in range(n_boot_bic)
                )
                hei_records = list(tqdm(
                    hei_records, total=n_boot_bic, desc="HeI BIC bootstrap",
                ))
                for v in hei_variants:
                    bic_hei_boot[v] = np.array(
                        [rec.get(v, np.nan) for rec in hei_records]
                    )
                    bic_hei[v] = float(np.nanmedian(bic_hei_boot[v]))
            else:
                flam_err = _ujy_to_flam(spectrum.err_ujy, spectrum.wave_um)
                for v in hei_variants:
                    hei_type = None if v == "off" else v
                    try:
                        fit_v, _ = _fit_model_variant(
                            spectrum, z, narrow_lines, grating, R, continuum, deg,
                            broad_type=baseline_broad, n_boot=0,
                            narrow_fit=fit_narrow,
                            n_jobs=n_jobs, sigma_factor=sigma_factor,
                            centroid_vmax=centroid_vmax,
                            moving_average=moving_average,
                            tie_uv_doublets=tie_uv_doublets,
                            tie_uv_centroids=tie_uv_centroids,
                            tie_uv_widths=tie_uv_widths,
                            sigma_overrides=sigma_overrides,
                            centroid_overrides=centroid_overrides,
                            niv_doublet_ratio=niv_doublet_ratio,
                            ciii_doublet_ratio=ciii_doublet_ratio,
                            oiii_broad_type=oiii_broad_type_sel,
                            hei_broad_type=hei_type,
                        )
                    except (ValueError, RuntimeError):
                        bic_hei[v] = float("nan")
                        continue
                    flam_data = _ujy_to_flam(
                        spectrum.flux_ujy - fit_v.continuum, spectrum.wave_um
                    )
                    flam_model = _ujy_to_flam(fit_v.model_flux, spectrum.wave_um)
                    valid = spectrum.mask_valid() & hei_mask
                    resid = flam_data - flam_model
                    nf = (
                        int(np.sum(fit_v.constraints.free_mask()))
                        if fit_v.constraints is not None
                        else len(fit_v.params)
                    )
                    bic_hei[v] = _compute_bic(resid, flam_err, valid, nf)

            finite = {k: v for k, v in bic_hei.items() if np.isfinite(v)}
            if finite:
                best_hei = min(finite, key=finite.get)
                if best_hei != "off":
                    delta = bic_hei["off"] - finite[best_hei]
                    if delta < bic_delta:
                        best_hei = "off"
                        logger.info(
                            "HeI ΔBIC=%.1f < %.1f; keeping no HeI broad.",
                            delta, bic_delta,
                        )
                hei_selected = best_hei
                bic_src = "median" if n_boot_bic > 0 else "single-point"
                logger.info(
                    "HeI BIC (%s): off=%.1f, broad1=%.1f, broad2=%.1f, "
                    "both=%.1f → %s",
                    bic_src, bic_hei["off"], bic_hei["broad1"],
                    bic_hei["broad2"], bic_hei["both"], hei_selected,
                )

    hei_broad_type_sel = None if hei_selected == "off" else hei_selected

    # --- Phase 4: re-fit only the selected model with bootstrap ---
    selected_key_parts = []
    if best_name != "narrow":
        selected_key_parts.append(best_name)
    if oiii_broad_type_sel is not None:
        selected_key_parts.append(f"oiii_{oiii_broad_type_sel}")
    if hei_broad_type_sel is not None:
        selected_key_parts.append(f"hei_{hei_broad_type_sel}")
    selected_key = "+".join(selected_key_parts) if selected_key_parts else "narrow"

    any_broad_sel = (
        best_name != "narrow"
        or oiii_broad_type_sel is not None
        or hei_broad_type_sel is not None
    )

    if n_boot > 0:
        broad_type = None if best_name == "narrow" else best_name
        best_fit, _ = _fit_model_variant(
            spectrum, z, narrow_lines, grating, R, continuum, deg,
            broad_type=broad_type, n_boot=n_boot,
            narrow_fit=fit_narrow if any_broad_sel else None,
            n_jobs=n_jobs, sigma_factor=sigma_factor,
            centroid_vmax=centroid_vmax,
            moving_average=moving_average, tie_uv_doublets=tie_uv_doublets,
            tie_uv_centroids=tie_uv_centroids,
            tie_uv_widths=tie_uv_widths,
            sigma_overrides=sigma_overrides,
            centroid_overrides=centroid_overrides,
            niv_doublet_ratio=niv_doublet_ratio,
            ciii_doublet_ratio=ciii_doublet_ratio,
            oiii_broad_type=oiii_broad_type_sel,
            hei_broad_type=hei_broad_type_sel,
        )
        all_fits[selected_key] = best_fit
    elif any_broad_sel and selected_key not in all_fits:
        best_fit, _ = _fit_model_variant(
            spectrum, z, narrow_lines, grating, R, continuum, deg,
            broad_type=None if best_name == "narrow" else best_name,
            n_boot=0,
            narrow_fit=fit_narrow,
            n_jobs=n_jobs, sigma_factor=sigma_factor,
            centroid_vmax=centroid_vmax,
            moving_average=moving_average,
            tie_uv_doublets=tie_uv_doublets,
            tie_uv_centroids=tie_uv_centroids,
            tie_uv_widths=tie_uv_widths,
            sigma_overrides=sigma_overrides,
            centroid_overrides=centroid_overrides,
            niv_doublet_ratio=niv_doublet_ratio,
            ciii_doublet_ratio=ciii_doublet_ratio,
            oiii_broad_type=oiii_broad_type_sel,
            hei_broad_type=hei_broad_type_sel,
        )
        all_fits[selected_key] = best_fit

    return BroadFitResult(
        best_fit=all_fits[selected_key],
        selected_model=best_name,
        bic_narrow=bic_narrow,
        bic_broad1=bic_b1,
        bic_broad2=bic_b2,
        bic_both=bic_both,
        all_fits=all_fits,
        bic_bootstrap=bic_boot_dist,
        oiii_selected=oiii_selected,
        bic_oiii_off=bic_oiii["off"],
        bic_oiii_broad1=bic_oiii["broad1"],
        bic_oiii_broad2=bic_oiii["broad2"],
        bic_oiii_both=bic_oiii["both"],
        bic_oiii_bootstrap=bic_oiii_boot,
        hei_selected=hei_selected,
        bic_hei_off=bic_hei["off"],
        bic_hei_broad1=bic_hei["broad1"],
        bic_hei_broad2=bic_hei["broad2"],
        bic_hei_both=bic_hei["both"],
        bic_hei_bootstrap=bic_hei_boot,
    )
