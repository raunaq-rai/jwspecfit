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


# Speed of light in km/s for velocity calculations.
_C_KMS = 2.99792458e5

# Velocity half-window for Balmer pixel mask (km/s).
_BALMER_MASK_DV = 3000.0


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

    return results


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
    if broad_type is not None:
        fit_lines_list = _add_broad_lines(line_names, broad_type)
    else:
        fit_lines_list = list(line_names)

    # Build parameter hints from narrow-only fit to seed narrow lines.
    p0_hint = None
    if narrow_fit is not None and broad_type is not None:
        p0_hint = {}
        nL_narrow = len(narrow_fit.line_names)
        for i, name in enumerate(narrow_fit.line_names):
            p0_hint[name] = (
                narrow_fit.params[i],                    # amplitude
                narrow_fit.params[nL_narrow + i],        # centroid
                narrow_fit.params[2 * nL_narrow + i],    # sigma
            )

    variant_label = broad_type or "narrow"
    result = fit_lines(
        spec, z, grating=grating, R=R, lines=fit_lines_list, deg=deg, n_boot=n_boot,
        n_jobs=n_jobs, sigma_factor=sigma_factor, centroid_vmax=centroid_vmax,
        moving_average=moving_average, tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids,
        tie_uv_widths=tie_uv_widths,
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
    mode: str = "auto",
    n_boot: int = 1000,
    n_boot_bic: int = 100,
    n_jobs: int = -1,
    snr_threshold: float = 5.0,
    bic_delta: float = BIC_DELTA_THRESHOLD,
    sigma_factor: float = 1.0,
    centroid_vmax: float = 500.0,
    moving_average: bool | int = False,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,
    _print_R: bool = True,
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
        Number of bootstrap iterations for flux uncertainties (default 1000).
    n_boot_bic : int
        Number of bootstrap iterations for BIC model selection (default 100).
        Set to 0 for single-point BIC (legacy behaviour).
    n_jobs : int
        Number of parallel jobs for bootstrap. ``-1`` uses all cores,
        ``1`` runs sequentially. Default ``-1``.
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
    )

    bic_b1 = np.nan
    bic_b2 = np.nan
    bic_both = np.nan
    all_fits: dict[str, FitResult] = {"narrow": fit_narrow}

    bic_boot_dist: dict[str, np.ndarray] = {}

    if mode == "off":
        # Re-fit with bootstrap for the narrow model.
        if n_boot > 0:
            fit_narrow, bic_narrow = _fit_model_variant(
                spectrum, z, narrow_lines, grating, R, continuum, deg,
                broad_type=None, n_boot=n_boot, n_jobs=n_jobs,
                sigma_factor=sigma_factor, centroid_vmax=centroid_vmax,
                moving_average=moving_average, tie_uv_doublets=tie_uv_doublets,
                tie_uv_centroids=tie_uv_centroids,
                tie_uv_widths=tie_uv_widths,
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
                broad_type=None, n_boot=n_boot, n_jobs=n_jobs,
                sigma_factor=sigma_factor, centroid_vmax=centroid_vmax,
                moving_average=moving_average, tie_uv_doublets=tie_uv_doublets,
                tie_uv_centroids=tie_uv_centroids,
                tie_uv_widths=tie_uv_widths,
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

    # Determine which variants to fit.
    variants_to_fit: list[str] = ["narrow"]
    if mode in ("auto", "broad1", "both"):
        variants_to_fit.append("broad1")
    if mode in ("auto", "broad2", "both"):
        variants_to_fit.append("broad2")
    if mode in ("auto", "both"):
        variants_to_fit.append("both")

    # --- Phase 2: BIC model selection ---
    if n_boot_bic > 0 and mode == "auto":
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
                tie_uv_centroids, tie_uv_widths,
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
                )
            for variant, fut in futures.items():
                fit_v, _ = fut.result()
                all_fits[variant] = fit_v

    else:
        # Legacy single-point BIC (n_boot_bic=0 or forced mode).
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

    # --- Phase 3: select best model by BIC ---
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

        bic_src = "median" if n_boot_bic > 0 else "single-point"
        logger.info(
            "BIC selection (%s): narrow=%.1f, broad1=%.1f, broad2=%.1f, both=%.1f → %s",
            bic_src, bic_narrow, bic_b1, bic_b2, bic_both, best_name,
        )
    elif mode == "broad1":
        best_name = "broad1"
    elif mode == "broad2":
        best_name = "broad2"
    elif mode == "both":
        best_name = "both"
    else:
        best_name = "narrow"

    # --- Phase 4: re-fit only the selected model with bootstrap ---
    if n_boot > 0:
        broad_type = None if best_name == "narrow" else best_name
        best_fit, _ = _fit_model_variant(
            spectrum, z, narrow_lines, grating, R, continuum, deg,
            broad_type=broad_type, n_boot=n_boot,
            narrow_fit=fit_narrow if broad_type is not None else None,
            n_jobs=n_jobs, sigma_factor=sigma_factor,
            centroid_vmax=centroid_vmax,
            moving_average=moving_average, tie_uv_doublets=tie_uv_doublets,
            tie_uv_centroids=tie_uv_centroids,
            tie_uv_widths=tie_uv_widths,
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
        bic_bootstrap=bic_boot_dist,
    )
