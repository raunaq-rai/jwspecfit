"""jwspecfit — JWST NIRSpec emission-line fitting."""

from __future__ import annotations

__version__ = "0.1.0"

from pathlib import Path
from typing import Callable

from .broad import BroadFitResult, fit_with_broad
from .fitter import FitResult, LineResult
from .fitter import fit_lines as _fit_lines_narrow
from .io import (
    Spectrum, export_lines_txt, load_result, read_dict, read_fits, read_npz, save_result,
)
from .lines import REST_LINES_A, get_line_list, observable_lines
from .lyman_alpha import igm_transmission, lya_model
from .plotting import plot_fit, plot_fit_interactive
from .resolution import R_from_pixels, R_prism, resolve_R, sigma_inst_A

__all__ = [
    "BroadFitResult",
    "FitResult",
    "LineResult",
    "REST_LINES_A",
    "R_from_pixels",
    "R_prism",
    "Spectrum",
    "fit_lines",
    "fit_with_broad",
    "get_line_list",
    "igm_transmission",
    "lya_model",
    "observable_lines",
    "plot_fit",
    "plot_fit_interactive",
    "read_dict",
    "read_fits",
    "read_npz",
    "resolve_R",
    "save_result",
    "load_result",
    "export_lines_txt",
    "sigma_inst_A",
]


def fit_lines(
    spectrum: Spectrum,
    z: float,
    *,
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    wave_range_A: tuple[float, float] | None = None,
    deg: int = 2,
    n_boot: int = 1000,
    clip_sigma: float = 2.5,
    n_jobs: int = -1,
    save_path: str | Path | None = None,
    mode: str = "auto",
    n_boot_bic: int = 100,
    snr_threshold: float = 5.0,
    bic_delta: float = 6.0,
    sigma_factor: float = 1.0,
) -> FitResult | BroadFitResult:
    """Fit emission lines in a spectrum.

    By default (``mode="auto"``), performs BIC-based broad Balmer
    component selection via :func:`fit_with_broad`.  Set ``mode="off"``
    for narrow-only fitting.

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
        Lines to fit.
    wave_range_A : tuple, optional
        Observed wavelength range (Angstrom).
    deg : int
        Continuum polynomial degree (default 2).
    n_boot : int
        Bootstrap iterations for flux uncertainties (default 1000).
    clip_sigma : float
        Continuum sigma-clipping threshold (default 2.5).
    n_jobs : int
        Parallel jobs for bootstrap (default ``-1``).
    save_path : str or Path, optional
        Path to save the result.
    mode : str
        Broad component mode (default ``"auto"``):
        - ``"auto"``: BIC-based selection.
        - ``"off"``: Narrow-only (no broad component search).
        - ``"broad1"``: Force intermediate broad.
        - ``"broad2"``: Force very broad.
        - ``"both"``: Force both broad components.
    n_boot_bic : int
        Bootstrap iterations for BIC model selection (default 100).
        Only used when ``mode != "off"``.
    snr_threshold : float
        Minimum Ha SNR to attempt broad fitting (default 5.0).
    bic_delta : float
        ΔBIC threshold for model selection (default 6.0).
    sigma_factor : float
        Multiplicative factor on the upper line-width bound.
        Use values > 1 for stacked spectra (default 1.0).

    Returns
    -------
    BroadFitResult
        When ``mode != "off"``.  Delegates all :class:`FitResult`
        attributes (``lines``, ``params``, ``model_flux``, etc.) via
        properties, so it can be used as a drop-in replacement.
    FitResult
        When ``mode="off"``.
    """
    if mode == "off":
        return _fit_lines_narrow(
            spectrum, z,
            grating=grating, R=R, lines=lines,
            wave_range_A=wave_range_A, deg=deg,
            n_boot=n_boot, clip_sigma=clip_sigma,
            n_jobs=n_jobs, save_path=save_path,
            sigma_factor=sigma_factor,
        )

    return fit_with_broad(
        spectrum, z,
        grating=grating, R=R, lines=lines,
        deg=deg, mode=mode,
        n_boot=n_boot,
        n_boot_bic=n_boot_bic,
        n_jobs=n_jobs,
        snr_threshold=snr_threshold,
        bic_delta=bic_delta,
        sigma_factor=sigma_factor,
    )
