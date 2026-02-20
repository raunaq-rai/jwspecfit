"""jwspecfit — JWST NIRSpec emission-line fitting."""

from __future__ import annotations

__version__ = "0.1.0"

from pathlib import Path
from typing import Callable

import numpy as np

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


def _merge_window_results(
    spectrum: Spectrum,
    window_results: list[FitResult | BroadFitResult],
    wave_windows_A: list[tuple[float, float]],
) -> FitResult:
    """Merge per-window fit results into a single full-spectrum result.

    Parameters
    ----------
    spectrum : Spectrum
        Original full spectrum.
    window_results : list of FitResult or BroadFitResult
        Fit result for each wavelength window.
    wave_windows_A : list of (float, float)
        Wavelength windows in Angstroms (observed frame).

    Returns
    -------
    FitResult
        Merged result covering all windows.  Pixels outside any window
        have ``NaN`` continuum and residuals, and zero model flux.
    """
    n_pix = spectrum.n_pix
    full_wave_A = spectrum.wave_A

    # Initialize full-spectrum arrays.
    continuum = np.full(n_pix, np.nan)
    model_flux = np.zeros(n_pix)
    residuals = np.full(n_pix, np.nan)

    all_lines: dict[str, LineResult] = {}
    all_line_names: list[str] = []

    for (lo, hi), res in zip(wave_windows_A, window_results):
        # Extract FitResult from BroadFitResult if needed.
        fit = res.best_fit if hasattr(res, "best_fit") else res

        # Map window pixels to full spectrum.
        mask = (full_wave_A >= lo) & (full_wave_A <= hi)

        continuum[mask] = fit.continuum
        model_flux[mask] = fit.model_flux
        residuals[mask] = fit.residuals

        # Collect line measurements (first window wins for duplicates).
        for name in fit.line_names:
            if name not in all_lines:
                all_lines[name] = fit.lines[name]
                all_line_names.append(name)

    # Build a synthetic params vector [amplitudes, centroids, sigmas]
    # from LineResult attributes so plotting functions can decompose
    # individual components.
    nL = len(all_line_names)
    params = np.zeros(3 * nL)
    for i, name in enumerate(all_line_names):
        lr = all_lines[name]
        params[i] = lr.amplitude
        params[nL + i] = lr.centroid_A
        params[2 * nL + i] = lr.sigma_A

    return FitResult(
        lines=all_lines,
        params=params,
        model_flux=model_flux,
        continuum=continuum,
        residuals=residuals,
        chi2=np.nan,
        spectrum=spectrum,
        line_names=all_line_names,
        constraints=None,
        success=all(
            (r.success if not hasattr(r, "best_fit") else r.best_fit.success)
            for r in window_results
        ),
    )


def fit_lines(
    spectrum: Spectrum,
    z: float,
    *,
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    wave_range_A: tuple[float, float] | None = None,
    wave_windows_A: list[tuple[float, float]] | None = None,
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
        Observed wavelength range (Angstrom).  Mutually exclusive with
        *wave_windows_A*.
    wave_windows_A : list of (float, float), optional
        Fit the spectrum in multiple independent wavelength windows,
        each with its own continuum subtraction.  Useful for stacked
        spectra where the continuum shape varies across the full
        wavelength range (e.g. UV-normalised stacks).  Each tuple is
        ``(lo, hi)`` in observed-frame Angstroms.  Results from all
        windows are merged into a single :class:`FitResult`.  Mutually
        exclusive with *wave_range_A*.
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
        When ``mode != "off"`` and *wave_windows_A* is ``None``.
        Delegates all :class:`FitResult` attributes (``lines``,
        ``params``, ``model_flux``, etc.) via properties, so it can be
        used as a drop-in replacement.
    FitResult
        When ``mode="off"`` or when *wave_windows_A* is specified
        (merged result).
    """
    # --- Multi-window fitting ---
    if wave_windows_A is not None:
        if wave_range_A is not None:
            raise ValueError(
                "Cannot specify both wave_range_A and wave_windows_A."
            )
        if len(wave_windows_A) == 0:
            raise ValueError("wave_windows_A must contain at least one window.")

        # Print resolving power once before the window loop.
        _grating = grating or spectrum.grating
        _R = R or spectrum.R
        if _grating is None and _R is None:
            _R = R_from_pixels(spectrum.wave_um)
        _R_arr = resolve_R(spectrum.wave_um, grating=_grating, R=_R)
        _R_med = float(np.median(_R_arr))
        _R_lo = float(np.min(_R_arr))
        _R_hi = float(np.max(_R_arr))
        if abs(_R_hi - _R_lo) < 10:
            print(f"Resolving power: R = {_R_med:.0f}")
        else:
            print(
                f"Resolving power: R \u2248 {_R_med:.0f} "
                f"(range {_R_lo:.0f}\u2013{_R_hi:.0f})"
            )

        window_results: list[FitResult | BroadFitResult] = []
        for i, (lo, hi) in enumerate(wave_windows_A):
            mask_win = (spectrum.wave_A >= lo) & (spectrum.wave_A <= hi)
            if np.sum(mask_win) < 10:
                raise ValueError(
                    f"Window [{lo:.0f}, {hi:.0f}] \u00c5 contains only "
                    f"{np.sum(mask_win)} pixels."
                )
            cropped = Spectrum(
                wave_um=spectrum.wave_um[mask_win],
                flux_ujy=spectrum.flux_ujy[mask_win],
                err_ujy=spectrum.err_ujy[mask_win],
                grating=spectrum.grating,
                z=spectrum.z,
                R=spectrum.R,
                meta=spectrum.meta,
            )
            win_label = f"Window {i + 1}/{len(wave_windows_A)}"
            print(f"--- {win_label}: {lo:.0f}\u2013{hi:.0f} \u00c5 ---")

            if mode == "off":
                res = _fit_lines_narrow(
                    cropped, z,
                    grating=grating, R=R, lines=lines, deg=deg,
                    n_boot=n_boot, clip_sigma=clip_sigma,
                    n_jobs=n_jobs, sigma_factor=sigma_factor,
                    _label=win_label,
                )
            else:
                res = fit_with_broad(
                    cropped, z,
                    grating=grating, R=R, lines=lines,
                    deg=deg, mode=mode,
                    n_boot=n_boot, n_boot_bic=n_boot_bic,
                    n_jobs=n_jobs, snr_threshold=snr_threshold,
                    bic_delta=bic_delta, sigma_factor=sigma_factor,
                    _print_R=False,
                )
            window_results.append(res)

        merged = _merge_window_results(spectrum, window_results, wave_windows_A)

        if save_path is not None:
            export_lines_txt(merged, save_path, z=z)

        return merged

    # --- Single-window fitting (default) ---
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
