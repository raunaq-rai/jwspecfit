"""MCMC result containers.

:class:`MCMCResult` holds the full posterior chains and derived
quantities.  :meth:`MCMCResult.to_fit_result` converts the median
posterior to a :class:`jwspecfit.fitter.FitResult` for compatibility
with :func:`jwspecfit.plotting.plot_fit`.

:class:`MCMCBroadFitResult` wraps an :class:`MCMCResult` with BIC
model-selection metadata for broad Balmer component fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import sqrt, pi
from typing import Any

import numpy as np

from jwspecfit.constraints import ConstraintSet
from jwspecfit.fitter import FitResult, LineResult
from jwspecfit.io import Spectrum, _ujy_to_flam, _flam_to_ujy
from jwspecfit.lines import REST_LINES_A
from jwspecfit.models import build_model

logger = logging.getLogger(__name__)

_SQRT2PI = sqrt(2.0 * pi)


@dataclass
class MCMCLineResult:
    """MCMC posterior summary for a single emission line.

    Parameters
    ----------
    name : str
        Line name.
    rest_wave_A : float
        Rest-frame wavelength (Angstrom).
    amplitude : float
        Median posterior amplitude.
    amplitude_err : tuple of float
        (lower, upper) 68% credible interval half-widths on amplitude.
    centroid_A : float
        Median posterior centroid (Angstrom).
    centroid_err : tuple of float
        (lower, upper) 68% CI half-widths on centroid.
    sigma_A : float
        Median posterior sigma (Angstrom).
    sigma_err : tuple of float
        (lower, upper) 68% CI half-widths on sigma.
    flux : float
        Median integrated flux (= amplitude for area-normalised Gaussians).
    flux_err : tuple of float
        (lower, upper) 68% CI half-widths on flux.
    flux_posterior : np.ndarray
        Full flux posterior samples.
    ew_A : float
        Median rest-frame equivalent width (Angstrom).
    snr : float
        Signal-to-noise ratio (flux / mean of flux_err tuple).
    """

    name: str
    rest_wave_A: float
    amplitude: float
    amplitude_err: tuple[float, float]
    centroid_A: float
    centroid_err: tuple[float, float]
    sigma_A: float
    sigma_err: tuple[float, float]
    flux: float
    flux_err: tuple[float, float]
    flux_posterior: np.ndarray
    ew_A: float
    snr: float


@dataclass
class MCMCResult:
    """Container for a complete MCMC fitting result.

    Parameters
    ----------
    lines : dict of MCMCLineResult
        Per-line posterior summaries, keyed by line name.
    flat_chains : np.ndarray
        Flattened posterior samples in the **full** parameter space,
        shape ``(n_samples, 3 * n_lines)``.
    flat_chains_free : np.ndarray
        Flattened posterior samples in the **free** parameter space,
        shape ``(n_samples, n_free)``.
    flat_log_prob : np.ndarray
        Log-posterior for each sample.
    chains : np.ndarray or None
        Raw chains of shape ``(n_walkers, n_steps, n_free)`` (emcee)
        or ``None`` (nautilus).
    params : np.ndarray
        Median posterior in the full parameter space.
    model_flux : np.ndarray
        Median model flux (µJy, continuum-subtracted).
    continuum : np.ndarray
        Continuum estimate (µJy).
    spectrum : Spectrum
        Input spectrum.
    line_names : list of str
        Ordered line names.
    constraints : ConstraintSet
        Applied constraints.
    convergence : dict
        Convergence diagnostics (R-hat, ESS).
    sampler_name : str
        Name of the sampler used (``"emcee"`` or ``"nautilus"``).
    sampler_meta : dict
        Additional sampler metadata (n_walkers, n_steps, etc.).
    """

    lines: dict[str, MCMCLineResult]
    flat_chains: np.ndarray
    flat_chains_free: np.ndarray
    flat_log_prob: np.ndarray
    chains: np.ndarray | None
    params: np.ndarray
    model_flux: np.ndarray
    continuum: np.ndarray
    spectrum: Spectrum
    line_names: list[str] = field(default_factory=list)
    constraints: ConstraintSet | None = None
    convergence: dict[str, Any] = field(default_factory=dict)
    sampler_name: str = ""
    sampler_meta: dict[str, Any] = field(default_factory=dict)

    def to_fit_result(self) -> FitResult:
        """Convert to a :class:`jwspecfit.fitter.FitResult`.

        Uses the median posterior as the best fit and the mean of the
        asymmetric 68% CI as the symmetric flux error.

        Returns
        -------
        FitResult
        """
        # Convert median model to residuals.
        spec = self.spectrum
        resid_ujy = spec.flux_ujy - self.continuum - self.model_flux

        # Reduced chi-squared from median model.
        flam = _ujy_to_flam(spec.flux_ujy - self.continuum, spec.wave_um)
        flam_err = _ujy_to_flam(spec.err_ujy, spec.wave_um)
        model_flam = _ujy_to_flam(self.model_flux, spec.wave_um)
        valid = np.isfinite(flam) & np.isfinite(flam_err) & (flam_err > 0)
        r = np.zeros_like(flam)
        r[valid] = (flam[valid] - model_flam[valid]) / flam_err[valid]
        n_data = np.sum(valid)
        n_free = self.flat_chains_free.shape[1] if self.flat_chains_free.ndim == 2 else 0
        dof = max(n_data - n_free, 1)
        chi2_red = float(np.nansum(r**2)) / dof

        line_results = {}
        for name, mlr in self.lines.items():
            # Symmetric flux error = mean of asymmetric CI half-widths.
            flux_err_sym = 0.5 * (mlr.flux_err[0] + mlr.flux_err[1])
            line_results[name] = LineResult(
                name=mlr.name,
                rest_wave_A=mlr.rest_wave_A,
                amplitude=mlr.amplitude,
                centroid_A=mlr.centroid_A,
                sigma_A=mlr.sigma_A,
                flux=mlr.flux,
                flux_err=flux_err_sym,
                ew_A=mlr.ew_A,
                snr_int_err=mlr.snr,
                snr_int_cont=0.0,
                snr_peak_err=0.0,
                snr_peak_cont=0.0,
            )

        return FitResult(
            lines=line_results,
            params=self.params,
            model_flux=self.model_flux,
            continuum=self.continuum,
            residuals=resid_ujy,
            chi2=chi2_red,
            spectrum=spec,
            line_names=self.line_names,
            constraints=self.constraints,
            success=True,
        )

    def flux_ratio_posterior(
        self,
        line_a: str,
        line_b: str,
    ) -> np.ndarray:
        """Compute the posterior distribution of a flux ratio.

        Parameters
        ----------
        line_a : str
            Numerator line name.
        line_b : str
            Denominator line name.

        Returns
        -------
        np.ndarray
            Posterior samples of ``flux(line_a) / flux(line_b)``.

        Raises
        ------
        KeyError
            If either line is not in the result.
        """
        if line_a not in self.lines:
            raise KeyError(f"Line '{line_a}' not found in result.")
        if line_b not in self.lines:
            raise KeyError(f"Line '{line_b}' not found in result.")

        flux_a = self.lines[line_a].flux_posterior
        flux_b = self.lines[line_b].flux_posterior

        # Avoid division by zero: mask samples where denominator is non-positive.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(flux_b > 0, flux_a / flux_b, np.nan)
        return ratio


@dataclass
class MCMCBroadFitResult:
    """MCMC result with BIC-based broad Balmer component selection.

    Wraps an :class:`MCMCResult` (full MCMC posteriors on the winning
    model) together with BIC model-selection metadata.

    Parameters
    ----------
    mcmc_result : MCMCResult
        Full MCMC posteriors for the selected model.
    selected_model : str
        Model name: ``"narrow"``, ``"broad1"``, ``"broad2"``, or ``"both"``.
    bic_narrow : float
        BIC for narrow-only model.
    bic_broad1 : float
        BIC for narrow + intermediate broad model (NaN if not attempted).
    bic_broad2 : float
        BIC for narrow + very broad model (NaN if not attempted).
    bic_both : float
        BIC for narrow + both broad components (NaN if not attempted).
    """

    mcmc_result: MCMCResult
    selected_model: str
    bic_narrow: float
    bic_broad1: float
    bic_broad2: float
    bic_both: float

    # Delegate all MCMCResult attributes for full API compatibility.

    @property
    def lines(self) -> dict[str, MCMCLineResult]:
        """Per-line posterior summaries."""
        return self.mcmc_result.lines

    @property
    def flat_chains(self) -> np.ndarray:
        """Flattened posterior samples (full parameter space)."""
        return self.mcmc_result.flat_chains

    @property
    def flat_chains_free(self) -> np.ndarray:
        """Flattened posterior samples (free parameter space)."""
        return self.mcmc_result.flat_chains_free

    @property
    def flat_log_prob(self) -> np.ndarray:
        """Log-posterior for each sample."""
        return self.mcmc_result.flat_log_prob

    @property
    def chains(self) -> np.ndarray | None:
        """Raw walker chains (emcee) or None (nautilus)."""
        return self.mcmc_result.chains

    @property
    def params(self) -> np.ndarray:
        """Median posterior in the full parameter space."""
        return self.mcmc_result.params

    @property
    def model_flux(self) -> np.ndarray:
        """Median model flux (continuum-subtracted)."""
        return self.mcmc_result.model_flux

    @property
    def continuum(self) -> np.ndarray:
        """Continuum estimate."""
        return self.mcmc_result.continuum

    @property
    def spectrum(self) -> Spectrum:
        """Input spectrum."""
        return self.mcmc_result.spectrum

    @property
    def line_names(self) -> list[str]:
        """Ordered line names."""
        return self.mcmc_result.line_names

    @property
    def constraints(self) -> ConstraintSet | None:
        """Applied constraints."""
        return self.mcmc_result.constraints

    @property
    def convergence(self) -> dict[str, Any]:
        """Convergence diagnostics."""
        return self.mcmc_result.convergence

    @property
    def sampler_name(self) -> str:
        """Name of the sampler used."""
        return self.mcmc_result.sampler_name

    @property
    def sampler_meta(self) -> dict[str, Any]:
        """Additional sampler metadata."""
        return self.mcmc_result.sampler_meta

    def to_fit_result(self) -> FitResult:
        """Convert to a :class:`jwspecfit.fitter.FitResult`.

        Delegates to :meth:`MCMCResult.to_fit_result`.

        Returns
        -------
        FitResult
        """
        return self.mcmc_result.to_fit_result()

    def flux_ratio_posterior(self, line_a: str, line_b: str) -> np.ndarray:
        """Compute the posterior distribution of a flux ratio.

        Delegates to :meth:`MCMCResult.flux_ratio_posterior`.

        Parameters
        ----------
        line_a : str
            Numerator line name.
        line_b : str
            Denominator line name.

        Returns
        -------
        np.ndarray
            Posterior samples of ``flux(line_a) / flux(line_b)``.
        """
        return self.mcmc_result.flux_ratio_posterior(line_a, line_b)
