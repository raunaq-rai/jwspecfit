"""jwspecmcmc — MCMC emission-line fitting for JWST NIRSpec spectra.

A companion to ``jwspecfit`` that replaces bootstrap uncertainties with
full Bayesian posterior sampling via **emcee** or **nautilus**.

Example
-------
>>> import jwspecfit
>>> import jwspecmcmc
>>> spec = jwspecfit.read_fits("spectrum.fits")
>>> result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="emcee", n_steps=2000)
>>> result.lines["OIII_5007"].flux_err  # asymmetric 68% CI
>>> ratio = result.flux_ratio_posterior("OIII_5007", "HBETA")
"""

from __future__ import annotations

__version__ = "0.1.0"

from typing import Any, Callable

from jwspecfit.io import Spectrum

from ._engine import _fit_lines_mcmc
from .priors import GaussianPrior, LogUniformPrior, PriorSet, UniformPrior, priors_from_bounds
from .result import MCMCLineResult, MCMCResult

__all__ = [
    "fit_lines",
    "MCMCResult",
    "MCMCLineResult",
    "UniformPrior",
    "GaussianPrior",
    "LogUniformPrior",
    "PriorSet",
    "priors_from_bounds",
    "plot_corner",
    "plot_traces",
    "plot_flux_posterior",
]


def fit_lines(
    spectrum: Spectrum,
    z: float,
    *,
    sampler: str = "emcee",
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    wave_range_A: tuple[float, float] | None = None,
    deg: int = 2,
    clip_sigma: float = 2.5,
    init_from_mle: bool = True,
    prior_overrides: dict[str, Any] | None = None,
    n_walkers: int = 64,
    n_steps: int = 2000,
    n_burn: int | None = None,
    n_live: int = 2000,
    n_eff: int = 10000,
    progress: bool = True,
    seed: int = 42,
) -> MCMCResult:
    """Fit emission lines using MCMC sampling.

    Thin wrapper around the internal engine; see
    :func:`jwspecmcmc._engine._fit_lines_mcmc` for full documentation.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum.
    z : float
        Source redshift.
    sampler : str
        ``"emcee"`` (default) or ``"nautilus"``.
    grating : str, optional
        Grating name.
    R : float or callable, optional
        Resolving power.
    lines : list of str, optional
        Lines to fit.
    wave_range_A : tuple, optional
        Observed wavelength range (Angstrom).
    deg : int
        Continuum polynomial degree.
    clip_sigma : float
        Continuum sigma-clipping threshold.
    init_from_mle : bool
        Initialise walkers from a least-squares MLE (default ``True``).
    prior_overrides : dict, optional
        Per-parameter prior overrides keyed by name, e.g.
        ``{"A_OIII_5007": GaussianPrior(1e-17, 1e-18, 0, 1e-15)}``.
    n_walkers : int
        Emcee walkers (default 64).
    n_steps : int
        Emcee steps (default 2000).
    n_burn : int or None
        Emcee burn-in (auto if ``None``).
    n_live : int
        Nautilus live points (default 2000).
    n_eff : int
        Nautilus target effective samples (default 10000).
    progress : bool
        Show progress bar.
    seed : int
        Random seed.

    Returns
    -------
    MCMCResult
    """
    return _fit_lines_mcmc(
        spectrum, z,
        sampler=sampler,
        grating=grating,
        R=R,
        lines=lines,
        wave_range_A=wave_range_A,
        deg=deg,
        clip_sigma=clip_sigma,
        init_from_mle=init_from_mle,
        prior_overrides=prior_overrides,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        n_live=n_live,
        n_eff=n_eff,
        progress=progress,
        seed=seed,
    )


def plot_corner(*args, **kwargs):
    """Corner plot of posterior samples.

    See :func:`jwspecmcmc.plotting.plot_corner` for full documentation.
    """
    from .plotting import plot_corner as _plot_corner
    return _plot_corner(*args, **kwargs)


def plot_traces(*args, **kwargs):
    """Trace plots of MCMC chains.

    See :func:`jwspecmcmc.plotting.plot_traces` for full documentation.
    """
    from .plotting import plot_traces as _plot_traces
    return _plot_traces(*args, **kwargs)


def plot_flux_posterior(*args, **kwargs):
    """Flux posterior histogram for a single line.

    See :func:`jwspecmcmc.plotting.plot_flux_posterior` for full documentation.
    """
    from .plotting import plot_flux_posterior as _plot_flux_posterior
    return _plot_flux_posterior(*args, **kwargs)
