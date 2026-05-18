"""jwspecmcmc — MCMC emission-line fitting for JWST NIRSpec spectra.

A companion to ``jwspecfit`` that replaces bootstrap uncertainties with
full Bayesian posterior sampling via **emcee**, **nautilus**, or **NUTS**.

By default, ``fit_lines()`` runs a narrow-only MCMC fit.  Two
independent BIC-based broad component selections can be opted in:

- ``fit_balmer_broad=True`` — Balmer broad (narrow vs. intermediate /
  very-broad / both, on Balmer pixels).
- ``fit_oiii_broad=True``   — [OIII] outflow broad (on OIII pixels).

Either or both can be enabled.

Example
-------
>>> import jwspecfit, jwspecmcmc
>>> spec = jwspecfit.read_fits("spectrum.fits")
>>> result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="nuts")
>>> result.selected_model          # "narrow" | "broad1" | "broad2" | "both"
>>> result.oiii_broad_selected     # bool — independent of selected_model
>>> result.lines["OIII_5007"].flux_err  # asymmetric 68% CI
>>> ratio = result.flux_ratio_posterior("OIII_5007", "HBETA")
"""

from __future__ import annotations

__version__ = "1.0.1"

from typing import Any, Callable

from jwspecfit.io import Spectrum

from ._engine import _fit_lines_mcmc, _fit_with_broad_mcmc
from .io import load_mcmc_result, save_mcmc_result
from .priors import GaussianPrior, LogUniformPrior, PriorSet, UniformPrior, priors_from_bounds
from .result import MCMCBroadFitResult, MCMCLineResult, MCMCResult

__all__ = [
    "fit_lines",
    "fit_with_broad",
    "MCMCResult",
    "MCMCBroadFitResult",
    "MCMCLineResult",
    "UniformPrior",
    "GaussianPrior",
    "LogUniformPrior",
    "PriorSet",
    "priors_from_bounds",
    "save_mcmc_result",
    "load_mcmc_result",
    "plot_corner",
    "plot_traces",
    "plot_flux_posterior",
]


def fit_lines(
    spectrum: Spectrum,
    z: float,
    *,
    sampler: str = "nuts",
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    wave_range_A: tuple[float, float] | None = None,
    deg: int = 2,
    clip_sigma: float = 2.5,
    init_from_mle: bool = True,
    prior_overrides: dict[str, Any] | None = None,
    n_walkers: int | str = "auto",
    n_steps: int = 2000,
    n_burn: int | None = None,
    n_live: int = 2000,
    n_eff: int = 10000,
    n_warmup: int = 500,
    n_samples_nuts: int = 2000,
    n_chains: int = 6,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    progress: bool = True,
    seed: int = 42,
    fit_balmer_broad: bool = False,
    fit_oiii_broad: bool = False,
    fit_hei_broad: bool = False,
    n_boot_bic: int = 100,
    n_jobs: int = -1,
    snr_threshold: float = 5.0,
    oiii_snr_threshold: float = 5.0,
    hei_snr_threshold: float = 5.0,
    bic_delta: float = 6.0,
    sigma_factor: float = 1.0,
    moving_average: bool | int = False,
    tie_balmer_to_oiii: bool = True,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,
    sigma_overrides: dict[str, tuple[float, float]] | None = None,
    centroid_overrides: dict[str, tuple[float, float]] | None = None,
    niv_doublet_ratio: float | None = None,
    ciii_doublet_ratio: float | None = None,
) -> MCMCResult | MCMCBroadFitResult:
    """Fit emission lines using MCMC sampling.

    Narrow-only MCMC fit by default.  Three independent BIC-based
    broad component tests can be opted in to:

    - ``fit_balmer_broad=True`` — Balmer broad selection.
    - ``fit_oiii_broad=True``   — [OIII] outflow selection.
    - ``fit_hei_broad=True``    — He I broad selection (shared
      kinematics across all observable HeI lines).

    Any combination can be enabled.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum.
    z : float
        Source redshift.
    sampler : str
        ``"nuts"`` (default), ``"emcee"``, or ``"nautilus"``.
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
        Emcee walkers.  ``"auto"`` (default) picks based on n_dim and CPU cores.
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
    fit_balmer_broad : bool
        If ``True``, run BIC selection for a broad Balmer component
        (narrow vs. intermediate vs. very-broad vs. both), gated by
        Hα SNR.  Default ``False`` (narrow-only).
    fit_oiii_broad : bool
        If ``True``, run an independent BIC test for a broad
        component on [OIII] 5007/4959 (outflow signature), gated by
        [OIII] 5007 SNR.  Default ``False`` (narrow-only).
    fit_hei_broad : bool
        If ``True``, run an independent BIC test for a broad He I
        component on all observable He I lines (5877/6680/4472/...),
        with all broad HeI lines sharing kinematics within each tier.
        Gated by the best narrow HeI SNR.  Default ``False``.
    n_boot_bic : int
        Bootstrap iterations for BIC model selection (default 100).
        Only used when at least one of the broad flags is True.
    n_jobs : int
        Parallel jobs for BIC bootstrap (default ``-1``).
    snr_threshold : float
        Minimum Hα SNR to attempt Balmer broad fitting (default 5.0).
    oiii_snr_threshold : float
        Minimum [OIII] 5007/4959 SNR to attempt OIII broad fitting
        (default 5.0).
    bic_delta : float
        ΔBIC threshold for model selection (default 6.0).
    sigma_factor : float
        Multiplicative factor on the upper line-width bound.
        Use values > 1 for stacked spectra (default 1.0).
    moving_average : bool or int
        If ``False`` (default), use polynomial continuum.  If ``True``,
        use a median filter with a default window of 75 pixels.  If an
        ``int``, use that as the median-filter window size.
    tie_uv_doublets : bool
        Tie UV doublet kinematics and fix resonance-line flux ratios.
        Recommended for stacked spectra where doublets are poorly
        resolved (default ``False``).

    Returns
    -------
    MCMCBroadFitResult
        When at least one broad flag is True.  Delegates all
        :class:`MCMCResult` attributes via properties.
    MCMCResult
        When both broad flags are False.
    """
    if not fit_balmer_broad and not fit_oiii_broad and not fit_hei_broad:
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
            n_warmup=n_warmup,
            n_samples_nuts=n_samples_nuts,
            n_chains=n_chains,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            progress=progress,
            seed=seed,
            sigma_factor=sigma_factor,
            moving_average=moving_average,
            tie_balmer_to_oiii=tie_balmer_to_oiii,
            tie_uv_doublets=tie_uv_doublets,
            tie_uv_centroids=tie_uv_centroids,
            tie_uv_widths=tie_uv_widths,
            sigma_overrides=sigma_overrides,
            centroid_overrides=centroid_overrides,
            niv_doublet_ratio=niv_doublet_ratio,
            ciii_doublet_ratio=ciii_doublet_ratio,
        )

    return _fit_with_broad_mcmc(
        spectrum, z,
        sampler=sampler,
        grating=grating,
        R=R,
        lines=lines,
        wave_range_A=wave_range_A,
        deg=deg,
        clip_sigma=clip_sigma,
        fit_balmer_broad=fit_balmer_broad,
        fit_oiii_broad=fit_oiii_broad,
        fit_hei_broad=fit_hei_broad,
        n_boot_bic=n_boot_bic,
        n_jobs=n_jobs,
        snr_threshold=snr_threshold,
        oiii_snr_threshold=oiii_snr_threshold,
        hei_snr_threshold=hei_snr_threshold,
        bic_delta=bic_delta,
        prior_overrides=prior_overrides,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        n_live=n_live,
        n_eff=n_eff,
        n_warmup=n_warmup,
        n_samples_nuts=n_samples_nuts,
        n_chains=n_chains,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        progress=progress,
        seed=seed,
        sigma_factor=sigma_factor,
        moving_average=moving_average,
        tie_balmer_to_oiii=tie_balmer_to_oiii,
        tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids,
        tie_uv_widths=tie_uv_widths,
        sigma_overrides=sigma_overrides,
        centroid_overrides=centroid_overrides,
        niv_doublet_ratio=niv_doublet_ratio,
        ciii_doublet_ratio=ciii_doublet_ratio,
    )


def fit_with_broad(
    spectrum: Spectrum,
    z: float,
    *,
    sampler: str = "nuts",
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    wave_range_A: tuple[float, float] | None = None,
    deg: int = 2,
    clip_sigma: float = 2.5,
    fit_balmer_broad: bool = False,
    fit_oiii_broad: bool = False,
    fit_hei_broad: bool = False,
    n_boot_bic: int = 100,
    n_jobs: int = -1,
    snr_threshold: float = 5.0,
    oiii_snr_threshold: float = 5.0,
    hei_snr_threshold: float = 5.0,
    bic_delta: float = 6.0,
    prior_overrides: dict[str, Any] | None = None,
    n_walkers: int | str = "auto",
    n_steps: int = 2000,
    n_burn: int | None = None,
    n_live: int = 2000,
    n_eff: int = 10000,
    n_warmup: int = 500,
    n_samples_nuts: int = 2000,
    n_chains: int = 6,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    progress: bool = True,
    seed: int = 42,
    sigma_factor: float = 1.0,
    moving_average: bool | int = False,
    tie_balmer_to_oiii: bool = True,
    tie_uv_doublets: bool = True,
    tie_uv_centroids: bool = True,
    tie_uv_widths: bool = True,
    sigma_overrides: dict[str, tuple[float, float]] | None = None,
    centroid_overrides: dict[str, tuple[float, float]] | None = None,
    niv_doublet_ratio: float | None = None,
    ciii_doublet_ratio: float | None = None,
) -> MCMCBroadFitResult:
    """Fit emission lines with BIC-based broad selection, then MCMC.

    Phase 1 uses :func:`jwspecfit.fit_with_broad` (fast least-squares)
    for BIC model selection.  Phase 2 runs MCMC on the winning model.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum.
    z : float
        Source redshift.
    sampler : str
        ``"nuts"`` (default), ``"emcee"``, or ``"nautilus"``.
    fit_balmer_broad : bool
        Run BIC selection for Balmer broad (default ``False``).
    fit_oiii_broad : bool
        Run independent BIC test for [OIII] broad (default ``False``).
    fit_hei_broad : bool
        Run independent BIC test for HeI broad with shared kinematics
        across all observable HeI lines (default ``False``).
    snr_threshold : float
        Minimum Hα SNR for Balmer broad (default 5.0).
    oiii_snr_threshold : float
        Minimum [OIII] 5007 SNR for OIII broad (default 5.0).
    bic_delta : float
        ΔBIC threshold (default 6.0).

    See :func:`fit_lines` for the full parameter list.

    Returns
    -------
    MCMCBroadFitResult
    """
    return _fit_with_broad_mcmc(
        spectrum, z,
        sampler=sampler,
        grating=grating,
        R=R,
        lines=lines,
        wave_range_A=wave_range_A,
        deg=deg,
        clip_sigma=clip_sigma,
        fit_balmer_broad=fit_balmer_broad,
        fit_oiii_broad=fit_oiii_broad,
        fit_hei_broad=fit_hei_broad,
        n_boot_bic=n_boot_bic,
        n_jobs=n_jobs,
        snr_threshold=snr_threshold,
        oiii_snr_threshold=oiii_snr_threshold,
        hei_snr_threshold=hei_snr_threshold,
        bic_delta=bic_delta,
        prior_overrides=prior_overrides,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        n_live=n_live,
        n_eff=n_eff,
        n_warmup=n_warmup,
        n_samples_nuts=n_samples_nuts,
        n_chains=n_chains,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        progress=progress,
        seed=seed,
        sigma_factor=sigma_factor,
        moving_average=moving_average,
        tie_balmer_to_oiii=tie_balmer_to_oiii,
        tie_uv_doublets=tie_uv_doublets,
        tie_uv_centroids=tie_uv_centroids,
        tie_uv_widths=tie_uv_widths,
        sigma_overrides=sigma_overrides,
        centroid_overrides=centroid_overrides,
        niv_doublet_ratio=niv_doublet_ratio,
        ciii_doublet_ratio=ciii_doublet_ratio,
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
