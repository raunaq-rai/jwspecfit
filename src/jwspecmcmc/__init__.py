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

from importlib.metadata import PackageNotFoundError, version

try:  # jwspecmcmc ships as part of the jwspecfit distribution
    __version__ = version("jwspecfit")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"

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
    centroid_vmax: float = 500.0,
    centroid_max_sigma: float = 1.0,
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

    A grouped, tabular version of every argument below is available in
    the user guide: :doc:`/user_guide/jwspecmcmc`.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum (observed wavelength, flux, and error arrays).
    z : float
        Source redshift.  Sets each observed line position via
        ``lambda_obs = lambda_rest * (1 + z)``.
    sampler : str
        Posterior-sampling backend: ``"nuts"`` (default; NumPyro
        Hamiltonian Monte Carlo), ``"emcee"`` (affine-invariant
        ensemble), or ``"nautilus"`` (importance nested sampling, which
        also returns the Bayesian evidence).
    grating : str, optional
        NIRSpec grating name (e.g. ``"G395H"``), used to set the
        resolution curve.  If ``None``, estimated from the pixel spacing.
    R : float or callable, optional
        Resolving power as a scalar or a callable ``R(lambda_um)``.
        Overrides *grating* when given; if ``None`` it is taken from
        *grating* or the pixel spacing.
    lines : list of str, optional
        Names of emission lines to fit (keys of
        :data:`jwspecfit.lines.REST_LINES_A`).  If ``None``, the lines
        that fall inside the spectrum at redshift *z* are auto-selected.
    wave_range_A : tuple of float, optional
        ``(lo, hi)`` observed-frame wavelength window in Angstrom to
        restrict the fit to.  If ``None``, the full spectrum is used.
    deg : int
        Degree of the polynomial continuum (default 2).
    clip_sigma : float
        Sigma-clip threshold used when fitting the continuum (default 2.5).
    moving_average : bool or int
        Continuum model.  ``False`` (default) fits a polynomial of degree
        *deg*; ``True`` uses a 75-pixel running-median filter; an ``int``
        uses a running-median filter of that window size.
    init_from_mle : bool
        If ``True`` (default), seed the sampler from a fast least-squares
        MLE (via :func:`jwspecfit.fit_lines`).  Strongly recommended; it
        speeds convergence and reduces stuck chains.
    n_warmup : int
        NUTS only.  Warm-up (tuning) iterations per chain before samples
        are kept; adapts the step size and mass matrix (default 500).
    n_samples_nuts : int
        NUTS only.  Posterior draws kept per chain after warm-up; the
        total number of samples is ``n_chains * n_samples_nuts``
        (default 2000).
    n_chains : int
        NUTS only.  Number of independent chains run in parallel; enables
        the Gelman-Rubin R-hat diagnostic (default 6).
    target_accept_prob : float
        NUTS only.  Target acceptance probability for step-size
        adaptation.  Raise toward 0.95 if the sampler reports divergences
        (default 0.8).
    max_tree_depth : int
        NUTS only.  Maximum NUTS binary-tree depth, capping the leapfrog
        steps per iteration at ``2 ** max_tree_depth`` (default 10).
    n_walkers : int or str
        emcee only.  Number of ensemble walkers; ``"auto"`` (default)
        chooses a value from the parameter count and CPU cores (must be
        at least ``2 * n_dim``).
    n_steps : int
        emcee only.  Number of steps per walker (default 2000).
    n_burn : int or None
        emcee only.  Burn-in steps discarded; if ``None`` (default) it is
        estimated automatically from the integrated autocorrelation time.
    n_live : int
        nautilus only.  Number of live points (default 2000).
    n_eff : int
        nautilus only.  Target effective posterior sample size
        (default 10000).
    fit_balmer_broad : bool
        If ``True``, run BIC selection for a broad Balmer component
        (narrow vs. intermediate vs. very-broad vs. both), gated by the
        Hα SNR.  Default ``False`` (narrow-only).
    fit_oiii_broad : bool
        If ``True``, run an independent BIC test for a broad component on
        [OIII] 4959/5007 (an outflow signature), gated by the [OIII] 5007
        SNR.  Default ``False``.
    fit_hei_broad : bool
        If ``True``, run an independent BIC test for a broad He I
        component shared (with common kinematics per tier) across all
        observable He I lines, gated by the best narrow He I SNR.
        Default ``False``.
    n_boot_bic : int
        Bootstrap iterations for the BIC model selection; ``0`` uses a
        single-point BIC.  Only used when a broad flag is set
        (default 100).
    n_jobs : int
        Number of parallel workers for the BIC bootstrap; ``-1`` uses all
        cores (default -1).
    snr_threshold : float
        Minimum Hα SNR required to attempt Balmer broad fitting
        (default 5.0).
    oiii_snr_threshold : float
        Minimum [OIII] 5007 SNR required to attempt [OIII] broad fitting
        (default 5.0).
    hei_snr_threshold : float
        Minimum He I SNR (of the best narrow He I line) required to
        attempt He I broad fitting (default 5.0).
    bic_delta : float
        Minimum ΔBIC by which a more complex model must beat the simpler
        one to be selected (default 6.0).
    tie_balmer_to_oiii : bool
        If ``True`` (default), tie the narrow Balmer (and [NII]) line
        widths and centroids to [OIII] 5007 in velocity space.
    tie_uv_doublets : bool
        If ``True`` (default), tie UV doublet kinematics and fix
        resonance-line flux ratios.  Recommended for stacked or poorly
        resolved spectra.
    tie_uv_centroids : bool
        When *tie_uv_doublets* is on, tie each UV doublet's secondary
        centroid to its primary in velocity space; if ``False`` the
        secondary centroids are free (default ``True``).
    tie_uv_widths : bool
        If ``True`` (default), tie the widths of the UV intercombination
        lines (CIII], NIV, NIII, SiIII) to a single shared velocity
        dispersion.
    niv_doublet_ratio : float or None
        If given, fix the N IV] flux ratio ``F(1483) / F(1486)`` to this
        value; ``None`` (default) leaves the two amplitudes free.
    ciii_doublet_ratio : float or None
        If given, fix the C III] flux ratio ``F(1909) / F(1907)`` to this
        value; ``None`` (default) leaves the two amplitudes free.
    sigma_overrides : dict, optional
        Per-line width bounds ``{line_name: (sigma_lo, sigma_hi)}`` in
        Angstrom, overriding the automatic grating-based bounds.
    centroid_overrides : dict, optional
        Per-line centroid bounds ``{line_name: (mu_lo, mu_hi)}`` in
        Angstrom, overriding the automatic velocity-based bounds.
    sigma_factor : float
        Multiplier applied to the upper line-width bound; use values > 1
        for stacked spectra whose lines are broadened (default 1.0).
    centroid_vmax : float
        Maximum centroid offset, in km/s, allowed for each line
        (default 500.0).
    centroid_max_sigma : float
        Extra resolution-aware cap on narrow-line centroid offsets: the
        margin is ``min(centroid_vmax-based, centroid_max_sigma *
        sigma_instrument)`` (default 1.0).
    prior_overrides : dict, optional
        Per-parameter prior overrides keyed by parameter name, e.g.
        ``{"A_OIII_5007": GaussianPrior(1e-17, 1e-18, 0, 1e-15)}``.
        Keys follow the ``A_<line>`` / ``mu_<line>`` / ``sigma_<line>``
        convention; values are :class:`Prior` instances.
    progress : bool
        Show a progress bar during sampling (default ``True``).
    seed : int
        Random seed for initialisation and sampling (default 42).

    Returns
    -------
    MCMCBroadFitResult
        When at least one broad flag is ``True``.  Delegates all
        :class:`MCMCResult` attributes via properties.
    MCMCResult
        When all broad flags are ``False``.
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
            centroid_vmax=centroid_vmax,
            centroid_max_sigma=centroid_max_sigma,
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
        centroid_vmax=centroid_vmax,
        centroid_max_sigma=centroid_max_sigma,
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
    centroid_vmax: float = 500.0,
    centroid_max_sigma: float = 1.0,
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

    Unlike :func:`fit_lines`, this entry point *always* performs the
    BIC broad-selection step, so the broad flags below are its primary
    controls.  Every other argument (samplers, continuum, kinematic
    ties, doublet ratios, line-bound overrides, priors) is identical to
    :func:`fit_lines`; see that function's docstring or the grouped
    tables in :doc:`/user_guide/jwspecmcmc` for the full reference.

    Parameters
    ----------
    spectrum : Spectrum
        Input spectrum (observed wavelength, flux, and error arrays).
    z : float
        Source redshift.
    fit_balmer_broad : bool
        If ``True``, run BIC selection for a broad Balmer component
        (narrow vs. intermediate vs. very-broad vs. both), gated by the
        Hα SNR.  Default ``False``.
    fit_oiii_broad : bool
        If ``True``, run an independent BIC test for a broad [OIII]
        4959/5007 component (outflow signature), gated by the [OIII] 5007
        SNR.  Default ``False``.
    fit_hei_broad : bool
        If ``True``, run an independent BIC test for a broad He I
        component shared across all observable He I lines, gated by the
        best narrow He I SNR.  Default ``False``.
    n_boot_bic : int
        Bootstrap iterations for the BIC model selection; ``0`` uses a
        single-point BIC (default 100).
    n_jobs : int
        Parallel workers for the BIC bootstrap; ``-1`` uses all cores
        (default -1).
    snr_threshold : float
        Minimum Hα SNR to attempt Balmer broad fitting (default 5.0).
    oiii_snr_threshold : float
        Minimum [OIII] 5007 SNR to attempt [OIII] broad fitting
        (default 5.0).
    hei_snr_threshold : float
        Minimum He I SNR to attempt He I broad fitting (default 5.0).
    bic_delta : float
        Minimum ΔBIC by which a more complex model must beat the simpler
        one to be selected (default 6.0).

    Other Parameters
    ----------------
    sampler, grating, R, lines, wave_range_A, deg, clip_sigma, \
    moving_average, n_walkers, n_steps, n_burn, n_live, n_eff, n_warmup, \
    n_samples_nuts, n_chains, target_accept_prob, max_tree_depth, \
    tie_balmer_to_oiii, tie_uv_doublets, tie_uv_centroids, tie_uv_widths, \
    niv_doublet_ratio, ciii_doublet_ratio, sigma_overrides, \
    centroid_overrides, sigma_factor, centroid_vmax, centroid_max_sigma, \
    prior_overrides, progress, seed
        Identical to the corresponding arguments of :func:`fit_lines`;
        see that function or the grouped tables in
        :doc:`/user_guide/jwspecmcmc`.  (``init_from_mle`` is not exposed
        here because Phase 2 always initialises from the Phase 1 fit.)

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
        centroid_vmax=centroid_vmax,
        centroid_max_sigma=centroid_max_sigma,
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
