"""Diagnostic plots for MCMC results.

Corner plots, trace plots, and flux posterior histograms.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .result import MCMCResult

logger = logging.getLogger(__name__)


def plot_corner(
    result: MCMCResult,
    *,
    params: list[str] | None = None,
    truths: np.ndarray | None = None,
    quantiles: list[float] | None = None,
    y_scale: str = "linear",
    **corner_kwargs,
) -> Figure:
    """Corner plot of posterior samples.

    Parameters
    ----------
    result : MCMCResult
        MCMC result.
    params : list of str, optional
        Parameter names to include (e.g. ``["A_OIII_5007", "sigma_Ha"]``).
        If ``None``, plots all free parameters.
    truths : np.ndarray, optional
        True values to mark on the plot.
    quantiles : list of float, optional
        Quantiles for corner (default ``[0.16, 0.5, 0.84]``).
    y_scale : {"linear", "log"}
        Scaling of the counts axis on the diagonal 1-D marginal
        histograms (default ``"linear"``), applied via corner's
        ``hist_kwargs``.  The off-diagonal panels are 2-D parameter-vs-
        parameter densities, so their axes are unaffected.
    **corner_kwargs
        Additional keyword arguments passed to :func:`corner.corner`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import corner

    from jwspecfit.plotting import _validate_y_scale

    y_scale = _validate_y_scale(y_scale)

    if y_scale == "log":
        hist_kwargs = dict(corner_kwargs.pop("hist_kwargs", None) or {})
        hist_kwargs.setdefault("log", True)
        corner_kwargs["hist_kwargs"] = hist_kwargs

    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]

    if params is not None:
        # Resolve named params to column indices in flat_chains.
        nL = len(result.line_names)
        idx_map = {name: i for i, name in enumerate(result.line_names)}
        col_indices = []
        labels = []
        for pname in params:
            if pname.startswith("A_"):
                line_key = pname[2:]
                if line_key in idx_map:
                    col_indices.append(idx_map[line_key])
                    labels.append(pname)
            elif pname.startswith("mu_"):
                line_key = pname[3:]
                if line_key in idx_map:
                    col_indices.append(nL + idx_map[line_key])
                    labels.append(pname)
            elif pname.startswith("sigma_"):
                line_key = pname[6:]
                if line_key in idx_map:
                    col_indices.append(2 * nL + idx_map[line_key])
                    labels.append(pname)

        data = result.flat_chains[:, col_indices]
        if truths is not None:
            truths = [truths[i] for i in col_indices]
    else:
        data = result.flat_chains_free
        n_free = data.shape[1]
        labels = [f"p{i}" for i in range(n_free)]

    with warnings.catch_warnings():
        if y_scale == "log":
            # corner sets the diagonal ylim bottom to 0, which matplotlib
            # ignores (and warns about) on a log axis — the autoscaled
            # positive limits it falls back to are what we want anyway.
            warnings.filterwarnings(
                "ignore",
                message=".*non-positive ylim on a log-scaled axis.*",
                category=UserWarning,
            )
        fig = corner.corner(
            data,
            labels=labels,
            truths=truths,
            quantiles=quantiles,
            show_titles=True,
            title_kwargs={"fontsize": 10},
            **corner_kwargs,
        )
    return fig


def plot_traces(
    result: MCMCResult,
    *,
    params: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    y_scale: str = "linear",
) -> Figure:
    """Trace plots of MCMC chains.

    Parameters
    ----------
    result : MCMCResult
        MCMC result.  Must have ``chains`` (i.e. from emcee).
    params : list of str, optional
        Parameter names to plot.  If ``None``, plots all free parameters.
    figsize : tuple, optional
        Figure size.
    y_scale : {"linear", "log"}
        Scaling of the parameter axis on every trace panel (default
        ``"linear"``).  ``"log"`` draws decade ticks (10⁻², 10⁻¹,
        10⁰ …); non-positive samples (e.g. absorption amplitudes) are
        not representable and are clipped from view.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If per-chain samples are not available (e.g. nautilus result).
    """
    from jwspecfit.plotting import _validate_y_scale

    y_scale = _validate_y_scale(y_scale)

    if result.chains is None:
        raise ValueError(
            "Trace plots require per-chain samples; the nautilus sampler "
            "does not produce them. Use sampler='emcee' or 'nuts' instead."
        )

    chains = result.chains  # (n_walkers, n_steps, n_dim)
    n_walkers, n_steps, n_dim = chains.shape

    if params is not None:
        # Resolve named params to free-parameter indices.
        nL = len(result.line_names)
        idx_map = {name: i for i, name in enumerate(result.line_names)}
        free_mask = result.constraints.free_mask() if result.constraints else np.ones(3 * nL, dtype=bool)

        col_indices = []
        labels = []
        for pname in params:
            if pname.startswith("A_"):
                line_key = pname[2:]
                full_idx = idx_map.get(line_key)
            elif pname.startswith("mu_"):
                line_key = pname[3:]
                full_idx = nL + idx_map.get(line_key, -nL)
            elif pname.startswith("sigma_"):
                line_key = pname[6:]
                full_idx = 2 * nL + idx_map.get(line_key, -2 * nL)
            else:
                continue

            if full_idx is not None and free_mask[full_idx]:
                free_idx = int(np.sum(free_mask[:full_idx]))
                col_indices.append(free_idx)
                labels.append(pname)

        n_plot = len(col_indices)
    else:
        col_indices = list(range(n_dim))
        labels = [f"p{i}" for i in range(n_dim)]
        n_plot = n_dim

    if figsize is None:
        figsize = (10, 2.5 * n_plot)

    fig, axes = plt.subplots(n_plot, 1, figsize=figsize, sharex=True)
    if n_plot == 1:
        axes = [axes]

    n_burn = result.sampler_meta.get("n_burn", 0)

    for ax, ci, label in zip(axes, col_indices, labels):
        for w in range(n_walkers):
            ax.plot(chains[w, :, ci], alpha=0.2, lw=0.5, color="C0")
        if n_burn > 0:
            ax.axvline(0, color="red", ls="--", lw=0.8, label=f"burn-in={n_burn}")
        if y_scale == "log":
            ax.set_yscale("log")
        ax.set_ylabel(label, fontsize=9)

    axes[-1].set_xlabel("Step (post burn-in)")
    fig.tight_layout()
    return fig


def plot_flux_posterior(
    result: MCMCResult,
    line_name: str,
    *,
    bins: int = 50,
    ax: plt.Axes | None = None,
    y_scale: str = "linear",
) -> plt.Axes:
    """Histogram of the flux posterior for a single line.

    Parameters
    ----------
    result : MCMCResult
        MCMC result.
    line_name : str
        Line name (e.g. ``"OIII_5007"``).
    bins : int
        Number of histogram bins.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.  If ``None``, creates a new figure.
    y_scale : {"linear", "log"}
        Scaling of the probability-density axis (default ``"linear"``).
        ``"log"`` draws decade ticks (10⁻², 10⁻¹, 10⁰ …), which makes
        the tails of the posterior visible; empty bins are dropped.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from jwspecfit.plotting import _validate_y_scale

    y_scale = _validate_y_scale(y_scale)

    if line_name not in result.lines:
        raise KeyError(f"Line '{line_name}' not found in result.")

    mlr = result.lines[line_name]
    flux_post = mlr.flux_posterior

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.hist(flux_post, bins=bins, density=True, alpha=0.7, color="C0", edgecolor="C0")
    ax.axvline(mlr.flux, color="C1", ls="-", lw=1.5, label="Median")
    ax.axvline(mlr.flux - mlr.flux_err[0], color="C1", ls="--", lw=1.0, label="16th pctl")
    ax.axvline(mlr.flux + mlr.flux_err[1], color="C1", ls="--", lw=1.0, label="84th pctl")

    if y_scale == "log":
        ax.set_yscale("log")

    ax.set_xlabel(f"Flux [{line_name}] (erg/s/cm$^2$)")
    ax.set_ylabel("Probability density")
    ax.legend(fontsize=8)
    ax.set_title(f"{line_name} flux posterior")
    return ax
