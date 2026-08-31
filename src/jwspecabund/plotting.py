"""Diagnostic plots for abundance measurements.

Currently one figure: the self-consistent O++ T_e-n_e solve, drawn the way
Hsiao et al. (2026, arXiv:2608.20339) present it in their figure 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .direct import (
    OIII_GRID_LOGNE,
    OIII_GRID_LOGTE,
    _oiii_te_ne_grid,
    _te_curve,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_te_ne_diagnostic"]


#: Curve styling: dataclass field, ratio key on the grid, colour, label.
#: Colours follow Hsiao et al. figure 3 (blue / red / green).
_CURVES = (
    ("Te_curve_4363", "log_R_4363", "4363", "tab:blue",
     r"[O III] $\lambda 4363/(\lambda 5007+\lambda 4959)$"),
    ("Te_curve_1666_4363", "log_R_1666_4363", "1666_4363", "tab:red",
     r"O III] $\lambda 1666$ / [O III] $\lambda 4363$"),
    ("Te_curve_5007_1666", "log_R_5007_1666", "5007_1666", "tab:green",
     r"[O III] $\lambda 5007$ / O III] $\lambda 1666$"),
)


def _as_solution(obj: Any) -> Any:
    """Accept a SelfConsistentOIII or an AbundanceResult carrying one."""
    if hasattr(obj, "Te_curve_4363"):
        return obj
    sc = getattr(obj, "Te_ne_selfconsistent", None)
    if sc is None:
        raise ValueError(
            "No self-consistent O III solution on this result. It is computed "
            "only when O III] 1666, [OIII] 4363 and [OIII] 5007 are all "
            "detected above snr_auroral; pass self_consistent_OIII=True to "
            "force the attempt, or call compute_Te_ne_OIII directly."
        )
    return sc


def plot_te_ne_diagnostic(
    solution: Any,
    *,
    ax: Any = None,
    show_bands: bool = True,
    show_posterior: bool = True,
    ne_range: tuple[float, float] | None = None,
    Te_range: tuple[float, float] | None = None,
    legend: bool = True,
    save_path: str | None = None,
) -> "Figure":
    """Plot the O++ T_e-n_e curves and their intersection.

    Reproduces figure 3 of Hsiao et al. (2026): the temperature each
    observed O III ratio implies as a function of density, with 1 sigma
    bands from the flux errors.  A single ratio traces a whole curve of
    (T_e, n_e) pairs; where two curves cross, one pair reproduces both, and
    that is the self-consistent solution.  Curves that stay parallel mean
    the density is unconstrained and only an upper limit follows.

    Parameters
    ----------
    solution : SelfConsistentOIII or AbundanceResult
        Output of :func:`~jwspecabund.direct.compute_Te_ne_OIII`, or an
        :class:`~jwspecabund.result.AbundanceResult` whose
        ``Te_ne_selfconsistent`` holds one.
    ax : matplotlib Axes, optional
        Axes to draw on.  A new figure is created when ``None``.
    show_bands : bool
        Shade the 1 sigma band around each curve, from the flux errors
        (default ``True``; ignored when no errors were supplied).
    show_posterior : bool
        Overlay the 1 sigma and 2 sigma posterior contours (default
        ``True``; ignored when no posterior was run).
    ne_range, Te_range : tuple of float, optional
        Axis limits as ``(min, max)`` in cm^-3 and K.  Default to the
        model grid.
    legend : bool
        Draw the legend (default ``True``).
    save_path : str, optional
        Path to save the figure to.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    import matplotlib.pyplot as plt

    sc = _as_solution(solution)
    if sc.logne_grid is None:
        raise ValueError("This solution carries no T_e(n_e) curves to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
    else:
        fig = ax.figure

    ne = 10.0 ** np.asarray(sc.logne_grid)
    grid = None
    errs = sc.log_ratio_errs if show_bands else None

    for field_name, ratio_key, err_key, colour, label in _CURVES:
        curve = getattr(sc, field_name, None)
        if curve is None:
            continue
        ax.plot(ne, curve, color=colour, lw=1.8, label=label, zorder=3)

        sigma = (errs or {}).get(err_key)
        if not sigma or sc.log_ratios is None:
            continue
        # Re-invert the ratio at +/- 1 sigma to get the band.  The curves
        # are stored as temperatures, so the band cannot be derived from
        # them directly.
        if grid is None:
            grid = _oiii_te_ne_grid(sc.coll_file)
        log_obs = sc.log_ratios[err_key]
        lo = 10.0 ** _te_curve(grid[ratio_key], log_obs - sigma, grid["logTe"])
        hi = 10.0 ** _te_curve(grid[ratio_key], log_obs + sigma, grid["logTe"])
        band_lo, band_hi = np.minimum(lo, hi), np.maximum(lo, hi)
        ok = np.isfinite(band_lo) & np.isfinite(band_hi)
        ax.fill_between(
            ne[ok], band_lo[ok], band_hi[ok],
            color=colour, alpha=0.15, lw=0, zorder=1,
        )

    if show_posterior and sc.ne_posterior is not None and sc.ne_posterior.size > 20:
        _draw_posterior_contours(ax, sc)

    # Adopted solution.
    ax.axvline(sc.ne, color="k", ls=":", lw=1.0, zorder=4)
    ax.axhline(sc.Te, color="k", ls=":", lw=1.0, zorder=4)
    ax.plot(
        [sc.ne], [sc.Te], marker="*", ms=15, color="k", zorder=5,
        label=(
            f"adopted: $T_e$ = {sc.Te:.0f} K, "
            + (f"$n_e$ < {sc.ne_upper_limit:.3g}" if sc.ne_is_upper_limit
               else f"$n_e$ = {sc.ne:.3g}")
            + r" cm$^{-3}$"
        ),
    )
    if sc.ne_is_upper_limit and sc.ne_upper_limit:
        ax.axvspan(
            sc.ne_upper_limit, 10.0 ** OIII_GRID_LOGNE[1],
            color="0.85", alpha=0.6, lw=0, zorder=0,
        )

    ax.set_xscale("log")
    ax.set_xlim(*(ne_range or (10.0 ** OIII_GRID_LOGNE[0], 10.0 ** OIII_GRID_LOGNE[1])))
    ax.set_ylim(*(Te_range or (10.0 ** OIII_GRID_LOGTE[0], 10.0 ** OIII_GRID_LOGTE[1])))
    ax.set_xlabel(r"$n_e$ (cm$^{-3}$)")
    ax.set_ylabel(r"$T_e$ (K)")
    title = "Self-consistent $T_e$-$n_e$ (O$^{++}$ zone)"
    if not sc.converged:
        title += " — curves do not intersect"
    ax.set_title(title)
    if legend:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _draw_posterior_contours(ax: Any, sc: Any) -> None:
    """Shade the 1 sigma and 2 sigma regions of the (n_e, T_e) posterior."""
    log_ne = np.log10(sc.ne_posterior)
    Te = sc.Te_posterior
    counts, xedges, yedges = np.histogram2d(log_ne, Te, bins=24)
    if counts.max() <= 0:
        return
    try:
        from scipy.ndimage import gaussian_filter
        # Smooth by one bin: with a few hundred draws the raw histogram is
        # speckled enough that the contours read as noise rather than shape.
        counts = gaussian_filter(counts, 1.0)
    except ImportError:
        pass

    # Contour levels enclosing 68 % and 95 % of the draws.
    flat = np.sort(counts.ravel())[::-1]
    cumulative = np.cumsum(flat) / flat.sum()
    levels = []
    for frac in (0.95, 0.68):
        idx = np.searchsorted(cumulative, frac)
        levels.append(flat[min(idx, flat.size - 1)])
    levels = sorted(set(levels))
    if not levels:
        return

    xc = 10.0 ** (0.5 * (xedges[:-1] + xedges[1:]))
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    ax.contourf(
        xc, yc, counts.T, levels=levels + [counts.max() + 1],
        colors=["#f6c28b", "#e08214"][-len(levels):], alpha=0.55, zorder=2,
    )
