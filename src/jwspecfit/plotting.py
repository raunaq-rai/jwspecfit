"""Publication-quality visualisation of spectral fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .fitter import FitResult


def plot_fit(
    result: "FitResult",
    *,
    fig: "Figure | None" = None,
    wave_unit: str = "um",
    show_residuals: bool = True,
    show_components: bool = True,
    label_lines: bool = True,
) -> "Figure":
    """Plot a spectral fit with data, model, continuum, and residuals.

    Parameters
    ----------
    result : FitResult
        Output of :func:`~jwspecfit.fitter.fit_lines`.
    fig : Figure, optional
        Matplotlib figure to draw on.  If ``None``, creates a new one.
    wave_unit : str
        ``"um"`` for microns (default) or ``"A"`` for Angstroms.
    show_residuals : bool
        Show residual panel below the main plot (default True).
    show_components : bool
        Show individual line components (default True).
    label_lines : bool
        Annotate line identifications (default True).

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from .models import build_model

    spec = result.spectrum

    if wave_unit == "A":
        wave = spec.wave_A
        xlabel = r"Wavelength [$\mathrm{\AA}$]"
    else:
        wave = spec.wave_um
        xlabel = r"Wavelength [$\mu$m]"

    flux = spec.flux_ujy
    err = spec.err_ujy
    cont = result.continuum
    model_total = result.model_flux + cont
    resid = result.residuals

    # Figure setup.
    if fig is None:
        if show_residuals:
            fig, (ax_main, ax_res) = plt.subplots(
                2, 1, figsize=(10, 6), height_ratios=[3, 1],
                sharex=True, gridspec_kw={"hspace": 0.05},
            )
        else:
            fig, ax_main = plt.subplots(1, 1, figsize=(10, 4.5))
            ax_res = None
    else:
        axes = fig.get_axes()
        ax_main = axes[0]
        ax_res = axes[1] if len(axes) > 1 else None

    # Main panel: data + model + continuum.
    valid = spec.mask_valid()
    ax_main.step(wave[valid], flux[valid], where="mid", color="0.3", lw=0.8, label="Data")
    ax_main.fill_between(
        wave[valid],
        (flux - err)[valid],
        (flux + err)[valid],
        step="mid",
        alpha=0.15,
        color="0.5",
    )
    ax_main.plot(wave, cont, "--", color="C2", lw=1.0, alpha=0.7, label="Continuum")
    ax_main.plot(wave, model_total, color="C3", lw=1.5, label="Model")

    # Individual line components.
    if show_components and len(result.line_names) > 0:
        edges = spec.wave_edges_A
        nL = len(result.line_names)
        colours = plt.cm.tab10(np.linspace(0, 1, min(nL, 10)))

        for i, name in enumerate(result.line_names):
            p_single = np.zeros(3 * nL)
            p_single[i] = result.params[i]
            p_single[nL + i] = result.params[nL + i]
            p_single[2 * nL + i] = result.params[2 * nL + i]
            comp_flam = build_model(p_single, edges, nL)

            from .io import _flam_to_ujy
            comp_ujy = _flam_to_ujy(comp_flam, spec.wave_um) + cont

            colour = colours[i % len(colours)]
            ax_main.plot(wave, comp_ujy, "-", color=colour, lw=0.7, alpha=0.6)

            if label_lines and result.params[i] > 0:
                centroid_A = result.params[nL + i]
                if wave_unit == "A":
                    x_label = centroid_A
                else:
                    x_label = centroid_A * 1e-4
                y_label = comp_ujy[np.argmin(np.abs(spec.wave_A - centroid_A))]
                # Clean up name for display.
                display_name = name.replace("_", " ")
                ax_main.annotate(
                    display_name,
                    xy=(x_label, y_label),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=7,
                    ha="center",
                    color=colour,
                    rotation=45,
                )

    ax_main.set_ylabel(r"Flux density [$\mu$Jy]")
    ax_main.legend(fontsize=8, loc="upper right")

    if result.spectrum.z is not None:
        ax_main.set_title(f"z = {result.spectrum.z:.4f}   |   χ²/dof = {result.chi2:.2f}")

    # Residual panel.
    if show_residuals and ax_res is not None:
        ax_res.step(wave[valid], resid[valid], where="mid", color="0.3", lw=0.8)
        ax_res.axhline(0, color="C3", lw=0.8, ls="--")
        ax_res.fill_between(
            wave[valid], -err[valid], err[valid],
            step="mid", alpha=0.15, color="0.5",
        )
        ax_res.set_ylabel("Residual")
        ax_res.set_xlabel(xlabel)
    else:
        ax_main.set_xlabel(xlabel)

    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig
