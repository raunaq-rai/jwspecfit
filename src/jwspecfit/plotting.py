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
    wave_unit: str = "A",
    show_residuals: bool = True,
    show_components: bool = True,
    label_lines: bool = True,
    y_pad: float = 1.3,
) -> "Figure":
    """Plot a spectral fit with data, model, continuum, and residuals.

    The y-axis upper limit is set to the peak of the tallest emission
    line (above continuum) times *y_pad*, so the plot is scaled to the
    lines rather than noise spikes.

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
        Show individual Gaussian components as filled curves (default True).
        Broad components are drawn with hatching for clarity.
    label_lines : bool
        Annotate line identifications (default True).
    y_pad : float
        Multiplicative padding above the tallest line peak (default 1.3).

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from .models import build_model
    from .io import _flam_to_ujy

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

    valid = spec.mask_valid()

    # Disable scientific notation / offset on axes so full numbers are shown.
    from matplotlib.ticker import ScalarFormatter
    sfmt = ScalarFormatter(useOffset=False)
    sfmt.set_scientific(False)

    # --- Individual line components (smooth Gaussians, behind data) ---
    if show_components and len(result.line_names) > 0:
        edges = spec.wave_edges_A
        nL = len(result.line_names)

        # Colour map: narrow lines in blues/greens, broad in reds/oranges.
        narrow_names = [n for n in result.line_names if "BROAD" not in n]
        broad_names = [n for n in result.line_names if "BROAD" in n]

        narrow_colours = plt.cm.Set2(np.linspace(0, 0.8, max(len(narrow_names), 1)))
        broad_colours = plt.cm.Oranges(np.linspace(0.4, 0.8, max(len(broad_names), 1)))

        n_narrow = 0
        n_broad = 0

        for i, name in enumerate(result.line_names):
            amp = result.params[i]

            p_single = np.zeros(3 * nL)
            p_single[i] = amp
            p_single[nL + i] = result.params[nL + i]
            p_single[2 * nL + i] = result.params[2 * nL + i]
            comp_flam = build_model(p_single, edges, nL)
            comp_ujy = _flam_to_ujy(comp_flam, spec.wave_um) + cont

            is_broad = "BROAD" in name

            # Fractional uncertainty for shading.
            lr = result.lines.get(name)
            frac_err = 0.0
            if lr is not None and lr.flux > 0 and lr.flux_err > 0:
                frac_err = lr.flux_err / lr.flux

            if is_broad:
                colour = broad_colours[n_broad % len(broad_colours)]
                n_broad += 1
                ax_main.fill_between(
                    wave, cont, comp_ujy,
                    alpha=0.25, color=colour, hatch="//", linewidth=0,
                )
                ax_main.plot(wave, comp_ujy, "-", color=colour, lw=1.2, alpha=0.8)
            else:
                colour = narrow_colours[n_narrow % len(narrow_colours)]
                n_narrow += 1
                ax_main.fill_between(
                    wave, cont, comp_ujy,
                    alpha=0.20, color=colour, linewidth=0,
                )
                ax_main.plot(wave, comp_ujy, "-", color=colour, lw=0.8, alpha=0.7)

            # Uncertainty shading (±1σ on the Gaussian profile).
            if frac_err > 0:
                line_only = comp_ujy - cont
                comp_hi = cont + line_only * (1.0 + frac_err)
                comp_lo = cont + line_only * max(1.0 - frac_err, 0.0)
                ax_main.fill_between(
                    wave, comp_lo, comp_hi,
                    alpha=0.12, color=colour, linewidth=0,
                )

            if label_lines:
                centroid_A = result.params[nL + i]
                if wave_unit == "A":
                    x_label = centroid_A
                else:
                    x_label = centroid_A * 1e-4
                y_label = comp_ujy[np.argmin(np.abs(spec.wave_A - centroid_A))]
                display_name = name.replace("_", " ")
                ax_main.annotate(
                    display_name,
                    xy=(x_label, y_label),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=7,
                    ha="center",
                    color=colour,
                    fontweight="bold" if is_broad else "normal",
                    rotation=45,
                )

    # Main panel: data + model + continuum.
    ax_main.step(wave[valid], flux[valid], where="mid", color="0.3", lw=0.8,
                 label="Data", zorder=3)
    ax_main.fill_between(
        wave[valid],
        (flux - err)[valid],
        (flux + err)[valid],
        step="mid", alpha=0.12, color="0.5", zorder=2,
    )
    ax_main.step(wave, cont, where="mid", color="C2", lw=1.0, alpha=0.7,
                 label="Continuum", linestyle="--", zorder=4)
    ax_main.step(wave, model_total, where="mid", color="C3", lw=1.2,
                 alpha=0.5, label="Model", zorder=5)

    ax_main.set_ylabel(r"Flux density [$\mu$Jy]")
    ax_main.legend(fontsize=8, loc="upper right")
    ax_main.xaxis.set_major_formatter(sfmt)

    if result.spectrum.z is not None:
        ax_main.set_title(f"z = {result.spectrum.z:.4f}   |   χ²/dof = {result.chi2:.2f}")

    # --- Y-axis limits based on emission-line peaks ---
    model_peak = np.nanmax(model_total[valid]) if np.any(valid) else 1.0
    cont_median = np.nanmedian(cont[valid]) if np.any(valid) else 0.0
    # Upper limit: tallest line peak × y_pad
    y_upper = cont_median + (model_peak - cont_median) * y_pad
    # Lower limit: slightly below zero or the minimum continuum
    y_lower = min(0.0, np.nanmin(cont[valid]) * 1.1) if np.any(valid) else -0.1
    if y_upper > y_lower:
        ax_main.set_ylim(y_lower, y_upper)

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
        ax_res.xaxis.set_major_formatter(sfmt)
    else:
        ax_main.set_xlabel(xlabel)

    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def plot_fit_interactive(
    result: "FitResult",
    *,
    wave_unit: str = "A",
    show_components: bool = True,
    y_pad: float = 1.3,
) -> "go.Figure":
    """Interactive plotly plot of a spectral fit with zoom and hover.

    Parameters
    ----------
    result : FitResult
        Output of :func:`~jwspecfit.fitter.fit_lines`.
    wave_unit : str
        ``"A"`` for Angstroms (default) or ``"um"`` for microns.
    show_components : bool
        Show individual line components (default True).
    y_pad : float
        Multiplicative padding above tallest line (default 1.3).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from .models import build_model
    from .io import _flam_to_ujy

    spec = result.spectrum
    if wave_unit == "A":
        wave = spec.wave_A
        xlabel = "Wavelength [Å]"
    else:
        wave = spec.wave_um
        xlabel = "Wavelength [µm]"

    flux = spec.flux_ujy
    err = spec.err_ujy
    cont = result.continuum
    model_total = result.model_flux + cont
    valid = spec.mask_valid()

    fig = go.Figure()

    # Error band.
    fig.add_trace(go.Scatter(
        x=np.concatenate([wave[valid], wave[valid][::-1]]),
        y=np.concatenate([(flux + err)[valid], (flux - err)[valid][::-1]]),
        fill="toself", fillcolor="rgba(150,150,150,0.15)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Individual line components (smooth Gaussians).
    if show_components and len(result.line_names) > 0:
        edges = spec.wave_edges_A
        nL = len(result.line_names)

        import plotly.express as px
        palette = px.colors.qualitative.Set2

        for i, name in enumerate(result.line_names):
            amp = result.params[i]

            p_single = np.zeros(3 * nL)
            p_single[i] = amp
            p_single[nL + i] = result.params[nL + i]
            p_single[2 * nL + i] = result.params[2 * nL + i]
            comp_flam = build_model(p_single, edges, nL)
            comp_ujy = _flam_to_ujy(comp_flam, spec.wave_um) + cont

            is_broad = "BROAD" in name
            colour = "rgba(255,140,0,0.6)" if is_broad else palette[i % len(palette)]
            display_name = name.replace("_", " ")
            dash = "dot" if is_broad else "solid"

            # Uncertainty shading (±1σ).
            lr = result.lines.get(name)
            frac_err = 0.0
            if lr is not None and lr.flux > 0 and lr.flux_err > 0:
                frac_err = lr.flux_err / lr.flux

            if frac_err > 0:
                line_only = comp_ujy - cont
                comp_hi = cont + line_only * (1.0 + frac_err)
                comp_lo = cont + line_only * max(1.0 - frac_err, 0.0)
                # Shaded band (smooth).
                fig.add_trace(go.Scatter(
                    x=np.concatenate([wave, wave[::-1]]),
                    y=np.concatenate([comp_hi, comp_lo[::-1]]),
                    fill="toself", fillcolor=colour.replace("0.6", "0.12")
                    if "rgba" in colour else f"rgba(150,150,150,0.12)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))

            # Smooth Gaussian line.
            fig.add_trace(go.Scatter(
                x=wave, y=comp_ujy,
                mode="lines", name=f"{'[B] ' if is_broad else ''}{display_name}",
                line=dict(color=colour, width=1.5, dash=dash),
                hovertemplate=f"{display_name}<br>λ=%{{x:.1f}}<br>flux=%{{y:.4f}} µJy<extra></extra>",
            ))

    # Data (steps).
    fig.add_trace(go.Scatter(
        x=wave[valid], y=flux[valid],
        mode="lines", name="Data",
        line=dict(color="grey", width=0.8, shape="hvh"),
        hovertemplate="Data<br>λ=%{x:.1f}<br>flux=%{y:.4f} µJy<extra></extra>",
    ))

    # Continuum (steps).
    fig.add_trace(go.Scatter(
        x=wave, y=cont,
        mode="lines", name="Continuum",
        line=dict(color="green", width=1, dash="dash", shape="hvh"),
    ))

    # Model (steps, semi-transparent).
    fig.add_trace(go.Scatter(
        x=wave, y=model_total,
        mode="lines", name="Model",
        line=dict(color="rgba(255,0,0,0.4)", width=1.5, shape="hvh"),
        hovertemplate="Model<br>λ=%{x:.1f}<br>flux=%{y:.4f} µJy<extra></extra>",
    ))

    # Y limits.
    model_peak = np.nanmax(model_total[valid]) if np.any(valid) else 1.0
    cont_median = np.nanmedian(cont[valid]) if np.any(valid) else 0.0
    y_upper = cont_median + (model_peak - cont_median) * y_pad
    y_lower = min(0.0, np.nanmin(cont[valid]) * 1.1) if np.any(valid) else -0.1

    title = ""
    if spec.z is not None:
        title = f"z = {spec.z:.4f}  |  χ²/dof = {result.chi2:.2f}"

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Flux density [µJy]",
        yaxis_range=[y_lower, y_upper],
        xaxis=dict(exponentformat="none"),
        template="plotly_white",
        hovermode=False,
        dragmode="zoom",
        legend=dict(x=1.0, y=1.0, xanchor="right"),
        width=1000,
        height=500,
    )

    return fig
