"""Publication-quality visualisation of spectral fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .fitter import FitResult


def _build_exclude_mask(
    wave_A: np.ndarray,
    exclude_wave_A: list[tuple[float, float]] | None,
) -> np.ndarray:
    """Return a boolean mask that is True for pixels to KEEP.

    Parameters
    ----------
    wave_A : np.ndarray
        Wavelength array in Angstroms.
    exclude_wave_A : list of (lo, hi) tuples, optional
        Wavelength ranges to exclude.

    Returns
    -------
    np.ndarray
        Boolean mask (True = keep).
    """
    keep = np.ones(len(wave_A), dtype=bool)
    if exclude_wave_A is not None:
        for lo, hi in exclude_wave_A:
            keep &= ~((wave_A >= lo) & (wave_A <= hi))
    return keep


def plot_fit(
    result: "FitResult",
    *,
    fig: "Figure | None" = None,
    wave_unit: str = "A",
    flux_unit: str = "fnu",
    show_residuals: bool = True,
    show_components: bool = True,
    label_lines: bool = True,
    y_pad: float = 1.3,
    exclude_wave_A: list[tuple[float, float]] | None = None,
    save_path: str | None = None,
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
        ``"A"`` for Angstroms (default) or ``"um"`` for microns.
    flux_unit : str
        ``"fnu"`` for µJy (default) or ``"flam"`` for erg/s/cm²/Å.
    show_residuals : bool
        Show residual panel below the main plot (default True).
    show_components : bool
        Show individual Gaussian components as filled curves (default True).
        Broad components are drawn with hatching for clarity.
    label_lines : bool
        Annotate line identifications (default True).
    y_pad : float
        Multiplicative padding above the tallest line peak (default 1.3).
    exclude_wave_A : list of (float, float), optional
        Wavelength ranges in Angstroms to hide from the plot.  Each tuple
        is ``(lo, hi)``.  Useful for masking noisy detector regions.
    save_path : str, optional
        If given, save the figure to this file path (e.g. ``"fit.pdf"``).

    Returns
    -------
    Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from .models import build_model
    from .io import _flam_to_ujy, _ujy_to_flam

    spec = result.spectrum

    if wave_unit == "A":
        wave = spec.wave_A
        xlabel = r"Wavelength [$\mathrm{\AA}$]"
    else:
        wave = spec.wave_um
        xlabel = r"Wavelength [$\mu$m]"

    use_flam = flux_unit.lower() == "flam"

    if use_flam:
        flux = _ujy_to_flam(spec.flux_ujy, spec.wave_um)
        err = _ujy_to_flam(spec.err_ujy, spec.wave_um)
        cont = _ujy_to_flam(result.continuum, spec.wave_um)
        model_total = _ujy_to_flam(result.model_flux + result.continuum, spec.wave_um)
        resid = _ujy_to_flam(result.residuals, spec.wave_um)
    else:
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

    # Apply wavelength exclusion mask.
    keep = _build_exclude_mask(spec.wave_A, exclude_wave_A)
    show = valid & keep

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

        # NaN-out excluded regions so components don't plot through them.
        wave_plot = wave.copy().astype(float)
        wave_plot[~keep] = np.nan

        for i, name in enumerate(result.line_names):
            amp = result.params[i]

            p_single = np.zeros(3 * nL)
            p_single[i] = amp
            p_single[nL + i] = result.params[nL + i]
            p_single[2 * nL + i] = result.params[2 * nL + i]
            comp_flam = build_model(p_single, edges, nL)
            if use_flam:
                comp_plot = comp_flam + _ujy_to_flam(result.continuum, spec.wave_um)
            else:
                comp_plot = _flam_to_ujy(comp_flam, spec.wave_um) + cont

            # NaN-out excluded regions.
            comp_plot_masked = comp_plot.copy()
            comp_plot_masked[~keep] = np.nan
            cont_masked = cont.copy()
            cont_masked[~keep] = np.nan

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
                    wave_plot, cont_masked, comp_plot_masked,
                    alpha=0.25, color=colour, hatch="//", linewidth=0,
                )
                ax_main.plot(wave_plot, comp_plot_masked, "-", color=colour, lw=1.2, alpha=0.8)
            else:
                colour = narrow_colours[n_narrow % len(narrow_colours)]
                n_narrow += 1
                ax_main.fill_between(
                    wave_plot, cont_masked, comp_plot_masked,
                    alpha=0.20, color=colour, linewidth=0,
                )
                ax_main.plot(wave_plot, comp_plot_masked, "-", color=colour, lw=0.8, alpha=0.7)

            # Uncertainty shading (±1σ on the Gaussian profile).
            if frac_err > 0:
                line_only = comp_plot_masked - cont_masked
                comp_hi = cont_masked + line_only * (1.0 + frac_err)
                comp_lo = cont_masked + line_only * max(1.0 - frac_err, 0.0)
                ax_main.fill_between(
                    wave_plot, comp_lo, comp_hi,
                    alpha=0.12, color=colour, linewidth=0,
                )

            if label_lines:
                centroid_A = result.params[nL + i]
                if wave_unit == "A":
                    x_label = centroid_A
                else:
                    x_label = centroid_A * 1e-4
                y_label = comp_plot[np.argmin(np.abs(spec.wave_A - centroid_A))]
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
    ax_main.step(wave[show], flux[show], where="mid", color="0.3", lw=0.8,
                 label="Data", zorder=3)
    ax_main.fill_between(
        wave[show],
        (flux - err)[show],
        (flux + err)[show],
        step="mid", alpha=0.12, color="0.5", zorder=2,
    )
    ax_main.step(wave[keep], cont[keep], where="mid", color="C2", lw=1.0, alpha=0.7,
                 label="Continuum", linestyle="--", zorder=4)
    ax_main.step(wave[keep], model_total[keep], where="mid", color="C3", lw=1.2,
                 alpha=0.5, label="Model", zorder=5)

    ylabel = r"$f_\lambda$ [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]" if use_flam else r"Flux density [$\mu$Jy]"
    ax_main.set_ylabel(ylabel)
    ax_main.legend(fontsize=8, loc="upper right")
    ax_main.xaxis.set_major_formatter(sfmt)

    if result.spectrum.z is not None:
        ax_main.set_title(f"z = {result.spectrum.z:.4f}   |   χ²/dof = {result.chi2:.2f}")

    # --- Y-axis limits based on emission-line peaks ---
    model_peak = np.nanmax(model_total[show]) if np.any(show) else 1.0
    cont_median = np.nanmedian(cont[show]) if np.any(show) else 0.0
    # Upper limit: tallest line peak × y_pad
    y_upper = cont_median + (model_peak - cont_median) * y_pad
    # Lower limit: slightly below zero or the minimum continuum
    y_lower = min(0.0, np.nanmin(cont[show]) * 1.1) if np.any(show) else -0.1
    if y_upper > y_lower:
        ax_main.set_ylim(y_lower, y_upper)

    # Residual panel.
    if show_residuals and ax_res is not None:
        ax_res.step(wave[show], resid[show], where="mid", color="0.3", lw=0.8)
        ax_res.axhline(0, color="C3", lw=0.8, ls="--")
        ax_res.fill_between(
            wave[show], -err[show], err[show],
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

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_fit_interactive(
    result: "FitResult",
    *,
    wave_unit: str = "A",
    flux_unit: str = "fnu",
    show_components: bool = True,
    show_residuals: bool = True,
    y_pad: float = 1.3,
    exclude_wave_A: list[tuple[float, float]] | None = None,
) -> "go.Figure":
    """Interactive plotly plot of a spectral fit with zoom and hover.

    Parameters
    ----------
    result : FitResult
        Output of :func:`~jwspecfit.fitter.fit_lines`.
    wave_unit : str
        ``"A"`` for Angstroms (default) or ``"um"`` for microns.
    flux_unit : str
        ``"fnu"`` for µJy (default) or ``"flam"`` for erg/s/cm²/Å.
    show_components : bool
        Show individual line components (default True).
    show_residuals : bool
        Show residual panel below the main plot (default True).
    y_pad : float
        Multiplicative padding above tallest line (default 1.3).
    exclude_wave_A : list of (float, float), optional
        Wavelength ranges in Angstroms to hide from the plot.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from .models import build_model
    from .io import _flam_to_ujy, _ujy_to_flam

    spec = result.spectrum
    if wave_unit == "A":
        wave = spec.wave_A
        xlabel = "Wavelength [Å]"
    else:
        wave = spec.wave_um
        xlabel = "Wavelength [µm]"

    use_flam = flux_unit.lower() == "flam"

    if use_flam:
        flux = _ujy_to_flam(spec.flux_ujy, spec.wave_um)
        err = _ujy_to_flam(spec.err_ujy, spec.wave_um)
        cont = _ujy_to_flam(result.continuum, spec.wave_um)
        model_total = _ujy_to_flam(result.model_flux + result.continuum, spec.wave_um)
        resid = _ujy_to_flam(result.residuals, spec.wave_um)
        flux_label = "erg/s/cm²/Å"
    else:
        flux = spec.flux_ujy
        err = spec.err_ujy
        cont = result.continuum
        model_total = result.model_flux + cont
        resid = result.residuals
        flux_label = "µJy"

    valid = spec.mask_valid()
    keep = _build_exclude_mask(spec.wave_A, exclude_wave_A)
    show = valid & keep

    # Insert NaN breaks at excluded-region boundaries so traces don't
    # draw lines through masked regions.
    def _nan_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = arr.copy().astype(float)
        out[~mask] = np.nan
        return out

    wave_k = _nan_mask(wave, keep)
    flux_s = _nan_mask(flux, show)
    err_s = _nan_mask(err, show)
    cont_k = _nan_mask(cont, keep)
    model_k = _nan_mask(model_total, keep)
    resid_s = _nan_mask(resid, show)

    # Build figure — with or without residuals subplot.
    if show_residuals:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.04,
        )
    else:
        fig = go.Figure()

    main_row = 1 if show_residuals else None

    def _add(trace, row=None):
        if show_residuals and row is not None:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

    # Error band.
    _add(go.Scatter(
        x=np.concatenate([wave[show], wave[show][::-1]]),
        y=np.concatenate([(flux + err)[show], (flux - err)[show][::-1]]),
        fill="toself", fillcolor="rgba(150,150,150,0.15)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1)

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
            if use_flam:
                comp_plot = comp_flam + _ujy_to_flam(result.continuum, spec.wave_um)
            else:
                comp_plot = _flam_to_ujy(comp_flam, spec.wave_um) + cont

            comp_k = _nan_mask(comp_plot, keep)

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
                line_only = comp_plot - cont
                comp_hi = _nan_mask(cont + line_only * (1.0 + frac_err), keep)
                comp_lo = _nan_mask(cont + line_only * max(1.0 - frac_err, 0.0), keep)
                _add(go.Scatter(
                    x=np.concatenate([wave, wave[::-1]]),
                    y=np.concatenate([comp_hi, comp_lo[::-1]]),
                    fill="toself", fillcolor=colour.replace("0.6", "0.12")
                    if "rgba" in colour else f"rgba(150,150,150,0.12)",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ), row=1)

            # Smooth Gaussian line.
            _add(go.Scatter(
                x=wave_k, y=comp_k,
                mode="lines", name=f"{'[B] ' if is_broad else ''}{display_name}",
                line=dict(color=colour, width=1.5, dash=dash),
                hovertemplate=f"{display_name}<br>λ=%{{x:.1f}}<br>flux=%{{y:.4e}} {flux_label}<extra></extra>",
            ), row=1)

    # Data (steps).
    _add(go.Scatter(
        x=wave[show], y=flux[show],
        mode="lines", name="Data",
        line=dict(color="grey", width=0.8, shape="hvh"),
        hovertemplate=f"Data<br>λ=%{{x:.1f}}<br>flux=%{{y:.4e}} {flux_label}<extra></extra>",
    ), row=1)

    # Continuum (steps).
    _add(go.Scatter(
        x=wave_k, y=cont_k,
        mode="lines", name="Continuum",
        line=dict(color="green", width=1, dash="dash", shape="hvh"),
    ), row=1)

    # Model (steps, semi-transparent).
    _add(go.Scatter(
        x=wave_k, y=model_k,
        mode="lines", name="Model",
        line=dict(color="rgba(255,0,0,0.4)", width=1.5, shape="hvh"),
        hovertemplate=f"Model<br>λ=%{{x:.1f}}<br>flux=%{{y:.4e}} {flux_label}<extra></extra>",
    ), row=1)

    # --- Residual panel ---
    if show_residuals:
        # Error band on residuals.
        _add(go.Scatter(
            x=np.concatenate([wave[show], wave[show][::-1]]),
            y=np.concatenate([err[show], (-err)[show][::-1]]),
            fill="toself", fillcolor="rgba(150,150,150,0.15)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=2)

        # Zero line.
        _add(go.Scatter(
            x=[wave[show].min(), wave[show].max()],
            y=[0, 0],
            mode="lines", showlegend=False,
            line=dict(color="rgba(255,0,0,0.4)", width=1, dash="dash"),
        ), row=2)

        # Residual data.
        _add(go.Scatter(
            x=wave[show], y=resid[show],
            mode="lines", name="Residual", showlegend=False,
            line=dict(color="grey", width=0.8, shape="hvh"),
            hovertemplate=f"Residual<br>λ=%{{x:.1f}}<br>resid=%{{y:.4e}} {flux_label}<extra></extra>",
        ), row=2)

    # Y limits.
    model_peak = np.nanmax(model_total[show]) if np.any(show) else 1.0
    cont_median = np.nanmedian(cont[show]) if np.any(show) else 0.0
    y_upper = cont_median + (model_peak - cont_median) * y_pad
    y_lower = min(0.0, np.nanmin(cont[show]) * 1.1) if np.any(show) else -0.1

    title = ""
    if spec.z is not None:
        title = f"z = {spec.z:.4f}  |  χ²/dof = {result.chi2:.2f}"

    ylabel = f"fλ [{flux_label}]" if use_flam else f"Flux density [{flux_label}]"

    if show_residuals:
        fig.update_layout(
            title=title,
            template="plotly_white",
            hovermode=False,
            dragmode="zoom",
            legend=dict(x=1.0, y=1.0, xanchor="right"),
            width=1000,
            height=650,
        )
        fig.update_xaxes(title_text=xlabel, exponentformat="none", row=2, col=1)
        fig.update_xaxes(exponentformat="none", row=1, col=1)
        fig.update_yaxes(title_text=ylabel, range=[y_lower, y_upper], row=1, col=1)
        fig.update_yaxes(title_text="Residual", row=2, col=1)
    else:
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
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
