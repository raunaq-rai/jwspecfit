"""Publication-quality visualisation of spectral fits."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .fitter import FitResult
    from .io import Spectrum


# Default emission-line markers (keys into REST_LINES_A) and their
# preferred display labels.  Shared between plot_spectrum_interactive
# and plot_2d_1d so labels stay consistent.
_DEFAULT_MARKER_NAMES: list[str] = [
    "Lya", "NIV_doublet", "CIV_doublet", "HEII_1640",
    "NIII_doublet", "CIII]",
    "OII_doublet", "NeIII_3869",
    "HEI_4027", "HDELTA", "HEI_4145", "HEII_4200", "HGAMMA", "OIII_4363",
    "FeII_4584", "NIII_4642", "FeIII_4660", "HeII_4687",
    "ArIV_4713", "FeII_4732", "ArIV_4741",
    "HBETA", "OIII_4959", "OIII_5007",
    "CIV_5803", "CIV_5814",
    "HEI_5877", "OI_6302",
    "Ha", "NII_6585", "HEI_6680",
    "SII_6718", "SII_6732",
    "HEI_7067", "ArIII_7138",
]

_DEFAULT_MARKER_LABELS: dict[str, str] = {
    "Lya": "Lyα", "NIV_doublet": "NIV", "CIV_doublet": "CIV",
    "HEII_1640": "HeII 1640",
    "NIII_doublet": "NIII 1750", "CIII]": "CIII]",
    "OII_doublet": "[OII]", "NeIII_3869": "[NeIII]",
    "HEI_4027": "HeI 4027",
    "HDELTA": "Hδ",
    "HEI_4145": "HeI 4145",
    "HEII_4200": "HeII 4200",
    "HGAMMA": "Hγ",
    "OIII_4363": "[OIII]4363",
    "FeII_4584": "FeII 4584", "NIII_4642": "NIII 4642",
    "FeIII_4660": "FeIII 4660", "HeII_4687": "HeII 4687",
    "ArIV_4713": "ArIV+HeI 4705",
    "FeII_4732": "FeII 4732", "ArIV_4741": "ArIV 4741",
    "HBETA": "Hβ",
    "OIII_4959": "[OIII]4959", "OIII_5007": "[OIII]5007",
    "CIV_5803": "CIV 5803", "CIV_5814": "CIV 5814",
    "HEI_5877": "HeI 5877",
    "OI_6302": "[OI] 6302",
    "Ha": "Hα", "NII_6585": "[NII] 6585",
    "HEI_6680": "HeI 6680",
    "SII_6718": "[SII]6716", "SII_6732": "[SII]6731",
    "HEI_7067": "HeI 7065",
    "ArIII_7138": "[ArIII] 7138",
}


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
    rest_frame: bool = False,
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
    rest_frame : bool
        If ``True``, plot wavelengths in the rest frame by dividing by
        ``(1 + z)`` using the redshift stored in the spectrum.  Default
        ``False`` (observed frame).
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

    # Auto-convert MCMCResult / MCMCBroadFitResult to FitResult.
    if hasattr(result, "to_fit_result") and not hasattr(result, "residuals"):
        result = result.to_fit_result()

    spec = result.spectrum

    # Rest-frame scaling factor.
    zp1 = 1.0
    if rest_frame and spec.z is not None:
        zp1 = 1.0 + spec.z

    if wave_unit == "A":
        wave = spec.wave_A / zp1
        xlabel = r"Rest Wavelength [$\mathrm{\AA}$]" if rest_frame else r"Wavelength [$\mathrm{\AA}$]"
    else:
        wave = spec.wave_um / zp1
        xlabel = r"Rest Wavelength [$\mu$m]" if rest_frame else r"Wavelength [$\mu$m]"

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

        # Build index mapping: line_names may include "Lya" which is not
        # in the Gaussian params vector.  Skip it for component plotting.
        _gauss_names = [n for n in result.line_names if n != "Lya"]
        _gauss_idx = {n: j for j, n in enumerate(_gauss_names)}
        nL_gauss = len(_gauss_names)

        for i, name in enumerate(result.line_names):
            if name == "Lya":
                continue  # Lyα uses skewed Gaussian, not in params vector
            gi = _gauss_idx[name]
            amp = result.params[gi]

            p_single = np.zeros(3 * nL_gauss)
            p_single[gi] = amp
            p_single[nL_gauss + gi] = result.params[nL_gauss + gi]
            p_single[2 * nL_gauss + gi] = result.params[2 * nL_gauss + gi]
            comp_flam = build_model(p_single, edges, nL_gauss)
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

            is_abs = name.startswith("abs_")

            # Fractional uncertainty for shading.
            lr = result.lines.get(name)
            frac_err = 0.0
            if lr is not None and abs(lr.flux) > 0 and lr.flux_err > 0:
                frac_err = lr.flux_err / abs(lr.flux)

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
                centroid_obs_A = result.params[nL_gauss + gi]
                centroid_A = centroid_obs_A / zp1
                if wave_unit == "A":
                    x_label = centroid_A
                else:
                    x_label = centroid_A * 1e-4
                y_label = comp_plot[np.argmin(np.abs(spec.wave_A - centroid_obs_A))]
                display_name = name.replace("abs_", "").replace("_", " ")
                # Absorption labels below the trough; emission labels above.
                y_offset = -10 if is_abs else 8
                va = "top" if is_abs else "baseline"
                ax_main.annotate(
                    display_name,
                    xy=(x_label, y_label),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    fontsize=7,
                    ha="center",
                    va=va,
                    color=colour,
                    fontweight="bold" if is_broad else "normal",
                    rotation=45,
                )

    # Lyα asymmetric Gaussian overlay for static plot.
    if show_components:
        _lya_p_s = getattr(result, "lya_params", None)
        if _lya_p_s is not None and len(_lya_p_s) == 4:
            from .models import asymmetric_gaussian as _ag

            centres_s = 0.5 * (edges[:-1] + edges[1:])
            lya_flam_s = _ag(centres_s, _lya_p_s[0], _lya_p_s[1], _lya_p_s[2], _lya_p_s[3])

            if use_flam:
                comp_s = lya_flam_s + _ujy_to_flam(result.continuum, spec.wave_um)
            else:
                comp_s = _flam_to_ujy(lya_flam_s, spec.wave_um) + cont
            comp_s_m = comp_s.copy()
            comp_s_m[~keep] = np.nan
            cont_m = cont.copy()
            cont_m[~keep] = np.nan
            ax_main.fill_between(
                wave_plot, cont_m, comp_s_m,
                alpha=0.25, color="C0", linewidth=0,
            )
            ax_main.plot(
                wave_plot, comp_s_m, "-",
                color="C0", lw=0.8, alpha=0.7,
            )

            if label_lines:
                # Peak of the asymmetric Gaussian (find numerically).
                _peak_idx_s = np.argmax(lya_flam_s)
                mu_s = centres_s[_peak_idx_s]
                x_lbl = mu_s / zp1 if wave_unit == "A" else mu_s * 1e-4 / zp1
                peak_flam = lya_flam_s[_peak_idx_s]
                cont_at = np.interp(mu_s * 1e-4, spec.wave_um, result.continuum)
                if use_flam:
                    y_lbl = peak_flam + _ujy_to_flam(np.array([cont_at]), np.array([mu_s * 1e-4]))[0]
                else:
                    y_lbl = _flam_to_ujy(np.array([peak_flam]), np.array([mu_s * 1e-4]))[0] + cont_at
                ax_main.annotate(
                    "Lyα", xy=(x_lbl, y_lbl), xytext=(0, 8),
                    textcoords="offset points", fontsize=7, ha="center",
                    va="baseline", color="C0", rotation=45,
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

    # --- Y-axis limits based on emission-line peaks / absorption troughs ---
    model_peak = np.nanmax(model_total[show]) if np.any(show) else 1.0
    model_trough = np.nanmin(model_total[show]) if np.any(show) else 0.0
    cont_median = np.nanmedian(cont[show]) if np.any(show) else 0.0
    # Upper limit: tallest line peak × y_pad
    y_upper = cont_median + (model_peak - cont_median) * y_pad
    # Lower limit: accommodate absorption troughs or minimum continuum
    y_lower_cont = np.nanmin(cont[show]) * 1.1 if np.any(show) else -0.1
    y_lower = min(0.0, y_lower_cont, model_trough - abs(model_trough) * 0.15)
    if y_upper > y_lower:
        ax_main.set_ylim(y_lower, y_upper)

    # Residual panel — x-range limited to the extent of the fitted lines.
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

        # Clip x-range to the outermost fitted lines ± 5σ margin.
        if len(result.line_names) > 0:
            nL = len(result.line_names)
            centroids_A = result.params[nL: 2 * nL] / zp1
            sigmas_A = result.params[2 * nL: 3 * nL] / zp1
            xlim_lo_A = np.min(centroids_A - 5 * sigmas_A)
            xlim_hi_A = np.max(centroids_A + 5 * sigmas_A)
            if wave_unit == "A":
                ax_res.set_xlim(xlim_lo_A, xlim_hi_A)
                ax_main.set_xlim(xlim_lo_A, xlim_hi_A)
            else:
                ax_res.set_xlim(xlim_lo_A * 1e-4, xlim_hi_A * 1e-4)
                ax_main.set_xlim(xlim_lo_A * 1e-4, xlim_hi_A * 1e-4)
    else:
        ax_main.set_xlabel(xlabel)

    try:
        fig.tight_layout()
    except Exception:
        pass

    if save_path is not None:
        save_path = Path(save_path).with_suffix(".png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _to_rgba(colour: str, alpha: float) -> str:
    """Convert any CSS colour string to ``rgba(r,g,b,alpha)``."""
    if colour.startswith("rgba("):
        # Replace existing alpha.
        return colour.rsplit(",", 1)[0] + f",{alpha})"
    if colour.startswith("rgb("):
        return colour.replace("rgb(", "rgba(").replace(")", f",{alpha})")
    if colour.startswith("#"):
        h = colour.lstrip("#")
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"
    return f"rgba(150,150,150,{alpha})"


_MULTI_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

# Component colours for plot_fit_interactive: every narrow line shares
# one colour, every _BROAD shares another, every _BROAD2 a third, so
# the eye groups them by kinematic class rather than by line identity.
_COMP_COLOUR_NARROW = "rgba(31,119,180,0.85)"   # plotly C0 blue
_COMP_COLOUR_BROAD = "rgba(255,127,14,0.75)"    # plotly C1 orange
_COMP_COLOUR_BROAD2 = "rgba(214,39,40,0.75)"    # plotly C3 red
_COMP_COLOUR_ABS = "rgba(70,130,180,0.8)"       # steel blue


def _component_colour(name: str, is_abs: bool) -> str:
    """Pick the fixed colour for a fitted component by class."""
    if is_abs:
        return _COMP_COLOUR_ABS
    if "_BROAD2" in name:
        return _COMP_COLOUR_BROAD2
    if "_BROAD" in name:
        return _COMP_COLOUR_BROAD
    return _COMP_COLOUR_NARROW


def _draw_emission_line_markers(
    fig,
    *,
    z_for_lines: "float | None",
    wave_unit: str,
    x_lo: "float | None",
    x_hi: "float | None",
    lines: "Sequence[str] | bool | None" = None,
    add_lines: "dict[str, float] | Sequence[str] | None" = None,
    line_color: str = "darkred",
    use_paper_shapes: bool = False,
) -> None:
    """Draw curated emission-line markers + labels onto a plotly figure.

    Shared by :func:`plot_spectrum_interactive` and
    :func:`plot_fit_interactive` so both honour the same ``lines`` /
    ``add_lines`` / ``line_color`` semantics and produce identical
    marker styling.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to draw markers on, in place.
    z_for_lines : float or None
        Redshift used to place markers.  ``0.0`` puts them at rest
        wavelengths; ``None`` skips drawing entirely.
    wave_unit : str
        ``"A"`` (Å) or ``"um"`` (µm) — must match the x-axis.
    x_lo, x_hi : float or None
        Plotted x-range.  Markers outside the range are dropped.  When
        either is ``None`` no markers are drawn.
    lines, add_lines, line_color
        Same semantics as the public plot functions.  See
        :func:`plot_spectrum_interactive` for details.
    use_paper_shapes : bool
        ``True`` when *fig* is a multi-row subplot — the marker is a
        paper-coord shape (spans every row) plus a single annotation
        pinned to the figure top.  ``False`` uses ``add_vline`` and
        works for a single-panel figure.
    """
    if z_for_lines is None or x_lo is None or x_hi is None:
        return

    from .lines import REST_LINES_A

    default_names = _DEFAULT_MARKER_NAMES
    display = _DEFAULT_MARKER_LABELS

    markers: list[tuple[float, str]] = []

    if lines is not False:
        names = default_names if lines is None else list(lines)
        for nm in names:
            rest_A = REST_LINES_A.get(nm)
            if rest_A is None:
                continue
            obs_A = rest_A * (1.0 + z_for_lines)
            x = obs_A if wave_unit == "A" else obs_A * 1e-4
            markers.append((x, display.get(nm, nm)))

    if add_lines:
        if isinstance(add_lines, dict):
            add_items = list(add_lines.items())
        else:
            add_items = []
            for nm in add_lines:
                rest_A = REST_LINES_A.get(nm)
                if rest_A is None:
                    continue
                label = display.get(nm, nm.replace("_", " "))
                add_items.append((label, rest_A))
        for label, rest_A in add_items:
            obs_A = float(rest_A) * (1.0 + z_for_lines)
            x = obs_A if wave_unit == "A" else obs_A * 1e-4
            markers.append((x, str(label)))

    # Clip to plotted range, then stagger so close labels don't overlap.
    markers = [(x, lab) for x, lab in markers if x_lo <= x <= x_hi]
    markers.sort(key=lambda m: m[0])
    if not markers:
        return
    threshold = 0.03 * (x_hi - x_lo)
    row_last_x: list[float] = []
    rows: list[int] = []
    for x, _ in markers:
        placed = False
        for r, last_x in enumerate(row_last_x):
            if x - last_x >= threshold:
                row_last_x[r] = x
                rows.append(r)
                placed = True
                break
        if not placed:
            row_last_x.append(x)
            rows.append(len(row_last_x) - 1)

    row_spacing_px = 14
    for (x, label), r in zip(markers, rows):
        if use_paper_shapes:
            # Paper-coord shape spans every subplot row; a single
            # annotation at the top avoids the per-row duplication
            # that add_vline(row="all", ...) produces.
            fig.add_shape(
                type="line",
                xref="x", yref="paper",
                x0=x, x1=x, y0=0, y1=1,
                line=dict(color=line_color, width=0.8, dash="dash"),
                opacity=0.6,
                layer="below",
            )
            fig.add_annotation(
                x=x, xref="x",
                y=1.0, yref="paper",
                yshift=r * row_spacing_px,
                text=label,
                showarrow=False,
                font=dict(size=9, color=line_color),
                xanchor="center", yanchor="bottom",
            )
        else:
            fig.add_vline(
                x=x,
                line_width=0.8,
                line_dash="dash",
                line_color=line_color,
                opacity=0.6,
                annotation_text=label,
                annotation_position="top",
                annotation_font_size=9,
                annotation_font_color=line_color,
                annotation_yshift=r * row_spacing_px,
                layer="below",
            )

    # Grow top margin so stacked rows fit above the plot.
    if row_last_x:
        n_rows = len(row_last_x)
        desired_t = 60 + (n_rows - 1) * row_spacing_px + 14
        fig.update_layout(margin=dict(t=desired_t))


def plot_spectrum_interactive(
    source: "Spectrum | str | Path | Sequence[Spectrum | str | Path]",
    *,
    z: float | None = None,
    wave_unit: str = "A",
    flux_unit: str = "fnu",
    rest_frame: bool = False,
    exclude_wave_A: list[tuple[float, float]] | None = None,
    title: str | None = None,
    labels: "str | Sequence[str] | None" = None,
    lines: "Sequence[str] | bool | None" = None,
    add_lines: "dict[str, float] | Sequence[str] | None" = None,
    line_color: str = "darkred",
    show_zero: bool = True,
    show_2d: "bool | str" = "auto",
    cmap_2d: str = "Blues",
    vmin_pct: float = 5.0,
    vmax_pct: float = 99.5,
    y_crop: tuple[float, float] = (0.25, 0.75),
    **read_kwargs,
) -> "go.Figure":
    """Open and interactively plot one or more 1-D spectra.

    Accepts a :class:`~jwspecfit.io.Spectrum` object, a path to a
    ``.fits`` / ``.npz`` file, or a list / tuple of such items.  When
    given a path, the file is read via :func:`~jwspecfit.io.read_fits`
    (or :func:`read_npz` for ``.npz``) and any extra ``read_kwargs`` are
    forwarded to the reader (e.g. ``hdu=``, ``wave_col=`` for FITS
    overrides).

    Parameters
    ----------
    source : Spectrum, str, Path, or sequence of these
        A single spectrum / file path, or a list / tuple of them to
        overplot.
    z : float, optional
        Source redshift.  Used only when a *source* item is a path; for
        an existing :class:`Spectrum`, its own ``z`` is preserved.
        Forwarded to every reader call.
    wave_unit : str
        ``"A"`` for Angstroms (default) or ``"um"`` for microns.
    flux_unit : str
        ``"fnu"`` for µJy (default) or ``"flam"`` for erg/s/cm²/Å.
    rest_frame : bool
        If ``True`` and a spectrum has a redshift, divide wavelengths
        by ``(1 + z)``.  Default ``False`` (observed frame).  Applied
        per spectrum.
    exclude_wave_A : list of (float, float), optional
        Wavelength ranges in Angstroms to hide from the plot.
    title : str, optional
        Figure title.  When a single spectrum is supplied, defaults to
        filename + redshift + grating if available; when multiple
        spectra are supplied, defaults to no title.
    labels : str or sequence of str, optional
        Legend label(s).  When ``None`` (default), a single spectrum
        uses ``"Data"`` and multiple spectra use each spectrum's
        filename (or ``"Spectrum {i}"`` as a fallback).
    lines : sequence of str, bool, or None
        Emission lines to mark as vertical dashed lines at the supplied
        redshift.  ``None`` (default) draws a curated list of common
        UV/optical lines.  Pass an explicit list of keys from
        :data:`jwspecfit.lines.REST_LINES_A` to override, or ``False``
        to disable.  The effective redshift is taken from ``z`` if
        given, else from a single spectrum's own ``spec.z``.  In
        rest-frame mode the markers sit at the rest wavelengths.
    add_lines : dict[str, float] or sequence of str, optional
        Extra lines to overlay on top of *lines*.  Two accepted forms:

        - **dict** — ``{label: rest_wavelength_A}``.  Free-form labels
          with explicit rest-frame wavelengths in **Angstroms**.  Use
          this for lines not in :data:`jwspecfit.lines.REST_LINES_A`,
          e.g. ``add_lines={"Mg II 2796": 2796.352}``.
        - **list of str** — names from :data:`REST_LINES_A` (e.g.
          ``add_lines=["H8", "HEPSILON", "FeII_2382"]``).  The rest
          wavelength is looked up automatically.  Call
          :func:`jwspecfit.show_lines` to see what's available.

        Each entry is redshifted by ``(1 + z)`` and staggered alongside
        the default markers.
    line_color : str
        Colour for the emission-line markers and their labels
        (default ``"darkred"``).
    show_zero : bool
        Draw a light-grey dashed horizontal line at ``y = 0`` to make
        continuum detection easier to gauge by eye (default ``True``).
    show_2d : bool or {"auto"}
        Whether to render the 2-D rectified spectrum image above the 1-D
        panel.  ``"auto"`` (default) shows it iff a *single* spectrum is
        supplied **and** its :attr:`~jwspecfit.io.Spectrum.sci_2d` is
        populated (e.g. when read from an msaexp ``.spec.fits`` file).
        ``True`` forces it (silently no-ops when multiple spectra are
        supplied or no 2-D is available); ``False`` always shows the 1-D
        only.  The two panels share the wavelength axis; emission-line
        markers span both.
    cmap_2d : str
        Plotly colorscale for the 2-D panel (default ``"Blues"`` to
        mirror :func:`plot_2d_1d`).
    vmin_pct, vmax_pct : float
        Percentile clip for the 2-D colour scale (default ``5`` /
        ``99.5``).
    y_crop : (float, float)
        Fractional crop ``(lo, hi)`` of the spatial axis on the 2-D
        panel (default ``(0.25, 0.75)`` keeps the middle 50 % of rows).
    **read_kwargs
        Forwarded to the file reader when a *source* item is a path.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from .io import Spectrum, read_fits, read_npz, _ujy_to_flam

    # Normalise sources / labels to parallel lists.
    if isinstance(source, (list, tuple)):
        sources_list = list(source)
    else:
        sources_list = [source]

    if labels is None:
        labels_in = [None] * len(sources_list)
    elif isinstance(labels, str):
        labels_in = [labels]
    else:
        labels_in = list(labels)
    if len(labels_in) != len(sources_list):
        raise ValueError(
            f"labels has length {len(labels_in)} but {len(sources_list)} "
            f"sources were given."
        )

    multi = len(sources_list) > 1

    # Resolve every source to a Spectrum.
    specs: list[Spectrum] = []
    for s in sources_list:
        if isinstance(s, Spectrum):
            specs.append(s)
        else:
            path = Path(s)
            suffix = path.suffix.lower()
            if suffix in (".fits", ".fit", ".fz"):
                specs.append(read_fits(path, z=z, **read_kwargs))
            elif suffix == ".npz":
                specs.append(read_npz(path, z=z, **read_kwargs))
            else:
                raise ValueError(
                    f"Unsupported file extension {suffix!r}: pass a .fits "
                    f"or .npz file, or a Spectrum object."
                )

    use_flam = flux_unit.lower() == "flam"

    # Decide whether to stack a 2-D panel above the 1-D.  Only meaningful
    # for a single spectrum with a populated sci_2d (msaexp .spec.fits).
    # In multi-spec mode the 2-D panel is silently omitted (the explicit
    # user contract: "if multiple spectra, show 1-D only").
    if show_2d == "auto":
        _show_2d = (not multi) and (specs[0].sci_2d is not None)
    else:
        _show_2d = bool(show_2d) and (not multi) and (specs[0].sci_2d is not None)

    if _show_2d:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.26, 0.74],
            vertical_spacing=0.02,
        )
        panel_kw = dict(row=2, col=1)
    else:
        fig = go.Figure()
        panel_kw = {}

    all_flux_show: list[np.ndarray] = []
    x_mins: list[float] = []
    x_maxs: list[float] = []
    err_legend_done = False
    xlabel = ylabel = flux_label = None

    for i, spec in enumerate(specs):
        # Rest-frame scaling (per spectrum).  Source of redshift:
        # prefer the Spectrum's own z (e.g. set by read_fits(z=...)),
        # else fall back to the explicit ``z`` kwarg so users who read
        # a file without z=... can still get a rest-frame plot.  Only
        # silently disable rest-frame when neither is available.
        rf = rest_frame
        zp1 = 1.0
        z_data: float | None = spec.z if spec.z is not None else z
        if rf and z_data is not None:
            zp1 = 1.0 + float(z_data)
        elif rf:
            rf = False  # No z anywhere — silently observed-frame.

        if wave_unit == "A":
            wave = spec.wave_A / zp1
            xlabel = "Rest Wavelength [Å]" if rf else "Wavelength [Å]"
        else:
            wave = spec.wave_um / zp1
            xlabel = "Rest Wavelength [µm]" if rf else "Wavelength [µm]"

        if use_flam:
            flux = _ujy_to_flam(spec.flux_ujy, spec.wave_um)
            err = _ujy_to_flam(spec.err_ujy, spec.wave_um)
            flux_label = "erg/s/cm²/Å"
            ylabel = f"fλ [{flux_label}]"
        else:
            flux = spec.flux_ujy
            err = spec.err_ujy
            flux_label = "µJy"
            ylabel = f"Flux density [{flux_label}]"

        # For plotting, only require finite flux — be lenient about errors so
        # spectra without an error array (e.g. image HDUs) still render.
        valid = np.isfinite(flux)
        keep = _build_exclude_mask(spec.wave_A, exclude_wave_A)
        show = valid & keep
        has_err = np.any(np.isfinite(err) & (err > 0))
        err_show = show & np.isfinite(err) & (err > 0)

        # Per-trace colour and label.
        if multi:
            colour = _MULTI_PALETTE[i % len(_MULTI_PALETTE)]
            band_fill = _to_rgba(colour, 0.18)
        else:
            colour = "black"
            band_fill = "rgba(150,150,150,0.20)"

        if labels_in[i] is not None:
            name = labels_in[i]
        elif multi:
            fname = spec.meta.get("filename")
            name = str(fname) if fname else f"Spectrum {i + 1}"
        else:
            name = "Data"

        # Error band — step-shaped fill between ±1σ.  Uses two traces
        # with `fill='tonexty'` so the upper/lower edges are drawn as
        # step functions matching the data trace shape.  Legend entry
        # appears once across all spectra.
        if has_err and np.any(err_show):
            fig.add_trace(go.Scatter(
                x=wave[err_show],
                y=(flux - err)[err_show],
                mode="lines",
                line=dict(width=0, shape="hvh"),
                showlegend=False, hoverinfo="skip",
            ), **panel_kw)
            fig.add_trace(go.Scatter(
                x=wave[err_show],
                y=(flux + err)[err_show],
                mode="lines",
                line=dict(width=0, shape="hvh"),
                fill="tonexty", fillcolor=band_fill,
                name="±1σ",
                showlegend=not err_legend_done,
                hoverinfo="skip",
            ), **panel_kw)
            err_legend_done = True

        # Data trace (histogram-step style).
        fig.add_trace(go.Scatter(
            x=wave[show], y=flux[show],
            mode="lines", name=name,
            line=dict(color=colour, width=0.9, shape="hvh"),
            hovertemplate=f"λ=%{{x:.3f}}<br>flux=%{{y:.4e}} {flux_label}<extra></extra>",
        ), **panel_kw)

        if np.any(show):
            all_flux_show.append(flux[show])
            x_mins.append(float(np.nanmin(wave[show])))
            x_maxs.append(float(np.nanmax(wave[show])))

    # Title.
    if title is None:
        if not multi:
            spec = specs[0]
            bits = []
            fname = spec.meta.get("filename")
            if fname:
                bits.append(str(fname))
            # Title z: prefer spec.z, fall back to explicit z= kwarg.
            _z_title = spec.z if spec.z is not None else z
            if _z_title is not None:
                bits.append(f"z = {float(_z_title):.4f}")
            if spec.grating:
                bits.append(spec.grating)
            title = "  |  ".join(bits)
        else:
            title = ""

    # Y-limits — show the full vertical range so emission lines (including
    # faint ones like [OIII]λ4363) are visible by default.  Lower bound is
    # the 2nd percentile (floored at 0) to avoid noise outliers dominating
    # the axis; upper bound is the data max with a small pad on top.
    if all_flux_show:
        f_show = np.concatenate(all_flux_show)
        finite = np.isfinite(f_show)
        if np.any(finite):
            lo = float(np.nanpercentile(f_show[finite], 2))
            hi = float(np.nanmax(f_show[finite]))
            pad = 0.05 * (hi - lo if hi > lo else max(abs(hi), 1.0))
            y_lower = min(0.0, lo - pad)
            y_upper = hi + pad
        else:
            y_lower, y_upper = -1.0, 1.0
    else:
        y_lower, y_upper = -1.0, 1.0

    # --- 2-D rectified-spectrum panel (row=1 when enabled) ---
    if _show_2d:
        spec0 = specs[0]
        sci = np.asarray(spec0.sci_2d, dtype=float)
        # x-axis matches the 1-D wavelength array (and inherits rest-frame
        # scaling from the spec0 branch of the per-spectrum loop above).
        # Same z-resolution policy as the 1-D loop above: spec.z first,
        # explicit z= kwarg as fallback, else 1.0 (observed frame).
        _z_2d: float | None = spec0.z if spec0.z is not None else z
        zp1_0 = (1.0 + float(_z_2d)) if (rest_frame and _z_2d is not None) else 1.0
        if wave_unit == "A":
            wave_2d = spec0.wave_A / zp1_0
        else:
            wave_2d = spec0.wave_um / zp1_0
        ny = sci.shape[0]
        y_lo_pix = int(round(ny * y_crop[0]))
        y_hi_pix = max(y_lo_pix + 1, int(round(ny * y_crop[1])))
        sci_crop = sci[y_lo_pix:y_hi_pix, :]
        y_pix = np.arange(y_lo_pix, y_hi_pix)
        finite = np.isfinite(sci_crop)
        if np.any(finite):
            vmin, vmax = np.nanpercentile(
                sci_crop[finite], [vmin_pct, vmax_pct],
            )
        else:
            vmin, vmax = 0.0, 1.0
        fig.add_trace(
            go.Heatmap(
                x=wave_2d, y=y_pix, z=sci_crop,
                colorscale=cmap_2d, zmin=vmin, zmax=vmax,
                showscale=False, hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # Legend below the plot at all times — top-right covered emission
    # lines on tall axes, and horizontal-bottom looks the same whether
    # one spectrum or several are overlaid.
    legend = dict(orientation="h", x=0.5, xanchor="center",
                  y=-0.22, yanchor="top")
    bottom_margin = 110

    # Layout differs between single-panel and subplot modes: subplots
    # need per-axis updates (update_xaxes / update_yaxes with row/col).
    if _show_2d:
        fig.update_layout(
            title=title,
            template="plotly_white",
            hovermode="x unified",
            dragmode="zoom",
            legend=legend,
            width=1000,
            height=620,
        )
        fig.update_xaxes(exponentformat="none")
        # Bottom (1-D) panel labels + range.
        fig.update_xaxes(title_text=xlabel, row=2, col=1)
        fig.update_yaxes(
            title_text=ylabel, range=[y_lower, y_upper], row=2, col=1,
        )
        # Top (2-D) panel: hide tick labels, label spatial axis.
        fig.update_yaxes(
            title_text="Spatial pix", showticklabels=False, row=1, col=1,
        )
        if bottom_margin is not None:
            fig.update_layout(margin=dict(b=bottom_margin))
    else:
        layout_kwargs = dict(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            yaxis_range=[y_lower, y_upper],
            xaxis=dict(exponentformat="none"),
            template="plotly_white",
            hovermode="x unified",
            dragmode="zoom",
            legend=legend,
            width=1000,
            height=500,
        )
        if bottom_margin is not None:
            layout_kwargs["margin"] = dict(b=bottom_margin)
        fig.update_layout(**layout_kwargs)

    # --- Zero-flux reference (light grey dashed) ---
    if show_zero:
        if _show_2d:
            fig.add_hline(
                y=0,
                line_width=1,
                line_dash="dash",
                line_color="lightgrey",
                layer="below",
                row=2, col=1,
            )
        else:
            fig.add_hline(
                y=0,
                line_width=1,
                line_dash="dash",
                line_color="lightgrey",
                layer="below",
            )

    # --- Emission-line markers at supplied redshift ---
    z_eff = z
    if z_eff is None and not multi:
        z_eff = specs[0].z

    if rest_frame:
        z_for_lines: float | None = 0.0
    elif z_eff is not None:
        z_for_lines = float(z_eff)
    else:
        z_for_lines = None

    _draw_emission_line_markers(
        fig,
        z_for_lines=z_for_lines,
        wave_unit=wave_unit,
        x_lo=min(x_mins) if x_mins else None,
        x_hi=max(x_maxs) if x_maxs else None,
        lines=lines,
        add_lines=add_lines,
        line_color=line_color,
        use_paper_shapes=_show_2d,
    )

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
    rest_frame: bool = False,
    z: float | None = None,
    lines: "Sequence[str] | bool | None" = False,
    add_lines: "dict[str, float] | Sequence[str] | None" = None,
    line_color: str = "darkred",
    show_2d: "bool | str" = "auto",
    cmap_2d: str = "Blues",
    vmin_pct: float = 5.0,
    vmax_pct: float = 99.5,
    y_crop: tuple[float, float] = (0.25, 0.75),
) -> "go.Figure":
    """Interactive plotly plot of a spectral fit with zoom and hover.

    Optionally stacks a 2-D rectified-spectrum panel above the main fit
    panel (when ``result.spectrum.sci_2d`` is populated by
    :func:`read_fits`) and a residual panel below.  By default no
    curated emission-line markers are drawn — every fitted line is
    already labelled at its peak above the model.  Pass ``lines=None``
    to add the package-default markers, or an explicit list of
    :data:`jwspecfit.lines.REST_LINES_A` keys to mark only those.

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
    rest_frame : bool
        If ``True``, plot wavelengths in the rest frame by dividing by
        ``(1 + z)``.  Default ``False`` (observed frame).  Emission-line
        markers are placed at rest wavelengths when ``True``.
    z : float, optional
        Redshift override used for rest-frame conversion and
        marker placement.  When ``None`` (default),
        ``result.spectrum.z`` is used.  Raises ``ValueError`` if
        ``rest_frame=True`` and neither source provides a redshift.
    lines : sequence of str, bool, or None
        Curated emission-line markers (vertical dashed lines + labels).
        ``False`` (default) draws none — the fit-component peak
        annotations already identify every fitted line.  ``None``
        opts in to the package-default marker set; an explicit list of
        :data:`jwspecfit.lines.REST_LINES_A` keys marks only those
        names.
    add_lines : dict[str, float] or sequence of str, optional
        Extra markers to overlay on top of *lines*.  Same semantics as
        in :func:`plot_spectrum_interactive`.
    line_color : str
        Colour for the emission-line markers and their labels
        (default ``"darkred"``).
    show_2d : bool or {"auto"}
        Whether to stack the 2-D rectified spectrum image above the fit
        panel.  ``"auto"`` (default) shows it iff
        ``result.spectrum.sci_2d`` is populated.  ``True`` forces it
        (silently no-ops when no 2-D is available); ``False`` always
        hides it.
    cmap_2d : str
        Plotly colorscale for the 2-D panel (default ``"Blues"``).
    vmin_pct, vmax_pct : float
        Percentile clip for the 2-D colour scale (default ``5`` /
        ``99.5``).
    y_crop : (float, float)
        Fractional crop ``(lo, hi)`` of the spatial axis on the 2-D
        panel (default ``(0.25, 0.75)``).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from .io import _flam_to_ujy, _ujy_to_flam

    # Auto-convert MCMCResult / MCMCBroadFitResult to FitResult.
    if hasattr(result, "to_fit_result") and not hasattr(result, "residuals"):
        result = result.to_fit_result()

    spec = result.spectrum

    # Rest-frame scaling factor.  Fail loudly if rest_frame is requested
    # but no z is available — silently falling back to zp1=1 produced a
    # plot that looked observed-frame with no warning.
    z_used = z if z is not None else spec.z
    if rest_frame and z_used is None:
        raise ValueError(
            "rest_frame=True but no redshift available: result.spectrum.z "
            "is None and no z= override was supplied.  Pass z=... or set "
            "spec.z before calling."
        )
    zp1 = (1.0 + float(z_used)) if rest_frame else 1.0

    if wave_unit == "A":
        wave = spec.wave_A / zp1
        xlabel = "Rest Wavelength [Å]" if rest_frame else "Wavelength [Å]"
    else:
        wave = spec.wave_um / zp1
        xlabel = "Rest Wavelength [µm]" if rest_frame else "Wavelength [µm]"

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

    # Decide whether to stack a 2-D panel on top.  When the spectrum
    # carries no 2-D, "auto" silently no-ops, and an explicit True is
    # also no-op'd (so the call doesn't fail on synthetic spectra).
    if show_2d == "auto":
        _show_2d = spec.sci_2d is not None
    else:
        _show_2d = bool(show_2d) and spec.sci_2d is not None

    # Figure layout: 1, 2, or 3 rows depending on which panels are on.
    # Row indices and heights are derived once so the rest of the
    # function can refer to _main_row / _resid_row / _viz_2d_row
    # without re-deriving them.
    if _show_2d and show_residuals:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.20, 0.60, 0.20], vertical_spacing=0.03,
        )
        _viz_2d_row, _main_row, _resid_row = 1, 2, 3
    elif _show_2d and not show_residuals:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.26, 0.74], vertical_spacing=0.03,
        )
        _viz_2d_row, _main_row, _resid_row = 1, 2, None
    elif show_residuals:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.04,
        )
        _viz_2d_row, _main_row, _resid_row = None, 1, 2
    else:
        fig = go.Figure()
        _viz_2d_row = _main_row = _resid_row = None

    _has_subplots = _show_2d or show_residuals

    def _add(trace, row=None):
        if _has_subplots and row is not None:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

    # Error band.
    _add(go.Scatter(
        x=np.concatenate([wave[show], wave[show][::-1]]),
        y=np.concatenate([(flux + err)[show], (flux - err)[show][::-1]]),
        fill="toself", fillcolor="rgba(150,150,150,0.15)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=_main_row)

    # Individual line components as smooth analytical Gaussians.
    if show_components and len(result.line_names) > 0:
        from math import sqrt, pi
        nL = len(result.line_names)

        import plotly.express as px
        palette = px.colors.qualitative.Set2

        # Interpolate continuum onto a fine grid for smooth component curves.
        cont_interp_fn = np.interp
        peak_info = []

        # Build index mapping: skip Lyα (not in Gaussian params vector).
        _gauss_names_i = [n for n in result.line_names if n != "Lya"]
        _gauss_idx_i = {n: j for j, n in enumerate(_gauss_names_i)}
        nL_gauss_i = len(_gauss_names_i)

        for i, name in enumerate(result.line_names):
            if name == "Lya":
                continue  # Lyα uses skewed Gaussian, not in params vector
            gi = _gauss_idx_i[name]
            amp = result.params[gi]
            mu_A = result.params[nL_gauss_i + gi]
            sig_A = result.params[2 * nL_gauss_i + gi]

            if amp == 0 or sig_A <= 0:
                continue

            is_abs = name.startswith("abs_")

            # Fine wavelength grid around ±5σ of the line.
            w_lo = max(mu_A - 5 * sig_A, spec.wave_A.min())
            w_hi = min(mu_A + 5 * sig_A, spec.wave_A.max())
            n_fine = max(int((w_hi - w_lo) / (sig_A / 5)), 100)
            wave_fine_A = np.linspace(w_lo, w_hi, n_fine)
            wave_fine_um = wave_fine_A * 1e-4

            # Analytical Gaussian in F_λ: G(λ) = A / (√(2π) σ) × exp(...)
            gauss_flam = (amp / (sqrt(2 * pi) * sig_A)) * np.exp(
                -0.5 * ((wave_fine_A - mu_A) / sig_A) ** 2
            )

            # Continuum at fine grid points (interpolated).
            cont_fine_ujy = np.interp(wave_fine_um, spec.wave_um, result.continuum)

            if use_flam:
                gauss_plot = gauss_flam + _ujy_to_flam(cont_fine_ujy, wave_fine_um)
                cont_fine = _ujy_to_flam(cont_fine_ujy, wave_fine_um)
            else:
                gauss_plot = _flam_to_ujy(gauss_flam, wave_fine_um) + cont_fine_ujy
                cont_fine = cont_fine_ujy

            # Convert to the chosen wave unit (with rest-frame scaling).
            wave_fine = wave_fine_A / zp1 if wave_unit == "A" else wave_fine_um / zp1

            # Apply exclusion mask.
            keep_fine = _build_exclude_mask(wave_fine_A, exclude_wave_A)
            gauss_masked = gauss_plot.copy()
            gauss_masked[~keep_fine] = np.nan
            cont_fine_masked = cont_fine.copy()
            cont_fine_masked[~keep_fine] = np.nan

            is_broad = "BROAD" in name
            colour = _component_colour(name, is_abs)
            display_name = name.replace("abs_", "").replace("_", " ")
            dash = "dot" if is_broad else "solid"

            # Uncertainty shading (±1σ).
            lr = result.lines.get(name)
            frac_err = 0.0
            if lr is not None and abs(lr.flux) > 0 and lr.flux_err > 0:
                frac_err = lr.flux_err / abs(lr.flux)

            if frac_err > 0:
                line_only = gauss_masked - cont_fine_masked
                comp_hi = cont_fine_masked + line_only * (1.0 + frac_err)
                comp_lo = cont_fine_masked + line_only * max(1.0 - frac_err, 0.0)
                fill = _to_rgba(colour, 0.12)
                _add(go.Scatter(
                    x=np.concatenate([wave_fine, wave_fine[::-1]]),
                    y=np.concatenate([comp_hi, comp_lo[::-1]]),
                    fill="toself", fillcolor=fill,
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ), row=_main_row)

            # Smooth Gaussian curve.
            _add(go.Scatter(
                x=wave_fine, y=gauss_masked,
                mode="lines", name=f"{'[B] ' if is_broad else ''}{display_name}",
                line=dict(color=colour, width=1.5, dash=dash),
                hovertemplate=f"{display_name}<br>λ=%{{x:.1f}}<br>flux=%{{y:.4e}} {flux_label}<extra></extra>",
                showlegend=False,
            ), row=_main_row)

            # Store peak/trough position for annotation label.
            gauss_peak_flam = amp / (sqrt(2 * pi) * sig_A)
            cont_at_peak_ujy = np.interp(
                mu_A * 1e-4, spec.wave_um, result.continuum,
            )
            if use_flam:
                y_peak = gauss_peak_flam + _ujy_to_flam(
                    np.array([cont_at_peak_ujy]),
                    np.array([mu_A * 1e-4]),
                )[0]
            else:
                y_peak = (
                    _flam_to_ujy(
                        np.array([gauss_peak_flam]),
                        np.array([mu_A * 1e-4]),
                    )[0]
                    + cont_at_peak_ujy
                )
            x_peak = mu_A / zp1 if wave_unit == "A" else mu_A * 1e-4 / zp1
            peak_info.append((name, x_peak, float(y_peak), colour, is_abs))

        # Lyα asymmetric Gaussian overlay.
        _lya_p = getattr(result, "lya_params", None)
        if _lya_p is not None and len(_lya_p) == 4:
            from .models import asymmetric_gaussian as _ag

            _A_pk, _mu_lya, _sig_lya, _alpha_lya = _lya_p
            comp_col = _COMP_COLOUR_NARROW

            # Fine wavelength grid around the line.
            w_lo_c = max(_mu_lya - 8 * _sig_lya, spec.wave_A.min())
            w_hi_c = min(_mu_lya + 12 * _sig_lya, spec.wave_A.max())
            n_fine_c = max(int((w_hi_c - w_lo_c) / (_sig_lya / 5)), 200)
            wave_fine_c = np.linspace(w_lo_c, w_hi_c, n_fine_c)
            wave_fine_c_um = wave_fine_c * 1e-4

            prof_flam = _ag(wave_fine_c, _A_pk, _mu_lya, _sig_lya, _alpha_lya)

            cont_fine_c_ujy = np.interp(wave_fine_c_um, spec.wave_um, result.continuum)
            if use_flam:
                prof_plot = prof_flam + _ujy_to_flam(cont_fine_c_ujy, wave_fine_c_um)
                cont_fine_c = _ujy_to_flam(cont_fine_c_ujy, wave_fine_c_um)
            else:
                prof_plot = _flam_to_ujy(prof_flam, wave_fine_c_um) + cont_fine_c_ujy
                cont_fine_c = cont_fine_c_ujy

            wave_fine_c_plot = wave_fine_c / zp1 if wave_unit == "A" else wave_fine_c_um / zp1
            keep_c = _build_exclude_mask(wave_fine_c, exclude_wave_A)
            prof_masked = prof_plot.copy()
            prof_masked[~keep_c] = np.nan
            cont_fine_c_masked = cont_fine_c.copy()
            cont_fine_c_masked[~keep_c] = np.nan

            # Uncertainty shading.
            lr_lya = result.lines.get("Lya")
            frac_err_c = 0.0
            if lr_lya is not None and abs(lr_lya.flux) > 0:
                fe = lr_lya.flux_err
                fe_val = 0.5 * (fe[0] + fe[1]) if isinstance(fe, tuple) else fe
                if fe_val > 0:
                    frac_err_c = fe_val / abs(lr_lya.flux)

            if frac_err_c > 0:
                line_only = prof_masked - cont_fine_c_masked
                comp_hi = cont_fine_c_masked + line_only * (1.0 + frac_err_c)
                comp_lo = cont_fine_c_masked + line_only * max(1.0 - frac_err_c, 0.0)
                fill_c = _to_rgba(comp_col, 0.12)
                _add(go.Scatter(
                    x=np.concatenate([wave_fine_c_plot, wave_fine_c_plot[::-1]]),
                    y=np.concatenate([comp_hi, comp_lo[::-1]]),
                    fill="toself", fillcolor=fill_c,
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ), row=_main_row)

            _add(go.Scatter(
                x=wave_fine_c_plot, y=prof_masked,
                mode="lines", name="Lyα",
                line=dict(color=comp_col, width=1.5, dash="solid"),
                hovertemplate="Lyα<br>λ=%{x:.1f}<br>flux=%{y:.4e} " + flux_label + "<extra></extra>",
                showlegend=False,
            ), row=_main_row)

            # Peak annotation.
            _pk_idx = np.argmax(prof_flam)
            peak_flam_c = prof_flam[_pk_idx]
            mu_pk = wave_fine_c[_pk_idx]
            cont_at_peak_c = np.interp(mu_pk * 1e-4, spec.wave_um, result.continuum)
            if use_flam:
                y_peak_c = peak_flam_c + _ujy_to_flam(
                    np.array([cont_at_peak_c]), np.array([mu_pk * 1e-4]))[0]
            else:
                y_peak_c = _flam_to_ujy(
                    np.array([peak_flam_c]), np.array([mu_pk * 1e-4]))[0] + cont_at_peak_c
            x_peak_c = mu_pk / zp1 if wave_unit == "A" else mu_pk * 1e-4 / zp1
            peak_info.append(("Lyα", x_peak_c, float(y_peak_c), comp_col, False))

        # --- Line-name annotations with row staggering to avoid overlap ---
        # Separate emission and absorption labels so each side staggers
        # independently (emission rises above peaks, absorption drops
        # below troughs).  Within each side, sort by x and place each
        # label in the first row where it sits ≥ `threshold` away from
        # the last label in that row.
        emission_lbls = [info for info in peak_info if not info[4]]
        absorption_lbls = [info for info in peak_info if info[4]]
        row_spacing_px = 14

        if peak_info:
            x_positions = [info[1] for info in peak_info]
            x_range = max(x_positions) - min(x_positions)
            threshold = 0.025 * x_range if x_range > 0 else 0.0
        else:
            threshold = 0.0

        def _assign_rows(labels, thr):
            labels_sorted = sorted(labels, key=lambda info: info[1])
            row_last_x: list[float] = []
            rows: list[int] = []
            for _, x, _, _, _ in labels_sorted:
                placed = False
                for r, last_x in enumerate(row_last_x):
                    if x - last_x >= thr:
                        row_last_x[r] = x
                        rows.append(r)
                        placed = True
                        break
                if not placed:
                    row_last_x.append(x)
                    rows.append(len(row_last_x) - 1)
            return labels_sorted, rows

        for labels_group, sign in ((emission_lbls, +1), (absorption_lbls, -1)):
            if not labels_group:
                continue
            labels_sorted, rows = _assign_rows(labels_group, threshold)
            base_yshift = 10 if sign > 0 else -12
            yanchor = "bottom" if sign > 0 else "top"
            for (name, x_peak, y_peak, colour, _is_abs), row in zip(
                labels_sorted, rows,
            ):
                display_name = name.replace("abs_", "").replace("_", " ")
                is_broad = "BROAD" in name
                if "rgba" in colour:
                    ann_colour = colour.rsplit(",", 1)[0] + ",1.0)"
                else:
                    ann_colour = colour
                yshift = base_yshift + sign * row * row_spacing_px
                fig.add_annotation(
                    x=x_peak,
                    y=y_peak,
                    xref="x",
                    yref="y",
                    text=f"<b>{display_name}</b>" if is_broad else display_name,
                    showarrow=False,
                    yshift=yshift,
                    font=dict(size=9, color=ann_colour),
                    xanchor="center",
                    yanchor=yanchor,
                )

    # --- 2-D rectified-spectrum panel (row=_viz_2d_row when enabled) ---
    if _show_2d:
        sci = np.asarray(spec.sci_2d, dtype=float)
        # Mirror the 1-D wavelength scaling so the 2-D heatmap aligns
        # column-for-column with the data trace below.
        if wave_unit == "A":
            wave_2d = spec.wave_A / zp1
        else:
            wave_2d = spec.wave_um / zp1
        ny = sci.shape[0]
        y_lo_pix = int(round(ny * y_crop[0]))
        y_hi_pix = max(y_lo_pix + 1, int(round(ny * y_crop[1])))
        sci_crop = sci[y_lo_pix:y_hi_pix, :]
        y_pix = np.arange(y_lo_pix, y_hi_pix)
        finite = np.isfinite(sci_crop)
        if np.any(finite):
            vmin, vmax = np.nanpercentile(
                sci_crop[finite], [vmin_pct, vmax_pct],
            )
        else:
            vmin, vmax = 0.0, 1.0
        _add(go.Heatmap(
            x=wave_2d, y=y_pix, z=sci_crop,
            colorscale=cmap_2d, zmin=vmin, zmax=vmax,
            showscale=False, hoverinfo="skip",
        ), row=_viz_2d_row)

    # Data (steps).
    _add(go.Scatter(
        x=wave[show], y=flux[show],
        mode="lines", name="Data",
        line=dict(color="grey", width=0.8, shape="hvh"),
        hovertemplate=f"Data<br>λ=%{{x:.1f}}<br>flux=%{{y:.4e}} {flux_label}<extra></extra>",
    ), row=_main_row)

    # Continuum (steps).
    _add(go.Scatter(
        x=wave_k, y=cont_k,
        mode="lines", name="Continuum",
        line=dict(color="green", width=1, dash="dash", shape="hvh"),
    ), row=_main_row)

    # Model (steps, semi-transparent).
    _add(go.Scatter(
        x=wave_k, y=model_k,
        mode="lines", name="Model",
        line=dict(color="rgba(255,0,0,0.4)", width=1.5, shape="hvh"),
        hovertemplate=f"Model<br>λ=%{{x:.1f}}<br>flux=%{{y:.4e}} {flux_label}<extra></extra>",
    ), row=_main_row)

    # --- Residual panel ---
    if show_residuals:
        # Error band on residuals.
        _add(go.Scatter(
            x=np.concatenate([wave[show], wave[show][::-1]]),
            y=np.concatenate([err[show], (-err)[show][::-1]]),
            fill="toself", fillcolor="rgba(150,150,150,0.15)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=_resid_row)

        # Zero line.
        _add(go.Scatter(
            x=[wave[show].min(), wave[show].max()],
            y=[0, 0],
            mode="lines", showlegend=False,
            line=dict(color="rgba(255,0,0,0.4)", width=1, dash="dash"),
        ), row=_resid_row)

        # Residual data.
        _add(go.Scatter(
            x=wave[show], y=resid[show],
            mode="lines", name="Residual", showlegend=False,
            line=dict(color="grey", width=0.8, shape="hvh"),
            hovertemplate=f"Residual<br>λ=%{{x:.1f}}<br>resid=%{{y:.4e}} {flux_label}<extra></extra>",
        ), row=_resid_row)

    # Y limits — account for absorption troughs.
    model_peak = np.nanmax(model_total[show]) if np.any(show) else 1.0
    model_trough = np.nanmin(model_total[show]) if np.any(show) else 0.0
    cont_median = np.nanmedian(cont[show]) if np.any(show) else 0.0
    y_upper = cont_median + (model_peak - cont_median) * y_pad
    y_lower_cont = np.nanmin(cont[show]) * 1.1 if np.any(show) else -0.1
    y_lower = min(0.0, y_lower_cont, model_trough - abs(model_trough) * 0.15)

    title_bits = []
    if z_used is not None:
        title_bits.append(f"z = {float(z_used):.4f}")
    title_bits.append(f"χ²/dof = {result.chi2:.2f}")
    if rest_frame:
        title_bits.append("rest frame")
    title = "  |  ".join(title_bits)

    ylabel = f"fλ [{flux_label}]" if use_flam else f"Flux density [{flux_label}]"

    # Bottom-horizontal legend in every mode — the previous top-right
    # placement covered emission-line markers and labels.  Margin
    # leaves room for the horizontal legend (and any stacked line
    # labels above the plot — _draw_emission_line_markers grows the
    # top margin to fit).
    _legend = dict(
        orientation="h", x=0.5, xanchor="center", y=-0.18, yanchor="top",
    )

    if _has_subplots:
        # Figure height by row count: 1 + 2D, 1 + residual, or 1+2D+residual.
        if _show_2d and show_residuals:
            fig_height = 760
        elif _show_2d:
            fig_height = 620
        else:
            fig_height = 650  # legacy show_residuals-only height

        fig.update_layout(
            title=title,
            template="plotly_white",
            hovermode=False,
            dragmode="zoom",
            legend=_legend,
            width=1000,
            height=fig_height,
            margin=dict(b=110),
        )
        fig.update_xaxes(exponentformat="none")
        # Main panel always gets the wavelength label so the unit is
        # visible without scrolling; bottom-most panel also gets it.
        fig.update_xaxes(title_text=xlabel, row=_main_row, col=1)
        fig.update_yaxes(
            title_text=ylabel, range=[y_lower, y_upper],
            row=_main_row, col=1,
        )
        if _show_2d:
            fig.update_yaxes(
                title_text="Spatial pix", showticklabels=False,
                row=_viz_2d_row, col=1,
            )
        if show_residuals:
            # Clip residual y-axis to ±5× median error so noise spikes
            # don't dominate the panel height.
            med_err = float(np.nanmedian(err[show])) if np.any(show) else 1.0
            res_ylim = 5.0 * med_err
            fig.update_yaxes(
                title_text="Residual", range=[-res_ylim, res_ylim],
                row=_resid_row, col=1,
            )
            fig.update_xaxes(title_text=xlabel, row=_resid_row, col=1)
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
            legend=_legend,
            width=1000,
            height=500,
            margin=dict(b=110),
        )

    # --- Curated emission-line markers (rest-frame aware) ---
    if rest_frame:
        z_for_lines: float | None = 0.0
    elif z_used is not None:
        z_for_lines = float(z_used)
    else:
        z_for_lines = None

    if np.any(show):
        x_data = wave[show]
        x_lo_m = float(np.nanmin(x_data))
        x_hi_m = float(np.nanmax(x_data))
    else:
        x_lo_m = x_hi_m = None

    _draw_emission_line_markers(
        fig,
        z_for_lines=z_for_lines,
        wave_unit=wave_unit,
        x_lo=x_lo_m,
        x_hi=x_hi_m,
        lines=lines,
        add_lines=add_lines,
        line_color=line_color,
        use_paper_shapes=_has_subplots,
    )

    return fig


def plot_2d_1d(
    path: str | Path,
    z: float | None = None,
    *,
    sci_ext: str = "SCI",
    hdu: str | int | None = None,
    wave_col: str | None = None,
    flux_col: str | None = None,
    err_col: str | None = None,
    flux_scale: float = 1e3,
    flux_label: str = r"$F_\nu$ [nJy]",
    xlabel: str = r"$\lambda_{\rm obs}\,[\mu{\rm m}]$",
    cmap: str = "Blues",
    vmin_pct: float = 5.0,
    vmax_pct: float = 99.5,
    y_crop: tuple[float, float] = (0.25, 0.75),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    line_colour: str = "steelblue",
    line_width: float = 0.5,
    err_colour: str = "grey",
    err_alpha: float = 0.35,
    figsize: tuple[float, float] = (8, 4),
    dpi: float = 300,
    title: str | None = None,
    lines: list[str] | bool | None = None,
    add_lines: list[str] | dict[str, float] | None = None,
    height_ratios: tuple[float, float] = (0.35, 1.0),
) -> tuple["Figure", tuple["Axes", "Axes"]]:
    """Plot a JWST/NIRSpec 2D + 1D spectrum from a ``.spec.fits`` file.

    Reads the 2D image from ``sci_ext`` and the 1D extraction from
    ``SPEC1D`` via :func:`jwspecfit.read_fits`, then renders a stacked
    figure (2D pcolormesh above, 1D flux below) with optional emission-
    line markers at the supplied redshift.

    Parameters
    ----------
    path
        Path to the ``.spec.fits`` file.
    z
        Redshift used to place observed-frame line markers.  If
        ``None``, no markers are drawn.
    sci_ext
        Name of the 2D image extension (default ``"SCI"``).
    hdu, wave_col, flux_col, err_col
        Forwarded to :func:`read_fits` for 1D extraction (override the
        default SPEC1D / column auto-detection).
    flux_scale, flux_label
        Multiplier and y-axis label applied to the 1D flux
        (default converts µJy → nJy).
    xlabel
        Shared x-axis label (default observed wavelength in µm).
    cmap
        Colormap for the 2D panel.
    vmin_pct, vmax_pct
        Percentile clip for the 2D colour scale.
    y_crop
        ``(y_lo_frac, y_hi_frac)`` fractional crop of the 2D vertical
        extent (default keeps the middle 50 % rows).
    xlim, ylim
        Optional axis limits.  ``xlim`` applies to both panels.
    line_colour, line_width
        Colour and line width of the 1D flux trace.
    err_colour, err_alpha
        Fill colour and alpha for the ``±1σ`` error band.
    figsize, dpi, height_ratios
        Matplotlib figure size, resolution (default 300), and 2D-vs-1D
        height ratio.
    title
        Optional title placed above the 2D panel.
    lines
        Default-marker control: ``None`` uses the package defaults;
        a list of :data:`REST_LINES_A` keys restricts to those names;
        ``False`` disables defaults entirely.
    add_lines
        Extra markers: either REST_LINES_A keys, or a
        ``{label: rest_wavelength_A}`` dict.

    Returns
    -------
    fig, (ax_2d, ax_1d)
        Matplotlib figure and the two axes.
    """
    import matplotlib.pyplot as plt
    from astropy.io import fits

    from .io import read_fits
    from .lines import REST_LINES_A

    path = Path(path)

    spec = read_fits(
        path, hdu=hdu, wave_col=wave_col, flux_col=flux_col, err_col=err_col,
    )
    with fits.open(path) as hdul:
        ext_names = [h.name for h in hdul]
        if sci_ext not in ext_names:
            raise KeyError(
                f"Extension {sci_ext!r} not found in {path.name} "
                f"(available: {ext_names})"
            )
        sci2d = np.asarray(hdul[sci_ext].data)

    if sci2d.ndim != 2:
        raise ValueError(
            f"Expected 2D array in {sci_ext!r}, got shape {sci2d.shape}"
        )

    wave = np.asarray(spec.wave_um)
    flux = np.asarray(spec.flux_ujy)
    err = np.asarray(spec.err_ujy) if spec.err_ujy is not None else None

    # 2D wavelength axis must match the 1D wave grid in length.
    if sci2d.shape[1] != wave.size:
        raise ValueError(
            f"SCI wavelength axis ({sci2d.shape[1]} pix) does not match "
            f"SPEC1D wave length ({wave.size})."
        )

    fig, (ax2d, ax1d) = plt.subplots(
        2, 1,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0.05},
        sharex=True,
    )

    # --- 2D panel --------------------------------------------------
    dw = float(np.median(np.diff(wave)))
    wave_edges = np.concatenate([
        [wave[0] - dw / 2.0],
        0.5 * (wave[1:] + wave[:-1]),
        [wave[-1] + dw / 2.0],
    ])
    ny = sci2d.shape[0]
    y_edges = np.arange(ny + 1)
    vmin, vmax = np.nanpercentile(sci2d, [vmin_pct, vmax_pct])
    ax2d.pcolormesh(
        wave_edges, y_edges, sci2d,
        shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax2d.set_ylim(ny * y_crop[0], ny * y_crop[1])
    ax2d.tick_params(labelbottom=False, labelleft=False)
    ax2d.set_ylabel("Spatial pix", fontsize=9)
    if title is not None:
        ax2d.set_title(title, fontsize=10)

    # --- 1D panel --------------------------------------------------
    f_scaled = flux * flux_scale
    ax1d.plot(wave, f_scaled, color=line_colour, lw=line_width)
    if err is not None:
        ax1d.fill_between(
            wave,
            (flux - err) * flux_scale,
            (flux + err) * flux_scale,
            color=err_colour, alpha=err_alpha, lw=0,
        )
    ax1d.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax1d.tick_params(direction="in", top=True, right=True, labelsize=8)
    ax1d.set_xlabel(xlabel, fontsize=10)
    ax1d.set_ylabel(flux_label, fontsize=10)

    if xlim is not None:
        ax1d.set_xlim(*xlim)
        ax2d.set_xlim(*xlim)

    if ylim is not None:
        ax1d.set_ylim(*ylim)
    else:
        # Auto-scale to the brightest emission line in the visible window
        # (typically [OIII]5007 for low-z optical spectra) — matches the
        # plot_spectrum_interactive behaviour: 2nd-percentile floor, data
        # max with a small pad.
        x_lo, x_hi = ax1d.get_xlim()
        in_view = (wave >= x_lo) & (wave <= x_hi) & np.isfinite(f_scaled)
        if np.any(in_view):
            lo = float(np.nanpercentile(f_scaled[in_view], 2))
            hi = float(np.nanmax(f_scaled[in_view]))
            pad = 0.05 * (hi - lo if hi > lo else max(abs(hi), 1.0))
            y_lower = min(0.0, lo - pad)
            y_upper = hi + pad
            if y_upper > y_lower:
                ax1d.set_ylim(y_lower, y_upper)

    # --- Emission-line markers ------------------------------------
    if z is not None:
        x_lo, x_hi = ax1d.get_xlim()

        markers: list[tuple[float, str]] = []
        if lines is not False:
            names = _DEFAULT_MARKER_NAMES if lines is None else list(lines)
            for nm in names:
                rest_A = REST_LINES_A.get(nm)
                if rest_A is None:
                    continue
                obs_um = float(rest_A) * (1.0 + z) / 1e4
                if x_lo < obs_um < x_hi:
                    label = _DEFAULT_MARKER_LABELS.get(nm, nm)
                    markers.append((obs_um, label))

        if add_lines:
            if isinstance(add_lines, dict):
                add_items = list(add_lines.items())
            else:
                add_items = []
                for nm in add_lines:
                    rest_A = REST_LINES_A.get(nm)
                    if rest_A is None:
                        continue
                    label = _DEFAULT_MARKER_LABELS.get(nm, nm.replace("_", " "))
                    add_items.append((label, rest_A))
            for label, rest_A in add_items:
                obs_um = float(rest_A) * (1.0 + z) / 1e4
                if x_lo < obs_um < x_hi:
                    markers.append((obs_um, str(label)))

        y_top = ax1d.get_ylim()[1]
        for x_obs, label in markers:
            for ax in (ax2d, ax1d):
                ax.axvline(x_obs, color="gray", ls="--", lw=0.7, alpha=0.6)
            ax1d.text(
                x_obs, y_top * 0.95, label,
                color="gray", rotation=90,
                ha="center", va="top", fontsize=7,
            )

    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.12)

    return fig, (ax2d, ax1d)
