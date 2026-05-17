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
    **read_kwargs
        Forwarded to the file reader when a *source* item is a path.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
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

    fig = go.Figure()
    all_flux_show: list[np.ndarray] = []
    x_mins: list[float] = []
    x_maxs: list[float] = []
    err_legend_done = False
    xlabel = ylabel = flux_label = None

    for i, spec in enumerate(specs):
        # Rest-frame scaling (per spectrum).
        rf = rest_frame
        zp1 = 1.0
        if rf and spec.z is not None:
            zp1 = 1.0 + spec.z
        elif rf and spec.z is None:
            rf = False  # Silently fall back; no z available.

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
            ))
            fig.add_trace(go.Scatter(
                x=wave[err_show],
                y=(flux + err)[err_show],
                mode="lines",
                line=dict(width=0, shape="hvh"),
                fill="tonexty", fillcolor=band_fill,
                name="±1σ",
                showlegend=not err_legend_done,
                hoverinfo="skip",
            ))
            err_legend_done = True

        # Data trace (histogram-step style).
        fig.add_trace(go.Scatter(
            x=wave[show], y=flux[show],
            mode="lines", name=name,
            line=dict(color=colour, width=0.9, shape="hvh"),
            hovertemplate=f"λ=%{{x:.3f}}<br>flux=%{{y:.4e}} {flux_label}<extra></extra>",
        ))

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
            if spec.z is not None:
                bits.append(f"z = {spec.z:.4f}")
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

    if multi:
        legend = dict(orientation="h", x=0.5, xanchor="center",
                      y=-0.22, yanchor="top")
        bottom_margin = 110
    else:
        legend = dict(x=1.0, y=1.0, xanchor="right")
        bottom_margin = None

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

    if z_for_lines is not None and x_mins and x_maxs:
        from .lines import REST_LINES_A

        default_names = [
            "Lya", "NIV_doublet", "CIV_doublet", "HEII_1640",
            "NIII_doublet", "CIII]",
            "OII_doublet", "NeIII_3869",
            "HEI_4027", "HDELTA", "HEI_4145", "HGAMMA", "OIII_4363",
            "FeII_4584", "NIII_4642", "FeIII_4660", "HeII_4687",
            "ArIV_4713", "FeII_4732", "ArIV_4741",
            "HBETA", "OIII_4959", "OIII_5007",
            "HEI_5877", "OI_6302",
            "Ha", "NII_6585", "HEI_6680",
            "SII_6718", "SII_6732",
            "HEI_7067", "ArIII_7138",
        ]
        display = {
            "Lya": "Lyα", "NIV_doublet": "NIV", "CIV_doublet": "CIV",
            "HEII_1640": "HeII 1640",
            "NIII_doublet": "NIII 1750", "CIII]": "CIII]",
            "OII_doublet": "[OII]", "NeIII_3869": "[NeIII]",
            "HEI_4027": "HeI 4027",
            "HDELTA": "Hδ",
            "HEI_4145": "HeI 4145",
            "HGAMMA": "Hγ",
            "OIII_4363": "[OIII]4363",
            "FeII_4584": "FeII 4584", "NIII_4642": "NIII 4642",
            "FeIII_4660": "FeIII 4660", "HeII_4687": "HeII 4687",
            "ArIV_4713": "ArIV+HeI 4705",
            "FeII_4732": "FeII 4732", "ArIV_4741": "ArIV 4741",
            "HBETA": "Hβ",
            "OIII_4959": "[OIII]4959", "OIII_5007": "[OIII]5007",
            "HEI_5877": "HeI 5877",
            "OI_6302": "[OI] 6302",
            "Ha": "Hα", "NII_6585": "[NII] 6585",
            "HEI_6680": "HeI 6680",
            "SII_6718": "[SII]6716", "SII_6732": "[SII]6731",
            "HEI_7067": "HeI 7067",
            "ArIII_7138": "[ArIII] 7138",
        }

        x_lo = min(x_mins)
        x_hi = max(x_maxs)

        # Collect markers from defaults / user list, then from add_lines.
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
                # Sequence of REST_LINES_A keys: look up rest wavelengths
                # and apply the friendly display label where known.
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
    rest_frame : bool
        If ``True``, plot wavelengths in the rest frame by dividing by
        ``(1 + z)`` using the redshift stored in the spectrum.  Default
        ``False`` (observed frame).

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

    # Rest-frame scaling factor.
    zp1 = 1.0
    if rest_frame and spec.z is not None:
        zp1 = 1.0 + spec.z

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
            if is_abs:
                colour = "rgba(70,130,180,0.8)"  # steel blue for absorption
            elif is_broad:
                colour = "rgba(255,140,0,0.6)"
            else:
                colour = palette[i % len(palette)]
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
                ), row=1)

            # Smooth Gaussian curve.
            _add(go.Scatter(
                x=wave_fine, y=gauss_masked,
                mode="lines", name=f"{'[B] ' if is_broad else ''}{display_name}",
                line=dict(color=colour, width=1.5, dash=dash),
                hovertemplate=f"{display_name}<br>λ=%{{x:.1f}}<br>flux=%{{y:.4e}} {flux_label}<extra></extra>",
                showlegend=False,
            ), row=1)

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
            comp_col = palette[0]

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
                ), row=1)

            _add(go.Scatter(
                x=wave_fine_c_plot, y=prof_masked,
                mode="lines", name="Lyα",
                line=dict(color=comp_col, width=1.5, dash="solid"),
                hovertemplate="Lyα<br>λ=%{x:.1f}<br>flux=%{y:.4e} " + flux_label + "<extra></extra>",
                showlegend=False,
            ), row=1)

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

        # Line name annotations.
        for name, x_peak, y_peak, colour, is_abs in peak_info:
            display_name = name.replace("abs_", "").replace("_", " ")
            is_broad = "BROAD" in name
            # Full opacity for readable text.
            if "rgba" in colour:
                ann_colour = colour.rsplit(",", 1)[0] + ",1.0)"
            else:
                ann_colour = colour
            # Absorption labels below the trough; emission labels above the peak.
            fig.add_annotation(
                x=x_peak,
                y=y_peak,
                xref="x",
                yref="y",
                text=f"<b>{display_name}</b>" if is_broad else display_name,
                showarrow=False,
                yshift=-12 if is_abs else 10,
                font=dict(size=9, color=ann_colour),
                xanchor="center",
                yanchor="top" if is_abs else "bottom",
            )

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

    # Y limits — account for absorption troughs.
    model_peak = np.nanmax(model_total[show]) if np.any(show) else 1.0
    model_trough = np.nanmin(model_total[show]) if np.any(show) else 0.0
    cont_median = np.nanmedian(cont[show]) if np.any(show) else 0.0
    y_upper = cont_median + (model_peak - cont_median) * y_pad
    y_lower_cont = np.nanmin(cont[show]) * 1.1 if np.any(show) else -0.1
    y_lower = min(0.0, y_lower_cont, model_trough - abs(model_trough) * 0.15)

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
        # Clip residual y-axis to ±5× median error so noise spikes
        # don't dominate the panel height.
        med_err = float(np.nanmedian(err[show])) if np.any(show) else 1.0
        res_ylim = 5.0 * med_err
        fig.update_yaxes(title_text="Residual", range=[-res_ylim, res_ylim], row=2, col=1)
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
