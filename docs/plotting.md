# Plotting & visualisation

Every fit, posterior, and spectrum produced by the suite can be visualised
with a small, consistent set of plotting helpers. They fall into three
families:

- **Static (matplotlib)** — publication-quality figures you save to PDF/PNG.
- **Interactive (plotly)** — zoom/pan/hover figures for exploration, with
  an optional stacked 2-D rectified-spectrum panel.
- **MCMC diagnostics** — corner plots, trace plots, and flux posteriors for
  [`jwspecmcmc`](user_guide/jwspecmcmc) results.

In addition, the [redshift](#redshift-scan-diagnostic) and
[DLA](#dla-fit-overlay) result objects carry their own `.plot()` convenience
methods.

```{contents}
:local:
:depth: 2
```

## At a glance

| Function | Package | Backend | Input | Use for |
|----------|---------|---------|-------|---------|
| {func}`~jwspecfit.plot_fit` | `jwspecfit` | matplotlib | `FitResult` | Publication figure of a completed fit (data + model + components + residuals) |
| {func}`~jwspecfit.plot_fit_interactive` | `jwspecfit` | plotly | `FitResult` | Interactive inspection of a fit; optional 2-D panel |
| {func}`~jwspecfit.plot_spectrum_interactive` | `jwspecfit` | plotly | `Spectrum` / path(s) | Look at one or more spectra *before* fitting; overplot lines |
| {func}`~jwspecfit.plot_2d_1d` | `jwspecfit` | matplotlib | `.spec.fits` path | Static stacked 2-D + 1-D figure straight from a file |
| {func}`~jwspecmcmc.plot_corner` | `jwspecmcmc` | matplotlib | `MCMCResult` | Posterior covariances / marginals |
| {func}`~jwspecmcmc.plot_traces` | `jwspecmcmc` | matplotlib | `MCMCResult` | Chain mixing / convergence (emcee chains) |
| {func}`~jwspecmcmc.plot_flux_posterior` | `jwspecmcmc` | matplotlib | `MCMCResult` | Flux posterior for one line |

```{tip}
An `MCMCResult` is accepted directly by {func}`~jwspecfit.plot_fit` and
{func}`~jwspecfit.plot_fit_interactive` — they auto-convert via
`to_fit_result()`. You do **not** need to convert manually to draw the fit.
```

---

## Static fit figures — `plot_fit`

The workhorse for papers. Given a `FitResult` (from
{func}`~jwspecfit.fit_lines`, {func}`~jwspecfit.fit_with_broad`, or an
`MCMCResult`), it draws the data, the total model, the continuum, each
Gaussian component, and a residual panel — with the y-axis scaled to the
tallest emission line rather than to noise spikes.

```python
import jwspecfit

result = jwspecfit.fit_lines("spectrum.fits", z=6.0)

# Quick look
fig = jwspecfit.plot_fit(result)

# Publication figure, saved straight to PDF
fig = jwspecfit.plot_fit(
    result,
    flux_unit="flam",          # erg/s/cm²/Å instead of µJy
    rest_frame=True,           # divide wavelengths by (1+z)
    show_components=True,      # filled Gaussians; broad lines hatched
    exclude_wave_A=[(13000, 13500)],  # mask a noisy detector gap
    save_path="fit.pdf",
)
```

Key parameters:

`flux_unit`
: `"fnu"` (µJy, default) or `"flam"` (erg/s/cm²/Å).

`wave_unit`
: `"A"` (Ångström, default) or `"um"` (microns).

`rest_frame`
: When `True`, divides wavelengths by `(1 + z)` using the spectrum's stored
  redshift. Axis labels switch to "Rest Wavelength".

`show_components`
: Draw each Gaussian as a filled curve. Broad Balmer components are drawn
  with hatching so they read clearly against the narrow lines.

`show_residuals`
: Toggle the lower residual panel (default `True`).

`y_pad`
: Multiplicative headroom above the tallest line peak (default `1.3`).

`exclude_wave_A`
: List of `(lo, hi)` ranges in Ångström to hide — useful for masking
  detector gaps or contaminated regions.

`save_path`
: Write the figure to disk (`.pdf`, `.png`, …). Returns the figure either way.

```{seealso}
Full parameter list: {func}`jwspecfit.plot_fit`.
```

---

## Static 2-D + 1-D — `plot_2d_1d`

Renders a stacked figure — the 2-D rectified spectrum on top, the 1-D
extraction below — directly from an msaexp-style `.spec.fits` file, with no
fit required. Ideal for a first look at the data or a context panel in a
figure.

```python
fig, (ax_2d, ax_1d) = jwspecfit.plot_2d_1d(
    "target.spec.fits",
    z=6.0,                     # places observed-frame line markers
    cmap="Blues",
    flux_scale=1000.0,         # µJy → nJy on the 1-D panel
    flux_label=r"$F_\nu$ [nJy]",
    add_lines={"Mg II 2796": 2796.352},  # extra label not in the defaults
)
fig.savefig("context.pdf", dpi=300)
```

It returns the figure **and** both axes, so you can annotate or restyle
afterwards (e.g. `ax_1d.set_xlim(...)`). The `lines` / `add_lines` arguments
take the same forms as the interactive helpers — see
[Marking emission lines](#marking-emission-lines).

```{seealso}
Full parameter list: {func}`jwspecfit.plot_2d_1d`.
```

---

## Interactive spectra — `plot_spectrum_interactive`

Open one or more 1-D spectra in plotly **before** fitting — zoom, pan, hover
read-outs, an optional 2-D panel, and overplotted line markers. Accepts a
`Spectrum`, a file path, or a list of either (to overplot several spectra).

```python
# Single file
fig = jwspecfit.plot_spectrum_interactive("spectrum.fits", z=6.0)

# A rest-frame stack saved as .npz
fig = jwspecfit.plot_spectrum_interactive("stack.npz", rest_frame=True)

# An in-memory Spectrum, in F_lambda units
fig = jwspecfit.plot_spectrum_interactive(spec, flux_unit="flam")

# Overplot several spectra with custom legend labels
fig = jwspecfit.plot_spectrum_interactive(
    ["g140m.fits", "g235m.fits"],
    z=6.0,
    labels=["G140M", "G235M"],
)
fig.show()
```

The 2-D rectified panel appears automatically (`show_2d="auto"`) whenever a
**single** spectrum is supplied and its `sci_2d` array is populated (e.g.
read from a `.spec.fits` file). Set `show_2d=False` to force 1-D only.

Any extra keyword arguments are forwarded to the file reader, so FITS column
overrides work inline:

```python
fig = jwspecfit.plot_spectrum_interactive(
    "custom.fits", z=6.0, hdu="EXTRACT1D", wave_col="WAVELENGTH",
)
```

```{seealso}
Full parameter list: {func}`jwspecfit.plot_spectrum_interactive`.
```

---

## Interactive fits — `plot_fit_interactive`

The plotly counterpart to `plot_fit`. Same data/model/components/residual
layout, but zoomable and with an optional stacked 2-D panel above the fit.

```python
fig = jwspecfit.plot_fit_interactive(result)
fig.show()

# Rest-frame, with a curated set of line markers added on top
fig = jwspecfit.plot_fit_interactive(
    result, rest_frame=True, lines=None,  # None opts in to default markers
)
```

By default (`lines=False`) **no** curated markers are drawn, because every
fitted line is already labelled at its peak above the model. Pass
`lines=None` for the package-default marker set, or an explicit list of line
keys to mark only those.

```{seealso}
Full parameter list: {func}`jwspecfit.plot_fit_interactive`.
```

---

(marking-emission-lines)=
## Marking emission lines

The interactive helpers and `plot_2d_1d` share a common vocabulary for
emission-line markers via the `lines` and `add_lines` arguments.

`lines`
: Controls the curated default set. `None` draws the package defaults,
  `False` draws none, and a list of keys from
  {data}`jwspecfit.lines.REST_LINES_A` marks only those.

`add_lines`
: Extra markers layered on top, in **two** accepted forms:

  - **list of names** from `REST_LINES_A` — wavelengths looked up
    automatically, e.g. `add_lines=["H8", "HEPSILON", "FeII_2382"]`.
  - **dict** of `{label: rest_wavelength_A}` for free-form labels at an
    explicit rest wavelength in Ångström, e.g.
    `add_lines={"Mg II 2796": 2796.352}`.

```python
# What line names are available?
jwspecfit.show_lines()
```

Markers are redshifted by `(1 + z)` (the effective `z` comes from the `z`
argument, else the spectrum's own `spec.z`); in rest-frame mode they sit at
the rest wavelengths.

---

## MCMC diagnostics

These live in `jwspecmcmc` and take an `MCMCResult`. See the
[`jwspecmcmc` guide](user_guide/jwspecmcmc) for how to produce one.

### Corner plots — `plot_corner`

Posterior marginals and pairwise covariances.

```python
import jwspecmcmc

result = jwspecmcmc.fit_lines("spectrum.fits", z=6.0)

# All free parameters
fig = jwspecmcmc.plot_corner(result)

# A focused subset
fig = jwspecmcmc.plot_corner(
    result,
    params=["A_OIII_5007", "A_HBETA", "sigma_Ha"],
    quantiles=[0.16, 0.5, 0.84],   # default: 16/50/84th percentiles
)
```

Extra keyword arguments are forwarded to `corner.corner`.

### Trace plots — `plot_traces`

Per-walker chains, for eyeballing mixing and convergence.

```python
fig = jwspecmcmc.plot_traces(result)
fig = jwspecmcmc.plot_traces(result, params=["A_HBETA"])
```

```{note}
`plot_traces` needs per-chain samples, so it works for **emcee** results.
A nautilus/NUTS result without stored chains raises `ValueError` — judge
those runs with R̂ and ESS from `result.summary()` and with `plot_corner`.
```

### Flux posteriors — `plot_flux_posterior`

Histogram of the integrated-flux posterior for a single line — the honest
picture of a flux measurement and its (often asymmetric) uncertainty.

```python
ax = jwspecmcmc.plot_flux_posterior(result, "OIII_5007", bins=50)

# Compose several onto shared axes
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for line in ("OIII_5007", "HBETA"):
    jwspecmcmc.plot_flux_posterior(result, line, ax=ax)
```

```{seealso}
Full parameter lists: {func}`jwspecmcmc.plot_corner`,
{func}`jwspecmcmc.plot_traces`, {func}`jwspecmcmc.plot_flux_posterior`.
```

---

## Result-object convenience plots

(redshift-scan-diagnostic)=
### Redshift scan — `RedshiftResult.plot()`

{func}`~jwspecfit.fit_redshift` returns a `RedshiftResult` whose `.plot()`
draws a two-panel plotly diagnostic: Δχ²(z) with 1/2/3σ thresholds and
candidate peaks on top, and the spectrum at the best redshift below.

```python
zres = jwspecfit.fit_redshift("spectrum.fits")
zres.plot().show()
```

(dla-fit-overlay)=
### DLA fit — `DLAResult.plot()`

{func}`~jwspecfit.fit_NHI` returns a `DLAResult` whose `.plot()` overlays the
fitted Lyα/DLA profile on the data with a residual panel.

```python
result = jwspecfit.fit_NHI(spec, z=6.5, R=spec.R, n_live=400, seed=42)
print(result.summary())
result.plot(flux_unit="flam")
```

---

## Choosing a backend

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - 
  - Static (matplotlib)
  - Interactive (plotly)
* - Best for
  - Paper figures, batch scripts
  - Exploration, picking masks, QA
* - Output
  - PDF / PNG via `save_path` / `savefig`
  - `fig.show()` in a notebook/browser
* - 2-D panel
  - `plot_2d_1d`
  - `show_2d="auto"` in both interactive helpers
* - Returns
  - `Figure` (and axes for `plot_2d_1d`)
  - `plotly.graph_objects.Figure`
```

A common workflow: explore with `plot_spectrum_interactive` to choose
`exclude_wave_A` masks and confirm the redshift, fit, then render the final
figure with `plot_fit(..., save_path=...)`.
