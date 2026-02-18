# jwspecfit

JWST NIRSpec emission-line fitting with resolution-aware Gaussian models.

`jwspecfit` fits Gaussian emission-line profiles to 1-D extracted JWST NIRSpec
spectra, handling the wavelength-dependent spectral resolution of the prism,
medium-resolution, and high-resolution gratings.  It also supports stacked
spectra with user-specified or auto-detected resolving power.

## Installation

```bash
git clone https://github.com/raunaq-rai/jwspecfit.git
cd jwspecfit
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, numpy, scipy, astropy, matplotlib, lmfit, tqdm, plotly.

---

## Quick start

```python
import jwspecfit

# Load a FITS spectrum — grating and resolution are auto-detected from the header.
spec = jwspecfit.read_fits("data/borg-v4_prism-clear_1747_732.spec.fits", z=6.0)

# Fit all observable emission lines (bootstrap uncertainties by default).
result = jwspecfit.fit_lines(spec, z=6.0)

# Inspect per-line results.
for name, line in result.lines.items():
    if line.snr > 3:
        print(f"{name}: flux = {line.flux:.2e} ± {line.flux_err:.2e}, SNR = {line.snr:.1f}")

# Plot (matplotlib).
fig = jwspecfit.plot_fit(result)

# Interactive plot (plotly) — zoom and hover on individual components.
fig_i = jwspecfit.plot_fit_interactive(result)
fig_i.show()

# Save the fit for later replotting.
jwspecfit.save_result(result, "my_fit.npz")

# Export line measurements as a text table.
jwspecfit.export_lines_txt(result, "lines.txt")
```

---

## Core fitting

### `fit_lines()`

The main entry point.  Fits Gaussian emission lines to a spectrum with
continuum subtraction, parameter constraints, and bootstrap uncertainties.

```python
result = jwspecfit.fit_lines(
    spectrum,                    # Spectrum object
    z,                           # Source redshift
    grating=None,                # Grating name (auto-detected from spectrum if None)
    R=None,                      # Resolving power: float, callable R(λ), or None
    lines=None,                  # List of line names to fit (None = auto-detect)
    wave_range_A=None,           # (lo, hi) observed wavelength window in Å
    deg=2,                       # Continuum polynomial degree
    n_boot=200,                  # Bootstrap iterations (0 for analytic errors)
    clip_sigma=2.5,              # Sigma-clipping threshold for continuum
    save_path=None,              # Export line measurements to this text file
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spectrum` | `Spectrum` | — | Input spectrum object |
| `z` | `float` | — | Source redshift |
| `grating` | `str \| None` | `None` | Grating name (`"PRISM"`, `"G395M"`, etc.). Falls back to `spectrum.grating` |
| `R` | `float \| Callable \| None` | `None` | Resolving power. Overrides `grating`. Can be a constant or a callable `R(lam_um)` |
| `lines` | `list[str] \| None` | `None` | Lines to fit (keys of `REST_LINES_A`). If `None`, auto-detected from grating and wavelength coverage |
| `wave_range_A` | `tuple[float, float] \| None` | `None` | Restrict fitting to this observed wavelength window (Angstroms). Pixels outside are excluded from continuum and line fitting |
| `deg` | `int` | `2` | Polynomial degree for continuum fitting |
| `n_boot` | `int` | `200` | Number of bootstrap iterations. Set to 0 for fast analytic error estimates |
| `clip_sigma` | `float` | `2.5` | Sigma-clipping threshold for iterative continuum fitting |
| `save_path` | `str \| Path \| None` | `None` | If given, automatically export per-line measurements to this text file path after fitting |

Returns a `FitResult`.

### `FitResult`

Returned by `fit_lines()`.

| Field | Type | Description |
|-------|------|-------------|
| `lines` | `dict[str, LineResult]` | Per-line results, keyed by line name |
| `params` | `np.ndarray` | Full best-fit parameter vector `[A_0..A_n, mu_0..mu_n, sigma_0..sigma_n]` |
| `model_flux` | `np.ndarray` | Best-fit emission-line model (continuum-subtracted, uJy) |
| `continuum` | `np.ndarray` | Best-fit polynomial continuum (uJy) |
| `residuals` | `np.ndarray` | Fit residuals: data - continuum - model (uJy) |
| `chi2` | `float` | Reduced chi-squared of the fit |
| `spectrum` | `Spectrum` | The input spectrum |
| `line_names` | `list[str]` | Ordered line names matching the parameter vector |
| `constraints` | `ConstraintSet \| None` | Applied parameter constraints |
| `success` | `bool` | Whether the optimiser converged |

### `LineResult`

Per-line measurements stored in `FitResult.lines`.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Line name |
| `rest_wave_A` | `float` | Rest-frame wavelength (Angstroms) |
| `amplitude` | `float` | Best-fit Gaussian amplitude (area-normalised) |
| `centroid_A` | `float` | Best-fit observed centroid (Angstroms) |
| `sigma_A` | `float` | Best-fit observed Gaussian sigma (Angstroms) |
| `flux` | `float` | Integrated line flux (erg/s/cm2) |
| `flux_err` | `float` | Flux uncertainty (bootstrap or analytic) |
| `ew_A` | `float` | Rest-frame equivalent width (Angstroms) |
| `snr` | `float` | Signal-to-noise ratio (flux / flux_err) |

---

## Broad Balmer component detection

### `fit_with_broad()`

Compares narrow-only vs narrow+broad models using the Bayesian Information
Criterion (BIC).  All model variants are fit quickly without bootstrap first,
then only the winning model is re-fit with bootstrap uncertainties.

```python
result = jwspecfit.fit_with_broad(
    spectrum, z,
    grating=None,                # Grating name
    R=None,                      # Resolving power
    lines=None,                  # Narrow line list
    deg=2,                       # Continuum polynomial degree
    mode="auto",                 # "auto", "off", "broad1", "broad2", "both"
    n_boot=200,                  # Bootstrap iterations for the winning model
    snr_threshold=5.0,           # Minimum Ha SNR to attempt broad fitting
    bic_delta=6.0,               # ΔBIC threshold for accepting a more complex model
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"auto"` | `"auto"`: BIC-based selection. `"off"`: narrow-only. `"broad1"`: force intermediate broad. `"broad2"`: force very broad. `"both"`: force both broad components |
| `snr_threshold` | `float` | `5.0` | Minimum Ha SNR (from the quick narrow fit) required before attempting broad models |
| `bic_delta` | `float` | `6.0` | A more complex model is only accepted if ΔBIC >= this threshold |

The four models compared are:
- **narrow**: narrow lines only
- **broad1**: narrow + intermediate broad Balmer (sigma_v ~ 3x narrow)
- **broad2**: narrow + very broad Balmer (sigma_v ~ 7x narrow)
- **both**: narrow + both broad components

Broad Balmer components are added for Ha, Hbeta, Hdelta, and Hgamma.
NII kinematics are tied to OIII to prevent broad Ha from absorbing NII flux.

### `BroadFitResult`

| Field | Type | Description |
|-------|------|-------------|
| `best_fit` | `FitResult` | The selected best-fit result |
| `selected_model` | `str` | `"narrow"`, `"broad1"`, `"broad2"`, or `"both"` |
| `bic_narrow` | `float` | BIC for the narrow-only model |
| `bic_broad1` | `float` | BIC for narrow + BROAD1 (NaN if not attempted) |
| `bic_broad2` | `float` | BIC for narrow + BROAD2 |
| `bic_both` | `float` | BIC for narrow + BROAD1 + BROAD2 |
| `all_fits` | `dict[str, FitResult]` | All fitted model variants for manual inspection |

---

## Spectral resolution

Resolution is handled automatically from the FITS header, or can be specified
manually for stacked spectra.

```python
# Auto-detected from FITS header
result = jwspecfit.fit_lines(spec, z=6.0)

# User-specified constant R for a stacked spectrum
stack = jwspecfit.read_npz("stack.npz", z=6.0, R=150)
result = jwspecfit.fit_lines(stack, z=6.0)

# Auto-estimated R from pixel spacing (fallback when no grating or R given)
result = jwspecfit.fit_lines(spec_no_header, z=6.0)
```

### Resolution functions

| Function | Description |
|----------|-------------|
| `R_prism(lam_um)` | Polynomial R(lambda) for PRISM/CLEAR, clipped to [30, 300]. `R = 50 + 50*(lam-1) + 15*(lam-1)^2` |
| `R_grating(name)` | Constant R for named gratings: 1000 for medium (G140M/G235M/G395M), 2700 for high (G140H/G235H/G395H) |
| `resolve_R(lam_um, grating=None, R=None)` | Returns R(lambda) array from grating name, constant R, or callable. `R` overrides `grating` |
| `R_from_pixels(lam_um)` | Estimates R ~ lambda / (2*dlambda) from pixel spacing. Returns a callable `R(lam)`. Fallback only |
| `sigma_inst_A(lam_um, grating=None, R=None)` | Instrumental Gaussian sigma in Angstroms: `sigma = lam_A / (R * 2.3548)` |
| `sigma_inst_kms(lam_um, grating=None, R=None)` | Instrumental Gaussian sigma in km/s: `sigma_v = c / (R * 2.3548)` |

---

## Line fitting details

### Bin-averaged Gaussians

Profiles are integrated over each pixel bin using the error function,
avoiding sampling bias when lines are narrower than pixels (prism regime).

### Continuum fitting

Iterative sigma-clipped polynomial fit, masking +/-6 instrumental sigma
around known emission lines.

```python
from jwspecfit.continuum import fit_continuum

continuum = fit_continuum(
    wave_um, flux_ujy, err_ujy, z, line_names,
    grating=None,                # Grating name
    R=None,                      # Resolving power
    deg=2,                       # Polynomial degree
    clip_sigma=2.5,              # Sigma-clipping threshold
    n_iter=5,                    # Number of clipping iterations
    line_mask_nsigma=6.0,        # Instrumental sigma to mask around each line
)
```

### Parameter constraints

Applied automatically during fitting:

- **[NII] doublet ratio**: `A(6549) / A(6585) = 1 / 2.96` (Storey & Zeippen 2000), with tied kinematics (centroid and width)
- **Balmer width tying**: narrow Balmer lines (Ha, Hbeta, Hdelta, Hgamma) and NII_6585 have their Gaussian widths tied to OIII_5007 in velocity space (`sigma_line = sigma_OIII * lam_line / lam_OIII`)
- **Broad lines are unconstrained**: components with `_BROAD` suffix are not subject to width tying

### Centroid bounds

Line centroids are free within +/-20 sigma_inst of the expected observed
wavelength (or +/-6 pixel widths, whichever is larger).

### Optimisation

Uses `scipy.optimize.least_squares` with grating-specific bounds on
amplitudes (>= 0), centroids, and widths.  Pixel weighting:
`w = (median_dlam / dlam)^0.35`.

---

## Uncertainties

**Bootstrap (default, `n_boot=200`):** the spectrum is perturbed by its error
array 200 times and refit, giving the standard deviation of recovered fluxes
as the uncertainty.  A `tqdm` progress bar shows progress.

**Analytic (`n_boot=0`):** local noise RMS scaled by the effective line width.
Faster but less accurate.

```python
# Default: bootstrap
result = jwspecfit.fit_lines(spec, z=6.0)

# Fast analytic errors
result = jwspecfit.fit_lines(spec, z=6.0, n_boot=0)
```

---

## Lyman-alpha

Lya is modelled as a skewed Gaussian attenuated by the mean IGM
transmission from Inoue et al. (2014).

```python
from jwspecfit.lyman_alpha import igm_transmission, lya_model, skewed_gaussian_binned
```

| Function | Description |
|----------|-------------|
| `igm_transmission(wave_obs_A, z_source)` | Mean IGM transmission T(lambda) in [0, 1] using Inoue+2014. T=1 redward of Lya at the source redshift |
| `skewed_gaussian_binned(lam_left_A, lam_right_A, amplitude, mu_A, sigma_A, skew)` | Bin-averaged skewed Gaussian using `scipy.stats.skewnorm`. Positive skew = red tail |
| `lya_model(lam_left_A, lam_right_A, z, amplitude, mu_A, sigma_A, skew)` | Full Lya forward model: `skewed_gaussian * igm_transmission` |

---

## I/O

### Loading spectra

```python
# From JWST FITS file (SPEC1D HDU with wave/flux/err columns)
spec = jwspecfit.read_fits("spectrum.fits", z=6.0)

# From a stacked .npz file (keys: wave_angstrom, flux, err)
spec = jwspecfit.read_npz("stack.npz", z=6.0, R=150.0)

# From a dict of arrays
spec = jwspecfit.read_dict(
    {"wave": wave_um, "flux": flux_ujy, "err": err_ujy},
    z=6.0, R=100.0,
)
```

### `Spectrum` container

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wave_um` | `np.ndarray` | — | Observed wavelength in microns |
| `flux_ujy` | `np.ndarray` | — | Flux density in microJansky |
| `err_ujy` | `np.ndarray` | — | 1-sigma uncertainty in microJansky |
| `grating` | `str \| None` | `None` | Grating name (e.g. `"PRISM"`, `"G395M"`) |
| `z` | `float \| None` | `None` | Source redshift |
| `R` | `float \| None` | `None` | Spectral resolving power (overrides grating) |
| `meta` | `dict` | `{}` | Arbitrary metadata |

**Computed properties:**

| Property | Description |
|----------|-------------|
| `wave_A` | Wavelength in Angstroms |
| `n_pix` | Number of pixels |
| `wave_edges_A` | Pixel-edge wavelengths in Angstroms (length n_pix + 1) |
| `dlam_A` | Pixel widths in Angstroms |
| `flux_flam` | Flux density in erg/s/cm2/A |
| `err_flam` | Error in erg/s/cm2/A |

**Methods:** `mask_valid()` (boolean mask: finite flux, finite err, err > 0), `copy()`.

### Saving and exporting

```python
# Save full FitResult to .npz for later replotting
jwspecfit.save_result(result, "fit_result.npz")

# Reload without re-fitting
loaded = jwspecfit.load_result("fit_result.npz")
fig = jwspecfit.plot_fit(loaded)

# Export line measurements as a text table
# Columns: name, rest_wave_A, centroid_A, flux (erg/s/cm2), flux_err,
#          EW_A (rest-frame), sigma_v_kms, SNR_integrated, SNR_peak
jwspecfit.export_lines_txt(result, "lines.txt")

# Or export automatically during fitting via save_path:
result = jwspecfit.fit_lines(spec, z=6.0, save_path="lines.txt")

# Save the plot directly:
fig = jwspecfit.plot_fit(result, save_path="fit.pdf")
```

---

## Plotting

### `plot_fit()` — matplotlib

Publication-quality static plot with data (steps), smooth Gaussian
components, continuum, total model, and residuals.  Gaussian uncertainty
is shaded.  Broad components are hatched.

```python
fig = jwspecfit.plot_fit(
    result,
    wave_unit="A",               # "A" (Angstroms) or "um" (microns)
    flux_unit="fnu",             # "fnu" (µJy) or "flam" (erg/s/cm²/Å)
    show_residuals=True,         # Show residual panel below
    show_components=True,        # Show individual Gaussian components
    label_lines=True,            # Annotate line names
    y_pad=1.3,                   # Y-axis padding above tallest line peak
    save_path=None,              # Save figure to file (e.g. "fit.pdf")
)

# Plot in f_lambda units:
fig = jwspecfit.plot_fit(result, flux_unit="flam")
```

### `plot_fit_interactive()` — plotly

Interactive zoomable plot.  Click and drag to zoom into any spectral
region.  Hover to see the flux of each component at any wavelength.

```python
fig = jwspecfit.plot_fit_interactive(
    result,
    wave_unit="A",               # "A" (Angstroms) or "um" (microns)
    flux_unit="fnu",             # "fnu" (µJy) or "flam" (erg/s/cm²/Å)
    show_components=True,        # Show individual line components
    y_pad=1.3,                   # Y-axis padding above tallest line peak
)
fig.show()
```

---

## Line database

`REST_LINES_A` contains rest-frame vacuum wavelengths in Angstroms:

| Line | Wavelength (A) | Line | Wavelength (A) |
|------|---------------|------|---------------|
| Lya | 1215.67 | OIII_4363 | 4364.44 |
| NV_1 | 1238.82 | OIII_4959 | 4961.68 |
| NV_2 | 1242.80 | OIII_5007 | 5009.64 |
| NV_doublet | 1240.81 | NII_5756 | 5757.79 |
| NIV_1 | 1486.50 | HEI_5877 | 5878.88 |
| CIV_1 | 1548.19 | NII_6549 | 6551.67 |
| CIV_2 | 1550.77 | Ha | 6566.42 |
| CIV_doublet | 1549.48 | NII_6585 | 6587.09 |
| HEII_1640 | 1640.42 | SII_6718 | 6720.15 |
| OIII_1663 | 1663.48 | SII_6732 | 6734.53 |
| SiIII_1 | 1882.71 | | |
| SiIII_2 | 1892.03 | **Semi-forbidden** | |
| CIII] | 1908.73 | OIII]_2321 | 2322.41 |
| OII_3726 | 3727.09 | OIII]_2331 | 2332.02 |
| OII_3729 | 3729.88 | OII]_2471 | 2471.72 |
| OII_doublet | 3728.48 | FeII*_2396 | 2397.09 |
| HDELTA | 4104.05 | | |
| HGAMMA | 4342.90 | | |
| HBETA | 4864.04 | | |

Default line lists are selected by grating:
- **Prism**: merged doublets (OII_doublet, CIV_doublet) + Lya
- **Medium/High**: resolved doublets (OII_3726, OII_3729) + auroral lines

---

## Modules

| Module | Description |
|--------|-------------|
| `io` | `Spectrum` container, FITS/npz/dict readers, save/load/export |
| `lines` | Rest-frame line database (`REST_LINES_A`), line-list helpers |
| `resolution` | R(lambda) models, instrumental sigma, auto-detection |
| `continuum` | Polynomial continuum with iterative sigma-clipping |
| `models` | Bin-averaged Gaussian profiles via erf |
| `constraints` | [NII] ratio, Balmer-[OIII] width tying |
| `fitter` | Core `fit_lines()` engine |
| `broad` | Broad component fitting + BIC selection |
| `lyman_alpha` | Skewed Gaussian + Inoue+2014 IGM absorption |
| `plotting` | Static (matplotlib) and interactive (plotly) visualisation |

## Example notebooks

See `docs/notebooks/` for worked examples:

- **01_prism_fit.ipynb** — basic prism fitting, save/load, text export, plotly
- **02_grating_broad.ipynb** — G395M grating with BIC broad-line detection
- **03_stacked_spectrum.ipynb** — stacked spectrum with custom R, IGM demo

## Tests

```bash
pytest tests/ -v
```

64 tests covering I/O, resolution, models, fitting, broad components, and
Lyman-alpha.
