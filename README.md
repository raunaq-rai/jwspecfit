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

## Quick start

```python
import jwspecfit

# Load a FITS spectrum — grating and resolution are auto-detected from the header.
spec = jwspecfit.read_fits("data/borg-v4_prism-clear_1747_732.spec.fits", z=6.0)

# Fit all observable emission lines.
result = jwspecfit.fit_lines(spec, z=6.0)

# Inspect results.
for name, line in result.lines.items():
    if line.snr > 3:
        print(f"{name}: flux = {line.flux:.2e}, SNR = {line.snr:.1f}")

# Plot.
fig = jwspecfit.plot_fit(result)
fig.savefig("fit.pdf")
```

## Features

### Spectral resolution handling

The resolving power R(λ) is determined automatically from the FITS header
grating keyword.  For the prism, a wavelength-dependent polynomial model
is used; for gratings, a constant R.  You can also pass a numeric `R` for
stacked spectra, or omit both to let the code estimate R from the pixel
spacing:

```python
# Auto-detected from FITS header (PRISM, G395M, etc.)
result = jwspecfit.fit_lines(spec, z=6.0)

# User-specified R for a stacked spectrum
stack = jwspecfit.read_npz("stack.npz", z=6.0, R=150)
result = jwspecfit.fit_lines(stack, z=6.0)

# Auto-estimated R from pixel spacing (when no grating or R given)
result = jwspecfit.fit_lines(spec_no_header, z=6.0)
```

### Line fitting

- **Bin-averaged Gaussians**: profiles are integrated over each pixel bin
  using the error function, avoiding sampling bias when lines are narrower
  than pixels (prism regime).
- **Continuum**: iterative σ-clipped polynomial, masking ±6σ around known
  lines.
- **Constraints**: [NII] 6549/6585 flux ratio fixed to 1/2.96 with tied
  kinematics; narrow Balmer and [NII] widths tied to [OIII] 5007 in
  velocity space.
- **Optimisation**: `scipy.optimize.least_squares` with grating-specific
  bounds on centroids, widths, and amplitudes.

### Broad Balmer components

Broad Hα (and Hβ) components are detected via BIC model selection:

```python
result = jwspecfit.fit_with_broad(spec, z=3.5, mode="auto")
print(result.selected_model)  # "narrow", "broad1", "broad2", or "both"
```

Four models are compared (narrow-only, +intermediate broad, +very broad,
+both), and the simplest model is preferred unless ΔBIC ≥ 6.

### Lyman-alpha

Lyα is modelled as a skewed Gaussian attenuated by the mean IGM
transmission from Inoue et al. (2014):

```python
from jwspecfit.lyman_alpha import igm_transmission
T = igm_transmission(wave_obs_A, z_source=7.0)
```

### Uncertainties

Two methods are available:

- **Analytic** (default, `n_boot=0`): estimates flux errors from the local
  noise RMS around each line, scaled by the effective number of resolution
  elements.
- **Bootstrap** (`n_boot=200`): perturbs the spectrum by its error array
  200 times and refits, returning the standard deviation of recovered
  fluxes.  More robust but slower (~200× the single-fit time).

```python
result = jwspecfit.fit_lines(spec, z=6.0, n_boot=200)
for name, line in result.lines.items():
    print(f"{name}: flux = {line.flux:.2e} ± {line.flux_err:.2e}")
```

## Modules

| Module | Description |
|--------|-------------|
| `io` | `Spectrum` container, FITS / npz / dict readers |
| `lines` | Rest-frame line database (`REST_LINES_A`), line-list helpers |
| `resolution` | R(λ) models, instrumental σ, auto-detection |
| `continuum` | Polynomial continuum with iterative σ-clipping |
| `models` | Bin-averaged Gaussian profiles via erf |
| `constraints` | [NII] ratio, Balmer–[OIII] width tying |
| `fitter` | Core `fit_lines()` engine |
| `broad` | Broad component fitting + BIC selection |
| `lyman_alpha` | Skewed Gaussian + Inoue+2014 IGM absorption |
| `plotting` | Publication-quality fit visualisation |

## Example notebooks

See `docs/notebooks/` for worked examples:

- **01_prism_fit.ipynb** — fitting a prism spectrum with automatic resolution
- **02_grating_broad.ipynb** — grating spectrum with broad-line detection
- **03_stacked_spectrum.ipynb** — fitting a stacked spectrum with custom R

## Data

The `data/` directory contains test spectra:

- `borg-v4_prism-clear_1747_732.spec.fits` — NIRSpec PRISM spectrum
- `excels-uds04-v4_g395m-f290lp_3543_63107.spec.fits` — G395M grating
- `stark-rxcj2248-v4_g395m-f290lp_2478_3.spec.fits` — G395M grating
- `stack_all_Muv19_21_DustCorrected.npz` — stacked rest-frame spectrum
- `inoue2014_table2.txt` — IGM absorption coefficients

## Tests

```bash
pytest tests/ -v
```

## Requirements

Python ≥ 3.10, numpy, scipy, astropy, matplotlib, lmfit.
