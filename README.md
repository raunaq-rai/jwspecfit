# jwspecfit

**Resolution-aware emission-line fitting for JWST NIRSpec spectra.**

jwspecfit fits Gaussian emission-line profiles to 1-D extracted JWST NIRSpec
spectra, properly accounting for the wavelength-dependent resolving power of
the prism (R ~ 30-300), fixed-resolution gratings (R ~ 1000-2700), and
user-supplied resolving powers for stacked/combined spectra.

Three integrated sub-packages cover the full analysis chain from raw line
measurements to chemical abundances:

| Package | Purpose |
|---------|---------|
| **`jwspecfit`** | Least-squares Gaussian fitting with bootstrap uncertainties |
| **`jwspecmcmc`** | Bayesian MCMC fitting (NUTS, emcee, nautilus) with full posteriors |
| **`jwspecabund`** | Chemical abundances: direct T_e, Bayesian forward model, strong-line calibrations |

---

## Installation

```bash
git clone https://github.com/raunaq-rai/jwspecfit.git
cd jwspecfit
pip install -e ".[dev]"
```

Optional extras:

```bash
# NUTS sampler (recommended) — requires JAX + NumPyro
pip install -e ".[nuts]"

# emcee + nautilus samplers
pip install -e ".[mcmc]"

# Chemical abundances (installs PyNEB)
pip install -e ".[abund]"

# Everything
pip install -e ".[dev,nuts,mcmc,abund]"
```

**Requirements:** Python >= 3.10, numpy, scipy, astropy, matplotlib, tqdm, joblib, plotly.

---

## Quick start

```python
import jwspecfit

# Load a JWST x1d FITS file (grating and R read from header)
spec = jwspecfit.read_fits("spectrum.fits", z=6.0)

# Fit all observable emission lines with bootstrap uncertainties
result = jwspecfit.fit_lines(spec, z=6.0)

# Print detections
for name, line in result.lines.items():
    if line.snr > 3:
        print(f"{name}: flux={line.flux:.2e} +/- {line.flux_err:.2e}, SNR={line.snr:.1f}")

# Plot
jwspecfit.plot_fit(result, save_path="fit.pdf")
```

---

## 1. jwspecfit -- Least-squares fitting

### Loading spectra

```python
# From a JWST x1d FITS file
spec = jwspecfit.read_fits("spectrum.fits", z=6.0)

# From numpy arrays (wavelength in microns, flux/error in micro-Jansky)
spec = jwspecfit.read_dict(
    {"wave": wave_um, "flux": flux_ujy, "err": err_ujy},
    z=6.0,
)

# From a stacked .npz file with custom resolving power
spec = jwspecfit.read_npz("stack.npz", z=2.0, R=1000)
```

### Fitting lines

```python
result = jwspecfit.fit_lines(spec, z=6.0)
```

This automatically:
1. Identifies observable lines from the grating wavelength coverage and redshift
2. Subtracts a polynomial continuum (with iterative sigma-clipping and line masking)
3. Fits resolution-aware, bin-averaged Gaussians to all lines simultaneously
4. Estimates uncertainties via bootstrap resampling (1000 iterations by default)
5. Optionally detects broad Balmer components via BIC model comparison (`mode="auto"`)

Common options:

```python
result = jwspecfit.fit_lines(
    spec, z=6.0,
    lines=["OIII_5007", "OIII_4959", "HBETA"],  # specific lines only
    wave_range_A=(35000, 50000),                  # restrict wavelength window
    n_boot=500,                                   # fewer bootstrap iterations
    mode="auto",                                  # broad Balmer detection (auto/off)
)
```

### Line results

Each `LineResult` contains:

| Field | Description |
|-------|-------------|
| `flux` | Integrated line flux (erg/s/cm^2) |
| `flux_err` | 1-sigma uncertainty from bootstrap |
| `snr` | Signal-to-noise ratio |
| `ew_A` | Rest-frame equivalent width (Angstroms) |
| `amplitude` | Peak amplitude |
| `centroid_A` | Observed centroid wavelength (Angstroms) |
| `sigma_A` | Gaussian sigma (Angstroms) |

### Broad Balmer components

jwspecfit tests four models for the Balmer lines and selects via BIC:
- Narrow only
- Narrow + intermediate broad (FWHM ~ 500-2000 km/s)
- Narrow + very broad (FWHM ~ 2000-5000 km/s)
- Narrow + both broad components

```python
result = jwspecfit.fit_with_broad(spec, z=2.0, n_boot_bic=100)
print(result.selected_model)  # "narrow", "broad1", "broad2", or "both"
```

### Absorption lines

Lines with the `abs_` prefix are fitted as negative Gaussians:

```python
uv_lines = [
    "NV_1", "NV_2", "CIV_1", "CIV_2", "HEII_1640",
    "CIII]_1907", "CIII]",
    "abs_SiII1260", "abs_CII1334", "abs_SiIV1394", "abs_SiIV1403",
]
result = jwspecfit.fit_lines(spec, z=6.0, lines=uv_lines)
```

### Plotting

```python
# Static (matplotlib)
fig = jwspecfit.plot_fit(result, save_path="fit.pdf")

# Interactive (plotly) -- zoomable with hover info
fig = jwspecfit.plot_fit_interactive(result)
fig.show()
```

### Save and reload

```python
jwspecfit.save_result(result, "fit.npz")
loaded = jwspecfit.load_result("fit.npz")

# Export a text table of line measurements
jwspecfit.export_lines_txt(result, "lines.txt")
```

---

## 2. jwspecmcmc -- Bayesian MCMC fitting

jwspecmcmc replaces bootstrap resampling with full Bayesian posterior sampling,
giving asymmetric uncertainties, parameter correlations, and flux ratio
posteriors.

### Recommended sampler: NUTS

The **NUTS** (No-U-Turn Sampler) via NumPyro is recommended for most use
cases. It is a Hamiltonian Monte Carlo method that efficiently explores
high-dimensional, correlated posteriors with adaptive step sizes and tree
depth:

```python
import jwspecmcmc

result = jwspecmcmc.fit_lines(
    spec, z=6.0,
    sampler="nuts",
    n_warmup=500,           # warm-up / adaptation steps
    n_samples_nuts=2000,    # posterior samples per chain
    n_chains=6,             # parallel chains
    target_accept_prob=0.8, # NUTS acceptance probability
)
```

NUTS requires JAX and NumPyro (`pip install -e ".[nuts]"`).

### Alternative samplers

**emcee** -- affine-invariant ensemble sampler:

```python
result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="emcee", n_steps=2000)
```

**nautilus** -- importance nested sampling (useful for evidence estimation):

```python
result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="nautilus", n_live=2000)
```

### MCMC results

`MCMCLineResult` provides asymmetric uncertainties and full posteriors:

```python
line = result.lines["OIII_5007"]
print(line.flux)                 # median flux
print(line.flux_err)             # (lower, upper) 68% CI half-widths
print(line.flux_posterior.shape) # (n_samples,) array

# Flux ratio posterior (sample-by-sample division)
r32 = result.flux_ratio_posterior("OIII_5007", "HBETA")
```

### Custom priors

Default priors are uniform within parameter bounds. Override per-parameter:

```python
from jwspecmcmc import GaussianPrior, LogUniformPrior

result = jwspecmcmc.fit_lines(
    spec, z=6.0, sampler="nuts",
    prior_overrides={
        "A_OIII_5007": GaussianPrior(mean=8e-18, std=1e-18, lo=0, hi=1e-15),
    },
)
```

Available prior classes: `UniformPrior`, `GaussianPrior`, `LogUniformPrior`.

### Convergence diagnostics

```python
print(result.convergence)  # R-hat and ESS per parameter
```

### Plotting

```python
jwspecmcmc.plot_corner(result, params=["A_OIII_5007", "A_HBETA"])
jwspecmcmc.plot_traces(result)
jwspecmcmc.plot_flux_posterior(result, "OIII_5007")
```

### Save/load (HDF5)

```python
jwspecmcmc.save_mcmc_result(result, "mcmc.h5")
loaded = jwspecmcmc.load_mcmc_result("mcmc.h5")

# Convert to FitResult for jwspecfit plotting
fit_result = result.to_fit_result()
jwspecfit.plot_fit(fit_result)
```

---

## 3. jwspecabund -- Chemical abundances

jwspecabund derives element abundances from emission-line fluxes measured by
jwspecfit or jwspecmcmc. It accepts either a `FitResult` or `MCMCResult`.

### Basic usage

```python
import jwspecabund

abund = jwspecabund.compute_abundances(result, z=6.0)
print(abund.summary())
```

### How it works

`compute_abundances` runs the following steps in order:

**Step 1: Dust correction.**
A_V is derived from the Balmer decrement (Hgamma/Hbeta, intrinsic ratio
0.468). All line fluxes are corrected using the Salim+18 attenuation curve
(default) or Cardelli+89 extinction curve. When multiple Balmer lines are
available (Hgamma through H10), a multi-line weighted average is used.

```python
# Override dust settings
abund = jwspecabund.compute_abundances(result, z=6.0,
    dust_law="cardelli",  # "salim" (default) or "cardelli"
    Av=0.5,               # fix A_V manually
    dust_correct=False,    # skip dust correction entirely
)
```

**Step 2: Electron density.**
Density is measured from three ionisation zones when the relevant lines are
available:

| Zone | Diagnostic | Fallback |
|------|-----------|----------|
| Low-ionisation | [SII] 6718/6732 | [OII] 3726/3729, then 300 cm^-3 |
| Mid-ionisation | CIII] 1907/1909 | -- |
| High-ionisation | NIV] 1483/1486 | n_e(mid), then n_e(low) |

**Step 3: Method selection.**
The method is chosen automatically based on available lines, or can be forced
with `method=`:

| Method | When used | What it does |
|--------|-----------|--------------|
| `"direct"` | [OIII] 4363 detected (SNR >= 3) | Electron temperature from [OIII] 4363/(4959+5007); ionic abundances from PyNEB emissivities; ICFs for total abundances |
| `"strong_line"` | No auroral line detected | Simultaneous polynomial calibrations from Sanders+25 using O3, O2, R23, O32 ratios |
| `"forward"` | Explicitly requested | Bayesian forward model (Cullen+25) sampling physical parameters to match observed line ratios |

**Step 4 (direct method): Electron temperature.**
T_e(O2+) is measured from the [OIII] auroral-to-nebular ratio. T_e for the
low-ionisation zone is derived via a T_e-T_e relation:
- `"desi"` (default): DESI DR2 calibration
- `"classical"`: Garnett (1992)

**Step 5 (direct method): Ionic and total abundances.**
PyNEB computes ionic abundance ratios (O+/H+, O++/H+, N+/H+, C++/H+, etc.)
from the dust-corrected fluxes, T_e, and n_e. Ionisation correction factors
(Martinez+25 for N/O; Izotov+06 for S, Ne, Ar) convert ionic ratios to total
element abundances: O/H, N/O, C/O, S/O, Ne/O, Ar/O.

### Key options

```python
abund = jwspecabund.compute_abundances(
    result, z=6.0,
    method=None,            # None (auto), "direct", "strong_line", "forward"
    dust_law="salim",       # "salim" or "cardelli"
    Av=None,                # None = derive from Balmer decrement
    Te_relation="desi",     # "desi" or "classical"
    n_mc=1000,              # MC iterations for error propagation
)
```

### Result fields

| Field | Description |
|-------|-------------|
| `OH` | 12 + log(O/H) |
| `OH_err` | Uncertainty (symmetric or asymmetric) |
| `NO` | log(N/O) |
| `CO` | log(C/O) |
| `SO`, `NeO`, `ArO` | Other element ratios |
| `Te_high`, `Te_low` | Electron temperatures (K) |
| `ne`, `ne_low`, `ne_mid`, `ne_high` | Electron densities (cm^-3) |
| `Av` | Dust attenuation |
| `ionic` | Dict of ionic abundance ratios (O+/H+, O++/H+, ...) |
| `OH_posterior`, `NO_posterior` | Full posterior arrays (MC/MCMC) |
| `method` | Method used (`"direct"`, `"strong_line"`, or `"forward"`) |

---

## Physical model details

### Resolution-aware Gaussians

Line profiles are evaluated as bin-averaged Gaussians using the error function
over each pixel's wavelength bin edges. This avoids the sampling bias that
arises from evaluating a Gaussian at pixel centres, which matters for the
prism where lines can be narrower than a pixel.

### Automatic parameter constraints

The following constraints are applied automatically during fitting:

- **[NII] doublet**: flux ratio 6549/6585 fixed to 1/2.96 (theoretical)
- **Balmer-[OIII] width tying**: narrow Balmer line widths share the [OIII] 5007 velocity dispersion
- **[NII] kinematics**: centroid tied to [OIII] to prevent broad Ha from absorbing [NII] flux
- **UV doublet ratios**: CIV, NV, NIII, OIII] amplitude ratios fixed at low-density limits
- **Centroid offsets**: limited to +/-500 km/s (configurable)

### Dust correction

Two attenuation/extinction curves are available:
- **Salim+18** (default): attenuation curve appropriate for star-forming galaxies
- **Cardelli+89**: Milky Way extinction curve

A_V is derived from the Balmer decrement. When Halpha is unavailable (e.g. at
high redshift), the Hgamma/Hbeta ratio is used (intrinsic ratio 0.468). A
multi-Balmer average (Hgamma through H10) weights by SNR when multiple lines
are available.

---

## Example notebooks

Worked examples in [`docs/notebooks/`](docs/notebooks/):

| Notebook | Description |
|----------|-------------|
| `01_prism_fit` | Prism fitting, save/load, plotting |
| `02_grating_broad` | G395M grating with broad Balmer detection |
| `03_stacked_spectrum` | Stacked spectrum with custom R |
| `04_mcmc_prism` | MCMC fitting with emcee |
| `05_mcmc_grating` | MCMC fitting for grating spectra |
| `06_mcmc_stack` | MCMC fitting for stacked spectra |
| `07_abundances` | Chemical abundances: direct, forward, strong-line |
| `08_nitrogen` | Nitrogen abundance diagnostics |
| `09_uv_abundances` | UV line fitting with absorption lines |

---

## Further documentation

- **Full API reference**: [`docs/api.md`](docs/api.md)
- **Abundance methodology**: [`docs/abundance_methodology.md`](docs/abundance_methodology.md)
- **References**: [`docs/references.md`](docs/references.md)

---

## Tests

```bash
pytest tests/ -v
```

## Licence

MIT -- see [LICENCE](LICENCE).
