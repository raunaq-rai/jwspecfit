# jwspecfit

**Resolution-aware emission-line fitting for JWST NIRSpec spectra.**

`jwspecfit` fits Gaussian emission-line profiles to 1-D extracted JWST NIRSpec
spectra.  It handles the strongly wavelength-dependent resolution of the prism
(R ~ 30-300), the fixed resolution of medium- and high-resolution gratings,
and user-supplied resolving powers for stacked spectra.

Key capabilities:

- **Automatic line detection** — selects observable lines based on redshift,
  grating, and wavelength coverage; merged doublets for the prism,
  resolved components for gratings.
- **Resolution-aware profiles** — Gaussians are bin-averaged via the error
  function, avoiding sampling bias when lines are narrower than pixels.
- **Continuum subtraction** — iterative sigma-clipped polynomial fit with
  emission-line masking and Lyman-break exclusion.
- **Bootstrap uncertainties** — perturb-and-refit approach (default 1000
  iterations) for robust flux, EW, and SNR errors.
- **Broad Balmer detection** — BIC-based model selection with optional
  bootstrapped BIC comparison for robust narrow vs. broad discrimination.
- **Lyman-alpha modelling** — skewed Gaussian attenuated by mean IGM
  transmission (Inoue et al. 2014).
- **Publication-quality plots** — static (matplotlib) and interactive
  (plotly) visualisation with residuals, smooth Gaussian components, and
  wavelength exclusion regions.
- **I/O** — read FITS, NPZ, and dict spectra; save/load full fit results;
  export line measurements as text tables.

---

## Installation

```bash
git clone https://github.com/raunaq-rai/jwspecfit.git
cd jwspecfit
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.10, numpy, scipy, astropy, matplotlib, lmfit, tqdm, joblib, plotly.

---

## Quick start

```python
import jwspecfit

# 1. Load a FITS spectrum — grating and resolution are auto-detected.
spec = jwspecfit.read_fits("spectrum.fits", z=6.0)

# 2. Fit all observable emission lines with bootstrap uncertainties.
result = jwspecfit.fit_lines(spec, z=6.0)

# 3. Inspect per-line results.
for name, line in result.lines.items():
    if line.snr > 3:
        print(f"{name}: flux={line.flux:.2e} ± {line.flux_err:.2e}, SNR={line.snr:.1f}")

# 4. Plot.
fig = jwspecfit.plot_fit(result)                   # matplotlib
fig_i = jwspecfit.plot_fit_interactive(result)      # plotly (zoomable)
fig_i.show()

# 5. Save and export.
jwspecfit.save_result(result, "fit.npz")            # reload later without re-fitting
jwspecfit.export_lines_txt(result, "lines.txt")     # text table of measurements
```

---

## Loading spectra

```python
# JWST FITS file (SPEC1D HDU with wave/flux/err columns)
spec = jwspecfit.read_fits("spectrum.fits", z=6.0)

# Stacked .npz file (keys: wave_angstrom, flux, err)
spec = jwspecfit.read_npz("stack.npz", z=6.0)

# From numpy arrays
spec = jwspecfit.read_dict(
    {"wave": wave_um, "flux": flux_ujy, "err": err_ujy},
    z=6.0,
)
```

The `Spectrum` container holds wavelength (µm), flux (µJy), uncertainty (µJy),
and metadata.  Properties include `wave_A`, `dlam_A`, `wave_edges_A`,
`flux_flam`, `err_flam`, and `mask_valid()`.

### Resolving power (R)

R is resolved automatically — you almost never need to specify it:

- **FITS spectra**: grating is read from the header; R is looked up from
  the grating name (PRISM → R(λ), G395M → 1000, etc.).
- **No grating and no R**: `fit_lines()` automatically calls
  `R_from_pixels()` to estimate R ≈ λ / (2Δλ) from the pixel spacing.
- **Manual override**: pass `R=` (float or callable) to `read_dict()`,
  `read_npz()`, or set `spec.R = ...` after loading if you know the
  true resolving power.

### Loading non-standard formats

`jwspecfit` internally requires wavelength in **microns (µm)** and
flux / error in **micro-Jansky (µJy)**.  If your data uses different
units or file formats, convert before passing to `read_dict()` or
constructing a `Spectrum` directly.

**Wavelength conversions:**

```python
from jwspecfit.io import Spectrum

# Angstroms → microns
wave_um = wave_angstrom * 1e-4

# Nanometres → microns
wave_um = wave_nm * 1e-3
```

**Flux conversions:**

```python
# Jansky → µJy
flux_ujy = flux_jy * 1e6

# erg/s/cm²/Å (F_λ) → µJy
#   F_ν = F_λ × λ² / c  (CGS),  then → µJy
c_cgs = 2.99792458e18   # Å/s
flux_ujy = flux_flam * (wave_angstrom ** 2) / c_cgs * 1e29
err_ujy  = err_flam  * (wave_angstrom ** 2) / c_cgs * 1e29

# erg/s/cm²/Hz (F_ν in CGS) → µJy
flux_ujy = flux_fnu * 1e29
```

**Loading from a CSV or ASCII table:**

```python
import numpy as np

data = np.genfromtxt("spectrum.csv", delimiter=",", names=True)

# Suppose columns are: wavelength_A, flux_flam, err_flam
wave_um = data["wavelength_A"] * 1e-4
c_cgs = 2.99792458e18
flux_ujy = data["flux_flam"] * (data["wavelength_A"] ** 2) / c_cgs * 1e29
err_ujy  = data["err_flam"]  * (data["wavelength_A"] ** 2) / c_cgs * 1e29

spec = jwspecfit.read_dict(
    {"wave": wave_um, "flux": flux_ujy, "err": err_ujy},
    z=2.5,
)
# R will be estimated automatically from pixel spacing when fit_lines() is called.
```

**Loading from a non-standard FITS file:**

```python
from astropy.io import fits

with fits.open("other_pipeline.fits") as hdul:
    tbl = hdul[1].data
    wave_um = tbl["WAVELENGTH"] * 1e-4   # e.g. Å → µm
    flux_ujy = tbl["FLUX"] * 1e6         # e.g. Jy → µJy
    err_ujy  = tbl["ERROR"] * 1e6

# If the grating is known, pass it so R is looked up automatically.
# Otherwise, omit it and R will be estimated from pixel spacing.
spec = jwspecfit.read_dict(
    {"wave": wave_um, "flux": flux_ujy, "err": err_ujy},
    z=3.0, grating="G395M",
)
```

**Building a `Spectrum` directly (maximum control):**

```python
from jwspecfit.io import Spectrum

spec = Spectrum(
    wave_um=wave_um,
    flux_ujy=flux_ujy,
    err_ujy=err_ujy,
    grating=None,       # or "PRISM", "G395M", etc.
    z=4.5,
    # R is optional — if omitted, fit_lines() estimates it from pixel spacing.
    # Set it explicitly only if you know the true resolving power:
    # R=150.0,          # constant R
    # R=R_callable,     # or a callable R(lam_um) → array
    meta={"source": "my_pipeline", "target": "GN-z11"},
)
```

**Summary of expected units:**

| Field | Unit | Notes |
|-------|------|-------|
| `wave_um` | µm | Observed-frame wavelength |
| `flux_ujy` | µJy | f_ν flux density |
| `err_ujy` | µJy | 1σ uncertainty (same units as flux) |
| `z` | dimensionless | Source redshift (0.0 for rest-frame stacks) |
| `grating` | str or None | If set, R is looked up automatically |
| `R` | dimensionless | Optional override; estimated from pixels if omitted |

---

## Fitting emission lines

### `fit_lines()`

The main entry point.  Fits Gaussian emission lines to a spectrum with
continuum subtraction, parameter constraints, and bootstrap uncertainties.

```python
result = jwspecfit.fit_lines(
    spec, z=6.0,
    grating=None,        # auto-detected from FITS header
    R=None,              # override resolving power (float or callable)
    lines=None,          # restrict to specific lines (default: auto-detect)
    wave_range_A=None,   # (lo, hi) restrict to wavelength window (Å, observed)
    wave_windows_A=None, # list of (lo, hi) for multi-window fitting (stacks)
    deg=2,               # continuum polynomial degree
    n_boot=1000,         # bootstrap iterations (0 = analytic errors)
    clip_sigma=2.5,      # continuum sigma-clipping threshold
    n_jobs=-1,           # parallel bootstrap jobs (-1 = all cores)
    save_path=None,      # auto-export line table to this path
)
```

The returned `FitResult` contains:

| Field | Description |
|-------|-------------|
| `lines` | `dict[str, LineResult]` — per-line flux, EW, SNR, centroid, sigma |
| `params` | Full parameter vector `[amplitudes, centroids, sigmas]` |
| `model_flux` | Best-fit emission-line model (µJy, continuum-subtracted) |
| `continuum` | Best-fit polynomial continuum (µJy) |
| `residuals` | Data - continuum - model (µJy) |
| `chi2` | Reduced chi-squared |
| `spectrum` | The input `Spectrum` |
| `line_names` | Ordered line names matching the parameter vector |
| `success` | Whether the optimiser converged |

Each `LineResult` has: `name`, `rest_wave_A`, `amplitude`, `centroid_A`,
`sigma_A`, `flux` (erg/s/cm²), `flux_err`, `ew_A` (rest-frame), `snr`.

### Fast analytic errors

For quick exploration, skip bootstrap:

```python
result = jwspecfit.fit_lines(spec, z=6.0, n_boot=0)
```

### Fit specific lines only

```python
result = jwspecfit.fit_lines(spec, z=6.0, lines=["OIII_4959", "OIII_5007", "HBETA"])
```

### Fit a wavelength window

```python
result = jwspecfit.fit_lines(spec, z=6.0, wave_range_A=(35000, 50000))
```

### Multi-window fitting

For stacked spectra where the continuum shape varies across the full
wavelength range (e.g. UV-normalised stacks), fitting in a single pass
can produce poor continuum estimates at the blue or red end.  Use
`wave_windows_A` to fit multiple independent wavelength windows, each
with its own continuum subtraction:

```python
result = jwspecfit.fit_lines(
    spec, z=0.0,
    wave_windows_A=[
        (3500, 5200),   # blue window: [OII] → Hβ + [OIII]
        (5500, 7000),   # red window:  [NII] + Hα + [SII]
    ],
    sigma_factor=2.0,   # wider width bounds for stacked spectra
)
```

Each window gets its own polynomial continuum fit and line detection.
Results are merged into a single `FitResult` with the full spectrum
attached — compatible with `plot_fit()`, `plot_fit_interactive()`,
`export_lines_txt()`, and downstream packages (`jwspecmcmc`,
`jwspecabund`).

Pixels outside all windows have `NaN` continuum and residuals (no fit
attempted there), so the plot naturally shows gaps between windows.

`wave_windows_A` is mutually exclusive with `wave_range_A` and works
with all broad-component modes (`mode="auto"`, `"off"`, etc.).

---

## Broad Balmer component detection

### `fit_with_broad()`

Compares narrow-only vs narrow+broad Balmer models using BIC.  Four model
variants are tested:

| Model | Description |
|-------|-------------|
| **narrow** | Narrow lines only |
| **broad1** | + intermediate broad (FWHM ~ 500-2000 km/s) |
| **broad2** | + very broad / BLR (FWHM ~ 2000-5000 km/s) |
| **both** | + both broad components |

```python
result = jwspecfit.fit_with_broad(
    spec, z=2.5,
    mode="auto",            # "auto" | "off" | "broad1" | "broad2" | "both"
    n_boot=1000,            # bootstrap for flux uncertainties on winning model
    n_boot_bic=100,         # bootstrap iterations for robust BIC comparison
    snr_threshold=5.0,      # min Ha SNR to attempt broad fitting
    bic_delta=6.0,          # ΔBIC threshold for model acceptance
)

print(f"Selected model: {result.selected_model}")
print(f"BIC narrow={result.bic_narrow:.1f}, broad1={result.bic_broad1:.1f}")
fig = jwspecfit.plot_fit(result.best_fit)
```

Broad components are added for Hα, Hβ, Hδ, and Hγ.  NII kinematics are
tied to OIII to prevent broad Hα from absorbing NII flux.

---

## Plotting

### Static plot (matplotlib)

```python
fig = jwspecfit.plot_fit(
    result,
    wave_unit="A",             # "A" or "um"
    flux_unit="fnu",           # "fnu" (µJy) or "flam" (erg/s/cm²/Å)
    show_residuals=True,       # residual panel below
    show_components=True,      # individual Gaussian components
    label_lines=True,          # annotate line names
    y_pad=1.3,                 # y-axis padding above tallest line
    exclude_wave_A=None,       # list of (lo, hi) ranges to hide (Å)
    save_path="fit.pdf",       # save to file
)
```

### Interactive plot (plotly)

```python
fig = jwspecfit.plot_fit_interactive(
    result,
    wave_unit="A",
    flux_unit="fnu",
    show_components=True,      # smooth analytical Gaussian components
    show_residuals=True,       # residual subplot
    y_pad=1.3,
    exclude_wave_A=None,       # list of (lo, hi) ranges to hide (Å)
)
fig.show()
```

The interactive plot renders individual line components as smooth analytical
Gaussians (not bin-averaged), with hover information showing flux values.

### Hiding noisy regions

Both plotting functions accept `exclude_wave_A` to mask noisy detector
regions without affecting the fit:

```python
fig = jwspecfit.plot_fit(result, exclude_wave_A=[(5000, 6000), (48000, 50000)])
```

---

## Spectral resolution

Resolution is handled automatically from the FITS header, or can be specified
manually.

```python
from jwspecfit.resolution import R_prism, sigma_inst_A, resolve_R

# Prism R(λ): polynomial fit, R ~ 30-300
R = R_prism(np.array([1.0, 3.0, 5.0]))

# Instrumental sigma in Angstroms
sig = sigma_inst_A(wave_um, grating="PRISM")

# Resolve R from grating name, constant, or callable
R_arr = resolve_R(wave_um, grating="G395M")
```

| Grating | Resolving power |
|---------|----------------|
| PRISM   | R(λ) ~ 30-300 (wavelength-dependent) |
| G140M, G235M, G395M | R ~ 1000 |
| G140H, G235H, G395H | R ~ 2700 |

---

## Line database

`REST_LINES_A` contains 34 rest-frame vacuum wavelengths in Angstroms,
including hydrogen Balmer series, oxygen, nitrogen, carbon, silicon, and
helium lines.  Default line lists are selected by grating:

- **Prism**: merged doublets (OII_doublet, NV_doublet, CIV_doublet) + Lya
- **Medium/High**: resolved doublets (OII_3726, OII_3729) + auroral lines

Lines at or blueward of NV (1238.8 Å rest-frame) are automatically excluded
from fitting at the Lyman break.

---

## Parameter constraints

Applied automatically during fitting:

- **[NII] doublet ratio**: A(6549) / A(6585) = 1/2.96 (Storey & Zeippen 2000)
  with tied kinematics.
- **Balmer-OIII width tying**: Hα, Hβ, Hδ, Hγ, and NII_6585 widths are tied
  to OIII_5007 in velocity space.
- **Centroid bounds**: ±20 σ_inst of expected observed wavelength (or ±6 pixel
  widths, whichever is larger).
- **Broad components**: unconstrained (not subject to width tying).

---

## Saving and loading

```python
# Save full result (spectrum, parameters, per-line measurements)
jwspecfit.save_result(result, "fit.npz")

# Reload without re-fitting
loaded = jwspecfit.load_result("fit.npz")
fig = jwspecfit.plot_fit(loaded)

# Export line measurements as a text table
jwspecfit.export_lines_txt(result, "lines.txt")
# Columns: name, rest_wave_A, centroid_A, flux, flux_err, EW_A, sigma_v_kms,
#          SNR_integrated, SNR_peak
```

---

## Lyman-alpha modelling

Lya is modelled as a skewed Gaussian attenuated by mean IGM transmission
(Inoue et al. 2014).

```python
from jwspecfit.lyman_alpha import igm_transmission, lya_model

T = igm_transmission(wave_obs_A, z_source=7.0)  # T(λ) in [0, 1]
model = lya_model(lam_left, lam_right, z, amplitude, mu, sigma, skew)
```

---

## MCMC fitting with `jwspecmcmc`

`jwspecmcmc` is a companion package that replaces bootstrap uncertainties
with full Bayesian posterior sampling via **emcee** or **nautilus**.
It reuses the same `Spectrum`, line database, and plotting infrastructure
from `jwspecfit`.

Key capabilities:

- **Drop-in MCMC replacement** — `jwspecmcmc.fit_lines()` mirrors the
  `jwspecfit` API but returns full posterior chains.
- **BIC broad selection + MCMC** — by default (`mode="auto"`), performs
  fast least-squares BIC model selection, then runs MCMC on the winning model.
- **Asymmetric credible intervals** — proper (16th, 84th) percentile errors
  for flux, centroid, and sigma.
- **Flux-ratio posteriors** — `result.flux_ratio_posterior("OIII_5007", "HBETA")`
  for line-ratio diagnostics with full uncertainty propagation.
- **Convergence diagnostics** — Gelman–Rubin R-hat and effective sample
  size (ESS) computed automatically.
- **Custom priors** — override default uniform priors with `GaussianPrior`,
  `LogUniformPrior`, or any custom `Prior` subclass.
- **Diagnostic plots** — corner plots, trace plots, and flux posterior
  histograms.
- **Sampler choice** — `"emcee"` for MCMC or `"nautilus"` for nested sampling.

### Quick start

```python
import jwspecfit
import jwspecmcmc

spec = jwspecfit.read_fits("spectrum.fits", z=6.0)

# MCMC fit — BIC broad selection is on by default
result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="emcee", n_steps=2000)

print(result.selected_model)                # "narrow", "broad1", "broad2", or "both"
print(result.lines["OIII_5007"].flux_err)   # asymmetric (lo, hi) 68% CI

# Flux-ratio posterior for metallicity diagnostics
ratio = result.flux_ratio_posterior("OIII_5007", "HBETA")

# Convergence diagnostics
print(result.convergence)  # {'r_hat_max': ..., 'ess_min': ..., 'converged': ...}

# Diagnostic plots
jwspecmcmc.plot_traces(result, params=["A_OIII_5007", "A_HBETA"])
jwspecmcmc.plot_corner(result, params=["A_OIII_5007", "A_HBETA"])
jwspecmcmc.plot_flux_posterior(result, "OIII_5007")

# Convert to FitResult for jwspecfit plotting
fig = jwspecfit.plot_fit(result.to_fit_result())
```

### Custom priors

```python
from jwspecmcmc import GaussianPrior

result = jwspecmcmc.fit_lines(
    spec, z=6.0,
    mode="off",
    prior_overrides={
        "A_OIII_5007": GaussianPrior(mean=8e-18, std=2e-18, lo=0, hi=1e-15),
    },
)
```

### Nautilus nested sampling

```python
result = jwspecmcmc.fit_lines(
    spec, z=6.0,
    sampler="nautilus",
    mode="off",
    n_live=1000,
    n_eff=5000,
)
```

---

## Chemical abundances with `jwspecabund`

`jwspecabund` derives chemical abundances (O/H, N/O, S/O, Ne/O, Ar/O)
from the emission-line fluxes produced by `jwspecfit` or `jwspecmcmc`.
It accepts any result type (`FitResult`, `BroadFitResult`, `MCMCResult`,
`MCMCBroadFitResult`) and automatically selects the appropriate
uncertainty propagation (bootstrap MC or full posterior sampling).

Three methods are supported:

1. **Direct T_e method** — when the [OIII] 4363 auroral line is detected
   (SNR >= 3 by default), uses PyNEB for electron temperature, density,
   and ionic abundance calculations with Izotov+06 ICFs.
2. **Bayesian forward model** — free parameters (log T_e, log n_e,
   ionic abundances) sampled via emcee/dynesty, predicting line ratios
   with PyNEB CEL emissivities and the Aller (1984) Hbeta formula
   (Cullen+25 approach).
3. **Strong-line calibrations** — Sanders et al. (2025) simultaneous
   polynomial fit across O3, O2, R23, O32 diagnostics with MC
   error propagation including calibration scatter.

Dust correction (Balmer decrement + Salim+18 or Cardelli+89) is applied
automatically.  Broad Balmer components are summed with narrow
components for correct hydrogen flux normalisation.

### Installation

```bash
pip install -e ".[abund]"   # installs PyNEB >= 1.1.25
```

### Quick start

```python
import jwspecfit
import jwspecabund

# From a bootstrap fit result:
result = jwspecfit.fit_lines(spec, z=6.0)
abund = jwspecabund.compute_abundances(result, z=6.0)

# From an MCMC result (full posterior propagation):
import jwspecmcmc
mcmc_result = jwspecmcmc.fit_lines(spec, z=6.0)
abund = jwspecabund.compute_abundances(mcmc_result, z=6.0)

print(abund.summary())
```

### Method selection

```python
# Auto (default): direct if [OIII] 4363 SNR >= 3, else strong-line
abund = jwspecabund.compute_abundances(result, z=6.0)

# Force direct T_e method
abund = jwspecabund.compute_abundances(result, z=6.0, method="direct")

# Force strong-line calibrations
abund = jwspecabund.compute_abundances(result, z=6.0, method="strong_line")

# Bayesian forward model (Cullen+25)
abund = jwspecabund.compute_abundances(result, z=6.0, method="forward")
```

### Key options

```python
abund = jwspecabund.compute_abundances(
    result, z=6.0,
    dust_law="salim",         # "salim" (default) or "cardelli"
    Av=None,                  # derive from Balmer decrement (default)
    Te_relation="desi",       # "desi" (DESI DR2) or "classical" (Garnett 1992)
    n_mc=1000,                # MC iterations for bootstrap results
    n_posterior=1000,          # max posterior samples to propagate (MCMC results)
    progress=True,            # show tqdm progress bars
)
```

### Result object

`AbundanceResult` holds the full output:

| Field | Type | Description |
|-------|------|-------------|
| `method` | `str` | `"direct"`, `"forward"`, or `"strong_line"` |
| `OH` | `float` | 12 + log(O/H) |
| `OH_err` | `float \| tuple` | Symmetric error or (lo, hi) 68% CI |
| `NO` | `float \| None` | log(N/O) |
| `Te_high`, `Te_low` | `float \| None` | Electron temperatures (K) |
| `ne` | `float \| None` | Electron density (cm^-3) |
| `Av` | `float \| None` | Dust attenuation A_V |
| `ionic` | `dict \| None` | Ionic abundances (O+/H+, O++/H+, ...) |
| `OH_posterior` | `np.ndarray \| None` | Full posterior samples |
| `SO`, `NeO`, `ArO` | `float \| None` | log(S/O), log(Ne/O), log(Ar/O) |
| `ratios_used` | `list \| None` | Diagnostic ratios (strong-line) |

---

## Example notebooks

See `docs/notebooks/` for worked examples:

| Notebook | Description |
|----------|-------------|
| `01_prism_fit.ipynb` | Basic prism fitting, save/load, text export, plotly |
| `02_grating_broad.ipynb` | G395M grating with BIC broad-line detection |
| `03_stacked_spectrum.ipynb` | Stacked spectrum with custom R, IGM demo |
| `04_mcmc_prism.ipynb` | MCMC fitting on a PRISM spectrum with emcee |
| `05_mcmc_grating.ipynb` | MCMC fitting on a G395M grating spectrum, nautilus demo |
| `06_mcmc_stack.ipynb` | MCMC fitting on a stacked spectrum with R estimation |
| `07_abundances.ipynb` | Chemical abundances: direct T_e, forward model, strong-line |
| `08_nitrogen.ipynb` | N/O ratio calculation via the direct method |

---

## Modules

### `jwspecfit`

| Module | Description |
|--------|-------------|
| `io` | `Spectrum` container, FITS/npz/dict readers, save/load/export |
| `lines` | Rest-frame line database, line-list selection |
| `resolution` | R(λ) models, instrumental sigma |
| `continuum` | Polynomial continuum with sigma-clipping |
| `models` | Bin-averaged Gaussian profiles via erf |
| `constraints` | [NII] ratio, Balmer-OIII width tying |
| `fitter` | Core `fit_lines()` engine |
| `broad` | Broad component fitting + BIC model selection |
| `lyman_alpha` | Skewed Gaussian + IGM absorption |
| `plotting` | Static and interactive visualisation |

### `jwspecmcmc`

| Module | Description |
|--------|-------------|
| `__init__` | Public API: `fit_lines()`, `fit_with_broad()`, plotting wrappers |
| `_engine` | MCMC fitting engine (emcee / nautilus backends) |
| `samplers` | `run_emcee()`, `run_nautilus()` sampler wrappers |
| `likelihood` | Log-likelihood for Gaussian emission-line models |
| `priors` | `UniformPrior`, `GaussianPrior`, `LogUniformPrior`, `PriorSet` |
| `result` | `MCMCResult`, `MCMCBroadFitResult`, `MCMCLineResult` |
| `diagnostics` | Gelman–Rubin R-hat, effective sample size (ESS) |
| `plotting` | Corner plots, trace plots, flux posterior histograms |

### `jwspecabund`

| Module | Description |
|--------|-------------|
| `_core` | `compute_abundances()` orchestrator, method selection, dust correction |
| `direct` | Direct T_e method via PyNEB: T_e, n_e, ionic abundances |
| `forward` | Bayesian forward model (Cullen+25): emcee / dynesty sampling |
| `strong_line` | Sanders+25 simultaneous polynomial calibrations |
| `dust` | Salim+18 and Cardelli+89 attenuation curves, Balmer decrement A_V |
| `icf` | Ionisation correction factors (Izotov+06) |
| `result` | `AbundanceResult` dataclass |

See [`docs/api.md`](docs/api.md) for the full API reference.

---

## Tests

```bash
pytest tests/ -v
```

---

## License

MIT
