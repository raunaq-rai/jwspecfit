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
spec = jwspecfit.read_npz("stack.npz", z=6.0, R=150.0)

# From numpy arrays
spec = jwspecfit.read_dict(
    {"wave": wave_um, "flux": flux_ujy, "err": err_ujy},
    z=6.0, R=100.0,
)
```

The `Spectrum` container holds wavelength (µm), flux (µJy), uncertainty (µJy),
and metadata.  Properties include `wave_A`, `dlam_A`, `wave_edges_A`,
`flux_flam`, `err_flam`, and `mask_valid()`.

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

## Example notebooks

See `docs/notebooks/` for worked examples:

| Notebook | Description |
|----------|-------------|
| `01_prism_fit.ipynb` | Basic prism fitting, save/load, text export, plotly |
| `02_grating_broad.ipynb` | G395M grating with BIC broad-line detection |
| `03_stacked_spectrum.ipynb` | Stacked spectrum with custom R, IGM demo |

---

## Modules

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

See [`docs/api.md`](docs/api.md) for the full API reference.

---

## Tests

```bash
pytest tests/ -v
```

---

## License

MIT
