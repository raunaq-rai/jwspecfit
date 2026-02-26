# jwspecfit

**Resolution-aware emission-line fitting for JWST NIRSpec spectra.**

`jwspecfit` fits Gaussian emission-line profiles to 1-D extracted JWST NIRSpec
spectra, handling the wavelength-dependent resolution of the prism (R ~ 30--300),
fixed-resolution gratings (R ~ 1000--2700), and user-supplied resolving powers
for stacked spectra.  Companion packages `jwspecmcmc` (Bayesian MCMC fitting)
and `jwspecabund` (chemical abundances) build on top.

---

## Installation

```bash
git clone https://github.com/raunaq-rai/jwspecfit.git
cd jwspecfit
pip install -e ".[dev]"

# For chemical abundances (installs PyNEB):
pip install -e ".[abund]"
```

**Requirements:** Python >= 3.10, numpy, scipy, astropy, matplotlib, tqdm, joblib, plotly.

---

## Tutorial 1: Fitting emission lines

### Load a spectrum

```python
import jwspecfit

# From a JWST x1d FITS file — grating and R are read from the header.
spec = jwspecfit.read_fits("spectrum.fits", z=6.0)

# From numpy arrays
spec = jwspecfit.read_dict(
    {"wave": wave_um, "flux": flux_ujy, "err": err_ujy},
    z=6.0,
)
```

Wavelength must be in **microns** and flux/error in **micro-Jansky**.
See [`docs/api.md`](docs/api.md) for unit conversion recipes and
alternative loaders (`read_npz`, direct `Spectrum` construction).

### Fit lines

```python
result = jwspecfit.fit_lines(spec, z=6.0)
```

This auto-detects observable lines from the grating and wavelength
coverage, subtracts a polynomial continuum, fits resolution-aware
Gaussians, and estimates uncertainties via bootstrap (1000 iterations
by default).

Common options:

```python
result = jwspecfit.fit_lines(
    spec, z=6.0,
    lines=["OIII_5007", "OIII_4959", "HBETA"],  # fit specific lines
    wave_range_A=(35000, 50000),                  # restrict wavelength window
    n_boot=0,                                     # skip bootstrap for speed
    mode="auto",                                  # broad Balmer detection (auto/off)
)
```

### Inspect results

```python
for name, line in result.lines.items():
    if line.snr > 3:
        print(f"{name}: flux={line.flux:.2e} +/- {line.flux_err:.2e}, SNR={line.snr:.1f}")
```

Each `LineResult` contains: `flux`, `flux_err`, `snr`, `ew_A`
(rest-frame equivalent width), `amplitude`, `centroid_A`, `sigma_A`.

### Plot

```python
# Static (matplotlib)
fig = jwspecfit.plot_fit(result, save_path="fit.pdf")

# Interactive (plotly) — zoomable with hover info
fig = jwspecfit.plot_fit_interactive(result)
fig.show()
```

Both show individual Gaussian components, residuals, and line labels.
Absorption lines (`abs_` prefix) are rendered as downward troughs with
labels below the feature.

### Save and reload

```python
jwspecfit.save_result(result, "fit.npz")
loaded = jwspecfit.load_result("fit.npz")

# Export a text table of line measurements
jwspecfit.export_lines_txt(result, "lines.txt")
```

### Absorption lines

Lines named with the `abs_` prefix (e.g. `abs_SiII1260`, `abs_CII1334`)
are fitted as negative Gaussians.  Pass them explicitly in the `lines=`
list:

```python
uv_lines = [
    "NV_1", "NV_2", "CIV_1", "CIV_2", "HEII_1640",
    "CIII]_1907", "CIII]",
    "abs_SiII1260", "abs_CII1334", "abs_SiIV1394", "abs_SiIV1403",
]

result = jwspecfit.fit_lines(spec, z=6.0, lines=uv_lines)
```

Fitted absorption lines have negative `flux` and `amplitude`.

---

## Tutorial 2: Chemical abundances

`jwspecabund` derives oxygen abundance (12 + log O/H), nitrogen-to-oxygen
(log N/O), and other ratios from the emission-line fluxes measured by
`jwspecfit` or `jwspecmcmc`.

### Basic usage

```python
import jwspecabund

abund = jwspecabund.compute_abundances(result, z=6.0)
print(abund.summary())
```

### Method selection

`compute_abundances` automatically chooses between the direct Te method
(when [OIII] 4363 is detected at SNR >= 3) and strong-line calibrations
(Sanders+25).  You can override:

```python
# Force direct Te method
abund = jwspecabund.compute_abundances(result, z=6.0, method="direct")

# Force strong-line calibrations
abund = jwspecabund.compute_abundances(result, z=6.0, method="strong_line")

# Bayesian forward model (Cullen+25)
abund = jwspecabund.compute_abundances(result, z=6.0, method="forward")
```

### What happens under the hood

1. **Dust correction** — A_V derived from the Hgamma/Hbeta Balmer
   decrement (intrinsic ratio 0.468).  Dust law: `"salim"` (Salim+18,
   default) or `"cardelli"` (CCM89).  Pass `Av=` to fix it manually,
   or `dust_correct=False` to skip.

2. **Electron density** — two-zone model:
   - Low-ionisation: [SII] 6718/6732 (fallback: [OII] 3726/3729, then 100 cm^-3)
   - High-ionisation: NIV] 1483/1486 (fallback: CIII] 1907/1909, then ne_low)

3. **Electron temperature** — Te(O2+) from [OIII] 4363/(4959+5007).
   Te(low) derived via a Te-Te relation: `"desi"` (DESI DR2, default)
   or `"classical"` (Garnett 1992).

4. **Ionic and total abundances** — PyNEB emissivities with ICFs
   (Martinez+25 / Izotov+06) for N/O, S/O, Ne/O, Ar/O, C/O.

### Key options

```python
abund = jwspecabund.compute_abundances(
    result, z=6.0,
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
| `Te_high`, `Te_low` | Electron temperatures (K) |
| `ne` | Electron density (cm^-3) |
| `Av` | Dust attenuation |
| `ionic` | Ionic abundances dict (O+/H+, O++/H+, ...) |
| `OH_posterior` | Full posterior array (for MC/MCMC) |

---

## Example notebooks

Worked examples in [`docs/notebooks/`](docs/notebooks/):

| Notebook | Description |
|----------|-------------|
| `01_prism_fit` | Prism fitting, save/load, plotting |
| `02_grating_broad` | G395M grating with broad-line detection |
| `03_stacked_spectrum` | Stacked spectrum with custom R |
| `04_mcmc_prism` | MCMC fitting with emcee |
| `07_abundances` | Chemical abundances: direct, forward, strong-line |
| `09_uv_abundances` | UV line fitting with absorption lines |

---

## Further documentation

- **Full API reference**: [`docs/api.md`](docs/api.md) — all functions,
  classes, parameters, and module descriptions for `jwspecfit`,
  `jwspecmcmc`, and `jwspecabund`.

---

## Tests

```bash
pytest tests/ -v
```

## Licence

MIT — see [LICENCE](LICENCE).
