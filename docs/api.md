# jwspecfit, jwspecmcmc & jwspecabund API Reference

Full API documentation for all public modules, classes, and functions.
For tutorials, see the [README](../README.md).

---

## Module overview

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
| `_engine` | MCMC fitting engine (NUTS / emcee / nautilus backends) |
| `samplers` | `run_nuts()`, `run_emcee()`, `run_nautilus()` sampler wrappers |
| `likelihood` | Log-likelihood for Gaussian emission-line models |
| `priors` | `UniformPrior`, `GaussianPrior`, `LogUniformPrior`, `PriorSet` |
| `result` | `MCMCResult`, `MCMCBroadFitResult`, `MCMCLineResult` |
| `diagnostics` | Gelman-Rubin R-hat, effective sample size (ESS) |
| `plotting` | Corner plots, trace plots, flux posterior histograms |

### `jwspecabund`

| Module | Description |
|--------|-------------|
| `_core` | `compute_abundances()` orchestrator, method selection, dust correction |
| `direct` | Direct T_e method via PyNEB: T_e, n_e, ionic abundances |
| `forward` | Bayesian forward model (Cullen+25): emcee / dynesty sampling |
| `strong_line` | Sanders+25 simultaneous polynomial calibrations |
| `dust` | Salim+18 and Cardelli+89 attenuation curves, Balmer decrement A_V, multi-Balmer weighted average |
| `icf` | Ionisation correction factors (Izotov+06, Garnett+97) |
| `result` | `AbundanceResult` dataclass |

---

## Loading non-standard spectra

`jwspecfit` requires wavelength in **microns** and flux/error in
**micro-Jansky**.  Below are recipes for common conversions.

### Unit conversions

```python
# Angstroms -> microns
wave_um = wave_angstrom * 1e-4

# Nanometres -> microns
wave_um = wave_nm * 1e-3

# Jansky -> µJy
flux_ujy = flux_jy * 1e6

# erg/s/cm²/Å (F_lambda) -> µJy
c_cgs = 2.99792458e18   # Å/s
flux_ujy = flux_flam * (wave_angstrom ** 2) / c_cgs * 1e29

# erg/s/cm²/Hz (F_nu in CGS) -> µJy
flux_ujy = flux_fnu * 1e29
```

### From a CSV or ASCII table

```python
import numpy as np

data = np.genfromtxt("spectrum.csv", delimiter=",", names=True)
wave_um = data["wavelength_A"] * 1e-4
c_cgs = 2.99792458e18
flux_ujy = data["flux_flam"] * (data["wavelength_A"] ** 2) / c_cgs * 1e29
err_ujy  = data["err_flam"]  * (data["wavelength_A"] ** 2) / c_cgs * 1e29

spec = jwspecfit.read_dict(
    {"wave": wave_um, "flux": flux_ujy, "err": err_ujy},
    z=2.5,
)
```

### From a non-standard FITS file

```python
from astropy.io import fits

with fits.open("other_pipeline.fits") as hdul:
    tbl = hdul[1].data
    wave_um = tbl["WAVELENGTH"] * 1e-4   # e.g. Å -> µm
    flux_ujy = tbl["FLUX"] * 1e6         # e.g. Jy -> µJy
    err_ujy  = tbl["ERROR"] * 1e6

spec = jwspecfit.read_dict(
    {"wave": wave_um, "flux": flux_ujy, "err": err_ujy},
    z=3.0, grating="G395M",
)
```

### Building a `Spectrum` directly

```python
from jwspecfit.io import Spectrum

spec = Spectrum(
    wave_um=wave_um,
    flux_ujy=flux_ujy,
    err_ujy=err_ujy,
    grating=None,       # or "PRISM", "G395M", etc.
    z=4.5,
    R=150.0,            # optional — estimated from pixels if omitted
    meta={"source": "my_pipeline", "target": "GN-z11"},
)
```

---

## Multi-window fitting

For stacked spectra where the continuum shape varies across the full
wavelength range, fitting in a single pass can produce poor continuum
estimates.  Use `wave_windows_A` to fit multiple independent wavelength
windows, each with its own continuum subtraction:

```python
result = jwspecfit.fit_lines(
    spec, z=0.0,
    wave_windows_A=[
        (3500, 5200),   # blue window: [OII] -> Hbeta + [OIII]
        (5500, 7000),   # red window:  [NII] + Halpha + [SII]
    ],
    sigma_factor=2.0,   # wider width bounds for stacked spectra
)
```

Each window gets its own continuum fit and automatic line detection.
Results are merged into a single `FitResult`.  `wave_windows_A` is
mutually exclusive with `wave_range_A`.

---

## Broad Balmer component detection

`fit_with_broad()` compares narrow-only vs narrow+broad Balmer models
using BIC.  Four model variants are tested:

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
    n_boot_bic=100,         # bootstrap iterations for robust BIC comparison
    snr_threshold=5.0,      # min Halpha SNR to attempt broad fitting
    bic_delta=6.0,          # delta-BIC threshold for model acceptance
)

print(f"Selected model: {result.selected_model}")
```

Broad components are added for Halpha, Hbeta, Hdelta, and Hgamma.  NII
kinematics are tied to OIII to prevent broad Halpha from absorbing NII
flux.

---

## Parameter constraints

Applied automatically during fitting:

- **[NII] doublet ratio**: A(6549) / A(6585) = 1/2.96 (Storey & Zeippen
  2000) with tied kinematics.
- **Balmer-OIII width tying**: Halpha, Hbeta, Hdelta, Hgamma, and
  NII_6585 widths are tied to OIII_5007 in velocity space.
- **Centroid bounds**: limited by `centroid_vmax` (default 500 km/s),
  capped at half the separation to the nearest line.
- **Broad components**: unconstrained (not subject to width tying).

---

## Lyman-alpha modelling

Lya is modelled as a skewed Gaussian attenuated by mean IGM transmission
(Inoue et al. 2014).

```python
from jwspecfit.lyman_alpha import igm_transmission, lya_model

T = igm_transmission(wave_obs_A, z_source=7.0)  # T(lambda) in [0, 1]
model = lya_model(lam_left, lam_right, z, amplitude, mu, sigma, skew)
```

---

## MCMC fitting with `jwspecmcmc`

`jwspecmcmc` replaces bootstrap uncertainties with full Bayesian
posterior sampling via **emcee** or **nautilus**.  It reuses the same
`Spectrum`, line database, and plotting infrastructure from `jwspecfit`.

### Quick start

```python
import jwspecmcmc

result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="emcee", n_steps=2000)

print(result.selected_model)
print(result.lines["OIII_5007"].flux_err)   # asymmetric (lo, hi) 68% CI

# Flux-ratio posterior
ratio = result.flux_ratio_posterior("OIII_5007", "HBETA")

# Convergence diagnostics
print(result.convergence)

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

## `jwspecfit.io` — Spectrum I/O

### `Spectrum`

```python
@dataclass
class Spectrum:
    wave_um: np.ndarray        # Observed wavelength (µm)
    flux_ujy: np.ndarray       # Flux density (µJy)
    err_ujy: np.ndarray        # 1σ uncertainty (µJy)
    grating: str | None        # Grating name (e.g. "PRISM", "G395M")
    z: float | None            # Source redshift
    R: float | None            # Resolving power (overrides grating)
    meta: dict[str, Any]       # Arbitrary metadata
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `wave_A` | `np.ndarray` | Wavelength in Angstroms (`wave_um × 1e4`) |
| `n_pix` | `int` | Number of pixels |
| `wave_edges_A` | `np.ndarray` | Pixel-edge wavelengths (Å), length `n_pix + 1` |
| `dlam_A` | `np.ndarray` | Pixel widths (Å) |
| `flux_flam` | `np.ndarray` | Flux density in erg/s/cm²/Å |
| `err_flam` | `np.ndarray` | Error in erg/s/cm²/Å |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `mask_valid()` | `np.ndarray[bool]` | True where flux and err are finite and err > 0 |
| `copy()` | `Spectrum` | Shallow copy with copied arrays |

**Required units:**

| Field | Unit | Notes |
|-------|------|-------|
| `wave_um` | µm | Observed-frame wavelength |
| `flux_ujy` | µJy | f_ν flux density |
| `err_ujy` | µJy | 1σ uncertainty (same units as flux) |

**Resolving power** is determined automatically: from the grating header
for FITS files, or estimated from pixel spacing (`R ≈ λ / 2Δλ`) when
neither `grating` nor `R` is set.  You only need to set `R` explicitly
if you know the true resolving power and want to override the estimate.

If your data uses different units, convert before constructing a `Spectrum`:

```python
# Angstroms → µm
wave_um = wave_angstrom * 1e-4

# erg/s/cm²/Å → µJy
c_cgs = 2.99792458e18   # Å/s
flux_ujy = flux_flam * (wave_angstrom ** 2) / c_cgs * 1e29

# Jy → µJy
flux_ujy = flux_jy * 1e6

# Direct construction — R is optional (auto-estimated if omitted)
spec = Spectrum(
    wave_um=wave_um, flux_ujy=flux_ujy, err_ujy=err_ujy,
    z=4.5,
)
```

---

### `read_fits(path, z=None)`

Read a JWST NIRSpec 1-D extracted spectrum from FITS.

Expects an HDU named `SPEC1D` with columns `wave` (µm), `flux` (µJy),
`err` (µJy).  Grating is auto-detected from the `GRATING` header keyword.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to the FITS file |
| `z` | `float \| None` | Source redshift to attach |

Returns `Spectrum`.

---

### `read_npz(path, z=None, R=None)`

Read a stacked spectrum from a NumPy `.npz` file.

Expected keys: `wave_angstrom`, `flux`, `err`.  Optionally `n_stacked`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to the `.npz` file |
| `z` | `float \| None` | Source redshift |
| `R` | `float \| None` | Effective resolving power |

Returns `Spectrum`.

---

### `read_dict(data, z=None, grating=None, R=None)`

Create a `Spectrum` from a dict with keys `wave`/`lam` (µm), `flux` (µJy),
`err` (µJy).

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `dict` | Must contain `"wave"` or `"lam"`, `"flux"`, `"err"` |
| `z` | `float \| None` | Source redshift |
| `grating` | `str \| None` | Grating name |
| `R` | `float \| None` | Resolving power |

Returns `Spectrum`.

---

### `save_result(result, path)`

Save a `FitResult` to a `.npz` file for later replotting without re-fitting.

| Parameter | Type | Description |
|-----------|------|-------------|
| `result` | `FitResult` | Fit result to save |
| `path` | `str \| Path` | Output file path (`.npz`) |

---

### `load_result(path)`

Load a `FitResult` from a `.npz` file saved by `save_result()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to the `.npz` file |

Returns `FitResult`.

---

### `export_lines_txt(result, path, z=None)`

Export per-line measurements to a text file.

Columns: `name`, `rest_wave_A`, `centroid_A`, `flux` (erg/s/cm²),
`flux_err`, `EW_A` (rest-frame), `sigma_v_kms`, `SNR_integrated`, `SNR_peak`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `result` | `FitResult` | Fit result |
| `path` | `str \| Path` | Output text file path |
| `z` | `float \| None` | Redshift for velocity calculation (default: `result.spectrum.z`) |

---

## `jwspecfit.fitter` — Core fitting engine

### `fit_lines()`

Fit emission lines in a spectrum with continuum subtraction, parameter
constraints, and bootstrap uncertainties.

```python
def fit_lines(
    spectrum: Spectrum,
    z: float,
    *,
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    wave_range_A: tuple[float, float] | None = None,
    deg: int = 2,
    n_boot: int = 1000,
    clip_sigma: float = 2.5,
    n_jobs: int = -1,
    save_path: str | Path | None = None,
) -> FitResult:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spectrum` | `Spectrum` | — | Input spectrum |
| `z` | `float` | — | Source redshift |
| `grating` | `str \| None` | `None` | Grating name. Falls back to `spectrum.grating` |
| `R` | `float \| Callable \| None` | `None` | Resolving power. Overrides `grating`. Can be constant or callable `R(lam_um)` |
| `lines` | `list[str] \| None` | `None` | Lines to fit. If `None`, auto-detected from grating and wavelength coverage |
| `wave_range_A` | `tuple[float, float] \| None` | `None` | Observed wavelength window (Å). Pixels outside are excluded |
| `deg` | `int` | `2` | Continuum polynomial degree |
| `n_boot` | `int` | `1000` | Bootstrap iterations. `0` for analytic errors |
| `clip_sigma` | `float` | `2.5` | Continuum sigma-clipping threshold |
| `n_jobs` | `int` | `-1` | Parallel bootstrap jobs (`-1` = all cores, `1` = sequential) |
| `save_path` | `str \| Path \| None` | `None` | Auto-export line table to this path |

Returns `FitResult`.

**Algorithm:**

1. Determine observable lines from redshift, grating, and wavelength range
2. Fit iterative sigma-clipped polynomial continuum (masking emission lines)
3. Set up Gaussian model with resolution-aware width bounds
4. Apply parameter constraints (NII ratio, Balmer-OIII width tying)
5. Optimise with `scipy.optimize.least_squares` (TRF method, `x_scale='jac'`)
6. Bootstrap (if `n_boot > 0`): perturb spectrum 1000× and refit for flux uncertainties

---

### `FitResult`

```python
@dataclass
class FitResult:
    lines: dict[str, LineResult]     # Per-line results
    params: np.ndarray               # Full parameter vector [A, mu, sigma]
    model_flux: np.ndarray           # Emission-line model (µJy, continuum-subtracted)
    continuum: np.ndarray            # Polynomial continuum (µJy)
    residuals: np.ndarray            # Data - continuum - model (µJy)
    chi2: float                      # Reduced χ²
    spectrum: Spectrum               # Input spectrum
    line_names: list[str]            # Ordered line names
    constraints: ConstraintSet | None  # Applied constraints
    success: bool                    # Optimiser convergence
```

The parameter vector layout is:
```
[A_0, A_1, ..., A_{n-1}, mu_0, mu_1, ..., mu_{n-1}, sigma_0, sigma_1, ..., sigma_{n-1}]
```
where `A` = amplitude (flux in erg/s/cm²), `mu` = centroid (Å), `sigma` = width (Å).

**Methods:**

#### `flux_upper_limit(line_name, n_sigma=3.0)`

Compute a noise-based flux upper limit for a single line from the local
continuum RMS.  Measures the RMS of fit residuals in a window around the
expected line position (±5σ, excluding the central ±2σ), then returns
`n_sigma × RMS × σ_inst × √(2π)`.

#### `flux_upper_limits(line_names=None, n_sigma=3.0)`

Compute upper limits for all non-detected lines (SNR < `n_sigma`) or for
a specific list of lines.  Returns `{line_name: flux_upper_limit}`.

---

### `LineResult`

```python
@dataclass
class LineResult:
    name: str               # Line name
    rest_wave_A: float      # Rest-frame wavelength (Å)
    amplitude: float        # Best-fit amplitude (flux × Å)
    centroid_A: float       # Observed centroid (Å)
    sigma_A: float          # Observed Gaussian σ (Å)
    flux: float             # Integrated line flux (erg/s/cm²)
    flux_err: float         # Flux uncertainty (bootstrap or analytic)
    ew_A: float             # Rest-frame equivalent width (Å)
    snr: float              # Signal-to-noise ratio (flux / flux_err)
```

---

## `jwspecfit.broad` — Broad Balmer component detection

### `fit_with_broad()`

Compare narrow-only vs narrow+broad Balmer models using BIC-based
model selection.

```python
def fit_with_broad(
    spectrum: Spectrum,
    z: float,
    *,
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    deg: int = 2,
    mode: str = "auto",
    n_boot: int = 1000,
    n_boot_bic: int = 100,
    snr_threshold: float = 5.0,
    bic_delta: float = 6.0,
    n_jobs: int = -1,
) -> BroadFitResult:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"auto"` | `"auto"`: BIC selection. `"off"`: narrow-only. `"broad1"` / `"broad2"` / `"both"`: force model |
| `n_boot` | `int` | `1000` | Bootstrap iterations for flux uncertainties (winning model) |
| `n_boot_bic` | `int` | `100` | Bootstrap iterations for robust BIC comparison |
| `snr_threshold` | `float` | `5.0` | Minimum Hα SNR to attempt broad fitting |
| `bic_delta` | `float` | `6.0` | ΔBIC threshold for accepting a more complex model |

**Velocity bounds for broad components:**

| Component | σ_v range (km/s) | FWHM range (km/s) | Physical origin |
|-----------|------------------|--------------------|-----------------|
| BROAD1 | 210-850 | 500-2000 | AGN NLR outflows, turbulent ISM |
| BROAD2 | 850-2120 | 2000-5000 | BLR virial motions |

Broad components are added for: Hα, Hβ, Hδ, Hγ.

---

### `BroadFitResult`

```python
@dataclass
class BroadFitResult:
    best_fit: FitResult                      # Selected best-fit
    selected_model: str                      # "narrow", "broad1", "broad2", "both"
    bic_narrow: float                        # Median BIC for narrow-only
    bic_broad1: float                        # Median BIC for narrow + BROAD1
    bic_broad2: float                        # Median BIC for narrow + BROAD2
    bic_both: float                          # Median BIC for narrow + BROAD1 + BROAD2
    all_fits: dict[str, FitResult]           # All fitted model variants
    bic_bootstrap: dict[str, np.ndarray]     # Full BIC distributions
```

---

## `jwspecfit.lines` — Line database

### `REST_LINES_A`

`dict[str, float]` — Rest-frame vacuum wavelengths in Angstroms.

Sources: NIST ASD, Morton (2003), Storey & Zeippen (2000).

**UV lines:**

| Name | λ_rest (Å) |
|------|-----------|
| `Lya` | 1215.670 |
| `NV_1` | 1238.821 |
| `NV_2` | 1242.804 |
| `NV_doublet` | 1240.81 |
| `NIV_1486` | 1486.496 |
| `CIV_1` | 1548.187 |
| `CIV_2` | 1550.772 |
| `CIV_doublet` | 1549.48 |
| `HEII_1640` | 1640.42 |
| `OIII_1661` | 1660.81 |
| `OIII_1666` | 1666.15 |
| `SiIII_1` | 1882.71 |
| `SiIII_2` | 1892.03 |
| `CIII]` | 1908.734 |

**Semi-forbidden / weak:**

| Name | λ_rest (Å) |
|------|-----------|
| `OIII]_2321` | 2322.41 |
| `OIII]_2331` | 2332.02 |
| `OII]_2471` | 2471.72 |
| `FeII*_2396` | 2397.09 |

**Optical:**

| Name | λ_rest (Å) |
|------|-----------|
| `OII_3726` | 3727.09 |
| `OII_3729` | 3729.88 |
| `OII_doublet` | 3728.48 |
| `HDELTA` | 4104.05 |
| `HGAMMA` | 4342.90 |
| `OIII_4363` | 4364.44 |
| `HBETA` | 4864.04 |
| `OIII_4959` | 4961.68 |
| `OIII_5007` | 5009.64 |
| `NII_5756` | 5757.79 |
| `HEI_5877` | 5878.88 |
| `NII_6549` | 6551.67 |
| `Ha` | 6566.42 |
| `NII_6585` | 6587.09 |
| `SII_6718` | 6720.15 |
| `SII_6732` | 6734.53 |

---

### `get_line_list(grating="prism")`

Return the default line names for a grating.

- `"prism"` → merged doublets + Lya
- `"medium"` / `"g395m"` etc. → resolved doublets + auroral lines
- `"high"` / `"g395h"` etc. → same as medium

Returns `list[str]`.

---

### `observable_lines(line_names, z, wave_min_um, wave_max_um)`

Filter lines to those observable in the wavelength range at redshift `z`.
Lines blueward of NV (IGM-absorbed region) are excluded.

Returns `list[str]`.

---

### `rest_wave_A(name)` / `observed_wave_A(name, z)` / `observed_wave_um(name, z)`

Convenience functions for wavelength lookups.

---

## `jwspecfit.resolution` — Spectral resolution

### `R_prism(lam_um)`

Polynomial R(λ) for NIRSpec PRISM/CLEAR, clipped to [30, 300].

Formula: `R = 50 + 50×(λ-1) + 15×(λ-1)²` (λ in µm).

---

### `R_grating(name)`

Constant R for named gratings:
- G140M, G235M, G395M → 1000
- G140H, G235H, G395H → 2700

---

### `resolve_R(lam_um, grating=None, R=None)`

Return R(λ) array from a grating name, constant R, or callable.
`R` takes precedence over `grating`.

---

### `R_from_pixels(lam_um)`

Estimate R from pixel spacing: R ≈ λ / (2Δλ).  Returns a callable
`R(lam_um)` that interpolates the estimated resolving power.  Fallback
only — use grating or explicit R when available.

---

### `sigma_inst_A(lam_um, grating=None, R=None)`

Instrumental Gaussian σ in Angstroms: `σ = λ / (R × 2.3548)`.

---

### `sigma_inst_kms(lam_um, grating=None, R=None)`

Instrumental Gaussian σ in km/s: `σ_v = c / (R × 2.3548)`.

---

## `jwspecfit.continuum` — Continuum fitting

### `fit_continuum()`

```python
def fit_continuum(
    wave_um: np.ndarray,
    flux_ujy: np.ndarray,
    err_ujy: np.ndarray,
    z: float,
    line_names: list[str],
    *,
    grating: str | None = None,
    R: float | None = None,
    deg: int = 2,
    clip_sigma: float = 2.5,
    n_iter: int = 5,
    line_mask_nsigma: float = 6.0,
) -> np.ndarray:
```

Algorithm:
1. Mask pixels blueward of NV (Lyman break region)
2. Mask pixels within ±`line_mask_nsigma` × σ_inst of every known line
3. Fit weighted polynomial of degree `deg` to unmasked pixels
4. Iteratively sigma-clip positive residuals and refit (emission rejection)

Returns continuum evaluated at each pixel (µJy).

---

## `jwspecfit.models` — Gaussian profiles

### `gaussian_binned(lam_left_A, lam_right_A, mu_A, sigma_A)`

Bin-averaged, area-normalised Gaussian profile using the error function.
Returns the mean value (Å⁻¹) integrated over each pixel bin.  Multiply by
amplitude (flux in erg/s/cm²) to get flux density.

This avoids sampling bias when lines are narrower than pixels (prism regime).

---

### `build_model(params, wave_edges_A, n_lines)`

Build multi-line emission model from a flat parameter vector.  Uses
vectorised numpy broadcasting for all lines simultaneously.

Parameter layout: `[A_0..A_n, mu_0..mu_n, sigma_0..sigma_n]`.

Returns flux density per pixel (Å⁻¹ × amplitude units).

---

### `pixel_weight(dlam_A, power=0.35)`

Pixel-width weighting: `w = (median_dλ / dλ)^power`.  Down-weights
atypically wide or narrow pixels.

---

## `jwspecfit.constraints` — Parameter constraints

### `ConstraintSet`

```python
@dataclass
class ConstraintSet:
    line_names: list[str]
    tie_nii: bool = True
    tie_balmer_to_oiii: bool = True
```

**Methods:**

| Method | Description |
|--------|-------------|
| `apply(params)` | Apply constraints to a parameter vector, returning a copy |
| `free_mask()` | Boolean mask of free (unconstrained) parameters |
| `expand_free_to_full(p_free)` | Insert free parameters into a full-length vector and apply constraints |

**Constraints applied:**

1. **[NII] 6549 / 6585 ratio**: `A(6549) = A(6585) × 1/2.96` with tied kinematics
2. **Balmer width tying**: Hα, Hβ, Hδ, Hγ widths tied to OIII_5007 in velocity space
3. **Balmer centroid tying**: same lines, centroids tied to OIII in velocity space
4. **Broad Balmer centroids**: tied to their narrow counterparts

---

## `jwspecfit.lyman_alpha` — Lyman-alpha modelling

### `igm_transmission(wave_obs_A, z_source)`

Mean IGM transmission T(λ) ∈ [0, 1] using Inoue et al. (2014).  Accounts
for Lyman-series line absorption (LAF) and damped Lyman-alpha systems (DLA).
T = 1 redward of Lyα at the source redshift.

---

### `skewed_gaussian_binned(lam_left_A, lam_right_A, amplitude, mu_A, sigma_A, skew)`

Bin-averaged skewed Gaussian using `scipy.stats.skewnorm`.  Positive
`skew` = red tail (typical for Lyα).  `amplitude` = integrated flux.

---

### `lya_model(lam_left_A, lam_right_A, z, amplitude, mu_A, sigma_A, skew)`

Full Lyα forward model: `skewed_gaussian × igm_transmission`.

---

## `jwspecfit.plotting` — Visualisation

### `plot_fit()`

```python
def plot_fit(
    result: FitResult,
    *,
    fig: Figure | None = None,
    wave_unit: str = "A",
    flux_unit: str = "fnu",
    show_residuals: bool = True,
    show_components: bool = True,
    label_lines: bool = True,
    y_pad: float = 1.3,
    exclude_wave_A: list[tuple[float, float]] | None = None,
    save_path: str | None = None,
) -> Figure:
```

Publication-quality static plot with data (steps), smooth Gaussian
components (filled curves), continuum (dashed), total model, and residuals.
Broad components are drawn with hatching.  Uncertainty is shaded as ±1σ
on each component (fractional bootstrap error).

Y-axis is automatically scaled to the tallest emission line peak.

---

### `plot_fit_interactive()`

```python
def plot_fit_interactive(
    result: FitResult,
    *,
    wave_unit: str = "A",
    flux_unit: str = "fnu",
    show_components: bool = True,
    show_residuals: bool = True,
    y_pad: float = 1.3,
    exclude_wave_A: list[tuple[float, float]] | None = None,
) -> go.Figure:
```

Interactive plotly plot with zoom, pan, and hover.  Individual line
components are rendered as smooth analytical Gaussians (not bin-averaged)
on a fine 100-point grid per line.

When `show_residuals=True`, a residual subplot is shown below the main
plot with error band and zero line.

---

### Wavelength exclusion

Both plotting functions accept `exclude_wave_A`: a list of `(lo, hi)` tuples
in Angstroms specifying wavelength regions to hide.  Excluded regions appear
as gaps — traces do not draw through them.  This does not affect the fit,
only the visualisation.

```python
fig = jwspecfit.plot_fit(result, exclude_wave_A=[(5000, 6000), (48000, 50000)])
```

---

## Constants

| Constant | Module | Value | Description |
|----------|--------|-------|-------------|
| `NII_RATIO` | `constraints` | `1 / 2.96` | [NII] 6549/6585 flux ratio |
| `BIC_DELTA_THRESHOLD` | `broad` | `6.0` | ΔBIC for model acceptance |
| `BROAD1_SIGMA_V_LO` | `broad` | `210.0` km/s | BROAD1 lower σ_v bound |
| `BROAD1_SIGMA_V_SEED` | `broad` | `420.0` km/s | BROAD1 seed σ_v |
| `BROAD1_SIGMA_V_HI` | `broad` | `850.0` km/s | BROAD1 upper σ_v bound |
| `BROAD2_SIGMA_V_LO` | `broad` | `850.0` km/s | BROAD2 lower σ_v bound |
| `BROAD2_SIGMA_V_SEED` | `broad` | `1270.0` km/s | BROAD2 seed σ_v |
| `BROAD2_SIGMA_V_HI` | `broad` | `2120.0` km/s | BROAD2 upper σ_v bound |

---

# `jwspecmcmc` API Reference

MCMC companion to `jwspecfit` for full Bayesian posterior sampling.

---

## `jwspecmcmc` — Public API

### `fit_lines()`

```python
def fit_lines(
    spectrum: Spectrum,
    z: float,
    *,
    sampler: str = "emcee",
    grating: str | None = None,
    R: float | Callable | None = None,
    lines: list[str] | None = None,
    wave_range_A: tuple[float, float] | None = None,
    deg: int = 2,
    clip_sigma: float = 2.5,
    init_from_mle: bool = True,
    prior_overrides: dict[str, Any] | None = None,
    n_walkers: int | str = "auto",
    n_steps: int = 2000,
    n_burn: int | None = None,
    n_live: int = 2000,
    n_eff: int = 10000,
    progress: bool = True,
    seed: int = 42,
    mode: str = "auto",
    n_boot_bic: int = 100,
    n_jobs: int = -1,
    snr_threshold: float = 5.0,
    bic_delta: float = 6.0,
) -> MCMCResult | MCMCBroadFitResult:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spectrum` | `Spectrum` | — | Input spectrum |
| `z` | `float` | — | Source redshift |
| `sampler` | `str` | `"emcee"` | `"emcee"` or `"nautilus"` |
| `grating` | `str \| None` | `None` | Grating name (falls back to `spectrum.grating`) |
| `R` | `float \| Callable \| None` | `None` | Resolving power (overrides grating) |
| `lines` | `list[str] \| None` | `None` | Lines to fit (default: auto-detect) |
| `wave_range_A` | `tuple \| None` | `None` | Observed wavelength range (Å) |
| `deg` | `int` | `2` | Continuum polynomial degree |
| `clip_sigma` | `float` | `2.5` | Continuum sigma-clipping threshold |
| `init_from_mle` | `bool` | `True` | Initialise walkers from least-squares MLE |
| `prior_overrides` | `dict \| None` | `None` | Per-parameter prior overrides, keyed by name (e.g. `"A_OIII_5007"`) |
| `n_walkers` | `int \| str` | `"auto"` | Emcee walkers (`"auto"` picks from n_dim and CPU cores) |
| `n_steps` | `int` | `2000` | Emcee steps |
| `n_burn` | `int \| None` | `None` | Emcee burn-in (auto if `None`) |
| `n_live` | `int` | `2000` | Nautilus live points |
| `n_eff` | `int` | `10000` | Nautilus target effective samples |
| `progress` | `bool` | `True` | Show progress bar |
| `seed` | `int` | `42` | Random seed |
| `mode` | `str` | `"auto"` | Broad mode: `"auto"` (BIC), `"off"`, `"broad1"`, `"broad2"`, `"both"` |
| `n_boot_bic` | `int` | `100` | Bootstrap iterations for BIC selection |
| `n_jobs` | `int` | `-1` | Parallel jobs for BIC bootstrap |
| `snr_threshold` | `float` | `5.0` | Minimum Hα SNR for broad fitting |
| `bic_delta` | `float` | `6.0` | ΔBIC threshold for model acceptance |

Returns `MCMCBroadFitResult` when `mode != "off"`, `MCMCResult` when `mode="off"`.

---

### `fit_with_broad()`

```python
def fit_with_broad(
    spectrum: Spectrum,
    z: float,
    *,
    sampler: str = "emcee",
    mode: str = "auto",
    ...  # same parameters as fit_lines()
) -> MCMCBroadFitResult:
```

Explicit BIC-based broad Balmer selection followed by MCMC.
Phase 1 uses `jwspecfit.fit_with_broad()` (fast least-squares) for
BIC model selection.  Phase 2 runs MCMC on the winning model.

Always returns `MCMCBroadFitResult`.

---

## `jwspecmcmc.result` — Result containers

### `MCMCLineResult`

```python
@dataclass
class MCMCLineResult:
    name: str                           # Line name
    rest_wave_A: float                  # Rest-frame wavelength (Å)
    amplitude: float                    # Median posterior amplitude
    amplitude_err: tuple[float, float]  # (lo, hi) 68% CI half-widths
    centroid_A: float                   # Median centroid (Å)
    centroid_err: tuple[float, float]   # (lo, hi) 68% CI half-widths
    sigma_A: float                      # Median sigma (Å)
    sigma_err: tuple[float, float]      # (lo, hi) 68% CI half-widths
    flux: float                         # Median flux (erg/s/cm²)
    flux_err: tuple[float, float]       # (lo, hi) 68% CI half-widths
    flux_posterior: np.ndarray          # Full flux posterior samples
    ew_A: float                         # Median rest-frame EW (Å)
    snr: float                          # flux / mean(flux_err)
```

---

### `MCMCResult`

```python
@dataclass
class MCMCResult:
    lines: dict[str, MCMCLineResult]    # Per-line posterior summaries
    flat_chains: np.ndarray             # (n_samples, 3*n_lines) full param space
    flat_chains_free: np.ndarray        # (n_samples, n_free) free param space
    flat_log_prob: np.ndarray           # Log-posterior per sample
    chains: np.ndarray | None           # (n_walkers, n_steps, n_free) or None
    params: np.ndarray                  # Median posterior (full param space)
    model_flux: np.ndarray              # Median model flux (µJy)
    continuum: np.ndarray               # Continuum (µJy)
    spectrum: Spectrum                  # Input spectrum
    line_names: list[str]               # Ordered line names
    constraints: ConstraintSet | None   # Applied constraints
    convergence: dict[str, Any]         # R-hat, ESS diagnostics
    sampler_name: str                   # "emcee" or "nautilus"
    sampler_meta: dict[str, Any]        # n_walkers, n_steps, n_burn, etc.
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `to_fit_result()` | `FitResult` | Convert median posterior to `jwspecfit.FitResult` for plotting |
| `flux_ratio_posterior(line_a, line_b)` | `np.ndarray` | Posterior samples of `flux(a) / flux(b)` |
| `doublet_flux_posterior(line_a, line_b)` | `np.ndarray` | Combined doublet flux posterior |
| `flux_upper_limit(line_name, n_sigma)` | `float` | Noise-based flux upper limit from continuum RMS |

---

### `MCMCBroadFitResult`

```python
@dataclass
class MCMCBroadFitResult:
    mcmc_result: MCMCResult     # Full MCMC posteriors for selected model
    selected_model: str         # "narrow", "broad1", "broad2", or "both"
    bic_narrow: float           # BIC for narrow-only
    bic_broad1: float           # BIC for narrow + BROAD1
    bic_broad2: float           # BIC for narrow + BROAD2
    bic_both: float             # BIC for narrow + both
```

Delegates all `MCMCResult` attributes (`.lines`, `.flat_chains`,
`.convergence`, `.to_fit_result()`, `.flux_ratio_posterior()`, etc.)
via properties for full API compatibility.

---

## `jwspecmcmc.priors` — Prior distributions

### `UniformPrior`

```python
@dataclass
class UniformPrior(Prior):
    lo: float       # Lower bound
    hi: float       # Upper bound
```

Flat prior on `[lo, hi]`.

---

### `GaussianPrior`

```python
@dataclass
class GaussianPrior(Prior):
    mean: float             # Gaussian mean
    std: float              # Standard deviation
    lo: float = -inf        # Hard lower bound
    hi: float = +inf        # Hard upper bound
```

Truncated Gaussian prior.

---

### `LogUniformPrior`

```python
@dataclass
class LogUniformPrior(Prior):
    lo: float       # Lower bound (must be > 0)
    hi: float       # Upper bound
```

Log-uniform (Jeffreys) prior on `[lo, hi]`.

---

### `PriorSet`

```python
@dataclass
class PriorSet:
    priors: list[Prior]     # One prior per free parameter
```

| Method | Returns | Description |
|--------|---------|-------------|
| `log_prior(p_free)` | `float` | Total log-prior for a free-parameter vector |
| `sample(rng)` | `np.ndarray` | Draw one sample from the joint prior |
| `n_dim` | `int` | Number of free parameters (property) |

---

### `priors_from_bounds(lb_free, ub_free, overrides=None)`

Build a `PriorSet` from parameter bounds.  Creates `UniformPrior` for
each parameter, with optional per-index overrides.

---

## `jwspecmcmc.diagnostics` — Convergence diagnostics

### `gelman_rubin(chains)`

Compute the Gelman–Rubin R-hat statistic per parameter.

| Parameter | Type | Description |
|-----------|------|-------------|
| `chains` | `np.ndarray` | Shape `(n_walkers, n_steps, n_dim)` |

Returns `np.ndarray` of R-hat values (length `n_dim`).  Values near
1.0 indicate convergence; above ~1.05 suggests poor mixing.

---

### `effective_sample_size(chains)`

Estimate ESS per parameter via FFT autocorrelation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `chains` | `np.ndarray` | Shape `(n_walkers, n_steps, n_dim)` |

Returns `np.ndarray` of ESS values (length `n_dim`).

---

### `summarise_convergence(chains)`

Return a convergence summary dict with keys: `r_hat` (array),
`ess` (array), `r_hat_max` (float), `ess_min` (float),
`converged` (bool — True if R-hat < 1.05 and ESS > 100 for all
parameters).

---

## `jwspecmcmc.plotting` — Diagnostic plots

### `plot_corner()`

```python
def plot_corner(
    result: MCMCResult,
    *,
    params: list[str] | None = None,
    truths: np.ndarray | None = None,
    quantiles: list[float] | None = None,
    **corner_kwargs,
) -> Figure:
```

Corner plot of posterior samples using the `corner` library.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `MCMCResult` | — | MCMC result |
| `params` | `list[str] \| None` | `None` | Named parameters (e.g. `["A_OIII_5007", "sigma_Ha"]`). All free if `None` |
| `truths` | `np.ndarray \| None` | `None` | True values to mark |
| `quantiles` | `list[float] \| None` | `[0.16, 0.5, 0.84]` | Quantile lines |

Returns `matplotlib.figure.Figure`.

---

### `plot_traces()`

```python
def plot_traces(
    result: MCMCResult,
    *,
    params: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
```

Trace plots of MCMC chains (emcee only — raises `ValueError` for nautilus).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `MCMCResult` | — | MCMC result with `.chains` |
| `params` | `list[str] \| None` | `None` | Named parameters to plot |
| `figsize` | `tuple \| None` | `None` | Figure size (auto-scaled if `None`) |

Returns `matplotlib.figure.Figure`.

---

### `plot_flux_posterior()`

```python
def plot_flux_posterior(
    result: MCMCResult,
    line_name: str,
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> Axes:
```

Histogram of the flux posterior for a single line, with median and 68%
credible interval marked.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `MCMCResult` | — | MCMC result |
| `line_name` | `str` | — | Line name (e.g. `"OIII_5007"`) |
| `bins` | `int` | `50` | Histogram bins |
| `ax` | `Axes \| None` | `None` | Axes to plot on (creates new if `None`) |

Returns `matplotlib.axes.Axes`.

---

# `jwspecabund` API Reference

Chemical abundance calculations from emission-line fluxes.  Accepts any
`jwspecfit` or `jwspecmcmc` result object and computes O/H, N/O, and other
element ratios via three methods: direct T_e, Bayesian forward modelling
(Cullen+25), or strong-line calibrations (Sanders+25).

**Installation:** `pip install -e ".[abund]"` (adds PyNEB >= 1.1.25).

---

## `jwspecabund` — Public API

### `compute_abundances()`

```python
def compute_abundances(
    result: Any,
    z: float,
    *,
    dust_correct: bool = True,
    dust_law: str = "salim",
    Av: float | None = None,
    method: str = "auto",
    snr_auroral: float = 3.0,
    n_mc: int = 1000,
    Te_relation: str = "desi",
    Rv: float = 3.15,
    delta: float = -0.35,
    B_bump: float = 2.27,
    forward_sampler: str = "emcee",
    forward_n_walkers: int = 32,
    forward_n_steps: int = 5000,
    forward_n_burn: int = 1000,
    forward_n_live: int = 500,
    forward_seed: int = 42,
    progress: bool = True,
    n_posterior: int = 1000,
) -> AbundanceResult:
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `FitResult \| BroadFitResult \| MCMCResult \| MCMCBroadFitResult` | — | Fitting result from `jwspecfit` or `jwspecmcmc` |
| `z` | `float` | — | Source redshift |
| `dust_correct` | `bool` | `True` | Apply dust correction via Balmer decrement |
| `dust_law` | `str` | `"salim"` | `"salim"` (Salim+18/Noll+09) or `"cardelli"` (CCM89) |
| `Av` | `float \| None` | `None` | V-band attenuation; derived from Balmer decrement if `None` |
| `method` | `str` | `"auto"` | `"auto"`, `"direct"`, `"forward"`, or `"strong_line"` |
| `snr_auroral` | `float` | `3.0` | Minimum [OIII] 4363 SNR for direct method (auto mode) |
| `n_mc` | `int` | `1000` | MC iterations for error propagation (bootstrap results) |
| `Te_relation` | `str` | `"desi"` | T_e-T_e relation: `"desi"` (DESI DR2) or `"classical"` (Garnett 1992) |
| `Rv` | `float` | `3.15` | Total-to-selective ratio for Salim law |
| `delta` | `float` | `-0.35` | Slope deviation for Salim law |
| `B_bump` | `float` | `2.27` | UV bump strength for Salim law |
| `forward_sampler` | `str` | `"emcee"` | Sampler for forward model: `"emcee"` or `"dynesty"` |
| `forward_n_walkers` | `int` | `32` | Emcee walkers for forward model |
| `forward_n_steps` | `int` | `5000` | Emcee steps for forward model |
| `forward_n_burn` | `int` | `1000` | Emcee burn-in for forward model |
| `forward_n_live` | `int` | `500` | Dynesty live points for forward model |
| `forward_seed` | `int` | `42` | Random seed for forward model |
| `progress` | `bool` | `True` | Show tqdm progress bars for MC/posterior loops |
| `n_posterior` | `int` | `1000` | Max posterior samples to propagate (MCMC results are thinned) |

Returns `AbundanceResult`.

**Method selection logic** (`method="auto"`):

1. If [OIII] 4363 SNR >= `snr_auroral` → direct T_e method
2. Otherwise → strong-line calibrations (Sanders+25)

**Input handling:**

- `FitResult` / `BroadFitResult` → Gaussian MC error propagation using `n_mc` iterations
- `MCMCResult` / `MCMCBroadFitResult` → full posterior propagation (thinned to `n_posterior` samples)
- Broad Balmer components are automatically summed with narrow components for correct hydrogen flux

---

### `AbundanceResult`

```python
@dataclass
class AbundanceResult:
    method: str                                 # "direct", "forward", or "strong_line"
    OH: float                                   # 12 + log(O/H)
    OH_err: float | tuple[float, float]         # symmetric or (lo, hi) 68% CI
    NO: float | None                            # log(N/O)
    NO_err: float | tuple[float, float] | None  # error on log(N/O)
    CO: float | None                            # log(C/O) (UV lines)
    CO_err: float | tuple[float, float] | None
    Te_high: float | None                       # T_e(O++) in K
    Te_low: float | None                        # T_e(O+/N+) in K
    ne: float | None                            # n_e in cm^-3
    Av: float | None                            # dust attenuation A_V
    ionic: dict[str, float] | None              # ionic abundances
    OH_posterior: np.ndarray | None              # full O/H posterior samples
    NO_posterior: np.ndarray | None              # full N/O posterior samples
    CO_posterior: np.ndarray | None              # full C/O posterior samples
    ratios_used: list[str] | None               # strong-line diagnostics used
    chi2: float | None                          # chi2 (strong-line)
    SO: float | None                            # log(S/O)
    NeO: float | None                           # log(Ne/O)
    ArO: float | None                           # log(Ar/O)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Human-readable multi-line summary |

---

## `jwspecabund.direct` — Direct T_e method

### `compute_ne()`

```python
def compute_ne(
    flux_line1: float,
    flux_line2: float,
    doublet: str = "SII",
    Te_guess: float = 1e4,
) -> float:
```

Compute electron density from a density-sensitive doublet via PyNEB.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flux_line1` | `float` | — | Blue member flux ([SII] 6718 or [OII] 3726) |
| `flux_line2` | `float` | — | Red member flux ([SII] 6732 or [OII] 3729) |
| `doublet` | `str` | `"SII"` | `"SII"` or `"OII"` |
| `Te_guess` | `float` | `1e4` | Temperature guess (K) |

Returns `float` — n_e in cm^-3.

---

### `compute_Te_OIII()`

```python
def compute_Te_OIII(
    flux_4363: float,
    flux_5007: float,
    flux_4959: float,
    ne: float,
) -> float:
```

Compute T_e(O++) from the [OIII] auroral/nebular ratio using PyNEB.
Uses `log=True` with `start_x=3.0, end_x=5.0` for robust root-finding
at high temperatures (T > 25,000 K).

Returns `float` — T_e in K.

---

### `compute_Te_OIII_1666()`

```python
def compute_Te_OIII_1666(
    flux_1666: float,
    flux_5007: float,
    flux_4959: float,
    ne: float,
) -> float:
```

Compute T_e(O++) from the O III] UV/optical ratio 1666/(5007+4959).
Uses the 1666 Å intercombination line as a UV auroral diagnostic when
[OIII] 4363 is unavailable.  The emissivity ratio is monotonically
increasing with T_e and more temperature-sensitive than 4363/(5007+4959)
due to the larger energy gap (7.5 eV vs 2.8 eV).

Returns `float` — T_e in K.

---

### `compute_Te_NII()`

```python
def compute_Te_NII(
    flux_5756: float,
    flux_6585: float,
    ne: float,
) -> float:
```

Compute T_e(N+) from the [NII] 5756/6585 auroral/nebular ratio.

Returns `float` — T_e in K.

---

### `Te_low_from_high()`

```python
def Te_low_from_high(Te_high: float, relation: str = "desi") -> float:
```

Derive T_e(low) from T_e(high) using an empirical T_e-T_e relation.

| Relation | Formula | Reference |
|----------|---------|-----------|
| `"desi"` | T_low = 0.648 × T_high + 3270 | DESI DR2 (arXiv:2601.02463) |
| `"classical"` | T_low = 0.7 × T_high + 3000 | Garnett (1992) |

---

### `compute_ionic_abundances()`

```python
def compute_ionic_abundances(
    fluxes: dict[str, float],
    Te_high: float,
    Te_low: float,
    ne: float,
) -> dict[str, float]:
```

Compute all available ionic abundances via PyNEB.  Requires `"HBETA"` in fluxes.

**Supported ions and temperature zones:**

| Ion | Line(s) | T_e zone |
|-----|---------|----------|
| O++/H+ | [OIII] 5007 | T_high |
| O+/H+ | [OII] 3726+3729 | T_low |
| N+/H+ | [NII] 6585 | T_low |
| C+/H+ | CII] 2324+2326 | T_low |
| S+/H+ | [SII] 6718+6732 | T_low |
| S++/H+ | [SIII] 9069 | T_mid = (T_high + T_low)/2 |
| Ne++/H+ | [NeIII] 3869 | T_high |
| Ar++/H+ | [ArIII] 7136 | T_mid |

For T > 30,000 K (beyond PyNEB H I tables), uses the Aller (1984) Hbeta
emissivity formula automatically.

Returns `dict` — e.g. `{"O+/H+": val, "O++/H+": val, "N+/H+": val, ...}`.

---

### `compute_total_abundances()`

```python
def compute_total_abundances(ionic: dict[str, float]) -> dict[str, float]:
```

Derive total element abundances from ionic abundances using Izotov+06 ICFs
(N, Ne, S, Ar) and Garnett+97 ICF (C/O when CII] 2326 is not detected).

Returns `dict` — `{"O/H": val, "N/O": val, "C/O": val, "S/O": val, "Ne/O": val, "Ar/O": val}`.

---

## `jwspecabund.forward` — Bayesian forward model

### `forward_model()`

```python
def forward_model(
    line_fluxes: dict[str, float],
    line_errors: dict[str, float],
    *,
    sampler: str = "emcee",
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burn: int = 1000,
    n_live: int = 500,
    seed: int = 42,
    progress: bool = True,
) -> dict[str, Any]:
```

Bayesian forward model (Cullen+25 approach).  Free parameters: log(T_e),
log(n_e), and log(ionic abundance) per detected ion.  Predicts line/Hbeta
flux ratios using PyNEB CEL emissivities and the Aller (1984) Hbeta formula.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_fluxes` | `dict` | — | `{line_name: flux}`, must include `"HBETA"` |
| `line_errors` | `dict` | — | `{line_name: flux_err}` |
| `sampler` | `str` | `"emcee"` | `"emcee"` or `"dynesty"` |
| `n_walkers` | `int` | `32` | Emcee walkers |
| `n_steps` | `int` | `5000` | Emcee total steps |
| `n_burn` | `int` | `1000` | Emcee burn-in |
| `n_live` | `int` | `500` | Dynesty live points |
| `seed` | `int` | `42` | Random seed |
| `progress` | `bool` | `True` | Show progress bar |

Returns `dict` with keys: `OH`, `OH_err`, `NO`, `NO_err`, `Te`, `ne`,
`ionic`, `OH_posterior`, `NO_posterior`, `NeO`, etc.

---

### `hbeta_emissivity_aller84()`

```python
def hbeta_emissivity_aller84(Te: float) -> float:
```

Hbeta volume emissivity using the Aller (1984) Case B formula:
alpha_Hb = 3.03e-14 × (T/10^4)^{-0.874} cm^3 s^-1.  Valid at all
temperatures (no 30,000 K upper limit).

Returns `float` — emissivity in erg cm^3 s^-1.

---

## `jwspecabund.strong_line` — Sanders+25 calibrations

### `sanders25_metallicity()`

```python
def sanders25_metallicity(
    line_fluxes: dict[str, float],
    line_errors: dict[str, float],
    n_mc: int = 1000,
    snr_thresh: float = 1.5,
    seed: int = 42,
    progress: bool = True,
) -> tuple[float, float, float, float, list[str], np.ndarray]:
```

Derive 12+log(O/H) via simultaneous Sanders+25 polynomial calibrations.
Uses bounded minimisation across all available diagnostic ratios.

MC error propagation perturbs each ratio by
sqrt(sigma_obs^2 + sigma_cal^2) — both measurement and calibration
scatter are included.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_fluxes` | `dict` | — | `{line_name: flux}` |
| `line_errors` | `dict` | — | `{line_name: flux_err}` |
| `n_mc` | `int` | `1000` | MC iterations |
| `snr_thresh` | `float` | `1.5` | Minimum SNR per line |
| `seed` | `int` | `42` | Random seed |
| `progress` | `bool` | `True` | Show tqdm progress bar |

Returns `(Z_best, Z_lo, Z_hi, chi2, ratios_used, Z_mc_samples)`.

**Calibrations (Sanders+25 Table 3):**

| Ratio | Definition | Coefficients | sigma_R_fit | sigma_R_int | sigma_OH_int |
|-------|-----------|-------------|-------------|-------------|--------------|
| O3 | [OIII]5007 / Hbeta | 0.852, -0.162, -1.149, -0.553 | 0.04 | 0.13 | 0.14 |
| O2 | [OII] / Hbeta | 0.172, 0.954, -0.832 | 0.03 | 0.25 | 0.22 |
| R23 | ([OIII]+[OII]) / Hbeta | 0.998, 0.053, -0.141, -0.493, -0.774 | 0.03 | 0.07 | 0.13 |
| O32 | [OIII]5007 / [OII] | 0.697, -1.245, -0.869 | 0.09 | 0.29 | 0.25 |

---

### `compute_line_ratios()`

```python
def compute_line_ratios(
    line_fluxes: dict[str, float],
    line_errors: dict[str, float],
    snr_thresh: float = 1.5,
) -> dict[str, dict[str, float]]:
```

Compute available strong-line diagnostic ratios from emission-line fluxes.
Accepts resolved (`OII_3726` + `OII_3729`) or unresolved (`OII_doublet`)
[OII] input.

Returns `{ratio_name: {"val": log10_ratio, "err": error}}`.

---

## `jwspecabund.dust` — Dust correction

### `salim_attenuation()`

```python
def salim_attenuation(
    wave_A: np.ndarray,
    Av: float,
    Rv: float = 3.15,
    delta: float = -0.35,
    B: float = 2.27,
) -> np.ndarray:
```

Salim+18/Noll+09 modified Calzetti attenuation curve.  Includes a UV bump
(Drude profile) and power-law slope deviation.

Returns A(lambda) in magnitudes.

---

### `cardelli_extinction()`

```python
def cardelli_extinction(
    wave_A: np.ndarray,
    Av: float,
    Rv: float = 3.1,
) -> np.ndarray:
```

Cardelli, Clayton & Mathis (1989) Milky Way extinction curve.

Returns A(lambda) in magnitudes.

---

### `dust_correct_fluxes()`

```python
def dust_correct_fluxes(
    line_fluxes: dict[str, tuple[float, float, float]],
    Av: float,
    law: str = "salim",
    **kwargs,
) -> dict[str, tuple[float, float]]:
```

Apply dust correction to emission-line fluxes.

| Parameter | Type | Description |
|-----------|------|-------------|
| `line_fluxes` | `dict` | `{name: (flux, flux_err, rest_wave_A)}` |
| `Av` | `float` | V-band attenuation |
| `law` | `str` | `"salim"` or `"cardelli"` |

Returns `{name: (corrected_flux, corrected_err)}`.

---

### `compute_Av_from_balmer()`

```python
def compute_Av_from_balmer(
    flux_num: float,
    flux_den: float,
    flux_num_err: float,
    flux_den_err: float,
    law: str = "salim",
    intrinsic_ratio: float = 2.86,
    wave_num_A: float = 6564.61,
    wave_den_A: float = 4862.68,
    **kwargs,
) -> tuple[float, float]:
```

Derive A_V from a Balmer decrement (default Ha/Hb; any pair supported).

Returns `(Av, Av_err)`.

---

### `compute_Av_multi_balmer()`

```python
def compute_Av_multi_balmer(
    fluxes: dict[str, float],
    errors: dict[str, float],
    law: str = "salim",
    snr_min: float = 3.0,
    **kwargs,
) -> dict[str, object]:
```

Derive A_V from every available Balmer decrement (Hgamma through H10
relative to Hbeta), returning individual and SNR-weighted average A_V
values.  Excludes Hepsilon (blended with [NeIII] 3968) and H8 (blended
with HeI 3889).

Returns `dict` with keys: `"Av_weighted"`, `"Av_err_weighted"`,
`"individual"` (per-line A_V values), `"weights"`.

---

## `jwspecabund.icf` — Ionisation correction factors

All ICFs from Izotov et al. (2006, A&A, 448, 955).

### `icf_nitrogen(O_plus, O_total)`

ICF for nitrogen (eq. 18).  N/O = ICF_N × N+/O+.

### `icf_neon(O_plus, O_total)`

ICF for neon (eq. 19).  Ne/O = ICF_Ne × Ne++/O++.

### `icf_sulfur(O_plus, O_total)`

ICF for sulfur (eq. 20).  S/O = ICF_S × (S+ + S++)/O.

### `icf_argon(O_plus, O_total)`

ICF for argon (eqs. 22/23).  Ar/O = ICF_Ar × Ar++/O++.

### `icf_carbon(O_plus, O_pp)`

ICF for carbon (Garnett+1997).  Corrects (C2+ + C3+)/O2+ for missing C+
in the low-ionisation zone: ICF_C = (O+ + O2+) / O2+.  Applied
automatically when CII] 2326 is not detected.
