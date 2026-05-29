# `jwspecfit` — least-squares fitting

`jwspecfit` is the least-squares engine: it fits resolution-aware,
bin-averaged Gaussian profiles to all observable emission lines
simultaneously, and returns a `FitResult` with fluxes, SNRs, centroids,
widths, and equivalent widths.

```{note}
For science-quality uncertainties the authors recommend the Bayesian MCMC
fitter, [`jwspecmcmc`](jwspecmcmc.md). Use this least-squares engine for quick
looks, initial guesses, and fast BIC model selection.
```

## Loading spectra

```python
import jwspecfit

spec = jwspecfit.read_fits("spectrum.fits", z=6.0)
```

`read_fits` accepts both the JWST `SPEC1D` BinTable convention
(default) and arbitrary 1-D FITS files: it auto-detects column names
(`wave/wavelength/lam/lambda/loglam`, `flux`, `err/sigma/ivar`),
reads units from the `TUNIT` keywords (μm/Å/nm, μJy/mJy/Jy,
erg/s/cm²/Å) and falls through to image HDUs with WCS keywords
(`CRVAL1`, `CDELT1`, `CRPIX1`, `CTYPE1`, `CUNIT1`, `BUNIT`).
Override the auto-detection with `hdu=`, `wave_col=`, `flux_col=`,
`err_col=` if needed.

See the {doc}`quickstart guide <../quickstart>` for the full list of
readers (`read_fits`, `read_dict`, `read_npz`).

### Quick interactive view

To pop a spectrum open in plotly without first running a fit, call
`plot_spectrum_interactive` directly on a path or a `Spectrum`:

```python
fig = jwspecfit.plot_spectrum_interactive("spectrum.fits", z=6.0)
fig = jwspecfit.plot_spectrum_interactive("stack.npz", rest_frame=True)
fig = jwspecfit.plot_spectrum_interactive(spec, flux_unit="flam")
fig.show()
```

The plot has zoom/pan, hover read-outs, and a ±1σ band whenever an
error array is available.

## Fitting

```python
result = jwspecfit.fit_lines(spec, z=6.0)
```

Common options:

```python
result = jwspecfit.fit_lines(
    spec, z=6.0,
    lines=["OIII_5007", "OIII_4959", "HBETA"],   # restrict line list
    wave_range_A=(35000, 50000),                  # wavelength window
    wave_windows_A=[(13000, 17000), (35000, 52000)],  # multi-window
    n_boot=500,                                    # bootstrap iterations
    mode="auto",                                   # broad-Balmer mode
    niv_doublet_ratio=None,                        # fix NIV ratio
    ciii_doublet_ratio=None,                       # fix CIII] ratio
)
```

## Inspecting results

Each `LineResult` contains:

| Field        | Description                                         |
| ------------ | --------------------------------------------------- |
| `flux`       | Integrated line flux (erg / s / cm²)                |
| `flux_err`   | 1σ uncertainty from bootstrap                       |
| `snr`        | Signal-to-noise ratio                               |
| `ew_A`       | Rest-frame equivalent width (Å)                     |
| `amplitude`  | Peak amplitude                                      |
| `centroid_A` | Observed centroid wavelength (Å)                    |
| `sigma_A`    | Gaussian sigma (Å, incl. instrumental broadening)   |

Upper limits for undetected lines:

```python
ul = result.flux_upper_limit("OIII_4363", n_sigma=3.0)
uls = result.flux_upper_limits(n_sigma=3.0)
```

## Broad Balmer components

Four nested models are tested and selected via BIC when `mode="auto"`:

1. Narrow only.
2. Narrow + intermediate broad (FWHM ~ 500–2000 km s⁻¹).
3. Narrow + very broad (FWHM ~ 2000–5000 km s⁻¹).
4. Narrow + both broad components.

```python
result = jwspecfit.fit_with_broad(spec, z=2.0, n_boot_bic=100)
print(result.selected_model)     # "narrow" | "broad1" | "broad2" | "both"
```

## UV and absorption lines

Lines prefixed with `abs_` are fit as negative Gaussians (low-ionisation
ISM absorption, DLA flanks):

```python
uv_lines = [
    "NV_1", "NV_2", "CIV_1", "CIV_2", "HEII_1640",
    "CIII]_1907", "CIII]",
    "abs_SiII1260", "abs_CII1334", "abs_SiIV1394", "abs_SiIV1403",
]
result = jwspecfit.fit_lines(spec, z=6.0, lines=uv_lines)
```

## DLA fitting

`jwspecfit.fit_NHI` performs a **joint Bayesian fit of the local DLA
column density and the IGM damping wing**, following the methodology
of Pollock et al. (2026) Eq. 4:

$$F(\lambda) = \big[F_0 (\lambda/\lambda_\text{piv})^{\beta_\text{UV}}
+ \mathrm{Ly}\alpha(\lambda)\big]
\;\exp(-\tau_\text{DLA})\;\exp(-\tau_\text{IGM})$$

The posterior over `(log_F0, β_UV, log_NHI)` (and optionally `x_HI`,
plus a Lyα emission profile) is sampled with `dynesty` nested
sampling. The Voigt profile is exact (Faddeeva); the IGM term uses
the Miralda-Escudé (1998) closed form integrated from `igm_z_min`
(default 5.3, Bosman+22) up to the source redshift.

Install the optional dependency first:

```bash
pip install jwspecfit[dla]   # installs dynesty
```

### Standard call

```python
import jwspecfit

spec = jwspecfit.read_npz("stack_zgt6_Muv19_21_prism.npz", z=0.0, R=400.0)

result = jwspecfit.fit_NHI(
    spec.wave_A, spec.flux_ujy, spec.err_ujy,
    z=0.0,                       # 0 for rest-frame stacks
    R=spec.R,                    # LSF convolution
    fit_x_HI=False,              # set True for individual z>6 spectra
    n_live=400, seed=42,
)
print(result.summary())
result.plot()
```

The defaults match Pollock+26: priors `log_NHI ∈ [18, 24]`,
`x_HI ∈ [0, 1]`, `β_UV ∈ [-4, 0]`, `log_F0` auto-centred ±3 dex
around the data; fit window 1216–3000 Å rest-frame; emission lines
masked with `mask_width_A=20 Å`; Lyα emission optionally added
*inside* the exp(−τ) factor.

### Useful options

| Argument          | Purpose                                                       |
| ----------------- | ------------------------------------------------------------- |
| `fit_x_HI`        | Sample IGM neutral fraction jointly (Pollock+26 z > 6 mode)    |
| `igm_z_min`       | Lower-z bound of the IGM integration (default 5.3)            |
| `b_kms`           | Voigt Doppler parameter (default 30 km s⁻¹; insensitive at log N_HI > 20.3) |
| `prior_log_NHI`, `prior_x_HI`, `prior_beta_UV`, `prior_log_F0` | Override the uniform prior bounds |
| `mask_lines`, `mask_width_A` | Default `True`, 20 Å (PRISM-appropriate)              |
| `fit_range_A`     | Default (1216, 3000) Å rest-frame                             |
| `mask_regions_A`  | Extra rest-frame regions to ignore                             |
| `Av`, `dust_law`  | Pre-correct the spectrum for foreground attenuation            |
| `fit_lya`, `lya_params` | Joint or fixed Lyα emission profile                       |
| `n_live`, `dlogz`, `seed` | dynesty controls                                       |

The returned `DLAResult` carries posterior samples for every free
parameter (`result.samples` is a dict of arrays), median ± 16/84 %
percentiles, log-evidence, and a 95th-percentile **upper limit** on
`log_NHI` (`result.log_NHI_upper95`, `result.is_upper_limit=True`)
for spectra where the wing is unconstrained — Pollock+26 do exactly
this for 21 of their 48 sources.

### D_Lyα equivalent-width statistic (Heintz+25)

When the N_HI–β degeneracy bites at low spectral resolution, the more
robust diagnostic is the rest-frame equivalent width of the Lyα
absorption integrated 1180–1350 Å (Heintz+25, Eq. 1). It is
β-prior–independent to first order and directly comparable across
the high-z DLA literature.

```python
DLya = jwspecfit.compute_D_Lya(spec.wave_A, spec.flux_ujy, spec.err_ujy, z=0.0)
print(f"D_Lyα = {DLya['D_Lya']:.1f} ± {DLya['D_Lya_err']:.1f} Å")
# Interpretation (Heintz+25 §3.1):
#   D_Lyα < 0    → Lyα emitter
#   ~0 to 35 Å   → weak / no absorption
#   35 to 50 Å   → IGM-dominated regime
#   > 50 Å       → DLA-dominated (log N_HI ≳ 21.5)
```

### Caveats at PRISM resolution

The N_HI–β posterior covariance is **r ≈ −0.96** at PRISM R~100
(Keating+25), so log N_HI shifts by ~1 dex when the β prior is
tightened from `[-4, 0]` to a star-forming-galaxy range like
`[-3, -1.5]`. For a stack at z > 6 with PRISM data, **report D_Lyα
as the primary statistic** and treat any specific log N_HI as a
prior-conditioned posterior. Robust column densities require
medium-resolution gratings (G140M / G235M, R ≳ 1000).

## Plotting

```python
fig = jwspecfit.plot_fit(result, save_path="fit.pdf")
fig = jwspecfit.plot_fit_interactive(result)   # plotly
```

See the [Plotting & visualisation](../plotting) page for the full set of
static and interactive helpers, line markers, and the 2-D + 1-D viewer.

## Save / load

```python
jwspecfit.save_result(result, "fit.npz")
loaded = jwspecfit.load_result("fit.npz")
jwspecfit.export_lines_txt(result, "lines.txt")
```

## Why resolution-aware?

NIRSpec prism resolving power R varies from ~30 at 1 μm to ~300 at 5 μm,
so narrow emission lines are often **sub-pixel** in the blue and
**super-pixel** in the red. Evaluating a Gaussian at pixel centres
biases the recovered flux systematically downward for under-sampled
lines. `jwspecfit` computes the analytic integral of each Gaussian
across each pixel's bin edges using the error function, eliminating this
bias.
