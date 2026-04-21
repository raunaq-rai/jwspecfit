# `jwspecfit` — least-squares fitting

`jwspecfit` is the least-squares engine: it fits resolution-aware,
bin-averaged Gaussian profiles to all observable emission lines
simultaneously, and returns a `FitResult` with fluxes, SNRs, centroids,
widths, and equivalent widths.

## Loading spectra

```python
import jwspecfit

spec = jwspecfit.read_fits("spectrum.fits", z=6.0)
```

See the {doc}`quickstart guide <../quickstart>` for the full list of
readers (`read_fits`, `read_dict`, `read_npz`).

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

## Plotting

```python
fig = jwspecfit.plot_fit(result, save_path="fit.pdf")
fig = jwspecfit.plot_fit_interactive(result)   # plotly
```

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
