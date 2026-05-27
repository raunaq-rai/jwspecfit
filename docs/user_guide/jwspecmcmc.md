# `jwspecmcmc` — Bayesian MCMC fitting

```{tip}
**This is the authors' recommended fitter for science-quality results.**
The least-squares `jwspecfit` engine is best for quick looks and initial
guesses; for published measurements and any quantity that needs faithful
uncertainties, use `jwspecmcmc`.
```

`jwspecmcmc` replaces the bootstrap uncertainties of `jwspecfit` with
full Bayesian posterior sampling. You get asymmetric errors, parameter
correlations, flux-ratio posteriors, and (with nautilus) the Bayesian
evidence.

## Default sampler: NUTS

The **No-U-Turn Sampler** (via NumPyro) is the default backend. It is a
Hamiltonian Monte Carlo variant with adaptive tree depth that navigates
high-dimensional correlated posteriors efficiently:

```python
import jwspecmcmc

result = jwspecmcmc.fit_lines(
    spec, z=6.0,
    sampler="nuts",
    n_warmup=500,
    n_samples_nuts=2000,
    n_chains=4,
    target_accept_prob=0.8,
)
```

NUTS requires JAX + NumPyro — install via `pip install -e ".[nuts]"`.

## Alternative samplers

**emcee** — affine-invariant ensemble:

```python
result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="emcee", n_steps=2000)
```

**nautilus** — importance nested sampling, useful for evidence:

```python
result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="nautilus", n_live=2000)
```

## Results

```python
line = result.lines["OIII_5007"]
line.flux                    # median flux
line.flux_err                # (lower, upper) 68% CI half-widths
line.flux_posterior          # 1-D array of posterior samples

ratio = result.flux_ratio_posterior("OIII_5007", "HBETA")  # sample-by-sample
```

## Custom priors

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

## Convergence

```python
result.convergence   # {"R_hat": {...}, "ESS": {...}}
```

Gelman-Rubin R̂ should sit below 1.05 and ESS above a few hundred per
parameter for a trusted fit.

## Plotting

```python
jwspecmcmc.plot_corner(result, params=["A_OIII_5007", "A_HBETA"])
jwspecmcmc.plot_traces(result)
jwspecmcmc.plot_flux_posterior(result, "OIII_5007")
```

## HDF5 persistence

```python
jwspecmcmc.save_mcmc_result(result, "mcmc.h5")
loaded = jwspecmcmc.load_mcmc_result("mcmc.h5")

# Convert back for jwspecfit plotting:
fit_result = result.to_fit_result()
jwspecfit.plot_fit(fit_result)
```

## Broad Balmer + MCMC

`jwspecmcmc.fit_lines` delegates BIC broad-Balmer model selection to
`jwspecfit.fit_with_broad`, then MCMC-samples the winning model. Set
`fit_balmer_broad=True` (and/or `fit_oiii_broad`, `fit_hei_broad`) to
opt in; the default is narrow-only.

## Full parameter reference

These tables document every argument of `fit_lines`. They apply equally
to `fit_with_broad`, which takes the same arguments but always performs
the BIC broad-selection step (and does not expose `init_from_mle`).
Required arguments have no default.

### Core

| Argument | Default | What it does |
|---|---|---|
| `spectrum` | *(required)* | Input `Spectrum` — observed wavelength, flux, and error arrays. |
| `z` | *(required)* | Source redshift. Sets each observed line position via `lambda_obs = lambda_rest * (1 + z)`. |
| `sampler` | `"nuts"` | Posterior backend: `"nuts"` (NumPyro HMC), `"emcee"` (ensemble), or `"nautilus"` (nested sampling — also returns the evidence). |

### Instrument & line selection

| Argument | Default | What it does |
|---|---|---|
| `grating` | `None` | NIRSpec grating name (e.g. `"G395H"`) setting the resolution curve. If `None`, estimated from the pixel spacing. |
| `R` | `None` | Resolving power as a scalar or callable `R(lambda_um)`. Overrides `grating`; if `None`, taken from `grating`/pixels. |
| `lines` | `None` | Names of lines to fit (keys of `jwspecfit.lines.REST_LINES_A`). If `None`, lines falling in range at `z` are auto-selected. |
| `wave_range_A` | `None` | `(lo, hi)` observed-frame wavelength window in Å to restrict the fit. If `None`, the full spectrum is used. |

### Continuum

| Argument | Default | What it does |
|---|---|---|
| `deg` | `2` | Degree of the polynomial continuum. |
| `clip_sigma` | `2.5` | Sigma-clip threshold used when fitting the continuum. |
| `moving_average` | `False` | Continuum model: `False` → polynomial of degree `deg`; `True` → 75-pixel running-median filter; `int` → running-median filter of that window. |

### Initialisation

| Argument | Default | What it does |
|---|---|---|
| `init_from_mle` | `True` | Seed the sampler from a fast least-squares MLE (`jwspecfit.fit_lines`). Recommended — speeds convergence. *(Not exposed by `fit_with_broad`.)* |

### NUTS controls (`sampler="nuts"`)

| Argument | Default | What it does |
|---|---|---|
| `n_warmup` | `500` | Warm-up (tuning) iterations per chain before samples are kept; adapts step size and mass matrix. |
| `n_samples_nuts` | `2000` | Posterior draws kept per chain after warm-up. Total samples = `n_chains * n_samples_nuts`. |
| `n_chains` | `6` | Number of independent chains run in parallel; enables the R-hat diagnostic. |
| `target_accept_prob` | `0.8` | Target acceptance probability for step-size adaptation. Raise toward 0.95 if you see divergences. |
| `max_tree_depth` | `10` | Maximum NUTS tree depth; caps leapfrog steps per iteration at `2 ** max_tree_depth`. |

### emcee controls (`sampler="emcee"`)

| Argument | Default | What it does |
|---|---|---|
| `n_walkers` | `"auto"` | Number of ensemble walkers. `"auto"` chooses from the parameter count and CPU cores (must be ≥ `2 * n_dim`). |
| `n_steps` | `2000` | Number of steps per walker. |
| `n_burn` | `None` | Burn-in steps discarded. If `None`, estimated from the integrated autocorrelation time. |

### nautilus controls (`sampler="nautilus"`)

| Argument | Default | What it does |
|---|---|---|
| `n_live` | `2000` | Number of live points. |
| `n_eff` | `10000` | Target effective posterior sample size. |

### Broad-component selection (BIC; opt-in)

| Argument | Default | What it does |
|---|---|---|
| `fit_balmer_broad` | `False` | Run BIC selection for a broad Balmer component (narrow / intermediate / very-broad / both), gated by Hα SNR. |
| `fit_oiii_broad` | `False` | Independent BIC test for a broad [OIII] 4959/5007 component (outflow signature), gated by [OIII] 5007 SNR. |
| `fit_hei_broad` | `False` | Independent BIC test for a broad He I component shared across all observable He I lines, gated by the best narrow He I SNR. |
| `n_boot_bic` | `100` | Bootstrap iterations for BIC model selection; `0` uses a single-point BIC. Only used when a broad flag is set. |
| `n_jobs` | `-1` | Parallel workers for the BIC bootstrap; `-1` uses all cores. |
| `snr_threshold` | `5.0` | Minimum Hα SNR to attempt Balmer broad fitting. |
| `oiii_snr_threshold` | `5.0` | Minimum [OIII] 5007 SNR to attempt [OIII] broad fitting. |
| `hei_snr_threshold` | `5.0` | Minimum He I SNR (best narrow He I line) to attempt He I broad fitting. |
| `bic_delta` | `6.0` | Minimum ΔBIC by which a more complex model must beat the simpler one to be selected. |

### Kinematic ties & doublets

| Argument | Default | What it does |
|---|---|---|
| `tie_balmer_to_oiii` | `True` | Tie narrow Balmer (and [NII]) widths and centroids to [OIII] 5007 in velocity space. |
| `tie_uv_doublets` | `True` | Tie UV doublet kinematics and fix resonance-line flux ratios. Recommended for stacked / poorly resolved spectra. |
| `tie_uv_centroids` | `True` | When `tie_uv_doublets` is on, tie each secondary centroid to its primary in velocity space; if `False`, secondary centroids are free. |
| `tie_uv_widths` | `True` | Tie the widths of the UV intercombination lines (CIII], NIV, NIII, SiIII) to a single shared velocity dispersion. |
| `niv_doublet_ratio` | `None` | If set, fix the N IV] flux ratio `F(1483) / F(1486)` to this value; `None` leaves the amplitudes free. |
| `ciii_doublet_ratio` | `None` | If set, fix the C III] flux ratio `F(1909) / F(1907)` to this value; `None` leaves the amplitudes free. |

### Line-bound overrides

| Argument | Default | What it does |
|---|---|---|
| `sigma_overrides` | `None` | Per-line width bounds `{line_name: (sigma_lo, sigma_hi)}` in Å, overriding the automatic grating-based bounds. |
| `centroid_overrides` | `None` | Per-line centroid bounds `{line_name: (mu_lo, mu_hi)}` in Å, overriding the automatic velocity-based bounds. |
| `sigma_factor` | `1.0` | Multiplier on the upper line-width bound; use > 1 for stacked spectra with broadened lines. |
| `centroid_vmax` | `500.0` | Maximum centroid offset, in km/s, allowed for each line. |
| `centroid_max_sigma` | `1.0` | Extra resolution-aware cap on narrow-line centroid offsets: `min(centroid_vmax-based, centroid_max_sigma * sigma_instrument)`. |

### Priors & misc

| Argument | Default | What it does |
|---|---|---|
| `prior_overrides` | `None` | Per-parameter prior overrides keyed by name (`A_<line>`, `mu_<line>`, `sigma_<line>`), e.g. `{"A_OIII_5007": GaussianPrior(1e-17, 1e-18, 0, 1e-15)}`. |
| `progress` | `True` | Show a progress bar during sampling. |
| `seed` | `42` | Random seed for initialisation and sampling. |
