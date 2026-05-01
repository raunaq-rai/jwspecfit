# Quick start

This page walks through a minimal end-to-end session: load a spectrum,
fit emission lines, compute abundances.

## 1. Load a spectrum

`jwspecfit` accepts a JWST `x1d` FITS file, a numpy/dict in memory, or a
stacked `.npz`:

```python
import jwspecfit

# From a JWST x1d FITS file (grating and R read from header):
spec = jwspecfit.read_fits("spectrum.fits", z=6.0)

# From arrays (wavelength in μm, flux / error in μJy):
spec = jwspecfit.read_dict(
    {"wave": wave_um, "flux": flux_ujy, "err": err_ujy},
    z=6.0,
)

# From a stacked .npz with custom resolving power:
spec = jwspecfit.read_npz("stack.npz", z=2.0, R=1000)
```

## 2. Fit emission lines

```python
result = jwspecfit.fit_lines(spec, z=6.0)   # default: BIC broad-Balmer
```

By default, `fit_lines` performs:

1. Line-list selection from the grating coverage and redshift.
2. Polynomial continuum subtraction with iterative sigma-clipping.
3. Simultaneous resolution-aware Gaussian fit to all observable lines.
4. 1000-iteration bootstrap for flux uncertainties.
5. BIC-based selection of broad-Balmer components.

Inspect detected lines:

```python
for name, line in result.lines.items():
    if line.snr > 3:
        print(f"{name:12s}  flux={line.flux:.2e}±{line.flux_err:.2e}  SNR={line.snr:.1f}")
```

Plot the fit:

```python
jwspecfit.plot_fit(result, save_path="fit.pdf")

# Or interactive (plotly) — zoomable with hover info:
fig = jwspecfit.plot_fit_interactive(result)
fig.show()
```

## 3. MCMC sampling

Swap the least-squares fit for a Bayesian one with full posteriors:

```python
import jwspecmcmc

mc = jwspecmcmc.fit_lines(
    spec, z=6.0,
    sampler="nuts",       # "nuts" (default), "emcee", or "nautilus"
    n_warmup=500,
    n_samples_nuts=2000,
    n_chains=4,
)

line = mc.lines["OIII_5007"]
print(line.flux)                 # median
print(line.flux_err)             # (lower, upper) 68% CI half-widths
print(mc.convergence)            # R̂ and ESS per parameter
```

Sample-level flux-ratio posteriors come for free:

```python
r32 = mc.flux_ratio_posterior("OIII_5007", "HBETA")
```

## 4. Chemical abundances

Feed either a `FitResult` (from `jwspecfit`) or an `MCMCResult` (from
`jwspecmcmc`) into `compute_abundances`:

```python
import jwspecabund

abund = jwspecabund.compute_abundances(
    result, z=6.0,
    method="auto",            # "direct" / "strong_line" / "forward"
    balmer_anchor="HBETA",    # or "Ha" if Hα is your best Balmer line
    dust_law="salim",         # or "cardelli"
)

print(abund.summary())
print(abund.OH)               # 12 + log(O/H)
print(abund.NO)               # log(N/O)
```

`compute_abundances` internally:

1. Derives A_V from the multi-Balmer decrement (or takes your `Av=` value).
2. Corrects every line flux at its own wavelength.
3. Measures electron density across three ionisation zones.
4. Selects direct-T_e, strong-line, or forward-model based on line availability.
5. Applies ICFs (Martinez+25 for N/O, Izotov+06 for S, Ne, Ar).

## 5. DLA fitting (optional)

For high-redshift spectra showing damped Lyα absorption, fit the
column density via nested sampling:

```bash
pip install jwspecfit[dla]
```

```python
result = jwspecfit.fit_NHI(
    wave_A=spec.wave_A,
    flux=spec.flux_ujy,
    flux_err=spec.err_ujy,
    z=0.0,                       # 0 for rest-frame stacks
    fit_range_A=(1100., 2000.),
    R=spec.R,
    n_live=200,
)
print(result.summary())          # log(N_HI), beta_UV, Sigma_HI, log(Z)
result.plot()
```

See the [user guide](user_guide/jwspecfit.md#dla-fitting) for the full
option list.

## Quick interactive view

To open a FITS or NPZ spectrum directly in plotly without fitting:

```python
fig = jwspecfit.plot_spectrum_interactive("spectrum.fits", z=6.0)
fig.show()
```

## Next steps

- Worked notebooks: see `docs/notebooks/`.
- Full API reference: [API index](api.md).
- Abundance methodology in detail:
  [Abundance methodology](abundance_methodology.md).
