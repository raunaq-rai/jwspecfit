# `jwspecmcmc` — Bayesian MCMC fitting

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
`mode="off"` to force narrow-only MCMC.
