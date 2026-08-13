# jwspecfit

**Resolution-aware emission-line fitting, MCMC sampling, and chemical
abundance derivation for JWST NIRSpec spectra.**

`jwspecfit` is a coordinated suite of three Python packages covering the
complete analysis path from 1-D extracted NIRSpec spectra to element
abundances:

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} `jwspecfit`
:link: user_guide/jwspecfit
:link-type: doc

Least-squares Gaussian line fitting with resolution-aware, bin-averaged
profiles; broad-Balmer BIC selection; UV doublet tying; Lyα and DLA
handling; bootstrap uncertainties.
:::

:::{grid-item-card} `jwspecmcmc`
:link: user_guide/jwspecmcmc
:link-type: doc

Bayesian MCMC replacement for the bootstrap fitter. **NUTS** (via
NumPyro) by default, with emcee and nautilus backends. Full posteriors,
asymmetric errors, flux-ratio posteriors, R̂ and ESS diagnostics.
:::

:::{grid-item-card} `jwspecabund`
:link: user_guide/jwspecabund
:link-type: doc

Chemical abundances: direct T_e (PyNEB), Bayesian forward model
(Cullen+25), strong-line calibrations (Sanders+25); multi-Balmer dust
correction anchored on Hα or Hβ; Martinez N/O and C/O ICFs; Lyα escape
fraction.
:::
::::

```{tip}
**Recommended method.** For science-quality measurements the authors
recommend the Bayesian MCMC fitter, [`jwspecmcmc`](user_guide/jwspecmcmc),
over the least-squares `jwspecfit` bootstrap. MCMC returns full posteriors,
asymmetric uncertainties, and parameter covariances, and propagates them
correctly into derived quantities (flux ratios, abundances). Use the
least-squares engine for quick looks, initial guesses, and BIC model
selection; use `jwspecmcmc` for published results.
```

```{note}
**Citing these packages.** Please cite Singh Rai & Roberts-Borsani (2026),
[arXiv:2608.10063](https://arxiv.org/abs/2608.10063) — see {doc}`citation`
for the BibTeX entry.
```

---

## Getting started

```{toctree}
:caption: Getting started
:maxdepth: 2

installation
quickstart
```

## jwspecfit

```{toctree}
:caption: jwspecfit
:maxdepth: 2

user_guide/jwspecfit
api_reference/jwspecfit
```

## jwspecmcmc

```{toctree}
:caption: jwspecmcmc
:maxdepth: 2

user_guide/jwspecmcmc
api_reference/jwspecmcmc
```

## jwspecabund

```{toctree}
:caption: jwspecabund
:maxdepth: 2

user_guide/jwspecabund
api_reference/jwspecabund
```

## Visualisation

```{toctree}
:caption: Visualisation
:maxdepth: 2

plotting
```

## Methodology

```{toctree}
:caption: Methodology
:maxdepth: 2

abundance_methodology
abundance
references
```

## Project

```{toctree}
:caption: Project
:maxdepth: 1

citation
changelog
```

---

## Indices and search

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
