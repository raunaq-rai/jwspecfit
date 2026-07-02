<p align="center">
  <img src="docs/_static/logos/logo.svg" alt="jwspecfit logo" width="128"/>
</p>

# jwspecfit

**Emission-line fitting, MCMC, and chemical abundances for JWST NIRSpec spectra.**

[![PyPI](https://img.shields.io/pypi/v/jwspecfit.svg)](https://pypi.org/project/jwspecfit/)
[![Python](https://img.shields.io/pypi/pyversions/jwspecfit.svg)](https://pypi.org/project/jwspecfit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENCE)
[![Tests](https://img.shields.io/badge/tests-pytest-lightgrey.svg)](tests/)
[![Docs](https://readthedocs.org/projects/jwspecfit/badge/?version=latest)](https://jwspecfit.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19679793.svg)](https://doi.org/10.5281/zenodo.19679793)

Three packages, one pipeline — from 1-D NIRSpec spectra to element abundances.

| Package             | What it does                                                       |
| ------------------- | ------------------------------------------------------------------ |
| **`jwspecfit`**     | Resolution-aware Gaussian line fitting with bootstrap errors       |
| **`jwspecmcmc`** ⭐ | Bayesian MCMC fitting (NUTS · emcee · nautilus) — **recommended**  |
| **`jwspecabund`**   | Chemical abundances — direct T_e · forward model · strong-line     |

> **Recommended fitter:** for science-quality results the authors recommend the Bayesian MCMC fitter `jwspecmcmc` (full posteriors and faithful uncertainties). Use the least-squares `jwspecfit` engine for quick looks, initial guesses, and BIC model selection.

## Key features

- **Resolution-aware line profiles**: bin-averaged Gaussians via erf — correct for prism, gratings, and stacks.
- **Broad-Balmer detection**: BIC-based selection across four nested models.
- **UV doublets**: flux-ratio and kinematic tying for C IV, N V, N III], O III], C III], N IV].
- **Lyα + DLA**: skewed Gaussian + IGM transmission + dynesty N_HI retrieval.
- **Dust correction**: multi-Balmer A_V anchored on Hβ **or** Hα, Salim+18 or Cardelli+89 curves.
- **Abundances**: direct T_e ([O III] 4363 or UV 1666), Cullen+25 forward model, Sanders+25 strong-line.
- **ICFs**: Martinez+25 (N/O) · Izotov+06 (S, Ne, Ar) · Garnett+97 (C/O).
- **Lyα escape fraction** with Monte Carlo propagation of A_V uncertainty.

## Install

```bash
pip install jwspecfit
```

Or with all optional extras (MCMC backends, abundances, DLA fitter):

```bash
pip install "jwspecfit[nuts,mcmc,abund,dla]"
```

For development (editable install from source):

```bash
git clone https://github.com/raunaq-rai/jwspecfit.git
cd jwspecfit
pip install -e ".[dev,nuts,mcmc,abund]"
```

Requires Python ≥ 3.10. See the [installation guide](https://jwspecfit.readthedocs.io/en/latest/installation.html) for individual extras.

## Example

```python
import numpyro
numpyro.set_host_device_count(6)   # before sampling: one host device per chain

import jwspecfit, jwspecmcmc, jwspecabund

spec   = jwspecfit.read_fits("spectrum.fits", z=6.0)
result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="nuts")   # Bayesian NUTS fit

# Assess convergence before trusting the posteriors
conv = result.convergence
print(f"converged={conv['converged']}  "        # True iff R-hat<1.05 & ESS>100 (all params)
      f"R-hat_max={conv['r_hat_max']:.3f}  "     # worst R-hat — quote this
      f"ESS_min={conv['ess_min']:.0f}")

abund = jwspecabund.compute_abundances(result, z=6.0)
print(abund.summary())
```

> For a quick, non-Bayesian look you can swap the fit for the least-squares
> engine — `result = jwspecfit.fit_lines(spec, z=6.0)` — but it has no chains, so
> `result.convergence` is empty; use the NUTS fit above for science-quality
> posteriors and a meaningful $\hat{R}$.

## Line fitting

The recommended fitter is the Bayesian `jwspecmcmc` engine sampled with NumPyro
NUTS:

```python
import numpyro
numpyro.set_host_device_count(6)  # required: run this before sampling

import jwspecmcmc

result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="nuts")
```

> **Important:** call `numpyro.set_host_device_count(6)` (matching the number of
> CPU cores you want to use) **before** any sampling. NumPyro otherwise sees a
> single host device, which forces multiple chains to run sequentially and can
> prevent NUTS from working properly. Set it once, at the top of your script or
> notebook, before importing/calling the fitter.

### Model and parameters

The continuum is first estimated with a low-order polynomial fitted to the
spectrum after masking the regions around known emission lines and iteratively
rejecting positive outliers, and is then subtracted. Each emission line is
modelled as a resolution-aware Gaussian, integrated over the wavelength range
spanned by each spectral pixel (via the error function) so that the model is
compared to the data on the same footing for the prism, the gratings, and
stacked spectra alike. The line model is the sum of these Gaussians on top of
the fitted continuum.

For every line three parameters are fit: an amplitude (the integrated line
flux), a central wavelength, and a Gaussian width. Physically related
transitions are tied to reduce the dimensionality and to respect atomic physics:
the Balmer series can share a common velocity and width with the bright
rest-optical lines (e.g. [O III] λ5007), doublets share kinematics, and
unresolved doublets are held at fixed or bounded flux ratios. Centroids and
widths are bounded to physically motivated ranges set by the instrumental
resolution and a maximum velocity offset from the systemic redshift.

### Sampling and inference

Inference is fully Bayesian: full posteriors are obtained via Markov Chain Monte
Carlo using a Gaussian likelihood (identical to the weighted chi-squared used by
the least-squares engine, so both fitters treat the data identically) with
uniform priors over the parameter bounds. Sampling uses the No-U-Turn Sampler
(NUTS), a self-tuning variant of Hamiltonian Monte Carlo. The likelihood is
JIT-compiled in JAX and differentiated automatically, so the sampler uses the
gradient to propose distant, high-acceptance moves and explores the
high-dimensional parameter space far more efficiently than random-walk methods.

During warmup the step size is adapted to a target acceptance probability
(`target_accept_prob = 0.8`) and the trajectory length is bounded by a maximum
tree depth (`max_tree_depth = 10`), with the No-U-Turn criterion terminating
each trajectory automatically; there is no hand-tuned proposal scale. By default
the chain is run for 500 warmup (adaptation) steps followed by 2000 posterior
samples, initialised at a validated finite-likelihood seed (a least-squares
pre-fit) so that every transition starts in bounds.

Uncertainties on the line fluxes are taken as the 16th–84th percentiles of the
posterior, and all quantities are reported as posterior medians with the
corresponding credible intervals. The fit yields full posterior chains for every
parameter, including a posterior distribution of the integrated flux of each
line; these per-line flux posteriors are resampled by the downstream abundance
routines to propagate measurement uncertainties through the (non-linear)
abundance calculation.

### Convergence and inspecting the chains

Convergence is assessed with the Gelman–Rubin statistic ($\hat{R}$) and the
effective sample size (ESS), computed for every free parameter. These are
attached to the result in the `convergence` dictionary:

```python
conv = result.convergence
conv["converged"]   # bool: True if R-hat < 1.05 and ESS > 100 for all params
conv["r_hat_max"]   # worst (largest) R-hat across parameters
conv["ess_min"]     # smallest effective sample size
conv["r_hat"]       # per-parameter R-hat array
conv["ess"]         # per-parameter ESS array
```

The Gelman–Rubin statistic compares the within-chain and between-chain variance,
so it is only meaningful with **two or more chains**. Run multiple chains (and
set `numpyro.set_host_device_count` to at least that many cores, as above) to get
a usable $\hat{R}$:

```python
import numpyro
numpyro.set_host_device_count(4)

result = jwspecmcmc.fit_lines(spec, z=6.0, sampler="nuts", n_chains=4)
print(result.convergence["r_hat_max"], result.convergence["ess_min"])
```

The chains themselves can be inspected directly. The raw, per-chain samples are
available as `result.chains` with shape `(n_chains, n_steps, n_free)` — ideal for
trace plots to check that the chains are well-mixed and stationary:

```python
import matplotlib.pyplot as plt

chains = result.chains                       # (n_chains, n_steps, n_free)
fig, ax = plt.subplots()
for c in range(chains.shape[0]):
    ax.plot(chains[c, :, 0], alpha=0.6)      # trace of the first free parameter
ax.set_xlabel("step"); ax.set_ylabel("parameter 0")
```

The flattened posterior across all chains is `result.flat_chains`, of shape
`(n_samples, 3 * n_lines)`, ordered as all amplitudes, then all central
wavelengths, then all widths, following `result.line_names`. So for line `i`:

```python
nL = len(result.line_names)
amp  = result.flat_chains[:, i]            # amplitude (= integrated flux)
cen  = result.flat_chains[:, nL + i]       # central wavelength
wid  = result.flat_chains[:, 2 * nL + i]   # width
```

For convenience, each line also carries its full flux posterior directly:

```python
result.lines["OIII_5007"].flux_posterior   # 1-D array of flux samples
```

Two other backends are available for cross-checks (`sampler="emcee"`, an
affine-invariant ensemble sampler, and `sampler="nautilus"`, an importance-nested
sampler), but NUTS is the default and recommended choice.

## Abundances

`jwspecabund` turns a set of fitted line fluxes into gas-phase chemical
abundances. Fluxes are dust-corrected from the Balmer decrement before any
abundance is computed, and measurement uncertainties are propagated by resampling
the line-flux posteriors.

```python
import jwspecabund

abund = jwspecabund.compute_abundances(result, z=6.0)
print(abund.summary())
```

### Direct (T_e-based) metallicity

The direct method measures the gas-phase oxygen abundance from the electron
temperature, with all atomic physics handled by **PyNEB**. The temperature is set
by an auroral-to-nebular line ratio — primarily [O III] λ4363/(λ4959+λ5007), or
the UV intercombination line O III] λ1666 relative to λ5007 when λ4363 is
unavailable — giving the temperature of the high-ionisation (O²⁺) zone. The
low-ionisation (O⁺) temperature is obtained either from [N II] λ5755/λ6585 or
from a temperature–temperature relation. The electron density comes from a
density-sensitive doublet appropriate to the zone ([S II] λ6716/λ6731 or
[O II] λ3726/λ3729 at low ionisation, with C III] λ1907/λ1909 and N IV]
λ1483/λ1486 available for higher-ionisation zones).

The ionic abundances are then computed at the relevant temperature and density:
O⁺/H⁺ from the [O II] λ3726,3729 doublet and O²⁺/H⁺ from [O III] λ5007, each
relative to a hydrogen Balmer line. The total oxygen abundance is the direct sum

```
O/H = O⁺/H⁺ + O²⁺/H⁺
```

so no ionisation-correction factor is required for the dominant ionisation
stages.

### Strong-line metallicity

When the auroral lines are too faint for a direct temperature, metallicity is
estimated from bright strong-line ratios using the **Sanders et al. (2025)**
calibrations. The implemented diagnostics are **O3** ([O III] λ5007/Hβ),
**O2** ([O II]/Hβ), **R23** (([O III] λ5007 + [O II])/Hβ), and **O32**
([O III] λ5007/[O II]). Available diagnostics are solved simultaneously against
the calibration, with Monte Carlo error propagation.

### C/O and N/O

Carbon and nitrogen abundance ratios are built from whichever ionic stages are
detected:

- **C/O** uses carbon from C II] λ2326, C III] λ1907,1909, and C IV λ1548,1551,
  with oxygen from [O II] λ3726,3729 and [O III] λ5007.
- **N/O** uses nitrogen from [N II] λ6585, N III] λ1750, N IV] λ1486, and
  N V λ1240, again with oxygen from [O II] λ3726,3729 and [O III] λ5007.

The code automatically uses the ions present in the fit, so UV-only spectra
(C III], C IV, N III], N IV]) and rest-optical spectra ([N II], [O II], [O III])
are both supported.

## Documentation

Usage guides, API reference, and methodology:
**https://jwspecfit.readthedocs.io/**

Worked examples: [`docs/notebooks/`](docs/notebooks/).

## Tests

```bash
pytest tests/
```

## Citation

If you use `jwspecfit` in your research, **please cite it**. Choose
whichever format your reference manager or journal prefers.

> ### 📖 DOI: [10.5281/zenodo.19679793](https://doi.org/10.5281/zenodo.19679793)
>
> Concept DOI — always resolves to the latest Zenodo-archived release.

### Plain text

> Rai, R. (2026). *jwspecfit: Resolution-aware emission-line fitting,
> MCMC sampling, and chemical abundances for JWST NIRSpec* (v1.0.1).
> Zenodo. <https://doi.org/10.5281/zenodo.19679793>

### BibTeX

```bibtex
@software{rai_jwspecfit,
  author       = {Rai, Raunaq},
  title        = {{jwspecfit}: Resolution-aware emission-line fitting,
                  MCMC sampling, and chemical abundances for JWST NIRSpec},
  year         = {2026},
  version      = {1.0.1},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19679793},
  url          = {https://doi.org/10.5281/zenodo.19679793},
}
```

### Other formats

APA · Chicago · IEEE · Harvard · MLA · CSL-JSON · DataCite XML are all
available from the [Zenodo record page](https://zenodo.org/records/19679793)
(Export panel on the right).

GitHub's **"Cite this repository"** button (top-right of the repo page)
reads [`CITATION.cff`](CITATION.cff) and produces APA/BibTeX on the fly.

### Pinning a specific version

The concept DOI above always points to the latest release. If a paper
needs to cite the *exact* code version used for reproducibility, pick
the per-version DOI from the "Versions" list on the Zenodo page.

## Licence

MIT — see [LICENCE](LICENCE).

## AI usage declaration

AI-based coding assistants were used during the development of this
package (for example to help draft code, tests, and documentation). All
contributions were reviewed, tested, and validated by the author, who
takes full responsibility for the scientific correctness of the code.
