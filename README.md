# jwspecfit

**Emission-line fitting, MCMC, and chemical abundances for JWST NIRSpec spectra.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENCE)
[![Tests](https://img.shields.io/badge/tests-pytest-lightgrey.svg)](tests/)
[![Docs](https://readthedocs.org/projects/jwspecfit/badge/?version=latest)](https://jwspecfit.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19679794.svg)](https://doi.org/10.5281/zenodo.19679794)

Three packages, one pipeline — from 1-D NIRSpec spectra to element abundances.

| Package             | What it does                                                       |
| ------------------- | ------------------------------------------------------------------ |
| **`jwspecfit`**     | Resolution-aware Gaussian line fitting with bootstrap errors       |
| **`jwspecmcmc`**    | Bayesian MCMC fitting (NUTS · emcee · nautilus)                    |
| **`jwspecabund`**   | Chemical abundances — direct T_e · forward model · strong-line     |

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
git clone https://github.com/raunaq-rai/jwspecfit.git
cd jwspecfit
pip install -e ".[dev,nuts,mcmc,abund]"
```

Requires Python ≥ 3.10. See the [installation guide](https://jwspecfit.readthedocs.io/en/latest/installation.html) for individual extras.

## Example

```python
import jwspecfit, jwspecabund

spec   = jwspecfit.read_fits("spectrum.fits", z=6.0)
result = jwspecfit.fit_lines(spec, z=6.0)
abund  = jwspecabund.compute_abundances(result, z=6.0)

print(abund.summary())
```

## Documentation

Usage guides, API reference, and methodology:
**https://jwspecfit.readthedocs.io/**

Worked examples: [`docs/notebooks/`](docs/notebooks/).

## Tests

```bash
pytest tests/
```

## Citation

> **DOI:** [10.5281/zenodo.19679794](https://doi.org/10.5281/zenodo.19679794)

Export BibTeX from the [Zenodo record](https://zenodo.org/records/19679794), or use GitHub's "Cite this repository" button (reads `CITATION.cff`).

## Licence

MIT — see [LICENCE](LICENCE).
