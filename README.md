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

If you use `jwspecfit` in your research, **please cite it**. Choose
whichever format your reference manager or journal prefers.

> ### 📖 DOI: [10.5281/zenodo.19679794](https://doi.org/10.5281/zenodo.19679794)
>
> Concept DOI — always resolves to the latest Zenodo-archived release.

### Plain text

> Rai, R. (2026). *jwspecfit: Resolution-aware emission-line fitting,
> MCMC sampling, and chemical abundances for JWST NIRSpec* (v1.0.1).
> Zenodo. <https://doi.org/10.5281/zenodo.19679794>

### BibTeX

```bibtex
@software{rai_jwspecfit,
  author       = {Rai, Raunaq},
  title        = {{jwspecfit}: Resolution-aware emission-line fitting,
                  MCMC sampling, and chemical abundances for JWST NIRSpec},
  year         = {2026},
  version      = {1.0.1},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19679794},
  url          = {https://doi.org/10.5281/zenodo.19679794},
}
```

### Other formats

APA · Chicago · IEEE · Harvard · MLA · CSL-JSON · DataCite XML are all
available from the [Zenodo record page](https://zenodo.org/records/19679794)
(Export panel on the right).

GitHub's **"Cite this repository"** button (top-right of the repo page)
reads [`CITATION.cff`](CITATION.cff) and produces APA/BibTeX on the fly.

### Pinning a specific version

The concept DOI above always points to the latest release. If a paper
needs to cite the *exact* code version used for reproducibility, pick
the per-version DOI from the "Versions" list on the Zenodo page.

## Licence

MIT — see [LICENCE](LICENCE).
