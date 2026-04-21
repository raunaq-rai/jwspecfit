# jwspecfit

**Resolution-aware emission-line fitting, MCMC sampling, and chemical abundance
derivation for JWST NIRSpec spectra.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENCE)
[![Tests](https://img.shields.io/badge/tests-pytest-lightgrey.svg)](tests/)
[![Docs](https://readthedocs.org/projects/jwspecfit/badge/?version=latest)](https://jwspecfit.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19679794.svg)](https://doi.org/10.5281/zenodo.19679794)

`jwspecfit` is a coordinated suite of three Python packages covering the full
analysis chain from 1-D extracted NIRSpec spectra to element abundances. It is
built for high-redshift galaxy science — prism, medium-resolution and
high-resolution gratings, and stacked spectra — with explicit handling of the
wavelength-dependent line-spread function, broad Balmer detection, UV
absorption, Lyα radiative transfer, and the direct-T_e / Bayesian-forward /
strong-line abundance pathways.

| Package         | Purpose                                                                               |
| --------------- | ------------------------------------------------------------------------------------- |
| **`jwspecfit`** | Least-squares Gaussian fitting with bootstrap uncertainties and broad-line selection  |
| **`jwspecmcmc`**| Bayesian MCMC fitting via **NUTS**, emcee, or nautilus — full posteriors and evidence |
| **`jwspecabund`**| Chemical abundances: direct T_e (PyNEB), Bayesian forward model, strong-line calibrations |

Documentation: **https://jwspecfit.readthedocs.io/** · Source:
**https://github.com/raunaq-rai/jwspecfit**

---

## Why this exists

JWST/NIRSpec spectra span resolving powers from R ~ 30 (prism) to R ~ 2700
(high-resolution gratings), with an R(λ) that varies by an order of magnitude
across a single exposure. Naive Gaussian fitters that assume a single R, or
that sample profiles at pixel centres, introduce large biases in the prism
regime where lines are often narrower than a pixel. `jwspecfit` evaluates
bin-averaged Gaussians via the error function over each pixel's bin edges,
and picks up R(λ) from either the grating header, a user-supplied scalar, or
a callable — so stacked spectra and custom LSFs are first-class citizens.

Beyond narrow-line fitting, the suite handles the surrounding analysis
questions that typically consume the majority of spectral-fitting time:

- **BIC-based broad-Balmer detection** with four nested models (narrow,
  narrow + intermediate broad, narrow + very broad, both).
- **UV doublet kinematics and flux-ratio tying** for C IV, N V, N III], O III],
  and the density-sensitive C III] and N IV] pairs.
- **Lyα emission + DLA absorption** with IGM transmission, skewed-Gaussian
  profile, and dynesty-based N_H I retrieval.
- **Dust correction** from the multi-Balmer decrement anchored on Hβ *or* Hα
  (user's choice), using either Salim+18 or Cardelli+89 extinction.
- **Abundances** via direct T_e ([O III] 4363 *or* O III] 1666 UV auroral),
  the Cullen+25 Bayesian forward model, or Sanders+25 strong-line calibrations,
  with Martinez+25 ICFs for N/O and Izotov+06 for S, Ne, Ar.

---

## Installation

```bash
git clone https://github.com/raunaq-rai/jwspecfit.git
cd jwspecfit
pip install -e ".[dev]"
```

Optional extras (combine freely):

```bash
pip install -e ".[nuts]"     # JAX + NumPyro — the recommended MCMC backend
pip install -e ".[mcmc]"     # emcee + nautilus samplers
pip install -e ".[abund]"    # PyNEB for chemical abundances
pip install -e ".[dev,nuts,mcmc,abund]"   # everything
```

**Requirements:** Python ≥ 3.10, plus numpy, scipy, astropy, matplotlib, tqdm,
joblib, plotly. Sampler and abundance extras pull in their own dependencies.

---

## Quick start

```python
import jwspecfit

spec = jwspecfit.read_fits("spectrum.fits", z=6.0)
result = jwspecfit.fit_lines(spec, z=6.0)       # BIC broad-Balmer by default

for name, line in result.lines.items():
    if line.snr > 3:
        print(f"{name:12s}  flux = {line.flux:.2e} ± {line.flux_err:.2e}"
              f"  SNR = {line.snr:.1f}")

jwspecfit.plot_fit(result, save_path="fit.pdf")
```

Full tutorial notebooks under
[`docs/notebooks/`](docs/notebooks/) walk through prism fits, grating fits
with broad-Balmer detection, stacked-spectrum R overrides, MCMC sampling,
and all three abundance pathways.

---

## Feature highlights

### `jwspecfit` — least-squares fitting
- Resolution-aware bin-averaged Gaussian profiles (prism, gratings, stacks).
- Automatic line-list selection from grating coverage and redshift.
- Polynomial or median-filter continuum with iterative sigma-clipping.
- Bootstrap uncertainties, parallelised across cores.
- Upper-limit machinery for non-detected lines.
- Absorption-line fitting (`abs_*` line names) for Lyα-break DLAs and
  low-ionisation ISM absorbers.
- Interactive plotly plotting in addition to matplotlib.

### `jwspecmcmc` — Bayesian MCMC
- **NUTS** (via NumPyro) by default — HMC with adaptive tree depth,
  handles correlated posteriors efficiently.
- `emcee` and `nautilus` backends for comparison and evidence estimation.
- Custom priors per parameter (`UniformPrior`, `GaussianPrior`, `LogUniformPrior`).
- Gelman-Rubin R-hat and ESS convergence diagnostics.
- Sample-level flux-ratio posteriors (`result.flux_ratio_posterior(...)`).
- HDF5 save/load round-trip and conversion to `FitResult` for plotting.

### `jwspecabund` — chemical abundances
- **Dust:** multi-Balmer A_V with `balmer_anchor="HBETA"` (default) or
  `"Ha"`. Salim+18 attenuation or Cardelli+89 extinction.
- **Electron density:** [S II], [O II], C III], N IV] across three
  ionisation zones, with manual overrides per zone.
- **Method auto-selection:** direct T_e when [O III] 4363 is detected;
  UV fallback via O III] 1666 when 4363 is unavailable; Sanders+25
  strong-line otherwise; Bayesian forward model on request.
- **Nitrogen:** Martinez+25 ICFs with per-tier locking (`icf_tier=...`).
- **Carbon:** C II] 2326 multiplet for C+/H+ when available.
- **Lyα escape fraction** from the Balmer-decrement prediction of
  intrinsic Lyα, with Monte Carlo propagation of A_V uncertainty.

---

## Documentation

The full user guide and API reference is on ReadTheDocs:

> **https://jwspecfit.readthedocs.io/**

Local Markdown originals live in [`docs/`](docs/):

- [`docs/api.md`](docs/api.md) — API reference
- [`docs/abundance_methodology.md`](docs/abundance_methodology.md) — full
  derivation of the abundance pipeline
- [`docs/references.md`](docs/references.md) — literature references

To build the docs locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

---

## Example notebooks

Worked examples in [`docs/notebooks/`](docs/notebooks/):

| Notebook                    | Description                                        |
| --------------------------- | -------------------------------------------------- |
| `01_prism_fit`              | Prism fitting, save/load, plotting                 |
| `02_grating_broad`          | G395M grating with broad-Balmer detection          |
| `03_stacked_spectrum`       | Stacked spectrum with custom R                     |
| `04_mcmc_prism`             | MCMC fitting (emcee)                               |
| `05_mcmc_grating`           | MCMC for grating spectra                           |
| `06_mcmc_stack`             | MCMC for stacked spectra                           |
| `07_abundances`             | Direct-T_e, forward, strong-line                   |
| `08_nitrogen`               | Nitrogen abundance diagnostics                     |
| `08b_nitrogen_combined`     | N/O with ICF tiers                                 |
| `09_uv_abundances`          | UV emission + absorption line fitting              |

> Some notebooks may lag behind the main branch during active development.
> The test suite is the authoritative source of truth for current behaviour.

---

## Tests

```bash
pytest tests/ -v
```

The suite covers the fitter, MCMC engines, abundance pathways, dust
correction, Lyα escape fraction, DLA retrieval, UV doublet constraints,
and the resolution model.

---

## Citation

If you use `jwspecfit` in academic work, please cite it using the DOI
below (also available via GitHub's "Cite this repository" button, which
reads the `CITATION.cff` in the repo root):

> **DOI:** [10.5281/zenodo.19679794](https://doi.org/10.5281/zenodo.19679794)

This is a concept-DOI — it always resolves to the latest Zenodo-archived
release. Follow the DOI link for per-version DOIs if you need to pin a
specific release.

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

Once a companion paper is on arXiv/accepted, cite both — the software
DOI for the exact code version used, the paper for the methods.

### Citing the methods

`jwspecfit` wraps established methods — please also cite the originals as
appropriate:

- **Salim+18** attenuation (dust correction)
- **Cardelli+89** extinction (MW law)
- **Storey & Hummer 1995** (H I recombination coefficients via PyNEB)
- **Osterbrock & Ferland 2006** (Case B intrinsic ratios)
- **Sanders et al. 2025** (strong-line calibrations)
- **Cullen et al. 2025** (Bayesian forward model)
- **Martinez+25** (N/O ICFs)
- **Izotov+06** (S, Ne, Ar ICFs)
- **Garnett 1992** / **DESI DR2** (T_e–T_e relations)

Full citation keys are in [`docs/references.md`](docs/references.md).

---

## Contributing

Contributions, bug reports, and feature requests are welcome via GitHub
issues and pull requests. Please include:

- Python and package versions (`python -c "import jwspecfit; print(jwspecfit.__version__)"`).
- A minimal reproducer for bugs.
- Tests for new features (`pytest tests/` must pass).

Code style is enforced with `ruff`:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

---

## Licence

MIT — see [LICENCE](LICENCE). You are free to use, modify, and redistribute
with attribution.

---

## Contact

Raunaq Rai — PhD student, UCL Department of Physics & Astronomy

- GitHub: [@raunaq-rai](https://github.com/raunaq-rai)
- Email: raunaq.rai.22@ucl.ac.uk
