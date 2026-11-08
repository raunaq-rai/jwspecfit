# Installation

## From PyPI (recommended)

```bash
pip install jwspecfit
```

This installs everything: the least-squares fitter, all MCMC backends
(NUTS/JAX, emcee, nautilus), PyNeb for abundances, `dynesty` for the DLA
fitter, and `ipywidgets`/`nbformat` so progress bars and interactive plotly
figures work in Jupyter.  No extras are needed.

## From source (development)

```bash
git clone https://github.com/raunaq-rai/jwspecfit.git
cd jwspecfit
pip install -e ".[dev]"
```

The `-e` (editable) install is recommended during development so local
edits are picked up without reinstalling.

## Extras

Only `dev` (`pytest`, `ruff`) adds anything.  The old `nuts`, `mcmc`,
`abund` and `dla` extras are kept as empty aliases, so existing
`pip install "jwspecfit[nuts,mcmc,abund]"` commands still work.

## Requirements

- **Python:** ≥ 3.10
- **Installed:** `numpy`, `scipy`, `astropy`, `matplotlib`, `tqdm`, `joblib`,
  `plotly`, `pandas`, `jax`, `jaxlib`, `numpyro`, `emcee`, `nautilus-sampler`,
  `corner`, `h5py`, `pyneb`, `dynesty`, `ipywidgets`, `nbformat`

## Verifying the install

```python
import jwspecfit, jwspecmcmc, jwspecabund
print(jwspecfit.__version__)
print(jwspecmcmc.__version__)
print(jwspecabund.__version__)
```

Run the test suite:

```bash
pytest tests/ -v
```

## Conda

If you maintain a conda environment (recommended for JWST work so the
pipeline and CRDS tooling sit alongside `jwspecfit`), create it once:

```bash
conda create -n jwst python=3.11
conda activate jwst
pip install -e ".[dev]"
```
