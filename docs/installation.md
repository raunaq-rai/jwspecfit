# Installation

## From PyPI (recommended)

```bash
pip install jwspecfit
```

Or with all optional extras (MCMC backends, abundances, DLA fitter):

```bash
pip install "jwspecfit[nuts,mcmc,abund,dla]"
```

## From source (development)

```bash
git clone https://github.com/raunaq-rai/jwspecfit.git
cd jwspecfit
pip install -e ".[dev]"
```

The `-e` (editable) install is recommended during development so local
edits are picked up without reinstalling.

## Optional extras

`jwspecfit` ships with optional dependency groups. Combine them as
needed (works for both PyPI and editable installs):

```bash
pip install "jwspecfit[nuts]"     # JAX + NumPyro — recommended MCMC backend
pip install "jwspecfit[mcmc]"     # emcee + nautilus + corner + h5py
pip install "jwspecfit[abund]"    # PyNEB for chemical abundances
pip install "jwspecfit[dev,nuts,mcmc,abund]"   # everything
```

| Extra   | Installs                                  | Enables                                          |
| ------- | ----------------------------------------- | ------------------------------------------------ |
| `dev`   | `pytest`, `ruff`                          | Test suite and linting                           |
| `nuts`  | `jax`, `jaxlib`, `numpyro`                | NUTS HMC sampler (default backend for `jwspecmcmc`) |
| `mcmc`  | `emcee`, `nautilus-sampler`, `corner`, `h5py` | `emcee` and `nautilus` backends, HDF5 save/load  |
| `abund` | `pyneb`                                   | Direct-T_e abundance computations in `jwspecabund` |

## Requirements

- **Python:** ≥ 3.10
- **Core:** `numpy`, `scipy`, `astropy`, `matplotlib`, `tqdm`, `joblib`, `plotly`
- **Optional:** as listed above

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
pip install -e ".[dev,nuts,mcmc,abund]"
```
