# jwspecfit — Project Instructions

## What This Is
JWST NIRSpec emission-line fitting package. Fits Gaussian profiles to spectral
lines with proper resolution handling (prism, grating, stacked spectra).

## Architecture
- `src/jwspecfit/` — installable package (src-layout)
- `data/` — real JWST spectra for testing (3 FITS + 1 stacked .npz + Inoue2014 table)
- `tests/` — pytest suite

## Key Design Decisions
- All wavelengths internally in **Angstroms** for line fitting; µm at I/O boundaries
- Gaussian profiles are **bin-averaged via erf** (not sampled at pixel centres)
- Resolution: accepts `grating="prism"` or `R=100` (numeric) for flexibility
- NII doublet: flux ratio 1/2.96, kinematics tied
- Broad components: BIC model selection with ΔBIC ≥ 6 threshold
- Lyα: skewed Gaussian × Inoue+2014 IGM transmission

## Conventions
- Follow PEP 8, ruff formatter, NumPy-style docstrings, type hints
- `scipy.optimize.least_squares` for fitting (not lmfit — kept as dependency for future use)
- Auto-commit after every valid change
