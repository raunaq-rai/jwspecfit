# Citation

`jwspecfit` has not yet been published. If you use it in academic work,
please cite it as unreleased software using the metadata in
[`CITATION.cff`](https://github.com/raunaq-rai/jwspecfit/blob/main/CITATION.cff),
which GitHub renders as a "Cite this repository" button and is
consumable by standard reference managers.

## How to get a citeable DOI

Because the package is not yet associated with a journal paper, the
recommended way to obtain a permanent citation handle is a **Zenodo**
DOI:

1. Sign in to [Zenodo](https://zenodo.org/) with your GitHub account
   and enable the `raunaq-rai/jwspecfit` repository on the
   [GitHub integration page](https://zenodo.org/account/settings/github/).
2. On GitHub, create a release (e.g. `v1.0.0`). Zenodo automatically
   archives that tag, mints a DOI, and returns a concept-DOI that
   always resolves to the latest release.
3. Paste the DOI back into `CITATION.cff` (`doi:` field) and into the
   BibTeX entry below.

## BibTeX (placeholder — fill in DOI once minted)

```bibtex
@software{rai_jwspecfit,
  author       = {Rai, Raunaq},
  title        = {{jwspecfit}: Resolution-aware emission-line fitting,
                  MCMC sampling, and chemical abundances for JWST NIRSpec},
  year         = {2026},
  version      = {1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/raunaq-rai/jwspecfit},
}
```

## Citing the methods

`jwspecfit` wraps published methods — please cite the originals:

- **Salim+18** — attenuation curve
- **Cardelli+89** — MW extinction
- **Storey & Hummer 1995** — H I recombination via PyNEB
- **Osterbrock & Ferland 2006** — Case B ratios
- **Sanders+25** — strong-line calibrations
- **Cullen+25** — Bayesian forward model
- **Martinez+25** — N/O ICFs
- **Izotov+06** — S, Ne, Ar ICFs
- **Garnett 1992** / **DESI DR2** — T_e–T_e relations

Full citation keys in {doc}`references`.
