# Citation

A paper describing `jwspecfit` is **in preparation** (Rai & Roberts-Borsani,
in prep.) and is expected to be submitted shortly. **This page will be updated
with the full reference and a bibliographic DOI on submission.** Until then, if
you use `jwspecfit` in your research **please cite the software archive** via
its Zenodo DOI, in whichever format your reference manager or journal prefers.

> ## 📖 DOI: [10.5281/zenodo.19679793](https://doi.org/10.5281/zenodo.19679793)
>
> Concept DOI — always resolves to the latest Zenodo-archived release.

## Plain text

> Rai, R. (2026). *jwspecfit: Resolution-aware emission-line fitting,
> MCMC sampling, and chemical abundances for JWST NIRSpec* (v1.1.7).
> Zenodo. <https://doi.org/10.5281/zenodo.19679793>

## BibTeX

```bibtex
% Companion paper (in preparation) — this entry will be updated on submission.
@article{rai_jwspecfit_paper,
  author = {Rai, Raunaq Singh and Roberts-Borsani, Guido},
  title  = {{jwspecfit}: Emission-line fitting, MCMC sampling, and chemical
            abundances for JWST NIRSpec spectra},
  year   = {2026},
  note   = {in preparation},
}

% Software archive.
@software{rai_jwspecfit,
  author       = {Rai, Raunaq},
  title        = {{jwspecfit}: Resolution-aware emission-line fitting,
                  MCMC sampling, and chemical abundances for JWST NIRSpec},
  year         = {2026},
  version      = {1.1.7},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19679793},
  url          = {https://doi.org/10.5281/zenodo.19679793},
}
```

## Other formats

APA · Chicago · IEEE · Harvard · MLA · CSL-JSON · DataCite XML are all
available from the
[Zenodo record page](https://zenodo.org/records/19679793) — click the
**Export** panel on the right-hand side.

GitHub's **"Cite this repository"** button on the repository page reads
[`CITATION.cff`](https://github.com/raunaq-rai/jwspecfit/blob/main/CITATION.cff)
and produces APA / BibTeX on the fly.

## Pinning a specific version

The concept DOI above always points to the latest release. If a paper
needs to cite the *exact* code version used for reproducibility, pick
the per-version DOI from the **Versions** list on the Zenodo record
page.

## Citing the methods

`jwspecfit` wraps published methods — please cite the originals:

- **NUTS / NumPyro** (Phan, Pradhan & Jankowiak 2019; Bingham et al. 2019) and
  **JAX** (Bradbury et al. 2018) — default MCMC sampler
- **Salim+18** — attenuation curve
- **Cardelli+89** — MW extinction
- **PyNEB** (Luridiana, Morisset & Shaw 2015); **Storey & Hummer 1995** — H I recombination
- **Osterbrock & Ferland 2006** — Case B ratios
- **Sanders+25** — strong-line calibrations
- **Cullen+25** — Bayesian forward model
- **Martinez+25** — N/O ICFs and O32 / N43 log(U) calibrations (Zorayda
  Martinez et al., "Under Pressure", arXiv:2510.21960)
- **Martinez, in prep. (2026)** — C²⁺/O²⁺ → C/O ICF (Zorayda Martinez et al.)
- **Izotov+06** — S, Ne, Ar ICFs; **Garnett+97** — legacy C/O ICF
- **Garnett 1992** — T_e–T_e relations (3-tier zones, the default)

Full citation keys in {doc}`references`.
