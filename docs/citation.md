# Citation

If you use `jwspecfit` in your research, **please cite the paper**:

> ## 📄 Singh Rai & Roberts-Borsani (2026), [arXiv:2608.10063](https://arxiv.org/abs/2608.10063)
>
> *Unveiling and Characterising Ubiquitous Nitrogen Enhancement in
> 6 ≤ z ≤ 10 Galaxies with JWST Spectroscopy*
>
> [ADS record](https://ui.adsabs.harvard.edu/abs/2026arXiv260810063S)

This paper is the reference for the software — there is no separate
software citation.

## Plain text

> Singh Rai, R. & Roberts-Borsani, G. (2026). *Unveiling and Characterising
> Ubiquitous Nitrogen Enhancement in 6 ≤ z ≤ 10 Galaxies with JWST
> Spectroscopy*. arXiv e-prints, arXiv:2608.10063.
> <https://arxiv.org/abs/2608.10063>

## BibTeX

```bibtex
@ARTICLE{rai2026,
       author = {{Singh Rai}, Raunaq and {Roberts-Borsani}, Guido},
        title = "{Unveiling and Characterising Ubiquitous Nitrogen Enhancement in $6 \leq z \leq 10$ Galaxies with JWST Spectroscopy}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies},
         year = 2026,
        month = aug,
          eid = {arXiv:2608.10063},
        pages = {arXiv:2608.10063},
archivePrefix = {arXiv},
       eprint = {2608.10063},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260810063S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Other formats

APA · Chicago · IEEE · Harvard · MLA · CSL-JSON · AASTeX are all available
from the
[ADS export panel](https://ui.adsabs.harvard.edu/abs/2026arXiv260810063S/exportcitation).

GitHub's **"Cite this repository"** button on the repository page reads
[`CITATION.cff`](https://github.com/raunaq-rai/jwspecfit/blob/main/CITATION.cff)
and returns the same paper.

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
