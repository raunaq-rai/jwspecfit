# Changelog

This project does not yet follow a formal release schedule. Key
additions are listed below in reverse chronological order. Commit
history on GitHub is the authoritative source.

## 1.0.1 — 2026-04-21

Patch release to retry Zenodo DOI minting. The v1.0.0 deposit failed
validation because `CITATION.cff` contained a placeholder ORCID; the
ORCID field has been removed and the citation metadata cleaned up. No
functional code changes.

## Unreleased

- **`jwspecabund`:** new `balmer_anchor` option on
  `compute_abundances` and `compute_Av_multi_balmer` — A_V can now be
  derived with Hα as the anchor (using Hβ/Hα, Hγ/Hα, Hδ/Hα, H9/Hα,
  H10/Hα) instead of the default Hβ anchor.
- **`jwspecabund`:** `ciii_doublet_ratio` parameter to fix the C III]
  ratio from a user-supplied density.
- **`jwspecabund`:** `niv_doublet_ratio` parameter to fix the N IV]
  ratio from the C III] density.
- **`jwspecabund`:** Cardelli extinction curve extended to the far-UV;
  DLA fitter overhaul.
- **`jwspecabund`:** `snr_balmer` parameter for the Balmer-line SNR
  floor used in A_V derivation.
- **`jwspecfit`:** hard IGM cutoff blueward of Lyα (Gunn–Peterson
  trough), smoothed by the LSF convolution.

## 1.0.0 — initial public development release

- `jwspecfit` — resolution-aware least-squares fitting with bootstrap
  uncertainties and BIC broad-Balmer selection.
- `jwspecmcmc` — NUTS / emcee / nautilus MCMC fitting.
- `jwspecabund` — direct-T_e, forward-model, and strong-line
  abundance pathways.
