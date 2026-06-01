# Changelog

This project does not yet follow a formal release schedule. Key
additions are listed below in reverse chronological order. Commit
history on GitHub is the authoritative source.

## Unreleased

### Behaviour changes

- **Multi-Balmer A_V now uses only lines bluer than the anchor.**
  `balmer_anchor="HBETA"` (default) uses Hγ/Hβ, Hδ/Hβ, H9/Hβ, H10/Hβ and
  **excludes Hα/Hβ**; `balmer_anchor="Ha"` uses every other Balmer line
  (all are bluer than Hα). Previously the Hβ anchor also included Hα/Hβ.
  Anchor on Hα to include Hα; on Hβ to use only the bluer Balmer series.

### New

- **`balmer_pair` option in `compute_abundances`** (and the public
  `compute_Av_balmer_pair`). Forces the A_V derivation onto a single
  Balmer decrement, e.g. `balmer_pair=("Ha", "HBETA")` for Hα/Hβ only,
  instead of the multi-line fit — useful to avoid low-SNR Balmer lines.
  Overrides `balmer_anchor`/`snr_balmer` for the A_V step; ignored when
  `Av` is supplied directly.

## 1.1.3 — 2026-06-01

### Behaviour changes

- **O²⁺ and Ne²⁺ are decoupled from the N IV] density.** `T_e([O III])`
  and the O²⁺/H⁺ and Ne²⁺/H⁺ abundances are now evaluated at the
  intermediate-ionisation density (C III] λ1907/1909, with a low-zone
  fallback) instead of the high-ionisation N IV] density. The [O III]
  5007/Hβ and [Ne III] 3869/Hβ abundances are density-insensitive below
  ~10⁴–10⁵ cm⁻³, and C III] (24–48 eV) overlaps the O²⁺ zone (35–55 eV),
  whereas N IV] (47–77 eV) traces more highly-ionised gas and can
  spuriously spike — dragging `T_e` down and inflating O/H. N³⁺ and C³⁺
  keep the high-ionisation density, so the Martinez+25 ICF 5
  ((N²⁺+N³⁺)/O²⁺) still uses the correct density for each nitrogen ion.
- **log(U) from O32 / N43 now takes an electron-density input.** The
  Martinez+25 O32 and N43 log(U) diagnostics are evaluated at the
  measured density rather than a fixed default.

### Documentation

- New **Plotting & visualisation** section consolidating all static
  (matplotlib) and interactive (plotly) plotting helpers across the
  suite, with a backend-choice guide and line-marker reference.

### Fixes

- `DLAResult.plot()` no longer emits a `tight_layout` warning on the
  two-panel (data + residual) figure (now uses constrained layout).

## 1.1.2 — 2026-05-29

### Behaviour changes

- **Martinez+2025 ICF/log(U) bounds are now enforced by rejection, not
  extrapolation.** Inputs outside the calibration domain
  (`log(O32)`, `log(N43)`, `Z/Z_sun`, or the resulting `log(U)`) are set
  to `NaN` instead of being extrapolated or clipped to the boundary, so
  uncalibrated values no longer enter the reported N/O.
- **Direct-`T_e` MC and MCMC loops resample to the requested count.** The
  loop keeps drawing until `n_mc` / `n_posterior` *in-bounds* N/O draws
  are collected (capped at 20× attempts; a `WARNING` is logged and N/O is
  left under-sampled if an object is centred outside the bounds).
- **O/H is decoupled from the N/O bounds.** Only N/O is gated on the
  Martinez calibration; O/H, `T_e`, and the C/O, S/O, Ne/O, Ar/O ratios
  (which do not use the Martinez ICF) are recorded for every drawn
  sample. As a result the O/H posterior generally holds **more** finite
  draws than the N/O posterior — the two arrays are independent and need
  not share a length. See `abundance_methodology` §8.2 and §11.3.

## 1.1.0 — 2026-05-20

### New public API

- **`jwspecfit.fit_redshift`** — strong-line redshift fitter spanning
  z = 0–20, with auto-detected spectral resolution (`spec.R` →
  `grating` → pixel-spacing fallback).
- **`jwspecfit.show_lines()`** — discovery helper for the line
  database; `plot_spectrum_interactive` and `plot_2d_1d` accept
  `add_lines=[…]` for ad-hoc marker additions.
- **`jwspecfit.plot_2d_1d`** — matplotlib panel showing the 2D SCI
  image and 1D extraction for a single FITS file, auto-scaled to the
  brightest emission line in view.
- **`jwspecmcmc.fit_with_broad` + `jwspecfit.fit_hei_broad`** —
  two-tier BIC selection for HeI broad components (analogous to the
  Balmer pipeline).

### Behaviour changes

- Broad-component control moved from `mode="…"` to dedicated boolean
  kwargs `fit_balmer_broad=` and `fit_oiii_broad=`; both default to
  `False`. Two-tier BIC selects between off / single broad / double
  broad / both, with kinematics tied across [O III] doublet
  components.
- `fit_redshift` excludes [N II] and [S II] from the default line
  list (avoids false matches in low-metallicity high-z spectra).
- Resolution-aware centroid bounds: σ-instrument-scaled limits on
  narrow-line centroid drift to prevent collapse into a neighbouring
  line.
- `plot_spectrum_interactive` overhaul — emission-line markers,
  staggered labels, step error band, multi-spectrum legend, custom
  `add_lines=[…]`, configurable y-range.

### Line database additions

- **He II 4200** (Pickering 11→4, vacuum 4201.013 Å)
- **He I**: 4027, 4145, 6680, 7065 (vacuum-converted from NIST air values)
- **[O I] 6302**
- **[Ar III] 7138**

### `jwspecabund` additions

- `balmer_anchor` option on `compute_abundances` and
  `compute_Av_multi_balmer` — A_V can be derived with Hα as the
  anchor (using Hβ/Hα, Hγ/Hα, Hδ/Hα, H9/Hα, H10/Hα) instead of the
  default Hβ anchor.
- `ciii_doublet_ratio` to fix the C III] ratio from a user-supplied
  density.
- `niv_doublet_ratio` to fix the N IV] ratio from the C III] density.
- Cardelli extinction curve extended to the far-UV; DLA fitter
  overhaul.
- `snr_balmer` parameter for the Balmer-line SNR floor used in A_V
  derivation.

### Fixes

- Centroid-bounds collapse for [O III] broad components in the MCMC
  engine (mirrored from the LS fitter).
- HEI_7067 marker label corrected to air-convention "HeI 7065"
  (vacuum wavelength 7067.138 Å unchanged).
- Hard IGM cutoff blueward of Lyα (Gunn–Peterson trough), smoothed by
  the LSF convolution.

### Docs / branding

- Adopted a hex + Gaussian logo; wired into RTD sidebar (`html_logo`),
  browser favicon (`html_favicon`), and README header. SVG source +
  PNG fallbacks (16/32/64/128/256 px) under `docs/_static/logos/`.
- New notebook walkthrough for `jwspecfit.fit_redshift`.

## 1.0.2 — 2026-05-14

Maintenance release; see commit history for details.

## 1.0.1 — 2026-04-21

Patch release to retry Zenodo DOI minting. The v1.0.0 deposit failed
validation because `CITATION.cff` contained a placeholder ORCID; the
ORCID field has been removed and the citation metadata cleaned up. No
functional code changes.

## 1.0.0 — initial public development release

- `jwspecfit` — resolution-aware least-squares fitting with bootstrap
  uncertainties and BIC broad-Balmer selection.
- `jwspecmcmc` — NUTS / emcee / nautilus MCMC fitting.
- `jwspecabund` — direct-T_e, forward-model, and strong-line
  abundance pathways.
