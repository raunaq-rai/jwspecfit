# Changelog

This project does not yet follow a formal release schedule. Key
additions are listed below in reverse chronological order. Commit
history on GitHub is the authoritative source.

## Unreleased

### `jwspecabund` — pure-UV C/O and an O III] doublet check

- **New `compute_CppOpp_uv()`: C/O from C III] λλ1907,1909 against
  O III] λλ1661,1666**, the classical rest-UV route (Garnett et al. 1995;
  Erb et al. 2010; Berg et al. 2016). Numerator and denominator sit ~240 Å
  apart, so it carries ~0.04 dex of reddening leverage per magnitude of A_V
  against ~0.6–1.3 dex for the C III]-vs-[O III] λ5007 pairing the adopted
  C/O uses — and it needs no Hβ, which removes the relative flux calibration
  between the gratings covering the UV and the optical. The cost is
  sensitivity to the assumed conditions (~0.03 dex per 1700 K in T_e, and
  −0.03 dex from n_e = 3×10⁴ → 10³, since C III] λ1907 has a low critical
  density).
- **It is computed and reported automatically**, with its own posterior, as
  `alt_results["CO_uv"]` whenever the four lines are present, and is *not*
  adopted in place of the existing C/O. C²⁺ is evaluated in the
  intermediate zone, matching the adopted C/O, so the comparison isolates
  the oxygen tracer; on three stacked spectra the two routes agree to
  0.06–0.14 dex (1–2σ).
- **New `check_oiii_uv_doublet()`, run automatically.** O III] λ1661 and
  λ1666 decay from the same upper level, so their ratio is a pure branching
  ratio — 0.402, with no T_e, n_e or abundance dependence. A departure is
  therefore a line-measurement problem, and it propagates into everything
  built on λ1666 (the pure-UV C/O, and the self-consistent T_e–n_e solve).
  The result lands in `AbundanceResult.oiii_uv_doublet` and `diagnostics`,
  with a warning past 3σ.
- The adopted C/O now has a `diagnostics["C/O"]` entry when the Martinez ICF
  is used; previously only the direct-sum and Garnett+97 routes described
  themselves.

### `jwspecabund` — self-consistent O²⁺ temperature and density

- **T_e and n_e are now solved together in the O²⁺ zone when O III] λ1666,
  [O III] λ4363 and [O III] λ5007 are all detected**, following Hsiao et al.
  (2026), [arXiv:2608.20339](https://arxiv.org/abs/2608.20339). [O III] λ5007
  has a critical density of only ~7 × 10⁵ cm⁻³, so above n_e ~ 10⁵ cm⁻³ it is
  collisionally de-excited and the classical λ4363/λ5007 ratio depends on both
  T_e and n_e. Solving it at an assumed low density then overestimates T_e and
  underestimates O/H — by up to 1.1 dex, which is how ordinary galaxies get
  misclassified as extremely metal-poor. λ1666 has a far higher critical
  density, so the two ratios give two independent constraints on the two
  unknowns. On a synthetic object at T_e = 13,000 K and n_e = 10⁶ cm⁻³ with
  12 + log(O/H) = 8.000, the joint solve returns 8.005 where λ4363 alone
  returns 6.849.
- **New `compute_Te_ne_OIII()`** builds the T_e(n_e) curve for each ratio over
  log(n_e/cm⁻³) = 0–7 in 1,000 steps and takes their intersection, with the
  closest approach as the fallback, exactly as the paper prescribes.
  Uncertainties come from the flux posterior.
- **Densities that cannot be constrained are reported as 1σ upper limits.**
  Below n_e ~ 10⁴·⁵ cm⁻³ both ratios go density-flat and the curves run
  parallel; the solve detects this, falls back to the single-ratio
  temperature, and caps the borrowed density at the bound the O III lines
  impose — but only when the curves genuinely cross. If they never do, the
  two ratios are not mutually consistent, the bound carries no density
  information, and nothing downstream is allowed to move because of it.
- **`compute_abundances()` gains `self_consistent_OIII` (default `"auto"`) and
  `oiii_coll_file`.** `"auto"` engages the joint solve whenever all three lines
  clear `snr_auroral`; pass `False` for the previous single-ratio behaviour.
  `oiii_coll_file="o_iii_coll_AK99.dat"` reproduces Hsiao et al.'s atomic data
  (1–2 % in T_e, ~0.02 dex in O²⁺/H⁺).
- **Every result now reports what each single ratio would have given on its
  own**, in `alt_results["direct_4363"]` / `alt_results["direct_1666"]` and as
  a side-by-side table in `AbundanceResult.summary()`, so the effect of the
  adopted diagnostic is visible rather than implied. `AbundanceResult` gains
  `Te_diagnostic`, `ne_Opp_err`, `ne_Opp_is_upper_limit` and
  `Te_ne_selfconsistent`.
- **New `plot_te_ne_diagnostic()`** reproduces figure 3 of Hsiao et al.: the
  three T_e(n_e) curves with 1σ bands, the posterior contours, and the adopted
  solution.

## 1.1.8 — 2026-08-14

### Citation

- **Citation updated to the published paper.** The companion paper is now on
  arXiv and indexed on ADS — Rai & Roberts-Borsani (2026),
  arXiv:2608.10063, *"Unveiling and Characterising Ubiquitous Nitrogen
  Enhancement in 6 ≤ z ≤ 10 Galaxies with JWST Spectroscopy"*. It replaces
  the "in preparation" placeholder in the README, the docs, and
  `CITATION.cff` (as `preferred-citation`, so GitHub's "Cite this
  repository" button now returns the paper).
- **The separate software citation has been dropped.** The paper is the only
  reference for `jwspecfit`: the `@software` BibTeX entry, the DOI badge and
  the version-pinning guidance have been removed from the README,
  `docs/citation.md` and `CITATION.cff`.

### `jwspecabund` — strong-line ratios

- **R23 now uses the [O III] λ4959 + λ5007 doublet sum**, as Sanders et al.
  (2025) Eq. 3 defines it; `compute_line_ratios` previously built the
  numerator from λ5007 alone. log R23 was underestimated by 0.09–0.12 dex,
  and because R23 carries the smallest calibration scatter of the four
  ratios the bias propagated efficiently into the strong-line metallicity —
  0.1–0.4 dex too low on the rising branch. The measured λ4959 is used when
  present and above the SNR cut, otherwise it is inferred from λ5007.

### `jwspecabund` — O III] λ1666 temperature

- **The λ1666 diagnostic now measures what it claims to.** It was inverting
  the λ2321 emissivity curve, because PyNEB's default 5-level O III atom
  does not reach level 6; the λ1666 atom is now built against the Tayal &
  Zatsarinny (2017) collision data. The switch is scoped to that atom, so
  the λ4363 temperature and every other calculation keep the SSB14 default
  and existing results are bit-identical. Expect a ~400 K (~2.4%)
  atomic-data systematic on any λ1666 temperature.
- **880× faster λ1666 solve.** Passing `NLevels=6` explicitly stops PyNEB
  re-interpolating the full 202-level collision array on every emissivity
  call: 8.4 ms per solve instead of 15–25 s.
- **The λ1666 cross-check is now a point estimate**, labelled "point
  estimate, not adopted", rather than carrying its own uncertainty
  propagation — it is a sanity check on the adopted λ4363 temperature, not
  a second measurement. This removes a redundant second direct run in
  `method="auto"` and restores the previous runtime.
- **Fixed a `TypeError`** raised whenever both λ4363 and λ1666 were detected
  in `method="auto"`.

### Plotting

- **New `y_scale` argument on every plotting entry point** — `"linear"`
  (default, unchanged) or `"log"` for decade-tick y-axes. Log mode replaces
  the lower limit with a positive one, omits the y = 0 reference lines, and
  leaves signed residual panels linear. Covers `plot_fit`,
  `plot_fit_interactive`, `plot_spectrum_interactive`, `plot_2d_1d`,
  `DLAResult.plot`, `RedshiftResult.plot`, `plot_corner`, `plot_traces` and
  `plot_flux_posterior`.

### Repository

- The example notebooks under `docs/notebooks/` have been untracked pending
  a rewrite.

## 1.1.7 — 2026-07-23

### `jwspecabund` — Martinez C/O ionisation-correction factor

- **New C²⁺/O²⁺ → C/O ICF from Martinez et al. (2026, in prep.)**, selectable
  via the new `co_icf_method` argument to `compute_abundances`. The ICF is
  interpolated in electron density and evaluated at the intermediate (C²⁺)
  zone density, consistent with the ion's ionisation zone. The legacy
  Garnett (1997) C/O ICF remains available as the alternative option.

### Line database additions

- **[S III] λ6312, λ9069, λ9531** added to `REST_LINES_A` and included as
  default plot markers.

### Behaviour changes

- **Invalid `icf_tier` now raises `ValueError`** instead of silently falling
  back to the Izotov tier, and **all string-choice inputs of
  `compute_abundances` are validated up front** so mistyped options fail
  immediately with a clear message rather than mid-computation.
- **Asymmetric uncertainties are printed as `(+hi/−lo)`** in
  `AbundanceResult.summary`.

### Documentation

- Consolidated **underlying-methods and references** section in the README,
  with explicit credit to **Zorayda Martinez et al. (2025, 2026 in prep.)**
  for the N/O and C/O ICFs and the O32 / N43 log(U) calibrations.
- Added a convergence (R̂ / ESS) check to the brief MCMC tutorial and a NUTS
  multi-component line-fit animation.

## 1.1.6 — 2026-06-23

### Bug fixes — Izotov+2006 ionisation correction factors

- **Corrected the neon, sulfur, argon and nitrogen ICFs**
  (`jwspecabund.icf`). The four Izotov+2006 ICFs had been implemented as
  invented polynomials in the O⁺ fraction that dropped the dominant `c/f`
  term of the true `a·f + b + c/f` form (eqs. 18–22), returning ICF ≈ 0.3–1.0
  where the correct value is ≈ 1.0–1.5. They are now the proper
  metallicity-dependent Izotov forms (three branches in 12+log(O/H), verified
  against PyNeb), return the elemental ICF (X/H = ICF·ion/H⁺), and the
  abundance ratios divide by the total oxygen abundance. Effect on reported
  ratios: **Ne/O ~+0.45 dex** (a spuriously low log(Ne/O) ≈ −1.2 is corrected
  to ≈ −0.8, consistent with the α-element locus), **S/O ~+0.17 dex**,
  **Ar/O ~+0.10 dex**. N/O via the Izotov tier changes by < 0.04 dex; the
  default Martinez+2025 / ICF-5 nitrogen path is unaffected. Carbon
  (Garnett+1997) was already correct. Regression tests added for all four.

## 1.1.5 — 2026-06-22

### Direct-Te abundances: three-zone temperature & density framework

- **New default `Te_relation="3_tier"`** (`jwspecabund`). The direct-Te
  method now resolves three ionisation zones with Garnett (1992) Te–Te
  relations throughout (following Martinez+2025, "Under Pressure",
  arXiv:2510.21960): `Te_int = 0.83·Te_high + 1700` and
  `Te_low = 0.70·Te_high + 3000`, with `Te_high` measured from
  [O III] λ4363 (or λ1666). Ions are assigned to zones by ionisation
  potential — O²⁺/Ne²⁺/N³⁺/C³⁺ to the high zone, N²⁺/C²⁺/S²⁺/Ar²⁺ to the
  intermediate zone, and O⁺/N⁺/S⁺ to the low zone. `AbundanceResult` gains
  `Te_mid`, `Te_mid_err`, and `ne_Opp`.

- **[Ar IV] λλ4711,4740 as the preferred O²⁺-zone density**, with the
  λ4711 / He I λ4714 blend deblended via the He I λ4472 anchor and a PyNEB
  He I emissivity ratio (`compute_ne_ArIV`, `heI_4714_over_4472`); C III]
  is the fallback.

- **[Si III] λλ1883,1892 added as a UV low-ion density fallback** for the
  O⁺/N⁺ zone (`compute_ne_SiIII`), for UV-only high-z stacks where the
  optical [S II]/[O II] doublets are unavailable.

- **Redshift-dependent density fallbacks** when a doublet is missing or its
  solve fails (`ne_zone_fallback`): `ne_low = 54·(1+z)^1.2`
  (Abdurro'uf+2024), and `ne_mid = 1110·(1+z)^1.93`,
  `ne_high = 5400·(1+z)^1.62` (Martinez+2025 Eqs 4/5). Densities are solved
  at a 1.5×10⁴ K fiducial and refined once at the zone temperature.

- **New `z_ne` parameter** on `compute_abundances` decouples the redshift
  used for density fallbacks from the geometric redshift (e.g. rest-frame
  z=0 stacks of a physically high-z sample).

### Bug fixes

- Intermediate-zone ions (C²⁺/N²⁺/S²⁺/Ar²⁺) now fall back to the O²⁺-zone
  density `ne_Opp` (carrying the z-dependent intermediate fallback) rather
  than the low-zone density when the C III] solve fails.
- `ne(low)`/`ne(high)` diagnostic labels are now truthful after a
  z-fallback, and the `Te(high)` diagnostic reports the actual auroral
  line used (λ4363 vs λ1666).

## 1.1.4 — 2026-06-07

### Bug fixes

- **Corrected air/vacuum wavelengths for 12 optical lines in
  `REST_LINES_A`.** A subset of lines (Hδ, Hγ, Hβ, Hα, [O III] λλ4959,5007,
  [N II] λλ5756,6549,6585, He I λ5877, [S II] λλ6718,6732) had the
  air→vacuum transform applied twice — once to values that were already in
  vacuum — inflating them by the refractive index of air (n ≈ 1.000279,
  ≈ +1.4 Å at 5000 Å, a +84 km/s offset). They now hold the correct vacuum
  wavelengths. The inconsistency (some lines, e.g. [O III] λ4363, were
  already correct) could bias fitted redshifts and line positions at
  grating/high resolution. A regression test
  (`tests/test_line_wavelengths.py`) now pins these against literature
  vacuum values.

### Line database additions

- **Seven UV interstellar absorption lines** added to `REST_LINES_A`
  (vacuum, Morton 2003): the O I λ1302 + Si II λ1304 blend
  (`abs_OISiII1303`, single feature), Si II λ1526 (`abs_SiII1526`),
  Fe II λ1608 (`abs_FeII1608`), the Al III λλ1854,1862 doublet
  (`abs_AlIII1854`, `abs_AlIII1862`), and the C IV λλ1548,1550 doublet
  (`abs_CIV1548`, `abs_CIV1550`). All but the C IV pair are included in
  the default prism and grating line lists. **`abs_CIV1548/1550` are in
  the database but *not* the default lists**, because they coincide
  exactly with the `CIV_1`/`CIV_2` emission components — fit them
  explicitly via `lines=[...]` when C IV is in absorption / P-Cygni.

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
