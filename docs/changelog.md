# Changelog

This project does not yet follow a formal release schedule. Key
additions are listed below in reverse chronological order. Commit
history on GitHub is the authoritative source.

## Unreleased

### N/O can measure O²⁺ from the rest-UV O III] lines

- **`no_oxygen_tracer=` on `compute_abundances`.** The Martinez+25 ICFs 2
  (N²⁺/O²⁺), 3 (N³⁺/O²⁺) and 5 ((N²⁺+N³⁺)/O²⁺) divide by O²⁺ alone, and an ICF
  is a function of the ionic ratio, log(U), Z and n_e — not of the line that
  measured O²⁺. They can therefore pair the UV nitrogen lines with
  O III] 1661,1666 instead of [O III] 5007, the nitrogen analogue of the
  pure-UV C/O. `"optical"` stays the default and is bit-identical to before;
  `"uv"` selects the UV tracer, `"auto"` switches on A_V as C/O does.
- **Why it is better conditioned**, measured with the package's own atomic data
  and dust curves: the N²⁺/O²⁺ conversion factor moves 0.09 dex across
  T_e = 10,000–25,000 K against 1.37 dex for the [O III] 5007 pairing, because
  λ1666 sits at nearly the same excitation energy as λ1749,1752 while λ5007
  (2.5 eV) does not. Reddening leverage on log(N/O) drops from +0.56 (Cardelli)
  and +1.32 (Salim) dex per magnitude of A_V to −0.006 and −0.047. For N³⁺ it is
  0.11 dex against 1.57, and +0.065/+0.166 against +0.626/+1.532. No Hβ is
  needed either, so the UV-to-optical grating calibration drops out.
- **ICF 5 is now genuinely "pure UV"**, as Martinez et al. describe it; its
  denominator had been optical all along.
- ICFs 1 and 4 keep the optical oxygen: they need O⁺, and summing a UV-traced
  O²⁺ with an optical O⁺ would put the reddening baseline back into the
  denominator. Monte-Carlo and posterior draws lock to the point estimate's
  tracer, so a resampled A_V cannot make a posterior a mixture of two tracers.
  `AbundanceResult.NO_oxygen_tracer` records which tracer produced the N/O.
- Z/Z_sun still comes from O/H, so this removes the optical oxygen from the
  ionic ratio, not from the ICF's metallicity argument.

### A He I-only broad request is no longer ignored

- `jwspecfit.fit_with_broad`'s narrow-only early return checked
  `fit_balmer_broad` and `fit_oiii_broad` but not `fit_hei_broad`, so asking
  for the He I broad test *alone* returned the narrow fit and skipped every BIC
  stage in silence. It reached `jwspecfit.fit_lines` and `jwspecmcmc` too, which
  both route He I-only requests through that function. Enabling any other broad
  flag alongside it had been masking the bug.
- **`centroid_max_sigma` now reaches the real-data broad fits.** The two
  `ThreadPoolExecutor` submissions passed their arguments positionally and
  stopped one short, so those fits used the 1.0 default whatever the caller
  set. They supply the single-point BIC values and, with `n_boot=0`, the
  best-fit handed to `jwspecmcmc`. Default runs are unaffected.

### The C/O oxygen tracer is chosen by A_V (documenting existing behaviour)

- The pure-UV C/O is not only reported alongside the optical one: above
  `CO_UV_AV_THRESHOLD` = 1 magnitude of A_V it is *adopted*, because past that
  point the reddening leverage on the UV-to-optical pairing outgrows the S/N
  cost of the faint λ1661,1666 denominator. This has been the behaviour for a
  while and was missing from both the changelog and §7.4.

### The low-ionisation density fallback is attributed correctly

- The `n_e(low)` z-fallback reported itself as `Topping+2025a`. The relation it
  actually evaluates, `54·(1+z)^1.2`, is from Abdurro'uf+2024 (arXiv:2404.16201,
  ApJ 973, 47) — as the code comment, the methodology document and the reference
  table all already said. Topping+2025a give only a slope (1.5 ± 0.6) with no
  published normalisation, which is why Abdurro'uf was adopted for the z > 6
  sample in the first place. The label was the only thing wrong: no density,
  temperature or abundance changes.

### Choose which Balmer lines get a broad component

- **`balmer_broad_lines=`** on `jwspecfit.fit_lines`,
  `jwspecfit.fit_with_broad`, `jwspecmcmc.fit_lines` and
  `jwspecmcmc.fit_with_broad`. `fit_balmer_broad=True` used to put a broad
  component on every one of Hα, Hβ, Hγ and Hδ; `balmer_broad_lines=["Ha"]`
  now tests a broad Hα alone. `None` (the default) keeps all four, and
  default results are bit-identical.
- **Why it matters for the BIC:** each broad line adds an amplitude and a
  width to the parameter count, but the bootstrap BIC is scored only on the
  Hα and Hβ pixels, so the Hγ and Hδ broads were charged a penalty they
  could never earn back. On the PRISM test spectrum `broad1` adds 8 free
  parameters with all four lines against 2 with Hα alone: a penalty of
  8 ln 17 ≈ 23 against 2 ln 11 ≈ 5, next to a ΔBIC threshold of 6.
- **The BIC window and SNR gate follow the selection.** The bootstrap BIC is
  scored on the Hα/Hβ windows among the requested lines (on the requested
  lines' own windows if neither is included). The gate uses the strongest
  requested line, Hα > Hβ > Hγ > Hδ, so an Hβ-only test at z ≳ 7, where Hα
  has left the band, is no longer gated on an absent Hα.
- Names other than Hα/Hβ/Hγ/Hδ raise: only those have their broad centroid
  tied to the narrow line.

### Upper limits are built from an assumed width and the local noise

- **`LineResult.upper_limit()` no longer returns `3 × flux_err`.** A line
  that is not detected has no meaningful fitted flux, so quoting three
  times its bootstrap error reported the degeneracies of the fit rather
  than the sensitivity of the spectrum. It now returns `n_sigma ×
  flux_err_cont`, measured from the local continuum noise. The old
  behaviour survives only as a fallback for hand-built `LineResult`s.
- **`FitResult.flux_upper_limit()` and `MCMCResult.flux_upper_limit()` no
  longer use the fitted width either.** They did measure the local
  residual RMS, but multiplied it by `lr.sigma_A` — the fitted σ of the
  undetected line, which is as unconstrained as its flux. On a synthetic
  non-detection the optimiser put [O III] 4363 at σ = 4.3 Å against an
  instrumental 13.0 Å, so the limit came out a factor of three too small.
- **The assumed width** is `σ² = σ_inst(λ)² + (σ_v/c × λ_obs)²`, with σ_v
  the intrinsic velocity width measured from the lines that *are*
  detected — so an undetected line inherits the width of the galaxy's
  real ones. Override with `sigma_v_kms=`. A measured width above
  500 km/s is warned about, never capped: narrowing the assumption
  tightens every limit, the one direction an upper limit must not err in.
- **The noise is measured, not evaluated.** The assumed profile is slid
  across the line-free residuals either side of the line and the scatter
  of the fluxes it recovers is the 1σ uncertainty. For white noise this
  reduces to `rms × √(2√π σ Δλ)` — the textbook amplitude error, and what
  "integrate the error spectrum over the line profile" evaluates to. The
  previous `rms × σ × √(2π)` assumed the noise was fully coherent across
  the line, ignoring pixel-to-pixel averaging.
- **Why measure it:** NIRSpec's 1D pixels are correlated by the 2D
  extraction and resampling, which no closed form captures. On the PRISM
  spectra tested the measured noise runs ~1.7× the white-noise value, on
  the medium gratings ~1.0×. Net effect on the limits themselves: ~0.7×
  the old values on the gratings, unchanged to within 5% on PRISM.
- New module `jwspecfit.limits`. `LineResult` and `MCMCLineResult` gain
  `flux_err_cont`, `sigma_assumed_A` and `ul_method`, all serialised and
  all optional, so old `.npz` files still load.
- **Results saved before this change are measured on load.** They carry no
  limits, so `upper_limit()` would have quietly fallen back to the
  bootstrap error — the behaviour this replaces, and ~1.5× off in the
  wrong direction. Everything needed (spectrum, model, continuum) is in
  the file, so `load_result()` and `load_mcmc_result()` now measure the
  limits rather than leave them absent. `jwspecfit.limits.backfill_flux_limits()`
  does the same for a result you already hold in memory.
- `jwspecabund` reuses the limits cached by the fit rather than
  recomputing them with the instrumental width alone, so the three
  packages now agree on what a non-detection means.

## 1.1.9 — 2026-09-04

### `jwspecabund` — flag a low-ionisation density that cannot be measured

- **The diagnostics now say when `n_e(low)` is uninformative.** [S II]
  6718/6732 and [O II] 3726/3729 have critical densities of only ~3×10³ and
  ~4×10³ cm⁻³, so their ratios sit on the high-density asymptote well before
  10⁴ and cannot distinguish 10⁴ from 10⁶ (Arellano-Córdova et al. 2026,
  arXiv:2602.13007). When the self-consistent O III solve measures the O²⁺
  zone above 10⁴ cm⁻³, `diagnostics["ne(low)"]` now records that O⁺, N⁺ and
  S⁺ are being evaluated in a zone that may be orders of magnitude denser
  than assumed, and a warning is logged. `ne_low_override` is the knob for
  following Arellano-Córdova's own prescription of 10⁴ cm⁻³.
- A measured `n_e(low)` that itself solves above 10⁴ cm⁻³ is now labelled a
  lower limit rather than reported as a value.
- Nothing is changed automatically: the low-ionisation density is still
  whatever the doublet or the redshift fallback gives. On OMEGA 77
  (n_e(O²⁺) = 1.6×10⁵) moving `ne_low` from the 673 cm⁻³ fallback to 10⁴
  raises 12+log(O/H) by 0.050 dex and lowers log(N/O) by 0.009.


### `jwspecabund` — A_V from the whole Balmer ladder at once

- **New `balmer_method="joint"` on `compute_abundances`** (and
  `compute_Av_joint_balmer()` in `jwspecabund.dust`). Instead of ratioing
  every line against a chosen anchor, it fits all the detected Balmer lines
  simultaneously with two free parameters — `A_V` and an overall
  normalisation — solving the normalisation in closed form at each `A_V` and
  scanning a profile χ². The default remains `"anchored"`; nothing changes
  unless you ask for the joint fit.
- **Why not just average more decrements?** Balmer decrements are exactly
  additive in magnitudes: `D(Hγ/Hα) = D(Hγ/Hβ) + D(Hβ/Hα)` holds to machine
  precision, noise included. A ladder of *N* lines therefore carries *N−1*
  independent constraints however many of the *N(N−1)/2* pairs you form.
  Adding the extra pairs contributes no information, and averaging them as
  though they were independent shrinks the quoted error spuriously.
- The joint fit is anchor-free, uses each *flux* exactly once, and returns
  `chi2`/`dof` — a large reduced χ² flags a ladder that no single `A_V` can
  reconcile (stellar Balmer absorption, or a flux-calibration step between
  gratings), which the anchored weighted mean silently averages away.

### `jwspecabund` — two fixes to the anchored Balmer estimator

- **Individual decrements are no longer clipped at `A_V ≥ 0` before
  averaging.** Attenuation cannot be negative, so clipping a *final* estimate
  is right, but clipping each decrement first folds every noise-driven
  negative to zero while leaving the positives alone, biasing the mean upward
  whenever the true `A_V` is near zero. Injecting `A_V = 0` into a four-line
  ladder at realistic SNR, the old estimator returned **+0.036**; it now
  returns 0.000. The clip now applies once, to the combined value.
  `compute_Av_from_balmer()` keeps its clip by default and grows a
  `clip=False` opt-out for use as an ingredient.
- **The error on the weighted mean now accounts for the shared anchor.**
  Every decrement divides by the same anchor flux, so their errors are
  positively correlated and inverse-variance weighting as if independent
  understates the result. Replaced with a generalised least squares using the
  full covariance (`C_ii = σ_i²`, `C_ij = s_i s_j` with `s` the anchor's
  share), falling back to the diagonal form if the matrix is singular. On a
  real four-line ladder anchored on Hα the quoted error went from 0.081 to
  0.100 against a Monte-Carlo truth of 0.097.


### `jwspecfit` — blended-doublet flux ratios corrected

- **O III] λ1661/λ1666 was fixed at 0.83; the true value is 0.4016.** Both
  components decay from the same upper level (⁵S₂ → ³P₁, ³P₂), so their ratio
  is a pure branching ratio set by the A-values and wavelengths — no density
  or temperature dependence whatsoever (verified against PyNEB at
  n_e = 10, 10³ and 10⁶ cm⁻³). The constant was wrong by a factor 2.07.
- **N III] λ1752/λ1749 was fixed at 0.67; PyNEB gives 2.75** at the
  low-density limit (rising to 3.5 by n_e = 10⁵). Wrong by a factor ~4.
- These ratios are applied **only when the doublet is unresolved**, so fits
  that resolve the pair are unaffected — the bug bit PRISM and other low-R
  data. The doublet *sum* was always right; what was wrong was the split
  between the two members, which matters for anything using one member alone
  (the self-consistent T_e–n_e solve uses λ1666 by itself).
- A new test pins every blended-doublet constant to PyNEB so they cannot
  drift again.

### `jwspecabund` — graceful failure on an unsolvable auroral ratio

- **A noisy or over-fit [O III] λ4363 no longer takes down the whole
  calculation.** Its ratio against λ5007 can land above the physical maximum,
  where PyNEB returns no solution; the resulting `ValueError` propagated out
  of `compute_abundances`. Each single-ratio diagnostic is now tried in turn,
  so an unsolvable λ4363 cannot hide a good O III] λ1666, and `method="auto"`
  falls back to the strong-line method with the reason recorded in
  `failures["direct method"]`. An explicit `method="direct"` still raises,
  now distinguishing an absent auroral line from an unsolvable one.

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
- **Which O III collision dataset to use is now argued out in the
  methodology page** rather than asserted. TZ17 stays the default on the
  strength of its 202-level close-coupling expansion against AK99's 6, with
  both comparably close to SSB14 on the quantities SSB14 can compute. The
  cost of the choice is strongly density-dependent — recovering a known
  O²⁺/H⁺ from TZ17 fluxes, AK99 differs by 0.014 dex at n_e = 10³ cm⁻³ but
  0.092 dex at 10⁶ — so above ~10⁵·⁵ cm⁻³ it is worth running both and
  quoting the spread.
- **Every result now reports what each single ratio would have given on its
  own**, in `alt_results["direct_4363"]` / `alt_results["direct_1666"]` and as
  a side-by-side table in `AbundanceResult.summary()`, so the effect of the
  adopted diagnostic is visible rather than implied. `AbundanceResult` gains
  `Te_diagnostic`, `ne_Opp_err`, `ne_Opp_is_upper_limit` and
  `Te_ne_selfconsistent`.
- **New `plot_te_ne_diagnostic()`** reproduces figure 3 of Hsiao et al.: the
  three T_e(n_e) curves with 1σ bands, the posterior contours, and the adopted
  solution.

### `jwspecfit` — N V in the default line markers

- **N V λλ1238,1242 is now drawn by `plot_spectrum_interactive` and
  `plot_2d_1d`.** It sits between Lyα and N IV] in the default marker list,
  so the highest-ionisation nitrogen line is labelled without having to pass
  a custom list.

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
