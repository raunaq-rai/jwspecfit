# Measurement Methodology

How every number that `jwspecfit` and `jwspecmcmc` report is calculated,
and what its uncertainty does and does not include.  Each section names
the function that implements it, so the description can be checked
against the code.

---

## Table of Contents

1. [Units and conventions](#1-units-and-conventions)
2. [Instrumental resolution](#2-instrumental-resolution)
3. [Continuum](#3-continuum)
4. [Line model and line flux](#4-line-model-and-line-flux)
5. [Tied parameters](#5-tied-parameters)
6. [The fit](#6-the-fit)
7. [Flux uncertainties — bootstrap](#7-flux-uncertainties--bootstrap)
8. [Flux uncertainties — MCMC](#8-flux-uncertainties--mcmc)
9. [Signal-to-noise ratios](#9-signal-to-noise-ratios)
10. [Equivalent widths](#10-equivalent-widths)
11. [Upper limits for non-detections](#11-upper-limits-for-non-detections)
12. [Broad components](#12-broad-components)
13. [Lyman-α](#13-lyman-α)
14. [What the uncertainties do not include](#14-what-the-uncertainties-do-not-include)

---

## 1. Units and conventions

| Quantity | Unit | Notes |
|---|---|---|
| Input flux density, error | µJy | `Spectrum.flux_ujy`, `Spectrum.err_ujy` |
| Wavelength | µm (input), Å (internally) | observed frame |
| Rest wavelengths | Å, **vacuum** | `jwspecfit.lines.REST_LINES_A` |
| Fitted flux density | erg s⁻¹ cm⁻² Å⁻¹ | continuum-subtracted |
| Line flux | erg s⁻¹ cm⁻² | integrated |
| Equivalent width | Å | rest frame unless stated (`ew0_A`) |
| Widths, centroids | Å, observed frame | `sigma_A` is a Gaussian σ, not a FWHM |

**µJy → F_λ.**  Each pixel is converted at its own centre,

$$F_\lambda = F_\nu \, \frac{c}{\lambda^2}, \qquad
F_\nu[\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}] = 10^{-29}\,F_\nu[\mu\mathrm{Jy}],$$

with λ in cm and a final factor 10⁻⁸ for "per Å" (`io._ujy_to_flam`).  The
error array is converted the same way.  The fit is done in F_λ, so line
fluxes come out directly in erg s⁻¹ cm⁻².

**Pixel edges.**  Edges are the midpoints between adjacent pixel centres;
the two outer edges are extrapolated by half a pixel
(`Spectrum.wave_edges_A`).  Pixel widths are the edge differences.

---

## 2. Instrumental resolution

The instrumental line-spread function is taken to be Gaussian with

$$\sigma_\mathrm{inst}(\lambda) = \frac{\lambda}{2.3548\,R(\lambda)}$$

(`resolution.sigma_inst_A`).  R is resolved in this order
(`resolution.resolve_R`):

1. an explicit `R=` (a number, a callable `R(λ_µm)`, or a
   `TabulatedR`, e.g. from the file's resolution column);
2. `Spectrum.R`;
3. the grating name:
   - **PRISM**: the Jakobsen et al. (2022) pre-launch curve
     (R ≈ 26–300) × 1.3 for the post-launch improvement;
   - **M gratings**: constant R = 1000;
   - **H gratings**: constant R = 2700;
4. if none of these is given, R is estimated from the pixel spacing
   (`R_from_pixels`, assuming Nyquist sampling).

```{important}
The grating values are **nominal, uniformly-illuminated-shutter**
resolutions.  A compact source underfills the shutter, so its true R is
often noticeably *higher* (lines narrower).  For point sources, pass an
`R(λ)` appropriate to the source morphology.  Everything below that
uses σ_inst — width bounds, width ties, the continuum line mask, upper
limits — inherits this choice.
```

---

## 3. Continuum

Implemented in `continuum.fit_continuum`.  Three options:

### 3.1 Polynomial (default)

1. **Valid pixels**: finite flux and error, error > 0, and redward of
   observed Lyα (unless `lya_break=True`).
2. **Line mask**: every pixel within ±max(6 σ_inst, 20 Å) of any line
   being fitted is excluded.
3. **Fit**: weighted polynomial of degree `deg` (default 2) in a
   normalised wavelength coordinate, weights 1/σ.
4. **Clipping**: 5 iterations; pixels more than `clip_sigma` (default
   2.5) error-σ *above* the fit are removed.  Only positive outliers are
   clipped, so unmasked emission cannot drag the continuum up, while
   absorption and negative noise are kept.
5. The final polynomial is evaluated at every pixel.

**Continuum uncertainty.**  With `return_err=True` the 1σ error at each
pixel is

$$\sigma_c^2(\lambda) = \mathbf{v}(\lambda)\, \mathbf{C}\, \mathbf{v}(\lambda)^\mathsf{T},$$

where **v** is the Vandermonde row of the polynomial and **C** the
coefficient covariance.  numpy's *scaled* covariance is used (multiplied
by the reduced χ² of the continuum fit), so a polynomial that describes
the continuum poorly gets a correspondingly larger error rather than an
optimistic formal one.

### 3.2 Median filter (`moving_average=True` or a window in pixels)

A median filter followed by a boxcar of the same window, applied to the
unmasked pixels and interpolated back onto the full grid; positive
outliers near the lines are clipped iteratively.  **There is no
parametric covariance, so its uncertainty is reported as NaN**
("unavailable"), never as zero.

### 3.3 External continuum (`continuum=`)

Any continuum on the spectrum's pixel grid in µJy (for example an SPS
fit with the lines masked) replaces the internal one.  Its uncertainty
can be supplied to `jwspecmcmc.fit_lines` as `continuum_err=`; if
omitted it is taken as zero.

### 3.4 Where the continuum uncertainty is used

| Output | Least squares (`jwspecfit`) | MCMC (`jwspecmcmc`) |
|---|---|---|
| Line flux error | included: the bootstrap redraws the continuum (§7) | not included (continuum held fixed) |
| EW value | uses the continuum | uses the continuum |
| EW error (`ew0_err`) | included: bootstrap (§7, §10) | included: continuum draws (§10) |
| Continuum error, S/N at the line (`cont_err_flam`, `cont_snr`) | available | available |

The continuum term is small for line *fluxes* — the continuum is
constrained by hundreds of pixels against a handful under the line
(< 1% of the flux error on G395M tests) — but **not** for equivalent
widths of faint-continuum sources: on G395M tests it raised the EW error
by 7–24% at a 0.2 µJy continuum and by up to 2.6× at 0.05 µJy.  An
external continuum is held fixed in both paths unless its error is
given (`continuum_err=`, MCMC only).

---

## 4. Line model and line flux

Each line is a Gaussian with amplitude A, centroid μ and width σ, and
the model in each pixel is the Gaussian's **mean over the pixel**, not
its value at the pixel centre (`models.gaussian_binned`,
`models.build_model`):

$$M_i = \frac{A}{\lambda_{i+1/2}-\lambda_{i-1/2}}
\left[\Phi\!\left(\frac{\lambda_{i+1/2}-\mu}{\sigma}\right)
-\Phi\!\left(\frac{\lambda_{i-1/2}-\mu}{\sigma}\right)\right],$$

with Φ the standard normal CDF.  This is exact for lines narrower than
a pixel, which the centre-sampled form is not (relevant on the PRISM).
The total model is the sum over lines.

**Line flux = A**, exactly: the profile has unit area, so the
integrated flux is the amplitude itself, in erg s⁻¹ cm⁻².  No numerical
integration and no aperture correction is involved.

Absorption lines (names starting `abs_`) are the same profile with
A ≤ 0.

---

## 5. Tied parameters

Every tie is an operation `p[dst] = p[src] × ratio` (or, for widths, the
velocity-plus-instrument rule below), held in one ordered list,
`ConstraintSet.tying_ops()`.  The least-squares fit, the bootstrap, the
NumPy MCMC likelihood and the JAX (NUTS) likelihood all use that list,
so they cannot disagree.  A tied parameter is not a free parameter: it
is not fitted, has no prior, and is not counted in the degrees of
freedom or the BIC.

| Tie | What is shared | Default |
|---|---|---|
| Balmer → [OIII] 5007 | velocity (centroid) and velocity width | on (`tie_balmer_to_oiii`) |
| [NII] 6549 → 6585 | flux ratio 1/2.96, velocity, width | on |
| [OIII] 4959 → 5007, [NeIII] 3968 → 3869, [OI] 6366 → 6302, O III] 1661 → 1666 | fixed branching ratio, velocity, width | on |
| H8 → H9 | Case B ratio 1.437 | on |
| UV doublets (N V, C IV, C III], N IV], Si III], C II]) | velocity and width; amplitudes free | on (`tie_uv_doublets`) |
| UV intercombination widths (C III], N IV], N III]) | one velocity width | **off** (`tie_uv_widths`) |
| Blended UV doublet flux ratio | fixed to the low-density value | **off** (`fix_blended_ratios`) |
| N III] 1750 quintuplet | one flux, velocity, width | on |
| Broad Balmer → narrow Balmer | centroid | when broad fitted |

**Velocity widths.**  Lines that share a velocity width share the
*intrinsic* dispersion σ_v; each line keeps its own instrumental width:

$$\sigma_t^2 = \left(\tfrac{\lambda_t}{\lambda_s}\right)^2
\left(\sigma_s^2-\sigma_{\mathrm{inst},s}^2\right)+\sigma_{\mathrm{inst},t}^2 .$$

This applies when the source line is wider than its assumed instrumental
width.  When it is not — unresolved, or a compact source whose true R
is higher than the assumed one — the target is scaled by the ratio of
instrumental widths, σ_t = σ_s σ_inst,t/σ_inst,s, which is exact for
unresolved lines even if R is wrong by a constant factor.  The two forms
meet at σ_s = σ_inst,s (`constraints.tied_width`).

Scaling the *observed* width by λ_t/λ_s instead would only be right for
constant R; on the PRISM, where R changes several-fold with wavelength,
it gives the wrong width ratio for unresolved lines and biased Hα/Hβ by
−14% to +23% (noise-free tests); with the rule above the same tests
recover every flux to < 0.3%, including when the true R is 1.3–1.5×
the assumed one.

**Blended doublets.**  With `fix_blended_ratios=True`, a UV doublet
closer than 4.71 σ_inst (two FWHM) has its flux ratio fixed to the
low-density / optically thin value
(`constraints._INTERCOM_LOW_DENSITY_RATIOS`).  By default both
amplitudes stay free: whether a doublet is resolved enough to split is
left to the user, who knows the source's actual resolution.

---

## 6. The fit

Implemented in `fitter.fit_lines`.

**Objective.**  Bounded least squares (`scipy.optimize.least_squares`,
trust-region reflective) minimising

$$\chi^2 = \sum_{i\,\in\,\mathrm{valid}}
\left(\frac{F_{\lambda,i}-C_i-M_i}{\sigma_i}\right)^2 ,$$

the plain inverse-variance χ²; the same χ² defines the MCMC likelihood
(§8).

**Valid pixels.**  Finite flux and error, error > 0, and redward of
observed N V 1240 (or of 1150 Å rest when Lyα is fitted).  A line is
dropped from the fit if fewer than half the pixels within ±5σ of it are
valid (detector gaps).

**Starting values and bounds** (per line, `fitter._grating_bounds`):

| | PRISM | M gratings | H gratings | No grating (stacks) |
|---|---|---|---|---|
| σ lower | 0.40 × pixel | max(0.12 pix, 0.22 σ_inst) | 0.10 × pixel | 0.20 × pixel |
| σ seed | 0.90 σ_inst | 0.60 σ_inst | 0.50 σ_inst | 1.0 σ_inst |
| σ upper | 1.7 σ_inst | 2.0 σ_inst | 1.8 σ_inst | 5.0 σ_inst |

σ_inst is evaluated at the line's own wavelength; the upper bound is
multiplied by `sigma_factor`, and every σ is capped at 500 Å.  A merged
entry (`OII_doublet`, `CIV_doublet`, `NV_doublet`, `NIV_doublet`,
`NIII_doublet`: one Gaussian standing for a blend) has the RMS spread of
its components, × (1 + z), added in quadrature to its seed and upper
bound (`lines.merged_sigma_A`); without it a partly resolved [O II] blend
at R = 2500 pinned at the cap and lost 21% of its flux.
Centroids may move by min(`centroid_vmax`, `centroid_max_sigma` ×
σ_inst), floored at two pixels and capped at half the distance to the
nearest other line.  Amplitudes are seeded from the peak near the line
(or the local noise level if nothing there is positive), bounded
0 ≤ A ≤ 150 × that peak × √(2π) σ_upper.  `sigma_overrides` and
`centroid_overrides` replace the automatic bounds for named lines.

**Reduced χ².**  `chi2` = χ²/(N_valid − N_free) at the best fit.

---

## 7. Flux uncertainties — bootstrap

`flux_err` from `jwspecfit.fit_lines` (`fitter._bootstrap_uncertainties`).

This is a **parametric (noise-injection) bootstrap**, not a resampling
of pixels:

1. draw an independent Gaussian realisation of the error array,
   ε_b ~ N(0, σ), for b = 1 … `n_boot` (default 1000);
2. **redraw the continuum.**  The continuum fit is linear in the data for
   a fixed line mask, so the continuum shift that ε_b produces is the
   same polynomial fit applied to ε_b itself, δC_b (clipping off: the
   perturbed data are √2 noisier, and clipping them would remove ~6× as
   many pixels as the real fit and inflate the spread).  An external
   continuum is held fixed (δC_b = 0);
3. refit all lines, Lyα included, to F − C + ε_b − δC_b with the same ties
   and bounds, and record each line's flux F_b and the continuum under it,
   C + δC_b;
4. `flux_err` and `ew0_err` are the standard deviations of F_b and of
   F_b / (C + δC_b) / (1 + z), after a 3σ clip that removes runaway refits
   of unconstrained parameters.

For a linear model this reproduces the estimator's sampling
distribution exactly; for the line fit it is the same to within the
non-linearity of the centroids and widths.  In injection–recovery tests
on G395M spectra (150–300 noise realisations) the scatter of the fitted
fluxes divided by the median `flux_err` was 0.94–1.08, and of the EWs
divided by the median `ew0_err` 0.93–1.06 (robust estimate) — consistent
with unity.

Notes:
- The noise is drawn from a fixed seed, so a rerun reproduces the same
  `flux_err`.  With fewer than ~300 draws the error on `flux_err`
  itself (≈ 1/√(2 n_boot)) becomes noticeable.
- The error array is treated as white (independent pixels).  NIRSpec 1-D
  pixels are correlated; see §14.
- **Lines near zero flux.**  Amplitudes are bounded at A ≥ 0, so for a
  line within ~2σ of zero the refitted fluxes pile up at the bound and
  their spread *underestimates* the real uncertainty (by up to ~2× in
  tests).  Such lines should be reported as upper limits (§11), not as
  flux ± `flux_err`.

---

## 8. Flux uncertainties — MCMC

`jwspecmcmc.fit_lines` (`_engine._fit_lines_mcmc`).

**Likelihood.**  ln 𝓛 = −χ²/2 with exactly the χ² of §6, on the same
pixels, ties and continuum.

**Priors.**  Uniform on each free parameter between the least-squares
bounds of §6 (`priors.priors_from_bounds`), unless overridden with
`prior_overrides={"A_HALPHA": GaussianPrior(...), ...}` (Gaussian,
log-uniform or uniform; honoured by every sampler).

**Initialisation.**  From a least-squares fit (`init_from_mle=True`).
NUTS chains start at least 10⁻³ of the prior range inside every bound:
NUTS samples in a transformed space in which a bound lies at infinity, so
a seed sitting on a bound (a width at its cap, say) cannot be sampled
from.

**Samplers.**  NUTS (NumPyro, JAX-differentiated likelihood; default),
emcee, or nautilus.  nautilus's weighted posterior is resampled to equal
weights.

**Summary statistics** for each line:

| Field | Definition |
|---|---|
| `flux` | posterior median of A |
| `flux_err` | (median − 16th percentile, 84th percentile − median) |
| `flux_posterior` | the samples themselves |
| `snr` | median / mean of the two half-widths |
| `centroid_A`, `sigma_A` | posterior medians, with the same 16/84 errors |

Line ratios should be formed **sample by sample** from
`flux_posterior` (both lines come from the same chain), not by dividing
summary values, so that their correlations are kept.

**Convergence** (`result.convergence`, NUTS and emcee): rank-normalised
split R̂ and bulk ESS per parameter.  nautilus reports none.

---

## 9. Signal-to-noise ratios

The least-squares `LineResult` carries four, because they answer
different questions:

| Field | Definition | Use |
|---|---|---|
| `snr_int_err` (`snr`) | flux / bootstrap `flux_err` | the standard detection significance |
| `snr_int_cont` | flux / (RMS of the fit residuals in a ±15σ window excluding ±3σ, × √N_pix within ±3σ) | empirical check against the scatter actually present |
| `snr_peak_err` | model peak A/(√(2π) σ) / error-array value at the peak pixel | per-pixel significance |
| `snr_peak_cont` | model peak / residual RMS as above | per-pixel, empirical |

The MCMC result has one, `snr` (§8).  A line is "detected" by whatever
threshold the user applies; the package never thresholds the reported
flux.

---

## 10. Equivalent widths

**Rest-frame EW** (`ew0_A`):

$$W_0 = \frac{F_\mathrm{line}}{C_\lambda(\mu)\,(1+z)} ,$$

with C_λ(μ) the continuum flux density (§3) at the pixel nearest the
fitted centroid μ.  If that continuum is ≤ 0, the median observed flux
density within ±5σ of the line is used; if that is also ≤ 0 the EW is
NaN.  The observed-frame EW is `ew_obs_A` = W₀ (1 + z).

The sign convention is positive for emission.

**Uncertainty.**

- *Least squares* (`ew0_err`): the bootstrap of §7, which redraws both
  the line and the continuum in every draw.
- *MCMC* (`ew0_err`, `ew0_posterior`): every flux sample is divided by
  an independent draw C ~ N(C_λ(μ), σ_c(μ)) of the continuum (§3.1), so
  both the line-flux and the continuum terms are propagated; draws with
  C ≤ 0 are discarded.  The summary is the median and 16/84 percentiles.
  With the median-filter continuum (σ_c = NaN) only the flux term is
  propagated.

```{warning}
The continuum draws for different lines are independent, while their
true continuum errors share the same polynomial coefficients.  Do not
take the ratio of two EW posteriors and expect the continuum error to
cancel.
```

**EW upper limit** (`ew0_upper_limit(n_sigma)`): the flux upper limit of
§11 divided by C_λ (1 + z), so the flux and EW limits always agree.

---

## 11. Upper limits for non-detections

`LineResult.upper_limit(n_sigma)` and `FitResult.flux_upper_limit()`,
implemented in `jwspecfit.limits`.  The fitted flux and width of an
undetected line are set by noise and bounds, so the limit is built from
the data instead:

1. **Assumed width.**  σ² = σ_inst² + (σ_v λ/c)², with σ_v the median
   intrinsic velocity width of the lines detected at SNR ≥ 5 in the same
   spectrum (at least two needed, else σ_v = 0)
   (`limits.intrinsic_sigma_kms`).
2. **Noise through that profile.**  For a unit-area Gaussian profile g
   at the expected position, the optimal-extraction flux error from the
   error array,

   $$\sigma_F = \frac{\sqrt{\sum_i g_i^2\sigma_i^2}}{\sum_i g_i^2}\,\Delta\lambda$$

   (for white noise, σ_rms √(2√π σ Δλ)).
3. **Correlated-noise factor.**  The same profile is slid across the
   line-free residuals on each side (out to ±50σ); each recovered flux is
   divided by its predicted error, and the RMS of those ratios is the
   correction factor.  The larger of the blue- and red-side factors is
   applied.  On PRISM data it is typically ~3.
4. The limit is n_sigma × σ_F (default 3σ).

This is reported as `flux_err_cont` with the assumed σ in
`sigma_assumed_A`; `ul_method` records which estimator was used.

---

## 12. Broad components

`jwspecfit.broad.fit_with_broad` (also reached through
`fit_lines(fit_balmer_broad=True, ...)`, and the BIC stage of
`jwspecmcmc.fit_with_broad`).

1. A narrow-only fit is made first.  Broad models are tried only if the
   strongest requested Balmer line **that the spectrum covers** has
   integrated SNR ≥ `snr_threshold`.
2. Every model (narrow, + intermediate broad, + very broad, + both) is
   fitted to the same pixels with the same continuum, and scored by

   $$\mathrm{BIC} = \chi^2 + k\ln N ,$$

   with k the number of free parameters (tied ones excluded) and N the
   number of pixels in the Hα/Hβ (or [OIII], He I) windows.
3. **Bootstrap BIC** (`n_boot_bic` > 0, default 100).  Each draw adds a
   noise realisation to the data; since the data already contain one, the
   perturbed spectrum has √2 × the error and is scored with it.  The
   improvement of a model is the **median over draws of the paired
   difference** BIC_simpler − BIC_complex (both models scored on the same
   draw).  With `n_boot_bic=0` the real-data difference is used.
4. A more complex model is adopted only if its improvement is finite and
   ≥ `bic_delta` (default 6).
5. The bootstrap noise is seeded (`seed`, default 42), so a rerun selects
   the same model.

The narrow-only fit seeds the narrow lines of the broad models, but sets
no bounds on them: a narrow-only fit absorbs broad wings into the narrow
lines, so a bound derived from it would hold [NII] or narrow Hα too high.

Broad widths are bounded in velocity: intermediate (`_BROAD`) and very
broad (`_BROAD2`) tiers have their own σ_v ranges
(`broad.BROAD1_SIGMA_V_*`, `BROAD2_SIGMA_V_*`).  Broad Balmer centroids
are tied to the narrow line.

---

## 13. Lyman-α

Lyα is fitted with the asymmetric (skew-normal) profile of
Bolan et al. (2025),

$$f(\lambda) = A_\mathrm{peak}\, e^{-t^2/2}\,
\left[1+\mathrm{erf}\!\left(\frac{\alpha t}{\sqrt2}\right)\right],
\qquad t = \frac{\lambda-\mu}{\sigma},$$

whose integral is exactly A_peak σ √(2π) (the erf term integrates to
zero against the Gaussian); the least-squares flux is the profile summed
over the pixels, the MCMC flux that analytic integral.

- **Width.**  The profile is not convolved with the LSF, so σ is bounded
  to [0.5 σ_inst, max(3 Å, 3 σ_inst) × `sigma_factor`]; the floor allows
  for a compact source whose R exceeds the assumed one.
- **Starting point.**  A pre-fit within ±max(10 Å, 5 σ_inst) of the peak,
  seeded with μ blueward of the peak pixel by σ δ √(2/π),
  δ = α/√(1+α²) (the peak of a skewed profile lies redward of μ).  The
  joint fit then keeps μ within ±max(1 Å, 2 σ_inst) of the pre-fit.
- **Continuum.**  Pixels blueward of observed Lyα never enter the
  continuum fit; with `lya_break` (automatic when Lyα is fitted) the
  continuum is zero there, rising over 5 Å redward of Lyα.
- **Errors.**  From the bootstrap (§7), which fits Lyα with the other
  lines.

```{note}
On the PRISM the IGM break is itself smeared by the ~100 Å LSF, which a
sharp-break continuum cannot follow; Lyα fluxes there are only as good as
the break model.  For damping-wing work use `jwspecfit.dla`.
```

---

## 14. What the uncertainties do not include

| Effect | Included? | Size / comment |
|---|---|---|
| Photon / read noise (error array) | yes | via the bootstrap or the posterior |
| Continuum on line flux | least squares: yes; MCMC: no | small: < 1% of the flux error on tested data (§3.4) |
| Continuum on EW | yes (both) | can dominate for faint continua (§3.4) |
| Pixel-to-pixel noise correlation | **no** for fluxes; yes for upper limits | NIRSpec 1-D pixels are correlated by extraction and resampling; the white-noise flux errors are then optimistic.  §11's measured factor quantifies it for a given spectrum. |
| Flux calibration, slit losses | no | apply externally |
| Resolution model (R) | no | sets width bounds and ties (§2) |
| Width tied to other lines | implicitly | a tied line's flux error assumes the tie is right |
| Fixed doublet / branching ratios | no | atomic constants or the stated density limit |
