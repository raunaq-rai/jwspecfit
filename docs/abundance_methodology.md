# Chemical Abundance Methodology

Comprehensive documentation of the emission-line abundance derivation
pipeline implemented in `jwspecabund`.  The methodology follows the
six-step procedure of Berg et al. (2025) for UV nitrogen abundances,
extended with strong-line calibrations from Sanders et al. (2025) and a
Bayesian forward model following Cullen et al. (2025).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dust Correction](#2-dust-correction)
3. [Electron Temperature](#3-electron-temperature)
4. [Electron Density](#4-electron-density)
5. [Oxygen Abundance — 12 + log(O/H)](#5-oxygen-abundance--12--logoh)
6. [Nitrogen-to-Oxygen Ratio — log(N/O)](#6-nitrogen-to-oxygen-ratio--logno)
7. [Carbon-to-Oxygen Ratio — log(C/O)](#7-carbon-to-oxygen-ratio--logco)
8. [Ionisation Correction Factors](#8-ionisation-correction-factors)
9. [Strong-Line Calibrations](#9-strong-line-calibrations)
10. [Bayesian Forward Model](#10-bayesian-forward-model)
11. [Uncertainty Propagation](#11-uncertainty-propagation)
12. [Upper Limits for Non-Detected Ions](#12-upper-limits-for-non-detected-ions)
13. [References](#13-references)

---

## 1. Overview

Three independent methods are available, orchestrated by
`compute_abundances()` in `jwspecabund._core`:

| Method | Key reference | When to use |
|--------|--------------|-------------|
| **Direct T_e** | Berg et al. (2025) | Auroral line detected (e.g. [OIII] 4363) |
| **Strong-line** | Sanders et al. (2025) | No auroral detection; uses O3, O2, R23, O32 |
| **Bayesian forward model** | Cullen et al. (2025) | MCMC/nested sampling over (T_e, n_e, ionic abundances) |

All three methods begin with the same dust correction step.  The direct
T_e method is documented in the most detail below, as it is the primary
approach for galaxies with UV spectroscopy.


---

## 2. Dust Correction

**Source:** `jwspecabund.dust`

### 2.1 Attenuation from the Balmer decrement

The dust attenuation A_V is derived from a Balmer line ratio (typically
Hgamma/Hbeta, as Halpha may fall outside the observed wavelength range
at high redshift).  The observed ratio R_obs is compared to the
intrinsic Case B recombination ratio R_int:

```
A_V = 2.5 × log10(R_obs / R_int) / (f_den − f_num)
```

where f_num and f_den are the normalised attenuation curve values
A(lambda)/A_V at the numerator and denominator wavelengths.  Negative
A_V values are clamped to zero.

**Intrinsic Case B ratios** at T_e = 10^4 K, n_e = 100 cm^-3
(Osterbrock & Ferland 2006):

| Ratio | Value |
|-------|------:|
| Halpha / Hbeta | 2.86 |
| Hgamma / Hbeta | 0.468 |
| Hdelta / Hbeta | 0.259 |

### 2.2 Extinction / attenuation laws

Two laws are implemented:

- **Cardelli, Clayton & Mathis (1989):** Milky Way extinction curve with
  R_V = 3.1.  Polynomial fits in 1/lambda across IR, optical/NIR, and
  UV regimes.  Used for individual galaxy spectra following Berg et al.
  (2025).

- **Salim et al. (2018) / Noll et al. (2009):** Modified Calzetti et al.
  (2000) attenuation curve with a UV bump and power-law slope deviation.
  Default parameters: R_V = 3.15, delta = -0.35, bump strength
  B = 2.27.

### 2.3 Dust correction of line fluxes

Each emission-line flux is corrected using:

```
F_corr = F_obs × 10^(0.4 × A_lambda)
```

where A_lambda = A_V × f(lambda) from the chosen attenuation law.

### 2.4 Error propagation

The uncertainty on A_V is propagated from the fractional error on the
observed ratio:

```
sigma_R = R_obs × sqrt[(sigma_num / F_num)^2 + (sigma_den / F_den)^2]
sigma_AV = |dA_V/dR| × sigma_R
         = 2.5 / [R_obs × ln(10) × (f_den − f_num)] × sigma_R
```


---

## 3. Electron Temperature

**Source:** `jwspecabund.direct`

### 3.1 Primary diagnostic: [OIII] auroral-to-nebular ratio

The electron temperature of the high-ionisation zone, T_e(O2+), is
derived from the collisionally-excited line ratio:

```
R_OIII = [OIII] 4363 / ([OIII] 5007 + [OIII] 4959)
```

PyNEB's `Atom("O", 3).getTemDen()` inverts the ratio to solve for
T_e.  We use `to_eval="(L(4363))/(L(5007)+L(4959))"` with log-space
root-finding (`log=True`, `start_x=3.0`, `end_x=5.0`) to ensure
convergence at T_e > 25,000 K, which is typical of metal-poor high-z
galaxies.

### 3.2 Secondary diagnostic: [NII] auroral-to-nebular ratio

When [NII] 5756 is detected, T_e(N+) is computed from:

```
R_NII = [NII] 5756 / [NII] 6585
```

using PyNEB's `Atom("N", 2).getTemDen()`.

### 3.3 T_e–T_e relations for zone temperatures

A single auroral line measures T_e in one ionisation zone.  Empirical
relations map T_e(O2+) (the high zone) to the intermediate and low
zones:

**Intermediate zone** (O+, C2+, Si2+) — Garnett (1992):

```
T_int = [0.243 + t × (1.031 − 0.184 × t)] × 10^4 K
```

where t = T_high / 10^4.

**Low zone** (N+, S+) — two options:

1. **DESI DR2** (arXiv:2601.02463, recommended):
   ```
   T_low = 0.648 × T_high + 3270 K
   ```

2. **Classical** (Campbell, Terlevich & Melnick 1986; used in
   Garnett 1992):
   ```
   T_low = 0.7 × T_high + 3000 K
   ```


---

## 4. Electron Density

**Source:** `jwspecabund.direct`, notebook 08

### 4.1 Density-sensitive doublets

Five diagnostic doublets span the full ionisation structure, each
measured via PyNEB's `getTemDen()` at the appropriate zone temperature:

| Doublet | Ion | Zone | T_e used | PyNEB wavelengths |
|---------|-----|------|----------|-------------------|
| [S II] 6718/6732 | S+ | Low | T_low | 6718, 6732 |
| [O II] 3726/3729 | O+ | Low | T_low | 3726, 3729 |
| C III] 1907/1909 | C2+ | Intermediate | T_int | 1907, 1909 |
| Si III] 1883/1892 | Si2+ | Intermediate | T_int | 1883, 1892 |
| N IV] 1483/1486 | N3+ | High | T_high | 1483, 1487 |
| [Ar IV] 4711/4740 | Ar3+ | High | T_high | 4711, 4740 |

### 4.2 Multi-phase density assignments (Berg et al. 2025, Section 4.1)

Each ionisation zone is assigned a representative density:

- **n_e,low:** from [S II] (preferred) or [O II]
- **n_e,int:** from C III] (preferred) or Si III]
- **n_e,high:** from N IV] (preferred) or [Ar IV]
- **Fallback:** n_e = 100 cm^-3 if no diagnostic is available

### 4.3 Iterative T_e–n_e refinement

T_e and n_e are coupled (density diagnostics require a temperature,
and the temperature diagnostic requires a density).  The pipeline uses
a two-pass procedure:

1. **Pass 1:** Compute T_e with an initial density (n_e = 10^4 cm^-3).
   Derive all zone densities at this T_e.
2. **Pass 2:** Recompute T_e using the refined high-zone density from
   Pass 1.  Update zone temperatures via the T_e–T_e relations.

This converges in a single iteration because density diagnostics are
only weakly sensitive to T_e.


---

## 5. Oxygen Abundance — 12 + log(O/H)

**Source:** `jwspecabund.direct.compute_ionic_abundances`,
`compute_total_abundances`

### 5.1 Ionic abundance formula

For any ion X^i+, the ionic abundance relative to hydrogen is:

```
X^i+/H+ = (F_line / F_Hbeta) × (epsilon_Hbeta / epsilon_line)
```

where epsilon denotes the volume emissivity computed by PyNEB at the
appropriate (T_e, n_e).

For doublets (e.g. [O II] 3726+3729), the emissivities of both
components are summed before dividing.

### 5.2 Hbeta emissivity

- **T_e <= 30,000 K:** PyNEB `RecAtom("H", 1).getEmissivity(Te, ne,
  wave=4861)`, based on the Storey & Hummer (1995) recombination tables.

- **T_e > 30,000 K:** The Storey & Hummer tables do not extend beyond
  30,000 K.  We use the Aller (1984) Case B formula:

  ```
  alpha_Hbeta_eff = 3.03 × 10^-14 × (T_e / 10^4)^(-0.874)   [cm^3 s^-1]
  epsilon_Hbeta   = alpha_Hbeta_eff × h × nu_Hbeta
  ```

  where h × nu_Hbeta = hc / lambda_Hbeta with lambda_Hbeta = 4862.68 A.

### 5.3 O2+/H+ from [OIII] 5007

| Parameter | Value |
|-----------|-------|
| Line | [OIII] 5007 |
| PyNEB ion | `Atom("O", 3)`, wave=5007 |
| Zone | T_high, n_e,high |

### 5.4 O+/H+ from [OII] 3726+3729

| Parameter | Value |
|-----------|-------|
| Lines | [OII] 3726 + [OII] 3729 (summed) |
| PyNEB ion | `Atom("O", 2)`, waves=[3726, 3729] |
| Zone | T_low, n_e,low |

If only the unresolved doublet is available (e.g. from prism data),
the blended flux is used directly.

### 5.5 Total oxygen abundance

No ionisation correction factor is required for oxygen:

```
O/H = O+/H+ + O++/H+
12 + log(O/H) = 12 + log10(O/H)
```

The fraction O+/O quantifies the contribution of the low-ionisation
zone, which feeds into the ICF calculations for other elements.


---

## 6. Nitrogen-to-Oxygen Ratio — log(N/O)

**Source:** `jwspecabund.direct.compute_ionic_abundances`,
`jwspecabund.martinez25_icf`

Nitrogen abundance determination requires up to four ionic species,
each measured in a different ionisation zone.  An ionisation correction
factor (ICF) then maps the observed ionic ratio to the total N/O.

### 6.1 N+ from [NII] 6585

| Parameter | Value |
|-----------|-------|
| Line | [NII] 6585 |
| PyNEB ion | `Atom("N", 2)`, wave=6584 |
| Zone | T_low, n_e,low |

### 6.2 N2+ from NIII] 1749+1752

| Parameter | Value |
|-----------|-------|
| Lines | NIII] 1749 + NIII] 1752 (summed) |
| PyNEB ion | `Atom("N", 3)`, waves=[1749, 1752] |
| Zone | T_high, n_e,high |

### 6.3 N3+ from NIV] 1483+1486

| Parameter | Value |
|-----------|-------|
| Lines | NIV] 1483 + NIV] 1486 (summed) |
| PyNEB ion | `Atom("N", 4)`, waves=[1483, 1487] |
| Zone | T_high, n_e,high |

### 6.4 N4+ from NV 1239+1243

PyNEB has no atomic data for N V.  We implement a manual three-level
Li-like atom model:

- **Level 1:** 2s ^2S_{1/2} (g = 2)
- **Level 2:** 2p ^2P_{1/2} (g = 2), lambda = 1242.804 A,
  A_{21} = 3.37 × 10^8 s^-1
- **Level 3:** 2p ^2P_{3/2} (g = 4), lambda = 1238.821 A,
  A_{31} = 3.40 × 10^8 s^-1

Effective collision strengths Upsilon(T) are interpolated from the
tabulation of Aggarwal & Keenan (2004) / CHIANTI v10.  Statistical
equilibrium is solved for the three-level system, and the emissivity
follows from:

```
epsilon = (n_upper / n_total) × A × h × nu / n_e
```

### 6.5 Direct-sum N/O (Berg et al. 2025, Eq. 5)

The raw ionic sum is:

```
(N/O)_raw = (N+ + N2+ + N3+ + N4+) / (O+ + O2+)
```

This is biased because the ionisation structure is non-uniform — the
N3+/N ratio differs from O2+/O.  An ICF is required to correct this
(see [Section 8](#8-ionisation-correction-factors)).

### 6.6 ICF-corrected N/O

The final N/O is obtained by applying the best available ICF from
Martinez et al. (2025):

```
log(N/O) = log10(ionic_ratio × ICF)
```

The ICF selection procedure is described in
[Section 8.2](#82-martinez-et-al-2025-density-dependent-icfs).


---

## 7. Carbon-to-Oxygen Ratio — log(C/O)

**Source:** `jwspecabund.direct.compute_ionic_abundances`,
`compute_total_abundances`

### 7.1 C2+ from CIII] 1907+1909

| Parameter | Value |
|-----------|-------|
| Lines | CIII] 1907 + CIII] 1909 (summed) |
| PyNEB ion | `Atom("C", 3)`, waves=[1907, 1909] |
| Zone | T_high, n_e,high |

### 7.2 C3+ from CIV 1548+1551

| Parameter | Value |
|-----------|-------|
| Lines | CIV 1548 + CIV 1551 (summed) |
| PyNEB ion | `Atom("C", 4)`, waves=[1548, 1551] |
| Zone | T_high, n_e,high |

Note that CIV can have both stellar wind and nebular components.
Only the nebular component should be used; broad stellar P Cygni
profiles must be excluded.

### 7.3 Total C/O (Jones et al. 2023)

The C/O ratio is computed as a direct ionic sum without ICF:

```
C/O = (C2+/H+ + C3+/H+) / (O2+/H+)
```

This assumes that C2+ and C3+ together account for essentially all
carbon in the O2+ zone, following the approach of Jones et al. (2023).
No optical carbon lines are available, so C/O is purely UV-derived.


---

## 8. Ionisation Correction Factors

### 8.1 Izotov et al. (2006) — classical optical ICFs

**Source:** `jwspecabund.icf`

These are polynomial fits in x = O+/O_total, calibrated against
photoionisation models.  Used as the fallback when UV lines or
log(U) are unavailable.

| Element | Equation | Formula | Applied as |
|---------|----------|---------|------------|
| Nitrogen | Eq. 18 | ICF_N = -0.825x^2 + 0.718x + 0.853 | N/O = ICF_N × (N+/O+) |
| Neon | Eq. 19 | ICF_Ne = -0.385x^2 + 0.405x + 0.335 | Ne/O = ICF_Ne × (Ne2+/O2+) |
| Sulfur | Eq. 20 | ICF_S = 0.013 + x(5.986 + x(-21.085 + x(26.462 - 11.282x))) | S/O = ICF_S × (S+ + S2+)/O |
| Argon | Eqs. 22-23 | ICF_Ar = 0.157 + x(3.119 + x(-6.185 + 4.517x)) | Ar/O = ICF_Ar × (Ar2+/O2+) |

### 8.2 Martinez et al. (2025) — density-dependent ICFs

**Source:** `jwspecabund.martinez25_icf`

These ICFs are calibrated on Cloudy photoionisation models and account
for the dependence of the ionisation correction on electron density,
ionisation parameter, and metallicity.  They are the recommended
approach for UV N/O determinations (Berg et al. 2025, Section 4.2).

#### Ionisation parameter diagnostics

Before applying an ICF, the ionisation parameter log(U) must be
estimated from one of two diagnostics:

1. **N43 diagnostic (recommended):**
   ```
   log(N43) = log10(NIV] 1486 / NIII] 1750)
   log(U_high) = f(log(N43), Z/Zsun; n_e)
   ```
   Density-insensitive.  Preferred when NIV] and NIII] are both
   detected.  Valid range: -3.0 <= log(N43) <= 0.5.

2. **O32 diagnostic (fallback):**
   ```
   log(O32) = log10([OIII] 5007 / [OII] 3727)
   log(U_int) = f(log(O32), Z/Zsun; n_e)
   ```
   Density-sensitive.  Valid range: -1.0 <= log(O32) <= 2.5.

Both diagnostics use bicubic surface fits (Martinez et al. 2025,
Table 3) interpolated between electron density grid points at
log10(n_e) = [2, 3, 4, 5, 6]:

```
f(x, y) = A + Bx + Cy + Dxy + Ex^2 + Fy^2 + Gxy^2 + Hyx^2 + Ix^3 + Jy^3
```

The metallicity input is Z/Z_sun = 10^(12+log(O/H) - 8.69), where
12 + log(O/H)_sun = 8.69 (Asplund et al. 2009).

Overall validity ranges: 0.05 <= Z/Z_sun <= 0.40, -3.5 <= log(U) <= -1.0.

#### Five N/O ICFs (Table 4)

Each ICF corrects a specific ionic ratio to total N/O:

| ICF | Ionic ratio | Notes |
|-----|-------------|-------|
| **ICF 5** | (N2+ + N3+) / O2+ | Pure UV.  Most robust at log(U) > -2.75 with corrections < 5%.  **Most preferred.** |
| **ICF 4** | (N+ + N2+) / (O+ + O2+) | Mixed optical + UV.  Robust across a wide range of conditions. |
| **ICF 2** | N2+ / O2+ | UV single-ion.  Overestimates at low log(U). |
| **ICF 1** | N+ / O+ | Classical optical.  Simplest but least accurate at high ionisation. |
| **ICF 3** | N3+ / O2+ | Large corrections (100–15,000%); returned in log-space.  **Least preferred.** |

#### Selection priority

`compute_NO_martinez25()` selects the best ICF based on which ionic
abundances are available, in order of decreasing preference:

1. ICF 5 — requires N2+, N3+, O2+
2. ICF 4 — requires N+, N2+, O+, O2+
3. ICF 2 — requires N2+, O2+
4. ICF 1 — requires N+, O+
5. ICF 3 — requires N3+, O2+ (last resort)

The final N/O is:

```
N/O = ionic_ratio × ICF(log(U), Z/Z_sun, n_e)
```


---

## 9. Strong-Line Calibrations

**Source:** `jwspecabund.strong_line`

When no auroral line is detected, 12 + log(O/H) is estimated from
strong-line ratios using the simultaneous polynomial calibrations of
Sanders et al. (2025).

### 9.1 Diagnostic ratios

| Ratio | Definition | Lines required |
|-------|-----------|----------------|
| O3 | log10([OIII] 5007 / Hbeta) | OIII_5007, HBETA |
| O2 | log10([OII] 3726+3729 / Hbeta) | OII_doublet, HBETA |
| R23 | log10(([OIII] 5007 + [OII] 3726+3729) / Hbeta) | OIII_5007, OII_doublet, HBETA |
| O32 | log10([OIII] 5007 / [OII] 3726+3729) | OIII_5007, OII_doublet |

Each ratio R is modelled as a polynomial in Z = 12 + log(O/H):

```
R(Z) = c0 + c1 × (Z - 8.0) + c2 × (Z - 8.0)^2 + ...
```

### 9.2 Simultaneous fit

All available ratios are fit simultaneously by minimising the combined
chi-squared:

```
chi^2(Z) = sum_i [(R_obs,i - R_model,i(Z))^2 / (sigma_obs,i^2 + sigma_cal,i^2)]
```

where sigma_cal = sqrt(sigma_R,fit^2 + sigma_R,int^2) accounts for
both the fit residual and intrinsic scatter in the calibration.  The
best-fit Z is found via bounded scalar minimisation over
6.5 <= Z <= 9.0.

### 9.3 Monte Carlo uncertainty

The uncertainty on Z is estimated by perturbing each ratio by its
observational error (1000 MC iterations) and re-fitting.  The reported
uncertainty is the standard deviation of the posterior.


---

## 10. Bayesian Forward Model

**Source:** `jwspecabund.forward`

An alternative approach following Cullen et al. (2025): free
parameters (T_e, n_e, ionic abundances) are sampled via MCMC or nested
sampling.  For each parameter vector, collisionally-excited line
emissivities from PyNEB and the Aller (1984) Hbeta recombination
formula predict line flux ratios that are compared to observations
via a Gaussian likelihood.

This method self-consistently accounts for the T_e–n_e coupling and
non-Gaussian posterior distributions, but is more computationally
expensive than the direct method.


---

## 11. Uncertainty Propagation

### 11.1 Monte Carlo approach (direct T_e method)

The full abundance pipeline is wrapped in a Monte Carlo loop
(default N_mc = 1000 iterations):

1. **Draw perturbed fluxes:** F_perturbed ~ N(F_observed, sigma_F) for
   each emission line.
2. **Re-derive all quantities:** dust correction, T_e, n_e, ionic
   abundances, log(U), ICFs, total abundances.
3. **Accumulate posterior arrays:** {O/H, N/O, C/O, T_e, ...}.
4. **Report:** median and 16th/84th percentile intervals.

### 11.2 MCMC posterior propagation

When MCMC chains are available from `jwspecmcmc`, posterior flux samples
are drawn directly from the chains rather than assuming Gaussian errors.
The dust correction is applied per-sample:

```
F_corr,i = F_obs,i × 10^(0.4 × k(lambda) × A_V,i)
```

where A_V,i is re-derived from the Balmer decrement in each sample.
This propagates the full posterior covariance between line fluxes
through to the final abundances.

### 11.3 Pre-computed emissivity grids

To avoid calling PyNEB ~10^6 times (N_mc × N_lines), emissivity grids
are pre-computed as a function of T_e(high) on a fine grid (e.g.
5,000–80,000 K, 500 points).  Each MC iteration interpolates into
these grids, giving a factor ~100× speedup while preserving the
parametric dependence on T_e.


---

## 12. Upper Limits for Non-Detected Ions

**Source:** `jwspecabund._core._compute_continuum_rms_limits`,
`_compute_ionic_upper_limits`

When an emission line is not detected (flux consistent with zero or
below the SNR threshold), we compute a 3σ upper limit on its flux
and convert this to an ionic abundance upper limit.  This is essential
for constraining N/O and C/O when UV lines (e.g. N III], N IV], C IV)
fall below the detection threshold.

### 12.1 Continuum-RMS flux upper limits

The flux upper limit is derived from the noise in the continuum around
the expected line position, independent of the emission-line fit:

```
F_ul = n_sigma × RMS_cont × sigma_inst × sqrt(2π)
```

where:

- **RMS_cont:** root-mean-square of the fit residuals (data minus
  model) in a window of ±5σ around the expected line centre, excluding
  the central ±2σ where the line would sit.  If fewer than 3 pixels
  fall in this annulus, the window is expanded to ±10σ with ±3σ
  exclusion.
- **sigma_inst:** the instrumental line width (in Angstrom), computed
  from the grating resolving power R as σ = λ_obs / (2.355 × R).
  When grating metadata is unavailable (e.g. spectra loaded from npz
  files), R is estimated from the pixel sampling via
  `R_from_pixels(wave_um)`.
- **n_sigma:** detection threshold, default 3.

This method is preferred over using the fit error because it measures
actual noise at the line position rather than relying on bootstrap or
MCMC error estimates, which can be inflated by unconstrained parameters
(particularly for faint UV secondary doublet members).

### 12.2 Dust correction of upper limits

Flux upper limits are corrected for dust attenuation using the same
A_V and attenuation law applied to detected lines:

```
F_ul,corr = F_ul × 10^(0.4 × A_lambda)
```

This ensures upper limits are on the same (intrinsic) flux scale as
the detected line fluxes used for ionic abundances.

### 12.3 Doublet upper limits

For doublets where both members are undetected (e.g. N III] 1749+1752),
the individual member upper limits are combined in quadrature:

```
F_ul,doublet = sqrt(F_ul,1^2 + F_ul,2^2)
```

This is the correct treatment for independent noise in each member,
rather than linear summation which would overestimate the upper limit.

### 12.4 Ionic abundance upper limits

The flux upper limit is converted to an ionic abundance upper limit
using the same emissivity-ratio formula as for detected lines
([Section 5.1](#51-ionic-abundance-formula)):

```
(X^i+/H+)_ul = (F_ul / F_Hbeta) × (epsilon_Hbeta / epsilon_line)
```

### 12.5 Abundance ratio upper limits

Upper limits on N/O and C/O are computed by summing detected ionic
abundances with upper limits for non-detected ions, divided by total
oxygen:

```
(N/O)_ul = (N_detected + N_upper_limits) / (O+ + O2+)
log(N/O)_ul = log10((N/O)_ul)
```

These are reported in the `AbundanceResult.summary()` output alongside
the ionic upper limit details.

### 12.6 Fallback to fit errors

If the continuum-RMS method cannot be applied (e.g. the spectrum
object is unavailable, or the residuals do not cover the line
position), the pipeline falls back to the fit-error method:

```
F_ul = n_sigma × sqrt(sum(sigma_i^2))
```

where sigma_i are the per-member flux errors from bootstrap or MCMC.
The `ionic_ul_details` dictionary records which method was used for
each ion (`"method": "continuum_rms"` or `"method": "fit_error"`).


---

## 13. References

| Reference | Context |
|-----------|---------|
| Aggarwal, K. M. & Keenan, F. P., 2004, A&A, 427, 763 | N V effective collision strengths (CHIANTI v10) |
| Aller, L. H., 1984, Physics of Thermal Gaseous Nebulae (Reidel) | Hbeta Case B recombination formula for T_e > 30,000 K |
| Asplund, M. et al., 2009, ARA&A, 47, 481 | Solar oxygen abundance: 12 + log(O/H)_sun = 8.69 |
| Berg, D. A. et al., 2025, arXiv:2511.13591 | Six-step UV nitrogen abundance procedure |
| Calzetti, D. et al., 2000, ApJ, 533, 682 | Starburst attenuation curve (base for Salim+18) |
| Campbell, A., Terlevich, R. & Melnick, J., 1986, MNRAS, 223, 811 | T_e–T_e relation: T_low = 0.7 × T_high + 3000 |
| Cardelli, J. A., Clayton, G. C. & Mathis, J. S., 1989, ApJ, 345, 245 | Milky Way extinction law |
| Cullen, F. et al., 2025, (EXCELS) | Bayesian forward model for emission-line abundances |
| DESI Collaboration, 2026, arXiv:2601.02463 | Updated T_e–T_e relation: T_low = 0.648 × T_high + 3270 |
| Garnett, D. R., 1992, AJ, 103, 1330 | Classical T_e–T_e relations for intermediate/low zones |
| Izotov, Y. I. et al., 2006, A&A, 448, 955 | ICF equations 18–23 for N, Ne, S, Ar |
| Jones, T. et al., 2023 | C/O from direct ionic sum: (C2+ + C3+) / O2+ |
| Martinez, M. A., Berg, D. A. et al., 2025, arXiv:2510.21960 | Density-dependent ICFs and log(U) diagnostics |
| Noll, S. et al., 2009, A&A, 499, 69 | UV bump + slope modification to Calzetti curve |
| Osterbrock, D. E. & Ferland, G. J., 2006, Astrophysics of Gaseous Nebulae and Active Galactic Nuclei (University Science Books) | Case B recombination ratios |
| Salim, S. et al., 2018, ApJ, 859, 11 | UV slope and bump calibration for dust attenuation |
| Sanders, R. L. et al., 2025 | Strong-line metallicity calibrations (O3, O2, R23, O32) |
| Storey, P. J. & Hummer, D. G., 1995, MNRAS, 272, 41 | H I recombination emissivity tables (valid to 30,000 K) |


---

## Appendix: Zone Assignment Summary

| Ion | Lines | Zone | T_e | n_e |
|-----|-------|------|-----|-----|
| O2+ | [OIII] 5007 | High | T_high | n_e,high |
| O+ | [OII] 3726+3729 | Low | T_low | n_e,low |
| N+ | [NII] 6585 | Low | T_low | n_e,low |
| N2+ | NIII] 1749+1752 | High | T_high | n_e,high |
| N3+ | NIV] 1483+1486 | High | T_high | n_e,high |
| N4+ | NV 1239+1243 | High | T_high | n_e,high |
| C2+ | CIII] 1907+1909 | High | T_high | n_e,high |
| C3+ | CIV 1548+1551 | High | T_high | n_e,high |
| S+ | [SII] 6718+6732 | Low | T_low | n_e,low |
| S2+ | [SIII] 9069 | Mid | (T_high+T_low)/2 | n_e,low |
| Ne2+ | [NeIII] 3869 | High | T_high | n_e,high |
| Ar2+ | [ArIII] 7136 | Mid | (T_high+T_low)/2 | n_e,low |
