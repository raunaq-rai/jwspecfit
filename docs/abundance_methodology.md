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

### 2.3 Multi-Balmer A_V

When multiple Balmer lines are detected (Hgamma through H10), individual
A_V values are derived from each line's ratio to Hbeta and combined as
an SNR-weighted average via `compute_Av_multi_balmer()`.  The weights
are the inverse-variance of each A_V estimate.

**Excluded lines** (blended at typical NIRSpec resolution):
- Hepsilon (3971 Å): blended with [NeIII] 3968 Å (Δλ = 3 Å)
- H8 (3890 Å): blended with HeI 3889 Å (Δλ = 0.4 Å)

**Included lines:**

| Line | λ_rest (Å) | Intrinsic ratio to Hβ |
|------|-----------|----------------------|
| Hgamma | 4342.90 | 0.468 |
| Hdelta | 4102.89 | 0.259 |
| H9 | 3836.48 | 0.0731 |
| H10 | 3799.00 | 0.0530 |

**Single-pair option.** Passing `balmer_pair=(numerator, denominator)`
to `compute_abundances` (or calling `compute_Av_balmer_pair()` directly)
restricts the derivation to one chosen decrement — e.g.
`("Ha", "HBETA")` for Hα/Hβ — bypassing the SNR-weighted multi-line fit.
This is useful when only one Balmer ratio is high-SNR and the fainter
lines would otherwise bias or inflate the A_V error.  The intrinsic
Case-B ratio is taken from the ladder above (extended with Hα = 2.86 and
Hβ = 1.00).  A_V is still applied as a single point value to all draws;
its formal error is reported on the result but not marginalised into the
abundances unless A_V resampling is explicitly enabled (§2.4).

### 2.4 Dust correction of line fluxes

Each emission-line flux is corrected using:

```
F_corr = F_obs × 10^(0.4 × A_lambda)
```

where A_lambda = A_V × f(lambda) from the chosen attenuation law.

### 2.5 Error propagation

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

### 3.2 UV fallback diagnostic: O III] 1666 / ([OIII] 5007 + 4959)

When [OIII] 4363 is unavailable or has insufficient SNR, the O III]
1666 Å intercombination line provides an alternative T_e diagnostic:

```
R_1666 = O III] 1666 / ([OIII] 5007 + [OIII] 4959)
```

The 1666 Å line arises from the 6→3 transition, whose upper level sits
7.48 eV above ground (against 5.35 eV for 4363 and 2.51 eV for 5007),
making this ratio more temperature-sensitive.  PyNEB's `Atom("O", 3).getTemDen()` is used
with `to_eval="(L(1666))/(L(5007)+L(4959))"` and solver range
extended to 250,000 K (`start_x=3.0, end_x=5.4`).  The ratio is
monotonically increasing with T_e, so the solution is unique.

This diagnostic is automatically used as a fallback in
`compute_abundances()` when [OIII] 4363 is not detected but O III]
1666 and the [OIII] 5007+4959 nebular lines are available.

### 3.3 Self-consistent T_e **and** n_e from UV + optical oxygen

**Source:** `jwspecabund.direct.compute_Te_ne_OIII`, following Hsiao et
al. (2026), [arXiv:2608.20339](https://arxiv.org/abs/2608.20339); the
method is due to Berg (2018) and Arellano-Córdova et al. (2020).

Sections 3.1 and 3.2 each solve T_e from **one** ratio at an assumed or
borrowed density.  That is safe only while the density stays well below
the critical density of the nebular line.  [OIII] 5007 has
n_crit ≈ 7 × 10⁵ cm⁻³, so above n_e ~ 10⁵ cm⁻³ it is collisionally
de-excited: the observed 4363/5007 ratio rises, and if it is inverted at
an assumed low density the temperature comes out too high and the oxygen
abundance too low.  Hsiao et al. show this misclassifies ordinary
galaxies as extremely metal-poor by up to **1.1 dex**.

O III] 1666 has a far higher critical density, so it does not suffer the
same suppression.  With all three lines measured there are two
independent ratios for the two unknowns, and both can be solved at once:

```
R_1 = [OIII] 4363 / ([OIII] 5007 + [OIII] 4959)
R_2 = O III] 1666 / [OIII] 4363
```

Each ratio alone traces a curve of (T_e, n_e) pairs that reproduce it.
Following Hsiao et al. §IV.3, those T_e(n_e) curves are built over
log(n_e/cm⁻³) = 0–7 in **1,000 steps**, and the solution is where they
cross — the one pair that reproduces both ratios.  Where they do not
cross, the closest approach is used, as in the paper.  A third curve
([OIII] 5007 / O III] 1666) is computed for display; it is the ratio of
the other two and carries no independent information.

Uncertainties come from the flux posterior, again as in the paper: every
draw is re-solved, and T_e and n_e are the posterior medians with 68 %
percentile errors.

**Upper limits.** Below n_e ~ 10⁴·⁵ cm⁻³ both ratios go density-flat and
the curves run parallel, so no density can be picked out — only bounded.
This is detected by comparing the 1σ lower bound of the density
posterior against the density at which R_1 first departs from its
low-density limit by more than the measured error on the ratio.  When it
falls below, `ne_is_upper_limit` is set, the 84th percentile is reported
as the 1σ limit, and the single-ratio temperature of §3.1 is adopted
instead.  Hsiao et al. report exactly such a limit for the one object in
their sample where the curves do not converge.

The borrowed density is capped at that limit **only if the curves
actually cross** (and do so in at least half the posterior draws).  When
they never cross, no (T_e, n_e) pair reproduces both ratios at any
density, and the "limit" is merely where near-parallel curves come
closest — a position that moves with a fraction of the atomic-data
systematic on λ1666.  Capping on that would let a non-detection of
density silently override a density measured from another ion; on a real
stacked spectrum it shifted 12+log(O/H) by 0.05 dex.  In that case the
limit is reported, a warning is logged, and nothing downstream changes.

**Atomic data.** All four lines must come from one atom, or the
"intersection" is between curves drawn from different atomic physics.
PyNEB's default `o_iii_coll_SSB14.dat` tabulates only 5 levels and cannot
represent λ1666 at all, so the joint solve uses the 6-level
`o_iii_coll_TZ17.dat` (Tayal & Zatsarinny 2017) — the same dataset §3.2
already uses.  `oiii_coll_file` selects an alternative:
`"o_iii_coll_AK99.dat"` (Aggarwal & Keenan 1999) reproduces Hsiao et al.
exactly and shifts T_e by 1–2 %, n_e by ≤0.1 dex and log(O²⁺/H⁺) by
~0.02 dex (worst case 0.09).

**When it is used.** `compute_abundances(..., self_consistent_OIII="auto")`
is the default and engages whenever λ1666, λ4363 and λ5007 all clear
`snr_auroral`.  Pass `True` to force the attempt or `False` to restore the
single-ratio behaviour.  Whichever method is adopted, the result always
reports what each single ratio would have given on its own, in
`alt_results["direct_4363"]` and `alt_results["direct_1666"]` and as a
side-by-side table in `summary()`.

`jwspecabund.plot_te_ne_diagnostic()` draws the curves, their 1σ bands
and the posterior contours — the figure 3 of Hsiao et al.

**Worked example.** Synthetic O III fluxes for
12 + log(O²⁺/H⁺) = 8.000 at 5 % flux errors:

| truth | self-consistent | [OIII] 4363 only (n_e borrowed) |
|---|---|---|
| T_e = 15,000 K, n_e = 10⁵ | 7.998 (T_e = 14,998 K) | 7.877 (T_e = 16,254 K) |
| T_e = 13,000 K, n_e = 10⁶ | 8.005 (T_e = 12,976 K) | 6.849 (T_e = 33,310 K) |

---

### 3.4 Secondary diagnostic: [NII] auroral-to-nebular ratio

When [NII] 5756 is detected, T_e(N+) is computed from:

```
R_NII = [NII] 5756 / [NII] 6585
```

using PyNEB's `Atom("N", 2).getTemDen()`.

### 3.5 T_e–T_e relations for zone temperatures

A single auroral line measures T_e in one ionisation zone.  The
`Te_relation` argument maps T_e(O2+) (the high zone) to the intermediate
and low zones.  Ions are assigned to zones by ionisation potential:

- **High zone** (O2+, Ne2+, N3+, C3+) — T_high = T_e([OIII])
- **Intermediate zone** (N2+, C2+, S2+, Ar2+) — T_int
- **Low zone** (O+, N+, S+) — T_low

**Default `"3_tier"`** — Garnett (1992) relations throughout, following
Martinez+2025 ("Under Pressure", arXiv:2510.21960):

```
T_int = 0.83 × T_high + 1700 K
T_low = 0.70 × T_high + 3000 K
```

**Other `Te_relation` options:**

- **`"classical"` / `"garnett"`** — Garnett (1992) low-zone relation
  (`T_low = 0.7 × T_high + 3000`; Campbell, Terlevich & Melnick 1986).
- **`"desi"`** — DESI DR2 (arXiv:2601.02463): `T_low = 0.648 × T_high + 3270`.


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

- **n_e,low:** from [S II] (preferred), [O II], or [Si III] (UV fallback)
- **n_e,int:** from C III]
- **n_e,high:** from N IV] (preferred) or [Ar IV]
- **n_e(O2+):** [Ar IV] 4711/4740 preferred for the O2+/Ne2+ zone, else n_e,int
- **Fallback:** a redshift-evolution fit `ne_zone_fallback(zone, z) = A·(1+z)^p`
  per zone when no diagnostic is available — **mid** `1110·(1+z)^1.93` and
  **high** `5400·(1+z)^1.62` from **Martinez+2025 (arXiv:2510.21960), Eqs 4 & 5**;
  **low** `54·(1+z)^1.2` from Abdurro'uf+2024 (arXiv:2404.16201). Martinez+2025
  fit only the intermediate and high zones and themselves adopt Abdurro'uf+2024
  for the low-ionisation zone, so no low-zone Martinez relation exists.

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

### 4.4 Ion-to-zone temperature and density assignments

Each ionic abundance is evaluated at the temperature **and** density of the
ionisation zone the ion traces (`compute_ionic_abundances`), following the
zone assignment of Martinez+2025 (arXiv:2510.21960).  Within the
high-ionisation side there are **two** densities: O2+ and Ne2+ use the
dedicated **O2+-zone density `n_e(O2+)`** ([Ar IV]-preferred), whereas the more
highly ionised N3+/N4+/C3+ use the **high-zone density `n_e,high`** (N IV] /
[Ar IV]).

| Ion (X^i+/H+) | Line(s) | Zone | T_e | n_e |
|---------------|---------|------|-----|-----|
| O+  | [O II] 3726+3729 | low | T_low | n_e,low |
| N+  | [N II] 6585 | low | T_low | n_e,low |
| S+  | [S II] 6718+6732 | low | T_low | n_e,low |
| C+  | C II] 2324+2326 | low | T_low | n_e,low |
| N2+ | N III] 1749+1752 | intermediate | T_int | n_e,int |
| C2+ | C III] 1907+1909 | intermediate | T_int | n_e,int |
| S2+ | [S III] 9069 | intermediate | T_int | n_e,int |
| Ar2+ | [Ar III] 7136 | intermediate | T_int | n_e,int |
| O2+ | [O III] 5007 | high (O2+) | T_high | n_e(O2+) |
| Ne2+ | [Ne III] 3869 | high (O2+) | T_high | n_e(O2+) |
| N3+ | N IV] 1483+1486 | high | T_high | n_e,high |
| N4+ | N V 1239+1243 | high | T_high | n_e,high |
| C3+ | C IV 1548+1551 | high | T_high | n_e,high |

where `n_e(O2+)` is the O2+/Ne2+-zone density ([Ar IV] 4711/4740 preferred →
C III] → z-fallback), and `n_e,int` / `n_e,high` are the intermediate / high
zone densities of §4.2.  Because these collisionally excited lines are
density-insensitive below ~10^4–10^5 cm^-3, the density choice affects the
ionic abundances only weakly; it matters most for the density-interpolated
Martinez ICFs.

**ICF interpolation densities.**  The Martinez N/O ICFs — e.g. ICF 5,
`NppNppp_Opp` = (N2+ + N3+) / O2+ — are interpolated at the **high-zone**
`log(U)` and `n_e,high`; the Martinez C/O ICF is interpolated at the
**intermediate-zone** `log(U_int)` and `n_e,int` (§8.2–8.3).  So for ICF 5 the
three ions enter the ratio at their own zones — N2+ at `(T_int, n_e,int)`, N3+
at `(T_high, n_e,high)`, O2+ at `(T_high, n_e(O2+))` — and the correction
factor itself is taken at the high-zone `log(U)` and `n_e,high`.


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
[Section 8.2](#82-martinez-et-al-2025--density-dependent-icfs).


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

### 7.3 C+ from CII] 2326

| Parameter | Value |
|-----------|-------|
| Lines | CII] 2324 + CII] 2326 (multiplet sum) |
| PyNEB ion | `Atom("C", 2)`, waves=[2323, 2325, 2326, 2327, 2328] |
| Zone | T_low, n_e,low |

The CII] λ2326 multiplet (five components from 2323 to 2328 Å) traces
C+ in the low-ionisation zone.  When detected, the C+ ionic abundance
is included in the carbon budget.

### 7.4 Pure-UV C/O — C III] 1907,1909 / O III] 1661,1666

**Source:** `jwspecabund.direct.compute_CppOpp_uv`; Garnett et al. (1995),
Erb et al. (2010), Berg et al. (2016).

The C/O of §7.5 divides C²⁺ (from C III] λ1907,1909) by O²⁺ (from
[O III] λ5007), both normalised to Hβ.  Hβ cancels algebraically, but the
surviving ratio still spans the whole UV-to-optical baseline.  Pairing
C III] with the *UV* oxygen lines instead keeps everything within 240 Å:

```
C2+/O2+ = I(C III] 1907+1909) / I(O III] 1661+1666)
          × eps_O(1661+1666) / eps_C(1907+1909)
```

C²⁺ (24.4–47.9 eV) and O²⁺ (35.1–54.9 eV) span nearly the same ionisation
range, so the same C²⁺/O²⁺ → C/O ICF applies (the Martinez ICF of §8; the
Garnett+97 ICF does **not**, as it assumes C³⁺ has been added in).

**Why it is better conditioned.** Reddening leverage on log(C/O), measured
on a real spectrum:

| | pure-UV (1907 vs 1666) | UV-optical (1907 vs 5007) |
|---|---|---|
| Cardelli | +0.042 dex/mag | +0.603 dex/mag |
| Salim | −0.036 dex/mag | +1.328 dex/mag |

A factor 14–37 less, with the sign flipping between laws — i.e.
consistent with zero.  It also needs no Hβ, which removes the relative
flux calibration between the gratings covering the UV and the optical.

**What it costs.** Sensitivity to the assumed conditions: about +0.03 dex
per +1700 K in T_e, and −0.03 dex if n_e falls from 3×10⁴ to 10³ cm⁻³
(C III] λ1907 has a low critical density, so this diagnostic wants the
density of §3.3).  And λ1661/λ1666 are ~20× fainter than λ5007.

Both ions are evaluated at a single T_e and n_e — the O²⁺-zone values —
because the diagnostic's validity rests on them being co-spatial.

**Reporting.** This is always computed alongside the adopted C/O when the
lines are present, as `alt_results["CO_uv"]`, with its own posterior. It
is not adopted automatically: the two routes can differ by ~0.2 dex, and
which to trust depends on the λ1661/λ1666 check below.

### 7.5 O III] 1661/1666 — a free consistency check

λ1660.81 and λ1666.15 are both decays of the same upper level (⁵S₂ → ³P₁
and ³P₂), so their ratio is a pure branching ratio: **0.402, with no
dependence on T_e, n_e, density or abundance**.  Any departure is a
problem with the line measurement, not the physics.

`check_oiii_uv_doublet()` tests it, and `compute_abundances` runs it
automatically whenever both components are detected, recording the result
in `AbundanceResult.oiii_uv_doublet` and `diagnostics`, and logging a
warning past 3σ.  It is worth checking before trusting anything that
leans on λ1666 — both the pure-UV C/O above and the joint T_e–n_e solve of
§3.3.

### 7.6 Total C/O

C/O uses the intermediate (C2+/O2+) ionisation zone.  The method is set by
`co_icf_method` (default `"auto"`).

**Default — Martinez (2026, in prep.) C2+/O2+ ICF (Section 8.3):**

Whenever log(U), Z, and both C2+ and O2+ are available, C/O is computed with
the Martinez (in prep.) density-interpolated C2+/O2+ ICF, evaluated at the
intermediate-zone log(U) and density:

```
C/O = ICF_C(Martinez; logU, Z, n_e) × (C2+/H+) / (O2+/H+)
```

**Fallback — Garnett+1997 ICF / direct sum:**

When the Martinez ICF cannot be computed (no log(U), or C2+/O2+ missing), the
legacy tiered path is used:

- **C+ detected** — direct ionic sum:
  ```
  C/O = (C+ + C2+ + C3+) / (O+ + O2+)
  ```
- **C+ not detected** — Garnett (1997) ICF (Jones et al. 2023 when O+ is also
  missing, ICF = 1):
  ```
  C/O = ICF_C × (C2+ + C3+) / O2+,    ICF_C = (O+ + O2+) / O2+
  ```


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

#### Inputs outside the calibration bounds

The Martinez et al. (2025) coefficients are polynomial fits over a
bounded calibration domain (the validity ranges above).  The bicubic
surface diverges if it is evaluated by extrapolation, so an input that
falls outside its range — `log(O32)`, `log(N43)`, `Z/Z_sun`, or the
resulting `log(U)` — is **rejected, not clipped**: the surface returns
`NaN` rather than a boundary value or an extrapolated number.  This
keeps unphysical, uncalibrated values out of the reported N/O entirely.

This rejection affects the ratios that depend on `log(U)` and `Z/Z_sun` —
**N/O** and, by default, **C/O** (Section 8.3).  O/H (direct `T_e`), `T_e`,
and the S/O, Ne/O and Ar/O ratios do not use a Martinez ICF and are
unaffected — see Section 11.3 for how this is handled in the Monte Carlo /
posterior loops.

### 8.3 Martinez et al. (2026, in prep.) — C²⁺/O²⁺ C/O ICF

**Source:** `jwspecabund.martinez25_icf` (`icf_CppOpp`)

The default C/O ICF (`co_icf_method="auto"`) is the Martinez (in prep.)
correction from the C2+/O2+ ionic ratio to total C/O.  Like the N/O ICFs it is
a density-interpolated function of log(U) and Z/Z_sun, evaluated at the
intermediate (C2+) ionisation zone: the log(U) is recomputed at the
intermediate-zone density (from the O32 diagnostic; N43 is density-insensitive)
and the ICF is interpolated in n_e.  It requires C2+, O2+, log(U) and Z; when
any of these is missing it falls back to the Garnett (1997) ICF or a direct
ionic sum (Section 7.4).  Inputs outside the calibration domain are rejected
(NaN), as for the N/O ICFs.


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

### 11.3 Out-of-bounds resampling and the O/H–N/O decoupling

The Martinez et al. (2025) log(U)/ICF surfaces are only valid inside
their calibration domain (Section 8.2).  Inputs outside that domain are
**rejected** (set to `NaN`), not clipped — see "Inputs outside the
calibration bounds" in Section 8.2.  In the uncertainty loops this is
handled per draw so the reported sample counts stay meaningful:

1. **Only N/O is gated.**  If a draw's `log(O32)`/`log(N43)`, `Z/Z_sun`,
   or resulting `log(U)` falls outside the bounds, `log(U)` is set to
   `NaN` for that draw.  The Martinez N/O ICF then returns `NaN`, so the
   draw contributes nothing to N/O.

2. **O/H and the other ratios are kept.**  O/H (direct `T_e`), `T_e`, and
   the X/O ratios (C/O, S/O, Ne/O, Ar/O) are computed and recorded for
   **every** drawn sample, in-bounds or not — an out-of-bounds draw only nulls
   N/O; the default Martinez C/O ICF simply falls back to Garnett+97 for that
   draw (Section 8.3).  They are never discarded on account of an
   out-of-bounds N/O.

3. **Resampling to the requested count.**  The loop keeps drawing until
   `n_mc` (Gaussian MC) or `n_posterior` (MCMC pool) *in-bounds* N/O
   draws have been collected, so N/O is sampled to the requested depth
   rather than silently under-sampled.  Solver failures and missing-line
   `NaN`s are *not* re-drawn (re-drawing cannot fix them) and count
   toward the target.

4. **Different lengths are expected.**  Because out-of-bounds draws are
   re-drawn for N/O but retained for O/H, the O/H posterior generally
   holds **more** finite samples than the N/O posterior.  The two arrays
   are independent; do not assume they are the same length.

5. **Safety cap.**  To guarantee termination for an object centred
   outside the bounds (e.g. `Z/Z_sun < 0.05`, where acceptance ≈ 0), the
   total number of attempts is capped at 20× the requested count.  If the
   target is not reached, a `WARNING` is logged
   (`"... in-bounds N/O draws ..."`) and N/O is reported from whatever
   in-bounds draws were collected; O/H and the other ratios remain fully
   sampled.

```{note}
If you set the `jwspecabund` logger above `WARNING`, the under-sampling
message is suppressed.  To detect objects where N/O could not be sampled
to the requested depth, inspect the posterior directly, e.g.
`np.isfinite(result.NO_posterior).sum()`.
```

### 11.4 Pre-computed emissivity grids

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
| Abdurro'uf et al., 2024, ApJ, 973, 47 (arXiv:2404.16201) | Low-ionisation n_e redshift-evolution fallback: 54·(1+z)^1.2 |
| Aggarwal, K. M. & Keenan, F. P., 1999, ApJS, 123, 311 | Alternative 6-level O III collision strengths (oiii_coll_file) |
| Aggarwal, K. M. & Keenan, F. P., 2004, A&A, 427, 763 | N V effective collision strengths (CHIANTI v10) |
| Aller, L. H., 1984, Physics of Thermal Gaseous Nebulae (Reidel) | Hbeta Case B recombination formula for T_e > 30,000 K |
| Asplund, M. et al., 2009, ARA&A, 47, 481 | Solar oxygen abundance: 12 + log(O/H)_sun = 8.69 |
| Berg, D. A. et al., 2025, arXiv:2511.13591 | Six-step UV nitrogen abundance procedure |
| Arellano-Córdova, K. Z. et al., 2020, MNRAS, 496, 1051 | Self-consistent T_e-n_e from UV + optical O III |
| Berg, D. A., 2018, ApJ, 859, 164 | Self-consistent T_e-n_e from UV + optical O III |
| Calzetti, D. et al., 2000, ApJ, 533, 682 | Starburst attenuation curve (base for Salim+18) |
| Campbell, A., Terlevich, R. & Melnick, J., 1986, MNRAS, 223, 811 | T_e–T_e relation: T_low = 0.7 × T_high + 3000 |
| Cardelli, J. A., Clayton, G. C. & Mathis, J. S., 1989, ApJ, 345, 245 | Milky Way extinction law |
| Cullen, F. et al., 2025, (EXCELS) | Bayesian forward model for emission-line abundances |
| DESI Collaboration, 2026, arXiv:2601.02463 | Updated T_e–T_e relation: T_low = 0.648 × T_high + 3270 |
| Garnett, D. R. et al., 1995, ApJ, 443, 64 | Pure-UV C/O from C III] 1907,09 / O III] 1661,66 |
| Garnett, D. R., 1992, AJ, 103, 1330 | Classical T_e–T_e relations for intermediate/low zones |
| Garnett, D. R. et al., 1997, ApJ, 489, 63 | C/O ionisation correction factor: ICF_C = (O+ + O++)/O++ |
| Hsiao, T. Y.-Y. et al., 2026, arXiv:2608.20339 | Self-consistent O III T_e-n_e; high-density EMPG impostors |
| Erb, D. K. et al., 2010, ApJ, 719, 1168 | Rest-UV C/O in a lensed star-forming galaxy |
| Berg, D. A. et al., 2016, ApJ, 827, 126 | Rest-UV C/O in low-metallicity dwarfs |
| Izotov, Y. I. et al., 2006, A&A, 448, 955 | ICF equations 18–23 for N, Ne, S, Ar |
| Jones, T. et al., 2023 | C/O from direct ionic sum: (C2+ + C3+) / O2+ |
| Martinez, Z. et al., 2025, "Under Pressure", arXiv:2510.21960 | Default N/O ICFs, O32/N43 log(U) diagnostics, 3-tier T_e zones, and mid/high n_e redshift evolution (Eqs 4 & 5) |
| Martinez, Z. et al., 2026, in prep. | Default C²⁺/O²⁺ → C/O ICF (density-interpolated) |
| Noll, S. et al., 2009, A&A, 499, 69 | UV bump + slope modification to Calzetti curve |
| Osterbrock, D. E. & Ferland, G. J., 2006, Astrophysics of Gaseous Nebulae and Active Galactic Nuclei (University Science Books) | Case B recombination ratios |
| Salim, S. et al., 2018, ApJ, 859, 11 | UV slope and bump calibration for dust attenuation |
| Sanders, R. L. et al., 2025 | Strong-line metallicity calibrations (O3, O2, R23, O32) |
| Tayal, S. S. & Zatsarinny, O., 2017, ApJ, 850, 147 | 6-level O III collision strengths (required for lambda1666) |
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
| C+ | CII] 2324+2326 | Low | T_low | n_e,low |
| C2+ | CIII] 1907+1909 | High | T_high | n_e,high |
| C3+ | CIV 1548+1551 | High | T_high | n_e,high |
| S+ | [SII] 6718+6732 | Low | T_low | n_e,low |
| S2+ | [SIII] 9069 | Mid | (T_high+T_low)/2 | n_e,low |
| Ne2+ | [NeIII] 3869 | High | T_high | n_e,high |
| Ar2+ | [ArIII] 7136 | Mid | (T_high+T_low)/2 | n_e,low |
