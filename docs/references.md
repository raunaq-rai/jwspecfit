# References and methodology

Comprehensive documentation of all equations, algorithms, software
dependencies, and literature references used across the `jwspecfit`,
`jwspecmcmc`, and `jwspecabund` packages.

---

## Software dependencies

| Package | Version | Role | Reference |
|---------|---------|------|-----------|
| NumPy | >= 1.24 | Array operations | Harris et al. (2020) |
| SciPy | >= 1.10 | Optimisation, special functions | Virtanen et al. (2020) |
| Astropy | >= 5.0 | Units, FITS I/O | Astropy Collaboration (2022) |
| matplotlib | >= 3.6 | Static plotting | Hunter (2007) |
| plotly | >= 5.0 | Interactive plotting | Plotly Technologies Inc. |
| lmfit | >= 1.2 | Parameter management | Newville et al. (2014) |
| tqdm | >= 4.60 | Progress bars | da Costa-Luis (2019) |
| joblib | >= 1.2 | Parallel bootstrap | Joblib Development Team |
| emcee | >= 3.1 | Affine-invariant MCMC | Foreman-Mackey et al. (2013) |
| nautilus | >= 1.0 | Importance nested sampling | Lange (2023) |
| dynesty | >= 2.0 | Dynamic nested sampling | Speagle (2020); Koposov et al. (2022) |
| PyNEB | >= 1.1.25 | Atomic physics, emissivities | Luridiana et al. (2015) |
| corner | >= 2.2 | Corner plots | Foreman-Mackey (2016) |

---

## 1. Emission-line fitting (`jwspecfit`)

### 1.1 Line profiles

Emission lines are modelled as Gaussians bin-averaged over each
spectral pixel using the error function:

$$
\bar{G}_i = \frac{A}{2\,\Delta\lambda_i}
\left[
  \mathrm{erf}\!\left(\frac{\lambda_{i+1} - \mu}{\sigma\sqrt{2}}\right)
  - \mathrm{erf}\!\left(\frac{\lambda_i - \mu}{\sigma\sqrt{2}}\right)
\right]
$$

where $A$ is the integrated flux (erg s$^{-1}$ cm$^{-2}$), $\mu$ is
the line centroid (Å), $\sigma$ is the Gaussian width (Å), and
$\lambda_i$, $\lambda_{i+1}$ are the pixel edges.  This avoids
sampling bias when lines are narrower than pixels (prism regime,
R ~ 30).

### 1.2 Spectral resolution

The instrumental line spread function is assumed Gaussian with
$\sigma_\mathrm{inst} = \lambda / (R \times 2.3548)$.  The resolving
power $R$ is determined as follows:

| Source | $R(\lambda)$ | Reference |
|--------|-------------|-----------|
| PRISM/CLEAR | $R = 50 + 50(\lambda - 1) + 15(\lambda - 1)^2$ (µm), clipped to [30, 300] | Heuristic fit to NIRSpec instrument model |
| G140M, G235M, G395M | $R = 1000$ | NIRSpec documentation |
| G140H, G235H, G395H | $R = 2700$ | NIRSpec documentation |
| Pixel estimate | $R \approx \lambda / (2\Delta\lambda)$ | Fallback when grating unknown |

### 1.3 Continuum subtraction

Iterative sigma-clipped polynomial fit:

1. Mask pixels within $\pm 6\sigma_\mathrm{inst}$ of every expected
   emission line and blueward of NV 1239 Å (Lyman break region).
2. Fit weighted polynomial of degree $d$ (default $d = 2$) to
   unmasked pixels.
3. Compute residuals $r_i = f_i - C_i$.
4. Reject pixels with $r_i > k\,\hat{\sigma}$ (default $k = 2.5$)
   to iteratively clip positive outliers (emission).
5. Repeat steps 2–4 for 5 iterations.

### 1.4 Optimisation

The model is fit using `scipy.optimize.least_squares` with the
Trust Region Reflective (TRF) algorithm, Jacobian-based parameter
scaling (`x_scale='jac'`), and box constraints on all parameters.

### 1.5 Parameter constraints

| Constraint | Implementation | Reference |
|------------|---------------|-----------|
| [NII] doublet ratio | $A(6549) = A(6585) / 2.96$ | Storey & Zeippen (2000) |
| Balmer–OIII width tying | $\sigma_v(\mathrm{H}\alpha, \mathrm{H}\beta, \mathrm{H}\gamma, \mathrm{H}\delta, \mathrm{[NII]}) = \sigma_v(\mathrm{[OIII]}\,5007)$ | Assumed common NLR kinematics |
| Centroid tying | Same lines, velocity-space centroid tied to [OIII] | |
| Broad Balmer centroids | Tied to narrow counterpart | |

### 1.6 Broad Balmer component detection

BIC-based model selection compares four models:

| Model | Components |
|-------|-----------|
| narrow | Narrow lines only |
| broad1 | + intermediate broad ($\mathrm{FWHM} \sim 500$–$2000$ km s$^{-1}$) |
| broad2 | + very broad / BLR ($\mathrm{FWHM} \sim 2000$–$5000$ km s$^{-1}$) |
| both | + both broad components |

The Bayesian Information Criterion is:

$$
\mathrm{BIC} = k \ln n - 2 \ln \hat{\mathcal{L}}
$$

where $k$ is the number of free parameters, $n$ is the number of
data points, and $\hat{\mathcal{L}}$ is the maximised likelihood.
A more complex model is accepted if $\Delta\mathrm{BIC} > 6$
(default threshold).  Robust comparison uses bootstrapped BIC
distributions ($n_\mathrm{boot,BIC} = 100$).

### 1.7 Bootstrap uncertainties

1. Perturb flux: $f'_i = f_i + \sigma_i \times \epsilon_i$,
   $\epsilon_i \sim \mathcal{N}(0, 1)$.
2. Refit with `least_squares` using the best-fit parameters as
   initial guess.
3. Repeat $N_\mathrm{boot}$ times (default 1000).
4. Report $\hat{\sigma}(\mathrm{flux})$ from the bootstrap
   distribution for each line.

Parallelised via `joblib` (`n_jobs = -1` uses all cores).

### 1.8 Lyman-alpha modelling

Lyα is modelled as a skewed Gaussian (Owen 1956;
`scipy.stats.skewnorm`) attenuated by mean IGM transmission:

$$
F_\mathrm{Ly\alpha}(\lambda) = S(\lambda;\, A,\, \mu,\, \sigma,\, \alpha) \times T_\mathrm{IGM}(\lambda,\, z)
$$

where $S$ is the bin-averaged skewed Gaussian (positive skew =
red tail) and $T_\mathrm{IGM}$ is the mean IGM transmission.

**IGM transmission** follows Inoue et al. (2014), accounting for
Lyman-series line absorption (LAF) and damped Lyman-alpha system
(DLA) continuum absorption.  Coefficients are loaded from their
Table 2 (shipped as `data/inoue2014_table2.txt`).

### 1.9 Line database

Rest-frame vacuum wavelengths from NIST Atomic Spectra Database,
Morton (2003), and Storey & Zeippen (2000).  34 lines from Lyα
(1215.67 Å) through [SII] 6732 (6734.53 Å).

---

## 2. MCMC fitting (`jwspecmcmc`)

### 2.1 Likelihood

Gaussian log-likelihood identical to the least-squares objective:

$$
\ln \mathcal{L} = -\frac{1}{2} \sum_i
\left(\frac{f_i - C_i - M_i(\boldsymbol{\theta})}{\sigma_i}\right)^2
$$

where $f_i$ is the observed flux, $C_i$ the continuum, $M_i$ the
emission-line model, and $\sigma_i$ the flux uncertainty.

### 2.2 Priors

Default priors are uniform within the same bounds used by
`least_squares`.  Per-parameter overrides are supported:

| Prior type | PDF | Use case |
|-----------|-----|----------|
| `UniformPrior(lo, hi)` | $\pi(\theta) \propto 1$ for $\theta \in [\mathrm{lo}, \mathrm{hi}]$ | Default for all parameters |
| `GaussianPrior(mean, std, lo, hi)` | Truncated Gaussian | Informative priors |
| `LogUniformPrior(lo, hi)` | $\pi(\theta) \propto 1/\theta$ for $\theta \in [\mathrm{lo}, \mathrm{hi}]$ | Scale-invariant parameters |

### 2.3 Samplers

**emcee** (Foreman-Mackey et al. 2013): affine-invariant ensemble
sampler with stretch moves.  Walkers are initialised from a
tight ball around the least-squares MLE.

**nautilus** (Lange 2023): importance nested sampler using neural
network-based proposal distributions.  Uses `n_eff` target
effective samples.

### 2.4 Convergence diagnostics

| Diagnostic | Criterion | Reference |
|-----------|-----------|-----------|
| Gelman–Rubin $\hat{R}$ | $\hat{R} < 1.05$ for all parameters | Gelman & Rubin (1992) |
| Effective sample size (ESS) | ESS > 100 for all parameters | Via FFT autocorrelation |

### 2.5 Posterior summaries

Point estimates: medians.  Uncertainties: (16th, 84th) percentile
half-widths, reported as asymmetric $(- \delta_\mathrm{lo},\, + \delta_\mathrm{hi})$
credible intervals.

---

## 3. Chemical abundances (`jwspecabund`)

### 3.1 Overview of methods

Three independent methods for gas-phase metallicity:

| Method | When used | Lines required | Uncertainty propagation |
|--------|----------|---------------|----------------------|
| Direct $T_e$ | [OIII] 4363 detected (SNR $\geq 3$) | [OIII] 4363, 5007, 4959, Hβ, + Balmer pair for dust | MC perturbation or MCMC posterior propagation through PyNEB |
| Bayesian forward model | User-selected (`method="forward"`) | Any subset of optical CELs + Hβ | Full MCMC/nested sampling of ionic abundances, $T_e$, $n_e$ |
| Strong-line calibrations | No auroral line | [OIII] 5007, Hβ, optionally [OII], Hα | MC perturbation including calibration scatter |

### 3.2 Dust correction

Dust correction is applied to all emission-line fluxes before
abundance calculations (for the direct and strong-line methods).
$A_V$ is derived from the Balmer decrement when not supplied.

#### 3.2.1 $A_V$ from the Balmer decrement

$$
A_V = \frac{2.5 \log_{10}(R_\mathrm{obs} / R_\mathrm{int})}{f(\lambda_\mathrm{den}) - f(\lambda_\mathrm{num})}
$$

where $R_\mathrm{obs}$ is the observed Balmer ratio (Hα/Hβ or
Hγ/Hβ), $R_\mathrm{int}$ is the intrinsic Case B ratio, and
$f(\lambda) = A(\lambda) / A_V$ is the normalised attenuation curve.

**Intrinsic Balmer ratios** (Case B, $T_e = 10^4$ K,
$n_e = 100$ cm$^{-3}$; Osterbrock & Ferland 2006):

| Ratio | Value |
|-------|-------|
| Hα/Hβ | 2.86 |
| Hγ/Hβ | 0.468 |
| Hδ/Hβ | 0.259 |

**Note on temperature dependence:** The intrinsic Balmer ratios
are temperature-dependent.  At $T_e = 2.5 \times 10^4$ K the
Hα/Hβ ratio is $\sim 2.72$, not 2.86.  The current implementation
uses fixed ratios at $10^4$ K for the external dust correction.
See Section 3.5 (forward model) for self-consistent treatment where
$A_V$ is a free parameter and Balmer ratios are recomputed at each
model $T_e$.

#### 3.2.2 Salim+18 / Noll+09 attenuation (default)

Modified Calzetti et al. (2000) attenuation with a UV bump and
power-law slope deviation (Noll et al. 2009; Salim et al. 2018):

$$
A(\lambda) = E(B-V) \left[ k'_\mathrm{Calz}(\lambda) \left(\frac{\lambda}{0.55\,\mu\mathrm{m}}\right)^\delta + B \cdot D(\lambda) \cdot R_V \right]
$$

where:
- $k'_\mathrm{Calz}(\lambda)$ is the Calzetti et al. (2000) starburst
  curve,
- $D(\lambda)$ is a Drude profile at 2175 Å modelling the UV bump:
  $D(\lambda) = (\lambda \gamma)^2 / [(\lambda^2 - \lambda_0^2)^2 + (\lambda\gamma)^2]$,
- $\delta$ is the slope deviation (default $-0.35$),
- $B$ is the UV bump strength (default $2.27$),
- $R_V = A_V / E(B-V)$ (default $3.15$).

**Default parameters** ($R_V = 3.15$, $\delta = -0.35$, $B = 2.27$)
follow the median values for star-forming galaxies from
Salim et al. (2018).

**References:**
- Calzetti, D. et al. 2000, ApJ, 533, 682
- Noll, S. et al. 2009, A&A, 507, 1793
- Salim, S. et al. 2018, ApJ, 859, 11

#### 3.2.3 Cardelli+89 extinction

Milky Way extinction curve (Cardelli, Clayton & Mathis 1989):

$$
A(\lambda) = A_V \left[ a(x) + \frac{b(x)}{R_V} \right]
$$

where $x = 1/\lambda$ (µm$^{-1}$) and $a(x)$, $b(x)$ are
piecewise polynomials for the IR ($0.3 \leq x < 1.1$),
optical/NIR ($1.1 \leq x < 3.3$), and UV ($3.3 \leq x < 8.0$)
regimes.  Default $R_V = 3.1$.

**Reference:** Cardelli, J. A., Clayton, G. C. & Mathis, J. S.
1989, ApJ, 345, 245

#### 3.2.4 Flux correction

Observed fluxes are corrected as:

$$
F_\mathrm{corr} = F_\mathrm{obs} \times 10^{0.4\,A(\lambda)}
$$

Errors are scaled by the same factor.  When MCMC posteriors are
available, each posterior sample is corrected individually.

---

### 3.3 Direct $T_e$ method

The direct method uses the temperature-sensitive auroral-to-nebular
line ratio to determine electron temperature, breaking the
degeneracy between temperature and ionic abundance inherent in
strong-line methods.  This is the gold standard for gas-phase
metallicity when the auroral line is detected.

#### 3.3.1 Two-zone ionisation model

A two-zone model is assumed:

| Zone | Traced by | Temperature | Lines |
|------|----------|-------------|-------|
| High ionisation | O<sup>2+</sup>, Ne<sup>2+</sup> | $T_e(\text{high}) = T_e(\text{[OIII]})$ | [OIII] 4363, 4959, 5007; [NeIII] 3869 |
| Low ionisation | O<sup>+</sup>, N<sup>+</sup>, S<sup>+</sup> | $T_e(\text{low}) = T_e(\text{[OII]})$ | [OII] 3726, 3729; [NII] 6585; [SII] 6718, 6732 |

Intermediate ions (S<sup>2+</sup>, Ar<sup>2+</sup>) use
$T_\text{mid} = (T_\text{high} + T_\text{low}) / 2$.

The contribution of O<sup>3+</sup>/H<sup>+</sup> to the total oxygen abundance
is assumed negligible, following Berg et al. (2021).

#### 3.3.2 Electron temperature: $T_e$([OIII])

Determined from the [OIII] auroral-to-nebular ratio using PyNEB:

$$
R_{\mathrm{[OIII]}} = \frac{I(4363)}{I(5007) + I(4959)}
$$

PyNEB solves for $T_e$ numerically using atomic data for the O<sup>2+</sup>
ion.  We use `getTemDen()` with `to_eval="(L(4363))/(L(5007)+L(4959))"`,
`log=True`, `start_x=3.0`, `end_x=5.0` for robust root-finding across
the range $10^3$–$10^5$ K (essential for metal-poor galaxies at
$z > 4$ where $T_e > 25\,000$ K is common).

**Underlying physics:** For a collisionally excited line (CEL) the
emissivity depends on $T_e$ through the Boltzmann factor.  For a
two-level simplification (Osterbrock & Ferland 2006, Chapter 5), the
emissivity of a transition from upper level $j$ to lower level $i$ is:

$$
\varepsilon_{ji} = n_j A_{ji} h\nu_{ji}
$$

where $A_{ji}$ is the spontaneous transition probability and $n_j$
is the population of level $j$, determined by solving the statistical
equilibrium equations.  In the low-density limit
($n_e \ll n_\text{crit}$), every collisional excitation is followed
by radiative de-excitation, and the emissivity ratio of the auroral
($^1S_0 \to\, ^1D_2$, 4363 Å) to nebular
($^1D_2 \to\, ^3P_{1,2}$, 4959+5007 Å) transitions depends
primarily on $T_e$:

$$
R_\text{[OIII]} = \frac{j_{4363}}{j_{4959} + j_{5007}}
= \frac{A_{4363}}{A_{4959} + A_{5007}}
\frac{\Omega(^1D_2,\, ^1S_0)}{\Omega(^3P,\, ^1D_2)}
\exp\!\left(-\frac{\Delta E_{43}}{k_B T_e}\right)
$$

where $\Omega$ are the collision strengths, $\Delta E_{43} = E(^1S_0) - E(^1D_2) = 3.29$ eV
is the energy gap between the auroral and nebular upper levels, and
$k_B$ is the Boltzmann constant.  This gives the strong exponential
sensitivity to $T_e$.

PyNEB solves the full multi-level (typically 5-level) statistical
equilibrium problem numerically, accounting for all collisional and
radiative transitions and the actual density dependence.  The
5-level model for O<sup>2+</sup> includes the ground term
$^3P_{0,1,2}$ and excited terms $^1D_2$ and $^1S_0$.  The
statistical equilibrium equations are:

$$
\sum_{j \neq i} n_j \left[ A_{ji} + n_e q_{ji}(T_e) \right]
= n_i \sum_{j \neq i} \left[ A_{ij} + n_e q_{ij}(T_e) \right]
$$

for each level $i$, where $q_{ij}$ is the collisional rate
coefficient:

$$
q_{ij}(T_e) = \frac{8.629 \times 10^{-6}}{g_i\, T_e^{1/2}}\, \Omega_{ij}(T_e) \quad \text{cm}^3\,\text{s}^{-1}
$$

with $g_i$ the statistical weight of level $i$ and
$\Omega_{ij}(T_e)$ the thermally averaged (Maxwellian) collision
strength.  PyNEB interpolates tabulated $\Omega_{ij}(T_e)$ from
the atomic data files.

#### 3.3.3 Electron temperature: $T_e$([NII])

When [NII] 5756 is detected, $T_e$(N<sup>+</sup>) is measured directly from
the [NII] 5756/6585 ratio via PyNEB, providing an independent
low-ionisation zone temperature:

$$
R_\text{[NII]} = \frac{I(5756)}{I(6549) + I(6585)}
$$

The same 5-level statistical equilibrium approach applies, with
the N<sup>+</sup> energy gap $\Delta E = E(^1S_0) - E(^1D_2) = 4.05$ eV
(the [NII] auroral line lies at shorter wavelength than [OIII] 4363
because the energy gap is larger).  This ratio is less commonly
detected because [NII] 5756 is intrinsically fainter than [OIII] 4363.

#### 3.3.4 $T_e$–$T_e$ relations

When $T_e$([OII]) cannot be measured directly (the usual case),
the low-ionisation zone temperature is derived from the high-ionisation
zone temperature via empirical relations.

**Available relations:**

| Name | Equation | Reference | Notes |
|------|---------|-----------|-------|
| `"desi"` | $T_e(\mathrm{low}) = 0.648\,T_e(\mathrm{high}) + 3270$ | DESI DR2 (arXiv:2601.02463) | Calibrated on the DESI Early Data Release galaxy sample |
| `"classical"` | $T_e(\mathrm{low}) = 0.7\,T_e(\mathrm{high}) + 3000$ | Garnett (1992) | Widely used; original form from Campbell, Terlevich & Melnick (1986) |

The $T_e$–$T_e$ relation is a significant source of systematic
uncertainty.  Other relations in the literature include:

| Equation | Reference |
|---------|-----------|
| $T_2 = 0.7\,T_3 + 1500$ | Campbell et al. (1986) / Garnett (1992), corrected by Andrews & Martini (2013) |
| Various parametric forms | Izotov et al. (2006); Pilyugin et al. (2009); Nicholls et al. (2014) |
| $T_2 = 1.0807\,T_3 - 0.0846\,T_3^2$ ($T$ in $10^4$ K) | Pagel et al. (1992) |

The $\sim 1000$–$2000$ K discrepancy between theoretically and
empirically calibrated relations (Andrews & Martini 2013; Curti
et al. 2017) arises partly from the difference between individual
H II regions and galaxy-integrated spectra, where flux originates
from a mixture of H II regions with varying properties plus diffuse
ionised gas.

#### 3.3.5 Electron density

$n_e$ is derived from density-sensitive doublet ratios via PyNEB:

| Doublet | Lines | Diagnostic ratio |
|---------|-------|-----------------|
| [SII] | 6718, 6732 | $I(6718)/I(6732)$ |
| [OII] | 3726, 3729 | $I(3726)/I(3729)$ |

[SII] is preferred when available; [OII] is used as a fallback.
If neither doublet is detected, $n_e = 100$ cm$^{-3}$ is assumed
(abundance diagnostics are insensitive to $n_e$ at
$n_e \lesssim 10^4$ cm$^{-3}$; e.g. Isobe et al. 2023).

**Physical basis:** The [SII] and [OII] doublets arise from
transitions between levels within the ground-term fine-structure
splitting ($^4S_{3/2}$, $^2D_{3/2,5/2}$ for S<sup>+</sup>; analogous
for O<sup>+</sup>).  The two lines in each doublet have different
critical densities, so their ratio depends on $n_e$:

$$
\frac{I(\lambda_1)}{I(\lambda_2)}
= \frac{n_{j_1}\,A_{j_1 \to i_1}\,\nu_1}{n_{j_2}\,A_{j_2 \to i_2}\,\nu_2}
$$

At low density ($n_e \ll n_\text{crit}$) the ratio approaches the
ratio of collision strengths; at high density
($n_e \gg n_\text{crit}$) it approaches the ratio of statistical
weights.  The transition occurs around
$n_\text{crit} \sim 10^3$–$10^4$ cm$^{-3}$, making the doublet ratio
a sensitive density probe in this regime.  PyNEB uses the full
multi-level solution to extract $n_e$ from the observed ratio at the
adopted $T_e$.

#### 3.3.6 Ionic abundances

Ionic abundances $X^{i+}/\mathrm{H}^+$ are computed via PyNEB's
`getIonAbundance()` method for $T_e \leq 30\,000$ K.  For
$T_e > 30\,000$ K (beyond the Storey & Hummer 1995 H I
recombination tables bundled with PyNEB), we compute abundances
manually:

$$
\frac{X^{i+}}{\mathrm{H}^+} = \frac{I_\mathrm{line}}{I_{\mathrm{H}\beta}} \times \frac{\varepsilon_{\mathrm{H}\beta}(T_e, n_e)}{\varepsilon_\mathrm{line}(T_e, n_e)}
$$

where $\varepsilon_\mathrm{line}$ is obtained from PyNEB's CEL
emissivity tables (valid at any temperature) and
$\varepsilon_{\mathrm{H}\beta}$ uses the Aller (1984) formula
(Section 3.5.3) which has no upper temperature limit.

The CEL emissivity is computed from the level populations obtained
by solving the statistical equilibrium equations (Section 3.3.2):

$$
\varepsilon_\mathrm{line}(T_e, n_e)
= n_{X^{i+}} \, n_e \, q_{ij}(T_e) \, \frac{A_{ji}}{A_{ji} + n_e\,q_{ji}(T_e)} \, h\nu_{ji}
$$

In the low-density limit this simplifies to
$\varepsilon \propto n_{X^{i+}} \, n_e \, q_{ij}(T_e) \, h\nu_{ji}$,
and the ionic abundance reduces to:

$$
\frac{X^{i+}}{\mathrm{H}^+}
= \frac{I_\mathrm{line}}{I_{\mathrm{H}\beta}}
\times \frac{\alpha_{\mathrm{H}\beta}^\mathrm{eff}(T_e)}{q_{ij}(T_e)\,h\nu_{ji} / (h\nu_{\mathrm{H}\beta})}
$$

where $\alpha_{\mathrm{H}\beta}^\mathrm{eff}$ is the effective
recombination coefficient for H$\beta$ (see Section 3.5.3).
PyNEB handles the full density-dependent multi-level solution
internally.

**Ions computed:**

| Ion | Line(s) | Zone | PyNEB atom |
|-----|---------|------|-----------|
| O<sup>2+</sup>/H<sup>+</sup> | [OIII] 5007 | High | `Atom("O", 3)` |
| O<sup>+</sup>/H<sup>+</sup> | [OII] 3726+3729 | Low | `Atom("O", 2)` |
| N<sup>+</sup>/H<sup>+</sup> | [NII] 6585 | Low | `Atom("N", 2)` |
| S<sup>+</sup>/H<sup>+</sup> | [SII] 6718+6732 | Low | `Atom("S", 2)` |
| S<sup>2+</sup>/H<sup>+</sup> | [SIII] 9069 | Mid | `Atom("S", 3)` |
| Ne<sup>2+</sup>/H<sup>+</sup> | [NeIII] 3869 | High | `Atom("Ne", 3)` |
| Ar<sup>2+</sup>/H<sup>+</sup> | [ArIII] 7136 | Mid | `Atom("Ar", 3)` |

#### 3.3.7 Atomic data

PyNEB (Luridiana et al. 2015) is used for all atomic physics
calculations.  The default atomic and collisional data shipped with
PyNEB $\geq 1.1.25$ include:

- **Transition probabilities:** Froese Fischer & Tachiev (2004)
  for most ions; Storey & Zeippen (2000) for [NII], [OIII].
- **Collision strengths:**
  - O<sup>+</sup>: Kisielius et al. (2009)
  - O<sup>2+</sup>: Aggarwal & Keenan (1999) / Storey, Sochi & Badnell (2014)
  - N<sup>+</sup>: Tayal (2011)
  - S<sup>+</sup>: Tayal & Zatsarinny (2010)
- **H I recombination:** Storey & Hummer (1995), valid to 30 000 K.

Users requiring specific atomic data versions (e.g. to match a
particular paper) should configure PyNEB's data files directly via
`pn.atomicData.setDataFile()`.

#### 3.3.8 Total abundances and ICFs

**Oxygen:**

$$
\frac{\mathrm{O}}{\mathrm{H}} = \frac{\mathrm{O}^+}{\mathrm{H}^+} + \frac{\mathrm{O}^{2+}}{\mathrm{H}^+}
$$

No ICF is applied for higher ionisation states; the O<sup>3+</sup>
contribution is negligible for typical H II regions (Berg et al. 2021).

**Nitrogen:**

$$
\frac{\mathrm{N}}{\mathrm{O}} = \mathrm{ICF}_\mathrm{N} \times \frac{\mathrm{N}^+/\mathrm{H}^+}{\mathrm{O}^+/\mathrm{H}^+}
$$

where $\mathrm{ICF}_\mathrm{N}$ is from Izotov et al. (2006, eq. 18):

$$
\mathrm{ICF}_\mathrm{N} = -0.825\,x^2 + 0.718\,x + 0.853, \qquad x = \mathrm{O}^+ / \mathrm{O}_\mathrm{total}
$$

clamped to $\mathrm{ICF}_\mathrm{N} \geq 1$.  For many applications
the standard assumption $\mathrm{N/O} \approx \mathrm{N}^+/\mathrm{O}^+$
(ICF = 1) is used, justified by the similar ionisation potentials of
O<sup>+</sup> and N<sup>+</sup> (e.g. Nava et al. 2006; Amayo et al. 2021).

**Sulfur:**

$$
\mathrm{S/O} = \mathrm{ICF}_\mathrm{S} \times (\mathrm{S}^+ + \mathrm{S}^{2+}) / \mathrm{O}
$$

ICF from Izotov et al. (2006, eq. 20), accounting for S<sup>3+</sup>:

$$
\mathrm{ICF}_\mathrm{S} = 0.013 + 5.986\,x - 21.085\,x^2 + 26.462\,x^3 - 11.282\,x^4
$$

clamped to $\mathrm{ICF}_\mathrm{S} \geq 1$.

**Neon:**

$$
\mathrm{Ne/O} = \mathrm{ICF}_\mathrm{Ne} \times \mathrm{Ne}^{2+}/\mathrm{O}^{2+}
$$

ICF from Izotov et al. (2006, eq. 19):

$$
\mathrm{ICF}_\mathrm{Ne} = -0.385\,x^2 + 0.405\,x + 0.335
$$

**Argon:**

$$
\mathrm{Ar/O} = \mathrm{ICF}_\mathrm{Ar} \times \mathrm{Ar}^{2+}/\mathrm{O}^{2+}
$$

ICF from Izotov et al. (2006, eqs. 22/23):

$$
\mathrm{ICF}_\mathrm{Ar} = 0.157 + 3.119\,x - 6.185\,x^2 + 4.517\,x^3
$$

clamped to $\mathrm{ICF}_\mathrm{Ar} \geq 1$.

In all ICF expressions, $x = \mathrm{O}^+ / \mathrm{O}_\mathrm{total}$ is
the fractional ionic abundance of O<sup>+</sup>.

**Reporting convention:**

$$
12 + \log(\mathrm{O/H}) = 12 + \log_{10}\!\left(\frac{\mathrm{O}^+/\mathrm{H}^+ + \mathrm{O}^{2+}/\mathrm{H}^+}{1}\right)
$$

$$
\log(\mathrm{N/O}) = \log_{10}\!\left(\mathrm{ICF}_\mathrm{N} \times \frac{\mathrm{N}^+/\mathrm{H}^+}{\mathrm{O}^+/\mathrm{H}^+}\right)
$$

The solar oxygen abundance is $12 + \log(\mathrm{O/H})_\odot = 8.69$
(Asplund et al. 2009).

**Reference:** Izotov, Y. I., Stasinska, G., Meynet, G., Guseva, N. G.
& Thuan, T. X. 2006, A&A, 448, 955

#### 3.3.9 Uncertainty propagation

**Bootstrap / least-squares results:** Gaussian Monte Carlo
perturbation ($N_\mathrm{MC} = 1000$ by default):

1. For each iteration, perturb every line flux:
   $F'_\mathrm{line} = \max(F_\mathrm{line} + \sigma_\mathrm{line} \times \epsilon,\; 10^{-50})$,
   $\epsilon \sim \mathcal{N}(0, 1)$.
2. Recompute $T_e$, ionic abundances, totals through the full PyNEB
   pipeline.
3. Report $\hat{\sigma}$ of the resulting 12+log(O/H) and
   log(N/O) distributions.

**MCMC results:** Each posterior sample's line fluxes are propagated
through the full PyNEB pipeline.  Posteriors are thinned to
$N_\mathrm{posterior}$ samples (default 1000) via random subsampling
to keep runtime manageable ($\sim 7$ ms per PyNEB evaluation).

---

### 3.4 Strong-line calibrations (Sanders+25)

Used when the [OIII] 4363 auroral line is not detected.
Strong-line methods use empirical calibrations between bright
emission-line ratios and metallicity, calibrated on samples of
galaxies with direct $T_e$ measurements.

#### 3.4.1 Diagnostic ratios

$$
\mathrm{O3} = \log_{10}\!\left(\frac{[\mathrm{OIII}]\,5007}{\mathrm{H}\beta}\right), \qquad
\mathrm{O2} = \log_{10}\!\left(\frac{[\mathrm{OII}]}{\mathrm{H}\beta}\right)
$$

$$
\mathrm{R23} = \log_{10}\!\left(\frac{[\mathrm{OIII}]\,5007 + [\mathrm{OII}]}{\mathrm{H}\beta}\right), \qquad
\mathrm{O32} = \log_{10}\!\left(\frac{[\mathrm{OIII}]\,5007}{[\mathrm{OII}]}\right)
$$

[OII] denotes the total [OII] $\lambda\lambda$3726,3729 flux
(resolved or unresolved).

#### 3.4.2 Calibration polynomials

Each diagnostic ratio $R$ is modelled as a polynomial in
$Z = 12 + \log(\mathrm{O/H})$:

$$
R_\mathrm{model}(Z) = \sum_{k=0}^{n} c_k \,(Z - Z_\mathrm{ref})^k, \qquad Z_\mathrm{ref} = 8.0
$$

**Coefficients (Sanders et al. 2025, Table 3):**

| Ratio | $c_0$ | $c_1$ | $c_2$ | $c_3$ | $c_4$ | $Z_\mathrm{min}$ | $Z_\mathrm{max}$ |
|-------|-------|-------|-------|-------|-------|-----------------|-----------------|
| O3 | 0.852 | $-0.162$ | $-1.149$ | $-0.553$ | — | 7.3 | 8.6 |
| O2 | 0.172 | 0.954 | $-0.832$ | — | — | 7.3 | 8.6 |
| R23 | 0.998 | 0.053 | $-0.141$ | $-0.493$ | $-0.774$ | 7.3 | 8.6 |
| O32 | 0.697 | $-1.245$ | $-0.869$ | — | — | 7.3 | 8.6 |

#### 3.4.3 Simultaneous chi-squared fit

The best-fit metallicity $Z_\mathrm{best}$ is found by minimising
the simultaneous chi-squared across all available diagnostics:

$$
\chi^2(Z) = \sum_{m \in \mathrm{diag}} \frac{(R_\mathrm{obs}^{(m)} - R_\mathrm{model}^{(m)}(Z))^2}{\sigma_{\mathrm{obs},m}^2 + \sigma_{\mathrm{cal},m}^2}
$$

where $\sigma_\mathrm{obs}$ is the measurement error on $\log_{10}$
of the ratio (Gaussian error propagation) and $\sigma_\mathrm{cal}$
is the total calibration scatter in ratio space:

$$
\sigma_\mathrm{cal} = \sqrt{\sigma_{R,\mathrm{fit}}^2 + \sigma_{R,\mathrm{int}}^2}
$$

Minimisation uses `scipy.optimize.minimize_scalar` with method
`"bounded"` on $Z \in [6.5, 9.0]$.

#### 3.4.4 Three scatter components

Sanders+25 report three distinct scatter values for each diagnostic:

| Symbol | Meaning | Role in code |
|--------|---------|-------------|
| $\sigma_{R,\mathrm{fit}}$ | RMS residual of the polynomial fit to the calibration sample | Combined into $\sigma_\mathrm{cal}$ |
| $\sigma_{R,\mathrm{int}}$ | Intrinsic scatter in the ratio at fixed $Z$ (physical scatter from galaxy-to-galaxy variations) | Combined into $\sigma_\mathrm{cal}$ |
| $\sigma_{\mathrm{O/H,int}}$ | Intrinsic scatter in $12+\log(\mathrm{O/H})$ at fixed ratio (the actual metallicity uncertainty floor per diagnostic) | Stored for reference |

**Values:**

| Ratio | $\sigma_{R,\mathrm{fit}}$ | $\sigma_{R,\mathrm{int}}$ | $\sigma_\mathrm{cal}$ | $\sigma_{\mathrm{O/H,int}}$ |
|-------|------|------|------|------|
| O3 | 0.04 | 0.13 | 0.136 | 0.14 |
| O2 | 0.03 | 0.25 | 0.252 | 0.22 |
| R23 | 0.03 | 0.07 | 0.076 | 0.13 |
| O32 | 0.09 | 0.29 | 0.304 | 0.25 |

#### 3.4.5 Uncertainty propagation

**Bootstrap / least-squares results:** Monte Carlo perturbation
($N_\mathrm{MC} = 1000$ by default).  Each iteration:

1. Perturb each observed log-ratio by
   $\mathcal{N}(0, \sqrt{\sigma_\mathrm{obs}^2 + \sigma_\mathrm{cal}^2})$
   — both measurement and calibration scatter are included.
2. Re-minimise the chi-squared to find a perturbed $Z$.
3. Report (16th, 84th) percentiles of the $Z$ distribution.

**MCMC results:** Each posterior sample provides a set of fluxes.
For each sample:

1. Compute line ratios from the posterior fluxes.
2. Perturb each ratio by $\mathcal{N}(0, \sigma_\mathrm{cal})$ to
   propagate calibration scatter.
3. Minimise to find $Z$ for that sample.

The MCMC posterior spread captures measurement uncertainty, and
the $\sigma_\mathrm{cal}$ perturbation adds the calibration scatter
on top.  Posteriors are thinned to $N_\mathrm{posterior}$ samples
(default 1000).

**Reference:** Sanders, R. L. et al. 2025, ApJ (submitted)

---

### 3.5 Bayesian forward model (Cullen+25 approach)

The forward model simultaneously constrains ionic abundances,
electron temperature, and electron density by predicting emission-line
flux ratios and comparing to observations via MCMC or nested sampling.

#### 3.5.1 Free parameters and priors

| Parameter | Symbol | Range | Prior | Unit |
|-----------|--------|-------|-------|------|
| Electron temperature | log<sub>10</sub> T<sub>e</sub> | [3.5, 5.0] | Uniform | K |
| Electron density | log<sub>10</sub> n<sub>e</sub> | [0.0, 5.0] | Uniform | cm<sup>-3</sup> |
| Per-ion abundance | log<sub>10</sub>(X<sup>i+</sup>/H<sup>+</sup>) | [-7, -2] (oxygen); [-8, -3] (N, Ne); [-9, -4] (S, Ar) | Uniform in log (= log-uniform in linear) | — |

Only ions with at least one detected emission line are included as
free parameters.  Typically 4–8 free parameters depending on the
available lines.

#### 3.5.2 Model prediction

For each parameter vector $\boldsymbol{\theta}$, the predicted
line/Hβ flux ratio is:

$$
\frac{F_\mathrm{line}}{F_{\mathrm{H}\beta}} = \frac{X^{i+}}{\mathrm{H}^+} \times \frac{\varepsilon_\mathrm{line}(T_e, n_e)}{\varepsilon_{\mathrm{H}\beta}(T_e, n_e)}
$$

where $\varepsilon_\mathrm{line}$ is the collisionally excited line
(CEL) emissivity from PyNEB's `Atom.getEmissivity()`, and
$\varepsilon_{\mathrm{H}\beta}$ is the Hβ recombination emissivity.

For lines consisting of multiple transitions (e.g. [OII] doublet),
the emissivities are summed.

#### 3.5.3 Hβ emissivity

For $T_e \leq 30\,000$ K, the Storey & Hummer (1995) H I
recombination tables in PyNEB are used via
`RecAtom("H", 1).getEmissivity(Te, ne, wave=4861)`.

For $T_e > 30\,000$ K, the Aller (1984) Case B formula is used:

$$
\alpha_{\mathrm{H}\beta}^{\mathrm{eff}} = 3.03 \times 10^{-14} \left(\frac{T_e}{10^4\,\mathrm{K}}\right)^{-0.874} \;\mathrm{cm}^3\,\mathrm{s}^{-1}
$$

$$
\varepsilon_{\mathrm{H}\beta} = \alpha_{\mathrm{H}\beta}^{\mathrm{eff}} \times h\nu_{\mathrm{H}\beta}
$$

This formula has no upper temperature limit, enabling abundance
calculations for the extremely metal-poor, high-$T_e$ galaxies
found at $z > 4$ with JWST.

**References:**
- Aller, L. H. 1984, *Physics of Thermal Gaseous Nebulae* (Reidel)
- Storey, P. J. & Hummer, D. G. 1995, MNRAS, 272, 41

#### 3.5.4 Likelihood

Gaussian log-likelihood on line/Hβ flux ratios:

$$
\ln \mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{2} \sum_{\mathrm{lines}\, l}
\frac{(R_{\mathrm{obs},l} - R_{\mathrm{model},l}(\boldsymbol{\theta}))^2}{\sigma_{R,l}^2}
$$

where $R_{\mathrm{obs},l} = F_l / F_{\mathrm{H}\beta}$ and
$\sigma_{R,l}$ is the error on the ratio propagated from the flux
uncertainties.

#### 3.5.5 Sampling

| Sampler | Algorithm | Key settings | Reference |
|---------|----------|-------------|-----------|
| emcee | Affine-invariant ensemble MCMC | 32 walkers, 5000 steps, 1000 burn-in | Foreman-Mackey et al. (2013) |
| dynesty | Dynamic nested sampling | 500 live points | Speagle (2020); Koposov et al. (2022) |

Walkers are initialised around a MAP estimate found by
`scipy.optimize.differential_evolution`.

#### 3.5.6 Total abundances from posteriors

$$
12 + \log(\mathrm{O/H}) = 12 + \log_{10}\!\left(10^{\log(\mathrm{O}^+/\mathrm{H}^+)} + 10^{\log(\mathrm{O}^{2+}/\mathrm{H}^+)}\right)
$$

computed sample-by-sample from the posterior chains.

N/O is derived as $\log(\mathrm{N}^+/\mathrm{H}^+) - \log(\mathrm{O}^+/\mathrm{H}^+)$
(= log(N<sup>+</sup>/O<sup>+</sup>), the ICF = 1 assumption).

#### 3.5.7 Current limitations and differences from Cullen+25

The current implementation differs from the full Cullen et al. (2025)
/ Scholte et al. (2025) EXCELS methodology in several ways:

1. **$A_V$ is not a free parameter.** Dust correction is applied
   externally before the forward model, using fixed Case B Balmer
   ratios at $10^4$ K.  Cullen+25 treat $A_V$ as a 6th free
   parameter and recompute temperature-dependent Balmer ratios at
   each likelihood evaluation, providing self-consistent dust
   correction.

2. **Single $T_e$ zone in the forward model.** The forward model
   uses a single `log_Te` parameter for all CEL emissivities.
   Cullen+25 apply a $T_e$–$T_e$ relation to derive $T_e$([OII])
   from $T_e$([OIII]) for the low-ionisation zone.  In practice
   this difference is absorbed into the fitted ionic abundances when
   both O<sup>+</sup> and O<sup>2+</sup> are free parameters.

3. **Prior bounds differ** (see Section 3.5.1).  Cullen+25 use
   $\log T_e \in [3.0, 4.52]$ and $\log n_e \in [1.0, 3.0]$.

---

### 3.6 Broad Balmer component handling

When a `BroadFitResult` or `MCMCBroadFitResult` is passed to
`compute_abundances()`, broad Balmer component fluxes (Hα, Hβ,
Hγ, Hδ) are automatically **summed** with the narrow component
fluxes.  This ensures the total hydrogen flux is used for the Balmer
decrement and abundance normalisation ($I_\mathrm{line}/I_{\mathrm{H}\beta}$).

For MCMC results, broad and narrow posteriors are summed
sample-by-sample to preserve correlations.

---

## 4. Full bibliography

### Atomic physics and emissivities

- Aggarwal, K. M. & Keenan, F. P. 1999, ApJS, 123, 311 — O<sup>2+</sup> collision strengths
- Aller, L. H. 1984, *Physics of Thermal Gaseous Nebulae* (Reidel) — Hβ emissivity formula
- Froese Fischer, C. & Tachiev, G. 2004, ADNDT, 87, 1 — Transition probabilities
- Kisielius, R. et al. 2009, ApJ, 694, 1209 — O<sup>+</sup> collision strengths
- Luridiana, V., Morisset, C. & Shaw, R. A. 2015, A&A, 573, A42 — PyNEB
- Storey, P. J. & Hummer, D. G. 1995, MNRAS, 272, 41 — H I recombination
- Storey, P. J., Sochi, T. & Badnell, N. R. 2014, MNRAS, 441, 3028 — O<sup>2+</sup> collision strengths
- Storey, P. J. & Zeippen, C. J. 2000, MNRAS, 312, 813 — [NII], [OIII] transition probabilities
- Tayal, S. S. 2011, ApJS, 195, 12 — N<sup>+</sup> collision strengths
- Tayal, S. S. & Zatsarinny, O. 2010, ApJS, 188, 32 — S<sup>+</sup> collision strengths

### Dust attenuation and extinction

- Calzetti, D. et al. 2000, ApJ, 533, 682 — Starburst attenuation law
- Cardelli, J. A., Clayton, G. C. & Mathis, J. S. 1989, ApJ, 345, 245 — MW extinction
- Noll, S. et al. 2009, A&A, 507, 1793 — UV bump + slope modification
- Osterbrock, D. E. & Ferland, G. J. 2006, *Astrophysics of Gaseous Nebulae and Active Galactic Nuclei*, 2nd ed. — Case B Balmer ratios
- Salim, S. et al. 2018, ApJ, 859, 11 — Modified Calzetti parameters

### Electron temperature and Te-Te relations

- Andrews, B. H. & Martini, P. 2013, ApJ, 765, 140 — Te-Te correction
- Campbell, A., Terlevich, R. & Melnick, J. 1986, MNRAS, 223, 811 — Original Te-Te relation
- Curti, M. et al. 2017, MNRAS, 465, 1384 — Te-Te discrepancy
- DESI Collaboration 2025, arXiv:2601.02463 — DESI DR2 Te-Te relation
- Garnett, D. R. 1992, AJ, 103, 1330 — Classical Te-Te relation
- Nicholls, D. C. et al. 2014, ApJ, 786, 155 — Te-Te relations
- Pagel, B. E. J. et al. 1992, MNRAS, 255, 325 — Te-Te relation
- Pilyugin, L. S. et al. 2009, A&A, 508, 1043 — Te-Te relation
- Pilyugin, L. S. et al. 2010, ApJ, 720, 1738 — Te-Te discrepancy

### Electron density

- Isobe, Y. et al. 2023, ApJ, 956, 139 — Density insensitivity at low $n_e$

### Ionic abundances and ICFs

- Amayo, A. et al. 2021, MNRAS, 505, 2361 — N/O = N+/O+ justification
- Asplund, M. et al. 2009, ARA&A, 47, 481 — Solar abundances
- Berg, D. A. et al. 2021, ApJ, 922, 170 — O<sup>3+</sup> contribution negligible
- Izotov, Y. I. et al. 2006, A&A, 448, 955 — ICFs (N, Ne, S, Ar)
- Nava, A. et al. 2006, ApJ, 645, 1076 — N/O = N+/O+ justification

### Strong-line calibrations

- Sanders, R. L. et al. 2025, ApJ (submitted) — Simultaneous O3/O2/R23/O32 calibrations

### Forward modelling

- Cullen, F. et al. 2025, MNRAS (EXCELS) — Forward modelling approach
- Scholte, D. et al. 2025, MNRAS (EXCELS) — Methodology section 4.3

### MCMC and sampling

- Foreman-Mackey, D. et al. 2013, PASP, 125, 306 — emcee
- Foreman-Mackey, D. 2016, JOSS, 1, 24 — corner.py
- Gelman, A. & Rubin, D. B. 1992, Statistical Science, 7, 457 — R-hat
- Koposov, S. et al. 2022, doi:10.5281/zenodo.7388523 — dynesty
- Lange, J. U. 2023, MNRAS, 525, 3181 — nautilus
- Speagle, J. S. 2020, MNRAS, 493, 3132 — dynesty

### IGM and Lyman-alpha

- Inoue, A. K. et al. 2014, MNRAS, 442, 1805 — IGM transmission model

### Line database

- Morton, D. C. 2003, ApJS, 149, 205 — UV line wavelengths
- NIST Atomic Spectra Database — Rest-frame vacuum wavelengths

### General

- Harris, C. R. et al. 2020, Nature, 585, 357 — NumPy
- Hunter, J. D. 2007, Computing in Science & Engineering, 9, 90 — matplotlib
- Virtanen, P. et al. 2020, Nature Methods, 17, 261 — SciPy
- Astropy Collaboration 2022, ApJ, 935, 167 — Astropy
