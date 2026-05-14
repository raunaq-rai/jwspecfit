# jwspecfit.fit_redshift — design spec

**Date:** 2026-05-14
**Status:** approved by user, ready for implementation plan

## 1. Purpose

Provide a robust, prior-aware redshift fitter for JWST NIRSpec spectra that:

- Searches z = 0 → 20 exhaustively, with no early stopping.
- Scores each trial z by jointly fitting the amplitudes of a curated strong-line set plus a low-order continuum, using non-negative least squares.
- Handles missing wavelength coverage gracefully: lines whose expected position falls outside the spectrum are simply skipped at that z.
- Accepts a non-Gaussian prior (uniform default, single window, list of disjoint windows, or arbitrary callable).
- Returns both a per-peak credible interval and a ranked list of secondary peaks, so the user can tell whether the best z is decisive or only narrowly preferred.

This is the redshift-finder companion to `fit_lines` (which requires z up front) and `fit_NHI` (which assumes z is known). It is the entry point when z is unknown or only weakly constrained from photometry.

## 2. Public API

```python
def fit_redshift(
    spec: Spectrum,
    *,
    lines: list[str] | None = None,            # default = curated strong-line set
    z_min: float = 0.0,
    z_max: float = 20.0,
    dz_coarse: float | None = None,            # default tied to spec.grating R
    prior: tuple | list[tuple] | Callable | None = None,
    n_peaks: int = 5,
    refine_dchi2: float = 9.0,
    continuum_order: int = 3,
    nonneg: bool = True,
    min_lines_in_range: int = 2,
) -> "RedshiftResult"
```

### Default line set

```python
DEFAULT_LINES = [
    "Lya", "CIV_doublet", "HEII_1640", "CIII]",
    "OII_doublet", "NeIII_3869",
    "HBETA", "OIII_4959", "OIII_5007",
    "Ha", "NII_6585",
    "SII_6718", "SII_6732",
]
```

All names are keys of `REST_LINES_A`. User can override with `lines=[...]`.

### `RedshiftResult` (dataclass)

```
z_best : float                          # MAP from refined sub-grid of best peak
z_ci68 : tuple[float, float]            # central 68% credible interval of best peak
z_ci95 : tuple[float, float]            # central 95% credible interval of best peak

peaks : list[Peak]                      # ranked by integrated probability
is_decisive : bool                      # dchi2(best, runner-up) > 25

z_grid_coarse : np.ndarray
chi2_coarse   : np.ndarray
P_z_coarse    : np.ndarray              # exp(-(chi2-chi2_min)/2) * prior, normalised

z_grid_fine : np.ndarray                # concatenated refined sub-grids
chi2_fine   : np.ndarray
P_z_fine    : np.ndarray

lines_used : list[str]                  # full default/user set (provenance)
grating    : str | None
spec       : Spectrum                   # for plot helper

def plot(self) -> "plotly.Figure": ...   # two-panel diagnostic
```

```
@dataclass
class Peak:
    z : float
    prob : float                        # integrated P inside this local mode
    dchi2 : float                       # relative to global best
    ci68 : tuple[float, float]
    ci95 : tuple[float, float]
    n_lines_used : int                  # in-range lines at this z
    lines_used   : list[str]
```

## 3. Algorithm

### 3.1 Coarse sweep

Build a uniform grid `z_coarse = np.arange(z_min, z_max + dz_coarse, dz_coarse)`.

Default `dz_coarse`:

```
R = R_grating(spec.grating or "prism", lam_ref)        # ~100 for prism, ~1000 for medium
dz_coarse = 0.5 / R                                     # one line moves ~half a resolution
                                                        # element per coarse step
```

For each `z` in `z_coarse`, compute `chi2(z)` via the per-z linear system (Section 3.2). **The full grid is always evaluated. No early stopping under any circumstance.**

### 3.2 Per-z score

1. Predict `mu_i_obs = lambda_rest_i * (1 + z)` for every line in `lines`.
2. Keep only lines satisfying `wave_min + 3*sigma_LSF <= mu_i_obs <= wave_max - 3*sigma_LSF`, where `sigma_LSF = mu_i_obs / (2.355 * R(mu_i_obs))`.
3. Let `m` be the number of in-range lines. If `m < min_lines_in_range`, mark this z as having insufficient coverage and assign `chi2(z) = chi2_continuum_only(z)` (i.e. just the polynomial continuum fit). These points form the floor of the chi^2 curve.
4. Build design matrix `D` of shape `(npix, m + continuum_order + 1)`:
   - First `m` columns: unit-amplitude Gaussian profiles at `mu_i_obs`, sigma from the LSF.
   - Last `continuum_order + 1` columns: orthonormal polynomial basis over wavelength.
5. Row-weight by `1/err` (skipping pixels with non-positive or non-finite err).
6. Solve `D @ a = data` (rows weighted by `1/err`) using `scipy.optimize.lsq_linear` with per-column bounds:
   - Line columns: `bounds=(0, inf)` — emission only.
   - Continuum columns: `bounds=(-inf, inf)` — unconstrained.
   This is a single call; no iteration, no manual reparameterisation. `lsq_linear` defaults (`method="trf"`) handle the mixed bounds correctly.
7. `chi2(z) = sum_pix ((data - D @ a) / err)^2` over the same pixel mask.

### 3.3 Peak detection and refinement

1. Find local minima of `chi2_coarse`: indices `i` with `chi2[i] < chi2[i-1]` and `chi2[i] < chi2[i+1]`.
2. Sort the minima by `chi2` ascending. Refine the top `n_peaks` AND any others within `refine_dchi2 = 9` of the global minimum (so a thicket of competing peaks is fully resolved, while a clear winner still yields a non-empty runner-up list).
3. For each kept minimum at `z_c`:
   - Build fine sub-grid: `np.linspace(z_c - 10*dz_coarse, z_c + 10*dz_coarse, 401)`. Clip to `[z_min, z_max]`.
   - Re-evaluate `chi2` on the fine sub-grid using the same per-z kernel.
   - Refined peak position: parabolic interpolation around the fine-grid minimum.
4. Concatenate all fine sub-grids into `z_grid_fine`, `chi2_fine`. Order does not matter; they are stored only for plotting and diagnostics.
5. **Final ranking**: after Section 3.4 computes integrated probabilities, the `peaks` list is sorted by `prob` descending (NOT by chi2). With a flat prior these orderings coincide; with a non-flat prior they can differ, and integrated probability is the right metric.

### 3.4 Credible intervals

For each refined peak:

1. Define the peak's local mode: the contiguous z-range around the peak bounded by where the fine-grid `chi2` first rises by `dchi2 = 25` on either side (or hits the grid edge).
2. Inside that range, compute the **unnormalised local posterior** `u(z) = exp(-(chi2(z) - chi2_peak)/2) * prior(z)`. The prior is applied here; nowhere else in this section.
3. Normalise `u` to a unit-area PDF over the local range → `P_local(z)`. CI68 / CI95 are the central 68%/95% quantiles of `P_local`.
4. The peak's **raw mass** is `m_k = integral(u(z) dz)` over the local range. The peak's reported `prob = m_k / sum_j m_j` over all peaks returned, so `sum(p.prob for p in peaks)` equals 1 by construction.

`is_decisive = (peaks[1].dchi2 > 25) if len(peaks) > 1 else True`.

## 4. Prior implementation

```python
def _evaluate_prior(prior, z):
    if prior is None:
        return np.ones_like(z)
    if callable(prior):
        return np.clip(prior(z), 0.0, None)
    if isinstance(prior, tuple) and len(prior) == 2 and np.isscalar(prior[0]):
        lo, hi = prior
        return ((z >= lo) & (z <= hi)).astype(float)
    # list of (lo, hi) windows, equal weight in each
    p = np.zeros_like(z)
    for lo, hi in prior:
        p[(z >= lo) & (z <= hi)] = 1.0
    return p
```

Hard top-hat semantics: P(z) is exactly zero outside the prior support, not merely small.

## 5. Module placement

- **New file**: `src/jwspecfit/redshift.py`.
  - `fit_redshift`, `RedshiftResult`, `Peak`, internal helpers `_per_z_chi2`, `_evaluate_prior`, `_default_dz_coarse`.
- **Reused** (no changes needed): `REST_LINES_A` from `lines.py`, `R_prism` / `R_grating` from `resolution.py`, the Gaussian profile from `models.py`.
- **Public re-exports**: `fit_redshift`, `RedshiftResult`, `Peak` added to `__init__.py`.

The kernel does **not** call `fit_lines`. `fit_lines` is too heavy for thousands of evaluations (constraints, broad components, bootstrap path). The per-z evaluator is a lean linear-system solve.

## 6. Plot helper

`RedshiftResult.plot()` returns a plotly Figure with two stacked subplots:

- **Top**: `dchi2 = chi2_coarse - chi2_coarse.min()` vs z, with horizontal lines at dchi2 = 1, 4, 9 and vertical dashed lines at each peak (annotated with `z` and `dchi2`).
- **Bottom**: the input spectrum, with vertical bars at the observed positions of the in-range lines at `z_best`, coloured by line, and labels in the legend below the plot — reuses `plot_spectrum_interactive` style.

## 7. Tests (`tests/test_redshift.py`)

| Test | Purpose |
|---|---|
| `test_recovery_synthetic` | Inject a clean strong-line spectrum at z ∈ {0.5, 2.0, 5.0, 8.0, 12.0}; recover each to within 2 × `dz_coarse`. |
| `test_no_data_tolerance` | PRISM coverage where Hα is off the red end at z=8; recovery succeeds from blue lines alone; `n_lines_used` reflects what was actually in range. |
| `test_prior_window_excludes_truth` | Synthetic z=8 spectrum with `prior=[(0, 5), (10, 20)]` should NOT recover z=8 (zero prior support), `is_decisive=False`. |
| `test_prior_callable` | A callable prior centred on z=8 reproduces the unprior result; a callable prior centred on z=2 shifts the MAP. |
| `test_decisiveness_clean` | Clean high-SNR strong-line spectrum → `is_decisive=True`. |
| `test_decisiveness_noise` | Continuum-only noise spectrum → `is_decisive=False`, no peak with `dchi2 < 9` of a baseline. |
| `test_aliasing_peak_reported` | Hα + [NII] only (no Hβ in range) → secondary peak misidentifying these as [OIII]4959/5007 must appear in `peaks`. Regression test for the classic z-finder failure mode. |
| `test_no_early_stopping` | Patch the per-z evaluator with a counter; confirm it is called exactly `len(z_grid_coarse)` times for the coarse pass, regardless of how good an early hit is. |
| `test_min_lines_in_range` | A spectrum where only one default line falls in range at z=X should NOT produce a refined peak at z=X. |
| `test_speed` | Full 0–20 coarse pass on a 430-pixel PRISM spectrum runs in < 5 s on a developer laptop. |

## 8. Documentation

- Add `docs/user_guide/redshift_fitting.md` (worked example, plot, prior options).
- Add API entry under `docs/api_reference/`.
- Mention `fit_redshift` in `quickstart.md` as the entry point when z is unknown.

## 9. Out of scope (deferred)

- Photo-z prior input from external catalogues (BPZ / EAZY) — user can pass any callable, so this is supported by composition.
- Continuum templates (galaxy SED libraries) instead of low-order polynomial — adds substantial complexity for marginal gain when strong lines are present. If needed later, add a `continuum_template=` kwarg.
- Velocity-offset fits per line (e.g. outflow components) — `fit_redshift` returns a single z; per-line velocity refinement is a job for `fit_lines` afterwards.
- Joint z + N_HI fit (for DLA-dominated UV continua) — separate problem, handled by `fit_NHI(... fit_z=True)` if ever needed.

## 10. Backward compatibility

All-new public symbol. No existing API is touched. No notebook regressions expected.
