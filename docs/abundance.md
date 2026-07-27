# jwspecabund — Code Workflow and Logic

Documentation of the internal code flow for `jwspecabund.compute_abundances()`,
covering orchestration, SNR gating, fallback chains, and the function call graph.
For the scientific methodology and equations, see `abundance_methodology.md`.

---

## Package Structure

```
src/jwspecabund/
  __init__.py           Public API re-exports
  _core.py              Orchestrator: compute_abundances() and all internal helpers
  direct.py             PyNEB-based Te, ne, ionic abundances, total abundances
  martinez25_icf.py     Martinez+25 bicubic ICF surfaces and logU diagnostics
  icf.py                Izotov+06 polynomial ICFs (N, Ne, S, Ar)
  forward.py            Bayesian forward model (Cullen+25), NV emissivity
  strong_line.py        Sanders+25 strong-line calibrations
  dust.py               Attenuation laws and Balmer-decrement Av
  result.py             AbundanceResult dataclass
```

---

## 1. Entry Point: `compute_abundances()`

**File:** `_core.py` (`compute_abundances`)

```python
compute_abundances(result, z, *, dust_correct=True, method="auto", ...)
```

Accepts any `FitResult`, `BroadFitResult`, `MCMCResult`, or `MCMCBroadFitResult`
from `jwspecfit` / `jwspecmcmc`.

### High-level flow

```
result ──> _extract_fluxes() ──> (fluxes, errors, is_mcmc)
                                       │
                               Dust correction (Av from Balmer or user-supplied)
                                       │
                               _filter_low_snr()  [per-line SNR gating]
                                       │
                          ┌─────── method selection ──────┐
                          │            │                   │
                     use_direct    use_forward        strong_line
                          │            │                   │
               ┌──── is_mcmc? ────┐   forward_model()    sanders25_metallicity()
               │                  │
      _run_direct_mcmc()    _run_direct()
               │                  │
               └─── AbundanceResult ──┘
```

---

## 2. Flux Extraction

### `_extract_fluxes(result)` -> `(fluxes, errors, is_mcmc)`

- Iterates `result.lines`, skipping `_BROAD` entries on first pass.
- If `flux_err` is a tuple `(lo, hi)`, symmetrises to `0.5*(lo+hi)` and sets `is_mcmc=True`.
- Second pass: sums broad Balmer components (`Ha_BROAD`, `HBETA_BROAD`, etc.) into
  the narrow totals so the Balmer decrement uses total hydrogen flux.

### `_extract_posteriors(result)` -> `{name: flux_posterior_array}`

- Only called when `is_mcmc=True`.
- Extracts `lr.flux_posterior` arrays, summing broad Balmer posteriors sample-by-sample.

---

## 3. Dust Correction

1. **Derive Av** (if not user-supplied): `compute_Av_from_balmer()` using Hgamma/Hbeta.
   Falls back to Av=0 if neither Hgamma nor Hbeta is available.
2. **Apply correction**: `_apply_dust_correction()` -> `dust_correct_fluxes()`.
   Each line corrected by `F_corr = F_obs * 10^(0.4 * A_lambda)`.
3. **Posteriors**: If MCMC, each posterior array is also dust-corrected with the
   same Av (not re-derived per sample).

---

## 4. SNR Gating: `_filter_low_snr()`

Removes lines with `flux / error < snr_line` (default 2.0).

**Protected lines** (`_SNR_PROTECTED`) are never removed:

```python
_SNR_PROTECTED = {
    "OIII_4363", "OIII_5007", "OIII_4959", "HBETA",
    "NII_6585",
    "NIII_1749", "NIII_1752",
    "NIV_1483", "NIV_1486",
    "NV_1", "NV_2",
    "CIV_1", "CIV_2",
    "CIII]_1907", "CIII]",
    "CII]_2324", "CII]_2326",
}
```

Rationale: Te-diagnostic lines, Hbeta (normalisation), and nitrogen/carbon UV lines
are gated by their own dedicated checks (auroral SNR, nitrogen gating, logU doublet checks).

---

## 5. Method Selection

| `method=` | Condition | Path |
|-----------|-----------|------|
| `"auto"` | OIII_4363 SNR >= `snr_auroral` | Direct Te (4363) |
| `"auto"` | OIII_4363 absent but O III] 1666 available | Direct Te (1666 fallback) |
| `"auto"` | No auroral line detected | Strong-line |
| `"direct"` | Always | Direct Te |
| `"forward"` | Always | Bayesian forward model |
| `"strong_line"` | Always | Sanders+25 calibrations |

---

## 6. Direct Te Method: `_run_direct()` / `_run_direct_mcmc()`

Implements the Berg+25 six-step procedure. The two variants differ only in
error propagation: `_run_direct()` uses parametric Gaussian MC perturbation of
fluxes (`n_mc`), while `_run_direct_mcmc()` draws from MCMC posterior chains
(`n_posterior`).

### Step 1 — Multi-phase electron density

**Function:** `_compute_multi_ne(fluxes, errors, snr_ne, ne_high_max)`

```
ne_low:  [SII] 6718/6732  -->  compute_ne(..., doublet="SII")
         fallback: [OII] 3726/3729  -->  compute_ne(..., doublet="OII")
         fallback: [Si III] 1883/1892 (UV)  -->  compute_ne_SiIII()
         fallback: redshift-dependent ne_zone_fallback("low", z)

ne_mid:  CIII] 1907/1909  -->  compute_ne_CIII()

ne_high: NIV] 1483/1486  -->  compute_ne_NIV()
         fallback: [Ar IV] 4711/4740  -->  compute_ne_ArIV()
         fallback: redshift-dependent ne_zone_fallback("high", z)
         clamp: if ne_high > ne_high_max (default 5e5), use ne_mid (else ne_low)

ne_Opp:  O2+/Ne2+ zone density -- [Ar IV] 4711/4740 (preferred),
         else ne_mid, else ne_zone_fallback("mid", z)
```

**SNR gate:** Each doublet requires both members to have `flux/error >= snr_ne`
(default 3.0), checked by `_doublet_snr_ok()`.

### Step 1b — Density overrides

When `ne_low_override`, `ne_mid_override`, or `ne_high_override` are set,
the corresponding diagnostic computation is bypassed and the user-supplied
value is used directly.

### Step 2 — Electron temperature

```python
# Primary: [OIII] 4363
Te_high = compute_Te_OIII(flux_4363, flux_5007, flux_4959, ne_high)

# Fallback: O III] 1666 (when 4363 is unavailable)
Te_high = compute_Te_OIII_1666(flux_1666, flux_5007, flux_4959, ne_high)

Te_low  = Te_low_from_high(Te_high, relation=Te_relation)
```

- `compute_Te_OIII()` uses PyNEB with `log=True, start_x=3.0, end_x=5.0` for
  convergence at Te > 25,000 K.
- `compute_Te_OIII_1666()` uses the UV 1666/(5007+4959) ratio with solver
  range extended to 250,000 K (`end_x=5.4`).
- `Te_int_from_high()` / `Te_low_from_high()`: default `"3_tier"` — Garnett
  (1992) throughout, as adopted by Martinez+2025 — `Te_int = 0.83 * Te_high +
  1700`, `Te_low = 0.70 * Te_high + 3000`; also `"classical"`/`"garnett"` (same
  low relation) and `"desi"` (`0.648 * Te_high + 3270`).

### Step 3 — Ionic abundances

**Function:** `compute_ionic_abundances(fluxes, Te_high, Te_low, ne_low, ne_high)`

For each ion, computes `X^i+/H+ = (F_line/F_Hb) * (eps_Hb/eps_line)` using PyNEB.

Zone assignments:
- **Te_high, ne_high:** C3+, N3+, N4+
- **Te_high, ne(O2+):** O2+, Ne2+
- **Te_high, ne_mid (C III]):** C2+, N2+ — Berg+2025 Table 3, "T_e,high ; n_e(C+2)"
- **Te_low, ne_low:** O+, N+, S+
- **Te_mid, ne_mid:** S2+, Ar2+

UV doublet handling: if both members present, sums fluxes and uses total emissivity.
If only one member present, uses single-member flux with single-line emissivity.

NV (N4+): Manual 3-level atom emissivity (`_nv_emissivity()` in `forward.py`)
because PyNEB has no N V atomic data.

### Step 4 — O/H and Z/Zsun

```python
OH = O+/H+ + O++/H+
OH_12 = 12 + log10(OH)
Z_Zsun = 10^(OH_12 - 8.69)
```

### Step 5 — Ionisation parameter

**Function:** `_compute_logU(fluxes, Z_Zsun, ne_high, errors, snr_logU)`

```
Priority 1: N43 = NIV]/NIII]  -->  log_U_from_N43(log_N43, Z_Zsun, ne)
  Requirements:
    - Both NIV] members present with total doublet SNR >= snr_logU
    - Both NIII] members present with total doublet SNR >= snr_logU
    - log(N43) within [-3.0, 0.5] validity range
    - Resulting logU within [-3.5, -1.0] validity range

Priority 2: O32 = [OIII]5007/[OII]3727  -->  log_U_from_O32(log_O32, Z_Zsun, ne)
  Requirements:
    - OIII_5007 > 0 and OII > 0
```

### Step 6 — Total abundances with ICFs

**Pre-step: Nitrogen SNR gating**

**Function:** `_gate_nitrogen_ions(ionic, fluxes, errors, snr_NO)`

Runs for **all** ICF methods (not just `direct_sum`). For each nitrogen ion:

- **Doublets** (NIII], NIV], NV): requires both members to have:
  1. Positive flux AND individual member SNR >= 1.0 (catches machine-zero fluxes
     like 1e-46 that are technically positive)
  2. Total doublet SNR >= `snr_NO` (default 1.5)
  - If either check fails, sets `ionic[ion_key] = 0.0` (excluded from all ICFs)

- **Singles** (NII 6585): requires line SNR >= `snr_NO`

**N/O computation: `compute_total_abundances()`**

The `icf_method` parameter selects the N/O path:

#### `icf_method="direct_sum"`

Tiered fallback chain (no ICF, no logU needed):

```
Tier 1: N+ > 0 AND (N2+ + N3+) > 0
  N/O = (N+ + N2+ + N3+ + N4+) / (O+ + O2+)
  Label: "Np_Npp_Nppp"

Tier 2/3: (N2+ + N3+) > 0 AND O2+ > 0
  N/O = (N2+ + N3+ + N4+) / O2+
  Label: "Npp_Nppp_Opp" | "Nppp_Opp" | "Npp_Opp"
  (sub-label depends on which UV ions are present)

Tier 4: N+ > 0 AND O+ > 0
  N/O = ICF_N(Izotov+06) * N+/O+
  Label: "izotov06_fallback"
```

#### `icf_method="martinez25"` (or `"auto"` with logU available)

Uses `compute_NO_martinez25()` with 5-priority ICF selection:

```
ICF 5: N2+ > 0 AND N3+ > 0 AND O2+ > 0  -->  (N2+ + N3+)/O2+ * ICF
ICF 4: N+ > 0 AND N2+ > 0 AND O+ > 0 AND O2+ > 0  -->  (N+ + N2+)/(O+ + O2+) * ICF
ICF 2: N2+ > 0 AND O2+ > 0  -->  N2+/O2+ * ICF
ICF 1: N+ > 0 AND O+ > 0  -->  N+/O+ * ICF
ICF 3: N3+ > 0 AND O2+ > 0  -->  N3+/O2+ * ICF  (log-space surface)
```

**Fallback:** If Martinez+25 returns `(None, None)` (no suitable ionic ratios
after SNR gating), falls through to the full `direct_sum` tier chain.

#### `icf_method="izotov06"`

Simple: `N/O = ICF_N(Izotov+06) * N+/O+`. Requires N+ and O+ only.

#### Other elements (always computed)

- **S/O:** `ICF_S(Izotov+06) * (S+ + S2+) / O/H`
- **Ne/O:** `ICF_Ne(Izotov+06) * Ne2+/O2+`
- **Ar/O:** `ICF_Ar(Izotov+06) * Ar2+/O2+`
- **C/O:** by default (`co_icf_method="auto"`) the **Martinez (2026, in prep.)
  C²⁺/O²⁺ ICF** — `ICF_C(Martinez) × C2+/O2+` — whenever logU, Z, C²⁺ and O²⁺
  are available; otherwise the legacy path — `(C+ + C2+ + C3+) / (O+ + O2+)`
  when CII] 2326 detected, else `ICF_C(Garnett+97) × (C2+ + C3+) / O2+`
- **N/O_UV_raw:** `(N2+ + N3+ + N4+) / O2+` (for comparison, no ICF)

**ICF tier locking:** When `icf_tier` is set (e.g. `"NppNppp_Opp"`), the
specified Martinez+25 ICF tier is used for both the point estimate and all
MC iterations, preventing tier switching across iterations.

### Upper Limits for Non-Detected Ions

**Functions:** `_compute_continuum_rms_limits()`, `_compute_ionic_upper_limits()`

When an ion is not detected (ionic abundance = 0 after SNR gating), a 3σ
upper limit is computed from the local continuum noise:

1. **Flux upper limit (continuum RMS method):**
   For each non-detected line, the RMS of the fit residuals is measured in
   a window around the expected line position (±5σ, excluding the central
   ±2σ).  The flux upper limit is:

   ```
   flux_ul = 3 × RMS_continuum × σ_inst × √(2π)
   ```

   where σ_inst = λ / (R × 2.3548) is the instrumental Gaussian width.
   This is independent of the bootstrap/MCMC error estimation and asks
   "given the noise here, what is the brightest line I could have missed?"

   For doublets, the individual member limits are added in quadrature
   (independent noise): `flux_ul_total = √(Σ flux_ul_i²)`.

   The flux limits are dust-corrected to match the corrected flux scale
   used for detected lines.

2. **Ionic abundance upper limit:**
   The flux upper limit is converted to an ionic abundance via PyNEB
   emissivities at the measured T_e and n_e, using the same formula as
   for detections:

   ```
   (X^i+/H+)_ul = (flux_ul / F_Hβ) × (ε_Hβ / ε_line)
   ```

3. **Abundance ratio upper limits:**
   For N/O, the detected nitrogen ions are summed with the upper limits
   for non-detected ions, then divided by total oxygen.  This gives a
   conservative upper limit: `log(N/O) < log10[(ΣN_detected + ΣN_ul) / O_total]`.
   The same approach is used for C/O.

**Fallback:** When the spectrum is not available (e.g. mock results),
the code falls back to using fit errors: `flux_ul = 3σ × √(Σ err²)`.

Results are stored in `AbundanceResult.ionic_upper_limits` (values) and
`AbundanceResult.ionic_ul_details` (metadata: source lines, flux limit,
method used).

### MC Error Propagation

**`_run_direct()` (least-squares input):**
```
for i in range(n_mc):
    mc_fluxes = {name: N(flux, error) for each line}
    redo steps 1-6 with mc_fluxes
    accumulate OH, NO, CO posteriors
```
- ne is held fixed (varying it adds noise without improving accuracy)
- Te_high and Te_low are re-derived per iteration → `Te_high_err`, `Te_low_err`
- logU is re-derived per iteration → `logU_err`
- logU diagnostic choice is locked to point estimate (prevents switching between
  N43 and O32 across iterations)
- logU value is clamped to [-3.5, -1.0] validity range per iteration
- A_V uncertainty is propagated from the Balmer decrement → `Av_err`
- S/O, Ne/O, Ar/O uncertainties are propagated → `SO_err`, `NeO_err`, `ArO_err`
- Nitrogen gating uses **original** errors (same ions included/excluded as point estimate)
- ICF tier is locked to match the point estimate (no tier switching in MC loop)

**`_run_direct_mcmc()` (MCMC input):**
- Draws from actual posterior chains instead of Gaussian perturbation
- Thins to `n_posterior` samples if chains are longer
- Same locking/clamping logic as above, using median errors for gating

---

## 7. Forward Model: `forward_model()`

**File:** `forward.py`

Free parameters: `log_Te`, `log_ne`, plus one `log(X^i+/H+)` per detected ion.

```
_build_observables()     -->  observed line/Hb ratios + errors
_determine_free_params() -->  parameter list from detected ions
_find_map()              -->  scipy.optimize.differential_evolution for MAP seed
_run_emcee()             -->  emcee ensemble sampler (or _run_dynesty() for nested)
_build_forward_result()  -->  posterior summary (median, 16/84 percentiles)
```

The model predicts `F_line/F_Hb = (X^i+/H+) * eps_line / eps_Hb` for each
observable, using PyNEB emissivities and the Aller (1984) Hb formula.

NV lines use `_nv_emissivity()` (3-level atom, Aggarwal & Keenan 2004 collision
strengths).

---

## 8. Strong-Line Method: `sanders25_metallicity()`

**File:** `strong_line.py`

1. `compute_line_ratios()` builds log10 ratios (O3, O2, R23, O32) with SNR gating.
2. `_chi2_simultaneous()` evaluates combined chi-squared across all available diagnostics.
3. `minimize_scalar()` finds best-fit `12+log(O/H)` in [6.5, 9.0].
4. MC loop perturbs ratios by `sqrt(sigma_obs^2 + sigma_cal^2)` for uncertainty.

When MCMC posteriors are available, posterior samples are propagated directly
(each sample is a separate chi-squared minimisation).

---

## 9. Key SNR Parameters

| Parameter | Default | What it gates | Where checked |
|-----------|---------|---------------|---------------|
| `snr_auroral` | 3.0 | OIII 4363 for direct vs strong-line selection | `compute_abundances()` |
| `snr_line` | 2.0 | All lines not in `_SNR_PROTECTED` | `_filter_low_snr()` |
| `snr_ne` | 3.0 | Both members of density-sensitive doublets | `_doublet_snr_ok()` |
| `snr_logU` | 1.5 | Total doublet SNR for NIV]/NIII] in N43 | `_compute_logU()` |
| `snr_NO` | 1.5 | Nitrogen ion doublet completeness + total SNR | `_gate_nitrogen_ions()` |

---

## 10. Complete Function Call Graph

```
compute_abundances(result, z, ...)
  │
  ├── _extract_fluxes(result)
  ├── _extract_posteriors(result)        [if MCMC]
  │
  ├── compute_Av_from_balmer()           [if dust_correct and Av not supplied]
  ├── _apply_dust_correction()
  │     └── dust_correct_fluxes()
  │           ├── salim_attenuation()    [if dust_law="salim"]
  │           └── cardelli_extinction()  [if dust_law="cardelli"]
  │
  ├── _filter_low_snr()
  ├── _compute_continuum_rms_limits()   [flux upper limits from residual RMS]
  │
  ├── [Direct method]
  │     ├── _run_direct_mcmc()           [if MCMC posteriors available]
  │     └── _run_direct()                [if least-squares]
  │           │
  │           ├── Step 1: _compute_multi_ne()
  │           │     ├── compute_ne()          [SII or OII]
  │           │     ├── compute_ne_NIV()      [NIV] 1483/1486]
  │           │     └── compute_ne_CIII()     [CIII] 1907/1909]
  │           │
  │           ├── Step 2: compute_Te_OIII()       [primary: 4363]
  │           │           compute_Te_OIII_1666()  [fallback: 1666]
  │           │           Te_low_from_high()
  │           │
  │           ├── Step 3: compute_ionic_abundances()
  │           │     ├── _ionic_abundance()    [per ion via PyNEB]
  │           │     └── _hbeta_emissivity()
  │           │           └── hbeta_emissivity_aller84()  [Te > 30000 K]
  │           │
  │           ├── Step 4: OH = O+/H+ + O++/H+, Z_Zsun
  │           │
  │           ├── Step 5: _compute_logU()
  │           │     ├── log_U_from_N43()     [preferred]
  │           │     └── log_U_from_O32()     [fallback]
  │           │
  │           ├── Step 6: _gate_nitrogen_ions()
  │           │           _compute_ionic_upper_limits()
  │           │           compute_total_abundances()
  │           │             ├── [direct_sum tiers]
  │           │             ├── compute_NO_martinez25()
  │           │             │     ├── icf_NppNppp_Opp()  [ICF 5]
  │           │             │     ├── icf_NpNpp_OpOpp()  [ICF 4]
  │           │             │     ├── icf_NppOpp()       [ICF 2]
  │           │             │     ├── icf_NpOp()         [ICF 1]
  │           │             │     └── icf_NpppOpp()      [ICF 3]
  │           │             ├── icf_nitrogen()           [Izotov+06]
  │           │             ├── icf_sulfur()
  │           │             ├── icf_neon()
  │           │             └── icf_argon()
  │           │
  │           └── MC loop: repeat steps 1-6 with perturbed fluxes
  │
  ├── [Forward model]
  │     └── forward_model()
  │           ├── _build_observables()
  │           ├── _determine_free_params()
  │           ├── _find_map()
  │           ├── _run_emcee() / _run_dynesty()
  │           │     └── _log_posterior() -> _log_likelihood() -> _predict_ratios()
  │           └── _build_forward_result()
  │
  └── [Strong-line]
        └── sanders25_metallicity()
              ├── compute_line_ratios()
              ├── _chi2_simultaneous()
              └── MC loop
```

---

## 11. AbundanceResult Fields

| Field | Type | Methods | Description |
|-------|------|---------|-------------|
| `method` | str | all | `"direct"`, `"forward"`, `"strong_line"` |
| `OH` | float | all | 12 + log(O/H) |
| `OH_err` | float or (lo, hi) | all | Symmetric std or asymmetric 16/84 percentile |
| `NO` | float or None | direct, forward | log(N/O) |
| `NO_err` | float or (lo, hi) or None | direct, forward | |
| `CO` | float or None | direct, forward | log(C/O) |
| `CO_err` | float or (lo, hi) or None | direct, forward | |
| `SO` | float or None | direct | log(S/O) |
| `SO_err` | float or None | direct | Uncertainty on log(S/O) |
| `NeO` | float or None | direct | log(Ne/O) |
| `NeO_err` | float or None | direct | Uncertainty on log(Ne/O) |
| `ArO` | float or None | direct | log(Ar/O) |
| `ArO_err` | float or None | direct | Uncertainty on log(Ar/O) |
| `Te_high` | float or None | direct, forward | T_e(O2+) in K |
| `Te_high_err` | float or None | direct | Uncertainty on T_e(high) |
| `Te_low` | float or None | direct | T_e(O+/N+) in K |
| `Te_low_err` | float or None | direct | Uncertainty on T_e(low) |
| `ne` | float or None | direct, forward | n_e (low-ionisation zone) |
| `ne_low` | float or None | direct | n_e from [SII]/[OII] |
| `ne_mid` | float or None | direct | n_e from CIII]/SiIII] |
| `ne_high` | float or None | direct | n_e from NIV]/CIII] |
| `logU` | float or None | direct | Ionisation parameter log(U) |
| `logU_err` | float or None | direct | Uncertainty on log(U) |
| `Av` | float or None | all | V-band attenuation |
| `Av_err` | float or None | all | Uncertainty on A_V |
| `ionic` | dict or None | direct, forward | Ionic abundances |
| `icf_method` | str or None | direct | `"martinez25"`, `"direct_sum"`, `"izotov06"` |
| `NO_icf_name` | str or None | direct | ICF identifier (e.g. `"NppNppp_Opp"`) |
| `OH_posterior` | ndarray or None | all | MC/posterior samples |
| `NO_posterior` | ndarray or None | direct, forward | |
| `CO_posterior` | ndarray or None | direct, forward | |
| `ratios_used` | list or None | strong_line | e.g. `["O3", "R23", "O32"]` |
| `chi2` | float or None | strong_line | Best-fit chi-squared |
| `excluded_lines` | list or None | all | Lines removed by SNR filter |

---

## 12. N/O ICF Label Reference

Labels reported in `AbundanceResult.NO_icf_name`:

| Label | Method | Ions Used | Meaning |
|-------|--------|-----------|---------|
| `NppNppp_Opp` | martinez25 | N2+, N3+, O2+ | Martinez+25 ICF 5 (best) |
| `NpNpp_OpOpp` | martinez25 | N+, N2+, O+, O2+ | Martinez+25 ICF 4 |
| `NppOpp` | martinez25 | N2+, O2+ | Martinez+25 ICF 2 |
| `NpOp` | martinez25 | N+, O+ | Martinez+25 ICF 1 |
| `NpppOpp` | martinez25 | N3+, O2+ | Martinez+25 ICF 3 (worst) |
| `Np_Npp_Nppp` | direct_sum | N+, N2+/N3+, O+, O2+ | Tier 1: all zones |
| `Npp_Nppp_Opp` | direct_sum | N2+, N3+, O2+ | Tier 2: both UV N ions |
| `Nppp_Opp` | direct_sum | N3+, O2+ | Tier 3: NIV] only |
| `Npp_Opp` | direct_sum | N2+, O2+ | Tier 3: NIII] only |
| `izotov06_fallback` | direct_sum | N+, O+ | Tier 4: optical fallback |
| (none) | izotov06 | N+, O+ | Standard Izotov+06 |
