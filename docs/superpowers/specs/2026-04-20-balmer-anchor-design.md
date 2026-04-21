# `balmer_anchor` parameter for multi-Balmer A_V derivation

**Date:** 2026-04-20
**Package:** `jwspecabund`
**Status:** approved, ready for implementation planning

## Motivation

`compute_Av_multi_balmer` currently hard-codes Hβ as the denominator in the
Balmer decrement (uses Hγ/Hβ, Hδ/Hβ, H9/Hβ, H10/Hβ). For spectra where Hα is
detected at high SNR (e.g. the Hα grating at moderate redshift, or low-z
sources where Hα and higher Balmer lines are in the same grating), it is
preferable to anchor the decrement to Hα instead of Hβ. Hα is brighter, so
its statistical uncertainty is smaller, and anchoring to the brightest line
generally gives the best-constrained A_V.

## Scope

- Add an `anchor` selection (Hα or Hβ) to the multi-Balmer A_V derivation.
- Surface the choice through `compute_abundances` as `balmer_anchor`.
- Default behaviour (anchor = Hβ) is byte-identical to the current behaviour.

### Explicit non-goals (YAGNI)
- **Not** generalising to arbitrary anchors (Hγ, Hδ, etc.). Only Hα and Hβ.
- **Not** changing the Lyα escape-fraction pipeline or `LYA_CASE_B_RATIOS`.
- **Not** changing dust-correction application — dust correction is already
  applied wavelength-dependently per line and is correct.
- **Not** touching the path where the user supplies a fixed `Av` — the anchor
  is irrelevant in that branch.

## Design

### 1. Unify the Balmer reference table in `src/jwspecabund/dust.py`

Replace the existing `_BALMER_DECREMENT_LINES` with a single master table
keyed by line name, holding rest wavelength and Case B ratio to Hβ
(T=10⁴ K, n_e=100 cm⁻³):

```python
_BALMER_LADDER: dict[str, tuple[float, float]] = {
    # name:  (rest λ in Å, intrinsic ratio to Hβ)
    "Ha":     (6564.61, 2.86),
    "HBETA":  (4862.68, 1.00),
    "HGAMMA": (4341.68, 0.468),
    "HDELTA": (4102.89, 0.259),
    "H9":     (3836.48, 0.0731),
    "H10":    (3799.00, 0.0530),
}
# Hε and H8 remain excluded (blended with [NeIII] 3968 and HeI 3889).
```

The old `_BALMER_DECREMENT_LINES` can be removed.

### 2. Add `anchor` argument to `compute_Av_multi_balmer`

```python
def compute_Av_multi_balmer(
    fluxes, errors, law="salim", snr_min=3.0,
    anchor: str = "HBETA",  # NEW — accepts "HBETA" or "Ha"
    **kwargs,
):
    ...
```

Implementation:

- Validate `anchor in {"HBETA", "Ha"}`; raise `ValueError` otherwise.
- Look up anchor wavelength and its Hβ-referenced ratio from `_BALMER_LADDER`.
- Numerators = every entry in `_BALMER_LADDER` other than the anchor, provided
  the line is present in `fluxes` with positive flux and SNR ≥ `snr_min`.
- Convert intrinsic ratios: `ratio_num_over_anchor = r_num_over_Hb / r_anchor_over_Hb`.
  (For anchor = Hβ this is a no-op; for anchor = Hα each ratio is divided by 2.86.)
- Short-circuit return when the anchor line is missing or ≤ 0.
- Logging and the returned `individual` dicts continue to report per-decrement
  A_V values, with line names reflecting the actual decrement used.

### 3. Surface in `compute_abundances` (`src/jwspecabund/_core.py`)

- Add kwarg `balmer_anchor: str = "HBETA"` to `compute_abundances`.
- Pass through to `compute_Av_multi_balmer` at the derive-from-decrement call
  site (current lines ~2288–2291). No effect when the user supplies `Av`
  directly.
- Update the diagnostic summary string (current line ~2540) to include the
  anchor, e.g. `"weighted mean of 3 decrements (anchor=Hα): …"`.
- Update the per-line `logger.info` line (current line ~2298) to print
  `"A_V from %s/%s = …"` with the anchor name substituted.

### 4. Tests

In `tests/`, add one focused test:

- Build a synthetic dict of Hα, Hβ, Hγ, Hδ fluxes at a chosen `Av_true` by
  forward-applying `salim_attenuation` to Case B ratios anchored on Hα.
- Call `compute_Av_multi_balmer(..., anchor="Ha")` and assert the returned
  weighted-mean A_V matches `Av_true` to within a tight numerical tolerance.
- Optional: a second test asserting that, for the same synthetic fluxes,
  `anchor="HBETA"` also recovers `Av_true` (cross-consistency check).

Existing tests using the default `anchor="HBETA"` continue unchanged.

### 5. Dust-correction audit (already correct; documented here for the record)

`dust_correct_fluxes` (`dust.py:213`) iterates per line and passes each
line's own rest wavelength to `salim_attenuation` / `cardelli_extinction`,
so `A(λ)` is evaluated at the correct wavelength and the correction factor
`10^(0.4·A(λ))` is wavelength-dependent as required. No change needed.

## Files touched

| File | Change |
|---|---|
| `src/jwspecabund/dust.py` | Replace `_BALMER_DECREMENT_LINES` with `_BALMER_LADDER`; add `anchor` arg to `compute_Av_multi_balmer`. |
| `src/jwspecabund/_core.py` | Add `balmer_anchor` kwarg to `compute_abundances`; pass through; update log/diagnostic strings. |
| `tests/test_dust.py` (or nearest existing test file) | Add Hα-anchor recovery test. |

## Backward compatibility

All existing call sites use the default `anchor="HBETA"` / `balmer_anchor="HBETA"`
and the result is numerically unchanged at the level of the weighted-mean A_V
recovered. One cosmetic difference: the hard-coded `hb_wave = 4864.04` inside
`compute_Av_multi_balmer` is replaced by the table's Hβ entry `4862.68` Å
(vacuum, consistent with the other ladder wavelengths). This shifts the Hβ
attenuation curve evaluation by ~0.03% of the wavelength — far below any
measurement precision — so it has no meaningful numerical effect but is worth
calling out.

The internal rename from `_BALMER_DECREMENT_LINES` to `_BALMER_LADDER` is a
private symbol and affects no public API.
