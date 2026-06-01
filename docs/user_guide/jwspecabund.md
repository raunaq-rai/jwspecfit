# `jwspecabund` — chemical abundances

`jwspecabund` turns a `FitResult` (from `jwspecfit`) or an `MCMCResult`
(from `jwspecmcmc`) into element abundances: 12 + log(O/H), log(N/O),
log(C/O), log(S/O), log(Ne/O), log(Ar/O), with electron temperatures,
densities, and dust attenuation.

## Basic usage

```python
import jwspecabund

abund = jwspecabund.compute_abundances(result, z=6.0)
print(abund.summary())
```

## Pipeline

`compute_abundances` runs five stages in order:

### 1 — Dust correction

A_V is derived from the **multi-Balmer decrement**. The anchor is the
denominator line; every *other* detected Balmer line above `snr_balmer`
is ratioed against it and the per-line A_V values are combined as an
inverse-variance weighted mean. With the default anchor Hβ that means
**Hα/Hβ** (when Hα is in band), Hγ/Hβ, Hδ/Hβ, H9/Hβ, H10/Hβ — i.e. Hα
*is* included and, having the smallest error, usually dominates the
mean. You can instead anchor on Hα:

```python
abund = jwspecabund.compute_abundances(
    result, z=2.5,
    balmer_anchor="Ha",    # or "HBETA" (default)
    dust_law="salim",       # or "cardelli"
)
```

A per-line A_V is computed against the anchor, then an inverse-variance
weighted mean is returned. You can also supply `Av=` directly to skip
the derivation entirely, or `dust_correct=False` to turn dust off.

**Forcing a single Balmer pair.** To derive A_V from exactly one
decrement — and ignore lower-SNR Balmer lines — pass `balmer_pair`:

```python
abund = jwspecabund.compute_abundances(
    result, z=6.0,
    balmer_pair=("Ha", "HBETA"),   # A_V from Hα/Hβ only
    dust_law="cardelli",
)
print(f"A_V = {abund.Av:.3f} ± {abund.Av_err:.3f}")
```

`balmer_pair=(numerator, denominator)` accepts ladder names `"Ha"`,
`"HBETA"`, `"HGAMMA"`, `"HDELTA"`, `"H9"`, `"H10"`; the intrinsic Case-B
ratio is looked up automatically. It overrides `balmer_anchor` /
`snr_balmer` for the A_V step and is ignored when `Av=` is supplied.
A_V is still applied as a single point value (the abundance error is
**not** widened by the A_V uncertainty), but both `abund.Av` and
`abund.Av_err` (the formal error of the chosen pair) are reported. If
either line is missing or non-positive, A_V falls back to 0 with
`Av_err = nan` — check for `0.000 ± nan` when looping over objects. The
same single-pair calculation is available standalone as
{func}`~jwspecabund.compute_Av_balmer_pair`.

All line fluxes are then corrected at each line's own rest wavelength —
the correction is fully wavelength-dependent.

```{note}
**A_V is held constant across MCMC draws by default.** When the input is
an `MCMCResult`, the single A_V derived above is applied once to every
posterior draw (the Balmer-propagated A_V error is reported on the
result but *not* marginalised into the abundance posteriors). To
marginalise over A_V instead — each draw dust-corrected with its own
A_V sample — opt in by passing `Av_err=<value>` explicitly (and
optionally `Av_prior="gaussian"` or `"uniform"`).
```

### 2 — Electron density

Three-zone density from whatever ratios are available:

| Zone              | Diagnostic       | Fallback                             |
| ----------------- | ---------------- | ------------------------------------ |
| Low-ionisation    | [S II] 6718/6732 | [O II] 3726/3729, else 300 cm⁻³      |
| Mid-ionisation    | C III] 1907/1909 | —                                    |
| High-ionisation   | N IV] 1483/1486  | mid-zone, else low-zone              |

Each zone can be forced manually:

```python
abund = jwspecabund.compute_abundances(
    result, z=6.0,
    ne_low_override=500.0,
    ne_mid_override=2.0e4,
    ne_high_override=1.0e5,
)
```

### 3 — Method selection

| Method          | Triggered when                        | What it does                                                              |
| --------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| `"direct"`      | [O III] 4363 detected (SNR ≥ `snr_auroral`) | T_e from [O III] 4363 / (5007 + 4959); ionic abundances via PyNEB         |
| `"direct"` (UV) | 4363 absent, O III] 1666 detected     | T_e from 1666 / (5007 + 4959); larger 7.5 eV gap → more sensitive         |
| `"strong_line"` | No auroral line available             | Sanders+25 simultaneous polynomial across O3, O2, R23, O32                |
| `"forward"`     | Requested explicitly                  | Cullen+25 Bayesian forward model, sampling T_e, n_e, log U, ionic abundances |

### 4 — Ionic and total abundances

For the direct method, PyNEB computes O⁺/H⁺, O²⁺/H⁺, N⁺/H⁺, C⁺/H⁺,
C²⁺/H⁺, Ne²⁺/H⁺, S⁺/H⁺, S²⁺/H⁺, Ar²⁺/H⁺, Ar³⁺/H⁺. ICFs convert ionic to
total:

- **N/O** — Martinez+25 (multiple tiers auto-selected, override via `icf_tier=`)
- **C/O** — Garnett+97
- **S/O, Ne/O, Ar/O** — Izotov+06

### 5 — T_e-T_e relation

When only the high-ionisation T_e is measured, the low-ionisation T_e
comes from:

- `"desi"` (default) — DESI DR2 calibration
- `"classical"` — Garnett 1992

## Result fields

| Field                                        | Description                                       |
| -------------------------------------------- | ------------------------------------------------- |
| `OH`, `OH_err`                               | 12 + log(O/H) and uncertainty                     |
| `NO`, `CO`, `SO`, `NeO`, `ArO` (+`_err`)     | Element ratios                                    |
| `Te_high`, `Te_low` (+`_err`)                | Electron temperatures (K)                         |
| `ne`, `ne_low`, `ne_mid`, `ne_high`          | Electron densities (cm⁻³)                         |
| `Av`, `Av_err`                               | Dust attenuation                                  |
| `logU`, `logU_err`                           | Ionisation parameter                              |
| `ionic`                                      | Dict of ionic abundance ratios                    |
| `icf_method`, `NO_icf_name`                  | ICF scheme and tier used for N/O                  |
| `OH_posterior`, `NO_posterior`               | Full posterior arrays (MC / MCMC)                 |
| `method`                                     | `"direct"` / `"strong_line"` / `"forward"`        |

## Lyα escape fraction

When Lyα is present, the intrinsic flux is predicted from the
dust-corrected Balmer lines via Case B, and the escape fraction
`f_esc = F_obs(Lyα) / F_int(Lyα)` is returned with MC propagation of
A_V uncertainty:

```python
print(abund.lya_f_esc)       # median
print(abund.lya_f_esc_err)   # (lo, hi)
```

See {doc}`../abundance_methodology` for the full derivation.
