# Plan — Decouple O²⁺ density from `ne_high` (N IV])

## Goal
Stop using the N IV]-derived high-ionisation density (`ne_high`) for O²⁺.
Feed O²⁺ the **intermediate-zone** density instead:
`ne_OIII = ne_mid (CIII]) if available, else ne_low`.
Keep N³⁺ (N IV], `ne_high`) and N²⁺ (N III], `ne_mid`) on their correct
zone densities so the ICF 5 calculation **(N²⁺+N³⁺)/O²⁺** uses the right
density for each nitrogen ion.

## Why (from the deep dive)
- O²⁺/H⁺ (and Tₑ(OIII)) are density-insensitive below ~10⁴ cm⁻³ but run
  away above it (Δ up to +0.8 dex by 5×10⁵).
- N IV] samples 47–77 eV, **above** the O²⁺ zone (35–55 eV); CIII] (24–48 eV)
  overlaps O²⁺'s lower half — a strictly better, less noisy density proxy.
- `ne_high` enters O²⁺ **twice**: the Tₑ(OIII) solve *and* the 5007/Hβ
  abundance. The Tₑ path dominates → **both** must move to `ne_OIII`, or the
  fix leaks through temperature.

## Scope decisions
- **One** O²⁺/H⁺ value at `ne_OIII`, used everywhere (reported, O/H total,
  and as the O²⁺ denominator in N/O ICF 5 and C/O). N ions keep correct ne.
- Tₑ(OIII) solved at `ne_OIII` → Tₑ_high becomes the correctly-density-solved
  [OIII] temperature. It is still the shared high-zone Tₑ applied to N³⁺/N²⁺/
  C³⁺/Ne²⁺ (unchanged densities). Net effect on N ions: negligible in the
  normal regime, protective in the spike regime.
- **Out of scope** (flagged, not changed): Ne²⁺ has the same issue but the
  user only asked for O²⁺; the Martinez ICF density (`ne=ne_high`) and logU
  (N43/O32 at `ne_high`) stay — correct for the high-ion N correction.

## Changes

### 1. `src/jwspecabund/direct.py` — `compute_ionic_abundances`
- [ ] O²⁺/H⁺ line: change density `ne_hi` → `ne_md` (which already = `ne_mid
      if not None else ne_low`). One-line change; propagates O²⁺ to **every**
      caller (point, MC, posterior, alt-1666, secondary path).
- [ ] Update the inline comment and the docstring `ne_high`/`ne_md` "Used
      for…" lists (O²⁺ moves from high → intermediate zone).
- [ ] N³⁺, C³⁺, Ne²⁺, N⁴⁺ stay on `ne_hi` (verify untouched).

### 2. `src/jwspecabund/_core.py` — Tₑ(OIII) solve density
Define a local `ne_OIII = ne_mid if ne_mid is not None else ne_low` and pass
it (instead of `ne_high`) to every `compute_Te_OIII` / `compute_Te_OIII_1666`
call. Sites:
- [ ] `_run_direct`: point estimate (~1192, 1196), MC loop (~1313, 1320),
      alt-1666 cross-check (~1490, 1511). Define `ne_OIII` after the
      `ne_mid_override`/`ne_high_override` block (~1180).
- [ ] `_run_direct_mcmc`: point (~1750, 1758), per-sample (~1837, 1844),
      alt-1666 (~2017, 2034). Define `ne_OIII` after override (~1743).
- [ ] Secondary 1666 path (~2765): use `ne_mid or ne_low` from
      `primary_result`.
- [ ] Leave all `_compute_logU(..., ne_high)` and
      `compute_total_abundances(..., ne=ne_high)` calls unchanged.

### 3. `src/jwspecabund/_core.py` — `_compute_ionic_upper_limits`
- [ ] O²⁺ row (~867) density `ne_high` → `ne_mid if ne_mid is not None else
      ne_low` (consistency for the rare non-detection case).

### 4. `src/jwspecabund/_core.py` — `_build_diagnostics`
- [ ] Tₑ description (~995): report the O²⁺-zone density actually used
      (`ne_OIII`), not `ne_high`.
- [ ] Add a short note that O²⁺/H⁺ uses the intermediate-zone density
      (CIII] → low fallback), decoupled from N IV], with the rationale.

## Tests (`tests/test_abundances.py` or new `tests/test_ne_oiii_zone.py`)
- [ ] Unit: `compute_ionic_abundances` with distinct `ne` (low), `ne_mid`,
      `ne_high` → O²⁺/H⁺ matches an independent `_ionic_abundance` at `ne_mid`
      (NOT `ne_high`); N³⁺/H⁺ matches at `ne_high` (regression guard).
- [ ] Unit: `ne_mid=None` → O²⁺ uses `ne_low`.
- [ ] Integration: flux dict where N IV] implies high `ne_high` and CIII]
      implies low `ne_mid`; run `_run_direct`; assert reported Tₑ_high and
      O²⁺/H⁺ equal the `ne_OIII`-based computation, and N³⁺/H⁺ unchanged.
- [ ] Regression: existing suite (110 abundance tests) still passes — they
      pass `ne` uniformly so O²⁺ value is unchanged for them.

## Verification
- [ ] `pytest tests/test_abundances.py tests/test_resample_count.py
      tests/test_martinez25_reject.py tests/test_dust_invariance.py -q`
- [ ] `ruff check` on changed files.
- [ ] Re-run the deep-dive sanity case: a high `ne_high` no longer shifts
      O²⁺/H⁺ or Tₑ_high.

## Commit
- [ ] Single logical commit once tests pass:
      "Use intermediate-zone density (CIII]→low) for O²⁺ instead of N IV]"
- [ ] (No Co-Authored-By trailer.)

## Review section

**Done (option a — density only).**

- `direct.py / compute_ionic_abundances`: O²⁺ row now uses `ne_md`
  (CIII]→low) instead of `ne_hi`. One-line behavioural change that
  propagates O²⁺ to every caller (point, MC, posterior, alt-1666,
  secondary). Docstring + comment updated.
- `_core.py`: added a local `ne_OIII = ne_mid if ne_mid is not None
  else ne_low` in `_run_direct` and `_run_direct_mcmc`, and routed it
  into all 12 Tₑ(OIII) solve sites (point, MC, per-sample, both
  alt-1666 cross-checks). Secondary 1666 path uses
  `primary_result.ne_mid or ne_low`.
- `_compute_ionic_upper_limits`: O²⁺ row `ne_high` → `ne_mid`.
- `_build_diagnostics`: Tₑ(high) description now reports the O²⁺-zone
  density; added an `O++/H+ density` diagnostic explaining the
  decoupling.
- Unchanged (as planned): N³⁺/C³⁺/Ne²⁺ on `ne_high`; N²⁺/C²⁺ on
  `ne_mid`; logU (`ne_high`) and Martinez ICF density (`ne=ne_high`).

**Verification.**
- New `tests/test_ne_oiii_zone.py` (4 tests): O²⁺ tracks `ne_mid`, falls
  back to `ne_low` when no CIII]; N³⁺ still tracks `ne_high`; N²⁺ tracks
  `ne_mid`. All pass.
- Full regression: 168 abundance-related tests pass.
- Integration: a spiked `ne_high = 51,015 cm⁻³` no longer drags Tₑ — it
  is solved at the CIII] density (304 cm⁻³); diagnostics report it
  correctly.
- `ruff`: no new errors in changed regions (repo has pre-existing
  E501/F401/I001 unrelated to this change).
