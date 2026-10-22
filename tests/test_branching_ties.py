"""Fixed branching ratios: [O III] 4959, O III] 1661, [Ne III] 3968, [O I] 6366, H8.

Each secondary decays from the same upper level as its primary, so its flux
ratio is set by the A-values alone; the fit derives its flux, velocity and
width from the primary.  H8 is tied to H9 by the Case B ratio, because it is
0.4 A from He I 3889 and their split is otherwise degenerate.

Covers:
    1. Constants and vacuum wavelengths pinned to PyNEB.
    2. ConstraintSet and the JAX tying ops: secondaries fully derived.
    3. End to end: fitting [Ne III] 3968 stops Hepsilon absorbing its flux.
"""

from __future__ import annotations

import numpy as np
import pytest

from jwspecfit.constraints import BRANCHING_RATIOS, H8_H9_RATIO, ConstraintSet
from jwspecfit.fitter import fit_lines
from jwspecfit.lines import REST_LINES_A

from .test_uv_doublets import _make_uv_spectrum

# Mirrors all_stack_fit.ipynb: UV centroids and widths left free.
LINES = [
    "OIII_1661", "OIII_1666", "H9", "NeIII_3869", "H8", "HEI_3889", "NeIII_3968",
    "HEPSILON", "HBETA", "OIII_4959", "OIII_5007", "OI_6302", "OI_6366",
]


def _air_to_vac(lam_air: float) -> float:
    """Air to vacuum wavelength (Morton 2000), Angstrom."""
    s2 = (1e4 / lam_air) ** 2
    return lam_air * (1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2))


def _cs(**kw) -> ConstraintSet:
    return ConstraintSet(LINES, tie_uv_centroids=False, tie_uv_widths=False, **kw)


# ===================================================================
#  1. Constants
# ===================================================================

@pytest.mark.parametrize(("el", "ion", "pri", "sec", "names"), [
    ("O", 3, 5007, 4959, ("OIII_5007", "OIII_4959")),
    ("Ne", 3, 3869, 3968, ("NeIII_3869", "NeIII_3968")),
    ("O", 1, 6300, 6364, ("OI_6302", "OI_6366")),
])
def test_ratio_matches_pyneb_at_any_te_ne(el, ion, pri, sec, names):
    import pyneb as pn

    atom = pn.Atom(el, ion)
    for te in (1e4, 2e4):
        for ne in (1e2, 1e5):
            r = atom.getEmissivity(te, ne, wave=sec) / atom.getEmissivity(te, ne, wave=pri)
            assert BRANCHING_RATIOS[names] == pytest.approx(r, rel=1e-3)


@pytest.mark.parametrize(("el", "ion", "label", "name"), [
    ("Ne", 3, 3968, "NeIII_3968"),
    ("O", 1, 6364, "OI_6366"),
])
def test_vacuum_wavelengths(el, ion, label, name):
    import pyneb as pn

    atom = pn.Atom(el, ion)
    lev_i, lev_j = atom.getTransition(label)
    lam_air = atom.wave_Ang[lev_i - 1, lev_j - 1]
    assert REST_LINES_A[name] == pytest.approx(_air_to_vac(lam_air), abs=0.01)


def test_h8_h9_case_b():
    import pyneb as pn

    h = pn.RecAtom("H", 1)
    for te in (1e4, 2e4):
        r = h.getEmissivity(te, 1e3, label="8_2") / h.getEmissivity(te, 1e3, label="9_2")
        assert H8_H9_RATIO == pytest.approx(r, rel=2e-3)


# ===================================================================
#  2. Constraints
# ===================================================================

class TestConstraints:

    def test_secondaries_fully_derived(self):
        cs, nL = _cs(), len(LINES)
        free = cs.free_mask()
        for _, sec in BRANCHING_RATIOS:
            i = LINES.index(sec)
            assert not free[i] and not free[nL + i] and not free[2 * nL + i]
        assert not free[LINES.index("H8")]
        assert free[LINES.index("HEI_3889")]          # He I stays free

    def test_apply_ratios_and_kinematics(self):
        cs, nL = _cs(), len(LINES)
        rng = np.random.default_rng(1)
        out = cs.apply(rng.uniform(0.5, 2.0, 3 * nL))
        for (pri, sec), ratio in BRANCHING_RATIOS.items():
            i, j = LINES.index(pri), LINES.index(sec)
            lam = REST_LINES_A[sec] / REST_LINES_A[pri]
            assert out[j] == pytest.approx(out[i] * ratio)
            assert out[nL + j] == pytest.approx(out[nL + i] * lam)
            assert out[2 * nL + j] == pytest.approx(out[2 * nL + i] * lam)
        assert out[LINES.index("H8")] == pytest.approx(out[LINES.index("H9")] * H8_H9_RATIO)

    def test_switch_off_leaves_them_free(self):
        free = _cs(fix_branching_ratios=False).free_mask()
        assert free[LINES.index("OIII_4959")] and free[LINES.index("H8")]

    def test_jax_ops_match_apply(self):
        from jwspecmcmc.jax_likelihood import _compile_tying_ops

        cs, nL = _cs(), len(LINES)
        p = np.random.default_rng(2).uniform(0.5, 2.0, 3 * nL)
        p[~cs.free_mask()] = 0.0
        q = p.copy()
        for d, s, r in _compile_tying_ops(cs):
            q[d] = q[s] * r
        np.testing.assert_allclose(q, cs.apply(p))


# ===================================================================
#  3. End to end
# ===================================================================

def test_neiii_3968_no_longer_absorbed_by_hepsilon():
    """A blended Hepsilon + [Ne III] 3968 pair: Hepsilon is recovered only if 3968 is fitted."""
    z, f_he, f_ne = 3.0, 1.5e-17, 4.0e-17
    inject = {"NeIII_3869": f_ne, "NeIII_3968": f_ne / 3.320, "HEPSILON": f_he}
    spec = _make_uv_spectrum(z=z, lines_to_inject=inject, R=1000.0, noise_level=0.002)

    with_3968 = fit_lines(spec, z=z, R=1000.0, lines=list(inject), n_boot=0)
    without = fit_lines(spec, z=z, R=1000.0, lines=["NeIII_3869", "HEPSILON"], n_boot=0)
    assert with_3968.success and without.success
    assert with_3968.lines["HEPSILON"].flux / f_he == pytest.approx(1.0, abs=0.05)
    assert without.lines["HEPSILON"].flux / f_he > 1.5        # absorbs most of 3968
