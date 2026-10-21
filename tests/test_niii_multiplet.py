"""N III] 1750 quintuplet: fitted as one blend, abundances from the sum.

The quintuplet's five members lie within 7 A of each other.  Fitting only
1748.6 + 1752.2 let the 1748.6 component soak up the 1749.7 flux (the
strongest member, ~half the total), while the abundance divided by the
emissivity of 1748.6 + 1752.2 alone (~37% of the multiplet at
ne >~ 1e4 cm^-3).  N2+ came out ~0.4 dex high.

Covers:
    1. Constraint templates pinned to PyNEB.
    2. ConstraintSet and the JAX tying ops: one free flux/velocity/width.
    3. End to end: a synthetic quintuplet recovers the injected N2+.
"""

from __future__ import annotations

import numpy as np
import pytest

from jwspecabund.direct import (
    NIII_MULTIPLET,
    _ionic_abundance,
    compute_ionic_abundances,
    niii_multiplet_flux,
)
from jwspecfit.constraints import (
    NIII_MULTIPLET_ANCHOR,
    NIII_MULTIPLET_RATIOS,
    ConstraintSet,
)
from jwspecfit.fitter import fit_lines
from jwspecfit.lines import REST_LINES_A

from .test_uv_doublets import _make_uv_spectrum

MEMBERS = list(NIII_MULTIPLET)
TE = 1.5e4


def niii_split(total: float) -> dict[str, float]:
    """Split a total N III] 1750 flux over the members as the fitter does."""
    w = {NIII_MULTIPLET_ANCHOR: 1.0, **NIII_MULTIPLET_RATIOS}
    norm = sum(w.values())
    return {n: total * w[n] / norm for n in MEMBERS}


def _pyneb_shares(Te: float, ne: float) -> dict[str, float]:
    """Fraction of the quintuplet emissivity in each member."""
    import pyneb as pn

    atom = pn.Atom("N", 3)
    eps = {n: atom.getEmissivity(Te, ne, wave=w) for n, w in NIII_MULTIPLET.items()}
    tot = sum(eps.values())
    return {n: e / tot for n, e in eps.items()}


# ===================================================================
#  1. Templates
# ===================================================================

class TestTemplate:

    def test_members_match_constraints(self):
        assert set(MEMBERS) == {NIII_MULTIPLET_ANCHOR, *NIII_MULTIPLET_RATIOS}

    def test_pyneb_labels_are_unique_transitions(self):
        """Each integer label maps to its own N III transition."""
        import pyneb as pn

        atom = pn.Atom("N", 3)
        trans = {tuple(atom.getTransition(w)) for w in NIII_MULTIPLET.values()}
        assert len(trans) == 5
        for name, w in NIII_MULTIPLET.items():
            lev_i, lev_j = atom.getTransition(w)
            assert atom.wave_Ang[lev_i - 1, lev_j - 1] == pytest.approx(
                REST_LINES_A[name], abs=0.01,
            )

    def test_ratios_match_pyneb_at_1e4(self):
        shares = _pyneb_shares(TE, 1e4)
        for name, ratio in NIII_MULTIPLET_RATIOS.items():
            assert ratio == pytest.approx(
                shares[name] / shares[NIII_MULTIPLET_ANCHOR], rel=0.005,
            )


# ===================================================================
#  2. Tying
# ===================================================================

class TestTying:

    def _cs(self, names):
        return ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                             tie_uv_doublets=True)

    def test_one_free_flux_velocity_width(self):
        cs = self._cs(MEMBERS)
        free = cs.free_mask()
        nL = len(MEMBERS)
        i_a = MEMBERS.index(NIII_MULTIPLET_ANCHOR)
        assert free.sum() == 3
        assert free[i_a] and free[nL + i_a] and free[2 * nL + i_a]

    def test_apply_ratios_and_kinematics(self):
        cs = self._cs(MEMBERS)
        nL = len(MEMBERS)
        i_a = MEMBERS.index(NIII_MULTIPLET_ANCHOR)
        p = np.zeros(3 * nL)
        p[i_a], p[nL + i_a], p[2 * nL + i_a] = 2.0, 7000.0, 3.0
        out = cs.apply(p)
        for name, ratio in NIII_MULTIPLET_RATIOS.items():
            i = MEMBERS.index(name)
            lr = REST_LINES_A[name] / REST_LINES_A[NIII_MULTIPLET_ANCHOR]
            assert out[i] == pytest.approx(2.0 * ratio)
            assert out[nL + i] == pytest.approx(7000.0 * lr)
            assert out[2 * nL + i] == pytest.approx(3.0 * lr)

    def test_members_follow_the_uv_width_anchor(self):
        """CIII] sets the UV width; the members inherit its final value."""
        names = ["CIII]_1907", "CIII]", *MEMBERS]
        cs = self._cs(names)
        nL = len(names)
        p = np.zeros(3 * nL)
        p[2 * nL] = 4.0                                # CIII]_1907 sigma
        p[2 * nL + names.index(NIII_MULTIPLET_ANCHOR)] = 99.0  # overwritten
        out = cs.apply(p)
        for name in MEMBERS:
            expect = 4.0 * REST_LINES_A[name] / REST_LINES_A["CIII]_1907"]
            assert out[2 * nL + names.index(name)] == pytest.approx(expect)

    def test_jax_ops_match_apply(self):
        """The JAX likelihood's tying ops reproduce ConstraintSet.apply."""
        from jwspecmcmc.jax_likelihood import _compile_tying_ops

        names = ["CIII]_1907", "CIII]", "NIV_1486", "NIV_1483", *MEMBERS]
        cs = self._cs(names)
        rng = np.random.default_rng(0)
        p = rng.uniform(1.0, 5.0, 3 * len(names))
        p[~cs.free_mask()] = 0.0
        q = p.copy()
        for d, s, r in _compile_tying_ops(cs):
            q[d] = q[s] * r
        np.testing.assert_allclose(q, cs.apply(p))


# ===================================================================
#  3. Abundance consumers
# ===================================================================

class TestFluxSum:

    def test_sums_present_members_and_errors_linearly(self):
        f = {"NIII_1750": 5.0, "NIII_1752": 3.0, "NIII_1747": 0.0}
        e = {"NIII_1750": 0.5, "NIII_1752": 0.3}
        flux, err, waves = niii_multiplet_flux(f, e)
        assert flux == 8.0
        assert err == pytest.approx(0.8)
        assert waves == [1750, 1752]

    def test_empty(self):
        assert niii_multiplet_flux({}) == (0.0, 0.0, [])


@pytest.mark.parametrize(("ne_true", "template_bias"), [
    (1e3, 0.9695),   # -0.013 dex
    (1e4, 1.0),
    (1e5, 1.0049),   # +0.002 dex
])
def test_synthetic_quintuplet_recovers_npp(ne_true, template_bias):
    """Inject a quintuplet at known N2+/H+ and recover it through the fit.

    The fit template is fixed at ne = 1e4; injecting at 1e3 and 1e5 pins
    the template systematic on the total flux (and so on N2+), measured
    noise-free.  At 1e2 it is -0.033 dex.
    """
    z, Hb, npp_true = 3.0, 1.0e-16, 1.0e-5
    # Flux of the whole quintuplet that gives npp_true at (Te, ne_true).
    f_tot = npp_true / _ionic_abundance(
        "N", 3, 1.0, Hb, TE, ne_true, list(NIII_MULTIPLET.values()),
    )
    shares = _pyneb_shares(TE, ne_true)
    spec = _make_uv_spectrum(
        z=z, lines_to_inject={n: f_tot * shares[n] for n in MEMBERS},
        R=1000.0, noise_level=0.002,
    )
    res = fit_lines(spec, z=z, R=1000.0, lines=MEMBERS, n_boot=0)
    assert res.success

    fluxes = {n: res.lines[n].flux for n in MEMBERS}
    assert sum(fluxes.values()) / f_tot == pytest.approx(template_bias, abs=0.005)

    ionic = compute_ionic_abundances(
        {"HBETA": Hb, **fluxes}, Te_high=TE, Te_low=TE, ne=ne_true,
        ne_mid=ne_true, Te_int=TE,
    )
    assert ionic["N++/H+"] / npp_true == pytest.approx(template_bias, abs=0.005)
