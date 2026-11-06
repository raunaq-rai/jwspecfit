"""Balmer ratios at the measured T_e / n_e (referee #52, #74).

``compute_abundances(..., balmer_Te_ne=True)`` iterates A_V with Case B
Balmer ratios evaluated at the H+-zone conditions.  The default path
(fixed Case B at 10^4 K) must be unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pytest

from jwspecabund import balmer_intrinsic_ratios, compute_abundances
from jwspecabund.dust import (
    _BALMER_LADDER,
    compute_Av_balmer_pair,
    compute_Av_joint_balmer,
    compute_Av_multi_balmer,
    salim_attenuation,
)


@dataclass
class _Line:
    name: str
    flux: float
    flux_err: float
    snr: float


@dataclass
class _Fit:
    lines: dict


def _balmer_obs(Av, Te, ne, scale=100.0):
    """Balmer ladder at (Te, ne), reddened by *Av* (Salim default)."""
    ratios = balmer_intrinsic_ratios(Te, ne)
    out = {}
    for name, (wave, _) in _BALMER_LADDER.items():
        A = salim_attenuation(np.array([wave]), Av)[0]
        out[name] = scale * ratios[name] * 10 ** (-0.4 * A)
    return out


def _fit(fluxes, frac_err=0.02):
    return _Fit({n: _Line(n, f, frac_err * f, 1 / frac_err)
                 for n, f in fluxes.items()})


# -- intrinsic ratios ------------------------------------------------------

def test_ratios_reproduce_case_b_table():
    r = balmer_intrinsic_ratios(1.0e4, 1.0e2)
    for name, (_, ratio) in _BALMER_LADDER.items():
        assert r[name] == pytest.approx(ratio, rel=3e-3), name


def test_ratios_temperature_trend():
    cold, hot = balmer_intrinsic_ratios(1e4, 1e2), balmer_intrinsic_ratios(2.5e4, 1e2)
    assert hot["HALPHA"] < cold["HALPHA"]          # 2.86 -> 2.72
    assert hot["HGAMMA"] > cold["HGAMMA"]          # 0.468 -> 0.477


def test_ratios_clip_above_table(caplog):
    with caplog.at_level(logging.WARNING):
        r = balmer_intrinsic_ratios(4.0e4, 1e2)
    assert np.isfinite(r["HALPHA"])
    assert r == balmer_intrinsic_ratios(3.0e4, 1e2)
    assert "outside the Storey & Hummer" in caplog.text


# -- A_V estimators accept the ratios --------------------------------------

@pytest.mark.parametrize("estimator", ["anchored_hb", "anchored_ha", "joint", "pair"])
def test_estimators_recover_av_with_matching_ratios(estimator):
    Te, ne, Av_true = 2.0e4, 1.0e3, 0.5
    f = _balmer_obs(Av_true, Te, ne)
    e = {k: 0.01 * v for k, v in f.items()}
    intr = balmer_intrinsic_ratios(Te, ne)

    def run(intrinsic):
        if estimator == "joint":
            return compute_Av_joint_balmer(f, e, intrinsic=intrinsic)["Av"]
        if estimator == "pair":
            return compute_Av_balmer_pair(f, e, ("HALPHA", "HBETA"),
                                          intrinsic=intrinsic)["Av"]
        anchor = "HALPHA" if estimator == "anchored_ha" else "HBETA"
        return compute_Av_multi_balmer(f, e, anchor=anchor,
                                       intrinsic=intrinsic)["Av"]

    assert run(intr) == pytest.approx(Av_true, abs=2e-3)
    # 10^4 K ratios on a 2x10^4 K nebula underestimate A_V.
    assert run(None) < Av_true - 0.02


# -- compute_abundances ----------------------------------------------------

def _hot_nebula(Av=0.5, Te_balmer=2.0e4, ne_balmer=1.0e3):
    """Direct-method mock: [O III] 4363/5007 implies T_e ~ 2x10^4 K."""
    f = _balmer_obs(Av, Te_balmer, ne_balmer)

    def k(w):
        return 10 ** (-0.4 * salim_attenuation(np.array([w]), Av)[0])

    f.update({
        "OIII_5007": 600.0 * k(5008.2),
        "OIII_4959": 201.0 * k(4960.3),
        "OIII_4363": 22.0 * k(4364.4),
        "OII_3726": 15.0 * k(3727.1),
        "OII_3729": 20.0 * k(3729.9),
    })
    return _fit(f)


_KW = dict(z=6.0, method="direct", n_mc=5, progress=False)


def test_default_unchanged_by_new_options():
    res = _hot_nebula()
    a = compute_abundances(res, **_KW)
    b = compute_abundances(res, balmer_Te_ne=False, **_KW)
    c = compute_abundances(res, balmer_intrinsic=balmer_intrinsic_ratios(1e4, 1e2), **_KW)
    assert a.Av == b.Av
    assert c.Av == pytest.approx(a.Av, abs=3e-3)   # PyNeb vs rounded table
    assert a.balmer_Te is None and a.balmer_iterations is None


def test_iteration_converges_self_consistently():
    from jwspecabund._core import _hplus_zone_conditions

    res = _hot_nebula()
    base = compute_abundances(res, **_KW)
    it = compute_abundances(res, balmer_Te_ne=True, **_KW)

    assert it.balmer_converged
    assert len(it.balmer_iterations) >= 2
    assert it.balmer_iterations[0]["Te"] == 1.0e4
    # Hot gas -> larger intrinsic Hγ/Hβ -> larger A_V than at 10^4 K.
    assert it.Av > base.Av + 0.02
    # Fixed point: the ratios used match the conditions this run measured.
    Te, ne = _hplus_zone_conditions(it)
    assert it.balmer_Te == pytest.approx(Te, rel=0.02)
    # And A_V is exactly what the ratios at (balmer_Te, balmer_ne) give.
    redo = compute_abundances(
        res, balmer_intrinsic=balmer_intrinsic_ratios(it.balmer_Te, it.balmer_ne),
        **_KW,
    )
    assert redo.Av == pytest.approx(it.Av, abs=1e-12)
    assert "iterated" in it.summary()


def test_iteration_ignored_when_av_supplied(caplog):
    res = _hot_nebula()
    with caplog.at_level(logging.WARNING):
        out = compute_abundances(res, Av=0.3, balmer_Te_ne=True, **_KW)
    assert out.Av == 0.3 and out.balmer_Te is None
    assert "balmer_Te_ne ignored" in caplog.text
