"""Tests for the stand-alone ``dust_correct`` helper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import jwspecabund
import jwspecfit
from jwspecabund.dust import _BALMER_LADDER, cardelli_extinction, salim_attenuation
from jwspecfit.lines import REST_LINES_A


@dataclass
class _MockLine:
    name: str
    flux: float
    flux_err: float


@dataclass
class _MockResult:
    lines: dict


def _reddened_result(Av_true: float, law: str = "salim", extra=("OIII_5007",)):
    """Case B Balmer ladder plus extra lines (intrinsic flux 5), reddened."""
    fn = salim_attenuation if law == "salim" else cardelli_extinction
    intrinsic = {n: r for n, (_, r) in _BALMER_LADDER.items()}
    intrinsic.update({n: 5.0 for n in extra})
    lines = {}
    for name, f_int in intrinsic.items():
        A = fn(np.array([REST_LINES_A[name]]), Av_true)[0]
        f = f_int * 10 ** (-0.4 * A)
        lines[name] = _MockLine(name, f, 0.01 * f)
    return _MockResult(lines), intrinsic


@pytest.mark.parametrize("law", ["salim", "cardelli"])
@pytest.mark.parametrize("method", ["anchored", "joint"])
def test_recovers_intrinsic_fluxes(law, method):
    res, intrinsic = _reddened_result(1.2, law)
    dc = jwspecabund.dust_correct(res, dust_law=law, balmer_method=method)
    assert dc.Av == pytest.approx(1.2, abs=2e-3)
    for name, f_int in intrinsic.items():
        assert dc.fluxes[name] == pytest.approx(f_int, rel=5e-3)


def test_jwspecfit_alias_and_pair():
    res, intrinsic = _reddened_result(0.8)
    dc = jwspecfit.dust_correct(res, balmer_pair=("HALPHA", "HBETA"))
    assert dc.method == "pair" and dc.n_balmer == 1
    assert dc.fluxes["OIII_5007"] == pytest.approx(5.0, rel=1e-3)


def test_user_av_and_error_propagation():
    res, _ = _reddened_result(0.5)
    no_prop = jwspecabund.dust_correct(res, Av=0.5, Av_err=0.2,
                                       propagate_av_err=False)
    prop = jwspecabund.dust_correct(res, Av=0.5, Av_err=0.2)
    assert no_prop.method == "user"
    # Without A_V error, SNR is preserved exactly.
    for n in no_prop.fluxes:
        assert no_prop.fluxes[n] / no_prop.errors[n] == pytest.approx(100.0)
        assert prop.errors[n] > no_prop.errors[n]


def test_no_balmer_raises():
    res = _MockResult({"OIII_5007": _MockLine("OIII_5007", 1.0, 0.1)})
    with pytest.raises(ValueError, match="Balmer"):
        jwspecabund.dust_correct(res)


def test_auto_anchor_prefers_halpha():
    res, _ = _reddened_result(1.0)
    dc = jwspecabund.dust_correct(res)
    assert dc.balmer_info["anchor"] == "HALPHA"
    used = {r["line"] for r in dc.balmer_info["individual"]}
    assert "HBETA" in used


def test_auto_anchor_falls_back_to_hbeta():
    res, intrinsic = _reddened_result(1.0)
    # Hα below the SNR-5 default -> anchor on Hβ.
    ha = res.lines["HALPHA"]
    res.lines["HALPHA"] = _MockLine("HALPHA", ha.flux, ha.flux / 4.0)
    dc = jwspecabund.dust_correct(res)
    assert dc.balmer_info["anchor"] == "HBETA"
    assert dc.Av == pytest.approx(1.0, abs=2e-3)
    assert dc.fluxes["OIII_5007"] == pytest.approx(5.0, rel=5e-3)


def test_snr_balmer_default_is_5():
    res, _ = _reddened_result(1.0)
    hg = res.lines["HGAMMA"]
    res.lines["HGAMMA"] = _MockLine("HGAMMA", hg.flux, hg.flux / 4.0)  # SNR 4
    dc = jwspecabund.dust_correct(res)
    assert "HGAMMA" not in {r["line"] for r in dc.balmer_info["individual"]}


def test_summary_reports_lines_ratios_and_rejections(capsys):
    res, _ = _reddened_result(1.0)
    hd = res.lines["HDELTA"]
    res.lines["HDELTA"] = _MockLine("HDELTA", hd.flux, hd.flux / 3.0)  # SNR 3
    dc = jwspecabund.dust_correct(res)
    out = capsys.readouterr().out
    assert out.strip() == dc.summary()
    assert "anchored on HALPHA" in out
    assert "HBETA/HALPHA" in out and "Case B 0.3497" in out
    assert dc.balmer_rejected == {"HDELTA": "SNR 3.0 < 5"}


def test_verbose_false_is_silent(capsys):
    res, _ = _reddened_result(1.0)
    jwspecabund.dust_correct(res, verbose=False)
    assert capsys.readouterr().out == ""


def test_low_snr_hbeta_anchor_rejected_with_reasons():
    # Hβ at SNR 2 used to anchor the decrement despite snr_balmer=5.
    res = _MockResult({
        "HBETA": _MockLine("HBETA", 1.0, 0.5),
        "HGAMMA": _MockLine("HGAMMA", 0.4, 0.01),
    })
    with pytest.raises(ValueError, match=r"HBETA: SNR 2\.0 < 5"):
        jwspecabund.dust_correct(res, verbose=False)
