"""The rest-UV oxygen tracer for N/O (Martinez+25 ICFs 2, 3 and 5).

The Martinez ICFs are functions of an ionic ratio, log(U), Z and n_e; they do
not care which line measured O²⁺.  So the three whose denominator is O²⁺ alone
can equally divide by O²⁺ from O III] 1661,1666 instead of [O III] 5007:

    ICF 2   N²⁺/O²⁺           NIII] 1749,1752 / O III] 1661,1666
    ICF 3   N³⁺/O²⁺           NIV] 1483,1486  / O III] 1661,1666
    ICF 5   (N²⁺+N³⁺)/O²⁺     both, the tier Martinez prefer

ICFs 1 and 4 need O⁺, whose only tracer is the optical [O II] 3727, so they
stay optical.  The default stays optical everywhere; ``no_oxygen_tracer``
selects the tracer, and ``"auto"`` reproduces the C/O A_V switch.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from jwspecabund.direct import (
    NO_UV_AV_THRESHOLD,
    _get_oiii_atom,
    _no_oxygen_tracer_order,
    compute_N_Opp_uv,
    compute_total_abundances,
    oiii_uv_branching_ratio,
)
from jwspecabund.martinez25_icf import (
    LOG_OH_SOLAR,
    VALID_NO_UV_ICF_TIERS,
    compute_NO_martinez25_uv,
    compute_NO_martinez25_uv_locked,
    icf_NppNppp_Opp,
    icf_NppOpp,
    icf_NpppOpp,
)

TE = 1.5e4
NE = 1e3


def _Z(oh):
    return 10 ** (oh - LOG_OH_SOLAR)


def _emissivities(Te=TE, ne=NE):
    """(eps_O_uv, eps_1749, eps_1752, eps_1483, eps_1486) at (Te, ne)."""
    import pyneb as pn

    o3 = _get_oiii_atom()
    n3, n4 = pn.Atom("N", 3), pn.Atom("N", 4)
    eps_O = (o3.getEmissivity(Te, ne, lev_i=6, lev_j=2)
             + o3.getEmissivity(Te, ne, lev_i=6, lev_j=3))
    return (eps_O,
            n3.getEmissivity(Te, ne, wave=1749),
            n3.getEmissivity(Te, ne, wave=1752),
            n4.getEmissivity(Te, ne, wave=1483),
            n4.getEmissivity(Te, ne, wave=1486))


def _uv_fluxes(npp=0.20, nppp=0.05, Te=TE, ne=NE, o_total=1.0):
    """Fluxes that encode exactly these two ionic ratios at (Te, ne)."""
    eps_O, e49, e52, e83, e86 = _emissivities(Te, ne)
    b = oiii_uv_branching_ratio()
    n3_sum = npp * o_total * (e49 + e52) / eps_O
    n4_sum = nppp * o_total * (e83 + e86) / eps_O
    return dict(
        flux_1666=o_total / (1 + b), flux_1661=o_total * b / (1 + b),
        flux_1749=n3_sum * e49 / (e49 + e52),
        flux_1752=n3_sum * e52 / (e49 + e52),
        flux_1483=n4_sum * e83 / (e83 + e86),
        flux_1486=n4_sum * e86 / (e83 + e86),
    )


# =========================================================================
# 1. The tracer function itself
# =========================================================================

class TestComputeNOppUv:

    @pytest.mark.parametrize("npp, nppp", [(0.2, 0.05), (0.02, 0.4), (1.0, 1.0)])
    def test_round_trip(self, npp, nppp):
        ratios, note = compute_N_Opp_uv(Te_O=TE, ne_O=NE,
                                        **_uv_fluxes(npp, nppp))
        assert ratios["N++/O++"] == pytest.approx(npp, rel=1e-9)
        assert ratios["N+++/O++"] == pytest.approx(nppp, rel=1e-9)
        assert "measured" in note

    def test_single_oiii_component_uses_branching_ratio(self):
        """λ1666 or λ1661 alone must reproduce the doublet answer."""
        f = _uv_fluxes()
        both = compute_N_Opp_uv(Te_O=TE, ne_O=NE, **f)[0]
        for drop, kept in (("flux_1661", "1666"), ("flux_1666", "1661")):
            one = dict(f)
            one[drop] = 0.0
            ratios, note = compute_N_Opp_uv(Te_O=TE, ne_O=NE, **one)
            assert ratios["N++/O++"] == pytest.approx(both["N++/O++"], rel=1e-9)
            assert kept in note

    def test_single_nitrogen_member_uses_its_own_emissivity(self):
        """A doublet member dropped by the SNR filter must not halve N²⁺."""
        f = _uv_fluxes()
        both = compute_N_Opp_uv(Te_O=TE, ne_O=NE, **f)[0]["N++/O++"]
        only = dict(f, flux_1752=0.0)
        ratios, note = compute_N_Opp_uv(Te_O=TE, ne_O=NE, **only)
        assert ratios["N++/O++"] == pytest.approx(both, rel=1e-9)
        assert "1749 alone" in note

    def test_only_one_nitrogen_ion_gives_only_that_key(self):
        f = _uv_fluxes()
        no_n4 = dict(f, flux_1483=0.0, flux_1486=0.0)
        ratios, _ = compute_N_Opp_uv(Te_O=TE, ne_O=NE, **no_n4)
        assert set(ratios) == {"N++/O++"}

    def test_nitrogen_zone_is_separable(self):
        """Te_Npp must move the answer and be recorded in the note."""
        f = _uv_fluxes()
        same = compute_N_Opp_uv(Te_O=TE, ne_O=NE, **f)[0]["N++/O++"]
        ratios, note = compute_N_Opp_uv(Te_O=TE, ne_O=NE, Te_Npp=1.2e4, **f)
        assert "NIII] at Te=12000" in note
        assert abs(np.log10(ratios["N++/O++"] / same)) > 0.02

    @pytest.mark.parametrize("missing, match", [
        (dict(flux_1661=0.0, flux_1666=0.0), "O III]"),
        (dict(flux_1749=0.0, flux_1752=0.0, flux_1483=0.0, flux_1486=0.0),
         "NIII]"),
    ])
    def test_raises_without_a_needed_ion(self, missing, match):
        with pytest.raises(ValueError, match=match):
            compute_N_Opp_uv(Te_O=TE, ne_O=NE, **dict(_uv_fluxes(), **missing))

    def test_barely_moves_with_temperature(self):
        """The point of the UV pairing: λ1666 and λ1750 share an excitation
        energy, so the emissivity factor nearly cancels.  Against λ5007 it
        does not."""
        import pyneb as pn

        f = _uv_fluxes()
        n3 = pn.Atom("N", 3)
        o3 = _get_oiii_atom()
        f_N = f["flux_1749"] + f["flux_1752"]
        uv, opt = [], []
        for Te in (1.0e4, 2.5e4):
            uv.append(compute_N_Opp_uv(Te_O=Te, ne_O=NE, **f)[0]["N++/O++"])
            # Same construction against [O III] 5007, at one fixed flux.
            eps_N = (n3.getEmissivity(Te, NE, wave=1749)
                     + n3.getEmissivity(Te, NE, wave=1752))
            opt.append((f_N / 1.0) * o3.getEmissivity(Te, NE, wave=5007) / eps_N)
        assert abs(np.log10(uv[1] / uv[0])) < 0.15
        assert abs(np.log10(opt[1] / opt[0])) > 1.0

    def test_nearly_reddening_free(self):
        """Dust-correcting with the wrong A_V barely moves the UV ratio."""
        from jwspecabund.dust import cardelli_extinction

        waves = np.array([1663.5, 1750.5])
        A = cardelli_extinction(waves, Av=1.0)
        f = _uv_fluxes()
        red = dict(f)
        for k, w in (("flux_1661", 0), ("flux_1666", 0),
                     ("flux_1749", 1), ("flux_1752", 1)):
            red[k] = f[k] * 10 ** (-0.4 * A[w])
        drift = np.log10(
            compute_N_Opp_uv(Te_O=TE, ne_O=NE, **red)[0]["N++/O++"]
            / compute_N_Opp_uv(Te_O=TE, ne_O=NE, **f)[0]["N++/O++"]
        )
        assert abs(drift) < 0.05


# =========================================================================
# 2. Tracer ordering
# =========================================================================

class TestTracerOrder:

    def test_optical_default_never_reaches_uv(self):
        assert _no_oxygen_tracer_order("optical", True) == (["optical"], None)

    def test_uv_prefers_uv_but_keeps_a_fallback(self):
        assert _no_oxygen_tracer_order("uv", True) == (["uv", "optical"], None)

    def test_auto_below_threshold_keeps_optical_first(self):
        order, reason = _no_oxygen_tracer_order(
            "auto", True, Av=NO_UV_AV_THRESHOLD - 0.5)
        assert order == ["optical", "uv"]
        assert reason is None

    def test_auto_above_threshold_promotes_uv(self):
        order, reason = _no_oxygen_tracer_order(
            "auto", True, Av=NO_UV_AV_THRESHOLD + 0.5)
        assert order == ["uv", "optical"]
        assert "A_V" in reason

    def test_threshold_is_strictly_greater(self):
        order, _ = _no_oxygen_tracer_order("auto", True, Av=NO_UV_AV_THRESHOLD)
        assert order == ["optical", "uv"]

    def test_custom_threshold_is_honoured(self):
        order, _ = _no_oxygen_tracer_order("auto", True, Av=0.6,
                                           Av_threshold=0.5)
        assert order == ["uv", "optical"]

    def test_unavailable_uv_is_dropped(self):
        assert _no_oxygen_tracer_order("uv", False) == (["optical"], None)
        assert _no_oxygen_tracer_order("auto", False, Av=5.0) == (["optical"],
                                                                  None)

    def test_lock_restricts_and_clears_a_moot_reason(self):
        order, reason = _no_oxygen_tracer_order("auto", True, Av=5.0,
                                                locked="optical")
        assert order == ["optical"]
        assert reason is None

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="no_oxygen_tracer"):
            _no_oxygen_tracer_order("5007", True)


# =========================================================================
# 3. ICF selection on UV ratios
# =========================================================================

class TestUvIcfSelection:
    LOGU, Z, NE_ICF = -2.5, _Z(8.0), 1e3

    def test_valid_tiers_are_the_three_Opp_only_icfs(self):
        assert set(VALID_NO_UV_ICF_TIERS) == {"NppOpp", "NpppOpp",
                                             "NppNppp_Opp"}

    def test_icf5_preferred_when_both_ions_present(self):
        val, name = compute_NO_martinez25_uv(
            {"N++/O++": 0.2, "N+++/O++": 0.05}, self.LOGU, self.Z, self.NE_ICF)
        assert name == "NppNppp_Opp"
        assert val == pytest.approx(
            0.25 * icf_NppNppp_Opp(self.LOGU, self.Z, self.NE_ICF))

    def test_icf2_when_only_Npp(self):
        val, name = compute_NO_martinez25_uv({"N++/O++": 0.2}, self.LOGU,
                                             self.Z, self.NE_ICF)
        assert name == "NppOpp"
        assert val == pytest.approx(
            0.2 * icf_NppOpp(self.LOGU, self.Z, self.NE_ICF))

    def test_icf3_when_only_Nppp(self):
        val, name = compute_NO_martinez25_uv({"N+++/O++": 0.05}, self.LOGU,
                                             self.Z, self.NE_ICF)
        assert name == "NpppOpp"
        assert val == pytest.approx(
            0.05 * icf_NpppOpp(self.LOGU, self.Z, self.NE_ICF))

    def test_nothing_usable(self):
        assert compute_NO_martinez25_uv({}, self.LOGU, self.Z) == (None, None)

    def test_locked_tier_does_not_cascade(self):
        ratios = {"N++/O++": 0.2, "N+++/O++": 0.05}
        val = compute_NO_martinez25_uv_locked(ratios, self.LOGU, self.Z,
                                              self.NE_ICF, "NppOpp")
        assert val == pytest.approx(
            0.2 * icf_NppOpp(self.LOGU, self.Z, self.NE_ICF))

    def test_locked_tier_unformable_returns_none(self):
        assert compute_NO_martinez25_uv_locked(
            {"N++/O++": 0.2}, self.LOGU, self.Z, self.NE_ICF,
            "NppNppp_Opp") is None

    @pytest.mark.parametrize("tier", ["NpOp", "NpNpp_OpOpp"])
    def test_optical_only_tiers_are_rejected(self, tier):
        with pytest.raises(ValueError, match="rest-UV"):
            compute_NO_martinez25_uv_locked({"N++/O++": 0.2}, self.LOGU,
                                            self.Z, self.NE_ICF, tier)


# =========================================================================
# 4. Tracer selection inside compute_total_abundances
# =========================================================================

IONIC = {"O+/H+": 2.0e-5, "O++/H+": 6.0e-5, "N+/H+": 1.0e-6,
         "N++/H+": 3.0e-6, "N+++/H+": 1.0e-6}
UV_RATIOS = {"N++/O++": 0.05, "N+++/O++": 0.01}
KW = dict(logU=-2.5, Z_Zsun=_Z(8.0), ne=1e3, icf_method="martinez25")


class TestTotalsTracerSelection:

    def test_default_is_optical_and_silent(self):
        t = compute_total_abundances(IONIC, no_uv_ratios=UV_RATIOS, **KW)
        assert t["NO_oxygen_tracer"] == "optical"
        assert "NO_tracer_reason" not in t

    def test_uv_is_adopted_when_asked_for(self):
        t = compute_total_abundances(IONIC, no_uv_ratios=UV_RATIOS,
                                     no_oxygen_tracer="uv", **KW)
        assert t["NO_oxygen_tracer"] == "uv"
        assert t["NO_icf_name"] == "NppNppp_Opp"
        assert t["N/O"] == pytest.approx(
            0.06 * icf_NppNppp_Opp(-2.5, _Z(8.0), 1e3))
        assert t["_icf_values"]["N/O"]["oxygen"] == "O III] 1661,1666"

    def test_uv_and_optical_differ(self):
        """Otherwise the rest of these tests prove nothing."""
        opt = compute_total_abundances(IONIC, **KW)["N/O"]
        uv = compute_total_abundances(IONIC, no_uv_ratios=UV_RATIOS,
                                      no_oxygen_tracer="uv", **KW)["N/O"]
        assert abs(np.log10(uv / opt)) > 0.01

    def test_uv_falls_back_with_a_recorded_reason(self):
        t = compute_total_abundances(IONIC, no_oxygen_tracer="uv", **KW)
        assert t["NO_oxygen_tracer"] == "optical"
        assert "O III] 1661/1666" in t["NO_tracer_reason"]

    def test_icf4_is_never_used_with_the_uv_tracer(self):
        """A UV O²⁺ summed with an optical O⁺ would restore the baseline."""
        t = compute_total_abundances(IONIC, no_uv_ratios={"N++/O++": 0.05},
                                     no_oxygen_tracer="uv", **KW)
        assert t["NO_oxygen_tracer"] == "uv"
        assert t["NO_icf_name"] == "NppOpp"

    def test_auto_switches_on_Av(self):
        hi = compute_total_abundances(
            IONIC, no_uv_ratios=UV_RATIOS, no_oxygen_tracer="auto",
            Av=NO_UV_AV_THRESHOLD + 0.5, **KW)
        lo = compute_total_abundances(
            IONIC, no_uv_ratios=UV_RATIOS, no_oxygen_tracer="auto",
            Av=NO_UV_AV_THRESHOLD - 0.5, **KW)
        assert hi["NO_oxygen_tracer"] == "uv"
        assert "A_V" in hi["NO_tracer_reason"]
        assert lo["NO_oxygen_tracer"] == "optical"
        assert "NO_tracer_reason" not in lo

    def test_auto_without_Av_keeps_optical(self):
        t = compute_total_abundances(IONIC, no_uv_ratios=UV_RATIOS,
                                     no_oxygen_tracer="auto", **KW)
        assert t["NO_oxygen_tracer"] == "optical"

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_unusable_ratios_are_never_adopted(self, bad):
        t = compute_total_abundances(
            IONIC, no_uv_ratios={"N++/O++": bad, "N+++/O++": bad},
            no_oxygen_tracer="uv", **KW)
        assert t["NO_oxygen_tracer"] == "optical"

    def test_lock_keeps_draws_on_the_point_estimate_tracer(self):
        t = compute_total_abundances(
            IONIC, no_uv_ratios=UV_RATIOS, no_oxygen_tracer="auto", Av=5.0,
            _lock_NO_tracer="optical", **KW)
        assert t["NO_oxygen_tracer"] == "optical"

    def test_locked_uv_that_cannot_form_fails_rather_than_substituting(self):
        t = compute_total_abundances(IONIC, no_oxygen_tracer="uv",
                                     _lock_NO_tracer="uv", **KW)
        assert "N/O" not in t
        assert "locked" in t["_failures"]["N/O"]


# =========================================================================
# 5. End to end through compute_abundances
# =========================================================================

@dataclass
class _Line:
    flux: float
    flux_err: float


class _MockFitResult:
    def __init__(self, fluxes, errors):
        self.lines = {k: _Line(fluxes[k], errors[k]) for k in fluxes}


def _uv_nitrogen_object(Te=2.0e4, ne=1000.0, O_pp=1.5e-5, O_p=1.8e-6,
                        N_pp=1.2e-6, N_ppp=3.0e-7, C_pp=4.0e-6, frac=0.04):
    """Synthetic object carrying both oxygen tracers and both UV N ions.

    CIII] is included so n_e(int) comes from the doublet rather than the
    z-fallback, which keeps the two oxygen tracers comparable.
    """
    import pyneb as pn

    o3 = _get_oiii_atom()
    o2, c3 = pn.Atom("O", 2), pn.Atom("C", 3)
    n3, n4 = pn.Atom("N", 3), pn.Atom("N", 4)
    jHb = pn.RecAtom("H", 1).getEmissivity(Te, ne, lev_i=4, lev_j=2)

    def _f(ion, emis):
        return 100.0 * ion * emis / jHb

    fluxes = {
        "HBETA": 100.0,
        "OII_3726": _f(O_p, o2.getEmissivity(Te, ne, wave=3726)),
        "OII_3729": _f(O_p, o2.getEmissivity(Te, ne, wave=3729)),
        "OIII_4363": _f(O_pp, o3.getEmissivity(Te, ne, wave=4363)),
        "OIII_4959": _f(O_pp, o3.getEmissivity(Te, ne, wave=4959)),
        "OIII_5007": _f(O_pp, o3.getEmissivity(Te, ne, wave=5007)),
        "OIII_1661": _f(O_pp, o3.getEmissivity(Te, ne, lev_i=6, lev_j=2)),
        "OIII_1666": _f(O_pp, o3.getEmissivity(Te, ne, lev_i=6, lev_j=3)),
        "NIII_1749": _f(N_pp, n3.getEmissivity(Te, ne, wave=1749)),
        "NIII_1752": _f(N_pp, n3.getEmissivity(Te, ne, wave=1752)),
        "NIV_1483": _f(N_ppp, n4.getEmissivity(Te, ne, wave=1483)),
        "NIV_1486": _f(N_ppp, n4.getEmissivity(Te, ne, wave=1486)),
        "CIII]_1907": _f(C_pp, c3.getEmissivity(Te, ne, wave=1907)),
        "CIII]": _f(C_pp, c3.getEmissivity(Te, ne, wave=1909)),
    }
    return _MockFitResult(fluxes, {k: frac * v for k, v in fluxes.items()})


class TestEndToEnd:
    """compute_abundances must carry the tracer choice through both runners."""

    @staticmethod
    def _run(**kw):
        from jwspecabund import compute_abundances

        kw.setdefault("dust_correct", False)
        return compute_abundances(
            _uv_nitrogen_object(), z=6.0, method="direct",
            n_mc=60, progress=False, **kw,
        )

    def test_default_uses_the_optical_tracer_silently(self):
        res = self._run()
        assert res.NO is not None and np.isfinite(res.NO)
        assert res.NO_oxygen_tracer == "optical"
        assert "N/O oxygen" not in res.diagnostics

    def test_uv_tracer_is_selectable_and_reported(self):
        res = self._run(no_oxygen_tracer="uv")
        assert res.NO_oxygen_tracer == "uv"
        assert "O III] 1661,1666" in res.diagnostics["N/O oxygen"]
        assert res.NO is not None and np.isfinite(res.NO)

    def test_both_tracers_agree_on_a_consistent_object(self):
        """Same ion, same ICF, fluxes built from one O²⁺ — they must agree."""
        assert abs(self._run().NO - self._run(no_oxygen_tracer="uv").NO) < 0.10

    def test_auto_switches_on_Av(self):
        assert self._run(no_oxygen_tracer="auto", Av=2.0,
                         dust_correct=True).NO_oxygen_tracer == "uv"
        assert self._run(no_oxygen_tracer="auto", Av=0.2,
                         dust_correct=True).NO_oxygen_tracer == "optical"

    def test_posterior_stays_on_one_tracer(self):
        """A mixture of two tracers would show up as an inflated spread."""
        res = self._run(no_oxygen_tracer="uv")
        post = np.asarray(res.NO_posterior, dtype=float)
        post = post[np.isfinite(post)]
        assert post.size > 10
        assert np.std(post) < 0.15

    def test_bad_tracer_name_raises(self):
        with pytest.raises(ValueError, match="no_oxygen_tracer"):
            self._run(no_oxygen_tracer="1666")
