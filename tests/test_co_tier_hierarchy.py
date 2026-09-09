"""Tests for the five-tier C/O hierarchy in compute_total_abundances.

Tier order, best oxygen tracer and ionisation correction first::

    1. Martinez (in prep.) ICF x C2+/O2+, O2+ from [OIII] 5007,4959
    2. Martinez (in prep.) ICF x C2+/O2+, O2+ from O III] 1661,1666
    3. Direct sum (C+ + C2+ + C3+) / (O+ + O2+)
    4. Garnett+1997 ICF x (C2+ + C3+) / O2+
    5. Raw (C2+ + C3+) / O2+ (ICF = 1)

Tiers 1 and 2 differ only in how O2+ is counted.  Both count the *same
ion*, so both need the same C2+/O2+ -> C/O ionisation correction, and
that correction is nowhere near unity: log(ICF) runs from -0.55 at
log(U) = -3.5 to +0.13 at log(U) = -1.0, because C2+ (24.4-47.9 eV) and
O2+ (35.1-54.9 eV) span overlapping but distinct ionisation ranges.
Tier 2 therefore requires log(U) and Z exactly as tier 1 does.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from jwspecabund.direct import CO_UV_AV_THRESHOLD, compute_total_abundances
from jwspecabund.martinez25_icf import LOG_OH_SOLAR, icf_CppOpp


def _Z(log_oh: float) -> float:
    return 10.0 ** (log_oh - LOG_OH_SOLAR)


#: Both oxygen tracers present.
IONIC = {
    "O+/H+": 8.2e-6, "O++/H+": 4.5e-5,
    "C++/H+": 2.4e-5, "C+++/H+": 4.3e-6,
}
#: No optical O2+, so tier 1 cannot be formed.
IONIC_NO_OPTICAL_O = {"O+/H+": 8.2e-6, "C++/H+": 2.4e-5, "C+++/H+": 4.3e-6}

LOGU, Z = -2.5, _Z(8.0)
NE = 1e3
UV_RATIO = 0.61          # C2+/O2+ from C III] 1907,1909 / O III] 1661,1666
KW = dict(logU=LOGU, Z_Zsun=Z, ne=NE, co_icf_method="martinez25")


class TestTierSelection:
    def test_tier1_wins_when_both_tracers_available(self):
        t = compute_total_abundances(IONIC, co_uv_ratio=UV_RATIO, **KW)
        assert t["CO_method"] == "martinez25"
        assert "CO_tracer_reason" not in t

    def test_tier2_when_optical_oxygen_missing(self):
        t = compute_total_abundances(
            IONIC_NO_OPTICAL_O, co_uv_ratio=UV_RATIO, **KW,
        )
        assert t["CO_method"] == "martinez25_uv"
        assert t["C/O"] == pytest.approx(icf_CppOpp(LOGU, Z, NE) * UV_RATIO)

    def test_tier2_skipped_without_the_uv_ratio(self):
        """No O III] 1661,1666 measured -> tier 2 cannot be formed.

        With no optical O2+ either, that leaves no C/O route at all, which
        is the honest outcome rather than a silently downgraded one.
        """
        t = compute_total_abundances(IONIC_NO_OPTICAL_O, **KW)
        assert "CO_method" not in t
        assert "C/O" not in t

    def test_tier2_needs_logU_and_Z_like_tier1(self):
        """Without the ICF, tier 2 is skipped and the chain drops to tier 4.

        The UV ionic ratio on its own is not C/O: the correction spans
        0.68 dex across the calibrated log(U) range.
        """
        t = compute_total_abundances(
            IONIC, co_uv_ratio=UV_RATIO, co_icf_method="martinez25",
        )
        assert t["CO_method"] == "garnett97_icf"

    def test_legacy_chain_untouched_without_martinez(self):
        t = compute_total_abundances(IONIC, co_uv_ratio=UV_RATIO)
        assert t["CO_method"] == "garnett97_icf"

    def test_direct_sum_still_wins_over_garnett(self):
        ionic = dict(IONIC, **{"C+/H+": 1.1e-5})
        t = compute_total_abundances(ionic, co_uv_ratio=UV_RATIO)
        assert t["CO_method"] == "direct_sum"


class TestSharedICF:
    """Tiers 1 and 2 count the same ion, so they share one ICF."""

    def test_both_tiers_use_the_same_icf_value(self):
        t1 = compute_total_abundances(IONIC, co_uv_ratio=UV_RATIO, **KW)
        t2 = compute_total_abundances(
            IONIC_NO_OPTICAL_O, co_uv_ratio=UV_RATIO, **KW,
        )
        assert t1["CO_icf_value"] == pytest.approx(t2["CO_icf_value"])
        assert t1["CO_icf_value"] == pytest.approx(icf_CppOpp(LOGU, Z, NE))

    def test_icf_is_not_unity_and_is_actually_applied(self):
        t = compute_total_abundances(
            IONIC_NO_OPTICAL_O, co_uv_ratio=UV_RATIO, **KW,
        )
        icf = icf_CppOpp(LOGU, Z, NE)
        assert abs(np.log10(icf)) > 0.01           # a real correction
        assert t["C/O"] != pytest.approx(UV_RATIO)  # not the bare ratio

    @pytest.mark.parametrize("logU,floor", [(-3.5, 0.4), (-1.0, 0.09)])
    def test_correction_is_large_at_the_ends_of_the_logU_range(self, logU, floor):
        """Where the 'C2+/O2+ ≈ C/O' shortcut would go badly wrong."""
        assert abs(np.log10(icf_CppOpp(logU, Z, 1e2))) > floor

    def test_tier2_uses_the_intermediate_zone_logU(self):
        """C/O sits in the C2+ zone, so co_logU governs tier 2 as well."""
        t = compute_total_abundances(
            IONIC_NO_OPTICAL_O, logU=-3.2, Z_Zsun=Z, ne=6e4,
            co_icf_method="martinez25", co_logU=-2.5, co_ne=2.7e4,
            co_uv_ratio=UV_RATIO,
        )
        assert t["CO_method"] == "martinez25_uv"
        assert t["C/O"] == pytest.approx(icf_CppOpp(-2.5, Z, 2.7e4) * UV_RATIO)


class TestAvSwitch:
    """A_V promotes the near-reddening-free UV tracer above the optical one."""

    def test_high_Av_promotes_tier2(self):
        t = compute_total_abundances(
            IONIC, co_uv_ratio=UV_RATIO, Av=CO_UV_AV_THRESHOLD + 0.5, **KW,
        )
        assert t["CO_method"] == "martinez25_uv"
        assert "A_V" in t["CO_tracer_reason"]

    def test_low_Av_keeps_tier1(self):
        t = compute_total_abundances(
            IONIC, co_uv_ratio=UV_RATIO, Av=CO_UV_AV_THRESHOLD - 0.5, **KW,
        )
        assert t["CO_method"] == "martinez25"
        assert "CO_tracer_reason" not in t

    def test_threshold_is_strict(self):
        t = compute_total_abundances(
            IONIC, co_uv_ratio=UV_RATIO, Av=CO_UV_AV_THRESHOLD, **KW,
        )
        assert t["CO_method"] == "martinez25"

    def test_custom_threshold_honoured(self):
        t = compute_total_abundances(
            IONIC, co_uv_ratio=UV_RATIO, Av=0.6, co_uv_Av_threshold=0.5, **KW,
        )
        assert t["CO_method"] == "martinez25_uv"

    def test_high_Av_without_uv_ratio_stays_on_tier1(self):
        """Nothing to promote: the swap must not strand C/O."""
        t = compute_total_abundances(IONIC, Av=5.0, **KW)
        assert t["CO_method"] == "martinez25"

    def test_missing_Av_keeps_tier1(self):
        t = compute_total_abundances(IONIC, co_uv_ratio=UV_RATIO, Av=None, **KW)
        assert t["CO_method"] == "martinez25"


class TestTierLocking:
    """Draws stay on the tier the point estimate picked."""

    def test_lock_forces_the_named_tier(self):
        """Tier 1 is available, but the lock holds the draw on tier 2."""
        t = compute_total_abundances(
            IONIC, co_uv_ratio=UV_RATIO, _lock_CO_tier="martinez25_uv", **KW,
        )
        assert t["CO_method"] == "martinez25_uv"
        assert t["C/O"] == pytest.approx(icf_CppOpp(LOGU, Z, NE) * UV_RATIO)

    def test_lock_survives_an_Av_that_would_reorder_tiers(self):
        t = compute_total_abundances(
            IONIC, co_uv_ratio=UV_RATIO, Av=5.0,
            _lock_CO_tier="martinez25", **KW,
        )
        assert t["CO_method"] == "martinez25"

    def test_lock_to_a_legacy_tier(self):
        ionic = dict(IONIC, **{"C+/H+": 1.1e-5})
        t = compute_total_abundances(
            ionic, co_uv_ratio=UV_RATIO, _lock_CO_tier="garnett97_icf", **KW,
        )
        assert t["CO_method"] == "garnett97_icf"

    def test_unformable_locked_tier_yields_no_CO(self):
        """A draw that cannot form the locked tier drops out, not falls back."""
        t = compute_total_abundances(
            IONIC, _lock_CO_tier="martinez25_uv", **KW,
        )
        assert "C/O" not in t
        assert "locked" in t["_failures"]["C/O"]


class TestBadUvRatios:
    """A malformed UV ratio must be ignored, never adopted."""

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_rejected(self, bad):
        t = compute_total_abundances(
            IONIC_NO_OPTICAL_O, co_uv_ratio=bad, **KW,
        )
        assert t.get("CO_method") != "martinez25_uv"


# =========================================================================
# End-to-end: the tier choice must survive the wiring in _core.py
# =========================================================================

@dataclass
class _Line:
    flux: float
    flux_err: float


class _MockFitResult:
    def __init__(self, fluxes, errors):
        self.lines = {k: _Line(fluxes[k], errors[k]) for k in fluxes}


def _metal_poor_result(Te=20000.0, ne=1000.0, O_pp=1.5e-5, O_p=1.8e-6,
                       C_pp=4.0e-6, frac=0.04):
    """A synthetic object at 12+log(O/H) ~ 7.23, below the published floor.

    Carries both oxygen tracers -- [O III] 5007,4959 and O III] 1661,1666 --
    so the C/O tier choice is genuinely free, plus the [O II] doublet that
    O32 needs to reach log(U).
    """
    import pyneb as pn

    from jwspecabund.direct import _get_oiii_atom

    o3 = _get_oiii_atom()
    o2 = pn.Atom("O", 2)
    c3 = pn.Atom("C", 3)
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
        "CIII]_1907": _f(C_pp, c3.getEmissivity(Te, ne, wave=1907)),
        "CIII]": _f(C_pp, c3.getEmissivity(Te, ne, wave=1909)),
    }
    return _MockFitResult(fluxes, {k: frac * v for k, v in fluxes.items()})


class TestEndToEnd:
    """compute_abundances must carry the tier choice through, not drop it."""

    @staticmethod
    def _run(**kw):
        from jwspecabund import compute_abundances

        return compute_abundances(
            _metal_poor_result(), z=6.0, method="direct", dust_correct=False,
            n_mc=60, progress=False, **kw,
        )

    def test_metal_poor_object_reaches_logU_at_all(self):
        """The whole point of the relaxed floor: it is no longer rejected."""
        res = self._run()
        assert res.OH < 7.389                       # below the published floor
        assert res.logU is not None and np.isfinite(res.logU)
        assert "Martinez calibration range" in res.diagnostics

    def test_default_adopts_the_optical_tracer(self):
        res = self._run()
        assert "[OIII] 5007" in res.diagnostics["C/O"]
        assert res.CO is not None and np.isfinite(res.CO)

    def test_high_Av_switches_to_the_uv_tracer(self):
        res = self._run(Av=2.0)
        assert "O III] 1661,1666" in res.diagnostics["C/O"]
        assert "A_V" in res.diagnostics["C/O"]
        assert res.CO is not None and np.isfinite(res.CO)

    def test_both_tracers_agree_to_within_a_tenth_of_a_dex(self):
        """Same ion, same ICF — the two routes must not diverge wildly."""
        assert abs(self._run().CO - self._run(Av=2.0).CO) < 0.10

    def test_posterior_stays_on_one_tier(self):
        """Locking keeps the C/O posterior from mixing two methods.

        A bimodal posterior would show up as a spread far larger than the
        4% flux errors can justify.
        """
        res = self._run(Av=2.0)
        post = np.asarray(res.CO_posterior, dtype=float)
        post = post[np.isfinite(post)]
        assert post.size > 10
        assert np.std(post) < 0.10
