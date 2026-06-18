"""Tests for the 3-tier T_e zones, [Ar IV] density, and z-dependent
electron-density fallbacks (task_ne_te.md).

Covers the pure helper functions in :mod:`jwspecabund.direct` and the
``compute_ionic_abundances`` wiring of ``Te_int`` / ``ne_Opp``.
"""

from __future__ import annotations

import numpy as np
import pytest

from jwspecabund.direct import (
    Te_int_from_high,
    Te_low_from_high,
    compute_ionic_abundances,
    compute_ne_ArIV,
    heI_4714_over_4472,
    ne_zone_fallback,
)


class TestTeZones:
    """3-tier Garnett (1992) temperature relations."""

    def test_3tier_low_is_garnett(self):
        # T_low = 0.70 T_high + 3000.
        assert Te_low_from_high(15000.0, "3_tier") == pytest.approx(13500.0)
        # Default relation is now 3_tier.
        assert Te_low_from_high(15000.0) == pytest.approx(13500.0)

    def test_3tier_int_is_garnett_sIII(self):
        # T_int = 0.83 T_high + 1700.
        assert Te_int_from_high(15000.0, "3_tier") == pytest.approx(14150.0)
        assert Te_int_from_high(15000.0) == pytest.approx(14150.0)

    def test_zones_are_monotone_in_direct_te_regime(self):
        # In the regime where direct-T_e applies (hot auroral lines,
        # T_high >~ 1.1e4 K, metal-poor high-z gas) the Garnett relations
        # are monotone: T_high >= T_int >= T_low.
        for Te_high in (12000.0, 18000.0, 25000.0):
            Te_int = Te_int_from_high(Te_high)
            Te_low = Te_low_from_high(Te_high)
            assert Te_high >= Te_int >= Te_low, (
                f"non-monotone at T_high={Te_high}: "
                f"int={Te_int}, low={Te_low}"
            )

    def test_garnett_crossover_below_1e4(self):
        # Documented caveat: the Garnett relations have positive intercepts,
        # so below T_high = 1e4 K (relatively metal-rich gas, rare for a
        # 4363 detection) T_int and T_low exceed T_high.  This matches
        # Martinez+2025, who apply the relations without clamping.
        assert Te_int_from_high(8000.0) > 8000.0
        # Crossover is exactly at T_high = 1e4 K.
        assert Te_int_from_high(1.0e4) == pytest.approx(1.0e4)
        assert Te_low_from_high(1.0e4) == pytest.approx(1.0e4)

    def test_classical_and_garnett_aliases(self):
        for rel in ("classical", "garnett"):
            assert Te_low_from_high(15000.0, rel) == pytest.approx(13500.0)
            assert Te_int_from_high(15000.0, rel) == pytest.approx(14150.0)

    def test_desi_low_unchanged(self):
        # DESI low relation must be untouched.
        assert Te_low_from_high(15000.0, "desi") == pytest.approx(12990.0)

    def test_desi_has_no_intermediate(self):
        # Relations without a defined intermediate return None so callers
        # fall back to the midpoint.
        assert Te_int_from_high(15000.0, "desi") is None

    def test_unknown_relation_raises(self):
        with pytest.raises(ValueError):
            Te_low_from_high(15000.0, "nonsense")


class TestNeZoneFallback:
    """Redshift-dependent electron-density fallbacks."""

    def test_z0_values(self):
        # At z=0, n_e = A.
        assert ne_zone_fallback("low", 0.0) == pytest.approx(54.0)
        assert ne_zone_fallback("mid", 0.0) == pytest.approx(1100.0)
        assert ne_zone_fallback("high", 0.0) == pytest.approx(5400.0)

    def test_z6_high_zone(self):
        # 5400 * 7^1.62 ~ 1.26e5.
        assert ne_zone_fallback("high", 6.0) == pytest.approx(
            5400.0 * 7.0 ** 1.62, rel=1e-6
        )

    def test_rises_with_redshift(self):
        for zone in ("low", "mid", "high"):
            assert ne_zone_fallback(zone, 6.0) > ne_zone_fallback(zone, 0.0)

    def test_unknown_zone_raises(self):
        with pytest.raises(ValueError):
            ne_zone_fallback("bogus", 3.0)


class TestArIVDensity:
    """[Ar IV] 4711/4740 density and the He I deblend ratio."""

    def test_arIV_density_in_range(self):
        # A ratio near 1.2 should solve to a finite, positive density.
        ne = compute_ne_ArIV(1.2, 1.0, Te_guess=1.5e4)
        assert np.isfinite(ne) and ne > 0

    def test_arIV_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            compute_ne_ArIV(1.2, 0.0)
        with pytest.raises(ValueError):
            compute_ne_ArIV(0.0, 1.0)

    def test_heI_ratio_physical(self):
        # He I 4714/4472 ~ 0.10-0.21 over the relevant T_e range.
        r1 = heI_4714_over_4472(1.0e4, 1e3)
        r2 = heI_4714_over_4472(2.0e4, 1e3)
        assert 0.05 < r1 < 0.25
        assert 0.05 < r2 < 0.25
        # Ratio increases with temperature.
        assert r2 > r1


class TestIonicWiring:
    """Te_int / ne_Opp threading through compute_ionic_abundances."""

    def _base_fluxes(self):
        # Minimal flux set: Hbeta + [OIII] + [SIII] + [NeIII].
        return {
            "HBETA": 100.0,
            "OIII_5007": 500.0,
            "OIII_4959": 167.0,
            "SIII_9069": 20.0,
            "NeIII_3869": 40.0,
        }

    def test_ne_Opp_changes_Opp(self):
        # O++/H+ should depend on the O²⁺-zone density when ne_Opp differs
        # strongly from ne_mid (5007 stays density-insensitive at low ne but
        # the call path must honour ne_Opp).
        fluxes = self._base_fluxes()
        a = compute_ionic_abundances(
            fluxes, 1.5e4, 1.2e4, 1e2, ne_mid=1e2, ne_high=1e2,
            Te_int=1.35e4, ne_Opp=1e2,
        )
        b = compute_ionic_abundances(
            fluxes, 1.5e4, 1.2e4, 1e2, ne_mid=1e2, ne_high=1e2,
            Te_int=1.35e4, ne_Opp=1e5,
        )
        assert "O++/H+" in a and "O++/H+" in b
        # Both finite/positive; the high-density O²⁺ abundance differs.
        assert a["O++/H+"] > 0 and b["O++/H+"] > 0
        assert a["O++/H+"] != pytest.approx(b["O++/H+"], rel=1e-6)

    def test_Te_int_used_for_sIII(self):
        # S++/H+ must respond to Te_int (intermediate-zone temperature).
        fluxes = self._base_fluxes()
        a = compute_ionic_abundances(
            fluxes, 1.5e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=1.0e4,
        )
        b = compute_ionic_abundances(
            fluxes, 1.5e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=1.6e4,
        )
        assert a["S++/H+"] != pytest.approx(b["S++/H+"], rel=1e-6)

    def test_Te_int_none_falls_back_to_midpoint(self):
        # When Te_int is None the legacy 0.5*(Th+Tl) midpoint is used.
        fluxes = self._base_fluxes()
        none_call = compute_ionic_abundances(
            fluxes, 1.5e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=None,
        )
        mid_call = compute_ionic_abundances(
            fluxes, 1.5e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=0.5 * (1.5e4 + 1.2e4),
        )
        assert none_call["S++/H+"] == pytest.approx(mid_call["S++/H+"], rel=1e-9)


class TestMultiNeWiring:
    """_compute_multi_ne: [Ar IV] preference, He I deblend, z-fallbacks."""

    def _arIV_fluxes(self, ne_true=3000.0, hei=0.5):
        """Build [Ar IV] + He I fluxes that deblend to a target density."""
        import pyneb as pn

        from jwspecabund.direct import heI_4714_over_4472

        ar = pn.Atom("Ar", 4)
        # [Ar IV] 4711/4740 emissivity ratio at the target density.
        e4711 = ar.getEmissivity(1.5e4, ne_true, wave=4711)
        e4740 = ar.getEmissivity(1.5e4, ne_true, wave=4740)
        f4740 = 1.0
        f4711_true = f4740 * e4711 / e4740
        # Blend He I 4714 into the fitted ArIV_4713 feature.
        ratio = heI_4714_over_4472(1.5e4, 1e4)
        f4713 = f4711_true + ratio * hei
        return {
            "ArIV_4713": f4713,
            "ArIV_4741": f4740,
            "HEI_4472": hei,
        }

    def test_returns_five_tuple(self):
        from jwspecabund._core import _compute_multi_ne

        out = _compute_multi_ne({}, errors={}, snr_ne=0.0, z=5.0)
        assert len(out) == 5

    def test_arIV_sets_ne_Opp_with_deblend(self):
        from jwspecabund._core import _compute_multi_ne

        ne_true = 3000.0
        fluxes = self._arIV_fluxes(ne_true=ne_true, hei=0.5)
        errors = {k: v / 50.0 for k, v in fluxes.items()}  # high SNR
        ne_low, ne_mid, ne_high, ne_Opp, fail = _compute_multi_ne(
            fluxes, errors=errors, snr_ne=3.0, z=0.0,
        )
        # O²⁺-zone density recovered from [Ar IV] near the injected value.
        assert ne_Opp == pytest.approx(ne_true, rel=0.25)
        assert "n_e(Ar IV)" not in fail

    def test_arIV_skipped_without_heI_anchor(self):
        from jwspecabund._core import _compute_multi_ne
        from jwspecabund.direct import ne_zone_fallback

        fluxes = self._arIV_fluxes(ne_true=3000.0, hei=0.5)
        del fluxes["HEI_4472"]  # remove the anchor -> cannot deblend
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        _, ne_mid, _, ne_Opp, fail = _compute_multi_ne(
            fluxes, errors=errors, snr_ne=3.0, z=0.0,
        )
        # No CIII], no anchor -> ne_Opp uses the mid z-fallback.
        assert ne_Opp == pytest.approx(ne_zone_fallback("mid", 0.0))
        assert "n_e(Ar IV)" in fail

    def test_arIV_rejected_when_heI_dominated(self):
        from jwspecabund._core import _compute_multi_ne

        # Huge He I anchor so the deblended 4711 goes negative.
        fluxes = self._arIV_fluxes(ne_true=3000.0, hei=0.5)
        fluxes["HEI_4472"] = 100.0
        errors = {k: max(v, 1e-3) / 50.0 for k, v in fluxes.items()}
        _, _, _, _, fail = _compute_multi_ne(
            fluxes, errors=errors, snr_ne=3.0, z=0.0,
        )
        assert "n_e(Ar IV)" in fail
        assert "He I-dominated" in fail["n_e(Ar IV)"]

    def test_ciii_fallback_when_no_arIV(self):
        from jwspecabund._core import _compute_multi_ne

        # CIII] doublet present, no [Ar IV] -> ne_Opp comes from CIII].
        fluxes = {"CIII]_1907": 1.5, "CIII]": 1.0}
        errors = {"CIII]_1907": 0.02, "CIII]": 0.02}
        _, ne_mid, _, ne_Opp, _ = _compute_multi_ne(
            fluxes, errors=errors, snr_ne=3.0, z=0.0,
        )
        assert ne_mid is not None
        assert ne_Opp == pytest.approx(ne_mid)


class TestDensityRefinement:
    """_solve_densities_refined: zone temperatures + refined densities."""

    def _oiii_fluxes(self):
        # [OIII] auroral + nebular for a hot (~1.5e4 K) metal-poor object,
        # plus an [SII] doublet for the low-zone density.
        return {
            "HBETA": 100.0,
            "OIII_4363": 12.0,
            "OIII_5007": 600.0,
            "OIII_4959": 200.0,
            "SII_6718": 1.4,
            "SII_6732": 1.0,
        }

    def test_zone_temperatures_monotone(self):
        from jwspecabund._core import _solve_densities_refined

        fluxes = self._oiii_fluxes()
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        (
            ne_low, ne_mid, ne_high, ne_Opp,
            Te_high, Te_int, Te_low, diag, fail,
        ) = _solve_densities_refined(
            fluxes, errors, z=6.0, Te_relation="3_tier",
            snr_ne=3.0, ne_high_max=5e5, niv_rejected=False,
        )
        assert Te_high is not None and Te_high > 1.1e4
        assert Te_high >= Te_int >= Te_low
        assert diag == "4363"

    def test_refinement_uses_zone_density(self):
        # ne_low solved from [SII] should be finite and physical after the
        # two-pass refinement.
        from jwspecabund._core import _solve_densities_refined

        fluxes = self._oiii_fluxes()
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        ne_low = _solve_densities_refined(
            fluxes, errors, z=0.0, Te_relation="3_tier",
            snr_ne=3.0, ne_high_max=5e5, niv_rejected=False,
        )[0]
        assert np.isfinite(ne_low) and 10 < ne_low < 1e4

    def test_no_auroral_returns_none_temps(self):
        from jwspecabund._core import _solve_densities_refined

        fluxes = {"HBETA": 100.0, "OIII_5007": 600.0}
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        out = _solve_densities_refined(
            fluxes, errors, z=0.0, Te_relation="3_tier",
            snr_ne=3.0, ne_high_max=5e5, niv_rejected=False,
        )
        # Te_high, Te_int, Te_low all None when no auroral line.
        assert out[4] is None and out[5] is None and out[6] is None
