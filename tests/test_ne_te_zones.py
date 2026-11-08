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
        # At z=0, n_e = A. low: Abdurro'uf+2024 (54 cm^-3, slope 1.2);
        # mid/high: Martinez+2025 Eqs (4)/(5).
        assert ne_zone_fallback("low", 0.0) == pytest.approx(54.0)
        assert ne_zone_fallback("mid", 0.0) == pytest.approx(1110.0)
        assert ne_zone_fallback("high", 0.0) == pytest.approx(5400.0)

    def test_literature_slopes(self):
        # Slopes: low=1.2 (Abdurro'uf+2024), mid=1.93, high=1.62 (Martinez+2025).
        z = 4.0
        assert ne_zone_fallback("low", z) == pytest.approx(54.0 * (1 + z) ** 1.2)
        assert ne_zone_fallback("mid", z) == pytest.approx(1110.0 * (1 + z) ** 1.93)
        assert ne_zone_fallback("high", z) == pytest.approx(5400.0 * (1 + z) ** 1.62)

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
            "SIII_9071": 20.0,
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

    def test_Te_int_used_for_nIII_cIII(self):
        # N²⁺ (NIII]) and C²⁺ (CIII]) are intermediate-zone ions and must
        # respond to Te_int, not Te_high (Martinez+2025 §5.4).
        fluxes = {
            "HBETA": 100.0,
            "NIII_1749": 5.0, "NIII_1752": 4.0,
            "CIII]_1907": 8.0, "CIII]_1909": 6.0,
        }
        lo = compute_ionic_abundances(
            fluxes, 2.0e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=1.3e4,
        )
        hi = compute_ionic_abundances(
            fluxes, 2.0e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=1.7e4,
        )
        assert lo["N++/H+"] != pytest.approx(hi["N++/H+"], rel=1e-6)
        assert lo["C++/H+"] != pytest.approx(hi["C++/H+"], rel=1e-6)

    def test_nIII_uses_intermediate_not_high(self):
        # Explicitly: changing only Te_high (not Te_int) must NOT move N²⁺.
        fluxes = {"HBETA": 100.0, "NIII_1749": 5.0, "NIII_1752": 4.0}
        a = compute_ionic_abundances(
            fluxes, 1.8e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=1.4e4,
        )
        b = compute_ionic_abundances(
            fluxes, 2.6e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=1.4e4,
        )
        assert a["N++/H+"] == pytest.approx(b["N++/H+"], rel=1e-9)

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

    def test_mid_density_falls_back_to_ne_Opp_not_low(self):
        # When CIII] fails (ne_mid=None), the intermediate ions C²⁺/N²⁺ must
        # use the supplied O²⁺-zone density (z-fallback mid), NOT ne_low.
        fluxes = {
            "HBETA": 100.0,
            "CIII]_1907": 8.0, "CIII]_1909": 6.0,
            "NIII_1749": 5.0, "NIII_1752": 4.0,
        }
        # ne_mid=None, ne_low=600, ne_Opp=6e4 (a z-fallback mid value).
        viafallback = compute_ionic_abundances(
            fluxes, 1.8e4, 1.2e4, 600.0, ne_mid=None, ne_Opp=6e4, Te_int=1.4e4,
        )
        # Same call but explicitly at ne_mid=6e4 should match the fallback.
        explicit = compute_ionic_abundances(
            fluxes, 1.8e4, 1.2e4, 600.0, ne_mid=6e4, ne_Opp=6e4, Te_int=1.4e4,
        )
        # And it must NOT equal the (wrong) ne_low=600 result.
        atlow = compute_ionic_abundances(
            fluxes, 1.8e4, 1.2e4, 600.0, ne_mid=600.0, ne_Opp=6e4, Te_int=1.4e4,
        )
        for ion in ("C++/H+", "N++/H+"):
            assert viafallback[ion] == pytest.approx(explicit[ion], rel=1e-9)
            assert viafallback[ion] != pytest.approx(atlow[ion], rel=1e-6)

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


class TestDefaultRelation:
    """The 3-tier Garnett scheme is the compute_abundances default."""

    def test_compute_abundances_default_is_3tier(self):
        import inspect

        from jwspecabund import compute_abundances

        sig = inspect.signature(compute_abundances)
        assert sig.parameters["Te_relation"].default == "3_tier"

    def test_diagnostics_report_te_method_4363(self):
        # The Te(high) method string must name the actual auroral line.
        from jwspecabund._core import _run_direct

        fluxes = {
            "HBETA": 100.0,
            "OIII_4363": 12.0,
            "OIII_5007": 600.0,
            "OIII_4959": 200.0,
            "OII_3726": 30.0,
            "OII_3729": 40.0,
        }
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=20, progress=False, seed=1, z=6.0,
        )
        diag = out["diagnostics"]
        assert "Te(high) method" in diag
        assert "4363" in diag["Te(high) method"]
        assert "1666" not in diag["Te(high) method"]

    def test_diagnostics_report_te_method_1666(self):
        # With only O III] 1666 (no 4363), the method must say 1666.
        from jwspecabund._core import _run_direct

        fluxes = {
            "HBETA": 100.0,
            "OIII_1666": 15.0,
            "OIII_5007": 600.0,
            "OIII_4959": 200.0,
            "OII_3726": 30.0,
            "OII_3729": 40.0,
        }
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=20, progress=False, seed=1, z=6.0,
        )
        diag = out["diagnostics"]
        assert "1666" in diag["Te(high) method"]

    def test_run_direct_reports_Te_mid(self):
        # A default ("3_tier") _run_direct run must populate Te_mid (the
        # intermediate-zone temperature) and keep zones monotone.
        from jwspecabund._core import _run_direct

        fluxes = {
            "HBETA": 100.0,
            "OIII_4363": 12.0,
            "OIII_5007": 600.0,
            "OIII_4959": 200.0,
            "OII_3726": 30.0,
            "OII_3729": 40.0,
        }
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=20, progress=False, seed=1, z=6.0,
        )
        assert out["Te_mid"] is not None
        assert out["Te_high"] >= out["Te_mid"] >= out["Te_low"]


class TestZneDecoupling:
    """z_ne drives the n_e fallback independently of the geometric z."""

    def test_z_ne_overrides_fallback_redshift(self):
        from jwspecabund._core import _solve_densities_refined
        from jwspecabund.direct import ne_zone_fallback

        # No high-zone doublet -> ne_high uses the z-fallback.  Passing a
        # high z must raise ne_high relative to z=0 even with no z change
        # elsewhere (mimics z=0 geometry, z_ne=7 physical).
        fluxes = {
            "HBETA": 100.0, "OIII_4363": 12.0,
            "OIII_5007": 600.0, "OIII_4959": 200.0,
        }
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        nh0 = _solve_densities_refined(
            fluxes, errors, z=0.0, Te_relation="3_tier",
            snr_ne=3.0, ne_high_max=5e5, niv_rejected=False,
        )[2]
        nh7 = _solve_densities_refined(
            fluxes, errors, z=7.0, Te_relation="3_tier",
            snr_ne=3.0, ne_high_max=5e5, niv_rejected=False,
        )[2]
        assert nh0 == pytest.approx(ne_zone_fallback("high", 0.0))
        assert nh7 == pytest.approx(ne_zone_fallback("high", 7.0))
        assert nh7 > nh0


class TestNeDiagnosticLabels:
    """ne(low)/ne(high) diagnostic strings name the true source."""

    def _oiii(self):
        return {
            "HBETA": 100.0, "OIII_4363": 12.0,
            "OIII_5007": 600.0, "OIII_4959": 200.0,
        }

    def test_ne_low_labels_sii_doublet(self):
        from jwspecabund._core import _run_direct

        fluxes = {**self._oiii(), "SII_6718": 1.4, "SII_6732": 1.0}
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=10, progress=False, seed=1, z=7.0,
        )
        assert "[SII]" in out["diagnostics"]["ne(low)"]
        assert "fallback" not in out["diagnostics"]["ne(low)"]

    def test_ne_low_labels_siIII_and_skipped_sii(self):
        from jwspecabund._core import _run_direct

        fluxes = {**self._oiii(), "SII_6718": 1.4, "SII_6732": 1.0,
                  "SiIII_1": 1.3, "SiIII_2": 1.0}
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=10, progress=False, seed=1, z=7.0,
        )
        s = out["diagnostics"]["ne(low)"]
        assert s.startswith("[Si III] 1883/1892")

    def test_ne_low_labels_sii_when_siIII_weak(self):
        from jwspecabund._core import _run_direct

        fluxes = {**self._oiii(), "SII_6718": 1.4, "SII_6732": 1.0,
                  "SiIII_1": 1.3, "SiIII_2": 1.0}
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        errors.update({"SiIII_1": 1.0, "SiIII_2": 1.0})
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=10, progress=False, seed=1, z=7.0,
        )
        s = out["diagnostics"]["ne(low)"]
        assert s.startswith("[SII] 6718/6732")
        assert "[Si III] 1883/1892 skipped" in s

    def test_ne_low_flags_siIII_below_its_floor(self):
        from jwspecabund._core import _run_direct

        # Ratio 1.55: ~4e2 cm^-3, in the insensitive low-density regime.
        fluxes = {**self._oiii(), "SiIII_1": 1.55, "SiIII_2": 1.0}
        errors = {k: v / 200.0 for k, v in fluxes.items()}
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=10, progress=False, seed=1, z=7.0,
        )
        s = out["diagnostics"]["ne(low)"]
        assert s.startswith("[Si III]")
        assert "upper limit" in s

    def test_ne_low_labels_zfallback(self):
        # No low-ion doublet -> ne(low) must say z-fallback, not a doublet.
        from jwspecabund._core import _run_direct

        fluxes = self._oiii()
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=10, progress=False, seed=1, z=7.0,
        )
        s = out["diagnostics"]["ne(low)"]
        assert s.startswith("z-fallback")
        assert "doublet ratio" not in s

    def test_ne_high_labels_zfallback(self):
        from jwspecabund._core import _run_direct

        fluxes = self._oiii()  # no NIV]/[Ar IV]
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=10, progress=False, seed=1, z=7.0,
        )
        assert "fallback" in out["diagnostics"]["ne(high)"].lower()

    def test_ne_mid_labels_zfallback(self):
        """No CIII] -> the intermediate zone uses the Martinez+25 z-fallback."""
        from jwspecabund._core import _run_direct
        from jwspecabund.direct import ne_zone_fallback

        fluxes = self._oiii()
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=10, progress=False, seed=1, z=7.0,
        )
        s = out["diagnostics"]["ne(mid)"]
        assert s.startswith("z-fallback")
        assert f"{ne_zone_fallback('mid', 7.0):.0f} cm^-3" in s
        assert "no CIII] doublet available" in s
        assert "ne(low)" not in s

    def test_ne_mid_labels_zfallback_when_ciii_fails_snr(self):
        from jwspecabund._core import _run_direct

        fluxes = {**self._oiii(), "CIII]_1907": 0.05, "CIII]_1909": 0.04}
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        errors["CIII]_1907"] = errors["CIII]_1909"] = 1.0  # SNR << snr_ne
        out = _run_direct(
            fluxes, errors, "3_tier", n_mc=10, progress=False, seed=1, z=7.0,
        )
        s = out["diagnostics"]["ne(mid)"]
        assert s.startswith("z-fallback")
        assert "CIII] failed SNR cut or solve" in s

    def test_ne_mid_labels_arIV_when_that_sets_the_opp_zone(self):
        from jwspecabund._core import _build_diagnostics

        diag = _build_diagnostics(
            {"HBETA": 1.0}, None, "3_tier", ne_low=500.0, ne_mid=None,
            ne_high=1e5, logU=None, logU_diag=None, icf_method=None,
            NO_icf_name=None, ne_default=300.0, ne_Opp=2.5e4, z=7.0,
        )
        assert diag["ne(mid)"].startswith("O2+-zone density (25000 cm^-3, [Ar IV])")

    def test_upper_limits_use_the_same_mid_density_as_abundances(self):
        """C2+/N2+ limits sat at ne(low) while detections used the z-fallback."""
        from jwspecabund._core import _ne_mid_used

        assert _ne_mid_used(800.0, 6e4, 500.0) == 800.0   # CIII] measured
        assert _ne_mid_used(None, 6e4, 500.0) == 6e4      # O2+ zone / z-fallback
        assert _ne_mid_used(None, None, 500.0) == 500.0


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

    def test_siIII_uv_low_ion_fallback(self):
        # No optical [SII]/[OII]; [Si III] 1883/1892 should set ne_low.
        from jwspecabund._core import _compute_multi_ne
        from jwspecabund.direct import ne_zone_fallback

        fluxes = {"SiIII_1": 1.5, "SiIII_2": 1.0}
        errors = {"SiIII_1": 0.02, "SiIII_2": 0.02}
        ne_low, _, _, _, _ = _compute_multi_ne(
            fluxes, errors=errors, snr_ne=3.0, z=0.0,
        )
        # Measured from [Si III], not the z-fallback.
        assert ne_low != pytest.approx(ne_zone_fallback("low", 0.0))
        assert 10 < ne_low < 1e5

    def test_siIII_preferred_over_optical(self):
        # When both [SII] and [Si III] are present, [Si III] wins.
        from jwspecabund._core import _compute_multi_ne
        from jwspecabund.direct import compute_ne_SiIII

        fluxes = {
            "SII_6718": 1.4, "SII_6732": 1.0,
            "SiIII_1": 1.3, "SiIII_2": 1.0,
        }
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        ne_low, _, _, _, fail = _compute_multi_ne(
            fluxes, errors=errors, snr_ne=3.0, z=0.0,
        )
        assert ne_low == pytest.approx(compute_ne_SiIII(1.3, 1.0, Te_guess=1.5e4))
        assert "n_e([Si III])" not in fail

    @pytest.mark.parametrize("bad", ["snr", "solve"])
    def test_weak_siIII_falls_through_to_sii(self, bad):
        # A present-but-unusable [Si III] must not block a good [SII].
        from jwspecabund._core import _compute_multi_ne
        from jwspecabund.direct import compute_ne

        si = (1.3, 1.0) if bad == "snr" else (5.0, 1.0)   # 5.0: above the limit
        fluxes = {"SII_6718": 1.4, "SII_6732": 1.0,
                  "SiIII_1": si[0], "SiIII_2": si[1]}
        errors = {"SII_6718": 0.028, "SII_6732": 0.02}
        errors.update({"SiIII_1": 1.0, "SiIII_2": 1.0} if bad == "snr"
                      else {"SiIII_1": 0.05, "SiIII_2": 0.05})
        ne_low, _, _, _, fail = _compute_multi_ne(
            fluxes, errors=errors, snr_ne=3.0, z=0.0,
        )
        assert ne_low == pytest.approx(
            compute_ne(1.4, 1.0, doublet="SII", Te_guess=1.5e4), rel=1e-6,
        )
        assert "n_e([Si III])" in fail
        assert "n_e(SII)" not in fail

    def test_sii_preferred_over_oii(self):
        from jwspecabund._core import _compute_multi_ne
        from jwspecabund.direct import compute_ne

        fluxes = {"SII_6718": 1.4, "SII_6732": 1.0,
                  "OII_3726": 1.0, "OII_3729": 1.3}
        errors = {k: v / 50.0 for k, v in fluxes.items()}
        ne_low = _compute_multi_ne(fluxes, errors=errors, snr_ne=3.0, z=0.0)[0]
        assert ne_low == pytest.approx(
            compute_ne(1.4, 1.0, doublet="SII", Te_guess=1.5e4), rel=1e-6,
        )

    def test_ciii_fallback_when_no_arIV(self):
        from jwspecabund._core import _compute_multi_ne

        # CIII] doublet present, no [Ar IV] -> ne_Opp comes from CIII].
        fluxes = {"CIII]_1907": 1.5, "CIII]_1909": 1.0}
        errors = {"CIII]_1907": 0.02, "CIII]_1909": 0.02}
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
            Te_high, Te_int, Te_low, diag, fail, sc, ne_Opp_single,
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
