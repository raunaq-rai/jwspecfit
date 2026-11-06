"""Renamed line keys: Ha -> HALPHA, CIII] -> CIII]_1909 (v1.1.11).

The old names must keep working, with a FutureWarning, wherever a user
passes a line name in or looks one up in a result.
"""

import warnings

import numpy as np
import pytest

from jwspecfit.constraints import ConstraintSet
from jwspecfit.fitter import FitResult, LineResult
from jwspecfit.lines import (
    LEGACY_LINE_NAMES,
    REST_LINES_A,
    LineDict,
    canonical_line_keys,
    canonical_line_name,
    canonical_line_names,
)


def _lr(name, flux=1.0):
    return LineResult(
        name=name, rest_wave_A=REST_LINES_A.get(name, 6564.6319),
        amplitude=1.0, centroid_A=6564.6, sigma_A=3.0,
        flux=flux, flux_err=0.1, ew0_A=10.0,
        snr_int_err=10.0, snr_int_cont=10.0, snr_peak_err=10.0, snr_peak_cont=10.0,
    )


def _fit_result(names):
    return FitResult(
        lines={n: _lr(n) for n in names},
        params=np.zeros(3 * len(names)), model_flux=np.zeros(3),
        continuum=np.zeros(3), residuals=np.zeros(3), chi2=1.0,
        spectrum=None, line_names=list(names),
    )


class TestDatabase:
    def test_new_keys_present_old_keys_gone(self):
        assert REST_LINES_A["HALPHA"] == pytest.approx(6564.6319)
        assert REST_LINES_A["CIII]_1909"] == pytest.approx(1908.734)
        assert REST_LINES_A["CIII]_1907"] == pytest.approx(1906.683)
        for old in LEGACY_LINE_NAMES:
            assert old not in REST_LINES_A

    def test_every_legacy_target_is_a_line_or_broad_component(self):
        for new in LEGACY_LINE_NAMES.values():
            assert new.split("_BROAD")[0] in REST_LINES_A


class TestCanonicalisation:
    def test_old_name_warns_and_maps(self):
        with pytest.warns(FutureWarning, match="'Ha' was renamed to 'HALPHA'"):
            assert canonical_line_name("Ha") == "HALPHA"

    def test_current_name_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert canonical_line_name("HALPHA") == "HALPHA"
            assert canonical_line_name("OIII_5007") == "OIII_5007"

    def test_container_type_is_kept(self):
        with pytest.warns(FutureWarning):
            assert canonical_line_names(("Ha", "HBETA")) == ("HALPHA", "HBETA")
        with pytest.warns(FutureWarning):
            assert canonical_line_names({"CIII]"}) == {"CIII]_1909"}
        assert canonical_line_names(None) is None

    def test_mapping_keys(self):
        with pytest.warns(FutureWarning):
            d = canonical_line_keys({"Ha": 3.0, "HBETA": 1.0})
        assert set(d) == {"HALPHA", "HBETA"}


class TestLineDict:
    def test_old_key_lookup_warns_and_resolves(self):
        d = LineDict({"HALPHA": 1, "CIII]_1909": 2})
        with pytest.warns(FutureWarning):
            assert d["Ha"] == 1
        with pytest.warns(FutureWarning):
            assert d.get("CIII]") == 2
        assert "Ha" in d and "CIII]" in d
        assert d.get("missing", 7) == 7
        with pytest.raises(KeyError):
            d["Hz"]

    def test_old_keys_are_stored_under_new_names(self):
        d = LineDict({"Ha": 1, "Ha_BROAD": 2})
        assert set(dict.keys(d)) == {"HALPHA", "HALPHA_BROAD"}


class TestResultsBuiltWithOldNames:
    """Saved files and older code hand the result classes the old keys."""

    def test_fit_result_keys_names_and_line_names(self):
        res = _fit_result(["Ha", "CIII]", "HBETA"])
        assert list(res.lines) == ["HALPHA", "CIII]_1909", "HBETA"]
        assert res.line_names == ["HALPHA", "CIII]_1909", "HBETA"]
        assert res.lines["HALPHA"].name == "HALPHA"
        with pytest.warns(FutureWarning):
            assert res.lines["Ha"] is res.lines["HALPHA"]

    def test_constraint_set(self):
        cs = ConstraintSet(["Ha", "CIII]_1907", "CIII]"], blended_doublets={"CIII]"})
        assert cs.line_names == ["HALPHA", "CIII]_1907", "CIII]_1909"]
        assert cs.blended_doublets == {"CIII]_1909"}

    def test_save_load_round_trip_of_old_file(self, tmp_path):
        from jwspecfit import Spectrum
        from jwspecfit.io import load_result, save_result

        spec = Spectrum(
            wave_um=np.linspace(3.8, 4.0, 50), flux_ujy=np.ones(50),
            err_ujy=np.full(50, 0.1), z=5.0,
        )
        res = _fit_result(["HALPHA"])
        res.spectrum = spec
        res.model_flux = res.continuum = res.residuals = np.zeros(50)
        save_result(res, tmp_path / "r.npz")

        # Rewrite the file as an older release would have written it.
        data = dict(np.load(tmp_path / "r.npz", allow_pickle=False))
        data["lines_json"] = np.array(
            [str(data["lines_json"][0]).replace('"HALPHA"', '"Ha"')]
        )
        data["line_names"] = np.array(["Ha"])
        np.savez(tmp_path / "old.npz", **data)

        old = load_result(tmp_path / "old.npz")
        assert list(old.lines) == ["HALPHA"] and old.line_names == ["HALPHA"]


class TestEntryPoints:
    def test_fit_lines_accepts_old_names(self, grating_spectrum):
        from jwspecfit.fitter import fit_lines

        with pytest.warns(FutureWarning, match="'Ha'"):
            res = fit_lines(
                grating_spectrum, z=5.0, grating="G395M", n_boot=0,
                lines=["Ha", "NII_6549", "NII_6585"],
            )
        assert "HALPHA" in dict.keys(res.lines)
        assert res.lines["HALPHA"].flux > 0

    def test_balmer_broad_lines_accepts_old_name(self):
        from jwspecfit.broad import _resolve_balmer_broad_lines

        with pytest.warns(FutureWarning):
            assert _resolve_balmer_broad_lines(["Ha"]) == ["HALPHA"]

    def test_dust_pair_and_anchor_accept_old_names(self):
        from jwspecabund.dust import compute_Av_balmer_pair

        fluxes = {"Ha": 2.86 * 1.3, "HBETA": 1.0}
        errors = {"Ha": 0.01, "HBETA": 0.01}
        with pytest.warns(FutureWarning):
            out = compute_Av_balmer_pair(fluxes, errors, pair=("Ha", "HBETA"))
        assert out["Av"] > 0

    def test_continuum_masks_old_name(self):
        """fit_continuum skips unknown names silently, so 'Ha' must map."""
        from jwspecfit.continuum import fit_continuum

        wave = np.linspace(3.8, 4.1, 400)
        lam = 6564.6319 * 6.0 / 1e4
        flux = 1.0 + 50.0 * np.exp(-0.5 * ((wave - lam) / 0.002) ** 2)
        err = np.full_like(wave, 0.05)
        new = fit_continuum(wave, flux, err, 5.0, ["HALPHA"], R=1000.0)
        with pytest.warns(FutureWarning):
            old = fit_continuum(wave, flux, err, 5.0, ["Ha"], R=1000.0)
        np.testing.assert_allclose(old, new)
