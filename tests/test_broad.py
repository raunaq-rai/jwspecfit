"""Tests for jwspecfit.broad — broad component fitting and BIC selection."""

import numpy as np
import pytest
import jwspecfit

from jwspecfit import fit_with_broad, read_fits
from jwspecfit.broad import (
    BROAD_CANDIDATES,
    BroadFitResult,
    _add_broad_lines,
    _balmer_pixel_mask,
    _resolve_balmer_broad_lines,
)
from jwspecfit.lines import REST_LINES_A
from .conftest import PRISM_FITS, G395M_FITS


class TestBalmerPixelMask:
    def test_mask_selects_ha_hb(self):
        """Mask should select pixels around Ha and Hb at given redshift."""
        z = 5.0
        wave_um = np.linspace(2.0, 5.0, 1000)
        mask = _balmer_pixel_mask(wave_um, z)
        assert mask.any(), "Mask should select some pixels"
        assert not mask.all(), "Mask should not select all pixels"
        selected = wave_um[mask]
        ha_obs = 6566.4 * (1.0 + z) * 1e-4
        hb_obs = 4864.0 * (1.0 + z) * 1e-4
        near_ha = np.any(np.abs(selected - ha_obs) < 0.1)
        near_hb = np.any(np.abs(selected - hb_obs) < 0.1)
        assert near_ha or near_hb

    def test_mask_empty_for_out_of_range(self):
        """Mask should be empty when lines are outside wavelength range."""
        wave_um = np.linspace(0.1, 0.3, 100)
        mask = _balmer_pixel_mask(wave_um, z=0.0)
        assert not mask.any()


class TestFitWithBroad:
    """Tests for the BIC-selected broad-component fitter.

    The legacy ``mode=`` kwarg was replaced with independent boolean
    flags ``fit_balmer_broad`` / ``fit_oiii_broad`` / ``fit_hei_broad``.
    When all are ``False`` (the default) only the narrow model is fit;
    when ``fit_balmer_broad=True`` BIC chooses among narrow / broad1
    (FWHM 500–2000 km/s) / broad2 (2000–5000 km/s) / both.
    """

    def test_narrow_only_default(self, prism_spectrum):
        """All broad flags off → narrow-only fit."""
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", n_boot=0,
        )
        assert isinstance(result, BroadFitResult)
        assert result.selected_model == "narrow"
        assert result.best_fit.success

    def test_balmer_bic_selection(self, grating_spectrum):
        """fit_balmer_broad=True triggers BIC; selected model must be valid."""
        result = fit_with_broad(
            grating_spectrum, z=5.0, grating="G395M",
            fit_balmer_broad=True, n_boot=0, n_boot_bic=0,
        )
        assert result.selected_model in ("narrow", "broad1", "broad2", "both")

    def test_bic_values_finite(self, prism_spectrum):
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM",
            fit_balmer_broad=True, n_boot=0, n_boot_bic=0,
        )
        assert np.isfinite(result.bic_narrow)

    def test_balmer_bic_populates_all_fits(self, prism_spectrum):
        """fit_balmer_broad=True must fit and store the narrow baseline."""
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM",
            fit_balmer_broad=True, n_boot=0, n_boot_bic=0,
        )
        assert "narrow" in result.all_fits

    def test_all_fits_dict(self, prism_spectrum):
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM",
            fit_balmer_broad=True, n_boot=0, n_boot_bic=0,
        )
        assert "narrow" in result.all_fits

    def test_bic_bootstrap_auto(self, grating_spectrum):
        """Bootstrap BIC with small n_boot_bic."""
        result = fit_with_broad(
            grating_spectrum, z=5.0, grating="G395M",
            fit_balmer_broad=True,
            n_boot=0, n_boot_bic=3, n_jobs=1, snr_threshold=0.0,
        )
        assert result.selected_model in ("narrow", "broad1", "broad2", "both")
        assert len(result.bic_bootstrap) > 0
        for variant, dist in result.bic_bootstrap.items():
            assert len(dist) == 3
            assert np.all(np.isfinite(dist))
        assert np.isfinite(result.bic_narrow)

    def test_bic_bootstrap_empty_when_no_flags(self, prism_spectrum):
        """No broad flag enabled → no BIC test → empty bic_bootstrap."""
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", n_boot=0,
        )
        assert result.bic_bootstrap == {}

    def test_bic_bootstrap_empty_when_zero(self, grating_spectrum):
        """Bootstrap BIC should be empty when n_boot_bic=0."""
        result = fit_with_broad(
            grating_spectrum, z=5.0, grating="G395M",
            fit_balmer_broad=True, n_boot=0, n_boot_bic=0,
        )
        assert result.bic_bootstrap == {}


class TestBroadFitPlumbing:
    """Flags and per-fit options must reach the fits they configure."""

    def test_hei_only_request_runs_the_hei_test(self, prism_spectrum):
        """fit_hei_broad alone used to return the narrow fit in silence.

        The narrow-only early return checked the Balmer and [OIII] flags but
        not the He I one, so a He I-only request skipped every BIC stage.
        """
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", fit_hei_broad=True,
            n_boot=0, n_boot_bic=0, hei_snr_threshold=0.0,
        )
        assert np.isfinite(result.bic_hei_off)
        assert result.hei_selected in ("off", "broad1", "broad2", "both")

    @pytest.mark.parametrize("n_boot_bic", [0, 2])
    def test_real_data_fits_honour_centroid_max_sigma(
        self, grating_spectrum, monkeypatch, n_boot_bic,
    ):
        """The ThreadPool fits passed their arguments positionally and stopped
        short of centroid_max_sigma, so those fits silently used the 1.0
        default however the caller had set it."""
        import inspect

        import jwspecfit.broad as broad

        real = broad._fit_model_variant
        sig = inspect.signature(real)
        seen = []

        def spy(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            seen.append(bound.arguments["centroid_max_sigma"])
            return real(*args, **kwargs)

        monkeypatch.setattr(broad, "_fit_model_variant", spy)
        fit_with_broad(
            grating_spectrum, z=5.0, grating="G395M", fit_balmer_broad=True,
            n_boot=0, n_boot_bic=n_boot_bic, n_jobs=1, snr_threshold=0.0,
            centroid_max_sigma=0.25,
        )
        assert seen
        assert all(s == 0.25 for s in seen)


class TestBalmerBroadLines:
    """``balmer_broad_lines`` chooses which Balmer lines get a broad component."""

    def test_default_is_every_candidate(self):
        assert _resolve_balmer_broad_lines(None) == BROAD_CANDIDATES

    def test_subset_is_ordered_and_deduplicated(self):
        """Caller order must not change the parameter vector."""
        assert _resolve_balmer_broad_lines(["HBETA", "Ha", "Ha"]) == ["Ha", "HBETA"]

    @pytest.mark.parametrize("bad", [["HEPSILON"], ["OIII_5007"], []])
    def test_rejects_unknown_or_empty(self, bad):
        with pytest.raises(ValueError, match="balmer_broad_lines"):
            _resolve_balmer_broad_lines(bad)

    def test_add_broad_lines_subset(self):
        narrow = ["Ha", "NII_6585", "HBETA", "HGAMMA", "HDELTA"]
        out = _add_broad_lines(narrow, "both", balmer_broad_lines=["Ha"])
        assert [n for n in out if "_BROAD" in n] == ["Ha_BROAD", "Ha_BROAD2"]

    def test_add_broad_lines_default_unchanged(self):
        out = _add_broad_lines(["Ha", "HBETA", "HGAMMA", "HDELTA"], "broad1")
        assert [n for n in out if "_BROAD" in n] == [
            f"{n}_BROAD" for n in BROAD_CANDIDATES
        ]

    def test_pixel_mask_follows_lines(self):
        z = 5.0
        wave_um = np.linspace(2.0, 5.0, 3000)
        both = _balmer_pixel_mask(wave_um, z)
        ha = _balmer_pixel_mask(wave_um, z, lines=("Ha",))
        hb_obs = REST_LINES_A["HBETA"] * (1.0 + z) * 1e-4
        assert np.array_equal(
            both, _balmer_pixel_mask(wave_um, z, lines=("Ha", "HBETA"))
        )
        assert 0 < ha.sum() < both.sum()
        assert not np.any(ha & (np.abs(wave_um - hb_obs) < 0.01))

    def test_fit_rejects_unknown_line(self, prism_spectrum):
        with pytest.raises(ValueError, match="HEPSILON"):
            fit_with_broad(
                prism_spectrum, z=6.0, grating="PRISM",
                fit_balmer_broad=True, balmer_broad_lines=["HEPSILON"],
            )

    def test_ha_only_fits_no_other_broad_balmer(self, prism_spectrum):
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM",
            fit_balmer_broad=True, balmer_broad_lines=["Ha"],
            n_boot=0, n_boot_bic=0, snr_threshold=0.0,
        )
        for variant in ("broad1", "broad2", "both"):
            broad = {
                n for n in result.all_fits[variant].line_names if "_BROAD" in n
            }
            assert broad and broad <= {"Ha_BROAD", "Ha_BROAD2"}

    def test_bootstrap_fits_receive_subset(self, grating_spectrum, monkeypatch):
        """Every broad fit, bootstrap iterations included, sees the subset."""
        import inspect

        import jwspecfit.broad as broad

        real = broad._fit_model_variant
        sig = inspect.signature(real)
        seen = []

        def spy(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            if bound.arguments["broad_type"] is not None:
                seen.append(bound.arguments["balmer_broad_lines"])
            return real(*args, **kwargs)

        monkeypatch.setattr(broad, "_fit_model_variant", spy)
        fit_with_broad(
            grating_spectrum, z=5.0, grating="G395M",
            fit_balmer_broad=True, balmer_broad_lines=["Ha"],
            n_boot=0, n_boot_bic=2, n_jobs=1, snr_threshold=0.0,
        )
        # 3 variants x 2 bootstrap draws + 3 real-data fits.
        assert len(seen) == 9
        assert all(s == ["Ha"] for s in seen)

    def test_snr_gate_uses_requested_line(self, prism_spectrum):
        """An Hβ-only test is gated on Hβ's SNR, not Hα's."""
        narrow = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", n_boot=0,
        ).best_fit
        if not {"Ha", "HBETA"} <= narrow.lines.keys():
            pytest.skip("needs both Hα and Hβ in the fit")
        ha, hb = narrow.lines["Ha"].snr, narrow.lines["HBETA"].snr
        if not ha > hb:
            pytest.skip("needs Hα detected more strongly than Hβ")
        kw = dict(
            z=6.0, grating="PRISM", fit_balmer_broad=True,
            n_boot=0, n_boot_bic=0, snr_threshold=0.5 * (ha + hb),
        )
        hb_only = fit_with_broad(prism_spectrum, balmer_broad_lines=["HBETA"], **kw)
        ha_only = fit_with_broad(prism_spectrum, balmer_broad_lines=["Ha"], **kw)
        assert np.isnan(hb_only.bic_broad1)
        assert np.isfinite(ha_only.bic_broad1)

    @pytest.mark.parametrize(
        "module, entry, target",
        [
            ("jwspecfit", "fit_lines", "jwspecfit"),
            ("jwspecmcmc", "fit_lines", "jwspecfit.broad"),
            ("jwspecmcmc", "fit_with_broad", "jwspecfit.broad"),
        ],
    )
    def test_entry_points_pass_subset(
        self, prism_spectrum, monkeypatch, module, entry, target,
    ):
        """Each public entry point hands the subset to the BIC stage."""
        import importlib

        class _Captured(Exception):
            pass

        def capture(*args, **kwargs):
            raise _Captured(kwargs.get("balmer_broad_lines"))

        monkeypatch.setattr(importlib.import_module(target), "fit_with_broad", capture)
        with pytest.raises(_Captured) as exc:
            getattr(importlib.import_module(module), entry)(
                prism_spectrum, z=6.0,
                fit_balmer_broad=True, balmer_broad_lines=["Ha"],
            )
        assert exc.value.args[0] == ["Ha"]
