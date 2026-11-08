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
        assert _resolve_balmer_broad_lines(["HBETA", "HALPHA", "HALPHA"]) == ["HALPHA", "HBETA"]

    @pytest.mark.parametrize("bad", [["HEPSILON"], ["OIII_5007"], []])
    def test_rejects_unknown_or_empty(self, bad):
        with pytest.raises(ValueError, match="balmer_broad_lines"):
            _resolve_balmer_broad_lines(bad)

    def test_add_broad_lines_subset(self):
        narrow = ["HALPHA", "NII_6585", "HBETA", "HGAMMA", "HDELTA"]
        out = _add_broad_lines(narrow, "both", balmer_broad_lines=["HALPHA"])
        assert [n for n in out if "_BROAD" in n] == ["HALPHA_BROAD", "HALPHA_BROAD2"]

    def test_add_broad_lines_default_unchanged(self):
        out = _add_broad_lines(["HALPHA", "HBETA", "HGAMMA", "HDELTA"], "broad1")
        assert [n for n in out if "_BROAD" in n] == [
            f"{n}_BROAD" for n in BROAD_CANDIDATES
        ]

    def test_pixel_mask_follows_lines(self):
        z = 5.0
        wave_um = np.linspace(2.0, 5.0, 3000)
        both = _balmer_pixel_mask(wave_um, z)
        ha = _balmer_pixel_mask(wave_um, z, lines=("HALPHA",))
        hb_obs = REST_LINES_A["HBETA"] * (1.0 + z) * 1e-4
        assert np.array_equal(
            both, _balmer_pixel_mask(wave_um, z, lines=("HALPHA", "HBETA"))
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
            fit_balmer_broad=True, balmer_broad_lines=["HALPHA"],
            n_boot=0, n_boot_bic=0, snr_threshold=0.0,
        )
        for variant in ("broad1", "broad2", "both"):
            broad = {
                n for n in result.all_fits[variant].line_names if "_BROAD" in n
            }
            assert broad and broad <= {"HALPHA_BROAD", "HALPHA_BROAD2"}

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
            fit_balmer_broad=True, balmer_broad_lines=["HALPHA"],
            n_boot=0, n_boot_bic=2, n_jobs=1, snr_threshold=0.0,
        )
        # 3 variants x 2 bootstrap draws + 3 real-data fits.
        assert len(seen) == 9
        assert all(s == ["HALPHA"] for s in seen)

    def test_snr_gate_uses_requested_line(self, prism_spectrum):
        """An Hβ-only test is gated on Hβ's SNR, not Hα's."""
        narrow = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", n_boot=0,
        ).best_fit
        if not {"HALPHA", "HBETA"} <= narrow.lines.keys():
            pytest.skip("needs both Hα and Hβ in the fit")
        ha, hb = narrow.lines["HALPHA"].snr, narrow.lines["HBETA"].snr
        if not ha > hb:
            pytest.skip("needs Hα detected more strongly than Hβ")
        kw = dict(
            z=6.0, grating="PRISM", fit_balmer_broad=True,
            n_boot=0, n_boot_bic=0, snr_threshold=0.5 * (ha + hb),
        )
        hb_only = fit_with_broad(prism_spectrum, balmer_broad_lines=["HBETA"], **kw)
        ha_only = fit_with_broad(prism_spectrum, balmer_broad_lines=["HALPHA"], **kw)
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
                fit_balmer_broad=True, balmer_broad_lines=["HALPHA"],
            )
        assert exc.value.args[0] == ["HALPHA"]


# --- Regression tests from the 2026-11 audit -------------------------------

def _synthetic_g395m(z, comps, err=0.01, seed=0):
    """Narrow + broad lines on a G395M-like grid; comps = [(rest_A, flux, sigma_v_kms)]."""
    from jwspecfit.io import Spectrum, _flam_to_ujy
    from jwspecfit.models import gaussian_binned

    wave = np.arange(2.87, 5.27, 0.00170476)
    s0 = Spectrum(wave_um=wave, flux_ujy=0 * wave, err_ujy=np.full_like(wave, err), grating="G395M")
    e = s0.wave_edges_A
    flam = np.zeros(len(wave))
    for rest, f, sv in comps:
        lam = rest * (1 + z)
        sig = np.hypot(lam / (2.3548 * 1000.0), sv / 2.998e5 * lam)
        flam += f * gaussian_binned(e[:-1], e[1:], lam, sig)
    rng = np.random.default_rng(seed)
    flux = 0.2 + _flam_to_ujy(flam, wave) + rng.normal(0, err, len(wave))
    return Spectrum(wave_um=wave, flux_ujy=flux, err_ujy=s0.err_ujy, grating="G395M")


class TestSelectByBic:
    def test_nan_comparison_never_adopts_the_complex_model(self):
        from jwspecfit.broad import _select_by_bic

        assert _select_by_bic({"narrow": np.nan, "broad1": 10.0}, "narrow", 6.0)[0] == "narrow"

    def test_paired_draws_decide_not_unpaired_medians(self):
        from jwspecfit.broad import _select_by_bic

        # Unpaired medians differ by only 5 (< 6); draw by draw the broad
        # model wins by 105 in two of three draws, so the paired median is 105.
        boot = {"narrow": np.array([100.0, 200.0, 300.0]),
                "broad1": np.array([290.0, 95.0, 195.0])}
        point = {k: float(np.median(v)) for k, v in boot.items()}
        assert _select_by_bic(point, "narrow", 6.0)[0] == "narrow"
        sel, gain = _select_by_bic(point, "narrow", 6.0, boot=boot)
        assert (sel, gain) == ("broad1", 105.0)

class TestBroadPathRegressions:
    Z = 5.0
    NARROW = [(REST_LINES_A["HBETA"], 3e-19, 60), (REST_LINES_A["OIII_4959"], 3.3e-19, 60),
              (REST_LINES_A["OIII_5007"], 1e-18, 60), (REST_LINES_A["HALPHA"], 9e-19, 60),
              (REST_LINES_A["NII_6585"], 1e-19, 60), (REST_LINES_A["NII_6549"], 1e-19 / 2.96, 60)]
    LINES = ["HBETA", "OIII_4959", "OIII_5007", "HALPHA", "NII_6549", "NII_6585"]

    def test_nii_not_pinned_by_a_floor_from_the_narrow_only_fit(self):
        """A narrow-only fit soaks broad wings into [NII]; no bound may be built from it."""
        broad = [(REST_LINES_A["HALPHA"], 2.7e-18, 1300)]
        spec = _synthetic_g395m(self.Z, self.NARROW + broad, seed=1)
        res = fit_with_broad(spec, self.Z, lines=self.LINES, fit_balmer_broad=True,
                             balmer_broad_lines=["HALPHA"], n_boot=50, n_boot_bic=0)
        assert res.selected_model != "narrow"
        nii = res.best_fit.lines["NII_6585"]
        assert nii.flux_err > 0
        assert nii.flux < 2.0e-19          # truth 1e-19; the old floor held it at ~2.6x

    def test_gate_uses_a_covered_balmer_line(self):
        """Halpha off the detector must not silently skip the Balmer test (G395M, z = 7)."""
        z = 7.0
        comps = [(REST_LINES_A["HBETA"], 1e-18, 60), (REST_LINES_A["OIII_5007"], 2e-18, 60),
                 (REST_LINES_A["OIII_4959"], 2e-18 / 2.984, 60), (REST_LINES_A["HBETA"], 3e-18, 1300)]
        spec = _synthetic_g395m(z, comps, seed=2)
        res = fit_with_broad(spec, z, lines=["HBETA", "OIII_4959", "OIII_5007"],
                             fit_balmer_broad=True, n_boot=0, n_boot_bic=0)
        assert res.selected_model != "narrow"

    def test_public_wrapper_keeps_wave_range_on_the_broad_path(self):
        spec = _synthetic_g395m(self.Z, self.NARROW, seed=3)
        lo, hi = 4600 * (1 + self.Z), 6900 * (1 + self.Z)
        res = jwspecfit.fit_lines(spec, self.Z, lines=self.LINES, wave_range_A=(lo, hi),
                                  fit_balmer_broad=True, n_boot=0, n_boot_bic=0)
        w = res.best_fit.spectrum.wave_A
        assert w.min() >= lo and w.max() <= hi

    def test_bootstrap_bic_selection_is_reproducible(self):
        spec = _synthetic_g395m(self.Z, self.NARROW, seed=4)
        kw = dict(lines=self.LINES, fit_balmer_broad=True, balmer_broad_lines=["HALPHA"],
                  n_boot=0, n_boot_bic=8, n_jobs=1)
        a, b = fit_with_broad(spec, self.Z, **kw), fit_with_broad(spec, self.Z, **kw)
        assert a.selected_model == b.selected_model
        assert a.bic_narrow == b.bic_narrow

    def test_bootstrap_chi2_is_on_the_noise_scale_of_the_perturbed_data(self):
        """Perturbed data carry sqrt(2) x err: their narrow chi2/N must be ~1, not ~2."""
        from jwspecfit.broad import _balmer_pixel_mask

        spec = _synthetic_g395m(self.Z, self.NARROW, seed=5)
        res = fit_with_broad(spec, self.Z, lines=self.LINES, fit_balmer_broad=True,
                             balmer_broad_lines=["HALPHA"], n_boot=0, n_boot_bic=12, n_jobs=1)
        n_pix = int(np.sum(_balmer_pixel_mask(spec.wave_um, self.Z, lines=["HALPHA"])))
        k = int(np.sum(res.all_fits["narrow"].constraints.free_mask()))
        chi2 = np.median(res.bic_bootstrap["narrow"]) - k * np.log(n_pix)
        assert 0.6 < chi2 / n_pix < 1.4
