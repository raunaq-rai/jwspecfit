"""Tests for assumed-width, continuum-noise flux limits.

The behaviour under test is that a non-detection's limit comes from the
spectrum's noise filtered through an *assumed* line profile, and never
from the fitted flux or the fitted width — both of which are
unconstrained when there is nothing to fit.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from jwspecfit import fit_lines
from jwspecfit.fitter import LineResult
from jwspecfit.io import Spectrum
from jwspecfit.limits import (
    SIGMA_V_FALLBACK_KMS,
    assumed_sigma_A,
    flux_noise_for_lines,
    flux_upper_limit,
    intrinsic_sigma_kms,
    matched_filter_flux_error,
)
from jwspecfit.lines import REST_LINES_A
from jwspecfit.resolution import sigma_inst_A

_SQRT2PI = np.sqrt(2.0 * np.pi)


def _white_noise_sigma(rms: float, sigma_A: float, dlam: float) -> float:
    """Textbook 1σ flux error for a known-shape Gaussian in white noise."""
    return rms * np.sqrt(2.0 * np.sqrt(np.pi) * sigma_A * dlam)


@pytest.fixture(scope="module")
def synthetic_fit():
    """A G395M-like spectrum with four real lines and two absent ones.

    [O III] 4363 and [N II] 6585 are *not* injected, so their fitted
    widths are pure noise — which is the case the limits must survive.
    """
    rng = np.random.default_rng(7)
    z = 6.0
    sigma_v = 120.0
    wave = np.linspace(2.4, 4.6, 1400)
    lam_A = wave * 1e4
    flux = np.full_like(wave, 0.05)
    for name, amp in {
        "HBETA": 2.0, "OIII_5007": 6.0, "OIII_4959": 2.0, "Ha": 6.0,
    }.items():
        lam_obs = REST_LINES_A[name] * (1.0 + z)
        s_inst = sigma_inst_A(np.array([lam_obs * 1e-4]), grating="G395M")[0]
        s = np.sqrt(s_inst**2 + (sigma_v / 2.99792458e5 * lam_obs) ** 2)
        flux += amp * 0.05 * np.exp(-0.5 * ((lam_A - lam_obs) / s) ** 2)
    flux += rng.normal(0.0, 0.02, wave.size)
    spec = Spectrum(
        wave_um=wave, flux_ujy=flux, err_ujy=np.full_like(wave, 0.02),
        grating="G395M", z=z,
    )
    return fit_lines(
        spec, z=z,
        lines=["HBETA", "OIII_5007", "OIII_4959", "OIII_4363", "Ha", "NII_6585"],
        n_boot=60,
    )


class TestMatchedFilter:
    """The noise measurement itself."""

    @pytest.mark.parametrize("sig_over_dlam", [0.93, 2.0, 4.0])
    def test_reproduces_the_white_noise_formula(self, sig_over_dlam):
        """With white noise it must land on rms·√(2√π σ Δλ)."""
        rng = np.random.default_rng(1)
        dlam, rms = 1.0, 1.0
        sigma_A = sig_over_dlam * dlam
        wave = np.arange(3000.0, 3000.0 + 400 * dlam, dlam)
        lam0 = wave[len(wave) // 2]
        valid = np.ones_like(wave, dtype=bool)

        measured = np.mean([
            matched_filter_flux_error(
                wave, rng.normal(0.0, rms, wave.size), valid, lam0, sigma_A,
            ).sigma_flux
            for _ in range(200)
        ])
        assert measured == pytest.approx(
            _white_noise_sigma(rms, sigma_A, dlam), rel=0.06,
        )

    def test_correlated_noise_inflates_the_limit(self):
        """Correlated pixels must give a *larger* error, not the white one.

        This is the whole reason the scatter is measured rather than
        evaluated: NIRSpec's resampled pixels are not independent, and
        a closed-form white-noise limit would be optimistic.
        """
        rng = np.random.default_rng(2)
        dlam, sigma_A = 1.0, 0.93
        wave = np.arange(3000.0, 3000.0 + 400 * dlam, dlam)
        lam0 = wave[200]
        valid = np.ones_like(wave, dtype=bool)

        white, correlated = [], []
        for _ in range(200):
            w = rng.normal(0.0, 1.0, wave.size)
            c = gaussian_filter1d(w, 1.0)
            c /= np.std(c)  # same per-pixel RMS, correlated neighbours
            white.append(
                matched_filter_flux_error(wave, w, valid, lam0, sigma_A).sigma_flux
            )
            correlated.append(
                matched_filter_flux_error(wave, c, valid, lam0, sigma_A).sigma_flux
            )
        assert np.mean(correlated) > 1.3 * np.mean(white)

    def test_scales_linearly_with_the_noise(self):
        rng = np.random.default_rng(3)
        wave = np.arange(3000.0, 3400.0, 1.0)
        valid = np.ones_like(wave, dtype=bool)
        base = rng.normal(0.0, 1.0, wave.size)
        one = matched_filter_flux_error(wave, base, valid, 3200.0, 2.0)
        ten = matched_filter_flux_error(wave, 10.0 * base, valid, 3200.0, 2.0)
        assert ten.sigma_flux == pytest.approx(10.0 * one.sigma_flux, rel=1e-9)

    def test_falls_back_when_too_few_positions(self):
        """A short valid stretch still yields the analytic estimate."""
        rng = np.random.default_rng(4)
        wave = np.arange(3000.0, 3400.0, 1.0)
        valid = np.zeros_like(wave, dtype=bool)
        valid[195:202] = True  # seven usable pixels, fewer than _MIN_POSITIONS
        noise = matched_filter_flux_error(
            wave, rng.normal(0.0, 1.0, wave.size), valid, 3198.0, 2.0,
        )
        assert noise.method == "matched_filter_analytic"
        assert noise.n_positions == 0
        assert noise.sigma_flux > 0

    def test_returns_none_without_usable_pixels(self):
        wave = np.arange(3000.0, 3100.0, 1.0)
        resid = np.full_like(wave, np.nan)
        assert matched_filter_flux_error(
            wave, resid, np.ones_like(wave, bool), 3050.0, 2.0,
        ) is None

    def test_rejects_a_nonsense_width(self):
        wave = np.arange(3000.0, 3100.0, 1.0)
        resid = np.zeros_like(wave)
        valid = np.ones_like(wave, bool)
        for bad in (0.0, -1.0, np.nan):
            assert matched_filter_flux_error(wave, resid, valid, 3050.0, bad) is None

    def test_upper_limit_is_n_sigma_times_the_error(self):
        rng = np.random.default_rng(5)
        wave = np.arange(3000.0, 3400.0, 1.0)
        resid = rng.normal(0.0, 1.0, wave.size)
        valid = np.ones_like(wave, bool)
        err = matched_filter_flux_error(wave, resid, valid, 3200.0, 2.0).sigma_flux
        ul = flux_upper_limit(wave, resid, valid, 3200.0, 2.0, n_sigma=3.0)
        assert ul == pytest.approx(3.0 * err, rel=1e-12)


class TestAssumedWidth:
    """Where the assumed σ comes from."""

    def test_instrumental_only_when_no_velocity_width(self):
        lam = 34000.0
        assert assumed_sigma_A(lam, 0.0, grating="G395M") == pytest.approx(
            sigma_inst_A(np.array([lam * 1e-4]), grating="G395M")[0],
        )

    def test_velocity_adds_in_quadrature(self):
        lam, sigma_v = 34000.0, 150.0
        s_inst = sigma_inst_A(np.array([lam * 1e-4]), grating="G395M")[0]
        expected = np.hypot(s_inst, sigma_v / 2.99792458e5 * lam)
        assert assumed_sigma_A(lam, sigma_v, grating="G395M") == pytest.approx(expected)

    def test_array_input_keeps_shape(self):
        out = assumed_sigma_A(np.array([20000.0, 34000.0]), 100.0, grating="G395M")
        assert out.shape == (2,)

    def test_velocity_width_recovered_from_detected_lines(self, synthetic_fit):
        """The injected 120 km/s must come back from the strong lines."""
        sigma_v = intrinsic_sigma_kms(synthetic_fit.lines, grating="G395M")
        assert sigma_v == pytest.approx(120.0, abs=25.0)

    def test_undetected_lines_do_not_drag_the_width(self, synthetic_fit):
        """Low-SNR lines are excluded, so a collapsed σ cannot bias it."""
        loose = intrinsic_sigma_kms(synthetic_fit.lines, grating="G395M", snr_min=0.0)
        strict = intrinsic_sigma_kms(synthetic_fit.lines, grating="G395M", snr_min=5.0)
        assert loose < strict

    def test_none_when_nothing_is_detected(self, synthetic_fit):
        assert intrinsic_sigma_kms(
            synthetic_fit.lines, grating="G395M", snr_min=1e6,
        ) is None

    def test_unresolved_lines_give_zero_not_nan(self):
        """σ_fit below σ_inst must clamp to zero, not go imaginary."""
        lam = 34000.0
        s_inst = float(sigma_inst_A(np.array([lam * 1e-4]), grating="G395M")[0])
        lines = {
            f"L{i}": LineResult(
                name=f"L{i}", rest_wave_A=lam / 7.0, amplitude=1.0,
                centroid_A=lam, sigma_A=0.5 * s_inst, flux=1.0, flux_err=0.01,
                ew_A=1.0, snr_int_err=100.0, snr_int_cont=100.0,
                snr_peak_err=100.0, snr_peak_cont=100.0,
            )
            for i in range(3)
        }
        assert intrinsic_sigma_kms(lines, grating="G395M") == 0.0

    def test_an_implausible_width_warns_but_is_not_capped(self, caplog):
        """Never silently narrow the assumption — that tightens the limit."""
        lam = 34000.0
        s_inst = float(sigma_inst_A(np.array([lam * 1e-4]), grating="G395M")[0])
        wide = np.hypot(s_inst, 2000.0 / 2.99792458e5 * lam)
        lines = {
            f"L{i}": LineResult(
                name=f"L{i}", rest_wave_A=lam / 7.0, amplitude=1.0,
                centroid_A=lam, sigma_A=wide, flux=1.0, flux_err=0.01,
                ew_A=1.0, snr_int_err=100.0, snr_int_cont=100.0,
                snr_peak_err=100.0, snr_peak_cont=100.0,
            )
            for i in range(3)
        }
        with caplog.at_level("WARNING", logger="jwspecfit.limits"):
            sigma_v = intrinsic_sigma_kms(lines, grating="G395M")
        assert sigma_v == pytest.approx(2000.0, rel=0.02)  # reported, not capped
        assert any("narrow-line region" in r.message for r in caplog.records)

    def test_broad_components_and_lya_are_ignored(self):
        lam = 34000.0
        s_inst = float(sigma_inst_A(np.array([lam * 1e-4]), grating="G395M")[0])
        def _lr(name, sig):
            return LineResult(
                name=name, rest_wave_A=lam / 7.0, amplitude=1.0, centroid_A=lam,
                sigma_A=sig, flux=1.0, flux_err=0.01, ew_A=1.0,
                snr_int_err=100.0, snr_int_cont=100.0, snr_peak_err=100.0,
                snr_peak_cont=100.0,
            )
        narrow = 2.0 * s_inst
        lines = {
            "Ha": _lr("Ha", narrow),
            "HBETA": _lr("HBETA", narrow),
            "Ha_BROAD": _lr("Ha_BROAD", 20.0 * s_inst),
            "Lya": _lr("Lya", 20.0 * s_inst),
        }
        both = intrinsic_sigma_kms(lines, grating="G395M")
        narrow_only = intrinsic_sigma_kms(
            {k: v for k, v in lines.items() if k in ("Ha", "HBETA")},
            grating="G395M",
        )
        assert both == pytest.approx(narrow_only)


class TestFitIntegration:
    """What fit_lines actually attaches, and what upper_limit() returns."""

    def test_every_line_gets_a_limit(self, synthetic_fit):
        for name, lr in synthetic_fit.lines.items():
            assert lr.flux_err_cont is not None and lr.flux_err_cont > 0, name
            assert lr.sigma_assumed_A is not None and lr.sigma_assumed_A > 0, name
            assert lr.ul_method in ("matched_filter", "matched_filter_analytic")

    def test_assumed_width_beats_a_collapsed_fitted_width(self, synthetic_fit):
        """[O III] 4363 is absent; its fitted σ falls below the instrument.

        The old limit used that σ directly and so under-reported the
        limit by the same factor.
        """
        lr = synthetic_fit.lines["OIII_4363"]
        s_inst = float(
            sigma_inst_A(np.array([lr.centroid_A * 1e-4]), grating="G395M")[0]
        )
        assert lr.sigma_A < s_inst          # physically impossible for a real line
        assert lr.sigma_assumed_A > s_inst  # assumed width is at least instrumental
        assert lr.sigma_assumed_A > 2.0 * lr.sigma_A

    def test_upper_limit_uses_the_continuum_noise(self, synthetic_fit):
        lr = synthetic_fit.lines["OIII_4363"]
        assert lr.upper_limit(3.0) == pytest.approx(
            3.0 * lr.flux_err_cont, rel=1e-12, abs=0.0,
        )
        # Nothing of the bootstrap error survives in the answer.
        assert abs(lr.upper_limit(3.0) - 3.0 * lr.flux_err) > 0.1 * lr.upper_limit(3.0)

    def test_upper_limit_scales_with_n_sigma(self, synthetic_fit):
        lr = synthetic_fit.lines["OIII_4363"]
        assert lr.upper_limit(5.0) == pytest.approx(
            5.0 / 3.0 * lr.upper_limit(3.0), rel=1e-12, abs=0.0,
        )

    def test_falls_back_to_the_bootstrap_error_when_unmeasured(self):
        """A hand-built LineResult keeps the old behaviour."""
        lr = LineResult(
            name="X", rest_wave_A=5007.0, amplitude=1.0, centroid_A=35000.0,
            sigma_A=10.0, flux=1.0, flux_err=0.25, ew_A=1.0, snr_int_err=4.0,
            snr_int_cont=4.0, snr_peak_err=4.0, snr_peak_cont=4.0,
        )
        assert lr.upper_limit(3.0) == pytest.approx(0.75)

    def test_result_method_matches_the_cached_value(self, synthetic_fit):
        for name, lr in synthetic_fit.lines.items():
            assert synthetic_fit.flux_upper_limit(name) == pytest.approx(
                3.0 * lr.flux_err_cont, rel=1e-12, abs=0.0,
            )

    def test_velocity_override_widens_the_limit(self, synthetic_fit):
        """A broader assumed line integrates more noise."""
        narrow = synthetic_fit.flux_upper_limit("OIII_4363", sigma_v_kms=0.0)
        broad = synthetic_fit.flux_upper_limit("OIII_4363", sigma_v_kms=600.0)
        assert broad > 1.5 * narrow

    def test_limits_dict_covers_only_low_snr_lines(self, synthetic_fit):
        uls = synthetic_fit.flux_upper_limits(snr_threshold=3.0)
        assert "OIII_4363" in uls
        assert "OIII_5007" not in uls

    def test_unknown_line_raises(self, synthetic_fit):
        with pytest.raises(KeyError):
            synthetic_fit.flux_upper_limit("NOT_A_LINE")

    def test_limits_survive_a_save_load_round_trip(self, synthetic_fit, tmp_path):
        from jwspecfit.io import load_result, save_result

        path = tmp_path / "fit.npz"
        save_result(synthetic_fit, path)
        loaded = load_result(path)
        for name, lr in synthetic_fit.lines.items():
            back = loaded.lines[name]
            assert back.flux_err_cont == pytest.approx(
                lr.flux_err_cont, rel=1e-9, abs=0.0,
            )
            assert back.sigma_assumed_A == pytest.approx(lr.sigma_assumed_A, rel=1e-9)
            assert back.ul_method == lr.ul_method
            assert back.upper_limit() == pytest.approx(
                lr.upper_limit(), rel=1e-9, abs=0.0,
            )


class TestBackfill:
    """Results saved before these fields existed must not go quiet."""

    @staticmethod
    def _strip(path_in, path_out):
        """Rewrite a saved .npz without the limit fields, as old files are."""
        import json

        raw = dict(np.load(path_in, allow_pickle=False))
        lines_data = json.loads(str(raw["lines_json"][0]))
        for entry in lines_data.values():
            for key in ("flux_err_cont", "sigma_assumed_A", "ul_method"):
                entry.pop(key, None)
        raw["lines_json"] = np.array([json.dumps(lines_data)])
        np.savez_compressed(path_out, **raw)

    @pytest.fixture
    def legacy_result(self, synthetic_fit, tmp_path):
        from jwspecfit.io import load_result, save_result

        save_result(synthetic_fit, tmp_path / "new.npz")
        self._strip(tmp_path / "new.npz", tmp_path / "old.npz")
        return load_result(tmp_path / "old.npz")

    def test_loading_an_old_file_measures_the_limits(
        self, legacy_result, synthetic_fit,
    ):
        for name, lr in synthetic_fit.lines.items():
            back = legacy_result.lines[name]
            assert back.flux_err_cont is not None, name
            assert back.upper_limit() == pytest.approx(
                lr.upper_limit(), rel=1e-9, abs=0.0,
            )

    def test_old_file_no_longer_falls_back_to_the_bootstrap_error(
        self, legacy_result,
    ):
        lr = legacy_result.lines["OIII_4363"]
        assert abs(lr.upper_limit() - 3.0 * lr.flux_err) > 0.1 * lr.upper_limit()

    def test_result_method_works_on_an_old_file(self, legacy_result):
        assert legacy_result.flux_upper_limit("OIII_4363") is not None

    def test_backfill_leaves_measured_lines_alone(self, synthetic_fit):
        from jwspecfit.limits import backfill_flux_limits

        before = {n: lr.flux_err_cont for n, lr in synthetic_fit.lines.items()}
        assert backfill_flux_limits(synthetic_fit) == 0
        for name, value in before.items():
            assert synthetic_fit.lines[name].flux_err_cont == value

    def test_backfill_without_a_spectrum_is_a_no_op(self):
        from jwspecfit.limits import backfill_flux_limits

        class _Bare:
            spectrum = None
            lines = {}

        assert backfill_flux_limits(_Bare()) == 0


class TestFluxNoiseForLines:
    """The per-fit driver."""

    def test_explicit_width_overrides_the_measurement(self, synthetic_fit):
        spec = synthetic_fit.spectrum
        from jwspecfit.io import _ujy_to_flam

        resid = _ujy_to_flam(synthetic_fit.residuals, spec.wave_um)
        _, used = flux_noise_for_lines(
            synthetic_fit.lines, spec.wave_A, resid, np.isfinite(resid), 6.0,
            grating="G395M", sigma_v_kms=333.0,
        )
        assert used == pytest.approx(333.0)

    def test_falls_back_when_nothing_is_detected(self):
        """No detections means no evidence for anything above instrumental."""
        rng = np.random.default_rng(6)
        wave = np.linspace(3.0, 4.0, 400)
        spec = Spectrum(
            wave_um=wave, flux_ujy=rng.normal(0.05, 0.02, wave.size),
            err_ujy=np.full_like(wave, 0.02), grating="G395M", z=6.0,
        )
        result = fit_lines(spec, z=6.0, lines=["HBETA", "OIII_5007"], n_boot=30)
        from jwspecfit.io import _ujy_to_flam

        resid = _ujy_to_flam(result.residuals, spec.wave_um)
        _, used = flux_noise_for_lines(
            result.lines, spec.wave_A, resid, np.isfinite(resid), 6.0,
            grating="G395M",
        )
        assert used == pytest.approx(SIGMA_V_FALLBACK_KMS)


class TestAbundanceIntegration:
    """jwspecabund must reuse what the fit already measured."""

    def test_reuses_the_cached_limits(self, synthetic_fit):
        from jwspecabund._core import _compute_continuum_rms_limits

        limits = _compute_continuum_rms_limits(
            synthetic_fit, z=6.0, Av=None, dust_law="salim",
        )
        assert limits, "no limits produced"
        for name, ul in limits.items():
            assert ul == pytest.approx(
                3.0 * synthetic_fit.lines[name].flux_err_cont, rel=1e-12, abs=0.0,
            )

    def test_n_sigma_scales_the_limits(self, synthetic_fit):
        from jwspecabund._core import _compute_continuum_rms_limits

        three = _compute_continuum_rms_limits(
            synthetic_fit, z=6.0, Av=None, dust_law="salim", n_sigma=3.0,
        )
        five = _compute_continuum_rms_limits(
            synthetic_fit, z=6.0, Av=None, dust_law="salim", n_sigma=5.0,
        )
        for name in three:
            assert five[name] == pytest.approx(
                5.0 / 3.0 * three[name], rel=1e-12, abs=0.0,
            )

    def test_dust_correction_raises_the_limits(self, synthetic_fit):
        from jwspecabund._core import _compute_continuum_rms_limits

        bare = _compute_continuum_rms_limits(
            synthetic_fit, z=6.0, Av=None, dust_law="salim",
        )
        reddened = _compute_continuum_rms_limits(
            synthetic_fit, z=6.0, Av=1.0, dust_law="salim",
        )
        for name in bare:
            assert reddened[name] > bare[name]

    def test_measures_limits_the_fit_did_not_cache(self, synthetic_fit):
        """Results predating this change still get proper limits."""
        import copy

        from jwspecabund._core import _compute_continuum_rms_limits

        stale = copy.deepcopy(synthetic_fit)
        for lr in stale.lines.values():
            lr.flux_err_cont = None
            lr.sigma_assumed_A = None
            lr.ul_method = None

        recomputed = _compute_continuum_rms_limits(
            stale, z=6.0, Av=None, dust_law="salim",
        )
        cached = _compute_continuum_rms_limits(
            synthetic_fit, z=6.0, Av=None, dust_law="salim",
        )
        assert set(recomputed) == set(cached)
        for name in cached:
            assert recomputed[name] == pytest.approx(cached[name], rel=1e-9, abs=0.0)

    def test_partial_cache_still_uses_the_detected_line_widths(self, synthetic_fit):
        """Refilling only the faint lines must not lose the velocity width.

        The lines needing a limit are the undetected ones, which carry no
        width information — so the width has to be measured from the whole
        fit, not from the subset being filled in.
        """
        import copy

        from jwspecabund._core import _compute_continuum_rms_limits

        partial = copy.deepcopy(synthetic_fit)
        faint = "OIII_4363"
        for name, lr in partial.lines.items():
            if name == faint:
                lr.flux_err_cont = None
                lr.sigma_assumed_A = None
                lr.ul_method = None

        refilled = _compute_continuum_rms_limits(
            partial, z=6.0, Av=None, dust_law="salim",
        )
        cached = _compute_continuum_rms_limits(
            synthetic_fit, z=6.0, Av=None, dust_law="salim",
        )
        assert refilled[faint] == pytest.approx(cached[faint], rel=1e-9, abs=0.0)

    def test_no_spectrum_yields_no_limits(self):
        from jwspecabund._core import _compute_continuum_rms_limits

        class _Bare:
            spectrum = None

        assert _compute_continuum_rms_limits(
            _Bare(), z=6.0, Av=None, dust_law="salim",
        ) == {}


class TestMCMCIntegration:
    """The MCMC path must attach the same fields, for the same reason."""

    @pytest.fixture(scope="class")
    def mcmc_fit(self):
        import jwspecmcmc

        rng = np.random.default_rng(11)
        z = 6.0
        wave = np.linspace(3.2, 3.7, 320)
        lam_A = wave * 1e4
        flux = np.full_like(wave, 0.05)
        for name, amp in {"HBETA": 2.0, "OIII_5007": 6.0, "OIII_4959": 2.0}.items():
            lam_obs = REST_LINES_A[name] * (1.0 + z)
            s_inst = sigma_inst_A(np.array([lam_obs * 1e-4]), grating="G395M")[0]
            s = np.hypot(s_inst, 120.0 / 2.99792458e5 * lam_obs)
            flux += amp * 0.05 * np.exp(-0.5 * ((lam_A - lam_obs) / s) ** 2)
        flux += rng.normal(0.0, 0.02, wave.size)
        spec = Spectrum(
            wave_um=wave, flux_ujy=flux, err_ujy=np.full_like(wave, 0.02),
            grating="G395M", z=z,
        )
        return jwspecmcmc.fit_lines(
            spec, z=z, lines=["HBETA", "OIII_5007", "OIII_4959"],
            n_walkers=24, n_steps=400, n_burn=150, sampler="emcee",
        )

    def test_every_line_gets_a_limit(self, mcmc_fit):
        for name, lr in mcmc_fit.lines.items():
            assert lr.flux_err_cont is not None and lr.flux_err_cont > 0, name
            assert lr.sigma_assumed_A is not None and lr.sigma_assumed_A > 0, name

    def test_upper_limit_ignores_the_posterior_width(self, mcmc_fit):
        for lr in mcmc_fit.lines.values():
            assert lr.upper_limit(3.0) == pytest.approx(
                3.0 * lr.flux_err_cont, rel=1e-12, abs=0.0,
            )

    def test_result_method_matches_the_cached_value(self, mcmc_fit):
        for name, lr in mcmc_fit.lines.items():
            assert mcmc_fit.flux_upper_limit(name) == pytest.approx(
                3.0 * lr.flux_err_cont, rel=1e-12, abs=0.0,
            )

    def test_to_fit_result_carries_the_limits(self, mcmc_fit):
        converted = mcmc_fit.to_fit_result()
        for name, lr in mcmc_fit.lines.items():
            assert converted.lines[name].flux_err_cont == pytest.approx(
                lr.flux_err_cont, rel=1e-12, abs=0.0,
            )
            assert converted.lines[name].ul_method == lr.ul_method

    def test_limits_survive_a_save_load_round_trip(self, mcmc_fit, tmp_path):
        from jwspecmcmc.io import load_mcmc_result, save_mcmc_result

        path = tmp_path / "mcmc.npz"
        save_mcmc_result(mcmc_fit, path)
        loaded = load_mcmc_result(path)
        for name, lr in mcmc_fit.lines.items():
            back = loaded.lines[name]
            assert back.flux_err_cont == pytest.approx(
                lr.flux_err_cont, rel=1e-9, abs=0.0,
            )
            assert back.sigma_assumed_A == pytest.approx(lr.sigma_assumed_A, rel=1e-9)
            assert back.ul_method == lr.ul_method
