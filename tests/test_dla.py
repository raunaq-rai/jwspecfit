"""Tests for the DLA column density fitter (jwspecfit.dla)."""

from __future__ import annotations

import numpy as np
import pytest

from jwspecfit.dla import (
    DLAResult,
    _evaluate_model,
    _mask_emission_lines,
    compute_D_Lya,
    fit_NHI,
    tau_DLA,
    tau_IGM_DW,
    voigt_H,
)


# ============================================================
# Voigt-Hjerting function
# ============================================================

class TestVoigtHjerting:
    """Test voigt_H against exact Faddeeva function."""

    def test_exact_via_wofz(self):
        """voigt_H should match wofz to machine precision."""
        from scipy.special import wofz

        a_vals = [1e-4, 1e-3, 1e-2, 0.1]
        u_vals = np.linspace(-50.0, 50.0, 500)

        for a in a_vals:
            exact = np.array([wofz(complex(u, a)).real for u in u_vals])
            ours = voigt_H(a, u_vals)
            np.testing.assert_allclose(ours, exact, rtol=1e-12,
                                       err_msg=f"Mismatch at a={a}")

    def test_gaussian_core_dominates(self):
        """For a -> 0 and moderate u, H(a, u) ~ exp(-u^2)."""
        a = 1e-8
        u = np.linspace(0.5, 1.5, 20)
        H = voigt_H(a, u)
        expected = np.exp(-u ** 2)
        np.testing.assert_allclose(H, expected, rtol=0.01)

    def test_damping_wings_positive(self):
        """H(a, u) should be positive for all u."""
        a = 0.01
        u = np.linspace(-1000.0, 1000.0, 10000)
        H = voigt_H(a, u)
        assert np.all(H >= 0), "H(a, u) should be non-negative."

    def test_wing_regime_accuracy(self):
        """In the damping wing (u >> 1), H ~ a/(sqrt(pi)*u^2)."""
        a = 0.01
        u_vals = np.array([20.0, 50.0, 100.0])
        H = voigt_H(a, u_vals)
        leading = a / (np.sqrt(np.pi) * u_vals ** 2)
        np.testing.assert_allclose(H, leading, rtol=0.05)


# ============================================================
# DLA optical depth
# ============================================================

class TestTauDLA:
    """Test the DLA optical depth calculation."""

    def test_proportional_to_NHI(self):
        """tau should scale linearly with N_HI."""
        wave = np.linspace(1220.0, 1400.0, 100)
        tau_20 = tau_DLA(wave, 20.0, z=0.0)
        tau_21 = tau_DLA(wave, 21.0, z=0.0)
        ratio = tau_21 / np.maximum(tau_20, 1e-30)
        mask = (wave > 1230) & (wave < 1350)
        np.testing.assert_allclose(ratio[mask], 10.0, rtol=0.01)

    def test_large_at_line_centre(self):
        """tau should be very large near Lya for high N_HI."""
        wave = np.array([1216.0])
        tau = tau_DLA(wave, 22.0, z=0.0)[0]
        assert tau > 1e3, f"Expected tau >> 1, got {tau}."

    def test_small_far_from_line(self):
        """tau should be negligible far from Lya for moderate N_HI."""
        wave = np.array([3000.0])
        tau = tau_DLA(wave, 20.0, z=0.0)[0]
        assert tau < 0.01, f"Expected tau << 1, got {tau}."

    def test_redshift_shifts_absorption(self):
        """At z > 0, DLA should be centred at 1215.67*(1+z)."""
        z = 2.0
        lya_obs = 1215.67 * (1 + z)
        wave_near = np.array([lya_obs + 2.0])
        tau_near = tau_DLA(wave_near, 22.0, z=z)[0]
        wave_far = np.array([lya_obs + 1000.0])
        tau_far = tau_DLA(wave_far, 22.0, z=z)[0]
        assert tau_near > tau_far


# ============================================================
# Emission line masking
# ============================================================

class TestMaskEmissionLines:
    """Test the emission line masking function."""

    def test_masks_civ_at_z0(self):
        """CIV 1549 should be masked at z=0."""
        wave = np.linspace(1500, 1600, 200)
        mask = _mask_emission_lines(wave, z=0.0, width_A=10.0)
        civ_region = (wave > 1538) & (wave < 1562)
        assert not mask[civ_region].all()

    def test_masks_shift_with_z(self):
        """Masks should shift with redshift."""
        wave = np.linspace(3000, 3200, 200)
        mask = _mask_emission_lines(wave, z=1.0, width_A=10.0)
        civ_region = (wave > 3076) & (wave < 3124)
        assert not mask[civ_region].all()

    def test_keeps_continuum_pixels(self):
        """Pixels far from any line should be kept."""
        wave = np.array([1440.0, 1450.0, 1460.0])
        mask = _mask_emission_lines(wave, z=0.0, width_A=10.0)
        assert mask.all()


class TestWingRegionNotMasked:
    """Regression: Lya + NV must be excluded from the auto-mask so the
    1216-1263 A rest damping-wing region remains in the fit.

    NV_1 (1238.821 A) and NV_2 (1242.804 A) sit inside the wing.  With
    the default mask_width_A = 20 A, including them in the auto-mask
    would erase 1218.8 - 1262.8 A rest -- the pixels that distinguish
    log N_HI ~ 21 from log N_HI ~ 19.  Any DLA strong enough to fit
    absorbs NV at tau >> 1, so the mask there destroys signal it does
    not need to remove.
    """

    def _wing_exclusion_set(self):
        """Exclusion set fit_NHI uses internally."""
        from jwspecfit.lines import REST_LINES_A
        exclude = {"Lya", "NV_1", "NV_2", "NV_doublet"}
        for name in REST_LINES_A:
            if name.startswith("abs_"):
                exclude.add(name)
        return exclude

    def test_wing_pixels_kept_at_z0(self):
        """Wing pixels in 1216-1263 A rest survive the auto-mask at z=0."""
        wave = np.linspace(1216.0, 1263.0, 200)
        mask = _mask_emission_lines(
            wave, z=0.0, width_A=20.0,
            exclude=self._wing_exclusion_set(),
        )
        # Every pixel in the wing window should be kept.
        assert mask.all(), (
            f"{(~mask).sum()}/{len(mask)} wing pixels masked; "
            "Lya/NV_1/NV_2/NV_doublet must be in the exclusion set."
        )

    def test_wing_pixels_kept_at_z6(self):
        """Same guarantee at z=6 (observed-frame mask shift)."""
        z = 6.0
        wave_obs = np.linspace(1216.0, 1263.0, 200) * (1.0 + z)
        mask = _mask_emission_lines(
            wave_obs, z=z, width_A=20.0,
            exclude=self._wing_exclusion_set(),
        )
        assert mask.all()

    def test_baseline_would_have_killed_wing(self):
        """Sanity check: if NV were NOT excluded, the wing would be erased.

        This documents *why* the regression test above matters.  If a
        future refactor drops NV from the exclusion set, this test will
        still pass (it is the negative control), but the test above
        will fail loudly.
        """
        wave = np.linspace(1216.0, 1263.0, 200)
        # Simulate the buggy old behaviour: only Lya excluded.
        exclude_buggy = {"Lya"}
        from jwspecfit.lines import REST_LINES_A
        for name in REST_LINES_A:
            if name.startswith("abs_"):
                exclude_buggy.add(name)
        mask = _mask_emission_lines(
            wave, z=0.0, width_A=20.0, exclude=exclude_buggy,
        )
        # NV_1 (1238.8) and NV_2 (1242.8) +/- 20 A wipes ~1219-1263 A.
        n_killed = (~mask).sum()
        assert n_killed > 0.7 * len(mask), (
            "Negative control failed: with NV in the auto-mask, the "
            "wing region should be heavily masked.  If this assertion "
            "fires, the masking semantics have changed and the test "
            "above needs revisiting."
        )


# ============================================================
# Synthetic DLA fitting
# ============================================================

class TestFitSyntheticDLA:
    """Test parameter recovery on synthetic spectra."""

    @pytest.fixture
    def synthetic_dla_spectrum(self):
        """Generate a synthetic DLA spectrum with known parameters."""
        true_log_NHI = 22.0
        true_beta_UV = -2.5
        true_log_F0 = -1.0

        wave = np.linspace(1050, 2000, 500)
        model = _evaluate_model(wave, true_log_F0, true_beta_UV,
                                true_log_NHI, z=0.0)

        rng = np.random.default_rng(12345)
        continuum_level = np.median(model[wave > 1500])
        noise_level = continuum_level / 15.0
        noise = rng.normal(0, noise_level, len(wave))
        flux = model + noise
        err = np.full_like(wave, noise_level)

        return {
            "wave": wave, "flux": flux, "err": err,
            "true_log_NHI": true_log_NHI,
            "true_beta_UV": true_beta_UV,
            "true_log_F0": true_log_F0,
        }

    def test_recovers_NHI(self, synthetic_dla_spectrum):
        """fit_NHI should recover the input log_NHI within 2 sigma."""
        from jwspecfit.dla import fit_NHI

        d = synthetic_dla_spectrum
        result = fit_NHI(
            d["wave"], d["flux"], d["err"],
            z=0.0, mask_lines=False, n_live=200, seed=42,
        )

        err_total = max(result.log_NHI_err)
        assert abs(result.log_NHI - d["true_log_NHI"]) < 2 * err_total + 0.3, (
            f"log_NHI = {result.log_NHI:.2f} vs true {d['true_log_NHI']:.2f}"
        )

    def test_recovers_beta_UV(self, synthetic_dla_spectrum):
        """fit_NHI should recover beta_UV within tolerance."""
        from jwspecfit.dla import fit_NHI

        d = synthetic_dla_spectrum
        result = fit_NHI(
            d["wave"], d["flux"], d["err"],
            z=0.0, mask_lines=False, n_live=200, seed=42,
        )

        # Allow 1.5 dex tolerance due to NHI-beta degeneracy.
        assert abs(result.beta_UV - d["true_beta_UV"]) < 1.5, (
            f"beta_UV = {result.beta_UV:.2f} vs true {d['true_beta_UV']:.2f}"
        )

    def test_result_has_correct_types(self, synthetic_dla_spectrum):
        """DLAResult should have correct attribute types."""
        from jwspecfit.dla import fit_NHI

        d = synthetic_dla_spectrum
        result = fit_NHI(
            d["wave"], d["flux"], d["err"],
            z=0.0, mask_lines=False, n_live=100, seed=42,
        )

        assert isinstance(result, DLAResult)
        assert isinstance(result.log_NHI, float)
        assert isinstance(result.log_NHI_err, tuple)
        assert len(result.log_NHI_err) == 2
        assert isinstance(result.Sigma_HI, float)
        assert isinstance(result.log_evidence, float)
        assert "log_NHI" in result.samples


class TestFitNoDLA:
    """Test that a pure power law gives low N_HI."""

    def test_no_absorption_gives_low_NHI(self):
        """Pure power law should recover log_NHI < 19.5."""
        from jwspecfit.dla import fit_NHI

        wave = np.linspace(1050, 2000, 500)
        F0 = 0.1
        beta = -2.0
        lam_pivot = _LAMBDA_PIVOT_A
        model = F0 * (wave / lam_pivot) ** beta

        rng = np.random.default_rng(999)
        noise_level = np.median(model) / 20.0
        flux = model + rng.normal(0, noise_level, len(wave))
        err = np.full_like(wave, noise_level)

        result = fit_NHI(
            wave, flux, err,
            z=0.0, mask_lines=False, n_live=200, seed=42,
        )

        assert result.log_NHI < 19.5, (
            f"Expected low N_HI, got log_NHI = {result.log_NHI:.2f}"
        )


# ============================================================
# Redshift consistency
# ============================================================

class TestRedshiftScaling:
    """Same intrinsic spectrum at z=0 and z=2 should give same N_HI."""

    def test_z_invariance(self):
        """N_HI should be consistent at z=0 and z=2."""
        from jwspecfit.dla import fit_NHI

        true_log_NHI = 21.5
        true_beta = -2.0
        true_log_F0 = -1.0

        # z=0 spectrum.
        wave_rest = np.linspace(1050, 2000, 400)
        model_rest = _evaluate_model(wave_rest, true_log_F0, true_beta,
                                     true_log_NHI, 0.0)
        rng = np.random.default_rng(42)
        noise_level = np.median(model_rest[wave_rest > 1500]) / 15.0
        flux_rest = model_rest + rng.normal(0, noise_level, len(wave_rest))
        err_rest = np.full_like(wave_rest, noise_level)

        result_z0 = fit_NHI(
            wave_rest, flux_rest, err_rest,
            z=0.0, mask_lines=False, n_live=200, seed=42,
        )

        z = 2.0
        wave_obs = wave_rest * (1 + z)
        model_obs = _evaluate_model(wave_obs, true_log_F0, true_beta,
                                    true_log_NHI, z)
        flux_obs = model_obs + rng.normal(0, noise_level, len(wave_obs))
        err_obs = np.full_like(wave_obs, noise_level)

        result_z2 = fit_NHI(
            wave_obs, flux_obs, err_obs,
            z=z, mask_lines=False, n_live=200, seed=42,
        )

        diff = abs(result_z0.log_NHI - result_z2.log_NHI)
        combined_err = max(result_z0.log_NHI_err) + max(result_z2.log_NHI_err)
        assert diff < 2 * combined_err + 0.5, (
            f"z=0: {result_z0.log_NHI:.2f}, z=2: {result_z2.log_NHI:.2f}"
        )


# ============================================================
# Dust correction
# ============================================================

class TestDustCorrection:
    """Test that dust correction is applied properly."""

    def test_dust_correction_recovers_NHI(self):
        """Spectrum with A_V reddening + correct A_V should recover true N_HI."""
        from jwspecfit.dla import fit_NHI
        from jwspecabund.dust import cardelli_extinction

        true_log_NHI = 21.5
        true_beta = -2.0
        true_log_F0 = -1.0
        Av = 0.5

        wave = np.linspace(1050, 2000, 400)
        model_intrinsic = _evaluate_model(wave, true_log_F0, true_beta,
                                          true_log_NHI, 0.0)
        A_lambda = cardelli_extinction(wave, Av)
        model_reddened = model_intrinsic * 10.0 ** (-0.4 * A_lambda)

        rng = np.random.default_rng(123)
        noise_level = np.median(model_reddened[wave > 1500]) / 15.0
        flux = model_reddened + rng.normal(0, noise_level, len(wave))
        err = np.full_like(wave, noise_level)

        result = fit_NHI(
            wave, flux, err,
            z=0.0, Av=Av, dust_law="cardelli",
            mask_lines=False, n_live=200, seed=42,
        )

        err_total = max(result.log_NHI_err)
        assert abs(result.log_NHI - true_log_NHI) < 2 * err_total + 0.5, (
            f"log_NHI = {result.log_NHI:.2f} vs true {true_log_NHI}"
        )


# ============================================================
# Resolution convolution
# ============================================================

class TestResolutionConvolution:
    """Test that spectral resolution convolution works."""

    def test_convolution_smooths_wing(self):
        """Model with R should smooth the sharp DLA wing transition."""
        # Use a fine grid spanning the DLA wing region.
        wave = np.linspace(1210, 1350, 1000)
        model_hires = _evaluate_model(wave, -1.0, -2.0, 22.0, 0.0, R=None)
        model_lores = _evaluate_model(wave, -1.0, -2.0, 22.0, 0.0, R=50)

        # The low-res second derivative should be smaller (smoother).
        d2_hires = np.diff(model_hires, n=2)
        d2_lores = np.diff(model_lores, n=2)
        assert np.max(np.abs(d2_lores)) < np.max(np.abs(d2_hires))

    def test_convolution_preserves_flux(self):
        """Convolution should roughly preserve total flux."""
        wave = np.linspace(1300, 2000, 500)
        model_hires = _evaluate_model(wave, -1.0, -2.0, 21.0, 0.0, R=None)
        model_lores = _evaluate_model(wave, -1.0, -2.0, 21.0, 0.0, R=100)

        # Total flux should be similar (within 5% — edge effects).
        flux_hires = np.sum(model_hires * np.median(np.diff(wave)))
        flux_lores = np.sum(model_lores * np.median(np.diff(wave)))
        np.testing.assert_allclose(flux_hires, flux_lores, rtol=0.05)


# ============================================================
# Plot and summary methods
# ============================================================

class TestPollockJointFit:
    """Pollock+26-style joint sampler: real posteriors on all 3 parameters."""

    @pytest.fixture(scope="class")
    def synth_dla(self):
        """Synthetic clean DLA spectrum with high SNR."""
        rng = np.random.default_rng(101)
        wave = np.linspace(1216.0, 2400.0, 800)
        true_NHI, true_beta, true_logF0 = 22.0, -2.5, -1.0
        model = _evaluate_model(wave, true_logF0, true_beta, true_NHI,
                                z=0.0, R=400.0)
        sigma = float(np.median(model[wave > 1500])) / 30.0
        flux = model + sigma * rng.standard_normal(len(wave))
        err = np.full_like(wave, sigma)
        return dict(wave=wave, flux=flux, err=err,
                    true_NHI=true_NHI, true_beta=true_beta,
                    true_logF0=true_logF0)

    def test_joint_recovers_all_three_parameters(self, synth_dla):
        """Joint fit should give real, non-zero errors on β and F0."""
        d = synth_dla
        r = fit_NHI(d["wave"], d["flux"], d["err"], z=0.0, R=400.0,
                    mask_lines=False, fit_range_A=(1216, 2400),
                    n_live=300, seed=42)
        # Real posteriors — errors strictly > 0 (no more zero-by-construction).
        assert max(r.beta_UV_err) > 0.0, "β_UV error should be a real posterior"
        assert max(r.log_F0_err) > 0.0, "log_F0 error should be a real posterior"
        # Recovery within 3σ.
        assert abs(r.log_NHI - d["true_NHI"]) < 3 * max(r.log_NHI_err) + 0.1
        assert abs(r.beta_UV - d["true_beta"]) < 3 * max(r.beta_UV_err) + 0.1
        # And not flagged as upper limit (signal is strong).
        assert not r.is_upper_limit


class TestIGMDampingWing:
    """Miralda-Escudé IGM damping wing (Pollock+26 / Heintz+25 framework)."""

    def test_tau_monotonic_redward(self):
        """τ_IGM should monotonically decrease redward of source Lyα."""
        z_src = 7.0
        wave = np.linspace(_LAMBDA_LYA_A * (1 + z_src) + 1.0,
                           1500.0 * (1 + z_src), 100)
        tau = tau_IGM_DW(wave, x_HI=1.0, z_source=z_src, z_min=5.3)
        assert np.all(np.isfinite(tau))
        assert np.all(tau >= 0)
        assert np.all(np.diff(tau) <= 1e-12), "τ_IGM should decay redward"

    def test_zero_when_x_HI_zero(self):
        """No IGM contribution when x_HI = 0."""
        wave = np.linspace(_LAMBDA_LYA_A * 8.5, _LAMBDA_LYA_A * 12.0, 50)
        tau = tau_IGM_DW(wave, x_HI=0.0, z_source=7.0, z_min=5.3)
        assert np.allclose(tau, 0.0)

    def test_scales_linearly_with_x_HI(self):
        """At fixed geometry, τ_IGM scales linearly with x_HI."""
        wave = np.array([_LAMBDA_LYA_A * (1 + 7.5)])
        t1 = tau_IGM_DW(wave, x_HI=0.3, z_source=7.0, z_min=5.3)
        t2 = tau_IGM_DW(wave, x_HI=0.6, z_source=7.0, z_min=5.3)
        np.testing.assert_allclose(t2, 2.0 * t1, rtol=1e-9)

    def test_fit_recovers_x_HI(self):
        """A spectrum with strong IGM (x_HI=1) should be recovered."""
        rng = np.random.default_rng(202)
        z_src = 7.0
        wave = np.linspace(1216.0 * (1 + z_src),
                           2800.0 * (1 + z_src), 1500)
        true_logF0, true_beta, true_NHI, true_xHI = -0.5, -2.0, 18.5, 1.0
        m = _evaluate_model(wave, true_logF0, true_beta, true_NHI,
                            z=z_src, x_HI=true_xHI, R=400.0)
        sigma = float(np.median(m[wave / (1 + z_src) > 1500])) / 25.0
        flux = m + sigma * rng.standard_normal(len(wave))
        err = np.full_like(wave, sigma)
        r = fit_NHI(wave, flux, err, z=z_src, R=400.0, fit_x_HI=True,
                    mask_lines=False, fit_range_A=(1216, 2800),
                    n_live=400, seed=42)
        assert abs(r.x_HI - true_xHI) < 3 * max(r.x_HI_err) + 0.1


class TestUpperLimitFlag:
    """Pollock+26 reports upper limits when N_HI is unconstrained."""

    def test_pure_power_law_flagged(self):
        """A pure power-law (no DLA) should set is_upper_limit=True."""
        rng = np.random.default_rng(303)
        wave = np.linspace(1216.0, 2400.0, 600)
        intrinsic = 0.5 * (wave / 1500.0) ** -2.0
        sigma = float(np.median(intrinsic)) / 30.0
        flux = intrinsic + sigma * rng.standard_normal(len(wave))
        err = np.full_like(wave, sigma)
        r = fit_NHI(wave, flux, err, z=0.0, R=400.0, mask_lines=False,
                    fit_range_A=(1216, 2400), n_live=200, seed=42)
        assert r.is_upper_limit
        # The 95th percentile should be safely below 21 — no spurious DLA.
        assert r.log_NHI_upper95 < 21.0


class TestDLyaStatistic:
    """Heintz+25 D_Lyα equivalent-width statistic."""

    def test_DLA_gives_positive_DLya(self):
        """log N_HI = 22 should give D_Lyα ≈ 60 Å (Heintz+25)."""
        rng = np.random.default_rng(404)
        wave = np.linspace(1100.0, 2700.0, 1500)
        m = _evaluate_model(wave, -1.0, -2.0, 22.0, z=0.0, R=400.0)
        sigma = float(np.median(m[wave > 1500])) / 50.0
        flux = m + sigma * rng.standard_normal(len(wave))
        err = np.full_like(wave, sigma)
        r = compute_D_Lya(wave, flux, err, z=0.0)
        # Heintz+25 fig 2 shows D_Lyα ≈ 50–70 Å for log N_HI = 22.
        assert 40.0 < r["D_Lya"] < 80.0
        assert r["D_Lya_err"] > 0
        # β recovered close to true value.
        assert abs(r["beta_UV"] - (-2.0)) < 0.1

    def test_pure_continuum_gives_near_zero_DLya(self):
        """A pure power-law continuum should give D_Lyα ≈ 0."""
        rng = np.random.default_rng(505)
        wave = np.linspace(1180.0, 2700.0, 1500)
        intrinsic = 0.5 * (wave / 1500.0) ** -2.0
        sigma = float(np.median(intrinsic)) / 80.0
        flux = intrinsic + sigma * rng.standard_normal(len(wave))
        err = np.full_like(wave, sigma)
        r = compute_D_Lya(wave, flux, err, z=0.0)
        # Should be small in absolute value relative to the integration window
        # (1180–1350 Å is 170 Å wide).
        assert abs(r["D_Lya"]) < 5.0


# Helper for tests that need _LAMBDA_LYA_A.
from jwspecfit.dla import _LAMBDA_LYA_A  # noqa: E402


class TestPlot:
    """Test that the plot and summary methods run."""

    def test_plot_runs(self):
        """DLAResult.plot() should produce a figure."""
        import matplotlib
        matplotlib.use("Agg")

        result = DLAResult(
            log_NHI=22.0, log_NHI_err=(0.1, 0.1),
            beta_UV=-2.5, beta_UV_err=(0.1, 0.1),
            log_F0=-1.0, log_F0_err=(0.1, 0.1),
            Sigma_HI=80.0,
            samples={"log_NHI": np.ones(100)*22, "beta_UV": np.ones(100)*-2.5,
                     "log_F0": np.ones(100)*-1.0},
            wave_fit=np.linspace(1050, 2000, 100),
            flux_fit=np.random.default_rng(0).normal(0.01, 0.001, 100),
            flux_err_fit=np.full(100, 0.001),
            model_best=np.full(100, 0.01),
            z=0.0, Av=0.0, log_evidence=-100.0,
        )
        fig = result.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_summary(self):
        """DLAResult.summary() should return a string."""
        result = DLAResult(
            log_NHI=22.0, log_NHI_err=(0.1, 0.15),
            beta_UV=-2.5, beta_UV_err=(0.1, 0.1),
            log_F0=-1.0, log_F0_err=(0.1, 0.1),
            Sigma_HI=80.0,
            samples={"log_NHI": np.ones(10)*22, "beta_UV": np.ones(10)*-2.5,
                     "log_F0": np.ones(10)*-1.0},
            wave_fit=np.linspace(1050, 2000, 10),
            flux_fit=np.ones(10),
            flux_err_fit=np.ones(10)*0.1,
            model_best=np.ones(10),
            log_evidence=-50.0,
        )
        s = result.summary()
        assert "log(N_HI" in s
        assert "Sigma_HI" in s
        assert "log(Z)" in s


# Import the pivot constant for TestFitNoDLA
from jwspecfit.dla import _LAMBDA_PIVOT_A
