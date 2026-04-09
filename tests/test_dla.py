"""Tests for the DLA column density fitter (jwspecfit.dla)."""

from __future__ import annotations

import numpy as np
import pytest

from jwspecfit.dla import (
    DLAResult,
    _evaluate_model,
    _mask_emission_lines,
    tau_DLA,
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
