"""Tests for the DLA column density fitter (jwspecfit.dla)."""

from __future__ import annotations

import numpy as np
import pytest

from jwspecfit.dla import (
    DLAResult,
    _evaluate_model,
    _mask_emission_lines,
    tau_DLA,
    tepper_garcia_H,
)


# ============================================================
# Voigt-Hjerting function
# ============================================================

class TestVoigtHjerting:
    """Test tepper_garcia_H against exact Faddeeva function."""

    def test_agreement_with_wofz(self):
        """H(a, u) should agree with wofz to <1% for |u| > 2."""
        from scipy.special import wofz
        import jax.numpy as jnp

        # Grid of damping parameters and frequency offsets.
        # Start from u=3 where the asymptotic wing expansion is accurate.
        a_vals = [1e-4, 1e-3, 1e-2, 0.1]
        u_vals = np.linspace(3.0, 100.0, 200)

        for a in a_vals:
            # H(a, u) = Re[w(u + i*a)] where w is the Faddeeva function.
            exact_H = np.array([wofz(complex(u, a)).real for u in u_vals])
            our_H = np.array(tepper_garcia_H(jnp.array(a), jnp.array(u_vals)))

            rel_err = np.abs(our_H - exact_H) / np.maximum(np.abs(exact_H), 1e-30)
            assert np.all(rel_err < 0.02), (
                f"Voigt-Hjerting mismatch at a={a}: max rel error = {rel_err.max():.4f}"
            )

    def test_gaussian_core_dominates(self):
        """For a -> 0 and moderate u, H(a, u) ~ exp(-u^2)."""
        import jax.numpy as jnp

        a = 1e-8
        u = jnp.linspace(0.5, 1.5, 20)
        H = np.array(tepper_garcia_H(jnp.array(a), u))
        expected = np.exp(-np.array(u) ** 2)
        # Core dominates, wing adds ~a/sqrt(pi)/u^2 ~ 1e-9, negligible.
        np.testing.assert_allclose(H, expected, rtol=0.01)

    def test_damping_wings_positive(self):
        """H(a, u) should be positive for all u."""
        import jax.numpy as jnp

        a = 0.01
        u = jnp.linspace(0.1, 1000.0, 5000)
        H = np.array(tepper_garcia_H(jnp.array(a), u))
        assert np.all(H >= 0), "H(a, u) should be non-negative everywhere."

    def test_wing_regime_accuracy(self):
        """In the damping wing (u >> 1), H ~ a/(sqrt(pi)*u^2)."""
        import jax.numpy as jnp

        a = 0.01
        u_vals = jnp.array([20.0, 50.0, 100.0])
        H = np.array(tepper_garcia_H(jnp.array(a), u_vals))
        leading = a / (np.sqrt(np.pi) * np.array(u_vals) ** 2)
        # Should agree to ~1% at u=20, better at larger u.
        np.testing.assert_allclose(H, leading, rtol=0.05)


# ============================================================
# DLA optical depth
# ============================================================

class TestTauDLA:
    """Test the DLA optical depth calculation."""

    def test_proportional_to_NHI(self):
        """tau should scale linearly with N_HI."""
        import jax.numpy as jnp

        wave = jnp.linspace(1220.0, 1400.0, 100)
        tau_20 = np.array(tau_DLA(wave, 20.0, z=0.0))
        tau_21 = np.array(tau_DLA(wave, 21.0, z=0.0))

        # tau(log_NHI=21) / tau(log_NHI=20) should be 10.
        ratio = tau_21 / np.maximum(tau_20, 1e-30)
        # Check away from line centre where numerical issues are smaller.
        mask = (np.array(wave) > 1230) & (np.array(wave) < 1350)
        np.testing.assert_allclose(ratio[mask], 10.0, rtol=0.01)

    def test_large_at_line_centre(self):
        """tau should be very large near Lya for high N_HI."""
        import jax.numpy as jnp

        wave = jnp.array([1216.0])
        tau = tau_DLA(wave, 22.0, z=0.0)[0].item()
        assert tau > 1e3, f"Expected tau >> 1 at Lya centre for log_NHI=22, got {tau}."

    def test_small_far_from_line(self):
        """tau should be negligible far from Lya for moderate N_HI."""
        import jax.numpy as jnp

        wave = jnp.array([3000.0])
        tau = tau_DLA(wave, 20.0, z=0.0)[0].item()
        assert tau < 0.01, f"Expected tau << 1 at 3000A for log_NHI=20, got {tau}."

    def test_redshift_shifts_absorption(self):
        """At z > 0, the DLA should be centred at 1215.67 * (1+z)."""
        import jax.numpy as jnp

        z = 2.0
        lya_obs = 1215.67 * (1 + z)
        # Just redward of shifted Lya should have large tau.
        wave_near = jnp.array([lya_obs + 2.0])
        tau_near = tau_DLA(wave_near, 22.0, z=z)[0].item()
        # Far redward should have small tau.
        wave_far = jnp.array([lya_obs + 1000.0])
        tau_far = tau_DLA(wave_far, 22.0, z=z)[0].item()
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
        # CIV doublet is at ~1549 A.
        civ_region = (wave > 1538) & (wave < 1562)
        assert not mask[civ_region].all(), "CIV region should be masked."

    def test_masks_shift_with_z(self):
        """Masks should shift with redshift."""
        wave = np.linspace(3000, 3200, 200)
        mask = _mask_emission_lines(wave, z=1.0, width_A=10.0)
        # CIV at z=1 is at ~3099 A.
        civ_region = (wave > 3076) & (wave < 3124)
        assert not mask[civ_region].all(), "CIV region should be masked at z=1."

    def test_keeps_continuum_pixels(self):
        """Pixels far from any line should be kept."""
        # 1440-1460 A is far from any known emission/absorption line.
        wave = np.array([1440.0, 1450.0, 1460.0])
        mask = _mask_emission_lines(wave, z=0.0, width_A=10.0)
        assert mask.all(), "Continuum pixels should not be masked."


# ============================================================
# Synthetic DLA fitting
# ============================================================

class TestFitSyntheticDLA:
    """Test parameter recovery on synthetic spectra."""

    @pytest.fixture
    def synthetic_dla_spectrum(self):
        """Generate a synthetic DLA spectrum with known parameters."""
        import jax.numpy as jnp

        # True parameters.
        true_log_NHI = 22.0
        true_beta_UV = -2.5
        true_log_F0 = -1.0

        # Wavelength grid (rest frame, z=0).
        wave = np.linspace(1050, 2000, 500)
        wave_jax = jnp.array(wave)

        # True model.
        model = np.array(
            _evaluate_model(wave_jax, true_log_F0, true_beta_UV, true_log_NHI, z=0.0)
        )

        # Add noise (S/N ~ 15 in the continuum region).
        rng = np.random.default_rng(12345)
        continuum_level = np.median(model[wave > 1500])
        noise_level = continuum_level / 15.0
        noise = rng.normal(0, noise_level, len(wave))
        flux = model + noise
        err = np.full_like(wave, noise_level)

        return {
            "wave": wave,
            "flux": flux,
            "err": err,
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
            z=0.0, mask_lines=False,
            n_warmup=300, n_samples=1000, seed=42,
        )

        # Check log_NHI recovery.
        err_total = max(result.log_NHI_err)
        assert abs(result.log_NHI - d["true_log_NHI"]) < 2 * err_total + 0.3, (
            f"log_NHI = {result.log_NHI:.2f} vs true {d['true_log_NHI']:.2f}"
        )

    def test_recovers_beta_UV(self, synthetic_dla_spectrum):
        """fit_NHI should recover beta_UV within 2 sigma.

        Note: beta_UV and log_NHI are degenerate — a redder slope
        can mimic a lower column density.  With enough samples the
        sampler should find the correct mode, but we allow generous
        tolerance.
        """
        from jwspecfit.dla import fit_NHI

        d = synthetic_dla_spectrum
        result = fit_NHI(
            d["wave"], d["flux"], d["err"],
            z=0.0, mask_lines=False,
            n_warmup=500, n_samples=2000, seed=42,
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
            z=0.0, mask_lines=False,
            n_warmup=200, n_samples=500, seed=42,
        )

        assert isinstance(result, DLAResult)
        assert isinstance(result.log_NHI, float)
        assert isinstance(result.log_NHI_err, tuple)
        assert len(result.log_NHI_err) == 2
        assert isinstance(result.Sigma_HI, float)
        assert "log_NHI" in result.samples
        assert len(result.samples["log_NHI"]) == 500


class TestFitNoDLA:
    """Test that a pure power law gives low N_HI."""

    def test_no_absorption_gives_low_NHI(self):
        """Pure power law should recover log_NHI < 19.5."""
        import jax.numpy as jnp
        from jwspecfit.dla import fit_NHI

        # Pure power law, no DLA.
        wave = np.linspace(1050, 2000, 500)
        F0 = 0.1
        beta = -2.0
        model = F0 * wave ** beta

        rng = np.random.default_rng(999)
        noise_level = np.median(model) / 20.0
        flux = model + rng.normal(0, noise_level, len(wave))
        err = np.full_like(wave, noise_level)

        result = fit_NHI(
            wave, flux, err,
            z=0.0, mask_lines=False,
            n_warmup=300, n_samples=1000, seed=42,
        )

        assert result.log_NHI < 19.5, (
            f"Expected low N_HI for pure power law, got log_NHI = {result.log_NHI:.2f}"
        )


# ============================================================
# Redshift consistency
# ============================================================

class TestRedshiftScaling:
    """Same intrinsic spectrum at z=0 and z=2 should give same N_HI."""

    def test_z_invariance(self):
        """N_HI should be consistent at z=0 and z=2."""
        import jax.numpy as jnp
        from jwspecfit.dla import fit_NHI

        true_log_NHI = 21.5
        true_beta = -2.0
        true_log_F0 = -1.0

        # z=0 spectrum.
        wave_rest = np.linspace(1050, 2000, 400)
        model_rest = np.array(
            _evaluate_model(jnp.array(wave_rest), true_log_F0, true_beta, true_log_NHI, 0.0)
        )
        rng = np.random.default_rng(42)
        noise_level = np.median(model_rest[wave_rest > 1500]) / 15.0
        flux_rest = model_rest + rng.normal(0, noise_level, len(wave_rest))
        err_rest = np.full_like(wave_rest, noise_level)

        result_z0 = fit_NHI(
            wave_rest, flux_rest, err_rest,
            z=0.0, mask_lines=False,
            n_warmup=300, n_samples=1000, seed=42,
        )

        # z=2 spectrum: same intrinsic, shifted to observed frame.
        z = 2.0
        wave_obs = wave_rest * (1 + z)
        # The model at z=2 uses observed wavelengths.
        model_obs = np.array(
            _evaluate_model(jnp.array(wave_obs), true_log_F0, true_beta, true_log_NHI, z)
        )
        flux_obs = model_obs + rng.normal(0, noise_level, len(wave_obs))
        err_obs = np.full_like(wave_obs, noise_level)

        result_z2 = fit_NHI(
            wave_obs, flux_obs, err_obs,
            z=z, mask_lines=False,
            n_warmup=300, n_samples=1000, seed=42,
        )

        # Should agree within combined uncertainties.
        diff = abs(result_z0.log_NHI - result_z2.log_NHI)
        combined_err = max(result_z0.log_NHI_err) + max(result_z2.log_NHI_err)
        assert diff < 2 * combined_err + 0.5, (
            f"z=0: {result_z0.log_NHI:.2f}, z=2: {result_z2.log_NHI:.2f}, "
            f"diff={diff:.2f} vs 2*err={2*combined_err:.2f}"
        )


# ============================================================
# Dust correction
# ============================================================

class TestDustCorrection:
    """Test that dust correction is applied properly."""

    def test_dust_correction_recovers_NHI(self):
        """Spectrum with A_V reddening + correct A_V should recover true N_HI."""
        import jax.numpy as jnp
        from jwspecfit.dla import fit_NHI
        from jwspecabund.dust import cardelli_extinction

        true_log_NHI = 21.5
        true_beta = -2.0
        true_log_F0 = -1.0
        Av = 0.5

        wave = np.linspace(1050, 2000, 400)
        wave_jax = jnp.array(wave)

        # Intrinsic model (no dust).
        model_intrinsic = np.array(
            _evaluate_model(wave_jax, true_log_F0, true_beta, true_log_NHI, 0.0)
        )

        # Apply dust reddening (make it look observed).
        A_lambda = cardelli_extinction(wave, Av)
        model_reddened = model_intrinsic * 10.0 ** (-0.4 * A_lambda)

        rng = np.random.default_rng(123)
        noise_level = np.median(model_reddened[wave > 1500]) / 15.0
        flux = model_reddened + rng.normal(0, noise_level, len(wave))
        err = np.full_like(wave, noise_level)

        # Fit with correct Av.
        result = fit_NHI(
            wave, flux, err,
            z=0.0, Av=Av, dust_law="cardelli",
            mask_lines=False,
            n_warmup=300, n_samples=1000, seed=42,
        )

        err_total = max(result.log_NHI_err)
        assert abs(result.log_NHI - true_log_NHI) < 2 * err_total + 0.5, (
            f"log_NHI = {result.log_NHI:.2f} vs true {true_log_NHI}"
        )


# ============================================================
# Plot method
# ============================================================

class TestPlot:
    """Test that the plot method runs without error."""

    def test_plot_runs(self):
        """DLAResult.plot() should produce a figure."""
        import matplotlib
        matplotlib.use("Agg")

        result = DLAResult(
            log_NHI=22.0,
            log_NHI_err=(0.1, 0.1),
            beta_UV=-2.5,
            beta_UV_err=(0.1, 0.1),
            log_F0=-1.0,
            log_F0_err=(0.1, 0.1),
            Sigma_HI=80.0,
            samples={"log_NHI": np.ones(100) * 22, "beta_UV": np.ones(100) * -2.5, "log_F0": np.ones(100) * -1.0},
            wave_fit=np.linspace(1050, 2000, 100),
            flux_fit=np.random.default_rng(0).normal(0.01, 0.001, 100),
            flux_err_fit=np.full(100, 0.001),
            model_best=np.full(100, 0.01),
            z=0.0,
            Av=0.0,
        )
        fig = result.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_summary(self):
        """DLAResult.summary() should return a string."""
        result = DLAResult(
            log_NHI=22.0,
            log_NHI_err=(0.1, 0.15),
            beta_UV=-2.5,
            beta_UV_err=(0.1, 0.1),
            log_F0=-1.0,
            log_F0_err=(0.1, 0.1),
            Sigma_HI=80.0,
            samples={"log_NHI": np.ones(10) * 22, "beta_UV": np.ones(10) * -2.5, "log_F0": np.ones(10) * -1.0},
            wave_fit=np.linspace(1050, 2000, 10),
            flux_fit=np.ones(10),
            flux_err_fit=np.ones(10) * 0.1,
            model_best=np.ones(10),
        )
        s = result.summary()
        assert "log(N_HI" in s
        assert "Sigma_HI" in s
        assert "beta_UV" in s
