"""Tests for Lyα fitting — asymmetric Gaussian model and IGM transmission."""

import numpy as np
import pytest

from jwspecfit.lyman_alpha import igm_transmission
from jwspecfit.models import asymmetric_gaussian


class TestIGMTransmission:
    def test_full_transmission_redward(self):
        """Redward of Lyα at source z, transmission should be 1."""
        wave = np.linspace(10000, 15000, 100)  # Well redward of Lyα at z=6
        T = igm_transmission(wave, z_source=6.0)
        np.testing.assert_array_almost_equal(T, 1.0)

    def test_reduced_transmission_blueward(self):
        """Blueward of Lyα at source z, transmission should be < 1."""
        wave = np.linspace(6000, 8000, 100)
        T = igm_transmission(wave, z_source=6.0)
        assert np.all(T < 1.0)
        assert np.all(T >= 0.0)

    def test_higher_z_more_absorption(self):
        """Higher redshift should have more IGM absorption."""
        wave = np.array([7000.0])
        T_z4 = igm_transmission(wave, z_source=4.0)
        T_z6 = igm_transmission(wave, z_source=6.0)
        assert T_z6[0] <= T_z4[0]

    def test_bounded(self):
        wave = np.linspace(3000, 15000, 500)
        T = igm_transmission(wave, z_source=7.0)
        assert np.all(T >= 0.0)
        assert np.all(T <= 1.0)


class TestAsymmetricGaussian:
    """Tests for the asymmetric Gaussian profile (Bolan+2025 form)."""

    def test_symmetric_when_alpha_zero(self):
        """With α=0, should be a symmetric Gaussian with peak A_peak."""
        wave = np.linspace(4900, 5100, 400)
        profile = asymmetric_gaussian(wave, 1.0, 5000.0, 20.0, 0.0)
        assert profile.shape == wave.shape
        # Peak should be at mu.
        idx_peak = np.argmax(profile)
        assert abs(wave[idx_peak] - 5000.0) < 1.0
        # Peak value should be A_peak (since erf(0)=0, so 1+erf=1).
        assert abs(profile[idx_peak] - 1.0) < 0.01

    def test_red_asymmetric_with_positive_alpha(self):
        """Positive α should produce a red-asymmetric profile."""
        wave = np.linspace(4900, 5200, 600)
        prof_sym = asymmetric_gaussian(wave, 1.0, 5000.0, 20.0, 0.0)
        prof_asym = asymmetric_gaussian(wave, 1.0, 5000.0, 20.0, 5.0)
        # The asymmetric profile peak should shift redward.
        assert np.argmax(prof_asym) > np.argmax(prof_sym)
        # Red side should have more flux than the symmetric case.
        red = wave > 5050
        assert np.sum(prof_asym[red]) > np.sum(prof_sym[red])

    def test_peak_higher_than_a_peak_with_alpha(self):
        """With α>0, the peak exceeds A_peak (erf term > 1 at peak)."""
        wave = np.linspace(4900, 5200, 600)
        profile = asymmetric_gaussian(wave, 1.0, 5000.0, 20.0, 5.0)
        # Peak should be > A_peak=1.0 because at the peak, erf > 0.
        assert np.max(profile) > 1.0

    def test_zero_sigma_returns_zeros(self):
        wave = np.linspace(4990, 5010, 20)
        profile = asymmetric_gaussian(wave, 1.0, 5000.0, 0.0, 0.0)
        np.testing.assert_array_equal(profile, 0.0)

    def test_flux_positive_definite(self):
        """Profile should be non-negative everywhere for α≥0."""
        wave = np.linspace(4800, 5200, 1000)
        for alpha in [0.0, 1.0, 5.0, 15.0, 30.0]:
            profile = asymmetric_gaussian(wave, 1.0, 5000.0, 20.0, alpha)
            assert np.all(profile >= 0), f"Negative values at α={alpha}"


class TestLyaFitIntegration:
    """Integration tests for Lyα fitting in the full pipeline."""

    @pytest.fixture
    def synthetic_lya_spectrum(self):
        """Create a synthetic rest-frame Lyα spectrum on a 0.5 Å grid."""
        from jwspecfit.io import Spectrum

        rng = np.random.default_rng(42)
        wave_A = np.arange(1180, 1260, 0.5)
        wave_um = wave_A * 1e-4

        # Asymmetric Gaussian Lyα profile in f_lambda.
        A_peak = 5e-17
        mu = 1216.5
        sigma = 1.5
        alpha = 5.0
        lya_flam = asymmetric_gaussian(wave_A, A_peak, mu, sigma, alpha)

        # Flat continuum.
        cont_flam = np.full_like(wave_A, 1e-18)

        # Convert to µJy.
        C_CGS = 2.99792458e10
        lam_cm = wave_um * 1e-4
        total_flam = lya_flam + cont_flam
        fnu_cgs = total_flam * 1e8 * lam_cm**2 / C_CGS
        flux_ujy = fnu_cgs / 1e-29

        noise = 0.02 * np.nanmax(flux_ujy)
        err_ujy = np.full_like(flux_ujy, noise)
        flux_ujy += rng.normal(0, noise, len(flux_ujy))

        return Spectrum(wave_um=wave_um, flux_ujy=flux_ujy, err_ujy=err_ujy)

    def test_lya_params_length_4(self, synthetic_lya_spectrum):
        """lya_params should have exactly 4 elements."""
        from jwspecfit.fitter import fit_lines

        result = fit_lines(
            synthetic_lya_spectrum, z=0.0, grating="G140M",
            lines=["Lya"], n_boot=0, lya_break=True,
            moving_average=25,
        )
        assert result.lya_params is not None
        assert len(result.lya_params) == 4

    def test_lya_params_none_without_lya(self, synthetic_lya_spectrum):
        """lya_params should be None when Lya not in line list."""
        from jwspecfit.fitter import fit_lines

        result = fit_lines(
            synthetic_lya_spectrum, z=0.0, grating="G140M",
            lines=["NV_1", "NV_2"], n_boot=0,
        )
        assert result.lya_params is None

    def test_lya_fit_recovers_params(self, synthetic_lya_spectrum):
        """Fit should approximately recover the input asymmetric Gaussian."""
        from jwspecfit.fitter import fit_lines

        result = fit_lines(
            synthetic_lya_spectrum, z=0.0, grating="G140M",
            lines=["Lya"], n_boot=0, lya_break=True,
            moving_average=25,
        )
        lp = result.lya_params
        # Check centroid is near 1216.5 Å (within 2 Å).
        assert abs(lp[1] - 1216.5) < 2.0, f"mu={lp[1]:.1f}, expected ~1216.5"
        # Check sigma is reasonable (0.5-5 Å).
        assert 0.5 < lp[2] < 5.0, f"sigma={lp[2]:.2f}"
        # Check alpha is positive (red-asymmetric).
        assert lp[3] >= 0, f"alpha={lp[3]:.1f}"
        # Check flux is positive.
        assert result.lines["Lya"].flux > 0

    def test_lya_in_line_results(self, synthetic_lya_spectrum):
        """Lya should appear in the LineResult dict."""
        from jwspecfit.fitter import fit_lines

        result = fit_lines(
            synthetic_lya_spectrum, z=0.0, grating="G140M",
            lines=["Lya"], n_boot=0, lya_break=True,
            moving_average=25,
        )
        assert "Lya" in result.lines
        lr = result.lines["Lya"]
        assert lr.name == "Lya"
        assert lr.flux > 0
        assert np.isfinite(lr.centroid_A)
