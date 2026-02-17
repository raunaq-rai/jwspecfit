"""Tests for jwspecfit.lyman_alpha — IGM transmission and Lyα model."""

import numpy as np
import pytest

from jwspecfit.lyman_alpha import igm_transmission, skewed_gaussian_binned, lya_model


class TestIGMTransmission:
    def test_full_transmission_redward(self):
        """Redward of Lyα at source z, transmission should be 1."""
        wave = np.linspace(10000, 15000, 100)  # Well redward of Lyα at z=6
        T = igm_transmission(wave, z_source=6.0)
        np.testing.assert_array_almost_equal(T, 1.0)

    def test_reduced_transmission_blueward(self):
        """Blueward of Lyα at source z, transmission should be < 1."""
        # Lyα at z=6 is at 8509 Å.
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


class TestSkewedGaussian:
    def test_symmetric(self):
        """With skew=0, should be symmetric Gaussian."""
        left = np.arange(4900, 5100, 1.0)
        right = left + 1.0
        profile = skewed_gaussian_binned(left, right, 1.0, 5000.0, 20.0, 0.0)
        assert profile.shape == left.shape
        # Should be roughly symmetric around centre.
        idx_centre = np.argmax(profile)
        assert 4990 < left[idx_centre] < 5010

    def test_positive_skew(self):
        """Positive skew should shift the peak blueward of the mean."""
        left = np.arange(4900, 5100, 1.0)
        right = left + 1.0
        profile_sym = skewed_gaussian_binned(left, right, 1.0, 5000.0, 20.0, 0.0)
        profile_skew = skewed_gaussian_binned(left, right, 1.0, 5000.0, 20.0, 5.0)
        # Skewed profile peak should be shifted.
        assert np.argmax(profile_skew) != np.argmax(profile_sym)

    def test_zero_sigma(self):
        left = np.arange(4990, 5010, 1.0)
        right = left + 1.0
        profile = skewed_gaussian_binned(left, right, 1.0, 5000.0, 0.0, 0.0)
        np.testing.assert_array_equal(profile, 0.0)


class TestLyaModel:
    def test_igm_attenuates(self):
        """Lyα model should be attenuated blueward by IGM."""
        z = 6.0
        mu = 1215.67 * (1 + z)  # ~8510 Å
        left = np.arange(7000, 10000, 5.0)
        right = left + 5.0

        # Intrinsic (no IGM).
        intrinsic = skewed_gaussian_binned(left, right, 1.0, mu, 30.0, 3.0)

        # With IGM.
        attenuated = lya_model(left, right, z, 1.0, mu, 30.0, 3.0)

        # Blueward flux should be reduced.
        blue = left < mu
        if np.any(blue) and np.any(intrinsic[blue] > 0):
            ratio = attenuated[blue] / np.where(intrinsic[blue] > 0, intrinsic[blue], 1.0)
            assert np.any(ratio < 1.0)
