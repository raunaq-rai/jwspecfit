"""Tests for jwspecfit.resolution."""

import numpy as np
import pytest

from jwspecfit.resolution import R_prism, R_grating, resolve_R, sigma_inst_A, sigma_inst_kms


class TestRPrism:
    def test_values_clipped(self):
        lam = np.array([0.6, 1.0, 3.0, 5.0])
        R = R_prism(lam)
        assert np.all(R >= 30)
        assert np.all(R <= 300)

    def test_monotonic_increase(self):
        lam = np.linspace(0.6, 5.0, 100)
        R = R_prism(lam)
        # R should generally increase with wavelength (quadratic).
        assert R[-1] > R[0]

    def test_at_1um(self):
        # At 1.0 µm: R = 50 + 0 + 0 = 50
        np.testing.assert_approx_equal(R_prism(np.array([1.0]))[0], 50.0)


class TestRGrating:
    def test_medium(self):
        assert R_grating("G395M") == 1000.0
        assert R_grating("G235M") == 1000.0

    def test_high(self):
        assert R_grating("G395H") == 2700.0

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            R_grating("UNKNOWN")


class TestResolveR:
    def test_grating(self):
        lam = np.array([1.0, 2.0, 3.0])
        R = resolve_R(lam, grating="PRISM")
        assert R.shape == lam.shape

    def test_numeric_R(self):
        lam = np.array([1.0, 2.0])
        R = resolve_R(lam, R=150.0)
        np.testing.assert_array_equal(R, [150.0, 150.0])

    def test_callable_R(self):
        lam = np.array([1.0, 2.0])
        R = resolve_R(lam, R=lambda x: x * 100)
        np.testing.assert_array_equal(R, [100.0, 200.0])

    def test_R_overrides_grating(self):
        lam = np.array([1.0])
        R = resolve_R(lam, grating="PRISM", R=500.0)
        assert R[0] == 500.0

    def test_neither_raises(self):
        with pytest.raises(ValueError):
            resolve_R(np.array([1.0]))


class TestSigmaInst:
    def test_sigma_positive(self):
        lam = np.linspace(1.0, 5.0, 50)
        sig = sigma_inst_A(lam, grating="PRISM")
        assert np.all(sig > 0)

    def test_sigma_kms_constant_for_grating(self):
        lam = np.linspace(3.0, 5.0, 50)
        sig_kms = sigma_inst_kms(lam, grating="G395M")
        # Constant R → constant σ_v.
        np.testing.assert_array_almost_equal(sig_kms, sig_kms[0])
