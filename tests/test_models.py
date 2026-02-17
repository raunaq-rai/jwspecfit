"""Tests for jwspecfit.models."""

import numpy as np
import pytest

from jwspecfit.models import gaussian_binned, build_model, pixel_weight


class TestGaussianBinned:
    def test_unit_area(self):
        """Integrated profile should sum to ~1 for fine bins."""
        left = np.arange(4900, 5100, 0.1)
        right = left + 0.1
        profile = gaussian_binned(left, right, mu_A=5000.0, sigma_A=10.0)
        # Area = sum(profile × bin_width)
        area = np.sum(profile * 0.1)
        np.testing.assert_approx_equal(area, 1.0, significant=3)

    def test_peak_at_centre(self):
        left = np.arange(4980, 5020, 0.5)
        right = left + 0.5
        profile = gaussian_binned(left, right, mu_A=5000.0, sigma_A=5.0)
        peak_idx = np.argmax(profile)
        # Peak should be near the centre bin.
        assert 4998.0 < left[peak_idx] < 5002.0

    def test_zero_sigma(self):
        left = np.arange(4990, 5010, 1.0)
        right = left + 1.0
        profile = gaussian_binned(left, right, mu_A=5000.0, sigma_A=0.0)
        np.testing.assert_array_equal(profile, 0.0)

    def test_negative_sigma(self):
        left = np.arange(4990, 5010, 1.0)
        right = left + 1.0
        profile = gaussian_binned(left, right, mu_A=5000.0, sigma_A=-5.0)
        np.testing.assert_array_equal(profile, 0.0)

    def test_coarse_bins(self):
        """Even with wide bins, area should be conserved."""
        left = np.arange(4000, 6000, 50.0)
        right = left + 50.0
        profile = gaussian_binned(left, right, mu_A=5000.0, sigma_A=100.0)
        area = np.sum(profile * 50.0)
        np.testing.assert_approx_equal(area, 1.0, significant=2)


class TestBuildModel:
    def test_single_line(self):
        edges = np.arange(4900, 5101, 1.0)
        params = np.array([1.0, 5000.0, 10.0])  # A, mu, sigma
        model = build_model(params, edges, n_lines=1)
        assert model.shape == (200,)
        assert np.max(model) > 0

    def test_two_lines(self):
        edges = np.arange(4900, 5201, 1.0)
        params = np.array([1.0, 0.5, 5000.0, 5100.0, 10.0, 10.0])
        model = build_model(params, edges, n_lines=2)
        assert model.shape == (300,)
        # Should have two peaks.
        assert np.sum(model > 0.001) > 10


class TestPixelWeight:
    def test_uniform_pixels(self):
        dlam = np.full(100, 5.0)
        w = pixel_weight(dlam)
        np.testing.assert_array_almost_equal(w, 1.0)

    def test_narrow_pixel_downweighted(self):
        dlam = np.full(100, 5.0)
        dlam[50] = 1.0
        w = pixel_weight(dlam)
        assert w[50] > w[49]  # Narrow pixel gets higher weight (med/dlam > 1)
