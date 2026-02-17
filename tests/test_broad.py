"""Tests for jwspecfit.broad — broad component fitting and BIC selection."""

import numpy as np
import pytest

from jwspecfit import fit_with_broad, read_fits
from jwspecfit.broad import BroadFitResult, _balmer_pixel_mask
from .conftest import PRISM_FITS, G395M_FITS


class TestBalmerPixelMask:
    def test_mask_selects_ha_hb(self):
        """Mask should select pixels around Hα and Hβ at given redshift."""
        z = 5.0
        # Hα at 6566.4 Å → obs ~ 3.9398 µm; Hβ at 4864.0 Å → obs ~ 2.9184 µm
        wave_um = np.linspace(2.0, 5.0, 1000)
        mask = _balmer_pixel_mask(wave_um, z)
        assert mask.any(), "Mask should select some pixels"
        assert not mask.all(), "Mask should not select all pixels"
        # Check that selected pixels are near the expected wavelengths.
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
    def test_narrow_mode(self, prism_spectrum):
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", mode="off", n_boot=0,
        )
        assert isinstance(result, BroadFitResult)
        assert result.selected_model == "narrow"
        assert result.best_fit.success

    def test_auto_mode(self, grating_spectrum):
        result = fit_with_broad(
            grating_spectrum, z=5.0, grating="G395M", mode="auto", n_boot=0,
            n_boot_bic=0,
        )
        assert result.selected_model in ("narrow", "broad1", "broad2", "both")

    def test_bic_values(self, prism_spectrum):
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", mode="auto", n_boot=0,
            n_boot_bic=0,
        )
        assert np.isfinite(result.bic_narrow)

    def test_forced_broad1(self, prism_spectrum):
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", mode="broad1", n_boot=0,
        )
        assert result.selected_model == "broad1"
        broad_lines = [n for n in result.best_fit.line_names if "BROAD" in n]
        assert len(broad_lines) > 0

    def test_all_fits_dict(self, prism_spectrum):
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", mode="auto", n_boot=0,
            n_boot_bic=0,
        )
        assert "narrow" in result.all_fits

    def test_bic_bootstrap_auto(self, grating_spectrum):
        """Bootstrap BIC should populate bic_bootstrap distributions."""
        result = fit_with_broad(
            grating_spectrum, z=5.0, grating="G395M", mode="auto",
            n_boot=0, n_boot_bic=5, n_jobs=1, snr_threshold=0.0,
        )
        assert result.selected_model in ("narrow", "broad1", "broad2", "both")
        assert len(result.bic_bootstrap) > 0
        for variant, dist in result.bic_bootstrap.items():
            assert len(dist) == 5
            assert np.all(np.isfinite(dist))
        # Median BIC values should be finite.
        assert np.isfinite(result.bic_narrow)

    def test_bic_bootstrap_empty_when_off(self, prism_spectrum):
        """Bootstrap BIC should be empty when mode='off'."""
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", mode="off", n_boot=0,
        )
        assert result.bic_bootstrap == {}

    def test_bic_bootstrap_empty_when_zero(self, grating_spectrum):
        """Bootstrap BIC should be empty when n_boot_bic=0."""
        result = fit_with_broad(
            grating_spectrum, z=5.0, grating="G395M", mode="auto",
            n_boot=0, n_boot_bic=0,
        )
        assert result.bic_bootstrap == {}
