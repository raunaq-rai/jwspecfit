"""Tests for jwspecfit.broad — broad component fitting and BIC selection."""

import numpy as np
import pytest
import jwspecfit

from jwspecfit import fit_with_broad, read_fits
from jwspecfit.broad import BroadFitResult, _balmer_pixel_mask
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
