"""Tests for jwspecfit.fitter — core fitting engine."""

import numpy as np
import pytest

from jwspecfit import fit_lines, read_fits
from jwspecfit.fitter import FitResult, LineResult
from .conftest import PRISM_FITS, G395M_FITS


class TestFitLinesPrism:
    def test_prism_fit_succeeds(self, prism_spectrum):
        result = fit_lines(prism_spectrum, z=6.0, grating="PRISM", n_boot=0)
        assert isinstance(result, FitResult)
        assert result.success

    def test_has_lines(self, prism_spectrum):
        result = fit_lines(prism_spectrum, z=6.0, grating="PRISM", n_boot=0)
        assert len(result.lines) > 0
        assert "OIII_5007" in result.lines

    def test_model_shape(self, prism_spectrum):
        result = fit_lines(prism_spectrum, z=6.0, grating="PRISM", n_boot=0)
        assert result.model_flux.shape == prism_spectrum.flux_ujy.shape
        assert result.continuum.shape == prism_spectrum.flux_ujy.shape
        assert result.residuals.shape == prism_spectrum.flux_ujy.shape

    def test_bootstrap_errors(self, prism_spectrum):
        """Test that bootstrap produces non-zero errors."""
        result = fit_lines(
            prism_spectrum, z=6.0, grating="PRISM",
            lines=["OIII_5007", "HBETA"], n_boot=20,
        )
        assert result.success
        for name, lr in result.lines.items():
            if lr.flux > 0:
                assert lr.flux_err > 0


class TestFitLinesGrating:
    def test_grating_fit_succeeds(self, grating_spectrum):
        result = fit_lines(grating_spectrum, z=5.0, grating="G395M", n_boot=0)
        assert result.success

    def test_chi2_reasonable(self, grating_spectrum):
        result = fit_lines(grating_spectrum, z=5.0, grating="G395M", n_boot=0)
        assert result.chi2 < 100


class TestFitLinesCustomR:
    def test_numeric_R(self, prism_spectrum):
        result = fit_lines(prism_spectrum, z=6.0, R=100.0, n_boot=0)
        assert result.success

    def test_no_grating_no_R_auto_detects(self, prism_spectrum):
        spec = prism_spectrum.copy()
        spec.grating = None
        spec.R = None
        result = fit_lines(spec, z=6.0, n_boot=0)
        assert result.success


class TestFitLinesCustomLines:
    def test_specific_lines(self, prism_spectrum):
        result = fit_lines(
            prism_spectrum, z=6.0, grating="PRISM",
            lines=["OIII_5007", "OIII_4959", "HBETA"], n_boot=0,
        )
        assert set(result.line_names) == {"OIII_5007", "OIII_4959", "HBETA"}

    def test_no_observable_lines(self, prism_spectrum):
        result = fit_lines(prism_spectrum, z=50.0, grating="PRISM", n_boot=0)
        assert not result.success or len(result.lines) == 0


class TestAbsorptionLines:
    """Test absorption line fitting (negative Gaussians)."""

    def test_absorption_fit_succeeds(self, absorption_spectrum):
        result = fit_lines(
            absorption_spectrum, z=1.0, R=500.0,
            lines=["abs_SiII1260"], n_boot=0,
        )
        assert result.success

    def test_absorption_amplitude_negative(self, absorption_spectrum):
        result = fit_lines(
            absorption_spectrum, z=1.0, R=500.0,
            lines=["abs_SiII1260"], n_boot=0,
        )
        lr = result.lines["abs_SiII1260"]
        assert lr.amplitude < 0, "Absorption line amplitude should be negative"

    def test_absorption_flux_negative(self, absorption_spectrum):
        result = fit_lines(
            absorption_spectrum, z=1.0, R=500.0,
            lines=["abs_SiII1260"], n_boot=0,
        )
        lr = result.lines["abs_SiII1260"]
        assert lr.flux < 0, "Absorption line flux should be negative"


class TestLineResult:
    def test_line_result_fields(self, prism_spectrum):
        result = fit_lines(prism_spectrum, z=6.0, grating="PRISM", n_boot=0)
        if "OIII_5007" in result.lines:
            lr = result.lines["OIII_5007"]
            assert isinstance(lr, LineResult)
            assert lr.flux > 0
            assert lr.sigma_A > 0
            assert lr.snr >= 0
            assert np.isfinite(lr.ew_A)
