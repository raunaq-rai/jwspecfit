"""Tests for jwspecfit.broad — broad component fitting and BIC selection."""

import numpy as np
import pytest

from jwspecfit import fit_with_broad, read_fits
from jwspecfit.broad import BroadFitResult
from .conftest import PRISM_FITS, G395M_FITS


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
        )
        assert result.selected_model in ("narrow", "broad1", "broad2", "both")

    def test_bic_values(self, prism_spectrum):
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", mode="auto", n_boot=0,
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
        )
        assert "narrow" in result.all_fits
