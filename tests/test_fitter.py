"""Tests for jwspecfit.fitter — core fitting engine."""

import numpy as np
import pytest

from jwspecfit.fitter import fit_lines, FitResult, LineResult
from .conftest import PRISM_FITS, G395M_FITS


class TestFitLinesPrism:
    def test_prism_fit_succeeds(self, prism_fit_result):
        assert isinstance(prism_fit_result, FitResult)
        assert prism_fit_result.success

    def test_has_lines(self, prism_fit_result):
        assert len(prism_fit_result.lines) > 0
        assert "OIII_5007" in prism_fit_result.lines

    def test_model_shape(self, prism_fit_result):
        spec = prism_fit_result.spectrum
        assert prism_fit_result.model_flux.shape == spec.flux_ujy.shape
        assert prism_fit_result.continuum.shape == spec.flux_ujy.shape
        assert prism_fit_result.residuals.shape == spec.flux_ujy.shape

    def test_bootstrap_errors(self, prism_spectrum):
        """Test that bootstrap produces non-zero errors (small line list)."""
        result = fit_lines(
            prism_spectrum, z=6.0, grating="PRISM",
            lines=["OIII_5007", "HBETA"], n_boot=20,
        )
        assert result.success
        for name, lr in result.lines.items():
            if lr.flux > 0:
                assert lr.flux_err > 0


class TestFitLinesGrating:
    def test_grating_fit_succeeds(self, grating_fit_result):
        assert grating_fit_result.success

    def test_chi2_reasonable(self, grating_fit_result):
        assert grating_fit_result.chi2 < 100


class TestFitLinesCustomR:
    def test_numeric_R(self, prism_spectrum):
        result = fit_lines(
            prism_spectrum, z=6.0, R=100.0,
            lines=["OIII_5007", "HBETA"], n_boot=0,
        )
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
    def test_line_result_fields(self, prism_fit_result):
        if "OIII_5007" in prism_fit_result.lines:
            lr = prism_fit_result.lines["OIII_5007"]
            assert isinstance(lr, LineResult)
            assert lr.flux > 0
            assert lr.sigma_A > 0
            assert lr.snr >= 0
            assert np.isfinite(lr.ew0_A)


def test_faint_line_on_negative_dip_keeps_sane_seed_and_errors():
    """A faint line with no positive pixel nearby must not get a ~1 erg/s/cm2/A seed.

    The old fallback seeded such a line at peak = 1.0 (~1e19 x the real flux
    scale); the joint fit then mis-converged and the bootstrap collapsed to
    identical garbage, returning flux_err = 0 for the bright lines.
    """
    import numpy as np

    from jwspecfit.fitter import _seed_peak

    flam = np.array([-3e-21, -2e-21, -4e-21, -1e-21])
    err = np.full(4, 1e-21)
    peak = _seed_peak(flam, err, np.ones(4, bool), np.arange(4), is_abs=False)
    assert peak == pytest.approx(1e-21, abs=0)


@pytest.mark.parametrize("grating,lo,hi,n", [("G395M", 2.87, 5.2, 3000), ("PRISM", 0.6, 5.3, 430)])
def test_grating_bounds_use_sigma_inst_at_the_line(grating, lo, hi, n):
    """Width bounds must use σ_inst at the line's own pixel, not elsewhere."""
    import numpy as np

    from jwspecfit.fitter import _grating_bounds
    from jwspecfit.resolution import sigma_inst_A

    w_um = np.geomspace(lo, hi, n)
    wave_A = w_um * 1e4
    dlam = np.gradient(wave_A)
    sig = sigma_inst_A(w_um, grating=grating)
    for lam in np.linspace(wave_A[50], wave_A[-50], 5):
        seed = _grating_bounds(grating, sig, dlam, lam, wave_A=wave_A)[1]
        expect = (0.90 if grating == "PRISM" else 0.60) * sig[np.argmin(abs(wave_A - lam))]
        assert seed == pytest.approx(expect)
