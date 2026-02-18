"""Tests for jwspecfit.io — Spectrum I/O."""

import numpy as np
import pytest

from jwspecfit import (
    Spectrum, read_dict, read_fits, read_npz, fit_lines,
    save_result, load_result, export_lines_txt,
)
from .conftest import G395M_FITS, PRISM_FITS, STACK_NPZ


class TestReadFits:
    def test_prism_loads(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        assert isinstance(spec, Spectrum)
        assert spec.n_pix > 100
        assert spec.grating == "PRISM"
        assert spec.z == 6.0

    def test_grating_loads(self):
        spec = read_fits(G395M_FITS, z=5.0)
        assert spec.grating == "G395M"
        assert spec.n_pix > 100

    def test_wave_units(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        # Wave in µm should be < 10
        assert spec.wave_um.max() < 10.0
        # Wave in Å should be > 1000
        assert spec.wave_A.max() > 1000.0
        np.testing.assert_allclose(spec.wave_A, spec.wave_um * 1e4)

    def test_edges_length(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        assert len(spec.wave_edges_A) == spec.n_pix + 1

    def test_dlam_positive(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        assert np.all(spec.dlam_A > 0)

    def test_mask_valid(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        mask = spec.mask_valid()
        assert mask.dtype == bool
        assert np.sum(mask) > 0


class TestReadNpz:
    def test_stack_loads(self):
        spec = read_npz(STACK_NPZ, z=6.0, R=100.0)
        assert spec.n_pix > 1000
        assert spec.R == 100.0
        assert spec.grating is None

    def test_stack_wave_range(self):
        spec = read_npz(STACK_NPZ, z=6.0, R=100.0)
        assert spec.wave_A.min() >= 900.0
        assert spec.wave_A.max() <= 80000.0


class TestReadDict:
    def test_from_dict(self):
        data = {
            "wave": np.linspace(1.0, 5.0, 100),
            "flux": np.random.default_rng(0).normal(0.1, 0.01, 100),
            "err": np.full(100, 0.01),
        }
        spec = read_dict(data, z=3.0, R=100.0)
        assert spec.n_pix == 100
        assert spec.z == 3.0


class TestSpectrum:
    def test_copy(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        spec2 = spec.copy()
        spec2.flux_ujy[0] = -999
        assert spec.flux_ujy[0] != -999

    def test_flux_flam_conversion(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        flam = spec.flux_flam
        assert flam.shape == spec.flux_ujy.shape
        # F_λ should generally be small positive numbers
        valid = spec.mask_valid()
        assert np.any(flam[valid] > 0)


class TestSaveLoadResult:
    @pytest.fixture
    def fit_result(self, prism_spectrum):
        return fit_lines(prism_spectrum, z=6.0, grating="PRISM", n_boot=0)

    def test_round_trip(self, fit_result, tmp_path):
        outfile = tmp_path / "result.npz"
        save_result(fit_result, outfile)
        loaded = load_result(outfile)
        assert loaded.success == fit_result.success
        np.testing.assert_allclose(loaded.model_flux, fit_result.model_flux, rtol=1e-10)
        np.testing.assert_allclose(loaded.chi2, fit_result.chi2)
        assert set(loaded.lines.keys()) == set(fit_result.lines.keys())
        for name in fit_result.lines:
            np.testing.assert_allclose(loaded.lines[name].flux, fit_result.lines[name].flux)

    def test_spectrum_preserved(self, fit_result, tmp_path):
        outfile = tmp_path / "result.npz"
        save_result(fit_result, outfile)
        loaded = load_result(outfile)
        np.testing.assert_allclose(loaded.spectrum.wave_um, fit_result.spectrum.wave_um)
        assert loaded.spectrum.grating == fit_result.spectrum.grating


class TestExportLinesTxt:
    def test_writes_file(self, prism_spectrum, tmp_path):
        result = fit_lines(prism_spectrum, z=6.0, grating="PRISM", n_boot=0)
        outfile = tmp_path / "lines.txt"
        export_lines_txt(result, outfile)
        assert outfile.exists()
        lines = outfile.read_text().strip().split("\n")
        # Header + data lines
        assert len(lines) >= 4  # 3 header + at least 1 line
        # Check header contains expected columns
        assert "flux" in lines[2]
        assert "EW_A" in lines[2]
        assert "SNR_peak" in lines[2]
        # Check units header
        assert "erg/s/cm2" in lines[1]
