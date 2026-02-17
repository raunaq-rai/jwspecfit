"""Tests for jwspecfit.io — Spectrum I/O."""

import numpy as np
import pytest

from jwspecfit import Spectrum, read_dict, read_fits, read_npz
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
