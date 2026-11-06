"""Per-source resolution curves from the FITS ``resolution`` column.

Combined multi-grating spectra (GRATING = "G140M+G235M+G395M") used to need
``spec.grating = None`` plus an explicit ``R=`` callable, and crashed in the
upper-limit code if the grating name was left set.  The file's resolution
column is now read into ``spec.R`` and the fit keeps the R it used.
"""

import pickle

import numpy as np
import pytest
from astropy.io import fits

from jwspecfit import TabulatedR, fit_lines, read_fits
from jwspecfit.lines import REST_LINES_A
from jwspecfit.resolution import R_grating, as_stored_R, resolve_R

Z = 5.0
COMBINED = "G140M+G235M+G395M"


def _R_curve(wave_um):
    return 1200.0 + 400.0 * (wave_um - 3.0)


def _write_spectrum(path, wave_um, R, grating=COMBINED, extra_hdu=None):
    """A Gaussian H-alpha line on a flat continuum, msaexp-style layout."""
    rng = np.random.default_rng(7)
    lam = REST_LINES_A["HALPHA"] * (1 + Z) / 1e4
    sig = lam / _R_curve(lam) / 2.355
    flux = 0.2 + 3.0 * np.exp(-0.5 * ((wave_um - lam) / sig) ** 2)
    err = np.full_like(wave_um, 0.02)
    flux = flux + rng.normal(0, 0.02, wave_um.size)
    cols = [
        fits.Column("wave", "D", unit="um", array=wave_um),
        fits.Column("flux", "D", unit="uJy", array=flux),
        fits.Column("err", "D", unit="uJy", array=err),
        fits.Column("resolution", "D", array=R),
    ]
    spec1d = fits.BinTableHDU.from_columns(cols, name="SPEC1D")
    spec1d.header["GRATING"] = grating
    hdus = [fits.PrimaryHDU(), spec1d]
    if extra_hdu is not None:
        hdus.append(extra_hdu)
    fits.HDUList(hdus).writeto(path)


@pytest.fixture
def combined_file(tmp_path):
    wave = np.linspace(3.80, 4.10, 600)
    R = _R_curve(wave)
    R[:5] = np.nan  # the real files carry NaN at the edges

    g395 = fits.BinTableHDU.from_columns([
        fits.Column("wave", "D", unit="um", array=wave),
        fits.Column("flux", "D", unit="uJy", array=np.ones_like(wave)),
        fits.Column("err", "D", unit="uJy", array=np.full_like(wave, 0.1)),
        fits.Column("resolution", "D", array=np.full_like(wave, 1500.0)),
    ], name="SPEC1D_G395M")
    g395.header["GRATING"] = "G395M"

    path = tmp_path / "combined.spec.fits"
    _write_spectrum(path, wave, R, extra_hdu=g395)
    return path


class TestReadFits:
    def test_resolution_column_becomes_spec_R(self, combined_file):
        spec = read_fits(combined_file, z=Z)
        assert spec.grating == COMBINED
        assert isinstance(spec.R, TabulatedR)
        lam = np.array([3.85, 3.95, 4.05])
        np.testing.assert_allclose(spec.R(lam), _R_curve(lam), rtol=1e-10)
        assert spec.meta["resolution_col"] == "resolution"

    def test_single_grating_extension_has_its_own_curve(self, combined_file):
        spec = read_fits(combined_file, z=Z, hdu="SPEC1D_G395M")
        assert spec.grating == "G395M"
        np.testing.assert_allclose(spec.R(np.array([3.9])), 1500.0)

    def test_explicit_R_overrides_the_column(self, combined_file):
        assert read_fits(combined_file, z=Z, R=2000.0).R == 2000.0


class TestCombinedGratingName:
    def test_nominal_R_of_a_combined_name(self):
        assert R_grating(COMBINED) == 1000.0
        np.testing.assert_allclose(resolve_R(np.array([2.0]), grating=COMBINED), 1000.0)

    def test_mixed_resolution_name_asks_for_R(self):
        with pytest.raises(ValueError, match="pass R="):
            R_grating("G235M+G395H")


class TestFitUsesFileR:
    def test_fit_needs_no_R_and_limits_do_not_crash(self, combined_file):
        spec = read_fits(combined_file, z=Z)
        res = fit_lines(spec, z=Z, n_boot=0, lines=["HALPHA", "NII_6585"])
        assert res.lines["HALPHA"].flux > 0
        # The non-detection's limit is built from the file's R, not "G140M+...".
        assert np.isfinite(res.flux_upper_limit("NII_6585"))
        assert isinstance(res.spectrum.R, TabulatedR)

    def test_callable_R_with_combined_name_set(self, combined_file):
        """The reported crash: R= passed while spec.grating is still combined."""
        spec = read_fits(combined_file, z=Z)
        spec.R = None
        res = fit_lines(
            spec, z=Z, n_boot=0, lines=["HALPHA", "NII_6585"],
            R=lambda lam: _R_curve(np.asarray(lam)),
        )
        assert spec.R is None  # the caller's spectrum is not modified
        np.testing.assert_allclose(
            res.spectrum.R(np.array([3.95])), _R_curve(np.array([3.95])), rtol=1e-6,
        )
        assert np.isfinite(res.flux_upper_limit("NII_6585"))
        pickle.loads(pickle.dumps(res))  # a lambda would not pickle


class TestTabulatedR:
    def test_drops_bad_values_sorts_and_pickles(self):
        tab = TabulatedR([3.0, 1.0, 2.0, 4.0], [300.0, 100.0, np.nan, -1.0])
        np.testing.assert_allclose(tab.wave_um, [1.0, 3.0])
        np.testing.assert_allclose(tab(np.array([2.0])), 200.0)
        assert pickle.loads(pickle.dumps(tab))(np.array([2.0])) == pytest.approx(200.0)

    def test_as_stored_R(self):
        w = np.linspace(1, 2, 5)
        assert as_stored_R(1000.0, w) == 1000.0
        assert as_stored_R(None, w) is None
        tab = as_stored_R(lambda lam: 100 * np.asarray(lam), w)
        assert isinstance(tab, TabulatedR)
        np.testing.assert_allclose(tab(w), 100 * w)
