"""Tests for equivalent-width uncertainties in jwspecmcmc.

EW = F_line / F_cont, so its uncertainty has two terms.  The line-flux
term was always available from the chain; the continuum term needed
``fit_continuum(return_err=True)``.  For a faint continuum the second
term is the larger one, so propagating only the first would return an
error bar that is confidently too small.
"""

import numpy as np
import pytest

from jwspecfit.continuum import fit_continuum
from jwspecmcmc._engine import _ew_posterior, _ew_summary


def _flat_spectrum(n=800, noise=0.02, seed=0):
    """A featureless continuum with known noise."""
    rng = np.random.default_rng(seed)
    wave_um = np.linspace(1.0, 5.0, n)
    flux = 0.5 + 0.1 * wave_um + rng.normal(0.0, noise, n)
    err = np.full(n, noise)
    return wave_um, flux, err


_LINES = ["HBETA", "OIII_5007"]
_KW = dict(R=1000.0)


# ---------------------------------------------------------------------------


class TestContinuumError:
    def test_default_still_returns_a_bare_array(self):
        """Existing callers must be untouched."""
        w, f, e = _flat_spectrum()
        out = fit_continuum(w, f, e, 6.0, _LINES, **_KW)
        assert isinstance(out, np.ndarray)

    def test_return_err_gives_a_pair(self):
        w, f, e = _flat_spectrum()
        out = fit_continuum(w, f, e, 6.0, _LINES, return_err=True, **_KW)
        assert isinstance(out, tuple) and len(out) == 2

    def test_continuum_is_unchanged_by_asking_for_the_error(self):
        w, f, e = _flat_spectrum()
        bare = fit_continuum(w, f, e, 6.0, _LINES, **_KW)
        cont, _ = fit_continuum(w, f, e, 6.0, _LINES, return_err=True, **_KW)
        np.testing.assert_allclose(bare, cont, rtol=1e-12)

    def test_error_is_positive_and_finite(self):
        w, f, e = _flat_spectrum()
        _, err = fit_continuum(w, f, e, 6.0, _LINES, return_err=True, **_KW)
        assert np.all(np.isfinite(err))
        assert np.all(err > 0)

    def test_error_is_small_next_to_the_continuum(self):
        """A well-sampled flat continuum should be pinned down well."""
        w, f, e = _flat_spectrum()
        cont, err = fit_continuum(w, f, e, 6.0, _LINES, return_err=True, **_KW)
        assert np.median(err) / np.median(cont) < 0.05

    def test_noisier_data_gives_a_bigger_error(self):
        """The whole point of propagating it."""
        _, errs = zip(*[
            fit_continuum(*_flat_spectrum(noise=n), 6.0, _LINES,
                          return_err=True, **_KW)
            for n in (0.02, 0.10)
        ])
        assert np.median(errs[1]) > 2.0 * np.median(errs[0])

    def test_error_rises_towards_the_edges(self):
        """A polynomial is least constrained where it is extrapolating."""
        w, f, e = _flat_spectrum()
        _, err = fit_continuum(w, f, e, 6.0, _LINES, return_err=True, **_KW)
        n = len(err)
        centre = np.median(err[n // 3: 2 * n // 3])
        edges = np.median(np.concatenate([err[:n // 10], err[-n // 10:]]))
        assert edges > centre

    def test_median_filter_path_reports_unavailable(self):
        """A median filter has no parametric covariance -- NaN, not zero."""
        w, f, e = _flat_spectrum()
        _, err = fit_continuum(w, f, e, 6.0, _LINES, moving_average=True,
                               return_err=True, **_KW)
        assert np.all(np.isnan(err))

    def test_too_few_pixels_reports_unavailable(self):
        w, f, e = _flat_spectrum(n=800)
        e = np.zeros_like(e)  # nothing valid
        cont, err = fit_continuum(w, f, e, 6.0, _LINES, return_err=True, **_KW)
        assert np.all(cont == 0)
        assert np.all(np.isnan(err))


class TestEWPosterior:
    FLUX = np.full(5000, 2.0)

    def test_without_continuum_error_it_is_a_plain_ratio(self):
        rng = np.random.default_rng(0)
        ew = _ew_posterior(self.FLUX, 0.5, np.nan, 1.0, rng)
        np.testing.assert_allclose(ew, 2.0 / 0.5 / 2.0)

    def test_zero_continuum_error_is_also_a_plain_ratio(self):
        rng = np.random.default_rng(0)
        ew = _ew_posterior(self.FLUX, 0.5, 0.0, 1.0, rng)
        np.testing.assert_allclose(ew, 2.0)

    def test_continuum_error_widens_the_posterior(self):
        """The reason the feature exists."""
        rng = np.random.default_rng(1)
        narrow = _ew_posterior(self.FLUX, 0.5, 0.0, 1.0, rng)
        wide = _ew_posterior(self.FLUX, 0.5, 0.05, 1.0, rng)
        assert np.nanstd(wide) > np.nanstd(narrow)
        assert np.nanstd(narrow) == pytest.approx(0.0, abs=1e-12)

    def test_redshift_puts_it_in_the_rest_frame(self):
        rng = np.random.default_rng(0)
        a = _ew_posterior(self.FLUX, 0.5, np.nan, 0.0, rng)
        b = _ew_posterior(self.FLUX, 0.5, np.nan, 3.0, rng)
        np.testing.assert_allclose(a / 4.0, b)

    def test_non_positive_continuum_gives_all_nan(self):
        rng = np.random.default_rng(0)
        assert np.all(np.isnan(_ew_posterior(self.FLUX, 0.0, 0.1, 1.0, rng)))
        assert np.all(np.isnan(_ew_posterior(self.FLUX, -0.5, 0.1, 1.0, rng)))

    def test_nan_continuum_gives_all_nan(self):
        rng = np.random.default_rng(0)
        assert np.all(np.isnan(_ew_posterior(self.FLUX, np.nan, 0.1, 1.0, rng)))

    def test_draws_crossing_zero_are_dropped_not_exploded(self):
        """A continuum consistent with zero must not blow the EW up."""
        rng = np.random.default_rng(2)
        ew = _ew_posterior(self.FLUX, 0.10, 0.10, 0.0, rng)
        assert np.isnan(ew).any()          # some draws went non-positive
        assert np.all(ew[np.isfinite(ew)] > 0)

    def test_shape_matches_the_flux_posterior(self):
        rng = np.random.default_rng(0)
        assert _ew_posterior(self.FLUX, 0.5, 0.01, 1.0, rng).shape == self.FLUX.shape


class TestEWSummary:
    def test_median_and_half_widths(self):
        rng = np.random.default_rng(0)
        d = rng.normal(50.0, 5.0, 100000)
        med, (lo, hi) = _ew_summary(d)
        assert med == pytest.approx(50.0, abs=0.1)
        assert lo == pytest.approx(5.0, abs=0.1)
        assert hi == pytest.approx(5.0, abs=0.1)

    def test_asymmetry_is_preserved(self):
        rng = np.random.default_rng(1)
        d = np.exp(rng.normal(0.0, 0.5, 100000))
        _, (lo, hi) = _ew_summary(d)
        assert hi > lo

    def test_nans_are_ignored(self):
        d = np.concatenate([np.full(1000, 10.0), np.full(500, np.nan)])
        med, _ = _ew_summary(d)
        assert med == pytest.approx(10.0)

    def test_too_few_finite_samples_gives_nan(self):
        d = np.concatenate([np.full(5, 10.0), np.full(1000, np.nan)])
        med, (lo, hi) = _ew_summary(d)
        assert np.isnan(med) and np.isnan(lo) and np.isnan(hi)

    def test_all_nan_gives_nan(self):
        med, _ = _ew_summary(np.full(100, np.nan))
        assert np.isnan(med)


class TestSerialisation:
    """ew_err and ew_posterior must survive a save/load round trip."""

    def _result(self):
        from tests.test_io import _make_synthetic_mcmc_result
        res = _make_synthetic_mcmc_result()
        rng = np.random.default_rng(7)
        for lr in res.lines.values():
            lr.ew_posterior = rng.normal(50.0, 5.0, len(lr.flux_posterior))
            lr.ew_err = (4.8, 5.2)
        return res

    def test_round_trip_preserves_ew_err(self, tmp_path):
        from jwspecmcmc import load_mcmc_result, save_mcmc_result

        original = self._result()
        out = tmp_path / "mcmc.npz"
        save_mcmc_result(original, out)
        loaded = load_mcmc_result(out)
        for name, lr in original.lines.items():
            assert loaded.lines[name].ew_err == pytest.approx(lr.ew_err)

    def test_round_trip_preserves_ew_posterior(self, tmp_path):
        from jwspecmcmc import load_mcmc_result, save_mcmc_result

        original = self._result()
        out = tmp_path / "mcmc.npz"
        save_mcmc_result(original, out)
        loaded = load_mcmc_result(out)
        for name, lr in original.lines.items():
            np.testing.assert_allclose(
                loaded.lines[name].ew_posterior, lr.ew_posterior, rtol=1e-10
            )

    def test_a_result_without_ew_posterior_still_round_trips(self, tmp_path):
        """Older saved files carry no EW posterior; loading must not break."""
        from jwspecmcmc import load_mcmc_result, save_mcmc_result
        from tests.test_io import _make_synthetic_mcmc_result

        original = _make_synthetic_mcmc_result()
        out = tmp_path / "mcmc.npz"
        save_mcmc_result(original, out)
        loaded = load_mcmc_result(out)
        for lr in loaded.lines.values():
            assert lr.ew_posterior is None
            assert lr.ew_err is None
