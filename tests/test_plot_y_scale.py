"""Tests for the ``y_scale`` option on every plotting function.

Covers the two contracts that matter for a log flux axis:

1. ``y_scale="linear"`` is the default and leaves the axis untouched.
2. ``y_scale="log"`` switches the axis to log **and** replaces the
   lower limit with a strictly positive one (matplotlib), or converts
   the range to log10 units (plotly, where ``range`` on a log axis is
   itself logarithmic).
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from .conftest import PRISM_FITS  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# Helper behaviour
# ============================================================

class TestHelpers:
    """The shared _log_ylim / _validate_y_scale helpers."""

    def test_validate_rejects_unknown(self):
        from jwspecfit.plotting import _validate_y_scale

        with pytest.raises(ValueError, match="y_scale"):
            _validate_y_scale("semilog")

    def test_validate_is_case_insensitive(self):
        from jwspecfit.plotting import _validate_y_scale

        assert _validate_y_scale("LOG") == "log"
        assert _validate_y_scale("Linear") == "linear"

    def test_log_ylim_is_positive_with_negatives_present(self):
        from jwspecfit.plotting import _log_ylim

        vals = np.array([-5.0, -0.1, 0.0, np.nan, 0.5, 2.0, 10.0])
        lo, hi = _log_ylim(vals, 13.0)
        assert lo > 0
        assert lo < hi
        assert hi == pytest.approx(13.0)

    def test_log_ylim_floors_dynamic_range(self):
        from jwspecfit.plotting import _log_ylim

        # Near-zero pixels must not stretch the axis over 20 decades.
        vals = np.concatenate([np.full(50, 1e-20), [1.0, 10.0]])
        lo, hi = _log_ylim(vals, 10.0, max_decades=4.0)
        assert lo == pytest.approx(hi / 1e4)

    def test_log_ylim_all_nonpositive(self):
        from jwspecfit.plotting import _log_ylim

        lo, hi = _log_ylim(np.array([-1.0, 0.0]), -3.0)
        assert 0 < lo < hi


# ============================================================
# jwspecfit.plotting.plot_fit  (matplotlib)
# ============================================================

class TestPlotFit:

    def test_default_is_linear(self, prism_fit_result):
        from jwspecfit import plot_fit

        fig = plot_fit(prism_fit_result)
        assert fig.get_axes()[0].get_yscale() == "linear"

    def test_log_sets_scale_and_positive_limits(self, prism_fit_result):
        from jwspecfit import plot_fit

        fig = plot_fit(prism_fit_result, y_scale="log")
        ax_main = fig.get_axes()[0]
        assert ax_main.get_yscale() == "log"
        assert ax_main.get_ylim()[0] > 0

    def test_upper_limit_matches_linear(self, prism_fit_result):
        """Log mode only changes the floor, not the line-driven ceiling."""
        from jwspecfit import plot_fit

        top_lin = plot_fit(prism_fit_result).get_axes()[0].get_ylim()[1]
        top_log = plot_fit(
            prism_fit_result, y_scale="log",
        ).get_axes()[0].get_ylim()[1]
        assert top_log == pytest.approx(top_lin, rel=1e-6)

    def test_residual_panel_stays_linear(self, prism_fit_result):
        from jwspecfit import plot_fit

        fig = plot_fit(prism_fit_result, y_scale="log", show_residuals=True)
        assert fig.get_axes()[1].get_yscale() == "linear"

    def test_invalid_raises(self, prism_fit_result):
        from jwspecfit import plot_fit

        with pytest.raises(ValueError, match="y_scale"):
            plot_fit(prism_fit_result, y_scale="logarithmic")


# ============================================================
# jwspecfit.plotting.plot_spectrum_interactive  (plotly)
# ============================================================

class TestPlotSpectrumInteractive:

    def test_default_is_linear(self, synthetic_spectrum):
        from jwspecfit import plot_spectrum_interactive

        fig = plot_spectrum_interactive(synthetic_spectrum)
        assert fig.layout.yaxis.type in (None, "linear")

    def test_log_axis_and_log10_range(self, synthetic_spectrum):
        from jwspecfit import plot_spectrum_interactive

        fig_lin = plot_spectrum_interactive(synthetic_spectrum)
        fig_log = plot_spectrum_interactive(synthetic_spectrum, y_scale="log")

        assert fig_log.layout.yaxis.type == "log"
        # Plotly log-axis ranges are in log10 units; the ceiling is the
        # same physical flux as in linear mode.
        assert 10 ** fig_log.layout.yaxis.range[1] == pytest.approx(
            fig_lin.layout.yaxis.range[1], rel=1e-6,
        )
        assert fig_log.layout.yaxis.range[0] < fig_log.layout.yaxis.range[1]

    def test_power_tick_format(self, synthetic_spectrum):
        from jwspecfit import plot_spectrum_interactive

        fig = plot_spectrum_interactive(synthetic_spectrum, y_scale="log")
        assert fig.layout.yaxis.exponentformat == "power"

    def test_log_with_2d_panel(self):
        """With the 2-D panel on, the 1-D flux axis is row 2 (yaxis2)."""
        from jwspecfit import plot_spectrum_interactive

        fig = plot_spectrum_interactive(PRISM_FITS, z=6.0, y_scale="log")
        assert fig.layout.yaxis2.type == "log"
        assert fig.layout.yaxis2.range[0] < fig.layout.yaxis2.range[1]
        assert fig.layout.yaxis.type in (None, "linear")  # 2-D spatial

    def test_zero_line_suppressed_on_log(self, synthetic_spectrum):
        from jwspecfit import plot_spectrum_interactive

        lin = plot_spectrum_interactive(synthetic_spectrum, show_zero=True)
        log = plot_spectrum_interactive(
            synthetic_spectrum, show_zero=True, y_scale="log",
        )
        assert len(lin.layout.shapes) > len(log.layout.shapes)

    def test_invalid_raises(self, synthetic_spectrum):
        from jwspecfit import plot_spectrum_interactive

        with pytest.raises(ValueError, match="y_scale"):
            plot_spectrum_interactive(synthetic_spectrum, y_scale="ln")


# ============================================================
# jwspecfit.plotting.plot_fit_interactive  (plotly)
# ============================================================

class TestPlotFitInteractive:

    def test_default_is_linear(self, prism_fit_result):
        from jwspecfit import plot_fit_interactive

        fig = plot_fit_interactive(prism_fit_result)
        assert fig.layout.yaxis.type in (None, "linear")

    def test_log_axis_and_log10_range(self, prism_fit_result):
        from jwspecfit import plot_fit_interactive

        kw = dict(show_2d=False)
        fig_lin = plot_fit_interactive(prism_fit_result, **kw)
        fig_log = plot_fit_interactive(prism_fit_result, y_scale="log", **kw)

        assert fig_log.layout.yaxis.type == "log"
        assert 10 ** fig_log.layout.yaxis.range[1] == pytest.approx(
            fig_lin.layout.yaxis.range[1], rel=1e-6,
        )

    def test_residual_panel_stays_linear(self, prism_fit_result):
        from jwspecfit import plot_fit_interactive

        fig = plot_fit_interactive(
            prism_fit_result, y_scale="log",
            show_residuals=True, show_2d=False,
        )
        # Main panel is row 1 (yaxis), residuals row 2 (yaxis2).
        assert fig.layout.yaxis.type == "log"
        assert fig.layout.yaxis2.type in (None, "linear")

    def test_log_with_2d_panel(self, prism_fit_result):
        """With the 2-D panel on, the fit panel moves to row 2 (yaxis2)."""
        from jwspecfit import plot_fit_interactive

        fig = plot_fit_interactive(
            prism_fit_result, y_scale="log",
            show_residuals=True, show_2d=True,
        )
        assert fig.layout.yaxis2.type == "log"
        assert fig.layout.yaxis.type in (None, "linear")   # 2-D spatial
        assert fig.layout.yaxis3.type in (None, "linear")  # residuals

    def test_single_panel_log(self, prism_fit_result):
        from jwspecfit import plot_fit_interactive

        fig = plot_fit_interactive(
            prism_fit_result, y_scale="log",
            show_residuals=False, show_2d=False,
        )
        assert fig.layout.yaxis.type == "log"
        assert fig.layout.yaxis.range[0] < fig.layout.yaxis.range[1]

    def test_invalid_raises(self, prism_fit_result):
        from jwspecfit import plot_fit_interactive

        with pytest.raises(ValueError, match="y_scale"):
            plot_fit_interactive(prism_fit_result, y_scale=True)


# ============================================================
# jwspecfit.plotting.plot_2d_1d  (matplotlib)
# ============================================================

class TestPlot2d1d:

    def test_default_is_linear(self):
        from jwspecfit import plot_2d_1d

        _, (_, ax1d) = plot_2d_1d(PRISM_FITS, z=6.0)
        assert ax1d.get_yscale() == "linear"

    def test_log_sets_scale_and_positive_limits(self):
        from jwspecfit import plot_2d_1d

        _, (_, ax1d) = plot_2d_1d(PRISM_FITS, z=6.0, y_scale="log")
        assert ax1d.get_yscale() == "log"
        assert ax1d.get_ylim()[0] > 0

    def test_line_labels_inside_axes(self):
        from jwspecfit import plot_2d_1d

        _, (_, ax1d) = plot_2d_1d(PRISM_FITS, z=6.0, y_scale="log")
        lo, hi = ax1d.get_ylim()
        texts = [t for t in ax1d.texts]
        assert texts, "expected emission-line labels on the 1-D panel"
        for t in texts:
            assert lo < t.get_position()[1] < hi

    def test_invalid_raises(self):
        from jwspecfit import plot_2d_1d

        with pytest.raises(ValueError, match="y_scale"):
            plot_2d_1d(PRISM_FITS, z=6.0, y_scale="log10")


# ============================================================
# jwspecfit.dla.DLAResult.plot  (matplotlib)
# ============================================================

@pytest.fixture(scope="module")
def dla_result():
    """Cheap DLA fit on a synthetic damped spectrum."""
    from jwspecfit.dla import _evaluate_model, fit_NHI

    wave = np.linspace(1050, 2000, 400)
    model = _evaluate_model(wave, -1.0, -2.5, 22.0, z=0.0)
    rng = np.random.default_rng(12345)
    noise_level = float(np.median(model[wave > 1500])) / 15.0
    flux = model + rng.normal(0, noise_level, len(wave))
    err = np.full_like(wave, noise_level)

    return fit_NHI(
        wave, flux, err, z=0.0, mask_lines=False, n_live=100, seed=42,
    )


class TestDLAPlot:

    def test_default_is_linear(self, dla_result):
        fig = dla_result.plot()
        assert fig.get_axes()[0].get_yscale() == "linear"

    def test_log_sets_scale_and_positive_limits(self, dla_result):
        fig = dla_result.plot(y_scale="log")
        ax_main = fig.get_axes()[0]
        assert ax_main.get_yscale() == "log"
        assert ax_main.get_ylim()[0] > 0

    def test_residual_panel_stays_linear(self, dla_result):
        fig = dla_result.plot(y_scale="log", show_residuals=True)
        assert fig.get_axes()[1].get_yscale() == "linear"

    def test_lya_label_inside_axes(self, dla_result):
        fig = dla_result.plot(y_scale="log")
        ax_main = fig.get_axes()[0]
        lo, hi = ax_main.get_ylim()
        assert ax_main.texts
        for t in ax_main.texts:
            assert lo < t.get_position()[1] < hi

    def test_invalid_raises(self, dla_result):
        with pytest.raises(ValueError, match="y_scale"):
            dla_result.plot(y_scale="lin")


# ============================================================
# jwspecfit.redshift.RedshiftResult.plot  (plotly)
# ============================================================

class TestRedshiftPlot:

    @staticmethod
    def _result(spec):
        from jwspecfit.redshift import RedshiftResult

        grid = np.linspace(0.9, 1.1, 21)
        chi2 = (grid - 1.0) ** 2 * 1e4
        return RedshiftResult(
            z_best=1.0, z_ci68=(0.99, 1.01), z_ci95=(0.98, 1.02),
            peaks=[], is_decisive=True,
            z_grid_coarse=grid, chi2_coarse=chi2,
            P_z_coarse=np.exp(-0.5 * chi2),
            z_grid_fine=grid, chi2_fine=chi2,
            P_z_fine=np.exp(-0.5 * chi2),
            lines_used=[], grating=None, spec=spec,
        )

    def test_default_is_linear(self, synthetic_spectrum):
        fig = self._result(synthetic_spectrum).plot()
        assert fig.layout.yaxis2.type in (None, "linear")

    def test_log_applies_to_spectrum_panel_only(self, synthetic_spectrum):
        fig = self._result(synthetic_spectrum).plot(y_scale="log")
        # Row 1 is Δχ²(z) (always linear), row 2 the spectrum.
        assert fig.layout.yaxis.type in (None, "linear")
        assert fig.layout.yaxis2.type == "log"
        assert fig.layout.yaxis2.exponentformat == "power"

    def test_invalid_raises(self, synthetic_spectrum):
        with pytest.raises(ValueError, match="y_scale"):
            self._result(synthetic_spectrum).plot(y_scale="log2")


# ============================================================
# jwspecmcmc.plotting  (matplotlib)
# ============================================================

@pytest.fixture
def fake_mcmc_result():
    """Minimal stand-in exposing only what the diagnostic plots read."""
    rng = np.random.default_rng(7)
    chains = np.abs(rng.normal(1.0, 0.1, size=(4, 50, 2)))
    flux_post = np.abs(rng.normal(3.0e-18, 4.0e-19, size=2000))
    line = SimpleNamespace(
        flux_posterior=flux_post,
        flux=float(np.median(flux_post)),
        flux_err=(4.0e-19, 4.0e-19),
    )
    return SimpleNamespace(
        chains=chains,
        flat_chains_free=chains.reshape(-1, 2),
        lines={"OIII_5007": line},
        line_names=["OIII_5007"],
        constraints=None,
        sampler_meta={},
    )


class TestMCMCTraces:

    def test_default_is_linear(self, fake_mcmc_result):
        from jwspecmcmc.plotting import plot_traces

        fig = plot_traces(fake_mcmc_result)
        assert all(ax.get_yscale() == "linear" for ax in fig.get_axes())

    def test_log_applies_to_every_panel(self, fake_mcmc_result):
        from jwspecmcmc.plotting import plot_traces

        fig = plot_traces(fake_mcmc_result, y_scale="log")
        assert all(ax.get_yscale() == "log" for ax in fig.get_axes())

    def test_invalid_raises(self, fake_mcmc_result):
        from jwspecmcmc.plotting import plot_traces

        with pytest.raises(ValueError, match="y_scale"):
            plot_traces(fake_mcmc_result, y_scale="nope")


class TestMCMCFluxPosterior:

    def test_default_is_linear(self, fake_mcmc_result):
        from jwspecmcmc.plotting import plot_flux_posterior

        ax = plot_flux_posterior(fake_mcmc_result, "OIII_5007")
        assert ax.get_yscale() == "linear"

    def test_log(self, fake_mcmc_result):
        from jwspecmcmc.plotting import plot_flux_posterior

        ax = plot_flux_posterior(
            fake_mcmc_result, "OIII_5007", y_scale="log",
        )
        assert ax.get_yscale() == "log"

    def test_invalid_raises(self, fake_mcmc_result):
        from jwspecmcmc.plotting import plot_flux_posterior

        with pytest.raises(ValueError, match="y_scale"):
            plot_flux_posterior(
                fake_mcmc_result, "OIII_5007", y_scale="nope",
            )


class TestMCMCCorner:

    def test_default_is_linear(self, fake_mcmc_result):
        from jwspecmcmc.plotting import plot_corner

        fig = plot_corner(fake_mcmc_result)
        assert fig.axes[0].get_yscale() == "linear"

    def test_log_applies_to_diagonal_marginals(self, fake_mcmc_result):
        from jwspecmcmc.plotting import plot_corner

        fig = plot_corner(fake_mcmc_result, y_scale="log")
        n_dim = fake_mcmc_result.flat_chains_free.shape[1]
        diag = [fig.axes[i * (n_dim + 1)] for i in range(n_dim)]
        assert all(ax.get_yscale() == "log" for ax in diag)

    def test_invalid_raises(self, fake_mcmc_result):
        from jwspecmcmc.plotting import plot_corner

        with pytest.raises(ValueError, match="y_scale"):
            plot_corner(fake_mcmc_result, y_scale="nope")


# ============================================================
# Public wrappers forward the kwarg
# ============================================================

class TestWrappersForward:

    def test_jwspecmcmc_toplevel_wrappers(self, fake_mcmc_result):
        import jwspecmcmc

        assert jwspecmcmc.plot_traces(
            fake_mcmc_result, y_scale="log",
        ).get_axes()[0].get_yscale() == "log"
        assert jwspecmcmc.plot_flux_posterior(
            fake_mcmc_result, "OIII_5007", y_scale="log",
        ).get_yscale() == "log"
