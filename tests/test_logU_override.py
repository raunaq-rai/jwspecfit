"""Tests for supplying log(U) instead of deriving it.

``logU_override`` hands log(U) in from outside -- an SED or
photoionisation fit -- and ``logU_override_err`` gives its uncertainty.
The contract: no emission-line ratio is consulted, the value flows into
the Martinez+2025 ICF surfaces exactly as a derived one would, and
nothing that does not depend on log(U) (O/H, T_e, n_e) moves at all.
"""

import numpy as np
import pytest

from jwspecabund import compute_abundances
from jwspecabund._core import _draw_logU

from .test_logU_ne import _run  # shared synthetic-object factory


# Each fixture is a full pipeline run; cache them for the module.
@pytest.fixture(scope="module")
def derived():
    return _run()


@pytest.fixture(scope="module")
def supplied():
    return _run(logU_override=-2.30)


@pytest.fixture(scope="module")
def supplied_err():
    return _run(logU_override=-2.30, logU_override_err=0.15)


@pytest.fixture(scope="module")
def supplied_asym():
    return _run(logU_override=-2.30, logU_override_err=(0.30, 0.10))


@pytest.fixture(scope="module")
def straddling_edge():
    """Central value 0.1 dex inside the floor, error bar well past it."""
    return _run(logU_override=-3.40, logU_override_err=0.30)


# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_raises(self, bad):
        with pytest.raises(ValueError, match="logU_override must be a finite"):
            compute_abundances(None, z=6.0, logU_override=bad)

    def test_conflicts_with_logU_diagnostic(self):
        with pytest.raises(ValueError, match="would have no effect"):
            compute_abundances(None, z=6.0, logU_override=-2.3,
                               logU_diagnostic="O32")

    def test_conflicts_with_logU_ne(self):
        with pytest.raises(ValueError, match="would have no effect"):
            compute_abundances(None, z=6.0, logU_override=-2.3, logU_ne=100.0)

    def test_conflict_message_names_both(self):
        with pytest.raises(ValueError, match="logU_diagnostic and logU_ne"):
            compute_abundances(None, z=6.0, logU_override=-2.3,
                               logU_diagnostic="O32", logU_ne=100.0)

    def test_err_without_value_raises(self):
        with pytest.raises(ValueError, match="without logU_override"):
            compute_abundances(None, z=6.0, logU_override_err=0.2)

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), (0.2, -0.1), (float("inf"), 0.1)])
    def test_bad_error_raises(self, bad):
        with pytest.raises(ValueError, match="non-negative finite"):
            compute_abundances(None, z=6.0, logU_override=-2.3,
                               logU_override_err=bad)

    def test_out_of_range_warns(self, caplog):
        with caplog.at_level("WARNING", logger="jwspecabund._core"):
            with pytest.raises(Exception):  # noqa: B017 - no fluxes, fails later
                compute_abundances(None, z=6.0, logU_override=-5.0)
        assert any("outside the Martinez+2025 ICF range" in r.message
                   for r in caplog.records)

    def test_in_range_does_not_warn(self, caplog):
        with caplog.at_level("WARNING", logger="jwspecabund._core"):
            _run(logU_override=-2.30)
        assert not any("outside the Martinez+2025 ICF range" in r.message
                       for r in caplog.records)


class TestDrawSampler:
    """`_draw_logU` is the whole uncertainty model; test it directly."""

    def test_no_error_is_exact_every_time(self):
        rng = np.random.default_rng(0)
        assert all(_draw_logU(rng, -2.3, None) == -2.3 for _ in range(50))

    def test_zero_error_is_exact(self):
        rng = np.random.default_rng(0)
        assert _draw_logU(rng, -2.3, 0.0) == -2.3

    def test_symmetric_recovers_mean_and_sigma(self):
        rng = np.random.default_rng(1)
        d = np.array([_draw_logU(rng, -2.3, 0.2) for _ in range(20000)])
        assert d.mean() == pytest.approx(-2.3, abs=0.01)
        assert d.std() == pytest.approx(0.2, abs=0.01)

    def test_asymmetric_is_skewed_the_right_way(self):
        """A long lower tail pulls the mean below the median."""
        rng = np.random.default_rng(2)
        d = np.array([_draw_logU(rng, -2.3, (0.4, 0.1)) for _ in range(40000)])
        assert d.mean() < -2.3

    def test_asymmetric_reproduces_the_quoted_interval(self):
        """The whole point: percentiles must come back as supplied.

        An SED fit quotes log(U) as a median with 16th/84th half-widths.
        Feeding those in must return them, or the uncertainty being
        propagated is not the one that was measured.
        """
        rng = np.random.default_rng(3)
        d = np.array([_draw_logU(rng, -2.45, (0.42, 0.18)) for _ in range(200000)])
        p16, p50, p84 = np.percentile(d, [16, 50, 84])
        assert p50 == pytest.approx(-2.45, abs=0.01)
        assert (p50 - p16) == pytest.approx(0.42, abs=0.015)
        assert (p84 - p50) == pytest.approx(0.18, abs=0.015)

    def test_symmetric_tuple_matches_the_scalar_form(self):
        rng = np.random.default_rng(4)
        a = np.array([_draw_logU(rng, -2.3, (0.2, 0.2)) for _ in range(20000)])
        rng = np.random.default_rng(4)
        b = np.array([_draw_logU(rng, -2.3, 0.2) for _ in range(20000)])
        assert a.std() == pytest.approx(b.std(), abs=0.01)

    def test_sides_are_equally_weighted(self):
        """Equal weighting is what puts the percentiles where quoted."""
        rng = np.random.default_rng(5)
        d = np.array([_draw_logU(rng, -2.3, (0.4, 0.1)) for _ in range(40000)])
        assert float(np.mean(d < -2.3)) == pytest.approx(0.5, abs=0.02)

    def test_one_sided_error(self):
        rng = np.random.default_rng(6)
        d = np.array([_draw_logU(rng, -2.3, (0.0, 0.2)) for _ in range(2000)])
        assert (d >= -2.3).all()


class TestSuppliedValue:
    def test_value_is_used_verbatim(self, supplied):
        assert supplied.logU == pytest.approx(-2.30)

    def test_it_differs_from_the_derived_one(self, derived, supplied):
        """Otherwise the test proves nothing."""
        assert abs(derived.logU - supplied.logU) > 0.1

    def test_no_uncertainty_gives_a_point_posterior(self, supplied):
        assert supplied.logU_err == pytest.approx(0.0, abs=1e-12)

    def test_symmetric_uncertainty_is_recovered(self, supplied_err):
        assert supplied_err.logU_err == pytest.approx(0.15, abs=0.02)

    def test_asymmetric_uncertainty_lands_between_the_sides(self, supplied_asym):
        assert 0.10 < supplied_asym.logU_err < 0.30


class TestNothingElseMoves:
    """log(U) enters only the ICFs; everything else must be untouched."""

    def test_oxygen_is_identical(self, derived, supplied):
        assert supplied.OH == pytest.approx(derived.OH, abs=1e-9)

    def test_oxygen_is_identical_with_an_error_bar(self, derived, supplied_err):
        """The supplied draws use their own RNG stream, so O/H cannot move."""
        assert supplied_err.OH == pytest.approx(derived.OH, abs=1e-9)

    def test_oxygen_posterior_is_identical(self, derived, supplied_err):
        assert np.allclose(supplied_err.OH_posterior, derived.OH_posterior)

    def test_temperature_is_identical(self, derived, supplied_err):
        assert supplied_err.Te_high == pytest.approx(derived.Te_high)

    def test_densities_are_identical(self, derived, supplied_err):
        assert supplied_err.ne_low == pytest.approx(derived.ne_low)
        assert supplied_err.ne_high == pytest.approx(derived.ne_high)


class TestDiagnostics:
    def test_says_it_was_supplied(self, supplied):
        assert "supplied via logU_override" in supplied.diagnostics["log(U)"]

    def test_names_no_line_ratio(self, supplied):
        d = supplied.diagnostics["log(U)"]
        assert "N43" not in d and "O32" not in d

    def test_quotes_the_value(self, supplied):
        assert "-2.30" in supplied.diagnostics["log(U)"]

    def test_quotes_a_symmetric_error(self, supplied_err):
        assert "+/- 0.15" in supplied_err.diagnostics["log(U)"]

    def test_quotes_an_asymmetric_error(self, supplied_asym):
        assert "+0.10 -0.30" in supplied_asym.diagnostics["log(U)"]

    def test_warns_that_no_error_means_no_scatter(self, supplied):
        assert "no uncertainty given" in supplied.diagnostics["log(U)"]

    def test_no_such_warning_when_an_error_is_given(self, supplied_err):
        assert "no uncertainty given" not in supplied_err.diagnostics["log(U)"]

    def test_says_one_value_serves_both_zones(self, supplied):
        assert "both the N/O (high) and C/O (intermediate) zones" in (
            supplied.diagnostics["log(U)"]
        )

    def test_derived_run_is_unaffected(self, derived):
        assert "supplied" not in derived.diagnostics["log(U)"]


class TestTruncationReporting:
    """The trap: an error bar straddling a calibration edge."""

    KEY = "log(U) draws rejected"

    def test_absent_when_safely_inside(self, supplied_err):
        assert self.KEY not in supplied_err.diagnostics

    def test_present_when_straddling(self, straddling_edge):
        assert self.KEY in straddling_edge.diagnostics

    def test_quotes_a_fraction(self, straddling_edge):
        assert "%" in straddling_edge.diagnostics[self.KEY]

    def test_explains_the_bias(self, straddling_edge):
        txt = straddling_edge.diagnostics[self.KEY]
        assert "truncates" in txt and "error bars are too small" in txt

    def test_suggests_a_fix(self, straddling_edge):
        assert "marginalize_logU=False" in straddling_edge.diagnostics[self.KEY]

    def test_the_truncation_is_real(self, straddling_edge):
        """The reported error must come back smaller than the one supplied."""
        assert straddling_edge.logU_err < 0.30
