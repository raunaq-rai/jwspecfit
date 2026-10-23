"""Rank-normalised split R-hat (Vehtari et al. 2021)."""

from __future__ import annotations

import numpy as np
import pytest

from jwspecmcmc.diagnostics import (
    gelman_rubin,
    rank_normalized_rhat,
    summarise_convergence,
)


def _chains(seed=0, shape=(4, 1000, 1)):
    return np.random.default_rng(seed).standard_normal(shape)


def test_mixed_chains_are_near_one():
    assert rank_normalized_rhat(_chains())[0] == pytest.approx(1.0, abs=0.01)


def test_shifted_chain_is_flagged():
    x = _chains()
    x[0] += 1.0
    assert rank_normalized_rhat(x)[0] > 1.05


def test_scale_difference_is_caught_by_the_tail_term():
    """Equal means, one chain 3x wider: the classic statistic misses it."""
    x = _chains(shape=(4, 2000, 1))
    x[0] *= 3.0
    assert gelman_rubin(x)[0] < 1.01
    assert rank_normalized_rhat(x)[0] > 1.01


def test_trend_within_a_chain_is_caught_by_the_split():
    x = _chains(shape=(4, 1000, 1))
    x[:, :, 0] += np.linspace(-1.0, 1.0, 1000)
    assert rank_normalized_rhat(x)[0] > 1.05


def test_matches_arviz():
    az = pytest.importorskip("arviz")
    x = np.random.default_rng(1).standard_t(3, size=(4, 1001, 2))
    x[2, :, 1] += 0.5
    ref = [float(az.rhat(az.convert_to_dataset(x[:, :, [d]]), method="rank")
                 .to_array().squeeze()) for d in range(2)]
    np.testing.assert_allclose(rank_normalized_rhat(x), ref, atol=1e-3)


def test_constant_parameter_is_nan():
    x = _chains(shape=(4, 100, 2))
    x[:, :, 1] = 3.0
    out = rank_normalized_rhat(x)
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_summary_uses_rank_rhat_and_1p01():
    x = _chains(shape=(4, 2000, 1))
    x[0] *= 3.0                       # classic R-hat would pass this
    conv = summarise_convergence(x)
    assert conv["r_hat_max"] == pytest.approx(rank_normalized_rhat(x)[0])
    assert not conv["converged"]
