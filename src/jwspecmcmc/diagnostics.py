"""MCMC convergence diagnostics.

Provides Gelman--Rubin R-hat and effective sample size (ESS) for
assessing chain convergence.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def gelman_rubin(chains: np.ndarray) -> np.ndarray:
    r"""Compute the Gelman--Rubin :math:`\hat{R}` statistic per parameter.

    Parameters
    ----------
    chains : np.ndarray
        MCMC chains of shape ``(n_walkers, n_steps, n_dim)``.

    Returns
    -------
    np.ndarray
        :math:`\hat{R}` per parameter (length ``n_dim``).  Values near
        1.0 indicate convergence; values above ~1.05 suggest the chains
        have not mixed.
    """
    n_walkers, n_steps, n_dim = chains.shape
    if n_walkers < 2:
        return np.full(n_dim, np.nan)

    # Per-chain means and variances.
    chain_means = chains.mean(axis=1)                  # (n_walkers, n_dim)
    chain_vars = chains.var(axis=1, ddof=1)            # (n_walkers, n_dim)

    # Between-chain variance.
    overall_mean = chain_means.mean(axis=0)            # (n_dim,)
    B = n_steps * np.var(chain_means, axis=0, ddof=1)  # (n_dim,)

    # Within-chain variance.
    W = chain_vars.mean(axis=0)                        # (n_dim,)

    # Pooled estimate.
    var_hat = (1.0 - 1.0 / n_steps) * W + (1.0 / n_steps) * B
    r_hat = np.sqrt(var_hat / np.where(W > 0, W, np.nan))
    return r_hat


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Estimate effective sample size (ESS) per parameter via FFT autocorrelation.

    Parameters
    ----------
    chains : np.ndarray
        MCMC chains of shape ``(n_walkers, n_steps, n_dim)``.

    Returns
    -------
    np.ndarray
        ESS per parameter (length ``n_dim``).
    """
    n_walkers, n_steps, n_dim = chains.shape
    ess = np.zeros(n_dim)

    for d in range(n_dim):
        tau_sum = 0.0
        for w in range(n_walkers):
            x = chains[w, :, d]
            x = x - x.mean()
            # FFT-based autocorrelation.
            n = len(x)
            f = np.fft.fft(x, n=2 * n)
            acf = np.fft.ifft(f * np.conj(f))[:n].real
            if acf[0] > 0:
                acf /= acf[0]
            else:
                continue
            # Integrated autocorrelation time: sum until first negative.
            tau = 1.0
            for k in range(1, n):
                if acf[k] < 0:
                    break
                tau += 2.0 * acf[k]
            tau_sum += tau

        avg_tau = tau_sum / max(n_walkers, 1)
        ess[d] = n_walkers * n_steps / max(avg_tau, 1.0)

    return ess


def summarise_convergence(chains: np.ndarray) -> dict[str, Any]:
    """Compute a convergence summary for MCMC chains.

    Parameters
    ----------
    chains : np.ndarray
        MCMC chains of shape ``(n_walkers, n_steps, n_dim)``.

    Returns
    -------
    dict
        Keys: ``r_hat`` (array), ``ess`` (array), ``r_hat_max`` (float),
        ``ess_min`` (float), ``converged`` (bool, True if R-hat < 1.05
        and ESS > 100 for all parameters).
    """
    r_hat = gelman_rubin(chains)
    ess = effective_sample_size(chains)
    r_hat_max = float(np.nanmax(r_hat)) if np.any(np.isfinite(r_hat)) else np.nan
    ess_min = float(np.nanmin(ess)) if np.any(np.isfinite(ess)) else 0.0
    converged = bool(r_hat_max < 1.05 and ess_min > 100)

    return {
        "r_hat": r_hat,
        "ess": ess,
        "r_hat_max": r_hat_max,
        "ess_min": ess_min,
        "converged": converged,
    }
