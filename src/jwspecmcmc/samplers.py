"""MCMC sampler wrappers for emcee and nautilus.

Both wrappers accept a :class:`~jwspecmcmc.likelihood.LikelihoodSpec`
and :class:`~jwspecmcmc.priors.PriorSet`, run the sampler, and return
a common result dict.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .likelihood import LikelihoodSpec, log_probability
from .priors import PriorSet

logger = logging.getLogger(__name__)


def run_emcee(
    spec: LikelihoodSpec,
    prior_set: PriorSet,
    p0_free: np.ndarray,
    *,
    n_walkers: int = 64,
    n_steps: int = 2000,
    n_burn: int | None = None,
    progress: bool = True,
    seed: int = 42,
    moves: Any = None,
) -> dict[str, Any]:
    """Run the emcee ensemble sampler.

    Parameters
    ----------
    spec : LikelihoodSpec
        Cached data for likelihood evaluation.
    prior_set : PriorSet
        Prior distributions.
    p0_free : np.ndarray
        MLE estimate in free-parameter space (used to initialise walkers).
    n_walkers : int
        Number of walkers (default 64).
    n_steps : int
        Number of MCMC steps (default 2000).
    n_burn : int or None
        Burn-in steps to discard.  If ``None``, estimated from the
        integrated autocorrelation time.
    progress : bool
        Show a progress bar.
    seed : int
        Random seed.
    moves : optional
        Custom emcee moves.  If ``None``, uses the default
        ``StretchMove``.

    Returns
    -------
    dict
        Keys: ``flat_chains`` (n_samples, n_dim), ``flat_log_prob``
        (n_samples,), ``chains`` (n_walkers, n_steps_kept, n_dim),
        ``log_prob_chains`` (n_walkers, n_steps_kept),
        ``n_burn`` (int), ``sampler_name`` (str),
        ``sampler_meta`` (dict).
    """
    import emcee

    from .priors import GaussianPrior, LogUniformPrior, UniformPrior

    n_dim = prior_set.n_dim

    # emcee requires n_walkers >= 2 * n_dim for the stretch move.
    min_walkers = 2 * n_dim + 2  # +2 for safety (must be even)
    if n_walkers < min_walkers:
        n_walkers = min_walkers + (min_walkers % 2)  # ensure even
        logger.info("Increased n_walkers to %d (>= 2 * n_dim = %d).", n_walkers, 2 * n_dim)

    rng = np.random.default_rng(seed)

    # Initialise walkers as a Gaussian ball around the MLE.
    # Use 1% of the prior range as the scatter scale to ensure walkers
    # are spread enough to be linearly independent while staying close
    # to the MLE.
    scale = np.zeros(n_dim)
    for i, prior in enumerate(prior_set.priors):
        if isinstance(prior, (UniformPrior, LogUniformPrior)):
            scale[i] = 0.01 * (prior.hi - prior.lo)
        elif isinstance(prior, GaussianPrior):
            scale[i] = 0.01 * prior.std
        else:
            scale[i] = max(np.abs(p0_free[i]) * 1e-3, 1e-30)
    scale = np.maximum(scale, 1e-30)

    p0 = p0_free[np.newaxis, :] + scale[np.newaxis, :] * rng.standard_normal(
        (n_walkers, n_dim)
    )

    # Clip to prior support to avoid -inf at start.
    for i, prior in enumerate(prior_set.priors):
        if isinstance(prior, (UniformPrior, LogUniformPrior)):
            lo, hi = prior.lo, prior.hi
            p0[:, i] = np.clip(p0[:, i], lo + 1e-30, hi - 1e-30)
        elif isinstance(prior, GaussianPrior):
            if np.isfinite(prior.lo):
                p0[:, i] = np.maximum(p0[:, i], prior.lo + 1e-30)
            if np.isfinite(prior.hi):
                p0[:, i] = np.minimum(p0[:, i], prior.hi - 1e-30)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        log_probability,
        args=(spec, prior_set),
        moves=moves,
    )

    logger.info(
        "Running emcee: %d walkers, %d steps, %d dims",
        n_walkers, n_steps, n_dim,
    )
    # skip_initial_state_check: emcee's condition number check can fail
    # when parameters span many orders of magnitude (e.g. amplitude ~1e-18,
    # centroid ~30000 Å).  This is cosmetic — the sampling is fine.
    sampler.run_mcmc(p0, n_steps, progress=progress, skip_initial_state_check=True)

    # Determine burn-in.
    if n_burn is None:
        try:
            tau = sampler.get_autocorr_time(quiet=True)
            n_burn = int(2.0 * np.nanmax(tau))
            logger.info("Auto burn-in from autocorrelation time: %d steps", n_burn)
        except Exception:
            n_burn = n_steps // 4
            logger.info("Autocorrelation time estimation failed; using n_burn=%d", n_burn)
    n_burn = min(n_burn, n_steps - 1)

    chains = sampler.get_chain(discard=n_burn)        # (n_steps_kept, n_walkers, n_dim)
    chains = chains.transpose(1, 0, 2)                # (n_walkers, n_steps_kept, n_dim)
    log_prob_chains = sampler.get_log_prob(discard=n_burn).T  # (n_walkers, n_steps_kept)

    flat_chains = sampler.get_chain(discard=n_burn, flat=True)  # (n_samples, n_dim)
    flat_log_prob = sampler.get_log_prob(discard=n_burn, flat=True)

    return {
        "flat_chains": flat_chains,
        "flat_log_prob": flat_log_prob,
        "chains": chains,
        "log_prob_chains": log_prob_chains,
        "n_burn": n_burn,
        "sampler_name": "emcee",
        "sampler_meta": {
            "n_walkers": n_walkers,
            "n_steps": n_steps,
            "n_dim": n_dim,
            "n_burn": n_burn,
        },
    }


def run_nautilus(
    spec: LikelihoodSpec,
    prior_set: PriorSet,
    *,
    n_live: int = 2000,
    n_eff: int = 10000,
    progress: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the nautilus nested sampler.

    Parameters
    ----------
    spec : LikelihoodSpec
        Cached data for likelihood evaluation.
    prior_set : PriorSet
        Prior distributions.
    n_live : int
        Number of live points (default 2000).
    n_eff : int
        Target effective sample size (default 10000).
    progress : bool
        Show a progress bar.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Same keys as :func:`run_emcee`, except ``chains`` and
        ``log_prob_chains`` are ``None`` (nautilus does not produce
        walker chains).
    """
    from nautilus import Prior as NautilusPrior, Sampler

    from .priors import GaussianPrior, LogUniformPrior, UniformPrior

    n_dim = prior_set.n_dim

    # Build nautilus Prior object from our PriorSet.
    naut_prior = NautilusPrior()
    for i, prior in enumerate(prior_set.priors):
        param_name = f"p{i}"
        if isinstance(prior, UniformPrior):
            naut_prior.add_parameter(param_name, dist=(prior.lo, prior.hi))
        elif isinstance(prior, LogUniformPrior):
            from scipy.stats import loguniform
            naut_prior.add_parameter(
                param_name,
                dist=loguniform(prior.lo, prior.hi),
            )
        elif isinstance(prior, GaussianPrior):
            from scipy.stats import truncnorm
            a = (prior.lo - prior.mean) / prior.std if np.isfinite(prior.lo) else -np.inf
            b = (prior.hi - prior.mean) / prior.std if np.isfinite(prior.hi) else np.inf
            naut_prior.add_parameter(
                param_name,
                dist=truncnorm(a, b, loc=prior.mean, scale=prior.std),
            )
        else:
            # Fallback: treat as uniform on a generous range.
            logger.warning("Unknown prior type for param %d; using uniform fallback.", i)
            naut_prior.add_parameter(param_name, dist=(-1e30, 1e30))

    def likelihood_fn(params: np.ndarray) -> float:
        return log_probability(params, spec, prior_set)

    sampler = Sampler(
        naut_prior,
        likelihood_fn,
        n_live=n_live,
        seed=seed,
    )

    logger.info(
        "Running nautilus: %d live points, target n_eff=%d, %d dims",
        n_live, n_eff, n_dim,
    )
    sampler.run(n_eff=n_eff, verbose=progress)

    points, log_w, log_l = sampler.posterior()
    # Resample to equally-weighted samples.
    weights = np.exp(log_w - np.max(log_w))
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=min(n_eff, len(points)), p=weights)
    flat_chains = points[indices]
    flat_log_prob = log_l[indices]

    return {
        "flat_chains": flat_chains,
        "flat_log_prob": flat_log_prob,
        "chains": None,
        "log_prob_chains": None,
        "n_burn": 0,
        "sampler_name": "nautilus",
        "sampler_meta": {
            "n_live": n_live,
            "n_eff": n_eff,
            "n_dim": n_dim,
            "n_samples": len(flat_chains),
        },
    }
