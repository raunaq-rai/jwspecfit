"""MCMC sampler wrappers for emcee, nautilus, and NumPyro NUTS.

All wrappers accept a :class:`~jwspecmcmc.likelihood.LikelihoodSpec`
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


def _auto_n_walkers(n_dim: int) -> int:
    """Choose n_walkers from n_dim and available CPU cores.

    The number of walkers must satisfy ``n_walkers >= 2 * n_dim`` for
    emcee's stretch move.  We round up to the nearest multiple of
    ``n_cores`` so that future pool-based parallelisation distributes
    likelihood evaluations evenly across cores.

    Parameters
    ----------
    n_dim : int
        Number of free parameters.

    Returns
    -------
    int
        Even number of walkers.
    """
    import os

    n_cores = os.cpu_count() or 4
    min_walkers = 2 * n_dim + 2
    # Round up to nearest multiple of n_cores.
    n_walkers = max(min_walkers, n_cores)
    remainder = n_walkers % n_cores
    if remainder:
        n_walkers += n_cores - remainder
    # Ensure even (emcee requirement).
    if n_walkers % 2:
        n_walkers += 1
    return n_walkers


def run_emcee(
    spec: LikelihoodSpec,
    prior_set: PriorSet,
    p0_free: np.ndarray,
    *,
    n_walkers: int | str = "auto",
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
    n_walkers : int or ``"auto"``
        Number of walkers.  ``"auto"`` (default) picks a value based
        on ``n_dim`` and the number of CPU cores.
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

    if n_walkers == "auto":
        n_walkers = _auto_n_walkers(n_dim)
        logger.info("Auto n_walkers=%d (n_dim=%d, n_cores=%d).",
                     n_walkers, n_dim, __import__("os").cpu_count() or 4)

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

    # nautilus v1.0.5 passes params as a dict keyed by parameter name
    # (e.g. {"p0": val0, "p1": val1, ...}).  Convert to a flat array.
    param_names = [f"p{i}" for i in range(n_dim)]

    def likelihood_fn(params: dict) -> float:
        p_arr = np.array([float(params[k]) for k in param_names])
        return log_probability(p_arr, spec, prior_set)

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


def run_nuts(
    spec: LikelihoodSpec,
    prior_set: PriorSet,
    p0_free: np.ndarray,
    *,
    n_warmup: int = 500,
    n_samples: int = 2000,
    n_chains: int = 6,
    progress: bool = True,
    seed: int = 42,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
) -> dict[str, Any]:
    """Run NumPyro NUTS (No-U-Turn Sampler) with JAX-accelerated likelihood.

    Uses automatic differentiation through the emission-line model for
    gradient-based HMC sampling, requiring far fewer likelihood evaluations
    than ensemble samplers like emcee.

    Parameters
    ----------
    spec : LikelihoodSpec
        Cached data for likelihood evaluation.
    prior_set : PriorSet
        Prior distributions.
    p0_free : np.ndarray
        MLE estimate in free-parameter space (used to initialise).
    n_warmup : int
        Number of warmup (adaptation) steps (default 500).
    n_samples : int
        Number of posterior samples per chain (default 2000).
    n_chains : int
        Number of independent chains (default 1).
    progress : bool
        Show a progress bar.
    seed : int
        Random seed.
    target_accept_prob : float
        Target acceptance probability for NUTS adaptation (default 0.8).
    max_tree_depth : int
        Maximum tree depth for NUTS (default 10).

    Returns
    -------
    dict
        Same keys as :func:`run_emcee`.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    from .jax_likelihood import make_jax_log_likelihood
    from .priors import GaussianPrior, LogUniformPrior, UniformPrior

    n_dim = prior_set.n_dim

    # Build JIT-compiled log-likelihood.
    log_likelihood_jax, static_data = make_jax_log_likelihood(spec)

    # Build bounds arrays for the uniform priors.
    lb = np.array([
        p.lo if isinstance(p, (UniformPrior, LogUniformPrior)) else -np.inf
        for p in prior_set.priors
    ])
    ub = np.array([
        p.hi if isinstance(p, (UniformPrior, LogUniformPrior)) else np.inf
        for p in prior_set.priors
    ])
    lb_jax = jnp.array(lb, dtype=jnp.float64)
    ub_jax = jnp.array(ub, dtype=jnp.float64)

    # Pre-compute log-prior normalisation for uniform priors.
    log_prior_norm = 0.0
    for prior in prior_set.priors:
        if isinstance(prior, UniformPrior):
            log_prior_norm -= np.log(prior.hi - prior.lo)

    def numpyro_model():
        # Sample each free parameter with its prior.
        p_free = numpyro.sample(
            "params",
            dist.Uniform(lb_jax, ub_jax),
        )
        # Add log-likelihood as a factor.
        ll = log_likelihood_jax(p_free)
        numpyro.factor("log_likelihood", ll)

    # Validate initial parameters before passing to NumPyro.
    p0_jax = jnp.array(p0_free, dtype=jnp.float64)
    oob_lo = p0_free <= lb
    oob_hi = p0_free >= ub
    if np.any(oob_lo) or np.any(oob_hi):
        bad = np.where(oob_lo | oob_hi)[0]
        for idx in bad:
            logger.warning(
                "  p0_free[%d] = %.6e  bounds = [%.6e, %.6e] %s",
                idx, p0_free[idx], lb[idx], ub[idx],
                "BELOW" if oob_lo[idx] else "ABOVE",
            )
        # Clip to strictly inside bounds.
        p0_free = np.clip(p0_free, lb + 1e-10 * np.abs(lb + 1), ub - 1e-10 * np.abs(ub + 1))
        p0_jax = jnp.array(p0_free, dtype=jnp.float64)

    ll_init = float(log_likelihood_jax(p0_jax))
    if not np.isfinite(ll_init):
        logger.error(
            "Initial log-likelihood is %s — NUTS cannot start. "
            "Check that bounds and overrides produce a valid model.",
            ll_init,
        )
        raise RuntimeError(
            f"NUTS initialisation failed: log-likelihood = {ll_init} at p0. "
            f"This usually means parameter bounds are too tight or inconsistent."
        )
    logger.info("Initial log-likelihood at p0: %.2f", ll_init)

    kernel = NUTS(
        numpyro_model,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=numpyro.infer.init_to_value(
            values={"params": p0_jax},
        ),
    )

    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
        progress_bar=progress,
    )

    logger.info(
        "Running NumPyro NUTS: %d warmup, %d samples, %d chain(s), %d dims",
        n_warmup, n_samples, n_chains, n_dim,
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key)

    # Extract samples.
    samples = mcmc.get_samples()["params"]  # (n_samples * n_chains, n_dim)
    flat_chains = np.array(samples)

    # Compute log-probabilities for each sample.
    log_prob_fn = jax.jit(lambda p: log_likelihood_jax(p) + log_prior_norm)
    flat_log_prob = np.array(jax.vmap(log_prob_fn)(jnp.array(flat_chains)))

    # Extract diagnostics.
    extra_fields = mcmc.get_extra_fields()
    diverging = extra_fields.get("diverging", None)
    n_divergent = int(np.sum(diverging)) if diverging is not None else 0
    if n_divergent > 0:
        logger.warning(
            "%d divergent transitions detected (%.1f%%). Consider increasing "
            "target_accept_prob or max_tree_depth.",
            n_divergent, 100.0 * n_divergent / len(flat_chains),
        )

    return {
        "flat_chains": flat_chains,
        "flat_log_prob": flat_log_prob,
        "chains": None,
        "log_prob_chains": None,
        "n_burn": 0,
        "sampler_name": "nuts",
        "sampler_meta": {
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "n_chains": n_chains,
            "n_dim": n_dim,
            "n_divergent": n_divergent,
            "target_accept_prob": target_accept_prob,
            "max_tree_depth": max_tree_depth,
        },
    }
