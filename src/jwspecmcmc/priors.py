"""Prior distributions for MCMC sampling.

Provides prior classes that can be composed into a :class:`PriorSet`
for use with the MCMC samplers.  Default priors are uniform within the
parameter bounds from :func:`jwspecfit.fitter._grating_bounds`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


class Prior(ABC):
    """Abstract base class for a 1-D prior distribution."""

    @abstractmethod
    def log_prob(self, x: float) -> float:
        """Return the log-probability at *x*.

        Parameters
        ----------
        x : float
            Parameter value.

        Returns
        -------
        float
            Log-probability (``-inf`` if outside support).
        """

    @abstractmethod
    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw random samples from the prior.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.
        size : int
            Number of samples.

        Returns
        -------
        np.ndarray
            Samples of shape ``(size,)``.
        """


@dataclass
class UniformPrior(Prior):
    """Uniform (flat) prior on ``[lo, hi]``.

    Parameters
    ----------
    lo : float
        Lower bound.
    hi : float
        Upper bound.
    """

    lo: float
    hi: float

    def log_prob(self, x: float) -> float:
        if self.lo <= x <= self.hi:
            return -np.log(self.hi - self.lo)
        return -np.inf

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        return rng.uniform(self.lo, self.hi, size=size)


@dataclass
class GaussianPrior(Prior):
    """Truncated Gaussian prior.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian.
    std : float
        Standard deviation.
    lo : float
        Hard lower bound (``-inf`` for unbounded).
    hi : float
        Hard upper bound (``+inf`` for unbounded).
    """

    mean: float
    std: float
    lo: float = -np.inf
    hi: float = np.inf

    def log_prob(self, x: float) -> float:
        if not (self.lo <= x <= self.hi):
            return -np.inf
        return -0.5 * ((x - self.mean) / self.std) ** 2

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        samples = rng.normal(self.mean, self.std, size=size)
        return np.clip(samples, self.lo, self.hi)


@dataclass
class LogUniformPrior(Prior):
    """Log-uniform (Jeffreys) prior on ``[lo, hi]`` with ``lo > 0``.

    Parameters
    ----------
    lo : float
        Lower bound (must be positive).
    hi : float
        Upper bound.
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo <= 0:
            raise ValueError("LogUniformPrior requires lo > 0.")

    def log_prob(self, x: float) -> float:
        if self.lo <= x <= self.hi:
            return -np.log(x) - np.log(np.log(self.hi / self.lo))
        return -np.inf

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        log_lo = np.log(self.lo)
        log_hi = np.log(self.hi)
        return np.exp(rng.uniform(log_lo, log_hi, size=size))


@dataclass
class PriorSet:
    """Collection of priors indexed by free-parameter position.

    Parameters
    ----------
    priors : list of Prior
        One prior per free parameter.
    """

    priors: list[Prior] = field(default_factory=list)

    @property
    def n_dim(self) -> int:
        """Number of free parameters."""
        return len(self.priors)

    def log_prior(self, p_free: np.ndarray) -> float:
        """Evaluate the total log-prior for a free-parameter vector.

        Parameters
        ----------
        p_free : np.ndarray
            Free parameter values (length ``n_dim``).

        Returns
        -------
        float
            Sum of individual log-priors (``-inf`` if any parameter
            is outside its support).
        """
        lp = 0.0
        for prior, val in zip(self.priors, p_free):
            lp_i = prior.log_prob(val)
            if not np.isfinite(lp_i):
                return -np.inf
            lp += lp_i
        return lp

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one sample from the joint prior.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Sample of shape ``(n_dim,)``.
        """
        return np.array([p.sample(rng, size=1)[0] for p in self.priors])


def priors_from_bounds(
    lb_free: np.ndarray,
    ub_free: np.ndarray,
    overrides: dict[int, Prior] | None = None,
) -> PriorSet:
    """Build a :class:`PriorSet` from parameter bounds.

    Parameters
    ----------
    lb_free : np.ndarray
        Lower bounds for free parameters.
    ub_free : np.ndarray
        Upper bounds for free parameters.
    overrides : dict mapping int to Prior, optional
        Per-index prior overrides.

    Returns
    -------
    PriorSet
    """
    overrides = overrides or {}
    priors = []
    for i, (lo, hi) in enumerate(zip(lb_free, ub_free)):
        if i in overrides:
            priors.append(overrides[i])
        else:
            priors.append(UniformPrior(lo=float(lo), hi=float(hi)))
    return PriorSet(priors=priors)
