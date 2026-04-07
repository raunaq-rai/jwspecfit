"""Log-likelihood and log-probability for MCMC sampling.

The likelihood uses the same weighted chi-squared as
:func:`jwspecfit.fitter.fit_lines`, ensuring identical statistical
treatment of the data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from typing import Callable

from jwspecfit.constraints import ConstraintSet
from jwspecfit.models import build_model

from .priors import PriorSet

logger = logging.getLogger(__name__)


@dataclass
class LikelihoodSpec:
    """Cached data needed for likelihood evaluation.

    All arrays are in F_lambda units (erg/s/cm^2/Angstrom).

    Parameters
    ----------
    flam : np.ndarray
        Continuum-subtracted flux density.
    flam_err : np.ndarray
        Flux density errors.
    valid : np.ndarray
        Boolean mask of valid pixels.
    edges : np.ndarray
        Pixel-edge wavelengths in Angstroms (length ``n_pix + 1``).
    n_lines : int
        Number of emission lines.
    constraints : ConstraintSet
        Parameter constraints.
    w_pix : np.ndarray
        Pixel weights from :func:`jwspecfit.models.pixel_weight`.
    n_lya : int
        Number of extra Lyα parameters appended to the free vector
        (4 when Lyα is being fit, 0 otherwise).
    lya_model_fn : callable or None
        Function ``(p_lya,) -> np.ndarray`` that evaluates the skewed
        Gaussian Lyα model from a 4-element parameter vector.
    """

    flam: np.ndarray
    flam_err: np.ndarray
    valid: np.ndarray
    edges: np.ndarray
    n_lines: int
    constraints: ConstraintSet
    w_pix: np.ndarray
    n_lya: int = 0
    lya_model_fn: Callable | None = None


def log_likelihood(p_free: np.ndarray, spec: LikelihoodSpec) -> float:
    """Gaussian log-likelihood for free parameters.

    Mirrors the residual function in ``jwspecfit.fitter.fit_lines``:
    ``-0.5 * sum(((flam - model) / flam_err * w_pix)^2)``.

    Parameters
    ----------
    p_free : np.ndarray
        Free parameter vector.  When Lyα is being fit, the last
        ``spec.n_lya`` elements are the Lyα parameters.
    spec : LikelihoodSpec
        Cached data for evaluation.

    Returns
    -------
    float
        Log-likelihood value.
    """
    n_gauss = len(p_free) - spec.n_lya
    p_gauss = p_free[:n_gauss]
    p_full = spec.constraints.expand_free_to_full(p_gauss)
    model = build_model(p_full, spec.edges, spec.n_lines)

    if spec.n_lya > 0 and spec.lya_model_fn is not None:
        model = model + spec.lya_model_fn(p_free[-spec.n_lya:])

    resid = np.zeros_like(spec.flam)
    v = spec.valid
    resid[v] = (spec.flam[v] - model[v]) / spec.flam_err[v] * spec.w_pix[v]

    return -0.5 * np.dot(resid, resid)


def log_probability(
    p_free: np.ndarray,
    spec: LikelihoodSpec,
    prior_set: PriorSet,
) -> float:
    """Log-posterior = log-prior + log-likelihood.

    Parameters
    ----------
    p_free : np.ndarray
        Free parameter vector.
    spec : LikelihoodSpec
        Cached data for evaluation.
    prior_set : PriorSet
        Prior distributions for free parameters.

    Returns
    -------
    float
        Log-posterior value (``-inf`` if outside prior support).
    """
    lp = prior_set.log_prior(p_free)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(p_free, spec)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll
