"""Bayesian forward model for emission-line abundance determination.

Implements the forward modelling approach of Cullen et al. (2025): free
parameters (ionic abundances, T_e, n_e) are sampled via MCMC or nested
sampling.  For each parameter vector, CEL emissivities from PyNEB and
the Aller (1984) Hbeta recombination formula predict line flux ratios
that are compared to the observed ratios via a Gaussian likelihood.

References
----------
- Aller L. H., 1984, Physics of Thermal Gaseous Nebulae (Reidel)
- Cullen et al. (2025) — forward modelling for EXCELS galaxies
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants and Aller (1984) Hbeta emissivity
# ---------------------------------------------------------------------------

# Planck constant * speed of light in CGS (erg cm).
_HC_CGS = 6.62607015e-27 * 2.99792458e10

# Hbeta vacuum wavelength in cm.
_LAMBDA_HB_CM = 4862.68e-8


def hbeta_emissivity_aller84(Te: float) -> float:
    r"""Hbeta volume emissivity coefficient using the Aller (1984) Case B formula.

    .. math::

        \varepsilon_{H\beta} = \alpha_{H\beta}^{\rm eff}(T)\;h\nu_{H\beta}

    where :math:`\alpha_{H\beta}^{\rm eff} = 3.03\times10^{-14}\,(T/10^4)^{-0.874}`
    cm\ :sup:`3` s\ :sup:`-1`.

    Unlike PyNEB's recombination tables, this formula is valid at
    *all* temperatures — no upper-bound limitation at 30,000 K.

    Parameters
    ----------
    Te : float
        Electron temperature in K.

    Returns
    -------
    float
        :math:`\varepsilon_{H\beta}` in erg cm\ :sup:`3` s\ :sup:`-1`.
    """
    alpha_hb = 3.03e-14 * (Te / 1e4) ** (-0.874)
    hnu_hb = _HC_CGS / _LAMBDA_HB_CM
    return alpha_hb * hnu_hb


# ---------------------------------------------------------------------------
# Line -> ion mapping
# ---------------------------------------------------------------------------

# Each line name maps to a list of (element, ion_stage, pyneb_wave) tuples.
# Most lines are single; blended doublets (e.g. OII_doublet) sum components.
_LINE_COMPONENTS: dict[str, list[tuple[str, int, int]]] = {
    "OIII_5007":   [("O", 3, 5007)],
    "OIII_4959":   [("O", 3, 4959)],
    "OIII_4363":   [("O", 3, 4363)],
    "OII_3726":    [("O", 2, 3726)],
    "OII_3729":    [("O", 2, 3729)],
    "OII_doublet": [("O", 2, 3726), ("O", 2, 3729)],
    "NII_6585":    [("N", 2, 6584)],
    "NII_6550":    [("N", 2, 6548)],
    "NeIII_3869":  [("Ne", 3, 3869)],
    "SII_6718":    [("S", 2, 6718)],
    "SII_6732":    [("S", 2, 6732)],
    "SIII_9069":   [("S", 3, 9069)],
    "ArIII_7136":  [("Ar", 3, 7136)],
    # UV semi-forbidden lines
    "NIV_1":       [("N", 4, 1487)],
    "NIII_1749":   [("N", 3, 1749)],
    "NIII_1752":   [("N", 3, 1752)],
    "NIII_doublet": [("N", 3, 1749), ("N", 3, 1752)],
    "CIII]":       [("C", 3, 1909)],
    "CIV_1":       [("C", 4, 1548)],
    "CIV_2":       [("C", 4, 1551)],
    "CIV_doublet": [("C", 4, 1548), ("C", 4, 1551)],
    "OIII_1661":   [("O", 3, 1661)],
    "OIII_1666":   [("O", 3, 1666)],
}

# Maps (element, ion_stage) -> parameter name for ionic abundance.
_ION_PARAM_NAME: dict[tuple[str, int], str] = {
    ("O", 3):  "log_Opp",
    ("O", 2):  "log_Op",
    ("N", 2):  "log_Np",
    ("N", 3):  "log_Npp",
    ("N", 4):  "log_Nppp",
    ("Ne", 3): "log_Nepp",
    ("S", 2):  "log_Sp",
    ("S", 3):  "log_Spp",
    ("Ar", 3): "log_Arpp",
    ("C", 3):  "log_Cpp",
    ("C", 4):  "log_Cppp",
}

# Human-readable labels for ionic abundance parameters.
_PARAM_LABELS: dict[str, str] = {
    "log_Te":    r"$\log\,T_e$ [K]",
    "log_ne":    r"$\log\,n_e$ [cm$^{-3}$]",
    "log_Opp":   r"$\log\,(\mathrm{O}^{2+}/\mathrm{H}^+)$",
    "log_Op":    r"$\log\,(\mathrm{O}^{+}/\mathrm{H}^+)$",
    "log_Np":    r"$\log\,(\mathrm{N}^{+}/\mathrm{H}^+)$",
    "log_Npp":   r"$\log\,(\mathrm{N}^{2+}/\mathrm{H}^+)$",
    "log_Nppp":  r"$\log\,(\mathrm{N}^{3+}/\mathrm{H}^+)$",
    "log_Nepp":  r"$\log\,(\mathrm{Ne}^{2+}/\mathrm{H}^+)$",
    "log_Sp":    r"$\log\,(\mathrm{S}^{+}/\mathrm{H}^+)$",
    "log_Spp":   r"$\log\,(\mathrm{S}^{2+}/\mathrm{H}^+)$",
    "log_Arpp":  r"$\log\,(\mathrm{Ar}^{2+}/\mathrm{H}^+)$",
    "log_Cpp":   r"$\log\,(\mathrm{C}^{2+}/\mathrm{H}^+)$",
    "log_Cppp":  r"$\log\,(\mathrm{C}^{3+}/\mathrm{H}^+)$",
}

# Uniform prior bounds for each parameter.
_PRIOR_BOUNDS: dict[str, tuple[float, float]] = {
    "log_Te":    (3.5, 5.0),
    "log_ne":    (0.0, 5.0),
    "log_Opp":   (-7.0, -2.0),
    "log_Op":    (-7.0, -2.0),
    "log_Np":    (-8.0, -3.0),
    "log_Npp":   (-8.0, -3.0),
    "log_Nppp":  (-8.0, -3.0),
    "log_Nepp":  (-8.0, -3.0),
    "log_Sp":    (-9.0, -4.0),
    "log_Spp":   (-9.0, -4.0),
    "log_Arpp":  (-9.0, -4.0),
    "log_Cpp":   (-8.0, -3.0),
    "log_Cppp":  (-8.0, -3.0),
}


# ---------------------------------------------------------------------------
# Observables and parameter construction
# ---------------------------------------------------------------------------

def _build_observables(
    line_fluxes: dict[str, float],
    line_errors: dict[str, float],
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    list[list[tuple[str, int, int]]],
]:
    """Build observed line/Hbeta flux ratios from available lines.

    Parameters
    ----------
    line_fluxes : dict
        ``{line_name: flux}``.  Must include ``"HBETA"``.
    line_errors : dict
        ``{line_name: flux_err}``.

    Returns
    -------
    ratio_names : list of str
        Human-readable labels for each observable, e.g. ``"OIII_5007/Hb"``.
    obs_ratios : ndarray
        Observed line/Hbeta ratios.
    ratio_errors : ndarray
        1-sigma errors on the ratios (Gaussian propagation).
    obs_components : list of list of tuple
        For each observable, the list of PyNEB ``(elem, ion, wave)``
        components that sum to give the line flux.
    """
    f_hb = line_fluxes.get("HBETA", 0.0)
    e_hb = line_errors.get("HBETA", 0.0)
    if f_hb <= 0:
        raise ValueError("HBETA flux is required and must be positive.")

    ratio_names: list[str] = []
    obs_ratios: list[float] = []
    ratio_errs: list[float] = []
    obs_components: list[list[tuple[str, int, int]]] = []

    for name, components in _LINE_COMPONENTS.items():
        if name not in line_fluxes or line_fluxes[name] <= 0:
            continue

        f_line = line_fluxes[name]
        e_line = line_errors.get(name, 0.0)

        ratio = f_line / f_hb
        # Error propagation: sigma_R = R * sqrt((sigma_l/f_l)^2 + (sigma_hb/f_hb)^2)
        if e_line > 0 and e_hb > 0:
            rel_err = ratio * np.sqrt(
                (e_line / f_line) ** 2 + (e_hb / f_hb) ** 2
            )
        else:
            rel_err = 0.1 * ratio  # 10 % default

        ratio_names.append(f"{name}/Hb")
        obs_ratios.append(ratio)
        ratio_errs.append(rel_err)
        obs_components.append(components)

    if not ratio_names:
        raise ValueError("No valid emission lines found for forward modelling.")

    return (
        ratio_names,
        np.array(obs_ratios),
        np.array(ratio_errs),
        obs_components,
    )


def _determine_free_params(
    obs_components: list[list[tuple[str, int, int]]],
) -> list[str]:
    """Determine free parameters from the ions present in the data.

    Always includes ``log_Te`` and ``log_ne``, plus one log-ionic-abundance
    parameter for each distinct ``(element, ion)`` pair.

    Parameters
    ----------
    obs_components : list
        Output of :func:`_build_observables`.

    Returns
    -------
    list of str
        Parameter names, e.g. ``["log_Te", "log_ne", "log_Opp", "log_Nepp"]``.
    """
    params = ["log_Te", "log_ne"]
    seen_ions: set[tuple[str, int]] = set()

    for components in obs_components:
        for elem, ion, _ in components:
            key = (elem, ion)
            if key not in seen_ions:
                seen_ions.add(key)
                param_name = _ION_PARAM_NAME.get(key)
                if param_name is not None:
                    params.append(param_name)

    return params


# ---------------------------------------------------------------------------
# Model prediction, likelihood, prior, posterior
# ---------------------------------------------------------------------------

def _predict_ratios(
    theta: np.ndarray,
    param_names: list[str],
    obs_components: list[list[tuple[str, int, int]]],
    pyneb_atoms: dict[tuple[str, int], Any],
) -> np.ndarray:
    """Predict line/Hbeta ratios for a given parameter vector.

    Parameters
    ----------
    theta : ndarray
        Current parameter values.
    param_names : list of str
        Names matching *theta*.
    obs_components : list
        Per-observable list of ``(elem, ion, wave)`` components.
    pyneb_atoms : dict
        Pre-initialised PyNEB ``Atom`` objects keyed by ``(elem, ion)``.

    Returns
    -------
    ndarray
        Predicted line/Hbeta ratios (same length as *obs_components*).
    """
    params = dict(zip(param_names, theta))
    Te = 10.0 ** params["log_Te"]
    ne = 10.0 ** params["log_ne"]

    eps_hb = hbeta_emissivity_aller84(Te)

    predicted = np.empty(len(obs_components))
    for i, components in enumerate(obs_components):
        # All components belong to the same ion.
        elem0, ion0, _ = components[0]
        ion_key = (elem0, ion0)
        param_name = _ION_PARAM_NAME[ion_key]
        abund = 10.0 ** params[param_name]

        atom = pyneb_atoms[ion_key]
        eps_line = sum(
            atom.getEmissivity(Te, ne, wave=w)
            for _, _, w in components
        )

        # F_line / F_Hb = (X^i+ / H+) * eps_line / eps_Hb
        predicted[i] = abund * eps_line / eps_hb

    return predicted


def _log_likelihood(
    theta: np.ndarray,
    param_names: list[str],
    obs_ratios: np.ndarray,
    ratio_errors: np.ndarray,
    obs_components: list[list[tuple[str, int, int]]],
    pyneb_atoms: dict[tuple[str, int], Any],
) -> float:
    """Gaussian log-likelihood comparing predicted to observed ratios."""
    try:
        predicted = _predict_ratios(
            theta, param_names, obs_components, pyneb_atoms,
        )
    except Exception:
        return -np.inf

    if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0):
        return -np.inf

    residuals = (obs_ratios - predicted) / ratio_errors
    return -0.5 * np.sum(residuals ** 2)


def _log_prior(theta: np.ndarray, param_names: list[str]) -> float:
    """Uniform log-prior within prescribed bounds."""
    for val, name in zip(theta, param_names):
        lo, hi = _PRIOR_BOUNDS[name]
        if val < lo or val > hi:
            return -np.inf
    return 0.0


def _log_posterior(
    theta: np.ndarray,
    param_names: list[str],
    obs_ratios: np.ndarray,
    ratio_errors: np.ndarray,
    obs_components: list[list[tuple[str, int, int]]],
    pyneb_atoms: dict[tuple[str, int], Any],
) -> float:
    """Log-posterior = log-prior + log-likelihood."""
    lp = _log_prior(theta, param_names)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(
        theta, param_names, obs_ratios, ratio_errors,
        obs_components, pyneb_atoms,
    )


# ---------------------------------------------------------------------------
# Sampler drivers
# ---------------------------------------------------------------------------

def _find_map(
    param_names: list[str],
    obs_ratios: np.ndarray,
    ratio_errors: np.ndarray,
    obs_components: list[list[tuple[str, int, int]]],
    pyneb_atoms: dict[tuple[str, int], Any],
    seed: int,
) -> np.ndarray:
    """Find a MAP estimate to seed the sampler.

    Runs ``scipy.optimize.differential_evolution`` to find a good
    starting point within the prior bounds.

    Parameters
    ----------
    param_names, obs_ratios, ratio_errors, obs_components, pyneb_atoms
        Model specification (see :func:`forward_model`).
    seed : int
        Random seed.

    Returns
    -------
    ndarray
        MAP parameter vector.
    """
    from scipy.optimize import differential_evolution

    bounds = [_PRIOR_BOUNDS[n] for n in param_names]

    def neg_logpost(theta):
        val = _log_posterior(
            theta, param_names, obs_ratios, ratio_errors,
            obs_components, pyneb_atoms,
        )
        return -val if np.isfinite(val) else 1e30

    result = differential_evolution(
        neg_logpost, bounds, seed=seed,
        maxiter=500, tol=1e-6, polish=True,
    )

    logger.info("MAP optimisation: success=%s, fun=%.2f", result.success, result.fun)
    return result.x


def _run_emcee(
    param_names: list[str],
    obs_ratios: np.ndarray,
    ratio_errors: np.ndarray,
    obs_components: list[list[tuple[str, int, int]]],
    pyneb_atoms: dict[tuple[str, int], Any],
    *,
    n_walkers: int,
    n_steps: int,
    n_burn: int,
    seed: int,
    progress: bool,
) -> np.ndarray:
    """Run emcee MCMC and return flattened posterior samples.

    Parameters
    ----------
    param_names, obs_ratios, ratio_errors, obs_components, pyneb_atoms
        Model specification.
    n_walkers : int
        Number of walkers.
    n_steps : int
        Total MCMC steps (including burn-in).
    n_burn : int
        Burn-in steps to discard.
    seed : int
        Random seed.
    progress : bool
        Show progress bar.

    Returns
    -------
    ndarray, shape ``(n_samples, n_dim)``
        Flattened posterior samples after burn-in removal.
    """
    import emcee

    n_dim = len(param_names)
    rng = np.random.default_rng(seed)

    # Seed walkers around the MAP estimate.
    map_est = _find_map(
        param_names, obs_ratios, ratio_errors,
        obs_components, pyneb_atoms, seed,
    )
    logger.info("MAP estimate: %s", dict(zip(param_names, map_est)))

    p0 = np.empty((n_walkers, n_dim))
    for j in range(n_dim):
        lo, hi = _PRIOR_BOUNDS[param_names[j]]
        scale = 0.01 * (hi - lo)
        p0[:, j] = np.clip(
            rng.normal(map_est[j], scale, n_walkers), lo + 1e-6, hi - 1e-6,
        )

    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, _log_posterior,
        args=(
            param_names, obs_ratios, ratio_errors,
            obs_components, pyneb_atoms,
        ),
    )
    sampler.run_mcmc(p0, n_steps, progress=progress)

    return sampler.get_chain(discard=n_burn, flat=True)


def _run_dynesty(
    param_names: list[str],
    obs_ratios: np.ndarray,
    ratio_errors: np.ndarray,
    obs_components: list[list[tuple[str, int, int]]],
    pyneb_atoms: dict[tuple[str, int], Any],
    *,
    n_live: int,
    seed: int,
) -> np.ndarray:
    """Run dynesty nested sampling and return weighted posterior samples.

    Parameters
    ----------
    param_names, obs_ratios, ratio_errors, obs_components, pyneb_atoms
        Model specification.
    n_live : int
        Number of live points.
    seed : int
        Random seed.

    Returns
    -------
    ndarray, shape ``(n_samples, n_dim)``
        Equally-weighted posterior samples.
    """
    try:
        import dynesty
    except ImportError as exc:
        raise ImportError(
            "dynesty is required for nested sampling. "
            "Install with: pip install dynesty"
        ) from exc

    n_dim = len(param_names)

    def loglike(theta):
        return _log_likelihood(
            theta, param_names, obs_ratios, ratio_errors,
            obs_components, pyneb_atoms,
        )

    def prior_transform(u):
        """Map unit cube [0, 1]^n to the prior volume."""
        theta = np.empty(n_dim)
        for j, name in enumerate(param_names):
            lo, hi = _PRIOR_BOUNDS[name]
            theta[j] = lo + u[j] * (hi - lo)
        return theta

    sampler = dynesty.NestedSampler(
        loglike, prior_transform, n_dim,
        nlive=n_live, rstate=np.random.default_rng(seed),
    )
    sampler.run_nested(print_progress=True)

    results = sampler.results
    # Resample to get equally-weighted samples.
    from dynesty.utils import resample_equal
    weights = np.exp(results.logwt - results.logz[-1])
    return resample_equal(results.samples, weights)


# ---------------------------------------------------------------------------
# Build result dict from posterior samples
# ---------------------------------------------------------------------------

def _build_forward_result(
    samples: np.ndarray,
    param_names: list[str],
    ratio_names: list[str],
) -> dict[str, Any]:
    """Summarise posterior samples into a result dictionary.

    Parameters
    ----------
    samples : ndarray, shape ``(n, n_dim)``
        Posterior samples.
    param_names : list of str
        Parameter names.
    ratio_names : list of str
        Observable names (for metadata).

    Returns
    -------
    dict
        Contains posteriors, medians, and 16/84-percentile intervals
        for Te, ne, ionic abundances, and 12+log(O/H).
    """
    result: dict[str, Any] = {
        "param_names": param_names,
        "param_labels": [_PARAM_LABELS.get(n, n) for n in param_names],
        "samples": samples,
        "ratio_names": ratio_names,
    }

    def _summarise(arr: np.ndarray) -> tuple[float, tuple[float, float]]:
        med = float(np.median(arr))
        lo = float(med - np.percentile(arr, 16))
        hi = float(np.percentile(arr, 84) - med)
        return med, (lo, hi)

    # Te
    idx_Te = param_names.index("log_Te")
    Te_post = 10.0 ** samples[:, idx_Te]
    result["Te_posterior"] = Te_post
    result["Te"], result["Te_err"] = _summarise(Te_post)

    # ne
    idx_ne = param_names.index("log_ne")
    ne_post = 10.0 ** samples[:, idx_ne]
    result["ne_posterior"] = ne_post
    result["ne"], result["ne_err"] = _summarise(ne_post)

    # Ionic abundance posteriors
    ionic_posteriors: dict[str, np.ndarray] = {}
    ionic_medians: dict[str, float] = {}
    for j, name in enumerate(param_names):
        if name.startswith("log_") and name not in ("log_Te", "log_ne"):
            ionic_posteriors[name] = samples[:, j]
            ionic_medians[name] = float(np.median(samples[:, j]))
    result["ionic_posteriors"] = ionic_posteriors
    result["ionic"] = ionic_medians

    # 12 + log(O/H)
    if "log_Opp" in ionic_posteriors:
        log_opp = ionic_posteriors["log_Opp"]
        if "log_Op" in ionic_posteriors:
            log_op = ionic_posteriors["log_Op"]
            OH = 10.0 ** log_opp + 10.0 ** log_op
        else:
            # No [OII] — assume O^2+ ~ O  (paper's assumption)
            OH = 10.0 ** log_opp
        OH_12 = 12.0 + np.log10(OH)
        result["OH_posterior"] = OH_12
        result["OH"], result["OH_err"] = _summarise(OH_12)
    else:
        result["OH"] = np.nan
        result["OH_err"] = np.nan
        result["OH_posterior"] = None

    # log(N/O) if N+ and O+ or O++ available
    if "log_Np" in ionic_posteriors:
        log_np = ionic_posteriors["log_Np"]
        if "log_Op" in ionic_posteriors:
            NO_post = log_np - ionic_posteriors["log_Op"]
        elif "log_Opp" in ionic_posteriors:
            # Approximate: N/O ~ N+/O++  (coarse)
            NO_post = log_np - ionic_posteriors["log_Opp"]
        else:
            NO_post = None

        if NO_post is not None:
            result["NO_posterior"] = NO_post
            result["NO"], result["NO_err"] = _summarise(NO_post)
        else:
            result["NO"] = None
            result["NO_err"] = None
            result["NO_posterior"] = None
    else:
        result["NO"] = None
        result["NO_err"] = None
        result["NO_posterior"] = None

    # log(Ne/O) if Ne^2+ and O^2+ available
    if "log_Nepp" in ionic_posteriors and "log_Opp" in ionic_posteriors:
        NeO_post = ionic_posteriors["log_Nepp"] - ionic_posteriors["log_Opp"]
        result["NeO_posterior"] = NeO_post
        result["NeO"], result["NeO_err"] = _summarise(NeO_post)
    else:
        result["NeO"] = None
        result["NeO_err"] = None

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forward_model(
    line_fluxes: dict[str, float],
    line_errors: dict[str, float],
    *,
    sampler: str = "emcee",
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burn: int = 1000,
    n_live: int = 500,
    seed: int = 42,
    progress: bool = True,
) -> dict[str, Any]:
    """Run a Bayesian forward model on emission-line fluxes.

    Free parameters are log(T_e), log(n_e), and one log(ionic abundance)
    per detected ion.  The model predicts line/Hbeta flux ratios using
    PyNEB CEL emissivities and the Aller (1984) Hbeta formula, and the
    Gaussian likelihood compares predictions to observations.

    Parameters
    ----------
    line_fluxes : dict
        ``{line_name: flux}``.  Must include ``"HBETA"``.
    line_errors : dict
        ``{line_name: flux_err}``.
    sampler : str
        ``"emcee"`` (default) or ``"dynesty"``.
    n_walkers : int
        Number of walkers for emcee (default 32).
    n_steps : int
        Total MCMC steps for emcee (default 5000).
    n_burn : int
        Burn-in steps to discard for emcee (default 1000).
    n_live : int
        Number of live points for dynesty (default 500).
    seed : int
        Random seed (default 42).
    progress : bool
        Show progress bar (default ``True``).

    Returns
    -------
    dict
        Result dictionary with keys:

        - ``param_names``, ``param_labels`` — parameter metadata
        - ``samples`` — raw posterior samples, shape ``(n, n_dim)``
        - ``Te``, ``Te_err``, ``Te_posterior``
        - ``ne``, ``ne_err``, ``ne_posterior``
        - ``OH``, ``OH_err``, ``OH_posterior``
        - ``NO``, ``NO_err``, ``NO_posterior`` (if nitrogen detected)
        - ``ionic`` — dict of median log-ionic abundances
        - ``ionic_posteriors`` — dict of log-ionic posterior arrays
        - ``ratio_names`` — observable labels

    Examples
    --------
    >>> import jwspecabund
    >>> fluxes = {
    ...     "HBETA": 1.0, "OIII_5007": 5.0,
    ...     "OIII_4363": 0.05, "NeIII_3869": 0.3,
    ... }
    >>> errors = {k: 0.05 * v for k, v in fluxes.items()}
    >>> res = jwspecabund.forward_model(fluxes, errors, n_steps=2000)
    >>> print(f"12+log(O/H) = {res['OH']:.2f}")
    """
    try:
        import pyneb as pn
    except ImportError as exc:
        raise ImportError(
            "PyNEB is required for the forward model. "
            "Install with: pip install 'jwspecfit[abund]'"
        ) from exc

    # Build observables.
    ratio_names, obs_ratios, ratio_errors, obs_components = _build_observables(
        line_fluxes, line_errors,
    )

    # Determine free parameters.
    param_names = _determine_free_params(obs_components)
    n_dim = len(param_names)

    logger.info(
        "Forward model: %d observables (%s), %d free parameters (%s)",
        len(obs_ratios),
        ", ".join(ratio_names),
        n_dim,
        ", ".join(param_names),
    )

    # Pre-initialise PyNEB Atom objects.
    pyneb_atoms: dict[tuple[str, int], Any] = {}
    for components in obs_components:
        for elem, ion, _ in components:
            key = (elem, ion)
            if key not in pyneb_atoms:
                pyneb_atoms[key] = pn.Atom(elem, ion)

    # Run sampler.
    if sampler == "emcee":
        samples = _run_emcee(
            param_names, obs_ratios, ratio_errors,
            obs_components, pyneb_atoms,
            n_walkers=n_walkers, n_steps=n_steps, n_burn=n_burn,
            seed=seed, progress=progress,
        )
    elif sampler == "dynesty":
        samples = _run_dynesty(
            param_names, obs_ratios, ratio_errors,
            obs_components, pyneb_atoms,
            n_live=n_live, seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown sampler: {sampler!r}. Use 'emcee' or 'dynesty'."
        )

    return _build_forward_result(samples, param_names, ratio_names)
