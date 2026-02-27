"""Save and load MCMC fitting results.

Serialises :class:`MCMCResult` and :class:`MCMCBroadFitResult` to
compressed NumPy ``.npz`` files, preserving posterior chains, per-line
flux posteriors, convergence diagnostics, and BIC metadata.

Example
-------
>>> import jwspecmcmc
>>> result = jwspecmcmc.fit_lines(spec, z=6.0, mode="off")
>>> jwspecmcmc.save_mcmc_result(result, "mcmc.npz")
>>> loaded = jwspecmcmc.load_mcmc_result("mcmc.npz")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from jwspecfit.constraints import ConstraintSet
from jwspecfit.io import Spectrum

from .result import MCMCBroadFitResult, MCMCLineResult, MCMCResult

logger = logging.getLogger(__name__)


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python types.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    Any
        Converted object suitable for ``json.dumps``.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


def save_mcmc_result(
    result: MCMCResult | MCMCBroadFitResult,
    path: str | Path,
) -> None:
    """Save an MCMC result to a compressed ``.npz`` file.

    Parameters
    ----------
    result : MCMCResult or MCMCBroadFitResult
        MCMC fitting result to save.
    path : str or Path
        Output file path (should end in ``.npz``).
    """
    path = Path(path)

    is_broad = isinstance(result, MCMCBroadFitResult)
    mcmc: MCMCResult = result.mcmc_result if is_broad else result

    # --- Serialise MCMCLineResult dicts ---
    lines_data: dict[str, dict[str, Any]] = {}
    flux_posteriors: dict[str, np.ndarray] = {}
    for name, lr in mcmc.lines.items():
        lines_data[name] = {
            "rest_wave_A": lr.rest_wave_A,
            "amplitude": lr.amplitude,
            "amplitude_err": list(lr.amplitude_err),
            "centroid_A": lr.centroid_A,
            "centroid_err": list(lr.centroid_err),
            "sigma_A": lr.sigma_A,
            "sigma_err": list(lr.sigma_err),
            "flux": lr.flux,
            "flux_err": list(lr.flux_err),
            "ew_A": lr.ew_A,
            "snr": lr.snr,
        }
        flux_posteriors[f"flux_post_{name}"] = lr.flux_posterior

    # --- Serialise constraints ---
    if mcmc.constraints is not None:
        cs = mcmc.constraints
        constraints_data = {
            "line_names": cs.line_names,
            "tie_nii": cs.tie_nii,
            "tie_balmer_to_oiii": cs.tie_balmer_to_oiii,
            "tie_uv_doublets": cs.tie_uv_doublets,
        }
        constraints_json = json.dumps(constraints_data)
    else:
        constraints_json = "null"

    # --- Build savez kwargs ---
    save_kwargs: dict[str, Any] = {
        # Spectrum
        "wave_um": mcmc.spectrum.wave_um,
        "flux_ujy": mcmc.spectrum.flux_ujy,
        "err_ujy": mcmc.spectrum.err_ujy,
        "grating": np.array([mcmc.spectrum.grating or ""]),
        "z": np.array([mcmc.spectrum.z or 0.0]),
        # Core arrays
        "flat_chains": mcmc.flat_chains,
        "flat_chains_free": mcmc.flat_chains_free,
        "flat_log_prob": mcmc.flat_log_prob,
        "chains": mcmc.chains if mcmc.chains is not None else np.array([]),
        "has_chains": np.array([mcmc.chains is not None]),
        "params": mcmc.params,
        "model_flux": mcmc.model_flux,
        "continuum": mcmc.continuum,
        # Metadata
        "line_names": np.array(mcmc.line_names),
        "lines_json": np.array([json.dumps(_to_python(lines_data))]),
        "constraints_json": np.array([constraints_json]),
        "convergence_json": np.array([json.dumps(_to_python(mcmc.convergence))]),
        "sampler_name": np.array([mcmc.sampler_name]),
        "sampler_meta_json": np.array([json.dumps(_to_python(mcmc.sampler_meta))]),
        # Broad fit flag
        "is_broad_fit": np.array([is_broad]),
    }

    # Per-line flux posteriors.
    save_kwargs.update(flux_posteriors)

    # BIC metadata for MCMCBroadFitResult.
    if is_broad:
        save_kwargs["selected_model"] = np.array([result.selected_model])
        save_kwargs["bic_narrow"] = np.array([result.bic_narrow])
        save_kwargs["bic_broad1"] = np.array([result.bic_broad1])
        save_kwargs["bic_broad2"] = np.array([result.bic_broad2])
        save_kwargs["bic_both"] = np.array([result.bic_both])

    np.savez_compressed(path, **save_kwargs)
    logger.info("Saved %s to %s", type(result).__name__, path)


def load_mcmc_result(
    path: str | Path,
) -> MCMCResult | MCMCBroadFitResult:
    """Load an MCMC result from a ``.npz`` file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file saved by :func:`save_mcmc_result`.

    Returns
    -------
    MCMCResult or MCMCBroadFitResult
        Reconstructed result with full posterior chains.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    # --- Spectrum ---
    grating_str = str(data["grating"][0])
    z_val = float(data["z"][0])
    spec = Spectrum(
        wave_um=data["wave_um"],
        flux_ujy=data["flux_ujy"],
        err_ujy=data["err_ujy"],
        grating=grating_str if grating_str else None,
        z=z_val if z_val != 0.0 else None,
    )

    # --- Lines ---
    lines_data = json.loads(str(data["lines_json"][0]))
    line_names = list(data["line_names"])
    lines: dict[str, MCMCLineResult] = {}
    for name, ld in lines_data.items():
        flux_post_key = f"flux_post_{name}"
        flux_posterior = data[flux_post_key] if flux_post_key in data else np.array([])
        lines[name] = MCMCLineResult(
            name=name,
            rest_wave_A=ld["rest_wave_A"],
            amplitude=ld["amplitude"],
            amplitude_err=tuple(ld["amplitude_err"]),
            centroid_A=ld["centroid_A"],
            centroid_err=tuple(ld["centroid_err"]),
            sigma_A=ld["sigma_A"],
            sigma_err=tuple(ld["sigma_err"]),
            flux=ld["flux"],
            flux_err=tuple(ld["flux_err"]),
            flux_posterior=flux_posterior,
            ew_A=ld["ew_A"],
            snr=ld["snr"],
        )

    # --- Constraints ---
    constraints_str = str(data["constraints_json"][0])
    if constraints_str == "null":
        constraints = None
    else:
        cd = json.loads(constraints_str)
        constraints = ConstraintSet(
            line_names=cd["line_names"],
            tie_nii=cd["tie_nii"],
            tie_balmer_to_oiii=cd["tie_balmer_to_oiii"],
            tie_uv_doublets=cd.get("tie_uv_doublets", False),
        )

    # --- Chains ---
    has_chains = bool(data["has_chains"][0])
    chains = data["chains"] if has_chains else None

    # --- Convergence and sampler metadata ---
    convergence = json.loads(str(data["convergence_json"][0]))
    sampler_meta = json.loads(str(data["sampler_meta_json"][0]))

    mcmc = MCMCResult(
        lines=lines,
        flat_chains=data["flat_chains"],
        flat_chains_free=data["flat_chains_free"],
        flat_log_prob=data["flat_log_prob"],
        chains=chains,
        params=data["params"],
        model_flux=data["model_flux"],
        continuum=data["continuum"],
        spectrum=spec,
        line_names=line_names,
        constraints=constraints,
        convergence=convergence,
        sampler_name=str(data["sampler_name"][0]),
        sampler_meta=sampler_meta,
    )

    # --- MCMCBroadFitResult wrapper ---
    is_broad = bool(data["is_broad_fit"][0])
    if is_broad:
        return MCMCBroadFitResult(
            mcmc_result=mcmc,
            selected_model=str(data["selected_model"][0]),
            bic_narrow=float(data["bic_narrow"][0]),
            bic_broad1=float(data["bic_broad1"][0]),
            bic_broad2=float(data["bic_broad2"][0]),
            bic_both=float(data["bic_both"][0]),
        )

    logger.info("Loaded %s from %s", type(mcmc).__name__, path)
    return mcmc
