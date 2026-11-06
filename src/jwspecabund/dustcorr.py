"""Stand-alone dust correction of fitted line fluxes.

Derives A_V from the Balmer decrement of a fit result and returns
dust-corrected fluxes, without running :func:`compute_abundances`.

Assumptions (state these when quoting the numbers): Case B intrinsic
Balmer ratios at T_e = 10^4 K, n_e = 100 cm^-3 (Osterbrock & Ferland
2006), and by default the Salim, Boquien & Lee (2018) attenuation curve
for local analogues of high-redshift galaxies (delta = -0.45, B = 2.27,
R_V = 2.64 from their Eq. 4), the same as
:func:`jwspecabund.compute_abundances`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._core import _LINE_WAVES, _extract_fluxes
from .dust import (
    _kappa,
    compute_Av_balmer_pair,
    compute_Av_joint_balmer,
    compute_Av_multi_balmer,
)

__all__ = ["DustCorrectedFluxes", "dust_correct"]


@dataclass
class DustCorrectedFluxes:
    """Dust-corrected line fluxes.

    Attributes
    ----------
    fluxes, errors : dict
        Dust-corrected flux and 1-sigma error per line.  Errors include
        the A_V uncertainty unless ``propagate_av_err=False`` was passed;
        that term is fully correlated between lines.
    observed_fluxes, observed_errors : dict
        Input (observed) fluxes; Balmer broad components are summed in.
    factors : dict
        Multiplicative correction ``10**(0.4 A_lambda)`` per line.
    Av, Av_err : float
        V-band attenuation applied and its uncertainty.
    law : str
        Attenuation law used.
    method : str
        How A_V was obtained (``"anchored"``, ``"joint"``, ``"pair"`` or
        ``"user"``).
    n_balmer : int
        Number of Balmer lines / decrements that constrained A_V.
    skipped : list of str
        Lines left uncorrected (no rest wavelength known).
    balmer_info : dict
        Full output of the A_V estimator.
    """

    fluxes: dict[str, float]
    errors: dict[str, float]
    observed_fluxes: dict[str, float]
    observed_errors: dict[str, float]
    factors: dict[str, float]
    Av: float
    Av_err: float
    law: str
    method: str
    n_balmer: int
    skipped: list[str] = field(default_factory=list)
    balmer_info: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"DustCorrectedFluxes(Av={self.Av:.3f} +/- {self.Av_err:.3f}, "
            f"law={self.law!r}, method={self.method!r}, "
            f"n_lines={len(self.fluxes)})"
        )


def _usable(fluxes: dict, errors: dict, name: str, snr_min: float) -> bool:
    """True if *name* is measured with positive flux and SNR >= *snr_min*."""
    f, e = fluxes.get(name, 0.0), errors.get(name, 0.0)
    return f > 0 and e > 0 and f / e >= snr_min


def dust_correct(
    result: Any,
    dust_law: str = "salim",
    Av: float | None = None,
    Av_err: float | None = None,
    balmer_method: str = "anchored",
    balmer_anchor: str = "auto",
    balmer_pair: tuple[str, str] | None = None,
    snr_balmer: float = 5.0,
    Rv: float | None = None,
    delta: float = -0.45,
    B_bump: float = 2.27,
    propagate_av_err: bool = True,
) -> DustCorrectedFluxes:
    """Dust-correct the line fluxes of a fit result via the Balmer decrement.

    Uses the point-estimate fluxes stored on the result (``lr.flux``:
    the best fit for :func:`jwspecfit.fit_lines`, the posterior point
    estimate for :func:`jwspecmcmc.fit_lines`).  Broad Balmer components
    are added to the narrow flux, as in :func:`compute_abundances`.

    Assumes Case B recombination (T_e = 10^4 K, n_e = 100 cm^-3) for
    the intrinsic Balmer ratios and, by default, the Salim+18 high-z
    analogue attenuation curve.

    Parameters
    ----------
    result : FitResult | BroadFitResult | MCMCResult | MCMCBroadFitResult
        Output of ``jwspecfit.fit_lines`` / ``jwspecmcmc.fit_lines`` (or
        their ``fit_with_broad`` variants).
    dust_law : str
        ``"salim"`` (default, Salim+18) or ``"cardelli"`` (Cardelli+89).
    Av, Av_err : float, optional
        Supply A_V directly instead of deriving it from the Balmer lines.
    balmer_method : str
        ``"anchored"`` (default; GLS mean of decrements against
        *balmer_anchor*) or ``"joint"`` (fit the whole Balmer ladder).
    balmer_anchor : str
        Denominator of the decrements (anchored method only).  ``"auto"``
        (default) uses Hα when it is measured at SNR >= *snr_balmer*
        (decrements Hβ/Hα, Hγ/Hα, ...), otherwise Hβ.  ``"HALPHA"`` or
        ``"HBETA"`` force the choice.
    balmer_pair : tuple of str, optional
        Force a single decrement, e.g. ``("HALPHA", "HBETA")``.
    snr_balmer : float
        Minimum SNR for a Balmer line to enter the A_V estimate
        (default 5).
    Rv : float or None
        Salim R_V; ``None`` (default) derives it from *delta* via
        Salim+18 Eq. 4.  Ignored for Cardelli (R_V = 3.1).
    delta, B_bump : float
        Salim+18 slope deviation and UV bump amplitude (defaults -0.45
        and 2.27, the high-z analogue curve).  Ignored for Cardelli.
    propagate_av_err : bool
        Add the A_V uncertainty to the corrected flux errors in
        quadrature (default ``True``).

    Returns
    -------
    DustCorrectedFluxes
        Corrected fluxes, errors, correction factors and the A_V used.

    Examples
    --------
    >>> res = jwspecfit.fit_lines(spec, z=6.0)
    >>> dc = jwspecfit.dust_correct(res, dust_law="salim")
    >>> dc.Av, dc.fluxes["OIII_5007"]
    """
    if dust_law not in ("salim", "cardelli"):
        raise ValueError(f"dust_law must be 'salim' or 'cardelli', got {dust_law!r}")
    if balmer_method not in ("anchored", "joint"):
        raise ValueError(
            f"balmer_method must be 'anchored' or 'joint', got {balmer_method!r}"
        )
    dust_kwargs = (
        {"Rv": Rv, "delta": delta, "B": B_bump} if dust_law == "salim" else {}
    )

    fluxes, errors, _ = _extract_fluxes(result)

    if Av is not None:
        method, info = "user", {}
        av = float(Av)
        av_err = float(Av_err) if Av_err is not None else 0.0
        n_balmer = 0
    else:
        if balmer_pair is not None:
            method = "pair"
            info = compute_Av_balmer_pair(
                fluxes, errors, balmer_pair, law=dust_law, **dust_kwargs
            )
        elif balmer_method == "joint":
            method = "joint"
            info = compute_Av_joint_balmer(
                fluxes, errors, law=dust_law, snr_min=snr_balmer, **dust_kwargs
            )
        else:
            method = "anchored"
            if balmer_anchor == "auto":
                balmer_anchor = "HALPHA" if _usable(fluxes, errors, "HALPHA",
                                                    snr_balmer) else "HBETA"
            info = compute_Av_multi_balmer(
                fluxes, errors, law=dust_law, snr_min=snr_balmer,
                anchor=balmer_anchor, **dust_kwargs,
            )
        n_balmer = int(info.get("n_lines", 0))
        if n_balmer == 0:
            raise ValueError(
                "No usable Balmer decrement (need two Balmer lines above "
                f"SNR {snr_balmer}); pass Av= explicitly."
            )
        av, av_err = float(info["Av"]), float(info["Av_err"])

    sigma_av = av_err if (propagate_av_err and np.isfinite(av_err)) else 0.0

    corr_f, corr_e, factors, skipped = {}, {}, {}, []
    for name, f in fluxes.items():
        wave = _LINE_WAVES.get(name)
        if wave is None:
            skipped.append(name)
            continue
        kappa = float(_kappa(np.array([wave]), dust_law, **dust_kwargs)[0])
        fac = 10.0 ** (0.4 * av * kappa)
        # d(fac)/d(Av) = 0.4 ln10 kappa fac
        dfac = 0.4 * np.log(10.0) * kappa * fac * sigma_av
        factors[name] = fac
        corr_f[name] = f * fac
        corr_e[name] = float(np.hypot(errors[name] * fac, f * dfac))

    return DustCorrectedFluxes(
        fluxes=corr_f,
        errors=corr_e,
        observed_fluxes=fluxes,
        observed_errors=errors,
        factors=factors,
        Av=av,
        Av_err=av_err,
        law=dust_law,
        method=method,
        n_balmer=n_balmer,
        skipped=skipped,
        balmer_info=info,
    )
