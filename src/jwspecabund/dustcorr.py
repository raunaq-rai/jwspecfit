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
    _BALMER_LADDER,
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
        Input (observed) narrow-component fluxes.
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
    balmer_rejected : dict
        ``{line: reason}`` for every Balmer line that did not enter the
        A_V estimate.
    snr_balmer : float
        SNR threshold applied to the Balmer lines.
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
    balmer_rejected: dict[str, str] = field(default_factory=dict)
    snr_balmer: float = np.nan

    def summary(self) -> str:
        """Human-readable report of the Balmer lines and ratios used.

        Returns
        -------
        str
            A_V, the method, each decrement (observed vs Case B ratio and
            its A_V) and the Balmer lines that were rejected, with why.
        """
        law = "Salim+18" if self.law == "salim" else "Cardelli+89"
        info = self.balmer_info
        if self.method == "user":
            return (f"A_V = {self.Av:.3f} +/- {self.Av_err:.3f} mag "
                    f"({law}; user-supplied, no Balmer lines used)")
        how = {"anchored": f"anchored on {info.get('anchor')}",
               "pair": "single pair",
               "joint": "joint ladder fit"}[self.method]
        out = [f"A_V = {self.Av:.3f} +/- {self.Av_err:.3f} mag "
               f"({law}, {how}, Case B 10^4 K)"]
        cut = "" if self.method == "pair" else f" (SNR >= {self.snr_balmer:g})"
        out.append(f"Balmer lines used{cut}:")
        f, e = self.observed_fluxes, self.observed_errors
        if self.method == "joint":
            for r in info.get("individual", []):
                out.append(f"  {r['line']:<7} SNR {r['flux'] / r['flux_err']:6.1f}"
                           f"   obs/model {r['flux'] / r['model']:.3f}")
            out.append(f"  chi2/dof = {info.get('chi2', np.nan):.2f}"
                       f"/{info.get('dof', 0)}")
        else:
            den = info.get("anchor")
            for r in info.get("individual", []):
                num = r["line"]
                rel = np.hypot(e.get(num, 0.0) / f[num],
                               e.get(den, 0.0) / f[den])
                obs = r["observed_ratio"]
                out.append(
                    f"  {num + '/' + den:<14} obs {obs:.4f} +/- {obs * rel:.4f}"
                    f"   Case B {r['intrinsic_ratio']:.4f}"
                    f"   A_V {r['Av']:+.3f} +/- {r['Av_err']:.3f}"
                )
        if self.balmer_rejected:
            out.append("Not used:")
            for name, why in self.balmer_rejected.items():
                out.append(f"  {name:<7} {why}")
        return "\n".join(out)

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


def _balmer_status(fluxes: dict, errors: dict, snr_min: float) -> dict[str, str]:
    """Why each Balmer line would fail the SNR cut (``""`` if it passes)."""
    status = {}
    for name in _BALMER_LADDER:
        f, e = fluxes.get(name), errors.get(name, 0.0)
        if f is None:
            status[name] = "not fitted"
        elif not f > 0:
            status[name] = f"flux {f:.3g} <= 0"
        elif not e > 0:
            status[name] = "no flux error"
        elif f / e < snr_min:
            status[name] = f"SNR {f / e:.1f} < {snr_min:g}"
        else:
            status[name] = ""
    return status


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
    verbose: bool = True,
) -> DustCorrectedFluxes:
    """Dust-correct the line fluxes of a fit result via the Balmer decrement.

    Uses the point-estimate fluxes stored on the result (``lr.flux``:
    the best fit for :func:`jwspecfit.fit_lines`, the posterior point
    estimate for :func:`jwspecmcmc.fit_lines`).  Only narrow components
    are used, as in :func:`compute_abundances`.

    Raises ``ValueError`` if fewer than two Balmer lines pass
    *snr_balmer* (the message lists each line and why it failed).

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
    verbose : bool
        Print :meth:`DustCorrectedFluxes.summary` (default ``True``): the
        Balmer lines and ratios used, and why any others were rejected.

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
    status = _balmer_status(fluxes, errors, snr_balmer)

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
        anchor = info.get("anchor")
        # The anchored estimator does not apply the SNR cut to its anchor.
        if method == "anchored" and status.get(anchor):
            n_balmer = 0
        if n_balmer == 0:
            detail = "; ".join(f"{n}: {why or 'ok'}" for n, why in status.items())
            raise ValueError(
                "No usable Balmer decrement (need two Balmer lines above "
                f"SNR {snr_balmer:g}"
                + (f", including the anchor {anchor}" if method == "anchored" else "")
                + f"). Balmer lines: {detail}. Pass Av= explicitly."
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

    if method == "user":
        used = set()
    else:
        used = {r["line"] for r in info.get("individual", [])}
        if method != "joint":
            used.add(info.get("anchor"))
    rejected = {}
    for name, why in status.items():
        if name in used:
            continue
        if not why and method == "anchored":
            why = f"redder than the anchor {info.get('anchor')}"
        elif not why:
            why = "not in the requested pair" if method == "pair" else "user A_V"
        rejected[name] = why
    if method == "user":
        rejected = {}

    dc = DustCorrectedFluxes(
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
        balmer_rejected=rejected,
        snr_balmer=snr_balmer,
    )
    if verbose:
        print(dc.summary())
    return dc
