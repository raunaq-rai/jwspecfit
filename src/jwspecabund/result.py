"""Abundance result containers.

Dataclasses holding the output of direct T_e, forward model, and
strong-line abundance calculations, including optional MCMC posterior arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AbundanceResult:
    """Container for a chemical abundance measurement.

    Parameters
    ----------
    method : str
        ``"direct"``, ``"forward"``, or ``"strong_line"``.
    OH : float
        12 + log(O/H).
    OH_err : float or tuple of float
        Symmetric error or ``(lo, hi)`` 68 % CI half-widths.
    NO : float or None
        log(N/O), if nitrogen lines available.
    NO_err : float or tuple of float or None
        Error on log(N/O).
    CO : float or None
        log(C/O), if UV lines present.
    CO_err : float or tuple of float or None
        Error on log(C/O).
    Te_high : float or None
        T_e(O++) in K (direct method only).
    Te_low : float or None
        T_e(O+/N+) in K (direct method only).
    ne : float or None
        Electron density in cm^-3 (direct method only).
    Av : float or None
        Dust attenuation A_V.
    ionic : dict or None
        Ionic abundance dict, e.g. ``{"O+/H+": val, "O++/H+": val, ...}``.
    OH_posterior : np.ndarray or None
        Full posterior samples of 12+log(O/H) (MCMC input).
    NO_posterior : np.ndarray or None
        Full posterior samples of log(N/O).
    CO_posterior : np.ndarray or None
        Full posterior samples of log(C/O).
    ratios_used : list of str or None
        Diagnostic ratios used (strong-line method).
    chi2 : float or None
        Goodness-of-fit chi-squared (strong-line method).
    SO : float or None
        log(S/O) if [SII] and [SIII] available.
    NeO : float or None
        log(Ne/O) if [NeIII] available.
    ArO : float or None
        log(Ar/O) if [ArIII] available.
    excluded_lines : list of str or None
        Line names excluded by the per-line SNR filter.
    """

    method: str
    OH: float
    OH_err: float | tuple[float, float]
    NO: float | None = None
    NO_err: float | tuple[float, float] | None = None
    CO: float | None = None
    CO_err: float | tuple[float, float] | None = None
    Te_high: float | None = None
    Te_low: float | None = None
    ne: float | None = None
    Av: float | None = None
    ionic: dict[str, float] | None = None
    OH_posterior: np.ndarray | None = None
    NO_posterior: np.ndarray | None = None
    CO_posterior: np.ndarray | None = None
    ratios_used: list[str] | None = None
    chi2: float | None = None
    SO: float | None = None
    NeO: float | None = None
    ArO: float | None = None
    logU: float | None = None
    ne_low: float | None = None
    ne_high: float | None = None
    icf_method: str | None = None
    NO_icf_name: str | None = None
    excluded_lines: list[str] | None = None
    # Internal: full forward model result dict (samples, param_names, etc.)
    _forward_result: dict[str, Any] | None = field(default=None, repr=False)

    def summary(self) -> str:
        """Return a human-readable summary string.

        Returns
        -------
        str
            Multi-line summary of the abundance measurement.
        """
        lines = [f"AbundanceResult (method={self.method})"]
        lines.append(f"  12+log(O/H) = {self.OH:.3f} +/- {self.OH_err}")
        if self.NO is not None:
            lines.append(f"  log(N/O)    = {self.NO:.3f} +/- {self.NO_err}")
        if self.CO is not None:
            lines.append(f"  log(C/O)    = {self.CO:.3f} +/- {self.CO_err}")
        if self.SO is not None:
            lines.append(f"  log(S/O)    = {self.SO:.3f}")
        if self.NeO is not None:
            lines.append(f"  log(Ne/O)   = {self.NeO:.3f}")
        if self.ArO is not None:
            lines.append(f"  log(Ar/O)   = {self.ArO:.3f}")
        if self.Te_high is not None:
            lines.append(f"  T_e(high)   = {self.Te_high:.0f} K")
        if self.Te_low is not None:
            lines.append(f"  T_e(low)    = {self.Te_low:.0f} K")
        if self.ne is not None:
            lines.append(f"  n_e         = {self.ne:.0f} cm^-3")
        if self.ne_low is not None and self.ne_high is not None:
            lines.append(f"  n_e(low)    = {self.ne_low:.0f} cm^-3")
            lines.append(f"  n_e(high)   = {self.ne_high:.0f} cm^-3")
        if self.logU is not None:
            lines.append(f"  log(U)      = {self.logU:.2f}")
        if self.Av is not None:
            lines.append(f"  A_V         = {self.Av:.3f}")
        if self.icf_method is not None:
            lines.append(f"  ICF method  = {self.icf_method}")
        if self.NO_icf_name is not None:
            lines.append(f"  N/O ICF     = {self.NO_icf_name}")
        if self.ratios_used:
            lines.append(f"  Ratios used = {self.ratios_used}")
        if self.chi2 is not None:
            lines.append(f"  chi2        = {self.chi2:.2f}")
        if self.excluded_lines:
            lines.append(f"  Excluded    = {self.excluded_lines}")
        return "\n".join(lines)
