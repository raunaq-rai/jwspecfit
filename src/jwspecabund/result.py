"""Abundance result containers.

Dataclasses holding the output of direct T_e, forward model, and
strong-line abundance calculations, including optional MCMC posterior arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Short labels for the T_e(O++) diagnostics, for the summary table.
_TE_DIAG_SHORT = {
    "self_consistent": "self-consistent (UV+opt)",
    "4363": "[OIII] 4363 only",
    "1666": "O III] 1666 only",
}


def _fmt_err_or_blank(err: float | tuple[float, float] | None, prec: int = 3) -> str:
    """Like :func:`_fmt_err`, but empty for a NaN error.

    The point-estimate cross-checks carry no error bar by design, and
    printing "+/- nan" reads as a bug rather than as "not sampled".
    """
    if err is None:
        return ""
    if not isinstance(err, (tuple, list)) and not np.isfinite(err):
        return ""
    return _fmt_err(err, prec)


def _fmt_err(err: float | tuple[float, float], prec: int = 3) -> str:
    """Format an uncertainty for display.

    Error tuples are stored as ``(lo, hi)`` 68 % CI half-widths
    (median − 16th pct, 84th pct − median), so the *upper* error is
    printed first: ``(+hi/-lo)``.  Scalars print as ``+/- err``.
    """
    if err is None:
        return ""
    if isinstance(err, (tuple, list)):
        lo, hi = err
        return f"(+{hi:.{prec}f}/-{lo:.{prec}f})"
    return f"+/- {err:.{prec}f}"


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
    ne_Opp_err : float or tuple of float or None
        Uncertainty on the O²⁺-zone density, when it was measured rather
        than borrowed.
    ne_Opp_is_upper_limit : bool
        ``True`` when the self-consistent solve could only bound the
        O²⁺-zone density from above (the density-insensitive regime), in
        which case the single-ratio temperature was adopted instead.
    Te_diagnostic : str or None
        Which T_e(O++) diagnostic was adopted: ``"self_consistent"``,
        ``"4363"`` or ``"1666"``.
    oiii_uv_doublet : dict or None
        Result of the O III] 1661/1666 branching-ratio check; see
        :func:`~jwspecabund.direct.check_oiii_uv_doublet`.
    Te_ne_selfconsistent : SelfConsistentOIII or None
        Full joint O++ T_e-n_e solution, including the T_e(n_e) curves, when
        one was computed.  See
        :func:`~jwspecabund.direct.compute_Te_ne_OIII`.
    excluded_lines : list of str or None
        Line names excluded by the per-line SNR filter.
    failures : dict or None
        Reasons why specific abundance ratios could not be computed,
        e.g. ``{"N/O": "no nitrogen ions detected"}``.
    """

    method: str
    OH: float
    OH_err: float | tuple[float, float]
    NO: float | None = None
    NO_err: float | tuple[float, float] | None = None
    CO: float | None = None
    CO_err: float | tuple[float, float] | None = None
    Te_high: float | None = None
    Te_high_err: float | tuple[float, float] | None = None
    Te_mid: float | None = None
    Te_mid_err: float | tuple[float, float] | None = None
    Te_low: float | None = None
    Te_low_err: float | tuple[float, float] | None = None
    ne: float | None = None
    Av: float | None = None
    Av_err: float | None = None
    Av_posterior: np.ndarray | None = field(default=None, repr=False)
    ionic: dict[str, float] | None = None
    OH_posterior: np.ndarray | None = None
    NO_posterior: np.ndarray | None = None
    CO_posterior: np.ndarray | None = None
    ratios_used: list[str] | None = None
    chi2: float | None = None
    SO: float | None = None
    SO_err: float | tuple[float, float] | None = None
    NeO: float | None = None
    NeO_err: float | tuple[float, float] | None = None
    ArO: float | None = None
    ArO_err: float | tuple[float, float] | None = None
    logU: float | None = None
    logU_err: float | tuple[float, float] | None = None
    ne_low: float | None = None
    ne_mid: float | None = None
    ne_high: float | None = None
    ne_Opp: float | None = None
    ne_Opp_err: float | tuple[float, float] | None = None
    ne_Opp_is_upper_limit: bool = False
    Te_diagnostic: str | None = None
    Te_ne_selfconsistent: Any | None = field(default=None, repr=False)
    oiii_uv_doublet: dict[str, Any] | None = field(default=None, repr=False)
    icf_method: str | None = None
    NO_icf_name: str | None = None
    lya_f_esc: float | None = None
    lya_f_esc_err: float | tuple[float, float] | None = None
    lya_f_esc_posterior: np.ndarray | None = field(default=None, repr=False)
    lya_f_esc_details: dict | None = field(default=None, repr=False)
    excluded_lines: list[str] | None = None
    ionic_upper_limits: dict[str, float] | None = field(default=None, repr=False)
    ionic_ul_details: dict[str, dict] | None = field(default=None, repr=False)
    NO_tiers: dict[str, float] | None = field(default=None, repr=False)
    icf_values: dict[str, dict] | None = field(default=None, repr=False)
    failures: dict[str, str] | None = field(default=None, repr=False)
    diagnostics: dict[str, str] | None = field(default=None, repr=False)
    alt_results: dict[str, AbundanceResult] | None = field(default=None, repr=False)
    # Internal: full forward model result dict (samples, param_names, etc.)
    _forward_result: dict[str, Any] | None = field(default=None, repr=False)

    def summary(self) -> str:
        """Return a human-readable summary string.

        Returns
        -------
        str
            Multi-line summary of the abundance measurement, including
            a diagnostics section explaining how each physical quantity
            was derived.
        """
        lines = [f"AbundanceResult (method={self.method})"]

        # --- Abundances ---
        _f = self.failures or {}
        lines.append(f"  12+log(O/H) = {self.OH:.3f} {_fmt_err(self.OH_err)}")
        if self.NO is not None:
            lines.append(f"  log(N/O)    = {self.NO:.3f} {_fmt_err(self.NO_err)}")
        elif "N/O" in _f:
            lines.append(f"  log(N/O)    = FAILED — {_f['N/O']}")
        if self.CO is not None:
            lines.append(f"  log(C/O)    = {self.CO:.3f} {_fmt_err(self.CO_err)}")
        elif "C/O" in _f:
            lines.append(f"  log(C/O)    = FAILED — {_f['C/O']}")
        if self.SO is not None:
            lines.append(f"  log(S/O)    = {self.SO:.3f} {_fmt_err(self.SO_err)}")
        elif "S/O" in _f:
            lines.append(f"  log(S/O)    = FAILED — {_f['S/O']}")
        if self.NeO is not None:
            lines.append(f"  log(Ne/O)   = {self.NeO:.3f} {_fmt_err(self.NeO_err)}")
        elif "Ne/O" in _f:
            lines.append(f"  log(Ne/O)   = FAILED — {_f['Ne/O']}")
        if self.ArO is not None:
            lines.append(f"  log(Ar/O)   = {self.ArO:.3f} {_fmt_err(self.ArO_err)}")
        elif "Ar/O" in _f:
            lines.append(f"  log(Ar/O)   = FAILED — {_f['Ar/O']}")

        # --- Physical conditions ---
        if self.Te_high is not None:
            if self.Te_high_err is not None:
                lines.append(f"  T_e(high)   = {self.Te_high:.0f} {_fmt_err(self.Te_high_err, 0)} K")
            else:
                lines.append(f"  T_e(high)   = {self.Te_high:.0f} K")
        if self.Te_mid is not None:
            if self.Te_mid_err is not None:
                lines.append(f"  T_e(int)    = {self.Te_mid:.0f} {_fmt_err(self.Te_mid_err, 0)} K")
            else:
                lines.append(f"  T_e(int)    = {self.Te_mid:.0f} K")
        if self.Te_low is not None:
            if self.Te_low_err is not None:
                lines.append(f"  T_e(low)    = {self.Te_low:.0f} {_fmt_err(self.Te_low_err, 0)} K")
            else:
                lines.append(f"  T_e(low)    = {self.Te_low:.0f} K")
        if self.ne is not None:
            lines.append(f"  n_e         = {self.ne:.0f} cm^-3")
        if self.ne_low is not None and self.ne_high is not None:
            lines.append(f"  n_e(low)    = {self.ne_low:.0f} cm^-3")
            if self.ne_mid is not None:
                lines.append(f"  n_e(mid)    = {self.ne_mid:.0f} cm^-3")
            if self.ne_Opp is not None:
                _sc = self.Te_ne_selfconsistent
                if self.ne_Opp_is_upper_limit and _sc is not None and _sc.ne_upper_limit:
                    _extra = (
                        "" if abs(self.ne_Opp - _sc.ne_upper_limit) < 1.0
                        else f"; adopted {self.ne_Opp:.0f}"
                    )
                    lines.append(
                        f"  n_e(O++)    < {_sc.ne_upper_limit:.0f} cm^-3 "
                        f"(1σ upper limit from O III{_extra})"
                    )
                elif self.ne_Opp_err is not None:
                    lines.append(
                        f"  n_e(O++)    = {self.ne_Opp:.0f} "
                        f"{_fmt_err(self.ne_Opp_err, 0)} cm^-3"
                    )
                else:
                    lines.append(f"  n_e(O++)    = {self.ne_Opp:.0f} cm^-3")
            lines.append(f"  n_e(high)   = {self.ne_high:.0f} cm^-3")
        if self.logU is not None:
            if self.logU_err is not None:
                lines.append(f"  log(U)      = {self.logU:.2f} {_fmt_err(self.logU_err, 2)}")
            else:
                lines.append(f"  log(U)      = {self.logU:.2f}")
        if self.Av is not None:
            if self.Av_err is not None:
                lines.append(f"  A_V         = {self.Av:.3f} +/- {self.Av_err:.3f}")
            else:
                lines.append(f"  A_V         = {self.Av:.3f}")

        # --- T_e(O++) diagnostics, side by side ---
        _alt_direct = {
            k[len("direct_"):]: v
            for k, v in (self.alt_results or {}).items()
            if k.startswith("direct_") and v.Te_high is not None
        }
        if self.Te_diagnostic is not None and _alt_direct:
            lines.append("")
            lines.append("T_e(O++) diagnostics:")
            rows = [(self.Te_diagnostic, self.Te_high, self.ne_Opp, self.OH, True)]
            rows += [
                (k, v.Te_high, v.ne_Opp, v.OH, False)
                for k, v in sorted(_alt_direct.items())
            ]
            for name, te, ne_opp, oh, adopted in rows:
                if te is None:
                    continue
                label = _TE_DIAG_SHORT.get(name, name)
                ne_txt = "" if ne_opp is None else f"  n_e(O++) = {ne_opp:.3g} cm^-3"
                oh_txt = "" if oh is None or not np.isfinite(oh) else (
                    f"  12+log(O/H) = {oh:.3f}"
                )
                mark = "*" if adopted else " "
                lines.append(
                    f"  {mark} {label:22s} T_e = {te:.0f} K{ne_txt}{oh_txt}"
                    + ("  [adopted]" if adopted else "")
                )
            _sc = self.Te_ne_selfconsistent
            if _sc is not None and not _sc.converged:
                lines.append(
                    "    (joint curves do not intersect; closest approach used)"
                )

        # --- Lyα escape fraction ---
        if self.lya_f_esc is not None and np.isfinite(self.lya_f_esc):
            if self.lya_f_esc_err is not None:
                if isinstance(self.lya_f_esc_err, tuple):
                    lines.append(
                        f"  f_esc(Lyα)  = {self.lya_f_esc:.3f}"
                        f" (+{self.lya_f_esc_err[1]:.3f}/-{self.lya_f_esc_err[0]:.3f})"
                    )
                else:
                    lines.append(
                        f"  f_esc(Lyα)  = {self.lya_f_esc:.3f}"
                        f" +/- {self.lya_f_esc_err:.3f}"
                    )
            else:
                lines.append(f"  f_esc(Lyα)  = {self.lya_f_esc:.3f}")
            if self.lya_f_esc_details:
                for r in self.lya_f_esc_details.get("individual", []):
                    name = r["line"]
                    fe = r["f_esc"]
                    if isinstance(r.get("f_esc_err"), tuple):
                        lines.append(
                            f"    via {name:8s}: f_esc = {fe:.3f}"
                            f" (+{r['f_esc_err'][1]:.3f}/-{r['f_esc_err'][0]:.3f})"
                        )
                    else:
                        fe_e = r.get("f_esc_err", np.nan)
                        lines.append(
                            f"    via {name:8s}: f_esc = {fe:.3f} +/- {fe_e:.3f}"
                        )

        # --- Density solve failures ---
        _ne_keys = [k for k in _f if k.startswith("n_e(")]
        for k in _ne_keys:
            lines.append(f"  {k:14s}= FAILED — {_f[k]}")

        # --- ICF info ---
        if self.icf_method is not None:
            lines.append(f"  ICF method  = {self.icf_method}")
        if self.NO_icf_name is not None:
            lines.append(f"  N/O ICF     = {self.NO_icf_name}")

        # --- ICF values ---
        if self.icf_values:
            lines.append("")
            lines.append("Ionisation correction factors:")
            for ratio_name, info in self.icf_values.items():
                icf_val = info.get("icf", 1.0)
                raw = info.get("raw")
                corrected = info.get("corrected")
                method = info.get("method", "")
                if raw is not None and corrected is not None:
                    delta = corrected - raw
                    lines.append(
                        f"  {ratio_name}: ICF = {icf_val:.4f} ({method})"
                        f"  raw = {raw:.3f}  →  corrected = {corrected:.3f}"
                        f"  (Δ = {delta:+.3f} dex)"
                    )
                else:
                    lines.append(f"  {ratio_name}: ICF = {icf_val:.4f} ({method})")

        # --- Ionic abundances ---
        if self.ionic:
            lines.append("")
            lines.append("Ionic abundances:")
            _ul = self.ionic_upper_limits or {}
            for key, val in self.ionic.items():
                if val > 0:
                    lines.append(f"  {key:14s}= {val:.4e}")
                elif key in _ul:
                    lines.append(f"  {key:14s}< {_ul[key]:.4e}  (3σ upper limit)")
                else:
                    lines.append(f"  {key:14s}  not detected")

        # --- N/O tiers (all eligible methods) ---
        if self.NO_tiers:
            # Filter out internal keys (prefixed with _).
            display_tiers = {k: v for k, v in self.NO_tiers.items()
                            if not k.startswith("_")}
            if display_tiers:
                lines.append("")
                lines.append("N/O by method (all eligible):")
                for tier_name, log_no in display_tiers.items():
                    selected = " ← selected" if (
                        self.NO is not None
                        and abs(log_no - self.NO) < 0.001
                    ) else ""
                    # ICF value lookup.
                    _ICF_KEY_MAP = {
                        "ICF 1:": "_icf1_value",
                        "ICF 2:": "_icf2_value",
                        "ICF 3:": "_icf3_value",
                        "ICF 4:": "_icf4_value",
                        "ICF 5:": "_icf5_value",
                        "Izotov": "_izotov06_icf_value",
                    }
                    icf_str = ""
                    for prefix, key in _ICF_KEY_MAP.items():
                        if prefix in tier_name:
                            icf_val = self.NO_tiers.get(key)
                            if icf_val is not None:
                                icf_str = f"  [ICF={icf_val:.3f}]"
                            break
                    # Uncertainty from MC/posterior propagation.
                    err_key = f"_err_{tier_name}"
                    err = self.NO_tiers.get(err_key)
                    if err is not None:
                        if isinstance(err, tuple):
                            err_str = f" (+{err[1]:.3f}/-{err[0]:.3f})"
                        else:
                            err_str = f" +/- {err:.3f}"
                    else:
                        err_str = ""
                    lines.append(
                        f"  {tier_name}: {log_no:.3f}{err_str}{icf_str}{selected}"
                    )

        # --- Strong-line info ---
        if self.ratios_used:
            lines.append(f"  Ratios used = {self.ratios_used}")
        if self.chi2 is not None:
            lines.append(f"  chi2        = {self.chi2:.2f}")

        if self.excluded_lines:
            lines.append(f"  Excluded    = {self.excluded_lines}")

        # --- Diagnostics section ---
        if self.diagnostics:
            lines.append("")
            lines.append("Derivation details:")
            for key, explanation in self.diagnostics.items():
                lines.append(f"  {key}: {explanation}")

        # --- Upper limit details ---
        if self.ionic_upper_limits and self.ionic_ul_details:
            lines.append("")
            lines.append("Upper limits (3σ):")
            _det = self.ionic_ul_details
            _ul = self.ionic_upper_limits
            for ion_key, abund_ul in _ul.items():
                info = _det.get(ion_key, {})
                src_lines = ", ".join(info.get("lines", []))
                flux_ul = info.get("flux_ul")
                n_sig = info.get("n_sigma", 3.0)
                lines.append(f"  {ion_key}:")
                lines.append(f"    Lines not detected: {src_lines}")
                ul_method = info.get("method", "fit_error")
                if flux_ul is not None:
                    if ul_method == "continuum_rms":
                        method_desc = f"{n_sig:.0f}σ × RMS_cont × σ_inst × √(2π)"
                    else:
                        method_desc = f"{n_sig:.0f}σ × sqrt(Σ err²)"
                    lines.append(
                        f"    Flux upper limit:   {flux_ul:.2e} ({method_desc})"
                    )
                lines.append(f"    Ionic abundance:    < {abund_ul:.4e}")

            # Upper limits on abundance ratios.
            O_total = 0.0
            if self.ionic:
                O_total = (
                    self.ionic.get("O+/H+", 0.0)
                    + self.ionic.get("O++/H+", 0.0)
                )
            if O_total > 0:
                # N/O upper limit: sum detected N ions + upper limits for
                # non-detected N ions, divided by total O.
                _n_ions = ["N+/H+", "N++/H+", "N+++/H+"]
                N_total = 0.0
                has_ul = False
                for nk in _n_ions:
                    detected = self.ionic.get(nk, 0.0) if self.ionic else 0.0
                    if detected > 0:
                        N_total += detected
                    elif nk in _ul:
                        N_total += _ul[nk]
                        has_ul = True
                if N_total > 0 and has_ul:
                    log_no_ul = np.log10(N_total / O_total)
                    lines.append(f"  → log(N/O) < {log_no_ul:.3f} (using upper"
                                 f" limits for non-detected N ions)")

                # C/O upper limit.
                _c_ions = ["C+/H+", "C++/H+", "C+++/H+"]
                C_total = 0.0
                has_ul_c = False
                for ck in _c_ions:
                    detected = self.ionic.get(ck, 0.0) if self.ionic else 0.0
                    if detected > 0:
                        C_total += detected
                    elif ck in _ul:
                        C_total += _ul[ck]
                        has_ul_c = True
                if C_total > 0 and has_ul_c:
                    log_co_ul = np.log10(C_total / O_total)
                    lines.append(f"  → log(C/O) < {log_co_ul:.3f} (using upper"
                                 f" limits for non-detected C ions)")

        # --- Alternative method results ---
        if self.alt_results:
            lines.append("")
            lines.append("Alternative methods (not selected):")
            for alt_name, alt in self.alt_results.items():
                lines.append(f"  [{alt_name}]")
                if alt.OH is not None and np.isfinite(alt.OH):
                    lines.append(f"    12+log(O/H) = {alt.OH:.3f} {_fmt_err_or_blank(alt.OH_err)}")
                if alt.NO is not None:
                    lines.append(f"    log(N/O)    = {alt.NO:.3f} {_fmt_err_or_blank(alt.NO_err)}")
                if alt.CO is not None:
                    lines.append(f"    log(C/O)    = {alt.CO:.3f} {_fmt_err_or_blank(alt.CO_err)}")
                if alt.ratios_used:
                    lines.append(f"    Ratios used = {alt.ratios_used}")
                if alt.Te_high is not None:
                    if alt.Te_high_err is not None:
                        lines.append(f"    T_e(high)   = {alt.Te_high:.0f} {_fmt_err(alt.Te_high_err, 0)} K")
                    else:
                        lines.append(f"    T_e(high)   = {alt.Te_high:.0f} K")

        return "\n".join(lines)
