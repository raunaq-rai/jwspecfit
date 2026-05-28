"""Strong-line redshift fitting via brute-force grid evaluation.

This module implements :func:`fit_redshift`, which determines the redshift
of a JWST NIRSpec spectrum by jointly fitting the amplitudes of a curated
set of strong emission lines on a dense grid of trial redshifts.  The
algorithm is deliberately exhaustive (no early stopping), it gracefully
skips lines that fall outside the spectrum's wavelength range at the
trial redshift, and it returns both a per-peak credible interval and a
ranked list of secondary peaks so that aliasing failures are visible.

The per-z score is the chi^2 of a non-negative joint line fit plus a
low-order polynomial continuum, computed via a single
:func:`scipy.optimize.lsq_linear` call with mixed bounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import lsq_linear

from .io import Spectrum
from .lines import REST_LINES_A
from .resolution import R_from_pixels, resolve_R

# NumPy 2.0 renamed ``np.trapz`` to ``np.trapezoid`` and removed the old
# name.  Resolve once at import so the package works under both old and
# new NumPy without per-call overhead.
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

logger = logging.getLogger(__name__)


DEFAULT_LINES: list[str] = [
    "Lya",
    "CIV_doublet",
    "HEII_1640",
    "CIII]",
    "OII_doublet",
    "NeIII_3869",
    "HBETA",
    "OIII_4959",
    "OIII_5007",
    "Ha",
]


# --- Result containers -----------------------------------------------------


@dataclass
class Peak:
    """A single local minimum of the chi^2(z) curve, refined and ranked."""

    z: float
    prob: float
    dchi2: float
    ci68: tuple[float, float]
    ci95: tuple[float, float]
    n_lines_used: int
    lines_used: list[str]


@dataclass
class RedshiftResult:
    """Output of :func:`fit_redshift`."""

    z_best: float
    z_ci68: tuple[float, float]
    z_ci95: tuple[float, float]

    peaks: list[Peak]
    is_decisive: bool

    z_grid_coarse: np.ndarray
    chi2_coarse: np.ndarray
    P_z_coarse: np.ndarray

    z_grid_fine: np.ndarray
    chi2_fine: np.ndarray
    P_z_fine: np.ndarray

    lines_used: list[str]
    grating: str | None
    spec: Spectrum = field(repr=False)

    def __repr__(self) -> str:
        return (
            f"RedshiftResult(z_best={self.z_best:.5f}, "
            f"ci68={self.z_ci68}, decisive={self.is_decisive}, "
            f"n_peaks={len(self.peaks)})"
        )

    def plot(self):  # pragma: no cover - thin plotly wrapper
        """Two-panel diagnostic: Delta chi^2(z) and the spectrum at z_best."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False,
            row_heights=[0.5, 0.5],
            vertical_spacing=0.12,
            subplot_titles=("Δχ²(z)", f"spectrum at z = {self.z_best:.5f}"),
        )

        dchi2 = self.chi2_coarse - np.nanmin(self.chi2_coarse)
        fig.add_trace(
            go.Scatter(x=self.z_grid_coarse, y=dchi2, mode="lines",
                       line=dict(color="black", width=1), name="Δχ² coarse",
                       showlegend=False),
            row=1, col=1,
        )
        for thr, lab in ((1.0, "1σ"), (4.0, "2σ"), (9.0, "3σ")):
            fig.add_hline(y=thr, line_dash="dot", line_color="grey",
                          annotation_text=lab, annotation_position="right",
                          row=1, col=1)
        for p in self.peaks:
            fig.add_vline(x=p.z, line_dash="dash", line_color="firebrick",
                          row=1, col=1,
                          annotation_text=f"z={p.z:.4f}<br>Δχ²={p.dchi2:.1f}",
                          annotation_position="top")

        spec = self.spec
        valid = spec.mask_valid()
        fig.add_trace(
            go.Scatter(x=spec.wave_A[valid], y=spec.flux_ujy[valid],
                       mode="lines",
                       line=dict(color="black", width=0.8, shape="hvh"),
                       name="data", showlegend=False),
            row=2, col=1,
        )
        for name in self.peaks[0].lines_used if self.peaks else []:
            lam_obs = REST_LINES_A[name] * (1.0 + self.z_best)
            fig.add_vline(x=lam_obs, line_dash="dot", line_color="steelblue",
                          row=2, col=1,
                          annotation_text=name, annotation_position="top")

        fig.update_xaxes(title_text="z", row=1, col=1)
        fig.update_yaxes(title_text="Δχ²", row=1, col=1)
        fig.update_xaxes(title_text="Wavelength [Å]", row=2, col=1)
        fig.update_yaxes(title_text="Flux [µJy]", row=2, col=1)
        fig.update_layout(
            template="plotly_white", height=700, width=1000,
            margin=dict(b=80),
        )
        return fig


# --- Prior -----------------------------------------------------------------


def _evaluate_prior(prior, z: np.ndarray) -> np.ndarray:
    """Return prior(z) as a non-negative array of the same shape as z."""
    z = np.asarray(z, dtype=float)
    if prior is None:
        return np.ones_like(z)
    if callable(prior):
        return np.clip(np.asarray(prior(z), dtype=float), 0.0, None)
    # Single (lo, hi) tuple of scalars.
    if (
        isinstance(prior, tuple)
        and len(prior) == 2
        and np.isscalar(prior[0])
        and np.isscalar(prior[1])
    ):
        lo, hi = float(prior[0]), float(prior[1])
        return ((z >= lo) & (z <= hi)).astype(float)
    # List/tuple of (lo, hi) windows.
    out = np.zeros_like(z)
    for win in prior:
        lo, hi = float(win[0]), float(win[1])
        out[(z >= lo) & (z <= hi)] = 1.0
    return out


# --- Coarse-grid step ------------------------------------------------------


def _resolve_R_callable(spec: Spectrum) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``R(lam_um) -> ndarray`` using the best resolution info available.

    Fallback chain:
      1. ``spec.R`` (scalar or callable) if set
      2. ``spec.grating`` (Jakobsen+22 PRISM curve, or constant for medium /
         high-res gratings) if set
      3. :func:`R_from_pixels` — estimate from the data's own pixel spacing
         (lambda / (2 * Delta_lambda))

    This lets ``fit_redshift`` work on stacked or co-added spectra where
    ``spec.grating`` is ``None``, by reading the resolution off the data
    itself.  When grating/R are both set, the package's calibrated R is
    used unchanged.
    """
    if spec.R is not None or spec.grating is not None:
        return lambda lam_um: resolve_R(lam_um, grating=spec.grating, R=spec.R)
    return R_from_pixels(spec.wave_um)


def _default_dz_coarse(spec: Spectrum) -> float:
    """Coarse-grid step in z, tied to the spectral resolving power.

    Picks ``dz = 0.5 / R`` so that a line moves about 1.18 sigma of the LSF
    per coarse step (Nyquist sampling of the per-line chi^2 peak).  Floored
    at 1e-4 to keep G395H from being undersampled while still bounding the
    grid size on extremely high-R user-supplied R callables.  The R used
    here is the same one :func:`_resolve_R_callable` would use, so
    grid-step and per-z line sigma are always consistent.
    """
    lam_mid_um = 0.5 * (spec.wave_um.min() + spec.wave_um.max())
    R_call = _resolve_R_callable(spec)
    R_mid = float(np.asarray(R_call(np.array([lam_mid_um])))[0])
    return max(1.0e-4, 0.5 / R_mid)


# --- Per-z chi^2 -----------------------------------------------------------


def _continuum_basis(wave_A: np.ndarray, order: int) -> np.ndarray:
    """Legendre-polynomial basis on the rescaled wavelength axis."""
    if order < 0:
        return np.empty((wave_A.size, 0))
    lo, hi = wave_A.min(), wave_A.max()
    x = 2.0 * (wave_A - lo) / (hi - lo) - 1.0
    # legvander returns shape (n, order+1) with columns L_0, L_1, ..., L_order.
    return np.polynomial.legendre.legvander(x, order)


def _per_z_chi2(
    z: float,
    wave_A: np.ndarray,
    data_w: np.ndarray,
    weights: np.ndarray,
    line_lambdas_rest_A: np.ndarray,
    line_names: list[str],
    R_callable: Callable[[np.ndarray], np.ndarray],
    cont_basis_w: np.ndarray,
    min_lines_in_range: int,
    chi2_cont_only: float,
    nonneg: bool,
) -> tuple[float, int, list[str]]:
    """Score a single trial z.

    Returns (chi2, n_lines_in_range, names_in_range).  When fewer than
    *min_lines_in_range* expected line positions fall inside the
    spectrum, returns the continuum-only chi^2 baseline and zero lines.
    """
    mu_A = line_lambdas_rest_A * (1.0 + z)
    R_at = R_callable(mu_A * 1e-4)  # R takes lam in microns
    sigma_LSF = mu_A / (2.3548 * R_at)
    margin = 3.0 * sigma_LSF
    in_range = (mu_A - margin >= wave_A[0]) & (mu_A + margin <= wave_A[-1])
    m = int(in_range.sum())
    if m < min_lines_in_range:
        return chi2_cont_only, 0, []

    mu_in = mu_A[in_range]
    sigma_in = sigma_LSF[in_range]
    names_in = [line_names[i] for i in np.where(in_range)[0]]

    # Weighted Gaussian columns: (npix, m).
    diff = wave_A[:, None] - mu_in[None, :]
    G = np.exp(-0.5 * (diff / sigma_in[None, :]) ** 2)
    D_lines = G * weights[:, None]

    n_cont = cont_basis_w.shape[1]

    # Solve via the normal equations.  D is (npix, m + n_cont) and m + n_cont
    # is always small (~17), so the (D.T D) system is microseconds to solve.
    # The continuum-continuum block of D.T D and the continuum part of D.T b
    # are precomputed up the call stack (cont_basis_w never changes); here we
    # only need the line-line and line-continuum cross blocks plus the line
    # projection onto the data.
    A_ll = D_lines.T @ D_lines                          # (m, m)
    A_lc = D_lines.T @ cont_basis_w                     # (m, n_cont)
    b_l = D_lines.T @ data_w                            # (m,)
    A = np.empty((m + n_cont, m + n_cont))
    A[:m, :m] = A_ll
    A[:m, m:] = A_lc
    A[m:, :m] = A_lc.T
    A[m:, m:] = _cont_AA(cont_basis_w)
    b = np.empty(m + n_cont)
    b[:m] = b_l
    b[m:] = _cont_Ab(cont_basis_w, data_w)

    # Tiny Tikhonov nudge: doublet lines whose Gaussians overlap heavily can
    # leave A almost singular, which makes np.linalg.solve return NaN/Inf
    # silently rather than raising.  Scaling by trace(A) keeps the bias far
    # below floating-point noise.
    if A.size:
        A.flat[:: m + n_cont + 1] += 1e-10 * (np.trace(A) / (m + n_cont))
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = None
    if x is None or not np.all(np.isfinite(x)):
        D = np.concatenate([D_lines, cont_basis_w], axis=1)
        x, *_ = np.linalg.lstsq(D, data_w, rcond=None)

    # Positivity post-fix: if any line amplitude went negative, drop those
    # lines and refit on the reduced design.
    if nonneg and np.any(x[:m] < 0):
        keep_line = x[:m] >= 0
        if not np.any(keep_line):
            # All lines went negative -> continuum-only is the answer.
            return chi2_cont_only, 0, []
        idx = np.concatenate([np.where(keep_line)[0],
                              np.arange(m, m + n_cont)])
        A2 = A[np.ix_(idx, idx)]
        b2 = b[idx]
        try:
            x2 = np.linalg.solve(A2, b2)
        except np.linalg.LinAlgError:
            x2 = None
        if x2 is None or not np.all(np.isfinite(x2)):
            D2 = np.concatenate(
                [D_lines[:, keep_line], cont_basis_w], axis=1)
            x2, *_ = np.linalg.lstsq(D2, data_w, rcond=None)
        # Reconstruct the model and chi^2.
        model = D_lines[:, keep_line] @ x2[: int(keep_line.sum())] \
                + cont_basis_w @ x2[int(keep_line.sum()):]
        r = model - data_w
        chi2 = float(np.dot(r, r))
        m = int(keep_line.sum())
        names_in = [n for n, k in zip(names_in, keep_line) if k]
    else:
        model = D_lines @ x[:m] + cont_basis_w @ x[m:]
        r = model - data_w
        chi2 = float(np.dot(r, r))

    return chi2, m, names_in


# Cache continuum-continuum normal-equation blocks (z-independent) across
# calls.  These are tiny (n_cont x n_cont and n_cont) so the cache is cheap.
_CONT_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _cont_AA(cont_basis_w: np.ndarray) -> np.ndarray:
    key = id(cont_basis_w)
    cached = _CONT_CACHE.get(key)
    if cached is None:
        AA = cont_basis_w.T @ cont_basis_w
        _CONT_CACHE[key] = (AA, None)
        return AA
    return cached[0]


def _cont_Ab(cont_basis_w: np.ndarray, data_w: np.ndarray) -> np.ndarray:
    key = id(cont_basis_w)
    AA, Ab = _CONT_CACHE.get(key, (None, None))
    if Ab is None:
        Ab = cont_basis_w.T @ data_w
        _CONT_CACHE[key] = (
            AA if AA is not None else cont_basis_w.T @ cont_basis_w, Ab)
    return Ab


# --- Peak detection, refinement, intervals --------------------------------


def _find_local_minima(chi2: np.ndarray) -> np.ndarray:
    """Indices of strict local minima of chi2 (interior points only)."""
    return np.where((chi2[1:-1] < chi2[:-2]) & (chi2[1:-1] < chi2[2:]))[0] + 1


def _refine_peak(
    z_centre: float,
    z_min: float,
    z_max: float,
    dz_coarse: float,
    score_fn: Callable[[float], tuple[float, int, list[str]]],
    n_fine: int = 401,
    half_window_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[int], list[list[str]]]:
    """Re-evaluate chi^2 on a fine grid around *z_centre*."""
    w = half_window_steps * dz_coarse
    z_lo = max(z_min, z_centre - w)
    z_hi = min(z_max, z_centre + w)
    if z_hi <= z_lo:
        return (np.array([z_centre]), np.array([np.inf]), [0], [[]])
    z_fine = np.linspace(z_lo, z_hi, n_fine)
    chi2 = np.empty_like(z_fine)
    n_lines = []
    names = []
    for i, z in enumerate(z_fine):
        c, n, nm = score_fn(z)
        chi2[i] = c
        n_lines.append(n)
        names.append(nm)
    return z_fine, chi2, n_lines, names


def _parabolic_refine(z: np.ndarray, chi2: np.ndarray, i: int) -> float:
    """Parabolic interpolation of the chi^2 minimum at index *i*."""
    if i == 0 or i == len(chi2) - 1:
        return float(z[i])
    y0, y1, y2 = chi2[i - 1], chi2[i], chi2[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom <= 0:
        return float(z[i])
    delta = 0.5 * (y0 - y2) / denom
    return float(z[i] + delta * (z[i + 1] - z[i]))


def _credible_interval(
    z: np.ndarray, chi2: np.ndarray, prior_vals: np.ndarray, frac: float
) -> tuple[float, float]:
    """Central `frac` credible interval from chi^2 + prior over a sub-grid."""
    if z.size < 2:
        return (float(z[0]), float(z[0]))
    chi2_min = float(np.min(chi2))
    u = np.exp(-0.5 * (chi2 - chi2_min)) * prior_vals
    if not np.any(u > 0):
        return (float(z[0]), float(z[-1]))
    # Trapezoidal CDF.
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (u[1:] + u[:-1]) * np.diff(z))])
    if cdf[-1] <= 0:
        return (float(z[0]), float(z[-1]))
    cdf /= cdf[-1]
    lo_q = 0.5 * (1.0 - frac)
    hi_q = 0.5 * (1.0 + frac)
    z_lo = float(np.interp(lo_q, cdf, z))
    z_hi = float(np.interp(hi_q, cdf, z))
    return (z_lo, z_hi)


def _local_mode_range(z: np.ndarray, chi2: np.ndarray, i: int,
                      dchi2_thresh: float = 25.0) -> tuple[int, int]:
    """Indices that bound the local mode around index *i*.

    Walks outward until chi^2 has risen by *dchi2_thresh* (or the grid
    edge is reached, or another minimum is encountered).
    """
    chi2_peak = chi2[i]
    lo = i
    while lo > 0 and chi2[lo - 1] - chi2_peak < dchi2_thresh:
        if chi2[lo - 1] < chi2[lo]:
            break  # rolled into another minimum
        lo -= 1
    hi = i
    while hi < len(chi2) - 1 and chi2[hi + 1] - chi2_peak < dchi2_thresh:
        if chi2[hi + 1] < chi2[hi]:
            break
        hi += 1
    return lo, hi


# --- Main entry point ------------------------------------------------------


def fit_redshift(
    spec: Spectrum,
    *,
    lines: list[str] | None = None,
    z_min: float = 0.0,
    z_max: float = 20.0,
    dz_coarse: float | None = None,
    prior=None,
    n_peaks: int = 5,
    refine_dchi2: float = 9.0,
    continuum_order: int = 3,
    nonneg: bool = True,
    min_lines_in_range: int = 2,
    verbose: bool = True,
) -> RedshiftResult:
    """Fit the redshift of *spec* by joint strong-line fitting on a z grid.

    Parameters
    ----------
    spec : Spectrum
        Input spectrum.  Must have valid `wave_um`, `flux_ujy`, `err_ujy`.
    lines : list of str, optional
        Line names (keys of :data:`REST_LINES_A`) to include.  Defaults
        to :data:`DEFAULT_LINES`.
    z_min, z_max : float
        Bounds of the redshift search (default 0 to 20).
    dz_coarse : float, optional
        Coarse-grid step.  Defaults to a value tied to the spectral
        resolving power (see :func:`_default_dz_coarse`).
    prior : None | (lo, hi) | list[(lo, hi)] | callable
        Optional non-Gaussian prior over z.  ``None`` means flat.
    n_peaks : int
        Minimum number of peaks to refine and return.
    refine_dchi2 : float
        All local minima within this Δχ² of the global minimum are
        refined, in addition to the top *n_peaks*.
    continuum_order : int
        Order of the Legendre-polynomial continuum that is added to the
        per-z linear fit.  Default 3.
    nonneg : bool
        If True, line amplitudes are constrained to be non-negative
        (emission only).  Strongly recommended.
    min_lines_in_range : int
        Trial redshifts where fewer than this many default lines fall
        inside the spectrum get the continuum-only chi^2 baseline.
    verbose : bool
        If True, log a few timing milestones at INFO level.

    Returns
    -------
    RedshiftResult
    """
    if lines is None:
        lines = list(DEFAULT_LINES)
    for name in lines:
        if name not in REST_LINES_A:
            raise KeyError(f"Unknown line name: {name!r}")

    # Mask invalid pixels.
    mask = spec.mask_valid()
    if mask.sum() < 10:
        raise ValueError("Spectrum has too few valid pixels for redshift fitting.")
    wave_A = spec.wave_A[mask]
    flux = spec.flux_ujy[mask]
    err = spec.err_ujy[mask]
    weights = 1.0 / err
    data_w = flux * weights

    # Precompute basis (z-independent).
    cont_basis = _continuum_basis(wave_A, continuum_order)
    cont_basis_w = cont_basis * weights[:, None]

    # Continuum-only chi^2 (used as the baseline when no lines are in range).
    x_cont, *_ = np.linalg.lstsq(cont_basis_w, data_w, rcond=None)
    r_cont = cont_basis_w @ x_cont - data_w
    chi2_cont_only = float(np.dot(r_cont, r_cont))

    # Line wavelengths and a resolution callable.  R is sourced from the
    # spectrum via the fallback chain (R -> grating -> pixel-spacing), so
    # stacked spectra without a grating header still get a sensible R(lam).
    line_lambdas = np.array([REST_LINES_A[n] for n in lines], dtype=float)
    R_callable = _resolve_R_callable(spec)

    # Coarse grid.
    if dz_coarse is None:
        dz_coarse = _default_dz_coarse(spec)
    z_coarse = np.arange(z_min, z_max + dz_coarse, dz_coarse)
    # Apply prior to bound the actual evaluation set; outside-prior
    # entries get chi2_cont_only without an lsq_linear call.
    prior_coarse = _evaluate_prior(prior, z_coarse)
    eval_mask = prior_coarse > 0

    if verbose:
        logger.info("fit_redshift: %d coarse grid points (%.1f%% with prior support)",
                    len(z_coarse), 100.0 * eval_mask.mean())

    chi2_coarse = np.full_like(z_coarse, chi2_cont_only)
    n_lines_coarse = np.zeros(len(z_coarse), dtype=int)
    eval_idx = np.where(eval_mask)[0]
    for j, idx in enumerate(eval_idx):
        c, n, _ = _per_z_chi2(
            z_coarse[idx], wave_A, data_w, weights, line_lambdas, lines,
            R_callable, cont_basis_w, min_lines_in_range, chi2_cont_only,
            nonneg,
        )
        chi2_coarse[idx] = c
        n_lines_coarse[idx] = n

    # Find and refine peaks.
    local_min_idx = _find_local_minima(chi2_coarse)
    # Keep only local minima with prior support and at least min_lines_in_range
    # lines actually used; otherwise they are noise-floor artefacts.
    local_min_idx = local_min_idx[
        (prior_coarse[local_min_idx] > 0)
        & (n_lines_coarse[local_min_idx] >= min_lines_in_range)
    ]
    if len(local_min_idx) == 0:
        raise RuntimeError(
            "No local minimum found inside the prior support with at least "
            f"{min_lines_in_range} lines in range."
        )

    chi2_at_min = chi2_coarse[local_min_idx]
    order = np.argsort(chi2_at_min)
    local_min_idx_sorted = local_min_idx[order]
    chi2_min_global = chi2_at_min[order[0]]
    # Keep top n_peaks AND any others within refine_dchi2 of best.
    keep_within = chi2_at_min[order] - chi2_min_global <= refine_dchi2
    n_keep = max(n_peaks, int(keep_within.sum()))
    n_keep = min(n_keep, len(local_min_idx_sorted))
    kept = local_min_idx_sorted[:n_keep]

    # Refine each kept peak.
    def score_fn(z):
        return _per_z_chi2(z, wave_A, data_w, weights, line_lambdas, lines,
                           R_callable, cont_basis_w, min_lines_in_range,
                           chi2_cont_only, nonneg)

    fine_z = []
    fine_chi2 = []
    peak_records = []  # (z_best, z_grid_local, chi2_local, prior_local, names_at_peak, n_lines_at_peak)

    for idx in kept:
        z_c = z_coarse[idx]
        z_fine, chi2_fine, n_fine, names_fine = _refine_peak(
            z_c, z_min, z_max, dz_coarse, score_fn,
            n_fine=401, half_window_steps=10,
        )
        prior_fine = _evaluate_prior(prior, z_fine)
        i_min = int(np.argmin(chi2_fine))
        z_refined = _parabolic_refine(z_fine, chi2_fine, i_min)
        peak_records.append({
            "z_refined": z_refined,
            "i_min": i_min,
            "z_fine": z_fine,
            "chi2_fine": chi2_fine,
            "prior_fine": prior_fine,
            "n_at_peak": int(n_fine[i_min]),
            "names_at_peak": list(names_fine[i_min]),
        })
        fine_z.append(z_fine)
        fine_chi2.append(chi2_fine)

    z_grid_fine = np.concatenate(fine_z)
    chi2_grid_fine = np.concatenate(fine_chi2)

    # Build Peak objects with per-peak credible intervals and integrated masses.
    chi2_best = min(rec["chi2_fine"][rec["i_min"]] for rec in peak_records)
    masses = []
    for rec in peak_records:
        z_f = rec["z_fine"]
        chi2_f = rec["chi2_fine"]
        prior_f = rec["prior_fine"]
        lo, hi = _local_mode_range(z_f, chi2_f, rec["i_min"])
        z_loc = z_f[lo:hi + 1]
        chi2_loc = chi2_f[lo:hi + 1]
        prior_loc = prior_f[lo:hi + 1]
        u = np.exp(-0.5 * (chi2_loc - chi2_best)) * prior_loc
        m = float(_trapezoid(u, z_loc)) if z_loc.size > 1 else float(u.sum())
        masses.append(m)
        rec["z_loc"] = z_loc
        rec["chi2_loc"] = chi2_loc
        rec["prior_loc"] = prior_loc
        rec["u_loc"] = u

    total_mass = float(sum(masses))
    if total_mass <= 0:
        total_mass = 1.0  # avoid div-by-zero; all peaks then get prob=0.

    peaks: list[Peak] = []
    for rec, m in zip(peak_records, masses):
        ci68 = _credible_interval(rec["z_loc"], rec["chi2_loc"],
                                  rec["prior_loc"], 0.68)
        ci95 = _credible_interval(rec["z_loc"], rec["chi2_loc"],
                                  rec["prior_loc"], 0.95)
        dchi2 = float(rec["chi2_fine"][rec["i_min"]] - chi2_best)
        peaks.append(Peak(
            z=rec["z_refined"],
            prob=float(m / total_mass),
            dchi2=dchi2,
            ci68=ci68,
            ci95=ci95,
            n_lines_used=rec["n_at_peak"],
            lines_used=rec["names_at_peak"],
        ))

    # Sort peaks by prob descending (with flat prior this matches chi^2 order).
    peaks.sort(key=lambda p: -p.prob)
    best = peaks[0]
    is_decisive = (len(peaks) == 1) or (peaks[1].dchi2 > 25.0)

    # Coarse-grid P(z), normalised.
    dchi2_coarse = chi2_coarse - np.nanmin(chi2_coarse)
    P_coarse_unnorm = np.exp(-0.5 * dchi2_coarse) * prior_coarse
    z_step = np.gradient(z_coarse) if z_coarse.size > 1 else np.array([1.0])
    norm_c = float(np.sum(P_coarse_unnorm * z_step))
    P_coarse = P_coarse_unnorm / norm_c if norm_c > 0 else P_coarse_unnorm

    # Fine-grid P(z) (concatenation; not necessarily normalised globally).
    dchi2_fine_arr = chi2_grid_fine - np.nanmin(chi2_grid_fine)
    prior_fine_arr = _evaluate_prior(prior, z_grid_fine)
    P_fine = np.exp(-0.5 * dchi2_fine_arr) * prior_fine_arr

    if verbose:
        logger.info(
            "fit_redshift: z_best=%.5f (Δχ²=0), CI68=[%.5f, %.5f], "
            "%d peaks, decisive=%s",
            best.z, best.ci68[0], best.ci68[1], len(peaks), is_decisive,
        )

    return RedshiftResult(
        z_best=best.z,
        z_ci68=best.ci68,
        z_ci95=best.ci95,
        peaks=peaks,
        is_decisive=is_decisive,
        z_grid_coarse=z_coarse,
        chi2_coarse=chi2_coarse,
        P_z_coarse=P_coarse,
        z_grid_fine=z_grid_fine,
        chi2_fine=chi2_grid_fine,
        P_z_fine=P_fine,
        lines_used=list(lines),
        grating=spec.grating,
        spec=spec,
    )
