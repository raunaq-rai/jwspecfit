"""Noise-based flux limits for non-detected emission lines.

A line that is not detected has no meaningful fitted flux and no
meaningful fitted width: the optimiser lands wherever the noise and the
bounds push it.  Quoting ``n_sigma * flux_err`` therefore reports the
degeneracies of the fit rather than the sensitivity of the spectrum.

This module builds the limit the other way round, from the data:

1. **Assume a width.**  The line is taken to have the instrumental
   width at its expected position, broadened by the intrinsic velocity
   width of the galaxy — measured from the lines that *are* detected in
   the same spectrum (:func:`intrinsic_sigma_kms`), so the assumed
   profile matches the real ones.
2. **Measure the noise with that profile.**  The assumed Gaussian is
   slid across the line-free residuals either side of the expected
   position and the scatter of the fluxes it recovers is taken as the
   1σ flux uncertainty (:func:`matched_filter_flux_error`).

Step 2 is an empirical matched filter.  For white noise it reduces
exactly to the textbook result

.. math::

    \\sigma_F = \\sigma_\\mathrm{rms}\\,\\sqrt{2\\sqrt{\\pi}\\,\\sigma\\,\\Delta\\lambda},

which is what "integrate the error spectrum in quadrature over the line
profile" evaluates to and what the literature normally quotes.  Measuring
it rather than evaluating it matters because NIRSpec's 1D spectra have
correlated adjacent pixels — the 2D extraction and resampling see to
that — so the white-noise formula is optimistic by a factor that depends
on the extraction and cannot be written down.  The sliding filter picks
that correlation up for free.

Note what this is *not*: ``rms * sigma * sqrt(2 * pi)``, the form used in
earlier versions of this package, assumes the noise is fully coherent
across the line and runs ~1.3x high at JWST's near-Nyquist sampling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import pi, sqrt
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .resolution import R_from_pixels, sigma_inst_A

logger = logging.getLogger(__name__)

_C_KMS = 2.99792458e5
_SQRT2PI = sqrt(2.0 * pi)

#: Minimum integrated SNR for a line to contribute to the intrinsic
#: velocity-width measurement.  Below this the fitted σ is noise.
SNR_RESOLVED_MIN: float = 5.0

#: Minimum number of such lines before the measured width is trusted.
N_LINES_RESOLVED_MIN: int = 2

#: Velocity width assumed when no line is detected well enough to
#: measure one.  Zero means "instrumental width only" — with nothing
#: detected there is no evidence for anything broader.
SIGMA_V_FALLBACK_KMS: float = 0.0

#: A measured width above this is flagged, not capped.  Narrow-line
#: regions do not reach it, so it usually means broad wings have been
#: absorbed into single Gaussians — common on PRISM, where every line is
#: instrumentally dominated and the deconvolution is a small difference
#: of large numbers.  It is left uncapped deliberately: silently
#: narrowing the assumed profile would make every limit *tighter*, which
#: is the one direction an upper limit must never be wrong in.  Pass
#: ``sigma_v_kms=`` to override.
SIGMA_V_WARN_KMS: float = 500.0

#: Template positions are drawn from an annulus spanning these multiples
#: of the assumed σ either side of the expected line centre.
_ANNULUS_INNER: float = 4.0
_ANNULUS_OUTER: float = 12.0

#: Gaussian template truncation radius, in units of the assumed σ.
_TEMPLATE_RADIUS: float = 4.0

#: Positions whose template is clipped by a spectrum edge or by masked
#: pixels are dropped; this is the fraction of the best position's
#: template weight they must retain to be kept.
_TEMPLATE_COMPLETENESS: float = 0.8

#: Fewest usable template positions before the empirical scatter is
#: trusted over the analytic white-noise formula.
_MIN_POSITIONS: int = 8


@dataclass(frozen=True)
class FluxNoise:
    """The 1σ flux uncertainty of an assumed-width line, and its provenance.

    Attributes
    ----------
    sigma_flux : float
        1σ uncertainty on the integrated flux of a Gaussian of width
        *sigma_A* centred at the expected position, in the flux units of
        the residuals used (f_lam × Angstrom for this package).
    sigma_A : float
        The assumed Gaussian σ (Angstrom) the filter was built with.
    method : str
        ``"matched_filter"`` when measured from the scatter of recovered
        fluxes, ``"matched_filter_analytic"`` when too few template
        positions were available and the white-noise formula was used.
    n_positions : int
        Number of template positions the scatter was measured over
        (zero for the analytic fallback).
    rms : float
        Local RMS of the residuals in the annulus, in f_lam units.
    """

    sigma_flux: float
    sigma_A: float
    method: str
    n_positions: int
    rms: float


def intrinsic_sigma_kms(
    lines: Mapping[str, Any],
    grating: str | None = None,
    R: float | Callable[[np.ndarray], np.ndarray] | None = None,
    snr_min: float = SNR_RESOLVED_MIN,
    min_lines: int = N_LINES_RESOLVED_MIN,
    skip: Iterable[str] = ("Lya",),
) -> float | None:
    """Measure the intrinsic velocity width from the detected lines.

    Each well-detected line contributes
    ``c * sqrt(sigma_fit^2 - sigma_inst^2) / lambda_obs``; the median is
    returned.  Lines narrower than the instrument contribute zero rather
    than an imaginary width, so an unresolved spectrum correctly yields
    a small or zero velocity width instead of ``None``.

    Parameters
    ----------
    lines : mapping
        ``{name: line_result}``.  Each value needs ``centroid_A``,
        ``sigma_A`` and either ``snr_int_err`` or ``snr``.
    grating : str, optional
        Grating name, for the instrumental width.
    R : float or callable, optional
        Resolving power (overrides *grating*).
    snr_min : float
        Minimum integrated SNR to contribute (default 5).
    min_lines : int
        Minimum number of contributing lines (default 2).  Below this
        ``None`` is returned and the caller should fall back.
    skip : iterable of str
        Line names to ignore.  Broad components (``"_BROAD"`` in the
        name) are always ignored, as is Lyα, whose profile is asymmetric
        and resonantly scattered rather than Gaussian.

    Returns
    -------
    float or None
        Intrinsic Gaussian σ in km/s, or ``None`` if too few lines are
        detected well enough to measure one.
    """
    skip_set = set(skip)
    widths: list[float] = []
    for name, lr in lines.items():
        if name in skip_set or "_BROAD" in name:
            continue
        snr = getattr(lr, "snr_int_err", None)
        if snr is None:
            snr = getattr(lr, "snr", 0.0)
        lam_obs = float(getattr(lr, "centroid_A", 0.0))
        sig_fit = float(getattr(lr, "sigma_A", 0.0))
        if not (np.isfinite(snr) and snr >= snr_min):
            continue
        if not (lam_obs > 0 and sig_fit > 0 and np.isfinite(sig_fit)):
            continue

        sig_inst = float(sigma_inst_A(np.array([lam_obs * 1e-4]), grating=grating, R=R)[0])
        sig_int = sqrt(max(sig_fit**2 - sig_inst**2, 0.0))
        widths.append(_C_KMS * sig_int / lam_obs)

    if len(widths) < min_lines:
        logger.debug(
            "Only %d line(s) above SNR %.1f; cannot measure an intrinsic width.",
            len(widths), snr_min,
        )
        return None

    sigma_v = float(np.median(widths))
    if sigma_v > SIGMA_V_WARN_KMS:
        logger.warning(
            "Intrinsic velocity width from %d line(s) is σ_v = %.0f km/s, "
            "above the %.0f km/s expected of a narrow-line region — flux "
            "limits will assume that width and so come out wide.  Likely "
            "broad wings absorbed into single Gaussians, or an "
            "instrumentally dominated spectrum where the deconvolution is "
            "unstable.  Pass sigma_v_kms= to override.",
            len(widths), sigma_v, SIGMA_V_WARN_KMS,
        )
    else:
        logger.debug(
            "Intrinsic velocity width from %d line(s): σ_v = %.1f km/s.",
            len(widths), sigma_v,
        )
    return sigma_v


def assumed_sigma_A(
    lam_obs_A: float | np.ndarray,
    sigma_v_kms: float = 0.0,
    grating: str | None = None,
    R: float | Callable[[np.ndarray], np.ndarray] | None = None,
) -> float | np.ndarray:
    """Gaussian σ assumed for an undetected line, in Angstrom.

    ``sigma^2 = sigma_inst(lambda)^2 + (sigma_v / c * lambda_obs)^2``.

    Parameters
    ----------
    lam_obs_A : float or array_like
        Expected observed wavelength(s) in Angstrom.
    sigma_v_kms : float
        Intrinsic velocity width (km/s).  Zero gives the instrumental
        width alone.
    grating : str, optional
        Grating name.
    R : float or callable, optional
        Resolving power (overrides *grating*).

    Returns
    -------
    float or np.ndarray
        Assumed σ in Angstrom, matching the shape of *lam_obs_A*.
    """
    lam = np.atleast_1d(np.asarray(lam_obs_A, dtype=float))
    sig_inst = sigma_inst_A(lam * 1e-4, grating=grating, R=R)
    sig_v = (float(sigma_v_kms) / _C_KMS) * lam
    out = np.sqrt(sig_inst**2 + sig_v**2)
    if np.isscalar(lam_obs_A) or np.ndim(lam_obs_A) == 0:
        return float(out[0])
    return out


def _gaussian_templates(
    lam_slab: np.ndarray,
    centres: np.ndarray,
    sigma_A: float,
    usable: np.ndarray,
) -> np.ndarray:
    """Build truncated unit-area Gaussians at each centre, one row each."""
    d = lam_slab[None, :] - centres[:, None]
    g = np.exp(-0.5 * (d / sigma_A) ** 2) / (sigma_A * _SQRT2PI)
    return np.where(usable[None, :] & (np.abs(d) < _TEMPLATE_RADIUS * sigma_A), g, 0.0)


def matched_filter_flux_error(
    wave_A: np.ndarray,
    residuals: np.ndarray,
    valid: np.ndarray,
    lam_obs_A: float,
    sigma_A: float,
) -> FluxNoise | None:
    """1σ flux uncertainty for a Gaussian of assumed width at a given position.

    The assumed profile is slid across the line-free residuals in an
    annulus around *lam_obs_A* and fitted for its amplitude at each
    position; the RMS of those amplitudes is the flux uncertainty.
    Because the residuals already have the continuum and every fitted
    line subtracted, the recovered amplitudes scatter about zero, and
    any pixel-to-pixel correlation in the data is carried through
    automatically.

    Parameters
    ----------
    wave_A : np.ndarray
        Observed wavelength array (Angstrom).
    residuals : np.ndarray
        Continuum- and model-subtracted residuals, in f_lam units.
    valid : np.ndarray
        Boolean mask of usable pixels.
    lam_obs_A : float
        Expected observed wavelength of the line (Angstrom).
    sigma_A : float
        Assumed Gaussian σ (Angstrom), from :func:`assumed_sigma_A`.

    Returns
    -------
    FluxNoise or None
        ``None`` when there are too few usable pixels around the line
        for even the analytic fallback.
    """
    if not (sigma_A > 0 and np.isfinite(sigma_A)):
        return None

    wave_A = np.asarray(wave_A, dtype=float)
    resid = np.asarray(residuals, dtype=float)
    usable = np.asarray(valid, dtype=bool) & np.isfinite(resid)

    near = np.abs(wave_A - lam_obs_A)
    slab = near < (_ANNULUS_OUTER + _TEMPLATE_RADIUS) * sigma_A
    if int(np.sum(slab & usable)) < 3:
        return None

    lam_slab = wave_A[slab]
    r_slab = np.where(usable[slab], resid[slab], 0.0)
    ok_slab = usable[slab]
    near_slab = near[slab]

    annulus = ok_slab & (near_slab > _ANNULUS_INNER * sigma_A) & (
        near_slab < _ANNULUS_OUTER * sigma_A
    )
    if int(np.sum(annulus)) < 3:
        # Too tight a window: widen it and drop the central exclusion.
        annulus = ok_slab & (near_slab < _ANNULUS_OUTER * sigma_A)
    if int(np.sum(annulus)) < 3:
        return None

    rms = float(np.sqrt(np.mean(r_slab[annulus] ** 2)))

    # --- Empirical matched filter over the annulus positions ---
    centres = lam_slab[annulus]
    g = _gaussian_templates(lam_slab, centres, sigma_A, ok_slab)
    weight = np.sum(g * g, axis=1)
    if weight.size and np.max(weight) > 0:
        keep = weight >= _TEMPLATE_COMPLETENESS * np.max(weight)
        if int(np.sum(keep)) >= _MIN_POSITIONS:
            amps = np.sum(g[keep] * r_slab[None, :], axis=1) / weight[keep]
            return FluxNoise(
                sigma_flux=float(np.sqrt(np.mean(amps**2))),
                sigma_A=float(sigma_A),
                method="matched_filter",
                n_positions=int(np.sum(keep)),
                rms=rms,
            )

    # --- Analytic fallback: white noise, template at the line itself ---
    g_line = _gaussian_templates(lam_slab, np.array([lam_obs_A]), sigma_A, ok_slab)
    weight_line = float(np.sum(g_line * g_line))
    if weight_line <= 0:
        return None
    logger.debug(
        "Too few template positions at %.1f Å; using the white-noise formula.",
        lam_obs_A,
    )
    return FluxNoise(
        sigma_flux=rms / sqrt(weight_line),
        sigma_A=float(sigma_A),
        method="matched_filter_analytic",
        n_positions=0,
        rms=rms,
    )


def flux_upper_limit(
    wave_A: np.ndarray,
    residuals: np.ndarray,
    valid: np.ndarray,
    lam_obs_A: float,
    sigma_A: float,
    n_sigma: float = 3.0,
) -> float | None:
    """Convenience wrapper: ``n_sigma`` times :func:`matched_filter_flux_error`.

    Parameters
    ----------
    wave_A, residuals, valid, lam_obs_A, sigma_A
        As for :func:`matched_filter_flux_error`.
    n_sigma : float
        Number of sigma for the limit (default 3).

    Returns
    -------
    float or None
        Integrated flux upper limit in f_lam × Angstrom, or ``None``.
    """
    noise = matched_filter_flux_error(
        wave_A, residuals, valid, lam_obs_A, sigma_A,
    )
    if noise is None:
        return None
    return float(n_sigma) * noise.sigma_flux


def resolution_of(spec: Any) -> tuple[str | None, Any]:
    """Return ``(grating, R)`` for a spectrum, inferring R when neither is set.

    Parameters
    ----------
    spec : Spectrum
        Spectrum carrying optional ``grating`` and ``R`` attributes.

    Returns
    -------
    tuple
        ``(grating, R)`` suitable for :func:`sigma_inst_A`.  When the
        spectrum declares neither, R is inferred from the pixel
        sampling via :func:`~jwspecfit.resolution.R_from_pixels`.
    """
    grating = getattr(spec, "grating", None)
    R = getattr(spec, "R", None)
    if grating is None and R is None:
        R = R_from_pixels(spec.wave_um)
    return grating, R


def flux_noise_for_lines(
    lines: Mapping[str, Any],
    wave_A: np.ndarray,
    residuals: np.ndarray,
    valid: np.ndarray,
    z: float,
    grating: str | None = None,
    R: float | Callable[[np.ndarray], np.ndarray] | None = None,
    sigma_v_kms: float | None = None,
) -> tuple[dict[str, FluxNoise], float]:
    """Assumed-width flux uncertainties for every line in a fit.

    The intrinsic velocity width is measured once from the detected
    lines and applied to all of them, so an undetected line inherits the
    width of the galaxy's real lines rather than whatever the optimiser
    left it at.

    Each line is filtered at its *expected* position,
    ``rest_wave_A * (1 + z)``, not its fitted centroid: for a
    non-detection the centroid is as unconstrained as the width.

    Parameters
    ----------
    lines : mapping
        ``{name: line_result}``; each value needs ``rest_wave_A``,
        ``centroid_A``, ``sigma_A`` and an SNR attribute.
    wave_A : np.ndarray
        Observed wavelength array (Angstrom).
    residuals : np.ndarray
        Continuum- and model-subtracted residuals in f_lam units.
    valid : np.ndarray
        Boolean mask of usable pixels.
    z : float
        Source redshift, for the expected line positions.
    grating : str, optional
        Grating name.
    R : float or callable, optional
        Resolving power (overrides *grating*).
    sigma_v_kms : float, optional
        Intrinsic velocity width to assume.  When ``None`` (the default)
        it is measured from the detected lines, falling back to
        :data:`SIGMA_V_FALLBACK_KMS`.

    Returns
    -------
    noises : dict of {str: FluxNoise}
        One entry per line the filter could be applied to.
    sigma_v_used : float
        The velocity width actually assumed, in km/s.
    """
    if sigma_v_kms is None:
        measured = intrinsic_sigma_kms(lines, grating=grating, R=R)
        sigma_v_used = SIGMA_V_FALLBACK_KMS if measured is None else measured
    else:
        sigma_v_used = float(sigma_v_kms)

    noises: dict[str, FluxNoise] = {}
    for name, lr in lines.items():
        lam_rest = float(getattr(lr, "rest_wave_A", 0.0))
        lam_obs = lam_rest * (1.0 + z) if lam_rest > 0 else float(
            getattr(lr, "centroid_A", 0.0)
        )
        if lam_obs <= 0:
            continue
        sig_A = assumed_sigma_A(lam_obs, sigma_v_used, grating=grating, R=R)
        noise = matched_filter_flux_error(wave_A, residuals, valid, lam_obs, sig_A)
        if noise is not None:
            noises[name] = noise

    return noises, sigma_v_used
