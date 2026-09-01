"""Dust correction for emission-line fluxes.

Implements Salim+18/Noll+09 (Calzetti base + UV bump + slope deviation)
and Cardelli+89 Milky Way extinction, plus A_V derivation from the
Balmer decrement and Lyα escape fraction computation.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Intrinsic Balmer decrements (Case B, T=10^4 K, n_e=100 cm^-3).
# Osterbrock & Ferland (2006), Table 4.2.
BALMER_RATIOS: dict[str, float] = {
    "Ha/Hb": 2.86,
    "Hg/Hb": 0.468,
    "Hd/Hb": 0.259,
    "He/Hb": 0.159,
    "H8/Hb": 0.105,
    "H9/Hb": 0.0731,
    "H10/Hb": 0.0530,
}


# ---------------------------------------------------------------------------
# Attenuation / extinction curves
# ---------------------------------------------------------------------------

def _calzetti_base(wave_um: np.ndarray) -> np.ndarray:
    """Calzetti+00 starburst attenuation curve k'(lambda).

    Parameters
    ----------
    wave_um : np.ndarray
        Rest-frame wavelength in microns.

    Returns
    -------
    np.ndarray
        k'(lambda) values.
    """
    k = np.zeros_like(wave_um, dtype=float)

    lo = wave_um < 0.63
    hi = ~lo

    # 0.12 <= lambda < 0.63 um
    w = wave_um[lo]
    k[lo] = (
        2.659 * (-2.156 + 1.509 / w - 0.198 / w**2 + 0.011 / w**3)
        + 4.05
    )

    # 0.63 <= lambda <= 2.20 um
    w = wave_um[hi]
    k[hi] = 2.659 * (-1.857 + 1.040 / w) + 4.05

    return np.clip(k, 0.0, None)


def _drude(wave_um: np.ndarray, x0: float = 0.2175, gamma: float = 0.035) -> np.ndarray:
    """Drude profile for the 2175 Å UV bump.

    Parameters
    ----------
    wave_um : np.ndarray
        Wavelength in microns.
    x0 : float
        Central wavelength of the bump in microns.
    gamma : float
        Width of the bump in microns.

    Returns
    -------
    np.ndarray
        Drude profile D(lambda).
    """
    return (wave_um * gamma) ** 2 / (
        (wave_um**2 - x0**2) ** 2 + (wave_um * gamma) ** 2
    )


def salim_attenuation(
    wave_A: np.ndarray,
    Av: float,
    Rv: float = 3.15,
    delta: float = -0.35,
    B: float = 2.27,
) -> np.ndarray:
    """Salim+18/Noll+09 modified Calzetti attenuation curve.

    A(lambda) = E(B-V) * [k'_Calzetti * (lambda/0.55)^delta + D_bump * B]

    Parameters
    ----------
    wave_A : np.ndarray
        Rest-frame wavelength in Angstroms.
    Av : float
        V-band attenuation A_V.
    Rv : float
        Total-to-selective ratio (default 3.15).
    delta : float
        Power-law slope deviation (default -0.35).
    B : float
        UV bump strength (default 2.27).

    Returns
    -------
    np.ndarray
        A(lambda) in magnitudes at each wavelength.
    """
    wave_um = np.asarray(wave_A, dtype=float) * 1e-4
    k_cal = _calzetti_base(wave_um)
    Ebv = Av / Rv

    # Slope modification
    k_mod = k_cal * (wave_um / 0.55) ** delta

    # UV bump
    D = _drude(wave_um)
    k_mod = k_mod + B * D * Rv

    A_lambda = Ebv * k_mod
    return np.clip(A_lambda, 0.0, None)


def cardelli_extinction(
    wave_A: np.ndarray,
    Av: float,
    Rv: float = 3.1,
) -> np.ndarray:
    """Cardelli, Clayton & Mathis (1989) MW extinction curve.

    Parameters
    ----------
    wave_A : np.ndarray
        Rest-frame wavelength in Angstroms.
    Av : float
        V-band extinction A_V.
    Rv : float
        Total-to-selective ratio (default 3.1).

    Returns
    -------
    np.ndarray
        A(lambda) in magnitudes at each wavelength.
    """
    wave_um = np.asarray(wave_A, dtype=float) * 1e-4
    x = 1.0 / wave_um  # inverse microns

    a = np.zeros_like(x, dtype=float)
    b = np.zeros_like(x, dtype=float)

    # Infrared: 0.3 <= x < 1.1
    ir = (x >= 0.3) & (x < 1.1)
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61

    # Optical/NIR: 1.1 <= x < 3.3
    opt = (x >= 1.1) & (x < 3.3)
    y = x[opt] - 1.82
    a[opt] = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b[opt] = (
        0.0
        + 1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )

    # UV: 3.3 <= x < 8.0 (Cardelli+89 Section 3a)
    uv = (x >= 3.3) & (x < 8.0)
    xu = x[uv]
    Fa = np.zeros_like(xu)
    Fb = np.zeros_like(xu)
    uv2 = xu >= 5.9
    Fa[uv2] = -0.04473 * (xu[uv2] - 5.9) ** 2 - 0.009779 * (xu[uv2] - 5.9) ** 3
    Fb[uv2] = 0.2130 * (xu[uv2] - 5.9) ** 2 + 0.1207 * (xu[uv2] - 5.9) ** 3
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + Fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + Fb

    # Far-UV: x >= 8.0 (lambda <= 1250 A).  Cardelli+89 Section 3b.
    fuv = x >= 8.0
    xf = np.clip(x[fuv], None, 10.0)  # formulae valid to x=10 (1000 A)
    a[fuv] = -1.073 - 0.628 * (xf - 8.0) + 0.137 * (xf - 8.0) ** 2 - 0.070 * (xf - 8.0) ** 3
    b[fuv] = 13.670 + 4.257 * (xf - 8.0) - 0.420 * (xf - 8.0) ** 2 + 0.374 * (xf - 8.0) ** 3

    A_over_Av = a + b / Rv
    A_lambda = Av * A_over_Av
    return np.clip(A_lambda, 0.0, None)


# ---------------------------------------------------------------------------
# Dust correction of line fluxes
# ---------------------------------------------------------------------------

def dust_correct_fluxes(
    line_fluxes: dict[str, tuple[float, float, float]],
    Av: float,
    law: str = "salim",
    **kwargs,
) -> dict[str, tuple[float, float]]:
    """Apply dust correction to a set of emission-line fluxes.

    Parameters
    ----------
    line_fluxes : dict
        ``{line_name: (flux, flux_err, rest_wave_A)}``.
    Av : float
        V-band attenuation.
    law : str
        ``"salim"`` (default) or ``"cardelli"``.
    **kwargs
        Extra keyword arguments passed to the attenuation law
        (e.g. ``Rv``, ``delta``, ``B`` for Salim).

    Returns
    -------
    dict
        ``{line_name: (corrected_flux, corrected_flux_err)}``.
    """
    corrected = {}
    for name, (flux, flux_err, wave_A) in line_fluxes.items():
        wave_arr = np.array([wave_A])
        if law == "salim":
            A_lam = salim_attenuation(wave_arr, Av, **kwargs)[0]
        elif law == "cardelli":
            A_lam = cardelli_extinction(wave_arr, Av, **kwargs)[0]
        else:
            raise ValueError(f"Unknown dust law: {law!r}. Use 'salim' or 'cardelli'.")

        factor = 10.0 ** (0.4 * A_lam)
        corrected[name] = (flux * factor, flux_err * factor)

    return corrected


# ---------------------------------------------------------------------------
# A_V from Balmer decrement
# ---------------------------------------------------------------------------

def compute_Av_from_balmer(
    flux_num: float,
    flux_den: float,
    flux_num_err: float,
    flux_den_err: float,
    law: str = "salim",
    intrinsic_ratio: float = 0.468,
    wave_num_A: float = 4341.68,
    wave_den_A: float = 4862.68,
    clip: bool = True,
    **kwargs,
) -> tuple[float, float]:
    """Derive A_V from a Balmer decrement.

    By default uses Hgamma/Hbeta, but any pair of Balmer lines can be
    used by specifying the wavelengths and intrinsic ratio.

    Parameters
    ----------
    flux_num : float
        Observed flux of the numerator line (e.g. Hgamma).
    flux_den : float
        Observed flux of the denominator line (e.g. Hbeta).
    flux_num_err : float
        Error on numerator flux.
    flux_den_err : float
        Error on denominator flux.
    law : str
        ``"salim"`` (default) or ``"cardelli"``.
    intrinsic_ratio : float
        Intrinsic flux ratio (default 0.468 for Hgamma/Hbeta).
    wave_num_A : float
        Rest wavelength of numerator line in Angstroms (default Hgamma).
    wave_den_A : float
        Rest wavelength of denominator line in Angstroms (default Hbeta).
    clip : bool
        Clip a negative A_V to zero (default ``True``).  Attenuation
        cannot be negative, so clipping a *final* estimate is right --
        but clipping each decrement *before* averaging several of them
        biases the mean upward whenever the true A_V is near zero, so
        :func:`compute_Av_multi_balmer` passes ``clip=False`` here and
        clips only the combined value.
    **kwargs
        Extra arguments to the attenuation law.

    Returns
    -------
    tuple of float
        ``(Av, Av_err)``.
    """
    if flux_den <= 0 or flux_num <= 0:
        return 0.0, np.nan

    observed_ratio = flux_num / flux_den

    # Compute differential attenuation per unit Av at the two wavelengths.
    # A(lambda) = Av * f(lambda), where f = A(lambda)/A_V.
    Av_test = 1.0
    waves = np.array([wave_num_A, wave_den_A])
    if law == "salim":
        A_test = salim_attenuation(waves, Av_test, **kwargs)
    elif law == "cardelli":
        A_test = cardelli_extinction(waves, Av_test, **kwargs)
    else:
        raise ValueError(f"Unknown dust law: {law!r}")

    # f_num, f_den are A(lambda)/A_V for the two lines.
    f_num = A_test[0] / Av_test
    f_den = A_test[1] / Av_test

    # A_V = 2.5 * log10(R_obs / R_int) / (f_den - f_num)
    delta_f = f_den - f_num
    if abs(delta_f) < 1e-10:
        return 0.0, np.nan

    Av = 2.5 * np.log10(observed_ratio / intrinsic_ratio) / delta_f

    # Error propagation: dAv/dR * sigma_R
    frac_err = np.sqrt(
        (flux_num_err / flux_num) ** 2 + (flux_den_err / flux_den) ** 2
    )
    # d(Av)/d(R) = 2.5 / (R * ln(10) * delta_f)
    dAv_dR = 2.5 / (observed_ratio * np.log(10.0) * delta_f)
    sigma_R = observed_ratio * frac_err
    Av_err = abs(dAv_dR * sigma_R)

    return (max(Av, 0.0) if clip else Av), Av_err


# Balmer-line reference table: {name: (rest wavelength in Å, Case B ratio to Hβ)}.
# Ratios are at T=10⁴ K, n_e=100 cm⁻³ (Osterbrock & Ferland 2006).
# EXCLUDED: Hε (blended with [NeIII] 3968, Δλ=3 Å) and
#           H8 (blended with HeI 3889, Δλ=0.4 Å).
_BALMER_LADDER: dict[str, tuple[float, float]] = {
    "Ha":     (6564.61, 2.86),
    "HBETA":  (4862.68, 1.00),
    "HGAMMA": (4341.68, 0.468),
    "HDELTA": (4102.89, 0.259),
    "H9":     (3836.48, 0.0731),
    "H10":    (3799.00, 0.0530),
}

_VALID_BALMER_ANCHORS: tuple[str, ...] = ("HBETA", "Ha")


def _kappa(wave_A: np.ndarray, law: str, **kwargs) -> np.ndarray:
    """Attenuation per unit A_V, kappa(lambda) = A(lambda) / A_V.

    Parameters
    ----------
    wave_A : np.ndarray
        Rest-frame wavelengths in Angstroms.
    law : str
        ``"salim"`` or ``"cardelli"``.
    **kwargs
        Extra arguments to the attenuation law.

    Returns
    -------
    np.ndarray
        kappa at each wavelength.
    """
    wave_A = np.atleast_1d(np.asarray(wave_A, dtype=float))
    if law == "salim":
        return salim_attenuation(wave_A, 1.0, **kwargs)
    if law == "cardelli":
        return cardelli_extinction(wave_A, 1.0, **kwargs)
    raise ValueError(f"Unknown dust law: {law!r}. Use 'salim' or 'cardelli'.")


def _delta_kappa(wave_num_A: float, wave_den_A: float, law: str, **kwargs) -> float:
    """kappa(denominator) - kappa(numerator), the lever arm of a decrement.

    Parameters
    ----------
    wave_num_A, wave_den_A : float
        Rest wavelengths of the two lines in Angstroms.
    law : str
        ``"salim"`` or ``"cardelli"``.
    **kwargs
        Extra arguments to the attenuation law.

    Returns
    -------
    float
        ``kappa(den) - kappa(num)``, matching the ``delta_f`` of
        :func:`compute_Av_from_balmer`.  It is *negative* when the
        numerator is the bluer line, which is the usual case; take the
        absolute value for the lever arm's magnitude.
    """
    k = _kappa(np.array([wave_num_A, wave_den_A]), law, **kwargs)
    return float(k[1] - k[0])


def compute_Av_multi_balmer(
    fluxes: dict[str, float],
    errors: dict[str, float],
    law: str = "salim",
    snr_min: float = 3.0,
    anchor: str = "HBETA",
    **kwargs,
) -> dict[str, object]:
    """Derive A_V from every available Balmer decrement.

    Ratios each detected Balmer line **bluer than** the chosen *anchor*
    against it, and returns individual and weighted-average A_V values.
    Only lines at shorter wavelength than the anchor are used: anchor Hβ
    therefore uses Hγ/Hβ, Hδ/Hβ, H9/Hβ, H10/Hβ (Hα is **excluded**),
    while anchor Hα uses Hβ/Hα, Hγ/Hα, Hδ/Hα, H9/Hα, H10/Hα (all of them).

    Parameters
    ----------
    fluxes : dict
        Emission-line fluxes (observed, not dust-corrected).
    errors : dict
        Corresponding 1σ errors.
    law : str
        ``"salim"`` (default) or ``"cardelli"``.
    snr_min : float
        Minimum SNR for a Balmer line to be included (default 3.0).
    anchor : str
        Reference (denominator) line for the decrement.  ``"HBETA"``
        (default) uses only bluer lines (Hγ/Hβ, Hδ/Hβ, H9/Hβ, H10/Hβ),
        excluding Hα.  ``"Ha"`` uses every other Balmer line
        (Hβ/Hα, Hγ/Hα, Hδ/Hα, H9/Hα, H10/Hα).
    **kwargs
        Extra arguments to the attenuation law (Rv, delta, B).

    Returns
    -------
    dict
        Keys: ``"Av"`` (weighted mean), ``"Av_err"`` (error on mean),
        ``"individual"`` (list of dicts with per-line results),
        ``"n_lines"`` (number of lines used), ``"anchor"`` (anchor name).
    """
    if anchor not in _VALID_BALMER_ANCHORS:
        raise ValueError(
            f"anchor must be one of {_VALID_BALMER_ANCHORS}, got {anchor!r}"
        )

    if anchor not in fluxes or fluxes[anchor] <= 0:
        return {
            "Av": 0.0, "Av_err": np.nan, "individual": [],
            "n_lines": 0, "anchor": anchor,
        }

    anchor_wave, anchor_ratio_over_Hb = _BALMER_LADDER[anchor]
    anchor_flux = fluxes[anchor]
    anchor_err = errors.get(anchor, 0.0)

    results = []
    for name, (wave, ratio_over_Hb) in _BALMER_LADDER.items():
        if name == anchor:
            continue
        # Only use lines bluer than the anchor.  Anchor=Hβ therefore uses
        # Hγ/Hβ, Hδ/Hβ, H9/Hβ, H10/Hβ (Hα excluded); anchor=Hα uses every
        # other Balmer line (all are bluer than Hα).
        if wave >= anchor_wave:
            continue
        if name not in fluxes or fluxes[name] <= 0:
            continue
        f = fluxes[name]
        e = errors.get(name, 0.0)
        if e > 0 and f / e < snr_min:
            continue

        intrinsic_ratio = ratio_over_Hb / anchor_ratio_over_Hb

        # clip=False: each decrement enters the mean unclipped, otherwise
        # noise-driven negative values are folded to zero and the average
        # is biased high whenever the true A_V is close to zero.
        av, av_err = compute_Av_from_balmer(
            f, anchor_flux, e, anchor_err,
            law=law, intrinsic_ratio=intrinsic_ratio,
            wave_num_A=wave, wave_den_A=anchor_wave,
            clip=False, **kwargs,
        )
        # Part of av_err contributed by the *shared* anchor flux.  Every
        # decrement is ratioed against the same line, so these pieces are
        # perfectly correlated between decrements and must not be treated
        # as independent when the values are averaged.
        dkappa = abs(_delta_kappa(wave, anchor_wave, law, **kwargs))
        if dkappa > 0 and anchor_flux > 0:
            shared = (2.5 / np.log(10.0)) * (anchor_err / anchor_flux) / dkappa
        else:
            shared = 0.0
        results.append({
            "line": name, "wave": wave, "Av": av, "Av_err": av_err,
            "observed_ratio": f / anchor_flux,
            "intrinsic_ratio": intrinsic_ratio,
            "Av_err_shared": shared,
        })

    if not results:
        return {
            "Av": 0.0, "Av_err": np.nan, "individual": results,
            "n_lines": 0, "anchor": anchor,
        }

    # Generalised-least-squares mean.  The decrements share the anchor
    # flux, so their errors are correlated; the covariance is
    # C_ii = sigma_i^2, C_ij = s_i s_j (i != j) with s the anchor share.
    # Ignoring the off-diagonal terms understates the error on the mean.
    avs = np.array([r["Av"] for r in results])
    errs = np.array([r["Av_err"] for r in results])
    shared = np.array([r.get("Av_err_shared", 0.0) for r in results])
    valid = np.isfinite(errs) & (errs > 0)
    if valid.sum() >= 2:
        a, e, s = avs[valid], errs[valid], shared[valid]
        cov = np.outer(s, s)
        np.fill_diagonal(cov, e**2)
        ones = np.ones_like(a)
        try:
            w = np.linalg.solve(cov, ones)
            norm = float(ones @ w)
            if not np.isfinite(norm) or norm <= 0:
                raise np.linalg.LinAlgError
            av_mean = float(a @ w) / norm
            av_mean_err = 1.0 / np.sqrt(norm)
        except np.linalg.LinAlgError:
            w = 1.0 / e**2
            av_mean = float(np.average(a, weights=w))
            av_mean_err = 1.0 / np.sqrt(float(np.sum(w)))
    elif valid.sum() == 1:
        av_mean = avs[valid][0]
        av_mean_err = errs[valid][0]
    else:
        av_mean = np.median(avs)
        av_mean_err = np.nan

    return {
        "Av": max(av_mean, 0.0),
        "Av_err": av_mean_err,
        "individual": results,
        "n_lines": len(results),
        "anchor": anchor,
    }


def compute_Av_joint_balmer(
    fluxes: dict[str, float],
    errors: dict[str, float],
    law: str = "salim",
    snr_min: float = 3.0,
    Av_min: float = -1.0,
    Av_max: float = 6.0,
    n_grid: int = 7001,
    **kwargs,
) -> dict[str, object]:
    """Derive A_V by fitting the whole Balmer ladder at once.

    Rather than forming decrements against a chosen anchor, this fits
    every detected Balmer line simultaneously with two free parameters --
    an overall normalisation and ``A_V``::

        F_i^model = C * R_i * 10 ** (-0.4 * A_V * kappa(lambda_i))

    where ``R_i`` is the Case B ratio to Hbeta and ``kappa = A(lambda)/A_V``.
    The normalisation enters linearly and is solved in closed form at each
    ``A_V``, so the fit reduces to a one-dimensional profile chi-squared
    that is scanned on a grid; ``Av_err`` is the half-width of the
    ``delta chi-squared = 1`` interval.

    Why not simply average more decrements?  Balmer decrements are not
    independent measurements.  In magnitudes the decrements are exactly
    additive -- ``D(Hg/Ha) = D(Hg/Hb) + D(Hb/Ha)`` identically -- so a set
    of ``N`` lines carries only ``N - 1`` independent constraints however
    many of the ``N(N-1)/2`` pairs are formed.  Adding the remaining pairs
    contributes no information and, if they are averaged as though they
    were independent, shrinks the quoted error spuriously.  This fit uses
    each *flux* once and is the well-posed way to use them all together.

    Compared with :func:`compute_Av_multi_balmer` it is anchor-free (no
    line is privileged), unbiased near ``A_V = 0`` (nothing is clipped
    before averaging), and returns a goodness of fit -- a large
    ``chi2/dof`` flags a ladder that no single ``A_V`` can reconcile,
    usually stellar Balmer absorption or a flux-calibration step between
    gratings.

    Parameters
    ----------
    fluxes : dict
        Emission-line fluxes (observed, not dust-corrected).
    errors : dict
        Corresponding 1 sigma errors.
    law : str
        ``"salim"`` (default) or ``"cardelli"``.
    snr_min : float
        Minimum SNR for a Balmer line to be included (default 3.0).
    Av_min, Av_max : float
        Bounds of the ``A_V`` scan.  ``Av_min`` is deliberately negative
        so that the minimum is bracketed for dust-free objects; the
        returned ``"Av"`` is clipped at zero afterwards.
    n_grid : int
        Number of grid points in the scan (default 7001, i.e. 0.001 mag).
    **kwargs
        Extra arguments to the attenuation law (Rv, delta, B).

    Returns
    -------
    dict
        Keys: ``"Av"`` (clipped at zero), ``"Av_err"``, ``"Av_fit"``
        (unclipped best fit), ``"chi2"``, ``"dof"``, ``"individual"``
        (per-line observed and model fluxes), ``"n_lines"``,
        ``"anchor"`` (``"joint"``), ``"method"`` (``"joint"``).
    """
    empty = {
        "Av": 0.0, "Av_err": np.nan, "Av_fit": np.nan,
        "chi2": np.nan, "dof": 0, "individual": [],
        "n_lines": 0, "anchor": "joint", "method": "joint",
    }

    names, wave, ratio, flux, err = [], [], [], [], []
    for name, (w, r) in _BALMER_LADDER.items():
        f = fluxes.get(name)
        if f is None or f <= 0:
            continue
        e = errors.get(name, 0.0)
        if e <= 0 or f / e < snr_min:
            continue
        names.append(name)
        wave.append(w)
        ratio.append(r)
        flux.append(f)
        err.append(e)

    # Two free parameters, so two detected lines are the minimum; with
    # exactly two the fit is exact (dof = 0) and reproduces that decrement.
    if len(names) < 2:
        return empty

    wave = np.array(wave)
    ratio = np.array(ratio)
    flux = np.array(flux)
    err = np.array(err)
    kappa = _kappa(wave, law, **kwargs)

    grid = np.linspace(float(Av_min), float(Av_max), int(n_grid))
    # shape (grid, line): the ladder shape at each A_V, in units of Hbeta
    model = ratio[None, :] * 10.0 ** (-0.4 * grid[:, None] * kappa[None, :])
    weight = model / err**2
    denom = np.sum(model * weight, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = (flux @ weight.T) / denom          # best C at each A_V
    resid = (flux[None, :] - norm[:, None] * model) / err
    chi2 = np.sum(resid**2, axis=1)

    if not np.any(np.isfinite(chi2)):
        return empty

    j = int(np.nanargmin(chi2))
    av_fit = float(grid[j])
    chi2_min = float(chi2[j])

    within = np.isfinite(chi2) & (chi2 <= chi2_min + 1.0)
    if np.any(within):
        lo, hi = float(grid[within].min()), float(grid[within].max())
        av_err = 0.5 * (hi - lo)
        # An interval running into a scan edge is a lower bound on the error.
        if lo <= grid[0] or hi >= grid[-1]:
            logger.warning(
                "A_V confidence interval reaches the scan edge "
                "[%.2f, %.2f]; Av_err is a lower bound.", grid[0], grid[-1]
            )
    else:
        av_err = np.nan

    best_norm = float(norm[j])
    individual = [
        {
            "line": n, "wave": float(w), "flux": float(f), "flux_err": float(e),
            "model": float(best_norm * r * 10.0 ** (-0.4 * av_fit * k)),
        }
        for n, w, r, f, e, k in zip(names, wave, ratio, flux, err, kappa)
    ]

    return {
        "Av": max(av_fit, 0.0),
        "Av_err": av_err,
        "Av_fit": av_fit,
        "chi2": chi2_min,
        "dof": len(names) - 2,
        "individual": individual,
        "n_lines": len(names),
        "anchor": "joint",
        "method": "joint",
    }


def compute_Av_balmer_pair(
    fluxes: dict[str, float],
    errors: dict[str, float],
    pair: tuple[str, str],
    law: str = "salim",
    **kwargs,
) -> dict[str, object]:
    """Derive A_V from a single, explicitly-chosen Balmer decrement.

    Like :func:`compute_Av_multi_balmer` but restricted to exactly one
    ``(numerator, denominator)`` pair — useful when you want to force the
    dust correction onto a high-SNR pair (e.g. ``("Ha", "HBETA")``) and
    ignore lower-SNR Balmer lines.

    Parameters
    ----------
    fluxes, errors : dict
        Observed (not dust-corrected) line fluxes and 1σ errors.
    pair : tuple of str
        ``(numerator, denominator)`` line names from the Balmer ladder
        (``"Ha"``, ``"HBETA"``, ``"HGAMMA"``, ``"HDELTA"``, ``"H9"``,
        ``"H10"``), e.g. ``("Ha", "HBETA")`` for Hα/Hβ.
    law : str
        ``"salim"`` (default) or ``"cardelli"``.
    **kwargs
        Extra arguments to the attenuation law (Rv, delta, B).

    Returns
    -------
    dict
        Same shape as :func:`compute_Av_multi_balmer`: ``"Av"``,
        ``"Av_err"``, ``"individual"``, ``"n_lines"`` (1 if usable, else
        0), ``"anchor"`` (the denominator line).

    Raises
    ------
    ValueError
        If either line name is not in the Balmer ladder.
    """
    num, den = pair
    for nm in (num, den):
        if nm not in _BALMER_LADDER:
            raise ValueError(
                f"balmer_pair line {nm!r} not recognised; "
                f"choose from {tuple(_BALMER_LADDER)}"
            )

    if (num not in fluxes or fluxes[num] <= 0
            or den not in fluxes or fluxes[den] <= 0):
        return {
            "Av": 0.0, "Av_err": np.nan, "individual": [],
            "n_lines": 0, "anchor": den,
        }

    wave_num, ratio_num = _BALMER_LADDER[num]
    wave_den, ratio_den = _BALMER_LADDER[den]
    intrinsic_ratio = ratio_num / ratio_den

    av, av_err = compute_Av_from_balmer(
        fluxes[num], fluxes[den], errors.get(num, 0.0), errors.get(den, 0.0),
        law=law, intrinsic_ratio=intrinsic_ratio,
        wave_num_A=wave_num, wave_den_A=wave_den, **kwargs,
    )
    return {
        "Av": max(av, 0.0),
        "Av_err": av_err,
        "individual": [{
            "line": num, "wave": wave_num, "Av": av, "Av_err": av_err,
            "observed_ratio": fluxes[num] / fluxes[den],
            "intrinsic_ratio": intrinsic_ratio,
        }],
        "n_lines": 1,
        "anchor": den,
    }


# ---------------------------------------------------------------------------
# Lyα escape fraction
# ---------------------------------------------------------------------------

# Case B intrinsic Lyα/Hx ratios at T=10^4 K, n_e=10^3 cm^-3.
# Computed directly from PyNEB RecAtom('H',1) emissivities using
# Storey & Hummer (1995, MNRAS 272, 41) recombination tables.
LYA_CASE_B_RATIOS: dict[str, tuple[float, float]] = {
    # line_name: (Lyα/Hx ratio, rest wavelength of Hx in Å)
    # Hα excluded: at high-z it falls in a different grating and is not
    # used for the Balmer decrement dust correction either.
    "HBETA":   (26.071,  4862.68),
    "HGAMMA":  (55.642,  4341.68),
    "HDELTA":  (100.593, 4102.89),
}


def compute_lya_escape_fraction(
    lya_flux: float,
    lya_flux_err: float,
    fluxes_corr: dict[str, float],
    errors_corr: dict[str, float],
    snr_min: float = 3.0,
) -> dict[str, object]:
    """Compute Lyα escape fraction from dust-corrected Balmer lines.

    Each available Balmer line predicts an intrinsic Lyα flux via
    Case B recombination (T=10⁴ K, n_e=100 cm⁻³).  The escape
    fraction is the ratio of observed Lyα to intrinsic Lyα.

    When multiple Balmer lines are available, f_esc is computed as
    the inverse-variance weighted mean of the individual estimates.

    The observed Lyα flux should NOT be dust-corrected — the escape
    fraction encapsulates both dust attenuation and resonant scattering.

    Parameters
    ----------
    lya_flux : float
        Observed (not dust-corrected) Lyα flux.
    lya_flux_err : float
        1σ error on the observed Lyα flux.
    fluxes_corr : dict
        Dust-corrected emission-line fluxes keyed by line name.
    errors_corr : dict
        Dust-corrected 1σ errors keyed by line name.
    snr_min : float
        Minimum SNR for a Balmer line to be included (default 3.0).

    Returns
    -------
    dict
        Keys: ``"f_esc"`` (weighted mean), ``"f_esc_err"``,
        ``"individual"`` (list of per-line dicts with ``line``,
        ``f_esc``, ``f_esc_err``, ``lya_intrinsic``,
        ``lya_intrinsic_err``), ``"n_lines"`` (number used).
    """
    if lya_flux <= 0:
        return {"f_esc": np.nan, "f_esc_err": np.nan, "individual": [], "n_lines": 0}

    results = []
    for name, (ratio, _wave) in LYA_CASE_B_RATIOS.items():
        if name not in fluxes_corr or fluxes_corr[name] <= 0:
            continue
        f_hx = fluxes_corr[name]
        e_hx = errors_corr.get(name, 0.0)
        if e_hx > 0 and f_hx / e_hx < snr_min:
            continue

        # Intrinsic Lyα predicted from this Balmer line.
        lya_int = f_hx * ratio
        lya_int_err = e_hx * ratio

        # Escape fraction.
        f_esc_i = lya_flux / lya_int

        # Error propagation: f_esc = F_obs / F_int
        # σ(f_esc) = f_esc × sqrt((σ_obs/F_obs)² + (σ_int/F_int)²)
        frac_err = np.sqrt(
            (lya_flux_err / lya_flux) ** 2
            + (lya_int_err / lya_int) ** 2
        )
        f_esc_err_i = f_esc_i * frac_err

        results.append({
            "line": name,
            "f_esc": f_esc_i,
            "f_esc_err": f_esc_err_i,
            "lya_intrinsic": lya_int,
            "lya_intrinsic_err": lya_int_err,
        })

    if not results:
        return {"f_esc": np.nan, "f_esc_err": np.nan, "individual": results, "n_lines": 0}

    # Inverse-variance weighted mean.
    fescs = np.array([r["f_esc"] for r in results])
    errs = np.array([r["f_esc_err"] for r in results])
    valid = np.isfinite(errs) & (errs > 0)
    if valid.sum() >= 2:
        w = 1.0 / errs[valid] ** 2
        f_esc_mean = np.average(fescs[valid], weights=w)
        f_esc_mean_err = 1.0 / np.sqrt(np.sum(w))
    elif valid.sum() == 1:
        f_esc_mean = fescs[valid][0]
        f_esc_mean_err = errs[valid][0]
    else:
        f_esc_mean = np.median(fescs)
        f_esc_mean_err = np.nan

    return {
        "f_esc": float(f_esc_mean),
        "f_esc_err": float(f_esc_mean_err),
        "individual": results,
        "n_lines": len(results),
    }


def _draw_Av(
    rng: np.random.Generator,
    Av: float,
    Av_err: float,
    prior: str = "gaussian",
) -> float:
    """Draw a single A_V sample, clipped to >= 0.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    Av : float
        Central A_V value.
    Av_err : float
        Uncertainty on A_V (1σ for Gaussian, half-width for uniform).
    prior : str
        ``"gaussian"`` or ``"uniform"``.

    Returns
    -------
    float
        Drawn A_V value, guaranteed >= 0.
    """
    if Av_err <= 0 or not np.isfinite(Av_err):
        return max(Av, 0.0)
    if prior == "uniform":
        lo = max(Av - Av_err, 0.0)
        hi = Av + Av_err
        return rng.uniform(lo, hi)
    # Default: Gaussian, clipped at 0.
    return max(rng.normal(Av, Av_err), 0.0)


def compute_lya_escape_fraction_mc(
    lya_flux: float,
    lya_flux_err: float,
    fluxes_corr: dict[str, float],
    errors_corr: dict[str, float],
    Av: float,
    Av_err: float,
    Av_prior: str = "gaussian",
    dust_law: str = "salim",
    snr_min: float = 3.0,
    n_mc: int = 1000,
    rng: np.random.Generator | None = None,
    **dust_kwargs,
) -> dict[str, object]:
    """Compute Lyα escape fraction with MC error propagation.

    Draws ``n_mc`` samples from the observed Lyα flux, dust-corrected
    Balmer fluxes, and A_V, recomputing the dust correction at each
    draw.  Returns the median and 16th/84th percentile uncertainties.

    Parameters
    ----------
    lya_flux : float
        Observed (not dust-corrected) Lyα flux.
    lya_flux_err : float
        1σ error on Lyα flux.
    fluxes_corr : dict
        Dust-corrected Balmer fluxes (used as central values; the MC
        re-applies dust correction from observed fluxes internally).
    errors_corr : dict
        Dust-corrected 1σ errors.
    Av : float
        Central A_V value.
    Av_err : float
        1σ error on A_V (Gaussian) or half-width (uniform).
    Av_prior : str
        Prior shape for A_V: ``"gaussian"`` (default, draws from
        N(Av, Av_err) clipped at 0) or ``"uniform"`` (draws from
        U(max(Av − Av_err, 0), Av + Av_err)).
    dust_law : str
        ``"salim"`` or ``"cardelli"``.
    snr_min : float
        Minimum SNR for a Balmer line to be included (default 3.0).
    n_mc : int
        Number of Monte Carlo draws (default 1000).
    rng : np.random.Generator or None
        Random number generator (for reproducibility).
    **dust_kwargs
        Extra arguments to the dust law (Rv, delta, B).

    Returns
    -------
    dict
        Keys: ``"f_esc"`` (median), ``"f_esc_err"`` (lo, hi) as
        16th/84th percentile half-widths, ``"f_esc_posterior"``
        (1D array of MC samples), ``"individual"`` (per-line results),
        ``"n_lines"``.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Identify which Balmer lines to use (from the point-estimate result).
    lines_used = []
    for name, (ratio, wave) in LYA_CASE_B_RATIOS.items():
        if name not in fluxes_corr or fluxes_corr[name] <= 0:
            continue
        e = errors_corr.get(name, 0.0)
        if e > 0 and fluxes_corr[name] / e < snr_min:
            continue
        lines_used.append(name)

    if not lines_used:
        return {
            "f_esc": np.nan,
            "f_esc_err": (np.nan, np.nan),
            "f_esc_posterior": np.array([]),
            "individual": [],
            "n_lines": 0,
        }

    # Pre-compute observed (un-dust-corrected) Balmer fluxes by inverting
    # the dust correction on the central values.
    obs_fluxes = {}
    obs_errors = {}
    for name in lines_used:
        ratio_cb, wave = LYA_CASE_B_RATIOS[name]
        wave_arr = np.array([wave])
        if dust_law == "salim":
            A_lam = salim_attenuation(wave_arr, Av, **dust_kwargs)[0]
        else:
            A_lam = cardelli_extinction(wave_arr, Av, **dust_kwargs)[0]
        factor = 10.0 ** (0.4 * A_lam)
        obs_fluxes[name] = fluxes_corr[name] / factor
        obs_errors[name] = errors_corr.get(name, 0.0) / factor

    # MC loop.
    f_esc_samples = np.full(n_mc, np.nan)
    per_line_samples: dict[str, np.ndarray] = {
        name: np.full(n_mc, np.nan) for name in lines_used
    }

    for i in range(n_mc):
        # Draw Lyα flux.
        lya_draw = rng.normal(lya_flux, lya_flux_err)
        if lya_draw <= 0:
            continue

        # Draw A_V.
        Av_draw = _draw_Av(rng, Av, Av_err, prior=Av_prior)

        # For each Balmer line, draw observed flux, dust-correct, predict Lyα.
        f_esc_per_line = []
        f_esc_err_per_line = []
        for name in lines_used:
            ratio_cb, wave = LYA_CASE_B_RATIOS[name]
            # Draw observed flux.
            f_draw = rng.normal(obs_fluxes[name], obs_errors[name])
            if f_draw <= 0:
                continue
            # Dust-correct this draw.
            wave_arr = np.array([wave])
            if dust_law == "salim":
                A_lam = salim_attenuation(wave_arr, Av_draw, **dust_kwargs)[0]
            else:
                A_lam = cardelli_extinction(wave_arr, Av_draw, **dust_kwargs)[0]
            f_corr = f_draw * 10.0 ** (0.4 * A_lam)

            # Intrinsic Lyα from this line.
            lya_int = f_corr * ratio_cb
            f_esc_i = lya_draw / lya_int
            per_line_samples[name][i] = f_esc_i

            # Use 1/variance weighting (variance from point estimate).
            pt = fluxes_corr[name] * ratio_cb
            e_pt = errors_corr.get(name, 0.0) * ratio_cb
            if pt > 0 and e_pt > 0:
                var_i = (lya_flux / pt) ** 2 * (
                    (lya_flux_err / lya_flux) ** 2 + (e_pt / pt) ** 2
                )
                if var_i > 0:
                    f_esc_per_line.append(f_esc_i)
                    f_esc_err_per_line.append(np.sqrt(var_i))

        if f_esc_per_line:
            arr = np.array(f_esc_per_line)
            earr = np.array(f_esc_err_per_line)
            w = 1.0 / earr ** 2
            f_esc_samples[i] = np.average(arr, weights=w)

    # Summarise.
    valid = np.isfinite(f_esc_samples)
    if valid.sum() < 10:
        return {
            "f_esc": np.nan,
            "f_esc_err": (np.nan, np.nan),
            "f_esc_posterior": f_esc_samples[valid],
            "individual": [],
            "n_lines": len(lines_used),
        }

    posterior = f_esc_samples[valid]
    med = float(np.median(posterior))
    lo = med - float(np.percentile(posterior, 16))
    hi = float(np.percentile(posterior, 84)) - med

    # Per-line summaries.
    individual = []
    for name in lines_used:
        pl = per_line_samples[name]
        pl_valid = pl[np.isfinite(pl)]
        if len(pl_valid) > 10:
            individual.append({
                "line": name,
                "f_esc": float(np.median(pl_valid)),
                "f_esc_err": (
                    float(np.median(pl_valid) - np.percentile(pl_valid, 16)),
                    float(np.percentile(pl_valid, 84) - np.median(pl_valid)),
                ),
                "lya_intrinsic": fluxes_corr[name] * LYA_CASE_B_RATIOS[name][0],
            })

    return {
        "f_esc": med,
        "f_esc_err": (lo, hi),
        "f_esc_posterior": posterior,
        "individual": individual,
        "n_lines": len(lines_used),
    }
