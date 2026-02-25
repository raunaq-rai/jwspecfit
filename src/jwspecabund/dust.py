"""Dust correction for emission-line fluxes.

Implements Salim+18/Noll+09 (Calzetti base + UV bump + slope deviation)
and Cardelli+89 Milky Way extinction, plus A_V derivation from the
Balmer decrement.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Intrinsic Balmer decrements (Case B, T=10^4 K, n_e=100 cm^-3).
# Osterbrock & Ferland (2006).
BALMER_RATIOS: dict[str, float] = {
    "Ha/Hb": 2.86,
    "Hg/Hb": 0.468,
    "Hd/Hb": 0.259,
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

    # UV: 3.3 <= x < 8.0
    uv = (x >= 3.3) & (x < 8.0)
    xu = x[uv]
    Fa = np.zeros_like(xu)
    Fb = np.zeros_like(xu)
    uv2 = xu >= 5.9
    Fa[uv2] = -0.04473 * (xu[uv2] - 5.9) ** 2 - 0.009779 * (xu[uv2] - 5.9) ** 3
    Fb[uv2] = 0.2130 * (xu[uv2] - 5.9) ** 2 + 0.1207 * (xu[uv2] - 5.9) ** 3
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + Fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + Fb

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

    return max(Av, 0.0), Av_err
