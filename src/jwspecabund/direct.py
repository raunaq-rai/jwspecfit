"""Direct T_e method for ionic and total abundances.

Uses PyNEB for the atomic physics: electron temperature from
auroral-to-nebular line ratios, electron density from density-sensitive
doublets, and ionic abundances via ``getIonAbundance()``.

References
----------
- DESI DR2 (arXiv:2601.02463) T_e-T_e relation
- Osterbrock & Ferland (2006) for Case B recombination
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _get_pyneb():
    """Import PyNEB lazily and return the module."""
    try:
        import pyneb as pn
    except ImportError as exc:
        raise ImportError(
            "PyNEB is required for the direct method. "
            "Install with: pip install 'jwspecfit[abund]'"
        ) from exc
    return pn


# ---------------------------------------------------------------------------
# Electron density
# ---------------------------------------------------------------------------

def compute_ne(
    flux_line1: float,
    flux_line2: float,
    doublet: str = "SII",
    Te_guess: float = 1e4,
) -> float:
    """Compute electron density from a density-sensitive doublet.

    Parameters
    ----------
    flux_line1 : float
        Flux of the blue member (e.g. [SII] 6718 or [OII] 3726).
    flux_line2 : float
        Flux of the red member (e.g. [SII] 6732 or [OII] 3729).
    doublet : str
        ``"SII"`` (default) or ``"OII"``.
    Te_guess : float
        Electron temperature guess in K (default 10^4).

    Returns
    -------
    float
        Electron density n_e in cm^-3.
    """
    pn = _get_pyneb()

    if flux_line2 <= 0:
        logger.warning("Denominator flux <= 0 for n_e; returning 100 cm^-3.")
        return 100.0

    ratio = flux_line1 / flux_line2

    if doublet == "SII":
        atom = pn.Atom("S", 2)
        ne = atom.getTemDen(
            ratio, tem=Te_guess, wave1=6718, wave2=6732
        )
    elif doublet == "OII":
        atom = pn.Atom("O", 2)
        ne = atom.getTemDen(
            ratio, tem=Te_guess, wave1=3726, wave2=3729
        )
    else:
        raise ValueError(f"Unknown doublet: {doublet!r}. Use 'SII' or 'OII'.")

    # PyNEB can return nan for extreme ratios; fall back to default.
    if np.isnan(ne) or ne <= 0:
        logger.warning("PyNEB returned invalid n_e=%.1f; using 100 cm^-3.", ne)
        return 100.0

    return float(ne)


# ---------------------------------------------------------------------------
# Electron temperature
# ---------------------------------------------------------------------------

def compute_Te_OIII(
    flux_4363: float,
    flux_5007: float,
    flux_4959: float,
    ne: float,
) -> float:
    """Compute T_e(O++) from the [OIII] auroral/nebular ratio.

    Uses PyNEB ``getTemDen()`` on the O++ atom with the standard
    diagnostic ratio [OIII] 4363 / ([OIII] 5007 + [OIII] 4959).

    Parameters
    ----------
    flux_4363 : float
        [OIII] 4363 flux.
    flux_5007 : float
        [OIII] 5007 flux.
    flux_4959 : float
        [OIII] 4959 flux.
    ne : float
        Electron density in cm^-3.

    Returns
    -------
    float
        T_e(O++) in K.
    """
    pn = _get_pyneb()

    nebular = flux_5007 + flux_4959
    if nebular <= 0:
        raise ValueError("[OIII] nebular flux (5007+4959) is non-positive.")
    if flux_4363 <= 0:
        raise ValueError("[OIII] 4363 flux is non-positive.")

    ratio = flux_4363 / nebular

    atom = pn.Atom("O", 3)
    # Use to_eval for the sum ratio, and log=True for robust root-finding
    # at high temperatures (T > 25,000 K, typical of metal-poor z > 4 galaxies).
    # start_x=3.0 → 1,000 K; end_x=5.0 → 100,000 K in log10(T) space.
    Te = atom.getTemDen(
        ratio, den=ne,
        to_eval="(L(4363))/(L(5007)+L(4959))",
        log=True, start_x=3.0, end_x=5.0,
    )

    if np.isnan(Te) or Te <= 0:
        raise ValueError(
            f"PyNEB returned invalid T_e for [OIII] 4363/(5007+4959)={ratio:.4f}. "
            f"Check that the auroral line is a real detection."
        )

    return float(Te)


def compute_Te_NII(
    flux_5756: float,
    flux_6585: float,
    ne: float,
) -> float:
    """Compute T_e(N+) from the [NII] auroral/nebular ratio.

    Parameters
    ----------
    flux_5756 : float
        [NII] 5756 flux.
    flux_6585 : float
        [NII] 6585 flux.
    ne : float
        Electron density in cm^-3.

    Returns
    -------
    float
        T_e(N+) in K.
    """
    pn = _get_pyneb()

    if flux_6585 <= 0:
        raise ValueError("[NII] 6585 flux is non-positive.")
    if flux_5756 <= 0:
        raise ValueError("[NII] 5756 flux is non-positive.")

    ratio = flux_5756 / flux_6585

    atom = pn.Atom("N", 2)
    # Use log=True for robust root-finding across a wide temperature range.
    Te = atom.getTemDen(
        ratio, den=ne, wave1=5755, wave2=6584,
        log=True, start_x=3.0, end_x=5.0,
    )

    if np.isnan(Te) or Te <= 0:
        raise ValueError(
            f"PyNEB returned invalid T_e for [NII] 5755/6584={ratio:.4f}."
        )

    return float(Te)


def Te_low_from_high(Te_high: float, relation: str = "desi") -> float:
    """Derive T_e(low) from T_e(high) using an empirical T_e-T_e relation.

    Parameters
    ----------
    Te_high : float
        T_e(O++) in K.
    relation : str
        ``"desi"`` (default) — DESI DR2 (arXiv:2601.02463):
        T_low = 0.648 * T_high + 3270
        ``"classical"`` — Garnett (1992):
        T_low = 0.7 * T_high + 3000

    Returns
    -------
    float
        T_e(low) in K.
    """
    if relation == "desi":
        return 0.648 * Te_high + 3270.0
    elif relation == "classical":
        return 0.7 * Te_high + 3000.0
    else:
        raise ValueError(f"Unknown T_e relation: {relation!r}. Use 'desi' or 'classical'.")


# ---------------------------------------------------------------------------
# Ionic abundances via PyNEB
# ---------------------------------------------------------------------------

# Maximum temperature where PyNEB's H I recombination tables are valid.
# Storey & Hummer (1995) tables in PyNEB go up to 30,000 K.
_PYNEB_HI_TMAX = 30000.0

# Power-law fit to the Hβ emissivity for extrapolation beyond 30,000 K.
# Fitted to PyNEB values at 10,000–30,000 K: ε_Hβ = 10^a × T^b.
# Accuracy < 1% within the fitted range; physically motivated for
# extrapolation up to ~60,000 K (Case B recombination scales as ~T^{-0.9}).
_HB_EMISS_LOGCOEFF = -21.178   # intercept (log10)
_HB_EMISS_TEXP = -0.932        # power-law exponent


def _hbeta_emissivity(Te: float, ne: float) -> float:
    """Return the Hβ volume emissivity, with extrapolation beyond 30,000 K.

    Uses PyNEB directly for T <= 30,000 K.  For higher temperatures,
    extrapolates with a power law fitted to Case B values.

    Parameters
    ----------
    Te : float
        Electron temperature in K.
    ne : float
        Electron density in cm^-3.

    Returns
    -------
    float
        Hβ emissivity (same units as PyNEB's ``RecAtom.getEmissivity``).
    """
    pn = _get_pyneb()
    if Te <= _PYNEB_HI_TMAX:
        H1 = pn.RecAtom("H", 1)
        return H1.getEmissivity(Te, ne, wave=4861)

    # Extrapolate using power law fitted to PyNEB values.
    return 10.0 ** (_HB_EMISS_LOGCOEFF + _HB_EMISS_TEXP * np.log10(Te))


def _ionic_abundance(
    element: str,
    ion: int,
    flux_line: float,
    flux_Hbeta: float,
    Te: float,
    ne: float,
    wave: int,
) -> float:
    """Compute an ionic abundance X^i+/H+ via PyNEB.

    For T <= 30,000 K, delegates to ``Atom.getIonAbundance()``.
    For T > 30,000 K, computes the abundance manually using the CEL
    emissivity from PyNEB and an extrapolated Hβ emissivity, since
    PyNEB's H I recombination tables only cover up to 30,000 K.

    Parameters
    ----------
    element : str
        Element symbol (e.g. ``"O"``, ``"N"``, ``"S"``).
    ion : int
        Ionisation stage (PyNEB convention: 2 = singly ionised, etc.).
    flux_line : float
        Emission-line flux.
    flux_Hbeta : float
        Hbeta flux (for normalisation).
    Te : float
        Electron temperature in K.
    ne : float
        Electron density in cm^-3.
    wave : int
        Approximate wavelength label for PyNEB (e.g. 5007, 6584).

    Returns
    -------
    float
        Ionic abundance X^i+/H+.
    """
    pn = _get_pyneb()

    if flux_Hbeta <= 0 or flux_line <= 0:
        return np.nan

    intensity = flux_line / flux_Hbeta * 100.0  # PyNEB convention: Hβ = 100

    atom = pn.Atom(element, ion)

    if Te <= _PYNEB_HI_TMAX:
        abund = atom.getIonAbundance(intensity, tem=Te, den=ne, wave=wave)
    else:
        # Manual computation: X^i+/H+ = (I_line/I_Hb) × (ε_Hb / ε_line)
        emiss_line = atom.getEmissivity(Te, ne, wave=wave)
        emiss_Hb = _hbeta_emissivity(Te, ne)
        if emiss_line > 0 and np.isfinite(emiss_Hb) and emiss_Hb > 0:
            abund = (intensity / 100.0) * (emiss_Hb / emiss_line)
        else:
            abund = np.nan

    if np.isnan(abund) or abund <= 0:
        return np.nan

    return float(abund)


def compute_ionic_abundances(
    fluxes: dict[str, float],
    Te_high: float,
    Te_low: float,
    ne: float,
) -> dict[str, float]:
    """Compute all available ionic abundances.

    Parameters
    ----------
    fluxes : dict
        Dust-corrected emission-line fluxes keyed by line name.
        Must include ``"HBETA"`` for normalisation.
    Te_high : float
        T_e(O++) in K.
    Te_low : float
        T_e(O+/N+) in K.
    ne : float
        Electron density in cm^-3.

    Returns
    -------
    dict
        Ionic abundances, e.g. ``{"O+/H+": val, "O++/H+": val, ...}``.
    """
    ionic = {}
    Hb = fluxes.get("HBETA", 0.0)
    if Hb <= 0:
        return ionic

    # O++/H+ from [OIII] 5007 — T_high zone
    if "OIII_5007" in fluxes and fluxes["OIII_5007"] > 0:
        ionic["O++/H+"] = _ionic_abundance("O", 3, fluxes["OIII_5007"], Hb, Te_high, ne, 5007)

    # O+/H+ from [OII] 3726+3729 — T_low zone
    oii = 0.0
    if "OII_3726" in fluxes and "OII_3729" in fluxes:
        oii = fluxes["OII_3726"] + fluxes["OII_3729"]
    elif "OII_doublet" in fluxes:
        oii = fluxes["OII_doublet"]
    if oii > 0:
        ionic["O+/H+"] = _ionic_abundance("O", 2, oii, Hb, Te_low, ne, 3726)

    # N+/H+ from [NII] 6585 — T_low zone
    if "NII_6585" in fluxes and fluxes["NII_6585"] > 0:
        ionic["N+/H+"] = _ionic_abundance("N", 2, fluxes["NII_6585"], Hb, Te_low, ne, 6584)

    # S+/H+ from [SII] 6718+6732 — T_low zone
    sii = 0.0
    if "SII_6718" in fluxes and "SII_6732" in fluxes:
        sii = fluxes["SII_6718"] + fluxes["SII_6732"]
    if sii > 0:
        ionic["S+/H+"] = _ionic_abundance("S", 2, sii, Hb, Te_low, ne, 6718)

    # S++/H+ from [SIII] 9069 — T_mid zone (use average of T_high, T_low)
    if "SIII_9069" in fluxes and fluxes["SIII_9069"] > 0:
        Te_mid = 0.5 * (Te_high + Te_low)
        ionic["S++/H+"] = _ionic_abundance("S", 3, fluxes["SIII_9069"], Hb, Te_mid, ne, 9069)

    # Ne++/H+ from [NeIII] 3869 — T_high zone
    if "NeIII_3869" in fluxes and fluxes["NeIII_3869"] > 0:
        ionic["Ne++/H+"] = _ionic_abundance("Ne", 3, fluxes["NeIII_3869"], Hb, Te_high, ne, 3869)

    # Ar++/H+ from [ArIII] 7136 — T_mid zone
    if "ArIII_7136" in fluxes and fluxes["ArIII_7136"] > 0:
        Te_mid = 0.5 * (Te_high + Te_low)
        ionic["Ar++/H+"] = _ionic_abundance("Ar", 3, fluxes["ArIII_7136"], Hb, Te_mid, ne, 7136)

    return ionic


def compute_total_abundances(
    ionic: dict[str, float],
) -> dict[str, float]:
    """Derive total element abundances from ionic abundances + ICFs.

    Parameters
    ----------
    ionic : dict
        Ionic abundance dict from :func:`compute_ionic_abundances`.

    Returns
    -------
    dict
        Total abundance ratios: ``"O/H"``, ``"N/O"``, ``"S/O"``,
        ``"Ne/O"``, ``"Ar/O"`` as available.
    """
    from .icf import icf_argon, icf_neon, icf_nitrogen, icf_sulfur

    totals: dict[str, float] = {}

    O_plus = ionic.get("O+/H+", 0.0)
    O_pp = ionic.get("O++/H+", 0.0)

    # O/H = O+/H+ + O++/H+ (no ICF needed)
    if O_plus > 0 or O_pp > 0:
        OH = O_plus + O_pp
        totals["O/H"] = OH

        # N/O
        N_plus = ionic.get("N+/H+", 0.0)
        if N_plus > 0 and O_plus > 0:
            icf_n = icf_nitrogen(O_plus, OH)
            totals["N/O"] = icf_n * N_plus / O_plus

        # S/O
        S_plus = ionic.get("S+/H+", 0.0)
        S_pp = ionic.get("S++/H+", 0.0)
        if S_plus > 0 or S_pp > 0:
            S_total_ion = S_plus + S_pp
            icf_s = icf_sulfur(O_plus, OH)
            totals["S/O"] = icf_s * S_total_ion / OH

        # Ne/O
        Ne_pp = ionic.get("Ne++/H+", 0.0)
        if Ne_pp > 0 and O_pp > 0:
            icf_ne = icf_neon(O_plus, OH)
            totals["Ne/O"] = icf_ne * Ne_pp / O_pp

        # Ar/O
        Ar_pp = ionic.get("Ar++/H+", 0.0)
        if Ar_pp > 0 and O_pp > 0:
            icf_ar = icf_argon(O_plus, OH)
            totals["Ar/O"] = icf_ar * Ar_pp / O_pp

    return totals
