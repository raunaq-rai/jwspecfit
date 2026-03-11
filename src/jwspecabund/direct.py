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

# Default electron density fallback (cm^-3).
# Matches the AURORA/EXCELS high-z median (~300-480 cm^-3 at z > 2).
NE_DEFAULT: float = 300.0


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
        raise ValueError(
            f"Denominator flux <= 0 for {doublet} density solve"
        )

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

    # PyNEB can return nan for extreme ratios.
    if np.isnan(ne) or ne <= 0:
        raise ValueError(
            f"PyNEB returned invalid n_e={ne:.1f} for {doublet} "
            f"ratio={ratio:.3f} (ratio out of valid range)"
        )

    return float(ne)


def compute_ne_CIII(
    flux_1907: float,
    flux_1909: float,
    Te_guess: float = 1e4,
) -> float:
    """Compute electron density from the CIII] 1907/1909 ratio.

    Probes the intermediate-ionisation zone.

    Parameters
    ----------
    flux_1907 : float
        CIII] 1907 flux.
    flux_1909 : float
        CIII] 1909 flux.
    Te_guess : float
        Electron temperature guess in K (default 10^4).

    Returns
    -------
    float
        Electron density n_e in cm^-3.
    """
    pn = _get_pyneb()

    if flux_1909 <= 0:
        raise ValueError("CIII] 1909 flux <= 0 for density solve")

    ratio = flux_1907 / flux_1909
    atom = pn.Atom("C", 3)
    ne = atom.getTemDen(ratio, tem=Te_guess, wave1=1907, wave2=1909)

    if np.isnan(ne) or ne <= 0:
        raise ValueError(
            f"PyNEB returned invalid n_e={ne:.1f} from CIII] "
            f"ratio={ratio:.3f} (ratio out of valid range)"
        )

    return float(ne)


def compute_ne_NIV(
    flux_1483: float,
    flux_1486: float,
    Te_guess: float = 1e4,
) -> float:
    """Compute electron density from the NIV] 1483/1486 ratio.

    Probes the high-ionisation zone.

    Parameters
    ----------
    flux_1483 : float
        NIV] 1483 flux.
    flux_1486 : float
        NIV] 1486 flux.
    Te_guess : float
        Electron temperature guess in K (default 10^4).

    Returns
    -------
    float
        Electron density n_e in cm^-3.
    """
    pn = _get_pyneb()

    if flux_1486 <= 0:
        raise ValueError("NIV] 1486 flux <= 0 for density solve")

    ratio = flux_1483 / flux_1486
    atom = pn.Atom("N", 4)
    ne = atom.getTemDen(ratio, tem=Te_guess, wave1=1483, wave2=1487)

    if np.isnan(ne) or ne <= 0:
        raise ValueError(
            f"PyNEB returned invalid n_e={ne:.1f} from NIV] "
            f"ratio={ratio:.3f} (ratio out of valid range)"
        )

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


def _hbeta_emissivity(Te: float, ne: float) -> float:
    """Return the Hbeta volume emissivity, using Aller (1984) beyond 30,000 K.

    Uses PyNEB directly for T <= 30,000 K.  For higher temperatures,
    uses the Aller (1984) formula which has no upper-bound limitation.

    Parameters
    ----------
    Te : float
        Electron temperature in K.
    ne : float
        Electron density in cm^-3.

    Returns
    -------
    float
        Hbeta emissivity (same units as PyNEB's ``RecAtom.getEmissivity``).
    """
    pn = _get_pyneb()
    if Te <= _PYNEB_HI_TMAX:
        H1 = pn.RecAtom("H", 1)
        return H1.getEmissivity(Te, ne, wave=4861)

    # Aller (1984) Case B formula, valid at any temperature.
    from .forward import hbeta_emissivity_aller84
    return hbeta_emissivity_aller84(Te)


def _ionic_abundance(
    element: str,
    ion: int,
    flux_line: float,
    flux_Hbeta: float,
    Te: float,
    ne: float,
    wave: int | list[int],
) -> float:
    """Compute an ionic abundance X^i+/H+ via PyNEB.

    For a single wavelength at T <= 30,000 K, delegates to
    ``Atom.getIonAbundance()``.  For T > 30,000 K or when *wave* is a
    list of wavelengths (doublet/multiplet), computes the abundance
    manually as (F_line/F_Hβ) × (ε_Hβ / Σε_line).

    Parameters
    ----------
    element : str
        Element symbol (e.g. ``"O"``, ``"N"``, ``"S"``).
    ion : int
        Ionisation stage (PyNEB convention: 2 = singly ionised, etc.).
    flux_line : float
        Emission-line flux.  When *wave* is a list this must be the
        **summed** flux of all components.
    flux_Hbeta : float
        Hbeta flux (for normalisation).
    Te : float
        Electron temperature in K.
    ne : float
        Electron density in cm^-3.
    wave : int or list[int]
        Wavelength label(s) for PyNEB (e.g. ``5007`` or ``[1907, 1909]``).
        Pass a list when *flux_line* is the combined doublet flux so that
        the total emissivity is used.

    Returns
    -------
    float
        Ionic abundance X^i+/H+.
    """
    pn = _get_pyneb()

    if flux_Hbeta <= 0 or flux_line <= 0:
        return np.nan

    atom = pn.Atom(element, ion)

    # Doublet / multiplet — always use the manual emissivity path
    # so that we sum ε over all components.
    if isinstance(wave, (list, tuple)):
        emiss_line = sum(atom.getEmissivity(Te, ne, wave=w) for w in wave)
        emiss_Hb = _hbeta_emissivity(Te, ne)
        if emiss_line > 0 and np.isfinite(emiss_Hb) and emiss_Hb > 0:
            abund = (flux_line / flux_Hbeta) * (emiss_Hb / emiss_line)
        else:
            abund = np.nan
    else:
        intensity = flux_line / flux_Hbeta * 100.0  # PyNEB convention: Hβ = 100
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
    ne_mid: float | None = None,
    ne_high: float | None = None,
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
        Electron density in cm^-3 (low-ionisation zone).
    ne_mid : float, optional
        Electron density for the intermediate-ionisation zone (cm^-3).
        Traced by CIII] 1907/1909 (~24 eV).  If ``None``, defaults to
        *ne*.  Used for C²⁺, N²⁺, S²⁺, Ar²⁺.
    ne_high : float, optional
        Electron density for the high-ionisation zone (cm^-3).
        Traced by NIV] 1483/1486 (~47 eV).  If ``None``, defaults to
        *ne_mid*.  Used for O²⁺, Ne²⁺, N³⁺, N⁴⁺, C³⁺.

    Returns
    -------
    dict
        Ionic abundances, e.g. ``{"O+/H+": val, "O++/H+": val, ...}``.
    """
    ionic = {}
    Hb = fluxes.get("HBETA", 0.0)
    if Hb <= 0:
        return ionic

    ne_lo = ne
    ne_md = ne_mid if ne_mid is not None else ne
    ne_hi = ne_high if ne_high is not None else ne_md

    # O++/H+ from [OIII] 5007 — T_high zone
    if "OIII_5007" in fluxes and fluxes["OIII_5007"] > 0:
        ionic["O++/H+"] = _ionic_abundance("O", 3, fluxes["OIII_5007"], Hb, Te_high, ne_hi, 5007)

    # O+/H+ from [OII] 3726+3729 — T_low zone
    oii = 0.0
    if "OII_3726" in fluxes and "OII_3729" in fluxes:
        oii = fluxes["OII_3726"] + fluxes["OII_3729"]
    elif "OII_doublet" in fluxes:
        oii = fluxes["OII_doublet"]
    if oii > 0:
        ionic["O+/H+"] = _ionic_abundance("O", 2, oii, Hb, Te_low, ne_lo, [3726, 3729])

    # N+/H+ from [NII] 6585 — T_low zone
    if "NII_6585" in fluxes and fluxes["NII_6585"] > 0:
        ionic["N+/H+"] = _ionic_abundance("N", 2, fluxes["NII_6585"], Hb, Te_low, ne_lo, 6584)

    # S+/H+ from [SII] 6718+6732 — T_low zone
    sii = 0.0
    if "SII_6718" in fluxes and "SII_6732" in fluxes:
        sii = fluxes["SII_6718"] + fluxes["SII_6732"]
    if sii > 0:
        ionic["S+/H+"] = _ionic_abundance("S", 2, sii, Hb, Te_low, ne_lo, [6718, 6732])

    # S++/H+ from [SIII] 9069 — T_mid zone (use average of T_high, T_low)
    if "SIII_9069" in fluxes and fluxes["SIII_9069"] > 0:
        Te_mid = 0.5 * (Te_high + Te_low)
        ionic["S++/H+"] = _ionic_abundance("S", 3, fluxes["SIII_9069"], Hb, Te_mid, ne_md, 9069)

    # Ne++/H+ from [NeIII] 3869 — T_high zone
    if "NeIII_3869" in fluxes and fluxes["NeIII_3869"] > 0:
        ionic["Ne++/H+"] = _ionic_abundance("Ne", 3, fluxes["NeIII_3869"], Hb, Te_high, ne_hi, 3869)

    # Ar++/H+ from [ArIII] 7136 — T_mid zone
    if "ArIII_7136" in fluxes and fluxes["ArIII_7136"] > 0:
        Te_mid = 0.5 * (Te_high + Te_low)
        ionic["Ar++/H+"] = _ionic_abundance("Ar", 3, fluxes["ArIII_7136"], Hb, Te_mid, ne_md, 7136)

    # --- UV ionic abundances (all use T_high zone) ---
    # For doublets: if both members are present, sum fluxes and use total
    # emissivity.  If only one member is present, use that member's flux
    # with its single-line emissivity to avoid underestimating the
    # abundance (the other member may have been excluded by SNR filtering).

    # C+/H+ from CII] 2324 + 2326 — low-ionisation zone
    # CII] 2326 is a 5-component multiplet (²P → ⁴P) spanning 2323–2329 Å.
    # The user fits 2 Gaussians that capture the full blended flux.
    # We must sum emissivities over ALL multiplet components so the
    # ionic abundance is correct (the 2 fitted lines are only ~13% of
    # the total emissivity).
    _CII_MULTIPLET_WAVES = [2323, 2325, 2326, 2327, 2328]
    _cii2324 = fluxes.get("CII]_2324", 0.0)
    _cii2326 = fluxes.get("CII]_2326", 0.0)
    if _cii2324 > 0 or _cii2326 > 0:
        cii_flux = _cii2324 + _cii2326
        ionic["C+/H+"] = _ionic_abundance(
            "C", 2, cii_flux, Hb, Te_low, ne_lo, _CII_MULTIPLET_WAVES,
        )

    # C2+/H+ from CIII] 1907 + 1909 — mid-ionisation zone
    _c1907 = fluxes.get("CIII]_1907", 0.0)
    _c1909 = fluxes.get("CIII]", 0.0)
    if _c1907 > 0 and _c1909 > 0:
        ionic["C++/H+"] = _ionic_abundance("C", 3, _c1907 + _c1909, Hb, Te_high, ne_md, [1907, 1909])
    elif _c1907 > 0:
        ionic["C++/H+"] = _ionic_abundance("C", 3, _c1907, Hb, Te_high, ne_md, 1907)
    elif _c1909 > 0:
        ionic["C++/H+"] = _ionic_abundance("C", 3, _c1909, Hb, Te_high, ne_md, 1909)

    # C3+/H+ from CIV 1548 + 1551 — T_high zone
    _civ1 = fluxes.get("CIV_1", 0.0)
    _civ2 = fluxes.get("CIV_2", 0.0)
    if _civ1 > 0 and _civ2 > 0:
        ionic["C+++/H+"] = _ionic_abundance("C", 4, _civ1 + _civ2, Hb, Te_high, ne_hi, [1548, 1551])
    elif _civ1 > 0:
        ionic["C+++/H+"] = _ionic_abundance("C", 4, _civ1, Hb, Te_high, ne_hi, 1548)
    elif _civ2 > 0:
        ionic["C+++/H+"] = _ionic_abundance("C", 4, _civ2, Hb, Te_high, ne_hi, 1551)

    # N2+/H+ from NIII] 1749 + 1752 — mid-ionisation zone
    _n1749 = fluxes.get("NIII_1749", 0.0)
    _n1752 = fluxes.get("NIII_1752", 0.0)
    if _n1749 > 0 and _n1752 > 0:
        ionic["N++/H+"] = _ionic_abundance("N", 3, _n1749 + _n1752, Hb, Te_high, ne_md, [1749, 1752])
    elif _n1749 > 0:
        ionic["N++/H+"] = _ionic_abundance("N", 3, _n1749, Hb, Te_high, ne_md, 1749)
    elif _n1752 > 0:
        ionic["N++/H+"] = _ionic_abundance("N", 3, _n1752, Hb, Te_high, ne_md, 1752)

    # N3+/H+ from NIV] 1483 + 1486 — T_high zone
    _n1483 = fluxes.get("NIV_1483", 0.0)
    _n1486 = fluxes.get("NIV_1486", 0.0)
    if _n1483 > 0 and _n1486 > 0:
        ionic["N+++/H+"] = _ionic_abundance("N", 4, _n1483 + _n1486, Hb, Te_high, ne_hi, [1483, 1486])
    elif _n1483 > 0:
        ionic["N+++/H+"] = _ionic_abundance("N", 4, _n1483, Hb, Te_high, ne_hi, 1483)
    elif _n1486 > 0:
        ionic["N+++/H+"] = _ionic_abundance("N", 4, _n1486, Hb, Te_high, ne_hi, 1486)

    # N4+/H+ from NV 1239 + 1243 — T_high zone (manual emissivity)
    _nv1 = fluxes.get("NV_1", 0.0)
    _nv2 = fluxes.get("NV_2", 0.0)
    if _nv1 > 0 or _nv2 > 0:
        from .forward import _nv_emissivity, hbeta_emissivity_aller84
        eps_Hb = _hbeta_emissivity(Te_high, ne_hi)
        if _nv1 > 0 and _nv2 > 0:
            nv_flux = _nv1 + _nv2
            eps_nv = _nv_emissivity(Te_high, ne_hi, 1239) + _nv_emissivity(Te_high, ne_hi, 1243)
        elif _nv1 > 0:
            nv_flux = _nv1
            eps_nv = _nv_emissivity(Te_high, ne_hi, 1239)
        else:
            nv_flux = _nv2
            eps_nv = _nv_emissivity(Te_high, ne_hi, 1243)
        if eps_nv > 0 and eps_Hb > 0:
            ionic["N4+/H+"] = (nv_flux / Hb) * (eps_Hb / eps_nv)

    return ionic


def _print_icf_reasoning(
    ionic: dict[str, float],
    logU: float | None,
    Z_Zsun: float | None,
    icf_method: str,
    use_martinez: bool,
    totals: dict[str, float],
    ionic_upper_limits: dict[str, float] | None = None,
) -> None:
    """Print the full N/O ICF decision breakdown (auto mode only)."""
    _ul = ionic_upper_limits or {}
    print("\n" + "=" * 60)
    print("N/O ICF REASONING (icf_method='auto')")
    print("=" * 60)

    # 1. Detected nitrogen ions
    n_ions = {
        "N+/H+": ionic.get("N+/H+", 0.0),
        "N++/H+": ionic.get("N++/H+", 0.0),
        "N+++/H+": ionic.get("N+++/H+", 0.0),
        "N4+/H+": ionic.get("N4+/H+", 0.0),
    }
    print("\n--- Detected nitrogen ions ---")
    for key, val in n_ions.items():
        if val > 0:
            status = f"{val:.4e}"
        elif key in _ul:
            status = f"< {_ul[key]:.4e}  (3σ upper limit)"
        else:
            status = "not detected"
        print(f"  {key}: {status}")

    # 2. Detected oxygen ions
    print("\n--- Detected oxygen ions ---")
    for key in ("O+/H+", "O++/H+"):
        val = ionic.get(key, 0.0)
        status = f"{val:.4e}" if val > 0 else "not detected"
        print(f"  {key}: {status}")

    # 3. Available diagnostics
    print("\n--- Available diagnostics ---")
    print(f"  logU:   {logU}" if logU is not None else "  logU:   not available")
    print(f"  Z/Zsun: {Z_Zsun}" if Z_Zsun is not None else "  Z/Zsun: not available")

    # 4. Requested method
    print(f"\n--- Requested method: '{icf_method}' ---")

    # 5. Eligibility check
    print("\n--- Method eligibility ---")
    has_logU_Z = logU is not None and Z_Zsun is not None
    N_plus = n_ions["N+/H+"]
    N_pp = n_ions["N++/H+"]
    N_ppp = n_ions["N+++/H+"]
    O_plus = ionic.get("O+/H+", 0.0)
    O_pp = ionic.get("O++/H+", 0.0)

    # martinez25
    if has_logU_Z:
        print("  martinez25: ELIGIBLE (logU and Z/Zsun available)")
    else:
        missing = []
        if logU is None:
            missing.append("logU")
        if Z_Zsun is None:
            missing.append("Z/Zsun")
        print(f"  martinez25: NOT ELIGIBLE (missing {', '.join(missing)})")

    # direct_sum tiers
    if N_plus > 0 and (N_pp + N_ppp) > 0:
        print("  direct_sum (Tier 1 — Np+Npp+Nppp/OH): ELIGIBLE")
    else:
        print("  direct_sum (Tier 1 — Np+Npp+Nppp/OH): NOT ELIGIBLE"
              " (need N+ and N++ or N+++)")
    if (N_pp + N_ppp) > 0 and O_pp > 0:
        print("  direct_sum (Tier 2/3 — UV N/O++): ELIGIBLE")
    else:
        print("  direct_sum (Tier 2/3 — UV N/O++): NOT ELIGIBLE"
              " (need N++ or N+++ and O++)")
    if N_plus > 0 and O_plus > 0:
        print("  izotov06 (Tier 4 — N+/O+ fallback): ELIGIBLE")
    else:
        print("  izotov06 (Tier 4 — N+/O+ fallback): NOT ELIGIBLE"
              " (need N+ and O+)")

    # 6. Final selection
    print("\n--- Final selection ---")
    if "N/O" in totals:
        method = totals.get("icf_method", "unknown")
        icf_name = totals.get("NO_icf_name", "N/A")
        print(f"  Chosen method:  {method}")
        print(f"  ICF name:       {icf_name}")
        print(f"  N/O value:      {totals['N/O']:.4e}")
    else:
        print("  N/O: could not be computed (no eligible method with detected ions)")

    print("=" * 60 + "\n")


def compute_total_abundances(
    ionic: dict[str, float],
    logU: float | None = None,
    Z_Zsun: float | None = None,
    ne: float | None = None,
    icf_method: str = "auto",
    ionic_upper_limits: dict[str, float] | None = None,
) -> dict[str, float]:
    """Derive total element abundances from ionic abundances + ICFs.

    Parameters
    ----------
    ionic : dict
        Ionic abundance dict from :func:`compute_ionic_abundances`.
    logU : float, optional
        Ionisation parameter log(U).  Required for Martinez+25 ICFs.
    Z_Zsun : float, optional
        Gas-phase metallicity in solar units.  Required for Martinez+25 ICFs.
    ne : float, optional
        Electron density in cm^-3 for Martinez+25 ICF density interpolation.
    icf_method : str
        ``"auto"`` (default): use Martinez+25 for N/O when logU is provided,
        fall back to Izotov+06 otherwise.
        ``"martinez25"``: force Martinez+25 ICFs (requires logU and Z_Zsun).
        ``"izotov06"``: use Izotov+06 ICFs only.
        ``"direct_sum"``: sum all detected nitrogen ions directly
        (Topping+2024, Yanagisawa+2025, Cameron+2023).  Tiered fallback:
        Tier 1 (N⁺ + N²⁺ + N³⁺) / (O⁺ + O²⁺),
        Tier 2/3 (N²⁺ + N³⁺) / O²⁺, Tier 4 Izotov+06 optical fallback.

    Returns
    -------
    dict
        Total abundance ratios: ``"O/H"``, ``"N/O"``, ``"S/O"``,
        ``"Ne/O"``, ``"Ar/O"``, ``"C/O"``, ``"N/O_UV"`` as available.
        When Martinez+25 is used, also includes ``"NO_icf_name"`` and
        ``"icf_method"`` keys.
    """
    from .icf import icf_argon, icf_carbon, icf_neon, icf_nitrogen, icf_sulfur

    totals: dict[str, float] = {}
    failures: dict[str, str] = {}

    O_plus = ionic.get("O+/H+", 0.0)
    O_pp = ionic.get("O++/H+", 0.0)

    # Decide whether to use Martinez+25 for N/O.
    use_martinez = False
    if icf_method == "martinez25":
        if logU is None or Z_Zsun is None:
            logger.warning("Martinez+25 ICFs require logU and Z_Zsun; falling back to Izotov+06.")
        else:
            use_martinez = True
    elif icf_method == "auto" and logU is not None and Z_Zsun is not None:
        use_martinez = True

    # O/H = O+/H+ + O++/H+ (no ICF needed)
    if O_plus > 0 or O_pp > 0:
        OH = O_plus + O_pp
        totals["O/H"] = OH

        # N/O — direct_sum: sum all detected nitrogen ions (no ICF/logU).
        if icf_method == "direct_sum":
            N_plus = ionic.get("N+/H+", 0.0)
            N_pp = ionic.get("N++/H+", 0.0)
            N_ppp = ionic.get("N+++/H+", 0.0)
            N_pppp = ionic.get("N4+/H+", 0.0)

            if N_plus > 0 and (N_pp + N_ppp) > 0:
                # Tier 1: all zones — Topping+2024
                totals["N/O"] = (N_plus + N_pp + N_ppp + N_pppp) / OH
                totals["icf_method"] = "direct_sum"
                totals["NO_icf_name"] = "Np_Npp_Nppp"
            elif (N_pp + N_ppp) > 0 and O_pp > 0:
                # Tier 2/3: UV only — Yanagisawa+25 / Cameron+23
                totals["N/O"] = (N_pp + N_ppp + N_pppp) / O_pp
                totals["icf_method"] = "direct_sum"
                if N_pp > 0 and N_ppp > 0:
                    totals["NO_icf_name"] = "Npp_Nppp_Opp"
                elif N_ppp > 0:
                    totals["NO_icf_name"] = "Nppp_Opp"
                else:
                    totals["NO_icf_name"] = "Npp_Opp"
            elif N_plus > 0 and O_plus > 0:
                # Tier 4: optical fallback — Izotov+06
                icf_n = icf_nitrogen(O_plus, OH)
                totals["N/O"] = icf_n * N_plus / O_plus
                totals["icf_method"] = "izotov06"
                totals["NO_icf_name"] = "izotov06_fallback"

        # N/O — Martinez+25 with direct_sum fallback
        elif use_martinez:
            from .martinez25_icf import compute_NO_martinez25
            ne_icf = ne if ne is not None else NE_DEFAULT
            NO_val, NO_icf_name = compute_NO_martinez25(ionic, logU, Z_Zsun, ne_icf)
            if NO_val is not None:
                totals["N/O"] = NO_val
                totals["NO_icf_name"] = NO_icf_name
                totals["icf_method"] = "martinez25"
            else:
                # Fall back to direct_sum tiers if Martinez+25 has no
                # suitable ionic ratios (e.g. nitrogen ions SNR-gated).
                N_plus = ionic.get("N+/H+", 0.0)
                N_pp = ionic.get("N++/H+", 0.0)
                N_ppp = ionic.get("N+++/H+", 0.0)
                N_pppp = ionic.get("N4+/H+", 0.0)
                if N_plus > 0 and (N_pp + N_ppp) > 0:
                    totals["N/O"] = (N_plus + N_pp + N_ppp + N_pppp) / OH
                    totals["icf_method"] = "direct_sum"
                    totals["NO_icf_name"] = "Np_Npp_Nppp"
                elif (N_pp + N_ppp) > 0 and O_pp > 0:
                    totals["N/O"] = (N_pp + N_ppp + N_pppp) / O_pp
                    totals["icf_method"] = "direct_sum"
                    if N_pp > 0 and N_ppp > 0:
                        totals["NO_icf_name"] = "Npp_Nppp_Opp"
                    elif N_ppp > 0:
                        totals["NO_icf_name"] = "Nppp_Opp"
                    else:
                        totals["NO_icf_name"] = "Npp_Opp"
                elif N_plus > 0 and O_plus > 0:
                    icf_n = icf_nitrogen(O_plus, OH)
                    totals["N/O"] = icf_n * N_plus / O_plus
                    totals["icf_method"] = "izotov06"
                    totals["NO_icf_name"] = "izotov06_fallback"
        else:
            N_plus = ionic.get("N+/H+", 0.0)
            if N_plus > 0 and O_plus > 0:
                icf_n = icf_nitrogen(O_plus, OH)
                totals["N/O"] = icf_n * N_plus / O_plus
                totals["icf_method"] = "izotov06"

        # Record N/O failure reason if not computed.
        if "N/O" not in totals:
            _n_ions = [k for k in ("N+/H+", "N++/H+", "N+++/H+", "N4+/H+")
                       if ionic.get(k, 0.0) > 0]
            _o_ions = [k for k in ("O+/H+", "O++/H+")
                       if ionic.get(k, 0.0) > 0]
            if not _n_ions:
                failures["N/O"] = "no nitrogen ions detected"
            elif icf_method in ("izotov06", "auto") and not use_martinez:
                failures["N/O"] = (
                    f"Izotov+06 requires N+ and O+; have {_n_ions} and {_o_ions}"
                )
            else:
                failures["N/O"] = (
                    f"no eligible ICF tier; detected N ions: {_n_ions}, "
                    f"O ions: {_o_ions}"
                )

        # Compute ALL eligible N/O tiers for comparison.
        # This mirrors Berg+2025 Section 4.3.2 (Eqs 2–5) which reports
        # individual-ion ICF results alongside the direct sum.
        NO_tiers: dict[str, float] = {}
        N_plus = ionic.get("N+/H+", 0.0)
        N_pp = ionic.get("N++/H+", 0.0)
        N_ppp = ionic.get("N+++/H+", 0.0)
        N_pppp = ionic.get("N4+/H+", 0.0)

        # Martinez+25 — compute ALL individual ICFs (not just the priority pick)
        if logU is not None and Z_Zsun is not None:
            from .martinez25_icf import (
                icf_NpOp, icf_NppOpp, icf_NpppOpp,
                icf_NpNpp_OpOpp, icf_NppNppp_Opp,
            )
            ne_icf = ne if ne is not None else NE_DEFAULT

            # ICF 1: N+/O+ × ICF  (Berg+2025 Eq. 2)
            if N_plus > 0 and O_plus > 0:
                icf1 = icf_NpOp(logU, Z_Zsun, ne_icf)
                val1 = (N_plus / O_plus) * icf1
                if val1 > 0:
                    NO_tiers["ICF 1: N⁺/O⁺ × ICF (Martinez+25)"] = np.log10(val1)
                    NO_tiers["_icf1_value"] = icf1

            # ICF 2: N²⁺/O²⁺ × ICF  (Berg+2025 Eq. 3)
            if N_pp > 0 and O_pp > 0:
                icf2 = icf_NppOpp(logU, Z_Zsun, ne_icf)
                val2 = (N_pp / O_pp) * icf2
                if val2 > 0:
                    NO_tiers["ICF 2: N²⁺/O²⁺ × ICF (Martinez+25)"] = np.log10(val2)
                    NO_tiers["_icf2_value"] = icf2

            # ICF 3: N³⁺/O²⁺ × ICF  (Berg+2025 Eq. 4)
            if N_ppp > 0 and O_pp > 0:
                icf3 = icf_NpppOpp(logU, Z_Zsun, ne_icf)
                val3 = (N_ppp / O_pp) * icf3
                if val3 > 0:
                    NO_tiers["ICF 3: N³⁺/O²⁺ × ICF (Martinez+25)"] = np.log10(val3)
                    NO_tiers["_icf3_value"] = icf3

            # ICF 4: (N⁺+N²⁺)/(O⁺+O²⁺) × ICF  (Martinez+25 Table 4)
            if N_plus > 0 and N_pp > 0 and O_plus > 0 and O_pp > 0:
                icf4 = icf_NpNpp_OpOpp(logU, Z_Zsun, ne_icf)
                val4 = ((N_plus + N_pp) / (O_plus + O_pp)) * icf4
                if val4 > 0:
                    NO_tiers["ICF 4: (N⁺+N²⁺)/(O⁺+O²⁺) × ICF (Martinez+25)"] = np.log10(val4)
                    NO_tiers["_icf4_value"] = icf4

            # ICF 5: (N²⁺+N³⁺)/O²⁺ × ICF  (Martinez+25 Table 4, recommended)
            if N_pp > 0 and N_ppp > 0 and O_pp > 0:
                icf5 = icf_NppNppp_Opp(logU, Z_Zsun, ne_icf)
                val5 = ((N_pp + N_ppp) / O_pp) * icf5
                if val5 > 0:
                    NO_tiers["ICF 5: (N²⁺+N³⁺)/O²⁺ × ICF (Martinez+25)"] = np.log10(val5)
                    NO_tiers["_icf5_value"] = icf5

        # Direct sum: (N⁺ + N²⁺ + N³⁺) / (O⁺ + O²⁺)  (Berg+2025 Eq. 5)
        if N_plus > 0 and (N_pp + N_ppp) > 0:
            t1 = (N_plus + N_pp + N_ppp + N_pppp) / OH
            if t1 > 0:
                NO_tiers["Direct sum: (N⁺+N²⁺+N³⁺)/(O⁺+O²⁺)"] = np.log10(t1)

        # UV-only direct sum: (N²⁺ + N³⁺) / O²⁺
        if (N_pp + N_ppp) > 0 and O_pp > 0:
            t23 = (N_pp + N_ppp + N_pppp) / O_pp
            if t23 > 0:
                NO_tiers["Direct sum UV: (N²⁺+N³⁺)/O²⁺"] = np.log10(t23)

        # Izotov+06: ICF × N⁺/O⁺
        if N_plus > 0 and O_plus > 0:
            icf_n_all = icf_nitrogen(O_plus, OH)
            t4 = icf_n_all * N_plus / O_plus
            if t4 > 0:
                NO_tiers["Izotov+06: ICF × N⁺/O⁺"] = np.log10(t4)
                NO_tiers["_izotov06_icf_value"] = icf_n_all

        if NO_tiers:
            totals["_NO_tiers"] = NO_tiers

        # Print ICF reasoning when auto mode is used.
        if icf_method == "auto":
            _print_icf_reasoning(ionic, logU, Z_Zsun, icf_method,
                                 use_martinez, totals,
                                 ionic_upper_limits=ionic_upper_limits)

        # S/O
        S_plus = ionic.get("S+/H+", 0.0)
        S_pp = ionic.get("S++/H+", 0.0)
        if S_plus > 0 or S_pp > 0:
            S_total_ion = S_plus + S_pp
            icf_s = icf_sulfur(O_plus, OH)
            totals["S/O"] = icf_s * S_total_ion / OH
        else:
            failures["S/O"] = "no S+ or S++ ions detected ([SII]/[SIII] missing)"

        # Ne/O
        Ne_pp = ionic.get("Ne++/H+", 0.0)
        if Ne_pp > 0 and O_pp > 0:
            icf_ne = icf_neon(O_plus, OH)
            totals["Ne/O"] = icf_ne * Ne_pp / O_pp
        elif Ne_pp <= 0:
            failures["Ne/O"] = "no Ne++ ion detected ([NeIII] 3869 missing)"
        else:
            failures["Ne/O"] = "no O++ ion for Ne/O normalisation"

        # Ar/O
        Ar_pp = ionic.get("Ar++/H+", 0.0)
        if Ar_pp > 0 and O_pp > 0:
            icf_ar = icf_argon(O_plus, OH)
            totals["Ar/O"] = icf_ar * Ar_pp / O_pp
        elif Ar_pp <= 0:
            failures["Ar/O"] = "no Ar++ ion detected ([ArIII] 7136 missing)"
        else:
            failures["Ar/O"] = "no O++ ion for Ar/O normalisation"

        # C/O — tiered approach:
        #   1. Direct sum (C+ + C2+ + C3+) / (O+ + O2+) when C+ detected
        #   2. Garnett+1997 ICF × (C2+ + C3+) / O2+ when C+ not detected
        #   3. Raw (C2+ + C3+) / O2+ when O+ also missing (ICF=1)
        C_p = ionic.get("C+/H+", 0.0)
        C_pp = ionic.get("C++/H+", 0.0)
        C_ppp = ionic.get("C+++/H+", 0.0)
        C_uv = C_pp + C_ppp
        if C_p > 0 and C_uv > 0 and OH > 0:
            # Direct sum — all C ions detected, use total O
            totals["C/O"] = (C_p + C_uv) / OH
            totals["CO_method"] = "direct_sum"
        elif C_uv > 0 and O_pp > 0:
            # Apply Garnett+1997 ICF to correct for missing C+
            icf_c = icf_carbon(O_plus, O_pp)
            totals["C/O"] = icf_c * C_uv / O_pp
            totals["CO_method"] = "garnett97_icf"
            totals["CO_icf_value"] = icf_c
        elif C_uv <= 0 and C_p <= 0:
            failures["C/O"] = "no carbon ions detected (CII]/CIII]/CIV missing)"
        else:
            failures["C/O"] = "no O++ ion for C/O normalisation"

        # UV N/O — raw ionic sum without ICF (for comparison).
        N_pp = ionic.get("N++/H+", 0.0)
        N_ppp = ionic.get("N+++/H+", 0.0)
        N_pppp = ionic.get("N4+/H+", 0.0)
        N_uv = N_pp + N_ppp + N_pppp
        if N_uv > 0 and O_pp > 0:
            totals["N/O_UV_raw"] = N_uv / O_pp

    totals["_failures"] = failures
    return totals
