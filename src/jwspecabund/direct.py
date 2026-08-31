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
from dataclasses import dataclass, field
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


#: Collision-strength dataset used *only* for the O III] 1666 temperature.
#: PyNEB's default (``o_iii_coll_SSB14.dat``) tabulates just 5 levels, so the
#: 5-S-2 upper level of lambda1661,1666 is out of range and lambda1666 cannot
#: be evaluated at all.  Tayal & Zatsarinny (2017) covers level 6 and shifts
#: the lambda4363-based T_e by only +137 K (0.8%) and log(O++/H+) by
#: +0.002 dex relative to SSB14, so adopting it here leaves the default
#: lambda4363 path untouched at negligible cost in consistency.
OIII_1666_COLL_FILE = "o_iii_coll_TZ17.dat"

_OIII_ATOMS: dict[str, Any] = {}


def _get_oiii_atom(coll_file: str = OIII_1666_COLL_FILE):
    """Return a 6-level O III atom capable of evaluating O III] 1666.

    Builds the atom against *coll_file*, then restores PyNEB's global
    collision-file selection so that every other calculation in the package
    (notably :func:`compute_Te_OIII`, which uses lambda4363) keeps the SSB14
    default.  PyNEB resolves data files at construction time, so the returned
    instance retains *coll_file* after the restore.  Atoms are cached per
    collision file because the switch is global state and must happen once.

    Parameters
    ----------
    coll_file : str
        PyNEB O III collision-strength file.  Must tabulate at least 6
        levels; ``o_iii_coll_SSB14.dat`` (PyNEB's default) does not.
        Valid 6-level choices are ``o_iii_coll_TZ17.dat`` (the package
        default, :data:`OIII_1666_COLL_FILE`), ``o_iii_coll_AK99.dat``
        (Aggarwal & Keenan 1999, used by Hsiao+2026) and
        ``o_iii_coll_MBZ20.dat``.

    Returns
    -------
    pyneb.Atom
        O III atom with at least 6 levels.

    Raises
    ------
    RuntimeError
        If the resulting atom still has fewer than 6 levels, which would
        silently mis-identify the lambda1666 transition.
    """
    cached = _OIII_ATOMS.get(coll_file)
    if cached is not None:
        return cached

    pn = _get_pyneb()
    prev = pn.atomicData.getDataFile("O3", "coll")
    try:
        pn.atomicData.setDataFile(coll_file)
        # NLevels=6 is a *performance* constraint, not a physical one: the
        # atom file caps the model at 6 levels either way, but without this
        # PyNEB re-interpolates TZ17's full 202-level collision array on
        # every getEmissivity call — 203 ms vs 0.23 ms, an 880x penalty that
        # makes each brentq T_e solve take seconds.  Emissivity ratios are
        # bit-identical with and without it.
        atom = pn.Atom("O", 3, NLevels=6)
    finally:
        if prev is not None:
            pn.atomicData.setDataFile(prev)

    if atom.NLevels < 6:
        raise RuntimeError(
            f"O III collision data {coll_file!r} yielded "
            f"{atom.NLevels} levels; O III] 1666 (the 6->3 transition) "
            f"needs at least 6."
        )
    _OIII_ATOMS[coll_file] = atom
    return atom


def _get_oiii_atom_1666():
    """Return the package-default 6-level O III atom (:data:`OIII_1666_COLL_FILE`).

    Thin wrapper kept for backwards compatibility; see :func:`_get_oiii_atom`.
    """
    return _get_oiii_atom(OIII_1666_COLL_FILE)


# ---------------------------------------------------------------------------
# Electron density
# ---------------------------------------------------------------------------

def compute_ne(
    flux_line1: float,
    flux_line2: float,
    doublet: str = "SII",
    Te_guess: float = 1.5e4,
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
    Te_guess: float = 1.5e4,
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


def ciii_ratio_at_density(
    ne: float,
    Te: float = 1e4,
) -> float:
    """Predict the CIII] 1909/1907 flux ratio at a given electron density.

    Useful for fixing the CIII] doublet ratio in the fitter when the
    density is assumed or known from another diagnostic.

    Parameters
    ----------
    ne : float
        Electron density in cm^-3.
    Te : float
        Electron temperature in K (default 10^4).

    Returns
    -------
    float
        Expected F(1909) / F(1907) ratio.
    """
    pn = _get_pyneb()
    atom = pn.Atom("C", 3)
    emiss_1909 = atom.getEmissivity(Te, ne, wave=1909)
    emiss_1907 = atom.getEmissivity(Te, ne, wave=1907)
    if emiss_1907 <= 0:
        raise ValueError(
            f"PyNEB returned zero emissivity for CIII] 1907 at "
            f"n_e={ne:.0f}, T_e={Te:.0f}"
        )
    return float(emiss_1909 / emiss_1907)


def niv_ratio_at_density(
    ne: float,
    Te: float = 1e4,
) -> float:
    """Predict the NIV] 1483/1486 flux ratio at a given electron density.

    Useful for fixing the NIV doublet ratio in the fitter when the
    density is known from another diagnostic (e.g. CIII]).

    Parameters
    ----------
    ne : float
        Electron density in cm^-3.
    Te : float
        Electron temperature in K (default 10^4).

    Returns
    -------
    float
        Expected F(1483) / F(1486) ratio.
    """
    pn = _get_pyneb()
    atom = pn.Atom("N", 4)
    emiss_1483 = atom.getEmissivity(Te, ne, wave=1483)
    emiss_1486 = atom.getEmissivity(Te, ne, wave=1487)
    if emiss_1486 <= 0:
        raise ValueError(
            f"PyNEB returned zero emissivity for NIV] 1486 at "
            f"n_e={ne:.0f}, T_e={Te:.0f}"
        )
    return float(emiss_1483 / emiss_1486)


def compute_ne_NIV(
    flux_1483: float,
    flux_1486: float,
    Te_guess: float = 1.5e4,
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


def compute_ne_SiIII(
    flux_1883: float,
    flux_1892: float,
    Te_guess: float = 1.5e4,
) -> float:
    """Compute electron density from the [Si III] 1883/1892 ratio.

    A UV density diagnostic (Si²⁺, 16-33 eV).  Used as the low-ionisation
    density fallback for O⁺/N⁺ when the optical [SII]/[OII] doublets are
    out of coverage (e.g. high-z UV-only stacks; Martinez+2025 Table 2
    list n_e(Si²⁺) for the low/intermediate zone).

    Parameters
    ----------
    flux_1883 : float
        [Si III] 1883 flux.
    flux_1892 : float
        Si III] 1892 flux.
    Te_guess : float
        Electron temperature guess in K (default 1.5e4).

    Returns
    -------
    float
        Electron density n_e in cm^-3.
    """
    pn = _get_pyneb()

    if flux_1892 <= 0:
        raise ValueError("Si III] 1892 flux <= 0 for density solve")

    ratio = flux_1883 / flux_1892
    atom = pn.Atom("Si", 3)
    ne = atom.getTemDen(ratio, tem=Te_guess, wave1=1883, wave2=1892)

    if np.isnan(ne) or ne <= 0:
        raise ValueError(
            f"PyNEB returned invalid n_e={ne:.1f} from [Si III] "
            f"ratio={ratio:.3f} (ratio out of valid range)"
        )

    return float(ne)


def compute_ne_ArIV(
    flux_4711: float,
    flux_4740: float,
    Te_guess: float = 1.5e4,
) -> float:
    """Compute electron density from the [Ar IV] 4711/4740 ratio.

    Probes the high-ionisation zone (Ar³⁺, 40.7-59.8 eV), overlapping the
    O²⁺ zone, so it is the preferred density for O²⁺ (Martinez+2025
    Table 2).  ``flux_4711`` must be the He I-deblended [Ar IV] 4711 flux
    (see :func:`heI_4714_over_4472`); the raw fitted ``ArIV_4713`` line is
    blended with He I 4714.

    Parameters
    ----------
    flux_4711 : float
        Deblended [Ar IV] 4711 flux.
    flux_4740 : float
        [Ar IV] 4740 flux.
    Te_guess : float
        Electron temperature guess in K (default 1.5e4).

    Returns
    -------
    float
        Electron density n_e in cm^-3.
    """
    pn = _get_pyneb()

    if flux_4740 <= 0:
        raise ValueError("[Ar IV] 4740 flux <= 0 for density solve")
    if flux_4711 <= 0:
        raise ValueError("[Ar IV] 4711 (deblended) flux <= 0 for density solve")

    ratio = flux_4711 / flux_4740
    atom = pn.Atom("Ar", 4)
    ne = atom.getTemDen(ratio, tem=Te_guess, wave1=4711, wave2=4740)

    if np.isnan(ne) or ne <= 0:
        raise ValueError(
            f"PyNEB returned invalid n_e={ne:.1f} from [Ar IV] "
            f"ratio={ratio:.3f} (ratio out of valid range)"
        )

    return float(ne)


def heI_4714_over_4472(Te: float, ne: float) -> float:
    """Predict the He I 4714 / He I 4472 recombination flux ratio.

    Used to deblend the [Ar IV] 4711 line from the He I 4714 line that
    falls in the same fitted feature (``ArIV_4713``).  The He I 4714
    contribution is estimated as ``ratio * F(HEI_4472)`` and subtracted:

        F([Ar IV] 4711) = F(ArIV_4713) - ratio * F(HEI_4472).

    The ratio is computed from PyNEB He I recombination emissivities
    (Storey & Hummer 1995) rather than hard-coded, so it tracks the
    assumed T_e and n_e (~0.11 at 10⁴ K to ~0.18 at 2x10⁴ K).

    Parameters
    ----------
    Te : float
        Electron temperature in K.
    ne : float
        Electron density in cm^-3.

    Returns
    -------
    float
        Expected F(He I 4714) / F(He I 4472).
    """
    pn = _get_pyneb()
    He1 = pn.RecAtom("He", 1)
    # PyNEB He I labels: 4471 (the 4472 anchor) and 4713 (the 4714 line).
    Te_eval = min(Te, _PYNEB_HI_TMAX) if Te > 0 else 1.5e4
    e_4714 = He1.getEmissivity(Te_eval, ne, wave=4713)
    e_4472 = He1.getEmissivity(Te_eval, ne, wave=4471)
    if e_4472 <= 0 or not np.isfinite(e_4714):
        raise ValueError(
            f"PyNEB returned invalid He I emissivity at T_e={Te:.0f}, n_e={ne:.0f}"
        )
    return float(e_4714 / e_4472)


# ---------------------------------------------------------------------------
# z-dependent electron-density fallbacks
# ---------------------------------------------------------------------------

# Redshift-evolution fits for the per-zone electron density, used when a
# zone's density-sensitive doublet is unavailable or fails the SNR/solve.
# Form: n_e = A * (1 + z)^p.
#
# - mid / high: Martinez+2025 (arXiv:2510.21960) Eqs (4) and (5):
#     ne,int  = 1.11e3 * (1 + z)^1.93   (intermediate-ionisation zone)
#     ne,high = 5.40e3 * (1 + z)^1.62   (high-ionisation zone)
# - low: Abdurro'uf+2024 (arXiv:2404.16201, ApJ 973, 47) fit the [O II]
#   3726/3729 low-ionisation density out to z ~ 10 as
#     ne,low = 54 (+31/-23) * (1 + z)^(1.2 +/- 0.4).
_NE_ZONE_FALLBACK: dict[str, tuple[float, float]] = {
    "low": (54.0, 1.2),
    "mid": (1110.0, 1.93),
    "high": (5400.0, 1.62),
}


def ne_zone_fallback(zone: str, z: float) -> float:
    """Return the redshift-dependent electron-density fallback for a zone.

    Parameters
    ----------
    zone : str
        ``"low"``, ``"mid"`` or ``"high"``.
    z : float
        Source redshift.

    Returns
    -------
    float
        Fallback electron density in cm^-3.
    """
    if zone not in _NE_ZONE_FALLBACK:
        raise ValueError(
            f"Unknown density zone: {zone!r}. Use 'low', 'mid' or 'high'."
        )
    A, p = _NE_ZONE_FALLBACK[zone]
    return float(A * (1.0 + z) ** p)


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


def compute_Te_OIII_1666(
    flux_1666: float,
    flux_5007: float,
    flux_4959: float,
    ne: float,
) -> float:
    """Compute T_e(O++) from the [OIII] UV/optical ratio 1666/(5007+4959).

    Uses the O III] 1666 Å intercombination line as a UV auroral
    diagnostic when [OIII] 4363 is unavailable or low-SNR.  The line is
    the 6→3 transition (⁵S₂ → ¹D₂), so it requires a 6-level O III atom;
    see :func:`_get_oiii_atom_1666`.  The emissivity ratio
    1666/(5007+4959) increases monotonically with T_e and is more
    temperature-sensitive than 4363/(5007+4959): λ1666 is excited from a
    level 7.48 eV above ground versus 5.35 eV for λ4363, against the
    2.51 eV λ5007 denominator.

    Notes
    -----
    This diagnostic uses the Tayal & Zatsarinny (2017) collision strengths
    (:data:`OIII_1666_COLL_FILE`) rather than PyNEB's SSB14 default, which
    stops at 5 levels and cannot represent λ1666 at all.  The λ4363 path
    (:func:`compute_Te_OIII`) is unaffected and keeps SSB14.  Because the
    two diagnostics are mutually exclusive per object, no single
    measurement mixes datasets; the residual inconsistency is that a
    λ1666-derived T_e is fed to SSB14 emissivities downstream, worth
    ~0.01 dex in log(O⁺⁺/H⁺).  Expect a ~400 K (~2.4%) atomic-data
    systematic on any λ1666 temperature, from the spread between
    published 6-level collision datasets.

    Parameters
    ----------
    flux_1666 : float
        O III] 1666 flux (dust-corrected).
    flux_5007 : float
        [OIII] 5007 flux (dust-corrected).
    flux_4959 : float
        [OIII] 4959 flux (dust-corrected).
    ne : float
        Electron density in cm^-3.

    Returns
    -------
    float
        T_e(O++) in K.
    """
    from scipy.optimize import brentq

    nebular = flux_5007 + flux_4959
    if nebular <= 0:
        raise ValueError("[OIII] nebular flux (5007+4959) is non-positive.")
    if flux_1666 <= 0:
        raise ValueError("O III] 1666 flux is non-positive.")

    observed_ratio = flux_1666 / nebular

    atom = _get_oiii_atom_1666()

    def _ratio_minus_obs(log_Te: float) -> float:
        Te = 10.0 ** log_Te
        e_1666 = atom.getEmissivity(Te, ne, lev_i=6, lev_j=3)
        e_5007 = atom.getEmissivity(Te, ne, wave=5007)
        e_4959 = atom.getEmissivity(Te, ne, wave=4959)
        if e_5007 + e_4959 <= 0:
            return -observed_ratio
        return e_1666 / (e_5007 + e_4959) - observed_ratio

    try:
        log_Te = brentq(_ratio_minus_obs, 3.5, 5.4, xtol=1e-6)
        Te = 10.0 ** log_Te
    except ValueError:
        # Check if the ratio exceeds the physical maximum.
        max_ratio = -_ratio_minus_obs(5.4) + observed_ratio
        if observed_ratio > max_ratio:
            raise ValueError(
                f"O III] 1666/(5007+4959)={observed_ratio:.6f} exceeds the "
                f"physical maximum ({max_ratio:.6f}) at ne={ne:.0f} cm^-3. "
                f"This can happen when dust correction over-inflates the UV "
                f"flux relative to optical. Consider using [OIII] 4363 instead."
            )
        raise ValueError(
            f"Could not solve T_e for O III] 1666/(5007+4959)={observed_ratio:.6f}. "
            f"Ratio may be outside the valid temperature range (3000–250000 K)."
        )

    if np.isnan(Te) or Te <= 0:
        raise ValueError(
            f"Invalid T_e from O III] 1666/(5007+4959)={observed_ratio:.6f}."
        )

    return float(Te)


# ---------------------------------------------------------------------------
# Self-consistent T_e and n_e in the O++ zone (Hsiao et al. 2026)
# ---------------------------------------------------------------------------

#: Temperature axis of the O III model grid, as ``(log10 T_min, log10 T_max,
#: n_points)``.  The 50,000 K ceiling is what makes the curve inversion
#: well-posed: every ratio below is strictly monotonic in T_e at *every*
#: density on the grid, whereas lambda4363/(lambda5007+lambda4959) turns over
#: above log T_e ~ 4.9 at n_e > 10^6 cm^-3 and would admit two roots.
OIII_GRID_LOGTE: tuple[float, float, int] = (3.6, 4.7, 441)

#: Density grid the curve intersection is searched on: Hsiao et al. (2026)
#: section IV.3 specify log(n_e/cm^-3) = 0-7 with 1,000 steps.
OIII_GRID_LOGNE: tuple[float, float, int] = (0.0, 7.0, 1000)

#: Density nodes actually evaluated with PyNEB before interpolating up to
#: :data:`OIII_GRID_LOGNE`.  Calling PyNEB at all 1,000 nodes costs ~12 s;
#: 141 nodes (0.05 dex) costs ~1.7 s and cubic interpolation in
#: log-emissivity reproduces the direct build to 3e-7 relative, which leaves
#: the recovered (T_e, n_e) bit-identical.  Guarded by a test.
OIII_GRID_LOGNE_NODES: int = 141

_OIII_GRIDS: dict[tuple[str, int], dict[str, Any]] = {}


def _oiii_te_ne_grid(
    coll_file: str = OIII_1666_COLL_FILE,
    *,
    n_nodes: int = OIII_GRID_LOGNE_NODES,
) -> dict[str, Any]:
    """Return the cached O III emissivity/ratio grid used by the joint solve.

    Tabulates the lambda1666, lambda4363, lambda5007 and lambda4959
    emissivities of a *single* 6-level O III atom over
    :data:`OIII_GRID_LOGTE` x :data:`OIII_GRID_LOGNE`.  Using one atom for
    all four lines is what makes the solve self-consistent: mixing collision
    datasets would intersect curves drawn from different atomic physics.

    Parameters
    ----------
    coll_file : str
        O III collision-strength file; see :func:`_get_oiii_atom`.
    n_nodes : int
        Number of density nodes evaluated with PyNEB before interpolating
        onto the full :data:`OIII_GRID_LOGNE` grid.  Defaults to
        :data:`OIII_GRID_LOGNE_NODES`; pass ``OIII_GRID_LOGNE[2]`` to build
        the grid directly with no interpolation.

    Returns
    -------
    dict
        ``logTe`` (n_T,), ``logne`` (n_n,), ``log_emis`` (dict of (n_T, n_n)
        arrays keyed by line name) and the three log10 line-ratio grids
        ``log_R_4363``, ``log_R_1666_4363`` and ``log_R_5007_1666``.

    Notes
    -----
    The grid is built lazily on first use and cached per
    ``(coll_file, n_nodes)`` for the lifetime of the process.
    """
    key = (coll_file, int(n_nodes))
    cached = _OIII_GRIDS.get(key)
    if cached is not None:
        return cached

    from scipy.interpolate import interp1d

    atom = _get_oiii_atom(coll_file)

    logTe = np.linspace(*OIII_GRID_LOGTE)
    logne = np.linspace(*OIII_GRID_LOGNE)
    Te = 10.0 ** logTe

    # PyNEB is called with a scalar density and the full temperature vector:
    # passing both as arrays makes it build an outer product internally and
    # blows up memory well before the grid is covered.
    nodes = np.linspace(OIII_GRID_LOGNE[0], OIII_GRID_LOGNE[1], int(n_nodes))
    _kwargs = {
        "1666": {"lev_i": 6, "lev_j": 3},
        "4363": {"wave": 4363},
        "5007": {"wave": 5007},
        "4959": {"wave": 4959},
    }
    logger.info(
        "Building O III T_e-n_e grid (%s, %d density nodes); "
        "this takes a few seconds and is cached.", coll_file, len(nodes),
    )
    raw = {k: np.empty((logTe.size, nodes.size)) for k in _kwargs}
    for j, ln in enumerate(nodes):
        ne = 10.0 ** ln
        for name, kw in _kwargs.items():
            raw[name][:, j] = atom.getEmissivity(Te, ne, **kw)

    if nodes.size == logne.size and np.allclose(nodes, logne):
        log_emis = {k: np.log10(v) for k, v in raw.items()}
    else:
        # Interpolate in log-emissivity: the emissivities span many decades
        # but are near-linear in log-log, so cubic interpolation here is far
        # more accurate than the same spline applied to the raw values.
        log_emis = {
            k: interp1d(nodes, np.log10(v), axis=1, kind="cubic")(logne)
            for k, v in raw.items()
        }

    e1666 = 10.0 ** log_emis["1666"]
    e4363 = 10.0 ** log_emis["4363"]
    e_neb = 10.0 ** log_emis["5007"] + 10.0 ** log_emis["4959"]

    grid = {
        "coll_file": coll_file,
        "logTe": logTe,
        "logne": logne,
        "log_emis": log_emis,
        # lambda4363/(lambda5007+lambda4959) -- the classical auroral ratio.
        # Hsiao et al. write it the other way up; the constraint is the same.
        "log_R_4363": np.log10(e4363 / e_neb),
        # O III] lambda1666/[O III] lambda4363 -- the UV/auroral ratio that
        # breaks the T_e-n_e degeneracy.
        "log_R_1666_4363": np.log10(e1666 / e4363),
        # [O III] lambda5007/O III] lambda1666 -- shown by Hsiao et al. but
        # not independent: it is the ratio of the other two.
        "log_R_5007_1666": np.log10(e_neb / e1666),
    }
    _OIII_GRIDS[key] = grid
    return grid


def _te_curve(log_ratio: np.ndarray, log_obs: float, logTe: np.ndarray) -> np.ndarray:
    """Invert a monotone-in-T_e ratio grid for log T_e at every density.

    Parameters
    ----------
    log_ratio : ndarray, shape (n_T, n_n)
        log10 model ratio, strictly increasing down the temperature axis.
    log_obs : float
        log10 of the observed ratio.
    logTe : ndarray, shape (n_T,)
        Temperature axis.

    Returns
    -------
    ndarray, shape (n_n,)
        log T_e reproducing *log_obs* at each density, or NaN where the
        observed ratio lies outside the model range at that density.
    """
    n_T = log_ratio.shape[0]
    idx = np.sum(log_ratio < log_obs, axis=0)
    out = np.full(log_ratio.shape[1], np.nan)
    j = np.flatnonzero((idx > 0) & (idx < n_T))
    if j.size == 0:
        return out
    i1 = idx[j]
    i0 = i1 - 1
    y0 = log_ratio[i0, j]
    y1 = log_ratio[i1, j]
    w = (log_obs - y0) / (y1 - y0)
    out[j] = logTe[i0] + w * (logTe[i1] - logTe[i0])
    return out


def _intersect_curves(
    logne: np.ndarray, T1: np.ndarray, T2: np.ndarray,
) -> tuple[float, float, bool] | None:
    """Locate the crossing of two T_e(n_e) curves.

    Follows Hsiao et al. (2026): the solution is the intersection of the
    curves, and "when no converged result is found, we determine the best
    n_e and T_e as the closest values".  The crossing is taken at the
    *global* minimum of ``|T1 - T2|`` rather than the first sign change:
    where the curves touch tangentially, or where sub-grid wiggles
    manufacture a spurious crossing in the density-insensitive regime, the
    first sign change picks the wrong root.

    Returns
    -------
    tuple or None
        ``(log n_e, log T_e, converged)``, or ``None`` if neither ratio is
        reproducible anywhere on the grid.  *converged* is ``True`` only for
        a genuine crossing; ``False`` marks the "closest values" fallback.
    """
    d = T1 - T2
    ok = np.isfinite(d)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return None

    k = int(idx[np.argmin(np.abs(d[idx]))])
    if d[k] == 0.0:
        return float(logne[k]), float(T1[k]), True

    for a, b in ((k - 1, k), (k, k + 1)):
        if a < 0 or b >= d.size or not (ok[a] and ok[b]):
            continue
        if d[a] * d[b] < 0:
            w = d[a] / (d[a] - d[b])
            return (
                float(logne[a] + w * (logne[b] - logne[a])),
                float(T1[a] + w * (T1[b] - T1[a])),
                True,
            )

    return float(logne[k]), float(T1[k]), False


def _ne_sensitivity_floor(
    grid: dict[str, Any], log_Te: float, sigma_log_ratio: float,
) -> float:
    """Lowest log n_e the data can actually distinguish from zero density.

    [O III] lambda4363/(lambda5007+lambda4959) is density-flat until
    lambda5007 starts to be collisionally de-excited, so below some density
    the observed ratio is consistent with *any* lower density and n_e is
    bounded only from above.  That threshold is not a fixed number: it
    depends on how precisely the ratio is measured.  This returns the
    smallest grid density at which the model ratio has moved away from its
    low-density limit by more than the measurement error.

    Parameters
    ----------
    grid : dict
        Model grid from :func:`_oiii_te_ne_grid`.
    log_Te : float
        log10 of the solution temperature.
    sigma_log_ratio : float
        1 sigma error on log10 of the observed auroral/nebular ratio.

    Returns
    -------
    float
        log10(n_e/cm^-3) below which the ratio carries no density
        information; the top of the grid if it never does.
    """
    logTe = grid["logTe"]
    i = int(np.argmin(np.abs(logTe - log_Te)))
    row = grid["log_R_4363"][i, :]
    moved = np.flatnonzero(np.abs(row - row[0]) > max(sigma_log_ratio, 0.0))
    if moved.size == 0:
        return float(grid["logne"][-1])
    return float(grid["logne"][moved[0]])


@dataclass
class SelfConsistentOIII:
    """Joint O++ temperature and density from UV + optical oxygen lines.

    Output of :func:`compute_Te_ne_OIII`.

    Attributes
    ----------
    Te : float
        T_e(O++) in K at the curve intersection.
    ne : float
        n_e(O++) in cm^-3 at the curve intersection.
    converged : bool
        ``True`` if the T_e(n_e) curves genuinely cross; ``False`` if the
        reported values are the "closest approach" fallback.
    Te_intersection, ne_intersection : float or None
        The raw curve-intersection solution for the unperturbed fluxes.
        Equal to *Te* / *ne* when no posterior was run; otherwise the
        posterior median is adopted instead (see :func:`compute_Te_ne_OIII`).
    Te_err, ne_err : tuple of float or None
        ``(lo, hi)`` 68 % CI half-widths from the flux posterior, or
        ``None`` if no flux errors were supplied.
    ne_is_upper_limit : bool
        ``True`` when the posterior density runs into the bottom of the
        grid, i.e. a low-density solution is allowed at 1 sigma and n_e is
        only bounded from above.
    ne_upper_limit : float or None
        The 1 sigma (84th percentile) upper limit on n_e when
        *ne_is_upper_limit* is set.
    converged_fraction : float or None
        Fraction of posterior draws that produced a genuine crossing.
    at_grid_edge : bool
        ``True`` if the solution sits on the boundary of the model grid, so
        the true value may lie outside it.
    coll_file : str
        O III collision-strength file the solve used.
    logne_grid, Te_curve_4363, Te_curve_1666_4363, Te_curve_5007_1666 : ndarray
        The three T_e(n_e) curves of Hsiao et al. figure 3, for plotting.
    Te_posterior, ne_posterior : ndarray or None
        Per-draw solutions, when flux errors were supplied.
    """

    Te: float
    ne: float
    converged: bool
    Te_intersection: float | None = None
    ne_intersection: float | None = None
    Te_err: tuple[float, float] | None = None
    ne_err: tuple[float, float] | None = None
    ne_is_upper_limit: bool = False
    ne_upper_limit: float | None = None
    converged_fraction: float | None = None
    at_grid_edge: bool = False
    coll_file: str = OIII_1666_COLL_FILE
    logne_grid: np.ndarray | None = field(default=None, repr=False)
    Te_curve_4363: np.ndarray | None = field(default=None, repr=False)
    Te_curve_1666_4363: np.ndarray | None = field(default=None, repr=False)
    Te_curve_5007_1666: np.ndarray | None = field(default=None, repr=False)
    Te_posterior: np.ndarray | None = field(default=None, repr=False)
    ne_posterior: np.ndarray | None = field(default=None, repr=False)


def _solve_oiii_once(
    grid: dict[str, Any],
    f1666: float,
    f4363: float,
    f_neb: float,
) -> tuple[float, float, bool, np.ndarray, np.ndarray, np.ndarray] | None:
    """Solve one (T_e, n_e) intersection and return it with the three curves."""
    if f1666 <= 0 or f4363 <= 0 or f_neb <= 0:
        return None

    logTe = grid["logTe"]
    T_4363 = _te_curve(grid["log_R_4363"], np.log10(f4363 / f_neb), logTe)
    T_1666_4363 = _te_curve(
        grid["log_R_1666_4363"], np.log10(f1666 / f4363), logTe,
    )
    T_5007_1666 = _te_curve(
        grid["log_R_5007_1666"], np.log10(f_neb / f1666), logTe,
    )

    hit = _intersect_curves(grid["logne"], T_4363, T_1666_4363)
    if hit is None:
        return None
    log_ne, log_Te, converged = hit
    return log_Te, log_ne, converged, T_4363, T_1666_4363, T_5007_1666


def compute_Te_ne_OIII(
    flux_1666: float,
    flux_4363: float,
    flux_5007: float,
    flux_4959: float = 0.0,
    *,
    err_1666: float | None = None,
    err_4363: float | None = None,
    err_5007: float | None = None,
    err_4959: float | None = None,
    coll_file: str = OIII_1666_COLL_FILE,
    n_draws: int = 500,
    seed: int = 42,
) -> SelfConsistentOIII:
    """Solve T_e and n_e simultaneously in the O++ zone (Hsiao et al. 2026).

    [O III] lambda5007 has a critical density of only ~7e5 cm^-3, so above
    n_e ~ 1e5 cm^-3 it is collisionally de-excited and the classical
    lambda4363/lambda5007 ratio depends on *both* T_e and n_e.  Solving it
    at an assumed low density then overestimates T_e and underestimates
    O/H -- by up to 1.1 dex in the Hsiao et al. sample.  O III] lambda1666
    has a far higher critical density, so adding it gives two independent
    ratios for the two unknowns and breaks the degeneracy.

    Following Hsiao et al. (2026) section IV.3, T_e(n_e) curves are built
    for each ratio over log(n_e/cm^-3) = 0-7 with 1,000 steps, and the
    solution is their intersection; when the curves do not cross, the
    closest approach is used instead.  Uncertainties come from the flux
    posterior, as in the paper.

    Parameters
    ----------
    flux_1666 : float
        O III] 1666 flux (dust-corrected).
    flux_4363 : float
        [O III] 4363 flux (dust-corrected).
    flux_5007 : float
        [O III] 5007 flux (dust-corrected).
    flux_4959 : float
        [O III] 4959 flux (dust-corrected).  Optional; the nebular
        denominator is ``flux_5007 + flux_4959``.
    err_1666, err_4363, err_5007, err_4959 : float or None
        1 sigma flux errors.  When *err_1666*, *err_4363* and *err_5007*
        are all given and positive, a *n_draws* posterior is run and the
        error bars, upper-limit flag and convergence fraction are filled in.
    coll_file : str
        O III collision-strength file; see :func:`_get_oiii_atom`.  The
        default is the package's :data:`OIII_1666_COLL_FILE` (TZ17);
        ``"o_iii_coll_AK99.dat"`` reproduces Hsiao et al. exactly and
        shifts T_e by 1-2 % and log(O++/H+) by ~0.02 dex.
    n_draws : int
        Posterior draws for the uncertainties (default 500).
    seed : int
        RNG seed for the posterior draws.

    Returns
    -------
    SelfConsistentOIII
        The joint solution, its uncertainties and the three T_e(n_e) curves.

    Raises
    ------
    ValueError
        If a flux is non-positive, or the observed ratios cannot be
        reproduced anywhere on the model grid.

    References
    ----------
    Hsiao et al. (2026), arXiv:2608.20339; the method is due to Berg (2018)
    and Arellano-Cordova et al. (2020).
    """
    f_neb = float(flux_5007) + float(flux_4959)
    if f_neb <= 0:
        raise ValueError("[OIII] nebular flux (5007+4959) is non-positive.")
    if flux_4363 <= 0:
        raise ValueError("[OIII] 4363 flux is non-positive.")
    if flux_1666 <= 0:
        raise ValueError("O III] 1666 flux is non-positive.")

    grid = _oiii_te_ne_grid(coll_file)
    solved = _solve_oiii_once(grid, float(flux_1666), float(flux_4363), f_neb)
    if solved is None:
        raise ValueError(
            f"No self-consistent (T_e, n_e) reproduces "
            f"[OIII] 4363/(5007+4959) = {flux_4363 / f_neb:.6g} and "
            f"O III] 1666/[OIII] 4363 = {flux_1666 / flux_4363:.6g} anywhere on "
            f"log T_e = {OIII_GRID_LOGTE[0]}-{OIII_GRID_LOGTE[1]}, "
            f"log n_e = {OIII_GRID_LOGNE[0]}-{OIII_GRID_LOGNE[1]}. "
            f"Check the dust correction: the UV/optical ratio is the most "
            f"reddening-sensitive of the three."
        )
    log_Te, log_ne, converged, c4363, c1666, c5007 = solved

    logTe_ax, logne_ax = grid["logTe"], grid["logne"]
    step_T = logTe_ax[1] - logTe_ax[0]
    step_n = logne_ax[1] - logne_ax[0]
    at_edge = bool(
        log_Te <= logTe_ax[0] + step_T or log_Te >= logTe_ax[-1] - step_T
        or log_ne <= logne_ax[0] + step_n or log_ne >= logne_ax[-1] - step_n
    )

    out = SelfConsistentOIII(
        Te=float(10.0 ** log_Te),
        ne=float(10.0 ** log_ne),
        converged=converged,
        Te_intersection=float(10.0 ** log_Te),
        ne_intersection=float(10.0 ** log_ne),
        at_grid_edge=at_edge,
        coll_file=coll_file,
        logne_grid=logne_ax,
        Te_curve_4363=10.0 ** c4363,
        Te_curve_1666_4363=10.0 ** c1666,
        Te_curve_5007_1666=10.0 ** c5007,
    )

    errs = (err_1666, err_4363, err_5007)
    if not all(e is not None and e > 0 for e in errs):
        return out

    rng = np.random.default_rng(seed)
    e4959 = err_4959 if (err_4959 is not None and err_4959 > 0) else 0.0
    lTe_d, lne_d = [], []
    n_conv = 0
    for _ in range(int(n_draws)):
        d1666 = rng.normal(flux_1666, err_1666)
        d4363 = rng.normal(flux_4363, err_4363)
        dneb = rng.normal(flux_5007, err_5007) + rng.normal(flux_4959, e4959)
        hit = _solve_oiii_once(grid, d1666, d4363, dneb)
        if hit is None:
            continue
        lTe_d.append(hit[0])
        lne_d.append(hit[1])
        n_conv += bool(hit[2])

    if not lTe_d:
        return out

    lTe_d = np.asarray(lTe_d)
    lne_d = np.asarray(lne_d)
    out.Te_posterior = 10.0 ** lTe_d
    out.ne_posterior = 10.0 ** lne_d
    out.converged_fraction = n_conv / len(lTe_d)

    tlo, tmed, thi = np.percentile(out.Te_posterior, [16, 50, 84])
    out.Te_err = (float(tmed - tlo), float(thi - tmed))
    nlo, nmed, nhi = np.percentile(out.ne_posterior, [16, 50, 84])
    out.ne_err = (float(nmed - nlo), float(nhi - nmed))

    # Adopt the posterior median rather than the intersection of the
    # unperturbed curves.  In the density-sensitive regime the two agree to
    # well under a grid step; in the flat regime the single intersection is
    # placed by sub-grid noise and the median is far more stable (recovering
    # log n_e = 4.26 against a truth of 4.48 where the raw crossing lands at
    # 1.44).  This is also what Hsiao et al. mean by propagating through the
    # posterior samples.
    out.Te, out.ne = float(tmed), float(nmed)
    log_Te_med = float(np.log10(tmed))

    # Below the density at which lambda5007 starts to be collisionally
    # de-excited, the auroral/nebular ratio is flat and any lower density
    # fits equally well -- n_e is then bounded only from above.  Compare the
    # 1 sigma lower bound of the posterior against that floor, computed at
    # the measured precision of the ratio.  Hsiao et al. report exactly such
    # a limit for their non-converged object.
    sig_neb = float(np.hypot(err_5007, e4959))
    sigma_log_ratio = float(
        np.hypot(err_4363 / flux_4363, sig_neb / f_neb) / np.log(10.0)
    )
    floor = _ne_sensitivity_floor(grid, log_Te_med, sigma_log_ratio)
    if np.percentile(lne_d, 16) < floor:
        out.ne_is_upper_limit = True
        out.ne_upper_limit = float(nhi)

    return out



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


def Te_low_from_high(Te_high: float, relation: str = "3_tier") -> float:
    """Derive T_e(low) from T_e(high) using an empirical T_e-T_e relation.

    The low-ionisation zone traces O⁺/N⁺ (14.5-29.6 eV).

    Parameters
    ----------
    Te_high : float
        T_e(O++) in K.
    relation : str
        ``"3_tier"`` (default) — Garnett (1992) O⁺ zone, as adopted by
        Martinez+2025 (arXiv:2510.21960): T_low = 0.70 * T_high + 3000.
        Pairs with :func:`Te_int_from_high` for the intermediate zone so
        the zones stay monotone (T_high >= T_int >= T_low).
        ``"classical"`` / ``"garnett"`` — alias of the Garnett (1992) low
        relation (identical to ``"3_tier"`` for the low zone).
        ``"desi"`` — DESI DR2 (arXiv:2601.02463):
        T_low = 0.648 * T_high + 3270.

    Returns
    -------
    float
        T_e(low) in K.
    """
    if relation in ("3_tier", "classical", "garnett"):
        return 0.7 * Te_high + 3000.0
    elif relation == "desi":
        return 0.648 * Te_high + 3270.0
    else:
        raise ValueError(
            f"Unknown T_e relation: {relation!r}. "
            "Use '3_tier', 'classical', 'garnett', or 'desi'."
        )


def Te_int_from_high(Te_high: float, relation: str = "3_tier") -> float | None:
    """Derive T_e(intermediate) from T_e(high) for the S²⁺ zone.

    The intermediate-ionisation zone traces S²⁺ (23.3-34.8 eV) and is
    used for S²⁺ and Ar²⁺.  Garnett (1992) gives t(S III) = 0.83 t(O III)
    + 0.17 (in 10⁴ K), i.e. T_int = 0.83 * T_high + 1700.  This is the
    relation adopted by Martinez+2025 (arXiv:2510.21960) for the
    intermediate zone.

    Parameters
    ----------
    Te_high : float
        T_e(O++) in K.
    relation : str
        ``"3_tier"`` (default), ``"classical"`` or ``"garnett"`` return the
        Garnett (1992) S III relation.  Any other relation (e.g. ``"desi"``)
        returns ``None`` — no intermediate relation is defined, and callers
        fall back to the T_high/T_low midpoint as before.

    Returns
    -------
    float or None
        T_e(intermediate) in K, or ``None`` if the relation defines no
        intermediate zone.
    """
    if relation in ("3_tier", "classical", "garnett"):
        return 0.83 * Te_high + 1700.0
    return None


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
    Te_int: float | None = None,
    ne_Opp: float | None = None,
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
        *ne*.  Used for O²⁺, Ne²⁺, C²⁺, N²⁺, S²⁺, Ar²⁺.  O²⁺ and Ne²⁺
        use this intermediate-zone density (not *ne_high*): the
        [OIII] 5007/Hβ and [NeIII] 3869/Hβ abundances are density-
        insensitive below ~10⁴–10⁵ cm⁻³, and CIII] (24–48 eV) overlaps
        the O²⁺ zone (35–55 eV), whereas NIV] (47–77 eV) traces more
        highly-ionised gas.  This decouples O²⁺/Ne²⁺ from the noisy
        high-ionisation N IV] density.
    ne_high : float, optional
        Electron density for the high-ionisation zone (cm^-3).
        Traced by NIV] 1483/1486 (~47 eV).  If ``None``, defaults to
        *ne_mid*.  Used for N³⁺, N⁴⁺, C³⁺.
    Te_int : float, optional
        Intermediate-zone (S²⁺) electron temperature in K, used for S²⁺
        and Ar²⁺.  If ``None`` (default), the legacy ``0.5*(Te_high +
        Te_low)`` midpoint is used.
    ne_Opp : float, optional
        Electron density for the O²⁺/Ne²⁺ zone (cm^-3), preferentially
        from [Ar IV] 4711/4740 (Martinez+2025 Table 2).  If ``None``
        (default), falls back to *ne_mid* (CIII] → *ne*), preserving the
        previous behaviour.

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
    # O²⁺/Ne²⁺ zone density: [Ar IV] when available, else the CIII]
    # intermediate density, else the (z-dependent) value supplied as ne_mid;
    # always a valid number when the caller passes ne_Opp.
    ne_opp = ne_Opp if ne_Opp is not None else (ne_mid if ne_mid is not None else ne)
    # Intermediate-zone density for C²⁺/N²⁺/S²⁺/Ar²⁺: CIII] when measured,
    # otherwise fall back to the O²⁺-zone density (which carries the
    # z-dependent intermediate fallback) rather than the low-ionisation ne.
    ne_md = ne_mid if ne_mid is not None else ne_opp
    ne_hi = ne_high if ne_high is not None else ne_opp
    # Intermediate-zone (S²⁺) temperature: Garnett relation when supplied,
    # else the legacy T_high/T_low midpoint.
    Te_mid = Te_int if Te_int is not None else 0.5 * (Te_high + Te_low)

    # O++/H+ from [OIII] 5007 — T_high zone, O²⁺-zone density.
    # Prefers ne_opp ([Ar IV], high-ionisation, overlaps O²⁺); falls back
    # to ne_md (CIII]→low).  5007/Hβ is density-insensitive below ~10⁴ cm⁻³.
    if "OIII_5007" in fluxes and fluxes["OIII_5007"] > 0:
        ionic["O++/H+"] = _ionic_abundance("O", 3, fluxes["OIII_5007"], Hb, Te_high, ne_opp, 5007)

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

    # S++/H+ from [SIII] 9069 — intermediate (S²⁺) zone temperature
    if "SIII_9069" in fluxes and fluxes["SIII_9069"] > 0:
        ionic["S++/H+"] = _ionic_abundance("S", 3, fluxes["SIII_9069"], Hb, Te_mid, ne_md, 9069)

    # Ne++/H+ from [NeIII] 3869 — T_high zone, O²⁺-zone density.
    # Like O²⁺, [NeIII] 3869 is density-insensitive below ~10⁵ cm⁻³ and Ne²⁺
    # overlaps the O²⁺ zone, so it uses ne_opp ([Ar IV] → CIII] → low).
    if "NeIII_3869" in fluxes and fluxes["NeIII_3869"] > 0:
        ionic["Ne++/H+"] = _ionic_abundance("Ne", 3, fluxes["NeIII_3869"], Hb, Te_high, ne_opp, 3869)

    # Ar++/H+ from [ArIII] 7136 — intermediate (S²⁺) zone temperature
    if "ArIII_7136" in fluxes and fluxes["ArIII_7136"] > 0:
        ionic["Ar++/H+"] = _ionic_abundance("Ar", 3, fluxes["ArIII_7136"], Hb, Te_mid, ne_md, 7136)

    # --- UV ionic abundances ---
    # Zone assignment follows Martinez+2025 (arXiv:2510.21960, §5.4/6.1):
    # the intermediate-ionisation ions C²⁺ and N²⁺ use the intermediate
    # temperature (Te_mid) and density (ne_md); the high-ionisation ions
    # C³⁺, N³⁺, N⁴⁺ use Te_high and ne_hi.
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

    # C2+/H+ from CIII] 1907 + 1909 — intermediate zone (Te_mid, ne_md)
    _c1907 = fluxes.get("CIII]_1907", 0.0)
    _c1909 = fluxes.get("CIII]", 0.0)
    if _c1907 > 0 and _c1909 > 0:
        ionic["C++/H+"] = _ionic_abundance("C", 3, _c1907 + _c1909, Hb, Te_mid, ne_md, [1907, 1909])
    elif _c1907 > 0:
        ionic["C++/H+"] = _ionic_abundance("C", 3, _c1907, Hb, Te_mid, ne_md, 1907)
    elif _c1909 > 0:
        ionic["C++/H+"] = _ionic_abundance("C", 3, _c1909, Hb, Te_mid, ne_md, 1909)

    # C3+/H+ from CIV 1548 + 1551 — T_high zone
    _civ1 = fluxes.get("CIV_1", 0.0)
    _civ2 = fluxes.get("CIV_2", 0.0)
    if _civ1 > 0 and _civ2 > 0:
        ionic["C+++/H+"] = _ionic_abundance("C", 4, _civ1 + _civ2, Hb, Te_high, ne_hi, [1548, 1551])
    elif _civ1 > 0:
        ionic["C+++/H+"] = _ionic_abundance("C", 4, _civ1, Hb, Te_high, ne_hi, 1548)
    elif _civ2 > 0:
        ionic["C+++/H+"] = _ionic_abundance("C", 4, _civ2, Hb, Te_high, ne_hi, 1551)

    # N2+/H+ from NIII] 1749 + 1752 — intermediate zone (Te_mid, ne_md)
    # Martinez+2025 §5.4: N²⁺ uses Te_int and ne_int.
    _n1749 = fluxes.get("NIII_1749", 0.0)
    _n1752 = fluxes.get("NIII_1752", 0.0)
    if _n1749 > 0 and _n1752 > 0:
        ionic["N++/H+"] = _ionic_abundance("N", 3, _n1749 + _n1752, Hb, Te_mid, ne_md, [1749, 1752])
    elif _n1749 > 0:
        ionic["N++/H+"] = _ionic_abundance("N", 3, _n1749, Hb, Te_mid, ne_md, 1749)
    elif _n1752 > 0:
        ionic["N++/H+"] = _ionic_abundance("N", 3, _n1752, Hb, Te_mid, ne_md, 1752)

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
    co_icf_method: str = "auto",
    co_logU: float | None = None,
    co_ne: float | None = None,
    ionic_upper_limits: dict[str, float] | None = None,
    _lock_NO_icf: str | None = None,
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
    icf_dict: dict[str, dict] = {}

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

    # C/O uses the intermediate (C2+/O2+) ionisation zone: Martinez confirms
    # log(U_int) is the appropriate ionisation parameter for the C2+/O2+ ICF,
    # and the matching density is the intermediate zone (from CIII], the C2+
    # density diagnostic).  ``co_logU``/``co_ne`` carry those; they fall back
    # to the shared logU/ne (high zone) when not supplied.
    _co_logU = co_logU if co_logU is not None else logU
    _co_ne = co_ne if co_ne is not None else ne

    # Decide whether to use the Martinez (in prep.) C/O ICF (C2+/O2+).
    use_martinez_co = False
    if co_icf_method == "martinez25":
        if _co_logU is None or Z_Zsun is None:
            logger.warning(
                "Martinez C/O ICF requires logU and Z_Zsun; "
                "falling back to Garnett+97."
            )
        else:
            use_martinez_co = True
    elif co_icf_method == "auto" and _co_logU is not None and Z_Zsun is not None:
        use_martinez_co = True

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
                icf_dict["N/O"] = {
                    "icf": 1.0, "method": "direct sum (no ICF)",
                    "raw": np.log10(totals["N/O"]) if totals["N/O"] > 0 else None,
                    "corrected": np.log10(totals["N/O"]) if totals["N/O"] > 0 else None,
                }
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
                icf_dict["N/O"] = {
                    "icf": 1.0, "method": "direct sum UV (no ICF)",
                    "raw": np.log10(totals["N/O"]) if totals["N/O"] > 0 else None,
                    "corrected": np.log10(totals["N/O"]) if totals["N/O"] > 0 else None,
                }
            elif N_plus > 0 and O_plus > 0:
                # Tier 4: optical fallback — Izotov+06
                icf_n = icf_nitrogen(O_plus, OH)
                raw_no_iz = N_plus / OH if OH > 0 else 0
                totals["N/O"] = icf_n * raw_no_iz
                totals["icf_method"] = "izotov06"
                totals["NO_icf_name"] = "izotov06_fallback"
                icf_dict["N/O"] = {
                    "icf": icf_n, "method": "Izotov+06",
                    "raw": np.log10(raw_no_iz) if raw_no_iz > 0 else None,
                    "corrected": np.log10(totals["N/O"]) if totals["N/O"] > 0 else None,
                }

        # N/O — Martinez+25 with direct_sum fallback
        elif use_martinez:
            from .martinez25_icf import compute_NO_martinez25, compute_NO_martinez25_locked
            ne_icf = ne if ne is not None else NE_DEFAULT
            # If locked to a specific ICF tier, use it directly.
            # When locked, do NOT fall back — return no N/O so the MC
            # iteration gets NaN rather than mixing tiers.
            ionic_with_ul = ionic  # default; overwritten if ULs used
            if _lock_NO_icf is not None:
                NO_val = compute_NO_martinez25_locked(ionic, logU, Z_Zsun, ne_icf, _lock_NO_icf)
                NO_icf_name = _lock_NO_icf
                if NO_val is not None:
                    totals["N/O"] = NO_val
                    totals["NO_icf_name"] = NO_icf_name
                    totals["icf_method"] = "martinez25"
                elif ionic_upper_limits:
                    # Ions not detected — try with 3σ upper-limit ionic
                    # abundances to produce an N/O upper limit.
                    ionic_with_ul = dict(ionic)
                    for ul_key, ul_val in ionic_upper_limits.items():
                        if ionic_with_ul.get(ul_key, 0.0) <= 0:
                            ionic_with_ul[ul_key] = ul_val
                    NO_ul_val = compute_NO_martinez25_locked(
                        ionic_with_ul, logU, Z_Zsun, ne_icf, _lock_NO_icf,
                    )
                    if NO_ul_val is not None:
                        NO_val = NO_ul_val  # so ICF dict block below fires
                        totals["N/O"] = NO_ul_val
                        totals["NO_icf_name"] = NO_icf_name
                        totals["icf_method"] = "martinez25"
                        totals["NO_is_upper_limit"] = True
                # else: N/O stays unset → NaN in MC loop
            else:
                NO_val, NO_icf_name = compute_NO_martinez25(ionic, logU, Z_Zsun, ne_icf)
            if _lock_NO_icf is None and NO_val is not None:
                totals["N/O"] = NO_val
                totals["NO_icf_name"] = NO_icf_name
                totals["icf_method"] = "martinez25"
            # Store N/O ICF value for the selected tier.
            if NO_val is not None and NO_icf_name is not None:
                _icf_key_map = {
                    "NppNppp_Opp": "_icf5_value",
                    "NpNpp_OpOpp": "_icf4_value",
                    "NpppOpp": "_icf3_value",
                    "NppOpp": "_icf2_value",
                    "NpOp": "_icf1_value",
                }
                # Use UL-augmented ionic dict if upper limits were used.
                _ion = ionic_with_ul if totals.get("NO_is_upper_limit") else ionic
                # Compute raw ionic ratio (without ICF)
                _no_icf_funcs = {
                    "NppNppp_Opp": lambda: (
                        (_ion.get("N++/H+", 0) + _ion.get("N+++/H+", 0)) / O_pp if O_pp > 0 else 0
                    ),
                    "NpOp": lambda: _ion.get("N+/H+", 0) / O_plus if O_plus > 0 else 0,
                    "NppOpp": lambda: _ion.get("N++/H+", 0) / O_pp if O_pp > 0 else 0,
                    "NpppOpp": lambda: _ion.get("N+++/H+", 0) / O_pp if O_pp > 0 else 0,
                    "NpNpp_OpOpp": lambda: (
                        (_ion.get("N+/H+", 0) + _ion.get("N++/H+", 0)) / (O_plus + O_pp)
                        if (O_plus + O_pp) > 0 else 0
                    ),
                }
                raw_ratio = _no_icf_funcs.get(NO_icf_name, lambda: 0)()
                if raw_ratio > 0 and NO_val > 0:
                    icf_no = NO_val / raw_ratio
                    icf_dict["N/O"] = {
                        "icf": icf_no, "method": f"Martinez+25 ({NO_icf_name})",
                        "raw": np.log10(raw_ratio),
                        "corrected": np.log10(NO_val),
                    }
            elif _lock_NO_icf is None and NO_val is None:
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
                    totals["N/O"] = icf_n * N_plus / OH if OH > 0 else 0
                    totals["icf_method"] = "izotov06"
                    totals["NO_icf_name"] = "izotov06_fallback"
        else:
            N_plus = ionic.get("N+/H+", 0.0)
            if N_plus > 0 and O_plus > 0:
                icf_n = icf_nitrogen(O_plus, OH)
                totals["N/O"] = icf_n * N_plus / OH if OH > 0 else 0
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
            raw_so = S_total_ion / OH if OH > 0 else 0
            totals["S/O"] = icf_s * S_total_ion / OH
            icf_dict["S/O"] = {
                "icf": icf_s, "method": "Izotov+06",
                "raw": np.log10(raw_so) if raw_so > 0 else None,
                "corrected": np.log10(totals["S/O"]) if totals["S/O"] > 0 else None,
            }
        else:
            failures["S/O"] = "no S+ or S++ ions detected ([SII]/[SIII] missing)"

        # Ne/O
        Ne_pp = ionic.get("Ne++/H+", 0.0)
        if Ne_pp > 0 and O_pp > 0:
            icf_ne = icf_neon(O_plus, OH)
            raw_neo = Ne_pp / OH if OH > 0 else 0
            totals["Ne/O"] = icf_ne * raw_neo
            icf_dict["Ne/O"] = {
                "icf": icf_ne, "method": "Izotov+06",
                "raw": np.log10(raw_neo) if raw_neo > 0 else None,
                "corrected": np.log10(totals["Ne/O"]) if totals["Ne/O"] > 0 else None,
            }
        elif Ne_pp <= 0:
            failures["Ne/O"] = "no Ne++ ion detected ([NeIII] 3869 missing)"
        else:
            failures["Ne/O"] = "no O++ ion for Ne/O normalisation"

        # Ar/O
        Ar_pp = ionic.get("Ar++/H+", 0.0)
        if Ar_pp > 0 and O_pp > 0:
            icf_ar = icf_argon(O_plus, OH)
            raw_aro = Ar_pp / OH if OH > 0 else 0
            totals["Ar/O"] = icf_ar * raw_aro
            icf_dict["Ar/O"] = {
                "icf": icf_ar, "method": "Izotov+06",
                "raw": np.log10(raw_aro) if raw_aro > 0 else None,
                "corrected": np.log10(totals["Ar/O"]) if totals["Ar/O"] > 0 else None,
            }
        elif Ar_pp <= 0:
            failures["Ar/O"] = "no Ar++ ion detected ([ArIII] 7136 missing)"
        else:
            failures["Ar/O"] = "no O++ ion for Ar/O normalisation"

        # C/O:
        #   Martinez (in prep.) ICF × (C2+ / O2+) when logU+Z available
        #   (co_icf_method 'auto'/'martinez25'), otherwise the legacy tiered
        #   approach:
        #     1. Direct sum (C+ + C2+ + C3+) / (O+ + O2+) when C+ detected
        #     2. Garnett+1997 ICF × (C2+ + C3+) / O2+ when C+ not detected
        #     3. Raw (C2+ + C3+) / O2+ when O+ also missing (ICF=1)
        C_p = ionic.get("C+/H+", 0.0)
        C_pp = ionic.get("C++/H+", 0.0)
        C_ppp = ionic.get("C+++/H+", 0.0)
        C_uv = C_pp + C_ppp
        co_done = False
        if use_martinez_co and C_pp > 0 and O_pp > 0:
            from .martinez25_icf import icf_CppOpp
            ne_icf = _co_ne if _co_ne is not None else NE_DEFAULT
            icf_c = icf_CppOpp(_co_logU, Z_Zsun, ne_icf)
            if np.isfinite(icf_c):
                raw_co = C_pp / O_pp
                totals["C/O"] = icf_c * raw_co
                totals["CO_method"] = "martinez25"
                totals["CO_icf_value"] = icf_c
                icf_dict["C/O"] = {
                    "icf": icf_c, "method": "Martinez (in prep.)",
                    "raw": np.log10(raw_co) if raw_co > 0 else None,
                    "corrected": np.log10(totals["C/O"]) if totals["C/O"] > 0 else None,
                }
                co_done = True
        if not co_done:
            if co_icf_method == "martinez25" and use_martinez_co:
                logger.warning(
                    "Martinez C/O ICF unavailable (needs C2+ and O2+ within "
                    "calibration validity); falling back to Garnett+97."
                )
            if C_p > 0 and C_uv > 0 and OH > 0:
                # Direct sum — all C ions detected, use total O
                totals["C/O"] = (C_p + C_uv) / OH
                totals["CO_method"] = "direct_sum"
            elif C_uv > 0 and O_pp > 0:
                # Apply Garnett+1997 ICF to correct for missing C+
                icf_c = icf_carbon(O_plus, O_pp)
                raw_co = C_uv / O_pp
                totals["C/O"] = icf_c * raw_co
                totals["CO_method"] = "garnett97_icf"
                totals["CO_icf_value"] = icf_c
                icf_dict["C/O"] = {
                    "icf": icf_c, "method": "Garnett+97",
                    "raw": np.log10(raw_co) if raw_co > 0 else None,
                    "corrected": np.log10(totals["C/O"]) if totals["C/O"] > 0 else None,
                }
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
    if icf_dict:
        totals["_icf_values"] = icf_dict
    return totals
