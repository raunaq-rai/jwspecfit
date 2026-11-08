"""Parameter constraints: tied kinematics and fixed flux ratios.

Constraints are applied as transformations on the free-parameter vector
before the model is evaluated.  This keeps the optimiser working in
an unconstrained space while enforcing physical relationships.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .lines import REST_LINES_A, canonical_line_names

logger = logging.getLogger(__name__)

# [NII] 6549/6585 theoretical flux ratio (Storey & Zeippen 2000).
NII_RATIO = 1.0 / 2.96

# Resonance-doublet optically-thin amplitude ratio (secondary/primary),
# set by statistical weights g = 2J+1.  Used as the blended-fallback ratio
# for N V and C IV; the ratio is left free when the doublet is resolved,
# because resonance scattering / optical depth drives it between 2:1
# (optically thin) and 1:1 (optically thick).
NV_RATIO = 0.5       # F(NV_2) / F(NV_1), optically thin limit

# Low-density / optically-thin flux ratios for doublets.
# Used as the default ratio when a doublet is blended (unresolved).
_INTERCOM_LOW_DENSITY_RATIOS: dict[tuple[str, str], float] = {
    # (primary, secondary): F(secondary) / F(primary)
    ("NV_1",      "NV_2"):      NV_RATIO,  # resonance, optically thin limit
    ("CIV_1",     "CIV_2"):     0.50,   # optically thin limit
    # Pure branching ratio: both components decay from the same upper level
    # (5S2 -> 3P1, 3P2), so this is fixed by the A-values and wavelengths
    # alone -- A(6->2)*lam(1666) / (A(6->3)*lam(1661)) -- with no density or
    # temperature dependence at all.  Verified against PyNEB at
    # ne = 1e1, 1e3 and 1e6: 0.4016 to four figures throughout.
    ("OIII_1666", "OIII_1661"): 0.4016,
    ("CIII]_1907", "CIII]_1909"):     0.65,   # Keenan+1992
    ("CII]_2326", "CII]_2324"):  0.49,   # A-value ratio, same upper level
    ("NIV_1486",   "NIV_1483"):  1.48,   # low-density limit (PyNEB); drops below 1 at ne > 5e4
    ("SiIII_1",   "SiIII_2"):   0.67,
}

# N III] 1750 quintuplet: F(member) / F(NIII_1750), from PyNEB
# (n_iii_atom_GMZ98, n_iii_coll_BP92) at Te = 1.5e4 K, ne = 1e4 cm^-3.
# The five members lie within 7 A of each other, so they cannot be
# separated at R ~ 1000 and freeing their ratios would be degenerate: the
# whole multiplet is fitted with ONE free flux (that of 1749.7, the
# strongest member) plus one shared velocity and width.  Above ~1e4 cm^-3
# the shares barely move (1e5 changes them by <= 5%); at 1e3 the weak
# members rise by up to 40% relative to 1749.7.  The fitted total flux,
# which is what the abundances use, barely notices: on a noise-free R = 1000
# blend injected at ne = 1e2 / 1e3 / 1e5 it is biased by -0.033 / -0.013 /
# +0.002 dex (tests/test_niii_multiplet.py).
NIII_MULTIPLET_ANCHOR = "NIII_1750"
NIII_MULTIPLET_RATIOS: dict[str, float] = {
    "NIII_1747": 0.1002,   # 4P3/2 -> 2P1/2
    "NIII_1749": 0.1732,   # 4P1/2 -> 2P1/2
    "NIII_1752": 0.5961,   # 4P3/2 -> 2P3/2
    "NIII_1754": 0.1844,   # 4P1/2 -> 2P3/2
}

# Pure branching ratios: both members decay from the same upper level, so
# F(secondary) / F(primary) is fixed by the A-values alone.  PyNEB gives the
# same value to four figures for Te = 1-2e4 K and ne = 1e2-1e5 cm^-3.  The
# secondary shares the primary's flux ratio, velocity and width.
BRANCHING_RATIOS: dict[tuple[str, str], float] = {
    # (primary, secondary): F(secondary) / F(primary)
    ("OIII_5007", "OIII_4959"): 1.0 / 2.984,
    ("OIII_1666", "OIII_1661"): 0.4016,
    ("NeIII_3869", "NeIII_3968"): 1.0 / 3.320,
    ("OI_6302", "OI_6366"): 1.0 / 3.131,
}

# Case B H8/H9 (PyNEB, ne = 1e3 cm^-3): 1.436 / 1.437 / 1.438 at Te = 1 / 1.5 /
# 2e4 K.  The lines are 54 A apart, so dust cancels.  H8 is 0.4 A from
# He I 3889, which is free, so without this tie their split is degenerate.
H8_H9_RATIO = 1.437


# Narrow Balmer lines whose width and centroid follow [OIII] 5007.
_BALMER_TIED_TO_OIII = (
    "HALPHA", "HBETA", "HGAMMA", "HDELTA", "HEPSILON", "H8", "H9", "H10",
)

# Broad Balmer components whose centroid follows the narrow line.
_BROAD_PAIRS = (
    ("HALPHA_BROAD", "HALPHA"), ("HALPHA_BROAD2", "HALPHA"),
    ("HBETA_BROAD", "HBETA"), ("HBETA_BROAD2", "HBETA"),
    ("HDELTA_BROAD", "HDELTA"), ("HDELTA_BROAD2", "HDELTA"),
    ("HGAMMA_BROAD", "HGAMMA"), ("HGAMMA_BROAD2", "HGAMMA"),
)

# (primary, secondary) doublets with shared kinematics.
_KINEMATIC_TIED = (
    ("NV_1", "NV_2"),
    ("CIV_1", "CIV_2"),
    ("OIII_1666", "OIII_1661"),
    ("CIII]_1907", "CIII]_1909"),
    ("NIV_1486", "NIV_1483"),
    ("SiIII_1", "SiIII_2"),
    ("CII]_2326", "CII]_2324"),
)

# UV intercombination lines sharing one velocity width (first present = anchor).
_UV_INTERCOM = ("CIII]_1907", "NIV_1486", "NIII_1750")

# Doublets whose members always share a velocity width.
_ALWAYS_TIE_WIDTH = (("CIII]_1907", "CIII]_1909"),)


def _check_op_order(ops: list[tuple[int, int, float]]) -> None:
    """Raise if an op copies a source that a later op still overwrites."""
    written_later: set[int] = set()
    for dst, src, _ in reversed(ops):
        if src in written_later:
            raise RuntimeError(
                f"Tying op copies parameter {src} before its final value is set."
            )
        written_later.add(dst)


@dataclass
class ConstraintSet:
    """Collection of parameter constraints for a line fit.

    Attributes
    ----------
    line_names : list of str
        Ordered line names matching the parameter vector layout.
    tie_nii : bool
        Tie [NII] 6549 amplitude and kinematics to [NII] 6585.
    tie_balmer_to_oiii : bool
        Tie narrow Balmer (and [NII]) widths to [OIII] 5007 in velocity space.
    blended_doublets : set of str
        Secondary line names of kinematic-tied doublets that are
        unresolved and should have their amplitude fixed to the
        low-density-limit ratio.  Populated by the fitter based on
        spectral resolution.
    fix_branching_ratios : bool
        Fix every pair in :data:`BRANCHING_RATIOS` (flux ratio, velocity and
        width, overriding ``tie_uv_centroids`` for O III] 1661) and H8 to
        :data:`H8_H9_RATIO` x H9.
    """

    line_names: list[str]
    tie_nii: bool = True
    tie_balmer_to_oiii: bool = True
    tie_uv_doublets: bool = True
    tie_uv_centroids: bool = True
    tie_uv_widths: bool = True
    blended_doublets: set[str] | None = None
    niv_doublet_ratio: float | None = None
    ciii_doublet_ratio: float | None = None
    fix_branching_ratios: bool = True

    def __post_init__(self) -> None:
        # Constraints saved before a line key was renamed carry the old name.
        self.line_names = canonical_line_names(list(self.line_names), warn=False)
        if self.blended_doublets is not None:
            self.blended_doublets = canonical_line_names(
                set(self.blended_doublets), warn=False,
            )

    def tying_ops(self) -> list[tuple[int, int, float]]:
        """Ordered tying operations: the single source of truth for every tie.

        Each operation ``(dst, src, ratio)`` means ``p[dst] = p[src] * ratio``.
        :meth:`apply`, :meth:`free_mask` and the JAX likelihood all consume
        this list, so they cannot disagree.  Operations are ordered so that
        every source is final before it is copied (checked on return).

        Returns
        -------
        list of (int, int, float)
            Ordered tying operations on the full parameter vector
            ``[A_0..A_n, mu_0..mu_n, sigma_0..sigma_n]``.
        """
        nL = len(self.line_names)
        idx = {name: i for i, name in enumerate(self.line_names)}
        ops: list[tuple[int, int, float]] = []

        def amp(n: str) -> int:
            return idx[n]

        def cen(n: str) -> int:
            return nL + idx[n]

        def wid(n: str) -> int:
            return 2 * nL + idx[n]

        def lam_ratio(dst: str, src: str) -> float:
            return REST_LINES_A[dst] / REST_LINES_A[src]

        # --- Narrow Balmer kinematics tied to [OIII] 5007 (same velocity) ---
        if self.tie_balmer_to_oiii and "OIII_5007" in idx:
            for name in _BALMER_TIED_TO_OIII:
                if name in idx:
                    ops.append((wid(name), wid("OIII_5007"), lam_ratio(name, "OIII_5007")))
            for name in _BALMER_TIED_TO_OIII:
                if name in idx:
                    ops.append((cen(name), cen("OIII_5007"), lam_ratio(name, "OIII_5007")))

        # --- Broad Balmer centroids tied to their narrow counterparts ---
        for broad_name, narrow_name in _BROAD_PAIRS:
            if broad_name in idx and narrow_name in idx:
                ops.append((cen(broad_name), cen(narrow_name), 1.0))

        # --- [OIII] broad doublet kinematics (same outflowing gas) ---
        for suffix in ("_BROAD", "_BROAD2"):
            pri, sec = f"OIII_5007{suffix}", f"OIII_4959{suffix}"
            if pri in idx and sec in idx:
                r = REST_LINES_A["OIII_4959"] / REST_LINES_A["OIII_5007"]
                ops.append((cen(sec), cen(pri), r))
                ops.append((wid(sec), wid(pri), r))

        # --- He I broad kinematics: one sigma_v / dv per tier ---
        from .broad import HEI_BROAD_CANDIDATES as _HEI_BC
        for suffix in ("_BROAD", "_BROAD2"):
            present = [f"{n}{suffix}" for n in _HEI_BC if f"{n}{suffix}" in idx]
            for tgt in present[1:]:
                ops.append((cen(tgt), cen(present[0]), lam_ratio(tgt, present[0])))
                ops.append((wid(tgt), wid(present[0]), lam_ratio(tgt, present[0])))

        # --- [NII] doublet: flux ratio and kinematics ---
        if self.tie_nii and "NII_6549" in idx and "NII_6585" in idx:
            r = lam_ratio("NII_6549", "NII_6585")
            ops.append((amp("NII_6549"), amp("NII_6585"), NII_RATIO))
            ops.append((cen("NII_6549"), cen("NII_6585"), r))
            ops.append((wid("NII_6549"), wid("NII_6585"), r))

        if self.tie_uv_doublets:
            # UV intercombination widths share one velocity dispersion.  This
            # runs BEFORE the doublet ties below: a doublet primary can itself
            # be a width target (N IV] 1486 -> C III] 1907), and its secondary
            # must copy the primary's final width.
            if self.tie_uv_widths:
                present = [n for n in _UV_INTERCOM if n in idx]
                for name in present[1:]:
                    ops.append((wid(name), wid(present[0]), lam_ratio(name, present[0])))

            # Kinematically tied doublets: amplitude free when resolved, fixed
            # to the low-density / optically-thin ratio when blended.
            _blended = self.blended_doublets or set()
            for pri, sec in _KINEMATIC_TIED:
                if pri in idx and sec in idx:
                    r = lam_ratio(sec, pri)
                    if self.tie_uv_centroids:
                        ops.append((cen(sec), cen(pri), r))
                    ops.append((wid(sec), wid(pri), r))
                    fixed = {
                        ("NIV_1486", "NIV_1483"): self.niv_doublet_ratio,
                        ("CIII]_1907", "CIII]_1909"): self.ciii_doublet_ratio,
                    }.get((pri, sec))
                    if fixed is not None:
                        ops.append((amp(sec), amp(pri), fixed))
                    elif sec in _blended:
                        ops.append((amp(sec), amp(pri),
                                    _INTERCOM_LOW_DENSITY_RATIOS.get((pri, sec), 0.67)))

            # N III] quintuplet: one flux, one velocity, one width.
            if NIII_MULTIPLET_ANCHOR in idx:
                for name, ratio in NIII_MULTIPLET_RATIOS.items():
                    if name in idx:
                        r = lam_ratio(name, NIII_MULTIPLET_ANCHOR)
                        ops.append((amp(name), amp(NIII_MULTIPLET_ANCHOR), ratio))
                        ops.append((cen(name), cen(NIII_MULTIPLET_ANCHOR), r))
                        ops.append((wid(name), wid(NIII_MULTIPLET_ANCHOR), r))

        # --- Doublet-internal width tying (always active) ---
        for pri, sec in _ALWAYS_TIE_WIDTH:
            if pri in idx and sec in idx:
                op = (wid(sec), wid(pri), lam_ratio(sec, pri))
                if op not in ops:
                    ops.append(op)

        # --- Fixed branching ratios (last: every primary is final by now) ---
        if self.fix_branching_ratios:
            for (pri, sec), ratio in BRANCHING_RATIOS.items():
                if pri in idx and sec in idx:
                    r = lam_ratio(sec, pri)
                    ops.append((amp(sec), amp(pri), ratio))
                    ops.append((cen(sec), cen(pri), r))
                    ops.append((wid(sec), wid(pri), r))
            if "H8" in idx and "H9" in idx:
                ops.append((amp("H8"), amp("H9"), H8_H9_RATIO))

        _check_op_order(ops)
        return ops

    def apply(self, params: np.ndarray) -> np.ndarray:
        """Apply constraints to a parameter vector (returns a copy).

        Parameters
        ----------
        params : np.ndarray
            Raw parameter vector ``[A_0..A_n, mu_0..mu_n, sigma_0..sigma_n]``.

        Returns
        -------
        np.ndarray
            Constrained parameter vector.
        """
        p = np.array(params, dtype=float, copy=True)
        for dst, src, ratio in self.tying_ops():
            p[dst] = p[src] * ratio
        return p

    def free_mask(self) -> np.ndarray:
        """Boolean mask of free (unconstrained) parameters.

        A parameter is free exactly when no tying operation writes to it.

        Returns
        -------
        np.ndarray
            Boolean array of length ``3 * n_lines``.
        """
        free = np.ones(3 * len(self.line_names), dtype=bool)
        for dst, _, _ in self.tying_ops():
            free[dst] = False
        return free

    def expand_free_to_full(self, p_free: np.ndarray) -> np.ndarray:
        """Insert free parameters into a full-length vector and apply ties.

        Parameters
        ----------
        p_free : np.ndarray
            Free parameter values.

        Returns
        -------
        np.ndarray
            Full parameter vector (length ``3 * n_lines``).
        """
        full = np.zeros(3 * len(self.line_names))
        full[self.free_mask()] = p_free
        return self.apply(full)
