"""Parameter constraints: tied kinematics and fixed flux ratios.

Constraints are applied as transformations on the free-parameter vector
before the model is evaluated.  This keeps the optimiser working in
an unconstrained space while enforcing physical relationships.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .lines import REST_LINES_A

logger = logging.getLogger(__name__)

# [NII] 6549/6585 theoretical flux ratio (Storey & Zeippen 2000).
NII_RATIO = 1.0 / 2.96

# Resonance doublets: secondary/primary amplitude ratio (optically thin).
# Set by statistical weights g = 2J+1; not density-sensitive.
NV_RATIO = 0.5       # F(NV_2) / F(NV_1)

# Low-density-limit flux ratios for intercombination doublets.
# Used as default when doublet is unresolved.
_INTERCOM_LOW_DENSITY_RATIOS: dict[tuple[str, str], float] = {
    # (primary, secondary): F(secondary) / F(primary)
    ("CIV_1",     "CIV_2"):     0.50,   # optically thin limit
    ("OIII_1666", "OIII_1661"): 0.83,   # 1/1.20, low-density limit
    ("CIII]_1907", "CIII]"):     0.65,   # Keenan+1992
    ("NIV_1486",   "NIV_1483"):  0.67,
    ("NIII_1749",  "NIII_1752"): 0.67,
    ("SiIII_1",   "SiIII_2"):   0.67,
}


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
    """

    line_names: list[str]
    tie_nii: bool = True
    tie_balmer_to_oiii: bool = True
    tie_uv_doublets: bool = True
    tie_uv_centroids: bool = True
    blended_doublets: set[str] | None = None

    def apply(self, params: np.ndarray) -> np.ndarray:
        """Apply constraints to a parameter vector (in-place copy).

        Parameters
        ----------
        params : np.ndarray
            Raw parameter vector ``[A_0..A_n, mu_0..mu_n, sigma_0..sigma_n]``.

        Returns
        -------
        np.ndarray
            Constrained parameter vector.
        """
        p = params.copy()
        nL = len(self.line_names)
        idx = {name: i for i, name in enumerate(self.line_names)}

        # --- Narrow-line kinematics tying to [OIII] 5007 ---
        # Widths are tied in velocity space for all narrow lines.
        # Centroids are tied for Balmer lines only (same systemic velocity).
        # NII_6585 width is tied but centroid is free so it can fit its peak.
        if self.tie_balmer_to_oiii and "OIII_5007" in idx:
            i_o3 = idx["OIII_5007"]
            lam_o3 = REST_LINES_A["OIII_5007"]
            sigma_o3 = p[2 * nL + i_o3]
            mu_o3 = p[nL + i_o3]

            # Width tying: Balmer lines share OIII velocity dispersion.
            # NII width is free (tied only via the doublet constraint).
            width_targets = [
                "Ha", "HBETA", "HGAMMA", "HDELTA",
                "HEPSILON", "H8", "H9", "H10",
            ]
            for name in width_targets:
                if name in idx:
                    i_t = idx[name]
                    ratio = REST_LINES_A[name] / lam_o3
                    p[2 * nL + i_t] = sigma_o3 * ratio

            # Centroid tying: Balmer lines only (not NII — its centroid is free).
            centroid_targets = [
                "Ha", "HBETA", "HGAMMA", "HDELTA",
                "HEPSILON", "H8", "H9", "H10",
            ]
            for name in centroid_targets:
                if name in idx:
                    i_t = idx[name]
                    ratio = REST_LINES_A[name] / lam_o3
                    p[nL + i_t] = mu_o3 * ratio

        # --- Tie broad centroids to their narrow counterparts ---
        _BROAD_PAIRS = [
            ("Ha_BROAD", "Ha"), ("Ha_BROAD2", "Ha"),
            ("HBETA_BROAD", "HBETA"), ("HBETA_BROAD2", "HBETA"),
            ("HDELTA_BROAD", "HDELTA"), ("HDELTA_BROAD2", "HDELTA"),
            ("HGAMMA_BROAD", "HGAMMA"), ("HGAMMA_BROAD2", "HGAMMA"),
        ]
        for broad_name, narrow_name in _BROAD_PAIRS:
            if broad_name in idx and narrow_name in idx:
                p[nL + idx[broad_name]] = p[nL + idx[narrow_name]]

        # --- [NII] doublet constraint (after Balmer tying so NII_6585 σ is set) ---
        if self.tie_nii and "NII_6549" in idx and "NII_6585" in idx:
            i49 = idx["NII_6549"]
            i85 = idx["NII_6585"]
            lam_ratio = REST_LINES_A["NII_6549"] / REST_LINES_A["NII_6585"]

            # Amplitude: A_6549 = A_6585 × NII_RATIO
            p[i49] = p[i85] * NII_RATIO

            # Centroid: tied in velocity space
            p[nL + i49] = p[nL + i85] * lam_ratio

            # Width: tied in velocity space
            p[2 * nL + i49] = p[2 * nL + i85] * lam_ratio

        # --- UV doublet constraints ---
        if self.tie_uv_doublets:
            # Amplitude-tied doublets: resonance lines with fixed flux ratios
            # and OIII] UV intercombination (weakly density-dependent).
            _AMPLITUDE_TIED = [
                # (primary, secondary, ratio = F_secondary / F_primary)
                ("NV_1",      "NV_2",      NV_RATIO),
            ]
            for pri, sec, ratio in _AMPLITUDE_TIED:
                if pri in idx and sec in idx:
                    i_pri = idx[pri]
                    i_sec = idx[sec]
                    lam_ratio = REST_LINES_A[sec] / REST_LINES_A[pri]
                    p[i_sec] = p[i_pri] * ratio
                    if self.tie_uv_centroids:
                        p[nL + i_sec] = p[nL + i_pri] * lam_ratio
                    p[2 * nL + i_sec] = p[2 * nL + i_pri] * lam_ratio

            # Intercombination doublets: density-sensitive lines.
            # If resolved, only kinematics are tied (amplitude free).
            # If blended (unresolved), amplitude is also fixed to the
            # low-density-limit ratio for fitting stability.
            _KINEMATIC_TIED = [
                # (primary, secondary) — amplitudes free when resolved
                ("CIV_1",     "CIV_2"),
                ("OIII_1666", "OIII_1661"),
                ("CIII]_1907", "CIII]"),
                ("NIV_1486",   "NIV_1483"),
                ("NIII_1749",  "NIII_1752"),
                ("SiIII_1",   "SiIII_2"),
            ]
            _blended = self.blended_doublets or set()
            for pri, sec in _KINEMATIC_TIED:
                if pri in idx and sec in idx:
                    i_pri = idx[pri]
                    i_sec = idx[sec]
                    lam_ratio = REST_LINES_A[sec] / REST_LINES_A[pri]
                    if self.tie_uv_centroids:
                        p[nL + i_sec] = p[nL + i_pri] * lam_ratio
                    p[2 * nL + i_sec] = p[2 * nL + i_pri] * lam_ratio
                    # Fix amplitude for unresolved doublets.
                    if sec in _blended:
                        ratio = _INTERCOM_LOW_DENSITY_RATIOS.get(
                            (pri, sec), 0.67,
                        )
                        p[i_sec] = p[i_pri] * ratio

            # Tie UV intercombination line widths together in velocity
            # space.  The first available line becomes the anchor whose
            # width is free; all others are derived from it.  The
            # shared width is jointly constrained by all tied lines
            # through the residual.
            # CIV excluded (resonance line), OIII] excluded (its own
            # doublet constraint already ties 1661 to 1666).
            _UV_INTERCOM = [
                "CIII]_1907", "NIV_1486", "NIII_1749",
            ]
            _uv_present = [n for n in _UV_INTERCOM if n in idx]
            if len(_uv_present) >= 2:
                anchor = _uv_present[0]
                i_anchor = idx[anchor]
                lam_anchor = REST_LINES_A[anchor]
                sigma_anchor = p[2 * nL + i_anchor]
                for name in _uv_present[1:]:
                    i_t = idx[name]
                    ratio = REST_LINES_A[name] / lam_anchor
                    p[2 * nL + i_t] = sigma_anchor * ratio

        return p

    def free_mask(self) -> np.ndarray:
        """Boolean mask of free (unconstrained) parameters.

        Parameters that are determined by constraints are marked False.
        The optimiser should only vary free parameters.

        Returns
        -------
        np.ndarray
            Boolean array of length ``3 * n_lines``.
        """
        nL = len(self.line_names)
        free = np.ones(3 * nL, dtype=bool)
        idx = {name: i for i, name in enumerate(self.line_names)}

        if self.tie_nii and "NII_6549" in idx:
            i49 = idx["NII_6549"]
            # Amplitude, centroid, and sigma are derived
            free[i49] = False
            free[nL + i49] = False
            free[2 * nL + i49] = False

        if self.tie_balmer_to_oiii and "OIII_5007" in idx:
            # Width tied for Balmer lines only (NII width is free).
            width_targets = [
                "Ha", "HBETA", "HGAMMA", "HDELTA",
                "HEPSILON", "H8", "H9", "H10",
            ]
            for name in width_targets:
                if name in idx:
                    free[2 * nL + idx[name]] = False

            # Centroid tied for Balmer lines only (NII centroid stays free).
            centroid_targets = [
                "Ha", "HBETA", "HGAMMA", "HDELTA",
                "HEPSILON", "H8", "H9", "H10",
            ]
            for name in centroid_targets:
                if name in idx:
                    free[nL + idx[name]] = False

        # Broad centroids tied to narrow.
        _BROAD_PAIRS = [
            ("Ha_BROAD", "Ha"), ("Ha_BROAD2", "Ha"),
            ("HBETA_BROAD", "HBETA"), ("HBETA_BROAD2", "HBETA"),
            ("HDELTA_BROAD", "HDELTA"), ("HDELTA_BROAD2", "HDELTA"),
            ("HGAMMA_BROAD", "HGAMMA"), ("HGAMMA_BROAD2", "HGAMMA"),
        ]
        for broad_name, narrow_name in _BROAD_PAIRS:
            if broad_name in idx and narrow_name in idx:
                free[nL + idx[broad_name]] = False

        # UV doublet constraints.
        if self.tie_uv_doublets:
            # Amplitude-tied: all 3 parameters of secondary are derived.
            _AMPLITUDE_TIED = [
                ("NV_1",      "NV_2"),
            ]
            for pri, sec in _AMPLITUDE_TIED:
                if pri in idx and sec in idx:
                    i_sec = idx[sec]
                    free[i_sec] = False
                    if self.tie_uv_centroids:
                        free[nL + i_sec] = False
                    free[2 * nL + i_sec] = False

            # Intercombination doublets: sigma derived, centroid conditional.
            # Amplitude is free when resolved, derived when blended.
            _KINEMATIC_TIED = [
                ("CIV_1",     "CIV_2"),
                ("OIII_1666", "OIII_1661"),
                ("CIII]_1907", "CIII]"),
                ("NIV_1486",   "NIV_1483"),
                ("NIII_1749",  "NIII_1752"),
                ("SiIII_1",   "SiIII_2"),
            ]
            _blended = self.blended_doublets or set()
            for pri, sec in _KINEMATIC_TIED:
                if pri in idx and sec in idx:
                    i_sec = idx[sec]
                    if self.tie_uv_centroids:
                        free[nL + i_sec] = False
                    free[2 * nL + i_sec] = False
                    if sec in _blended:
                        free[i_sec] = False

            # UV intercombination widths: first available is anchor (free),
            # all others are derived (not free).
            # OIII] excluded — its width is independent.
            _UV_INTERCOM = [
                "CIII]_1907", "NIV_1486", "NIII_1749",
            ]
            _uv_present = [n for n in _UV_INTERCOM if n in idx]
            if len(_uv_present) >= 2:
                for name in _uv_present[1:]:
                    free[2 * nL + idx[name]] = False

        return free

    def expand_free_to_full(self, p_free: np.ndarray) -> np.ndarray:
        """Insert free parameters into a full-length vector.

        Constrained slots are filled with placeholder values that will be
        overwritten by :meth:`apply`.

        Parameters
        ----------
        p_free : np.ndarray
            Free parameter values.

        Returns
        -------
        np.ndarray
            Full parameter vector (length ``3 * n_lines``).
        """
        nL = len(self.line_names)
        full = np.zeros(3 * nL)
        mask = self.free_mask()
        full[mask] = p_free
        return self.apply(full)
