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
    """

    line_names: list[str]
    tie_nii: bool = True
    tie_balmer_to_oiii: bool = True

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

        # --- Balmer width tying to [OIII] 5007 (must run BEFORE NII tying) ---
        if self.tie_balmer_to_oiii and "OIII_5007" in idx:
            i_o3 = idx["OIII_5007"]
            lam_o3 = REST_LINES_A["OIII_5007"]
            sigma_o3 = p[2 * nL + i_o3]

            # Lines to tie (narrow Balmer + [NII])
            tie_targets = ["HBETA", "H⍺", "HDELTA", "HGAMMA", "NII_6585"]
            for name in tie_targets:
                if name in idx:
                    i_t = idx[name]
                    lam_t = REST_LINES_A[name]
                    ratio = lam_t / lam_o3
                    p[2 * nL + i_t] = sigma_o3 * ratio

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
            tie_targets = ["HBETA", "H⍺", "HDELTA", "HGAMMA", "NII_6585"]
            for name in tie_targets:
                if name in idx:
                    # Width is tied (sigma slot)
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
