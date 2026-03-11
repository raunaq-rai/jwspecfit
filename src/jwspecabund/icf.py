"""Ionisation correction factors.

References
----------
Izotov, Y. I., Stasinska, G., Meynet, G., Guseva, N. G., & Thuan, T. X.
2006, A&A, 448, 955.  Equations 18--23.

Garnett, D. R., Shields, G. A., Skillman, E. D., Sagan, S. P., &
Dufour, R. J.  1997, ApJ, 489, 63.
"""

from __future__ import annotations

import numpy as np


def icf_nitrogen(O_plus: float, O_total: float) -> float:
    """ICF for nitrogen (Izotov+06 eq. 18).

    N/O = ICF_N * N+/O+, where ICF_N accounts for the contribution of
    higher ionisation states of nitrogen.

    Parameters
    ----------
    O_plus : float
        Ionic abundance O+/H+.
    O_total : float
        Total oxygen abundance O/H (= O+/H+ + O++/H+).

    Returns
    -------
    float
        ICF_N.  Returns 1.0 for the standard assumption N/O ~ N+/O+.
    """
    if O_total <= 0 or O_plus <= 0:
        return 1.0
    x = O_plus / O_total
    # Eq. 18: ICF(N) = (O/O+) * correction factor
    # For low-ionisation galaxies (x > 0.5), ICF ~ 1.
    # Standard assumption: ICF_N = 1 (widely used).
    # Izotov+06 eq. 18 gives a small correction:
    icf = -0.825 * x**2 + 0.718 * x + 0.853
    # Note: this is a fit to ICF = N_total/N+ * O+/O_total
    return max(icf, 1.0)


def icf_neon(O_plus: float, O_total: float) -> float:
    """ICF for neon (Izotov+06 eq. 19).

    Ne/O = ICF_Ne * Ne++/O++.

    Parameters
    ----------
    O_plus : float
        Ionic abundance O+/H+.
    O_total : float
        Total oxygen abundance O/H.

    Returns
    -------
    float
        ICF_Ne.
    """
    if O_total <= 0 or O_plus <= 0:
        return 1.0
    x = O_plus / O_total
    # Eq. 19
    icf = -0.385 * x**2 + 0.405 * x + 0.335
    # ICF = Ne_total / Ne++ * O++ / O_total
    # Normalised so that when x ~ 0 (fully ionised), ICF ~ 0.335 + ...
    # The actual prescription: Ne/O = ICF * Ne++/O++
    # ICF ~ 1 for typical HII regions.
    return max(icf, 0.1)


def icf_sulfur(O_plus: float, O_total: float) -> float:
    """ICF for sulfur (Izotov+06 eq. 20).

    S/O = ICF_S * (S+ + S++)/O.

    Parameters
    ----------
    O_plus : float
        Ionic abundance O+/H+.
    O_total : float
        Total oxygen abundance O/H.

    Returns
    -------
    float
        ICF_S.
    """
    if O_total <= 0 or O_plus <= 0:
        return 1.0
    x = O_plus / O_total
    # Eq. 20: accounts for S3+ not captured by optical lines
    icf = 0.013 + x * (5.986 + x * (-21.085 + x * (26.462 - x * 11.282)))
    return max(icf, 1.0)


def icf_carbon(O_plus: float, O_pp: float) -> float:
    """ICF for carbon (Garnett+1997).

    When C+ is not detected, the observed (C2+ + C3+)/O2+ underestimates
    C/O because C+ in the low-ionisation zone is missing.  The correction
    assumes C+/C_total ~ O+/O_total (similar ionisation structure):

        C/O = ICF_C * (C2+ + C3+) / O2+
        ICF_C = (O+ + O2+) / O2+

    Parameters
    ----------
    O_plus : float
        Ionic abundance O+/H+.
    O_pp : float
        Ionic abundance O++/H+.

    Returns
    -------
    float
        ICF_C.  Returns 1.0 when O+ is negligible or not detected.
    """
    if O_pp <= 0:
        return 1.0
    if O_plus <= 0:
        # No O+ detected -> ICF = 1 (high-ionisation object)
        return 1.0
    icf = (O_plus + O_pp) / O_pp
    return max(icf, 1.0)


def icf_argon(O_plus: float, O_total: float) -> float:
    """ICF for argon (Izotov+06 eqs. 22/23).

    Ar/O = ICF_Ar * Ar++/O++.

    Parameters
    ----------
    O_plus : float
        Ionic abundance O+/H+.
    O_total : float
        Total oxygen abundance O/H.

    Returns
    -------
    float
        ICF_Ar.
    """
    if O_total <= 0 or O_plus <= 0:
        return 1.0
    x = O_plus / O_total
    # Eqs. 22/23: combined ICF for Ar++ → Ar_total
    icf = 0.157 + x * (3.119 + x * (-6.185 + x * 4.517))
    return max(icf, 1.0)
