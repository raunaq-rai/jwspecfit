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

    Elemental correction: N/H = ICF_N * N+/H+.  To form N/O, divide the
    corrected nitrogen abundance by the total oxygen abundance, i.e.
    N/O = ICF_N * (N+/H+) / (O/H).  Izotov+06 eq. 18 is a fit in the O+
    ionisation fraction v = O+/(O+ + O++), with three metallicity branches.

    Parameters
    ----------
    O_plus : float
        Ionic abundance O+/H+.
    O_total : float
        Total oxygen abundance O/H (= O+/H+ + O++/H+).

    Returns
    -------
    float
        ICF_N (>= 1; accounts for unseen N++, N+++).
    """
    if O_total <= 0 or O_plus <= 0:
        return 1.0
    v = float(np.clip(O_plus / O_total, 1e-3, 1.0))
    oh = 12.0 + np.log10(O_total)
    if oh <= 7.2:
        icf = -0.825 * v + 0.718 + 0.853 / v  # eq. 18a (low Z)
    elif oh >= 8.2:
        icf = -1.476 * v + 1.752 + 0.688 / v  # eq. 18c (high Z)
    else:
        icf = -0.809 * v + 0.712 + 0.852 / v  # eq. 18b (intermediate Z)
    return max(icf, 1.0)


def icf_neon(O_plus: float, O_total: float) -> float:
    """ICF for neon (Izotov+06 eq. 19).

    Elemental correction: Ne/H = ICF_Ne * Ne++/H+.  To form Ne/O, divide
    the corrected Ne abundance by the total oxygen abundance, i.e.
    Ne/O = ICF_Ne * (Ne++/H+) / (O/H).

    Parameters
    ----------
    O_plus : float
        Ionic abundance O+/H+.
    O_total : float
        Total oxygen abundance O/H (= O+/H+ + O++/H+).

    Returns
    -------
    float
        ICF_Ne (>= 1; ~1 for high-ionisation HII regions).
    """
    if O_total <= 0 or O_plus <= 0:
        return 1.0
    # Elemental ICF: Ne/H = ICF * Ne++/H+.  Izotov+06 eq. 19 is a fit in
    # the O++ ionisation fraction w = O++/(O+ + O++), with three branches
    # selected by metallicity.  ICF -> 1 at high ionisation (w -> 1, all
    # oxygen doubly ionised so Ne++ captures all Ne) and rises as w falls
    # (unseen Ne+ in the low-ionisation zone).
    w = float(np.clip((O_total - O_plus) / O_total, 1e-3, 1.0))
    oh = 12.0 + np.log10(O_total)
    if oh <= 7.2:
        icf = -0.385 * w + 1.365 + 0.022 / w  # eq. 19a (low Z)
    elif oh >= 8.2:
        icf = -0.591 * w + 0.927 + 0.546 / w  # eq. 19c (high Z)
    else:
        icf = -0.405 * w + 1.382 + 0.021 / w  # eq. 19b (intermediate Z)
    return max(icf, 1.0)


def icf_sulfur(O_plus: float, O_total: float) -> float:
    """ICF for sulfur (Izotov+06 eq. 20).

    Elemental correction: S/H = ICF_S * (S+ + S++)/H+, accounting for
    unseen S+++ in the high-ionisation zone.  Izotov+06 eq. 20 is a fit
    in v = O+/(O+ + O++), with three metallicity branches.

    Parameters
    ----------
    O_plus : float
        Ionic abundance O+/H+.
    O_total : float
        Total oxygen abundance O/H (= O+/H+ + O++/H+).

    Returns
    -------
    float
        ICF_S (>= 1).
    """
    if O_total <= 0 or O_plus <= 0:
        return 1.0
    v = float(np.clip(O_plus / O_total, 1e-3, 1.0))
    oh = 12.0 + np.log10(O_total)
    if oh <= 7.2:
        icf = 0.121 * v + 0.511 + 0.161 / v  # eq. 20a (low Z)
    elif oh >= 8.2:
        icf = 0.178 * v + 0.610 + 0.153 / v  # eq. 20c (high Z)
    else:
        icf = 0.155 * v + 0.849 + 0.062 / v  # eq. 20b (intermediate Z)
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
    """ICF for argon (Izotov+06 eq. 22).

    Elemental correction from Ar++ alone: Ar/H = ICF_Ar * Ar++/H+,
    accounting for unseen Ar+ and Ar+++.  To form Ar/O, divide by the
    total oxygen abundance: Ar/O = ICF_Ar * (Ar++/H+) / (O/H).  Izotov+06
    eq. 22 is a fit in v = O+/(O+ + O++), with three metallicity branches.

    Parameters
    ----------
    O_plus : float
        Ionic abundance O+/H+.
    O_total : float
        Total oxygen abundance O/H (= O+/H+ + O++/H+).

    Returns
    -------
    float
        ICF_Ar (>= 1).
    """
    if O_total <= 0 or O_plus <= 0:
        return 1.0
    v = float(np.clip(O_plus / O_total, 1e-3, 1.0))
    oh = 12.0 + np.log10(O_total)
    if oh <= 7.2:
        icf = 0.278 * v + 0.836 + 0.121 / v  # eq. 22a (low Z)
    elif oh >= 8.2:
        icf = 0.517 * v + 0.763 + 0.042 / v  # eq. 22c (high Z)
    else:
        icf = 0.285 * v + 0.833 + 0.051 / v  # eq. 22b (intermediate Z)
    return max(icf, 1.0)
