"""Tests for O²⁺ density decoupling from the N IV] high-ionisation zone.

O²⁺/H+ and the T_e(O++) solve must use the intermediate-zone density
(CIII] 1907/1909, with a low-zone fallback) rather than the noisy
high-ionisation N IV] density.  N³⁺ and C³⁺ must keep the high-zone
density so the Martinez+25 ICF 5 ((N²⁺+N³⁺)/O²⁺) uses the correct
density for each nitrogen ion.
"""

import numpy as np

from jwspecabund.direct import _ionic_abundance, compute_ionic_abundances

# Fluxes with all the ions needed to exercise the three density zones.
_FLUXES = {
    "HBETA": 100.0,
    "OIII_5007": 500.0,
    "NeIII_3869": 80.0,
    "OII_3726": 30.0,
    "OII_3729": 30.0,
    "NIV_1483": 5.0,
    "NIV_1486": 5.0,
    "NIII_1749": 3.0,
    "NIII_1752": 3.0,
}

_TE_HIGH = 18000.0
_TE_LOW = 14000.0
# Deliberately distinct so the zone a value came from is unambiguous.
_NE_LOW = 100.0
_NE_MID = 3000.0
_NE_HIGH = 1.0e5


def test_Opp_uses_intermediate_zone_density():
    """O²⁺/H+ must be computed at ne_mid, not ne_high."""
    ionic = compute_ionic_abundances(
        _FLUXES, _TE_HIGH, _TE_LOW, _NE_LOW, ne_mid=_NE_MID, ne_high=_NE_HIGH,
    )
    expected_mid = _ionic_abundance(
        "O", 3, _FLUXES["OIII_5007"], _FLUXES["HBETA"], _TE_HIGH, _NE_MID, 5007,
    )
    expected_high = _ionic_abundance(
        "O", 3, _FLUXES["OIII_5007"], _FLUXES["HBETA"], _TE_HIGH, _NE_HIGH, 5007,
    )
    assert np.isclose(ionic["O++/H+"], expected_mid, rtol=1e-6)
    # The two zones differ enough that picking the wrong one is detectable.
    assert not np.isclose(ionic["O++/H+"], expected_high, rtol=1e-3)


def test_Nepp_uses_intermediate_zone_density():
    """Ne²⁺/H+ ([NeIII] 3869) must use ne_mid, not ne_high."""
    ionic = compute_ionic_abundances(
        _FLUXES, _TE_HIGH, _TE_LOW, _NE_LOW, ne_mid=_NE_MID, ne_high=_NE_HIGH,
    )
    expected_mid = _ionic_abundance(
        "Ne", 3, _FLUXES["NeIII_3869"], _FLUXES["HBETA"], _TE_HIGH, _NE_MID, 3869,
    )
    assert np.isclose(ionic["Ne++/H+"], expected_mid, rtol=1e-6)


def test_Nppp_still_uses_high_zone_density():
    """N³⁺/H+ (N IV]) must keep the high-ionisation density (regression)."""
    ionic = compute_ionic_abundances(
        _FLUXES, _TE_HIGH, _TE_LOW, _NE_LOW, ne_mid=_NE_MID, ne_high=_NE_HIGH,
    )
    expected_high = _ionic_abundance(
        "N", 4, _FLUXES["NIV_1483"] + _FLUXES["NIV_1486"], _FLUXES["HBETA"],
        _TE_HIGH, _NE_HIGH, [1483, 1486],
    )
    assert np.isclose(ionic["N+++/H+"], expected_high, rtol=1e-6)


def test_Npp_uses_intermediate_zone_density():
    """N²⁺/H+ (N III]) must use the intermediate density and temperature.

    Since commit f97b7d3, N²⁺ sits in the intermediate zone for *both* the
    density (ne_mid) and the temperature (Te_int, defaulting to the
    midpoint 0.5*(Te_high+Te_low) when Te_int is not supplied).
    """
    ionic = compute_ionic_abundances(
        _FLUXES, _TE_HIGH, _TE_LOW, _NE_LOW, ne_mid=_NE_MID, ne_high=_NE_HIGH,
    )
    te_mid = 0.5 * (_TE_HIGH + _TE_LOW)
    expected_mid = _ionic_abundance(
        "N", 3, _FLUXES["NIII_1749"] + _FLUXES["NIII_1752"], _FLUXES["HBETA"],
        te_mid, _NE_MID, [1749, 1752],
    )
    assert np.isclose(ionic["N++/H+"], expected_mid, rtol=1e-6)


def test_Opp_falls_back_to_low_when_no_mid():
    """With ne_mid=None, O²⁺ falls back to the low-zone density."""
    ionic = compute_ionic_abundances(
        _FLUXES, _TE_HIGH, _TE_LOW, _NE_LOW, ne_mid=None, ne_high=_NE_HIGH,
    )
    expected_low = _ionic_abundance(
        "O", 3, _FLUXES["OIII_5007"], _FLUXES["HBETA"], _TE_HIGH, _NE_LOW, 5007,
    )
    assert np.isclose(ionic["O++/H+"], expected_low, rtol=1e-6)
