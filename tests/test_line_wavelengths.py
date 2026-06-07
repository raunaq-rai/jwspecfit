"""Guard against air/vacuum errors in the rest-frame line database.

``REST_LINES_A`` is documented as holding *vacuum* wavelengths.  A subset of
optical lines once had the air->vacuum transform applied twice (once to values
that were already in vacuum), inflating them by the refractive index of air
(n ~ 1.000279, i.e. ~+1.4 A at 5000 A / +84 km/s).  These tests pin the
wavelengths to literature vacuum values so the drift cannot silently recur.

Reference air wavelengths (lambda > 2000 A) are from the NMSU table of emission
lines (astronomy.nmsu.edu/drewski/tableofemissionlines.html), which lists air
above 2000 A and vacuum below.  Vacuum values below are those air wavelengths
converted once via the Edlen (1966) air->vacuum relation, cross-checked against
the SDSS vacuum line list.
"""

from __future__ import annotations

import numpy as np
import pytest

from jwspecfit.lines import REST_LINES_A


def _air_to_vac(lam_air: float) -> float:
    """Edlen (1966) air->vacuum conversion (as used by SDSS idlutils)."""
    s = 1e4 / lam_air
    n = 1 + 6.4328e-5 + 2.94981e-2 / (146 - s**2) + 2.5540e-4 / (41 - s**2)
    return lam_air * n


# Known vacuum wavelengths (Angstroms) for key optical lines.
EXPECTED_VACUUM = {
    "HDELTA": 4102.8996,
    "HGAMMA": 4341.6913,
    "HBETA": 4862.6910,
    "Ha": 6564.6319,
    "OIII_4363": 4364.4363,
    "OIII_4959": 4960.2949,
    "OIII_5007": 5008.2396,
    "NII_5756": 5756.1861,
    "NII_6549": 6549.8589,
    "NII_6585": 6585.2784,
    "HEI_5877": 5877.2525,
    "OII_3726": 3727.0917,
    "OII_3729": 3729.8754,
    "NeIII_3869": 3869.8568,
    "OI_6302": 6302.0464,
    "SII_6718": 6718.2942,
    "SII_6732": 6732.6681,
}


@pytest.mark.parametrize("name, vac", EXPECTED_VACUUM.items())
def test_line_matches_literature_vacuum(name: str, vac: float) -> None:
    """Each tabulated line sits within 0.05 A of its literature vacuum value."""
    assert name in REST_LINES_A
    assert REST_LINES_A[name] == pytest.approx(vac, abs=0.05)


def test_no_double_air_to_vacuum_drift() -> None:
    """No optical line is inflated by an extra air->vacuum factor (~+84 km/s).

    A double conversion shows up as code/vacuum ~ 1.000279.  Assert every
    checked line is consistent with single (or zero) conversion, not double.
    """
    for name, vac in EXPECTED_VACUUM.items():
        ratio = REST_LINES_A[name] / vac
        assert abs(ratio - 1.0) < 1e-5, (
            f"{name}: code/vacuum = {ratio:.7f} "
            f"({(ratio - 1) * 3e5:+.1f} km/s) — suspected air/vacuum error"
        )
