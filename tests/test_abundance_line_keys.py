"""jwspecabund must look lines up under the keys jwspecfit produces.

Regression: direct.py read ``SIII_9069`` / ``ArIII_7136`` and forward.py
``NII_6550``, but the fitter names those lines ``SIII_9071``,
``ArIII_7138`` and ``NII_6549``.  S++ and Ar++ were therefore never
computed from a real fit, and nothing failed.
"""

import re
from pathlib import Path

import pytest

from jwspecfit.lines import LEGACY_LINE_NAMES, REST_LINES_A
from jwspecabund.direct import compute_ionic_abundances

_SRC = Path(__file__).resolve().parents[1] / "src" / "jwspecabund"
# Quoted names shaped like a line key: element + ionisation + "_" + label.
_KEY = re.compile(r'"((?:H|He|HE|C|N|O|Ne|S|Si|Ar|Mg|Fe)I*V?I*\]?_[0-9]+[a-z]?)"')


@pytest.mark.parametrize("module", ["direct.py", "_core.py", "forward.py", "dust.py"])
def test_every_line_key_is_a_fitter_key(module):
    keys = set(_KEY.findall((_SRC / module).read_text()))
    unknown = sorted(k for k in keys if k not in REST_LINES_A and k not in LEGACY_LINE_NAMES)
    assert not unknown, f"{module} uses line keys the fitter never produces: {unknown}"


def test_sulfur_and_argon_from_fitter_keys():
    fluxes = {"HBETA": 100.0, "OIII_5007": 500.0, "SIII_9071": 20.0, "ArIII_7138": 8.0}
    ionic = compute_ionic_abundances(fluxes, 1.5e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=1.35e4)
    assert ionic.get("S++/H+", 0) > 0
    assert ionic.get("Ar++/H+", 0) > 0


def test_old_air_keys_still_accepted_with_warning():
    fluxes = {"HBETA": 100.0, "SIII_9069": 20.0, "ArIII_7136": 8.0}
    with pytest.warns(FutureWarning, match="SIII_9069"):
        old = compute_ionic_abundances(fluxes, 1.5e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=1.35e4)
    new = compute_ionic_abundances(
        {"HBETA": 100.0, "SIII_9071": 20.0, "ArIII_7138": 8.0},
        1.5e4, 1.2e4, 1e2, ne_mid=1e2, Te_int=1.35e4,
    )
    assert old["S++/H+"] == new["S++/H+"]
    assert old["Ar++/H+"] == new["Ar++/H+"]
