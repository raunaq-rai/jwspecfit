"""Tests for rejection-resampling of the direct-Te Monte-Carlo loop.

When an MC draw falls outside the Martinez+2025 calibration bounds it is
rejected and re-drawn, so the number of *valid* draws still reaches n_mc.
If an object is centred outside the bounds (acceptance ~0) the loop stops
at a safety cap, warns, and pads the remainder with NaN — it never hangs.
"""

import logging
from dataclasses import dataclass

import numpy as np

from jwspecabund import compute_abundances


@dataclass
class _MockLine:
    name: str
    flux: float
    flux_err: float
    snr: float


@dataclass
class _MockFit:
    lines: dict


def _fit(line_fluxes):
    lines = {
        name: _MockLine(name, f, e, f / e if e > 0 else 0.0)
        for name, (f, e) in line_fluxes.items()
    }
    return _MockFit(lines=lines)


# A direct-Te object with [OIII]4363 (for Te) and an in-range O32 diagnostic.
_IN_RANGE = {
    "OIII_4363": (0.03, 0.005),
    "OIII_5007": (3.0, 0.05),
    "OIII_4959": (1.0, 0.02),
    "HBETA": (1.0, 0.02),
    "Ha": (2.86, 0.05),
    "NII_6585": (0.3, 0.02),
    "OII_doublet": (1.0, 0.05),   # O32 = 3.0 -> log(O32) ~ 0.48, in range
}


def test_in_range_object_fills_to_n_mc():
    """An in-range object yields a full n_mc posterior, mostly finite."""
    abund = compute_abundances(
        _fit(_IN_RANGE), z=2.0, method="direct", n_mc=100, progress=False,
    )
    assert abund.NO_posterior is not None
    assert len(abund.NO_posterior) == 100               # count preserved
    assert np.isfinite(abund.NO_posterior).sum() >= 90  # resampling kept it full


def test_out_of_range_object_caps_warns_and_pads(caplog):
    """An object centred out of range terminates at the cap and warns.

    [OII] is tiny so O32 is huge (log(O32) >> 2.5): essentially every draw
    is rejected, so the loop hits the attempt cap, warns, and pads the
    posterior to n_mc with NaN rather than hanging.
    """
    out_of_range = dict(_IN_RANGE)
    out_of_range["OII_doublet"] = (0.005, 0.001)  # O32 ~ 600 -> log ~ 2.78

    with caplog.at_level(logging.WARNING, logger="jwspecabund._core"):
        abund = compute_abundances(
            _fit(out_of_range), z=2.0, method="direct", n_mc=20, progress=False,
        )

    # Array length is still n_mc (padded), so downstream shapes are intact.
    assert abund.NO_posterior is not None
    assert len(abund.NO_posterior) == 20
    # The N/O posterior is essentially all NaN (object outside the bounds).
    assert np.isfinite(abund.NO_posterior).sum() <= 2
    # The resample cap warning fired.
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "valid draws" in msgs
