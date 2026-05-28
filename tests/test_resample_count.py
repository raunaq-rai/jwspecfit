"""Tests for rejection-resampling of the direct-Te Monte-Carlo loop.

Only N/O is gated on the Martinez+2025 calibration bounds: out-of-bounds
draws have their N/O rejected (NaN) and another draw is taken, so the
number of *in-bounds* N/O draws still reaches n_mc.  O/H, Te and the other
X/O ratios do not use the Martinez ICF, so they are recorded for every
drawn sample and remain fully sampled even when N/O is rejected.  An
object centred outside the bounds stops at a safety cap and warns; it
never hangs.
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
    assert len(abund.NO_posterior) == 100               # in-bounds target met
    assert np.isfinite(abund.NO_posterior).sum() >= 90  # resampling kept it full
    assert np.isfinite(abund.OH_posterior).sum() >= 90


def test_out_of_range_object_keeps_OH_rejects_NO(caplog):
    """An object centred out of range: O/H survives, only N/O is rejected.

    [OII] is tiny so O32 is huge (log(O32) >> 2.5): essentially every draw
    is out of the Martinez bounds.  N/O is therefore (almost) all NaN and
    the loop hits its safety cap and warns, but O/H is computed from the
    direct Te (independent of the Martinez ICF) and stays finite — so its
    posterior is still well sampled, with more finite draws than N/O.
    """
    out_of_range = dict(_IN_RANGE)
    out_of_range["OII_doublet"] = (0.005, 0.001)  # O32 ~ 600 -> log ~ 2.78

    with caplog.at_level(logging.WARNING, logger="jwspecabund._core"):
        abund = compute_abundances(
            _fit(out_of_range), z=2.0, method="direct", n_mc=20, progress=False,
        )

    n_oh = np.isfinite(abund.OH_posterior).sum()
    n_no = np.isfinite(abund.NO_posterior).sum()

    # O/H is recovered (decoupled from the Martinez bounds)...
    assert abund.OH is not None and np.isfinite(abund.OH)
    assert n_oh >= 20
    # ...while N/O is essentially all rejected.
    assert n_no <= 2
    # O/H has strictly more finite draws than N/O.
    assert n_oh > n_no
    # The under-sampling warning fired and names N/O specifically.
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "in-bounds N/O" in msgs
