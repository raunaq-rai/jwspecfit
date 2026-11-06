"""Upper limits from the error spectrum at the line, times a measured factor.

These pin the properties a 3 sigma upper limit must have, measured over many
noise realisations: it follows the noise at the line's own position (a
combined spectrum changes noise level at each grating boundary), it picks up
pixel correlation the error array does not know about, and pure noise
exceeds it about as rarely as 3 sigma promises.  The previous estimator (the
residual scatter in a +/-12 sigma annulus, kept as the no-error fallback)
was ~5% low with +/-25% scatter, so noise beat its "3 sigma" ~1% of the time.
"""

import numpy as np
import pytest

from jwspecfit.limits import _gaussian_templates, matched_filter_flux_error

WAVE = np.linspace(30000.0, 50000.0, 2400)  # Angstrom, ~8.3 A pixels
SIG = 17.0                                  # assumed line sigma, ~2 pixels
OK = np.ones(WAVE.size, dtype=bool)
LEVEL = np.where(WAVE < 40000.0, 1.0, 4.0)  # noise jumps x4 at 4.0 um
N_REAL = 3000


def _noise(correlated_above=False, correlated_everywhere=False, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N_REAL, WAVE.size))
    k = np.array([0.25, 0.5, 0.25])
    xc = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, x)
    xc /= np.sqrt(np.sum(k**2))
    if correlated_everywhere:
        x = xc
    elif correlated_above:
        x = np.where(WAVE < 40000.0, x, xc)
    return x * LEVEL


def _true_and_estimates(R, lam, err=LEVEL):
    """Real scatter of the flux at the line, and the estimate per realisation."""
    g = _gaussian_templates(WAVE, np.array([lam]), SIG, OK)[0]
    flux = R @ g / (g @ g)
    est = np.array([
        matched_filter_flux_error(WAVE, r, OK, lam, SIG, err=err).sigma_flux
        for r in R
    ])
    return flux, est


@pytest.fixture(scope="module")
def correlated_above():
    return _noise(correlated_above=True, seed=1)


def test_follows_the_noise_level_at_the_line():
    R = _noise(seed=2)
    _, below = _true_and_estimates(R[:200], 39500.0)
    _, above = _true_and_estimates(R[:200], 40500.0)
    # 0.05 um either side of the x4 jump, each limit sees its own grating.
    assert np.median(above) / np.median(below) == pytest.approx(4.0, rel=0.1)


def test_picks_up_correlation_the_error_array_misses():
    R = _noise(correlated_everywhere=True, seed=3)
    flux, est = _true_and_estimates(R[:400], 36000.0)
    white = matched_filter_flux_error(WAVE, np.zeros(WAVE.size), OK, 36000.0, SIG,
                                      err=LEVEL)
    true = flux.std()
    assert white.method == "error_spectrum"
    assert np.median(est) / true == pytest.approx(1.0, abs=0.12)
    # The error array alone would have been ~0.63x too optimistic here.
    g = _gaussian_templates(WAVE, np.array([36000.0]), SIG, OK)[0]
    assert np.sqrt(g**2 @ LEVEL**2) / (g @ g) / true < 0.7


@pytest.mark.parametrize("lam", [36000.0, 39500.0, 40500.0, 44000.0])
def test_noise_rarely_beats_three_sigma(correlated_above, lam):
    """Pure noise must exceed the 3 sigma limit about as rarely as promised:
    0.13% for an exact sigma; the estimate's own scatter adds a little."""
    flux, est = _true_and_estimates(correlated_above, lam)
    assert np.mean(flux > 3.0 * est) < 0.006
    # ...and never by being biased low, including next to the boundary
    # where the correlation changes.
    assert np.median(est) / flux.std() > 0.95


def test_without_an_error_spectrum_the_annulus_estimator_is_used():
    R = _noise(seed=4)[0]
    noise = matched_filter_flux_error(WAVE, R, OK, 36000.0, SIG)
    assert noise.method == "matched_filter"
    assert noise.corr_factor == 1.0
