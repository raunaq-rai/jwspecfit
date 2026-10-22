"""External continuum: ``continuum=`` replaces the internal continuum fit.

Covers:
    1. MLE fitter: the supplied continuum is used verbatim and line fluxes are
       recovered on a continuum the polynomial cannot follow.
    2. Shape mismatch raises.
    3. MCMC engine: the supplied continuum and its error reach the result.
"""

from __future__ import annotations

import numpy as np
import pytest

from jwspecfit.fitter import fit_lines
from jwspecfit.io import Spectrum

from .test_uv_doublets import _make_uv_spectrum

Z, F_CIV1, F_CIII = 3.0, 3.0e-17, 4.0e-17
INJECT = {"CIV_1": F_CIV1, "CIII]_1907": F_CIII}


def _bumpy_spectrum() -> tuple[Spectrum, np.ndarray]:
    """Lines on a continuum with a sharp step that a quadratic cannot fit."""
    base = _make_uv_spectrum(z=Z, lines_to_inject=INJECT, R=1000.0, noise_level=0.002)
    lam = base.wave_um * 1e4
    cont = 0.5 + 0.4 * np.tanh((lam - 1700.0 * (1 + Z)) / 40.0)
    spec = Spectrum(wave_um=base.wave_um, flux_ujy=base.flux_ujy - 0.5 + cont,
                    err_ujy=base.err_ujy, grating=base.grating, z=base.z, R=base.R)
    return spec, cont


def test_mle_uses_supplied_continuum():
    spec, cont = _bumpy_spectrum()
    res = fit_lines(spec, z=Z, R=1000.0, lines=list(INJECT), n_boot=0, continuum=cont)
    assert res.success
    np.testing.assert_allclose(res.continuum, cont)
    assert res.lines["CIV_1"].flux / F_CIV1 == pytest.approx(1.0, abs=0.05)
    assert res.lines["CIII]_1907"].flux / F_CIII == pytest.approx(1.0, abs=0.05)


def test_mle_shape_mismatch_raises():
    spec, cont = _bumpy_spectrum()
    with pytest.raises(ValueError, match="continuum has shape"):
        fit_lines(spec, z=Z, R=1000.0, lines=list(INJECT), n_boot=0, continuum=cont[:-1])


def test_mcmc_uses_supplied_continuum_and_error():
    import jwspecmcmc

    spec, cont = _bumpy_spectrum()
    err = np.full_like(cont, 0.01)
    res = jwspecmcmc.fit_lines(
        spec, z=Z, R=1000.0, lines=list(INJECT), sampler="emcee",
        n_walkers=16, n_steps=300, n_burn=100, progress=False,
        continuum=cont, continuum_err=err,
    )
    np.testing.assert_allclose(res.continuum, cont)
    assert res.lines["CIII]_1907"].flux / F_CIII == pytest.approx(1.0, abs=0.1)
