"""Every tie lives in ConstraintSet.tying_ops(); NumPy, JAX and the fitter must agree."""

import numpy as np
import pytest

from jwspecfit.constraints import ConstraintSet
from jwspecfit.lines import REST_LINES_A, get_line_list
from jwspecfit.models import build_model

_UV = [n for n in get_line_list("g140m") if REST_LINES_A[n] < 2400]
CONFIGS = [
    ("g140m", get_line_list("g140m"), {}),
    ("g395m", get_line_list("g395m"), {}),
    ("prism", get_line_list("prism"), {}),
    ("uv", _UV, {}),
    ("uv blended", _UV, dict(blended_doublets={"CIII]_1909", "NIV_1483", "CII]_2324"})),
    ("uv ratios", _UV, dict(niv_doublet_ratio=0.5, ciii_doublet_ratio=1.3)),
    ("uv no centroid tie", _UV, dict(tie_uv_centroids=False)),
]


def _random_full(names, rng):
    nL = len(names)
    lam = np.array([REST_LINES_A[n] for n in names]) * 4.0
    return np.r_[rng.uniform(1, 10, nL), lam, rng.uniform(1, 8, nL)]


@pytest.mark.parametrize("label,names,kw", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_expand_free_matches_apply_and_no_zero_width(label, names, kw):
    """A tied slot must never be left at its placeholder (σ = 0 drops the line)."""
    cs = ConstraintSet(names, **kw)
    full = _random_full(names, np.random.default_rng(0))
    expanded = cs.expand_free_to_full(full[cs.free_mask()])
    np.testing.assert_allclose(expanded, cs.apply(full), rtol=1e-14)
    assert np.all(expanded[2 * len(names):] > 0)


def test_niv_1483_width_follows_final_niv_1486_width():
    """N IV] 1486 is itself tied to C III] 1907; 1483 must copy its *final* width."""
    names = ["CIII]_1907", "CIII]_1909", "NIV_1486", "NIV_1483"]
    cs = ConstraintSet(names)
    p = cs.expand_free_to_full(np.array([1.0, 1.0, 1.0, 1.0, 7631.0, 5946.0, 3.0]))
    s1907, s1486, s1483 = p[8], p[10], p[11]
    assert s1486 == pytest.approx(s1907 * REST_LINES_A["NIV_1486"] / REST_LINES_A["CIII]_1907"])
    assert s1483 == pytest.approx(s1486 * REST_LINES_A["NIV_1483"] / REST_LINES_A["NIV_1486"])


@pytest.mark.parametrize("label,names,kw", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_jax_loglike_equals_numpy(label, names, kw):
    pytest.importorskip("jax")
    from jwspecmcmc.jax_likelihood import make_jax_log_likelihood
    from jwspecmcmc.likelihood import LikelihoodSpec, log_likelihood

    rng = np.random.default_rng(1)
    cs = ConstraintSet(names, **kw)
    full = _random_full(names, rng)
    lam = full[len(names):2 * len(names)]
    edges = np.linspace(lam.min() - 300, lam.max() + 300, 3001)
    pf = full[cs.free_mask()]
    flam = build_model(cs.expand_free_to_full(pf), edges, len(names)) + rng.normal(0, 0.05, 3000)
    spec = LikelihoodSpec(
        flam=flam, flam_err=np.full(3000, 0.05), valid=np.ones(3000, bool), edges=edges,
        n_lines=len(names), constraints=cs,
    )
    f_jax, _ = make_jax_log_likelihood(spec)
    for _ in range(3):
        q = pf * rng.uniform(0.9, 1.1, len(pf))
        assert float(f_jax(q)) == pytest.approx(log_likelihood(q, spec), rel=1e-10)


def test_likelihood_is_plain_chi2_on_nonuniform_pixels():
    """No pixel-width weight: ln L = -chi2/2 exactly, so posterior widths are not rescaled.

    A (median dlam / dlam)^0.35 weight used to multiply every residual, which
    scaled MCMC posterior widths by 1/w (up to -33%/+29% on the PRISM).
    """
    from jwspecmcmc.likelihood import LikelihoodSpec, log_likelihood

    rng = np.random.default_rng(2)
    edges = np.cumsum(np.r_[5000.0, np.geomspace(5.0, 40.0, 400)])  # 8x range of pixel widths
    names = ["HBETA"]
    cs = ConstraintSet(names)
    p = np.array([3e-17, 0.5 * (edges[0] + edges[-1]), 30.0])
    model = build_model(p, edges, 1)
    err = rng.uniform(1e-19, 3e-19, 400)
    flam = model + rng.normal(0, 1, 400) * err
    spec = LikelihoodSpec(flam=flam, flam_err=err, valid=np.ones(400, bool), edges=edges,
                          n_lines=1, constraints=cs)
    q = p * 1.01
    chi2 = np.sum(((flam - build_model(q, edges, 1)) / err) ** 2)
    assert log_likelihood(q, spec) == pytest.approx(-0.5 * chi2, rel=1e-12)


def test_tied_width_limits_and_continuity():
    """Resolved: quadrature on the intrinsic part; unresolved: instrumental ratio."""
    from jwspecfit.constraints import tied_width

    r, a_s, a_d = 1.31, 40.0**2, 32.0**2   # PRISM-like: sigma_inst ratio 0.8, lambda ratio 1.31
    # Purely instrumental source -> purely instrumental target.
    assert tied_width(40.0, r, a_s, a_d) == pytest.approx(32.0)
    # Narrower than the assumed instrument (true R higher) -> scale by sigma_inst ratio.
    assert tied_width(30.0, r, a_s, a_d) == pytest.approx(30.0 * 0.8)
    # Resolved: sigma_v shared, each line keeps its own instrument.
    sv_s = 50.0
    assert tied_width(np.hypot(sv_s, 40.0), r, a_s, a_d) == pytest.approx(np.hypot(r * sv_s, 32.0))
    # Continuous at the branch point.
    eps = 1e-9
    assert tied_width(40.0 - eps, r, a_s, a_d) == pytest.approx(tied_width(40.0 + eps, r, a_s, a_d))


def test_jax_width_ties_with_sigma_inst_match_numpy():
    pytest.importorskip("jax")
    from jwspecmcmc.jax_likelihood import make_jax_log_likelihood
    from jwspecmcmc.likelihood import LikelihoodSpec, log_likelihood

    rng = np.random.default_rng(3)
    names = ["HBETA", "OIII_4959", "OIII_5007", "HALPHA", "NII_6549", "NII_6585"]
    si = {n: 30.0 + 10 * k for k, n in enumerate(names)}
    cs = ConstraintSet(names, sigma_inst_A=si)
    full = _random_full(names, rng)
    full[2 * len(names):] = rng.uniform(20, 80, len(names))   # straddles both branches
    lam = full[len(names):2 * len(names)]
    edges = np.linspace(lam.min() - 600, lam.max() + 600, 3001)
    pf = full[cs.free_mask()]
    flam = build_model(cs.expand_free_to_full(pf), edges, len(names)) + rng.normal(0, 0.05, 3000)
    spec = LikelihoodSpec(flam=flam, flam_err=np.full(3000, 0.05), valid=np.ones(3000, bool),
                          edges=edges, n_lines=len(names), constraints=cs)
    f_jax, _ = make_jax_log_likelihood(spec)
    for _ in range(4):
        q = pf * rng.uniform(0.7, 1.3, len(pf))
        assert float(f_jax(q)) == pytest.approx(log_likelihood(q, spec), rel=1e-10)


@pytest.mark.parametrize("z", [1.5, 6.5])
def test_prism_balmer_tie_unbiased_for_unresolved_lines(z):
    """Tied to [OIII], unresolved PRISM Balmer fluxes must come back exact (noise-free).

    Scaling the observed width by lambda forced sigma(Ha)/sigma(5007) = 1.31
    against a true 0.8-0.94 and biased Ha/Hb by -14% (z=1.5) to +23% (z=6.5).
    """
    from jwspecfit.fitter import fit_lines
    from jwspecfit.io import Spectrum, _flam_to_ujy
    from jwspecfit.resolution import sigma_inst_A

    wave = np.geomspace(0.6, 5.3, 450)
    names = ["HBETA", "OIII_4959", "OIII_5007", "HALPHA"]
    lam = np.array([REST_LINES_A[n] for n in names]) * (1 + z)
    si = sigma_inst_A(wave, grating="PRISM")
    sig = np.array([si[np.argmin(abs(wave * 1e4 - x))] for x in lam])
    flux = np.array([6e-19, 9.7e-19, 2.9e-18, 1.75e-18])
    s0 = Spectrum(wave_um=wave, flux_ujy=0 * wave, err_ujy=np.full_like(wave, 0.01), grating="PRISM")
    f = 0.05 + _flam_to_ujy(build_model(np.r_[flux, lam, sig], s0.wave_edges_A, 4), wave)
    spec = Spectrum(wave_um=wave, flux_ujy=f, err_ujy=s0.err_ujy, grating="PRISM")
    res = fit_lines(spec, z, lines=names, n_boot=0, _label="t")
    for k, n in enumerate(names):
        assert res.lines[n].flux == pytest.approx(flux[k], rel=0.01, abs=0), n


def test_merged_oii_doublet_not_clipped_by_width_bound():
    """A merged [O II] blend is wider than sigma_inst; its width bound must allow that."""
    from jwspecfit.fitter import fit_lines
    from jwspecfit.io import Spectrum, _flam_to_ujy
    from jwspecfit.models import gaussian_binned

    z, R = 5.77, 2500.0     # point-source R on G235M: the blend is wider than 2 sigma_inst
    wave = np.arange(2.40, 2.65, 0.0004)
    s0 = Spectrum(wave_um=wave, flux_ujy=0 * wave, err_ujy=np.full_like(wave, 0.01),
                  grating="G235M", R=R)
    e = s0.wave_edges_A
    flam = np.zeros(len(wave))
    for name, f in (("OII_3726", 1e-18), ("OII_3729", 1e-18)):
        lam = REST_LINES_A[name] * (1 + z)
        flam += f * gaussian_binned(e[:-1], e[1:], lam, lam / (2.3548 * R))
    spec = Spectrum(wave_um=wave, flux_ujy=0.05 + _flam_to_ujy(flam, wave), err_ujy=s0.err_ujy,
                    grating="G235M", R=R)
    res = fit_lines(spec, z, lines=["OII_doublet"], n_boot=0, _label="t")
    # One Gaussian on a partly resolved blend runs ~5% high; pinned at the
    # 2 sigma_inst cap it came back 21% low.
    assert res.lines["OII_doublet"].flux == pytest.approx(2e-18, rel=0.08, abs=0)
