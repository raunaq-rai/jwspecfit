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
        n_lines=len(names), constraints=cs, w_pix=np.ones(3000),
    )
    f_jax, _ = make_jax_log_likelihood(spec)
    for _ in range(3):
        q = pf * rng.uniform(0.9, 1.1, len(pf))
        assert float(f_jax(q)) == pytest.approx(log_likelihood(q, spec), rel=1e-10)
