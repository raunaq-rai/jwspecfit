"""Regression tests: dust-correction path invariance.

Asserts that

    compute_abundances(raw_fluxes, dust_correct=True)              # path A

produces an identical AbundanceResult to

    Av  = compute_Av_multi_balmer(raw_fluxes, ...)
    f'  = dust_correct_fluxes(raw_fluxes, Av, ...)
    compute_abundances(f', dust_correct=False, Av=Av)              # path B

i.e. it doesn't matter whether the user dust-corrects up-front and feeds
dereddened fluxes in, or feeds raw fluxes and lets compute_abundances
handle dust internally.  The downstream abundance arithmetic must be
bit-for-bit (or float-noise) identical because dust correction is a
purely multiplicative per-line scaling.

Covered axes:
    * Salim and Cardelli laws
    * Av_true = 0, 0.5, 1.5
    * balmer_anchor = HBETA, Ha
    * auto-derived Av (Av=None, Path A internally derives) and
      user-supplied Av (both paths receive the same Av directly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _MockLine:
    name: str
    flux: float
    flux_err: float
    snr: float


@dataclass
class _MockResult:
    lines: dict


def _attenuate(flux_intrinsic: float, wave_A: float, Av: float, law: str,
               **dust_kwargs) -> float:
    """Apply attenuation to a single intrinsic flux at one wavelength."""
    from jwspecabund.dust import salim_attenuation, cardelli_extinction
    fn = salim_attenuation if law == "salim" else cardelli_extinction
    A_lam = fn(np.array([wave_A]), Av, **dust_kwargs)[0]
    return flux_intrinsic * 10.0 ** (-0.4 * A_lam)


def _synthetic_intrinsic_fluxes() -> dict[str, float]:
    """Intrinsic line fluxes for a typical SF galaxy with auroral [OIII] 4363.

    Hbeta = 1.0; everything else scaled by realistic ratios so the direct
    method has an auroral line, both Te zones, ne diagnostics, N+/H+ and
    O+/O++ ionic abundances.  Numbers are chosen for a moderately
    metal-poor HII region (Te ~ 1.4e4 K, ne ~ 200 cm^-3).
    """
    return {
        # Balmer ladder (Case B for Te ~ 1.4e4 K is close to standard).
        "Ha":         2.86,
        "HBETA":      1.00,
        "HGAMMA":     0.468,
        "HDELTA":     0.259,
        # [OIII] auroral and nebular.
        "OIII_4363":  0.05,   # ~5% of Hbeta -> Te-detectable
        "OIII_4959":  1.65,   # 5007/4959 = 3 by atomic physics
        "OIII_5007":  4.95,
        # [OII] doublet (low-density, ratio ~1.4 → ne ~ 200 cm^-3).
        "OII_3726":   0.85,
        "OII_3729":   1.20,
        # [NII] for N+/H+
        "NII_6585":   0.30,
    }


def _build_observed(intrinsic: dict[str, float], Av_true: float, law: str,
                    flux_err_frac: float = 0.02, **dust_kwargs):
    """Apply attenuation to every line, return (fluxes, errors)."""
    from jwspecabund._core import _LINE_WAVES

    fluxes, errors = {}, {}
    for name, f_int in intrinsic.items():
        wave = _LINE_WAVES[name]   # MUST use the same dict the abundance
                                   # code uses internally — otherwise
                                   # path A and path B see different A_lambda
                                   # at lines that have a slightly
                                   # different wavelength elsewhere
                                   # (e.g. _BALMER_LADDER).
        f_obs = _attenuate(f_int, wave, Av_true, law, **dust_kwargs)
        fluxes[name] = f_obs
        errors[name] = flux_err_frac * f_obs
    return fluxes, errors


def _make_mock_result(fluxes: dict, errors: dict) -> _MockResult:
    lines = {
        name: _MockLine(name=name, flux=f, flux_err=errors[name],
                        snr=f / errors[name] if errors[name] > 0 else 0.0)
        for name, f in fluxes.items()
    }
    return _MockResult(lines=lines)


def _extract_centrals(abund) -> dict:
    """Pull every central value we want to compare."""
    out = {
        "Av": abund.Av,
        "OH": abund.OH,
        "NO": abund.NO,
        "Te_high": abund.Te_high,
        "Te_low":  abund.Te_low,
        "ne_low":  abund.ne_low,
        "ne_mid":  abund.ne_mid,
        "ne_high": abund.ne_high,
    }
    if abund.ionic:
        # Compare every ionic abundance the direct method produced.
        for k, v in abund.ionic.items():
            out[f"ionic[{k}]"] = v
    return out


def _compare_centrals(a: dict, b: dict, *, rtol: float, atol: float):
    """Assert two central-value dicts agree on every shared key."""
    keys_a = set(a)
    keys_b = set(b)
    assert keys_a == keys_b, (
        f"Path A and B disagree on which fields exist:\n"
        f"  only-A: {keys_a - keys_b}\n  only-B: {keys_b - keys_a}"
    )
    for key in sorted(keys_a):
        va, vb = a[key], b[key]
        if va is None and vb is None:
            continue
        assert (va is None) == (vb is None), (
            f"{key}: path A = {va}, path B = {vb} (one is None)"
        )
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            np.testing.assert_allclose(
                vb, va, rtol=rtol, atol=atol,
                err_msg=f"Mismatch on {key}: A={va!r}, B={vb!r}",
            )


# ---------------------------------------------------------------------------
# The main invariance tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("law", ["salim", "cardelli"])
@pytest.mark.parametrize("Av_true", [0.0, 0.5, 1.5])
@pytest.mark.parametrize("anchor", ["HBETA", "Ha"])
def test_auto_av_invariance(law, Av_true, anchor):
    """Path A (auto-derive Av) ≡ Path B (external derive + deredden).

    Path A:  compute_abundances(raw, dust_correct=True, Av=None)
    Path B:  Av  = compute_Av_multi_balmer(raw, ...)
             f'  = dust_correct_fluxes(raw, Av, law=...)
             compute_abundances(f', dust_correct=False, Av=Av, Av_err=Av_err)

    Both must produce identical OH, NO, Te, ne, ionic abundances and Av.
    """
    from jwspecabund import compute_abundances
    from jwspecabund._core import _LINE_WAVES
    from jwspecabund.dust import compute_Av_multi_balmer, dust_correct_fluxes

    # Salim defaults match compute_abundances defaults (Rv=3.15, delta=-0.35,
    # B_bump=2.27).  Cardelli takes no extra args.
    if law == "salim":
        dust_kwargs = {"Rv": 3.15, "delta": -0.35, "B": 2.27}
    else:
        dust_kwargs = {}

    intrinsic = _synthetic_intrinsic_fluxes()
    fluxes_obs, errors_obs = _build_observed(
        intrinsic, Av_true, law, **dust_kwargs,
    )

    # ---- Path A: feed raw fluxes, let compute_abundances handle dust ----
    result_obs = _make_mock_result(fluxes_obs, errors_obs)
    abund_A = compute_abundances(
        result_obs, z=2.0, method="direct",
        dust_correct=True, dust_law=law, balmer_anchor=anchor,
        n_mc=200, progress=False,
    )

    # ---- Path B: external derivation + dereddening, then no internal dust ----
    bal = compute_Av_multi_balmer(
        fluxes_obs, errors_obs, law=law, snr_min=3.0, anchor=anchor,
        **dust_kwargs,
    )
    Av_ext = bal["Av"]
    Av_ext_err = bal["Av_err"]

    # Build the line-data dict expected by dust_correct_fluxes:
    # {name: (flux, err, wave_A)}.  Use the SAME wavelength source as
    # _apply_dust_correction inside compute_abundances.
    line_data = {
        name: (fluxes_obs[name], errors_obs[name], _LINE_WAVES[name])
        for name in fluxes_obs
    }
    corrected = dust_correct_fluxes(line_data, Av_ext, law=law, **dust_kwargs)
    fluxes_dered = {n: v[0] for n, v in corrected.items()}
    errors_dered = {n: v[1] for n, v in corrected.items()}

    result_dered = _make_mock_result(fluxes_dered, errors_dered)
    abund_B = compute_abundances(
        result_dered, z=2.0, method="direct",
        dust_correct=False, dust_law=law,
        Av=Av_ext, Av_err=Av_ext_err if Av_ext_err > 0 else None,
        balmer_anchor=anchor,
        n_mc=200, progress=False,
    )

    # ---- Compare ----
    centrals_A = _extract_centrals(abund_A)
    centrals_B = _extract_centrals(abund_B)

    # Av must match the input we externally derived (both paths use the
    # same compute_Av_multi_balmer on the same raw fluxes).
    assert centrals_A["Av"] is not None
    assert centrals_B["Av"] is not None
    np.testing.assert_allclose(centrals_A["Av"], Av_ext, rtol=0, atol=1e-12)
    np.testing.assert_allclose(centrals_B["Av"], Av_ext, rtol=0, atol=1e-12)

    # Every central value should match to floating-point noise.  The two
    # paths perform the same multiplicative scaling and feed the same
    # (dereddened, errored) numbers into the same downstream code with
    # the same RNG seed (=42), so even MC-derived centrals should agree.
    _compare_centrals(centrals_A, centrals_B, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("law", ["salim", "cardelli"])
@pytest.mark.parametrize("Av_user", [0.0, 0.7, 2.0])
def test_user_supplied_av_invariance(law, Av_user):
    """When Av is supplied directly, the same invariance must hold.

    Path A: compute_abundances(raw, dust_correct=True, Av=Av_user)
    Path B: deredden externally with Av_user, then dust_correct=False.

    No Balmer derivation involved here, so this isolates the *application*
    of dust correction from its derivation.
    """
    from jwspecabund import compute_abundances
    from jwspecabund._core import _LINE_WAVES
    from jwspecabund.dust import dust_correct_fluxes

    if law == "salim":
        dust_kwargs = {"Rv": 3.15, "delta": -0.35, "B": 2.27}
    else:
        dust_kwargs = {}

    intrinsic = _synthetic_intrinsic_fluxes()
    fluxes_obs, errors_obs = _build_observed(
        intrinsic, Av_user, law, **dust_kwargs,
    )

    # Path A — let compute_abundances apply dust given Av_user.
    result_obs = _make_mock_result(fluxes_obs, errors_obs)
    abund_A = compute_abundances(
        result_obs, z=2.0, method="direct",
        dust_correct=True, dust_law=law,
        Av=Av_user, Av_err=None,
        n_mc=200, progress=False,
    )

    # Path B — externally deredden with the same Av_user.
    line_data = {
        name: (fluxes_obs[name], errors_obs[name], _LINE_WAVES[name])
        for name in fluxes_obs
    }
    corrected = dust_correct_fluxes(line_data, Av_user, law=law, **dust_kwargs)
    fluxes_dered = {n: v[0] for n, v in corrected.items()}
    errors_dered = {n: v[1] for n, v in corrected.items()}

    result_dered = _make_mock_result(fluxes_dered, errors_dered)
    abund_B = compute_abundances(
        result_dered, z=2.0, method="direct",
        dust_correct=False, dust_law=law,
        Av=Av_user, Av_err=None,
        n_mc=200, progress=False,
    )

    centrals_A = _extract_centrals(abund_A)
    centrals_B = _extract_centrals(abund_B)
    _compare_centrals(centrals_A, centrals_B, rtol=1e-10, atol=1e-12)


def test_av_zero_is_a_perfect_noop():
    """Av=0 must leave fluxes literally untouched."""
    from jwspecabund._core import _LINE_WAVES
    from jwspecabund.dust import dust_correct_fluxes, salim_attenuation

    intrinsic = _synthetic_intrinsic_fluxes()
    line_data = {
        name: (f, 0.02 * f, _LINE_WAVES[name]) for name, f in intrinsic.items()
    }

    out = dust_correct_fluxes(line_data, Av=0.0, law="salim",
                              Rv=3.15, delta=-0.35, B=2.27)

    for name, (flux_in, err_in, _wave) in line_data.items():
        flux_out, err_out = out[name]
        np.testing.assert_allclose(flux_out, flux_in, rtol=0, atol=0)
        np.testing.assert_allclose(err_out, err_in, rtol=0, atol=0)

    # Sanity-check the underlying attenuation curve.
    waves = np.array([3000.0, 5000.0, 7000.0, 10000.0])
    A = salim_attenuation(waves, Av=0.0, Rv=3.15, delta=-0.35, B=2.27)
    np.testing.assert_allclose(A, 0.0, atol=1e-15)


def test_dereddening_recovers_intrinsic_fluxes():
    """Round-trip: attenuate, then deredden with the same Av, recover input.

    This is the algebraic identity that the invariance test relies on.
    If this fails, the other tests are testing nothing.
    """
    from jwspecabund._core import _LINE_WAVES
    from jwspecabund.dust import dust_correct_fluxes

    Av = 1.2
    law = "salim"
    dk = {"Rv": 3.15, "delta": -0.35, "B": 2.27}

    intrinsic = _synthetic_intrinsic_fluxes()
    fluxes_obs, errors_obs = _build_observed(intrinsic, Av, law, **dk)

    line_data = {
        name: (fluxes_obs[name], errors_obs[name], _LINE_WAVES[name])
        for name in fluxes_obs
    }
    corrected = dust_correct_fluxes(line_data, Av, law=law, **dk)

    for name, f_int in intrinsic.items():
        f_back = corrected[name][0]
        np.testing.assert_allclose(f_back, f_int, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Real-data sanity check
# ---------------------------------------------------------------------------
#
# Synthetic tests above prove the algebraic invariance.  This test exercises
# the same property on a real FitResult produced by jwspecfit.fit_lines on
# the public stack data/stack_all_Muv19_21.npz (a rest-frame composite).
# It catches any bug where _extract_fluxes, _extract_posteriors, or downstream
# code reaches a different branch when fed a real result vs synthetic one.

@pytest.fixture(scope="module")
def stack_optical_fit():
    """Fit optical emission lines on the (un-dust-corrected) stack.

    Uses moderate bootstrap n_boot=50 to keep runtime down — central values
    don't depend on n_boot, only the flux uncertainties.
    """
    import jwspecfit

    from tests.conftest import DATA_DIR
    npz_path = DATA_DIR / "stack_all_Muv19_21.npz"
    spec = jwspecfit.read_npz(npz_path, z=0.0)

    optical_lines = [
        "OII_doublet", "HEPSILON", "HDELTA", "HGAMMA",
        "OIII_4363", "OIII_4959", "OIII_5007",
        "HBETA", "Ha", "NII_6585",
    ]
    return jwspecfit.fit_lines(
        spec, z=0.0, lines=optical_lines,
        wave_range_A=(3600, 6800),
        moving_average=100, n_boot=50, mode="off",
    )


@pytest.mark.parametrize("law", ["salim", "cardelli"])
def test_real_data_dust_invariance(stack_optical_fit, law):
    """End-to-end: feeding the real FitResult through path A and path B.

    Tolerance is looser than the synthetic test (rtol=1e-9) to absorb any
    floating-point reordering when the real result also goes through
    _extract_fluxes, broad-component summing, etc.  The dust correction
    itself is still exact, so values must agree to ~9–10 decimal places.
    """
    from jwspecabund import compute_abundances
    from jwspecabund._core import _LINE_WAVES, _extract_fluxes
    from jwspecabund.dust import compute_Av_multi_balmer, dust_correct_fluxes

    if law == "salim":
        dust_kwargs = {"Rv": 3.15, "delta": -0.35, "B": 2.27}
    else:
        dust_kwargs = {}

    # ---- Path A: feed the real FitResult and let compute_abundances
    #              derive Av and apply dust internally. ----
    abund_A = compute_abundances(
        stack_optical_fit, z=0.0, method="auto",
        dust_correct=True, dust_law=law,
        balmer_anchor="HBETA", n_mc=200, progress=False,
    )

    # ---- Path B: extract the same fluxes _extract_fluxes would,
    #              derive Av externally, deredden, then call with
    #              dust_correct=False. ----
    fluxes, errors, _ = _extract_fluxes(stack_optical_fit)
    bal = compute_Av_multi_balmer(
        fluxes, errors, law=law, snr_min=3.0, anchor="HBETA",
        **dust_kwargs,
    )
    Av_ext = bal["Av"]
    Av_ext_err = bal["Av_err"]

    line_data = {
        name: (fluxes[name], errors[name], _LINE_WAVES[name])
        for name in fluxes if name in _LINE_WAVES
    }
    corrected = dust_correct_fluxes(line_data, Av_ext, law=law, **dust_kwargs)

    # Build a mock FitResult that mirrors _extract_fluxes' output so path B
    # sees an identical set of lines and uncertainties.
    fluxes_dered = {n: v[0] for n, v in corrected.items()}
    errors_dered = {n: v[1] for n, v in corrected.items()}
    result_dered = _make_mock_result(fluxes_dered, errors_dered)

    abund_B = compute_abundances(
        result_dered, z=0.0, method="auto",
        dust_correct=False, dust_law=law,
        Av=Av_ext, Av_err=Av_ext_err if Av_ext_err > 0 else None,
        balmer_anchor="HBETA", n_mc=200, progress=False,
    )

    # ---- Sanity: the test is only meaningful if the stack actually
    # produced a non-trivial Av and reached the direct method. ----
    assert abund_A.Av is not None and abund_A.Av > 0, (
        "Stack produced Av <= 0; real-data test loses its bite. "
        f"Got Av={abund_A.Av}."
    )
    assert abund_A.method == "direct", (
        f"Expected direct method on this stack; got {abund_A.method}."
    )

    centrals_A = _extract_centrals(abund_A)
    centrals_B = _extract_centrals(abund_B)
    _compare_centrals(centrals_A, centrals_B, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("law", ["salim", "cardelli"])
def test_external_deredden_then_av_zero(stack_optical_fit, law):
    """User-workflow test: deredden externally, then compute_abundances(Av=0).

    This is the workflow a typical user is most likely to write by hand:

        1. fit emission lines
        2. compute Av from Balmer decrement
        3. dust-correct ALL line fluxes manually
        4. compute_abundances on the dereddened fluxes, telling the
           function Av=0 (i.e. "I've already handled dust").

    Path A is the canonical one: hand the raw FitResult to
    compute_abundances and let it auto-derive and apply Av internally.
    Both must produce identical OH, NO, Te, ne, ionic abundances.

    Note: the only field that is *expected* to differ is `result.Av`
    itself — Path A reports the derived value, Path B reports 0.0
    because that is what we told it.  The downstream chemistry must
    nevertheless be identical because both paths feed the same
    (dereddened) numbers into the same downstream code with the same
    RNG seed.
    """
    from jwspecabund import compute_abundances
    from jwspecabund._core import _LINE_WAVES, _extract_fluxes
    from jwspecabund.dust import compute_Av_multi_balmer, dust_correct_fluxes

    if law == "salim":
        dust_kwargs = {"Rv": 3.15, "delta": -0.35, "B": 2.27}
    else:
        dust_kwargs = {}

    # ---- Path A: fit → compute_abundances handles dust internally ----
    abund_A = compute_abundances(
        stack_optical_fit, z=0.0, method="auto",
        dust_correct=True, dust_law=law,
        balmer_anchor="HBETA", n_mc=200, progress=False,
    )

    # ---- Path B: fit → external Balmer Av → external deredden →
    #              compute_abundances with Av=0 ("I've already done it") ----
    fluxes, errors, _ = _extract_fluxes(stack_optical_fit)
    bal = compute_Av_multi_balmer(
        fluxes, errors, law=law, snr_min=3.0, anchor="HBETA", **dust_kwargs,
    )
    Av_balmer = bal["Av"]

    line_data = {
        name: (fluxes[name], errors[name], _LINE_WAVES[name])
        for name in fluxes if name in _LINE_WAVES
    }
    corrected = dust_correct_fluxes(line_data, Av_balmer, law=law, **dust_kwargs)
    fluxes_dered = {n: v[0] for n, v in corrected.items()}
    errors_dered = {n: v[1] for n, v in corrected.items()}
    result_dered = _make_mock_result(fluxes_dered, errors_dered)

    # We dereddened the fluxes ourselves; tell compute_abundances Av=0
    # so it does not double-correct.
    abund_B = compute_abundances(
        result_dered, z=0.0, method="auto",
        dust_correct=True, dust_law=law,
        Av=0.0, Av_err=None,
        balmer_anchor="HBETA", n_mc=200, progress=False,
    )

    # ---- Sanity guards: real test must trigger non-trivial paths ----
    assert abund_A.method == "direct" and abund_B.method == "direct", (
        f"Expected direct method on stack; got A={abund_A.method}, "
        f"B={abund_B.method}."
    )
    assert abund_A.Av is not None and abund_A.Av > 0, (
        f"Stack produced Av <= 0; test loses its bite. Got {abund_A.Av}."
    )
    # Path B's reported Av is whatever we passed in (0.0), by design.
    assert abund_B.Av == 0.0

    # ---- Compare every other central value ----
    centrals_A = _extract_centrals(abund_A)
    centrals_B = _extract_centrals(abund_B)
    # Av is *expected* to differ here (A: derived, B: forced 0); strip it
    # before invoking the dict comparator.
    centrals_A.pop("Av", None)
    centrals_B.pop("Av", None)
    _compare_centrals(centrals_A, centrals_B, rtol=1e-9, atol=1e-12)
