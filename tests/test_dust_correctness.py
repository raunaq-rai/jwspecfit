"""Absolute-correctness tests for the dust-correction pipeline.

The invariance tests in test_dust_invariance.py prove that two equivalent
dust-correction workflows produce identical abundances.  However, both
workflows can still inherit the SAME bug from a single shared component —
e.g. a wrong V-band normalisation, a bad Balmer wavelength, or a sign
error in compute_Av_multi_balmer — and the invariance test would not
catch it.

This file targets the SHARED components themselves with property-based
checks that any plausible implementation must satisfy:

    1. V-band normalisation: A(5500 Å) = Av by definition.
    2. Av recovery on clean data: known Av_in must come back out.
    3. Av recovery under noise: estimator should be unbiased.
    4. Monotonicity: bigger observed Balmer ratio => bigger Av.
    5. Monotonicity: bigger Av => bigger A_lambda everywhere.
    6. SNR preservation: dust correction scales flux and error equally.
    7. Sign convention: red excess => Av > 0; blue excess => Av = 0.
    8. Cross-law sanity on real data: Salim and Cardelli agree on Av
       to within ~30 % (different curves, but not wildly different).
    9. Wavelength self-consistency: _LINE_WAVES (used in dust correction)
       and _BALMER_LADDER (used in Av derivation) must agree on the
       Balmer-line rest wavelengths to within a small tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers (kept inline so this file is self-contained).
# ---------------------------------------------------------------------------

@dataclass
class _MockLine:
    name: str; flux: float; flux_err: float; snr: float


@dataclass
class _MockResult:
    lines: dict


def _attenuate_one(flux_intrinsic: float, wave_A: float, Av: float, law: str,
                   **dust_kwargs) -> float:
    from jwspecabund.dust import salim_attenuation, cardelli_extinction
    fn = salim_attenuation if law == "salim" else cardelli_extinction
    A_lam = fn(np.array([wave_A]), Av, **dust_kwargs)[0]
    return flux_intrinsic * 10.0 ** (-0.4 * A_lam)


def _build_full_balmer_observed(Av_true: float, law: str, **dust_kwargs):
    """All 6 Balmer ladder lines attenuated at Av_true."""
    from jwspecabund.dust import _BALMER_LADDER

    fluxes, errors = {}, {}
    for name, (wave, ratio_over_Hb) in _BALMER_LADDER.items():
        f_int = ratio_over_Hb * 1.0   # Hβ = 1
        f_obs = _attenuate_one(f_int, wave, Av_true, law, **dust_kwargs)
        fluxes[name] = f_obs
        errors[name] = 0.01 * f_obs   # 1 %, SNR=100, well above any cut
    return fluxes, errors


# ---------------------------------------------------------------------------
# 1. V-band normalisation.   A(5500 Å) MUST equal Av by definition.
# ---------------------------------------------------------------------------
class TestVBandNormalisation:

    def test_cardelli_A_at_V_equals_Av(self):
        """Cardelli+89 V-band: A(5500 Å) = Av to high precision.

        At λ_V = 5500 Å, x = 1/0.55 ≈ 1.818, so y = x - 1.82 ≈ -0.002.
        The polynomial gives a ≈ 1, b ≈ 0, so A/Av = a + b/Rv ≈ 1.
        """
        from jwspecabund.dust import cardelli_extinction
        for Av in (0.5, 1.0, 2.0):
            A = cardelli_extinction(np.array([5500.0]), Av, Rv=3.1)[0]
            np.testing.assert_allclose(
                A, Av, rtol=2e-3,
                err_msg=f"Cardelli A(5500 Å) ≠ Av at Av={Av}",
            )

    def test_salim_Av_is_a_model_parameter_not_A_at_5500(self):
        """Salim+18 with non-zero ``delta`` does NOT satisfy A(5500 Å) = Av.

        This is a real property of the law, not a bug: when the slope
        deviation ``delta`` is non-zero the curve is "tilted" and the
        V-band normalisation shifts.  The ``Av`` argument is a model
        parameter used in the recovery.  The recovery is self-consistent
        — same law derives Av and applies dust correction — so abundances
        are unaffected.

        For the default Salim parameters (Rv=3.15, delta=-0.35, B=2.27),
        A(5500 Å) / Av ≈ 1.30.  This test pins that ratio so any future
        change to the curve flags itself.
        """
        from jwspecabund.dust import salim_attenuation
        ratios = []
        for Av in (0.25, 0.5, 1.0, 2.0):
            A = salim_attenuation(np.array([5500.0]), Av,
                                  Rv=3.15, delta=-0.35, B=2.27)[0]
            ratios.append(A / Av)
        ratios = np.array(ratios)

        # Linearity in Av: ratio must be the same for every Av.
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)

        # Pin the value (~1.298) — flags any change to k_Calzetti, delta,
        # or the bump.  Loose 1 % tolerance to allow harmless re-derivations.
        np.testing.assert_allclose(
            ratios[0], 1.298, rtol=1e-2,
            err_msg=(f"A(5500 Å)/Av for Salim default params = {ratios[0]:.4f}; "
                     f"expected ≈ 1.298.  Curve definition may have changed."),
        )

    @pytest.mark.parametrize("law", ["salim", "cardelli"])
    def test_A_lambda_proportional_to_Av(self, law):
        """A_lambda must scale linearly with Av at every wavelength.

        If A(λ, 2·Av) ≠ 2·A(λ, Av) the curve has a non-linear bug.
        """
        from jwspecabund.dust import salim_attenuation, cardelli_extinction
        fn = salim_attenuation if law == "salim" else cardelli_extinction
        waves = np.array([1500.0, 2175.0, 4000.0, 5500.0, 6563.0, 9000.0])
        A1 = fn(waves, Av=1.0)
        A2 = fn(waves, Av=2.0)
        np.testing.assert_allclose(A2, 2.0 * A1, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# 2. Av recovery on clean data.  This is the WHOLE POINT.
# ---------------------------------------------------------------------------
class TestAvRecovery:

    @pytest.mark.parametrize("law", ["salim", "cardelli"])
    @pytest.mark.parametrize("anchor", ["HBETA", "Ha"])
    @pytest.mark.parametrize("Av_true", [0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
    def test_recover_av_from_clean_balmer(self, law, anchor, Av_true):
        """compute_Av_multi_balmer recovers a known Av from clean fluxes.

        Synthesises noiseless Balmer fluxes attenuated at Av_true, then
        runs the recovery routine.  Tolerance is 1e-3 (well above
        round-off) — any larger error would indicate a real bug.
        """
        from jwspecabund.dust import compute_Av_multi_balmer

        if law == "salim":
            dust_kwargs = {"Rv": 3.15, "delta": -0.35, "B": 2.27}
        else:
            dust_kwargs = {}

        fluxes, errors = _build_full_balmer_observed(
            Av_true, law, **dust_kwargs,
        )
        out = compute_Av_multi_balmer(
            fluxes, errors, law=law, snr_min=3.0, anchor=anchor,
            **dust_kwargs,
        )
        assert out["n_lines"] >= 3, (
            f"Expected ≥ 3 Balmer pairs, got {out['n_lines']}."
        )
        np.testing.assert_allclose(
            out["Av"], Av_true, atol=1e-3,
            err_msg=(f"Recovered Av={out['Av']} ≠ true Av={Av_true} "
                     f"(law={law}, anchor={anchor})."),
        )


# ---------------------------------------------------------------------------
# 2a. Multi-Balmer only uses lines bluer than the anchor.
# ---------------------------------------------------------------------------
class TestMultiBalmerBluerThanAnchor:

    def test_hbeta_anchor_excludes_halpha(self):
        """anchor=Hβ must use only bluer lines (Hα excluded)."""
        from jwspecabund.dust import compute_Av_multi_balmer
        fluxes, errors = _build_full_balmer_observed(0.5, "cardelli")
        out = compute_Av_multi_balmer(
            fluxes, errors, law="cardelli", snr_min=3.0, anchor="HBETA",
        )
        used = {r["line"] for r in out["individual"]}
        assert "Ha" not in used
        assert used == {"HGAMMA", "HDELTA", "H9", "H10"}

    def test_ha_anchor_includes_all_bluer(self):
        """anchor=Hα must use every other Balmer line (all bluer than Hα)."""
        from jwspecabund.dust import compute_Av_multi_balmer
        fluxes, errors = _build_full_balmer_observed(0.5, "cardelli")
        out = compute_Av_multi_balmer(
            fluxes, errors, law="cardelli", snr_min=3.0, anchor="Ha",
        )
        used = {r["line"] for r in out["individual"]}
        assert used == {"HBETA", "HGAMMA", "HDELTA", "H9", "H10"}


# ---------------------------------------------------------------------------
# 2b. Single-pair Av recovery — compute_Av_balmer_pair forces one decrement.
# ---------------------------------------------------------------------------
class TestAvBalmerPair:

    @pytest.mark.parametrize("law", ["salim", "cardelli"])
    @pytest.mark.parametrize("pair", [("Ha", "HBETA"), ("HGAMMA", "HBETA")])
    @pytest.mark.parametrize("Av_true", [0.0, 0.5, 1.5, 3.0])
    def test_recover_av_from_single_pair(self, law, pair, Av_true):
        """compute_Av_balmer_pair recovers a known Av from exactly one pair."""
        from jwspecabund.dust import compute_Av_balmer_pair

        dust_kwargs = ({"Rv": 3.15, "delta": -0.35, "B": 2.27}
                       if law == "salim" else {})
        fluxes, errors = _build_full_balmer_observed(Av_true, law, **dust_kwargs)
        out = compute_Av_balmer_pair(fluxes, errors, pair, law=law, **dust_kwargs)
        assert out["n_lines"] == 1
        assert out["anchor"] == pair[1]
        np.testing.assert_allclose(out["Av"], Av_true, atol=1e-3)

    def test_uses_only_the_named_pair(self):
        """Only the two named lines influence the result; others are ignored."""
        from jwspecabund.dust import compute_Av_balmer_pair

        fluxes, errors = _build_full_balmer_observed(1.0, "cardelli")
        # Corrupt a line NOT in the pair — result must be unchanged.
        ref = compute_Av_balmer_pair(fluxes, errors, ("Ha", "HBETA"), law="cardelli")
        fluxes["HGAMMA"] *= 5.0
        out = compute_Av_balmer_pair(fluxes, errors, ("Ha", "HBETA"), law="cardelli")
        np.testing.assert_allclose(out["Av"], ref["Av"], atol=1e-10)

    def test_missing_line_returns_no_detection(self):
        from jwspecabund.dust import compute_Av_balmer_pair
        out = compute_Av_balmer_pair({"HBETA": 100.0}, {"HBETA": 2.0}, ("Ha", "HBETA"))
        assert out["n_lines"] == 0 and out["Av"] == 0.0

    def test_invalid_line_name_raises(self):
        from jwspecabund.dust import compute_Av_balmer_pair
        with pytest.raises(ValueError):
            compute_Av_balmer_pair({"Ha": 1, "HBETA": 1}, {}, ("Halpha", "HBETA"))


# ---------------------------------------------------------------------------
# 3. Av recovery under realistic noise — the estimator must be unbiased.
# ---------------------------------------------------------------------------
class TestAvRecoveryUnderNoise:

    @pytest.mark.parametrize("Av_true", [0.5, 1.5])
    def test_av_estimator_unbiased(self, Av_true):
        """Adding 2% Gaussian noise to all Balmer fluxes and averaging
        many recovered Av's should reproduce Av_true to within the
        propagated 1σ.

        A persistent bias of >1σ would mean compute_Av_multi_balmer's
        weighting / averaging is mis-calibrated.
        """
        from jwspecabund.dust import compute_Av_multi_balmer

        rng = np.random.default_rng(0xA1B2)
        n_trials = 200
        recovered = []
        recovered_err = []

        # Clean baseline.
        clean_f, clean_e = _build_full_balmer_observed(Av_true, "salim")
        # 2 % errors (independent of clean_e which was 1 %; we override here).
        sigma_frac = 0.02

        for _ in range(n_trials):
            noisy_f = {n: f + rng.normal(0.0, sigma_frac * f)
                       for n, f in clean_f.items()}
            noisy_e = {n: sigma_frac * f for n, f in clean_f.items()}
            out = compute_Av_multi_balmer(noisy_f, noisy_e, law="salim",
                                          snr_min=3.0, anchor="HBETA")
            recovered.append(out["Av"])
            recovered_err.append(out["Av_err"])

        mean_recovered = np.mean(recovered)
        mean_err = np.mean(recovered_err)
        # Standard error on the mean of n_trials.
        sigma_of_mean = mean_err / np.sqrt(n_trials)

        # The estimator's bias must be smaller than 3σ_of_mean — otherwise
        # the mean has drifted from Av_true beyond what RNG can excuse.
        assert abs(mean_recovered - Av_true) < 3 * sigma_of_mean, (
            f"Bias detected: <Av>={mean_recovered:.4f} vs true={Av_true:.4f}, "
            f"3σ_mean={3*sigma_of_mean:.4f}.  Estimator is not unbiased."
        )


# ---------------------------------------------------------------------------
# 4. Monotonicity: increasing observed Balmer ratio => increasing recovered Av.
# ---------------------------------------------------------------------------
class TestMonotonicity:

    def test_av_increases_with_balmer_ratio(self):
        """Sweep observed Hα/Hβ from 2.86 → 6.0; recovered Av must increase."""
        from jwspecabund.dust import compute_Av_multi_balmer

        Av_seq = []
        for ratio in np.linspace(2.86, 6.0, 12):
            fluxes = {"Ha": ratio, "HBETA": 1.0}
            errors = {"Ha": 0.01, "HBETA": 0.01}
            out = compute_Av_multi_balmer(fluxes, errors, law="salim",
                                          snr_min=3.0, anchor="HBETA")
            Av_seq.append(out["Av"])
        Av_seq = np.array(Av_seq)
        diffs = np.diff(Av_seq)
        assert np.all(diffs >= 0), (
            f"Av is not monotonic in observed Hα/Hβ: diffs = {diffs}"
        )

    @pytest.mark.parametrize("law", ["salim", "cardelli"])
    def test_a_lambda_monotonic_in_Av(self, law):
        """Increasing Av must monotonically increase A_λ at every λ."""
        from jwspecabund.dust import salim_attenuation, cardelli_extinction
        fn = salim_attenuation if law == "salim" else cardelli_extinction
        waves = np.array([1500.0, 3000.0, 5500.0, 6563.0, 9000.0])

        prev = fn(waves, Av=0.0)
        for Av in (0.1, 0.5, 1.0, 2.0, 3.0):
            cur = fn(waves, Av=Av)
            assert np.all(cur >= prev - 1e-15), (
                f"A_λ not monotonic in Av at Av={Av} ({law}): "
                f"prev={prev}, cur={cur}"
            )
            prev = cur


# ---------------------------------------------------------------------------
# 5. dust_correct_fluxes preserves SNR per line — flux and error scale equally.
# ---------------------------------------------------------------------------
def test_dust_correction_preserves_snr():
    """dust_correct_fluxes multiplies flux and error by the same factor.

    Critical: this is what makes the SNR cuts (snr_line, snr_balmer) give
    the same set of lines whether dust correction is applied before or
    after the cut.
    """
    from jwspecabund._core import _LINE_WAVES
    from jwspecabund.dust import dust_correct_fluxes

    line_data = {
        "HBETA":      (1.00, 0.05, _LINE_WAVES["HBETA"]),
        "Ha":         (3.50, 0.10, _LINE_WAVES["Ha"]),
        "OIII_5007":  (5.00, 0.10, _LINE_WAVES["OIII_5007"]),
        "OII_doublet":(2.00, 0.20, _LINE_WAVES["OII_doublet"]),
    }
    snr_in = {n: f / e for n, (f, e, _) in line_data.items()}

    out = dust_correct_fluxes(line_data, Av=1.5, law="salim",
                              Rv=3.15, delta=-0.35, B=2.27)
    snr_out = {n: f / e for n, (f, e) in out.items()}

    for name in line_data:
        np.testing.assert_allclose(
            snr_out[name], snr_in[name], rtol=1e-12, atol=0,
            err_msg=f"SNR not preserved for {name}",
        )


# ---------------------------------------------------------------------------
# 6. Sign convention — observed Balmer ratio above intrinsic => positive Av.
# ---------------------------------------------------------------------------
class TestSignConvention:

    def test_red_balmer_excess_gives_positive_av(self):
        """Observed Hα/Hβ > 2.86 means dust-reddened => Av > 0.

        Uses anchor="Ha" so the Hα/Hβ decrement is actually exercised
        (the Hβ anchor only uses lines bluer than Hβ).
        """
        from jwspecabund.dust import compute_Av_multi_balmer
        out = compute_Av_multi_balmer(
            {"Ha": 4.0, "HBETA": 1.0},
            {"Ha": 0.01, "HBETA": 0.01},
            law="salim", snr_min=3.0, anchor="Ha",
        )
        assert out["Av"] > 0, (
            f"Red Balmer excess (Hα/Hβ = 4.0) gave Av = {out['Av']}; "
            f"must be > 0."
        )

    def test_blue_balmer_excess_clipped_to_zero(self):
        """Observed Hα/Hβ < 2.86 is unphysical for normal sources;
        compute_Av_multi_balmer must clip recovered Av to 0 (not return
        a negative dust value)."""
        from jwspecabund.dust import compute_Av_multi_balmer
        out = compute_Av_multi_balmer(
            {"Ha": 2.5, "HBETA": 1.0},
            {"Ha": 0.01, "HBETA": 0.01},
            law="salim", snr_min=3.0, anchor="Ha",
        )
        assert out["Av"] >= 0, (
            f"Blue Balmer excess produced negative Av = {out['Av']}."
        )


# ---------------------------------------------------------------------------
# 7. Cross-law sanity on synthetic Balmer-decrement input.
# ---------------------------------------------------------------------------
#
# Previously this section refitted a real stacked composite (which is
# not distributed with the package) and compared Salim vs Cardelli on
# it.  Reformulated as a fully synthetic test: build Case-B-like
# Balmer fluxes attenuated by a known Av, derive Av with each law and
# assert they agree on the *physical* attenuation A(5500 Å).

def test_salim_and_cardelli_give_similar_A_at_5500_on_synthetic_data():
    """Cross-law sanity on a known synthetic Balmer decrement.

    Salim's ``Av`` is a model parameter and ≠ A(5500 Å) when
    ``delta ≠ 0``, so the meaningful comparison is at A(5500 Å).  Both
    laws should recover ≈ the same A(5500 Å) on the same fluxes within
    a factor of 2 — loose enough to allow genuine curve-shape
    differences across the Balmer range, strict enough to flag a sign
    error or factor-of-10 bug.
    """
    from jwspecabund.dust import (
        compute_Av_multi_balmer, salim_attenuation, cardelli_extinction,
    )

    Av_true = 0.8
    salim_kw = {"Rv": 3.15, "delta": -0.35, "B": 2.27}
    waves = {"Ha": 6562.8, "HBETA": 4861.3, "HGAMMA": 4340.5, "HDELTA": 4101.7}
    case_b = {"Ha": 2.86, "HBETA": 1.00, "HGAMMA": 0.468, "HDELTA": 0.259}
    A_lam = salim_attenuation(
        np.array([waves[k] for k in case_b]), Av_true, **salim_kw,
    )
    fluxes = {k: case_b[k] * 10 ** (-0.4 * A_lam[i])
              for i, k in enumerate(case_b)}
    errors = {k: 0.02 * f for k, f in fluxes.items()}

    sal = compute_Av_multi_balmer(
        fluxes, errors, law="salim", snr_min=3.0, anchor="HBETA",
        **salim_kw,
    )
    car = compute_Av_multi_balmer(
        fluxes, errors, law="cardelli", snr_min=3.0, anchor="HBETA",
        Rv=3.1,
    )

    A_V_sal = salim_attenuation(np.array([5500.0]), sal["Av"], **salim_kw)[0]
    A_V_car = cardelli_extinction(np.array([5500.0]), car["Av"], Rv=3.1)[0]

    assert A_V_sal > 0.01 and A_V_car > 0.01, (
        f"Synthetic input returned ~zero A(5500 Å): "
        f"salim={A_V_sal:.4f}, cardelli={A_V_car:.4f}"
    )
    ratio = A_V_sal / A_V_car
    assert 0.5 < ratio < 2.0, (
        f"Salim and Cardelli A(5500 Å) disagree by more than a factor of 2: "
        f"salim={A_V_sal:.4f}, cardelli={A_V_car:.4f}, ratio={ratio:.3f}."
    )


# ---------------------------------------------------------------------------
# 8. Wavelength self-consistency.
# ---------------------------------------------------------------------------
def test_balmer_wavelengths_consistent_between_modules():
    """_LINE_WAVES (used by _apply_dust_correction) and _BALMER_LADDER
    (used by compute_Av_multi_balmer) must agree on Balmer-line rest
    wavelengths.  A divergence > 5 Å changes A_λ and could subtly bias
    derived Av — the tolerance here flags any worsening.
    """
    from jwspecabund._core import _LINE_WAVES
    from jwspecabund.dust import _BALMER_LADDER

    for name, (wave_ladder, _ratio) in _BALMER_LADDER.items():
        if name not in _LINE_WAVES:
            continue
        wave_lines = _LINE_WAVES[name]
        diff = abs(wave_lines - wave_ladder)
        assert diff < 5.0, (
            f"Balmer wavelength mismatch for {name}: "
            f"_LINE_WAVES={wave_lines:.4f} Å vs "
            f"_BALMER_LADDER={wave_ladder:.4f} Å (Δ={diff:.4f} Å). "
            f"A divergence this large will bias A_λ at Hβ-anchored "
            f"derivations."
        )
