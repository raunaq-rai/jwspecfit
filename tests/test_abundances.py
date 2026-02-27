"""Unit tests for jwspecabund."""

from __future__ import annotations

import numpy as np
import pytest


# -----------------------------------------------------------------------
# Dust correction tests
# -----------------------------------------------------------------------

class TestDustCorrection:
    """Tests for dust.py attenuation curves and correction."""

    def test_salim_attenuation_zero_Av(self):
        """A_V = 0 should give zero attenuation everywhere."""
        from jwspecabund.dust import salim_attenuation

        waves = np.array([3000.0, 5000.0, 7000.0, 10000.0])
        A = salim_attenuation(waves, Av=0.0)
        np.testing.assert_allclose(A, 0.0, atol=1e-15)

    def test_cardelli_extinction_zero_Av(self):
        """A_V = 0 should give zero extinction."""
        from jwspecabund.dust import cardelli_extinction

        waves = np.array([3000.0, 5000.0, 7000.0, 10000.0])
        A = cardelli_extinction(waves, Av=0.0)
        np.testing.assert_allclose(A, 0.0, atol=1e-15)

    def test_salim_attenuation_increases_blueward(self):
        """Attenuation should be larger at shorter wavelengths."""
        from jwspecabund.dust import salim_attenuation

        waves = np.array([3000.0, 5500.0, 10000.0])
        A = salim_attenuation(waves, Av=1.0)
        assert A[0] > A[1] > A[2]

    def test_cardelli_extinction_increases_blueward(self):
        """Extinction should be larger at shorter wavelengths (optical)."""
        from jwspecabund.dust import cardelli_extinction

        waves = np.array([4000.0, 5500.0, 8000.0])
        A = cardelli_extinction(waves, Av=1.0)
        assert A[0] > A[1] > A[2]

    def test_dust_correct_roundtrip(self):
        """Correcting and then attenuating should recover original flux."""
        from jwspecabund.dust import dust_correct_fluxes, salim_attenuation

        Av = 0.5
        flux_true = 1.0e-16
        wave = 5000.0

        # Attenuate the true flux.
        A_lam = salim_attenuation(np.array([wave]), Av)[0]
        flux_obs = flux_true * 10.0 ** (-0.4 * A_lam)

        # Now correct.
        corrected = dust_correct_fluxes(
            {"test": (flux_obs, 0.01 * flux_obs, wave)}, Av, law="salim"
        )
        np.testing.assert_allclose(corrected["test"][0], flux_true, rtol=1e-10)

    def test_compute_Av_from_balmer_no_dust(self):
        """Intrinsic ratio should give A_V = 0."""
        from jwspecabund.dust import compute_Av_from_balmer

        Av, Av_err = compute_Av_from_balmer(
            flux_num=2.86, flux_den=1.0,
            flux_num_err=0.01, flux_den_err=0.01,
            law="salim",
            intrinsic_ratio=2.86,
        )
        assert Av == 0.0

    def test_compute_Av_from_balmer_positive_dust(self):
        """Observed ratio > intrinsic should give positive A_V."""
        from jwspecabund.dust import compute_Av_from_balmer

        # Ratio of 4.0 > 2.86 → positive dust.
        Av, Av_err = compute_Av_from_balmer(
            flux_num=4.0, flux_den=1.0,
            flux_num_err=0.1, flux_den_err=0.05,
            law="salim",
        )
        assert Av > 0.0
        assert Av_err > 0.0


# -----------------------------------------------------------------------
# Strong-line tests
# -----------------------------------------------------------------------

class TestStrongLine:
    """Tests for strong_line.py Sanders+25 calibrations."""

    def test_sanders25_basic(self):
        """Known O3 ratio should give a sensible metallicity."""
        from jwspecabund.strong_line import sanders25_metallicity

        # Typical star-forming galaxy: log(O3/Hb) ~ 0.85 → 12+log(O/H) ~ 8.0
        fluxes = {"OIII_5007": 7.08, "HBETA": 1.0}  # log10(7.08) ~ 0.85
        errors = {"OIII_5007": 0.1, "HBETA": 0.02}

        Z, Z_lo, Z_hi, chi2, ratios, Z_mc = sanders25_metallicity(
            fluxes, errors, n_mc=100,
        )

        assert 7.0 < Z < 9.0, f"Z={Z} out of range"
        assert Z_lo < Z < Z_hi
        assert "O3" in ratios
        assert len(Z_mc) == 100

    def test_sanders25_with_OII(self):
        """Including [OII] should add O2, R23, O32 ratios."""
        from jwspecabund.strong_line import sanders25_metallicity

        fluxes = {"OIII_5007": 5.0, "HBETA": 1.0, "OII_doublet": 2.0}
        errors = {"OIII_5007": 0.1, "HBETA": 0.02, "OII_doublet": 0.1}

        Z, _, _, _, ratios, _ = sanders25_metallicity(
            fluxes, errors, n_mc=10,
        )
        assert "O3" in ratios
        assert "O2" in ratios
        assert "R23" in ratios
        assert "O32" in ratios

    def test_sanders25_no_lines_raises(self):
        """Should raise if no valid ratios."""
        from jwspecabund.strong_line import sanders25_metallicity

        with pytest.raises(ValueError, match="No valid strong-line ratios"):
            sanders25_metallicity({}, {}, n_mc=10)

    def test_compute_line_ratios_resolved_OII(self):
        """Resolved [OII] 3726+3729 should be combined."""
        from jwspecabund.strong_line import compute_line_ratios

        fluxes = {
            "OIII_5007": 5.0, "HBETA": 1.0,
            "OII_3726": 1.0, "OII_3729": 1.0,
        }
        errors = {
            "OIII_5007": 0.1, "HBETA": 0.02,
            "OII_3726": 0.05, "OII_3729": 0.05,
        }
        ratios = compute_line_ratios(fluxes, errors)
        assert "O2" in ratios
        # log10(2.0 / 1.0) ~ 0.301
        np.testing.assert_allclose(ratios["O2"]["val"], np.log10(2.0), atol=0.01)


# -----------------------------------------------------------------------
# ICF tests
# -----------------------------------------------------------------------

class TestICF:
    """Tests for Izotov+06 ICF prescriptions."""

    def test_icf_nitrogen_unity_approx(self):
        """ICF(N) should be near unity for typical HII regions."""
        from jwspecabund.icf import icf_nitrogen

        # x = O+/O ~ 0.5 for a moderately ionised nebula
        icf = icf_nitrogen(0.5e-4, 1.0e-4)
        assert 0.9 <= icf <= 1.5

    def test_icf_neon_positive(self):
        """ICF(Ne) should be positive."""
        from jwspecabund.icf import icf_neon

        icf = icf_neon(0.3e-4, 1.0e-4)
        assert icf > 0

    def test_icf_sulfur_ge_one(self):
        """ICF(S) should be >= 1 (accounts for unobserved S3+)."""
        from jwspecabund.icf import icf_sulfur

        icf = icf_sulfur(0.3e-4, 1.0e-4)
        assert icf >= 1.0

    def test_icf_argon_ge_one(self):
        """ICF(Ar) should be >= 1."""
        from jwspecabund.icf import icf_argon

        icf = icf_argon(0.3e-4, 1.0e-4)
        assert icf >= 1.0

    def test_icf_handles_zero_input(self):
        """ICFs should handle zero inputs gracefully."""
        from jwspecabund.icf import icf_nitrogen, icf_neon, icf_sulfur, icf_argon

        assert icf_nitrogen(0, 1e-4) == 1.0
        assert icf_neon(0, 1e-4) == 1.0
        assert icf_sulfur(0, 1e-4) == 1.0
        assert icf_argon(0, 1e-4) == 1.0


# -----------------------------------------------------------------------
# Direct method tests (require PyNEB)
# -----------------------------------------------------------------------

class TestDirect:
    """Tests for direct.py T_e method."""

    @pytest.fixture(autouse=True)
    def _check_pyneb(self):
        """Skip all tests in this class if PyNEB is not installed."""
        pytest.importorskip("pyneb")

    def test_Te_low_from_high_desi(self):
        """DESI relation: T_low = 0.648 * T_high + 3270."""
        from jwspecabund.direct import Te_low_from_high

        Te_low = Te_low_from_high(15000.0, relation="desi")
        expected = 0.648 * 15000.0 + 3270.0
        np.testing.assert_allclose(Te_low, expected)

    def test_Te_low_from_high_classical(self):
        """Classical relation: T_low = 0.7 * T_high + 3000."""
        from jwspecabund.direct import Te_low_from_high

        Te_low = Te_low_from_high(15000.0, relation="classical")
        expected = 0.7 * 15000.0 + 3000.0
        np.testing.assert_allclose(Te_low, expected)

    def test_compute_ne_SII(self):
        """Density from [SII] doublet with ratio ~ 1.0 (low density)."""
        from jwspecabund.direct import compute_ne

        # SII 6718/6732 ratio ~ 1.4 → low density regime (~100 cm^-3)
        ne = compute_ne(1.4, 1.0, doublet="SII", Te_guess=1e4)
        assert 10 < ne < 1000

    def test_compute_Te_OIII(self):
        """T_e from [OIII] with a known auroral/nebular ratio."""
        from jwspecabund.direct import compute_Te_OIII

        # Typical [OIII] 4363/(5007+4959) ~ 0.01 → T ~ 10,000-15,000 K
        Te = compute_Te_OIII(
            flux_4363=0.01,
            flux_5007=0.75,
            flux_4959=0.25,
            ne=100.0,
        )
        assert 5000 < Te < 30000

    def test_ionic_abundances(self):
        """Ionic O++/H+ from [OIII] 5007 should be reasonable."""
        from jwspecabund.direct import compute_ionic_abundances

        fluxes = {
            "OIII_5007": 3.0,  # relative to Hβ = 1.0
            "HBETA": 1.0,
        }
        ionic = compute_ionic_abundances(fluxes, Te_high=15000.0, Te_low=12000.0, ne=100.0)
        assert "O++/H+" in ionic
        assert ionic["O++/H+"] > 0
        # 12 + log(O++/H+) should be somewhere in 6.5–9.0
        assert 6.5 < 12 + np.log10(ionic["O++/H+"]) < 9.0

    def test_total_abundances_OH(self):
        """O/H = O+/H+ + O++/H+ with no ICF."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {"O+/H+": 1e-5, "O++/H+": 5e-5}
        totals = compute_total_abundances(ionic)
        np.testing.assert_allclose(totals["O/H"], 6e-5)

    def test_ionic_abundance_high_Te(self):
        """Ionic abundances should work at T > 30,000 K (PyNEB HI limit)."""
        from jwspecabund.direct import compute_ionic_abundances

        fluxes = {
            "OIII_5007": 5.0,
            "HBETA": 1.0,
        }
        # T_e = 36,000 K exceeds PyNEB's H I tables; the fallback should handle it.
        ionic = compute_ionic_abundances(fluxes, Te_high=36000.0, Te_low=26500.0, ne=100.0)
        assert "O++/H+" in ionic
        assert ionic["O++/H+"] > 0
        assert 6.0 < 12 + np.log10(ionic["O++/H+"]) < 9.0


# -----------------------------------------------------------------------
# Orchestrator tests
# -----------------------------------------------------------------------

class TestComputeAbundances:
    """Tests for _core.py compute_abundances orchestrator."""

    def _make_mock_fit_result(
        self,
        line_fluxes: dict[str, tuple[float, float]],
    ):
        """Create a minimal mock FitResult for testing."""
        from dataclasses import dataclass

        @dataclass
        class MockLineResult:
            name: str
            flux: float
            flux_err: float
            snr: float

        @dataclass
        class MockFitResult:
            lines: dict

        lines = {}
        for name, (flux, err) in line_fluxes.items():
            lines[name] = MockLineResult(
                name=name, flux=flux, flux_err=err,
                snr=flux / err if err > 0 else 0.0,
            )
        return MockFitResult(lines=lines)

    def test_strong_line_method(self):
        """Strong-line method on a mock FitResult."""
        from jwspecabund import compute_abundances

        result = self._make_mock_fit_result({
            "OIII_5007": (5.0, 0.1),
            "HBETA": (1.0, 0.02),
            "Ha": (2.86, 0.05),  # no dust
            "OII_doublet": (2.0, 0.1),
        })

        abund = compute_abundances(result, z=2.0, method="strong_line", n_mc=50)
        assert abund.method == "strong_line"
        assert 6.5 < abund.OH < 9.5
        assert abund.ratios_used is not None
        assert len(abund.ratios_used) > 0

    def test_auto_selects_strong_line_without_4363(self):
        """Auto method should use strong-line when 4363 is absent."""
        from jwspecabund import compute_abundances

        result = self._make_mock_fit_result({
            "OIII_5007": (5.0, 0.1),
            "HBETA": (1.0, 0.02),
            "Ha": (2.86, 0.05),
        })

        abund = compute_abundances(result, z=2.0, method="auto", n_mc=50)
        assert abund.method == "strong_line"

    @pytest.mark.skipif(
        not pytest.importorskip("pyneb", reason="PyNEB required"),
        reason="PyNEB not available",
    )
    def test_auto_selects_direct_with_4363(self):
        """Auto method should use direct when 4363 has high SNR."""
        from jwspecabund import compute_abundances

        result = self._make_mock_fit_result({
            "OIII_4363": (0.03, 0.005),  # SNR = 6
            "OIII_5007": (3.0, 0.05),
            "OIII_4959": (1.0, 0.05),
            "HBETA": (1.0, 0.02),
            "Ha": (2.86, 0.05),
            "NII_6585": (0.1, 0.01),
            "OII_doublet": (1.0, 0.05),
        })

        abund = compute_abundances(
            result, z=2.0, method="auto", snr_auroral=3.0, n_mc=50,
        )
        assert abund.method == "direct"
        assert abund.Te_high is not None
        assert abund.Te_low is not None
        assert 6.5 < abund.OH < 9.5

    def test_dust_correct_false(self):
        """dust_correct=False should skip correction."""
        from jwspecabund import compute_abundances

        result = self._make_mock_fit_result({
            "OIII_5007": (5.0, 0.1),
            "HBETA": (1.0, 0.02),
        })

        abund = compute_abundances(
            result, z=2.0, dust_correct=False, method="strong_line", n_mc=50,
        )
        assert abund.Av is None  # no Av computed when correction skipped

    def test_posterior_shape(self):
        """OH_posterior should have n_mc samples."""
        from jwspecabund import compute_abundances

        result = self._make_mock_fit_result({
            "OIII_5007": (5.0, 0.1),
            "HBETA": (1.0, 0.02),
            "Ha": (2.86, 0.05),
        })

        abund = compute_abundances(result, z=2.0, method="strong_line", n_mc=200)
        assert abund.OH_posterior is not None
        assert len(abund.OH_posterior) == 200


# -----------------------------------------------------------------------
# Result dataclass tests
# -----------------------------------------------------------------------

class TestAbundanceResult:
    """Tests for result.py AbundanceResult."""

    def test_summary(self):
        """Summary string should include key fields."""
        from jwspecabund.result import AbundanceResult

        res = AbundanceResult(
            method="direct",
            OH=8.0,
            OH_err=0.1,
            NO=-1.2,
            NO_err=0.15,
            Te_high=15000.0,
            Te_low=12000.0,
            ne=100.0,
            Av=0.5,
        )
        s = res.summary()
        assert "direct" in s
        assert "8.000" in s
        assert "15000" in s

    def test_default_none_fields(self):
        """Optional fields should default to None."""
        from jwspecabund.result import AbundanceResult

        res = AbundanceResult(method="strong_line", OH=8.0, OH_err=0.1)
        assert res.NO is None
        assert res.Te_high is None
        assert res.ionic is None
        assert res.ratios_used is None


# -----------------------------------------------------------------------
# Broad component summation tests
# -----------------------------------------------------------------------

class TestBroadComponentSummation:
    """Tests for narrow + broad Balmer component summation."""

    def _make_mock_result(self, lines_dict):
        """Create a minimal mock result with given lines."""
        from dataclasses import dataclass

        @dataclass
        class MockLineResult:
            name: str
            flux: float
            flux_err: float
            snr: float

        @dataclass
        class MockFitResult:
            lines: dict

        lines = {}
        for name, (flux, err) in lines_dict.items():
            lines[name] = MockLineResult(
                name=name, flux=flux, flux_err=err,
                snr=flux / err if err > 0 else 0.0,
            )
        return MockFitResult(lines=lines)

    def test_broad_added_to_narrow(self):
        """Broad Balmer flux should be summed into the narrow line."""
        from jwspecabund._core import _extract_fluxes

        result = self._make_mock_result({
            "Ha": (3.0, 0.1),
            "Ha_BROAD": (1.0, 0.2),
            "HBETA": (1.0, 0.05),
            "OIII_5007": (5.0, 0.1),
        })

        fluxes, errors, _ = _extract_fluxes(result)

        assert fluxes["Ha"] == pytest.approx(4.0)
        assert errors["Ha"] == pytest.approx(np.sqrt(0.1**2 + 0.2**2))
        # Non-Balmer lines unaffected.
        assert fluxes["OIII_5007"] == pytest.approx(5.0)

    def test_broad_without_narrow_ignored(self):
        """Broad component should be ignored if no narrow counterpart."""
        from jwspecabund._core import _extract_fluxes

        result = self._make_mock_result({
            "Ha_BROAD": (1.0, 0.2),
            "HBETA": (1.0, 0.05),
        })

        fluxes, _, _ = _extract_fluxes(result)

        assert "Ha" not in fluxes
        assert "Ha_BROAD" not in fluxes

    def test_non_balmer_broad_ignored(self):
        """Broad component of non-Balmer lines should not be summed."""
        from jwspecabund._core import _extract_fluxes

        result = self._make_mock_result({
            "OIII_5007": (5.0, 0.1),
            "OIII_5007_BROAD": (1.0, 0.2),
            "HBETA": (1.0, 0.05),
        })

        fluxes, _, _ = _extract_fluxes(result)

        assert fluxes["OIII_5007"] == pytest.approx(5.0)


# -----------------------------------------------------------------------
# Forward model tests (require PyNEB + emcee)
# -----------------------------------------------------------------------

class TestForwardModel:
    """Tests for forward.py Bayesian forward model."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        """Skip if PyNEB or emcee not installed."""
        pytest.importorskip("pyneb")
        pytest.importorskip("emcee")

    def test_hbeta_emissivity_aller84(self):
        """Aller (1984) Hbeta emissivity should be positive and decrease with T."""
        from jwspecabund.forward import hbeta_emissivity_aller84

        eps_1e4 = hbeta_emissivity_aller84(1e4)
        eps_3e4 = hbeta_emissivity_aller84(3e4)
        eps_5e4 = hbeta_emissivity_aller84(5e4)

        assert eps_1e4 > 0
        assert eps_1e4 > eps_3e4 > eps_5e4  # decreases with T

    def test_build_observables(self):
        """Should construct correct line/Hb ratios."""
        from jwspecabund.forward import _build_observables

        fluxes = {"HBETA": 1.0, "OIII_5007": 5.0, "OIII_4363": 0.05}
        errors = {"HBETA": 0.02, "OIII_5007": 0.1, "OIII_4363": 0.01}

        names, obs, errs, comps = _build_observables(fluxes, errors)

        assert len(names) == 2
        assert "OIII_5007/Hb" in names
        assert "OIII_4363/Hb" in names
        np.testing.assert_allclose(obs[names.index("OIII_5007/Hb")], 5.0)

    def test_determine_free_params(self):
        """Should add Te, ne, and one param per detected ion."""
        from jwspecabund.forward import _build_observables, _determine_free_params

        fluxes = {"HBETA": 1.0, "OIII_5007": 5.0, "NeIII_3869": 0.3}
        errors = {"HBETA": 0.02, "OIII_5007": 0.1, "NeIII_3869": 0.05}

        _, _, _, comps = _build_observables(fluxes, errors)
        params = _determine_free_params(comps)

        assert "log_Te" in params
        assert "log_ne" in params
        assert "log_Opp" in params
        assert "log_Nepp" in params
        assert len(params) == 4

    def test_predict_ratios(self):
        """Predicted ratios should be positive for valid parameters."""
        import pyneb as pn

        from jwspecabund.forward import (
            _build_observables,
            _determine_free_params,
            _predict_ratios,
        )

        fluxes = {"HBETA": 1.0, "OIII_5007": 5.0, "OIII_4363": 0.05}
        errors = {"HBETA": 0.02, "OIII_5007": 0.1, "OIII_4363": 0.01}

        _, _, _, comps = _build_observables(fluxes, errors)
        params = _determine_free_params(comps)

        pyneb_atoms = {("O", 3): pn.Atom("O", 3)}
        # theta: log_Te=4.2, log_ne=2.0, log_Opp=-4.0
        theta = np.array([4.2, 2.0, -4.0])

        pred = _predict_ratios(theta, params, comps, pyneb_atoms)
        assert pred.shape == (2,)
        assert np.all(pred > 0)
        assert np.all(np.isfinite(pred))

    def test_forward_model_runs(self):
        """Smoke test: forward_model should run and return sensible output."""
        from jwspecabund.forward import forward_model

        fluxes = {
            "HBETA": 1.0,
            "OIII_5007": 5.0,
            "OIII_4363": 0.05,
        }
        errors = {k: 0.05 * abs(v) for k, v in fluxes.items()}

        res = forward_model(
            fluxes, errors,
            sampler="emcee",
            n_walkers=16,
            n_steps=500,
            n_burn=200,
            seed=42,
            progress=False,
        )

        assert "OH" in res
        assert "Te" in res
        assert "samples" in res
        assert res["samples"].shape[1] == 3  # log_Te, log_ne, log_Opp
        assert 5.0 < res["OH"] < 10.0
        assert res["Te"] > 5000

    def test_forward_model_with_neon(self):
        """Forward model with [NeIII] should have 4 free parameters."""
        from jwspecabund.forward import forward_model

        fluxes = {
            "HBETA": 1.0,
            "OIII_5007": 5.0,
            "OIII_4363": 0.05,
            "NeIII_3869": 0.3,
        }
        errors = {k: 0.05 * abs(v) for k, v in fluxes.items()}

        res = forward_model(
            fluxes, errors,
            sampler="emcee",
            n_walkers=16,
            n_steps=500,
            n_burn=200,
            seed=42,
            progress=False,
        )

        assert res["samples"].shape[1] == 4  # log_Te, log_ne, log_Opp, log_Nepp
        assert "NeO" in res
        assert res["NeO"] is not None

    def test_compute_abundances_forward_method(self):
        """compute_abundances(method='forward') should dispatch correctly."""
        from jwspecabund import compute_abundances

        from dataclasses import dataclass

        @dataclass
        class MockLineResult:
            name: str
            flux: float
            flux_err: float
            snr: float

        @dataclass
        class MockFitResult:
            lines: dict

        line_data = {
            "OIII_4363": (0.05, 0.01),
            "OIII_5007": (5.0, 0.1),
            "HBETA": (1.0, 0.02),
        }
        lines = {}
        for name, (flux, err) in line_data.items():
            lines[name] = MockLineResult(
                name=name, flux=flux, flux_err=err,
                snr=flux / err if err > 0 else 0.0,
            )
        result = MockFitResult(lines=lines)

        abund = compute_abundances(
            result, z=6.0, method="forward",
            dust_correct=False,
            forward_n_walkers=16,
            forward_n_steps=500,
            forward_n_burn=200,
            forward_seed=42,
            forward_progress=False,
        )

        assert abund.method == "forward"
        assert 5.0 < abund.OH < 10.0
        assert abund.Te_high is not None
        assert abund._forward_result is not None


# -----------------------------------------------------------------------
# SNR gating tests
# -----------------------------------------------------------------------

class TestSNRGating:
    """Tests for per-line SNR filtering in the direct method."""

    def test_filter_low_snr_removes_noisy_lines(self):
        """Lines with SNR < threshold are excluded (except protected lines)."""
        from jwspecabund._core import _filter_low_snr

        fluxes = {
            "OIII_5007": 5.0,
            "HBETA": 1.0,
            "SII_6718": 0.01,    # SNR = 0.01/0.1 = 0.1 → excluded
            "Ha": 0.05,          # SNR = 0.05/0.025 = 2.0 → kept
        }
        errors = {
            "OIII_5007": 0.1,
            "HBETA": 0.02,
            "SII_6718": 0.1,
            "Ha": 0.025,
        }

        filt_f, filt_e, excluded = _filter_low_snr(fluxes, errors, snr_thresh=2.0)

        assert "SII_6718" in excluded
        assert "SII_6718" not in filt_f
        assert "SII_6718" not in filt_e
        assert "Ha" in filt_f  # SNR = 2.0, exactly at threshold
        assert "OIII_5007" in filt_f
        assert "HBETA" in filt_f

    def test_filter_low_snr_keeps_at_threshold(self):
        """Lines with SNR exactly at threshold are kept (>=)."""
        from jwspecabund._core import _filter_low_snr

        fluxes = {"Ha": 2.0, "HBETA": 1.0}
        errors = {"Ha": 1.0, "HBETA": 0.02}

        filt_f, _, excluded = _filter_low_snr(fluxes, errors, snr_thresh=2.0)

        assert "Ha" in filt_f  # SNR = 2.0 exactly
        assert "Ha" not in excluded

    def test_filter_low_snr_protects_auroral(self):
        """OIII_4363 is never filtered, even at very low SNR."""
        from jwspecabund._core import _filter_low_snr

        fluxes = {"OIII_4363": 0.01, "HBETA": 1.0, "OIII_5007": 5.0}
        errors = {"OIII_4363": 0.1, "HBETA": 0.02, "OIII_5007": 0.1}

        filt_f, _, excluded = _filter_low_snr(fluxes, errors, snr_thresh=2.0)

        # SNR = 0.1 but OIII_4363 is protected.
        assert "OIII_4363" in filt_f
        assert "OIII_4363" not in excluded

    def test_filter_low_snr_protects_hbeta(self):
        """HBETA is never filtered, even at low SNR."""
        from jwspecabund._core import _filter_low_snr

        fluxes = {"HBETA": 0.01, "OIII_5007": 5.0}
        errors = {"HBETA": 0.1, "OIII_5007": 0.1}

        filt_f, _, excluded = _filter_low_snr(fluxes, errors, snr_thresh=5.0)

        assert "HBETA" in filt_f
        assert "HBETA" not in excluded

    def test_filter_low_snr_protects_all_te_lines(self):
        """All four Te-essential lines are protected."""
        from jwspecabund._core import _filter_low_snr

        fluxes = {
            "OIII_4363": 0.001,
            "OIII_5007": 0.001,
            "OIII_4959": 0.001,
            "HBETA": 0.001,
            "NII_6585": 0.001,  # not protected
        }
        errors = {k: 1.0 for k in fluxes}  # all SNR = 0.001

        filt_f, _, excluded = _filter_low_snr(fluxes, errors, snr_thresh=2.0)

        assert "OIII_4363" in filt_f
        assert "OIII_5007" in filt_f
        assert "OIII_4959" in filt_f
        assert "HBETA" in filt_f
        assert "NII_6585" in excluded

    def test_filter_low_snr_zero_error_kept(self):
        """Lines with zero error (SNR=inf) are always kept."""
        from jwspecabund._core import _filter_low_snr

        fluxes = {"Ha": 1.0, "HBETA": 1.0}
        errors = {"Ha": 0.0, "HBETA": 0.0}

        filt_f, _, excluded = _filter_low_snr(fluxes, errors, snr_thresh=5.0)

        assert "Ha" in filt_f
        assert len(excluded) == 0

    def test_excluded_lines_in_summary(self):
        """AbundanceResult.summary() displays excluded lines."""
        from jwspecabund.result import AbundanceResult

        result = AbundanceResult(
            method="direct",
            OH=8.0,
            OH_err=0.1,
            excluded_lines=["NIII_1749", "NIII_1752"],
        )
        text = result.summary()
        assert "Excluded" in text
        assert "NIII_1749" in text
        assert "NIII_1752" in text

    def test_excluded_lines_none_not_in_summary(self):
        """Summary omits 'Excluded' line when no lines are excluded."""
        from jwspecabund.result import AbundanceResult

        result = AbundanceResult(
            method="direct",
            OH=8.0,
            OH_err=0.1,
        )
        text = result.summary()
        assert "Excluded" not in text


# -----------------------------------------------------------------------
# Electron density SNR gating tests
# -----------------------------------------------------------------------

class TestNeSNRGating:
    """Tests for SNR gating on density-sensitive doublets."""

    def test_low_snr_SII_falls_back_to_default(self):
        """Low-SNR [SII] doublet should fall back to NE_DEFAULT (300)."""
        from jwspecabund._core import _compute_multi_ne
        from jwspecabund.direct import NE_DEFAULT

        fluxes = {
            "SII_6718": 0.01,
            "SII_6732": 0.01,
        }
        errors = {
            "SII_6718": 0.1,   # SNR = 0.1
            "SII_6732": 0.1,   # SNR = 0.1
        }

        ne_low, ne_high = _compute_multi_ne(fluxes, errors=errors, snr_ne=3.0)
        assert ne_low == NE_DEFAULT
        assert ne_high == NE_DEFAULT

    @pytest.mark.skipif(
        not pytest.importorskip("pyneb", reason="PyNEB required"),
        reason="PyNEB not available",
    )
    def test_high_snr_SII_uses_measured_ne(self):
        """High-SNR [SII] doublet should produce a measured n_e != NE_DEFAULT."""
        from jwspecabund._core import _compute_multi_ne
        from jwspecabund.direct import NE_DEFAULT

        # Ratio ~1.4 → low density regime
        fluxes = {
            "SII_6718": 1.4,
            "SII_6732": 1.0,
        }
        errors = {
            "SII_6718": 0.05,  # SNR = 28
            "SII_6732": 0.05,  # SNR = 20
        }

        ne_low, ne_high = _compute_multi_ne(fluxes, errors=errors, snr_ne=3.0)
        # Should be a real measurement, not the default.
        assert ne_low != NE_DEFAULT
        assert 10 < ne_low < 5000

    def test_snr_ne_zero_disables_gating(self):
        """snr_ne=0 should disable the SNR check (all doublets pass)."""
        from jwspecabund._core import _compute_multi_ne, _doublet_snr_ok

        fluxes = {
            "SII_6718": 0.001,
            "SII_6732": 0.001,
        }
        errors = {
            "SII_6718": 1.0,   # SNR = 0.001
            "SII_6732": 1.0,   # SNR = 0.001
        }

        # With snr_ne=0, even very low SNR should pass the check.
        assert _doublet_snr_ok("SII_6718", "SII_6732", fluxes, errors, snr_ne=0.0)

    def test_one_member_low_snr_fails(self):
        """If only one doublet member has low SNR, the doublet fails."""
        from jwspecabund._core import _doublet_snr_ok

        fluxes = {
            "SII_6718": 1.0,
            "SII_6732": 0.01,
        }
        errors = {
            "SII_6718": 0.05,  # SNR = 20
            "SII_6732": 0.1,   # SNR = 0.1
        }

        assert not _doublet_snr_ok("SII_6718", "SII_6732", fluxes, errors, snr_ne=3.0)

    def test_NE_DEFAULT_is_300(self):
        """NE_DEFAULT constant should be 300 cm^-3."""
        from jwspecabund.direct import NE_DEFAULT

        assert NE_DEFAULT == 300.0

    def test_no_errors_dict_falls_back(self):
        """When errors=None, all doublets should still pass (no gating)."""
        from jwspecabund._core import _compute_multi_ne, _doublet_snr_ok

        fluxes = {
            "SII_6718": 1.4,
            "SII_6732": 1.0,
        }

        # _doublet_snr_ok should return False when lines not in errors.
        assert not _doublet_snr_ok("SII_6718", "SII_6732", fluxes, {}, snr_ne=3.0)

        # But _compute_multi_ne with errors=None should use empty dict
        # and fall back to NE_DEFAULT.
        from jwspecabund.direct import NE_DEFAULT
        ne_low, _ = _compute_multi_ne(fluxes, errors=None, snr_ne=3.0)
        assert ne_low == NE_DEFAULT

    def test_missing_doublet_member_fails(self):
        """Missing doublet member should fail the SNR check."""
        from jwspecabund._core import _doublet_snr_ok

        fluxes = {"SII_6718": 1.0}
        errors = {"SII_6718": 0.05}

        assert not _doublet_snr_ok("SII_6718", "SII_6732", fluxes, errors, snr_ne=3.0)


# -----------------------------------------------------------------------
# Doublet completeness tests
# -----------------------------------------------------------------------

class TestDoubletCompleteness:
    """Tests for partial-doublet handling in logU and ionic abundances."""

    @pytest.fixture(autouse=True)
    def _check_pyneb(self):
        """Skip all tests in this class if PyNEB is not installed."""
        pytest.importorskip("pyneb")

    def test_logU_skips_N43_when_NIV_incomplete(self):
        """_compute_logU should skip N43 when only one NIV member is present."""
        from jwspecabund._core import _compute_logU

        fluxes = {
            "NIV_1483": 1.0e-17,
            # NIV_1486 missing (excluded by SNR filter)
            "NIII_1749": 5.0e-18,
            "NIII_1752": 3.0e-18,
            "OIII_5007": 3.0e-16,
            "OII_doublet": 8.0e-17,
        }

        logU, diag = _compute_logU(fluxes, Z_Zsun=0.1, ne_high=300.0)

        # Should NOT use N43 (incomplete NIV doublet); should fall back to O32.
        assert diag != "N43"
        if logU is not None:
            assert diag == "O32"

    def test_logU_skips_N43_when_NIII_incomplete(self):
        """_compute_logU should skip N43 when only one NIII member is present."""
        from jwspecabund._core import _compute_logU

        fluxes = {
            "NIV_1483": 1.0e-17,
            "NIV_1486": 5.0e-18,
            "NIII_1749": 5.0e-18,
            # NIII_1752 missing
            "OIII_5007": 3.0e-16,
            "OII_doublet": 8.0e-17,
        }

        logU, diag = _compute_logU(fluxes, Z_Zsun=0.1, ne_high=300.0)

        assert diag != "N43"

    def test_logU_uses_N43_when_both_complete(self):
        """_compute_logU should use N43 when both doublets are complete."""
        from jwspecabund._core import _compute_logU

        # Use fluxes that give a log(N43) within the validity range
        # [-3.0, 0.5] and a logU within [-3.5, -1.0].
        # NIV total = 4e-19, NIII total = 8e-18 → N43 = 0.05, log(N43) = -1.3.
        fluxes = {
            "NIV_1483": 2.0e-19,
            "NIV_1486": 2.0e-19,
            "NIII_1749": 5.0e-18,
            "NIII_1752": 3.0e-18,
        }

        logU, diag = _compute_logU(fluxes, Z_Zsun=0.1, ne_high=300.0)

        assert diag == "N43"
        assert logU is not None

    def test_ionic_single_NIII_member(self):
        """Single NIII member should still give N++/H+ using single-line emissivity."""
        from jwspecabund.direct import compute_ionic_abundances

        fluxes_both = {
            "NIII_1749": 5.0e-18,
            "NIII_1752": 3.0e-18,
            "HBETA": 5.0e-17,
        }
        fluxes_one = {
            "NIII_1749": 5.0e-18,
            # NIII_1752 missing
            "HBETA": 5.0e-17,
        }

        ionic_both = compute_ionic_abundances(fluxes_both, Te_high=13000.0, Te_low=11000.0, ne=300.0)
        ionic_one = compute_ionic_abundances(fluxes_one, Te_high=13000.0, Te_low=11000.0, ne=300.0)

        # Both should produce N++/H+.
        assert "N++/H+" in ionic_both
        assert "N++/H+" in ionic_one
        # Single-member result should be larger (uses single-line emissivity,
        # which is smaller than total, so abundance = flux/Hb * eps_Hb/eps_line is larger).
        assert ionic_one["N++/H+"] > ionic_both["N++/H+"]

    def test_ionic_complete_doublet_unchanged(self):
        """Complete doublets should give the same result as before."""
        from jwspecabund.direct import compute_ionic_abundances

        fluxes = {
            "CIII]_1907": 1.5e-17,
            "CIII]": 1.2e-17,
            "HBETA": 5.0e-17,
        }

        ionic = compute_ionic_abundances(fluxes, Te_high=13000.0, Te_low=11000.0, ne=300.0)

        assert "C++/H+" in ionic
        assert ionic["C++/H+"] > 0

    def test_ionic_no_members_skips(self):
        """No doublet members → no ionic abundance computed."""
        from jwspecabund.direct import compute_ionic_abundances

        fluxes = {
            "HBETA": 5.0e-17,
            "OIII_5007": 3.0e-16,
        }

        ionic = compute_ionic_abundances(fluxes, Te_high=13000.0, Te_low=11000.0, ne=300.0)

        assert "N++/H+" not in ionic
        assert "N+++/H+" not in ionic
        assert "C++/H+" not in ionic


# -----------------------------------------------------------------------
# Direct-sum N/O tests
# -----------------------------------------------------------------------

class TestDirectSum:
    """Tests for icf_method='direct_sum' tiered N/O calculation."""

    def test_direct_sum_all_ions(self):
        """Tier 1: N+ + N++ + N+++ all present → denominator is O+ + O++."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {
            "O+/H+": 1e-5,
            "O++/H+": 5e-5,
            "N+/H+": 3e-7,
            "N++/H+": 1e-5,
            "N+++/H+": 8e-6,
        }
        totals = compute_total_abundances(ionic, icf_method="direct_sum")

        OH = 1e-5 + 5e-5
        expected_NO = (3e-7 + 1e-5 + 8e-6) / OH
        np.testing.assert_allclose(totals["N/O"], expected_NO)
        assert totals["icf_method"] == "direct_sum"
        assert totals["NO_icf_name"] == "Np_Npp_Nppp"

    def test_direct_sum_uv_only(self):
        """Tier 2: only N++ + N+++ (no N+) → denominator is O++."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {
            "O+/H+": 1e-5,
            "O++/H+": 5e-5,
            "N++/H+": 1e-5,
            "N+++/H+": 8e-6,
        }
        totals = compute_total_abundances(ionic, icf_method="direct_sum")

        expected_NO = (1e-5 + 8e-6) / 5e-5
        np.testing.assert_allclose(totals["N/O"], expected_NO)
        assert totals["icf_method"] == "direct_sum"
        assert totals["NO_icf_name"] == "Npp_Nppp_Opp"

    def test_direct_sum_npp_only(self):
        """Tier 3: only N++ (no N+, no N+++) → denominator is O++."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {
            "O+/H+": 1e-5,
            "O++/H+": 5e-5,
            "N++/H+": 1e-5,
        }
        totals = compute_total_abundances(ionic, icf_method="direct_sum")

        expected_NO = 1e-5 / 5e-5
        np.testing.assert_allclose(totals["N/O"], expected_NO)
        assert totals["icf_method"] == "direct_sum"
        assert totals["NO_icf_name"] == "Npp_Opp"

    def test_direct_sum_optical_fallback(self):
        """Tier 4: only N+ (no UV) → falls back to Izotov+06."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {
            "O+/H+": 1e-5,
            "O++/H+": 5e-5,
            "N+/H+": 3e-7,
        }
        totals = compute_total_abundances(ionic, icf_method="direct_sum")

        assert totals["icf_method"] == "izotov06"
        assert totals["NO_icf_name"] == "izotov06_fallback"
        # N/O should be ICF(N) × N+/O+, which is > 0.
        assert totals["N/O"] > 0

    def test_direct_sum_includes_n4plus(self):
        """N4+ (NV) should be included in the sum when detected."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {
            "O+/H+": 1e-5,
            "O++/H+": 5e-5,
            "N+/H+": 3e-7,
            "N++/H+": 1e-5,
            "N+++/H+": 8e-6,
            "N4+/H+": 1e-7,
        }
        totals = compute_total_abundances(ionic, icf_method="direct_sum")

        OH = 1e-5 + 5e-5
        expected_NO = (3e-7 + 1e-5 + 8e-6 + 1e-7) / OH
        np.testing.assert_allclose(totals["N/O"], expected_NO)

    def test_gate_nitrogen_ions_removes_low_snr(self):
        """Low-SNR UV nitrogen lines should be excluded from the sum."""
        from jwspecabund._core import _gate_nitrogen_ions

        ionic = {
            "O+/H+": 1e-5,
            "O++/H+": 5e-5,
            "N+/H+": 3e-7,    # from NII 6585
            "N++/H+": 1e-5,   # from NIII 1749+1752
            "N+++/H+": 8e-6,  # from NIV 1483+1486
        }
        fluxes = {
            "NII_6585": 0.1,          # SNR = 5.0 → kept
            "NIII_1749": 4.0e-18,      # doublet SNR = 5e-18 / 3.6e-18 = 1.4
            "NIII_1752": 1.0e-18,
            "NIV_1483": 2.0e-18,       # doublet SNR = 3e-18 / 2.2e-18 = 1.4
            "NIV_1486": 1.0e-18,
        }
        errors = {
            "NII_6585": 0.02,
            "NIII_1749": 3.0e-18,
            "NIII_1752": 2.0e-18,
            "NIV_1483": 2.0e-18,
            "NIV_1486": 1.0e-18,
        }

        _gate_nitrogen_ions(ionic, fluxes, errors, snr_NO=2.0)

        # N+ kept (SNR=5), N++ removed (SNR~1.4), N+++ removed (SNR~1.4)
        assert ionic["N+/H+"] == 3e-7
        assert ionic["N++/H+"] == 0.0
        assert ionic["N+++/H+"] == 0.0

    def test_gate_nitrogen_ions_keeps_high_snr(self):
        """High-SNR nitrogen lines should be kept."""
        from jwspecabund._core import _gate_nitrogen_ions

        ionic = {
            "O+/H+": 1e-5,
            "O++/H+": 5e-5,
            "N+/H+": 3e-7,
            "N++/H+": 1e-5,
        }
        fluxes = {
            "NII_6585": 0.1,
            "NIII_1749": 5.0e-18,
            "NIII_1752": 3.0e-18,
        }
        errors = {
            "NII_6585": 0.02,
            "NIII_1749": 1.0e-18,
            "NIII_1752": 0.5e-18,
        }

        _gate_nitrogen_ions(ionic, fluxes, errors, snr_NO=2.0)

        # doublet SNR = 8e-18 / sqrt(1e-36 + 0.25e-36) = 8e-18 / 1.12e-18 = 7.1
        assert ionic["N+/H+"] == 3e-7
        assert ionic["N++/H+"] == 1e-5

    def test_gate_disabled_when_snr_zero(self):
        """snr_NO=0 should disable gating."""
        from jwspecabund._core import _gate_nitrogen_ions

        ionic = {"N++/H+": 1e-5, "O++/H+": 5e-5}
        fluxes = {"NIII_1749": 1e-19, "NIII_1752": 1e-20}
        errors = {"NIII_1749": 1e-18, "NIII_1752": 1e-18}

        _gate_nitrogen_ions(ionic, fluxes, errors, snr_NO=0.0)

        assert ionic["N++/H+"] == 1e-5  # unchanged


class TestDoubletCompletenessLogU:
    """Tests for logU SNR gating with partial doublets (split from TestDoubletCompleteness)."""

    @pytest.fixture(autouse=True)
    def _check_pyneb(self):
        """Skip all tests in this class if PyNEB is not installed."""
        pytest.importorskip("pyneb")

    def test_logU_skips_N43_when_member_low_snr(self):
        """_compute_logU should skip N43 when a doublet member has low SNR.

        UV doublet members are protected from the per-line SNR filter so
        they remain in the flux dict for ionic abundances, but the logU
        diagnostic requires both members to have SNR >= snr_logU.
        """
        from jwspecabund._core import _compute_logU

        fluxes = {
            "NIV_1483": 1.0e-18,   # low SNR member
            "NIV_1486": 5.0e-18,   # decent SNR
            "NIII_1749": 5.0e-18,
            "NIII_1752": 3.0e-18,
            "OIII_5007": 3.0e-16,
            "OII_doublet": 8.0e-17,
        }
        errors = {
            "NIV_1483": 2.0e-18,   # SNR = 0.5 → below threshold
            "NIV_1486": 1.0e-18,   # SNR = 5.0 → OK
            "NIII_1749": 1.0e-18,  # SNR = 5.0 → OK
            "NIII_1752": 1.0e-18,  # SNR = 3.0 → OK
            "OIII_5007": 5.0e-17,
            "OII_doublet": 1.5e-17,
        }

        # With errors → SNR gating should block N43 (NIV_1483 SNR < 3).
        logU, diag = _compute_logU(fluxes, Z_Zsun=0.1, ne_high=300.0,
                                    errors=errors, snr_logU=3.0)
        assert diag != "N43"

        # Without errors → no SNR gating, N43 should be used.
        logU2, diag2 = _compute_logU(fluxes, Z_Zsun=0.1, ne_high=300.0)
        assert diag2 == "N43"

    def test_uv_doublet_members_snr_protected(self):
        """UV doublet members should be in _SNR_PROTECTED."""
        from jwspecabund._core import _SNR_PROTECTED

        for name in ("NIII_1749", "NIII_1752", "NIV_1483", "NIV_1486",
                      "CIV_1", "CIV_2", "NV_1", "NV_2",
                      "CIII]_1907", "CIII]"):
            assert name in _SNR_PROTECTED, f"{name} not in _SNR_PROTECTED"
