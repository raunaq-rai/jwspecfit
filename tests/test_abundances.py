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
