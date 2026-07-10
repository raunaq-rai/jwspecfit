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
        """Observed Hgamma/Hbeta below intrinsic should give positive A_V.

        For Hgamma/Hbeta the blue line (Hgamma) is preferentially
        attenuated, so dust *reduces* the ratio below the Case B value
        of 0.468.
        """
        from jwspecabund.dust import compute_Av_from_balmer

        # Ratio of 0.3 < 0.468 → positive dust (blue suppressed).
        Av, Av_err = compute_Av_from_balmer(
            flux_num=0.3, flux_den=1.0,
            flux_num_err=0.01, flux_den_err=0.01,
            law="salim",
        )
        assert Av > 0.0
        assert Av_err > 0.0


class TestBalmerAnchor:
    """Tests for the `anchor` option in compute_Av_multi_balmer."""

    @staticmethod
    def _attenuate(flux_intrinsic, wave_A, Av, law="salim"):
        """Apply attenuation at a single wavelength (for synthesising tests)."""
        from jwspecabund.dust import salim_attenuation, cardelli_extinction
        fn = salim_attenuation if law == "salim" else cardelli_extinction
        A_lam = fn(np.array([wave_A]), Av)[0]
        return flux_intrinsic * 10.0 ** (-0.4 * A_lam)

    def _synthesise_balmer_fluxes(self, Av_true, law="salim"):
        """Build observed Balmer fluxes at known A_V using the master ladder."""
        from jwspecabund.dust import _BALMER_LADDER

        fluxes = {}
        errors = {}
        # Intrinsic Hβ flux = 1.0; all others scaled by their ladder ratio.
        for name, (wave, ratio_over_Hb) in _BALMER_LADDER.items():
            f_int = ratio_over_Hb * 1.0
            f_obs = self._attenuate(f_int, wave, Av_true, law=law)
            fluxes[name] = f_obs
            # 1% errors so SNR >> 3.
            errors[name] = 0.01 * f_obs
        return fluxes, errors

    def test_ha_anchor_recovers_av(self):
        """compute_Av_multi_balmer with anchor='Ha' recovers known A_V."""
        from jwspecabund.dust import compute_Av_multi_balmer

        Av_true = 0.6
        fluxes, errors = self._synthesise_balmer_fluxes(Av_true)

        out = compute_Av_multi_balmer(fluxes, errors, anchor="Ha")

        assert out["anchor"] == "Ha"
        assert out["n_lines"] >= 3
        # Hα itself is the anchor, so not in numerator list.
        assert all(r["line"] != "Ha" for r in out["individual"])
        np.testing.assert_allclose(out["Av"], Av_true, atol=1e-3)

    def test_hbeta_anchor_recovers_av(self):
        """Default anchor='HBETA' recovers known A_V (regression)."""
        from jwspecabund.dust import compute_Av_multi_balmer

        Av_true = 0.4
        fluxes, errors = self._synthesise_balmer_fluxes(Av_true)

        out = compute_Av_multi_balmer(fluxes, errors)

        assert out["anchor"] == "HBETA"
        assert out["n_lines"] >= 3
        assert all(r["line"] != "HBETA" for r in out["individual"])
        np.testing.assert_allclose(out["Av"], Av_true, atol=1e-3)

    def test_ha_and_hbeta_anchors_agree_on_synthetic_data(self):
        """Both anchors should return the same Av for the same synthetic spectrum."""
        from jwspecabund.dust import compute_Av_multi_balmer

        Av_true = 0.8
        fluxes, errors = self._synthesise_balmer_fluxes(Av_true)

        out_hb = compute_Av_multi_balmer(fluxes, errors, anchor="HBETA")
        out_ha = compute_Av_multi_balmer(fluxes, errors, anchor="Ha")

        np.testing.assert_allclose(out_hb["Av"], out_ha["Av"], atol=2e-3)

    def test_invalid_anchor_raises(self):
        """Unknown anchor must raise ValueError."""
        from jwspecabund.dust import compute_Av_multi_balmer

        with pytest.raises(ValueError, match="anchor"):
            compute_Av_multi_balmer(
                {"HBETA": 1.0, "HGAMMA": 0.4},
                {"HBETA": 0.01, "HGAMMA": 0.01},
                anchor="Hgamma",
            )

    def test_missing_anchor_returns_zero(self):
        """If the anchor line is absent, function returns Av=0, n_lines=0."""
        from jwspecabund.dust import compute_Av_multi_balmer

        fluxes = {"HBETA": 1.0, "HGAMMA": 0.4}
        errors = {"HBETA": 0.01, "HGAMMA": 0.01}
        out = compute_Av_multi_balmer(fluxes, errors, anchor="Ha")

        assert out["Av"] == 0.0
        assert out["n_lines"] == 0
        assert out["anchor"] == "Ha"

    def test_numerator_ratios_are_anchor_relative(self):
        """Intrinsic ratios in `individual` should be numerator/anchor."""
        from jwspecabund.dust import compute_Av_multi_balmer, _BALMER_LADDER

        # No dust → observed ratios equal intrinsic.
        fluxes, errors = self._synthesise_balmer_fluxes(Av_true=0.0)

        out = compute_Av_multi_balmer(fluxes, errors, anchor="Ha")
        _, ha_ratio = _BALMER_LADDER["Ha"]

        for r in out["individual"]:
            _, num_ratio_over_Hb = _BALMER_LADDER[r["line"]]
            expected = num_ratio_over_Hb / ha_ratio
            np.testing.assert_allclose(r["intrinsic_ratio"], expected, rtol=1e-12)

    def test_compute_abundances_threads_anchor(self):
        """compute_abundances(balmer_anchor='Ha') reaches derivation and logs it."""
        from dataclasses import dataclass

        from jwspecabund import compute_abundances

        @dataclass
        class MockLineResult:
            name: str
            flux: float
            flux_err: float
            snr: float

        @dataclass
        class MockFitResult:
            lines: dict

        # Synthesise Balmer fluxes at Av_true, plus a couple of metal lines so
        # the strong-line method has what it needs.  We only care that the
        # anchor is threaded through and the derived Av is correct.
        Av_true = 0.5
        fluxes, errors = self._synthesise_balmer_fluxes(Av_true)
        line_fluxes = dict(fluxes)
        line_errors = dict(errors)
        # Minimal metal lines for strong-line method (observed, so they get
        # dust-corrected inside compute_abundances).
        Hb_int = 1.0
        line_fluxes["OIII_5007"] = self._attenuate(5.0 * Hb_int, 5008.24, Av_true)
        line_errors["OIII_5007"] = 0.05 * line_fluxes["OIII_5007"]
        line_fluxes["OII_doublet"] = self._attenuate(2.0 * Hb_int, 3728.0, Av_true)
        line_errors["OII_doublet"] = 0.05 * line_fluxes["OII_doublet"]

        lines = {
            name: MockLineResult(
                name=name, flux=f, flux_err=line_errors[name],
                snr=f / line_errors[name] if line_errors[name] > 0 else 0.0,
            )
            for name, f in line_fluxes.items()
        }
        result = MockFitResult(lines=lines)

        abund = compute_abundances(
            result, z=2.0, method="strong_line",
            balmer_anchor="Ha", n_mc=50,
        )

        # Av should be recovered to within a few percent — this confirms the
        # anchor flowed all the way through to compute_Av_multi_balmer.
        assert abund.Av is not None
        np.testing.assert_allclose(abund.Av, Av_true, atol=5e-3)


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

    def test_icf_nitrogen_ratio_near_unity(self):
        """N/O ~ N+/O+ for typical HII regions.

        icf_nitrogen returns the *elemental* ICF (N/H = ICF*N+/H+), which
        is ~2 at moderate ionisation.  The physically meaningful invariant
        is that the effective ratio correction ICF * v (v = O+/O) stays
        near unity, i.e. N/O = ICF*N+/(O+ + O++) ~ N+/O+.
        """
        from jwspecabund.icf import icf_nitrogen

        O_plus, O_total = 0.5e-4, 1.0e-4   # v = O+/O ~ 0.5
        icf = icf_nitrogen(O_plus, O_total)
        assert icf > 1.5                    # elemental ICF, not ~1
        v = O_plus / O_total
        assert 0.9 <= icf * v <= 1.15       # N/O ~ N+/O+

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
# ICF reasoning diagnostics tests
# -----------------------------------------------------------------------

class TestICFReasoning:
    """Tests for _print_icf_reasoning() diagnostics in auto mode."""

    def test_auto_prints_reasoning(self, capsys):
        """icf_method='auto' should print ICF reasoning to stdout."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {"O+/H+": 1e-5, "O++/H+": 5e-5, "N+/H+": 3e-7, "N++/H+": 2e-6}
        compute_total_abundances(ionic, logU=-2.5, Z_Zsun=0.1, icf_method="auto")
        captured = capsys.readouterr()
        assert "N/O ICF REASONING" in captured.out
        assert "Detected nitrogen ions" in captured.out
        assert "Detected oxygen ions" in captured.out
        assert "Final selection" in captured.out

    def test_non_auto_no_output(self, capsys):
        """icf_method != 'auto' should NOT print ICF reasoning."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {"O+/H+": 1e-5, "O++/H+": 5e-5, "N+/H+": 3e-7, "N++/H+": 2e-6}
        compute_total_abundances(ionic, icf_method="direct_sum")
        captured = capsys.readouterr()
        assert "N/O ICF REASONING" not in captured.out

    def test_auto_martinez_eligible(self, capsys):
        """When logU and Z_Zsun are provided, martinez25 should be ELIGIBLE."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {"O+/H+": 1e-5, "O++/H+": 5e-5, "N+/H+": 3e-7}
        compute_total_abundances(ionic, logU=-2.5, Z_Zsun=0.1, icf_method="auto")
        captured = capsys.readouterr()
        assert "martinez25: ELIGIBLE" in captured.out

    def test_auto_martinez_not_eligible_missing_logU(self, capsys):
        """Without logU, martinez25 should be NOT ELIGIBLE."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {"O+/H+": 1e-5, "O++/H+": 5e-5, "N+/H+": 3e-7}
        compute_total_abundances(ionic, Z_Zsun=0.1, icf_method="auto")
        captured = capsys.readouterr()
        assert "martinez25: NOT ELIGIBLE" in captured.out
        assert "missing logU" in captured.out

    def test_auto_shows_detected_ions(self, capsys):
        """Should display detected ion values and 'not detected' for absent ones."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {"O+/H+": 1e-5, "O++/H+": 5e-5, "N++/H+": 2e-6}
        compute_total_abundances(ionic, icf_method="auto")
        captured = capsys.readouterr()
        assert "N++/H+: 2.0000e-06" in captured.out
        assert "N+/H+: not detected" in captured.out
        assert "N+++/H+: not detected" in captured.out

    def test_auto_shows_chosen_method(self, capsys):
        """Final selection should show the chosen method and N/O value."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {"O+/H+": 1e-5, "O++/H+": 5e-5, "N+/H+": 3e-7, "N++/H+": 2e-6}
        result = compute_total_abundances(ionic, logU=-2.5, Z_Zsun=0.1, icf_method="auto")
        captured = capsys.readouterr()
        assert f"Chosen method:  {result['icf_method']}" in captured.out
        assert "N/O value:" in captured.out

    def test_auto_no_nitrogen_shows_no_method(self, capsys):
        """When no nitrogen ions are detected, should report N/O cannot be computed."""
        from jwspecabund.direct import compute_total_abundances

        ionic = {"O+/H+": 1e-5, "O++/H+": 5e-5}
        compute_total_abundances(ionic, logU=-2.5, Z_Zsun=0.1, icf_method="auto")
        captured = capsys.readouterr()
        assert "could not be computed" in captured.out


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
    def test_marginalize_logU_false_keeps_logU_error(self):
        """marginalize_logU=False fixes logU in the ICFs but keeps its error.

        Like a fixed A_V, the median-flux point-estimate log(U) is fed to
        every draw's ICFs (so logU scatter does not inflate N/O), but log(U)
        is still resampled for reporting — its error must NOT be zeroed.
        """
        from jwspecabund import compute_abundances

        result = self._make_mock_fit_result({
            "OIII_4363": (0.5, 0.03),
            "OIII_4959": (3.0, 0.10),
            "OIII_5007": (9.0, 0.50),
            "HBETA": (1.0, 0.03),
            "Ha": (2.86, 0.05),
            "OII_doublet": (2.0, 0.20),
        })
        common = dict(z=2.0, method="direct", dust_correct=False,
                      n_mc=80, progress=False)

        a_fixed = compute_abundances(result, marginalize_logU=False, **common)
        a_marg = compute_abundances(result, marginalize_logU=True, **common)

        assert a_fixed.logU is not None and np.isfinite(a_fixed.logU)
        # Error is preserved (not forced to zero) — logU reporting is
        # decoupled from whether it propagates into the ICFs.
        err_fixed = np.asarray(a_fixed.logU_err, dtype=float)
        assert np.any(err_fixed > 0.0)
        # Central value agrees between the two modes (same reporting posterior).
        assert abs(a_fixed.logU - a_marg.logU) < 0.5

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
        """Te-essential and nitrogen lines are protected."""
        from jwspecabund._core import _filter_low_snr

        fluxes = {
            "OIII_4363": 0.001,
            "OIII_5007": 0.001,
            "OIII_4959": 0.001,
            "HBETA": 0.001,
            "NII_6585": 0.001,  # protected (nitrogen gating handles it)
            "Ha": 0.001,        # not protected
        }
        errors = {k: 1.0 for k in fluxes}  # all SNR = 0.001

        filt_f, _, excluded = _filter_low_snr(fluxes, errors, snr_thresh=2.0)

        assert "OIII_4363" in filt_f
        assert "OIII_5007" in filt_f
        assert "OIII_4959" in filt_f
        assert "HBETA" in filt_f
        assert "NII_6585" in filt_f  # protected from general filter
        assert "Ha" in excluded

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
        """Low-SNR [SII] doublet should fall back to the z-dependent ne."""
        from jwspecabund._core import _compute_multi_ne
        from jwspecabund.direct import ne_zone_fallback

        fluxes = {
            "SII_6718": 0.01,
            "SII_6732": 0.01,
        }
        errors = {
            "SII_6718": 0.1,   # SNR = 0.1
            "SII_6732": 0.1,   # SNR = 0.1
        }

        ne_low, ne_mid, ne_high, ne_Opp, _ = _compute_multi_ne(
            fluxes, errors=errors, snr_ne=3.0, z=0.0,
        )
        assert ne_low == pytest.approx(ne_zone_fallback("low", 0.0))
        assert ne_mid is None
        # No high-zone doublet -> z-dependent high fallback.
        assert ne_high == pytest.approx(ne_zone_fallback("high", 0.0))
        # No [Ar IV]/CIII] -> O²⁺-zone uses the mid z-fallback.
        assert ne_Opp == pytest.approx(ne_zone_fallback("mid", 0.0))

    def test_niv_density_invalid_skips_solve_and_falls_back(self):
        """An unphysical NIV] ratio must not be used for the density.

        When ``niv_density_invalid`` is set, ``_compute_multi_ne`` must skip
        the NIV] density solve entirely (so the bad ratio cannot corrupt
        ne_high) and record the reason — while the caller keeps the summed
        NIV] flux for the N³⁺/H⁺ abundance.
        """
        from jwspecabund._core import _compute_multi_ne

        # NIV] present with an unphysical ratio (F1483/F1486 = 2.0 > 1.5),
        # no other high-zone doublet → ne_high should fall back, NOT solve.
        fluxes = {"NIV_1483": 2.0e-17, "NIV_1486": 1.0e-17}
        errors = {"NIV_1483": 1.0e-19, "NIV_1486": 1.0e-19}  # very high SNR

        _, _, ne_high, _, failures = _compute_multi_ne(
            fluxes, errors=errors, snr_ne=3.0, niv_density_invalid=True,
        )
        # The failure must be attributed to the ratio, not a PyNEB solve, and
        # ne_high must be a finite fallback (not a value solved from 2.0).
        assert "NIV]" in failures.get("n_e(NIV])", "")
        assert "outside physical range" in failures["n_e(NIV])"]
        assert np.isfinite(ne_high)

    @pytest.mark.skipif(
        not pytest.importorskip("pyneb", reason="PyNEB required"),
        reason="PyNEB not available",
    )
    def test_high_snr_SII_uses_measured_ne(self):
        """High-SNR [SII] doublet should produce a measured n_e (not fallback)."""
        from jwspecabund._core import _compute_multi_ne
        from jwspecabund.direct import ne_zone_fallback

        # Ratio ~1.4 → low density regime
        fluxes = {
            "SII_6718": 1.4,
            "SII_6732": 1.0,
        }
        errors = {
            "SII_6718": 0.05,  # SNR = 28
            "SII_6732": 0.05,  # SNR = 20
        }

        ne_low, ne_mid, ne_high, ne_Opp, _ = _compute_multi_ne(
            fluxes, errors=errors, snr_ne=3.0, z=0.0,
        )
        # Should be a real measurement, not the z-fallback.
        assert ne_low != pytest.approx(ne_zone_fallback("low", 0.0))
        assert 10 < ne_low < 5000
        assert ne_mid is None

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
        # and fall back to the z-dependent default (gating disabled but the
        # SNR check returns False without errors, so the doublet is skipped).
        from jwspecabund.direct import ne_zone_fallback
        ne_low, ne_mid, _, _, _f = _compute_multi_ne(
            fluxes, errors=None, snr_ne=3.0, z=0.0,
        )
        assert ne_low == pytest.approx(ne_zone_fallback("low", 0.0))
        assert ne_mid is None

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
        """Low-SNR UV nitrogen doublets should be excluded."""
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

        _gate_nitrogen_ions(ionic, fluxes, errors, snr_NO=1.5)

        # N+ kept (SNR=5), N++ removed (SNR~1.4), N+++ removed (SNR~1.4)
        assert ionic["N+/H+"] == 3e-7
        assert ionic["N++/H+"] == 0.0
        assert ionic["N+++/H+"] == 0.0

    def test_gate_nitrogen_ions_keeps_high_snr(self):
        """Complete high-SNR doublets should be kept."""
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

        _gate_nitrogen_ions(ionic, fluxes, errors, snr_NO=1.5)

        # doublet SNR = 8e-18 / sqrt(1e-36 + 0.25e-36) = 8e-18 / 1.12e-18 = 7.1
        assert ionic["N+/H+"] == 3e-7
        assert ionic["N++/H+"] == 1e-5

    def test_gate_single_member_kept(self):
        """Doublet with one well-detected member should be kept."""
        from jwspecabund._core import _gate_nitrogen_ions

        ionic = {
            "O++/H+": 5e-5,
            "N++/H+": 1e-5,
        }
        fluxes = {
            "NIII_1749": 5.0e-18,  # only one member, SNR=10
        }
        errors = {
            "NIII_1749": 0.5e-18,
        }

        _gate_nitrogen_ions(ionic, fluxes, errors, snr_NO=1.5)

        # Kept: one member has SNR >= 1 and total SNR > snr_NO.
        assert ionic["N++/H+"] == 1e-5

    def test_gate_no_member_above_snr1(self):
        """Doublet where no member has SNR >= 1 should be excluded."""
        from jwspecabund._core import _gate_nitrogen_ions

        ionic = {
            "O++/H+": 5e-5,
            "N++/H+": 1e-5,
        }
        fluxes = {
            "NIII_1749": 1e-20,
        }
        errors = {
            "NIII_1749": 1e-18,
        }

        _gate_nitrogen_ions(ionic, fluxes, errors, snr_NO=1.5)

        assert ionic["N++/H+"] == 0.0

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


# -----------------------------------------------------------------------
# Lyα escape fraction tests
# -----------------------------------------------------------------------

class TestLyaEscapeFraction:
    """Tests for Lyα escape fraction computation."""

    def test_perfect_escape_no_dust(self):
        """With no dust and full Case B Lyα, f_esc should be ~1."""
        from jwspecabund.dust import compute_lya_escape_fraction

        # Hβ flux = 1.0, intrinsic Lyα/Hβ = 26.071, so intrinsic Lyα = 26.071.
        lya_flux = 26.071
        lya_err = 0.1
        fluxes_corr = {"HBETA": 1.0}
        errors_corr = {"HBETA": 0.01}

        out = compute_lya_escape_fraction(lya_flux, lya_err, fluxes_corr, errors_corr)
        assert out["n_lines"] == 1
        assert abs(out["f_esc"] - 1.0) < 0.01

    def test_half_escape(self):
        """If observed Lyα is half the intrinsic, f_esc ~0.5."""
        from jwspecabund.dust import compute_lya_escape_fraction

        lya_flux = 26.071 * 0.5
        lya_err = 0.1
        fluxes_corr = {"HBETA": 1.0}
        errors_corr = {"HBETA": 0.01}

        out = compute_lya_escape_fraction(lya_flux, lya_err, fluxes_corr, errors_corr)
        assert abs(out["f_esc"] - 0.5) < 0.01

    def test_multiple_balmer_lines(self):
        """Using multiple Balmer lines gives consistent f_esc."""
        from jwspecabund.dust import compute_lya_escape_fraction, LYA_CASE_B_RATIOS

        # Set up so all Balmer lines predict the same intrinsic Lyα.
        f_esc_true = 0.3
        fluxes_corr = {}
        errors_corr = {}
        for name, (ratio, _wave) in LYA_CASE_B_RATIOS.items():
            # Intrinsic Lyα = flux * ratio, so flux = Lyα_int / ratio.
            lya_int = 100.0  # arbitrary intrinsic Lyα
            fluxes_corr[name] = lya_int / ratio
            errors_corr[name] = fluxes_corr[name] * 0.01  # 1% error

        lya_flux = lya_int * f_esc_true
        lya_err = lya_flux * 0.01

        out = compute_lya_escape_fraction(lya_flux, lya_err, fluxes_corr, errors_corr)
        assert out["n_lines"] == len(LYA_CASE_B_RATIOS)
        assert abs(out["f_esc"] - f_esc_true) < 0.01

    def test_snr_filtering(self):
        """Lines below SNR threshold are excluded."""
        from jwspecabund.dust import compute_lya_escape_fraction

        fluxes_corr = {"HBETA": 1.0, "HGAMMA": 0.01}
        errors_corr = {"HBETA": 0.01, "HGAMMA": 0.1}  # Hγ SNR=0.1

        out = compute_lya_escape_fraction(
            26.071, 0.1, fluxes_corr, errors_corr, snr_min=3.0
        )
        assert out["n_lines"] == 1
        assert out["individual"][0]["line"] == "HBETA"

    def test_no_balmer_lines(self):
        """No Balmer lines → nan result."""
        from jwspecabund.dust import compute_lya_escape_fraction

        out = compute_lya_escape_fraction(10.0, 1.0, {}, {})
        assert np.isnan(out["f_esc"])
        assert out["n_lines"] == 0

    def test_negative_lya_flux(self):
        """Negative Lyα flux → nan."""
        from jwspecabund.dust import compute_lya_escape_fraction

        out = compute_lya_escape_fraction(-1.0, 0.1, {"HBETA": 1.0}, {"HBETA": 0.01})
        assert np.isnan(out["f_esc"])

    def test_individual_results_have_intrinsic(self):
        """Each per-line result includes the predicted intrinsic Lyα."""
        from jwspecabund.dust import compute_lya_escape_fraction

        out = compute_lya_escape_fraction(
            10.0, 0.5, {"HBETA": 1.0}, {"HBETA": 0.05}
        )
        assert len(out["individual"]) == 1
        r = out["individual"][0]
        assert abs(r["lya_intrinsic"] - 26.071) < 0.01
        assert r["lya_intrinsic_err"] > 0

    def test_case_b_ratios_from_pyneb(self):
        """Lyα/Hx ratios should match PyNEB Storey & Hummer 1995 values."""
        from jwspecabund.dust import LYA_CASE_B_RATIOS

        # Direct PyNEB values at T=1e4K, ne=100 cm^-3.
        assert abs(LYA_CASE_B_RATIOS["HBETA"][0] - 26.071) < 0.01
        assert abs(LYA_CASE_B_RATIOS["HGAMMA"][0] - 55.642) < 0.01
        assert abs(LYA_CASE_B_RATIOS["HDELTA"][0] - 100.593) < 0.01
        assert "Ha" not in LYA_CASE_B_RATIOS


class TestLyaEscapeFractionMC:
    """Tests for MC escape fraction computation."""

    def test_mc_recovers_point_estimate(self):
        """MC result should be consistent with the analytic estimate."""
        from jwspecabund.dust import (
            compute_lya_escape_fraction,
            compute_lya_escape_fraction_mc,
        )

        lya_flux = 13.036  # ~50% escape
        lya_err = 0.5
        fluxes_corr = {"HBETA": 1.0}
        errors_corr = {"HBETA": 0.05}

        pt = compute_lya_escape_fraction(lya_flux, lya_err, fluxes_corr, errors_corr)
        mc = compute_lya_escape_fraction_mc(
            lya_flux, lya_err, fluxes_corr, errors_corr,
            Av=0.0, Av_err=0.0, n_mc=5000,
            rng=np.random.default_rng(42),
        )

        assert mc["n_lines"] == 1
        # MC median should be close to analytic value.
        assert abs(mc["f_esc"] - pt["f_esc"]) < 0.05
        # Posterior should exist.
        assert len(mc["f_esc_posterior"]) > 100

    def test_mc_with_dust_uncertainty(self):
        """Increasing A_V uncertainty should widen the posterior."""
        from jwspecabund.dust import compute_lya_escape_fraction_mc

        lya_flux = 5.0
        lya_err = 0.2
        fluxes_corr = {"HBETA": 1.0, "HGAMMA": 0.468}
        errors_corr = {"HBETA": 0.05, "HGAMMA": 0.02}

        mc_tight = compute_lya_escape_fraction_mc(
            lya_flux, lya_err, fluxes_corr, errors_corr,
            Av=0.5, Av_err=0.01, n_mc=3000,
            rng=np.random.default_rng(42),
        )
        mc_wide = compute_lya_escape_fraction_mc(
            lya_flux, lya_err, fluxes_corr, errors_corr,
            Av=0.5, Av_err=0.5, n_mc=3000,
            rng=np.random.default_rng(42),
        )

        # Wider A_V uncertainty → wider f_esc posterior.
        spread_tight = mc_tight["f_esc_err"][0] + mc_tight["f_esc_err"][1]
        spread_wide = mc_wide["f_esc_err"][0] + mc_wide["f_esc_err"][1]
        assert spread_wide > spread_tight

    def test_mc_returns_asymmetric_errors(self):
        """MC errors should be (lo, hi) tuple."""
        from jwspecabund.dust import compute_lya_escape_fraction_mc

        mc = compute_lya_escape_fraction_mc(
            10.0, 0.5, {"HBETA": 1.0}, {"HBETA": 0.05},
            Av=0.0, Av_err=0.0, n_mc=2000,
            rng=np.random.default_rng(42),
        )
        assert isinstance(mc["f_esc_err"], tuple)
        assert len(mc["f_esc_err"]) == 2
        assert mc["f_esc_err"][0] > 0
        assert mc["f_esc_err"][1] > 0

    def test_mc_multiple_lines_weighted(self):
        """MC with multiple Balmer lines should use all of them."""
        from jwspecabund.dust import compute_lya_escape_fraction_mc, LYA_CASE_B_RATIOS

        f_esc_true = 0.2
        lya_int = 50.0
        fluxes_corr = {}
        errors_corr = {}
        for name, (ratio, _wave) in LYA_CASE_B_RATIOS.items():
            fluxes_corr[name] = lya_int / ratio
            errors_corr[name] = fluxes_corr[name] * 0.02

        mc = compute_lya_escape_fraction_mc(
            lya_int * f_esc_true, lya_int * f_esc_true * 0.05,
            fluxes_corr, errors_corr,
            Av=0.0, Av_err=0.0, n_mc=3000,
            rng=np.random.default_rng(42),
        )
        assert mc["n_lines"] == len(LYA_CASE_B_RATIOS)
        assert abs(mc["f_esc"] - f_esc_true) < 0.03

    def test_mc_no_balmer_returns_nan(self):
        """No Balmer lines → nan."""
        from jwspecabund.dust import compute_lya_escape_fraction_mc

        mc = compute_lya_escape_fraction_mc(
            10.0, 0.5, {}, {},
            Av=0.0, Av_err=0.0, n_mc=100,
        )
        assert np.isnan(mc["f_esc"])
        assert mc["n_lines"] == 0


class TestLyaEscapeInAbundances:
    """Test f_esc integration in compute_abundances."""

    def _make_result_with_lya(self, lya_flux=10.0, lya_err=0.5):
        """Build a minimal FitResult with Lyα and Hβ."""
        from jwspecfit.fitter import FitResult, LineResult
        from jwspecfit.io import Spectrum

        # Minimal spectrum.
        wave_um = np.linspace(0.6, 5.3, 1000)
        flux_ujy = np.ones(1000) * 0.1
        err_ujy = np.ones(1000) * 0.01
        spec = Spectrum(wave_um=wave_um, flux_ujy=flux_ujy, err_ujy=err_ujy, z=6.0)

        lines = {
            "Lya": LineResult(
                name="Lya", rest_wave_A=1215.67,
                amplitude=lya_flux, centroid_A=1216.0, sigma_A=1.5,
                flux=lya_flux, flux_err=lya_err,
                ew_A=50.0,
                snr_int_err=lya_flux / lya_err,
                snr_int_cont=lya_flux / lya_err,
                snr_peak_err=lya_flux / lya_err,
                snr_peak_cont=lya_flux / lya_err,
            ),
            "HBETA": LineResult(
                name="HBETA", rest_wave_A=4862.68,
                amplitude=1.0, centroid_A=4862.68, sigma_A=3.0,
                flux=1.0, flux_err=0.05,
                ew_A=100.0, snr_int_err=20.0,
                snr_int_cont=20.0, snr_peak_err=20.0, snr_peak_cont=20.0,
            ),
            "OIII_5007": LineResult(
                name="OIII_5007", rest_wave_A=5008.24,
                amplitude=5.0, centroid_A=5008.24, sigma_A=3.0,
                flux=5.0, flux_err=0.1,
                ew_A=500.0, snr_int_err=50.0,
                snr_int_cont=50.0, snr_peak_err=50.0, snr_peak_cont=50.0,
            ),
            "OIII_4959": LineResult(
                name="OIII_4959", rest_wave_A=4960.295,
                amplitude=1.67, centroid_A=4960.295, sigma_A=3.0,
                flux=1.67, flux_err=0.05,
                ew_A=167.0, snr_int_err=33.0,
                snr_int_cont=33.0, snr_peak_err=33.0, snr_peak_cont=33.0,
            ),
        }

        return FitResult(
            lines=lines,
            params=np.array([]),
            model_flux=flux_ujy,
            continuum=np.zeros(1000),
            residuals=np.zeros(1000),
            chi2=1.0,
            spectrum=spec,
            line_names=["Lya", "HBETA", "OIII_5007", "OIII_4959"],
            constraints=None,
            success=True,
            lya_params=np.array([1.0, 1216.0, 1.5, 5.0]),
        )

    def test_f_esc_computed_when_lya_present(self):
        """compute_abundances should populate lya_f_esc when Lyα is fitted."""
        from jwspecabund import compute_abundances

        result = self._make_result_with_lya(lya_flux=10.0, lya_err=0.5)
        abund = compute_abundances(
            result, z=6.0, dust_correct=False, method="strong_line",
        )
        assert abund.lya_f_esc is not None
        assert np.isfinite(abund.lya_f_esc)
        assert abund.lya_f_esc > 0
        # f_esc = 10.0 / (1.0 * 26.071) ≈ 0.429
        assert abs(abund.lya_f_esc - 10.0 / 26.071) < 0.01

    def test_f_esc_none_without_lya(self):
        """Without Lyα, lya_f_esc should be None."""
        from jwspecabund import compute_abundances
        from jwspecfit.fitter import FitResult, LineResult
        from jwspecfit.io import Spectrum

        wave_um = np.linspace(0.6, 5.3, 1000)
        spec = Spectrum(
            wave_um=wave_um,
            flux_ujy=np.ones(1000) * 0.1,
            err_ujy=np.ones(1000) * 0.01,
            z=6.0,
        )
        lines = {
            "HBETA": LineResult(
                name="HBETA", rest_wave_A=4862.68,
                amplitude=1.0, centroid_A=4862.68, sigma_A=3.0,
                flux=1.0, flux_err=0.05,
                ew_A=100.0, snr_int_err=20.0,
                snr_int_cont=20.0, snr_peak_err=20.0, snr_peak_cont=20.0,
            ),
            "OIII_5007": LineResult(
                name="OIII_5007", rest_wave_A=5008.24,
                amplitude=5.0, centroid_A=5008.24, sigma_A=3.0,
                flux=5.0, flux_err=0.1,
                ew_A=500.0, snr_int_err=50.0,
                snr_int_cont=50.0, snr_peak_err=50.0, snr_peak_cont=50.0,
            ),
            "OIII_4959": LineResult(
                name="OIII_4959", rest_wave_A=4960.295,
                amplitude=1.67, centroid_A=4960.295, sigma_A=3.0,
                flux=1.67, flux_err=0.05,
                ew_A=167.0, snr_int_err=33.0,
                snr_int_cont=33.0, snr_peak_err=33.0, snr_peak_cont=33.0,
            ),
        }
        result = FitResult(
            lines=lines, params=np.array([]),
            model_flux=np.ones(1000) * 0.1, continuum=np.zeros(1000),
            residuals=np.zeros(1000), chi2=1.0, spectrum=spec,
            line_names=["HBETA", "OIII_5007", "OIII_4959"],
            constraints=None, success=True,
        )
        abund = compute_abundances(
            result, z=6.0, dust_correct=False, method="strong_line",
        )
        assert abund.lya_f_esc is None

    def test_f_esc_in_summary(self):
        """f_esc should appear in the summary string."""
        from jwspecabund import compute_abundances

        result = self._make_result_with_lya(lya_flux=10.0, lya_err=0.5)
        abund = compute_abundances(
            result, z=6.0, dust_correct=False, method="strong_line",
        )
        summary = abund.summary()
        assert "f_esc(Lyα)" in summary


class TestLogUDiagnosticLock:
    """log(U) diagnostic locking for the Monte-Carlo posterior.

    Each MC draw must use the SAME diagnostic (N43 or O32) that the point
    estimate selected, so the log(U) posterior is never a silent mixture
    of the two (which would manufacture a bimodal/skewed error bar).
    """

    Z = 0.2
    NE = 5.0e4
    # Tiny errors so both N doublets always pass the SNR gate; only the
    # N43 ratio (set per draw) governs which branch is taken.
    ERRS = {k: 1e-4 for k in (
        "NIV_1483", "NIV_1486", "NIII_1749", "NIII_1752",
        "OIII_5007", "OII_doublet",
    )}

    @staticmethod
    def _fluxes(log_n43: float) -> dict[str, float]:
        niii, niv = 1.0, 10.0 ** log_n43
        return {
            "NIV_1483": niv * 0.4, "NIV_1486": niv * 0.6,
            "NIII_1749": niii * 0.6, "NIII_1752": niii * 0.4,
            "OIII_5007": 10.0, "OII_doublet": 2.0,
        }

    def test_unlocked_can_switch_diagnostic(self):
        """Without a lock, draws near the N43 validity edge flip N43<->O32."""
        from jwspecabund._core import _compute_logU

        # These log(N43) values straddle the N43->log(U) validity edge:
        # the higher ones give an in-range N43 log(U); the lower ones spill
        # below it and fall back to O32.
        diags = set()
        for log_n43 in (-2.55, -2.40, -2.80, -2.90):
            _, d = _compute_logU(
                self._fluxes(log_n43), self.Z, self.NE, errors=self.ERRS,
            )
            diags.add(d)
        assert diags == {"N43", "O32"}  # mixture confirmed

    def test_lock_o32_uses_only_o32(self):
        """lock_diag='O32' skips N43 entirely for every input."""
        from jwspecabund._core import _compute_logU

        for log_n43 in (-2.55, -2.40, -2.80, -2.90):
            val, diag = _compute_logU(
                self._fluxes(log_n43), self.Z, self.NE,
                errors=self.ERRS, lock_diag="O32",
            )
            assert diag == "O32"
            assert val is not None

    def test_lock_n43_does_not_fall_back_to_o32(self):
        """lock_diag='N43' returns None (rejected draw) instead of O32."""
        from jwspecabund._core import _compute_logU

        # In-range N43 -> used.
        val, diag = _compute_logU(
            self._fluxes(-2.40), self.Z, self.NE,
            errors=self.ERRS, lock_diag="N43",
        )
        assert diag == "N43" and val is not None

        # Out-of-range N43 -> rejected, NOT switched to O32.
        val, diag = _compute_logU(
            self._fluxes(-2.90), self.Z, self.NE,
            errors=self.ERRS, lock_diag="N43",
        )
        assert val is None and diag is None


class TestNeonICF:
    """Izotov+06 eq. 19 ICF for neon (regression for the dropped 1/w term)."""

    def test_icf_unity_at_high_ionisation(self):
        # w = O++/(O++O++) ~ 1 (fully doubly ionised) => ICF -> ~1.
        from jwspecabund.icf import icf_neon
        O_total = 10 ** (7.8 - 12)
        O_plus = 0.02 * O_total  # 98% O++
        icf = icf_neon(O_plus, O_total)
        assert 0.95 < icf < 1.10

    def test_icf_increases_as_ionisation_drops(self):
        from jwspecabund.icf import icf_neon
        O_total = 10 ** (7.8 - 12)
        icf_high = icf_neon(0.05 * O_total, O_total)  # high ionisation
        icf_low = icf_neon(0.50 * O_total, O_total)   # more O+ => more unseen Ne+
        assert icf_low > icf_high >= 1.0

    def test_icf_never_below_unity(self):
        from jwspecabund.icf import icf_neon
        O_total = 10 ** (8.5 - 12)  # high-Z branch dips below 1 before clamp
        for f in (0.02, 0.1, 0.3, 0.6, 0.9):
            assert icf_neon(f * O_total, O_total) >= 1.0

    def test_metallicity_branches_selected(self):
        # Exercise all three eq. 19 branches at fixed ionisation.
        from jwspecabund.icf import icf_neon
        w_frac = 0.1  # O+/O => w = 0.9
        vals = {oh: icf_neon(w_frac * 10 ** (oh - 12), 10 ** (oh - 12))
                for oh in (7.0, 7.8, 8.5)}
        # All physical (~1) but the branches differ.
        assert all(0.9 < v < 1.3 for v in vals.values())
        assert len({round(v, 4) for v in vals.values()}) == 3

    def test_regression_not_old_buggy_value(self):
        # The old (buggy) formula returned ~0.37 here; the fix must be ~1.
        from jwspecabund.icf import icf_neon
        O_total = 10 ** (7.8 - 12)
        icf = icf_neon(0.1 * O_total, O_total)
        assert icf > 0.9  # would have been 0.372 with the dropped 1/w term


class TestSulfurArgonNitrogenICF:
    """Izotov+06 eqs. 18/20/22 (regression for the dropped 1/v term)."""

    def test_sulfur_above_unity_at_high_ionisation(self):
        from jwspecabund.icf import icf_sulfur
        O_total = 10 ** (7.8 - 12)
        icf = icf_sulfur(0.1 * O_total, O_total)
        assert 1.3 < icf < 1.7  # was clamped to 1.0 with the buggy quartic

    def test_argon_above_unity_at_high_ionisation(self):
        from jwspecabund.icf import icf_argon
        O_total = 10 ** (7.8 - 12)
        icf = icf_argon(0.1 * O_total, O_total)
        assert 1.2 < icf < 1.6  # was clamped to 1.0 with the buggy cubic

    def test_nitrogen_elemental_icf_large_at_high_ionisation(self):
        from jwspecabund.icf import icf_nitrogen
        O_total = 10 ** (7.8 - 12)
        # Elemental N/H ICF grows as ionisation rises (unseen N++, N+++).
        assert icf_nitrogen(0.1 * O_total, O_total) > 5.0

    def test_all_three_increase_as_ionisation_rises(self):
        from jwspecabund.icf import icf_sulfur, icf_argon, icf_nitrogen
        O_total = 10 ** (7.8 - 12)
        for fn in (icf_sulfur, icf_argon, icf_nitrogen):
            hi = fn(0.05 * O_total, O_total)  # high ionisation
            lo = fn(0.60 * O_total, O_total)  # low ionisation
            assert hi > lo >= 1.0

    def test_metallicity_branches_distinct(self):
        from jwspecabund.icf import icf_sulfur, icf_argon, icf_nitrogen
        for fn in (icf_sulfur, icf_argon, icf_nitrogen):
            vals = {oh: fn(0.1 * 10 ** (oh - 12), 10 ** (oh - 12))
                    for oh in (7.0, 7.8, 8.5)}
            assert len({round(v, 4) for v in vals.values()}) == 3


class TestIcfTierValidation:
    """compute_abundances must reject unknown icf_tier names up front.

    Regression test: ``icf_tier="Np_Op"`` (invalid; the tier is named
    ``"NpOp"``) used to fail silently and degrade N/O to an unlabelled
    Izotov+06 fallback instead of raising.
    """

    def _make_mock_fit_result(self):
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

        line_fluxes = {
            "OIII_5007": (5.0, 0.1),
            "HBETA": (1.0, 0.02),
            "Ha": (2.86, 0.05),
            "OII_doublet": (2.0, 0.1),
        }
        lines = {
            name: MockLineResult(name, flux, err, flux / err)
            for name, (flux, err) in line_fluxes.items()
        }
        return MockFitResult(lines=lines)

    def test_invalid_tier_raises_value_error(self):
        from jwspecabund import compute_abundances

        with pytest.raises(ValueError, match="Np_Op"):
            compute_abundances(
                self._make_mock_fit_result(), z=2.0, icf_tier="Np_Op",
            )

    def test_error_message_lists_valid_tiers(self):
        from jwspecabund import compute_abundances
        from jwspecabund.martinez25_icf import VALID_NO_ICF_TIERS

        with pytest.raises(ValueError) as excinfo:
            compute_abundances(
                self._make_mock_fit_result(), z=2.0, icf_tier="bogus",
            )
        for tier in VALID_NO_ICF_TIERS:
            assert tier in str(excinfo.value)

    def test_valid_tier_passes_validation(self):
        from jwspecabund import compute_abundances

        # Strong-line path ignores icf_tier; the call must complete
        # without an icf_tier ValueError.
        abund = compute_abundances(
            self._make_mock_fit_result(), z=2.0, method="strong_line",
            n_mc=50, icf_tier="NpOp", progress=False,
        )
        assert abund.method == "strong_line"
