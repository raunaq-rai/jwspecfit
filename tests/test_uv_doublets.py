"""Tests for UV doublet tying (tie_uv_doublets parameter).

Covers:
    1. ConstraintSet unit tests — free_mask, apply, expand_free_to_full
    2. Integration tests — fit_lines and fit_with_broad with tie_uv_doublets=True
    3. API plumbing tests — parameter accepted at every level
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from jwspecfit import Spectrum, fit_with_broad
from jwspecfit.fitter import fit_lines, FitResult
from jwspecfit.constraints import (
    CIV_RATIO,
    NII_RATIO,
    NV_RATIO,
    OIII_UV_RATIO,
    ConstraintSet,
)
from jwspecfit.lines import REST_LINES_A

# ---------------------------------------------------------------------------
#  Helper: build a synthetic UV spectrum with known doublet lines
# ---------------------------------------------------------------------------

def _make_uv_spectrum(
    z: float = 3.0,
    lines_to_inject: dict[str, float] | None = None,
    R: float = 1000.0,
    noise_level: float = 0.005,
    seed: int = 42,
) -> Spectrum:
    """Create a synthetic spectrum containing UV doublet lines."""
    if lines_to_inject is None:
        lines_to_inject = {
            "CIV_1": 3.0e-17,
            "CIV_2": 1.5e-17,
            "CIII]_1907": 4.0e-17,
            "CIII]": 2.5e-17,
        }

    rng = np.random.default_rng(seed)

    obs_wavelengths = [REST_LINES_A[n] * (1 + z) for n in lines_to_inject]
    lam_min_A = min(obs_wavelengths) - 200.0
    lam_max_A = max(obs_wavelengths) + 200.0
    wave_um = np.linspace(lam_min_A * 1e-4, lam_max_A * 1e-4, 600)
    wave_A = wave_um * 1e4

    continuum_ujy = np.full_like(wave_A, 0.5)

    C_CGS = 2.99792458e10
    total_line_ujy = np.zeros_like(wave_A)
    for name, amplitude in lines_to_inject.items():
        mu_A = REST_LINES_A[name] * (1 + z)
        sigma_A = mu_A / R / 2.355
        inv = 1.0 / (np.sqrt(2.0) * sigma_A)

        edges = np.zeros(len(wave_A) + 1)
        edges[1:-1] = 0.5 * (wave_A[:-1] + wave_A[1:])
        edges[0] = 2 * wave_A[0] - edges[1]
        edges[-1] = 2 * wave_A[-1] - edges[-2]
        left, right = edges[:-1], edges[1:]
        cdf_r = 0.5 * (1.0 + erf((right - mu_A) * inv))
        cdf_l = 0.5 * (1.0 + erf((left - mu_A) * inv))
        width = right - left
        profile = (cdf_r - cdf_l) / width
        line_flam = amplitude * profile

        lam_cm = wave_um * 1e-4
        fnu_cgs = line_flam * 1e8 * lam_cm**2 / C_CGS
        total_line_ujy += fnu_cgs / 1e-29

    flux_ujy = continuum_ujy + total_line_ujy
    err_ujy = np.full_like(flux_ujy, noise_level)
    flux_ujy += rng.normal(0, noise_level, len(flux_ujy))

    return Spectrum(
        wave_um=wave_um,
        flux_ujy=flux_ujy,
        err_ujy=err_ujy,
        grating=None,
        z=z,
        R=R,
    )


# ===================================================================
#  1. ConstraintSet unit tests
# ===================================================================

class TestConstraintSetFreeMask:
    """Test free_mask() with tie_uv_doublets on and off."""

    def test_no_uv_tying(self):
        """With tie_uv_doublets=False, all UV doublet params are free."""
        names = ["CIV_1", "CIV_2", "CIII]_1907", "CIII]"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=False)
        free = cs.free_mask()
        assert free.all(), "All 12 parameters should be free"

    def test_amplitude_tied_free_mask(self):
        """Amplitude-tied doublets: secondary has 0 free params."""
        names = ["CIV_1", "CIV_2"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free = cs.free_mask()
        nL = 2
        assert free[0] is np.True_
        assert free[nL + 0] is np.True_
        assert free[2 * nL + 0] is np.True_
        assert free[1] is np.False_
        assert free[nL + 1] is np.False_
        assert free[2 * nL + 1] is np.False_

    def test_kinematic_tied_free_mask(self):
        """Kinematics-only: secondary amplitude free, centroid/sigma derived."""
        names = ["CIII]_1907", "CIII]"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free = cs.free_mask()
        nL = 2
        assert free[0] is np.True_
        assert free[nL + 0] is np.True_
        assert free[2 * nL + 0] is np.True_
        # CIII] secondary: amplitude free (not blended), centroid/sigma derived
        assert free[1] is np.True_
        assert free[nL + 1] is np.False_
        assert free[2 * nL + 1] is np.False_

    def test_blended_doublet_free_mask(self):
        """Blended doublet: secondary amplitude also constrained."""
        names = ["CIII]_1907", "CIII]"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True,
                           blended_doublets={"CIII]"})
        free = cs.free_mask()
        nL = 2
        # Secondary: all 3 params constrained when blended
        assert free[1] is np.False_
        assert free[nL + 1] is np.False_
        assert free[2 * nL + 1] is np.False_

    def test_nv_free_mask(self):
        """NV doublet: amplitude-tied (secondary fully constrained)."""
        names = ["NV_1", "NV_2"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free = cs.free_mask()
        nL = 2
        assert free[1] is np.False_
        assert free[nL + 1] is np.False_
        assert free[2 * nL + 1] is np.False_

    def test_oiii_uv_free_mask(self):
        """OIII] UV doublet: amplitude-tied (secondary fully constrained)."""
        names = ["OIII_1666", "OIII_1661"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free = cs.free_mask()
        nL = 2
        assert free[1] is np.False_
        assert free[nL + 1] is np.False_
        assert free[2 * nL + 1] is np.False_

    def test_niv_free_mask(self):
        """NIV] doublet: kinematics-only (amplitude free)."""
        names = ["NIV_1486", "NIV_1483"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free = cs.free_mask()
        nL = 2
        assert free[1] is np.True_
        assert free[nL + 1] is np.False_
        assert free[2 * nL + 1] is np.False_

    def test_niii_free_mask(self):
        """NIII] doublet: kinematics-only."""
        names = ["NIII_1749", "NIII_1752"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free = cs.free_mask()
        nL = 2
        assert free[1] is np.True_
        assert free[nL + 1] is np.False_
        assert free[2 * nL + 1] is np.False_

    def test_siiii_free_mask(self):
        """SiIII] doublet: kinematics-only."""
        names = ["SiIII_1", "SiIII_2"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free = cs.free_mask()
        nL = 2
        assert free[1] is np.True_
        assert free[nL + 1] is np.False_
        assert free[2 * nL + 1] is np.False_

    def test_free_count_reduction(self):
        """Tying should reduce the number of free parameters."""
        names = ["CIV_1", "CIV_2", "CIII]_1907", "CIII]",
                 "NIV_1486", "NIV_1483"]
        cs_off = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                               tie_uv_doublets=False)
        cs_on = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                              tie_uv_doublets=True)
        n_free_off = int(cs_off.free_mask().sum())
        n_free_on = int(cs_on.free_mask().sum())
        assert n_free_off == 18
        # CIV_2: -3, CIII]: -2 (A free), NIV_1483: -2 (A free) = 18 - 7 = 11
        # UV intercom width tying: CIII]_1907 and NIV_1486 widths tied
        # to first anchor — but neither is the anchor here since OIII_1666
        # isn't present.  CIII]_1907 is anchor, NIV_1486 width derived: -1
        # Total: 18 - 7 - 1 = 10
        assert n_free_on == 10

    def test_missing_secondary_is_no_op(self):
        """If only one doublet member is present, no constraint is applied."""
        names = ["CIV_1"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free = cs.free_mask()
        assert free.all()


class TestConstraintSetApply:
    """Test apply() correctly computes derived parameters."""

    def _make_params(self, names: list[str]) -> np.ndarray:
        nL = len(names)
        p = np.zeros(3 * nL)
        for i in range(nL):
            p[i] = 1.0 + 0.1 * i
            p[nL + i] = 5000.0 + 100.0 * i
            p[2 * nL + i] = 10.0 + 1.0 * i
        return p

    def test_civ_amplitude_ratio(self):
        names = ["CIV_1", "CIV_2"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        p = self._make_params(names)
        p_out = cs.apply(p)
        assert p_out[1] == pytest.approx(p_out[0] * CIV_RATIO)

    def test_civ_centroid_velocity_space(self):
        names = ["CIV_1", "CIV_2"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        p = self._make_params(names)
        p_out = cs.apply(p)
        nL = 2
        lam_ratio = REST_LINES_A["CIV_2"] / REST_LINES_A["CIV_1"]
        assert p_out[nL + 1] == pytest.approx(p_out[nL + 0] * lam_ratio)

    def test_civ_sigma_velocity_space(self):
        names = ["CIV_1", "CIV_2"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        p = self._make_params(names)
        p_out = cs.apply(p)
        nL = 2
        lam_ratio = REST_LINES_A["CIV_2"] / REST_LINES_A["CIV_1"]
        assert p_out[2 * nL + 1] == pytest.approx(p_out[2 * nL + 0] * lam_ratio)

    def test_nv_amplitude_ratio(self):
        names = ["NV_1", "NV_2"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        p = self._make_params(names)
        p_out = cs.apply(p)
        assert p_out[1] == pytest.approx(p_out[0] * NV_RATIO)

    def test_oiii_uv_amplitude_ratio(self):
        names = ["OIII_1666", "OIII_1661"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        p = self._make_params(names)
        p_out = cs.apply(p)
        assert p_out[1] == pytest.approx(p_out[0] * OIII_UV_RATIO)

    def test_ciii_amplitude_free(self):
        """CIII] amplitude is NOT overwritten (kinematics-only, not blended)."""
        names = ["CIII]_1907", "CIII]"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        p = self._make_params(names)
        original_a_sec = p[1]
        p_out = cs.apply(p)
        nL = 2
        assert p_out[1] == pytest.approx(original_a_sec)
        lam_ratio = REST_LINES_A["CIII]"] / REST_LINES_A["CIII]_1907"]
        assert p_out[nL + 1] == pytest.approx(p_out[nL + 0] * lam_ratio)
        assert p_out[2 * nL + 1] == pytest.approx(p_out[2 * nL + 0] * lam_ratio)

    def test_ciii_blended_amplitude_fixed(self):
        """When blended, CIII] amplitude IS overwritten."""
        from jwspecfit.constraints import _INTERCOM_LOW_DENSITY_RATIOS
        names = ["CIII]_1907", "CIII]"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True,
                           blended_doublets={"CIII]"})
        p = self._make_params(names)
        p_out = cs.apply(p)
        expected_ratio = _INTERCOM_LOW_DENSITY_RATIOS[("CIII]_1907", "CIII]")]
        assert p_out[1] == pytest.approx(p_out[0] * expected_ratio)

    def test_niv_amplitude_free(self):
        names = ["NIV_1486", "NIV_1483"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        p = self._make_params(names)
        original_a = p[1]
        p_out = cs.apply(p)
        assert p_out[1] == pytest.approx(original_a)

    def test_apply_is_idempotent(self):
        names = ["CIV_1", "CIV_2", "CIII]_1907", "CIII]"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        p = self._make_params(names)
        p1 = cs.apply(p)
        p2 = cs.apply(p1)
        np.testing.assert_array_almost_equal(p1, p2)

    def test_apply_without_tying_is_identity(self):
        names = ["CIV_1", "CIV_2", "CIII]_1907", "CIII]"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=False)
        p = self._make_params(names)
        p_out = cs.apply(p)
        np.testing.assert_array_equal(p, p_out)


class TestExpandFreeToFull:
    """Test expand_free_to_full round-trip."""

    def test_round_trip_amplitude_tied(self):
        names = ["CIV_1", "CIV_2"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free_mask = cs.free_mask()
        assert int(free_mask.sum()) == 3
        p_free = np.array([2.0, 6200.0, 5.0])
        p_full = cs.expand_free_to_full(p_free)
        nL = 2
        assert p_full[0] == pytest.approx(2.0)
        assert p_full[nL + 0] == pytest.approx(6200.0)
        assert p_full[2 * nL + 0] == pytest.approx(5.0)
        assert p_full[1] == pytest.approx(2.0 * CIV_RATIO)
        lam_ratio = REST_LINES_A["CIV_2"] / REST_LINES_A["CIV_1"]
        assert p_full[nL + 1] == pytest.approx(6200.0 * lam_ratio)
        assert p_full[2 * nL + 1] == pytest.approx(5.0 * lam_ratio)

    def test_round_trip_kinematic_tied(self):
        names = ["CIII]_1907", "CIII]"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free_mask = cs.free_mask()
        # Primary: 3 free, secondary: 1 free (amplitude only)
        assert int(free_mask.sum()) == 4
        p_free = np.array([4.0, 2.5, 7630.0, 8.0])
        p_full = cs.expand_free_to_full(p_free)
        nL = 2
        assert p_full[0] == pytest.approx(4.0)
        assert p_full[nL + 0] == pytest.approx(7630.0)
        assert p_full[2 * nL + 0] == pytest.approx(8.0)
        assert p_full[1] == pytest.approx(2.5)
        lam_ratio = REST_LINES_A["CIII]"] / REST_LINES_A["CIII]_1907"]
        assert p_full[nL + 1] == pytest.approx(7630.0 * lam_ratio)
        assert p_full[2 * nL + 1] == pytest.approx(8.0 * lam_ratio)

    def test_mixed_doublets(self):
        names = ["CIV_1", "CIV_2", "CIII]_1907", "CIII]", "NIV_1486", "NIV_1483"]
        cs = ConstraintSet(names, tie_nii=False, tie_balmer_to_oiii=False,
                           tie_uv_doublets=True)
        free_mask = cs.free_mask()
        n_free = int(free_mask.sum())
        # CIV_1: 3 free, CIV_2: 0
        # CIII]_1907: 3, CIII]: 1 (A only)
        # NIV_1486: 2 (A, mu — sigma tied to CIII]_1907 anchor), NIV_1483: 1 (A only)
        # Total = 3 + 0 + 3 + 1 + 2 + 1 = 10
        assert n_free == 10

        p_free = np.arange(1.0, n_free + 1.0)
        p_full = cs.expand_free_to_full(p_free)
        assert len(p_full) == 3 * 6
        assert np.all(np.isfinite(p_full))


# ===================================================================
#  2. Integration tests — fit_lines with tie_uv_doublets
# ===================================================================

class TestFitLinesUVDoublets:
    """Integration tests: fit_lines with tie_uv_doublets=True."""

    @pytest.fixture
    def uv_spectrum_civ_ciii(self):
        return _make_uv_spectrum(
            z=3.0,
            lines_to_inject={
                "CIV_1": 3.0e-17,
                "CIV_2": 1.5e-17,
                "CIII]_1907": 4.0e-17,
                "CIII]": 2.5e-17,
            },
            R=1000.0,
            noise_level=0.003,
        )

    def test_fit_succeeds_with_tying(self, uv_spectrum_civ_ciii):
        result = fit_lines(
            uv_spectrum_civ_ciii, z=3.0, R=1000.0,
            lines=["CIV_1", "CIV_2", "CIII]_1907", "CIII]"],
            tie_uv_doublets=True, n_boot=0,
        )
        assert isinstance(result, FitResult)
        assert result.success

    def test_fit_succeeds_without_tying(self, uv_spectrum_civ_ciii):
        result = fit_lines(
            uv_spectrum_civ_ciii, z=3.0, R=1000.0,
            lines=["CIV_1", "CIV_2", "CIII]_1907", "CIII]"],
            tie_uv_doublets=False, n_boot=0,
        )
        assert result.success

    def test_civ_flux_ratio_enforced(self, uv_spectrum_civ_ciii):
        result = fit_lines(
            uv_spectrum_civ_ciii, z=3.0, R=1000.0,
            lines=["CIV_1", "CIV_2", "CIII]_1907", "CIII]"],
            tie_uv_doublets=True, n_boot=0,
        )
        civ1 = result.lines["CIV_1"]
        civ2 = result.lines["CIV_2"]
        measured_ratio = civ2.amplitude / civ1.amplitude
        assert measured_ratio == pytest.approx(CIV_RATIO, rel=1e-6)

    def test_ciii_kinematics_tied(self, uv_spectrum_civ_ciii):
        result = fit_lines(
            uv_spectrum_civ_ciii, z=3.0, R=1000.0,
            lines=["CIV_1", "CIV_2", "CIII]_1907", "CIII]"],
            tie_uv_doublets=True, n_boot=0,
        )
        c1907 = result.lines["CIII]_1907"]
        c1909 = result.lines["CIII]"]
        v_sigma_1907 = c1907.sigma_A / c1907.centroid_A
        v_sigma_1909 = c1909.sigma_A / c1909.centroid_A
        assert v_sigma_1907 == pytest.approx(v_sigma_1909, rel=1e-4)

    def test_bootstrap_with_tying(self, uv_spectrum_civ_ciii):
        result = fit_lines(
            uv_spectrum_civ_ciii, z=3.0, R=1000.0,
            lines=["CIV_1", "CIV_2", "CIII]_1907", "CIII]"],
            tie_uv_doublets=True, n_boot=20, n_jobs=1,
        )
        assert result.success
        for name, lr in result.lines.items():
            if lr.flux > 0:
                assert lr.flux_err > 0, f"{name} should have positive error"


class TestFitLinesNIVDoublet:
    """Test NIV] doublet specifically (kinematics-only)."""

    @pytest.fixture
    def uv_spectrum_niv(self):
        return _make_uv_spectrum(
            z=3.0,
            lines_to_inject={
                "NIV_1486": 3.0e-17,
                "NIV_1483": 1.0e-17,
            },
            R=1000.0,
            noise_level=0.003,
        )

    def test_niv_kinematics_tied(self, uv_spectrum_niv):
        result = fit_lines(
            uv_spectrum_niv, z=3.0, R=1000.0,
            lines=["NIV_1486", "NIV_1483"],
            tie_uv_doublets=True, n_boot=0,
        )
        assert result.success
        niv86 = result.lines["NIV_1486"]
        niv83 = result.lines["NIV_1483"]
        v_sigma_86 = niv86.sigma_A / niv86.centroid_A
        v_sigma_83 = niv83.sigma_A / niv83.centroid_A
        assert v_sigma_86 == pytest.approx(v_sigma_83, rel=1e-4)

    def test_niv_amplitudes_free(self, uv_spectrum_niv):
        result = fit_lines(
            uv_spectrum_niv, z=3.0, R=1000.0,
            lines=["NIV_1486", "NIV_1483"],
            tie_uv_doublets=True, n_boot=0,
        )
        niv86 = result.lines["NIV_1486"]
        niv83 = result.lines["NIV_1483"]
        ratio = niv83.amplitude / niv86.amplitude
        assert 0.1 < ratio < 0.8, (
            f"NIV ratio {ratio:.3f} should be freely fitted, not fixed"
        )


# ===================================================================
#  3. Integration tests — fit_with_broad with tie_uv_doublets
# ===================================================================

class TestFitWithBroadUVDoublets:
    """Test that fit_with_broad accepts and passes through tie_uv_doublets."""

    def test_broad_mode_off_with_tying(self, prism_spectrum):
        result = fit_with_broad(
            prism_spectrum, z=6.0, grating="PRISM", mode="off",
            n_boot=0, tie_uv_doublets=True,
        )
        assert result.selected_model == "narrow"
        assert result.best_fit.success

    def test_broad_mode_auto_with_tying(self, grating_spectrum):
        result = fit_with_broad(
            grating_spectrum, z=5.0, grating="G395M", mode="auto",
            n_boot=0, n_boot_bic=0, tie_uv_doublets=True,
        )
        assert result.selected_model in ("narrow", "broad1", "broad2", "both")

    def test_bic_bootstrap_with_tying(self, grating_spectrum):
        result = fit_with_broad(
            grating_spectrum, z=5.0, grating="G395M", mode="auto",
            n_boot=0, n_boot_bic=3, n_jobs=1,
            snr_threshold=0.0, tie_uv_doublets=True,
        )
        assert result.selected_model in ("narrow", "broad1", "broad2", "both")
        if result.bic_bootstrap:
            for variant, dist in result.bic_bootstrap.items():
                assert len(dist) == 3


# ===================================================================
#  4. API plumbing — parameter accepted at every level
# ===================================================================

class TestAPIPlumbing:
    """Verify tie_uv_doublets is accepted by every public function."""

    def test_jwspecfit_fit_lines_accepts(self):
        import inspect
        import jwspecfit
        sig = inspect.signature(jwspecfit.fit_lines)
        assert "tie_uv_doublets" in sig.parameters
        p = sig.parameters["tie_uv_doublets"]
        assert p.default is True

    def test_jwspecfit_fit_with_broad_accepts(self):
        import inspect
        import jwspecfit
        sig = inspect.signature(jwspecfit.fit_with_broad)
        assert "tie_uv_doublets" in sig.parameters

    def test_jwspecfit_fitter_fit_lines_accepts(self):
        import inspect
        from jwspecfit.fitter import fit_lines as _fit_lines
        sig = inspect.signature(_fit_lines)
        assert "tie_uv_doublets" in sig.parameters

    def test_jwspecmcmc_fit_lines_accepts(self):
        import inspect
        import jwspecmcmc
        sig = inspect.signature(jwspecmcmc.fit_lines)
        assert "tie_uv_doublets" in sig.parameters
        p = sig.parameters["tie_uv_doublets"]
        assert p.default is True

    def test_jwspecmcmc_fit_with_broad_accepts(self):
        import inspect
        import jwspecmcmc
        sig = inspect.signature(jwspecmcmc.fit_with_broad)
        assert "tie_uv_doublets" in sig.parameters

    def test_engine_fit_lines_mcmc_accepts(self):
        import inspect
        from jwspecmcmc._engine import _fit_lines_mcmc
        sig = inspect.signature(_fit_lines_mcmc)
        assert "tie_uv_doublets" in sig.parameters

    def test_engine_fit_with_broad_mcmc_accepts(self):
        import inspect
        from jwspecmcmc._engine import _fit_with_broad_mcmc
        sig = inspect.signature(_fit_with_broad_mcmc)
        assert "tie_uv_doublets" in sig.parameters

    def test_broad_bic_bootstrap_single_accepts(self):
        import inspect
        from jwspecfit.broad import _bic_bootstrap_single
        sig = inspect.signature(_bic_bootstrap_single)
        assert "tie_uv_doublets" in sig.parameters

    def test_broad_fit_model_variant_accepts(self):
        import inspect
        from jwspecfit.broad import _fit_model_variant
        sig = inspect.signature(_fit_model_variant)
        assert "tie_uv_doublets" in sig.parameters


# ===================================================================
#  5. Default value tests
# ===================================================================

class TestDefaultValues:
    """Verify that the default tie_uv_doublets is True."""

    def test_default_is_true(self):
        cs = ConstraintSet(["CIV_1", "CIV_2"])
        assert cs.tie_uv_doublets is True

    def test_prism_fit_with_default(self, prism_fit_result):
        """Prism fit with default tie_uv_doublets=True should work."""
        assert prism_fit_result.success
        assert "OIII_5007" in prism_fit_result.lines

    def test_constraints_stored_on_result(self, prism_fit_result):
        assert prism_fit_result.constraints is not None
