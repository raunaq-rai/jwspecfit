"""Tests for the strong-line redshift fitter (jwspecfit.redshift)."""

from __future__ import annotations

import numpy as np
import pytest

import jwspecfit as jw
from jwspecfit.lines import REST_LINES_A
from jwspecfit.redshift import (
    DEFAULT_LINES,
    Peak,
    RedshiftResult,
    _evaluate_prior,
    _find_local_minima,
    fit_redshift,
)


# --- Helpers ---------------------------------------------------------------


def _make_spec(
    z_true: float,
    wave_um_lo: float = 2.87,
    wave_um_hi: float = 5.27,
    n_pix: int = 1661,
    grating: str = "G395M",
    line_amps: dict[str, float] | None = None,
    sigma_A: float | None = None,
    continuum: float = 0.05,
    noise: float = 0.05,
    seed: int = 0,
) -> jw.Spectrum:
    """Build a synthetic spectrum with strong lines at the requested z.

    Lines are injected at the grating-appropriate LSF width unless
    *sigma_A* is set explicitly.
    """
    from jwspecfit.resolution import resolve_R

    rng = np.random.default_rng(seed)
    wave_um = np.linspace(wave_um_lo, wave_um_hi, n_pix)
    wave_A = wave_um * 1e4
    flux = np.full_like(wave_A, continuum)
    if line_amps is None:
        # Mix UV and optical lines so the helper still injects something
        # at very high z (where optical falls off the red end) and at very
        # low z (where UV is below the blue end).
        line_amps = {
            "Lya": 4.0,
            "CIV_doublet": 1.5,
            "HEII_1640": 1.0,
            "CIII]": 1.5,
            "OII_doublet": 1.0,
            "HBETA": 0.6,
            "OIII_4959": 1.5,
            "OIII_5007": 4.5,
            "Ha": 3.0,
            "NII_6585": 0.5,
        }
    for name, amp in line_amps.items():
        mu_A = REST_LINES_A[name] * (1 + z_true)
        if sigma_A is None:
            R = float(resolve_R(np.array([mu_A * 1e-4]), grating=grating)[0])
            sig = mu_A / (2.3548 * R)
        else:
            sig = sigma_A
        # Only inject if the line falls in range.
        if not (wave_A[0] - 3 * sig <= mu_A <= wave_A[-1] + 3 * sig):
            continue
        flux += amp * np.exp(-0.5 * ((wave_A - mu_A) / sig) ** 2)
    err = np.full_like(flux, noise)
    flux += err * rng.standard_normal(len(flux))
    return jw.Spectrum(
        wave_um=wave_um, flux_ujy=flux, err_ujy=err, grating=grating,
    )


# --- Smoke and recovery ----------------------------------------------------


class TestRecovery:
    """fit_redshift should recover injected redshifts across the grid."""

    @pytest.mark.parametrize("z_true", [0.5, 2.0, 5.0, 8.0, 12.0])
    def test_synthetic_recovery_prism(self, z_true):
        spec = _make_spec(z_true, wave_um_lo=0.6, wave_um_hi=5.3,
                          n_pix=430, grating="PRISM")
        res = fit_redshift(spec, verbose=False)
        # The algorithm's resolution floor is a few coarse steps; the
        # statistical CI does not include LSF-model systematics so we
        # check the recovered MAP against truth instead of CI coverage.
        assert abs(res.z_best - z_true) < 5e-3, (
            f"z_best={res.z_best:.5f} too far from {z_true}"
        )

    def test_g395m_synthetic_z6(self):
        spec = _make_spec(6.1052, grating="G395M")
        res = fit_redshift(spec, verbose=False)
        assert abs(res.z_best - 6.1052) < 5e-4
        assert res.is_decisive


# --- Missing-line tolerance ------------------------------------------------


class TestNoDataTolerance:
    def test_recovery_when_some_lines_off_red_end(self):
        # PRISM red end is ~5.3 microns.  At z=8: Halpha = 5.91 microns (off),
        # but Hbeta = 4.38 micron, [OIII]5007 = 4.51 micron, [OII] = 3.36 micron
        # are all in range.  The fitter should recover z=8 from the blue lines
        # alone, and Halpha must not appear in the best peak's lines_used.
        spec = _make_spec(
            8.0,
            wave_um_lo=0.6, wave_um_hi=5.3, n_pix=430, grating="PRISM",
            line_amps={"HBETA": 1.0, "OIII_4959": 1.5, "OIII_5007": 4.5,
                       "OII_doublet": 1.0, "NeIII_3869": 0.8},
        )
        res = fit_redshift(spec, verbose=False)
        assert abs(res.z_best - 8.0) < 0.05
        # The best peak's lines_used must exclude Halpha (out of range).
        assert "Ha" not in res.peaks[0].lines_used
        # And it should rest on at least 3 lines.
        assert res.peaks[0].n_lines_used >= 3


# --- Prior shapes ----------------------------------------------------------


class TestPriors:
    def test_evaluate_prior_none(self):
        z = np.linspace(0, 20, 5)
        np.testing.assert_array_equal(_evaluate_prior(None, z), np.ones_like(z))

    def test_evaluate_prior_tuple(self):
        z = np.linspace(0, 20, 21)
        p = _evaluate_prior((4.0, 8.0), z)
        assert np.all(p[(z >= 4.0) & (z <= 8.0)] == 1.0)
        assert np.all(p[(z < 4.0) | (z > 8.0)] == 0.0)

    def test_evaluate_prior_window_list(self):
        z = np.linspace(0, 20, 21)
        p = _evaluate_prior([(2.0, 4.0), (7.0, 9.0)], z)
        in_supp = ((z >= 2) & (z <= 4)) | ((z >= 7) & (z <= 9))
        assert np.all(p[in_supp] == 1.0)
        assert np.all(p[~in_supp] == 0.0)

    def test_evaluate_prior_callable(self):
        z = np.linspace(0, 20, 5)
        p = _evaluate_prior(lambda z: np.exp(-(z - 8) ** 2 / 4), z)
        # Peaks near z=8.
        assert np.argmax(p) == np.argmin(np.abs(z - 8))

    def test_prior_excludes_truth(self):
        spec = _make_spec(8.0, grating="G395M")
        res = fit_redshift(
            spec, prior=[(0.0, 5.0), (10.0, 20.0)], verbose=False,
        )
        # z=8 is not in the prior support -> z_best cannot be in (5, 10).
        assert not (5.0 < res.z_best < 10.0)
        # The reported z_best must lie inside the prior support.
        assert (res.z_best <= 5.0) or (res.z_best >= 10.0)

    def test_prior_window_restricts_peak(self):
        spec = _make_spec(6.1052, grating="G395M")
        res_flat = fit_redshift(spec, verbose=False)
        res_window = fit_redshift(spec, prior=(5.0, 7.0), verbose=False)
        assert abs(res_flat.z_best - res_window.z_best) < 5e-4


# --- Decisiveness ----------------------------------------------------------


class TestDecisiveness:
    def test_clean_spectrum_decisive(self):
        spec = _make_spec(6.1052, grating="G395M", noise=0.02)
        res = fit_redshift(spec, verbose=False)
        assert res.is_decisive
        assert res.peaks[0].prob > 0.5

    def test_pure_noise_not_decisive(self):
        rng = np.random.default_rng(42)
        wave_um = np.linspace(2.87, 5.27, 1661)
        flux = 0.05 + 0.05 * rng.standard_normal(len(wave_um))
        err = np.full_like(flux, 0.05)
        spec = jw.Spectrum(
            wave_um=wave_um, flux_ujy=flux, err_ujy=err, grating="G395M",
        )
        res = fit_redshift(spec, verbose=False)
        # On pure noise we shouldn't claim a decisive z.
        assert not res.is_decisive


# --- Aliasing peak reported ------------------------------------------------


class TestAliasing:
    def test_ha_nii_aliasing_peak_present(self):
        """Halpha + [NII] alone admits an alias as [OIII] 4959/5007."""
        # Build a G395M spectrum with only Halpha + [NII]6585 at z=6.1052.
        # Lambda(Ha at z=6.1052) ~ 4.66 um, lambda([NII] at z=6.1052) ~ 4.69 um.
        # These could be misread as [OIII]4959/5007 at some other z.
        spec = _make_spec(
            6.1052, grating="G395M",
            line_amps={"Ha": 4.0, "NII_6585": 0.8},
        )
        res = fit_redshift(spec, verbose=False, refine_dchi2=50.0)
        # There should be more than one peak (truth + alias).
        assert len(res.peaks) >= 2
        # Truth is the strongest.
        assert abs(res.peaks[0].z - 6.1052) < 5e-3


# --- No early stopping -----------------------------------------------------


class TestNoEarlyStopping:
    def test_full_coarse_grid_evaluated(self, monkeypatch):
        """Confirm the per-z evaluator is called once per coarse grid point
        with prior support, regardless of how good a hit appears early."""
        from jwspecfit import redshift as rzmod

        calls = []
        orig = rzmod._per_z_chi2

        def counted(*args, **kwargs):
            calls.append(args[0])
            return orig(*args, **kwargs)

        monkeypatch.setattr(rzmod, "_per_z_chi2", counted)

        spec = _make_spec(6.1052, grating="G395M")
        # Restrict the grid for speed and predictability.
        res = fit_redshift(spec, z_min=5.5, z_max=6.5, dz_coarse=0.005,
                           verbose=False)
        # Coarse grid: np.arange(5.5, 6.5+0.005, 0.005) -> 201 points.
        coarse_calls = [c for c in calls if 5.5 <= c <= 6.5]
        # The same z values may appear again during refinement; ensure
        # every coarse-grid point was visited at least once.
        z_coarse_set = set(np.round(res.z_grid_coarse, 6))
        called_set = set(np.round(coarse_calls, 6))
        assert z_coarse_set <= called_set, (
            f"Missed {len(z_coarse_set - called_set)} coarse points"
        )


# --- min_lines_in_range filter ---------------------------------------------


class TestMinLinesInRange:
    def test_single_line_does_not_create_refined_peak(self):
        # G395M (2.87 - 5.27 um) at z=0.5: NONE of the default lines fall in
        # range (the bluest default line is Lyalpha at z=0.5 -> 1.82 um,
        # the reddest is [SII] at z=0.5 -> 1.01 um, all below 2.87 um).
        # So fit_redshift should raise rather than report a spurious peak.
        spec = _make_spec(
            0.5, grating="G395M", line_amps={"Ha": 2.0},
        )
        with pytest.raises(RuntimeError, match="No local minimum"):
            fit_redshift(spec, z_min=0.0, z_max=1.0, verbose=False)


# --- Result object structure -----------------------------------------------


class TestResultObject:
    def test_result_fields_populated(self):
        spec = _make_spec(6.1052, grating="G395M")
        res = fit_redshift(spec, verbose=False)
        assert isinstance(res, RedshiftResult)
        assert isinstance(res.peaks[0], Peak)
        assert res.z_grid_coarse.shape == res.chi2_coarse.shape
        assert res.z_grid_coarse.shape == res.P_z_coarse.shape
        assert res.lines_used == DEFAULT_LINES
        assert res.grating == "G395M"
        # CI bounds are ordered.
        assert res.z_ci68[0] <= res.z_best <= res.z_ci68[1]
        assert res.z_ci95[0] <= res.z_ci68[0]
        assert res.z_ci68[1] <= res.z_ci95[1]


# --- Local-minima helper ---------------------------------------------------


class TestFindLocalMinima:
    def test_simple_quadratic_has_one_min(self):
        x = np.linspace(-1, 1, 21)
        y = (x - 0.1) ** 2 + 1.0
        idx = _find_local_minima(y)
        assert len(idx) == 1
        assert abs(x[idx[0]] - 0.1) < 0.1

    def test_two_wells(self):
        x = np.linspace(0, 2 * np.pi, 201)
        y = -np.cos(2 * x)  # minima at x = 0, pi, 2pi (edges excluded)
        idx = _find_local_minima(y)
        # We expect at least the interior minimum at x = pi.
        assert any(abs(x[i] - np.pi) < 0.05 for i in idx)
