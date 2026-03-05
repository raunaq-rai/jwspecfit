"""Tests for jwspecfit.io — Spectrum I/O."""

import numpy as np
import pytest

from jwspecfit import (
    Spectrum, read_dict, read_fits, read_npz,
    save_result, load_result, export_lines_txt,
)
from jwspecfit.fitter import fit_lines
from .conftest import G395M_FITS, PRISM_FITS, STACK_NPZ


class TestReadFits:
    def test_prism_loads(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        assert isinstance(spec, Spectrum)
        assert spec.n_pix > 100
        assert spec.grating == "PRISM"
        assert spec.z == 6.0

    def test_grating_loads(self):
        spec = read_fits(G395M_FITS, z=5.0)
        assert spec.grating == "G395M"
        assert spec.n_pix > 100

    def test_wave_units(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        assert spec.wave_um.max() < 10.0
        assert spec.wave_A.max() > 1000.0
        np.testing.assert_allclose(spec.wave_A, spec.wave_um * 1e4)

    def test_edges_length(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        assert len(spec.wave_edges_A) == spec.n_pix + 1

    def test_dlam_positive(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        assert np.all(spec.dlam_A > 0)

    def test_mask_valid(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        mask = spec.mask_valid()
        assert mask.dtype == bool
        assert np.sum(mask) > 0


class TestReadNpz:
    def test_stack_loads(self):
        spec = read_npz(STACK_NPZ, z=6.0, R=100.0)
        assert spec.n_pix > 1000
        assert spec.R == 100.0
        assert spec.grating is None

    def test_stack_wave_range(self):
        spec = read_npz(STACK_NPZ, z=6.0, R=100.0)
        assert spec.wave_A.min() >= 900.0
        assert spec.wave_A.max() <= 80000.0


class TestReadDict:
    def test_from_dict(self):
        data = {
            "wave": np.linspace(1.0, 5.0, 100),
            "flux": np.random.default_rng(0).normal(0.1, 0.01, 100),
            "err": np.full(100, 0.01),
        }
        spec = read_dict(data, z=3.0, R=100.0)
        assert spec.n_pix == 100
        assert spec.z == 3.0


class TestSpectrum:
    def test_copy(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        spec2 = spec.copy()
        spec2.flux_ujy[0] = -999
        assert spec.flux_ujy[0] != -999

    def test_flux_flam_conversion(self):
        spec = read_fits(PRISM_FITS, z=6.0)
        flam = spec.flux_flam
        assert flam.shape == spec.flux_ujy.shape
        valid = spec.mask_valid()
        assert np.any(flam[valid] > 0)


class TestSaveLoadResult:
    def test_round_trip(self, prism_fit_result, tmp_path):
        outfile = tmp_path / "result.npz"
        save_result(prism_fit_result, outfile)
        loaded = load_result(outfile)
        assert loaded.success == prism_fit_result.success
        np.testing.assert_allclose(loaded.model_flux, prism_fit_result.model_flux, rtol=1e-10)
        np.testing.assert_allclose(loaded.chi2, prism_fit_result.chi2)
        assert set(loaded.lines.keys()) == set(prism_fit_result.lines.keys())
        for name in prism_fit_result.lines:
            np.testing.assert_allclose(loaded.lines[name].flux, prism_fit_result.lines[name].flux)

    def test_spectrum_preserved(self, prism_fit_result, tmp_path):
        outfile = tmp_path / "result.npz"
        save_result(prism_fit_result, outfile)
        loaded = load_result(outfile)
        np.testing.assert_allclose(loaded.spectrum.wave_um, prism_fit_result.spectrum.wave_um)
        assert loaded.spectrum.grating == prism_fit_result.spectrum.grating


class TestExportLinesTxt:
    def test_writes_file(self, prism_fit_result, tmp_path):
        outfile = tmp_path / "lines.txt"
        export_lines_txt(prism_fit_result, outfile)
        assert outfile.exists()
        lines = outfile.read_text().strip().split("\n")
        assert len(lines) >= 4  # 3 header + at least 1 line
        assert "flux" in lines[2]
        assert "EW_A" in lines[2]
        assert "SNR_p_err" in lines[2]
        assert "erg/s/cm2" in lines[1]


# -----------------------------------------------------------------------
# MCMC result save/load tests
# -----------------------------------------------------------------------

def _make_synthetic_mcmc_result():
    """Build a synthetic MCMCResult for testing without running MCMC."""
    from jwspecfit.constraints import ConstraintSet
    from jwspecmcmc.result import MCMCLineResult, MCMCResult

    rng = np.random.default_rng(42)
    n_pix = 200
    n_samples = 500
    n_lines = 2
    line_names = ["OIII_5007", "HBETA"]

    spec = Spectrum(
        wave_um=np.linspace(0.8, 1.2, n_pix),
        flux_ujy=rng.normal(0.1, 0.01, n_pix),
        err_ujy=np.full(n_pix, 0.01),
        grating="PRISM",
        z=6.0,
    )

    lines = {}
    for name in line_names:
        flux_post = rng.normal(1e-17, 1e-18, n_samples)
        lines[name] = MCMCLineResult(
            name=name,
            rest_wave_A=5007.0 if "OIII" in name else 4862.68,
            amplitude=float(np.median(flux_post)),
            amplitude_err=(1e-18, 1.2e-18),
            centroid_A=5007.0 * 7.0 if "OIII" in name else 4862.68 * 7.0,
            centroid_err=(0.5, 0.6),
            sigma_A=10.0,
            sigma_err=(0.3, 0.4),
            flux=float(np.median(flux_post)),
            flux_err=(1e-18, 1.2e-18),
            flux_posterior=flux_post,
            ew_A=50.0,
            snr=10.0,
        )

    flat_chains = rng.normal(0, 1, (n_samples, 3 * n_lines))
    flat_chains_free = rng.normal(0, 1, (n_samples, 3 * n_lines))
    flat_log_prob = rng.normal(-100, 10, n_samples)
    chains = rng.normal(0, 1, (32, 100, 3 * n_lines))
    params = np.median(flat_chains, axis=0)

    constraints = ConstraintSet(
        line_names=line_names,
        tie_nii=True,
        tie_balmer_to_oiii=True,
        tie_uv_doublets=False,
    )

    return MCMCResult(
        lines=lines,
        flat_chains=flat_chains,
        flat_chains_free=flat_chains_free,
        flat_log_prob=flat_log_prob,
        chains=chains,
        params=params,
        model_flux=rng.normal(0, 0.01, n_pix),
        continuum=np.full(n_pix, 0.1),
        spectrum=spec,
        line_names=line_names,
        constraints=constraints,
        convergence={"rhat_max": 1.01, "ess_min": 450},
        sampler_name="emcee",
        sampler_meta={"n_walkers": 32, "n_steps": 100},
    )


class TestSaveLoadMCMCResult:
    """Tests for jwspecmcmc save/load round-trip."""

    def test_mcmc_round_trip(self, tmp_path):
        from jwspecmcmc import save_mcmc_result, load_mcmc_result

        original = _make_synthetic_mcmc_result()
        outfile = tmp_path / "mcmc.npz"
        save_mcmc_result(original, outfile)
        loaded = load_mcmc_result(outfile)

        assert isinstance(loaded, type(original))
        assert loaded.sampler_name == original.sampler_name
        np.testing.assert_allclose(
            loaded.flat_chains, original.flat_chains, rtol=1e-10
        )
        np.testing.assert_allclose(
            loaded.flat_log_prob, original.flat_log_prob, rtol=1e-10
        )
        np.testing.assert_allclose(loaded.params, original.params, rtol=1e-10)
        assert set(loaded.lines.keys()) == set(original.lines.keys())
        for name in original.lines:
            np.testing.assert_allclose(
                loaded.lines[name].flux, original.lines[name].flux
            )
            np.testing.assert_allclose(
                loaded.lines[name].flux_err, original.lines[name].flux_err
            )

    def test_mcmc_flux_posterior_preserved(self, tmp_path):
        from jwspecmcmc import save_mcmc_result, load_mcmc_result

        original = _make_synthetic_mcmc_result()
        outfile = tmp_path / "mcmc.npz"
        save_mcmc_result(original, outfile)
        loaded = load_mcmc_result(outfile)

        for name in original.lines:
            np.testing.assert_allclose(
                loaded.lines[name].flux_posterior,
                original.lines[name].flux_posterior,
                rtol=1e-10,
            )

    def test_mcmc_spectrum_preserved(self, tmp_path):
        from jwspecmcmc import save_mcmc_result, load_mcmc_result

        original = _make_synthetic_mcmc_result()
        outfile = tmp_path / "mcmc.npz"
        save_mcmc_result(original, outfile)
        loaded = load_mcmc_result(outfile)

        np.testing.assert_allclose(
            loaded.spectrum.wave_um, original.spectrum.wave_um
        )
        assert loaded.spectrum.grating == original.spectrum.grating
        assert loaded.spectrum.z == original.spectrum.z

    def test_mcmc_constraints_preserved(self, tmp_path):
        from jwspecmcmc import save_mcmc_result, load_mcmc_result

        original = _make_synthetic_mcmc_result()
        outfile = tmp_path / "mcmc.npz"
        save_mcmc_result(original, outfile)
        loaded = load_mcmc_result(outfile)

        assert loaded.constraints is not None
        assert loaded.constraints.tie_nii == original.constraints.tie_nii
        assert loaded.constraints.tie_balmer_to_oiii == original.constraints.tie_balmer_to_oiii
        assert loaded.constraints.line_names == original.constraints.line_names

    def test_mcmc_chains_preserved(self, tmp_path):
        from jwspecmcmc import save_mcmc_result, load_mcmc_result

        original = _make_synthetic_mcmc_result()
        outfile = tmp_path / "mcmc.npz"
        save_mcmc_result(original, outfile)
        loaded = load_mcmc_result(outfile)

        assert loaded.chains is not None
        np.testing.assert_allclose(loaded.chains, original.chains, rtol=1e-10)

    def test_mcmc_no_chains(self, tmp_path):
        from jwspecmcmc import save_mcmc_result, load_mcmc_result

        original = _make_synthetic_mcmc_result()
        original.chains = None
        original.sampler_name = "nautilus"

        outfile = tmp_path / "mcmc_nautilus.npz"
        save_mcmc_result(original, outfile)
        loaded = load_mcmc_result(outfile)

        assert loaded.chains is None
        assert loaded.sampler_name == "nautilus"

    def test_mcmc_broad_round_trip(self, tmp_path):
        from jwspecmcmc import save_mcmc_result, load_mcmc_result
        from jwspecmcmc.result import MCMCBroadFitResult

        mcmc = _make_synthetic_mcmc_result()
        original = MCMCBroadFitResult(
            mcmc_result=mcmc,
            selected_model="broad1",
            bic_narrow=1200.0,
            bic_broad1=1180.0,
            bic_broad2=1195.0,
            bic_both=1190.0,
        )

        outfile = tmp_path / "mcmc_broad.npz"
        save_mcmc_result(original, outfile)
        loaded = load_mcmc_result(outfile)

        assert isinstance(loaded, MCMCBroadFitResult)
        assert loaded.selected_model == "broad1"
        np.testing.assert_allclose(loaded.bic_narrow, 1200.0)
        np.testing.assert_allclose(loaded.bic_broad1, 1180.0)
        np.testing.assert_allclose(loaded.bic_broad2, 1195.0)
        np.testing.assert_allclose(loaded.bic_both, 1190.0)
        assert set(loaded.lines.keys()) == set(mcmc.lines.keys())
        np.testing.assert_allclose(loaded.flat_chains, mcmc.flat_chains, rtol=1e-10)

    def test_mcmc_convergence_preserved(self, tmp_path):
        from jwspecmcmc import save_mcmc_result, load_mcmc_result

        original = _make_synthetic_mcmc_result()
        outfile = tmp_path / "mcmc.npz"
        save_mcmc_result(original, outfile)
        loaded = load_mcmc_result(outfile)

        assert loaded.convergence["rhat_max"] == pytest.approx(1.01)
        assert loaded.convergence["ess_min"] == pytest.approx(450)
        assert loaded.sampler_meta["n_walkers"] == 32
