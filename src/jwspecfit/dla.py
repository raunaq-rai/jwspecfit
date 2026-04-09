"""DLA column density fitter — measure N_HI from Lya damping wings.

Fits a UV power-law continuum attenuated by a damped Lyman-alpha
absorber (DLA) to derive the neutral hydrogen column density N_HI
in the galaxy's ISM.

The model is:
    F(lambda) = F0 * lambda^beta_UV * exp(-tau_DLA(lambda, N_HI))

where tau_DLA uses the Tepper-Garcia (2006) analytic approximation
to the Voigt-Hjerting function.

Sampling is performed with NumPyro NUTS (JAX-based HMC).

References
----------
Pollock, C. L., et al. 2026, A&A, arXiv:2602.11783.
    Method and application to z > 9 galaxies.
Tepper-Garcia, T. 2006, MNRAS, 369, 2025.
    Analytic Voigt-Hjerting approximation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .lines import REST_LINES_A

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Physical constants (CGS)
# --------------------------------------------------------------------------
_LAMBDA_LYA_A = 1215.670       # Lya rest wavelength (Angstrom)
_F_ALPHA = 0.4162              # Lya oscillator strength
_GAMMA_ALPHA = 6.265e8         # Lya damping constant (s^-1)
_E_CGS = 4.8032e-10            # electron charge (esu)
_ME_CGS = 9.1094e-28           # electron mass (g)
_C_CGS = 2.9979e10             # speed of light (cm/s)
_B_DEFAULT_KMS = 30.0          # default Doppler parameter (km/s)
_LAMBDA_PIVOT_A = 1500.0       # pivot wavelength for power-law normalisation


# --------------------------------------------------------------------------
# Voigt-Hjerting function (Tepper-Garcia 2006)
# --------------------------------------------------------------------------

def tepper_garcia_H(a: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Voigt-Hjerting function H(a, u) for DLA damping wings.

    Uses a Gaussian core plus 4-term asymptotic Lorentzian wing
    expansion.  The wing series diverges for |u| < 2, so u^2 is
    floored at 4.0 — this introduces <0.5% error near the line
    centre, but the optical depth there is >> 10^6 for any DLA
    (N_HI > 10^{20} cm^{-2}) so exp(-tau) = 0 regardless.

    Parameters
    ----------
    a : array-like
        Voigt damping parameter.
    u : array-like
        Dimensionless frequency offset from line centre.

    Returns
    -------
    array-like
        H(a, u) evaluated at each (a, u) pair.

    Notes
    -----
    Accurate to <0.5% across all u for typical DLA damping
    parameters (a ~ 10^{-4} to 10^{-2}).  Pure JAX — fully
    differentiable for use with NUTS.
    """
    u2 = u ** 2
    core = jnp.exp(-u2)

    # Asymptotic wing expansion: H_wing ~ (a/sqrt(pi)) * sum_n c_n / u^{2n}
    # Floor u^2 at 4.0 to avoid divergence near line centre.
    u2_safe = jnp.maximum(u2, 4.0)
    u4 = u2_safe ** 2
    u6 = u2_safe * u4
    u8 = u4 ** 2
    wing = (a / jnp.sqrt(jnp.pi)) * (
        1.0 / u2_safe + 1.5 / u4 + 3.75 / u6 + 13.125 / u8
    )

    return core + wing


# --------------------------------------------------------------------------
# DLA optical depth
# --------------------------------------------------------------------------

def tau_DLA(
    wave_A: jnp.ndarray,
    log_NHI: float,
    z: float = 0.0,
    b_kms: float = _B_DEFAULT_KMS,
) -> jnp.ndarray:
    """Compute DLA optical depth at each wavelength.

    Parameters
    ----------
    wave_A : array-like
        Observed-frame wavelength in Angstrom.
    log_NHI : float
        log10(N_HI / cm^-2).
    z : float
        Source redshift (0 for rest-frame spectra).
    b_kms : float
        Doppler parameter in km/s (default 30).

    Returns
    -------
    array-like
        Optical depth tau(lambda).
    """
    NHI = 10.0 ** log_NHI

    # Doppler width in frequency space.
    b_cms = b_kms * 1e5  # km/s -> cm/s
    lambda_0_cm = _LAMBDA_LYA_A * (1.0 + z) * 1e-8  # observed Lya in cm
    nu_0 = _C_CGS / lambda_0_cm
    delta_nu_D = (b_cms / _C_CGS) * nu_0

    # Voigt parameters.
    a = _GAMMA_ALPHA / (4.0 * jnp.pi * delta_nu_D)

    # Frequency offset: u = (nu - nu_0) / delta_nu_D
    # In wavelength: nu = c / (lambda * 1e-8), so
    # u = (c/(lambda*1e-8) - nu_0) / delta_nu_D
    wave_cm = wave_A * 1e-8
    nu = _C_CGS / wave_cm
    u = (nu - nu_0) / delta_nu_D

    # Line-centre cross section.
    # sigma_0 = (sqrt(pi) * e^2 * f) / (m_e * c * delta_nu_D)
    sigma_0 = (
        jnp.sqrt(jnp.pi) * _E_CGS ** 2 * _F_ALPHA
        / (_ME_CGS * _C_CGS * delta_nu_D)
    )

    H = tepper_garcia_H(a, u)
    tau = NHI * sigma_0 * H

    # Ensure non-negative (numerical noise can give tiny negatives).
    return jnp.maximum(tau, 0.0)


# --------------------------------------------------------------------------
# Emission line masking
# --------------------------------------------------------------------------

def _mask_emission_lines(
    wave_A: np.ndarray,
    z: float = 0.0,
    width_A: float = 10.0,
) -> np.ndarray:
    """Create a boolean mask that is True for pixels to *keep*.

    Parameters
    ----------
    wave_A : np.ndarray
        Observed-frame wavelength array.
    z : float
        Source redshift.
    width_A : float
        Half-width of each mask region in rest-frame Angstrom.

    Returns
    -------
    np.ndarray
        Boolean array (True = keep, False = masked).
    """
    keep = np.ones(len(wave_A), dtype=bool)
    for name, lam_rest in REST_LINES_A.items():
        lam_obs = lam_rest * (1.0 + z)
        width_obs = width_A * (1.0 + z)
        keep &= (wave_A < lam_obs - width_obs) | (wave_A > lam_obs + width_obs)
    return keep


# --------------------------------------------------------------------------
# Dust correction helpers
# --------------------------------------------------------------------------

def _apply_dust_correction(
    wave_A: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    Av: float,
    dust_law: str,
    Rv: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply dust correction to flux and errors.

    Parameters
    ----------
    wave_A : np.ndarray
        Wavelength array in Angstrom.
    flux : np.ndarray
        Observed flux.
    flux_err : np.ndarray
        Observed flux errors.
    Av : float
        V-band extinction.
    dust_law : str
        ``"cardelli"`` or ``"salim"``.
    Rv : float
        Total-to-selective extinction ratio.

    Returns
    -------
    tuple of np.ndarray
        (corrected_flux, corrected_err).
    """
    if Av == 0.0:
        return flux.copy(), flux_err.copy()

    if dust_law == "cardelli":
        from jwspecabund.dust import cardelli_extinction
        A_lambda = cardelli_extinction(wave_A, Av, Rv=Rv)
    elif dust_law == "salim":
        from jwspecabund.dust import salim_attenuation
        A_lambda = salim_attenuation(wave_A, Av, Rv=Rv)
    else:
        raise ValueError(f"Unknown dust_law: {dust_law!r}. Use 'cardelli' or 'salim'.")

    correction = 10.0 ** (0.4 * A_lambda)
    return flux * correction, flux_err * correction


# --------------------------------------------------------------------------
# Result dataclass
# --------------------------------------------------------------------------

@dataclass
class DLAResult:
    """Result of a DLA column density fit.

    Attributes
    ----------
    log_NHI : float
        Median posterior log10(N_HI / cm^-2).
    log_NHI_err : tuple of float
        (lower, upper) 68% CI half-widths.
    beta_UV : float
        Median posterior UV spectral slope.
    beta_UV_err : tuple of float
        (lower, upper) 68% CI half-widths.
    log_F0 : float
        Median posterior log10(F0) continuum normalisation.
    log_F0_err : tuple of float
        (lower, upper) 68% CI half-widths.
    Sigma_HI : float
        H I gas surface density in M_sun pc^-2 (from log_NHI).
    samples : dict
        Full posterior samples: {"log_NHI": arr, "beta_UV": arr, "log_F0": arr}.
    wave_fit : np.ndarray
        Wavelengths used in fit (after masking), observed frame.
    flux_fit : np.ndarray
        Dust-corrected fluxes used in fit.
    flux_err_fit : np.ndarray
        Dust-corrected flux errors used in fit.
    model_best : np.ndarray
        Best-fit model evaluated on wave_fit.
    z : float
        Redshift used in fit.
    Av : float
        Dust correction applied.
    """

    log_NHI: float
    log_NHI_err: tuple[float, float]
    beta_UV: float
    beta_UV_err: tuple[float, float]
    log_F0: float
    log_F0_err: tuple[float, float]
    Sigma_HI: float
    samples: dict[str, np.ndarray] = field(repr=False)
    wave_fit: np.ndarray = field(repr=False)
    flux_fit: np.ndarray = field(repr=False)
    flux_err_fit: np.ndarray = field(repr=False)
    model_best: np.ndarray = field(repr=False)
    z: float = 0.0
    Av: float = 0.0

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "DLA Fit Result",
            "=" * 40,
            f"log(N_HI/cm^-2)  = {self.log_NHI:.2f} "
            f"(+{self.log_NHI_err[1]:.2f}, -{self.log_NHI_err[0]:.2f})",
            f"Sigma_HI         = {self.Sigma_HI:.1f} Msun/pc^2",
            f"beta_UV          = {self.beta_UV:.2f} "
            f"(+{self.beta_UV_err[1]:.2f}, -{self.beta_UV_err[0]:.2f})",
            f"log(F0)          = {self.log_F0:.2f} "
            f"(+{self.log_F0_err[1]:.2f}, -{self.log_F0_err[0]:.2f})",
            f"z                = {self.z}",
            f"Av               = {self.Av}",
            f"N pixels         = {len(self.wave_fit)}",
            f"N samples        = {len(self.samples['log_NHI'])}",
        ]
        return "\n".join(lines)

    def plot(
        self,
        ax: Any = None,
        show_residuals: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot the DLA fit over the data.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to plot on.  If None, creates a new figure.
            If show_residuals is True, this is ignored and a new
            figure with two panels is created.
        show_residuals : bool
            If True, show a residual panel below the main plot.
        **kwargs
            Passed to the data plot (e.g. ``color``, ``alpha``).

        Returns
        -------
        matplotlib Figure
            The figure object.
        """
        import matplotlib.pyplot as plt

        if show_residuals:
            fig, (ax_main, ax_res) = plt.subplots(
                2, 1, figsize=(8, 5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            )
        else:
            if ax is None:
                fig, ax_main = plt.subplots(figsize=(8, 4))
            else:
                ax_main = ax
                fig = ax.get_figure()
            ax_res = None

        # Convert to rest frame for display.
        wave_rest = self.wave_fit / (1.0 + self.z)

        # Data.
        data_kw = {"color": "k", "lw": 0.8, "alpha": 0.6, "label": "Data"}
        data_kw.update(kwargs)
        ax_main.step(wave_rest, self.flux_fit, where="mid", **data_kw)

        # Error band.
        ax_main.fill_between(
            wave_rest,
            self.flux_fit - self.flux_err_fit,
            self.flux_fit + self.flux_err_fit,
            color="grey", alpha=0.2, step="mid",
        )

        # Best-fit model.
        ax_main.plot(
            wave_rest, self.model_best,
            color="red", lw=1.5, label="DLA model",
        )

        # Intrinsic continuum (no DLA).
        lam_pivot = _LAMBDA_PIVOT_A * (1.0 + self.z)
        continuum = 10.0 ** self.log_F0 * (self.wave_fit / lam_pivot) ** self.beta_UV
        ax_main.plot(
            wave_rest, continuum,
            color="blue", lw=1, ls="--", alpha=0.5,
            label=rf"Continuum ($\beta_{{UV}}={self.beta_UV:.2f}$)",
        )

        # Lya marker.
        lya_rest = _LAMBDA_LYA_A
        ax_main.axvline(lya_rest, color="orange", ls=":", alpha=0.5, lw=1)
        ax_main.text(
            lya_rest + 5, ax_main.get_ylim()[1] * 0.9,
            r"Ly$\alpha$", color="orange", fontsize=9,
        )

        ax_main.set_ylabel("Flux density")
        ax_main.legend(fontsize=9, frameon=False)
        ax_main.set_title(
            rf"$\log(N_{{\rm HI}}/\mathrm{{cm}}^{{-2}}) = "
            rf"{self.log_NHI:.2f}^{{+{self.log_NHI_err[1]:.2f}}}"
            rf"_{{-{self.log_NHI_err[0]:.2f}}}$",
            fontsize=11,
        )

        if ax_res is not None:
            residuals = self.flux_fit - self.model_best
            normalised = residuals / self.flux_err_fit
            ax_res.step(wave_rest, normalised, where="mid", color="k", lw=0.5)
            ax_res.axhline(0, color="red", ls="-", lw=0.5)
            ax_res.fill_between(
                wave_rest, -1, 1, color="grey", alpha=0.15, step="mid",
            )
            ax_res.set_ylabel(r"Residual ($\sigma$)")
            ax_res.set_xlabel(r"Rest wavelength ($\mathrm{\AA}$)")
            ax_res.set_ylim(-5, 5)
        else:
            ax_main.set_xlabel(r"Rest wavelength ($\mathrm{\AA}$)")

        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------
# Model evaluation (pure JAX, for use inside and outside numpyro)
# --------------------------------------------------------------------------

def _evaluate_model(
    wave_A: jnp.ndarray,
    log_F0: float,
    beta_UV: float,
    log_NHI: float,
    z: float,
    b_kms: float = _B_DEFAULT_KMS,
) -> jnp.ndarray:
    """Evaluate the DLA-attenuated power-law model.

    Parameters
    ----------
    wave_A : array-like
        Observed-frame wavelengths.
    log_F0 : float
        log10 of continuum normalisation.
    beta_UV : float
        UV spectral slope.
    log_NHI : float
        log10(N_HI / cm^-2).
    z : float
        Source redshift.
    b_kms : float
        Doppler parameter in km/s.

    Returns
    -------
    array-like
        Model flux at each wavelength.
    """
    F0 = 10.0 ** log_F0
    # Normalise wavelengths to pivot (1500 A observed-frame equivalent)
    # so F0 is the flux density at 1500 A and beta_UV is well-conditioned.
    lam_pivot = _LAMBDA_PIVOT_A * (1.0 + z)
    continuum = F0 * (wave_A / lam_pivot) ** beta_UV
    tau = tau_DLA(wave_A, log_NHI, z=z, b_kms=b_kms)
    return continuum * jnp.exp(-tau)


# --------------------------------------------------------------------------
# Main fitting function
# --------------------------------------------------------------------------

def fit_NHI(
    wave_A: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    z: float = 0.0,
    *,
    Av: float = 0.0,
    dust_law: str = "cardelli",
    Rv: float = 3.1,
    mask_lines: bool = True,
    mask_width_A: float = 10.0,
    fit_range_A: tuple[float, float] = (1050.0, 2000.0),
    mask_regions_A: list[tuple[float, float]] | None = None,
    n_warmup: int = 500,
    n_samples: int = 2000,
    seed: int = 42,
) -> DLAResult:
    """Fit the DLA column density from a Lya damping wing.

    Parameters
    ----------
    wave_A : np.ndarray
        Wavelength array in Angstrom (observed frame).
    flux : np.ndarray
        Flux density array.
    flux_err : np.ndarray
        1-sigma flux errors.
    z : float
        Source redshift (0 for rest-frame spectra).
    Av : float
        Dust extinction A_V to correct for before fitting.
    dust_law : str
        Extinction law: ``"cardelli"`` or ``"salim"``.
    Rv : float
        Total-to-selective extinction ratio (default 3.1).
    mask_lines : bool
        If True, mask known emission lines.
    mask_width_A : float
        Half-width of line masks in rest-frame Angstrom.
    fit_range_A : tuple
        Rest-frame wavelength range for the fit.
    mask_regions_A : list of (float, float), optional
        Additional rest-frame wavelength regions to mask, e.g.
        ISM absorption features: ``[(1255, 1270), (1296, 1310)]``.
        Each tuple is ``(lo, hi)`` in rest-frame Angstrom.
    n_warmup : int
        NUTS warmup iterations.
    n_samples : int
        NUTS posterior samples.
    seed : int
        RNG seed.

    Returns
    -------
    DLAResult
        Fit results with posteriors.
    """
    wave_A = np.asarray(wave_A, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)

    # --- Dust correction ---
    flux_corr, err_corr = _apply_dust_correction(
        wave_A, flux, flux_err, Av, dust_law, Rv,
    )

    # --- Rest-frame wavelength range selection ---
    wave_rest = wave_A / (1.0 + z)
    in_range = (wave_rest >= fit_range_A[0]) & (wave_rest <= fit_range_A[1])

    # --- Mask blueward of Lya line centre (IGM-absorbed / zero flux) ---
    lya_obs = _LAMBDA_LYA_A * (1.0 + z)
    # Keep everything redward of Lya centre — the damping wing
    # signal is at 1216–1300 A rest, so we must include this region.
    red_of_lya = wave_A > lya_obs

    # --- Emission line masking ---
    if mask_lines:
        line_mask = _mask_emission_lines(wave_A, z=z, width_A=mask_width_A)
        # Mask only the narrow Lya emission spike (±8 A rest-frame).
        # The damping wing extends from ~1220 A outward and must not
        # be masked.
        lya_mask_width = 8.0 * (1.0 + z)
        lya_emission_mask = (
            (wave_A < lya_obs - lya_mask_width)
            | (wave_A > lya_obs + lya_mask_width)
        )
        line_mask &= lya_emission_mask
    else:
        line_mask = np.ones(len(wave_A), dtype=bool)

    # --- Custom region masking (e.g. ISM absorption features) ---
    if mask_regions_A:
        for lo, hi in mask_regions_A:
            lo_obs = lo * (1.0 + z)
            hi_obs = hi * (1.0 + z)
            line_mask &= (wave_A < lo_obs) | (wave_A > hi_obs)

    # --- Positive error filter ---
    good_err = err_corr > 0

    # --- Combined mask ---
    use = in_range & line_mask & good_err & red_of_lya
    if use.sum() < 10:
        raise ValueError(
            f"Only {use.sum()} pixels remain after masking. "
            "Check fit_range_A, mask_width_A, and data quality."
        )

    w = wave_A[use]
    f = flux_corr[use]
    e = err_corr[use]

    logger.info(
        "DLA fit: %d pixels in [%.0f, %.0f] A rest-frame (z=%.3f, Av=%.2f).",
        len(w), fit_range_A[0], fit_range_A[1], z, Av,
    )

    # --- Initial guess for F0 from data ---
    # F0 is the flux at the pivot wavelength (1500 A rest).
    # Use median flux near the pivot to estimate it.
    pivot_obs = _LAMBDA_PIVOT_A * (1.0 + z)
    wave_rest_use = w / (1.0 + z)
    near_pivot = np.abs(w - pivot_obs) < 200.0 * (1.0 + z)
    if near_pivot.sum() > 5:
        log_F0_guess = float(np.log10(np.maximum(np.median(f[near_pivot]), 1e-30)))
    else:
        log_F0_guess = float(np.log10(np.maximum(np.median(f), 1e-30)))

    # --- Convert to JAX arrays ---
    w_jax = jnp.array(w)
    f_jax = jnp.array(f)
    e_jax = jnp.array(e)

    # --- NumPyro model ---
    def numpyro_model():
        log_NHI = numpyro.sample("log_NHI", dist.Uniform(18.0, 24.0))
        beta_UV = numpyro.sample("beta_UV", dist.Uniform(-4.0, 0.0))
        log_F0 = numpyro.sample(
            "log_F0",
            dist.Uniform(log_F0_guess - 5.0, log_F0_guess + 5.0),
        )

        model_flux = _evaluate_model(w_jax, log_F0, beta_UV, log_NHI, z)
        numpyro.sample("obs", dist.Normal(model_flux, e_jax), obs=f_jax)

    # --- Run NUTS ---
    rng_key = jax.random.PRNGKey(seed)
    kernel = NUTS(numpyro_model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                progress_bar=True)
    mcmc.run(rng_key)

    samples = mcmc.get_samples()

    # --- Extract posteriors ---
    log_NHI_samples = np.array(samples["log_NHI"])
    beta_UV_samples = np.array(samples["beta_UV"])
    log_F0_samples = np.array(samples["log_F0"])

    def _median_ci(arr):
        med = float(np.median(arr))
        lo = med - float(np.percentile(arr, 16))
        hi = float(np.percentile(arr, 84)) - med
        return med, (lo, hi)

    log_NHI_med, log_NHI_err = _median_ci(log_NHI_samples)
    beta_UV_med, beta_UV_err = _median_ci(beta_UV_samples)
    log_F0_med, log_F0_err = _median_ci(log_F0_samples)

    # --- Surface density (Pollock+26 Eq. 7) ---
    Sigma_HI = 8e-21 * 10.0 ** log_NHI_med

    # --- Best-fit model ---
    model_best = np.array(
        _evaluate_model(w_jax, log_F0_med, beta_UV_med, log_NHI_med, z)
    )

    return DLAResult(
        log_NHI=log_NHI_med,
        log_NHI_err=log_NHI_err,
        beta_UV=beta_UV_med,
        beta_UV_err=beta_UV_err,
        log_F0=log_F0_med,
        log_F0_err=log_F0_err,
        Sigma_HI=Sigma_HI,
        samples={
            "log_NHI": log_NHI_samples,
            "beta_UV": beta_UV_samples,
            "log_F0": log_F0_samples,
        },
        wave_fit=np.array(w),
        flux_fit=np.array(f),
        flux_err_fit=np.array(e),
        model_best=model_best,
        z=z,
        Av=Av,
    )
