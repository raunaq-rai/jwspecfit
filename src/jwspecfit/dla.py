"""DLA column density fitter — measure N_HI from Lya damping wings.

Fits a UV power-law continuum attenuated by a damped Lyman-alpha
absorber (DLA) to derive the neutral hydrogen column density N_HI
in the galaxy's ISM.

The model (following Pollock et al. 2026, Eq. 4) is:

    F(lambda) = F0 * (lambda/lambda_pivot)^beta_UV * exp(-tau_DLA)

where tau_DLA = C * a * N_HI * H(a, x) uses the Voigt-Hjerting
function H(a, x) evaluated via the Faddeeva function (exact).

The model is convolved with the instrumental line-spread function
(Gaussian with FWHM = lambda / R) before comparison to data, as
in Pollock et al. (2026).

Sampling is performed with ``dynesty`` nested sampling (matching
the paper's methodology).

References
----------
Pollock, C. L., et al. 2026, A&A, arXiv:2602.11783.
    Method and application to z > 9 galaxies.
Tepper-Garcia, T. 2006, MNRAS, 369, 2025.
    Voigt-Hjerting function framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import wofz

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
# Voigt-Hjerting function (exact via Faddeeva)
# --------------------------------------------------------------------------

def voigt_H(a: float, u: np.ndarray) -> np.ndarray:
    """Voigt-Hjerting function H(a, u) via the Faddeeva function.

    Computes the exact H(a, u) = Re[w(u + i*a)] where w(z) is the
    Faddeeva function.  This is the function that Tepper-Garcia (2006)
    approximates analytically; here we use the full solution.

    Parameters
    ----------
    a : float
        Voigt damping parameter.
    u : np.ndarray
        Dimensionless frequency offset from line centre.

    Returns
    -------
    np.ndarray
        H(a, u) at each u.
    """
    z = u + 1j * a
    return wofz(z).real


# --------------------------------------------------------------------------
# DLA optical depth
# --------------------------------------------------------------------------

def tau_DLA(
    wave_A: np.ndarray,
    log_NHI: float,
    z: float = 0.0,
    b_kms: float = _B_DEFAULT_KMS,
) -> np.ndarray:
    """Compute DLA optical depth at each wavelength.

    Implements Eq. 1 of Pollock et al. (2026):
        tau_DLA = N_HI * sigma_0 * H(a, u)

    where sigma_0 = sqrt(pi) * e^2 * f_alpha / (m_e * c * Delta_nu_D),
    a = Gamma_alpha / (4 pi Delta_nu_D), and u is the dimensionless
    frequency offset from Lya.

    Parameters
    ----------
    wave_A : np.ndarray
        Observed-frame wavelength in Angstrom.
    log_NHI : float
        log10(N_HI / cm^-2).
    z : float
        Source redshift (0 for rest-frame spectra).
    b_kms : float
        Doppler parameter in km/s (default 30).

    Returns
    -------
    np.ndarray
        Optical depth tau(lambda).
    """
    NHI = 10.0 ** log_NHI

    # Doppler width in frequency space.
    b_cms = b_kms * 1e5  # km/s -> cm/s
    lambda_0_cm = _LAMBDA_LYA_A * (1.0 + z) * 1e-8  # observed Lya in cm
    nu_0 = _C_CGS / lambda_0_cm
    delta_nu_D = (b_cms / _C_CGS) * nu_0

    # Voigt damping parameter.
    a = _GAMMA_ALPHA / (4.0 * np.pi * delta_nu_D)

    # Frequency offset u = (nu - nu_0) / delta_nu_D.
    wave_cm = np.asarray(wave_A) * 1e-8
    nu = _C_CGS / wave_cm
    u = (nu - nu_0) / delta_nu_D

    # Line-centre cross section.
    sigma_0 = (
        np.sqrt(np.pi) * _E_CGS ** 2 * _F_ALPHA
        / (_ME_CGS * _C_CGS * delta_nu_D)
    )

    H = voigt_H(a, u)
    tau = NHI * sigma_0 * H

    return np.maximum(tau, 0.0)


# --------------------------------------------------------------------------
# Spectral resolution convolution
# --------------------------------------------------------------------------

def _convolve_resolution(
    wave_A: np.ndarray,
    flux: np.ndarray,
    R: float,
) -> np.ndarray:
    """Convolve a spectrum with a Gaussian LSF at resolving power R.

    Parameters
    ----------
    wave_A : np.ndarray
        Wavelength array (must be uniformly or near-uniformly spaced).
    flux : np.ndarray
        Flux array to convolve.
    R : float
        Spectral resolving power (lambda / FWHM).

    Returns
    -------
    np.ndarray
        Convolved flux array (same length as input).
    """
    # Median pixel scale.
    dlam = np.median(np.diff(wave_A))
    # FWHM at the midpoint wavelength.
    lam_mid = np.median(wave_A)
    fwhm_A = lam_mid / R
    sigma_A = fwhm_A / 2.3548  # FWHM -> sigma

    # Kernel half-width in pixels.
    sigma_pix = sigma_A / dlam
    hw = int(4.0 * sigma_pix) + 1
    x = np.arange(-hw, hw + 1)
    kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
    kernel /= kernel.sum()

    # Pad edges to avoid convolution artifacts.
    padded = np.pad(flux, hw, mode="edge")
    convolved = np.convolve(padded, kernel, mode="same")
    return convolved[hw:-hw]


# --------------------------------------------------------------------------
# Emission line masking
# --------------------------------------------------------------------------

def _mask_emission_lines(
    wave_A: np.ndarray,
    z: float = 0.0,
    width_A: float = 10.0,
    exclude: set[str] | None = None,
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
    exclude : set of str, optional
        Line names to skip (not mask).

    Returns
    -------
    np.ndarray
        Boolean array (True = keep, False = masked).
    """
    keep = np.ones(len(wave_A), dtype=bool)
    skip = exclude or set()
    for name, lam_rest in REST_LINES_A.items():
        if name in skip:
            continue
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
    log_evidence : float
        Log-evidence (logZ) from dynesty.
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
    log_evidence: float = 0.0
    _lya_subtracted: bool = False
    lya_params: np.ndarray | None = None

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
        ]
        if self.lya_params is not None:
            lp = self.lya_params
            lines.append(
                f"Lya A_peak       = {lp[0]:.2e}  "
                f"sigma={lp[2]:.2f} A  alpha={lp[3]:.2f}"
            )
            lines.append(
                f"Lya fitted       = {'log_A_lya' in self.samples}"
            )
        lines.extend([
            f"z                = {self.z}",
            f"Av               = {self.Av}",
            f"log(Z)           = {self.log_evidence:.1f}",
            f"N pixels         = {len(self.wave_fit)}",
            f"N samples        = {len(self.samples['log_NHI'])}",
        ])
        return "\n".join(lines)

    def plot(
        self,
        ax: Any = None,
        show_residuals: bool = True,
        flux_unit: str = "fnu",
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
        flux_unit : str
            ``"fnu"`` for F_nu (default, same units as input) or
            ``"flam"`` for F_lambda (converted via F_lam = F_nu * c / lam^2).
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

        # Unit conversion factor.
        if flux_unit == "flam":
            conv = 1.0 / (self.wave_fit ** 2)
            conv = conv / np.median(conv)
            ylabel = r"$F_\lambda$ (relative)"
        else:
            conv = np.ones_like(self.wave_fit)
            ylabel = r"$F_\nu$ (flux density)"

        flux_plot = self.flux_fit * conv
        err_plot = self.flux_err_fit * conv
        model_plot = self.model_best * conv

        # Data.
        _data_label = (r"Data (Ly$\alpha$-subtracted)"
                       if self._lya_subtracted else "Data")
        data_kw = {"color": "k", "lw": 0.8, "alpha": 0.6, "label": _data_label}
        data_kw.update(kwargs)
        ax_main.step(wave_rest, flux_plot, where="mid", **data_kw)

        # Error band.
        ax_main.fill_between(
            wave_rest,
            flux_plot - err_plot,
            flux_plot + err_plot,
            color="grey", alpha=0.2, step="mid",
        )

        # --- Component decomposition ---
        lam_pivot = _LAMBDA_PIVOT_A * (1.0 + self.z)
        continuum = 10.0 ** self.log_F0 * (self.wave_fit / lam_pivot) ** self.beta_UV
        lya_obs = _LAMBDA_LYA_A * (1.0 + self.z)
        _R = getattr(self, '_R', None)

        # 1) Intrinsic continuum (no DLA, no IGM).
        ax_main.plot(
            wave_rest, continuum * conv,
            color="blue", lw=1, ls="--", alpha=0.5,
            label=rf"Intrinsic cont. ($\beta_{{UV}}={self.beta_UV:.2f}$)",
        )

        # 2) Continuum × DLA × IGM (no Lyα).
        tau = tau_DLA(self.wave_fit, self.log_NHI, z=self.z)
        cont_dla = continuum * np.exp(-tau)
        cont_dla = np.where(self.wave_fit < lya_obs, 0.0, cont_dla)
        if _R is not None:
            cont_dla = _convolve_resolution(self.wave_fit, cont_dla, _R)
        ax_main.plot(
            wave_rest, cont_dla * conv,
            color="purple", lw=1.5, alpha=0.8,
            label=rf"Cont. $\times$ DLA ($\log N_{{HI}}={self.log_NHI:.1f}$)",
        )

        # 3) Lyα component (convolved, with IGM cutoff).
        if self.lya_params is not None:
            from .models import asymmetric_gaussian
            from .io import _flam_to_ujy

            lp = self.lya_params
            lya_flam = asymmetric_gaussian(
                self.wave_fit, lp[0], lp[1], lp[2], lp[3],
            )
            lya_ujy = _flam_to_ujy(lya_flam, self.wave_fit * 1e-4)
            lya_ujy = np.where(self.wave_fit < lya_obs, 0.0, lya_ujy)
            if _R is not None:
                lya_ujy = _convolve_resolution(self.wave_fit, lya_ujy, _R)
            ax_main.plot(
                wave_rest, lya_ujy * conv,
                color="green", lw=1.5, alpha=0.8,
                label=r"Ly$\alpha$ (escaped)",
            )


        # Lya marker.
        lya_rest = _LAMBDA_LYA_A
        ax_main.axvline(lya_rest, color="orange", ls=":", alpha=0.5, lw=1)
        ax_main.text(
            lya_rest + 5, ax_main.get_ylim()[1] * 0.9,
            r"Ly$\alpha$", color="orange", fontsize=9,
        )

        # Set y-axis from data, not the error envelope.
        finite = np.isfinite(flux_plot)
        if finite.any():
            ylo = float(np.nanpercentile(flux_plot[finite], 2))
            yhi = float(np.nanpercentile(flux_plot[finite], 98))
            margin = 0.15 * max(yhi - ylo, abs(yhi) * 0.1)
            ax_main.set_ylim(ylo - margin, yhi + margin)

        ax_main.set_ylabel(ylabel)
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

    def corner(self, **kwargs: Any) -> Any:
        """Plot a corner plot of the posterior samples.

        Requires the ``corner`` package.

        Parameters
        ----------
        **kwargs
            Passed to ``corner.corner()``.

        Returns
        -------
        matplotlib Figure
            The corner plot figure.
        """
        import corner as corner_pkg

        cols = [self.samples["log_NHI"]]
        labels = [r"$\log(N_{\rm HI}/\mathrm{cm}^{-2})$"]
        truths = [self.log_NHI]

        # Only include beta/F0 if they have a real posterior (not fixed).
        beta_samples = self.samples["beta_UV"]
        if np.std(beta_samples) > 1e-10:
            cols.append(beta_samples)
            cols.append(self.samples["log_F0"])
            labels.extend([r"$\beta_{UV}$", r"$\log(F_0)$"])
            truths.extend([self.beta_UV, self.log_F0])

        # Include Lyα params if fitted jointly.
        if "log_A_lya" in self.samples:
            cols.append(self.samples["log_A_lya"])
            cols.append(self.samples["sigma_lya"])
            cols.append(self.samples["alpha_lya"])
            labels.extend([
                r"$\log(A_{\rm Ly\alpha})$",
                r"$\sigma_{\rm Ly\alpha}$ (\AA)",
                r"$\alpha_{\rm Ly\alpha}$",
            ])
            truths.extend([
                float(np.median(self.samples["log_A_lya"])),
                float(np.median(self.samples["sigma_lya"])),
                float(np.median(self.samples["alpha_lya"])),
            ])

        if len(cols) == 1:
            # 1D posterior — corner can't handle it, plot histogram.
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(cols[0], bins=50, density=True, color="steelblue", alpha=0.7)
            ax.axvline(truths[0], color="red", lw=1.5, label="Median")
            lo = np.percentile(cols[0], 16)
            hi = np.percentile(cols[0], 84)
            ax.axvline(lo, color="grey", ls="--", lw=1)
            ax.axvline(hi, color="grey", ls="--", lw=1)
            ax.set_xlabel(labels[0], fontsize=12)
            ax.set_ylabel("Posterior density")
            ax.set_title(
                f"${np.median(cols[0]):.2f}^{{+{hi - np.median(cols[0]):.2f}}}"
                f"_{{-{np.median(cols[0]) - lo:.2f}}}$",
                fontsize=12,
            )
            ax.legend(fontsize=9)
            fig.tight_layout()
            return fig

        data = np.column_stack(cols)

        defaults = dict(
            labels=labels,
            truths=truths,
            show_titles=True,
            title_kwargs={"fontsize": 11},
            quantiles=[0.16, 0.5, 0.84],
        )
        defaults.update(kwargs)
        fig = corner_pkg.corner(data, **defaults)
        return fig


# --------------------------------------------------------------------------
# Model evaluation (numpy — used by dynesty likelihood)
# --------------------------------------------------------------------------

def _evaluate_model(
    wave_A: np.ndarray,
    log_F0: float,
    beta_UV: float,
    log_NHI: float,
    z: float,
    R: float | None = None,
    b_kms: float = _B_DEFAULT_KMS,
    lya_ujy_func: object | None = None,
    lya_free_params: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Evaluate the DLA-attenuated power-law model.

    Following Pollock et al. (2026) Eq. 4, the intrinsic spectrum
    (continuum + emission lines) is multiplied by exp(-tau_DLA):

        F = (F0*(lam/lam_pivot)^beta + Lya(lam)) * exp(-tau_DLA)

    Parameters
    ----------
    wave_A : np.ndarray
        Observed-frame wavelengths.
    log_F0 : float
        log10 of continuum normalisation at the pivot wavelength.
    beta_UV : float
        UV spectral slope.
    log_NHI : float
        log10(N_HI / cm^-2).
    z : float
        Source redshift.
    R : float or None
        Spectral resolving power.  If not None, the model is
        convolved with a Gaussian LSF of FWHM = lambda / R.
    b_kms : float
        Doppler parameter in km/s.
    lya_ujy_func : callable or None
        Function that takes wavelength array (Å) and returns the
        fixed Lyα emission profile in µJy.  Added to the intrinsic
        spectrum before DLA absorption.
    lya_free_params : tuple of (log_A, sigma, alpha), optional
        Free Lyα parameters: log10(A_peak) in flam, sigma in Å,
        and skewness alpha.  Centroid fixed at Lyα rest × (1+z).
        The Lyα emission is added **after** DLA absorption — the
        observed Lyα photons already escaped the neutral gas via
        resonant scattering into the damping wing frequencies.

    Returns
    -------
    np.ndarray
        Model flux at each wavelength.
    """
    from .io import _flam_to_ujy
    from .models import asymmetric_gaussian

    F0 = 10.0 ** log_F0
    lam_pivot = _LAMBDA_PIVOT_A * (1.0 + z)
    continuum = F0 * (wave_A / lam_pivot) ** beta_UV

    # DLA absorption on the continuum only.
    tau = tau_DLA(wave_A, log_NHI, z=z, b_kms=b_kms)
    model = continuum * np.exp(-tau)

    # IGM absorption: complete Gunn-Peterson trough blueward of Lya.
    lya_obs = _LAMBDA_LYA_A * (1.0 + z)
    model = np.where(wave_A < lya_obs, 0.0, model)

    # Add Lyα emission AFTER DLA and IGM absorption.
    # The observed Lyα photons already escaped the DLA (via resonant
    # scattering into the damping wing frequencies), so they should
    # not be attenuated again.  The IGM cutoff still applies to the
    # blue wing of Lyα (absorbed at z > z_source).
    lya_ujy = None
    if lya_ujy_func is not None:
        lya_ujy = lya_ujy_func(wave_A)

    if lya_free_params is not None:
        log_A, sigma, alpha = lya_free_params
        A_peak = 10.0 ** log_A
        mu = _LAMBDA_LYA_A * (1.0 + z)
        lya_flam = asymmetric_gaussian(wave_A, A_peak, mu, sigma, alpha)
        lya_ujy = _flam_to_ujy(lya_flam, wave_A * 1e-4)

    if lya_ujy is not None:
        # Apply IGM cutoff to Lyα blue wing too.
        lya_ujy = np.where(wave_A < lya_obs, 0.0, lya_ujy)
        model = model + lya_ujy

    if R is not None:
        model = _convolve_resolution(wave_A, model, R)

    return model


# --------------------------------------------------------------------------
# Main fitting function
# --------------------------------------------------------------------------

def fit_NHI(
    wave_A: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    z: float = 0.0,
    *,
    continuum: np.ndarray | None = None,
    Av: float = 0.0,
    dust_law: str = "cardelli",
    Rv: float = 3.1,
    R: float | None = None,
    mask_lines: bool = False,
    mask_width_A: float = 10.0,
    fit_range_A: tuple[float, float] = (1050.0, 2000.0),
    mask_regions_A: list[tuple[float, float]] | None = None,
    fit_lya: bool = False,
    lya_params: np.ndarray | list | None = None,
    n_live: int = 500,
    seed: int = 42,
) -> DLAResult:
    """Fit the DLA column density from a Lya damping wing.

    Uses ``dynesty`` nested sampling (matching Pollock et al. 2026)
    with an exact Voigt-Hjerting profile and optional spectral
    resolution convolution.

    Parameters
    ----------
    wave_A : np.ndarray
        Wavelength array in Angstrom (observed frame).
    flux : np.ndarray
        Flux density array (raw, not continuum-subtracted).
    flux_err : np.ndarray
        1-sigma flux errors.
    z : float
        Source redshift (0 for rest-frame spectra).
    continuum : np.ndarray, optional
        Moving-average continuum estimate (same length as ``flux``).
        When provided, the DLA model is fitted to this smooth
        continuum instead of the raw flux.  This removes emission
        lines automatically and gives a clean damping wing signal.
        Only 3 parameters are fitted (``log_NHI``, ``beta_UV``,
        ``log_F0``); ``fit_lya`` and ``lya_params`` are ignored.
    Av : float
        Dust extinction A_V to correct for before fitting.
    dust_law : str
        Extinction law: ``"cardelli"`` or ``"salim"``.
    Rv : float
        Total-to-selective extinction ratio (default 3.1).
    R : float or None
        Spectral resolving power.  If provided, the model is
        convolved with a Gaussian LSF at each likelihood evaluation.
    mask_lines : bool
        If True, mask known emission lines.
    mask_width_A : float
        Half-width of line masks in rest-frame Angstrom.
    fit_range_A : tuple
        Rest-frame wavelength range for the fit.
    mask_regions_A : list of (float, float), optional
        Additional rest-frame wavelength regions to mask.
    fit_lya : bool
        If True, jointly fit Lyα emission with DLA absorption.
        Ignored when ``continuum`` is provided.
    lya_params : array-like of length 4, optional
        Fixed or hint Lyα params.  Ignored when ``continuum``
        is provided.
    n_live : int
        Number of live points for dynesty (default 500).
    seed : int
        RNG seed.

    Returns
    -------
    DLAResult
        Fit results with posteriors and Bayesian evidence.
    """
    import dynesty

    wave_A = np.asarray(wave_A, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)

    # --- Continuum mode: fit to smooth continuum estimate ---
    _use_continuum = continuum is not None
    if _use_continuum:
        continuum = np.asarray(continuum, dtype=np.float64)
        fit_lya = False
        lya_params = None

    # --- Dust correction ---
    if _use_continuum:
        # Apply dust correction to continuum and errors.
        flux_corr, err_corr = _apply_dust_correction(
            wave_A, continuum, flux_err, Av, dust_law, Rv,
        )
    else:
        flux_corr, err_corr = _apply_dust_correction(
            wave_A, flux, flux_err, Av, dust_law, Rv,
        )

    # --- Build fixed Lya model (only when fit_lya=False, no continuum) ---
    _lya_ujy_func = None
    if lya_params is not None and not fit_lya:
        lp = np.asarray(lya_params)
        if len(lp) != 4:
            raise ValueError(
                f"lya_params must have 4 elements [A_peak, mu, sigma, alpha], "
                f"got {len(lp)}."
            )
        from .models import asymmetric_gaussian
        from .io import _flam_to_ujy

        def _lya_ujy_func(w):
            lya_flam = asymmetric_gaussian(w, lp[0], lp[1], lp[2], lp[3])
            return _flam_to_ujy(lya_flam, w * 1e-4)

        logger.info(
            "Including fixed Lya: A=%.2e, mu=%.1f, sigma=%.1f, alpha=%.1f",
            lp[0], lp[1], lp[2], lp[3],
        )

    # --- Rest-frame wavelength range selection ---
    wave_rest = wave_A / (1.0 + z)
    in_range = (wave_rest >= fit_range_A[0]) & (wave_rest <= fit_range_A[1])

    # --- Lya reference wavelength ---
    lya_obs = _LAMBDA_LYA_A * (1.0 + z)

    # --- Emission line masking ---
    if mask_lines and not _use_continuum:
        _exclude = {"Lya"} if fit_lya else None
        line_mask = _mask_emission_lines(
            wave_A, z=z, width_A=mask_width_A, exclude=_exclude,
        )
        if not fit_lya:
            lya_mask_width = 8.0 * (1.0 + z)
            lya_emission_mask = (
                (wave_A < lya_obs - lya_mask_width)
                | (wave_A > lya_obs + lya_mask_width)
            )
            line_mask &= lya_emission_mask
    else:
        # No line masking needed for continuum mode.
        line_mask = np.ones(len(wave_A), dtype=bool)

    # --- Custom region masking ---
    if mask_regions_A:
        for lo, hi in mask_regions_A:
            lo_obs = lo * (1.0 + z)
            hi_obs = hi * (1.0 + z)
            line_mask &= (wave_A < lo_obs) | (wave_A > hi_obs)

    # --- Positive error and finite flux filter ---
    good = (err_corr > 0) & np.isfinite(flux_corr)

    # --- Combined mask ---
    use = in_range & line_mask & good
    if use.sum() < 10:
        raise ValueError(
            f"Only {use.sum()} pixels remain after masking. "
            "Check fit_range_A, mask_width_A, and data quality."
        )

    w = wave_A[use]
    f = flux_corr[use]
    e = err_corr[use]

    _mode = "continuum" if _use_continuum else ("fit_lya" if fit_lya else "standard")
    logger.info(
        "DLA fit (%s): %d pixels in [%.0f, %.0f] A rest-frame "
        "(z=%.3f, Av=%.2f).",
        _mode, len(w), fit_range_A[0], fit_range_A[1], z, Av,
    )

    # --- Initial guess for F0 from data ---
    pivot_obs = _LAMBDA_PIVOT_A * (1.0 + z)
    near_pivot = np.abs(w - pivot_obs) < 200.0 * (1.0 + z)
    if near_pivot.sum() > 5:
        log_F0_guess = float(np.log10(np.maximum(np.median(f[near_pivot]), 1e-30)))
    else:
        log_F0_guess = float(np.log10(np.maximum(np.median(f), 1e-30)))

    # --- Precompute inverse variance ---
    inv_var = 1.0 / e ** 2

    # ================================================================
    # Stage 1: Fit continuum (F0, beta_UV) from the RED side only,
    # where the DLA optical depth is negligible.
    # ================================================================
    # The wing boundary separates the DLA-affected region (blue)
    # from the unabsorbed continuum (red).  Use 1280 Å so the
    # power-law fit is anchored close to the wing transition.
    _wing_boundary_A = 1280.0 * (1.0 + z)
    red_mask = w > _wing_boundary_A
    if red_mask.sum() < 5:
        _wing_boundary_A = lya_obs + 50.0
        red_mask = w > _wing_boundary_A

    w_red = w[red_mask]
    f_red = f[red_mask]
    e_red = e[red_mask]

    # Fit F = F0 * (lam/lam_pivot)^beta in linear space via curve_fit.
    from scipy.optimize import curve_fit
    lam_pivot = _LAMBDA_PIVOT_A * (1.0 + z)

    def _power_law(lam, log_F0, beta):
        return 10.0 ** log_F0 * (lam / lam_pivot) ** beta

    pos_red = (f_red > 0) & (e_red > 0)
    if pos_red.sum() > 5:
        try:
            popt, _ = curve_fit(
                _power_law,
                w_red[pos_red], f_red[pos_red],
                p0=[log_F0_guess, -1.5],
                sigma=e_red[pos_red], absolute_sigma=True,
                bounds=([-30, -4.0], [30, 1.0]),
            )
            _fixed_log_F0 = float(popt[0])
            _fixed_beta = float(popt[1])
        except (RuntimeError, ValueError):
            _fixed_log_F0 = log_F0_guess
            _fixed_beta = -1.5
    else:
        _fixed_log_F0 = log_F0_guess
        _fixed_beta = -1.5

    logger.info(
        "Stage 1 (red continuum): log_F0=%.3f, beta_UV=%.3f "
        "from %d pixels redward of %.0f A.",
        _fixed_log_F0, _fixed_beta, int(red_mask.sum()), _wing_boundary_A,
    )

    # ================================================================
    # Stage 2: Fit log_NHI with F0 and beta fixed, focusing on the
    # damping wing region where τ_DLA shapes the spectrum.
    # ================================================================
    # Use only the wing region for the NHI fit so the flat red
    # continuum doesn't dominate the likelihood.
    wing_mask = w <= _wing_boundary_A
    w_wing = w[wing_mask]
    f_wing = f[wing_mask]
    inv_var_wing = inv_var[wing_mask]

    if wing_mask.sum() < 5:
        logger.warning(
            "Only %d pixels in the wing region. Falling back to full range.",
            wing_mask.sum(),
        )
        w_wing = w
        f_wing = f
        inv_var_wing = inv_var

    logger.info(
        "Stage 2 (wing fit): %d pixels in wing region (< %.0f A).",
        len(w_wing), _wing_boundary_A,
    )

    # 1D dynesty: only log_NHI is free.
    ndim = 1

    def _log_like_nhi(log_NHI):
        model = _evaluate_model(
            w_wing, _fixed_log_F0, _fixed_beta, log_NHI, z, R=R,
            lya_ujy_func=_lya_ujy_func if not fit_lya else None,
        )
        resid = f_wing - model
        return -0.5 * np.sum(resid ** 2 * inv_var_wing)

    # Quick grid search to find the best log_NHI and centre the prior.
    _grid = np.linspace(18.0, 23.0, 100)
    _grid_ll = np.array([_log_like_nhi(v) for v in _grid])
    _best_nhi = float(_grid[np.argmax(_grid_ll)])
    # Centre prior ±2 dex around the grid optimum, clipped to [18, 23].
    prior_lo_nhi = np.array([max(18.0, _best_nhi - 2.0)])
    prior_hi_nhi = np.array([min(23.0, _best_nhi + 2.0)])
    logger.info(
        "Grid search best: log_NHI=%.2f, prior=[%.1f, %.1f].",
        _best_nhi, prior_lo_nhi[0], prior_hi_nhi[0],
    )

    def prior_transform(u):
        return prior_lo_nhi + u * (prior_hi_nhi - prior_lo_nhi)

    def log_likelihood(theta):
        return _log_like_nhi(theta[0])

    # --- Run dynesty ---
    sampler = dynesty.NestedSampler(
        log_likelihood, prior_transform, ndim=ndim,
        nlive=n_live, rstate=np.random.default_rng(seed),
    )
    sampler.run_nested(print_progress=True, dlogz=0.01)
    results = sampler.results

    # --- Extract weighted posterior samples ---
    from dynesty.utils import resample_equal
    weights = np.exp(results.logwt - results.logz[-1])
    samples_arr = resample_equal(results.samples, weights)

    log_NHI_samples = samples_arr[:, 0]

    def _median_ci(arr):
        med = float(np.median(arr))
        lo = med - float(np.percentile(arr, 16))
        hi = float(np.percentile(arr, 84)) - med
        return med, (lo, hi)

    log_NHI_med, log_NHI_err = _median_ci(log_NHI_samples)

    # F0 and beta are fixed — no posterior, just point estimates.
    beta_UV_med = _fixed_beta
    beta_UV_err = (0.0, 0.0)
    log_F0_med = _fixed_log_F0
    log_F0_err = (0.0, 0.0)

    samples_dict = {
        "log_NHI": log_NHI_samples,
        "beta_UV": np.full_like(log_NHI_samples, _fixed_beta),
        "log_F0": np.full_like(log_NHI_samples, _fixed_log_F0),
    }

    # --- Surface density (Pollock+26 Eq. 7) ---
    Sigma_HI = 8e-21 * 10.0 ** log_NHI_med

    # --- Best-fit model (evaluated on full fit range for plotting) ---
    model_best = _evaluate_model(
        w, log_F0_med, beta_UV_med, log_NHI_med, z, R=R,
        lya_ujy_func=_lya_ujy_func if not fit_lya else None,
    )

    # --- Log-evidence ---
    log_evidence = float(results.logz[-1])

    final_lya_params = None
    if lya_params is not None:
        final_lya_params = np.asarray(lya_params)

    result = DLAResult(
        log_NHI=log_NHI_med,
        log_NHI_err=log_NHI_err,
        beta_UV=beta_UV_med,
        beta_UV_err=beta_UV_err,
        log_F0=log_F0_med,
        log_F0_err=log_F0_err,
        Sigma_HI=Sigma_HI,
        samples=samples_dict,
        wave_fit=np.array(w),
        flux_fit=np.array(f),
        flux_err_fit=np.array(e),
        model_best=model_best,
        z=z,
        Av=Av,
        log_evidence=log_evidence,
        _lya_subtracted=False,
        lya_params=final_lya_params,
    )
    result._R = R
    return result
