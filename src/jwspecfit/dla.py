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
_R_ALPHA = _GAMMA_ALPHA * _LAMBDA_LYA_A * 1e-8 / (4.0 * np.pi * _C_CGS)  # ≈ 2.02e-8
_IGM_Z_MIN_DEFAULT = 5.3       # IGM integration lower bound (Bosman+22 reionisation end)


# --------------------------------------------------------------------------
# IGM damping wing (Miralda-Escudé 1998 / Totani et al. 2006 closed form)
# --------------------------------------------------------------------------

def _IGM_I(x: np.ndarray) -> np.ndarray:
    """Helper integral for the Miralda-Escudé (1998) damping wing.

    I(x) = x^(9/2)/(1-x) + (9/7) x^(7/2) + (9/5) x^(5/2) + 3 x^(3/2)
           + 9 x^(1/2) - (9/2) ln[(1+x^(1/2))/(1-x^(1/2))]

    Defined for x ∈ (0, 1).
    """
    sqx = np.sqrt(x)
    return (
        x ** 4.5 / (1.0 - x)
        + (9.0 / 7.0) * x ** 3.5
        + (9.0 / 5.0) * x ** 2.5
        + 3.0 * x ** 1.5
        + 9.0 * sqx
        - 4.5 * np.log((1.0 + sqx) / (1.0 - sqx))
    )


def tau_IGM_DW(
    wave_A: np.ndarray,
    x_HI: float,
    z_source: float,
    z_min: float = _IGM_Z_MIN_DEFAULT,
    cosmo: Any = None,
) -> np.ndarray:
    """IGM damping-wing optical depth from a uniformly neutral IGM.

    Implements the closed-form solution of Miralda-Escudé (1998), as
    written in Totani et al. (2006) Eq. 3 and used in Pollock et al.
    (2026) Eq. 3.  Computes the redward Lyα damping-wing absorption
    by neutral hydrogen of mean fraction ``x_HI`` distributed
    uniformly between redshift ``z_min`` and ``z_source``.

    Parameters
    ----------
    wave_A : np.ndarray
        Observed-frame wavelength array in Angstrom.
    x_HI : float
        Volume-averaged neutral hydrogen fraction of the IGM.
    z_source : float
        Source redshift (upper bound of the IGM integration).
    z_min : float
        Lower bound of the IGM integration; default 5.3 (Bosman+22).
    cosmo : astropy.cosmology, optional
        Cosmology used to compute the line-centre Gunn-Peterson optical
        depth.  Defaults to Planck18.

    Returns
    -------
    np.ndarray
        Optical depth τ_IGM(λ).  Zero for pixels redward of the
        source's Lyα (the formula returns zero by construction
        outside the integration range), and a finite damping-wing
        contribution between λ_α(1+z_source) and longer wavelengths.

    Notes
    -----
    For pixels blueward of λ_α(1+z_source) the photon would have been
    absorbed inside the GP forest already; the IGM damping-wing
    formula is not the right description there.  Callers should
    enforce a separate model = 0 cut blueward of source Lyα.
    """
    if cosmo is None:
        from astropy.cosmology import Planck18 as cosmo

    wave_A = np.asarray(wave_A, dtype=float)
    out = np.zeros_like(wave_A)
    if x_HI <= 0.0:
        return out

    # Redshift at which a photon of observed λ would have hit Lyα.
    z_obs = wave_A / _LAMBDA_LYA_A - 1.0

    # The damping wing applies redward of source Lyα (z_obs > z_source).
    valid = z_obs > z_source
    if not np.any(valid):
        return out

    z_o = z_obs[valid]
    x1 = (1.0 + z_min) / (1.0 + z_o)
    x2 = (1.0 + z_source) / (1.0 + z_o)

    eps = 1e-9
    x1 = np.clip(x1, eps, 1.0 - eps)
    x2 = np.clip(x2, eps, 1.0 - eps)

    # Line-centre Gunn-Peterson optical depth at z_obs (Madau 1995 form).
    h = float(cosmo.h)
    Ob_h2 = float(cosmo.Ob0) * h * h
    Om_h2 = float(cosmo.Om0) * h * h
    tau_GP_z = (
        6.45e5
        * (Ob_h2 / 0.022)
        * (Om_h2 / 0.13) ** -0.5
        * ((1.0 + z_o) / 10.0) ** 1.5
    )

    out[valid] = (
        tau_GP_z
        * x_HI
        * (_R_ALPHA / np.pi)
        * (_IGM_I(x2) - _IGM_I(x1))
    )
    return np.maximum(out, 0.0)


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

    # The Voigt absorption profile is a property of the absorber's rest
    # frame.  An observed-frame photon at wave_A interacts with the gas
    # at rest-frame wavelength wave_A / (1+z), so we evaluate everything
    # in the rest frame of the gas.  Using observed-frame frequencies
    # would inflate the wing optical depth by (1+z)^2 because both
    # sigma_0 ~ 1/Delta_nu_D and a (= gamma/(4 pi Delta_nu_D)) scale
    # with (1+z) when Delta_nu_D is built from the observed-frame
    # Lyalpha frequency.
    b_cms = b_kms * 1e5  # km/s -> cm/s
    lambda_0_cm = _LAMBDA_LYA_A * 1e-8                    # rest-frame Lya in cm
    nu_0 = _C_CGS / lambda_0_cm                            # rest-frame Lya freq
    delta_nu_D = (b_cms / _C_CGS) * nu_0                   # rest-frame Doppler width

    # Voigt damping parameter (rest-frame).
    a = _GAMMA_ALPHA / (4.0 * np.pi * delta_nu_D)

    # Rest-frame frequency offset of each pixel.
    wave_rest_cm = np.asarray(wave_A) / (1.0 + z) * 1e-8
    nu = _C_CGS / wave_rest_cm
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
    R,
) -> np.ndarray:
    """Convolve a spectrum with a Gaussian LSF at resolving power R.

    Parameters
    ----------
    wave_A : np.ndarray
        Wavelength array (must be uniformly or near-uniformly spaced).
    flux : np.ndarray
        Flux array to convolve.
    R : float, callable, or array_like
        Spectral resolving power (λ / FWHM).  If a callable, it is
        invoked as ``R(lam_um)`` (wavelength in microns) and must
        return an array the same shape as the input.  If an array,
        it must already be evaluated on ``wave_A``.  If a scalar,
        a single Gaussian kernel at the median wavelength is used.

    Returns
    -------
    np.ndarray
        Convolved flux array (same length as input).
    """
    wave_A = np.asarray(wave_A, dtype=float)
    flux = np.asarray(flux, dtype=float)

    # Resolve R into either a scalar or a per-pixel array.
    if callable(R):
        R_arr = np.asarray(R(wave_A * 1e-4), dtype=float)
    elif np.isscalar(R):
        R_arr = None
    else:
        R_arr = np.asarray(R, dtype=float)
        if R_arr.shape != wave_A.shape:
            raise ValueError(
                "R array must have same shape as wave_A "
                f"({R_arr.shape} != {wave_A.shape})"
            )

    if R_arr is None:
        # Single-kernel fast path (constant R).
        dlam = np.median(np.diff(wave_A))
        lam_mid = np.median(wave_A)
        sigma_A = (lam_mid / float(R)) / 2.3548
        sigma_pix = sigma_A / dlam
        hw = int(4.0 * sigma_pix) + 1
        x = np.arange(-hw, hw + 1)
        kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(flux, hw, mode="edge")
        convolved = np.convolve(padded, kernel, mode="same")
        return convolved[hw:-hw]

    # Wavelength-dependent kernel: build a per-pixel Gaussian and apply.
    sigma_A = (wave_A / R_arr) / 2.3548
    dlam = np.median(np.diff(wave_A))
    sigma_max = float(np.max(sigma_A))
    hw = int(4.0 * sigma_max / dlam) + 1
    out = np.empty_like(flux)
    pad = np.pad(flux, hw, mode="edge")
    # Offsets in wavelength relative to each pixel (uniform grid assumed).
    offsets = np.arange(-hw, hw + 1) * dlam
    for i in range(len(wave_A)):
        kernel = np.exp(-0.5 * (offsets / sigma_A[i]) ** 2)
        kernel /= kernel.sum()
        out[i] = np.dot(pad[i:i + 2 * hw + 1], kernel)
    return out


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
    # --- IGM damping-wing extension (Pollock+26) ---
    x_HI: float = 0.0
    x_HI_err: tuple[float, float] = (0.0, 0.0)
    fit_x_HI: bool = False
    igm_z_min: float = _IGM_Z_MIN_DEFAULT
    # --- Upper-limit reporting for unconstrained sources ---
    log_NHI_upper95: float = 0.0
    is_upper_limit: bool = False

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "DLA Fit Result (Pollock+26-style joint sampling)",
            "=" * 50,
        ]
        if self.is_upper_limit:
            lines.append(
                f"log(N_HI/cm^-2)  < {self.log_NHI_upper95:.2f}   (95% upper limit)"
            )
        else:
            lines.append(
                f"log(N_HI/cm^-2)  = {self.log_NHI:.2f} "
                f"(+{self.log_NHI_err[1]:.2f}, -{self.log_NHI_err[0]:.2f})"
            )
        lines.extend([
            f"Sigma_HI         = {self.Sigma_HI:.1f} Msun/pc^2",
            f"beta_UV          = {self.beta_UV:.2f} "
            f"(+{self.beta_UV_err[1]:.2f}, -{self.beta_UV_err[0]:.2f})",
            f"log(F0)          = {self.log_F0:.2f} "
            f"(+{self.log_F0_err[1]:.2f}, -{self.log_F0_err[0]:.2f})",
        ])
        if self.fit_x_HI:
            lines.append(
                f"x_HI (IGM)       = {self.x_HI:.2f} "
                f"(+{self.x_HI_err[1]:.2f}, -{self.x_HI_err[0]:.2f})"
            )
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
    R=None,
    b_kms: float = _B_DEFAULT_KMS,
    x_HI: float = 0.0,
    igm_z_min: float = _IGM_Z_MIN_DEFAULT,
    lya_ujy_func: object | None = None,
    lya_free_params: tuple[float, float, float] | None = None,
    emission_lines: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """Evaluate the joint DLA + IGM damping-wing model.

    Implements Pollock et al. (2026) Eq. 4 — the intrinsic continuum
    (single power-law) plus any Lyα emission is attenuated by the
    local DLA and the IGM damping wing simultaneously:

        F(λ) = [F0 (λ/λ_pivot)^β + Lyα(λ)] · exp(−τ_DLA) · exp(−τ_IGM)

    Parameters
    ----------
    wave_A : np.ndarray
        Observed-frame wavelengths in Angstrom.
    log_F0 : float
        log10 of the continuum normalisation at the (observer-frame)
        pivot wavelength λ_pivot · (1 + z), in the same flux units as
        the input data (typically µJy).
    beta_UV : float
        UV spectral slope of the intrinsic continuum.
    log_NHI : float
        log10(N_HI / cm^-2) of the local DLA.
    z : float
        Source redshift.
    R : float or None
        Spectral resolving power.  If given, the model is convolved
        with a Gaussian LSF of FWHM = λ / R.
    b_kms : float
        Doppler parameter in km/s used in the Voigt profile.  The
        damping wing is insensitive to ``b`` for log N_HI > 20.3.
    x_HI : float
        Volume-averaged neutral fraction of the IGM (Miralda-Escudé
        damping-wing term).  Default 0 (no IGM contribution).
    igm_z_min : float
        Lower-redshift bound of the IGM integration (default 5.3,
        Bosman+22 reionisation end).
    lya_ujy_func : callable or None
        Function returning a fixed Lyα emission profile in µJy.
        Added to the intrinsic continuum before attenuation.
    lya_free_params : tuple of (log_A, sigma, alpha), optional
        Free Lyα parameters: log10(A_peak) in F_λ (erg/s/cm²/Å),
        sigma in Å, and skewness α.  Centroid fixed at Lyα rest × (1+z).
        The profile is added to the continuum and attenuated by both
        DLA and IGM (Pollock+26 convention).

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

    # Build the intrinsic spectrum: continuum + (optional) Lyα emission +
    # (optional) fixed Gaussian emission lines (Pollock+26 Eq. 4 convention:
    # the lines are added *before* DLA+IGM attenuation).
    intrinsic = continuum.copy()
    if lya_ujy_func is not None:
        intrinsic = intrinsic + lya_ujy_func(wave_A)
    if lya_free_params is not None:
        log_A, sigma, alpha = lya_free_params
        A_peak = 10.0 ** log_A
        mu = _LAMBDA_LYA_A * (1.0 + z)
        lya_flam = asymmetric_gaussian(wave_A, A_peak, mu, sigma, alpha)
        intrinsic = intrinsic + _flam_to_ujy(lya_flam, wave_A * 1e-4)
    if emission_lines is not None:
        for A_peak, mu_obs, sigma_obs in emission_lines:
            intrinsic = intrinsic + A_peak * np.exp(
                -0.5 * ((wave_A - mu_obs) / sigma_obs) ** 2
            )

    # DLA + IGM attenuation (Pollock+26 Eq. 4).
    tau_total = tau_DLA(wave_A, log_NHI, z=z, b_kms=b_kms)
    if x_HI > 0.0:
        tau_total = tau_total + tau_IGM_DW(
            wave_A, x_HI, z_source=z, z_min=igm_z_min,
        )
    model = intrinsic * np.exp(-tau_total)

    # Hard zero blueward of source Lyα — the GP forest is essentially
    # saturated there.  The Miralda-Escudé damping wing only models the
    # red side of source Lyα; the formula is not applicable to forest
    # photons that would have hit Lyα at z < z_source.
    lya_obs = _LAMBDA_LYA_A * (1.0 + z)
    model = np.where(wave_A < lya_obs, 0.0, model)

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
    fit_x_HI: bool = False,
    igm_z_min: float = _IGM_Z_MIN_DEFAULT,
    b_kms: float = _B_DEFAULT_KMS,
    continuum: np.ndarray | None = None,
    Av: float = 0.0,
    dust_law: str = "cardelli",
    Rv: float = 3.1,
    R=None,
    mask_lines: bool = True,
    mask_width_A: float = 20.0,
    mask_lya_emission_width_A: float = 8.0,
    fit_range_A: tuple[float, float] = (1216.0, 3000.0),
    mask_regions_A: list[tuple[float, float]] | None = None,
    fit_lya: bool = False,
    lya_params: np.ndarray | list | None = None,
    emission_lines: list[tuple[float, float, float]] | None = None,
    prior_log_NHI: tuple[float, float] = (18.0, 24.0),
    prior_x_HI: tuple[float, float] = (0.0, 1.0),
    prior_beta_UV: tuple[float, float] = (-4.0, 0.0),
    prior_log_F0: tuple[float, float] | None = None,
    prior_log_A_lya: tuple[float, float] = (-22.0, -15.0),
    prior_sigma_lya: tuple[float, float] = (1.0, 30.0),
    prior_alpha_lya: tuple[float, float] = (0.0, 5.0),
    n_live: int = 500,
    dlogz: float = 0.5,
    seed: int = 42,
) -> DLAResult:
    """Joint Bayesian fit of the DLA + IGM damping wing (Pollock+26-style).

    Implements the Pollock et al. (2026) joint nested-sampling fit
    over (log_F0, β_UV, log_NHI[, x_HI[, Lyα]]) with an exact
    Voigt-Hjerting profile and the Miralda-Escudé (1998) IGM damping
    wing in the Totani+06 closed form.

    Parameters
    ----------
    wave_A, flux, flux_err : np.ndarray
        Wavelength (observed-frame Å) and flux density arrays.
    z : float
        Source redshift (use 0 for rest-frame stacks).
    fit_x_HI : bool
        If True, sample over IGM neutral fraction x_HI in addition to
        the local DLA.  Recommended at z > 6.
    igm_z_min : float
        Lower-redshift bound of the IGM integration (default 5.3,
        Bosman+22 reionisation end).
    b_kms : float
        Doppler parameter (km/s) for the Voigt profile.  The damping
        wing is insensitive to b for log N_HI > 20.3.
    continuum : np.ndarray, optional
        Pre-smoothed continuum estimate.  When provided the model is
        fit against this smoothed array instead of the raw flux —
        emission lines are then implicitly removed.
    Av, dust_law, Rv : float, str, float
        Optional foreground dust correction applied before fitting.
    R : float or None
        Spectral resolving power for LSF convolution.
    mask_lines : bool
        If True (default), mask known UV emission lines.  Lyα, the
        NV λλ1238.8, 1242.8 doublet, and any `abs_*` low-ionisation
        absorption features are excluded from this mask: Lyα and NV
        sit inside the damping wing (1216–1263 Å rest) where the N_HI
        constraint lives, and any DLA strong enough to fit absorbs NV
        at τ ≫ 1 (no emission left to contaminate); `abs_*` features
        are foreground-gas signatures aligned with the DLA.
    mask_width_A : float
        Half-width of the general emission-line mask (rest-frame Å).
        Default 20 Å is appropriate for NIRSpec PRISM resolution.
    mask_lya_emission_width_A : float
        Half-width of the *narrow* Lyα-emission core mask in
        rest-frame Å (default 8 Å).  Set this small enough not to eat
        the damping wing.  Ignored when ``fit_lya=True``.
    fit_range_A : tuple
        Rest-frame fit window (default 1216–3000 Å, matching
        Pollock+26 — Lyα to just below the Balmer break).
    mask_regions_A : list of (float, float), optional
        Extra rest-frame wavelength ranges to ignore.
    fit_lya : bool
        Jointly fit a skewed Lyα emission profile.
    lya_params : array of length 4, optional
        Pre-determined Lyα profile (A_peak, μ, σ, α) added to the
        intrinsic spectrum before attenuation.
    prior_log_NHI, prior_x_HI, prior_beta_UV : tuple
        Uniform-prior bounds for the named parameter (Pollock+26 Sect. 3.2
        defaults: log_NHI ∈ [18,24], x_HI ∈ [0,1], β_UV ∈ [-4,0]).
    prior_log_F0 : tuple, optional
        Uniform prior bounds on log10(F0) in the same flux units as the
        input data.  Defaults to ±3 dex around the median continuum
        flux derived from the input spectrum.
    prior_log_A_lya, prior_sigma_lya, prior_alpha_lya : tuple
        Priors used when ``fit_lya=True``.
    n_live : int
        Number of dynesty live points (default 500).
    dlogz : float
        dynesty convergence threshold on log Z (default 0.5).
    seed : int
        RNG seed.

    Returns
    -------
    DLAResult
        Posterior samples for every free parameter, median ± 16/84
        percentiles, log-evidence, and a 95th-percentile upper limit
        on log N_HI when the posterior is unconstrained.

    Notes
    -----
    - Pollock+26 Eq. 4: F = (F0·(λ/λ_pivot)^β + Lyα) · exp(−τ_DLA) · exp(−τ_IGM).
    - Lyα emission is added *before* attenuation (Pollock+26 convention).
    - Reports an upper limit when the posterior touches the prior
      lower bound (Pollock+26 do this for 21/48 sources).
    """
    import dynesty
    from dynesty.utils import resample_equal

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
        flux_corr, err_corr = _apply_dust_correction(
            wave_A, continuum, flux_err, Av, dust_law, Rv,
        )
    else:
        flux_corr, err_corr = _apply_dust_correction(
            wave_A, flux, flux_err, Av, dust_law, Rv,
        )

    # --- Build fixed Lya model (only when fit_lya=False) ---
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

    lya_obs = _LAMBDA_LYA_A * (1.0 + z)

    # --- Emission line masking ---
    # Lyα, NV, and any low-ionisation absorption (`abs_*`) lines are
    # excluded from the general mask because:
    #   (a) the broad Lyα damping wing is the actual DLA signal —
    #       masking ±20 Å around 1216 Å hides exactly the pixels that
    #       constrain N_HI;
    #   (b) NV λλ1238.8, 1242.8 sits at Δλ_rest ≈ 23–27 Å from Lyα,
    #       inside the wing.  Any DLA strong enough to leave a
    #       measurable absorption signature (log N_HI ≳ 20) absorbs the
    #       NV doublet at τ ≫ 1 — there is no NV emission left to
    #       contaminate, so masking it would only destroy the
    #       wing-discriminating pixels (cf. fitter table: log N_HI = 21
    #       gives a 40% wing depth at +30 Å rest, so masking ±20 Å
    #       around NV erases the 1219–1263 Å window that distinguishes
    #       log N_HI = 20 from 22);
    #   (c) `abs_*` features are foreground neutral-gas signatures
    #       aligned with the DLA, not contamination.
    # Lyα emission, when present, gets a separate narrow mask controlled
    # by ``mask_lya_emission_width_A``.
    if mask_lines and not _use_continuum:
        _exclude = {"Lya", "NV_1", "NV_2", "NV_doublet"}
        for _name in REST_LINES_A:
            if _name.startswith("abs_"):
                _exclude.add(_name)
        line_mask = _mask_emission_lines(
            wave_A, z=z, width_A=mask_width_A, exclude=_exclude,
        )
        if not fit_lya:
            lya_em_width = mask_lya_emission_width_A * (1.0 + z)
            line_mask &= (
                (wave_A < lya_obs - lya_em_width)
                | (wave_A > lya_obs + lya_em_width)
            )
    else:
        line_mask = np.ones(len(wave_A), dtype=bool)

    if mask_regions_A:
        for lo, hi in mask_regions_A:
            line_mask &= (wave_A < lo * (1.0 + z)) | (wave_A > hi * (1.0 + z))

    good = (err_corr > 0) & np.isfinite(flux_corr)
    use = in_range & line_mask & good
    if use.sum() < 10:
        raise ValueError(
            f"Only {use.sum()} pixels remain after masking. "
            "Check fit_range_A, mask_width_A, and data quality."
        )

    w = wave_A[use]
    f = flux_corr[use]
    e = err_corr[use]
    inv_var = 1.0 / e ** 2

    _mode_bits = ["joint"]
    if fit_x_HI:
        _mode_bits.append("DLA+IGM")
    else:
        _mode_bits.append("DLA-only")
    if fit_lya:
        _mode_bits.append("+Lyα")
    if _use_continuum:
        _mode_bits.append("continuum")
    logger.info(
        "DLA fit (%s): %d pixels in [%.0f, %.0f] Å rest-frame "
        "(z=%.3f, Av=%.2f).",
        " ".join(_mode_bits), len(w), fit_range_A[0], fit_range_A[1], z, Av,
    )

    # --- Construct the prior ---
    # Default log_F0 prior centred on data median if user did not pass one.
    if prior_log_F0 is None:
        f_med = float(np.nanmedian(np.maximum(f, 1e-30)))
        log_F0_centre = float(np.log10(max(f_med, 1e-30)))
        prior_log_F0 = (log_F0_centre - 3.0, log_F0_centre + 3.0)

    pri_lo = [prior_log_F0[0], prior_beta_UV[0], prior_log_NHI[0]]
    pri_hi = [prior_log_F0[1], prior_beta_UV[1], prior_log_NHI[1]]
    param_names = ["log_F0", "beta_UV", "log_NHI"]

    if fit_x_HI:
        pri_lo.append(prior_x_HI[0])
        pri_hi.append(prior_x_HI[1])
        param_names.append("x_HI")

    if fit_lya:
        pri_lo.extend([prior_log_A_lya[0], prior_sigma_lya[0], prior_alpha_lya[0]])
        pri_hi.extend([prior_log_A_lya[1], prior_sigma_lya[1], prior_alpha_lya[1]])
        param_names.extend(["log_A_lya", "sigma_lya", "alpha_lya"])

    pri_lo = np.asarray(pri_lo)
    pri_hi = np.asarray(pri_hi)
    pri_range = pri_hi - pri_lo
    ndim = len(pri_lo)

    def prior_transform(u):
        return pri_lo + u * pri_range

    def log_likelihood(theta):
        log_F0 = theta[0]
        beta_UV = theta[1]
        log_NHI = theta[2]
        idx = 3
        x_HI = float(theta[idx]) if fit_x_HI else 0.0
        if fit_x_HI:
            idx += 1
        lya_free = tuple(theta[idx:idx + 3]) if fit_lya else None

        m = _evaluate_model(
            w, log_F0, beta_UV, log_NHI, z,
            R=R, b_kms=b_kms,
            x_HI=x_HI, igm_z_min=igm_z_min,
            lya_ujy_func=_lya_ujy_func if not fit_lya else None,
            lya_free_params=lya_free,
            emission_lines=emission_lines,
        )
        resid = f - m
        return -0.5 * float(np.sum(resid * resid * inv_var))

    # --- Run dynesty ---
    sampler = dynesty.NestedSampler(
        log_likelihood, prior_transform, ndim=ndim,
        nlive=n_live, rstate=np.random.default_rng(seed),
    )
    sampler.run_nested(print_progress=True, dlogz=dlogz)
    results = sampler.results

    weights = np.exp(results.logwt - results.logz[-1])
    samples = resample_equal(results.samples, weights)

    samples_dict: dict[str, np.ndarray] = {
        name: samples[:, i] for i, name in enumerate(param_names)
    }

    def _median_ci(arr):
        m = float(np.median(arr))
        lo = m - float(np.percentile(arr, 16))
        hi = float(np.percentile(arr, 84)) - m
        return m, (lo, hi)

    log_F0_med, log_F0_err = _median_ci(samples_dict["log_F0"])
    beta_UV_med, beta_UV_err = _median_ci(samples_dict["beta_UV"])
    log_NHI_med, log_NHI_err = _median_ci(samples_dict["log_NHI"])

    if fit_x_HI:
        x_HI_med, x_HI_err = _median_ci(samples_dict["x_HI"])
    else:
        x_HI_med, x_HI_err = 0.0, (0.0, 0.0)
        # Keep an empty x_HI sample for downstream compatibility.
        samples_dict["x_HI"] = np.zeros_like(samples_dict["log_NHI"])

    # --- Upper-limit detection: posterior touches the prior lower bound ---
    nhi_p5 = float(np.percentile(samples_dict["log_NHI"], 5))
    is_upper_limit = nhi_p5 <= prior_log_NHI[0] + 0.1
    log_NHI_upper95 = float(np.percentile(samples_dict["log_NHI"], 95))
    if is_upper_limit:
        logger.info(
            "log_NHI posterior is unconstrained — reporting 95%% upper "
            "limit log_NHI < %.2f.", log_NHI_upper95,
        )

    # --- Surface density (Pollock+26 Eq. 7) ---
    Sigma_HI = 8e-21 * 10.0 ** log_NHI_med

    # --- Best-fit model on the full fit window (for plotting) ---
    lya_free_best = None
    if fit_lya:
        lya_free_best = (
            float(np.median(samples_dict["log_A_lya"])),
            float(np.median(samples_dict["sigma_lya"])),
            float(np.median(samples_dict["alpha_lya"])),
        )
    model_best = _evaluate_model(
        w, log_F0_med, beta_UV_med, log_NHI_med, z,
        R=R, b_kms=b_kms,
        x_HI=x_HI_med, igm_z_min=igm_z_min,
        lya_ujy_func=_lya_ujy_func if not fit_lya else None,
        lya_free_params=lya_free_best,
        emission_lines=emission_lines,
    )

    log_evidence = float(results.logz[-1])

    final_lya_params = np.asarray(lya_params) if lya_params is not None else None

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
        x_HI=x_HI_med,
        x_HI_err=x_HI_err,
        fit_x_HI=fit_x_HI,
        igm_z_min=igm_z_min,
        log_NHI_upper95=log_NHI_upper95,
        is_upper_limit=is_upper_limit,
    )
    result._R = R
    return result


# --------------------------------------------------------------------------
# D_Lyα equivalent-width statistic (Heintz et al. 2025, Eq. 1)
# --------------------------------------------------------------------------

def compute_D_Lya(
    wave_A: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    z: float,
    *,
    cont_range_A: tuple[float, float] = (1250.0, 2600.0),
    int_range_A: tuple[float, float] = (1180.0, 1350.0),
    mask_lines: bool = True,
    mask_width_A: float = 20.0,
    n_mc: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Heintz+25 D_Lyα equivalent-width statistic (Eq. 1).

    Fits a power-law continuum F_cont(λ) = F_1550 (λ/1550)^β over
    *cont_range_A* (lines masked) then integrates

        D_Lyα = ∫ (1 − F_λ / F_cont(λ)) dλ / (1 + z)

    over *int_range_A* (rest-frame).  D_Lyα < 0 indicates a
    Lyα emitter; ≳50 Å indicates a strong damped absorber.

    Parameters
    ----------
    wave_A, flux, flux_err : np.ndarray
        Spectrum in observed-frame Å and arbitrary linear flux units.
    z : float
        Source redshift.
    cont_range_A : tuple
        Rest-frame range used to fit the power-law continuum
        (default 1250–2600 Å, matching Heintz+25).
    int_range_A : tuple
        Rest-frame range over which the EW integral is taken
        (default 1180–1350 Å, matching Heintz+25 Eq. 1).
    mask_lines : bool
        Whether to mask known UV emission lines from the continuum
        fit (default True).  The integral is always taken over the
        full *int_range_A* — emission/absorption inside it is the
        signal.
    mask_width_A : float
        Half-width of line masks for the continuum fit (rest-frame Å).
    n_mc : int
        Monte-Carlo iterations for D_Lyα uncertainty propagation.
    seed : int
        RNG seed.

    Returns
    -------
    dict
        ``{"D_Lya": float, "D_Lya_err": float, "beta_UV": float,
        "beta_UV_err": float, "F_1550": float, "F_1550_err": float}``
    """
    from scipy.optimize import curve_fit

    wave_A = np.asarray(wave_A, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)

    wave_rest = wave_A / (1.0 + z)
    pivot = 1550.0  # rest-frame (Heintz+25)

    in_cont = (wave_rest >= cont_range_A[0]) & (wave_rest <= cont_range_A[1])
    if mask_lines:
        line_mask = _mask_emission_lines(wave_A, z=z, width_A=mask_width_A)
        # Also mask the broad Lyα region from the continuum fit.
        lya_obs = _LAMBDA_LYA_A * (1.0 + z)
        line_mask &= (
            (wave_A < lya_obs - 50.0 * (1.0 + z))
            | (wave_A > lya_obs + 50.0 * (1.0 + z))
        )
    else:
        line_mask = np.ones(len(wave_A), dtype=bool)
    cont_use = in_cont & line_mask & (flux_err > 0) & np.isfinite(flux)
    if cont_use.sum() < 5:
        raise ValueError(
            f"Only {cont_use.sum()} continuum pixels in {cont_range_A} Å; "
            "cannot fit β_UV."
        )

    def power_law(lam_rest, F_1550, beta):
        return F_1550 * (lam_rest / pivot) ** beta

    f_med = float(np.nanmedian(flux[cont_use]))
    popt, pcov = curve_fit(
        power_law,
        wave_rest[cont_use], flux[cont_use],
        p0=[max(f_med, 1e-30), -2.0],
        sigma=flux_err[cont_use], absolute_sigma=True,
        bounds=([1e-30, -5.0], [1e30, 1.0]),
    )
    F_1550_med, beta_med = popt
    perr = np.sqrt(np.diag(pcov))

    # Integral: D_Lyα = ∫ (1 - F/F_cont) dλ / (1+z), in rest-frame Å.
    in_int = (wave_rest >= int_range_A[0]) & (wave_rest <= int_range_A[1])
    in_int &= (flux_err > 0) & np.isfinite(flux)
    if in_int.sum() < 3:
        raise ValueError(
            f"Only {in_int.sum()} pixels in EW integral range "
            f"{int_range_A} Å — increase the spectrum coverage."
        )
    w_int = wave_rest[in_int]
    f_int = flux[in_int]
    fe_int = flux_err[in_int]
    order = np.argsort(w_int)
    w_int = w_int[order]
    f_int = f_int[order]
    fe_int = fe_int[order]

    rng = np.random.default_rng(seed)
    D_samples = np.empty(n_mc)
    for k in range(n_mc):
        F_1550_k = F_1550_med + perr[0] * rng.standard_normal()
        beta_k = beta_med + perr[1] * rng.standard_normal()
        f_k = f_int + fe_int * rng.standard_normal(len(f_int))
        cont_k = F_1550_k * (w_int / pivot) ** beta_k
        # Avoid divide-by-zero at the rare positive-continuum dips.
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = np.where(cont_k > 0, 1.0 - f_k / cont_k, 0.0)
        D_samples[k] = np.trapz(integrand, w_int)
    D_Lya_med = float(np.median(D_samples))
    D_Lya_err = float(np.std(D_samples))

    return {
        "D_Lya": D_Lya_med,
        "D_Lya_err": D_Lya_err,
        "beta_UV": float(beta_med),
        "beta_UV_err": float(perr[1]),
        "F_1550": float(F_1550_med),
        "F_1550_err": float(perr[0]),
        "n_pixels": int(in_int.sum()),
    }
