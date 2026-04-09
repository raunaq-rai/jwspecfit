# DLA Column Density Fitter — Design Spec

**Date:** 2026-04-09
**Module:** `src/jwspecfit/dla.py`
**Purpose:** Measure neutral hydrogen column density ($N_{\rm HI}$) from Ly$\alpha$ damping wing absorption in galaxy spectra.

---

## 1. Motivation

High-redshift galaxies observed with JWST show damped Ly$\alpha$ absorption (DLA) features indicating dense neutral gas reservoirs in the ISM. Following the method of Pollock et al. (2026, A&A, arXiv:2602.11783), we fit the red damping wing profile to derive $N_{\rm HI}$, which traces the galaxy's own neutral gas content.

This is a standalone fitting function in `jwspecfit` — not wired into `jwspecabund.compute_abundances()`. It operates on a spectrum (rest-frame or observed-frame) and returns the DLA column density with full posteriors.

---

## 2. Physical Model

The observed spectrum is modelled as:

$$F_\lambda = F_0 \, \lambda^{\beta_{UV}} \times \exp\left(-\tau_{\rm DLA}(\lambda, N_{\rm HI})\right)$$

where:

- $F_0 \lambda^{\beta_{UV}}$ is the intrinsic UV continuum power law.
- $\tau_{\rm DLA}$ is the optical depth from a damped Ly$\alpha$ absorber at the galaxy redshift.

### DLA optical depth

Following Tepper-Garcia (2006), the optical depth at wavelength $\lambda$ is:

$$\tau_{\rm DLA}(\lambda) = N_{\rm HI} \times \sigma_0 \times H(a, u)$$

where:

- $\sigma_0 = \frac{\sqrt{\pi} e^2 f_\alpha}{m_e c \Delta\nu_D}$ is the line-centre cross-section.
- $f_\alpha = 0.4162$ is the Ly$\alpha$ oscillator strength.
- $\lambda_\alpha = 1215.67$ A (rest-frame Ly$\alpha$).
- $a = \frac{\Gamma_\alpha}{4\pi \Delta\nu_D}$ is the Voigt damping parameter, with $\Gamma_\alpha = 6.265 \times 10^8$ s$^{-1}$.
- $u = \frac{c}{\Delta\nu_D} \left(\frac{1}{\lambda_\alpha(1+z)} - \frac{1}{\lambda}\right)$ is the dimensionless frequency offset.
- $\Delta\nu_D$ is the Doppler width. For DLA fitting the damping wings dominate and the result is insensitive to the assumed Doppler parameter; we fix $b = 30$ km/s (standard for DLA work).
- $H(a, u)$ is the Voigt-Hjerting function, evaluated using the Tepper-Garcia (2006) analytic approximation (see Section 3).

### Redshift handling

Ly$\alpha$ is placed at $\lambda_\alpha (1 + z)$. For rest-frame stacks, $z = 0$. For individual observed-frame spectra, pass the spectroscopic redshift.

### Dust correction

Before fitting, the input spectrum is corrected for dust attenuation:

$$F_{\rm corr}(\lambda) = F_{\rm obs}(\lambda) \times 10^{0.4 \, A_\lambda}$$

where $A_\lambda$ comes from either the Cardelli et al. (1989) or Salim et al. (2018) attenuation curve, evaluated at the user-supplied $A_V$. Errors are scaled by the same factor. If $A_V = 0$ (default), no correction is applied.

---

## 3. Voigt-Hjerting Function (Tepper-Garcia 2006)

The analytic approximation to $H(a, u)$:

$$H(a, u) \approx e^{-u^2} - \frac{a}{\sqrt{\pi} \, u^2} \left[ e^{-u^2} \left(4u^4 + 7u^2 + 4 + \frac{3}{2u^2}\right) - \frac{1}{u^2}\left(2u^2 + 1\right) - 1 \right]$$

valid for $|u| > 0$. For $|u| < 10^{-4}$, use the series expansion $H(a, 0) \approx 1 - 2a/\sqrt{\pi}$ (or evaluate at $|u| = 10^{-4}$).

This is accurate to <0.1% relative to the exact Voigt profile for the damping wing regime ($|u| \gg 1$) which is what matters for DLA fitting.

Implementation must be **pure JAX** (using `jax.numpy`) so that it is differentiable for NUTS sampling.

---

## 4. Function Signature

```python
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
    n_warmup: int = 500,
    n_samples: int = 2000,
    seed: int = 42,
) -> DLAResult
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `wave_A` | Wavelength array in Angstrom (observed frame). |
| `flux` | Flux density array (any units — $F_0$ absorbs the normalisation). |
| `flux_err` | 1$\sigma$ flux errors, same units as `flux`. |
| `z` | Source redshift. 0 for rest-frame stacks. |
| `Av` | Dust attenuation $A_V$ (mag). Applied before fitting. Default 0. |
| `dust_law` | `"cardelli"` (Cardelli+89) or `"salim"` (Salim+18). |
| `Rv` | Total-to-selective extinction ratio. Default 3.1. |
| `mask_lines` | If True, mask known emission lines from `jwspecfit.lines`. |
| `mask_width_A` | Half-width of emission line masks in rest-frame Angstrom. |
| `fit_range_A` | Rest-frame wavelength range for the fit (Angstrom). |
| `n_warmup` | NUTS warmup iterations. |
| `n_samples` | NUTS posterior samples. |
| `seed` | RNG seed for reproducibility. |

### Fit range

Default `(1050, 2000)` A rest-frame. The blue limit captures the Ly$\alpha$ red wing onset; the red limit provides continuum leverage. Everything below Ly$\alpha$ centre ($1215.67 \times (1+z)$ A observed) where the flux is zero from the Ly$\alpha$ forest/IGM is automatically excluded by the model (the DLA $\tau \to \infty$ there naturally suppresses model flux, but we also mask pixels with flux consistent with zero to avoid biasing the fit).

---

## 5. Return Object

```python
@dataclass
class DLAResult:
    log_NHI: float                     # Median posterior log10(N_HI / cm^-2)
    log_NHI_err: tuple[float, float]   # (lo, hi) 68% CI half-widths
    beta_UV: float                     # Median posterior UV slope
    beta_UV_err: tuple[float, float]   # (lo, hi) 68% CI half-widths
    log_F0: float                      # Median posterior log10(F0)
    log_F0_err: tuple[float, float]    # (lo, hi) 68% CI half-widths
    Sigma_HI: float                    # Gas surface density (M_sun pc^-2)
    samples: dict[str, np.ndarray]     # {"log_NHI": arr, "beta_UV": arr, "log_F0": arr}
    wave_fit: np.ndarray               # Wavelengths used in fit (after masking)
    flux_fit: np.ndarray               # Dust-corrected fluxes used
    model_best: np.ndarray             # Best-fit model on wave_fit
```

The surface density conversion (Pollock+26, Eq. 7):

$$\Sigma_{\rm HI} \,(M_\odot \,\text{pc}^{-2}) = 8 \times 10^{-21} \times N_{\rm HI} \,(\text{cm}^{-2})$$

---

## 6. Emission Line Masking

When `mask_lines=True`, the function imports the line list from `jwspecfit.lines.LINES` and masks all lines within $\pm$`mask_width_A` of each line's rest wavelength $\times (1+z)$. This removes NIV], CIV, HeII, OIII], NIII], CIII], etc. from the fit, leaving only continuum pixels + the DLA wing.

---

## 7. Sampler

**NumPyro NUTS** (No-U-Turn Sampler) via JAX. Three free parameters:

| Parameter | Prior |
|-----------|-------|
| $\log_{10}(N_{\rm HI})$ | Uniform(18, 24) |
| $\beta_{UV}$ | Uniform(-4, 0) |
| $\log_{10}(F_0)$ | Uniform($\mu - 5$, $\mu + 5$) where $\mu$ = log10(median flux in fit range) |

Likelihood: Gaussian, pixel-independent:

$$\ln \mathcal{L} = -\frac{1}{2} \sum_i \left(\frac{F_i^{\rm obs} - F_i^{\rm model}}{\sigma_i}\right)^2$$

The model is evaluated on the masked wavelength grid at each NUTS step.

---

## 8. Internal Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `tepper_garcia_H(a, u)` | `dla.py` | JAX Voigt-Hjerting function |
| `tau_DLA(wave_A, log_NHI, z, b_kms)` | `dla.py` | DLA optical depth array |
| `_dla_continuum_model(wave_A, log_F0, beta_UV)` | `dla.py` | Power-law continuum |
| `_mask_emission_lines(wave_A, z, width_A)` | `dla.py` | Boolean mask from line list |
| `fit_NHI()` | `dla.py` | Orchestrator |
| `DLAResult` | `dla.py` | Result dataclass |

---

## 9. Integration

- **Module:** `src/jwspecfit/dla.py` (new file)
- **Public API:** Add `fit_NHI` and `DLAResult` to `src/jwspecfit/__init__.py`
- **Dependencies:** `jax`, `numpyro` (already in environment from `jwspecmcmc`)
- **No coupling** to `jwspecabund` or `jwspecmcmc`

---

## 10. Testing

All tests in `tests/test_dla.py`.

| Test | Description |
|------|-------------|
| `test_voigt_hjerting_vs_scipy` | Compare `tepper_garcia_H(a, u)` against `scipy.special.voigt_profile` for a grid of $(a, u)$ values. Agreement <0.1% for $|u| > 1$. |
| `test_tau_DLA_known_values` | Verify $\tau_{\rm DLA}$ at line centre for $\log N_{\rm HI} = 20, 21, 22$. Cross-check: $\tau_0 \propto N_{\rm HI}$. |
| `test_fit_synthetic_DLA` | Generate synthetic spectrum: $\log N_{\rm HI} = 22$, $\beta_{UV} = -2.5$, $F_0 = 0.1$, add Gaussian noise (S/N ~ 10). Fit and verify all three parameters recovered within 2$\sigma$. |
| `test_fit_no_DLA` | Pure power law (no absorption). Should recover $\log N_{\rm HI} \lesssim 19$. |
| `test_mask_emission_lines` | Check that CIV 1549, NIII] 1750, etc. are masked at z=0 and z=2. |
| `test_z_scaling` | Same intrinsic spectrum at z=0 and z=2 should give same $N_{\rm HI}$. |
| `test_dust_correction_effect` | Synthetic spectrum with $A_V = 0.5$ reddening. Fit with correct $A_V$ should recover true $N_{\rm HI}$; fit with $A_V = 0$ should give biased $\beta_{UV}$ but $N_{\rm HI}$ within ~0.3 dex (since DLA wing shape is less sensitive to slope). |

---

## 11. Usage Example

```python
import jwspecfit

# Rest-frame stack
result = jwspecfit.fit_NHI(
    wave_rest, flux_stack, flux_err_stack,
    z=0.0,
    Av=0.25, dust_law="cardelli",
)

print(f"log(N_HI) = {result.log_NHI:.2f} (+{result.log_NHI_err[1]:.2f}, -{result.log_NHI_err[0]:.2f})")
print(f"Sigma_HI = {result.Sigma_HI:.1f} Msun/pc^2")
print(f"beta_UV = {result.beta_UV:.2f}")

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(result.wave_fit, result.flux_fit, 'k', lw=0.5)
ax.plot(result.wave_fit, result.model_best, 'r', lw=1.5)
ax.set_xlabel("Wavelength (A)")
ax.set_ylabel("Flux")
```
