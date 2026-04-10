"""jwspecabund — Chemical abundance calculations for JWST spectra.

Computes oxygen abundances (O/H), nitrogen-to-oxygen ratios (N/O),
and other element ratios from emission-line fluxes produced by
``jwspecfit`` or ``jwspecmcmc``.

Three pathways are supported:

1. **Direct T_e method** — when auroral lines (e.g. [OIII] 4363) are
   detected, uses PyNEB for electron temperature / density diagnostics
   and ionic abundance calculations.

2. **Bayesian forward model** — samples ionic abundances, T_e, and n_e
   via MCMC/nested sampling, predicting line ratios with PyNEB CEL
   emissivities and the Aller (1984) Hbeta formula (Cullen+25 approach).

3. **Strong-line calibrations** — Sanders et al. (2025) simultaneous
   polynomial fit across O3, O2, R23, O32 diagnostics.

Example
-------
>>> import jwspecfit, jwspecabund
>>> spec = jwspecfit.read_fits("spectrum.fits")
>>> result = jwspecfit.fit_lines(spec, z=6.0)
>>> abund = jwspecabund.compute_abundances(result, z=6.0)
>>> print(abund.OH)          # 12 + log(O/H)
>>> print(abund.method)      # "direct" or "strong_line"
"""

from __future__ import annotations

__version__ = "1.0.0"

from ._core import compute_abundances
from .forward import forward_model, hbeta_emissivity_aller84
from .direct import (
    Te_low_from_high,
    compute_ionic_abundances,
    compute_ne,
    compute_ne_CIII,
    compute_ne_NIV,
    compute_Te_NII,
    compute_Te_OIII,
    compute_Te_OIII_1666,
    compute_total_abundances,
    ciii_ratio_at_density,
    niv_ratio_at_density,
)
from .martinez25_icf import (
    compute_NO_martinez25,
    log_U_from_N43,
    log_U_from_O32,
)
from .dust import (
    cardelli_extinction,
    compute_Av_from_balmer,
    compute_Av_multi_balmer,
    compute_lya_escape_fraction,
    compute_lya_escape_fraction_mc,
    dust_correct_fluxes,
    salim_attenuation,
)
from .icf import icf_argon, icf_neon, icf_nitrogen, icf_sulfur
from .result import AbundanceResult
from .strong_line import compute_line_ratios, sanders25_metallicity

__all__ = [
    "AbundanceResult",
    "Te_low_from_high",
    "cardelli_extinction",
    "compute_abundances",
    "compute_Av_from_balmer",
    "compute_Av_multi_balmer",
    "compute_lya_escape_fraction",
    "compute_lya_escape_fraction_mc",
    "compute_ionic_abundances",
    "compute_line_ratios",
    "compute_ne",
    "compute_ne_CIII",
    "compute_ne_NIV",
    "compute_NO_martinez25",
    "compute_Te_NII",
    "compute_Te_OIII",
    "compute_Te_OIII_1666",
    "compute_total_abundances",
    "dust_correct_fluxes",
    "forward_model",
    "hbeta_emissivity_aller84",
    "icf_argon",
    "icf_neon",
    "icf_nitrogen",
    "icf_sulfur",
    "log_U_from_N43",
    "log_U_from_O32",
    "salim_attenuation",
    "sanders25_metallicity",
]
