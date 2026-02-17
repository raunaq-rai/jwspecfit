"""jwspecfit — JWST NIRSpec emission-line fitting."""

__version__ = "0.1.0"

from .broad import BroadFitResult, fit_with_broad
from .fitter import FitResult, LineResult, fit_lines
from .io import (
    Spectrum, export_lines_txt, load_result, read_dict, read_fits, read_npz, save_result,
)
from .lines import REST_LINES_A, get_line_list, observable_lines
from .lyman_alpha import igm_transmission, lya_model
from .plotting import plot_fit, plot_fit_interactive
from .resolution import R_from_pixels, R_prism, resolve_R, sigma_inst_A

__all__ = [
    "BroadFitResult",
    "FitResult",
    "LineResult",
    "REST_LINES_A",
    "R_from_pixels",
    "R_prism",
    "Spectrum",
    "fit_lines",
    "fit_with_broad",
    "get_line_list",
    "igm_transmission",
    "lya_model",
    "observable_lines",
    "plot_fit",
    "plot_fit_interactive",
    "read_dict",
    "read_fits",
    "read_npz",
    "resolve_R",
    "save_result",
    "load_result",
    "export_lines_txt",
    "sigma_inst_A",
]
