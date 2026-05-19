"""Sphinx configuration for jwspecfit documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the package importable so autodoc works.
sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------

project = "jwspecfit"
author = "Raunaq Rai"
copyright = f"{datetime.now().year}, {author}"

try:
    from jwspecfit import __version__ as _version
except Exception:  # noqa: BLE001 - docs build may happen without install
    _version = "1.0.0"

version = _version
release = _version


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

# MyST — allow .md source files with extended syntax.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "tasklist",
    "fieldlist",
]
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "notebooks/**",
    "superpowers/**",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}


# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
    "logo_only": False,
    "style_nav_header_background": "#2c3e50",
}
html_static_path = ["_static"]
html_show_sourcelink = True
html_title = f"{project} {version}"
html_logo = "_static/logos/logo.svg"
html_favicon = "_static/logos/logo-32.png"


# Suppress warnings for references that cannot be resolved (e.g. optional
# dependencies that aren't installed in the RTD build environment) and for
# the benign duplicates that autosummary produces for package-level
# re-exports in ``__init__`` files.
nitpicky = False
suppress_warnings = [
    "autodoc.import_object",
    "autosectionlabel.*",
    "ref.python",
]
