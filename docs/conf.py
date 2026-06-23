import os
import sys
from importlib.metadata import PackageNotFoundError, version

sys.path.insert(0, os.path.abspath("../src"))

project = "qsospec"
copyright = "2026, Yuming Fu"
author = "Yuming Fu"
try:
    release = version("qsospec")
except PackageNotFoundError:
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx_reredirects",
    "myst_parser",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_context = {
    "display_github": True,
    "github_user": "rudolffu",
    "github_repo": "qsospec",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

rst_prolog = """
.. |project_name| replace:: qsospec
.. |python_versions| replace:: 3.9–3.13
.. |repository| replace:: https://github.com/rudolffu/qsospec
.. |issues| replace:: https://github.com/rudolffu/qsospec/issues
.. |pypi| replace:: https://pypi.org/project/qsospec/
"""

redirects = {
    "examples": "how_to/index.html",
    "recipe_reference": "reference/recipes.html",
    "scientific_definitions": "science/measurements.html",
    "run_bundles": "reference/run_bundles.html",
    "api": "reference/api/index.html",
    "references": "science/references.html",
    "development_plan": "contributing/roadmap.html",
    "user_guide/configuration": "../reference/configuration.html",
    "user_guide/workflows": "index.html",
    "user_guide/results_warnings": "results.html",
}
