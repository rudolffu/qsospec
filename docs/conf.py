import os
import re
import sys
from pathlib import Path

# Build the API and version label from this checkout, even if an older
# editable installation has left stale distribution metadata behind.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "qsospec"
copyright = "2026, Yuming Fu"
author = "Yuming Fu"
release = re.search(
    r'^version\s*=\s*"([^"]+)"',
    (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    re.MULTILINE,
).group(1)
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
if os.environ.get("QSOSPEC_DOCS_OFFLINE") == "1":
    intersphinx_mapping = {}

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

html_theme = "furo"
html_title = f"qsospec {release} documentation"
html_theme_options = {
    "source_repository": "https://github.com/rudolffu/qsospec/",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
    "light_css_variables": {
        "color-brand-primary": "#245b83",
        "color-brand-content": "#245b83",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8ec5ed",
        "color-brand-content": "#8ec5ed",
    },
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

rst_prolog = """
.. |project_name| replace:: qsospec
.. |python_versions| replace:: 3.10 or later
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
