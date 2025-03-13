# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../src/imagej/"))


# -- Project information -----------------------------------------------------

project = "PyImageJ"
copyright = "2022 ImageJ2 developers"
author = "ImageJ2 developers"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_search.extension",
    "sphinx_copybutton",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "examples/README.md",
]

# -- MyST-Parser/MyST-NB configuration ---------------------------------------
myst_heading_anchors = 4
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------

# Always show the Edit on GitHub buttons
# Set the correct path for Edit on GitHub
html_context = {
    'display_github': True,
    'github_user': 'imagej',
    'github_repo': 'pyimagej',
    'github_version': 'main/doc/',
}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Add the PyImageJ logo
html_logo = "doc-images/logo.svg"
html_theme_options = {
    "logo_only": True,
}
