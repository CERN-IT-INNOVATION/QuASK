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
<<<<<<< HEAD
sys.path.insert(0, os.path.abspath('../../src'))
=======
sys.path.insert(0, os.path.abspath('../..'))
>>>>>>> origin


# -- Project information -----------------------------------------------------

project = 'QuASK'
<<<<<<< HEAD
copyright = '2023, Massimiliano Incudini, Michele Grossi'
author = 'Massimiliano Incudini, Michele Grossi'

# The full version, including alpha/beta/rc tags
release = '2.0.0-alpha1'
=======
copyright = '2022, Francesco Di Marcantonio, Massimiliano Incudini, Davide Tezza, Michele Grossi'
author = 'Francesco Di Marcantonio, Massimiliano Incudini, Davide Tezza, Michele Grossi'

# The full version, including alpha/beta/rc tags
release = '1.0.0'
>>>>>>> origin


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.viewcode',
<<<<<<< HEAD
  'sphinx.ext.napoleon'
=======
  'sphinx.ext.napoleon',
>>>>>>> origin
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

<<<<<<< HEAD
autodoc_mock_imports = ["skopt", "skopt.space", "django", "mushroom_rl", "opytimizer", "pennylane", "qiskit", "qiskit_ibm_runtime", "qiskit_aer"]
=======
>>>>>>> origin

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
<<<<<<< HEAD
# html_theme = 'sphinx_rtd_theme'
html_theme = 'furo'

=======
html_theme = 'sphinx_rtd_theme'
>>>>>>> origin
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_show_sourcelink = True
<<<<<<< HEAD

html_favicon = "images/favicon.ico"
html_logo = "images/logo_nobg.png"

html_theme_options = {
    "sidebar_hide_name": True
}
=======
>>>>>>> origin
