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
sys.path.insert(0, os.path.abspath('../../src/nbrefactor'))


# -- Project information -----------------------------------------------------

project = 'nbrefactor'
copyright = '2024, Mohamed Shahawy'
author = 'Mohamed Shahawy'

# The full version, including alpha/beta/rc tags
version = '0.1.0'
release = '0.1.0'


# -- General configuration ---------------------------------------------------

add_module_names = False

html_title = "nbrefactor - a Jupyter notebook refactoring tool"
html_short_title = "nbrefactor"
html_favicon = '_static/favicon.png'

# autoclass_content = 'both'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'nbsphinx',
	'sphinx.ext.autodoc',
	#'sphinx.ext.autosummary',
	'sphinx.ext.coverage',
	'sphinx.ext.napoleon',
	#'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'myst_parser',
	# 'sphinx.ext.graphviz',
	# 'sphinx.ext.inheritance_diagram'
]

myst_enable_extensions = [
	'html_admonition'
]
myst_heading_anchors = 3

source_suffix = ['.rst', '.md']

pygments_style = 'sphinx'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'api/src.rst', 'api/modules.rst']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo' # 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'sidebar_hide_name': True,
    'light_logo': 'nbrefactor_logo.svg',
    'dark_logo': 'nbrefactor_logo.svg'
}

html_show_sphinx = False

# -- Options for AutoAPI -----------------------------------------------------

autoapi_options = [
	'members',
	'undoc-members',
	'private-members',
	'show-inheritance',
	'show-module-summary',
	'special-members',
	'imported-members',
	'titlesonly'
]

toc_object_entries_show_parents = 'hide'
add_function_parentheses = False

# ----------------------------------------------------------------------------

def run_apidoc(_):
    # Run sphinx-apidoc with options
    from sphinx.ext.apidoc import main

    cur_dir = os.path.abspath(os.path.dirname(__file__))
    out_dir = os.path.join(cur_dir, 'api')
    module = os.path.join(cur_dir, '../../src/nbrefactor')
    template_dir = os.path.join(cur_dir, '_templates', 'apidoc')

    main(['-o', out_dir, module, '-fMe', '-t', template_dir])


def setup(app):
	app.connect('builder-inited', run_apidoc)
	app.add_css_file('css/custom.css')


