import irdl

project = "irdl"
author = irdl.__author__
copyright = f"{irdl.__date__.split(' ')[-1]}, {irdl.__author__}"
version = irdl.__version__

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

html_theme = "pydata_sphinx_theme"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "show-inheritance": True,
}

autosummary_generate = True
