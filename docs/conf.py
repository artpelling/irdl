import irdl

project = "irdl"
author = irdl.__author__
copyright = f"{irdl.__date__.split(' ')[-1]}, {irdl.__author__}"
version = irdl.__version__

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

html_theme = "pydata_sphinx_theme"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "show-inheritance": True,
}

autosummary_generate = True
numpydoc_show_class_members = False

intersphinx_mapping = {
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pyfar": ("https://pyfar-gallery.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3/", None),
}
