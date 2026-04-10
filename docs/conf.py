import importlib.metadata

metadata = importlib.metadata.metadata("irdl")

project = "irdl"
author = "Art J. R. Pelling"
copyright = f"2025-%Y, {author}"
version = metadata["Version"]

extensions = [
    "numpydoc",
    "sphinx_click",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "alt_text": "irdl",
        "text": "irdl",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/artpelling/irdl",
            "icon": "fa-brands fa-square-github",
        },
    ],
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}
html_sidebars = {
    "installation": [],
    "cli_ref": [],
}

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
    "pooch": ("https://www.fatiando.org/pooch/latest", None),
    "pyfar": ("https://pyfar.readthedocs.io/en/stable", None),
    "python": ("https://docs.python.org/3/", None),
}
