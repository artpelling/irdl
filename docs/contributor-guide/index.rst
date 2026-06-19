Contributor Guide
=================

Contributions are welcome. This section collects the development setup and the conceptual
material needed to extend ``irdl``.

Developer setup
---------------

``irdl`` uses `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ for
development. Start by syncing the development dependencies, then run tools through
``uv run``.

.. code-block:: console

   $ uv run python -m pytest
   $ uv run ruff check --fix
   $ uv run ruff format
   $ uv run make -C docs html

The documentation Makefile regenerates the dataset docs and CLI help snippets that are
included in the Sphinx documentation. Treat ``docs/build/`` as generated output; update
source files under ``docs/`` and then rebuild. ``README.md`` is not generated and may
need manual updates when user-facing behavior changes.

Coding style
------------

The source of truth for formatting and linting is ``pyproject.toml`` and is enforced by
``ruff``. Important rules are:

- Line length: 120 characters.
- Double quotes.
- NumPy-style docstrings.
- Prefer explicit ``ValueError`` exceptions for invalid user input rather than ``assert``.
- Add or update tests whenever retrieval semantics, cache behavior, Provider selection, or documentation-visible output changes.
- After parameter validation, access required parameters directly, for example
  ``dataset_kwargs["scenario"]`` rather than ``dataset_kwargs.get("scenario")``.

Architecture and extension guides
---------------------------------

.. toctree::
   :maxdepth: 1

   Processing flow <processing_flow>
   Adding a new Dataset <adding_dataset>
