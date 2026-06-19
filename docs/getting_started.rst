Getting started
===============

This guide shows one end-to-end MIRACLE retrieval through both public entry points.

Installation summary
--------------------

Install ``irdl`` into an environment first. For the full set of installation paths,
see :doc:`installation`.

.. code-block:: bash

   $ uv pip install irdl

Python
------

Use the MIRACLE Dataset class directly:

.. code-block:: python

   from irdl import MiracleDataset

   data = MiracleDataset.get(scenario="D1", provider="auto")

   print(data["impulse_response"])
   print(data["source_coordinates"])
   print(data["receiver_coordinates"])

By default, ``get()`` returns in-memory ``pyfar`` objects. Use ``output_format`` when you
want NumPy arrays or a file-backed return such as ``"sofa"`` or ``"hdf5"``. Use
``provider`` to pin a concrete fetch endpoint or leave the default ``"auto"`` in place.
For non-raw outputs, ``"auto"`` prefers a provider-native path when one is available.

CLI
---

The CLI exposes the same MIRACLE retrieval flow as a Dataset subcommand:

.. code-block:: bash

   $ irdl get miracle --scenario D1 --provider auto

Add ``--output-format sofa`` or ``--output-format hdf5`` when you want a file-backed
result. Use ``--provider depositonce`` to pin a specific Provider, or keep ``--provider auto``
for transparent automatic selection. Use ``--export-dir`` to copy the requested artifact to an Export Directory.

``--output-format raw`` is special: it always returns the canonical Provider artifact, even
when ``--provider auto`` would otherwise prefer a mirrored provider for processed outputs.

Next steps
----------

- Browse all supported :doc:`Datasets <datasets/index>`.
- Read the :doc:`Reference <reference/index>` for the Python API and CLI.
- See :doc:`contributor-guide/processing_flow` for the deeper architecture of ``get()``.
