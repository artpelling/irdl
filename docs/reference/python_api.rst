Python API
==========

.. currentmodule:: irdl

The public Python API is centered on concrete Dataset classes and a shared retrieval flow.

Streaming batch budget
----------------------

One package-wide setting controls streamed IR batches for Ingest, payload checks,
and SOFA-to-IRDL-HDF5 export:

.. code-block:: python

   import irdl

   irdl.config.batch_bytes = 16 * 1024**2  # 16 MiB; default is 64 MiB
   path = irdl.Meshgrid3dDataset.get(speaker=1, output_format="hdf5")

Set it before calling ``get()``. A running Get keeps its starting budget, even if
another caller changes the setting; subsequent calls use the updated value.
Python values must be integers of at least 8 bytes. There are no per-Dataset size
settings or extra ``get()`` arguments.

For CLI use, initialize the same setting with a positive integer number of MiB
in ``IRDL_BATCH_MIB`` before starting the process:

.. code-block:: console

   $ IRDL_BATCH_MIB=16 irdl get 3dmeshgrid --speaker 1 --output-format hdf5

Slices are calculated from array shapes and dtypes, rather than a fixed number
of measurements or receivers. Compressed SOFA storage chunks are chosen
automatically, capped at 1 MiB and the batch budget. Queued compression buffers
also respect the budget. Changing the setting does not rewrite existing files.

This is not a total-process memory limit: several arrays, compression buffers,
and library caches can coexist. NumPy and pyfar outputs still load all requested
IRs, and older non-streaming Ingest implementations are not made streamed by this
setting.

Dataset modules
---------------

.. autosummary::
   :caption: Core Python modules
   :toctree: ../_autosummary

   base
   config
   datasets.aalto
   datasets.akt
   datasets.barilan
   datasets.esat
   datasets.iks
   datasets.ista
   datasets.sonicom
   datasets.uec

Registry
--------

The checked-in ``registry/`` Hash Registry stores SHA-256 digests for Provider artifacts, including MIRD archives and direct-SOFA files. It is package data, not a public Python API.

Internal modules
----------------

.. autosummary::
   :caption: Internal modules
   :toctree: ../_autosummary

   cli
   downloader
   logging
   repositories
   utils
