Architecture and Processing Flow
================================

This page describes the contributor-facing architecture of ``irdl`` and the conceptual flow of
a ``Dataset.get(...)`` call. We focus on the extension points rather than every private implementation detail.

Core architecture
-----------------

``BaseDataset``
   The core abstraction for all Datasets. See :class:`~irdl.base.BaseDataset`. It defines the shared ``get`` pipeline:
   common parameter validation, Provider selection, cache path handling, retrieval/process orchestration, SOFA verification,
   and output conversion.

Optional Dataset Family classes
   When multiple Datasets share dataset-specific steps in the get pipeline (see :ref:`get-processing-flow`), a Dataset
   Family class can be introduced to lift that shared logic into an intermediate base class. Do this only when the
   behaviour is genuinely shared; otherwise keep it in the individual Dataset class.

Individual Dataset classes
   Each Dataset is implemented as an individual class. Its typed ``get()`` classmethod delegates to the shared pipeline,
   while the class itself implements that pipeline's dataset-specific steps (see :ref:`get-processing-flow`). The typed
   signature and NumPy-style docstring are also used to generate CLI parameters and help text automatically.

Support modules
   Retrieval/repository helpers, CLI generation, logging/progress helpers, and small utility functions live outside the
   Dataset classes.

Class hierarchy:

.. code-block:: text

   BaseDataset
   └── Optional Dataset Family class
       └── Individual Dataset class

Cache stages
------------

``irdl`` uses three canonical Cache Stages:

1. ``provider``

   Files exactly as the Dataset Provider delivers them.

2. ``ingest``

   The single ingest-ready file that ``irdl`` can read into the internal SOFA representation.

3. ``output``

   Files produced by converting the internal SOFA representation to disk-based
   Output Formats.

Use these names in code comments and documentation. "Ingest-ready" is an adjective for a
file in the ``ingest`` stage, not a separate stage name.

.. _get-processing-flow:

``get`` processing flow
-----------------------

A public ``Dataset.get(...)`` call delegates to the shared :class:`~irdl.base.BaseDataset` flow:

.. code-block:: text

   Dataset.get(...)
     └─ BaseDataset._get(...)
         ├─ validate common and Dataset-specific parameters
         ├─ resolve canonical / explicit / auto Provider choice
         ├─ resolve provider / ingest / output paths
         ├─ [optional] raw output: retrieve canonical provider artifact and return it
         ├─ reuse cached output if available
         ├─ retrieve provider artifact if needed
         ├─ [optional] materialize output directly from a SOFA provider artifact
         ├─ [optional] process provider file(s) into ingest file
         ├─ ingest non-SOFA artifacts to internal SOFA representation
         ├─ verify and upgrade SOFA convention
         └─ convert SOFA → requested Output Format

Each Dataset has the following extension points:

``_validate_params()``
   Mandatory. Validate Dataset-specific parameters and invalid parameter combinations.

``_source_filename()``
   Mandatory. Return the canonical basename for the ingest-ready file.

``_download()``
   Mandatory. Acquire provider-stage file(s) for a selected Provider and return the primary provider artifact.

``_process()``
   Optional. Transform provider-stage files into the single ingest-ready file. The default
   implementation skips processing and handles simple single-file promotion 
   from ``provider`` to ``ingest``.

``_ingest()``
   Mandatory. Read the ingest-ready file and return the internal SOFA representation.

``_to_output()``
   Optional. Conversion methods normally stay in :class:`~irdl.base.BaseDataset`. New Datasets should not implement
   output-specific conversion unless the shared conversion layer itself needs to change.

Output behavior
---------------

``output_format="raw"`` returns the canonical provider-stage artifact before ``irdl`` processing.
For all other Output Formats, ``irdl`` either materializes the output directly from a SOFA
provider artifact or ingests the data to SOFA first and then converts from SOFA to the requested
representation. ``provider="auto"`` is transparent: IRDL logs the provider-native and
ingest-derived Provider order, then reports which Provider won. Final non-raw output is reused
from the ``output`` cache before any Provider download starts, even for explicit Provider
requests. See the public class docs in :doc:`/reference/python_api`.

When ``export_dir`` is provided, ``irdl`` copies the requested artifact to the Export Directory.
The cache remains intact so later calls can reuse provider, ingest, or output artifacts.
