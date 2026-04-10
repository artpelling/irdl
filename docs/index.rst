``irdl``: Impulse Response Downloader
=====================================

Python package to download, unpack and process impulse response datasets in a unified way.

Usage (Python API)
------------------

The package can be included in a Python script as simple as:

.. code-block:: python

  from irdl import get_fabian

  data = get_fabian(kind='measured', hato=10)
  print(data)


.. code-block:: bash

  {'impulse_response': time domain energy Signal:
  (11950, 2) channels with 256 samples @ 44100.0 Hz sampling rate and none FFT normalization,
   'receiver_coordinates': 2D Coordinates object with 2 points of cshape (2, 1)
  Does not contain sampling weights,
   'source_coordinates': 1D Coordinates object with 11950 points of cshape (11950,)
  Does not contain sampling weights}

Usage (CLI)
-----------

Once installed, the package provides a convenient command line script which can be invoked with ``irdl``.

.. code-block:: bash

  $ irdl --help

.. literalinclude:: cli-help.txt
  :caption: Output:
  :language: bash

The supported datasets are available as subcommands, i.e.

.. code-block:: bash

  $ irdl miracle --help

.. literalinclude:: cli-miracle-help.txt
  :caption: Output:
  :language: bash


.. toctree::
   :hidden:

   Installation <installation>
   Available Datasets <datasets>
   CLI Reference <cli_ref>
   API Reference <api_ref>
