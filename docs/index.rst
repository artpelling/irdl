Impulse Response Downloader
===========================

``irdl`` retrieves, caches, and converts impulse response Datasets in a unified way.

.. code-block:: python

   from irdl import MiracleDataset

   data = MiracleDataset.get(scenario="D1", provider="auto")
   print(data["impulse_response"])

.. code-block:: bash

   $ irdl get miracle --scenario D1 --provider auto

``irdl`` follows a simple user-facing flow:

1. Choose a Dataset and parameters.
2. ``irdl`` selects a Provider, retrieves the data, and reuses cached source artifacts.
3. ``irdl`` processes the artifacts if needed and internally parses it to `SOFA standard <https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)>`_.
4. ``irdl`` returns the requested output format either as a path or in-memory objects.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      Install ``irdl`` and run the first dataset retrieval from Python or the CLI.

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Compare ``uv`` and ``pip`` installation paths and global tool setup.

   .. grid-item-card:: Datasets
      :link: datasets/index
      :link-type: doc

      Browse all available datasets.

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      Jump to the Python API and CLI reference.

.. toctree::
   :hidden:

   Introduction <self>
   Getting started <getting_started>
   Installation <installation>
   Datasets <datasets/index>
   Reference <reference/index>
   Contributor Guide <contributor-guide/index>
