Python API
==========

.. currentmodule:: irdl

The public Python API is centered on concrete Dataset classes and a shared retrieval flow.

Dataset modules
---------------

.. autosummary::
   :caption: Core Python modules
   :toctree: ../_autosummary

   aalto
   akt
   base
   dechorate
   esat
   iks
   ista
   sonicom

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
