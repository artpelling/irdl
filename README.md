[![PyPI](https://img.shields.io/pypi/pyversions/irdl.svg)](https://pypi.org/project/irdl)
[![PyPI](https://img.shields.io/pypi/v/irdl.svg)](https://pypi.org/project/irdl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19496801.svg)](https://zenodo.org/doi/10.5281/zenodo.19496801)

# `irdl`: Impulse Response Downloader
Python package to download, unpack and process impulse response datasets in a unified way.

## Highlights
- Returns data in a standardised format either
  - a dictionary of [`pyfar`](https://pyfar.org)-objects (default) (`'pyfar'`)
  - a dictionary of NumPy arrays (`'numpy'`)
  - a path to an HDF5-file for partial data access not having to load the entire data into memory (`'hdf5'`)
  - a path to a SOFA-file, the standardised format for spatially oriented acoustic data (`'sofa'`)
  - a path to the unprocessed provider files as downloaded (`'raw'`)
- Leverages [`pooch`](https://www.fatiando.org/pooch/latest/) to download impulse response datasets and verifies their integrity with DOI metadata or packaged hash registries.
- Only downloads, extracts and processes what is actually needed.
- Supports provider selection via ``provider=...`` / ``--provider ...`` with transparent ``auto`` resolution.
- Restricts ``output_format="raw"`` to the canonical Provider while allowing ``auto`` to prefer mirrored SOFA-native Providers for processed outputs.
- Adds `pooch`-support for dSpace repositories, such as TU Berlin [depositonce](https://depositonce.tu-berlin.de/home).
- Data storage location can be set by the `IRDL_CACHE_DIR` environment variable (defaults to the user cache directory).
- Output can be processed and exported to a custom location via the `export_dir` argument.

## Links
- [Documentation](https://artpelling.github.io/irdl/)
- [Installation instructions](https://artpelling.github.io/irdl/installation.html)
- [Available datasets](https://artpelling.github.io/irdl/datasets)
- [Adding a Dataset](https://artpelling.github.io/irdl/contributor-guide/adding_dataset.html)
- [Reference (CLI & API)](https://artpelling.github.io/irdl/reference)
- [Changelog](https://github.com/artpelling/irdl/blob/main/CHANGELOG.md)

## Usage (Python API)

The package can be included in a Python script as simple as:

``` python
from irdl import MiracleDataset

data = MiracleDataset.get(scenario='D1', provider='auto')
print(data)
```

``` shell
INFO     Downloading MIRACLE scenario D1                              
INFO     Downloading file 'D1.h5' from                                
         'https://api-depositonce.tu-berlin.de/server/api/core/bitstre
         ams/4b6cb9e5-f1e4-42c6-9b41-0a85e6ee9422/content' to         
         '/home/pelling/.cache/irdl/MIRACLE/provider'.                
D1.h5 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 302.3/302.3 MB 51.8 TB/s 0:00:00
INFO     Convention SingleRoomMIMOSRIR v1.0 is up to date             
{'impulse_response': time domain energy Signal:
(1, 1089, 64) channels with 1024 samples @ 32000.0 Hz sampling rate and none FFT normalization
, 'source_coordinates': 1D Coordinates object with 1089 points of cshape (1089,)

Does not contain sampling weights, 'receiver_coordinates': 2D Coordinates object with 64 points of cshape (64, 1)

Does not contain sampling weights}
```

## Usage (CLI)

Once installed, the package provides a convenient command line script which can be invoked with `irdl`.
Dataset downloads live under the `get` command.

``` shell                                                                   
 Usage: irdl [OPTIONS] COMMAND [ARGS]...                              
                                                                      
 Main callback for version flag handling.                             
                                                                      
╭─ Options ──────────────────────────────────────────────────────────╮
│ --version                     Show version and exit.               │
│ --install-completion          Install completion for the current   │
│                               shell.                               │
│ --show-completion             Show completion for the current      │
│                               shell, to copy it or customize the   │
│                               installation.                        │
│ --help                        Show this message and exit.          │
╰────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────╮
│ list   List all available datasets.                                │
│ cache  Manage cache directory.                                     │
│ get    Download datasets.                                          │
╰────────────────────────────────────────────────────────────────────╯
```

Example:

``` shell
$ irdl get miracle --scenario D1 --provider auto
```

Provider selection rules:

- ``--provider auto`` prefers a provider-native path when one is available for the requested output format.
- ``--provider <name>`` pins a concrete Provider but still reuses any matching cached final output.
- ``--output-format raw`` always uses the canonical Provider artifact.
