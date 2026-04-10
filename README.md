[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19496801.svg)](https://doi.org/10.5281/zenodo.19496801)

# `irdl`: Impulse Response Downloader
Python package to download, unpack and process impulse response datasets in a unified way.

## Highlights
- Returns a standardised dictionary of [`pyfar`](https://pyfar.org)-objects.
- Leverages [`pooch`](https://www.fatiando.org/pooch/latest/) to download impulse response datasets and verifies their integrity with a checksum. 
- Only downloads, extracts and processes what is actually needed.
- Adds `pooch`-support for dSpace repositories, such as [depositonce](https://depositonce.tu-berlin.de/home).
- Data storage location can be set by `IRDL_DATA_DIR` environmental variable (defaults to user cache directory).

## Links
- [Documentation](https://artpelling.github.io/irdl/)
- [Installation instructions](https://artpelling.github.io/irdl/installation.html)
- [Available Datasets](https://artpelling.github.io/irdl/api_ref.html)
- [CLI Reference](https://artpelling.github.io/irdl/cli_ref.html)

## Usage (Python API)

The package can be included in a Python script as simple as:

``` python
from irdl import get_fabian

data = get_fabian(kind='measured', hato=10)
print(data)
```

``` shell
{'impulse_response': time domain energy Signal:
(11950, 2) channels with 256 samples @ 44100.0 Hz sampling rate and none FFT normalization
,
 'receiver_coordinates': 2D Coordinates object with 2 points of cshape (2, 1)

Does not contain sampling weights,
 'source_coordinates': 1D Coordinates object with 11950 points of cshape (11950,)

Does not contain sampling weights}
```

## Usage (CLI)

Once installed, the package provides a convenient command line script which can be invoked with `irdl`.

``` shell
$ irdl --help

 Usage: irdl [OPTIONS] COMMAND [ARGS]...                                          
                                                                                  
╭─ Options ──────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.        │
│ --show-completion             Show completion for the current shell, to copy   │
│                               it or customize the installation.                │
│ --help                        Show this message and exit.                      │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────╮
│ fabian    Download and extract the FABIAN HRTF Database v4 from DepositOnce.   │
│ miracle   Download and extract the MIRACLE database from DepositOnce.          │
╰────────────────────────────────────────────────────────────────────────────────╯
```
