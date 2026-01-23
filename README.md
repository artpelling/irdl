# `irdl`: Impulse Response Downloader
Python package to download, unpack and process impulse response datasets.

## Highlights
- Returns a dictionary of [`pyfar`](https://pyfar.org)-objects in a standardised way.
- Leverages [`pooch`](https://www.fatiando.org/pooch/latest/) to download impulse response datasets and verifies their integrity with a checksum. 
- Only downloads, extracts and processes what is needed.
- Adds `pooch`-support for dSpace repositories, such as [depositonce](https://depositonce.tu-berlin.de/home).
- Data storage location can be set by `IRDL_DATA_DIR` environmental variable (defaults to user cache directory).

## Installation
Add
``` toml
dependencies = [
   "irdl @ git+https://github.com/artpelling/irdl"
]
```
to your `pyproject.toml` (or similarly to `requirements.txt`). 

## Available datasets

#### Room Impulse Responses
- [MIRACLE](https://doi.org/10.14279/depositonce-20837) via `irdl miracle`

#### Head-related Impulse Responses
- [FABIAN-HRTF](https://doi.org/10.14279/depositonce-5718.5) via `irdl fabian`

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

The supported datasets are available as subcommands, i.e.

``` shell
$ irdl miracle --help

 Usage: irdl miracle [OPTIONS]                                                    
                                                                                  
 Download and extract the MIRACLE database from DepositOnce.                      
                                                                                  
 DOI: 10.14279/depositonce-20837                                                  
                                                                                  
╭─ Options ──────────────────────────────────────────────────────────────────────╮
│ --scenario        TEXT  Name of the scenario to download. Either 'A1', 'A2',   │
│                         'D1', or 'R2'.                                         │
│                         [default: A1]                                          │
│ --path            TEXT  Path to the directory where the data should be stored. │
│                         Will be overwritten, if the environment variable       │
│                         `IRDL_DATA_DIR` is set. Default is the user cache      │
│                         directory.                                             │
│                         [default: ~/.cache/irdl]                   │
│ --help                  Show this message and exit.                            │
╰────────────────────────────────────────────────────────────────────────────────╯
```

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
