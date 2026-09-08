# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [1.0.0b5] - 2026-09-08

### Added
- `BrasRs8Dataset` for the BRAS-RS8 room impulse response dataset
- `MyriadDataset` for selectable MYRiAD room and microphone-array responses
- `MultiRoomTransitionDataset`  by [@Akinesia112](https://github.com/Akinesia112)

### Changed
- Reorganized dataset implementations by institution while retaining the top-level dataset API
- Centralized SOFA convention validation and improved streaming ingestion for ISTA datasets
- Updated contributor documentation for the dataset processing flow
- Upstreamed streaming SOFA verification to sofar; IRDL now requires `sofar>=1.3.0`

### Fixed
- Preserved file permissions when copying cached and exported artifacts
- Improved Windows compatibility for conversions and ISTA processing
- DepositOnce compatibility with the current DSpace API
- DepositOnce discovery of files across paginated repository responses
- Documentation build for Dataset overview pages

### Removed
- Obsolete SOFAcoustics implementation and checksum registry

---

## [1.0.0b4] - 2026-06-12

### Added
- `HutubsDataset` for SOFAcoustics HUTUBS dataset support
- `DatasetCategory` enum for organizing datasets by type (HRIR, Room Acoustics, etc.)
- New CLI commands: `irdl list` to show available datasets, `irdl cache size`, `irdl cache dir`, `irdl cache clean`, `irdl cache prune`
- Dataset registry refactoring with improved organization
- Shell integration support
- `simulated` parameter for FabianDataset (replaces `fabian` kwarg)
- `dataset_kwargs` support for FabianDataset

### Changed
- FabianDataset parameter renamed from `fabian` to `simulated` for clarity
- Improved CLI output formatting
- Refactored registry system for better dataset management
- Enhanced docstrings and documentation

### Fixed
- Sample rate handling for various dataset formats
- Process kwargs handling in dataset methods
- Cache pruning safety improvements
- Smoke tests and unit test fixes
- Documentation links and README formatting

---

## [1.0.0b3] - 2026-06-11

### Added
- Contributing guide and tutorial documentation
- Enhanced error handling with FileNotFound exceptions
- Improved logging with demoted log levels for cleaner output

### Changed
- Refactored SofaBase class for better SOFA file handling
- Improved export path construction and raw export functionality
- Cleaned up cache directory handling
- Enhanced directory setup logic
- Updated README with clearer data return format descriptions
- Improved error messages throughout the codebase

### Fixed
- Circular import issues
- Export logic clarity and robustness
- Code formatting and linting throughout
- Installation instructions and documentation
- Unreachable code removal and code simplification

---

## [1.0.0b2] - 2026-06-03

### Added
- Class-based Dataset architecture with `BaseDataset` abstract base class
- `IstaBaseDataset` for shared HDF5-based Dataset functionality (MIRACLE, SRIRACHA)
- `FabianDataset`, `MiracleDataset`, `SrirachaDataset` classes replacing wrapper functions
- SOFA as internal standard for all Datasets
- `output_format="raw"` support for accessing original downloaded files
- Separation of download and processing steps
- Automatic docstring composition for Dataset classes via `__init_subclass__`
- DOI as single source of truth (class attribute)
- Export directory support for moving final output files
- DSpace repository support for DepositOnce
- Improved download progress bar with retry logic and exponential backoff
- Comprehensive unit tests for base functionality and conversions

### Changed
- Public API: `get_miracle()`, `get_sriracha()`, `get_fabian()` → `MiracleDataset.get()`, `SrirachaDataset.get()`, `FabianDataset.get()`
- Internal HDF5 ingestion for MIRACLE and SRIRACHA datasets
- Cache directory environment variable: `CACHE_DIR` → `IRDL_CACHE_DIR`
- Progress bar implementation using Rich
- Logger module renamed to `logging` to avoid naming conflicts
- Documentation restructured with separate reference pages for API, CLI, and datasets
- Build backend switched to uv-build

### Fixed
- Sample rate handling for multi-dimensional arrays
- Memory gating for large datasets (falls back to HDF5 when data doesn't fit in memory)
- SRIRACHA non-dense scenario handling (4 split files merging)
- MIRACLE dataset split extraction
- SOFA convention compliance checking with automatic upgrading
- Cache directory property implementation for consistent path handling

### Removed
- Singleton dataset instances (`miracle_dataset`, `sriracha_dataset`, `fabian_dataset`)
- Redundant `FabianDataset.get()` override (base class handles raw correctly)
- Edge case tests that were redundant or too specific

---

## [1.0.0b1] - 2026-04-14

### Added
- Initial beta release
- Support for MIRACLE, SRIRACHA, and FABIAN datasets
- HDF5 and SOFA file format support
- Pooch-based download with checksum verification
- CLI with auto-generated help from Typer
- Python API with pyfar, numpy, hdf5, and sofa output formats
- Sphinx documentation with API reference
- GitHub Actions CI/CD workflows

### Changed
- Project renamed to IRDL (Impulse Response Downloader)
- Package structure reorganized to `src/irdl/`

### Fixed
- Initial implementation of dataset downloading and processing

---

## [1.0.0a3] - 2026-04-10

### Added
- SRIRACHA dataset support
- MIRACLE dataset split functionality
- ISTA module for shared functionality

---

## [1.0.0a2] - 2026-04-10

### Added
- Index page for documentation

---

## [1.0.0a1] - 2026-04-10

### Added
- Initial alpha release
- Basic dataset downloading infrastructure
- Zenodo repository support

---

[Unreleased]: https://github.com/artpelling/irdl/compare/v1.0.0b5...HEAD
[1.0.0b5]: https://github.com/artpelling/irdl/compare/v1.0.0b4...v1.0.0b5
[1.0.0b4]: https://github.com/artpelling/irdl/compare/v1.0.0b3...v1.0.0b4
[1.0.0b3]: https://github.com/artpelling/irdl/compare/v1.0.0b2...v1.0.0b3
[1.0.0b2]: https://github.com/artpelling/irdl/compare/v1.0.0b1...v1.0.0b2
[1.0.0b1]: https://github.com/artpelling/irdl/compare/v1.0.0a3...v1.0.0b1
[1.0.0a3]: https://github.com/artpelling/irdl/compare/v1.0.0a2...v1.0.0a3
[1.0.0a2]: https://github.com/artpelling/irdl/compare/v1.0.0a1...v1.0.0a2
[1.0.0a1]: https://github.com/artpelling/irdl/releases/tag/v1.0.0a1
