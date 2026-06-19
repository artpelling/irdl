"""Datasets from the Department of Engineering Acoustics, TU Berlin, Berlin Germany.

- MIRACLE: Microphone Array Impulse Response Dataset for Acoustic Learning.
- SRIRACHA: Shoebox Room Impulse Response Archive with Varying Absorption.
"""

from pathlib import Path

import h5py as h5
import numpy as np
import sofar as sf

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi


class IstaBaseDataset(BaseDataset):
    """Base class for HDF5-based datasets from ISTA (MIRACLE, SRIRACHA).

    Both MIRACLE and SRIRACHA share identical HDF5 file structure and can use
    the same ingestion logic to convert HDF5 to SOFA format.

    Attributes
    ----------
    room_volume : float
        Room volume in cubic meters, used for SOFA metadata.
    """

    canonical_provider = "depositonce"
    providers = ("depositonce",)

    def _provider_artifact_format(self, provider: str, **_dataset_kwargs) -> str:
        """Return the Provider-side artifact Data Format.

        Current ISTA datasets publish HDF5 Provider artifacts from their
        canonical Provider.
        """
        if provider != self.canonical_provider:
            msg = f"Unknown provider {provider!r} for {self.name.upper()}"
            raise ValueError(msg)
        return "hdf5"

    def _source_filename(self, **dataset_kwargs) -> str:
        """Construct the raw input filename with extension.

        Shared implementation for MIRACLE and SRIRACHA datasets.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain ``scenario``. May contain ``dataset_split``.

        Returns
        -------
        str
            Filename in format ``{scenario}[-{split}].h5``.
        """
        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")
        return f"{scenario}{('-' + split) if split else ''}.h5"

    def _ingest(self, ingest_path: Path) -> sf.Sofa:
        """Convert a MIRACLE/SRIRACHA HDF5 file into a SOFA object.

        Both datasets share an identical HDF5 layout, so this single
        implementation covers both subclasses. The output follows the
        SingleRoomMIMOSRIR SOFA convention.

        Parameters
        ----------
        ingest_path : :class:`pathlib.Path`
            Path to the HDF5 file.

        Returns
        -------
        :class:`sofar.Sofa`
            SOFA object in the SingleRoomMIMOSRIR convention.
        """
        with h5.File(ingest_path, "r") as f:
            ir = f["data"]["impulse_response"][()]
            receiver_pos = f["data"]["location"]["receiver"][()]
            source_pos = f["data"]["location"]["source"][()]
            sampling_rate = f["metadata"]["sampling_rate"][()]
            temperature = f["metadata"]["temperature"][()]
            speed_of_sound = f["metadata"]["c0"][()]
            humidity = f["metadata"]["humidity"][()] if "humidity" in f["metadata"] else None

        m, r, _ = ir.shape
        e = 1
        c = 3
        i = 1

        sofa = sf.Sofa("SingleRoomMIMOSRIR")
        sofa.GLOBAL_Title = self.name.upper()
        sofa.GLOBAL_AuthorContact = "a.pelling@tu-berlin.de; adam.kujawksi@tu-berlin.de"
        sofa.GLOBAL_Organization = "TU Berlin, Department of Engineering Acoustics"
        sofa.GLOBAL_License = "CC BY-NC-SA 4.0"
        sofa.GLOBAL_References = self.doi
        sofa.GLOBAL_DatabaseName = self.name.upper()
        sofa.GLOBAL_RoomLocation = "TU Berlin, Einsteinufer 25, 10587 Berlin"
        sofa.GLOBAL_ListenerShortName = "Custom planar microphone array"
        sofa.GLOBAL_ListenerDescription = (
            "64-channel planar microphone array "
            "(1.5 m x 1.5 m aluminium plate, Vogel's spiral, max spacing 1.47 m, 51.2 kHz sampling rate)"
        )
        sofa.GLOBAL_ReceiverShortName = "GRAS 40PL-1 Short CCP"
        sofa.GLOBAL_SourceShortName = "Loudspeaker"
        sofa.GLOBAL_SourceDescription = (
            "Dynamic 2” cone loudspeaker in a cylindrical enclosure (Frequency range 100 Hz-16 kHz)"
        )

        sofa.RoomVolume = self.room_volume
        sofa.MeasurementDate = np.full(m, self.measurement_date)
        sofa.RoomTemperature = temperature[np.newaxis, ...] + 273.15
        sofa.RoomTemperature_Units = "kelvin"
        sofa.ListenerPosition = np.zeros((m, c))
        sofa.ListenerPosition_Type = "cartesian"
        sofa.ListenerPosition_Units = "metre"
        sofa.ReceiverPosition = receiver_pos.reshape(r, c, i)
        sofa.ReceiverPosition_Type = "cartesian"
        sofa.ReceiverPosition_Units = "metre"
        sofa.ReceiverDescriptions = np.array(["GRAS 40PL-1 Short CCP"] * r)
        sofa.ReceiverView = np.tile([1.0, 0.0, 0.0], (r, 1))[..., np.newaxis]
        sofa.ReceiverUp = np.tile([0.0, 0.0, 1.0], (r, 1))[..., np.newaxis]
        sofa.SourcePosition = source_pos
        sofa.EmitterPosition = np.zeros((e, c, i))
        sofa.EmitterPosition_Type = "cartesian"
        sofa.EmitterPosition_Units = "metre"
        sofa.Data_IR = ir[..., np.newaxis]
        sofa.Data_SamplingRate = (
            float(sampling_rate)
            if np.isscalar(sampling_rate) or len(np.unique(sampling_rate)) == 1
            else np.full((i, m), sampling_rate)
        )
        sofa.Data_Delay = np.zeros((m, r, i))
        sofa.add_variable("SpeedOfSound", speed_of_sound.reshape(m, i), "double", "MI")
        if humidity is not None:
            sofa.add_variable("Humidity", humidity.reshape(m, i), "double", "MI")

        return sofa


class MiracleDataset(IstaBaseDataset):
    """Download the MIRACLE database from DepositOnce.

    Attributes
    ----------
    name : str
        Dataset name ("miracle").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-20837").
    room_volume : float
        Room volume in cubic meters (830).
    measurement_date : float
        Release date in POSIX seconds, used as the SOFA MeasurementDate
        (no per-measurement date is available).
    """

    name = "miracle"
    doi = "10.14279/depositonce-20837"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    room_volume = 830
    measurement_date = 1697068800.0

    @classmethod
    def get(
        cls,
        scenario: str = "A1",
        dataset_split: str | None = None,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
        provider: str = "auto",
    ) -> dict | Path | None:
        """
        scenario : str
            Scenario to download. One of 'A1', 'A2', 'D1', 'R2'.
        dataset_split : str or None, optional
            Artificial dataset split. One of 'C1', 'C2', 'C3', 'C4' or None.
            Dense scenarios (D1) cannot be split.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            scenario=scenario,
            dataset_split=dataset_split,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
            provider=provider,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate MIRACLE-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain ``scenario`` (one of ``'A1'``, ``'A2'``, ``'D1'``,
            ``'R2'``). May contain ``dataset_split`` (one of ``'C1'``-``'C4'``
            or ``None``). ``provider`` and ``output_format`` are passed through
            the shared pipeline but add no MIRACLE-specific restrictions beyond
            the common rules.

        Raises
        ------
        ValueError
            If ``scenario`` or ``dataset_split`` is invalid, or if ``'D1'`` is
            combined with a split.
        """
        scenario = dataset_kwargs["scenario"]
        dataset_split = dataset_kwargs.get("dataset_split")

        if scenario not in ["A1", "A2", "D1", "R2"]:
            msg = "scenario must be one of ['A1', 'A2', 'D1', 'R2']"
            raise ValueError(msg)
        if dataset_split not in [None, "C1", "C2", "C3", "C4"]:
            msg = "dataset_split must be None or one ['C1', 'C2', 'C3', 'C4']"
            raise ValueError(msg)
        if scenario == "D1" and dataset_split is not None:
            msg = "scenario D1 cannot be split"
            raise ValueError(msg)

    def _download(self, provider_dir: Path, provider: str, **dataset_kwargs) -> Path:
        """Download the MIRACLE Provider artifact.

        MIRACLE currently supports only its canonical Provider. The canonical
        artifact is always the full-scenario HDF5 file; split extraction is a
        processing step, not a Provider concern.

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Provider directory (for example ``cache/MIRACLE/provider/depositonce``).
        provider : str
            Provider name. Must be the canonical Provider.
        **dataset_kwargs : dict
            Must contain ``scenario``. May contain ``dataset_split``.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded full-scenario HDF5 file inside the Provider
            directory.
        """
        if provider != self.canonical_provider:
            msg = f"Unknown provider {provider!r} for {self.name.upper()}"
            raise ValueError(msg)
        full_path = provider_dir / self._source_filename(**{**dataset_kwargs, "dataset_split": None})
        self.logger.info("provider=%r artifact=%r -> download to provider cache", provider, full_path.name)
        pup = _pooch_from_doi(self.doi, path=provider_dir)
        _fetch(pup, full_path.name)
        return full_path

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process MIRACLE files if needed.

        If a dataset split is requested, extract the requested quadrant from the
        full-scenario Provider artifact into the ingest directory. Otherwise
        promote the Provider file to the ingest stage unchanged.
        """
        split = dataset_kwargs.get("dataset_split")
        if not split:
            return super()._process(provider_artifact, ingest_path, **dataset_kwargs)
        return self._extract_split(provider_artifact, split, ingest_path)

    def _extract_split(self, ingest_path: Path, dataset_split: str, output_path: Path) -> Path:
        """Extract a dataset split from a full MIRACLE HDF5 file.

        Reads the full file, indexes the requested quadrant of the source grid,
        and writes the result to a new HDF5 file.

        Parameters
        ----------
        ingest_path : :class:`pathlib.Path`
            Path to the full HDF5 file in the Provider directory.
        dataset_split : str
            Split to extract. One of ``'C1'``, ``'C2'``, ``'C3'``, ``'C4'``.
        output_path : :class:`pathlib.Path`
            Target path in the ingest directory.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the extracted split HDF5 file.
        """
        self.logger.info("Extracting split %s from %s -> %s", dataset_split, ingest_path.name, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5.File(ingest_path, "r") as f:
            data = {
                "impulse_response": f["data"]["impulse_response"][()],
                "receiver_coordinates": f["data"]["location"]["receiver"][()],
                "source_coordinates": f["data"]["location"]["source"][()],
                "speed_of_sound": f["metadata"]["c0"][()],
                "temperature": f["metadata"]["temperature"][()],
                "sampling_rate": f["metadata"]["sampling_rate"][()],
            }
            if "humidity" in f["metadata"]:
                data["humidity"] = f["metadata"]["humidity"][()]

        offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}
        row, column = offsets[dataset_split]
        n = int(np.sqrt(data["source_coordinates"].shape[0]))
        ir_shape = data["impulse_response"].shape

        data["source_coordinates"] = data["source_coordinates"].reshape(n, n, 3)[row::2, column::2, :].reshape(-1, 3)
        data["impulse_response"] = (
            data["impulse_response"].reshape(n, n, *ir_shape[1:])[row::2, column::2, :].reshape(-1, *ir_shape[1:])
        )
        data["temperature"] = data["temperature"].reshape(n, n)[row::2, column::2].reshape(-1)
        data["speed_of_sound"] = data["speed_of_sound"].reshape(n, n)[row::2, column::2].reshape(-1)
        if "humidity" in data:
            data["humidity"] = data["humidity"].reshape(n, n)[row::2, column::2].reshape(-1)

        with h5.File(output_path, "w") as f:
            data_group = f.create_group("data")
            data_group.create_dataset("impulse_response", data=data["impulse_response"])
            location_group = data_group.create_group("location")
            location_group.create_dataset("source", data=data["source_coordinates"])
            location_group.create_dataset("receiver", data=data["receiver_coordinates"])
            metadata_group = f.create_group("metadata")
            metadata_group.create_dataset("c0", data=data["speed_of_sound"])
            metadata_group.create_dataset("temperature", data=data["temperature"])
            metadata_group.create_dataset("sampling_rate", data=data["sampling_rate"])
            if "humidity" in data:
                metadata_group.create_dataset("humidity", data=data["humidity"])

        return output_path


class SrirachaDataset(IstaBaseDataset):
    """Download and merge the SRIRACHA database from DepositOnce.

    Attributes
    ----------
    name : str
        Dataset name ("sriracha").
    doi : str
        Digital Object Identifier ("10.14279/depositonce-23943").
    room_volume : float
        Room volume in cubic meters (73.5).
    measurement_date : float
        Release date in POSIX seconds, used as the SOFA MeasurementDate
        (no per-measurement date is available).
    """

    name = "sriracha"
    doi = "10.14279/depositonce-23943"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES
    room_volume = 73.5
    measurement_date = 1755648000.0

    @classmethod
    def get(
        cls,
        scenario: str = "SR1-D",
        dataset_split: str | None = None,
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
        provider: str = "auto",
    ) -> dict | Path | None:
        """
        scenario : str, optional
            Scenario to download. One of 'SR1', 'SRA1', 'SR1-D', 'SRA1-D',
            'SR2', 'SRA2', 'SR2-D', 'SRA2-D'. Default is 'SR1-D'.
        dataset_split : str or None, optional
            Optional dataset split for full-plane scenarios. One of 'C1',
            'C2', 'C3', 'C4' or None. Dense scenarios (ending in '-D') do not
            have splits. Default is None.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            scenario=scenario,
            dataset_split=dataset_split,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
            provider=provider,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate SRIRACHA-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain ``scenario`` (one of the supported SRIRACHA scenario
            names). May contain ``dataset_split``. ``output_format`` is used to
            forbid ``raw`` for non-dense full-plane retrieval because that case
            consists of four Provider artifacts rather than one.

        Raises
        ------
        ValueError
            If parameters are invalid, if a dense scenario is combined with a
            split, or if ``raw`` is requested for a non-dense full-plane
            scenario.
        """
        scenario = dataset_kwargs.get("scenario")
        dataset_split = dataset_kwargs.get("dataset_split")
        output_format = dataset_kwargs.get("output_format")

        if scenario not in ["SR1", "SRA1", "SR1-D", "SRA1-D", "SR2", "SRA2", "SR2-D", "SRA2-D"]:
            msg = "scenario must be one of [SR1, SRA1, SR1-D, SRA1-D, SR2, SRA2, SR2-D, SRA2-D]"
            raise ValueError(msg)
        if dataset_split not in [None, "C1", "C2", "C3", "C4"]:
            msg = "dataset_split must be None or in [C1, C2, C3, C4]"
            raise ValueError(msg)
        if scenario[-1] == "D" and dataset_split is not None:
            msg = "dense datasets do not have splits"
            raise ValueError(msg)
        if output_format == "raw" and scenario and scenario[-1] != "D" and dataset_split is None:
            msg = "raw output_format not supported for non-dense SRIRACHA scenarios without split"
            raise ValueError(msg)

    def _download(self, provider_dir: Path, provider: str, **dataset_kwargs) -> Path:
        """Download SRIRACHA Provider artifact(s) to the Provider directory.

        Dense scenarios and explicit splits map to one Provider file. Non-dense
        full-plane scenarios map to four Provider files that must later be
        merged into one ingest-ready HDF5 file.

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Provider directory (for example ``cache/SRIRACHA/provider/depositonce``).
        provider : str
            Provider name. Must be the canonical Provider.
        **dataset_kwargs : dict
            Must contain ``scenario``. May contain ``dataset_split``.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded file or to the Provider directory when the
            request expands to multiple split files.
        """
        if provider != self.canonical_provider:
            msg = f"Unknown provider {provider!r} for {self.name.upper()}"
            raise ValueError(msg)

        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")

        if scenario.endswith("D") or split is not None:
            fname = self._source_filename(**dataset_kwargs)
            target_file = provider_dir / fname
            self.logger.info("provider=%r artifact=%r -> download to provider cache", provider, target_file.name)
            pup = _pooch_from_doi(self.doi, path=provider_dir)
            _fetch(pup, fname)
            return target_file

        self.logger.info("Downloading split provider artifacts for scenario %s from %r", scenario, provider)
        pup = _pooch_from_doi(self.doi, path=provider_dir)
        for split_file in ["C1", "C2", "C3", "C4"]:
            fname = f"{scenario}-{split_file}.h5"
            _fetch(pup, fname)
        return provider_dir

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Post-process SRIRACHA Provider artifacts if needed.

        Dense scenarios and explicit splits promote one Provider artifact to the
        ingest stage. Non-dense full-plane scenarios merge four Provider split
        files into one ingest-ready HDF5 file.
        """
        scenario = dataset_kwargs["scenario"]
        split = dataset_kwargs.get("dataset_split")

        if scenario.endswith("D") or split is not None:
            return super()._process(provider_artifact, ingest_path, **dataset_kwargs)
        self.logger.debug("Merging split provider artifacts")
        return self._merge_split_files(scenario, provider_artifact, ingest_path)

    def _merge_split_files(self, scenario: str, provider_artifact: Path, ingest_path: Path) -> Path:
        """Merge four quadrant HDF5 files into a full-plane file.

        Reads metadata from the first split file in the Provider directory,
        allocates output datasets with the full source-grid shape, copies each
        split's measurements into the correct interleaved positions, and then
        deletes the split Provider files.

        Parameters
        ----------
        scenario : str
            Scenario name (for example ``'SR1'``).
        provider_artifact : Path
            Provider directory where split files are downloaded.
        ingest_path : Path
            Target path in the ingest directory for the merged file.

        Returns
        -------
        Path
            Path to the merged HDF5 file.
        """
        offsets = {"C1": (0, 0), "C2": (0, 1), "C3": (1, 0), "C4": (1, 1)}

        split_files = {split: provider_artifact / f"{scenario}-{split}.h5" for split in offsets}

        with h5.File(split_files["C1"], "r") as f:
            ir_shape = f["data"]["impulse_response"].shape
            ir_dtype = f["data"]["impulse_response"].dtype
            n_split = ir_shape[0]
            sampling_rate = f["metadata"]["sampling_rate"][()]
            receiver = f["data"]["location"]["receiver"][()]
            has_humidity = "humidity" in f["metadata"]

        n_sources = len(split_files) * n_split
        n_full_grid = int(np.sqrt(n_sources))
        n_split_grid = n_full_grid // 2

        with h5.File(ingest_path, "w") as out:
            data_grp = out.create_group("data")
            ir_ds = data_grp.create_dataset("impulse_response", shape=(n_sources, *ir_shape[1:]), dtype=ir_dtype)
            loc_grp = data_grp.create_group("location")
            src_ds = loc_grp.create_dataset("source", shape=(n_sources, 3), dtype="float64")
            loc_grp.create_dataset("receiver", data=receiver)

            meta_grp = out.create_group("metadata")
            meta_grp.create_dataset("sampling_rate", data=sampling_rate)
            c0_ds = meta_grp.create_dataset("c0", shape=(n_sources,), dtype="float32")
            temp_ds = meta_grp.create_dataset("temperature", shape=(n_sources,), dtype="float32")
            if has_humidity:
                hum_ds = meta_grp.create_dataset("humidity", shape=(n_sources,), dtype="float32")

            handles = {s: h5.File(f, "r") for s, f in split_files.items()}
            try:
                for split_name, (row, col) in offsets.items():
                    f = handles[split_name]
                    for r in range(n_split_grid):
                        src = slice(r * n_split_grid, (r + 1) * n_split_grid)
                        grid_row = 2 * r + row
                        dst = slice(grid_row * n_full_grid + col, grid_row * n_full_grid + n_full_grid, 2)

                        ir_ds[dst] = f["data"]["impulse_response"][src]
                        src_ds[dst] = f["data"]["location"]["source"][src]
                        c0_ds[dst] = f["metadata"]["c0"][src]
                        temp_ds[dst] = f["metadata"]["temperature"][src]
                        if has_humidity:
                            hum_ds[dst] = f["metadata"]["humidity"][src]
            finally:
                for fh in handles.values():
                    fh.close()

            for f in split_files.values():
                f.unlink()

        self.logger.debug("Split provider artifacts merged")

        return ingest_path
