"""Datasets from the Department of Electrical Engineering, KU Leuven, Leuven, Belgium.

Currently this module hosts:

- MYRiAD: A Multi-Array Room Acoustic Database

"""

from pathlib import Path
from typing import ClassVar
from zipfile import ZipFile

import numpy as np
import pyfar as pf
import sofar as sf

from irdl.base import BaseDataset, DatasetCategory
from irdl.downloader import _fetch, _pooch_from_doi
from irdl.logging import logger


class MyriadDataset(BaseDataset):
    """Download the MYRiAD database (RIR subset) from Zenodo.

    Attributes
    ----------
    name : str
        Dataset name ("myriad").
    doi : str
        Digital Object Identifier ("10.5281/zenodo.7389996").
    """

    name = "myriad"
    doi = "10.5281/zenodo.7389996"
    _category = DatasetCategory.ROOM_IMPULSE_RESPONSES

    # Selectable array-group labels and their corresponding microphone labels
    _ARRAY_GROUPS: ClassVar[dict[str, list[str]]] = {
        "dummy_head": ["DHL", "DHR"],
        "bte-pieces": ["BTELF", "BTELB", "BTERF", "BTERB"],
        "external-microphones": ["XM1", "XM2", "XM3", "XM4", "XM5"],
        "circular-microphone-array": [
            "CMA10_-90",
            "CMA10_0",
            "CMA10_90",
            "CMA10_180",
            "CMA20_-135",
            "CMA20_-90",
            "CMA20_-45",
            "CMA20_0",
            "CMA20_45",
            "CMA20_90",
            "CMA20_135",
            "CMA20_180",
        ],
    }

    @classmethod
    def get(
        cls,
        room: str = "SAL",
        array: str = "all",
        config: str = "P1",
        cache_dir: str | Path | None = None,
        export_dir: str | Path | None = None,
        output_format: str = "pyfar",
    ) -> dict | Path | None:
        """
        room : str
            Room to load. Either ``'SAL'`` or ``'AIL'``. Default is ``'SAL'``.
        array : str
            Microphone array group(s) to load. One of ``'dummy_head'``,
            ``'bte-pieces'``, ``'external-microphones'``, ``'circular-microphone-array'``,
            a comma-separated combination of these (e.g. ``'dummy_head,bte-pieces'``),
            or ``'all'`` for every group available in the room.
            A list of group names is also accepted via the Python API.
            The circular array is only available in ``'AIL'``. Default is ``'all'``.
        config : str
            Microphone configuration placement in the AIL. Either ``'P1'`` or ``'P2'``.
            Ignored for the SAL. Default is ``'P1'``.

        Returns
        -------
        dict or Path
            For 'pyfar' / 'numpy': dict of in-memory objects.
            For 'sofa' / 'hdf5' / 'raw': Path to file on disk.
        """  # noqa: D205, D403
        return cls()._get(
            room=room,
            array=array,
            config=config,
            cache_dir=cache_dir,
            export_dir=export_dir,
            output_format=output_format,
        )

    def _validate_params(self, **dataset_kwargs) -> None:
        """Validate MYRiAD-specific parameters.

        Parameters
        ----------
        **dataset_kwargs : dict
            Must contain ``'room'`` (one of ``'SAL'``, ``'AIL'``), ``'array'`` (one
            or more valid group names or ``'all'``) and ``'config'`` (one of ``'P1'``,
            ``'P2'``). ``output_format`` is also passed but unused here.

        Raises
        ------
        ValueError
            If any parameter is out of range, the circular array is requested for
            ``'SAL'``, or ``'MultiSpeakerBRIR'`` is combined with more than the
            ``'dummy_head'`` group.
        """
        selection = self._resolve(**dataset_kwargs)

        if selection["room"] not in ("SAL", "AIL"):
            msg = "room must be one of ['SAL', 'AIL']"
            raise ValueError(msg)

        for group in selection["groups"]:
            if group not in self._ARRAY_GROUPS:
                msg = f"array group(s) {group!r} must be one of {list(self._ARRAY_GROUPS)} or 'all'"
                raise ValueError(msg)
            if selection["room"] == "SAL" and group == "circular-microphone-array":
                msg = f"array group {group!r} is not available in the SAL"
                raise ValueError(msg)

        if selection["room"] == "AIL" and selection["config"] not in ("P1", "P2"):
            msg = "config must be one of ['P1', 'P2'] for the AIL"
            raise ValueError(msg)

    def _source_filename(self, **dataset_kwargs) -> str:
        """Build the ingest filename encoding the full selection.

        The name has the form ``MYRIAD_<room>[_<config>]_<groups>_<convention>``,
        where ``<config>`` is included only for the AIL and ``<groups>`` is the
        selected array groups as short tokens. ``<convention>`` is part of the name
        because the ingest archive embeds the convention in its metadata sidecar,
        so each convention must map to a distinct cache file.

        Returns
        -------
        str
            Filename for the ingest-stage archive.
        """
        # Short tokens used to build a unique, filesystem-friendly cache filename.
        group_token = {
            "dummy_head": "dh",
            "bte-pieces": "bte",
            "external-microphones": "xm",
            "circular-microphone-array": "cma",
        }
        selection = self._resolve(**dataset_kwargs)

        tokens = "-".join(group_token[g] for g in selection["groups"])
        parts = ["MYRIAD", selection["room"]]
        if selection["room"] == "AIL":
            parts.append(selection["config"])
        parts.append(tokens)
        return "_".join(parts)

    def _download(self, provider_dir: Path, **dataset_kwargs) -> Path:  # noqa: ARG002
        """Download MYRiAD ZIP archive to the provider directory.

        Only downloads the archive if it is not already cached. Returns the
        ZIP path so that ``_process`` can extract and process the requested files.

        Parameters
        ----------
        provider_dir : :class:`pathlib.Path`
            Provider directory (e.g., ``cache/MYRiAD/provider/``).
        **dataset_kwargs : dict
            Dataset keyword arguments.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the downloaded ZIP archive.
        """
        zipfile_name = "MYRiAD_V2_econ.zip"
        zip_path = provider_dir / zipfile_name
        if zip_path.exists():
            logger.info(f"MYRiAD ZIP archive already cached at {zip_path}, skipping download")
        else:
            logger.info("Downloading MYRiAD (econ RIR archive)")
            pup = _pooch_from_doi(self.doi, path=provider_dir)
            _fetch(pup, zipfile_name)
        return zip_path

    def _process(self, provider_artifact: Path, ingest_path: Path, **dataset_kwargs) -> Path:
        """Extract only the selected RIR WAVs and coordinate CSV into the ingest directory.

        Performs no signal processing: selects the WAVs and the room coordinate CSV for
        the requested selection and extracts them, preserving the archive tree, into
        ``ingest_path``. All SOFA construction is deferred to :meth:`_ingest`.

        Parameters
        ----------
        provider_artifact : :class:`pathlib.Path`
            Path to the downloaded MYRiAD ZIP archive.
        ingest_path : :class:`pathlib.Path`
            Target directory for the extracted files.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the ingest directory (``ingest_path``).
        """
        selection = self._resolve(**dataset_kwargs)
        ingest_path.mkdir(parents=True, exist_ok=True)

        # build strings for each file that needs to be extracted
        files = []
        for speaker in selection["speakers"]:
            base = f"MYRiAD_V2_econ/audio/{selection['room']}/{speaker}"
            if selection["room"] == "AIL":
                base = f"{base}/{selection['config']}"
            files += [f"{base}/{mic}_RIR.wav" for mic in selection["mics"]]
        files.append(f"MYRiAD_V2_econ/coord/{selection['room']}.csv")

        # extract files while preserving the archive tree
        with ZipFile(provider_artifact, "r") as zf:
            for member in files:
                logger.debug(f"Extracting {member}")
                zf.extract(member, path=ingest_path)

        return ingest_path

    def _ingest(self, ingest_artifact: Path, sofa_path: Path, **dataset_kwargs) -> Path:
        """Build the internal SOFA from the extracted files and write it to sofa_path.

        Stacks the selected loudspeaker/microphone WAVs into a ``Data.IR`` array of
        shape ``(M, R, N, E)``, attaches the geometry from the room coordinate CSV,
        layers on the descriptive metadata, and writes the result. The loudspeakers
        are the *emitters* (E) and the selected microphones are the *receivers* (R) —
        the inverse of the moving-source ISTA datasets, where source positions index M.

        Parameters
        ----------
        ingest_artifact : :class:`pathlib.Path`
            Ingest directory populated by :meth:`_process`.
        sofa_path : :class:`pathlib.Path`
            Target path for the retained SOFA file.

        Returns
        -------
        :class:`pathlib.Path`
            Path to the written SOFA file (``sofa_path``).
        """
        group_hardware = {
            "dummy_head": "Neumann KU 100 in-ear microphone",
            "bte-pieces": "Cochlear behind-the-ear microphone",
            "external-microphones": "External microphone",
            "circular-microphone-array": "Circular microphone array (DPA 4060 / AKG CK32)",
        }
        room_volumes = {"SAL": 102, "AIL": 208}
        sampling_rate = 44100
        n_samples = 132300

        selection = self._resolve(**dataset_kwargs)
        audio_root = ingest_artifact / "MYRiAD_V2_econ" / "audio" / selection["room"]

        # --- stack the impulse responses, shape (M=1, R, N, E) -------------------
        ir = np.zeros((1, len(selection["mics"]), n_samples, len(selection["speakers"])), dtype=np.float32)
        for ei, speaker in enumerate(selection["speakers"]):
            speaker_dir = (
                audio_root / speaker / selection["config"] if selection["room"] == "AIL" else audio_root / speaker
            )
            for ri, mic in enumerate(selection["mics"]):
                ir[0, ri, :, ei] = pf.io.read_audio(speaker_dir / f"{mic}_RIR.wav").time[0]

        # --- coordinates: CSV columns are Label, x, y, z -------------------------
        csv_path = ingest_artifact / "MYRiAD_V2_econ" / "coord" / f"{selection['room']}.csv"
        rows = np.atleast_2d(np.genfromtxt(csv_path, delimiter=",", dtype=str, skip_header=1))
        mic_type = "MIC" if selection["room"] == "SAL" else f"MIC {selection['config']}"

        speaker_coords, mic_coords = {}, {}
        for kind, label, *raw in rows:
            xyz = np.array(raw[:3], dtype=float)
            if kind.strip() == "LS":
                speaker_coords[label.strip()] = xyz
            elif kind.strip() == mic_type:
                mic_coords[label.strip()] = xyz

        source = np.array([speaker_coords[s] for s in selection["speakers"]])  # (E, C)
        receiver = np.array([mic_coords[mic] for mic in selection["mics"]])  # (R, C)

        m, r, _n, e = ir.shape

        label_to_group = {mic: group for group, labels in self._ARRAY_GROUPS.items() for mic in labels}
        descriptions = [group_hardware[label_to_group[mic]] for mic in selection["mics"]]

        # --- build the SOFA ------------------------------------------------------
        sofa = sf.Sofa("SingleRoomMIMOSRIR")
        sofa.GLOBAL_Title = "MYRiAD"
        sofa.GLOBAL_DatabaseName = "MYRiAD"
        sofa.GLOBAL_Organization = "KU Leuven, ESAT-STADIUS"
        sofa.GLOBAL_AuthorContact = "thomas.dietzen@esat.kuleuven.be"
        sofa.GLOBAL_References = f"https://doi.org/{self.doi}"
        sofa.GLOBAL_RoomType = "reverberant"
        sofa.GLOBAL_Comment = f"MYRiAD room {selection['room']}; emitters=loudspeakers, receivers=selected microphones."

        sofa.Data_IR = ir  # (M=1, R, N, E)
        sofa.Data_SamplingRate = float(sampling_rate)
        sofa.Data_Delay = np.zeros((m, r, e))
        sofa.RoomVolume = room_volumes[selection["room"]]

        sofa.ListenerPosition = np.zeros((1, 3))  # (M=1, C) origin
        sofa.SourcePosition = np.zeros((1, 3))  # (M=1, C) origin
        sofa.ReceiverView = np.tile([1.0, 0.0, 0.0], (r, 1))[..., np.newaxis]  # (R,C,I)
        sofa.ReceiverUp = np.tile([0.0, 0.0, 1.0], (r, 1))[..., np.newaxis]
        sofa.ReceiverPosition = receiver.reshape(r, 3, 1)  # (R, C, I) absolute room coords
        sofa.ReceiverDescriptions = np.array(descriptions)  # (R, S)
        sofa.EmitterView = np.tile([1.0, 0.0, 0.0], (e, 1))[..., np.newaxis]  # (E,C,I)
        sofa.EmitterUp = np.tile([0.0, 0.0, 1.0], (e, 1))[..., np.newaxis]
        sofa.EmitterPosition = source.reshape(e, 3, 1)  # (E, C, I) absolute room coords
        sofa.EmitterDescriptions = np.array(["Loudspeaker"] * e)  # (E,S)

        sofa_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write_sofa(sofa_path, sofa)  # verifies the convention on write
        return sofa_path

    @classmethod
    def _resolve(cls, *, room, array, config=None, **_ignored) -> dict:
        """Resolve the raw get() parameters into a concrete selection.

        Parses the array selection into a deduplicated, canonical-order group list
        and bundles it with the room, config, convention, and the resolved speaker
        and microphone labels. This method performs no validation: unknown group
        names are kept at the end of ``'groups'`` so that ``_validate_params`` can
        reject them, and ``config`` is passed through untouched.

        Parameters
        ----------
        room : str
            Room being loaded. Used to look up the speakers and to exclude the
            circular array from ``'all'`` for room SAL.
        array : str or list of str
            Requested group(s): ``'all'``, a comma-separated string, or a list of
            the array group names.
        config : str or None
            Microphone configuration placement in the AIL (``'P1'`` or ``'P2'``).
            Ignored for the SAL.

        Returns
        -------
        dict
            Resolved selection with keys ``'room'``, ``'config'``, ``'groups'``
            (requested groups in canonical order, unknowns kept at the end),
            ``'speakers'`` (emitter labels) and ``'mics'`` (receiver labels
            for the known groups).
        """
        # Loudspeaker labels per room, in canonical order
        speakers = {
            "SAL": ["S0_1", "S0_2", "S-30_1", "S30_1", "S-45_2", "S45_2", "S-60_1", "S60_1", "S-90_1", "S90_1"],
            "AIL": [
                "SL1",
                "SL2",
                "SL3",
                "SL4",
                "SL5",
                "SL6",
                "SL7",
                "SL8",
                "SU1",
                "SU2",
                "SU3",
                "SU4",
                "SU5",
                "SU6",
                "SU7",
                "SU8",
                "SU9",
                "SU10",
                "SU11",
                "SU12",
                "ST1",
                "ST2",
                "ST3",
                "ST4",
            ],
        }

        # get requested array groups
        if array == "all":
            groups = [g for g in cls._ARRAY_GROUPS if room != "SAL" or g != "circular-microphone-array"]
        elif isinstance(array, str):
            groups = [g.strip() for g in array.split(",")]
        else:
            groups = list(array)
        # canonical order for known groups, unknowns kept at the end for validation
        seen = set(groups)
        groups = [g for g in cls._ARRAY_GROUPS if g in seen] + [g for g in groups if g not in cls._ARRAY_GROUPS]

        return {
            "room": room,
            "config": config,
            "groups": groups,
            "speakers": speakers.get(room, []),
            "mics": [mic for g in groups if g in cls._ARRAY_GROUPS for mic in cls._ARRAY_GROUPS[g]],
        }
