"""Tests for the University of Electro-Communications 3D Meshgrid Dataset."""

import h5py as h5
import netCDF4
import numpy as np
import pytest
import sofar as sf
from typer.testing import CliRunner

import irdl
from irdl import Meshgrid3dDataset, _sofa, config
from irdl.base import BaseDataset
from irdl.cli import app
from irdl.datasets import uec
from irdl.datasets.ista import IstaBaseDataset
from irdl.utils import _atomic_write

_SAMPLING_RATE = 16000


def _provider(path, *, receivers=5, samples=17, dtype="f4"):
    """Write MATLAB-layout IRs, coordinates, and one-element scalar attributes."""
    responses = np.arange(receivers * samples, dtype=dtype).reshape(receivers, samples)
    responses[-1, -1] = -0.0
    if dtype == "f8":
        responses.flat[1] = np.nextafter(1.0, 2.0)
    grid = np.arange(receivers, dtype=np.float32)[None, :] / 5 + np.arange(3, dtype=np.float32)[:, None]
    with h5.File(path, "w") as source:
        source.create_dataset("rir", data=responses)
        source.create_dataset("grid", data=grid)
        source.attrs["fs"] = np.array([_SAMPLING_RATE], dtype=np.float32)
        source.attrs["c"] = np.array([343], dtype=np.float32)
        source.attrs["sp_pos"] = np.array([5.54, 0.94, 1.42], dtype=np.float32)
        source.attrs["room_size"] = np.array([8.4, 6.14, 2.66], dtype=np.float32)
    return responses, grid.T


@pytest.mark.parametrize("fail_first", [False, True])
def test_get_snapshots_the_global_budget_and_next_get_uses_updated_value(tmp_path, monkeypatch, fail_first):
    """Success and failure both release the starting budget for the next operation."""
    provider = tmp_path / "provider.h5"
    _provider(provider)
    initial_budget = 64
    next_budget = 4096
    monkeypatch.setattr(config, "batch_bytes", initial_budget)
    calls = 0

    def download(_self, _provider_dir, **_kwargs):
        nonlocal calls
        calls += 1
        config.batch_bytes = next_budget
        if fail_first and calls == 1:
            msg = "acquisition failed"
            raise OSError(msg)
        return provider

    monkeypatch.setattr(Meshgrid3dDataset, "_download", download)
    if fail_first:
        with pytest.raises(OSError, match="acquisition failed"):
            Meshgrid3dDataset.get(cache_dir=tmp_path / "first", output_format="sofa")
    else:
        first = Meshgrid3dDataset.get(cache_dir=tmp_path / "first", output_format="sofa")
        with h5.File(first) as sofa:
            assert np.prod(sofa["Data.IR"].chunks) * 8 <= initial_budget
    second = Meshgrid3dDataset.get(cache_dir=tmp_path / "second", output_format="sofa")
    with h5.File(second) as sofa:
        assert initial_budget < np.prod(sofa["Data.IR"].chunks) * 8 <= next_budget


def test_meshgrid_is_public_without_inheriting_ista_metadata():
    """Meshgrid reuses IR writing, not another Dataset Group's metadata."""
    assert "Meshgrid3dDataset" in irdl.__all__
    assert issubclass(Meshgrid3dDataset, BaseDataset)
    assert not issubclass(Meshgrid3dDataset, IstaBaseDataset)
    dataset = Meshgrid3dDataset()
    for speaker in range(1, 5):
        dataset._validate_params(speaker=speaker)
        assert dataset._source_filename(speaker=speaker) == f"A_ir_16k_sp{speaker}.h5"


@pytest.mark.parametrize("speaker", [0, 5, "1", 1.0, True, None])
def test_meshgrid_rejects_invalid_speaker_selectors(speaker):
    """Selectors are integers, not strings, booleans, or rounded numeric values."""
    with pytest.raises(ValueError, match="speaker"):
        Meshgrid3dDataset()._validate_params(speaker=speaker)


def test_meshgrid_acquires_the_selected_provider_file(tmp_path, monkeypatch):
    """Use the Dataset DOI and the Provider filename for the requested speaker."""
    calls = []
    pooch = object()

    def from_doi(doi, path):
        calls.append((doi, path))
        return pooch

    monkeypatch.setattr(uec, "_pooch_from_doi", from_doi)
    monkeypatch.setattr(uec, "_fetch", lambda pup, name: calls.append((pup, name)))
    assert Meshgrid3dDataset()._download(tmp_path, speaker=2) == tmp_path / "A_ir_16k_sp2.h5"
    assert calls == [(Meshgrid3dDataset.doi, tmp_path), (pooch, "A_ir_16k_sp2.h5")]


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_meshgrid_uses_shared_writer_with_its_own_geometry_and_metadata(tmp_path, monkeypatch, dtype):
    """The microphone grid becomes receivers, not moving sources or an ISTA array."""
    provider = tmp_path / "provider.h5"
    responses, grid = _provider(provider, dtype=dtype)
    provider.chmod(0o640)
    path = tmp_path / "meshgrid.sofa"
    writes = []
    original_write = uec.write_ir_chunks

    def observed_write(sofa_path, chunks):
        writes.append(sofa_path)
        return original_write(sofa_path, chunks)

    monkeypatch.setattr(uec, "write_ir_chunks", observed_write)
    dataset = Meshgrid3dDataset()
    monkeypatch.setattr(config, "batch_bytes", 2 * responses.shape[1] * 8)
    dataset._ingest(provider, path)

    assert writes == [path]
    with h5.File(path) as sofa, h5.File(provider) as source:
        ir = sofa["Data.IR"]
        assert ir.shape == (1, *responses.shape, 1)
        assert ir[0, :, :, 0].tobytes() == responses.astype("f8").tobytes()
        assert ir.chunks[0] == ir.chunks[-1] == 1
        assert np.prod(ir.chunks) * 8 <= config.batch_bytes
        np.testing.assert_array_equal(sofa["ReceiverPosition"][:, :, 0], grid)
        np.testing.assert_array_equal(sofa["SourcePosition"][:], source.attrs["sp_pos"].astype("f8")[None])
        np.testing.assert_array_equal(sofa["EmitterPosition"][:], np.zeros((1, 3, 1)))
        np.testing.assert_array_equal(sofa["ListenerPosition"][:], np.zeros((1, 3)))
        np.testing.assert_array_equal(sofa["RoomCornerB"][:], source.attrs["room_size"].astype("f8")[None])
        np.testing.assert_array_equal(sofa["RoomVolume"][:], [np.prod(source.attrs["room_size"].astype("f8"))])
        np.testing.assert_array_equal(sofa["SpeedOfSound"][:], source.attrs["c"].astype("f8").reshape(1, 1))
        assert "RoomTemperature" not in sofa
        assert "Humidity" not in sofa
        assert "MeasurementDate" not in sofa
    with netCDF4.Dataset(path) as sofa:
        assert sofa.Organization == "The University of Electro-Communications"
        assert sofa.License == "CC BY 4.0"
        assert "GRAS" not in sofa.ReceiverDescription
        assert sofa.variables["Data.SamplingRate"][0] == _SAMPLING_RATE
        assert sofa.variables["ReceiverPosition"].Type == "cartesian"
        assert sofa.variables["ReceiverPosition"].Units == "metre"
    with sf.SofaStream(path) as sofa:
        assert sofa.verify(issue_handling="return", mode="read") is None
    assert path.stat().st_mode & 0o777 == provider.stat().st_mode & 0o777


def test_meshgrid_hdf5_export_does_not_read_the_whole_microphone_grid(tmp_path, monkeypatch):
    """M=1 must not turn the HDF5 conversion into an unbounded receiver read."""
    provider = tmp_path / "provider.h5"
    expected, _ = _provider(provider)
    dataset = Meshgrid3dDataset()
    monkeypatch.setattr(config, "batch_bytes", 2 * expected.shape[1] * 8)
    sofa_path = tmp_path / "meshgrid.sofa"
    dataset._ingest(provider, sofa_path)
    sizes = []
    setitem = h5.Dataset.__setitem__

    def bounded_write(destination, selection, values):
        if destination.name == "/data/impulse_response":
            assert values.nbytes <= config.batch_bytes
            sizes.append(values.size)
        return setitem(destination, selection, values)

    monkeypatch.setattr(h5.Dataset, "__setitem__", bounded_write)
    output = tmp_path / "meshgrid.h5"
    dataset._to_hdf5(sofa_path, output)

    assert len(sizes) > 1
    with h5.File(output) as hdf5:
        np.testing.assert_array_equal(hdf5["data/impulse_response"][0], expected)


@pytest.mark.parametrize("samples", [4000, 200_000])
def test_meshgrid_write_and_verification_reads_stay_within_storage_chunks(tmp_path, monkeypatch, samples):
    """Both receiver tails and long sample tails are read without loading whole rows."""
    provider = tmp_path / "provider.h5"
    expected, grid = _provider(provider, samples=samples, dtype="f8")
    dataset = Meshgrid3dDataset()
    monkeypatch.setattr(config, "batch_bytes", 64008)
    reads = []
    getitem = h5.Dataset.__getitem__

    def bounded_read(variable, selection, **kwargs):
        values = getitem(variable, selection, **kwargs)
        if variable.name in ("/rir", "/Data.IR"):
            assert values.size * np.dtype("f8").itemsize <= config.batch_bytes
            reads.append(values.shape)
        return values

    monkeypatch.setattr(h5.Dataset, "__getitem__", bounded_read)
    path = tmp_path / "bounded.sofa"
    dataset._ingest(provider, path)

    assert reads
    with netCDF4.Dataset(path) as sofa:
        assert sofa.variables["Data.IR"][0, :, :, 0].tobytes() == expected.tobytes()
        np.testing.assert_array_equal(sofa.variables["ReceiverPosition"][:, :, 0], grid)


def test_meshgrid_verification_uses_a_smaller_budget_than_the_existing_storage_chunks(tmp_path, monkeypatch):
    """Changing the budget affects later reads without requiring a cached SOFA rewrite."""
    provider = tmp_path / "provider.h5"
    _provider(provider, dtype="f8")
    path = tmp_path / "existing.sofa"
    dataset = Meshgrid3dDataset()
    dataset._ingest(provider, path)
    budget = 64
    with h5.File(path) as sofa:
        assert np.prod(sofa["Data.IR"].chunks) * 8 > budget
    getitem = h5.Dataset.__getitem__
    reads = []

    def bounded_read(variable, selection, **kwargs):
        values = getitem(variable, selection, **kwargs)
        if variable.name in ("/rir", "/Data.IR"):
            assert values.nbytes <= budget
            reads.append(values.nbytes)
        return values

    monkeypatch.setattr(config, "batch_bytes", budget)
    monkeypatch.setattr(h5.Dataset, "__getitem__", bounded_read)
    dataset._verify_payload(path, provider)
    assert len(reads) > 2  # noqa: PLR2004 - Both source and destination need multiple reads.


@pytest.mark.parametrize(
    "field",
    [
        "Data.IR",
        "ReceiverPosition",
        "SourcePosition",
        "EmitterPosition",
        "ListenerPosition",
        "Data.SamplingRate",
        "Data.Delay",
        "ListenerView",
        "ListenerUp",
        "SourceView",
        "SourceUp",
        "SpeedOfSound",
        "RoomCornerA",
        "RoomCornerB",
        "RoomVolume",
    ],
)
def test_meshgrid_verification_rejects_even_sub_float32_changes(tmp_path, field):
    """Meshgrid's exact comparisons cover IR precision and supplied physical metadata."""
    provider = tmp_path / "provider.h5"
    _provider(provider, dtype="f8")
    path = tmp_path / "changed.sofa"
    dataset = Meshgrid3dDataset()
    dataset._ingest(provider, path)
    with h5.File(path, "r+") as sofa:
        variable = sofa[field]
        index = (0,) * variable.ndim
        variable[index] = np.nextafter(float(variable[index]), np.inf)

    with pytest.raises(ValueError, match="Meshgrid payload mismatch"):
        dataset._verify_payload(path, provider)


@pytest.mark.parametrize(
    "fault",
    [
        "rir_missing",
        "rir_rank",
        "rir_empty",
        "rir_integer",
        "rir_complex",
        "grid_transposed",
        "sp_pos_short",
        "sp_pos_nan",
        "room_size_short",
        "room_size_zero",
        "fs_missing",
        "fs_multiple",
        "fs_zero",
        "c_nan",
    ],
)
def test_meshgrid_rejects_invalid_provider_metadata_before_creating_output(tmp_path, fault):
    """Malformed matrices and physical attributes are not turned into plausible SOFA."""
    provider = tmp_path / "provider.h5"
    responses, grid = _provider(provider)
    with h5.File(provider, "a") as source:
        if fault.startswith("rir_"):
            del source["rir"]
            replacements = {
                "rir_rank": responses.ravel(),
                "rir_empty": responses[:0],
                "rir_integer": responses.astype("i8"),
                "rir_complex": responses.astype("c16"),
            }
            if fault != "rir_missing":
                source.create_dataset("rir", data=replacements[fault])
        elif fault == "grid_transposed":
            del source["grid"]
            source.create_dataset("grid", data=grid)
        elif fault == "fs_missing":
            del source.attrs["fs"]
        else:
            attributes = {
                "sp_pos_short": ("sp_pos", [1, 2]),
                "sp_pos_nan": ("sp_pos", [1, np.nan, 3]),
                "room_size_short": ("room_size", [1, 2]),
                "room_size_zero": ("room_size", [1, 0, 3]),
                "fs_multiple": ("fs", [16000, 48000]),
                "fs_zero": ("fs", [0]),
                "c_nan": ("c", [np.nan]),
            }
            name, value = attributes[fault]
            source.attrs[name] = value
    path = tmp_path / "invalid.sofa"

    with pytest.raises((ValueError, KeyError)):
        Meshgrid3dDataset()._ingest(provider, path)
    assert not path.exists()


@pytest.mark.parametrize("output_format", ["sofa", "hdf5", "numpy", "pyfar", "raw"])
def test_meshgrid_public_get_supports_all_formats_and_retains_canonical_sofa(tmp_path, monkeypatch, output_format):
    """The normal Get lifecycle works without Dataset-specific cache or export handling."""
    provider = tmp_path / "provider.h5"
    responses, grid = _provider(provider)
    calls = []

    def acquire(_self, _provider_dir, **kwargs):
        calls.append(kwargs["speaker"])
        return provider

    monkeypatch.setattr(Meshgrid3dDataset, "_download", acquire)
    result = Meshgrid3dDataset.get(speaker=2, cache_dir=tmp_path, output_format=output_format)
    canonical = tmp_path / "3DMESHGRID" / "output" / "A_ir_16k_sp2.sofa"
    assert calls == [2]
    if output_format == "raw":
        assert result == provider
        assert not canonical.exists()
        return
    assert canonical.is_file()
    if output_format == "sofa":
        assert result == canonical
    elif output_format == "hdf5":
        with h5.File(result) as output:
            np.testing.assert_array_equal(output["data/impulse_response"][0], responses)
            np.testing.assert_array_equal(output["data/location/receiver"][:], grid)
    elif output_format == "numpy":
        assert result["impulse_response"].shape == (1, *responses.shape, 1)
        np.testing.assert_array_equal(result["impulse_response"][0, :, :, 0], responses)
        np.testing.assert_array_equal(result["receiver_coordinates"][:, :, 0], grid)
        assert result["sampling_rate"] == _SAMPLING_RATE
    else:
        np.testing.assert_array_equal(result["impulse_response"].time[0, 0], responses)
        np.testing.assert_array_equal(result["receiver_coordinates"].cartesian[:, 0], grid)
    with h5.File(provider) as source:
        wanted_source = source.attrs["sp_pos"].astype("f8")[None]
    if isinstance(result, dict):
        actual_source = result["source_coordinates"]
        if output_format == "pyfar":
            actual_source = actual_source.cartesian
        np.testing.assert_array_equal(actual_source, wanted_source)


def test_meshgrid_cli_is_registered():
    """CLI discovery picks up the new public Dataset and its speaker selector."""
    result = CliRunner().invoke(app, ["get", "3dmeshgrid", "--help"])
    assert result.exit_code == 0
    assert "--speaker" in result.stdout


def test_meshgrid_reuses_canonical_sofa_after_source_files_are_removed(tmp_path, monkeypatch):
    """In-memory retrieval can reuse canonical SOFA without its lower Cache Stages."""
    provider = tmp_path / "provider.h5"
    responses, _ = _provider(provider)
    monkeypatch.setattr(Meshgrid3dDataset, "_download", lambda *_args, **_kwargs: provider)
    Meshgrid3dDataset.get(cache_dir=tmp_path, output_format="sofa")
    provider.unlink()
    ingest = tmp_path / "3DMESHGRID" / "ingest" / "A_ir_16k_sp1.h5"
    ingest.unlink()

    def fail_acquisition(*_args, **_kwargs):
        pytest.fail("A retained SOFA artifact should not require Provider files")

    monkeypatch.setattr(Meshgrid3dDataset, "_download", fail_acquisition)
    result = Meshgrid3dDataset.get(cache_dir=tmp_path, output_format="numpy")
    np.testing.assert_array_equal(result["impulse_response"][0, :, :, 0], responses)


@pytest.mark.parametrize("failure", ["source", "compression", "write", "payload", "convention"])
def test_meshgrid_get_discards_failed_output(tmp_path, monkeypatch, failure):
    """Shared writing integrates with BaseDataset staging; no second promotion is needed."""
    provider = tmp_path / "provider.h5"
    _provider(provider)
    monkeypatch.setattr(Meshgrid3dDataset, "_download", lambda *_args, **_kwargs: provider)

    def fail(*_args, **_kwargs):
        msg = "injected failure"
        raise OSError(msg)

    chunks = uec._meshgrid_ir_chunks

    def failed_source(*args):
        yield next(chunks(*args))
        fail()

    if failure == "source":
        monkeypatch.setattr(uec, "_meshgrid_ir_chunks", failed_source)
    elif failure == "compression":
        monkeypatch.setattr(_sofa, "_encode_ir_chunk", fail)
    elif failure == "write":
        monkeypatch.setattr(uec, "write_ir_chunks", fail)
    elif failure == "payload":
        monkeypatch.setattr(Meshgrid3dDataset, "_verify_payload", fail)
    elif failure == "convention":
        monkeypatch.setattr(Meshgrid3dDataset, "_verify_sofa_convention", fail)

    with pytest.raises(OSError, match="injected failure"):
        Meshgrid3dDataset.get(cache_dir=tmp_path, output_format="sofa")
    assert list((tmp_path / "3DMESHGRID" / "output").iterdir()) == []
    with h5.File(provider, "r+") as source:
        source.attrs["closed_after_failure"] = True


def test_meshgrid_failed_rewrite_keeps_existing_artifact(tmp_path, monkeypatch):
    """A failed writer cannot replace an existing retained artifact."""
    provider = tmp_path / "provider.h5"
    _provider(provider)
    path = tmp_path / "retained.sofa"
    dataset = Meshgrid3dDataset()
    dataset._ingest(provider, path)
    original = path.read_bytes()

    def fail(*_args, **_kwargs):
        msg = "injected compression failure"
        raise OSError(msg)

    monkeypatch.setattr(_sofa, "_encode_ir_chunk", fail)
    with pytest.raises(OSError, match="injected compression failure"), _atomic_write(path) as temporary:
        dataset._ingest(provider, temporary)
    assert path.read_bytes() == original
    assert not path.with_name(f".tmp-{path.name}").exists()
