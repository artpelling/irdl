"""Offline logging contracts shared by all Dataset Get implementations."""

from inspect import signature
from pathlib import Path

import h5py as h5
import numpy as np
import pytest
import sofar as sf
from typer.testing import CliRunner

import irdl
from irdl.base import BaseDataset, _get_dataset_classes
from irdl.cli import app
from irdl.datasets.ista import MiracleDataset
from irdl.utils import _link_or_copy


@pytest.mark.parametrize("dataset_class", _get_dataset_classes(irdl), ids=lambda cls: cls.name)
@pytest.mark.parametrize("output_format", ["raw", "sofa", "hdf5", "numpy", "pyfar"])
def test_all_datasets_report_context_and_one_ready_outcome(
    tmp_path, monkeypatch, sofa_object, caplog, dataset_class, output_format
):
    """Provider fixtures isolate the shared lifecycle without network requests."""
    provider = tmp_path / "provider.sofa"
    sf.write_sofa(provider, sofa_object)
    monkeypatch.setattr(dataset_class, "_direct_sofa", lambda *_args: (None, None))
    monkeypatch.setattr(dataset_class, "_download", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(dataset_class, "_process", BaseDataset._process)
    monkeypatch.setattr(
        dataset_class, "_ingest", lambda _self, artifact, destination, **_kwargs: _link_or_copy(artifact, destination)
    )
    kwargs = {
        key: parameter.default
        for key, parameter in signature(dataset_class.get).parameters.items()
        if key not in {"cache_dir", "export_dir", "output_format"}
    }
    caplog.clear()
    result = dataset_class.get(cache_dir=tmp_path / "cache", output_format=output_format, **kwargs)
    records = [record for record in caplog.records if record.name.startswith("irdl")]
    assert records[0].getMessage().startswith("Get started:")
    assert records[-1].getMessage().startswith(f"Ready ({output_format}):")
    assert sum(record.getMessage().startswith("Ready (") for record in records) == 1
    assert all(record.dataset == dataset_class.name.upper() and record.selection for record in records)
    assert ".tmp-" not in records[-1].getMessage()
    if isinstance(result, Path):
        assert result.is_file()
    if output_format in {"sofa", "hdf5"}:
        caplog.clear()
        assert dataset_class.get(cache_dir=tmp_path / "cache", output_format=output_format, **kwargs) == result
        assert any(message.startswith("Output cache hit") for message in caplog.messages)
        assert caplog.messages[-1].startswith(f"Ready ({output_format}):")


def test_cli_stdout_contains_only_the_requested_result(tmp_path):
    """A real cached Get remains usable in shell substitutions and pipelines."""
    output = tmp_path / "3DMESHGRID" / "output" / "A_ir_16k_sp1.h5"
    output.parent.mkdir(parents=True)
    output.touch()
    result = CliRunner().invoke(app, ["get", "3dmeshgrid", "--cache-dir", str(tmp_path), "--output-format", "hdf5"])
    assert result.exit_code == 0
    assert result.stdout == f"{output}\n"
    assert "Output cache hit" in result.stderr
    assert "3DMESHGRID" in result.stderr


def test_failed_final_validation_never_announces_ready(tmp_path, monkeypatch, caplog):
    """An actual ISTA Ingest must not announce its temporary file as finished."""
    provider = tmp_path / "provider.h5"
    with h5.File(provider, "w") as source:
        source.create_dataset("data/impulse_response", data=np.zeros((2, 2, 4), dtype="f4"))
        source.create_dataset("data/location/source", data=np.zeros((2, 3)))
        source.create_dataset("data/location/receiver", data=np.zeros((2, 3)))
        source.create_dataset("metadata/sampling_rate", data=48000)
        source.create_dataset("metadata/temperature", data=np.full(2, 20.0))
        source.create_dataset("metadata/c0", data=np.full(2, 343.0))

    def fail(*_args, **_kwargs):
        msg = "final convention failure"
        raise ValueError(msg)

    monkeypatch.setattr(MiracleDataset, "_direct_sofa", lambda *_args: (None, None))
    monkeypatch.setattr(MiracleDataset, "_download", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(MiracleDataset, "_verify_sofa_convention", fail)
    with pytest.raises(ValueError, match="final convention failure"):
        MiracleDataset.get(scenario="A1", cache_dir=tmp_path / "cache", output_format="sofa")
    assert not any(message.startswith(("Ready (", "Finished SOFA")) for message in caplog.messages)
    assert sum(message.startswith("Get failed:") for message in caplog.messages) == 1
    assert provider.exists()
    assert not list((tmp_path / "cache" / "MIRACLE" / "output").iterdir())


def test_ingest_failure_keeps_exception_type_and_reports_artifact_paths(tmp_path, monkeypatch, sofa_object, caplog):
    """The public error identifies both artifacts without replacing the original exception."""
    provider = tmp_path / "provider.sofa"
    sf.write_sofa(provider, sofa_object)
    failure = OSError("cannot read Data.IR")

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(MiracleDataset, "_direct_sofa", lambda *_args: (None, None))
    monkeypatch.setattr(MiracleDataset, "_download", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(MiracleDataset, "_ingest", fail)
    with pytest.raises(OSError, match=r"cannot read Data\.IR") as caught:
        MiracleDataset.get(scenario="A1", cache_dir=tmp_path / "cache", output_format="sofa")
    assert caught.value is failure
    ingest = tmp_path / "cache" / "MIRACLE" / "ingest" / "A1.h5"
    assert str(ingest) in caplog.messages[-1]
    assert ".tmp-A1.sofa" in caplog.messages[-1]
    assert provider.exists()
    assert ingest.exists()


def test_direct_sofa_route_is_explicit_and_ready_follows_validation(tmp_path, monkeypatch, sofa_object, caplog):
    """A direct Provider follows the same completion contract without fake Ingest work."""
    provider = tmp_path / "direct.sofa"
    sf.write_sofa(provider, sofa_object)
    monkeypatch.setattr(MiracleDataset, "_direct_sofa", lambda *_args: ("https://example.invalid/A1.sofa", None))
    monkeypatch.setattr(MiracleDataset, "_download_direct_sofa", lambda *_args: provider)
    result = MiracleDataset.get(scenario="A1", cache_dir=tmp_path / "cache", output_format="sofa")
    assert result.exists()
    assert "Provider: direct SOFA from example.invalid" in caplog.messages
    assert not any(message.startswith("Ingest:") for message in caplog.messages)
    assert caplog.messages[-1] == f"Ready (sofa): {result}"
