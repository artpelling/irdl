"""Tests for CLI wrapper behavior."""

from inspect import signature
from pathlib import Path

import pytest
import typer
from numpydoc.docscrape import FunctionDoc
from typer.main import get_command
from typer.testing import CliRunner

import irdl
from irdl.base import _get_dataset_classes
from irdl.cli import _make_wrapper, app

RESULT_DIR = Path("tests/_tmp_cli")


@pytest.mark.parametrize("dataset_class", _get_dataset_classes(irdl), ids=lambda cls: cls.name)
def test_dataset_help_documents_optional_parameters_and_defaults(dataset_class):
    """Document defaults explicitly only when Typer does not display them."""
    param_docs = {param.name: param for param in FunctionDoc(dataset_class.get)["Parameters"]}
    command = get_command(app).commands["get"].commands[dataset_class.name]
    options = {option.name: option for option in command.params}
    ctx = typer.Context(command)

    for name, param in signature(dataset_class.get).parameters.items():
        doc = param_docs[name]
        description = " ".join(doc.desc).replace("`", "")
        _, help_text = options[name].get_help_record(ctx)
        assert "optional" in doc.type
        if "[default:" in help_text:
            assert "Default is " not in description
        else:
            assert f"Default is {param.default!r}" in description

    result = CliRunner().invoke(app, ["get", dataset_class.name, "--help"])
    assert result.exit_code == 0, result.output


def test_list_displays_consistent_dataset_descriptions():
    """Keep verified expansions and leave other Dataset names unchanged."""
    expected_names = {
        "3dmeshgrid": "3D Meshgrid Room Impulse Response dataset",
        "ari": "ARI dataset",
        # Original paper: https://doi.org/10.71568/daga2026.367
        "bras-rs8": "Benchmark for Room Acoustical Simulation, Reference Scene 8 (BRAS-RS8) dataset",
        "cipic": "CIPIC dataset",
        "dechorate": "dEchorate dataset",
        "fabian": "FABIAN dataset",
        "hutubs": "HUTUBS dataset",
        # Original paper: https://doi.org/10.1186/s13636-024-00352-8
        "miracle": "Microphone Array Impulse Response Dataset for Acoustic Learning (MIRACLE)",
        # User-confirmed expansion.
        "mird": "Multi-Channel Impulse Response Database (MIRD)",
        # Paper title checked; full text unavailable: https://doi.org/10.1109/IWAENC61483.2024.10694356
        "mrtd": "Multi-Room Transition Dataset (MRTD)",
        # Original paper: https://doi.org/10.1186/s13636-023-00284-9
        "myriad": "Multi-Array Room Acoustic Database (MYRiAD)",
        # Original paper, section 1: https://doi.org/10.3390/app8112029
        "sadie": "Spatial Audio for Domestic Interactive Entertainment (SADIE II) dataset",
        # Original manuscript glossary: https://doi.org/10.1051/aacus/2026094
        "sriracha": "Shoebox Room Impulse Response Archive with Varying Absorption (SRIRACHA)",
    }
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0, result.output

    for dataset_class in _get_dataset_classes(irdl):
        summary = " ".join(dataset_class.__doc__.strip().split("\n\n", 1)[0].split())
        assert summary.startswith(f"Download the {expected_names[dataset_class.name]} from ")
        assert summary.endswith(".")
        assert " ".join(FunctionDoc(dataset_class.get)["Summary"]) == summary
        description, provider = summary.rsplit(" from ", 1)
        assert (
            f"    {description}.\n"
            f"    Provider: {provider.removesuffix('.')}\n"
            f"    DOI: https://doi.org/{dataset_class.doi}"
        ) in result.output


class DummyDataset:
    """Concrete stand-in for CLI wrapper tests."""

    name = "dummy"

    @classmethod
    def get(cls, output_format: str = "pyfar"):
        """Return representative values for different output formats."""
        if output_format in {"raw", "sofa", "hdf5"}:
            suffix = "h5" if output_format == "hdf5" else "sofa"
            return RESULT_DIR / f"result.{suffix}"
        return {"output_format": output_format}


def test_make_wrapper_echoes_returned_path(monkeypatch):
    """Verify file-based CLI results are printed to stdout."""
    echoed: list[object] = []
    monkeypatch.setattr(typer, "echo", echoed.append)

    wrapper = _make_wrapper(
        DummyDataset,
        DummyDataset.get,
        signature(DummyDataset.get.__func__).parameters,
        "help",
        DummyDataset.name,
        {},
    )

    result = wrapper(output_format="sofa")

    assert result == RESULT_DIR / "result.sofa"
    # typer.style adds ANSI color codes (\x1b[96m = bright cyan, \x1b[0m = reset)
    assert echoed == [f"\x1b[96m{RESULT_DIR / 'result.sofa'}\x1b[0m"]


def test_make_wrapper_echoes_formatted_dict(monkeypatch):
    """Verify in-memory CLI results are rendered as formatted text."""
    echoed: list[object] = []
    monkeypatch.setattr(typer, "echo", echoed.append)

    wrapper = _make_wrapper(
        DummyDataset,
        DummyDataset.get,
        signature(DummyDataset.get.__func__).parameters,
        "help",
        DummyDataset.name,
        {},
    )

    result = wrapper(output_format="numpy")

    assert result == {"output_format": "numpy"}
    # typer.style adds ANSI color codes (\x1b[96m = bright cyan, \x1b[0m = reset)
    assert echoed == ["\x1b[96moutput_format:\n  numpy\x1b[0m"]
