"""Tests for the one package-wide streaming batch setting."""

import runpy

import pytest

import irdl
from irdl import Meshgrid3dDataset, config


def test_config_is_public():
    """Python callers can set the same configuration used by every streamed Dataset."""
    assert irdl.config is config
    assert "config" in irdl.__all__


@pytest.mark.parametrize("budget", [None, True, False, 0, -1, 7, 8.0, "64"])
def test_invalid_budgets_fail_before_acquisition_or_cache_changes(tmp_path, monkeypatch, budget):
    """Invalid global settings must not start acquisition or create partial artifacts."""
    monkeypatch.setattr(config, "batch_bytes", budget)
    monkeypatch.setattr(Meshgrid3dDataset, "_download", lambda *_args, **_kwargs: pytest.fail("acquisition started"))
    with pytest.raises(ValueError, match=r"config\.batch_bytes"):
        Meshgrid3dDataset.get(cache_dir=tmp_path, output_format="sofa")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mib", [None, "2"])
def test_environment_initializes_the_same_byte_setting(monkeypatch, mib):
    """CLI processes can initialize the global setting without Dataset-specific options."""
    if mib is None:
        monkeypatch.delenv("IRDL_BATCH_MIB", raising=False)
    else:
        monkeypatch.setenv("IRDL_BATCH_MIB", mib)
    settings = runpy.run_path(config.__file__)
    assert settings["batch_bytes"] == (64 if mib is None else int(mib)) * 1024**2


@pytest.mark.parametrize("mib", ["0", "-1", "1.5", "many"])
def test_invalid_environment_values_have_a_clear_error(monkeypatch, mib):
    """An invalid environment default names the setting rather than failing during I/O."""
    monkeypatch.setenv("IRDL_BATCH_MIB", mib)
    with pytest.raises(ValueError, match="IRDL_BATCH_MIB"):
        runpy.run_path(config.__file__)
