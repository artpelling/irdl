"""Tests for BaseDataset abstract base class."""

from pathlib import Path

import pytest

from irdl.base import BaseDataset
from irdl.ista import IstaBaseDataset


class TestBaseDatasetAbstract:
    """Tests for BaseDataset abstract class behavior."""

    def test_cannot_instantiate_basedataset(self):
        """Verify BaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDataset()

    def test_abstract_methods_defined(self):
        """Verify BaseDataset has expected abstract methods."""
        expected_abstract = {
            "_source_filename",
            "_download",
            "_ingest",
            "_validate_params",
        }
        assert BaseDataset.__abstractmethods__ == expected_abstract


class TestBaseDatasetHelpers:
    """Tests for shared BaseDataset helper behavior."""

    def test_output_path_accepts_source_filename_string(self, tmp_path):
        """Verify file-based output paths work when _source_filename returns a string."""

        class DummyDataset(BaseDataset):
            name = "dummy"
            doi = "10.0000/dummy"
            canonical_provider = "depositonce"
            providers = ("depositonce",)

            def _validate_params(self, **dataset_kwargs):
                pass

            def _provider_artifact_format(self, provider: str, **_dataset_kwargs):
                assert provider == "depositonce"
                return "sofa"

            def _download(self, provider_dir: Path, provider: str, **_dataset_kwargs):
                assert provider == "depositonce"
                return provider_dir / "dummy.sofa"

            def _ingest(self, ingest_path: Path):
                raise NotImplementedError

            def _source_filename(self, **_dataset_kwargs):
                return "dummy.sofa"

        dataset = DummyDataset()

        output_dir = tmp_path / "out"

        assert dataset._output_path(output_dir, "dummy.sofa", "sofa") == output_dir / "dummy.sofa"
        assert dataset._output_path(output_dir, "dummy.sofa", "hdf5") == output_dir / "dummy.h5"

    def test_process_passes_dataset_kwargs_to_subclass(self, tmp_path):
        """Verify the process wrapper forwards dataset-specific parameters to _process."""

        class DummyDataset(BaseDataset):
            name = "dummy"
            doi = "10.0000/dummy"
            canonical_provider = "depositonce"
            providers = ("depositonce",)

            def _validate_params(self, **dataset_kwargs):
                pass

            def _provider_artifact_format(self, provider: str, **_dataset_kwargs):
                assert provider == "depositonce"
                return "bin"

            def _download(self, provider_dir: Path, provider: str, **_dataset_kwargs):
                assert provider == "depositonce"
                return provider_dir / "dummy.bin"

            def _ingest(self, ingest_path: Path):
                raise NotImplementedError

            def _source_filename(self, **_dataset_kwargs):
                return "dummy.bin"

            def _process(self, _provider_artifact: Path, ingest_path: Path, **dataset_kwargs):
                assert dataset_kwargs == {"provider": "depositonce", "scenario": "demo"}
                ingest_path.write_text("ok")
                return ingest_path

        dataset = DummyDataset()
        provider_artifact = tmp_path / "provider.bin"
        provider_artifact.write_text("provider")
        ingest_path = tmp_path / "ingest" / "dummy.bin"

        assert dataset.process(provider_artifact, ingest_path, provider="depositonce", scenario="demo") == ingest_path
        assert ingest_path.exists()


class TestProviderSelection:
    """Tests for provider selection behavior."""

    def test_raw_rejects_non_canonical_provider(self):
        """Verify raw output is restricted to the canonical provider."""

        class DummyDataset(BaseDataset):
            name = "dummy"
            doi = "10.0000/dummy"
            canonical_provider = "depositonce"
            providers = ("depositonce", "sonicom")

            def _validate_params(self, **dataset_kwargs):
                pass

            def _provider_artifact_format(self, provider: str, **_dataset_kwargs):
                return "sofa" if provider == "sonicom" else "hdf5"

            def _download(self, provider_dir: Path, provider: str, **_dataset_kwargs):
                raise NotImplementedError

            def _ingest(self, ingest_path: Path):
                raise NotImplementedError

            def _source_filename(self, **_dataset_kwargs):
                return "dummy.sofa"

        with pytest.raises(ValueError, match="raw output_format is only supported"):
            DummyDataset()._get(cache_dir=None, export_dir=None, output_format="raw", provider="sonicom")

    def test_auto_prefers_direct_provider_and_logs_choice(self, monkeypatch, tmp_path):
        """Verify auto provider selection prefers direct providers before convertible ones."""

        class DummyDataset(BaseDataset):
            name = "dummy"
            doi = "10.0000/dummy"
            canonical_provider = "depositonce"
            providers = ("depositonce", "sonicom")

            def _validate_params(self, **dataset_kwargs):
                pass

            def _provider_artifact_format(self, provider: str, **_dataset_kwargs):
                return "sofa" if provider == "sonicom" else "hdf5"

            def _download(self, provider_dir: Path, provider: str, **_dataset_kwargs):
                target = provider_dir / ("dummy.sofa" if provider == "sonicom" else "dummy.h5")
                target.parent.mkdir(parents=True, exist_ok=True)
                if provider == "sonicom":
                    target.write_text("sonicom-sofa")
                else:
                    target.write_text("canonical")
                return target

            def _ingest(self, ingest_path: Path):
                raise NotImplementedError

            def _source_filename(self, **_dataset_kwargs):
                return "dummy.sofa"

        dataset = DummyDataset()
        messages: list[str] = []

        def capture_info(message, *args):
            messages.append(message % args if args else message)

        monkeypatch.setattr(dataset.logger, "info", capture_info)

        result = dataset._get(
            cache_dir=tmp_path,
            export_dir=None,
            output_format="sofa",
            provider="auto",
        )

        assert result == tmp_path / "DUMMY" / "output" / "dummy.sofa"
        expected = "provider='sonicom' requested='auto' output_format='sofa' -> provider-native path"
        assert any(expected in message for message in messages)
        assert not any("provider='depositonce'" in message for message in messages)

    def test_explicit_provider_reuses_existing_output_without_download(self, tmp_path):
        """Verify explicit non-raw provider requests reuse cached output before download."""

        class DummyDataset(BaseDataset):
            name = "dummy"
            doi = "10.0000/dummy"
            canonical_provider = "depositonce"
            providers = ("depositonce", "sonicom")

            def _validate_params(self, **dataset_kwargs):
                pass

            def _provider_artifact_format(self, provider: str, **_dataset_kwargs):
                return "sofa" if provider == "sonicom" else "zip"

            def _download(self, provider_dir: Path, provider: str, **_dataset_kwargs):
                if provider == "depositonce":
                    msg = "depositonce download should not run when output cache exists"
                    raise AssertionError(msg)
                target = provider_dir / "dummy.sofa"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("sonicom-sofa")
                return target

            def _ingest(self, ingest_path: Path):
                raise NotImplementedError

            def _source_filename(self, **_dataset_kwargs):
                return "dummy.sofa"

        dataset = DummyDataset()
        output_path = tmp_path / "DUMMY" / "output" / "dummy.h5"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("cached-output")

        result = dataset._get(
            cache_dir=tmp_path,
            export_dir=None,
            output_format="hdf5",
            provider="depositonce",
        )

        assert result == output_path


class TestIstaBaseDatasetAbstract:
    """Tests for IstaBaseDataset abstract class behavior."""

    def test_cannot_instantiate_istabasedataset(self):
        """Verify IstaBaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IstaBaseDataset()

    def test_inherits_abstract_methods(self):
        """Verify IstaBaseDataset inherits and adds abstract methods."""
        # IstaBaseDataset implements _source_filename and _ingest,
        # so only _download and _validate_params are abstract
        expected_abstract = {
            "_download",
            "_validate_params",
        }
        assert IstaBaseDataset.__abstractmethods__ == expected_abstract
