"""Tests for BaseDataset abstract base class."""

from pathlib import Path
from zipfile import ZipFile

import pytest
import sofar as sf

from irdl.base import BaseDataset
from irdl.datasets.akt import AKTZipBaseDataset
from irdl.datasets.ista import IstaBaseDataset
from irdl.utils import _validate_hash_registry


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
            "_validate_params",
        }
        assert BaseDataset.__abstractmethods__ == expected_abstract

    def test_dataset_requires_doi(self):
        """Reject concrete Dataset definitions without a DOI."""
        with pytest.raises(TypeError, match="must define a DOI"):

            class MissingDoiDataset(BaseDataset):
                name = "missing-doi"


class TestBaseDatasetHelpers:
    """Tests for shared BaseDataset helper behavior."""

    def test_output_path_accepts_source_filename_string(self, tmp_path):
        """Verify file-based output paths work when _source_filename returns a string."""

        class DummyDataset(BaseDataset):
            name = "dummy"
            doi = "10.0000/dummy"

            def _validate_params(self, **dataset_kwargs):
                pass

            def _download(self, provider_dir: Path, **_dataset_kwargs):
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

            def _validate_params(self, **dataset_kwargs):
                pass

            def _download(self, provider_dir: Path, **_dataset_kwargs):
                return provider_dir / "dummy.bin"

            def _ingest(self, ingest_path: Path):
                raise NotImplementedError

            def _source_filename(self, **_dataset_kwargs):
                return "dummy.bin"

            def _process(self, _provider_artifact: Path, ingest_path: Path, **dataset_kwargs):
                assert dataset_kwargs == {"scenario": "demo"}
                ingest_path.write_text("ok")
                return ingest_path

        dataset = DummyDataset()
        provider_artifact = tmp_path / "provider.bin"
        provider_artifact.write_text("provider")
        ingest_path = tmp_path / "ingest" / "dummy.bin"

        assert dataset.process(provider_artifact, ingest_path, scenario="demo") == ingest_path
        assert ingest_path.exists()


class TestDirectSofaProvider:
    """Tests for optional direct-SOFA retrieval."""

    @pytest.mark.parametrize("expected_hash", ["sha256:known", None])
    def test_non_raw_uses_direct_sofa_without_ingest(self, monkeypatch, sofa_object, tmp_path, expected_hash):
        """Direct SOFA bypasses the DOI download and ingest stage."""

        class DirectSofaDataset(BaseDataset):
            name = "direct"
            doi = "10.0000/direct"

            def _validate_params(self, **_dataset_kwargs):
                pass

            def _source_filename(self, **_dataset_kwargs):
                return "canonical.h5"

            def _download(self, _provider_dir: Path, **_dataset_kwargs):
                msg = "DOI download must only serve raw requests"
                raise AssertionError(msg)

            def _direct_sofa(self, source_filename: str):
                assert source_filename == "canonical.h5"
                return "https://example.invalid/alternate.sofa", expected_hash

        dataset = DirectSofaDataset()

        def download_direct(provider_dir: Path, url: str, known_hash: str | None) -> Path:
            assert url == "https://example.invalid/alternate.sofa"
            assert known_hash == expected_hash
            provider_dir.mkdir(parents=True, exist_ok=True)
            path = provider_dir / "alternate.sofa"
            sf.write_sofa(path, sofa_object)
            return path

        monkeypatch.setattr(dataset, "_download_direct_sofa", download_direct)

        result = dataset._get(cache_dir=tmp_path, export_dir=None, output_format="sofa")

        assert result == tmp_path / "DIRECT" / "output" / "canonical.sofa"
        assert result.exists()
        assert (tmp_path / "DIRECT" / "provider" / "alternate.sofa").exists()
        assert not (tmp_path / "DIRECT" / "ingest").exists()

    @pytest.mark.parametrize("endpoint", ["download", "content"])
    def test_generic_downloader_still_requires_named_sofa_urls(self, tmp_path, endpoint):
        """Filename-free provider URLs remain the Dataset subclass's responsibility."""
        with pytest.raises(ValueError, match=r"must name a \.sofa file"):
            BaseDataset._download_direct_sofa(None, tmp_path, f"https://example.test/{endpoint}", "md5:known")

    def test_hash_registry_supports_pinned_sha256(self):
        """Keep packaged content pins separate from API-provided checksums."""
        _validate_hash_registry("test", {"example.sofa": "sha256:" + "a" * 64})

    @pytest.mark.parametrize("digest", ["md5:" + "a" * 32, "unknown:checksum"])
    def test_hash_registry_rejects_unsupported_digest(self, digest):
        """Packaged registries retain SHA256-only pins; API checksums bypass them."""
        with pytest.raises(ValueError, match="invalid digest"):
            _validate_hash_registry("test", {"example.sofa": digest})

    def test_non_raw_logs_canonical_fallback(self, caplog, sofa_object, tmp_path):
        """Log when no direct SOFA is available."""

        class CanonicalSofaDataset(BaseDataset):
            name = "canonical"
            doi = "10.0000/canonical"

            def _validate_params(self, **_dataset_kwargs):
                pass

            def _source_filename(self, **_dataset_kwargs):
                return "canonical.sofa"

            def _download(self, provider_dir: Path, **_dataset_kwargs):
                path = provider_dir / "canonical.sofa"
                sf.write_sofa(path, sofa_object)
                return path

        CanonicalSofaDataset()._get(cache_dir=tmp_path, export_dir=None, output_format="sofa")

        assert any("No direct SOFA available for canonical.sofa" in message for message in caplog.messages)


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


def _make_ingest_dataset(sofa_object, ingest_calls: list, *, fail_ingest: bool = False):
    """Create a Dataset whose _ingest writes sofa_object, or fails after a partial write."""

    class IngestDataset(BaseDataset):
        name = "atomic"
        doi = "10.0000/atomic"

        def _validate_params(self, **_dataset_kwargs):
            pass

        def _source_filename(self, **_dataset_kwargs):
            return "atomic.bin"

        def _download(self, provider_dir: Path, **_dataset_kwargs):
            path = provider_dir / "atomic.bin"
            path.write_text("provider")
            return path

        def _ingest(self, _ingest_artifact: Path, sofa_path: Path, **_dataset_kwargs):
            ingest_calls.append(sofa_path)
            if fail_ingest:
                sofa_path.write_bytes(b"partial")
                msg = "interrupted"
                raise RuntimeError(msg)
            sf.write_sofa(sofa_path, sofa_object)
            return sofa_path

    return IngestDataset()


class TestAtomicCacheWrites:
    """Failed or interrupted writes must never leave a trusted cache entry behind."""

    def test_failed_ingest_leaves_no_sofa_and_is_retried(self, sofa_object, tmp_path):
        """A partial SOFA from a failed ingest is discarded and the next Get rebuilds it."""
        calls = []
        sofa_path = tmp_path / "ATOMIC" / "output" / "atomic.sofa"

        with pytest.raises(RuntimeError, match="interrupted"):
            _make_ingest_dataset(sofa_object, calls, fail_ingest=True)._get(
                cache_dir=tmp_path, export_dir=None, output_format="sofa"
            )
        assert not sofa_path.exists()
        assert list(sofa_path.parent.iterdir()) == []

        dataset = _make_ingest_dataset(sofa_object, calls)
        result = dataset._get(cache_dir=tmp_path, export_dir=None, output_format="sofa")
        assert result == sofa_path
        assert sofa_path.exists()
        assert [path.name for path in calls] == [".tmp-atomic.sofa"] * 2

    def test_failed_verification_leaves_no_sofa(self, monkeypatch, sofa_object, tmp_path):
        """A completely written SOFA that fails verification is not promoted."""
        dataset = _make_ingest_dataset(sofa_object, [])

        def reject(_sofa_path):
            msg = "invalid"
            raise ValueError(msg)

        monkeypatch.setattr(dataset, "_verify_sofa_convention", reject)

        with pytest.raises(ValueError, match="invalid"):
            dataset._get(cache_dir=tmp_path, export_dir=None, output_format="sofa")
        assert list((tmp_path / "ATOMIC" / "output").iterdir()) == []

    def test_stale_tmp_file_is_replaced(self, sofa_object, tmp_path):
        """A leftover temporary file from a killed run does not block the next build."""
        output_dir = tmp_path / "ATOMIC" / "output"
        output_dir.mkdir(parents=True)
        (output_dir / ".tmp-atomic.sofa").write_bytes(b"stale")

        result = _make_ingest_dataset(sofa_object, [])._get(cache_dir=tmp_path, export_dir=None, output_format="sofa")

        assert result.exists()
        assert sorted(path.name for path in output_dir.iterdir()) == ["atomic.sofa"]

    def test_failed_hdf5_conversion_leaves_no_output(self, monkeypatch, sofa_object, tmp_path):
        """An HDF5 conversion that fails midway leaves no .h5 file behind."""
        dataset = _make_ingest_dataset(sofa_object, [])
        sofa_path = tmp_path / "atomic.sofa"
        sf.write_sofa(sofa_path, sofa_object)
        output_path = tmp_path / "out" / "atomic.h5"

        def fail(*_args, **_kwargs):
            msg = "interrupted"
            raise RuntimeError(msg)

        # np.squeeze runs after the impulse responses have been written
        monkeypatch.setattr("irdl.base.np.squeeze", fail)

        with pytest.raises(RuntimeError, match="interrupted"):
            dataset._to_hdf5(sofa_path, output_path)
        assert list(output_path.parent.iterdir()) == []

    def test_failed_process_leaves_no_ingest_artifact(self, tmp_path):
        """A partial ingest artifact from a failed _process is discarded."""

        class FailingProcessDataset(BaseDataset):
            name = "dummy"
            doi = "10.0000/dummy"

            def _validate_params(self, **_dataset_kwargs):
                pass

            def _download(self, provider_dir: Path, **_dataset_kwargs):
                return provider_dir / "dummy.bin"

            def _source_filename(self, **_dataset_kwargs):
                return "dummy"

            def _process(self, _provider_artifact: Path, ingest_path: Path, **_dataset_kwargs):
                ingest_path.mkdir()
                (ingest_path / "first.wav").write_bytes(b"partial")
                msg = "interrupted"
                raise RuntimeError(msg)

        ingest_path = tmp_path / "ingest" / "dummy"
        with pytest.raises(RuntimeError, match="interrupted"):
            FailingProcessDataset().process(tmp_path / "provider.bin", ingest_path)
        assert list(ingest_path.parent.iterdir()) == []

    def test_process_passes_through_in_place_artifact(self, tmp_path):
        """A _process that uses its input in place returns it and writes no ingest file."""

        class InPlaceDataset(BaseDataset):
            name = "dummy"
            doi = "10.0000/dummy"

            def _validate_params(self, **_dataset_kwargs):
                pass

            def _download(self, provider_dir: Path, **_dataset_kwargs):
                return provider_dir

            def _source_filename(self, **_dataset_kwargs):
                return "dummy.h5"

            def _process(self, provider_artifact: Path, _ingest_path: Path, **_dataset_kwargs):
                return provider_artifact

        provider_dir = tmp_path / "provider"
        ingest_path = tmp_path / "ingest" / "dummy.h5"

        assert InPlaceDataset().process(provider_dir, ingest_path) == provider_dir
        assert list(ingest_path.parent.iterdir()) == []


def test_akt_process_extracts_into_given_path(tmp_path):
    """AKT finds its ZIP entry by the final name and writes it to the temporary ingest path."""

    class DummyAktDataset(AKTZipBaseDataset):
        name = "dummy"
        doi = "10.0000/dummy"
        _zipfile = "dummy.zip"

        def _validate_params(self, **_dataset_kwargs):
            pass

        def _source_filename(self, **_dataset_kwargs):
            return "scene.sofa"

    archive = tmp_path / "dummy.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("nested/dir/scene.sofa", b"sofa-bytes")
    ingest_path = tmp_path / "ingest" / "scene.sofa"

    assert DummyAktDataset().process(archive, ingest_path) == ingest_path
    assert ingest_path.read_bytes() == b"sofa-bytes"
    assert sorted(path.name for path in ingest_path.parent.iterdir()) == ["scene.sofa"]
