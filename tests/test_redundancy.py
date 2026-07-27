"""Network-loss redundancy tests: retries and resumable uploads."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cctv_zarr.archive import (
    CloudArchive, MockStorageClient, UPLOAD_STATE_FILENAME,
)
from cctv_zarr.config import Settings
from cctv_zarr.exceptions import ArchiveError
from cctv_zarr.query import QueryClient
from cctv_zarr.writer import CctvZarrWriter

from conftest_util import approach_frames, static_frames, write_video

CONFIG = Path(__file__).parent.parent / "config.yaml"
T0 = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)


class FlakyBlob:
    """Blob wrapper that fails a set number of times per operation."""

    def __init__(self, inner, plan):
        self._inner = inner
        self._plan = plan

    def _maybe_fail(self, op):
        self._plan["calls"] += 1
        if self._plan["fail_remaining"] > 0:
            self._plan["fail_remaining"] -= 1
            raise ConnectionError("simulated network loss")

    def upload_from_filename(self, filename):
        self._maybe_fail("upload")
        return self._inner.upload_from_filename(filename)

    def download_to_filename(self, filename):
        self._maybe_fail("download")
        return self._inner.download_to_filename(filename)


class FlakyBucket:
    """Bucket wrapper injecting failures into blob transfers."""

    def __init__(self, inner, plan):
        self._inner = inner
        self._plan = plan
        self.name = inner.name

    def blob(self, blob_name):
        return FlakyBlob(self._inner.blob(blob_name), self._plan)

    def list_blobs(self, prefix=""):
        return self._inner.list_blobs(prefix=prefix)


class FlakyClient:
    """Storage client whose transfers fail fail_remaining times."""

    def __init__(self, root, fail_remaining):
        self._inner = MockStorageClient(root)
        self.plan = {"fail_remaining": fail_remaining, "calls": 0}

    def bucket(self, bucket_name):
        return FlakyBucket(self._inner.bucket(bucket_name), self.plan)


@pytest.fixture
def settings(tmp_path):
    base = Settings.load(CONFIG)
    return replace(
        base,
        gop=replace(base.gop, gop_frames=10),
        runtime=replace(
            base.runtime, store_directory=str(tmp_path / "stores"),
        ),
    )


@pytest.fixture
def store(settings, tmp_path):
    frames = static_frames(12) + approach_frames(16) + static_frames(12)
    video = write_video(tmp_path / "cam01.mp4", frames)
    dest = Path(settings.runtime.store_directory) / "cam01.zarr"
    CctvZarrWriter(settings).ingest(video, dest, start_time=T0)
    return dest


def _archive(client):
    return CloudArchive(
        client, "cctv-archive", "sites/gate3",
        retry_attempts=3, retry_base_delay_s=0.0,
    )


def _object_count(store):
    return sum(1 for p in store.rglob("*") if p.is_file())


class TestRetries:
    def test_transient_failures_are_retried(self, store, tmp_path):
        client = FlakyClient(tmp_path / "gcs", fail_remaining=2)
        archive = _archive(client)
        count = archive.upload_store(store, "cam01.zarr")
        assert count == _object_count(store)
        state = store.parent / (store.name + UPLOAD_STATE_FILENAME)
        assert not state.exists()

    def test_flaky_download_recovers(self, store, tmp_path):
        good = MockStorageClient(tmp_path / "gcs")
        _archive(good).upload_store(store, "cam01.zarr")
        client = FlakyClient(tmp_path / "gcs", fail_remaining=2)
        query = QueryClient.from_archive(
            _archive(client), "cam01.zarr", tmp_path / "cache",
        )
        frames, _ = query.fetch(query.select(movement=True))
        assert len(frames) > 0


class TestResumableUpload:
    def test_outage_leaves_journal_then_resumes(self, store, tmp_path):
        down = FlakyClient(tmp_path / "gcs", fail_remaining=10 ** 6)
        with pytest.raises(ArchiveError):
            _archive(down).upload_store(store, "cam01.zarr")
        state = store.parent / (store.name + UPLOAD_STATE_FILENAME)
        assert state.exists()
        assert CloudArchive.pending_objects(store) > 0

        good = MockStorageClient(tmp_path / "gcs")
        resumed = _archive(good).upload_store(store, "cam01.zarr")
        assert resumed == _object_count(store)
        assert not state.exists()
        assert CloudArchive.pending_objects(store) == 0

    def test_partial_outage_resumes_only_missing(self, store, tmp_path):
        total = _object_count(store)
        flaky = FlakyClient(tmp_path / "gcs", fail_remaining=0)
        archive = _archive(flaky)
        sent_some = 0
        for path, rel in CloudArchive._store_files(store):
            if sent_some >= 3:
                break
            archive._with_retries(
                f"seed {rel}",
                lambda p=path, k=archive._key("cam01.zarr", rel):
                    archive.bucket.blob(k).upload_from_filename(str(p)),
            )
            CloudArchive._write_state(
                store.parent / (store.name + UPLOAD_STATE_FILENAME),
                {r for _, r in list(
                    CloudArchive._store_files(store)
                )[:sent_some + 1]},
            )
            sent_some += 1
        assert CloudArchive.pending_objects(store) == total - sent_some
        good = MockStorageClient(tmp_path / "gcs")
        resumed = _archive(good).upload_store(store, "cam01.zarr")
        assert resumed == total - sent_some
        keys = _archive(good).list_store("cam01.zarr")
        assert len(keys) == total

    def test_second_upload_is_noop(self, store, tmp_path):
        good = MockStorageClient(tmp_path / "gcs")
        first = _archive(good).upload_store(store, "cam01.zarr")
        again = _archive(good).upload_store(store, "cam01.zarr")
        assert first > 0
        assert again == first

    def test_journal_excluded_from_archive(self, store, tmp_path):
        down = FlakyClient(tmp_path / "gcs", fail_remaining=10 ** 6)
        with pytest.raises(ArchiveError):
            _archive(down).upload_store(store, "cam01.zarr")
        good = MockStorageClient(tmp_path / "gcs")
        _archive(good).upload_store(store, "cam01.zarr")
        keys = _archive(good).list_store("cam01.zarr")
        assert all(UPLOAD_STATE_FILENAME not in k for k in keys)

    def test_bad_retry_budget_rejected(self, tmp_path):
        client = MockStorageClient(tmp_path / "gcs")
        with pytest.raises(ArchiveError):
            CloudArchive(client, "b", "p", retry_attempts=0)
