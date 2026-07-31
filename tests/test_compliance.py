"""Retention, erasure, redaction, and governance tests. Offline."""

import json
import os
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from cctv_zarr import audit, ome, retention, zarr_io
from cctv_zarr.archive import (
    CloudArchive, MockStorageClient, build_cipher,
)
from cctv_zarr.config import Settings
from cctv_zarr.query import QueryClient
from cctv_zarr.writer import CctvZarrWriter

from conftest_util import approach_frames, static_frames, write_video

CONFIG = Path(__file__).parent.parent / "config.yaml"
T0 = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)
FERNET_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="


def _settings(tmp_path, **overrides):
    base = Settings.load(CONFIG)
    settings = replace(
        base,
        gop=replace(base.gop, gop_frames=10),
        runtime=replace(
            base.runtime,
            store_directory=str(tmp_path / "stores"),
            video_directory=str(tmp_path / "videos"),
        ),
        archive=replace(
            base.archive, local_root=str(tmp_path / "mock_gcs"),
        ),
    )
    for key, value in overrides.items():
        settings = replace(settings, **{key: value})
    return settings


def _ingest(settings, tmp_path, name="cam01", start_time=T0):
    folder = Path(settings.runtime.video_directory)
    folder.mkdir(parents=True, exist_ok=True)
    frames = static_frames(12) + approach_frames(16) + static_frames(12)
    video = write_video(folder / f"{name}.mp4", frames)
    store = Path(settings.runtime.store_directory) / f"{name}.zarr"
    result = CctvZarrWriter(settings).ingest(
        video, store, start_time=start_time,
    )
    return video, store, result


class TestRetention:
    def test_differential_expiry(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, _ = _ingest(settings, tmp_path)
        now = T0 + timedelta(hours=100)
        result = retention.prune_store(
            store, settings.retention,
            Path(settings.runtime.store_directory), now=now,
        )
        assert result["objects"]
        assert result["whole_store"] is False
        records = ome.parse_chunk_records(zarr_io.read_attrs(store))
        image = zarr_io.ZarrArray.open(store / ome.IMAGE_PATH)
        for r in records:
            exists = image.chunk_path(
                (r.index, 0, 0, 0, 0),
            ).exists()
            assert exists == bool(r.movement)

    def test_whole_store_expiry(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, _ = _ingest(settings, tmp_path)
        now = T0 + timedelta(days=100)
        result = retention.prune_store(
            store, settings.retention,
            Path(settings.runtime.store_directory), now=now,
        )
        assert result["whole_store"] is True
        assert not store.exists()

    def test_deletions_are_journalled(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, _ = _ingest(settings, tmp_path)
        root = Path(settings.runtime.store_directory)
        retention.prune_store(
            store, settings.retention, root,
            now=T0 + timedelta(hours=100),
        )
        entries = audit.read_log(
            root / audit.DELETION_LOG_FILENAME,
        )
        assert entries
        assert entries[-1]["reason"] == "retention"
        assert entries[-1]["store"] == "cam01.zarr"

    def test_fresh_store_untouched(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, result = _ingest(settings, tmp_path)
        outcome = retention.prune_store(
            store, settings.retention,
            Path(settings.runtime.store_directory),
            now=T0 + timedelta(hours=1),
        )
        assert outcome["objects"] == []
        assert store.is_dir()

    def test_prune_all_sweeps_exports(self, tmp_path):
        settings = _settings(tmp_path)
        _ingest(settings, tmp_path)
        root = Path(settings.runtime.store_directory)
        exports = root / retention.EXPORTS_DIRNAME
        exports.mkdir(parents=True, exist_ok=True)
        clip = exports / "old_clip.mp4"
        clip.write_bytes(b"clip")
        old = time.time() - 90 * 3600
        os.utime(clip, (old, old))
        summary = retention.prune_all(settings)
        assert summary["exports_swept"] == 1
        assert not clip.exists()


class TestErasure:
    def test_erase_window_and_cache(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, _ = _ingest(settings, tmp_path)
        root = Path(settings.runtime.store_directory)
        cache = tmp_path / "query_cache"
        image_rel = f"{ome.IMAGE_PATH}/1.0.0.0.0"
        cached = cache / "cam01.zarr" / image_rel
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"cached")
        start = (T0 + timedelta(seconds=0.4)).isoformat()
        end = (T0 + timedelta(seconds=0.8)).isoformat()
        objects = retention.erase_window(store, start, end, root)
        assert image_rel in objects
        removed = retention.invalidate_cache(
            cache, "cam01.zarr", objects,
        )
        assert removed == 1
        image = zarr_io.ZarrArray.open(store / ome.IMAGE_PATH)
        assert image.read_chunk((1, 0, 0, 0, 0)).max() == 0

    def test_archive_objects_deleted(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, _ = _ingest(settings, tmp_path)
        client = MockStorageClient(tmp_path / "mock_gcs")
        archive = CloudArchive(client, "bucket", "sites/a")
        archive.upload_store(store, "cam01.zarr")
        rel = f"{ome.IMAGE_PATH}/1.0.0.0.0"
        assert archive.delete_objects("cam01.zarr", [rel]) == 1
        assert rel not in archive.list_store("cam01.zarr")


class TestMinimisation:
    def test_quiet_chunks_never_stored(self, tmp_path):
        settings = _settings(tmp_path)
        settings = replace(
            settings,
            retention=replace(
                settings.retention, keep_non_movement=False,
            ),
        )
        _, store, result = _ingest(settings, tmp_path)
        records = ome.parse_chunk_records(zarr_io.read_attrs(store))
        image = zarr_io.ZarrArray.open(store / ome.IMAGE_PATH)
        for r in records:
            exists = image.chunk_path(
                (r.index, 0, 0, 0, 0),
            ).exists()
            assert exists == bool(r.movement)
        attrs = zarr_io.read_attrs(store)
        assert attrs["cctv"]["non_movement_stored"] is False
        assert result.movement_chunks >= 1


class TestRedaction:
    def test_masked_export_blacks_region(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, _ = _ingest(settings, tmp_path)
        client = QueryClient.from_local(store)
        selection = client.select(movement=True)
        out = tmp_path / "masked"
        written = client.export_frames(
            selection, out,
            mask_regions=[(0.0, 0.0, 0.5, 0.5)],
        )
        assert written > 0
        import cv2
        frame = cv2.imread(str(sorted(out.iterdir())[0]))
        height, width = frame.shape[:2]
        assert frame[: height // 2 - 1, : width // 2 - 1].max() == 0
        assert frame.max() > 0

    def test_bad_mask_rejected(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, _ = _ingest(settings, tmp_path)
        client = QueryClient.from_local(store)
        selection = client.select(movement=True)
        from cctv_zarr.exceptions import QueryError
        with pytest.raises(QueryError):
            client.export_mp4(
                selection, tmp_path / "x.mp4",
                mask_regions=[(0.5, 0.5, 0.5, 0.5)],
            )


class TestGovernanceMetadata:
    def test_store_carries_governance_and_provenance(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, _ = _ingest(settings, tmp_path)
        attrs = zarr_io.read_attrs(store)
        block = attrs["cctv"]["governance"]
        assert block["biometric_source"] is False
        assert "retention_policy_hours" in block
        assert attrs["cctv"]["time_source"] == "explicit"

    def test_mtime_provenance_recorded(self, tmp_path):
        settings = _settings(tmp_path)
        folder = Path(settings.runtime.video_directory)
        folder.mkdir(parents=True, exist_ok=True)
        frames = static_frames(12) + approach_frames(16)
        video = write_video(folder / "m.mp4", frames)
        store = Path(settings.runtime.store_directory) / "m.zarr"
        CctvZarrWriter(settings).ingest(video, store)
        attrs = zarr_io.read_attrs(store)
        assert attrs["cctv"]["time_source"] == "file_mtime"

    def test_events_opt_out(self, tmp_path):
        settings = _settings(tmp_path)
        settings = replace(
            settings,
            flow=replace(settings.flow, record_events=False),
        )
        _, store, result = _ingest(settings, tmp_path)
        attrs = zarr_io.read_attrs(store)
        assert attrs["cctv"]["events"] == []
        assert result.movement_chunks >= 1


class TestSourceHandling:
    def test_delete_after_ingest(self, tmp_path):
        settings = _settings(tmp_path)
        video, _, _ = _ingest(settings, tmp_path)
        policy = replace(
            settings.retention, source_after_ingest="delete",
        )
        action = retention.handle_source(
            video, policy, Path(settings.runtime.store_directory),
        )
        assert action == "delete"
        assert not video.exists()

    def test_quarantine_after_ingest(self, tmp_path):
        settings = _settings(tmp_path)
        video, _, _ = _ingest(settings, tmp_path)
        policy = replace(
            settings.retention, source_after_ingest="quarantine",
        )
        action = retention.handle_source(
            video, policy, Path(settings.runtime.store_directory),
        )
        assert action == "quarantine"
        assert not video.exists()
        moved = (
            video.parent / retention.QUARANTINE_DIRNAME / video.name
        )
        assert moved.exists()


class TestArchiveEncryption:
    def test_objects_encrypted_and_roundtrip(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, _ = _ingest(settings, tmp_path)
        cipher = build_cipher(FERNET_KEY)
        client = MockStorageClient(tmp_path / "mock_gcs")
        archive = CloudArchive(
            client, "bucket", "sites/a", cipher=cipher,
        )
        archive.upload_store(store, "cam01.zarr")
        raw_object = (
            tmp_path / "mock_gcs" / "bucket" / "sites" / "a"
            / "cam01.zarr" / ".zattrs"
        ).read_bytes()
        plain = (store / ".zattrs").read_bytes()
        assert raw_object != plain
        fetched = archive.fetch(
            "cam01.zarr", [".zattrs"], tmp_path / "fetched",
        )
        assert json.loads(
            (fetched / ".zattrs").read_text()
        ) == json.loads(plain.decode("utf-8"))

    def test_bad_key_rejected(self):
        from cctv_zarr.exceptions import ArchiveError
        with pytest.raises(ArchiveError):
            build_cipher("not-a-key")


class TestAccessLog:
    def test_query_export_logged(self, tmp_path):
        settings = _settings(tmp_path)
        _, store, _ = _ingest(settings, tmp_path)
        root = Path(settings.runtime.store_directory)
        audit.log_access(
            root, "export", "cam01.zarr",
            destination="clip.mp4", purpose="insurance claim",
        )
        entries = audit.read_log(root / audit.ACCESS_LOG_FILENAME)
        assert entries[-1]["action"] == "export"
        assert entries[-1]["purpose"] == "insurance claim"
        assert "ts" in entries[-1]
