"""End-to-end: ingest -> mock-GCS archive -> chunk-level query."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from cctv_zarr import ome
from cctv_zarr.archive import CloudArchive, MockStorageClient
from cctv_zarr.config import Settings
from cctv_zarr.exceptions import QueryError
from cctv_zarr.query import QueryClient
from cctv_zarr.writer import CctvZarrWriter

from conftest_util import approach_frames, static_frames, write_video

CONFIG = Path(__file__).parent.parent / "config.yaml"
T0 = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings(tmp_path):
    base = Settings.load(CONFIG)
    return replace(
        base,
        gop=replace(base.gop, gop_frames=10),
        runtime=replace(base.runtime, store_directory=str(tmp_path)),
    )


@pytest.fixture
def archived(settings, tmp_path):
    frames = static_frames(12) + approach_frames(16) + static_frames(12)
    video = write_video(tmp_path / "cam01.mp4", frames)
    store = tmp_path / "cam01.zarr"
    CctvZarrWriter(settings).ingest(video, store, start_time=T0)
    client = MockStorageClient(tmp_path / "mock_gcs")
    archive = CloudArchive(client, "cctv-archive", "sites/gate3")
    archive.upload_store(store, "cam01.zarr")
    return archive, store


class TestLocalQuery:
    def test_time_window_selects_chunks(self, archived):
        _, store = archived
        q = QueryClient.from_local(store)
        sel = q.select(
            start=T0 + timedelta(seconds=0.4),
            end=T0 + timedelta(seconds=0.9),
        )
        assert sel.chunk_indices == (1, 2)

    def test_movement_only_selection(self, archived):
        _, store = archived
        q = QueryClient.from_local(store)
        sel = q.select(movement=True)
        assert len(sel.records) >= 1
        assert all(r.movement == 1 for r in sel.records)
        quiet = q.select(movement=False)
        assert all(r.movement == 0 for r in quiet.records)
        assert len(sel.records) + len(quiet.records) == 4

    def test_fetch_returns_frames_and_timestamps(self, archived):
        _, store = archived
        q = QueryClient.from_local(store)
        sel = q.select(end=T0 + timedelta(seconds=0.4))
        frames, stamps = q.fetch(sel)
        assert len(frames) == 10
        assert frames[0].shape == (240, 320, 3)
        assert stamps[0] == pytest.approx(T0.timestamp())
        assert np.all(np.diff(stamps) > 0)

    def test_iso_string_window(self, archived):
        _, store = archived
        q = QueryClient.from_local(store)
        sel = q.select(start="2026-07-20T14:00:00.400+00:00")
        assert sel.chunk_indices == (1, 2, 3)

    def test_bad_timestamp_raises(self, archived):
        _, store = archived
        q = QueryClient.from_local(store)
        with pytest.raises(QueryError):
            q.select(start="not a time")

    def test_movement_series_is_binary(self, archived):
        _, store = archived
        q = QueryClient.from_local(store)
        series = q.movement_series()
        assert set(np.unique(series)).issubset({0, 1})


class TestArchiveQuery:
    def test_metadata_only_open(self, archived, tmp_path):
        archive, _ = archived
        cache = tmp_path / "cache"
        q = QueryClient.from_archive(archive, "cam01.zarr", cache)
        assert len(q.records) == 4
        image_dir = cache / "cam01.zarr" / ome.IMAGE_PATH
        fetched = [p for p in image_dir.iterdir() if p.name != ".zarray"]
        assert fetched == []  # no pixel chunks moved yet

    def test_partial_fetch_only_selected_chunks(self, archived, tmp_path):
        archive, _ = archived
        q = QueryClient.from_archive(
            archive, "cam01.zarr", tmp_path / "cache",
        )
        sel = q.select(movement=True)
        frames, stamps = q.fetch(sel)
        assert len(frames) == 10 * len(sel.records)
        image_dir = Path(q.root) / ome.IMAGE_PATH
        fetched = sorted(
            p.name for p in image_dir.iterdir() if p.name != ".zarray"
        )
        expected = sorted(f"{i}.0.0.0.0" for i in sel.chunk_indices)
        assert fetched == expected  # unselected chunks stayed remote

    def test_fetched_section_matches_query_window(self, archived, tmp_path):
        archive, _ = archived
        q = QueryClient.from_archive(
            archive, "cam01.zarr", tmp_path / "cache",
        )
        start = T0 + timedelta(seconds=0.4)
        end = T0 + timedelta(seconds=0.8)
        frames, stamps = q.fetch(q.select(start=start, end=end))
        assert stamps.min() >= T0.timestamp()
        assert stamps.max() < end.timestamp() + 0.4  # within chunk pad

    def test_export_mp4(self, archived, tmp_path):
        archive, _ = archived
        q = QueryClient.from_archive(
            archive, "cam01.zarr", tmp_path / "cache",
        )
        out = tmp_path / "event.mp4"
        n = q.export_mp4(q.select(movement=True), out, fps=25.0)
        assert n > 0
        assert out.exists() and out.stat().st_size > 0

    def test_export_frames_named_by_timestamp(self, archived, tmp_path):
        archive, _ = archived
        q = QueryClient.from_archive(
            archive, "cam01.zarr", tmp_path / "cache",
        )
        out = tmp_path / "frames"
        n = q.export_frames(q.select(movement=True), out)
        pngs = sorted(out.glob("*.png"))
        assert len(pngs) == n
        assert pngs[0].name.startswith("20260720T14")

    def test_empty_selection_export_raises(self, archived, tmp_path):
        archive, _ = archived
        q = QueryClient.from_archive(
            archive, "cam01.zarr", tmp_path / "cache",
        )
        sel = q.select(start=T0 + timedelta(hours=5))
        with pytest.raises(QueryError):
            q.export_mp4(sel, tmp_path / "empty.mp4")
