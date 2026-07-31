"""Writer + OME metadata tests on a synthetic video. Fully offline."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from cctv_zarr import ome, zarr_io
from cctv_zarr.config import Settings
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
def store(settings, tmp_path):
    frames = (
        static_frames(12) + approach_frames(16) + static_frames(12)
    )
    video = write_video(tmp_path / "cam01.mp4", frames)
    dest = tmp_path / "cam01.zarr"
    result = CctvZarrWriter(settings).ingest(video, dest, start_time=T0)
    return dest, result


class TestStoreLayout:
    def test_ingest_counts(self, store):
        _, result = store
        assert result.frames_written == 40
        assert result.chunk_count == 4
        assert result.movement_chunks >= 1
        assert result.motion_events >= 1

    def test_compression_metrics_recorded(self, store):
        dest, result = store
        assert result.stored_mb > 0
        assert result.compression_ratio > 1.0
        assert result.footprint_ratio >= result.compression_ratio
        block = zarr_io.read_attrs(dest)["cctv"]["compression"]
        assert block["ratio"] > 1.0
        assert block["stored_mb"] > 0
        assert block["raw_mb"] > block["stored_mb"]
        assert block["source_mb"] > 0

    def test_image_array_is_tczyx_gop_chunked(self, store):
        dest, _ = store
        meta = json.loads((dest / "0" / ".zarray").read_text())
        assert meta["shape"][0] == 40
        assert meta["chunks"][0] == 10
        assert len(meta["shape"]) == 5
        assert meta["shape"][2] == 1

    def test_ome_multiscales_metadata(self, store):
        dest, _ = store
        attrs = zarr_io.read_attrs(dest)
        ms = attrs["multiscales"][0]
        assert ms["version"] == "0.4"
        assert [a["name"] for a in ms["axes"]] == ["t", "c", "z", "y", "x"]
        scale = ms["datasets"][0]["coordinateTransformations"][0]["scale"]
        assert scale[0] == pytest.approx(1.0 / 25.0)

    def test_movement_detected_is_binary(self, store):
        dest, _ = store
        movement = zarr_io.ZarrArray.open(
            dest / ome.MOVEMENT_PATH
        ).read_full()
        assert movement.dtype == np.uint8
        assert set(np.unique(movement)).issubset({0, 1})
        assert movement.shape == (40,)
        assert movement.sum() > 0

    def test_timestamps_per_frame_and_per_chunk(self, store):
        dest, result = store
        stamps = zarr_io.ZarrArray.open(
            dest / ome.TIMESTAMPS_PATH
        ).read_full()
        assert stamps.shape == (40,)
        assert stamps[0] == pytest.approx(T0.timestamp())
        assert stamps[25] == pytest.approx(T0.timestamp() + 25 / result.fps)
        records = ome.parse_chunk_records(zarr_io.read_attrs(dest))
        assert len(records) == 4
        assert records[0].start_time == T0
        for r in records:
            assert r.end_time > r.start_time
            assert r.movement in (0, 1)

    def test_manifest_flags_the_approach_window(self, store):
        dest, _ = store
        records = ome.parse_chunk_records(zarr_io.read_attrs(dest))
        assert records[0].movement == 0
        assert 1 in {records[1].movement, records[2].movement}
        assert any(r.trigger == "motion" for r in records)

    def test_pixels_roundtrip(self, store):
        dest, _ = store
        image = zarr_io.ZarrArray.open(dest / "0")
        chunk = image.read_chunk((0, 0, 0, 0, 0))
        assert chunk.shape == (10, 3, 1, 240, 320)
        assert chunk.dtype == np.uint8
        assert chunk.max() > 0
