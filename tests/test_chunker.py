"""GOP chunk-grid tests. Fully offline."""

from datetime import datetime, timedelta, timezone

import pytest

from cctv_zarr.chunker import build_chunk_records, count_events
from cctv_zarr.config import GopConfig

T0 = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)
GOP = GopConfig(gop_frames=10, max_video_frames=100000)


def _flows(flags):
    return list(flags)


class TestChunkGrid:
    def test_boundaries_align_to_gop(self):
        records = build_chunk_records(GOP, _flows([False] * 35), T0, 25.0)
        assert [(r.start_frame, r.end_frame) for r in records] == [
            (0, 10), (10, 20), (20, 30), (30, 35),
        ]

    def test_timestamps_recorded_per_chunk(self):
        records = build_chunk_records(GOP, _flows([False] * 25), T0, 25.0)
        assert records[0].start_time == T0
        assert records[1].start_time == T0 + timedelta(seconds=0.4)
        assert records[2].end_time == T0 + timedelta(seconds=1.0)

    def test_movement_is_binary_category(self):
        flags = [False] * 12 + [True] * 6 + [False] * 12
        records = build_chunk_records(GOP, _flows(flags), T0, 25.0)
        assert [r.movement for r in records] == [0, 1, 0]
        assert all(r.movement in (0, 1) for r in records)

    def test_event_snaps_to_gop_and_marks_trigger(self):
        flags = [False] * 12 + [True] * 6 + [False] * 12
        records = build_chunk_records(GOP, _flows(flags), T0, 25.0)
        assert records[1].trigger == "motion"
        assert records[0].trigger == "gop"
        assert records[2].trigger == "gop"
        assert count_events(records) == 1

    def test_event_spanning_chunks_counts_once(self):
        flags = [False] * 8 + [True] * 14 + [False] * 8
        records = build_chunk_records(GOP, _flows(flags), T0, 25.0)
        assert [r.movement for r in records] == [1, 1, 1]
        assert count_events(records) == 1

    def test_two_events_count_twice(self):
        flags = (
            [True] * 4 + [False] * 6
            + [False] * 10
            + [True] * 4 + [False] * 6
        )
        records = build_chunk_records(GOP, _flows(flags), T0, 25.0)
        assert [r.movement for r in records] == [1, 0, 1]
        assert count_events(records) == 2

    def test_empty_run_raises(self):
        with pytest.raises(ValueError):
            build_chunk_records(GOP, [], T0, 25.0)

    def test_naive_start_time_raises(self):
        with pytest.raises(ValueError):
            build_chunk_records(
                GOP, _flows([False]), datetime(2026, 7, 20), 25.0,
            )

    def test_overlap_window(self):
        records = build_chunk_records(GOP, _flows([False] * 30), T0, 25.0)
        window_start = T0 + timedelta(seconds=0.5)
        window_end = T0 + timedelta(seconds=0.9)
        hits = [r for r in records if r.overlaps(window_start, window_end)]
        assert [r.index for r in hits] == [1, 2]
