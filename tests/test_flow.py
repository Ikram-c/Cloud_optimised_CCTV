"""Motion tracker tests on synthetic scenes. Fully offline."""

from pathlib import Path

import numpy as np
import pytest

from cctv_zarr.config import Settings
from cctv_zarr.flow import MotionTracker, movement_flags
from cctv_zarr.models import MotionEvent

from conftest_util import (
    H, W, approach_frames, draw_square, lateral_frames, static_frames,
    textured_background,
)

CONFIG = Path(__file__).parent.parent / "config.yaml"
FPS = 2.0


@pytest.fixture
def config():
    return Settings.load(CONFIG).flow


def _tracker(config):
    return MotionTracker(config, fps=FPS)


def _sequence(*segments):
    """Frames of one square walking and pausing.

    Args:
        segments: (kind, n) pairs; kind "move" advances the square,
            "pause" holds it still.

    Returns:
        list: BGR frames.
    """
    scene = textured_background()
    frames = []
    x = 40
    for kind, count in segments:
        for _ in range(count):
            if kind == "move":
                x += 6
            frames.append(draw_square(scene, x, H // 2, 24))
    return frames


def _two_object_frames(count):
    """Two squares moving simultaneously, far apart."""
    scene = textured_background()
    frames = []
    for i in range(count):
        one = draw_square(scene, 40 + i * 5, 60, 20)
        frames.append(draw_square(one, 280 - i * 5, 180, 20))
    return frames


class TestMotionTracker:
    def test_lateral_movement_detected(self, config):
        tracker = _tracker(config)
        results = tracker.analyse(lateral_frames(20))
        assert any(r.moving for r in results)
        assert len(tracker.events()) >= 1

    def test_approaching_movement_detected(self, config):
        tracker = _tracker(config)
        tracker.analyse(approach_frames(20))
        assert len(tracker.events()) >= 1

    def test_static_scene_no_events(self, config):
        tracker = _tracker(config)
        results = tracker.analyse(static_frames(16))
        assert not any(r.moving for r in results)
        assert tracker.events() == []

    def test_pause_within_buffer_is_one_movement(self, config):
        tracker = _tracker(config)
        frames = _sequence(("move", 8), ("pause", 6), ("move", 8))
        tracker.analyse(frames)
        events = tracker.events()
        assert len(events) == 1
        assert events[0].start_frame <= 2
        assert events[0].end_frame >= len(frames) - 2
        flags = movement_flags(events, len(frames))
        assert flags[10] is True

    def test_pause_beyond_buffer_splits_movements(self, config):
        tracker = _tracker(config)
        frames = _sequence(("move", 8), ("pause", 14), ("move", 8))
        tracker.analyse(frames)
        events = tracker.events()
        assert len(events) == 2
        flags = movement_flags(events, len(frames))
        assert flags[14] is False

    def test_two_objects_are_two_movements(self, config):
        tracker = _tracker(config)
        tracker.analyse(_two_object_frames(16))
        events = tracker.events()
        assert len(events) == 2
        overlap = (
            events[0].start_frame <= events[1].end_frame
            and events[1].start_frame <= events[0].end_frame
        )
        assert overlap

    def test_short_blip_is_noise(self, config):
        tracker = _tracker(config)
        tracker.analyse(_sequence(("move", 2), ("pause", 14)))
        assert tracker.events() == []

    def test_first_frame_is_baseline(self, config):
        tracker = _tracker(config)
        results = tracker.analyse(lateral_frames(2))
        assert results[0].magnitude == 0.0
        assert not results[0].moving

    def test_reset_clears_state(self, config):
        tracker = _tracker(config)
        tracker.analyse(lateral_frames(12))
        tracker.reset()
        assert tracker.events() == []
        results = tracker.analyse(static_frames(6))
        assert not any(r.moving for r in results)

    def test_empty_frame_raises(self, config):
        tracker = _tracker(config)
        with pytest.raises(ValueError):
            tracker.update(np.empty((0, 0, 3), dtype=np.uint8), 0)

    def test_bad_fps_raises(self, config):
        with pytest.raises(ValueError):
            MotionTracker(config, fps=0.0)


class TestMovementFlags:
    def test_span_includes_bridged_pause(self):
        events = [MotionEvent(0, 2, 8)]
        flags = movement_flags(events, 12)
        assert flags[2] and flags[5] and flags[8]
        assert not flags[1] and not flags[9]

    def test_out_of_range_event_clamped(self):
        events = [MotionEvent(0, 5, 50)]
        flags = movement_flags(events, 10)
        assert flags[9] and not flags[4]
