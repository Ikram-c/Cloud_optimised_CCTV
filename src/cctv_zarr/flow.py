"""Optical-flow motion detection with per-object movement tracking.

Dense Farneback flow between consecutive downscaled grey frames; the
flow's direction is deliberately discarded and only its magnitude is
used, so every kind of movement is monitored equally. Pixels moving
faster than ``min_magnitude_px`` are grouped into object-sized blobs
by connected components, and blobs are associated frame-to-frame into
tracks by nearest-centroid matching.

A movement event is one track's continuous motion: it opens at the
object's first moving frame and closes at its last. A track survives
without motion for up to ``pause_buffer_s`` seconds, so an object or
person that pauses and then continues stays one single event; only a
pause longer than the buffer (or leaving the scene) ends it. Tracks
shorter than ``min_track_frames`` are treated as noise and dropped.
"""

import logging
from typing import List, Sequence

import cv2
import numpy as np

from .config import FlowConfig
from .models import FlowResult, MotionEvent

logger = logging.getLogger(__name__)

MAX_BLOBS_PER_FRAME = 32
MORPH_KERNEL_SIZE = 3


class _Track:
    """Mutable state for one tracked moving object."""

    __slots__ = (
        "track_id", "cx", "cy", "first_frame",
        "last_motion_frame", "motion_frames",
    )

    def __init__(self, track_id: int, cx: float, cy: float, frame: int):
        """Open a track at a blob position.

        Args:
            track_id (int): Stable identifier.
            cx (float): Blob centroid x (analysis pixels).
            cy (float): Blob centroid y (analysis pixels).
            frame (int): Frame index of first motion.
        """
        self.track_id = track_id
        self.cx = cx
        self.cy = cy
        self.first_frame = frame
        self.last_motion_frame = frame
        self.motion_frames = 1


class MotionTracker:
    """Turns frames into per-frame motion flags and per-object events."""

    def __init__(self, config: FlowConfig, fps: float):
        """Derive the pause buffer and association radius.

        Args:
            config (FlowConfig): Validated flow configuration.
            fps (float): Video frame rate; converts the pause buffer
                from seconds to frames.

        Raises:
            ValueError: If fps is not positive.
        """
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self.config = config
        self.fps = fps
        width, height = config.analysis_size
        self._min_area = config.min_area_frac * width * height
        self._max_dist = config.match_distance_frac * float(
            np.hypot(width, height)
        )
        self._gap_frames = max(1, int(round(config.pause_buffer_s * fps)))
        self._kernel = np.ones(
            (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8,
        )
        self._prev = None
        self._tracks: List[_Track] = []
        self._events: List[MotionEvent] = []
        self._next_id = 0

    def reset(self):
        """Clear all inter-frame state between videos."""
        self._prev = None
        self._tracks = []
        self._events = []
        self._next_id = 0

    def update(self, frame_bgr: np.ndarray, frame_index: int) -> FlowResult:
        """Measure one frame and advance every track.

        Args:
            frame_bgr (np.ndarray): BGR or grayscale frame.
            frame_index (int): Source frame index.

        Returns:
            FlowResult: Frame-level motion measurement.

        Raises:
            ValueError: If the frame is empty.
        """
        if frame_bgr.size == 0:
            raise ValueError("frame must be non-empty")
        gray = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if frame_bgr.ndim == 3 else frame_bgr
        )
        gray = cv2.resize(
            gray, tuple(self.config.analysis_size),
            interpolation=cv2.INTER_AREA,
        )
        if self._prev is None:
            self._prev = gray
            return FlowResult(frame_index, 0.0, False, 0)
        c = self.config
        flow = cv2.calcOpticalFlowFarneback(
            self._prev, gray, None,
            c.pyr_scale, c.levels, c.winsize,
            c.iterations, c.poly_n, c.poly_sigma, 0,
        )
        self._prev = gray
        magnitude_map = np.hypot(flow[..., 0], flow[..., 1])
        blobs = self._find_blobs(magnitude_map)
        matched = self._associate(blobs, frame_index)
        self._expire(frame_index)
        return FlowResult(
            frame_index,
            float(np.mean(magnitude_map)),
            len(blobs) > 0,
            matched,
        )

    def finish(self):
        """Close every open track at the end of a video."""
        for track in self._tracks:
            self._close(track)
        self._tracks = []

    def events(self) -> List[MotionEvent]:
        """The movement events recorded so far.

        Returns:
            List[MotionEvent]: Sorted by start frame.
        """
        return sorted(self._events, key=lambda e: e.start_frame)

    def analyse(self, frames, start_index: int = 0) -> List[FlowResult]:
        """Run the tracker over an iterable of frames and finish.

        Args:
            frames: Iterable of BGR frames.
            start_index (int): Index of the first frame.

        Returns:
            List[FlowResult]: One result per frame, in order.
        """
        self.reset()
        results = [
            self.update(frame, start_index + i)
            for i, frame in enumerate(frames)
        ]
        self.finish()
        return results

    def _find_blobs(self, magnitude_map: np.ndarray) -> List[tuple]:
        """Group moving pixels into object-sized blobs.

        Args:
            magnitude_map (np.ndarray): Per-pixel flow speed.

        Returns:
            List[tuple]: (cx, cy, area), largest first, bounded.
        """
        mask = (
            magnitude_map > self.config.min_magnitude_px
        ).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8,
        )
        blobs = []
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area >= self._min_area:
                blobs.append((
                    float(centroids[label][0]),
                    float(centroids[label][1]),
                    area,
                ))
        blobs.sort(key=lambda b: -b[2])
        return blobs[:MAX_BLOBS_PER_FRAME]

    def _associate(self, blobs: List[tuple], frame_index: int) -> int:
        """Match blobs to tracks by nearest centroid; open new tracks.

        Args:
            blobs (List[tuple]): This frame's (cx, cy, area) blobs.
            frame_index (int): Current frame index.

        Returns:
            int: Number of objects moving in this frame.
        """
        unmatched = list(range(len(blobs)))
        moving = 0
        for track in self._tracks:
            best, best_dist = None, self._max_dist
            for position in unmatched:
                cx, cy, _ = blobs[position]
                dist = float(np.hypot(cx - track.cx, cy - track.cy))
                if dist <= best_dist:
                    best, best_dist = position, dist
            if best is None:
                continue
            unmatched.remove(best)
            cx, cy, _ = blobs[best]
            track.cx = cx
            track.cy = cy
            track.last_motion_frame = frame_index
            track.motion_frames += 1
            moving += 1
        for position in unmatched:
            if len(self._tracks) >= self.config.max_tracks:
                break
            cx, cy, _ = blobs[position]
            self._tracks.append(
                _Track(self._next_id, cx, cy, frame_index)
            )
            self._next_id += 1
            moving += 1
        return moving

    def _expire(self, frame_index: int):
        """Close tracks whose pause exceeded the buffer.

        Args:
            frame_index (int): Current frame index.
        """
        kept: List[_Track] = []
        for track in self._tracks:
            if frame_index - track.last_motion_frame > self._gap_frames:
                self._close(track)
            else:
                kept.append(track)
        self._tracks = kept

    def _close(self, track: _Track):
        """Record a finished track as an event if it was real motion.

        Args:
            track (_Track): The track to close.
        """
        if track.motion_frames >= self.config.min_track_frames:
            self._events.append(MotionEvent(
                track_id=track.track_id,
                start_frame=track.first_frame,
                end_frame=track.last_motion_frame,
            ))


def movement_flags(
    events: Sequence[MotionEvent], total_frames: int,
) -> List[bool]:
    """Per-frame movement flags from event spans.

    Frames inside an event span are flagged, including pause frames
    the buffer bridged; quiet frames after an event's last motion are
    not.

    Args:
        events (Sequence[MotionEvent]): Recorded events.
        total_frames (int): Number of stored frames.

    Returns:
        List[bool]: One flag per frame.
    """
    flags = [False] * total_frames
    for event in events:
        start = max(0, event.start_frame)
        end = min(total_frames - 1, event.end_frame)
        for index in range(start, end + 1):
            flags[index] = True
    return flags
