"""Immutable data models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


@dataclass(frozen=True, slots=True)
class FlowResult:
    """Per-frame optical-flow motion measurement.

    Direction is deliberately ignored: ``magnitude`` is the mean flow
    speed over the analysis frame, ``moving`` is True when at least
    one object-sized region of pixels is in motion, and ``objects``
    counts the tracked objects moving in this frame.
    """

    frame_index: int
    magnitude: float
    moving: bool
    objects: int


@dataclass(frozen=True, slots=True)
class MotionEvent:
    """One object's continuous movement, pauses bridged.

    A movement belongs to a single tracked object or person and runs
    from its first moving frame to its last; pauses shorter than the
    configured buffer do not split it. ``end_frame`` is inclusive.
    """

    track_id: int
    start_frame: int
    end_frame: int

    def __post_init__(self):
        if self.end_frame < self.start_frame:
            raise ValueError("event must not end before it starts")
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One encoder-aligned time chunk (a closed GOP) in the store.

    ``movement`` is the binary "movement detected" category: 1 when
    any frame in the chunk carries an approach event, else 0. Start
    and end timestamps are recorded per chunk; ``end_time`` is the
    time of the frame after the last frame (half-open, like the frame
    range).
    """

    index: int
    start_frame: int
    end_frame: int
    start_time: datetime
    end_time: datetime
    movement: int
    trigger: str

    def __post_init__(self):
        if self.end_frame <= self.start_frame:
            raise ValueError("chunk must contain at least one frame")
        if self.movement not in (0, 1):
            raise ValueError("movement must be the binary category 0 or 1")
        if self.trigger not in ("gop", "motion"):
            raise ValueError("trigger must be 'gop' or 'motion'")

    def overlaps(
        self, start: Optional[datetime], end: Optional[datetime],
    ) -> bool:
        """Whether this chunk intersects a [start, end) time window.

        Args:
            start (Optional[datetime]): Window start; None = open.
            end (Optional[datetime]): Window end; None = open.

        Returns:
            bool: True when the chunk intersects the window.
        """
        if start is not None and self.end_time <= start:
            return False
        if end is not None and self.start_time >= end:
            return False
        return True


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Basic per-video properties."""

    total_frames: int
    fps: float
    width: int
    height: int

    def __post_init__(self):
        if self.total_frames <= 0:
            raise ValueError("total_frames must be positive")
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of ingesting one video into a store.

    ``compression_ratio`` compares the store against raw uncompressed
    frames at the stored resolution (the lossless codec's own work);
    ``footprint_ratio`` compares it against raw frames at the source
    resolution (resize and compression together). ``source_mb`` is the
    original video file, reported for transparency.
    """

    store_path: str
    frames_written: int
    chunk_count: int
    movement_chunks: int
    motion_events: int
    start_time: datetime
    fps: float
    stored_mb: float = 0.0
    raw_mb: float = 0.0
    source_mb: float = 0.0
    compression_ratio: float = 1.0
    footprint_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class QuerySelection:
    """The chunks a query resolved to, before any pixel data moves."""

    chunk_indices: Tuple[int, ...]
    records: Tuple[ChunkRecord, ...]
    frame_start: int
    frame_end: int
