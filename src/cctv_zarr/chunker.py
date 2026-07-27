"""Encoder-aligned chunking with movement-activated segments.

The Zarr time-chunk equals one closed GOP (``gop.gop_frames``), the
unit a standard H.264/H.263 CCTV encoder can cut on: every chunk
boundary is an IDR point, so any chunk is independently decodable and
independently fetchable from the archive.

Movement activates chunking the way an encoder inserts an IDR on
demand: each movement event is snapped outward to the GOP grid,
every covered chunk is flagged ``movement = 1`` (the binary
"movement detected" category), and the first covered chunk is
recorded with ``trigger = "motion"``. Chunks without any movement
frame carry ``movement = 0`` and ``trigger = "gop"``. Each chunk
records its start and end timestamps.
"""

from datetime import datetime, timedelta
from typing import List, Sequence

from .config import GopConfig
from .models import ChunkRecord


def build_chunk_records(
    config: GopConfig,
    frame_flags: Sequence[bool],
    start_time: datetime,
    fps: float,
) -> List[ChunkRecord]:
    """Partition a frame run into GOP chunks with movement flags.

    Args:
        config (GopConfig): Validated GOP configuration.
        frame_flags (Sequence[bool]): Per-frame movement flags, one
            per stored frame, in order (event spans with pauses
            bridged - see flow.movement_flags).
        start_time (datetime): Timezone-aware timestamp of frame 0.
        fps (float): Frame rate used to derive chunk timestamps.

    Returns:
        List[ChunkRecord]: One record per GOP chunk, in order.

    Raises:
        ValueError: If no frames are supplied, fps is not positive,
            or the start time is naive.
    """
    if not frame_flags:
        raise ValueError("cannot chunk an empty frame run")
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if start_time.tzinfo is None:
        raise ValueError("start_time must be timezone-aware")

    total = len(frame_flags)
    gop = config.gop_frames
    approach_flags = list(frame_flags)

    records: List[ChunkRecord] = []
    for index, chunk_start in enumerate(range(0, total, gop)):
        chunk_end = min(chunk_start + gop, total)
        window = approach_flags[chunk_start:chunk_end]
        movement = 1 if any(window) else 0
        first_active = window.index(True) if movement else -1
        starts_event = (
            movement == 1 and (
                (chunk_start + first_active == 0)
                or not approach_flags[chunk_start + first_active - 1]
            )
        )
        records.append(ChunkRecord(
            index=index,
            start_frame=chunk_start,
            end_frame=chunk_end,
            start_time=start_time + timedelta(seconds=chunk_start / fps),
            end_time=start_time + timedelta(seconds=chunk_end / fps),
            movement=movement,
            trigger="motion" if starts_event else "gop",
        ))
    return records


def count_events(records: Sequence[ChunkRecord]) -> int:
    """Count chunks that start a movement event.

    Args:
        records (Sequence[ChunkRecord]): Ordered chunk records.

    Returns:
        int: Number of chunks with trigger "motion".
    """
    return sum(1 for r in records if r.trigger == "motion")
