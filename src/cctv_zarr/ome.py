"""OME-NGFF (OME-Zarr v0.4) metadata for CCTV stores.

The image lives at array path ``0`` with axes ``t, c, z, y, x`` per
the NGFF multiscales spec (single resolution level - CCTV archives
gain nothing from pyramids and edge devices should not pay for them).
Alongside it, two coordinate-style 1D arrays:

- ``movement_detected`` - uint8 per frame, the binary category:
  0 = no movement toward the camera, 1 = movement detected.
- ``timestamps`` - float64 POSIX seconds (UTC) per frame.

The chunk manifest (one entry per GOP chunk: frame range, ISO start
and end timestamps, binary movement flag, trigger) is recorded under
the ``cctv`` namespace in the group attributes, so a query client can
plan a partial fetch after reading metadata alone.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Sequence

from .models import ChunkRecord, MotionEvent

NGFF_VERSION = "0.4"
IMAGE_PATH = "0"
MOVEMENT_PATH = "movement_detected"
TIMESTAMPS_PATH = "timestamps"


def _multiscales_entry(name: str, fps: float) -> dict:
    """Build the NGFF multiscales entry for the image.

    Args:
        name (str): Human-readable image name.
        fps (float): Frame rate; the t-axis scale is 1/fps seconds.

    Returns:
        dict: One multiscales list entry.
    """
    return {
        "version": NGFF_VERSION,
        "name": name,
        "axes": [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        "datasets": [{
            "path": IMAGE_PATH,
            "coordinateTransformations": [{
                "type": "scale",
                "scale": [1.0 / fps, 1.0, 1.0, 1.0, 1.0],
            }],
        }],
        "metadata": {
            "description": (
                "CCTV footage; time chunks are closed GOPs "
                "(encoder-aligned) with movement-activated "
                "flagged segments"
            ),
        },
    }


def _serialize_events(
    events: Sequence[MotionEvent],
    start_time: datetime,
    fps: float,
) -> list:
    """Serialise movement events with ISO UTC timestamps.

    Args:
        events (Sequence[MotionEvent]): Per-object movement events.
        start_time (datetime): Timezone-aware timestamp of frame 0.
        fps (float): Frame rate.

    Returns:
        list: JSON-safe event entries.
    """
    return [
        {
            "track_id": e.track_id,
            "start_frame": e.start_frame,
            "end_frame": e.end_frame,
            "start_time": (
                start_time + timedelta(seconds=e.start_frame / fps)
            ).astimezone(timezone.utc).isoformat(),
            "end_time": (
                start_time
                + timedelta(seconds=(e.end_frame + 1) / fps)
            ).astimezone(timezone.utc).isoformat(),
        }
        for e in events
    ]


def _serialize_records(records: Sequence[ChunkRecord]) -> list:
    """Serialise the chunk manifest.

    Args:
        records (Sequence[ChunkRecord]): The chunk manifest.

    Returns:
        list: JSON-safe manifest entries.
    """
    return [
        {
            "index": r.index,
            "start_frame": r.start_frame,
            "end_frame": r.end_frame,
            "start_time": r.start_time.astimezone(
                timezone.utc
            ).isoformat(),
            "end_time": r.end_time.astimezone(
                timezone.utc
            ).isoformat(),
            "movement_detected": r.movement,
            "trigger": r.trigger,
        }
        for r in records
    ]


def build_group_attrs(
    name: str,
    fps: float,
    gop_frames: int,
    records: Sequence[ChunkRecord],
    source_video: str,
    events: Sequence[MotionEvent] = (),
    start_time: datetime = None,
) -> dict:
    """Assemble the store's .zattrs document.

    Args:
        name (str): Human-readable image name.
        fps (float): Frame rate; the t-axis scale is 1/fps seconds.
        gop_frames (int): Frames per GOP chunk.
        records (Sequence[ChunkRecord]): The chunk manifest.
        source_video (str): Provenance path of the source file.
        events (Sequence[MotionEvent]): Per-object movement events.
        start_time (datetime): Timezone-aware timestamp of frame 0;
            required when events are supplied.

    Returns:
        dict: Attributes ready for zarr_io.write_group.

    Raises:
        ValueError: If events are supplied without a start time.
    """
    if events and start_time is None:
        raise ValueError("events require a start_time")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return {
        "multiscales": [_multiscales_entry(name, fps)],
        "cctv": {
            "source_video": source_video,
            "fps": fps,
            "gop_frames": gop_frames,
            "movement_array": MOVEMENT_PATH,
            "timestamps_array": TIMESTAMPS_PATH,
            "events": _serialize_events(events, start_time, fps),
            "chunks": _serialize_records(records),
        },
    }


def parse_chunk_records(attrs: dict) -> list:
    """Rebuild ChunkRecord objects from stored attributes.

    Args:
        attrs (dict): A store's .zattrs contents.

    Returns:
        list: ChunkRecord objects in index order.

    Raises:
        KeyError: If the cctv namespace is absent or malformed.
    """
    entries = attrs["cctv"]["chunks"]
    records = [
        ChunkRecord(
            index=e["index"],
            start_frame=e["start_frame"],
            end_frame=e["end_frame"],
            start_time=datetime.fromisoformat(e["start_time"]),
            end_time=datetime.fromisoformat(e["end_time"]),
            movement=e["movement_detected"],
            trigger=e["trigger"],
        )
        for e in entries
    ]
    return sorted(records, key=lambda r: r.index)


def parse_events(attrs: dict) -> List[dict]:
    """Rebuild movement events from stored attributes.

    Args:
        attrs (dict): A store's .zattrs contents.

    Returns:
        List[dict]: Events with aware datetimes and durations,
            sorted by start time; [] for stores without events.
    """
    entries = attrs.get("cctv", {}).get("events", [])
    events = []
    for e in entries:
        start = datetime.fromisoformat(e["start_time"])
        end = datetime.fromisoformat(e["end_time"])
        events.append({
            "track_id": e["track_id"],
            "start_frame": e["start_frame"],
            "end_frame": e["end_frame"],
            "start_time": start,
            "end_time": end,
            "seconds": round((end - start).total_seconds(), 1),
        })
    return sorted(events, key=lambda e: e["start_time"])
