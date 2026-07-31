"""Query the archive and fetch only the chunks that match.

A query never touches pixel data until the selection is known: the
client reads the store's metadata (group attributes with the chunk
manifest, plus the tiny movement and timestamp arrays), resolves a
time window and/or the binary movement category to a set of GOP-chunk
indices, and then fetches exactly those image-chunk objects - nothing
else leaves the archive.

Example:
    archive = CloudArchive(client, "cctv-archive", "sites/gate3")
    q = QueryClient.from_archive(archive, "cam01_20260720", cache_dir)
    sel = q.select(start="2026-07-20T14:03:00+00:00",
                   end="2026-07-20T14:07:00+00:00", movement=True)
    frames, stamps = q.fetch(sel)
    q.export_mp4(sel, Path("event.mp4"))
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from . import ome, zarr_io
from .archive import CloudArchive
from .exceptions import QueryError
from .models import ChunkRecord, QuerySelection

logger = logging.getLogger(__name__)

TimeLike = Union[str, datetime, None]


def _as_time(value: TimeLike) -> Optional[datetime]:
    """Coerce an ISO string or datetime to an aware datetime.

    Args:
        value (TimeLike): ISO-8601 string, datetime, or None.

    Returns:
        Optional[datetime]: Aware datetime (naive input = UTC).

    Raises:
        QueryError: On unparseable strings.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            raise QueryError(f"unparseable timestamp: {value!r}")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _redact(
    frame: np.ndarray, mask_regions: Optional[List[tuple]],
) -> np.ndarray:
    """Black out fractional rectangles of one frame.

    Args:
        frame (np.ndarray): The frame (grayscale or BGR).
        mask_regions (Optional[List[tuple]]): Fractional
            (x0, y0, x1, y1) rectangles in [0, 1].

    Returns:
        np.ndarray: The frame, redacted copies where masked.

    Raises:
        QueryError: On malformed regions.
    """
    if not mask_regions:
        return frame
    height, width = frame.shape[:2]
    out = frame.copy()
    for region in mask_regions:
        if len(region) != 4:
            raise QueryError(
                "mask region must be (x0, y0, x1, y1)",
            )
        x0, y0, x1, y1 = (max(0.0, min(1.0, v)) for v in region)
        if x1 <= x0 or y1 <= y0:
            raise QueryError("mask region is empty")
        out[
            int(y0 * height):max(int(y1 * height), int(y0 * height) + 1),
            int(x0 * width):max(int(x1 * width), int(x0 * width) + 1),
        ] = 0
    return out


class QueryClient:
    """Plans and executes chunk-level queries against one store."""

    def __init__(
        self,
        root: Path,
        records: List[ChunkRecord],
        archive: Optional[CloudArchive] = None,
        store_name: Optional[str] = None,
        events: Optional[List[dict]] = None,
    ):
        """Use from_local or from_archive instead of calling directly."""
        self.root = Path(root)
        self.records = records
        self.archive = archive
        self.store_name = store_name
        self.events = events or []

    @classmethod
    def from_local(cls, store_path: Path) -> "QueryClient":
        """Open a store on the local filesystem.

        Args:
            store_path (Path): Store root directory.

        Returns:
            QueryClient: Ready to select and fetch.

        Raises:
            QueryError: If the store has no cctv chunk manifest.
        """
        attrs = zarr_io.read_attrs(store_path)
        try:
            records = ome.parse_chunk_records(attrs)
        except KeyError:
            raise QueryError(f"not a cctv_zarr store: {store_path}")
        return cls(
            Path(store_path), records,
            events=ome.parse_events(attrs),
        )

    @classmethod
    def from_archive(
        cls,
        archive: CloudArchive,
        store_name: str,
        cache_dir: Path,
    ) -> "QueryClient":
        """Open a store in the cloud archive, fetching metadata only.

        Downloads the group documents, array headers, and the tiny
        movement/timestamp arrays into ``cache_dir``; image chunks
        stay in the archive until a selection asks for them.

        Args:
            archive (CloudArchive): The bound archive.
            store_name (str): Store name under the archive prefix.
            cache_dir (Path): Local mirror for fetched objects.

        Returns:
            QueryClient: Ready to select and fetch.

        Raises:
            QueryError: If the store's manifest is absent/malformed.
        """
        cache = Path(cache_dir) / store_name
        archive.fetch(store_name, [".zgroup", ".zattrs"], cache)
        attrs = zarr_io.read_attrs(cache)
        try:
            records = ome.parse_chunk_records(attrs)
        except KeyError:
            raise QueryError(f"not a cctv_zarr store: {store_name}")
        sidecar_keys = [f"{ome.IMAGE_PATH}/.zarray"]
        for array in (ome.MOVEMENT_PATH, ome.TIMESTAMPS_PATH):
            sidecar_keys.append(f"{array}/.zarray")
            sidecar_keys.extend(
                f"{array}/{i}" for i in range(len(records))
            )
        archive.fetch(store_name, sidecar_keys, cache)
        return cls(
            cache, records, archive=archive, store_name=store_name,
            events=ome.parse_events(attrs),
        )

    def select(
        self,
        start: TimeLike = None,
        end: TimeLike = None,
        movement: Optional[bool] = None,
    ) -> QuerySelection:
        """Resolve a query to the chunks it needs.

        Args:
            start (TimeLike): Window start (inclusive), or None.
            end (TimeLike): Window end (exclusive), or None.
            movement (Optional[bool]): True = only chunks whose binary
                movement category is 1; False = only 0; None = both.

        Returns:
            QuerySelection: Matching chunk indices and records.
        """
        start_dt, end_dt = _as_time(start), _as_time(end)
        matched = [
            r for r in self.records
            if r.overlaps(start_dt, end_dt)
            and (movement is None or r.movement == int(movement))
        ]
        return QuerySelection(
            chunk_indices=tuple(r.index for r in matched),
            records=tuple(matched),
            frame_start=matched[0].start_frame if matched else 0,
            frame_end=matched[-1].end_frame if matched else 0,
        )

    def fetch(
        self, selection: QuerySelection,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Materialise a selection's frames, fetching chunks on demand.

        Args:
            selection (QuerySelection): From select().

        Returns:
            Tuple[List[np.ndarray], np.ndarray]: BGR (or grayscale)
                frames in time order, and their POSIX timestamps.
        """
        if not selection.records:
            return [], np.empty(0, dtype=np.float64)
        self._ensure_chunks(selection.chunk_indices)
        image = zarr_io.ZarrArray.open(self.root / ome.IMAGE_PATH)
        stamps = zarr_io.ZarrArray.open(
            self.root / ome.TIMESTAMPS_PATH
        ).read_full()
        frames: List[np.ndarray] = []
        times: List[np.ndarray] = []
        for record in selection.records:
            chunk = image.read_chunk((record.index, 0, 0, 0, 0))
            valid = record.end_frame - record.start_frame
            for t in range(valid):
                frames.append(self._to_frame(chunk[t]))
            times.append(stamps[record.start_frame:record.end_frame])
        return frames, np.concatenate(times)

    def movement_series(self) -> np.ndarray:
        """The per-frame binary movement category (0/1).

        Returns:
            np.ndarray: uint8 array, one entry per stored frame.
        """
        return zarr_io.ZarrArray.open(
            self.root / ome.MOVEMENT_PATH
        ).read_full()

    def export_frames(
        self,
        selection: QuerySelection,
        out_dir: Path,
        mask_regions: Optional[List[tuple]] = None,
    ) -> int:
        """Write a selection's frames as PNGs named by timestamp.

        Args:
            selection (QuerySelection): From select().
            out_dir (Path): Destination directory.
            mask_regions (Optional[List[tuple]]): Fractional
                (x0, y0, x1, y1) rectangles blacked out of every
                frame - redaction of third parties for subject
                access copies (GDPR Art. 15).

        Returns:
            int: Number of frames written.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames, times = self.fetch(selection)
        for frame, ts in zip(frames, times):
            stamp = datetime.fromtimestamp(ts, tz=timezone.utc)
            name = stamp.strftime("%Y%m%dT%H%M%S_%f") + ".png"
            cv2.imwrite(
                str(out_dir / name),
                _redact(frame, mask_regions),
            )
        return len(frames)

    def export_mp4(
        self,
        selection: QuerySelection,
        out_path: Path,
        fps: float = 25.0,
        mask_regions: Optional[List[tuple]] = None,
    ) -> int:
        """Write a selection's frames as one H.264-family MP4 clip.

        Args:
            selection (QuerySelection): From select().
            out_path (Path): Destination .mp4 path.
            fps (float): Playback rate.
            mask_regions (Optional[List[tuple]]): Fractional
                (x0, y0, x1, y1) rectangles blacked out of every
                frame - redaction of third parties for subject
                access copies (GDPR Art. 15).

        Returns:
            int: Number of frames written.

        Raises:
            QueryError: If the selection is empty or the writer fails.
        """
        frames, _ = self.fetch(selection)
        if not frames:
            raise QueryError("selection matched no frames")
        first = frames[0]
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height), isColor=first.ndim == 3,
        )
        if not writer.isOpened():
            raise QueryError(f"cannot open video writer for {out_path}")
        for frame in frames:
            writer.write(_redact(frame, mask_regions))
        writer.release()
        return len(frames)

    def _ensure_chunks(self, chunk_indices: Tuple[int, ...]):
        """Fetch missing image chunks from the archive, if bound.

        Args:
            chunk_indices (Tuple[int, ...]): Selected t-chunk indices.
        """
        if self.archive is None or self.store_name is None:
            return
        missing = [
            f"{ome.IMAGE_PATH}/{i}.0.0.0.0"
            for i in chunk_indices
            if not (self.root / ome.IMAGE_PATH / f"{i}.0.0.0.0").exists()
        ]
        if missing:
            logger.info(
                "Fetching %d chunk object(s) from the archive", len(missing),
            )
            self.archive.fetch(self.store_name, missing, self.root)

    @staticmethod
    def _to_frame(sample: np.ndarray) -> np.ndarray:
        """Convert a stored (c, 1, y, x) sample to an OpenCV frame.

        Args:
            sample (np.ndarray): One time sample from an image chunk.

        Returns:
            np.ndarray: (y, x) grayscale or (y, x, 3) BGR frame.
        """
        squeezed = sample[:, 0]
        if squeezed.shape[0] == 1:
            return squeezed[0]
        return np.transpose(squeezed, (1, 2, 0)).copy()
