"""Ingest one video into an OME-Zarr CCTV store.

Layout produced (all standard Zarr v2 + OME-NGFF 0.4):

    <store>/
      .zgroup .zattrs           group + multiscales + cctv manifest
      0/                        image, uint8, (t, c, z, y, x)
      movement_detected/        uint8 per frame, chunked per GOP
      timestamps/               float64 POSIX seconds per frame

The image's t-chunk equals one GOP, so one archive object holds one
independently decodable, independently fetchable chunk - and the
manifest's per-chunk timestamps and binary movement flags let a query
client fetch exactly the chunks it needs.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import ome, zarr_io
from .capture import VideoCapture, iter_frames, read_info
from .chunker import build_chunk_records
from .config import Settings
from .flow import MotionTracker, movement_flags
from .models import IngestResult

logger = logging.getLogger(__name__)


class CctvZarrWriter:
    """Turns a CCTV video file into a chunk-addressable OME-Zarr store."""

    def __init__(self, settings: Settings):
        """Store settings; the tracker is built per video (fps).

        Args:
            settings (Settings): Root configuration.
        """
        self.settings = settings

    def ingest(
        self,
        video_path: Path,
        store_path: Path,
        start_time: Optional[datetime] = None,
    ) -> IngestResult:
        """Read, analyse, chunk, and write one video.

        Frames stream through in one bounded pass: each frame is
        measured by the motion tracker and buffered per GOP; a full
        GOP flushes as one image chunk, so peak memory is one GOP.
        Movement flags and per-object events are settled at the end,
        when every pause has either been bridged or expired.

        Args:
            video_path (Path): Source video file.
            store_path (Path): Destination store directory.
            start_time (Optional[datetime]): Timezone-aware timestamp
                of frame 0; defaults to the file's mtime in UTC.

        Returns:
            IngestResult: Store statistics.

        Raises:
            VideoOpenError: If the video cannot be opened.
            ValueError: On invalid configuration or empty video.
        """
        video_path = Path(video_path)
        store_path = Path(store_path)
        info = read_info(video_path, self.settings.runtime.default_fps)
        if start_time is None:
            start_time = datetime.fromtimestamp(
                video_path.stat().st_mtime, tz=timezone.utc,
            )
        if start_time.tzinfo is None:
            raise ValueError("start_time must be timezone-aware")

        out_w, out_h = self._output_size(info.width, info.height)
        channels = 1 if self.settings.zarr.grayscale else 3
        gop = self.settings.gop.gop_frames
        total_cap = min(info.total_frames, self.settings.gop.max_video_frames)

        image = zarr_io.ZarrArray.create(
            store_path / ome.IMAGE_PATH,
            shape=(total_cap, channels, 1, out_h, out_w),
            chunks=(gop, channels, 1, out_h, out_w),
            dtype="uint8",
            compression_level=self.settings.zarr.compression_level,
        )

        tracker = MotionTracker(self.settings.flow, fps=info.fps)
        frame_count = 0
        buffer: List[np.ndarray] = []
        written = 0
        with VideoCapture(video_path) as cap:
            for index, frame in iter_frames(
                cap, total_cap,
                self.settings.runtime.max_consecutive_fails,
            ):
                tracker.update(frame, frame_count)
                frame_count += 1
                buffer.append(self._prepare(frame, out_w, out_h, channels))
                if len(buffer) == gop:
                    image.write_chunk(
                        (written // gop, 0, 0, 0, 0), np.stack(buffer),
                    )
                    written += len(buffer)
                    buffer = []
        if buffer:
            image.write_chunk(
                (written // gop, 0, 0, 0, 0), np.stack(buffer),
            )
            written += len(buffer)
        if written == 0:
            raise ValueError(f"no readable frames in {video_path}")
        if written != total_cap:
            self._shrink(image, written, store_path)

        tracker.finish()
        events = tracker.events()
        flags = movement_flags(events, written)
        records = build_chunk_records(
            self.settings.gop, flags, start_time, info.fps,
        )
        self._write_sidecars(store_path, flags, start_time, info.fps)
        zarr_io.write_group(store_path, ome.build_group_attrs(
            name=video_path.stem,
            fps=info.fps,
            gop_frames=gop,
            records=records,
            source_video=str(video_path),
            events=events,
            start_time=start_time,
        ))
        movement_chunks = sum(r.movement for r in records)
        logger.info(
            "%s: %d frames, %d chunks (%d movement), %d movement(s)",
            video_path.name, written, len(records), movement_chunks,
            len(events),
        )
        return IngestResult(
            store_path=str(store_path),
            frames_written=written,
            chunk_count=len(records),
            movement_chunks=movement_chunks,
            motion_events=len(events),
            start_time=start_time,
            fps=info.fps,
        )

    def _output_size(self, width: int, height: int):
        """Resolve the stored frame size.

        Args:
            width (int): Source width.
            height (int): Source height.

        Returns:
            Tuple[int, int]: (width, height) after optional resize.
        """
        resize = self.settings.zarr.resize_width
        if resize is None or resize == width:
            return width, height
        return resize, max(1, int(round(height * resize / width)))

    @staticmethod
    def _shrink(image: zarr_io.ZarrArray, written: int, store_path: Path):
        """Rewrite the image .zarray shape after a short read.

        Args:
            image (zarr_io.ZarrArray): The image array.
            written (int): Frames actually written.
            store_path (Path): Store root (for logging).
        """
        import json
        image.meta["shape"][0] = written
        (image.path / ".zarray").write_text(json.dumps(image.meta, indent=2))
        logger.warning(
            "%s: video shorter than reported; store truncated to %d frames",
            store_path, written,
        )

    def _prepare(
        self, frame: np.ndarray, out_w: int, out_h: int, channels: int,
    ) -> np.ndarray:
        """Resize/convert one frame to the stored (c, z, y, x) layout.

        Args:
            frame (np.ndarray): BGR source frame.
            out_w (int): Stored width.
            out_h (int): Stored height.
            channels (int): 1 (grayscale) or 3 (BGR).

        Returns:
            np.ndarray: uint8 array shaped (c, 1, y, x).
        """
        if (frame.shape[1], frame.shape[0]) != (out_w, out_h):
            frame = cv2.resize(
                frame, (out_w, out_h), interpolation=cv2.INTER_AREA,
            )
        if channels == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return gray[np.newaxis, np.newaxis, :, :]
        return np.transpose(frame, (2, 0, 1))[:, np.newaxis, :, :]

    def _write_sidecars(
        self,
        store_path: Path,
        frame_flags: List[bool],
        start_time: datetime,
        fps: float,
    ):
        """Write the movement_detected and timestamps arrays.

        Args:
            store_path (Path): Store root.
            frame_flags (List[bool]): Per-frame movement flags.
            start_time (datetime): Timestamp of frame 0.
            fps (float): Frame rate.
        """
        total = len(frame_flags)
        gop = self.settings.gop.gop_frames
        movement = zarr_io.ZarrArray.create(
            store_path / ome.MOVEMENT_PATH,
            shape=(total,), chunks=(gop,), dtype="uint8",
            compression_level=self.settings.zarr.compression_level,
        )
        stamps = zarr_io.ZarrArray.create(
            store_path / ome.TIMESTAMPS_PATH,
            shape=(total,), chunks=(gop,), dtype="float64",
            compression_level=self.settings.zarr.compression_level,
        )
        flags = np.array(
            [1 if f else 0 for f in frame_flags], dtype=np.uint8,
        )
        base = start_time.timestamp()
        times = base + np.arange(total, dtype=np.float64) / fps
        for start in range(0, total, gop):
            end = min(start + gop, total)
            movement.write_chunk((start // gop,), flags[start:end])
            stamps.write_chunk((start // gop,), times[start:end])
