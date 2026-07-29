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
        total_cap = min(
            info.total_frames, self.settings.gop.max_video_frames,
        )
        image = self._create_image(
            store_path, total_cap, out_w, out_h, channels,
        )
        tracker = MotionTracker(self.settings.flow, fps=info.fps)
        written = self._stream_frames(
            video_path, image, tracker, total_cap, out_w, out_h,
            channels,
        )
        if written == 0:
            raise ValueError(f"no readable frames in {video_path}")
        if written != total_cap:
            self._shrink(image, written, store_path)
        tracker.finish()
        return self._finalize(
            store_path, video_path, info, written, channels,
            out_w, out_h, start_time, tracker.events(),
        )

    def _create_image(
        self,
        store_path: Path,
        total_cap: int,
        out_w: int,
        out_h: int,
        channels: int,
    ) -> zarr_io.ZarrArray:
        """Create the (t, c, z, y, x) image array, GOP-chunked.

        Args:
            store_path (Path): Store root.
            total_cap (int): Frame capacity.
            out_w (int): Stored width.
            out_h (int): Stored height.
            channels (int): Stored channel count.

        Returns:
            zarr_io.ZarrArray: The created array.
        """
        gop = self.settings.gop.gop_frames
        return zarr_io.ZarrArray.create(
            store_path / ome.IMAGE_PATH,
            shape=(total_cap, channels, 1, out_h, out_w),
            chunks=(gop, channels, 1, out_h, out_w),
            dtype="uint8",
            compression_level=self.settings.zarr.compression_level,
        )

    def _finalize(
        self,
        store_path: Path,
        video_path: Path,
        info,
        written: int,
        channels: int,
        out_w: int,
        out_h: int,
        start_time: datetime,
        events,
    ) -> IngestResult:
        """Write sidecars and metadata, then report the ingest.

        Args:
            store_path (Path): Store root.
            video_path (Path): Source video file.
            info: VideoInfo of the source.
            written (int): Frames written.
            channels (int): Stored channel count.
            out_w (int): Stored width.
            out_h (int): Stored height.
            start_time (datetime): Timestamp of frame 0.
            events: Per-object movement events.

        Returns:
            IngestResult: Store statistics.
        """
        flags = movement_flags(events, written)
        records = build_chunk_records(
            self.settings.gop, flags, start_time, info.fps,
        )
        self._write_sidecars(store_path, flags, start_time, info.fps)
        sizes = self._measure(
            store_path, video_path, info, written, channels,
            out_w, out_h,
        )
        self._write_attrs(
            store_path, video_path, info, records, events,
            start_time, sizes, written,
        )
        return IngestResult(
            store_path=str(store_path),
            frames_written=written,
            chunk_count=len(records),
            movement_chunks=sum(r.movement for r in records),
            motion_events=len(events),
            start_time=start_time,
            fps=info.fps,
            stored_mb=sizes["stored_mb"],
            raw_mb=sizes["raw_mb"],
            source_mb=sizes["source_mb"],
            compression_ratio=sizes["ratio"],
            footprint_ratio=sizes["footprint_ratio"],
        )

    def _write_attrs(
        self,
        store_path: Path,
        video_path: Path,
        info,
        records,
        events,
        start_time: datetime,
        sizes: dict,
        written: int,
    ):
        """Write group attributes and log the ingest summary.

        Args:
            store_path (Path): Store root.
            video_path (Path): Source video file.
            info: VideoInfo of the source.
            records: Chunk manifest records.
            events: Per-object movement events.
            start_time (datetime): Timestamp of frame 0.
            sizes (dict): Compression measurements.
            written (int): Frames written.
        """
        attrs = ome.build_group_attrs(
            name=video_path.stem,
            fps=info.fps,
            gop_frames=self.settings.gop.gop_frames,
            records=records,
            source_video=str(video_path),
            events=events,
            start_time=start_time,
        )
        attrs["cctv"]["compression"] = dict(
            sizes,
            zlib_level=self.settings.zarr.compression_level,
            resize_width=self.settings.zarr.resize_width,
            grayscale=self.settings.zarr.grayscale,
        )
        zarr_io.write_group(store_path, attrs)
        logger.info(
            "%s: %d frames, %d chunks (%d movement), "
            "%d movement(s), %.1fx smaller than raw",
            video_path.name, written, len(records),
            sum(r.movement for r in records), len(events),
            sizes["footprint_ratio"],
        )

    def _stream_frames(
        self,
        video_path: Path,
        image: zarr_io.ZarrArray,
        tracker: MotionTracker,
        total_cap: int,
        out_w: int,
        out_h: int,
        channels: int,
    ) -> int:
        """Stream frames through the tracker into GOP chunks.

        Args:
            video_path (Path): Source video file.
            image (zarr_io.ZarrArray): Destination image array.
            tracker (MotionTracker): Motion tracker to feed.
            total_cap (int): Hard bound on frames read.
            out_w (int): Stored width.
            out_h (int): Stored height.
            channels (int): Stored channel count.

        Returns:
            int: Frames written.
        """
        gop = self.settings.gop.gop_frames
        buffer: List[np.ndarray] = []
        written = 0
        frame_count = 0
        with VideoCapture(video_path) as cap:
            for _, frame in iter_frames(
                cap, total_cap,
                self.settings.runtime.max_consecutive_fails,
            ):
                tracker.update(frame, frame_count)
                frame_count += 1
                buffer.append(
                    self._prepare(frame, out_w, out_h, channels),
                )
                if len(buffer) == gop:
                    image.write_chunk(
                        (written // gop, 0, 0, 0, 0),
                        np.stack(buffer),
                    )
                    written += len(buffer)
                    buffer = []
        if buffer:
            image.write_chunk(
                (written // gop, 0, 0, 0, 0), np.stack(buffer),
            )
            written += len(buffer)
        return written

    @staticmethod
    def _measure(
        store_path: Path,
        video_path: Path,
        info,
        written: int,
        channels: int,
        out_w: int,
        out_h: int,
    ) -> dict:
        """Measure stored size against raw and source baselines.

        Args:
            store_path (Path): Store root directory.
            video_path (Path): Source video file.
            info: VideoInfo of the source.
            written (int): Frames written.
            channels (int): Stored channel count.
            out_w (int): Stored width.
            out_h (int): Stored height.

        Returns:
            dict: stored_mb, raw_mb, source_mb, ratio,
            footprint_ratio.
        """
        stored_bytes = sum(
            p.stat().st_size for p in store_path.rglob("*")
            if p.is_file()
        )
        raw_stored = written * channels * out_h * out_w
        raw_source = written * 3 * info.height * info.width
        source_bytes = video_path.stat().st_size
        return {
            "stored_mb": round(stored_bytes / 1e6, 2),
            "raw_mb": round(raw_source / 1e6, 2),
            "source_mb": round(source_bytes / 1e6, 2),
            "ratio": round(raw_stored / max(stored_bytes, 1), 2),
            "footprint_ratio": round(
                raw_source / max(stored_bytes, 1), 2,
            ),
        }

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
