"""Bounded, fault-tolerant video reading (adapted from video_zarr)."""

import logging
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from .exceptions import VideoOpenError
from .models import VideoInfo

logger = logging.getLogger(__name__)


class VideoCapture:
    """Context-managed cv2.VideoCapture."""

    __slots__ = ("_path", "_capture")

    def __init__(self, path: Path):
        self._path = path
        self._capture = None

    def __enter__(self) -> cv2.VideoCapture:
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise VideoOpenError(self._path)
        return self._capture

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._capture is not None:
            self._capture.release()
        return False


def read_info(video_path: Path, default_fps: float) -> VideoInfo:
    """Read frame count, fps, and dimensions.

    Args:
        video_path (Path): Video file.
        default_fps (float): Used when the container reports none.

    Returns:
        VideoInfo: Validated properties.

    Raises:
        VideoOpenError: On unreadable files or non-positive counts.
    """
    with VideoCapture(video_path) as cap:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        raise VideoOpenError(video_path)
    return VideoInfo(
        total_frames=total,
        fps=fps if fps > 0 else default_fps,
        width=width, height=height,
    )


def iter_frames(
    capture: cv2.VideoCapture,
    max_frames: int,
    max_fails: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (index, frame), tolerating corrupt mid-file segments.

    Consecutive read failures beyond ``max_fails`` stop iteration;
    a successful read resets the budget. Both bounds are hard.

    Args:
        capture (cv2.VideoCapture): An opened capture.
        max_frames (int): Hard bound on frames yielded.
        max_fails (int): Consecutive failure budget.

    Yields:
        Tuple[int, np.ndarray]: Frame index and BGR frame.
    """
    yielded, fails, pos = 0, 0, 0
    limit = (max_frames + 1) * (max_fails + 1)
    for _ in range(limit):
        if yielded >= max_frames:
            return
        ret, frame = capture.read()
        if not ret or frame is None:
            fails += 1
            if fails > max_fails:
                logger.warning(
                    "Failure budget exhausted at frame %d", pos,
                )
                return
            pos += 1
            continue
        fails = 0
        yield pos, frame
        pos += 1
        yielded += 1
