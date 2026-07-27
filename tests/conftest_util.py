"""Shared synthetic-scene builders for the offline test suite."""

from pathlib import Path

import cv2
import numpy as np

W, H = 320, 240


def textured_background(seed: int = 7) -> np.ndarray:
    """A static, textured BGR background (flow needs texture)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 180, (H, W, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (5, 5), 0)


def draw_square(scene: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    """Stamp a textured square of the given half-size onto a copy."""
    out = scene.copy()
    rng = np.random.default_rng(half)
    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(W, cx + half)
    patch = rng.integers(140, 255, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
    out[y0:y1, x0:x1] = patch
    return out


def approach_frames(n: int, start_half: int = 12, grow: float = 2.5):
    """An object growing frame-on-frame: movement toward the camera."""
    scene = textured_background()
    return [
        draw_square(scene, W // 2, H // 2, int(start_half + i * grow))
        for i in range(n)
    ]


def lateral_frames(n: int, half: int = 24, step: int = 6):
    """An object of fixed size translating: movement across the view."""
    scene = textured_background()
    return [
        draw_square(scene, 40 + i * step, H // 2, half)
        for i in range(n)
    ]


def static_frames(n: int):
    """No motion at all."""
    scene = textured_background()
    return [scene.copy() for _ in range(n)]


def write_video(path: Path, frames, fps: float = 25.0) -> Path:
    """Encode frames to an MP4 file (mp4v)."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    assert writer.isOpened(), "video writer failed to open"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path
