#!/usr/bin/env python3
"""Generate a tiny synthetic CCTV clip for zero-footage demos.

Produces a 12-second 320x240 MP4 (about 1-2 MB): a static textured
scene, then one object crossing the frame with a mid-walk pause
shorter than the tracker's 5-second buffer, then stillness. Ingested
with config.mini.yaml this exercises every demo surface - GOP
chunking, one bridged movement event, movement-only search, and the
compression figures - in a few CPU-seconds, so the whole pipeline
can be shown on a free hosting tier without uploading any footage.

Usage:
    python scripts/make_sample_video.py --out videos/sample.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

WIDTH = 320
HEIGHT = 240
FPS = 25
STATIC_LEAD_S = 3.0
WALK_1_S = 3.0
PAUSE_S = 1.5
WALK_2_S = 2.0
STATIC_TAIL_S = 2.5
BOX_SIZE = 40
SEED = 20260731


def _background() -> np.ndarray:
    """Build a fixed textured background.

    Returns:
        np.ndarray: BGR frame the scene starts from.
    """
    rng = np.random.default_rng(SEED)
    base = rng.integers(60, 120, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (7, 7), 0)


def _object_patch() -> np.ndarray:
    """Build the textured moving object.

    Dense optical flow needs gradient inside the object, not just
    at its edges, so the patch carries strong random texture.

    Returns:
        np.ndarray: BGR patch of BOX_SIZE x BOX_SIZE.
    """
    rng = np.random.default_rng(SEED + 1)
    patch = rng.integers(
        0, 255, (BOX_SIZE, BOX_SIZE, 3), dtype=np.uint8,
    )
    patch[:, :, 2] = np.clip(
        patch[:, :, 2].astype(np.int32) + 80, 0, 255,
    ).astype(np.uint8)
    return patch


def _frame_at(
    background: np.ndarray, patch: np.ndarray, x,
) -> np.ndarray:
    """Render one frame with the moving object at column x.

    Args:
        background (np.ndarray): The static scene.
        patch (np.ndarray): The textured object.
        x: Left edge of the object; None hides it.

    Returns:
        np.ndarray: The rendered BGR frame.
    """
    frame = background.copy()
    if x is None:
        return frame
    top = HEIGHT // 2 - BOX_SIZE // 2
    left = max(0, x)
    right = min(WIDTH, x + BOX_SIZE)
    if right <= left:
        return frame
    frame[top:top + BOX_SIZE, left:right] = (
        patch[:, left - x:right - x]
    )
    return frame


def _positions() -> list:
    """Plan the object's x position for every frame.

    Returns:
        list: One x (or -1 for absent) per frame.
    """
    lead = [None] * int(STATIC_LEAD_S * FPS)
    walk1_n = int(WALK_1_S * FPS)
    start, middle = -BOX_SIZE, WIDTH // 2
    walk1 = [
        start + int((middle - start) * i / walk1_n)
        for i in range(walk1_n)
    ]
    pause = [middle] * int(PAUSE_S * FPS)
    walk2_n = int(WALK_2_S * FPS)
    walk2 = [
        middle + int((WIDTH - middle) * i / walk2_n)
        for i in range(walk2_n)
    ]
    tail = [None] * int(STATIC_TAIL_S * FPS)
    return lead + walk1 + pause + walk2 + tail


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Generate the mini demo clip",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("videos") / "sample_footage.mp4",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
        FPS, (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        print("could not open video writer", file=sys.stderr)
        return 1
    background = _background()
    patch = _object_patch()
    positions = _positions()
    for x in positions:
        writer.write(_frame_at(background, patch, x))
    writer.release()
    size_mb = args.out.stat().st_size / 1e6
    print(
        f"wrote {args.out} ({len(positions)} frames, "
        f"{size_mb:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
