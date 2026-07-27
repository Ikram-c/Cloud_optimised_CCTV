#!/usr/bin/env python3
"""Run cctv_zarr over a real CCTV dataset and report per-video results.

Designed for a dataset laid out like:

    ~/cctv/
      Videos/                  video files, possibly in class subfolders
      Test_Train_Splits/       optional split lists (testlist*.txt etc.)

Two modes:

- default (analysis-only): stream each video through the optical-flow
  approach detector and the GOP chunker; no store is written. Fast,
  answers "what would be flagged, and where?"
- --full: additionally write the OME-Zarr store, upload it to the
  (mock or real, per config) archive, then run a movement-only query
  back against the archive and verify the fetched section matches the
  manifest - the whole pipeline on real footage.

Outputs one CSV row per video plus a console summary.

Examples:
    python scripts/run_dataset_tests.py --root ~/cctv
    python scripts/run_dataset_tests.py --root ~/cctv \
        --split ~/cctv/Test_Train_Splits/testlist01.txt --limit 20
    python scripts/run_dataset_tests.py --root ~/cctv --full \
        --work-dir dataset_run --max-frames 1500
"""

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

# Allow running from a source checkout without installation.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv_zarr.archive import CloudArchive, get_storage_client  # noqa: E402
from cctv_zarr.capture import VideoCapture, iter_frames, read_info  # noqa: E402
from cctv_zarr.chunker import build_chunk_records, count_events  # noqa: E402
from cctv_zarr.config import Settings  # noqa: E402
from cctv_zarr.exceptions import ArchiveError, VideoOpenError  # noqa: E402
from cctv_zarr.flow import ApproachDetector  # noqa: E402
from cctv_zarr.query import QueryClient  # noqa: E402
from cctv_zarr.writer import CctvZarrWriter  # noqa: E402

logger = logging.getLogger("run_dataset_tests")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".mpg", ".mpeg", ".webm"}

CSV_COLUMNS = [
    "video", "label", "status", "frames", "fps", "duration_s",
    "chunks", "movement_chunks", "movement_ratio", "events",
    "first_event_offset_s", "peak_expansion", "process_fps",
    "store_mb", "query_chunks_fetched", "query_frames", "error",
]


def find_split_file(root: Path) -> Optional[Path]:
    """Pick the most likely test-split list under Test_Train_Splits.

    Args:
        root (Path): Dataset root (contains Test_Train_Splits).

    Returns:
        Optional[Path]: The first file whose name contains 'test'
            (case-insensitive), else None.
    """
    splits_dir = root / "Test_Train_Splits"
    if not splits_dir.is_dir():
        return None
    candidates = sorted(
        p for p in splits_dir.rglob("*")
        if p.is_file() and "test" in p.name.lower()
    )
    return candidates[0] if candidates else None


def parse_split(split_path: Path) -> List[Tuple[str, str]]:
    """Parse a split list into (video reference, label) pairs.

    Tolerates the common dataset formats: one entry per line, the
    first whitespace-separated token being a relative path (with /
    or \\ separators), an optional second token being a label.

    Args:
        split_path (Path): The split list file.

    Returns:
        List[Tuple[str, str]]: (reference, label) pairs; label ''.
    """
    entries = []
    for line in split_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        ref = tokens[0].replace("\\", "/")
        label = tokens[1] if len(tokens) > 1 else ""
        entries.append((ref, label))
    return entries


def resolve_video(root: Path, ref: str) -> Optional[Path]:
    """Resolve a split-file reference to an actual video path.

    Tries, in order: root/Videos/<ref>, root/<ref>, then a recursive
    search for the basename under root/Videos.

    Args:
        root (Path): Dataset root.
        ref (str): Reference from the split file.

    Returns:
        Optional[Path]: The resolved path, or None.
    """
    for candidate in (root / "Videos" / ref, root / ref):
        if candidate.is_file():
            return candidate
    basename = Path(ref).name
    videos_dir = root / "Videos"
    if videos_dir.is_dir():
        matches = sorted(videos_dir.rglob(basename))
        if matches:
            return matches[0]
    return None


def discover_videos(root: Path) -> List[Tuple[Path, str]]:
    """Recursively find all videos under root/Videos (label = folder).

    Args:
        root (Path): Dataset root.

    Returns:
        List[Tuple[Path, str]]: (path, parent-folder label) pairs.
    """
    videos_dir = root / "Videos"
    base = videos_dir if videos_dir.is_dir() else root
    found = sorted(
        p for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return [
        (p, p.parent.name if p.parent != base else "")
        for p in found
    ]


def analyse_video(
    settings: Settings,
    detector: ApproachDetector,
    video_path: Path,
    max_frames: int,
) -> dict:
    """Analysis-only pass: flow decisions + chunk records, no store.

    Args:
        settings (Settings): Root configuration.
        detector (ApproachDetector): Shared detector (reset per video).
        video_path (Path): The video.
        max_frames (int): Hard per-video frame cap.

    Returns:
        dict: Partial CSV row.
    """
    info = read_info(video_path, settings.runtime.default_fps)
    started = time.perf_counter()
    detector.reset()
    flow_results = []
    frame_cap = min(
        info.total_frames, max_frames, settings.gop.max_video_frames,
    )
    with VideoCapture(video_path) as cap:
        for index, frame in iter_frames(
            cap, frame_cap,
            settings.runtime.max_consecutive_fails,
        ):
            flow_results.append(detector.update(frame, index))
    elapsed = time.perf_counter() - started
    if not flow_results:
        raise VideoOpenError(video_path)
    start_time = datetime.fromtimestamp(
        video_path.stat().st_mtime, tz=timezone.utc,
    )
    records = build_chunk_records(
        settings.gop, flow_results, start_time, info.fps,
    )
    movement_chunks = sum(r.movement for r in records)
    first_event = next(
        (r for r in records if r.trigger == "approach"), None,
    )
    active = [r.expansion for r in flow_results if r.approaching]
    return {
        "frames": len(flow_results),
        "fps": round(info.fps, 3),
        "duration_s": round(len(flow_results) / info.fps, 2),
        "chunks": len(records),
        "movement_chunks": movement_chunks,
        "movement_ratio": round(movement_chunks / len(records), 3),
        "events": count_events(records),
        "first_event_offset_s": (
            round(
                (first_event.start_time - start_time).total_seconds(), 2,
            ) if first_event else ""
        ),
        "peak_expansion": (
            round(max(active), 4) if active else ""
        ),
        "process_fps": round(len(flow_results) / elapsed, 1)
        if elapsed > 0 else "",
    }


def full_pipeline(
    settings: Settings,
    video_path: Path,
    work_dir: Path,
) -> dict:
    """Full pass: store -> archive -> movement query verification.

    Args:
        settings (Settings): Root configuration.
        video_path (Path): The video.
        work_dir (Path): Working directory for stores/mock/cache.

    Returns:
        dict: Extra CSV fields (store_mb, query results).

    Raises:
        ArchiveError, VideoOpenError, ValueError: On pipeline failure.
    """
    store_name = video_path.stem + ".zarr"
    store_path = work_dir / "stores" / store_name
    result = CctvZarrWriter(settings).ingest(video_path, store_path)
    store_mb = sum(
        p.stat().st_size for p in store_path.rglob("*") if p.is_file()
    ) / 1e6

    mock_root = (
        Path(settings.archive.local_root)
        if settings.archive.local_root else work_dir / "mock_gcs"
    )
    if settings.archive.use_mock_gcs:
        mock_root = work_dir / "mock_gcs"
    client = get_storage_client(
        use_mock=settings.archive.use_mock_gcs, mock_root=mock_root,
    )
    archive = CloudArchive(
        client, settings.archive.gcs_bucket, settings.archive.gcs_prefix,
    )
    archive.upload_store(store_path, store_name)

    query = QueryClient.from_archive(
        archive, store_name, work_dir / "query_cache",
    )
    selection = query.select(movement=True)
    frames, stamps = query.fetch(selection)
    expected = sum(
        r.end_frame - r.start_frame for r in selection.records
    )
    if len(frames) != expected:
        raise ValueError(
            f"query returned {len(frames)} frames, manifest says {expected}"
        )
    del result
    return {
        "store_mb": round(store_mb, 2),
        "query_chunks_fetched": len(selection.chunk_indices),
        "query_frames": len(frames),
    }


def main() -> int:
    """Entry point."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run cctv_zarr over a CCTV dataset",
    )
    parser.add_argument("--root", type=Path, default=Path.home() / "cctv",
                        help="dataset root containing Videos/ (default ~/cctv)")
    parser.add_argument("--split", type=Path, default=None,
                        help="split list file (default: auto-detect a "
                             "'test' list in Test_Train_Splits, else scan)")
    parser.add_argument("--no-split", action="store_true",
                        help="ignore split files; scan Videos/ directly")
    parser.add_argument("--config", type=Path, default=None,
                        help="cctv_zarr config.yaml (default: package default)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N videos")
    parser.add_argument("--max-frames", type=int, default=3000,
                        help="per-video frame cap (default 3000)")
    parser.add_argument("--full", action="store_true",
                        help="also write stores, upload to the archive, "
                             "and verify a movement query per video")
    parser.add_argument("--work-dir", type=Path,
                        default=Path("dataset_run"),
                        help="working directory for --full artefacts")
    parser.add_argument("--out", type=Path,
                        default=Path("dataset_report.csv"),
                        help="CSV report path (default dataset_report.csv)")
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.is_dir():
        print(f"error: dataset root not found: {root}", file=sys.stderr)
        return 2

    config_path = args.config or (
        Path(__file__).resolve().parent.parent / "config.yaml"
    )
    settings = Settings.load(config_path)
    detector = ApproachDetector(settings.flow)

    # Build the worklist: split file if available, else directory scan.
    videos: List[Tuple[Path, str]] = []
    split_path = None if args.no_split else (
        args.split.expanduser() if args.split else find_split_file(root)
    )
    if split_path is not None and split_path.is_file():
        print(f"split list: {split_path}")
        unresolved = 0
        for ref, label in parse_split(split_path):
            resolved = resolve_video(root, ref)
            if resolved is None:
                unresolved += 1
                continue
            videos.append((resolved, label))
        if unresolved:
            print(f"warning: {unresolved} split entries did not resolve "
                  f"to files under {root}")
    if not videos:
        print(f"scanning {root / 'Videos'} ...")
        videos = discover_videos(root)
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        print("error: no videos found", file=sys.stderr)
        return 2
    print(f"{len(videos)} video(s) to process "
          f"({'full pipeline' if args.full else 'analysis only'}, "
          f"cap {args.max_frames} frames each)\n")

    rows = []
    totals = {"ok": 0, "failed": 0, "with_events": 0,
              "chunks": 0, "movement_chunks": 0}
    for i, (video_path, label) in enumerate(videos, 1):
        rel = (
            video_path.relative_to(root).as_posix()
            if root in video_path.parents else str(video_path)
        )
        row = {c: "" for c in CSV_COLUMNS}
        row.update({"video": rel, "label": label})
        try:
            row.update(analyse_video(
                settings, detector, video_path, args.max_frames,
            ))
            if args.full:
                row.update(full_pipeline(
                    settings, video_path, args.work_dir,
                ))
            row["status"] = "ok"
            totals["ok"] += 1
            totals["chunks"] += row["chunks"]
            totals["movement_chunks"] += row["movement_chunks"]
            if row["events"]:
                totals["with_events"] += 1
        except (VideoOpenError, ArchiveError, ValueError, OSError,
                cv2.error) as e:
            row["status"] = "failed"
            row["error"] = str(e)
            totals["failed"] += 1
        rows.append(row)
        marker = (
            f"{row['events']} event(s), "
            f"{row['movement_chunks']}/{row['chunks']} movement chunks"
            if row["status"] == "ok" else f"FAILED: {row['error']}"
        )
        print(f"[{i}/{len(videos)}] {rel}: {marker}")

    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 60}")
    print(f"processed : {totals['ok']} ok, {totals['failed']} failed")
    if totals["ok"]:
        flagged_pct = (
            100.0 * totals["movement_chunks"] / max(totals["chunks"], 1)
        )
        print(f"movement  : {totals['with_events']}/{totals['ok']} videos "
              f"with approach events; "
              f"{totals['movement_chunks']}/{totals['chunks']} chunks "
              f"flagged ({flagged_pct:.1f}%)")
    print(f"report    : {args.out}")
    if args.full:
        print(f"stores    : {args.work_dir / 'stores'}")
    return 1 if totals["failed"] and not totals["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
