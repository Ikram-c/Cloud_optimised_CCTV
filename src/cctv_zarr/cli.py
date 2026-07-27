#!/usr/bin/env python3
"""CLI: ingest CCTV footage, push to the archive, query it back."""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from .archive import CloudArchive, get_storage_client
from .config import Settings
from .exceptions import ArchiveError
from .query import QueryClient
from .writer import CctvZarrWriter


def _client_and_archive(settings: Settings):
    """Build the archive binding from settings.

    Args:
        settings (Settings): Root configuration.

    Returns:
        CloudArchive: Bound archive.
    """
    c = settings.archive
    client = get_storage_client(
        use_mock=c.use_mock_gcs,
        mock_root=Path(c.local_root) if c.local_root else None,
    )
    return CloudArchive(client, c.gcs_bucket, c.gcs_prefix)


def _cmd_ingest(settings: Settings, args) -> int:
    """Ingest one video (and optionally upload it)."""
    writer = CctvZarrWriter(settings)
    store_name = args.video.stem
    store_root = Path(settings.runtime.store_directory)
    store_path = store_root / (store_name + ".zarr")
    start_time = (
        datetime.fromisoformat(args.start_time)
        if args.start_time else None
    )
    if start_time is not None and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    result = writer.ingest(args.video, store_path, start_time=start_time)
    print(
        f"{result.store_path}: {result.frames_written} frames, "
        f"{result.chunk_count} chunks, "
        f"{result.movement_chunks} movement, "
        f"{result.motion_events} movement event(s)"
    )
    if args.upload:
        archive = _client_and_archive(settings)
        try:
            count = archive.upload_store(store_path, store_name + ".zarr")
            print(f"uploaded {count} objects")
        except ArchiveError as e:
            pending = CloudArchive.pending_objects(store_path)
            print(f"upload interrupted ({e}); {pending} object(s) "
                  f"still pending - the footage is safe locally, run "
                  f"'cctv-zarr push' to resume")
    return 0


def _cmd_query(settings: Settings, args) -> int:
    """Query a store (local or archived) and optionally export."""
    if args.local:
        client = QueryClient.from_local(args.store)
    else:
        archive = _client_and_archive(settings)
        client = QueryClient.from_archive(
            archive, str(args.store), Path(args.cache),
        )
    movement = True if args.movement_only else None
    selection = client.select(
        start=args.start, end=args.end, movement=movement,
    )
    for r in selection.records:
        print(
            f"chunk {r.index:4d}  {r.start_time.isoformat()} -> "
            f"{r.end_time.isoformat()}  movement={r.movement}  "
            f"trigger={r.trigger}"
        )
    print(f"{len(selection.records)} chunk(s) matched")
    if args.export_mp4 and selection.records:
        n = client.export_mp4(selection, args.export_mp4)
        print(f"wrote {n} frames to {args.export_mp4}")
    if args.export_frames and selection.records:
        n = client.export_frames(selection, args.export_frames)
        print(f"wrote {n} frames to {args.export_frames}")
    return 0


def _cmd_push(settings: Settings, args) -> int:
    """Resume pending uploads for every local store."""
    archive = _client_and_archive(settings)
    store_root = Path(settings.runtime.store_directory)
    if not store_root.is_dir():
        print(f"no stores under {store_root}")
        return 0
    failures = 0
    for store_path in sorted(p for p in store_root.iterdir() if p.is_dir()):
        try:
            count = archive.upload_store(store_path, store_path.name)
            if count:
                print(f"{store_path.name}: resumed, {count} object(s) sent")
            elif args.verbose:
                print(f"{store_path.name}: already synced")
        except ArchiveError as e:
            failures += 1
            pending = CloudArchive.pending_objects(store_path)
            print(f"{store_path.name}: still offline ({e}); "
                  f"{pending} object(s) pending")
    return 1 if failures else 0


def main():
    """Entry point for the cctv-zarr console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="CCTV-optimised OME-Zarr archive tooling",
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="video file -> OME-Zarr store")
    p_ingest.add_argument("video", type=Path)
    p_ingest.add_argument("--start-time", default=None,
                          help="ISO timestamp of frame 0 (default: mtime)")
    p_ingest.add_argument("--upload", action="store_true",
                          help="push the store to the cloud archive")

    p_query = sub.add_parser("query", help="fetch matching chunks")
    p_query.add_argument("store", help="store name (or path with --local)")
    p_query.add_argument("--local", action="store_true",
                         help="treat STORE as a local path")
    p_query.add_argument("--cache", type=Path, default=Path("query_cache"))
    p_query.add_argument("--start", default=None, help="ISO window start")
    p_query.add_argument("--end", default=None, help="ISO window end")
    p_query.add_argument("--movement-only", action="store_true",
                         help="only chunks with movement_detected == 1")
    p_query.add_argument("--export-mp4", type=Path, default=None)
    p_query.add_argument("--export-frames", type=Path, default=None)

    p_push = sub.add_parser(
        "push", help="resume uploads interrupted by network loss",
    )
    p_push.add_argument("--verbose", action="store_true",
                        help="also report stores already synced")

    args = parser.parse_args()
    settings = Settings.load(args.config)
    if args.command == "ingest":
        raise SystemExit(_cmd_ingest(settings, args))
    if args.command == "push":
        raise SystemExit(_cmd_push(settings, args))
    raise SystemExit(_cmd_query(settings, args))


if __name__ == "__main__":
    main()
