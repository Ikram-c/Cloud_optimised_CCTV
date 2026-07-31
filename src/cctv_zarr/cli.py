#!/usr/bin/env python3
"""CLI: ingest, query, and govern CCTV footage.

Beyond ingest/query/push, the governance verbs keep an operator
compliant: ``prune`` applies the retention policy to stores, the
archive, exports, and the query cache (journalled); ``erase``
honours an Art. 17 request for an explicit time window across
every copy; ``access-request`` produces an Art. 15 copy with
third-party regions redacted, logged with its purpose.
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import audit, retention
from .archive import CloudArchive, build_cipher, get_storage_client
from .config import Settings
from .exceptions import ArchiveError, QueryError
from .query import QueryClient
from .writer import CctvZarrWriter


def _client_and_archive(settings: Settings):
    """Build the archive binding from settings.

    Applies client-side encryption when configured and fails
    closed on a bucket-location mismatch (GDPR Chapter V).

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
    cipher = None
    if settings.security.encrypt_archive:
        cipher = build_cipher(settings.security.encryption_key)
    archive = CloudArchive(
        client, c.gcs_bucket, c.gcs_prefix, cipher=cipher,
    )
    archive.verify_location(c.bucket_location)
    return archive


def _parse_masks(values) -> list:
    """Parse repeated x0,y0,x1,y1 mask arguments.

    Args:
        values: Raw strings from argparse, or None.

    Returns:
        list: Fractional rectangles.

    Raises:
        SystemExit: On malformed regions.
    """
    regions = []
    for text in values or []:
        parts = text.split(",")
        if len(parts) != 4:
            raise SystemExit(
                f"mask must be x0,y0,x1,y1 (got {text!r})",
            )
        regions.append(tuple(float(p) for p in parts))
    return regions


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
    action = retention.handle_source(
        args.video, settings.retention, store_root,
    )
    if action != "keep":
        print(f"source file: {action}")
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
    audit.log_access(
        Path(settings.runtime.store_directory), "query",
        str(args.store), matched=len(selection.records),
        purpose=args.purpose or "",
        exported=bool(args.export_mp4 or args.export_frames),
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


def _cmd_prune(settings: Settings, args) -> int:
    """Apply the retention policy everywhere, journalled."""
    summary = retention.prune_all(settings, cache_dir=args.cache)
    archive = _client_and_archive(settings) if args.archive else None
    for result in summary["stores"]:
        if not result["objects"]:
            continue
        label = (
            "whole store" if result["whole_store"]
            else f"{len(result['objects'])} chunk object(s)"
        )
        print(f"{result['store']}: expired {label}")
        if archive is not None:
            archive.delete_objects(
                result["store"], result["objects"],
            )
        if args.cache:
            retention.invalidate_cache(
                args.cache, result["store"], result["objects"],
            )
    print(
        f"exports swept: {summary['exports_swept']}; "
        f"cache swept: {summary['cache_swept']}"
    )
    return 0


def _cmd_erase(settings: Settings, args) -> int:
    """Erase one time window from every copy (Art. 17)."""
    store_root = Path(settings.runtime.store_directory)
    store_path = store_root / args.store
    if not store_path.is_dir():
        print(f"store not found: {store_path}")
        return 1
    objects = retention.erase_window(
        store_path, args.start, args.end, store_root,
    )
    removed_cache = 0
    if args.cache:
        removed_cache = retention.invalidate_cache(
            args.cache, args.store, objects,
        )
    if args.archive and objects:
        archive = _client_and_archive(settings)
        archive.delete_objects(args.store, objects)
    print(
        f"{args.store}: erased {len(objects)} chunk object(s); "
        f"{removed_cache} cache file(s) removed"
    )
    return 0


def _cmd_access_request(settings: Settings, args) -> int:
    """Produce a redacted Art. 15 copy of one time window."""
    store_root = Path(settings.runtime.store_directory)
    store_path = store_root / args.store
    try:
        client = QueryClient.from_local(store_path)
        selection = client.select(start=args.start, end=args.end)
        masks = _parse_masks(args.mask)
        written = client.export_mp4(
            selection, args.out, mask_regions=masks,
        )
    except QueryError as e:
        print(f"access request failed: {e}")
        return 1
    audit.log_access(
        store_root, "access_request", args.store,
        window=[args.start, args.end],
        destination=str(args.out),
        purpose=args.purpose or "subject access request",
        masked_regions=len(masks),
        frames=written,
    )
    print(
        f"wrote {written} frame(s) to {args.out} "
        f"({len(masks)} masked region(s)); logged"
    )
    return 0


def _add_governance_parsers(sub):
    """Attach the prune, erase, and access-request subcommands.

    Args:
        sub: The subparsers action.
    """
    p_prune = sub.add_parser(
        "prune", help="apply the retention policy (journalled)",
    )
    p_prune.add_argument("--archive", action="store_true",
                         help="also delete expired archive objects")
    p_prune.add_argument("--cache", type=Path, default=None,
                         help="query cache directory to sweep")

    p_erase = sub.add_parser(
        "erase", help="erase a time window from every copy",
    )
    p_erase.add_argument("store")
    p_erase.add_argument("--start", required=True)
    p_erase.add_argument("--end", required=True)
    p_erase.add_argument("--archive", action="store_true",
                         help="also delete the archive objects")
    p_erase.add_argument("--cache", type=Path, default=None,
                         help="query cache directory to purge")

    p_access = sub.add_parser(
        "access-request",
        help="export a redacted subject-access copy (logged)",
    )
    p_access.add_argument("store")
    p_access.add_argument("--start", default=None)
    p_access.add_argument("--end", default=None)
    p_access.add_argument("--out", type=Path, required=True)
    p_access.add_argument("--mask", action="append", default=None,
                          help="redact region x0,y0,x1,y1 "
                               "(fractions; repeatable)")
    p_access.add_argument("--purpose", default=None)


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
    p_query.add_argument("--purpose", default=None,
                         help="reason recorded in the access log")

    p_push = sub.add_parser(
        "push", help="resume uploads interrupted by network loss",
    )
    p_push.add_argument("--verbose", action="store_true",
                        help="also report stores already synced")
    _add_governance_parsers(sub)
    args = parser.parse_args()
    settings = Settings.load(args.config)
    handlers = {
        "ingest": _cmd_ingest,
        "query": _cmd_query,
        "push": _cmd_push,
        "prune": _cmd_prune,
        "erase": _cmd_erase,
        "access-request": _cmd_access_request,
    }
    raise SystemExit(handlers[args.command](settings, args))


if __name__ == "__main__":
    main()
