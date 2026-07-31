"""Storage limitation and erasure (GDPR Arts. 5(1)(e), 17).

Retention is enforced chunk-by-chunk from the manifest's per-chunk
timestamps - no video is decoded to expire it. Movement and
non-movement chunks age out on separate clocks; a store whose
every chunk has expired is deleted whole. The same primitives
serve Art. 17 erasure requests (delete an explicit time window,
policy or not). Every deletion lands in the deletion journal, so
an operator can evidence erasure, and expired exports and query
caches are swept on the same run so no copy outlives its chunk.
"""

import logging
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from . import audit, ome, zarr_io
from .config import RetentionConfig, Settings
from .models import ChunkRecord
from .query import _as_time

logger = logging.getLogger(__name__)

MAX_STORES_PER_RUN = 1000
MAX_FILES_PER_SWEEP = 10000
MAX_QUARANTINE_RENAMES = 100
EXPORTS_DIRNAME = "exports"
QUARANTINE_DIRNAME = "_ingested"


def _now(now: Optional[datetime]) -> datetime:
    """Default the clock to current UTC.

    Args:
        now (Optional[datetime]): Injected clock for tests.

    Returns:
        datetime: An aware datetime.
    """
    if now is None:
        return datetime.now(timezone.utc)
    return now


def chunk_expired(
    record: ChunkRecord, policy: RetentionConfig, now: datetime,
) -> bool:
    """Decide whether one chunk has aged out.

    Args:
        record (ChunkRecord): The chunk's manifest row.
        policy (RetentionConfig): The retention policy.
        now (datetime): The clock.

    Returns:
        bool: True when the chunk is past its limit.
    """
    hours = (
        policy.movement_max_age_hours if record.movement
        else policy.non_movement_max_age_hours
    )
    return record.end_time + timedelta(hours=hours) < now


def expired_indices(
    records: List[ChunkRecord],
    policy: RetentionConfig,
    now: datetime,
) -> List[int]:
    """List the chunk indices a policy expires.

    Args:
        records (List[ChunkRecord]): The chunk manifest.
        policy (RetentionConfig): The retention policy.
        now (datetime): The clock.

    Returns:
        List[int]: Expired chunk indices.
    """
    return [
        r.index for r in records if chunk_expired(r, policy, now)
    ]


def delete_chunks(
    store_path: Path,
    indices: List[int],
    reason: str,
    journal_root: Path,
) -> List[str]:
    """Delete image chunks from one store, journalled.

    The manifest and sidecar arrays stay intact; reads of deleted
    ranges return fill values, so queries degrade cleanly.

    Args:
        store_path (Path): Store root directory.
        indices (List[int]): t-chunk indices to delete.
        reason (str): Journal reason (retention or erasure).
        journal_root (Path): Directory holding the journal.

    Returns:
        List[str]: Store-relative paths of deleted objects.
    """
    image = zarr_io.ZarrArray.open(
        Path(store_path) / ome.IMAGE_PATH,
    )
    deleted = []
    for index in indices:
        if image.delete_chunk((index, 0, 0, 0, 0)):
            deleted.append(f"{ome.IMAGE_PATH}/{index}.0.0.0.0")
    if deleted:
        audit.log_deletion(
            journal_root, reason, Path(store_path).name,
            chunks=indices, objects=deleted,
        )
    return deleted


def delete_store(
    store_path: Path, reason: str, journal_root: Path,
) -> List[str]:
    """Delete one whole store, journalled.

    Args:
        store_path (Path): Store root directory.
        reason (str): Journal reason.
        journal_root (Path): Directory holding the journal.

    Returns:
        List[str]: Store-relative paths of deleted objects.
    """
    store_path = Path(store_path)
    objects = [
        p.relative_to(store_path).as_posix()
        for p in sorted(store_path.rglob("*")) if p.is_file()
    ]
    shutil.rmtree(store_path)
    audit.log_deletion(
        journal_root, reason, store_path.name,
        whole_store=True, object_count=len(objects),
    )
    return objects


def prune_store(
    store_path: Path,
    policy: RetentionConfig,
    journal_root: Path,
    now: Optional[datetime] = None,
) -> dict:
    """Apply the retention policy to one store.

    Args:
        store_path (Path): Store root directory.
        policy (RetentionConfig): The retention policy.
        journal_root (Path): Directory holding the journal.
        now (Optional[datetime]): Injected clock for tests.

    Returns:
        dict: store, deleted chunk objects, whole-store flag.
    """
    now = _now(now)
    store_path = Path(store_path)
    attrs = zarr_io.read_attrs(store_path)
    try:
        records = ome.parse_chunk_records(attrs)
    except KeyError:
        return {"store": store_path.name, "objects": [],
                "whole_store": False}
    expired = expired_indices(records, policy, now)
    if len(expired) == len(records) and records:
        objects = delete_store(
            store_path, "retention", journal_root,
        )
        return {"store": store_path.name, "objects": objects,
                "whole_store": True}
    objects = delete_chunks(
        store_path, expired, "retention", journal_root,
    )
    return {"store": store_path.name, "objects": objects,
            "whole_store": False}


def erase_window(
    store_path: Path,
    start,
    end,
    journal_root: Path,
) -> List[str]:
    """Erase every chunk overlapping a time window (Art. 17).

    Args:
        store_path (Path): Store root directory.
        start: Window start (ISO string or datetime).
        end: Window end (ISO string or datetime).
        journal_root (Path): Directory holding the journal.

    Returns:
        List[str]: Store-relative paths of deleted objects.
    """
    attrs = zarr_io.read_attrs(store_path)
    records = ome.parse_chunk_records(attrs)
    start_dt, end_dt = _as_time(start), _as_time(end)
    indices = [
        r.index for r in records if r.overlaps(start_dt, end_dt)
    ]
    return delete_chunks(
        store_path, indices, "erasure", journal_root,
    )


def sweep_directory(
    directory: Path,
    max_age_hours: float,
    journal_root: Path,
    label: str,
    now: Optional[datetime] = None,
) -> int:
    """Delete files older than a limit from one flat directory.

    Args:
        directory (Path): The directory (exports, cache).
        max_age_hours (float): Age limit.
        journal_root (Path): Directory holding the journal.
        label (str): Journal label for the sweep.
        now (Optional[datetime]): Injected clock for tests.

    Returns:
        int: Files deleted.
    """
    now = _now(now)
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    cutoff = now - timedelta(hours=max_age_hours)
    removed = []
    candidates = sorted(directory.rglob("*"))[:MAX_FILES_PER_SWEEP]
    for path in candidates:
        if not path.is_file():
            continue
        modified = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc,
        )
        if modified < cutoff:
            path.unlink()
            removed.append(path.name)
    if removed:
        audit.log_deletion(
            journal_root, "sweep", label, files=removed,
        )
    return len(removed)


def handle_source(
    video_path: Path,
    policy: RetentionConfig,
    journal_root: Path,
) -> str:
    """Apply the post-ingest source-file policy.

    The store never carries audio; the source file may. ``keep``
    retains it (with a warning), ``delete`` removes it
    (journalled), ``quarantine`` moves it into a holding folder
    the exports sweep can expire.

    Args:
        video_path (Path): The ingested source file.
        policy (RetentionConfig): The retention policy.
        journal_root (Path): Directory holding the journal.

    Returns:
        str: The action taken.
    """
    video_path = Path(video_path)
    if policy.source_after_ingest == "keep":
        logger.warning(
            "Source file retained after ingest (may carry an "
            "audio track): %s", video_path,
        )
        return "keep"
    if policy.source_after_ingest == "delete":
        video_path.unlink(missing_ok=True)
        audit.log_deletion(
            journal_root, "source", video_path.name,
            path=str(video_path),
        )
        return "delete"
    quarantine = video_path.parent / QUARANTINE_DIRNAME
    quarantine.mkdir(parents=True, exist_ok=True)
    dest = quarantine / video_path.name
    for attempt in range(1, MAX_QUARANTINE_RENAMES + 1):
        if not dest.exists():
            break
        dest = quarantine / (
            f"{video_path.stem}_{attempt}{video_path.suffix}"
        )
    video_path.rename(dest)
    logger.info("Source quarantined: %s -> %s", video_path, dest)
    return "quarantine"


def invalidate_cache(
    cache_dir: Path, store_name: str, rel_paths: List[str],
) -> int:
    """Remove erased objects from a query cache (Art. 17 reach).

    Args:
        cache_dir (Path): The query cache root.
        store_name (str): Store name within the cache.
        rel_paths (List[str]): Store-relative object paths.

    Returns:
        int: Cache files removed.
    """
    base = Path(cache_dir) / store_name
    removed = 0
    for rel in rel_paths:
        target = base / rel
        if target.is_file():
            target.unlink()
            removed += 1
    return removed


def prune_all(
    settings: Settings,
    cache_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Apply retention to every store, the exports, and the cache.

    Args:
        settings (Settings): Root configuration.
        cache_dir (Optional[Path]): Query cache to sweep.
        now (Optional[datetime]): Injected clock for tests.

    Returns:
        dict: Per-store results plus sweep counts; remote deletion
        of the same objects is the caller's step (CLI --archive).
    """
    now = _now(now)
    policy = settings.retention
    root = Path(settings.runtime.store_directory)
    results = []
    if policy.enabled and root.is_dir():
        stores = sorted(p for p in root.iterdir() if p.is_dir())
        for store_path in stores[:MAX_STORES_PER_RUN]:
            if store_path.name.startswith("_"):
                continue
            if store_path.name == EXPORTS_DIRNAME:
                continue
            results.append(
                prune_store(store_path, policy, root, now=now),
            )
    exports_swept = sweep_directory(
        root / EXPORTS_DIRNAME, policy.exports_max_age_hours,
        root, EXPORTS_DIRNAME, now=now,
    )
    cache_swept = 0
    if cache_dir is not None:
        cache_swept = sweep_directory(
            Path(cache_dir), policy.cache_max_age_hours,
            root, "query_cache", now=now,
        )
    return {
        "stores": results,
        "exports_swept": exports_swept,
        "cache_swept": cache_swept,
    }
