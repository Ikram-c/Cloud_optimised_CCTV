"""Append-only accountability records (GDPR Arts. 5(2), 32).

Two JSONL files live beside the stores: ``access_log.jsonl``
records who fetched, previewed, or exported which chunks and for
what purpose; ``deletion_log.jsonl`` records every governed
deletion (retention pruning, erasure requests, store replacement)
so erasure is provable, not just performed. Both are append-only
and bounded per entry.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

ACCESS_LOG_FILENAME = "access_log.jsonl"
DELETION_LOG_FILENAME = "deletion_log.jsonl"
MAX_ENTRY_CHARS = 4000


def _append(path: Path, entry: dict):
    """Append one timestamped entry to a JSONL file.

    Args:
        path (Path): The log file.
        entry (dict): JSON-safe fields.
    """
    record = dict(entry)
    record["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(record)[:MAX_ENTRY_CHARS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def log_access(root: Path, action: str, store: str, **fields):
    """Record one access to footage.

    Args:
        root (Path): The store directory (log lives beside stores).
        action (str): One of query, fetch, frames, export,
            access_request.
        store (str): Store name.
        **fields: Extra JSON-safe context (chunks, client,
            purpose, destination).
    """
    entry = {"action": action, "store": store}
    entry.update(fields)
    _append(Path(root) / ACCESS_LOG_FILENAME, entry)


def log_deletion(root: Path, reason: str, store: str, **fields):
    """Record one governed deletion.

    Args:
        root (Path): The store directory (log lives beside stores).
        reason (str): One of retention, erasure, replace,
            minimisation, sweep.
        store (str): Store name (or directory swept).
        **fields: Extra JSON-safe context (chunks, paths, window).
    """
    entry = {"reason": reason, "store": store}
    entry.update(fields)
    _append(Path(root) / DELETION_LOG_FILENAME, entry)


def read_log(path: Path) -> list:
    """Read a JSONL log, tolerating absence and damage.

    Args:
        path (Path): The log file.

    Returns:
        list: Parsed entries in order.
    """
    if not Path(path).is_file():
        return []
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except ValueError:
            logger.warning("Damaged log line skipped in %s", path)
    return entries
