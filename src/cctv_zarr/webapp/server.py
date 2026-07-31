"""Web control panel: browse, ingest, and query CCTV stores.

FastAPI app serving a single-page panel built to platform interface
guidelines: fluid system typography (base 17px), light and dark
modes with at least 4.5:1 text contrast, safe-area insets, a fixed
49px tab bar with four destinations that becomes a 260-320px
sidebar on wide screens, 44px minimum touch targets, 17px text
fields, pill toggles, contained scroll views, spinners deferred one
second, explicit-dismiss sheets at ten percent inset, an activity
ring for the movement ratio, 44px list rows with inset separators, a
context menu on chunk rows, haptic triggers, a single-series
movement timeline, and implicit auto-save of panel state at most
every thirty seconds.

Videos are selected rather than typed: the panel browses the server
filesystem from the configured video directory, marks files that
already have a store, and ingests either an explicit selection or a
whole folder as one sequential batch with live per-item progress.

Built to be shared: batches from concurrent users wait in a bounded
queue served by one worker instead of turning each other away, each
browser polls its own job and auto-saves its own panel state, and a
configurable per-video upload cap (``ui.max_upload_mb``) is enforced
server-side and shown in the panel before anything is copied.
"""

import argparse
import base64
import binascii
from dataclasses import dataclass
import json
import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .. import audit, ome, retention, zarr_io
from ..config import Settings
from ..exceptions import QueryError, VideoOpenError
from ..models import QuerySelection
from ..query import QueryClient
from ..writer import CctvZarrWriter

logger = logging.getLogger(__name__)

MAX_STORES_LISTED = 500
MAX_FOLDERS_LISTED = 200
MAX_VIDEOS_LISTED = 500
MAX_BATCH_VIDEOS = 500
MAX_ACTIVE_UPLOADS = 8
MAX_UPLOAD_PART_BYTES = 16 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_NAME_COLLISIONS = 100
MAX_JOB_HISTORY = 32
MAX_WORKER_JOBS = 1000
MAX_PREFS_CLIENTS = 64
PREFS_FILENAME = "panel_prefs.json"
EXPORT_DIRNAME = "exports"
UPLOAD_PREFIX = ".upload_"
CLIENT_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
CLIENT_ID_RE = re.compile(CLIENT_ID_PATTERN)
DEFAULT_PREFS_CLIENT = "default"


class BrowseRequest(BaseModel):
    """Folder listing request."""

    path: Optional[str] = None


class UploadRequest(BaseModel):
    """One part of a chunked video upload from the browser."""

    upload_id: str = Field(min_length=1, max_length=64,
                           pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=255)
    seq: int = Field(ge=0)
    last: bool = False
    data: str = ""


class IngestRequest(BaseModel):
    """Batch ingest request: explicit videos, a folder, or both."""

    videos: List[str] = Field(default_factory=list)
    folder: Optional[str] = None


class QueryRequest(BaseModel):
    """Chunk-selection request against one local store."""

    store: str
    start: Optional[str] = None
    end: Optional[str] = None
    movement_only: bool = False
    client: Optional[str] = Field(
        default=None, pattern=CLIENT_ID_PATTERN,
    )


class FramesRequest(BaseModel):
    """Preview request for one chunk of one store."""

    store: str
    chunk_index: int = Field(ge=0)
    client: Optional[str] = Field(
        default=None, pattern=CLIENT_ID_PATTERN,
    )


class ExportRequest(BaseModel):
    """Export request for the chunks a query matches."""

    store: str
    start: Optional[str] = None
    end: Optional[str] = None
    movement_only: bool = False
    fps: float = Field(default=25.0, gt=0.0)
    client: Optional[str] = Field(
        default=None, pattern=CLIENT_ID_PATTERN,
    )


class PrefsRequest(BaseModel):
    """Implicit auto-save payload for one client's panel state."""

    prefs: Dict[str, object] = Field(default_factory=dict)
    client: Optional[str] = Field(
        default=None, pattern=CLIENT_ID_PATTERN,
    )


class JobState:
    """Thread-safe state of one batch-ingest job."""

    def __init__(self, job_id: str, total: int):
        """Initialise a queued job.

        Args:
            job_id (str): Queue-assigned identifier.
            total (int): Number of videos in the batch.
        """
        self._lock = threading.Lock()
        self.job_id = job_id
        self.state = "queued"
        self.total = total
        self.done = 0
        self.current = ""
        self.items: List[dict] = []
        self.error: Optional[str] = None

    def begin(self):
        """Mark the job as running."""
        with self._lock:
            self.state = "running"

    def start_item(self, video: str):
        """Record the video currently being ingested.

        Args:
            video (str): Video path.
        """
        with self._lock:
            self.current = video

    def item_done(self, entry: dict):
        """Record one finished batch item.

        Args:
            entry (dict): Per-video outcome fields.
        """
        with self._lock:
            self.items.append(entry)
            self.done += 1

    def finish(self):
        """Mark the batch complete."""
        with self._lock:
            self.state = "done"
            self.current = ""

    def fail(self, error: str):
        """Record a launch-level failure.

        Args:
            error (str): Human-readable failure description.
        """
        with self._lock:
            self.state = "failed"
            self.error = error

    def snapshot(self) -> dict:
        """Atomically copy the state for the status endpoint.

        Returns:
            dict: job_id, state, total, done, current, items, error.
        """
        with self._lock:
            return {
                "job_id": self.job_id,
                "state": self.state,
                "total": self.total,
                "done": self.done,
                "current": self.current,
                "items": list(self.items),
                "error": self.error,
            }


_IDLE_SNAPSHOT = {
    "job_id": None,
    "state": "idle",
    "total": 0,
    "done": 0,
    "current": "",
    "items": [],
    "error": None,
    "queued_ahead": 0,
}


class JobQueue:
    """Bounded FIFO of batch jobs served by one worker thread.

    Concurrent panel users each submit their own batch; jobs wait in
    line and run one at a time, so simultaneous submissions queue
    instead of failing and never contend for the CPU. Finished jobs
    are kept (bounded) so every user can keep polling their own
    job_id after it completes.
    """

    def __init__(self):
        """Initialise the empty queue."""
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobState] = {}
        self._order: List[str] = []
        self._pending: List[tuple] = []
        self._running: Optional[JobState] = None
        self._worker: Optional[threading.Thread] = None
        self._counter = 0

    def submit(
        self, settings: Settings, videos: List[Path], max_queued: int,
    ) -> Optional[str]:
        """Add a batch to the queue and ensure the worker runs.

        Args:
            settings (Settings): Root configuration for the batch.
            videos (List[Path]): Videos to ingest, in order.
            max_queued (int): Maximum jobs allowed to wait in line.

        Returns:
            Optional[str]: The job id, or None when the queue is full.
        """
        with self._lock:
            if len(self._pending) >= max_queued:
                return None
            self._counter += 1
            job_id = f"job-{self._counter}"
            job = JobState(job_id, len(videos))
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._pending.append((settings, videos, job))
            self._evict_finished()
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._work, daemon=True,
                )
                self._worker.start()
            return job_id

    def _evict_finished(self):
        """Drop oldest finished jobs beyond MAX_JOB_HISTORY."""
        for _ in range(MAX_JOB_HISTORY):
            if len(self._order) <= MAX_JOB_HISTORY:
                return
            terminal = [
                j for j in self._order
                if self._jobs[j].snapshot()["state"]
                in ("done", "failed")
            ]
            if not terminal:
                return
            victim = terminal[0]
            self._order.remove(victim)
            self._jobs.pop(victim, None)

    def _next(self) -> Optional[tuple]:
        """Pop the next task, retiring the worker when idle.

        Returns:
            Optional[tuple]: (settings, videos, job), or None.
        """
        with self._lock:
            self._running = None
            if not self._pending:
                self._worker = None
                return None
            task = self._pending.pop(0)
            self._running = task[2]
            return task

    def _work(self):
        """Worker-thread body: run queued batches in order."""
        for _ in range(MAX_WORKER_JOBS):
            task = self._next()
            if task is None:
                return
            settings, videos, job = task
            job.begin()
            try:
                _run_batch(settings, videos, job)
            except (OSError, ValueError, cv2.error) as e:
                logger.exception("Batch worker failed")
                job.fail(str(e))
        with self._lock:
            self._worker = None
            self._running = None

    def snapshot(self, job_id: Optional[str]) -> dict:
        """Report one job's state, defaulting to the newest job.

        Args:
            job_id (Optional[str]): Job to report, or None.

        Returns:
            dict: JobState snapshot plus queued_ahead, or idle.
        """
        with self._lock:
            if job_id is None:
                job_id = self._order[-1] if self._order else None
            job = self._jobs.get(job_id) if job_id else None
            if job is None:
                return dict(_IDLE_SNAPSHOT)
            ahead = 0
            if self._running is not None and self._running is not job:
                ahead += 1
            for position, task in enumerate(self._pending):
                if task[2] is job:
                    ahead += position
                    break
        snap = job.snapshot()
        snap["queued_ahead"] = ahead if snap["state"] == "queued" else 0
        return snap


def _store_summary(store_path: Path) -> Optional[dict]:
    """Summarise one store from its manifest alone.

    Args:
        store_path (Path): Store root directory.

    Returns:
        Optional[dict]: Summary fields, or None when unreadable.
    """
    try:
        attrs = zarr_io.read_attrs(store_path)
        records = ome.parse_chunk_records(attrs)
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        logger.warning("Unreadable store skipped: %s", store_path)
        return None
    if not records:
        return None
    movement = sum(r.movement for r in records)
    size = sum(
        p.stat().st_size for p in store_path.rglob("*") if p.is_file()
    )
    return {
        "name": store_path.name,
        "chunks": len(records),
        "movement_chunks": movement,
        "movement_ratio": round(movement / len(records), 3),
        "frames": records[-1].end_frame,
        "fps": attrs["cctv"]["fps"],
        "start_time": records[0].start_time.isoformat(),
        "end_time": records[-1].end_time.isoformat(),
        "size_mb": round(size / 1e6, 2),
        "seconds": round(
            records[-1].end_frame / attrs["cctv"]["fps"], 1,
        ),
        "compression": attrs["cctv"].get("compression"),
    }


def _record_dict(record) -> dict:
    """Serialise one ChunkRecord for the panel.

    Args:
        record: The ChunkRecord.

    Returns:
        dict: JSON-safe fields.
    """
    return {
        "index": record.index,
        "start_frame": record.start_frame,
        "end_frame": record.end_frame,
        "start_time": record.start_time.isoformat(),
        "end_time": record.end_time.isoformat(),
        "movement_detected": record.movement,
        "trigger": record.trigger,
    }


def _discover_folder_videos(folder: Path, extensions: tuple) -> List[Path]:
    """List a folder's videos, non-recursively, sorted and bounded.

    Args:
        folder (Path): The folder.
        extensions (tuple): Accepted suffixes, lowercase.

    Returns:
        List[Path]: At most MAX_BATCH_VIDEOS video paths.
    """
    allowed = {e.lower() for e in extensions}
    found = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    )
    return found[:MAX_BATCH_VIDEOS]


def _run_batch(settings: Settings, videos: List[Path], job: JobState):
    """Daemon-thread body: ingest a batch sequentially.

    Args:
        settings (Settings): Root configuration.
        videos (List[Path]): Videos to ingest, in order.
        job (JobState): Shared job state.
    """
    writer = CctvZarrWriter(settings)
    store_root = Path(settings.runtime.store_directory)
    for video in videos[:MAX_BATCH_VIDEOS]:
        job.start_item(video.name)
        entry: Dict[str, object] = {"video": video.name}
        store = store_root / (video.stem + ".zarr")
        try:
            if store.is_dir():
                _tombstone_store(store_root, store)
            result = writer.ingest(video, store)
            retention.handle_source(
                video, settings.retention, store_root,
            )
            entry.update({
                "store": store.name,
                "seconds": round(
                    result.frames_written / result.fps, 1,
                ),
                "frames": result.frames_written,
                "chunks": result.chunk_count,
                "movement_chunks": result.movement_chunks,
                "events": result.motion_events,
                "stored_mb": result.stored_mb,
                "footprint_ratio": result.footprint_ratio,
            })
        except (VideoOpenError, ValueError, OSError, cv2.error) as e:
            logger.exception("Ingest failed for %s", video)
            entry["error"] = str(e)
        job.item_done(entry)
    job.finish()


def _check_auth(settings: Settings, authorization: Optional[str]):
    """Enforce the panel access token when one is configured.

    Args:
        settings (Settings): Root configuration.
        authorization (Optional[str]): The Authorization header.

    Raises:
        HTTPException: 401 when the token is missing or wrong.
    """
    token = settings.security.auth_token
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(
            status_code=401,
            detail="An access code is required.",
        )


def _within(path: Path, root: Path) -> bool:
    """Whether a resolved path sits inside a resolved root.

    Args:
        path (Path): Candidate path (resolved).
        root (Path): Confinement root (resolved).

    Returns:
        bool: True for the root itself or any descendant.
    """
    return path == root or root in path.parents


def _confine_or_400(path: Path, root: Path, what: str) -> Path:
    """Resolve a path and require it inside a root directory.

    Args:
        path (Path): Candidate path.
        root (Path): Confinement root.
        what (str): Human label for the error message.

    Returns:
        Path: The resolved path.

    Raises:
        HTTPException: When outside the root or unresolvable.
    """
    try:
        resolved = Path(path).expanduser().resolve()
        root = Path(root).resolve()
    except OSError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not _within(resolved, root):
        raise HTTPException(
            status_code=400,
            detail=f"That {what} is outside the video area.",
        )
    return resolved


def _settings_or_400(config_path: Path) -> Settings:
    """Load settings or raise a 400.

    Args:
        config_path (Path): Path to config.yaml.

    Returns:
        Settings: Validated settings.

    Raises:
        HTTPException: When the config cannot be loaded.
    """
    try:
        return Settings.load(config_path)
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


def _store_root(settings: Settings, name: str) -> Path:
    """Resolve and validate a store name inside store_directory.

    Args:
        settings (Settings): Root configuration.
        name (str): Store directory name.

    Returns:
        Path: The store root.

    Raises:
        HTTPException: On traversal attempts or missing stores.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(
            status_code=400,
            detail="That recording name is not valid.",
        )
    root = Path(settings.runtime.store_directory) / name
    if not root.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"That recording was not found: {name}",
        )
    return root


def _open_query(settings: Settings, name: str) -> QueryClient:
    """Open a local store for querying.

    Args:
        settings (Settings): Root configuration.
        name (str): Store directory name.

    Returns:
        QueryClient: Bound client.

    Raises:
        HTTPException: When the store is not a cctv_zarr store.
    """
    root = _store_root(settings, name)
    try:
        return QueryClient.from_local(root)
    except QueryError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _validate_upload_name(settings: Settings, name: str) -> str:
    """Validate a picked file name and its extension.

    Args:
        settings (Settings): Root configuration.
        name (str): The browser-supplied file name.

    Returns:
        str: The safe file name.

    Raises:
        HTTPException: On unsafe names or non-video extensions.
    """
    if Path(name).name != name or name.startswith("."):
        raise HTTPException(
            status_code=400,
            detail="That file name is not allowed.",
        )
    allowed = {e.lower() for e in settings.runtime.video_extensions}
    if Path(name).suffix.lower() not in allowed:
        raise HTTPException(
            status_code=400,
            detail="That file does not look like a video.",
        )
    return name


def _final_upload_path(folder: Path, name: str) -> Path:
    """Pick a non-colliding destination for an uploaded video.

    Args:
        folder (Path): The video directory.
        name (str): The safe file name.

    Returns:
        Path: A free destination path.

    Raises:
        HTTPException: When too many name collisions exist.
    """
    candidate = folder / name
    stem, suffix = candidate.stem, candidate.suffix
    for attempt in range(1, MAX_NAME_COLLISIONS + 1):
        if not candidate.exists():
            return candidate
        candidate = folder / f"{stem}_{attempt}{suffix}"
    raise HTTPException(
        status_code=400,
        detail="Too many files with that name already exist.",
    )


def _read_prefs_file(path: Path) -> Dict[str, dict]:
    """Read all clients' saved panel state.

    Args:
        path (Path): The prefs file.

    Returns:
        Dict[str, dict]: Per-client prefs; legacy flat files map
        to the default client.
    """
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    clients = data.get("clients")
    if isinstance(clients, dict):
        return {
            k: v for k, v in clients.items()
            if isinstance(v, dict)
        }
    return {DEFAULT_PREFS_CLIENT: data}


def _upload_limit_bytes(settings: Settings) -> int:
    """Resolve the per-video upload byte limit.

    Args:
        settings (Settings): Root configuration.

    Returns:
        int: The effective limit in bytes.
    """
    limit = MAX_UPLOAD_TOTAL_BYTES
    if settings.ui.max_upload_mb > 0:
        limit = min(
            limit, settings.ui.max_upload_mb * 1000 * 1000,
        )
    return limit


def _decode_part(data: str) -> bytes:
    """Decode and bound one base64 upload part.

    Args:
        data (str): Base64 payload.

    Returns:
        bytes: Decoded bytes.

    Raises:
        HTTPException: On invalid base64 or oversized parts.
    """
    try:
        payload = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=400,
            detail="That upload was not readable.",
        )
    if len(payload) > MAX_UPLOAD_PART_BYTES:
        raise HTTPException(
            status_code=400,
            detail="That upload part is too large.",
        )
    return payload


@dataclass(slots=True)
class PanelContext:
    """Shared mutable state for one panel application."""

    config_path: Path
    queue: JobQueue
    uploads: Dict[str, dict]
    uploads_lock: threading.Lock
    prefs_lock: threading.Lock


def _upload_begin(
    ctx: PanelContext, req: UploadRequest, name: str, temp: Path,
) -> dict:
    """Open a new upload slot for part zero.

    Args:
        ctx (PanelContext): Panel state; uploads_lock is held.
        req (UploadRequest): The first part.
        name (str): Validated file name.
        temp (Path): Hidden staging file.

    Returns:
        dict: The new upload entry.

    Raises:
        HTTPException: On a non-zero first part or full slots.
    """
    if req.seq != 0:
        raise HTTPException(
            status_code=400,
            detail="That upload was interrupted. Please try again.",
        )
    if len(ctx.uploads) >= MAX_ACTIVE_UPLOADS:
        raise HTTPException(
            status_code=409,
            detail="Too many uploads at once. Please wait a moment.",
        )
    entry = {"name": name, "next_seq": 0, "bytes": 0}
    ctx.uploads[req.upload_id] = entry
    temp.write_bytes(b"")
    return entry


def _upload_apply_part(
    ctx: PanelContext,
    settings: Settings,
    req: UploadRequest,
    name: str,
    payload: bytes,
    temp: Path,
) -> dict:
    """Append one part; finalise on the last part.

    Args:
        ctx (PanelContext): Panel state; uploads_lock is held.
        settings (Settings): Root configuration.
        req (UploadRequest): The part.
        name (str): Validated file name.
        payload (bytes): Decoded part bytes.
        temp (Path): Hidden staging file.

    Returns:
        dict: Progress or completion payload.

    Raises:
        HTTPException: On sequence gaps or oversized videos.
    """
    entry = ctx.uploads.get(req.upload_id)
    if entry is None:
        entry = _upload_begin(ctx, req, name, temp)
    if req.seq != entry["next_seq"]:
        ctx.uploads.pop(req.upload_id, None)
        temp.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="That upload was interrupted. Please try again.",
        )
    if entry["bytes"] + len(payload) > _upload_limit_bytes(settings):
        ctx.uploads.pop(req.upload_id, None)
        temp.unlink(missing_ok=True)
        detail = "That video is too large to copy."
        if settings.ui.max_upload_mb > 0:
            detail = (
                f"That video is over the "
                f"{settings.ui.max_upload_mb} MB limit."
            )
        raise HTTPException(status_code=400, detail=detail)
    with temp.open("ab") as handle:
        handle.write(payload)
    entry["next_seq"] += 1
    entry["bytes"] += len(payload)
    if not req.last:
        return {"done": False, "received": entry["bytes"]}
    ctx.uploads.pop(req.upload_id, None)
    final = _final_upload_path(temp.parent, entry["name"])
    temp.rename(final)
    return {"done": True, "path": str(final)}


def _collect_batch_videos(
    settings: Settings, req: IngestRequest,
) -> List[Path]:
    """Resolve and deduplicate the requested batch videos.

    Args:
        settings (Settings): Root configuration.
        req (IngestRequest): The batch request.

    Returns:
        List[Path]: Unique, existing videos in order.

    Raises:
        HTTPException: On missing folders/videos or empty requests.
    """
    video_root = Path(settings.runtime.video_directory)
    videos: List[Path] = []
    if req.folder:
        folder = _confine_or_400(
            Path(req.folder), video_root, "folder",
        )
        if not folder.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"That folder was not found: {folder}",
            )
        videos.extend(_discover_folder_videos(
            folder, settings.runtime.video_extensions,
        ))
    for raw in req.videos[:MAX_BATCH_VIDEOS]:
        path = _confine_or_400(Path(raw), video_root, "video")
        if not path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"That video was not found: {path}",
            )
        videos.append(path)
    unique: List[Path] = []
    seen = set()
    for path in videos:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise HTTPException(
            status_code=400,
            detail="Choose at least one video first.",
        )
    return unique


def _encode_previews(images, stamps, settings: Settings) -> List[dict]:
    """JPEG-encode a bounded preview strip of one section.

    Args:
        images: Frames of the fetched section.
        stamps: Per-frame POSIX timestamps.
        settings (Settings): Root configuration.

    Returns:
        List[dict]: base64 JPEGs with timestamps.
    """
    limit = settings.ui.preview_max_frames
    stride = max(1, len(images) // limit)
    encoded: List[dict] = []
    for position in range(0, len(images), stride):
        frame = images[position]
        width = settings.ui.preview_width
        if frame.shape[1] > width:
            height = max(
                1, int(frame.shape[0] * width / frame.shape[1]),
            )
            frame = cv2.resize(
                frame, (width, height),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72],
        )
        if not ok:
            continue
        encoded.append({
            "jpeg": base64.b64encode(
                buffer.tobytes()
            ).decode("ascii"),
            "timestamp": float(stamps[position]),
        })
    return encoded


def _add_basic_routes(app: FastAPI, ctx: PanelContext):
    """Register the panel page, status, and config routes.

    Args:
        app (FastAPI): The application.
        ctx (PanelContext): Shared panel state.
    """

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """Serve the panel."""
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/status")
    def status(
        job: Optional[str] = None,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """Report one batch job's state, newest by default."""
        settings = _settings_or_400(ctx.config_path)
        _check_auth(settings, authorization)
        return ctx.queue.snapshot(job)

    @app.get("/api/config")
    def config_info(
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """Expose the panel-relevant limits."""
        settings = _settings_or_400(ctx.config_path)
        _check_auth(settings, authorization)
        return {
            "max_upload_mb": settings.ui.max_upload_mb,
            "max_queued_jobs": settings.ui.max_queued_jobs,
        }


def _add_stores_route(app: FastAPI, ctx: PanelContext):
    """Register the store-listing route.

    Args:
        app (FastAPI): The application.
        ctx (PanelContext): Shared panel state.
    """

    @app.get("/api/stores")
    def stores(
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """List readable stores under store_directory."""
        settings = _settings_or_400(ctx.config_path)
        _check_auth(settings, authorization)
        root = Path(settings.runtime.store_directory)
        found: List[dict] = []
        if root.is_dir():
            candidates = sorted(
                p for p in root.iterdir()
                if p.is_dir() and not p.name.startswith("_")
            )
            for path in candidates[:MAX_STORES_LISTED]:
                summary = _store_summary(path)
                if summary is not None:
                    found.append(summary)
        return {"stores": found}


def _add_browse_route(app: FastAPI, ctx: PanelContext):
    """Register the folder-browsing route.

    Args:
        app (FastAPI): The application.
        ctx (PanelContext): Shared panel state.
    """

    @app.post("/api/browse")
    def browse(
        req: BrowseRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """List one folder's subfolders and selectable videos."""
        settings = _settings_or_400(ctx.config_path)
        _check_auth(settings, authorization)
        video_root = Path(settings.runtime.video_directory)
        base = _confine_or_400(
            Path(req.path) if req.path else video_root,
            video_root, "folder",
        )
        if not base.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"That folder was not found: {base}",
            )
        store_root = Path(settings.runtime.store_directory)
        try:
            children = sorted(base.iterdir())
        except OSError as e:
            raise HTTPException(status_code=400, detail=str(e))
        folders = [
            p.name for p in children
            if p.is_dir() and not p.name.startswith(".")
        ][:MAX_FOLDERS_LISTED]
        allowed = {
            e.lower() for e in settings.runtime.video_extensions
        }
        videos = []
        for p in children:
            if len(videos) >= MAX_VIDEOS_LISTED:
                break
            if not p.is_file() or p.suffix.lower() not in allowed:
                continue
            videos.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(p.stat().st_size / 1e6, 2),
                "ingested": (
                    store_root / (p.stem + ".zarr")
                ).is_dir(),
            })
        return {
            "path": str(base),
            "parent": str(base.parent),
            "folders": folders,
            "videos": videos,
        }


def _add_upload_route(app: FastAPI, ctx: PanelContext):
    """Register the chunked-upload route.

    Args:
        app (FastAPI): The application.
        ctx (PanelContext): Shared panel state.
    """

    @app.post("/api/upload")
    def upload(
        req: UploadRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """Receive one part of a clicked-or-dropped video upload.

        Parts arrive in order per upload_id and append to a hidden
        temporary file in the video directory; the last part renames
        it to its final name and returns the saved path.
        """
        settings = _settings_or_400(ctx.config_path)
        _check_auth(settings, authorization)
        name = _validate_upload_name(settings, req.name)
        payload = _decode_part(req.data)
        folder = Path(settings.runtime.video_directory)
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder / (UPLOAD_PREFIX + req.upload_id)
        with ctx.uploads_lock:
            return _upload_apply_part(
                ctx, settings, req, name, payload, temp,
            )


def _add_ingest_route(app: FastAPI, ctx: PanelContext):
    """Register the batch-ingest route.

    Args:
        app (FastAPI): The application.
        ctx (PanelContext): Shared panel state.
    """

    @app.post("/api/ingest")
    def ingest(
        req: IngestRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """Queue a batch ingest for the worker thread."""
        settings = _settings_or_400(ctx.config_path)
        _check_auth(settings, authorization)
        unique = _collect_batch_videos(settings, req)
        job_id = ctx.queue.submit(
            settings, unique, settings.ui.max_queued_jobs,
        )
        if job_id is None:
            raise HTTPException(
                status_code=409, detail="The waiting line is full "
                       "right now. Please try again in a moment.",
            )
        return {"started": True, "count": len(unique),
                "job_id": job_id}


def _add_query_route(app: FastAPI, ctx: PanelContext):
    """Register the chunk-query route.

    Args:
        app (FastAPI): The application.
        ctx (PanelContext): Shared panel state.
    """

    @app.post("/api/query")
    def query(
        req: QueryRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """Resolve a time/movement query to chunk records."""
        settings = _settings_or_400(ctx.config_path)
        _check_auth(settings, authorization)
        client = _open_query(settings, req.store)
        movement = True if req.movement_only else None
        try:
            selection = client.select(
                start=req.start, end=req.end, movement=movement,
            )
        except QueryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        audit.log_access(
            Path(settings.runtime.store_directory), "query",
            req.store, matched=len(selection.records),
            client=req.client or "",
        )
        all_records = client.records
        return {
            "records": [
                _record_dict(r) for r in selection.records
            ],
            "all_chunks": [_record_dict(r) for r in all_records],
            "events": [
                {
                    "start_time": e["start_time"].isoformat(),
                    "end_time": e["end_time"].isoformat(),
                    "seconds": e["seconds"],
                }
                for e in client.events
            ],
            "matched": len(selection.records),
            "total": len(all_records),
        }


def _add_frames_route(app: FastAPI, ctx: PanelContext):
    """Register the section-preview route.

    Args:
        app (FastAPI): The application.
        ctx (PanelContext): Shared panel state.
    """

    @app.post("/api/frames")
    def frames(
        req: FramesRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """Return JPEG previews for one chunk's section."""
        settings = _settings_or_400(ctx.config_path)
        _check_auth(settings, authorization)
        audit.log_access(
            Path(settings.runtime.store_directory), "frames",
            req.store, chunk=req.chunk_index,
            client=req.client or "",
        )
        client = _open_query(settings, req.store)
        matched = [
            r for r in client.records
            if r.index == req.chunk_index
        ]
        if not matched:
            raise HTTPException(
                status_code=404,
                detail="That section was not found.",
            )
        selection = QuerySelection(
            chunk_indices=(matched[0].index,),
            records=(matched[0],),
            frame_start=matched[0].start_frame,
            frame_end=matched[0].end_frame,
        )
        images, stamps = client.fetch(selection)
        return {
            "frames": _encode_previews(images, stamps, settings),
            "chunk": _record_dict(matched[0]),
        }


def _add_export_route(app: FastAPI, ctx: PanelContext):
    """Register the clip-export route.

    Args:
        app (FastAPI): The application.
        ctx (PanelContext): Shared panel state.
    """

    @app.post("/api/export")
    def export(
        req: ExportRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """Export the matching section as an MP4 clip (logged)."""
        settings = _settings_or_400(ctx.config_path)
        _check_auth(settings, authorization)
        client = _open_query(settings, req.store)
        movement = True if req.movement_only else None
        try:
            selection = client.select(
                start=req.start, end=req.end, movement=movement,
            )
        except QueryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not selection.records:
            raise HTTPException(
                status_code=400,
                detail="Nothing matched that search.",
            )
        out_dir = (
            Path(settings.runtime.store_directory) / EXPORT_DIRNAME
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = selection.records[0].start_time.strftime(
            "%Y%m%dT%H%M%S"
        )
        out_path = out_dir / f"{req.store}_{stamp}.mp4"
        try:
            written = client.export_mp4(
                selection, out_path, fps=req.fps,
            )
        except QueryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        store_root = Path(settings.runtime.store_directory)
        audit.log_access(
            store_root, "export", req.store,
            destination=str(out_path), frames=written,
            client=req.client or "",
        )
        retention.sweep_directory(
            out_dir, settings.retention.exports_max_age_hours,
            store_root, EXPORT_DIRNAME,
        )
        return {
            "path": str(out_path),
            "frames": written,
            "chunks": len(selection.records),
        }


def _add_prefs_routes(app: FastAPI, ctx: PanelContext):
    """Register the panel-state read and write routes.

    Args:
        app (FastAPI): The application.
        ctx (PanelContext): Shared panel state.
    """

    @app.get("/api/prefs")
    def read_prefs(
        client: Optional[str] = None,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """Return one client's auto-saved panel state."""
        try:
            settings = Settings.load(ctx.config_path)
        except (FileNotFoundError, KeyError, ValueError):
            return {"prefs": {}}
        _check_auth(settings, authorization)
        key = (
            client if client and CLIENT_ID_RE.match(client)
            else DEFAULT_PREFS_CLIENT
        )
        path = (
            Path(settings.runtime.store_directory) / PREFS_FILENAME
        )
        if not path.is_file():
            return {"prefs": {}}
        with ctx.prefs_lock:
            return {"prefs": _read_prefs_file(path).get(key, {})}

    @app.post("/api/prefs")
    def write_prefs(
        req: PrefsRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> dict:
        """Persist one client's auto-saved panel state."""
        settings = _settings_or_400(ctx.config_path)
        _check_auth(settings, authorization)
        root = Path(settings.runtime.store_directory)
        root.mkdir(parents=True, exist_ok=True)
        key = req.client or DEFAULT_PREFS_CLIENT
        path = root / PREFS_FILENAME
        with ctx.prefs_lock:
            clients = _read_prefs_file(path)
            clients.pop(key, None)
            clients[key] = req.prefs
            names = list(clients)
            excess = max(len(names) - MAX_PREFS_CLIENTS, 0)
            for name in names[:excess]:
                clients.pop(name, None)
            path.write_text(json.dumps({"clients": clients}))
        return {"saved": True}


_ROUTE_REGISTRARS = (
    _add_basic_routes,
    _add_stores_route,
    _add_browse_route,
    _add_upload_route,
    _add_ingest_route,
    _add_query_route,
    _add_frames_route,
    _add_export_route,
    _add_prefs_routes,
)


def _tombstone_store(store_root: Path, store: Path):
    """Move an existing store aside instead of destroying it.

    Re-ingest never silently erases prior footage: the old store
    moves under ``_replaced`` and the move lands in the deletion
    journal, so an operator can evidence what the archive held.

    Args:
        store_root (Path): The stores directory.
        store (Path): The store being replaced.
    """
    graveyard = store_root / "_replaced"
    graveyard.mkdir(parents=True, exist_ok=True)
    dest = graveyard / store.name
    for attempt in range(1, MAX_NAME_COLLISIONS + 1):
        if not dest.exists():
            break
        dest = graveyard / f"{store.name}.{attempt}"
    store.rename(dest)
    audit.log_deletion(
        store_root, "replace", store.name, moved_to=str(dest),
    )


def create_app(config_path: Path) -> FastAPI:
    """Build the control-panel app bound to one config file.

    Args:
        config_path (Path): Path to config.yaml.

    Returns:
        FastAPI: The application.
    """
    app = FastAPI(title="CCTV Zarr")
    ctx = PanelContext(
        config_path=Path(config_path),
        queue=JobQueue(),
        uploads={},
        uploads_lock=threading.Lock(),
        prefs_lock=threading.Lock(),
    )
    for register in _ROUTE_REGISTRARS:
        register(app, ctx)
    return app


def main():
    """Entry point for the cctv-zarr-ui console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="CCTV Zarr web panel")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    settings = Settings.load(args.config)
    loopback = settings.ui.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not (
        settings.security.auth_token and settings.security.allow_remote
    ):
        raise SystemExit(
            "refusing to serve footage beyond loopback: set "
            "security.auth_token and security.allow_remote, and put "
            "TLS termination in front (GDPR Art. 32)"
        )
    import uvicorn
    uvicorn.run(
        create_app(args.config),
        host=settings.ui.host, port=settings.ui.port,
    )


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>CCTV Archive</title>
<style>
:root {
  color-scheme: light dark;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --ink-1: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --quiet: #e1e0d9;
  --series-raw: #9ec5f4;
  --danger: #d03b3b; --good: #006300;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19; --page: #0d0d0d;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --quiet: #2c2c2a;
    --series-raw: #184f95;
    --danger: #e66767; --good: #0ca30c;
  }
}
* { box-sizing: border-box; margin: 0; }
html, body { height: 100%; }
body {
  font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
  font-size: clamp(17px, 0.5vw + 15px, 19px); line-height: 1.45;
  -webkit-text-size-adjust: 100%;
  background: var(--page); color: var(--ink-1);
  padding: env(safe-area-inset-top) env(safe-area-inset-right)
           0 env(safe-area-inset-left);
}
.shell { display: block; min-height: 100%; }
.sidebar { display: none; }
.content {
  padding: 16px;
  padding-bottom: calc(49px + 16px + env(safe-area-inset-bottom));
  max-width: 760px; margin: 0 auto;
}
.masthead { display: flex; align-items: center; gap: 12px;
  margin-bottom: 16px; }
.appicon {
  width: 44px; height: 44px; border-radius: 22.5%;
  background: linear-gradient(135deg, var(--series-1), #104281);
  flex: none;
}
h1 { font-size: clamp(22px, 2.2vw + 17px, 28px); }
h2 { font-size: clamp(19px, 1vw + 17px, 22px); margin-bottom: 12px; }
.sub { color: var(--ink-2); font-size: 17px; }
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin-bottom: 16px;
}
button {
  font: inherit; font-size: 17px;
  min-height: 44px; min-width: 44px; padding: 0 18px;
  border-radius: 10px; border: 1px solid var(--border);
  background: var(--surface-1); color: var(--ink-1); cursor: pointer;
}
button.primary {
  background: var(--series-1); border-color: var(--series-1);
  color: #ffffff; font-weight: 600;
}
button:disabled { opacity: 0.5; cursor: default; }
input[type=text] {
  font: inherit; font-size: 17px; appearance: none;
  width: 100%; min-height: 44px; padding: 8px 12px;
  border-radius: 10px; border: 1px solid var(--baseline);
  background: var(--surface-1); color: var(--ink-1);
}
label.field {
  display: block; color: var(--ink-2); font-size: 17px;
  margin: 12px 0 4px;
}
.row { display: flex; gap: 12px; align-items: center;
  flex-wrap: wrap; margin-top: 12px; }
.toggle { display: inline-flex; align-items: center; gap: 10px;
  min-height: 44px; cursor: pointer; }
.toggle input { position: absolute; opacity: 0; }
.knob {
  width: 51px; height: 31px; border-radius: 999px;
  background: var(--baseline); position: relative;
  transition: background .2s; flex: none;
}
.knob::after {
  content: ""; position: absolute; top: 2px; left: 2px;
  width: 27px; height: 27px; border-radius: 999px;
  background: #ffffff; transition: left .2s;
  box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}
.toggle input:checked + .knob { background: var(--series-1); }
.toggle input:checked + .knob::after { left: 22px; }
.list { overscroll-behavior: contain; touch-action: pan-y;
  max-height: 420px; overflow-y: auto; }
.cell {
  display: flex; align-items: center; gap: 12px;
  min-height: 44px; padding: 6px 0; cursor: pointer;
}
.cell + .cell { border-top: 1px solid var(--grid); }
.cell .grow { flex: 1; min-width: 0; }
.cell .title { font-size: 17px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.cell .meta { color: var(--ink-muted); font-size: 15px; }
.cell .badge { color: var(--good); font-size: 15px; flex: none; }
.cell .chev { color: var(--ink-muted); flex: none; }
.check {
  width: 28px; height: 28px; border-radius: 999px;
  border: 2px solid var(--baseline); flex: none;
  display: flex; align-items: center; justify-content: center;
  color: transparent; font-size: 16px; font-weight: 700;
}
.cell.selected .check {
  background: var(--series-1); border-color: var(--series-1);
  color: #ffffff;
}
.dot { width: 10px; height: 10px; border-radius: 999px;
  background: var(--quiet); flex: none; }
.dot.on { background: var(--series-1); }
.crumb { color: var(--ink-muted); font-size: 15px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  margin-bottom: 4px; }
.drop {
  border: 2px dashed var(--baseline); border-radius: 12px;
  padding: 20px 16px; text-align: center; color: var(--ink-2);
  margin-top: 12px;
}
.drop.hover { border-color: var(--series-1); color: var(--ink-1); }
.drop button { margin-top: 8px; }
.hiddeninput { display: none; }
.progress { height: 8px; border-radius: 999px; background: var(--grid);
  overflow: hidden; flex: 1; min-width: 120px; }
.progress > div { height: 100%; width: 0%;
  background: var(--series-1); border-radius: 999px;
  transition: width .3s; }
.tab-bar {
  position: fixed; bottom: 0; left: 0; right: 0;
  height: calc(49px + env(safe-area-inset-bottom));
  padding-bottom: env(safe-area-inset-bottom);
  display: flex; background: var(--surface-1);
  border-top: 1px solid var(--border); z-index: 40;
}
.tab-bar button {
  flex: 1; border: 0; border-radius: 0; background: none;
  color: var(--ink-muted); font-size: 15px; min-height: 49px;
  padding: 0 4px; min-width: 0;
}
.tab-bar button.active { color: var(--series-1); font-weight: 600; }
.view { display: none; }
.view.active { display: block; }
.timeline { display: flex; gap: 2px; align-items: flex-end;
  height: clamp(56px, 9vh, 88px); padding: 8px 0; }
.timeline .bar {
  flex: 1; min-width: 3px; border-radius: 4px 4px 0 0;
  background: var(--quiet); height: 30%; cursor: pointer;
}
.timeline .bar.movement { background: var(--series-1); height: 100%; }
.timeline .bar.dim { opacity: 0.35; }
.axis { display: flex; justify-content: space-between;
  color: var(--ink-muted); font-size: 13px;
  border-top: 1px solid var(--baseline); padding-top: 4px; }
.ringwrap { display: flex; gap: 16px; align-items: center; }
.ring { width: clamp(84px, 12vw, 112px);
  height: clamp(84px, 12vw, 112px); transform: rotate(-90deg); }
.ring .track { fill: none; stroke: var(--grid); stroke-width: 10; }
.ring .val {
  fill: none; stroke: var(--series-1); stroke-width: 10;
  stroke-linecap: round; stroke-dasharray: 100;
  stroke-dashoffset: var(--val, 100);
  transition: stroke-dashoffset .6s ease;
}
.ringnum { font-size: clamp(22px, 2vw + 17px, 28px); font-weight: 700; }
.spinner {
  width: 28px; height: 28px; border-radius: 999px;
  border: 3px solid var(--grid); border-top-color: var(--series-1);
  animation: spin 1s linear infinite; display: none;
}
@keyframes spin { to { transform: rotate(360deg); } }
.sheet-backdrop {
  position: fixed; inset: 0; background: rgba(0,0,0,0.45);
  display: none; z-index: 45;
}
.sheet {
  position: fixed; inset: 10%; z-index: 46; display: none;
  background: var(--surface-1); border-radius: 16px;
  border: 1px solid var(--border); padding: 16px;
  overflow-y: auto; overscroll-behavior: contain;
}
.sheet.open, .sheet-backdrop.open { display: block; }
.sheet header { display: flex; justify-content: space-between;
  align-items: center; margin-bottom: 12px; }
.preview { display: flex; gap: 8px; overflow-x: auto;
  overscroll-behavior: contain; touch-action: pan-x; }
.preview img { height: clamp(120px, 24vh, 200px); border-radius: 8px;
  border: 1px solid var(--border); }
.menu {
  position: absolute; z-index: 50; min-width: 180px;
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 6px; display: none;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.menu button { display: block; width: 100%; text-align: left;
  border: 0; background: none; }
.hero { display: flex; flex-direction: column; gap: 4px;
  margin-bottom: 12px; }
.heronum { font-size: clamp(28px, 3vw + 17px, 40px);
  font-weight: 700; line-height: 1.1; }
.legend { display: flex; gap: 16px; flex-wrap: wrap;
  color: var(--ink-2); font-size: 15px; margin-bottom: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 3px;
  flex: none; }
.swatch.raw { background: var(--series-raw); }
.swatch.stored { background: var(--series-1); }
.bars { display: flex; flex-direction: column; gap: 14px; }
.barrow .title { font-size: 17px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.barline { display: flex; align-items: center; gap: 8px;
  margin-top: 4px; }
.bartrack { flex: 1; height: 12px; border-radius: 999px;
  background: none; }
.hbar { height: 12px; border-radius: 999px; min-width: 2px; }
.hbar.raw { background: var(--series-raw); }
.hbar.stored { background: var(--series-1); }
.barvalue { color: var(--ink-2); font-size: 15px; flex: none;
  min-width: 72px; text-align: right;
  font-variant-numeric: tabular-nums; }
.footnote { color: var(--ink-muted); font-size: 13px;
  margin-top: 12px; }
.error { color: var(--danger); margin-top: 8px; min-height: 22px; }
.ok { color: var(--good); }
@media (min-width: 900px) {
  .shell {
    display: grid;
    grid-template-columns: minmax(260px, 320px) 1fr;
    min-height: 100vh;
  }
  .sidebar {
    display: block; background: var(--surface-1);
    border-right: 1px solid var(--border); padding: 16px;
  }
  .tab-bar { display: none; }
  .content { padding-bottom: 16px; }
  .sidebar nav button {
    display: block; width: 100%; text-align: left; border: 0;
    background: none; color: var(--ink-1); margin-bottom: 2px;
  }
  .sidebar nav button.active {
    background: var(--page); color: var(--series-1); font-weight: 600;
    border-radius: 10px;
  }
  .only-narrow { display: none; }
}
@media (min-width: 1280px) {
  .content { max-width: 860px; }
}
</style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="masthead">
      <div class="appicon"></div>
      <div><h1>CCTV Archive</h1>
      <div class="sub">Your camera footage, organised</div></div>
    </div>
    <nav>
      <button data-tab="ingest" class="active">Add footage</button>
      <button data-tab="stores">Library</button>
      <button data-tab="query">Search</button>
      <button data-tab="storage">Results</button>
    </nav>
  </aside>
  <main class="content">
    <div class="masthead only-narrow">
      <div class="appicon"></div>
      <div><h1>CCTV Archive</h1>
      <div class="sub">Your camera footage, organised</div></div>
    </div>

    <div class="card" id="auth-card" hidden>
      <h2>Enter your access code</h2>
      <div class="sub">This archive is protected. Ask the person
      who runs it for the code.</div>
      <label class="field" for="auth-code">Access code</label>
      <input type="text" id="auth-code" autocomplete="off">
      <div class="row">
        <button class="primary" id="btn-auth">Unlock</button>
      </div>
      <div class="error" id="auth-error"></div>
    </div>

    <section class="view active" id="view-ingest">
      <div class="card">
        <h2>Choose videos</h2>
        <div class="drop" id="dropzone">
          <div>Drop videos here, or pick them with a click</div>
          <div class="sub" id="upload-limit"></div>
          <div class="row" style="justify-content: center;">
            <button class="primary" id="btn-pick-files">
              Choose files</button>
            <button id="btn-pick-folder">Choose a folder</button>
          </div>
          <input type="file" id="file-input" class="hiddeninput"
                 multiple accept="video/*">
          <input type="file" id="folder-input" class="hiddeninput"
                 webkitdirectory>
        </div>
        <div class="row" id="upload-row" hidden>
          <div class="progress"><div id="upload-fill"></div></div>
          <span class="sub" id="upload-label"></span>
        </div>
        <label class="field" for="folder-path">
          Or browse the archive computer</label>
        <input type="text" id="folder-path" autocomplete="off"
               placeholder="videos">
        <div class="row">
          <button id="btn-open">Open folder</button>
          <button id="btn-select-all">Select all</button>
        </div>
        <div class="crumb" id="crumb"></div>
        <div class="list" id="browser"></div>
        <div class="row">
          <button class="primary" id="btn-ingest-selected" disabled>
            Add selected</button>
          <button id="btn-ingest-folder">Add whole folder</button>
          <div class="spinner" id="ingest-spinner"></div>
        </div>
        <div class="error" id="ingest-error"></div>
      </div>
      <div class="card" id="batch-card" hidden>
        <h2>Adding footage</h2>
        <div class="row">
          <div class="progress"><div id="batch-fill"></div></div>
          <span class="sub" id="batch-count"></span>
        </div>
        <div class="sub" id="batch-current"></div>
        <div class="list" id="batch-items"></div>
      </div>
    </section>

    <section class="view" id="view-stores">
      <div class="card">
        <h2>Library</h2>
        <div class="sub" id="library-compression"></div>
        <div class="list" id="store-list"></div>
      </div>
    </section>

    <section class="view" id="view-query">
      <div class="card">
        <h2>Search footage</h2>
        <label class="field">Recording - tap to choose</label>
        <input type="text" id="query-store" class="hiddeninput"
               autocomplete="off">
        <div class="list" id="query-store-list"></div>
        <label class="field" for="query-start">From (date and time)</label>
        <input type="text" id="query-start" autocomplete="off"
               placeholder="2026-07-20 14:03">
        <label class="field" for="query-end">To (date and time)</label>
        <input type="text" id="query-end" autocomplete="off"
               placeholder="2026-07-20 14:07">
        <div class="row">
          <label class="toggle">
            <input type="checkbox" id="movement-only">
            <span class="knob"></span>
            <span>Only show movement</span>
          </label>
        </div>
        <div class="row">
          <button class="primary" id="btn-query">Search</button>
          <button id="btn-export">Save video clip</button>
          <div class="spinner" id="query-spinner"></div>
        </div>
        <div class="error" id="query-error"></div>
      </div>
      <div class="card" id="result-card" hidden>
        <div class="ringwrap">
          <svg class="ring" viewBox="0 0 40 40">
            <circle class="track" cx="20" cy="20" r="15.9155"></circle>
            <circle class="val" id="ring-val" cx="20" cy="20"
                    r="15.9155" pathLength="100"></circle>
          </svg>
          <div>
            <div class="ringnum" id="ring-num">0%</div>
            <div class="sub" id="ring-sub">of this recording has movement</div>
          </div>
        </div>
        <div class="sub" id="events-line"></div>
        <div class="list" id="event-list"></div>
        <div class="timeline" id="timeline"></div>
        <div class="axis"><span id="axis-start"></span>
        <span id="axis-end"></span></div>
        <div class="sub" id="match-line"></div>
        <div class="list" id="chunk-list"></div>
      </div>
    </section>

    <section class="view" id="view-storage">
      <div class="card">
        <h2>Storage savings</h2>
        <div class="hero">
          <div class="heronum" id="storage-hero">-</div>
          <div class="sub" id="storage-hero-sub"></div>
        </div>
        <div class="legend">
          <span><span class="swatch raw"></span>Uncompressed</span>
          <span><span class="swatch stored"></span>Stored</span>
        </div>
        <div class="bars" id="storage-bars"></div>
        <div class="footnote">Uncompressed means the raw video
        frames before any compression, not the original camera
        file.</div>
        <div class="error" id="storage-error"></div>
      </div>
    </section>
  </main>
</div>

<nav class="tab-bar">
  <button data-tab="ingest" class="active">Add footage</button>
  <button data-tab="stores">Library</button>
  <button data-tab="query">Search</button>
  <button data-tab="storage">Results</button>
</nav>

<div class="sheet-backdrop" id="backdrop"></div>
<div class="sheet" id="sheet">
  <header>
    <h2 id="sheet-title">Section</h2>
    <button id="sheet-close">Close</button>
  </header>
  <div class="sub" id="sheet-meta"></div>
  <div class="row">
    <div class="spinner" id="sheet-spinner"></div>
  </div>
  <div class="preview" id="sheet-preview"></div>
</div>

<div class="menu" id="menu">
  <button id="menu-view">View this moment</button>
  <button id="menu-export">Save this moment as a clip</button>
</div>

<script>
"use strict";
const AUTO_SAVE_MS = 30000;
const SPINNER_DELAY_MS = 1000;
const POLL_MS = 1500;
const selected = new Set();
let currentFolder = "";
let lastQuery = null;
let menuChunk = null;
let pollTimer = null;
let dirty = false;
let currentJobId = null;
let maxUploadMb = 0;

function initClientId() {
  let id = "";
  try { id = window.localStorage.getItem("cctvPanelClient") || ""; }
  catch (e) { id = ""; }
  if (!id) {
    id = "c" + Math.random().toString(36).slice(2, 12);
    try { window.localStorage.setItem("cctvPanelClient", id); }
    catch (e) {}
  }
  return id;
}
const clientId = initClientId();

async function loadConfig() {
  try {
    const data = await api("/api/config");
    maxUploadMb = data.max_upload_mb || 0;
    document.getElementById("upload-limit").textContent = maxUploadMb
      ? "Videos up to " + maxUploadMb + " MB each" : "";
  } catch (e) {}
}

function haptic(ms) {
  if (navigator.vibrate) { navigator.vibrate(ms); }
}

function fmtTime(iso) {
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function fmtDate(iso) {
  return new Date(iso).toLocaleDateString([], {
    day: "numeric", month: "short", year: "numeric",
  });
}

function fmtSecs(s) {
  if (s === undefined || s === null) { return ""; }
  if (s >= 60) {
    return Math.floor(s / 60) + " min " + Math.round(s % 60) + " s";
  }
  return Math.round(s) + " s";
}

function sectionLabel(c) {
  const secs =
    (Date.parse(c.end_time) - Date.parse(c.start_time)) / 1000;
  const kind = c.movement_detected
    ? (c.trigger === "motion"
       ? "movement starts here" : "movement")
    : "no movement";
  return fmtSecs(secs) + " - " + kind;
}

function delayedSpinner(el, promise) {
  const timer = setTimeout(() => { el.style.display = "block"; },
                           SPINNER_DELAY_MS);
  return promise.finally(() => {
    clearTimeout(timer);
    el.style.display = "none";
  });
}

function loadAuthToken() {
  try {
    return window.localStorage.getItem("cctvPanelAuth") || "";
  } catch (e) { return ""; }
}
let authToken = loadAuthToken();

async function api(path, body) {
  const headers = {};
  if (authToken) { headers.Authorization = "Bearer " + authToken; }
  let options = {headers: headers};
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    options = {
      method: "POST", headers: headers,
      body: JSON.stringify(body),
    };
  }
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (res.status === 401) {
      document.getElementById("auth-card").hidden = false;
    }
    const err = new Error(data.detail || ("HTTP " + res.status));
    err.status = res.status;
    throw err;
  }
  return data;
}

document.getElementById("btn-auth")
  .addEventListener("click", async () => {
    haptic(10);
    authToken =
      document.getElementById("auth-code").value.trim();
    try {
      window.localStorage.setItem("cctvPanelAuth", authToken);
    } catch (e) {}
    try {
      await api("/api/config");
      document.getElementById("auth-card").hidden = true;
      document.getElementById("auth-error").textContent = "";
      restore();
    } catch (e) {
      document.getElementById("auth-error").textContent =
        "That code did not work. Please try again.";
    }
  });

function switchTab(name) {
  document.querySelectorAll("[data-tab]").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".view").forEach(v =>
    v.classList.toggle("active", v.id === "view-" + name));
  if (name === "stores") { loadStores(); }
  if (name === "query") { loadQueryStores(); }
  if (name === "storage") { loadStorage(); }
  markDirty();
}
document.querySelectorAll("[data-tab]").forEach(b =>
  b.addEventListener("click", () => {
    haptic(10); switchTab(b.dataset.tab);
  }));

function activeTab() {
  const btn = document.querySelector(".tab-bar button.active");
  return btn ? btn.dataset.tab : "ingest";
}

function updateSelectedButton() {
  const btn = document.getElementById("btn-ingest-selected");
  btn.disabled = selected.size === 0;
  btn.textContent = selected.size
    ? "Add selected (" + selected.size + ")" : "Add selected";
}

function videoCell(v) {
  const cell = document.createElement("div");
  cell.className = "cell" + (selected.has(v.path) ? " selected" : "");
  cell.innerHTML =
    '<span class="check">\\u2713</span><div class="grow">' +
    '<div class="title">' + v.name + '</div><div class="meta">' +
    v.size_mb + " MB</div></div>" +
    (v.ingested ? '<span class="badge">In library</span>' : "");
  cell.addEventListener("click", () => {
    haptic(10);
    if (selected.has(v.path)) { selected.delete(v.path); }
    else { selected.add(v.path); }
    cell.classList.toggle("selected", selected.has(v.path));
    updateSelectedButton();
  });
  return cell;
}

function folderCell(name, target) {
  const cell = document.createElement("div");
  cell.className = "cell";
  cell.innerHTML =
    '<div class="grow"><div class="title">' + name +
    '</div></div><span class="chev">\\u203a</span>';
  cell.addEventListener("click", () => { haptic(10); browse(target); });
  return cell;
}

async function browse(path) {
  const errorEl = document.getElementById("ingest-error");
  errorEl.textContent = "";
  try {
    const data = await api("/api/browse", path ? {path: path} : {});
    currentFolder = data.path;
    selected.clear();
    updateSelectedButton();
    document.getElementById("folder-path").value = data.path;
    document.getElementById("crumb").textContent = data.path;
    const list = document.getElementById("browser");
    list.textContent = "";
    list.appendChild(folderCell("..", data.parent));
    for (const name of data.folders) {
      list.appendChild(folderCell(name, data.path + "/" + name));
    }
    for (const v of data.videos) { list.appendChild(videoCell(v)); }
    if (!data.folders.length && !data.videos.length) {
      const none = document.createElement("div");
      none.className = "cell";
      none.innerHTML = '<div class="grow">' +
        '<div class="meta">Empty folder</div></div>';
      list.appendChild(none);
    }
    markDirty();
  } catch (e) {
    haptic(30);
    errorEl.textContent = e.message;
  }
}
document.getElementById("btn-open").addEventListener("click", () =>
  browse(document.getElementById("folder-path").value.trim() || null));

function blobToB64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      resolve(String(reader.result).split(",")[1] || "");
    reader.onerror = () =>
      reject(new Error("Could not read that file"));
    reader.readAsDataURL(blob);
  });
}

function isVideoName(name) {
  return /\.(mp4|avi|mkv|mov|mpg|mpeg|webm)$/i.test(name);
}

async function uploadPicked(fileList) {
  const all = Array.from(fileList)
    .filter(f => isVideoName(f.name)).slice(0, 50);
  const errorEl = document.getElementById("ingest-error");
  errorEl.textContent = "";
  const limitBytes = maxUploadMb * 1000 * 1000;
  const oversized = limitBytes
    ? all.filter(f => f.size > limitBytes) : [];
  const files = limitBytes
    ? all.filter(f => f.size <= limitBytes) : all;
  if (oversized.length) {
    errorEl.textContent = "Skipped " + oversized.length +
      " video(s) over the " + maxUploadMb + " MB limit: " +
      oversized.map(f => f.name).join(", ");
  }
  if (!files.length) {
    if (!oversized.length) {
      errorEl.textContent = "No videos were picked.";
    }
    return;
  }
  const row = document.getElementById("upload-row");
  const fill = document.getElementById("upload-fill");
  const label = document.getElementById("upload-label");
  row.hidden = false;
  const partSize = 8 * 1024 * 1024;
  const totalBytes = files.reduce((n, f) => n + Math.max(f.size, 1), 0);
  let sentBytes = 0;
  const paths = [];
  try {
    for (let f = 0; f < files.length; f++) {
      const file = files[f];
      label.textContent = "Copying " + (f + 1) + " of " +
        files.length + ": " + file.name;
      const id = "u" + Date.now().toString(36) +
        Math.random().toString(36).slice(2, 10);
      const parts = Math.max(1, Math.ceil(file.size / partSize));
      for (let i = 0; i < parts; i++) {
        const blob = file.slice(i * partSize, (i + 1) * partSize);
        const data = await blobToB64(blob);
        const res = await api("/api/upload", {
          upload_id: id, name: file.name, seq: i,
          last: i === parts - 1, data: data,
        });
        sentBytes += Math.max(blob.size, 1);
        fill.style.width =
          Math.round(100 * sentBytes / totalBytes) + "%";
        if (res.path) { paths.push(res.path); }
      }
    }
    label.textContent = "Copied " + paths.length + " video(s)";
    haptic(10);
    await startIngest({videos: paths});
  } catch (e) {
    haptic(30);
    errorEl.textContent = e.message;
  } finally {
    setTimeout(() => { row.hidden = true; }, 1500);
  }
}

document.getElementById("btn-pick-files")
  .addEventListener("click", () => {
    haptic(10);
    document.getElementById("file-input").click();
  });
document.getElementById("btn-pick-folder")
  .addEventListener("click", () => {
    haptic(10);
    document.getElementById("folder-input").click();
  });
document.getElementById("file-input")
  .addEventListener("change", ev => {
    uploadPicked(ev.target.files);
    ev.target.value = "";
  });
document.getElementById("folder-input")
  .addEventListener("change", ev => {
    uploadPicked(ev.target.files);
    ev.target.value = "";
  });
const dropzone = document.getElementById("dropzone");
dropzone.addEventListener("dragover", ev => {
  ev.preventDefault();
  dropzone.classList.add("hover");
});
dropzone.addEventListener("dragleave", () =>
  dropzone.classList.remove("hover"));
dropzone.addEventListener("drop", ev => {
  ev.preventDefault();
  dropzone.classList.remove("hover");
  haptic(10);
  if (ev.dataTransfer && ev.dataTransfer.files) {
    uploadPicked(ev.dataTransfer.files);
  }
});
document.getElementById("btn-select-all")
  .addEventListener("click", () => {
    haptic(10);
    const cells = document.querySelectorAll("#browser .cell .check");
    const all = document.querySelectorAll("#browser .cell");
    let any = false;
    all.forEach(cell => {
      if (cell.querySelector(".check") &&
          !cell.classList.contains("selected")) {
        any = true;
        cell.click();
      }
    });
    if (!any) {
      all.forEach(cell => {
        if (cell.querySelector(".check") &&
            cell.classList.contains("selected")) {
          cell.click();
        }
      });
    }
  });

async function startIngest(body) {
  const errorEl = document.getElementById("ingest-error");
  errorEl.textContent = "";
  try {
    const res = await api("/api/ingest", body);
    currentJobId = res.job_id || null;
    haptic(10);
    document.getElementById("batch-card").hidden = false;
    document.getElementById("batch-items").textContent = "";
    pollStatus();
  } catch (e) {
    haptic(30);
    errorEl.textContent = e.message;
  }
}
document.getElementById("btn-ingest-selected")
  .addEventListener("click", () =>
    startIngest({videos: Array.from(selected)}));
document.getElementById("btn-ingest-folder")
  .addEventListener("click", () =>
    startIngest({folder: currentFolder ||
      document.getElementById("folder-path").value.trim() || null}));

function renderBatch(s) {
  const pct = s.total ? Math.round(100 * s.done / s.total) : 0;
  document.getElementById("batch-fill").style.width = pct + "%";
  document.getElementById("batch-count").textContent =
    s.done + " / " + s.total;
  document.getElementById("batch-current").textContent =
    s.state === "queued"
      ? "Waiting in line - " + (s.queued_ahead || 0) +
        " task(s) ahead"
      : (s.state === "running" && s.current
         ? "Working on " + s.current : "");
  const list = document.getElementById("batch-items");
  list.textContent = "";
  for (const item of s.items) {
    const cell = document.createElement("div");
    cell.className = "cell";
    if (item.error) {
      cell.innerHTML =
        '<span class="dot"></span><div class="grow">' +
        '<div class="title">' + item.video + '</div>' +
        '<div class="meta error">' + item.error + "</div></div>";
    } else {
      cell.innerHTML =
        '<span class="dot ' + (item.movement_chunks ? "on" : "") +
        '"></span><div class="grow"><div class="title">' + item.video +
        '</div><div class="meta">' + fmtSecs(item.seconds) +
        " of footage - " +
        (item.events
          ? item.events + " movement(s) across " +
            item.movement_chunks + " of " + item.chunks + " sections"
          : "no movement") +
        (item.footprint_ratio
          ? " - " + item.footprint_ratio.toFixed(1) +
            "x smaller than raw"
          : "") + "</div></div>" +
        '<span class="badge">\\u2713</span>';
    }
    list.appendChild(cell);
  }
}

function pollStatus() {
  clearTimeout(pollTimer);
  const spinner = document.getElementById("ingest-spinner");
  const timer = setTimeout(() => { spinner.style.display = "block"; },
                           SPINNER_DELAY_MS);
  const tick = async () => {
    if (document.hidden) {
      pollTimer = setTimeout(tick, POLL_MS);
      return;
    }
    try {
      const s = await api("/api/status" +
        (currentJobId ? "?job=" + currentJobId : ""));
      renderBatch(s);
      if (s.state === "running" || s.state === "queued") {
        pollTimer = setTimeout(tick, POLL_MS);
        return;
      }
      clearTimeout(timer);
      spinner.style.display = "none";
      if (s.state === "done") {
        haptic(10);
        loadStores();
        browse(currentFolder || null);
      } else if (s.state === "failed") {
        haptic(30);
        document.getElementById("ingest-error").textContent =
          s.error || "batch failed";
      }
    } catch (e) {
      clearTimeout(timer);
      spinner.style.display = "none";
      document.getElementById("ingest-error").textContent = e.message;
    }
  };
  pollTimer = setTimeout(tick, POLL_MS);
}

async function loadStores() {
  const list = document.getElementById("store-list");
  const banner = document.getElementById("library-compression");
  try {
    const data = await api("/api/stores");
    list.textContent = "";
    if (!data.stores.length) {
      banner.textContent = "";
      list.innerHTML = '<div class="cell"><div class="grow">' +
        '<div class="meta">Your library is empty. ' +
        "Add footage first." +
        "</div></div></div>";
      return;
    }
    let rawTotal = 0;
    let storedTotal = 0;
    for (const s of data.stores) {
      if (s.compression) {
        rawTotal += s.compression.raw_mb;
        storedTotal += s.compression.stored_mb;
      }
    }
    banner.textContent = storedTotal > 0
      ? "Altogether: " + Math.round(rawTotal) +
        " MB of raw frames stored in " + Math.round(storedTotal) +
        " MB - " + (rawTotal / storedTotal).toFixed(1) +
        "x smaller than raw"
      : "";
    for (const s of data.stores) {
      const squeeze = s.compression
        ? " - " + s.compression.footprint_ratio.toFixed(1) +
          "x smaller than raw"
        : "";
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.innerHTML =
        '<span class="dot ' + (s.movement_chunks ? "on" : "") +
        '"></span><div class="grow"><div class="title">' + s.name +
        '</div><div class="meta">' + fmtSecs(s.seconds) +
        " - movement in " + s.movement_chunks + " of " + s.chunks +
        " sections - " + s.size_mb + " MB" + squeeze +
        "</div></div>";
      cell.addEventListener("click", () => {
        haptic(10);
        document.getElementById("query-store").value = s.name;
        switchTab("query");
        runQuery();
      });
      list.appendChild(cell);
    }
  } catch (e) {
    list.innerHTML = '<div class="cell"><div class="grow">' +
      '<div class="meta">' + e.message + "</div></div></div>";
  }
}

function fmtMb(mb) {
  if (mb >= 1024) { return (mb / 1024).toFixed(1) + " GB"; }
  return Math.round(mb) + " MB";
}

function storageRow(name, rawMb, storedMb, maxRaw) {
  const rawPct = Math.max(100 * rawMb / maxRaw, 0.5);
  const storedPct = Math.max(100 * storedMb / maxRaw, 0.5);
  const row = document.createElement("div");
  row.className = "barrow";
  row.title = name + ": " + fmtMb(rawMb) +
    " uncompressed, " + fmtMb(storedMb) + " stored";
  row.innerHTML =
    '<div class="title">' + name + "</div>" +
    '<div class="barline"><div class="bartrack">' +
    '<div class="hbar raw" style="width:' + rawPct +
    '%"></div></div>' +
    '<span class="barvalue">' + fmtMb(rawMb) + "</span></div>" +
    '<div class="barline"><div class="bartrack">' +
    '<div class="hbar stored" style="width:' + storedPct +
    '%"></div></div>' +
    '<span class="barvalue">' + fmtMb(storedMb) + "</span></div>";
  return row;
}

async function loadStorage() {
  const bars = document.getElementById("storage-bars");
  const hero = document.getElementById("storage-hero");
  const heroSub = document.getElementById("storage-hero-sub");
  const errorEl = document.getElementById("storage-error");
  errorEl.textContent = "";
  try {
    const data = await api("/api/stores");
    bars.textContent = "";
    const rows = data.stores.filter(s => s.compression);
    if (!rows.length) {
      hero.textContent = "-";
      heroSub.textContent =
        "Add footage first to see how much space you save.";
      return;
    }
    let rawTotal = 0;
    let storedTotal = 0;
    let maxRaw = 0;
    for (const s of rows) {
      rawTotal += s.compression.raw_mb;
      storedTotal += s.compression.stored_mb;
      maxRaw = Math.max(maxRaw, s.compression.raw_mb);
    }
    const ratio = storedTotal > 0 ? rawTotal / storedTotal : 0;
    hero.textContent = ratio.toFixed(1) + "x smaller";
    heroSub.textContent = fmtMb(storedTotal) +
      " stored instead of " + fmtMb(rawTotal) + " uncompressed";
    for (const s of rows) {
      bars.appendChild(storageRow(
        s.name, s.compression.raw_mb,
        s.compression.stored_mb, maxRaw));
    }
  } catch (e) {
    errorEl.textContent = e.message;
  }
}

function queryStoreCell(s) {
  const cell = document.createElement("div");
  const current = document.getElementById("query-store").value;
  cell.className = "cell" + (current === s.name ? " selected" : "");
  cell.innerHTML =
    '<span class="check">\\u2713</span>' +
    '<span class="dot ' + (s.movement_chunks ? "on" : "") +
    '"></span><div class="grow"><div class="title">' + s.name +
    '</div><div class="meta">' + fmtSecs(s.seconds) +
    " - movement in " + s.movement_chunks + " of " + s.chunks +
    " sections</div></div>";
  cell.addEventListener("click", () => {
    haptic(10);
    document.getElementById("query-store").value = s.name;
    document.querySelectorAll("#query-store-list .cell")
      .forEach(c => c.classList.remove("selected"));
    cell.classList.add("selected");
    markDirty();
  });
  return cell;
}

async function loadQueryStores() {
  const list = document.getElementById("query-store-list");
  try {
    const data = await api("/api/stores");
    list.textContent = "";
    if (!data.stores.length) {
      list.innerHTML = '<div class="cell"><div class="grow">' +
        '<div class="meta">Your library is empty. ' +
        "Add footage first.</div></div></div>";
      return;
    }
    for (const s of data.stores) {
      list.appendChild(queryStoreCell(s));
    }
  } catch (e) {
    list.innerHTML = '<div class="cell"><div class="grow">' +
      '<div class="meta">' + e.message + "</div></div></div>";
  }
}

function queryBody() {
  const body = {
    store: document.getElementById("query-store").value.trim(),
    movement_only: document.getElementById("movement-only").checked,
    client: clientId,
  };
  const start = document.getElementById("query-start").value.trim();
  const end = document.getElementById("query-end").value.trim();
  if (start) { body.start = start; }
  if (end) { body.end = end; }
  return body;
}

async function runQuery() {
  const errorEl = document.getElementById("query-error");
  errorEl.textContent = "";
  const body = queryBody();
  if (!body.store) {
    errorEl.textContent = "Choose a recording first.";
    return;
  }
  const spinner = document.getElementById("query-spinner");
  try {
    const data = await delayedSpinner(
      spinner, api("/api/query", body));
    haptic(10);
    lastQuery = data;
    renderQuery(data);
  } catch (e) {
    haptic(30);
    errorEl.textContent = e.message;
  }
}
document.getElementById("btn-query").addEventListener("click", () => {
  runQuery(); markDirty();
});

function renderQuery(data) {
  document.getElementById("result-card").hidden = false;
  const matchedSet = new Set(data.records.map(r => r.index));
  const all = data.all_chunks;
  const movement = all.filter(c => c.movement_detected === 1).length;
  const pct = all.length ? Math.round(100 * movement / all.length) : 0;
  document.getElementById("ring-val")
    .style.setProperty("--val", String(100 - pct));
  document.getElementById("ring-num").textContent = pct + "%";
  document.getElementById("match-line").textContent =
    data.matched + " of " + data.total +
    " sections match your search";
  const events = data.events || [];
  document.getElementById("events-line").textContent = events.length
    ? events.length + " separate movement(s) in this recording"
    : "No movements in this recording";
  const eventList = document.getElementById("event-list");
  eventList.textContent = "";
  for (let i = 0; i < Math.min(events.length, 50); i++) {
    const e = events[i];
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.innerHTML =
      '<span class="dot on"></span><div class="grow">' +
      '<div class="title">Movement ' + (i + 1) + " - " +
      fmtTime(e.start_time) + " - " + fmtTime(e.end_time) +
      '</div><div class="meta">' + fmtSecs(e.seconds) +
      "</div></div>";
    cell.addEventListener("click", () => {
      haptic(10);
      document.getElementById("query-start").value = e.start_time;
      document.getElementById("query-end").value = e.end_time;
      runQuery();
    });
    eventList.appendChild(cell);
  }
  const timeline = document.getElementById("timeline");
  timeline.textContent = "";
  const shown = all.slice(0, 240);
  for (const c of shown) {
    const bar = document.createElement("div");
    bar.className = "bar" +
      (c.movement_detected === 1 ? " movement" : "") +
      (matchedSet.has(c.index) ? "" : " dim");
    bar.title = "Section " + c.index + " - " +
      fmtTime(c.start_time) +
      (c.movement_detected ? " - movement" : " - quiet");
    bar.addEventListener("click", () => openSheet(c));
    timeline.appendChild(bar);
  }
  if (all.length) {
    document.getElementById("axis-start").textContent =
      fmtDate(all[0].start_time) + " " + fmtTime(all[0].start_time);
    document.getElementById("axis-end").textContent =
      fmtTime(all[all.length - 1].end_time);
  }
  const list = document.getElementById("chunk-list");
  list.textContent = "";
  for (const c of data.records) {
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.innerHTML =
      '<span class="dot ' + (c.movement_detected ? "on" : "") +
      '"></span><div class="grow"><div class="title">' +
      fmtTime(c.start_time) + " - " + fmtTime(c.end_time) +
      '</div><div class="meta">' + fmtDate(c.start_time) + " - " +
      sectionLabel(c) + "</div></div>";
    cell.addEventListener("click", () => openSheet(c));
    cell.addEventListener("contextmenu", ev => {
      ev.preventDefault();
      openMenu(ev, c);
    });
    list.appendChild(cell);
  }
}

function openMenu(ev, chunk) {
  menuChunk = chunk;
  const menu = document.getElementById("menu");
  menu.style.display = "block";
  const pad = 12;
  const x = Math.min(ev.pageX, window.innerWidth - menu.offsetWidth - pad);
  const y = Math.min(ev.pageY,
    window.innerHeight + window.scrollY - menu.offsetHeight - pad);
  menu.style.left = Math.max(pad, x) + "px";
  menu.style.top = Math.max(pad, y) + "px";
}
document.addEventListener("click", ev => {
  const menu = document.getElementById("menu");
  if (!menu.contains(ev.target)) { menu.style.display = "none"; }
});
document.getElementById("menu-view").addEventListener("click", () => {
  document.getElementById("menu").style.display = "none";
  if (menuChunk) { openSheet(menuChunk); }
});
document.getElementById("menu-export").addEventListener("click", () => {
  document.getElementById("menu").style.display = "none";
  if (menuChunk) { exportSection(menuChunk); }
});

async function openSheet(chunk) {
  haptic(10);
  document.getElementById("sheet-title").textContent =
    fmtTime(chunk.start_time) + " - " + fmtTime(chunk.end_time);
  document.getElementById("sheet-meta").textContent =
    fmtDate(chunk.start_time) + " - " + sectionLabel(chunk);
  document.getElementById("sheet-preview").textContent = "";
  document.getElementById("sheet").classList.add("open");
  document.getElementById("backdrop").classList.add("open");
  const spinner = document.getElementById("sheet-spinner");
  try {
    const data = await delayedSpinner(spinner, api("/api/frames", {
      store: document.getElementById("query-store").value.trim(),
      chunk_index: chunk.index,
      client: clientId,
    }));
    const preview = document.getElementById("sheet-preview");
    for (const f of data.frames) {
      const img = document.createElement("img");
      img.src = "data:image/jpeg;base64," + f.jpeg;
      img.alt = "frame at " + f.timestamp;
      preview.appendChild(img);
    }
  } catch (e) {
    document.getElementById("sheet-meta").textContent = e.message;
  }
}
function closeSheet() {
  document.getElementById("sheet").classList.remove("open");
  document.getElementById("backdrop").classList.remove("open");
}
document.getElementById("sheet-close")
  .addEventListener("click", closeSheet);
document.addEventListener("keydown", ev => {
  if (ev.key === "Escape") { closeSheet(); }
});

async function exportSection(chunk) {
  const errorEl = document.getElementById("query-error");
  const body = queryBody();
  if (chunk) {
    body.start = chunk.start_time;
    body.end = chunk.end_time;
    body.movement_only = false;
  }
  try {
    const data = await delayedSpinner(
      document.getElementById("query-spinner"),
      api("/api/export", body));
    haptic(10);
    errorEl.classList.add("ok");
    errorEl.textContent = "Clip saved to " + data.path;
  } catch (e) {
    haptic(30);
    errorEl.classList.remove("ok");
    errorEl.textContent = e.message;
  }
}
document.getElementById("btn-export")
  .addEventListener("click", () => exportSection(null));

function markDirty() { dirty = true; }
["query-store", "query-start", "query-end", "folder-path"]
  .forEach(id => document.getElementById(id)
    .addEventListener("input", markDirty));
document.getElementById("movement-only")
  .addEventListener("change", () => { haptic(10); markDirty(); });

async function autoSave() {
  if (!dirty) { return; }
  dirty = false;
  const prefs = {
    tab: activeTab(),
    folder: currentFolder,
    store: document.getElementById("query-store").value,
    start: document.getElementById("query-start").value,
    end: document.getElementById("query-end").value,
    movement_only: document.getElementById("movement-only").checked,
  };
  try {
    await api("/api/prefs", {prefs: prefs, client: clientId});
  } catch (e) {}
}
setInterval(autoSave, AUTO_SAVE_MS);
window.addEventListener("pagehide", autoSave);

async function restore() {
  let folder = null;
  loadConfig();
  try {
    const data = await api("/api/prefs?client=" + clientId);
    const p = data.prefs || {};
    if (p.store) {
      document.getElementById("query-store").value = p.store;
    }
    if (p.start) {
      document.getElementById("query-start").value = p.start;
    }
    if (p.end) { document.getElementById("query-end").value = p.end; }
    document.getElementById("movement-only").checked =
      Boolean(p.movement_only);
    if (p.folder) { folder = p.folder; }
    if (p.tab) { switchTab(p.tab); }
  } catch (e) {}
  browse(folder);
  loadStores();
}
restore();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
