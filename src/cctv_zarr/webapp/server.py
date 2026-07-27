"""Web control panel: browse, ingest, and query CCTV stores.

FastAPI app serving a single-page panel built to platform interface
guidelines: fluid system typography (base 17px), light and dark
modes with at least 4.5:1 text contrast, safe-area insets, a fixed
49px tab bar with three destinations that becomes a 260-320px
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
"""

import argparse
import base64
import binascii
import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .. import ome, zarr_io
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
PREFS_FILENAME = "panel_prefs.json"
EXPORT_DIRNAME = "exports"
UPLOAD_PREFIX = ".upload_"


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


class FramesRequest(BaseModel):
    """Preview request for one chunk of one store."""

    store: str
    chunk_index: int = Field(ge=0)


class ExportRequest(BaseModel):
    """Export request for the chunks a query matches."""

    store: str
    start: Optional[str] = None
    end: Optional[str] = None
    movement_only: bool = False
    fps: float = Field(default=25.0, gt=0.0)


class PrefsRequest(BaseModel):
    """Implicit auto-save payload for panel state."""

    prefs: Dict[str, object] = Field(default_factory=dict)


class JobState:
    """Thread-safe batch-ingest state."""

    def __init__(self):
        """Initialise the idle state."""
        self._lock = threading.Lock()
        self.state = "idle"
        self.total = 0
        self.done = 0
        self.current = ""
        self.items: List[dict] = []
        self.error: Optional[str] = None

    def try_start(self, total: int) -> bool:
        """Claim the batch slot.

        Args:
            total (int): Number of videos in the batch.

        Returns:
            bool: False when a batch is already running.
        """
        with self._lock:
            if self.state == "running":
                return False
            self.state = "running"
            self.total = total
            self.done = 0
            self.current = ""
            self.items = []
            self.error = None
            return True

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
            dict: state, total, done, current, items, error.
        """
        with self._lock:
            return {
                "state": self.state,
                "total": self.total,
                "done": self.done,
                "current": self.current,
                "items": list(self.items),
                "error": self.error,
            }


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
                shutil.rmtree(store)
            result = writer.ingest(video, store)
            entry.update({
                "store": store.name,
                "seconds": round(
                    result.frames_written / result.fps, 1,
                ),
                "frames": result.frames_written,
                "chunks": result.chunk_count,
                "movement_chunks": result.movement_chunks,
                "events": result.motion_events,
            })
        except (VideoOpenError, ValueError, OSError, cv2.error) as e:
            logger.exception("Ingest failed for %s", video)
            entry["error"] = str(e)
        job.item_done(entry)
    job.finish()


def create_app(config_path: Path) -> FastAPI:
    """Build the control-panel app bound to one config file.

    Args:
        config_path (Path): Path to config.yaml.

    Returns:
        FastAPI: The application.
    """
    app = FastAPI(title="CCTV Zarr")
    job = JobState()
    uploads: Dict[str, dict] = {}
    uploads_lock = threading.Lock()
    config_path = Path(config_path)

    def _load_settings() -> Settings:
        """Load settings lazily so config errors surface as 400s.

        Returns:
            Settings: Validated settings.
        """
        return Settings.load(config_path)

    def _settings_or_400() -> Settings:
        """Load settings or raise a 400.

        Returns:
            Settings: Validated settings.

        Raises:
            HTTPException: When the config cannot be loaded.
        """
        try:
            return _load_settings()
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

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        """Serve the panel."""
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/status")
    def status() -> dict:
        """Report the batch-ingest state."""
        return job.snapshot()

    @app.get("/api/stores")
    def stores() -> dict:
        """List readable stores under store_directory."""
        settings = _settings_or_400()
        root = Path(settings.runtime.store_directory)
        found: List[dict] = []
        if root.is_dir():
            candidates = sorted(p for p in root.iterdir() if p.is_dir())
            for path in candidates[:MAX_STORES_LISTED]:
                summary = _store_summary(path)
                if summary is not None:
                    found.append(summary)
        return {"stores": found}

    @app.post("/api/browse")
    def browse(req: BrowseRequest) -> dict:
        """List one folder's subfolders and selectable videos."""
        settings = _settings_or_400()
        base = (
            Path(req.path).expanduser()
            if req.path else Path(settings.runtime.video_directory)
        )
        try:
            base = base.resolve()
        except OSError as e:
            raise HTTPException(status_code=400, detail=str(e))
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
        allowed = {e.lower() for e in settings.runtime.video_extensions}
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
                "ingested": (store_root / (p.stem + ".zarr")).is_dir(),
            })
        return {
            "path": str(base),
            "parent": str(base.parent),
            "folders": folders,
            "videos": videos,
        }

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

    @app.post("/api/upload")
    def upload(req: UploadRequest) -> dict:
        """Receive one part of a clicked-or-dropped video upload.

        Parts arrive in order per upload_id and append to a hidden
        temporary file in the video directory; the last part renames
        it to its final name and returns the saved path.
        """
        settings = _settings_or_400()
        name = _validate_upload_name(settings, req.name)
        try:
            payload = base64.b64decode(req.data, validate=True)
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
        folder = Path(settings.runtime.video_directory)
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder / (UPLOAD_PREFIX + req.upload_id)
        with uploads_lock:
            entry = uploads.get(req.upload_id)
            if entry is None:
                if req.seq != 0:
                    raise HTTPException(
                        status_code=400,
                        detail="That upload was interrupted. "
                               "Please try again.",
                    )
                if len(uploads) >= MAX_ACTIVE_UPLOADS:
                    raise HTTPException(
                        status_code=409,
                        detail="Too many uploads at once. "
                               "Please wait a moment.",
                    )
                entry = {"name": name, "next_seq": 0, "bytes": 0}
                uploads[req.upload_id] = entry
                temp.write_bytes(b"")
            if req.seq != entry["next_seq"]:
                uploads.pop(req.upload_id, None)
                temp.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail="That upload was interrupted. "
                           "Please try again.",
                )
            if entry["bytes"] + len(payload) > MAX_UPLOAD_TOTAL_BYTES:
                uploads.pop(req.upload_id, None)
                temp.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail="That video is too large to copy.",
                )
            with temp.open("ab") as handle:
                handle.write(payload)
            entry["next_seq"] += 1
            entry["bytes"] += len(payload)
            if not req.last:
                return {"done": False, "received": entry["bytes"]}
            uploads.pop(req.upload_id, None)
            final = _final_upload_path(folder, entry["name"])
            temp.rename(final)
        return {"done": True, "path": str(final)}

    @app.post("/api/ingest")
    def ingest(req: IngestRequest) -> dict:
        """Launch a batch ingest in a daemon thread."""
        settings = _settings_or_400()
        videos: List[Path] = []
        if req.folder:
            folder = Path(req.folder).expanduser()
            if not folder.is_dir():
                raise HTTPException(
                    status_code=400,
                    detail=f"That folder was not found: {folder}",
                )
            videos.extend(_discover_folder_videos(
                folder, settings.runtime.video_extensions,
            ))
        for raw in req.videos[:MAX_BATCH_VIDEOS]:
            path = Path(raw).expanduser()
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
                status_code=400, detail="Choose at least one video first.",
            )
        if not job.try_start(len(unique)):
            raise HTTPException(
                status_code=409, detail="Another task is still running. "
                       "Please wait for it to finish.",
            )
        threading.Thread(
            target=_run_batch, args=(settings, unique, job), daemon=True,
        ).start()
        return {"started": True, "count": len(unique)}

    @app.post("/api/query")
    def query(req: QueryRequest) -> dict:
        """Resolve a time/movement query to chunk records."""
        settings = _settings_or_400()
        client = _open_query(settings, req.store)
        movement = True if req.movement_only else None
        try:
            selection = client.select(
                start=req.start, end=req.end, movement=movement,
            )
        except QueryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        all_records = client.records
        return {
            "records": [_record_dict(r) for r in selection.records],
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

    @app.post("/api/frames")
    def frames(req: FramesRequest) -> dict:
        """Return JPEG previews for one chunk's section."""
        settings = _settings_or_400()
        client = _open_query(settings, req.store)
        matched = [r for r in client.records if r.index == req.chunk_index]
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
                    frame, (width, height), interpolation=cv2.INTER_AREA,
                )
            ok, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72],
            )
            if not ok:
                continue
            encoded.append({
                "jpeg": base64.b64encode(buffer.tobytes()).decode("ascii"),
                "timestamp": float(stamps[position]),
            })
        return {"frames": encoded, "chunk": _record_dict(matched[0])}

    @app.post("/api/export")
    def export(req: ExportRequest) -> dict:
        """Export the matching section as an MP4 clip."""
        settings = _settings_or_400()
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
                status_code=400, detail="Nothing matched that search.",
            )
        out_dir = Path(settings.runtime.store_directory) / EXPORT_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = selection.records[0].start_time.strftime("%Y%m%dT%H%M%S")
        out_path = out_dir / f"{req.store}_{stamp}.mp4"
        try:
            written = client.export_mp4(selection, out_path, fps=req.fps)
        except QueryError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "path": str(out_path),
            "frames": written,
            "chunks": len(selection.records),
        }

    @app.get("/api/prefs")
    def read_prefs() -> dict:
        """Return the auto-saved panel state."""
        try:
            settings = _load_settings()
        except (FileNotFoundError, KeyError, ValueError):
            return {"prefs": {}}
        path = Path(settings.runtime.store_directory) / PREFS_FILENAME
        if not path.is_file():
            return {"prefs": {}}
        try:
            return {"prefs": json.loads(path.read_text())}
        except (ValueError, OSError):
            return {"prefs": {}}

    @app.post("/api/prefs")
    def write_prefs(req: PrefsRequest) -> dict:
        """Persist the auto-saved panel state."""
        settings = _settings_or_400()
        root = Path(settings.runtime.store_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / PREFS_FILENAME).write_text(json.dumps(req.prefs))
        return {"saved": True}

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
  --danger: #d03b3b; --good: #006300;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19; --page: #0d0d0d;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --quiet: #2c2c2a;
    --danger: #e66767; --good: #0ca30c;
  }
}
* { box-sizing: border-box; margin: 0; }
html, body { height: 100%; }
body {
  font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
  font-size: 17px; line-height: 1.45;
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
}
.tab-bar button.active { color: var(--series-1); font-weight: 600; }
.view { display: none; }
.view.active { display: block; }
.timeline { display: flex; gap: 2px; align-items: flex-end;
  height: 72px; padding: 8px 0; }
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
.ring { width: 96px; height: 96px; transform: rotate(-90deg); }
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
.preview img { height: 160px; border-radius: 8px;
  border: 1px solid var(--border); }
.menu {
  position: absolute; z-index: 50; min-width: 180px;
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 6px; display: none;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.menu button { display: block; width: 100%; text-align: left;
  border: 0; background: none; }
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
    </nav>
  </aside>
  <main class="content">
    <div class="masthead only-narrow">
      <div class="appicon"></div>
      <div><h1>CCTV Archive</h1>
      <div class="sub">Your camera footage, organised</div></div>
    </div>

    <section class="view active" id="view-ingest">
      <div class="card">
        <h2>Choose videos</h2>
        <div class="drop" id="dropzone">
          <div>Drop videos here, or pick them with a click</div>
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
        <div class="list" id="store-list"></div>
      </div>
    </section>

    <section class="view" id="view-query">
      <div class="card">
        <h2>Search footage</h2>
        <label class="field" for="query-store">Recording</label>
        <input type="text" id="query-store" placeholder="cam01.zarr"
               autocomplete="off">
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
  </main>
</div>

<nav class="tab-bar">
  <button data-tab="ingest" class="active">Add footage</button>
  <button data-tab="stores">Library</button>
  <button data-tab="query">Search</button>
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

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  };
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || ("HTTP " + res.status));
  }
  return data;
}

function switchTab(name) {
  document.querySelectorAll("[data-tab]").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".view").forEach(v =>
    v.classList.toggle("active", v.id === "view-" + name));
  if (name === "stores") { loadStores(); }
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
      list.innerHTML += '<div class="cell"><div class="grow">' +
        '<div class="meta">Empty folder</div></div></div>';
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
  const files = Array.from(fileList)
    .filter(f => isVideoName(f.name)).slice(0, 50);
  const errorEl = document.getElementById("ingest-error");
  errorEl.textContent = "";
  if (!files.length) {
    errorEl.textContent = "No videos were picked.";
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
    await api("/api/ingest", body);
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
    s.state === "running" && s.current
      ? "Working on " + s.current : "";
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
          : "no movement") + "</div></div>" +
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
      const s = await api("/api/status");
      renderBatch(s);
      if (s.state === "running") {
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
  try {
    const data = await api("/api/stores");
    list.textContent = "";
    if (!data.stores.length) {
      list.innerHTML = '<div class="cell"><div class="grow">' +
        '<div class="meta">Your library is empty. ' +
        "Add footage first." +
        "</div></div></div>";
      return;
    }
    for (const s of data.stores) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.innerHTML =
        '<span class="dot ' + (s.movement_chunks ? "on" : "") +
        '"></span><div class="grow"><div class="title">' + s.name +
        '</div><div class="meta">' + fmtSecs(s.seconds) +
        " - movement in " + s.movement_chunks + " of " + s.chunks +
        " sections - " + s.size_mb + " MB</div></div>";
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

function queryBody() {
  const body = {
    store: document.getElementById("query-store").value.trim(),
    movement_only: document.getElementById("movement-only").checked,
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
  const spinner = document.getElementById("query-spinner");
  try {
    const data = await delayedSpinner(
      spinner, api("/api/query", queryBody()));
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
  try { await api("/api/prefs", {prefs: prefs}); } catch (e) {}
}
setInterval(autoSave, AUTO_SAVE_MS);
window.addEventListener("pagehide", autoSave);

async function restore() {
  let folder = null;
  try {
    const data = await api("/api/prefs");
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
