"""Typed YAML configuration loading into frozen dataclasses.

Every numeric parameter originates from ``config.yaml``; module code
contains no magic numbers.
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional

import yaml

MAX_CONFIG_NODES = 100_000


def _to_tuple(root: Any) -> Any:
    """Convert nested YAML lists to tuples, iteratively and bounded.

    Post-order traversal with an explicit stack: children convert
    before their container, no recursion, at most MAX_CONFIG_NODES
    visits.

    Args:
        root: A YAML-decoded value.

    Returns:
        Any: The value with every list converted to a tuple.

    Raises:
        ValueError: If the document exceeds MAX_CONFIG_NODES nodes.
    """
    if not isinstance(root, (list, dict)):
        return root
    stack = [(root, False)]
    done: dict = {}
    for _ in range(MAX_CONFIG_NODES):
        if not stack:
            return done[id(root)]
        node, expanded = stack.pop()
        if not isinstance(node, (list, dict)):
            done[id(node)] = node
        elif not expanded:
            stack.append((node, True))
            children = (
                node if isinstance(node, list) else node.values()
            )
            stack.extend((child, False) for child in children)
        elif isinstance(node, list):
            done[id(node)] = tuple(done[id(c)] for c in node)
        else:
            done[id(node)] = {
                k: done[id(c)] for k, c in node.items()
            }
    raise ValueError("config document too large")


def _build_defaulted(cls, section: Optional[dict]):
    """Instantiate a fully-defaulted section, absent = defaults.

    Args:
        cls: The dataclass type (every field has a default).
        section (Optional[dict]): Raw YAML mapping, or None.

    Returns:
        The dataclass instance.

    Raises:
        KeyError: On unknown keys.
        ValueError: On invalid values.
    """
    if section is None:
        return cls()
    return _build(cls, section)


def _build(cls, section: Optional[dict]):
    """Instantiate a frozen dataclass from a YAML section.

    Args:
        cls: The dataclass type.
        section (Optional[dict]): Raw YAML mapping.

    Raises:
        KeyError: If the section is missing or has unknown keys.
    """
    if section is None:
        raise KeyError(f"Missing config section for {cls.__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(section) - known
    if unknown:
        raise KeyError(f"Unknown keys in {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: _to_tuple(v) for k, v in section.items()})


@dataclass(frozen=True, slots=True)
class GopConfig:
    """Encoder-aligned chunking.

    ``gop_frames`` mirrors the camera encoder's closed-GOP length
    (H.264/H.263 CCTV encoders typically emit an IDR every 1-2 s, e.g.
    25/30/50/60 frames). The Zarr time-chunk equals one GOP, so a
    chunk boundary is always a legal cut point, and movement-activated
    segments snap to GOP boundaries exactly as an encoder inserts an
    IDR on demand.
    """

    gop_frames: int
    max_video_frames: int

    def __post_init__(self):
        if self.gop_frames <= 0:
            raise ValueError("gop_frames must be positive")
        if self.max_video_frames <= 0:
            raise ValueError("max_video_frames must be positive")


@dataclass(frozen=True, slots=True)
class FlowConfig:
    """Optical-flow motion detection and per-object tracking.

    Farneback dense flow on downscaled grey frames; only the flow
    magnitude is used, so movement in any direction is monitored.
    Moving pixels form blobs of at least ``min_area_frac`` of the
    analysis frame; blobs associate into per-object tracks within
    ``match_distance_frac`` of the frame diagonal. A track survives a
    pause of up to ``pause_buffer_s`` seconds, so one object pausing
    and continuing stays one movement; tracks shorter than
    ``min_track_frames`` are noise and are dropped.
    """

    analysis_size: tuple
    pyr_scale: float
    levels: int
    winsize: int
    iterations: int
    poly_n: int
    poly_sigma: float
    min_magnitude_px: float
    min_area_frac: float
    pause_buffer_s: float
    match_distance_frac: float
    min_track_frames: int
    max_tracks: int
    record_events: bool = True

    def __post_init__(self):
        if len(self.analysis_size) != 2:
            raise ValueError("analysis_size must be (width, height)")
        if self.analysis_size[0] <= 0 or self.analysis_size[1] <= 0:
            raise ValueError("analysis_size must be positive")
        if not 0.0 < self.pyr_scale < 1.0:
            raise ValueError("pyr_scale must be in (0, 1)")
        if self.levels <= 0 or self.winsize <= 0 or self.iterations <= 0:
            raise ValueError("levels, winsize, iterations must be positive")
        if self.min_magnitude_px <= 0.0:
            raise ValueError("min_magnitude_px must be positive")
        if not 0.0 < self.min_area_frac < 1.0:
            raise ValueError("min_area_frac must be in (0, 1)")
        if self.pause_buffer_s <= 0.0:
            raise ValueError("pause_buffer_s must be positive")
        if not 0.0 < self.match_distance_frac <= 1.0:
            raise ValueError("match_distance_frac must be in (0, 1]")
        if self.min_track_frames <= 0:
            raise ValueError("min_track_frames must be positive")
        if self.max_tracks <= 0:
            raise ValueError("max_tracks must be positive")


@dataclass(frozen=True, slots=True)
class ZarrConfig:
    """OME-Zarr store layout."""

    resize_width: Optional[int]
    grayscale: bool
    compression_level: int

    def __post_init__(self):
        if self.resize_width is not None and self.resize_width <= 0:
            raise ValueError("resize_width must be positive when set")
        if not 0 <= self.compression_level <= 9:
            raise ValueError("compression_level must be in [0, 9]")


@dataclass(frozen=True, slots=True)
class ArchiveConfig:
    """Cloud archive (GCS) destination.

    ``bucket_location`` pins where footage may physically live
    (GDPR Chapter V): when set, binding to a real bucket in any
    other location fails closed. The mock archive is always local.
    """

    gcs_bucket: Optional[str]
    gcs_prefix: str
    use_mock_gcs: bool
    local_root: Optional[str]
    bucket_location: Optional[str] = None


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    """Storage limitation (GDPR Art. 5(1)(e)) and minimisation.

    Movement and non-movement chunks expire separately;
    ``keep_non_movement: false`` never stores quiet footage at all
    (data minimisation by design, Art. 25). Exports and query
    caches expire on their own clocks so no copy outlives its
    chunk. ``source_after_ingest`` governs the original file:
    ``keep``, ``quarantine`` (moved aside for a short hold), or
    ``delete``.
    """

    enabled: bool = True
    movement_max_age_hours: float = 720.0
    non_movement_max_age_hours: float = 72.0
    exports_max_age_hours: float = 72.0
    cache_max_age_hours: float = 24.0
    keep_non_movement: bool = True
    source_after_ingest: str = "keep"

    def __post_init__(self):
        if self.movement_max_age_hours <= 0:
            raise ValueError("movement_max_age_hours must be positive")
        if self.non_movement_max_age_hours <= 0:
            raise ValueError(
                "non_movement_max_age_hours must be positive",
            )
        if self.exports_max_age_hours <= 0:
            raise ValueError("exports_max_age_hours must be positive")
        if self.cache_max_age_hours <= 0:
            raise ValueError("cache_max_age_hours must be positive")
        if self.source_after_ingest not in (
            "keep", "quarantine", "delete",
        ):
            raise ValueError(
                "source_after_ingest must be keep, quarantine, "
                "or delete",
            )


@dataclass(frozen=True, slots=True)
class GovernanceConfig:
    """Accountability metadata written into every store.

    Travels with the store (GDPR Arts. 5(2), 30): controller
    identity, site, purpose, legal basis, retention policy, and
    DPO contact. Empty values are tolerated only against the mock
    archive; real-archive ingest warns loudly.
    """

    controller: str = ""
    site_id: str = ""
    purpose: str = ""
    legal_basis: str = ""
    retention_policy: str = ""
    dpo_contact: str = ""


@dataclass(frozen=True, slots=True)
class SecurityConfig:
    """Processing security (GDPR Art. 32).

    ``auth_token`` protects every panel endpoint; the panel
    refuses to leave loopback unless a token is set and
    ``allow_remote`` is explicitly true (behind TLS termination).
    ``encrypt_archive`` encrypts every object client-side (Fernet,
    AES-128-CBC + HMAC) before it reaches the bucket.
    """

    auth_token: str = ""
    allow_remote: bool = False
    encrypt_archive: bool = False
    encryption_key: str = ""

    def __post_init__(self):
        if self.encrypt_archive and not self.encryption_key:
            raise ValueError(
                "encrypt_archive requires encryption_key",
            )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Paths and behaviour flags."""

    video_directory: str
    store_directory: str
    default_fps: float
    max_consecutive_fails: int
    video_extensions: tuple
    timezone: str

    def __post_init__(self):
        if self.default_fps <= 0.0:
            raise ValueError("default_fps must be positive")
        if self.max_consecutive_fails <= 0:
            raise ValueError("max_consecutive_fails must be positive")


@dataclass(frozen=True, slots=True)
class UIConfig:
    """Web control panel host/port, preview and sharing limits.

    ``max_upload_mb`` caps each video copied through the panel
    (0 disables the cap); ``max_queued_jobs`` bounds how many batch
    tasks may wait in line behind the running one, so several
    concurrent users queue instead of being turned away.
    """

    host: str
    port: int
    preview_width: int
    preview_max_frames: int
    max_upload_mb: int = 0
    max_queued_jobs: int = 4

    def __post_init__(self):
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if self.preview_width <= 0:
            raise ValueError("preview_width must be positive")
        if self.preview_max_frames <= 0:
            raise ValueError("preview_max_frames must be positive")
        if self.max_upload_mb < 0:
            raise ValueError("max_upload_mb must be >= 0")
        if self.max_queued_jobs <= 0:
            raise ValueError("max_queued_jobs must be positive")


@dataclass(frozen=True, slots=True)
class Settings:
    """Root configuration aggregating all typed sections."""

    gop: GopConfig
    flow: FlowConfig
    zarr: ZarrConfig
    archive: ArchiveConfig
    runtime: RuntimeConfig
    ui: UIConfig
    retention: RetentionConfig
    governance: GovernanceConfig
    security: SecurityConfig

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load and validate settings from a YAML file.

        Args:
            path (Path): Path to config.yaml.

        Returns:
            Settings: Fully validated, immutable settings.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: On missing sections or unknown keys.
            ValueError: On invalid values.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a mapping")
        return cls(
            gop=_build(GopConfig, raw.get("gop")),
            flow=_build(FlowConfig, raw.get("flow")),
            zarr=_build(ZarrConfig, raw.get("zarr")),
            archive=_build(ArchiveConfig, raw.get("archive")),
            runtime=_build(RuntimeConfig, raw.get("runtime")),
            ui=_build(UIConfig, raw.get("ui")),
            retention=_build_defaulted(
                RetentionConfig, raw.get("retention"),
            ),
            governance=_build_defaulted(
                GovernanceConfig, raw.get("governance"),
            ),
            security=_build_defaulted(
                SecurityConfig, raw.get("security"),
            ),
        )

    def with_overrides(
        self,
        video_directory: Optional[Path] = None,
        store_directory: Optional[Path] = None,
    ) -> "Settings":
        """Return a copy with CLI path overrides applied."""
        runtime = self.runtime
        if video_directory is not None:
            runtime = replace(runtime, video_directory=str(video_directory))
        if store_directory is not None:
            runtime = replace(runtime, store_directory=str(store_directory))
        return replace(self, runtime=runtime)
