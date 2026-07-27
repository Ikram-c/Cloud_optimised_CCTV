"""Package exceptions."""

from pathlib import Path


class VideoOpenError(Exception):
    """Raised when a video file cannot be opened."""

    __slots__ = ("path",)

    def __init__(self, path: Path):
        self.path = path
        super().__init__(f"Failed to open video: {path}")


class ArchiveError(Exception):
    """Raised when the cloud archive cannot serve a request."""


class QueryError(Exception):
    """Raised when a query is invalid or matches a corrupt store."""
