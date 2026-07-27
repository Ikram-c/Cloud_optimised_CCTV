"""Cloud archive: upload stores to GCS, fetch chunks selectively.

The store's on-disk layout maps one file to one object, so a query's
chunk selection maps directly to a small set of object downloads:
metadata documents first (.zattrs, .zarray, the movement and
timestamp arrays - all tiny), then only the image chunks the query
matched. The offline mock mirrors a bucket in a local directory so
the whole test suite runs with no network or credentials.

Network loss is survivable by design. Every object transfer retries
with bounded exponential backoff, and uploads keep a journal
(.upload_state.json beside the store) recording which objects have
landed: if the connection drops mid-upload, the store stays intact
on local disk and a later upload_store call resumes from the journal,
re-sending only what is missing. Downloads that already exist in the
local cache are never re-fetched, so an interrupted query resumes
where it stopped.
"""

import json
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

from .exceptions import ArchiveError

logger = logging.getLogger(__name__)

RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 0.5
RETRY_MAX_DELAY_S = 8.0
UPLOAD_STATE_FILENAME = ".upload_state.json"

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


class MockBlob:
    """A blob backed by a file under the mock bucket root."""

    __slots__ = ("_path", "name")

    def __init__(self, root: Path, name: str):
        self._path = Path(root) / name
        self.name = name

    def upload_from_filename(self, filename: str):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(Path(filename).read_bytes())

    def download_to_filename(self, filename: str):
        if not self._path.exists():
            raise FileNotFoundError(f"mock blob missing: {self.name}")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_bytes(self._path.read_bytes())

    def exists(self) -> bool:
        return self._path.exists()


class MockBucket:
    """A bucket backed by a local directory."""

    __slots__ = ("_root", "name")

    def __init__(self, root: Path, name: str):
        self._root = Path(root) / name
        self.name = name

    def blob(self, blob_name: str) -> MockBlob:
        return MockBlob(self._root, blob_name)

    def list_blobs(self, prefix: str = "") -> List[MockBlob]:
        if not self._root.exists():
            return []
        blobs = []
        for path in sorted(self._root.rglob("*")):
            if path.is_file():
                name = path.relative_to(self._root).as_posix()
                if name.startswith(prefix):
                    blobs.append(MockBlob(self._root, name))
        return blobs


class MockStorageClient:
    """Drop-in replacement for google.cloud.storage.Client.

    Args:
        root (Path): Directory that stands in for the GCS service.
    """

    def __init__(self, root: Path):
        self._root = Path(root)

    def bucket(self, bucket_name: str) -> MockBucket:
        return MockBucket(self._root, bucket_name)

    def list_blobs(self, bucket_name: str, prefix: str = ""):
        return self.bucket(bucket_name).list_blobs(prefix=prefix)


def get_storage_client(use_mock: bool, mock_root: Optional[Path] = None):
    """Return a real or mock storage client.

    Args:
        use_mock (bool): Use the offline mock.
        mock_root (Optional[Path]): Mock service directory; required
            when use_mock is True.

    Returns:
        A client exposing bucket(name) with blob/list operations.

    Raises:
        ArchiveError: If the mock root is missing, or the real client
            is requested without google-cloud-storage installed.
    """
    if use_mock:
        if mock_root is None:
            raise ArchiveError("mock GCS requires a mock_root directory")
        return MockStorageClient(mock_root)
    if not GCS_AVAILABLE:
        raise ArchiveError(
            "google-cloud-storage is required unless use_mock_gcs is true"
        )
    return storage.Client()


class CloudArchive:
    """One store's home in the archive: <bucket>/<prefix>/<store_name>."""

    def __init__(
        self,
        client,
        bucket_name: str,
        prefix: str,
        retry_attempts: int = RETRY_ATTEMPTS,
        retry_base_delay_s: float = RETRY_BASE_DELAY_S,
    ):
        """Bind to a bucket and prefix.

        Args:
            client: Real or mock storage client.
            bucket_name (str): Bucket name.
            prefix (str): Key prefix ('' for bucket root).
            retry_attempts (int): Bounded tries per object transfer.
            retry_base_delay_s (float): First backoff delay; doubles
                per retry up to RETRY_MAX_DELAY_S.

        Raises:
            ArchiveError: On a non-positive retry budget.
        """
        if retry_attempts <= 0:
            raise ArchiveError("retry_attempts must be positive")
        if retry_base_delay_s < 0.0:
            raise ArchiveError("retry_base_delay_s must be non-negative")
        self.client = client
        self.bucket = client.bucket(bucket_name)
        self.prefix = prefix.strip("/")
        self.retry_attempts = retry_attempts
        self.retry_base_delay_s = retry_base_delay_s

    def _with_retries(self, label: str, operation: Callable):
        """Run one network operation with bounded backoff.

        Args:
            label (str): Description for logs and errors.
            operation (Callable): Zero-argument transfer callable.

        Returns:
            The operation's return value.

        Raises:
            ArchiveError: When every attempt failed; the original
                error message is preserved.
        """
        delay = self.retry_base_delay_s
        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return operation()
            except FileNotFoundError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.retry_attempts:
                    logger.warning(
                        "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                        label, attempt, self.retry_attempts, e, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2.0, RETRY_MAX_DELAY_S)
        raise ArchiveError(
            f"{label} failed after {self.retry_attempts} attempts: "
            f"{last_error}"
        )

    def _key(self, store_name: str, rel: str) -> str:
        parts = [p for p in (self.prefix, store_name, rel) if p]
        return "/".join(parts)

    def upload_store(self, store_path: Path, store_name: str) -> int:
        """Upload a local store resumably, one object per file.

        A journal beside the store records every object that has
        landed; a network failure mid-upload leaves the journal in
        place and a later call re-sends only the missing objects. The
        journal is removed once the store is fully uploaded.

        Args:
            store_path (Path): Local store root.
            store_name (str): Name under the archive prefix.

        Returns:
            int: Objects uploaded by this call (0 = already synced).

        Raises:
            ArchiveError: If the store directory does not exist, or
                the network stayed down through every retry (the
                store and journal remain on disk for a later resume).
        """
        store_path = Path(store_path)
        if not store_path.is_dir():
            raise ArchiveError(f"store not found: {store_path}")
        state_path = store_path.parent / (
            store_path.name + UPLOAD_STATE_FILENAME
        )
        uploaded = self._read_state(state_path)
        self._write_state(state_path, uploaded)
        count = 0
        for path, rel in self._store_files(store_path):
            if rel in uploaded:
                continue
            key = self._key(store_name, rel)
            self._with_retries(
                f"upload {rel}",
                lambda p=path, k=key:
                    self.bucket.blob(k).upload_from_filename(str(p)),
            )
            uploaded.add(rel)
            self._write_state(state_path, uploaded)
            count += 1
        state_path.unlink(missing_ok=True)
        logger.info(
            "Uploaded %s: %d objects to %s/%s",
            store_path, count, self.bucket.name, self._key(store_name, ""),
        )
        return count

    @staticmethod
    def pending_objects(store_path: Path) -> int:
        """Objects still waiting to reach the archive.

        Args:
            store_path (Path): Local store root.

        Returns:
            int: 0 when fully uploaded or never attempted with a
                surviving journal; otherwise the remaining count.
        """
        store_path = Path(store_path)
        state_path = store_path.parent / (
            store_path.name + UPLOAD_STATE_FILENAME
        )
        if not state_path.is_file():
            return 0
        uploaded = CloudArchive._read_state(state_path)
        total = sum(
            1 for _ in CloudArchive._store_files(store_path)
        )
        return max(0, total - len(uploaded))

    @staticmethod
    def _store_files(store_path: Path):
        """Yield (path, store-relative key) for every store file.

        Args:
            store_path (Path): Local store root.

        Yields:
            Tuple[Path, str]: File path and relative key.
        """
        for path in sorted(Path(store_path).rglob("*")):
            if path.is_file():
                yield path, path.relative_to(store_path).as_posix()

    @staticmethod
    def _read_state(state_path: Path) -> set:
        """Read the upload journal, tolerating absence or damage.

        Args:
            state_path (Path): Journal path.

        Returns:
            set: Relative keys already uploaded.
        """
        if not state_path.is_file():
            return set()
        try:
            payload = json.loads(state_path.read_text())
            return set(payload.get("uploaded", []))
        except (ValueError, OSError):
            logger.warning("Damaged upload journal reset: %s", state_path)
            return set()

    @staticmethod
    def _write_state(state_path: Path, uploaded: set):
        """Persist the upload journal atomically.

        Args:
            state_path (Path): Journal path.
            uploaded (set): Relative keys already uploaded.
        """
        import os
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"uploaded": sorted(uploaded)}))
        os.replace(tmp, state_path)

    def fetch(self, store_name: str, rel_paths: List[str], dest: Path) -> Path:
        """Download a selected subset of one store's objects.

        Args:
            store_name (str): Store name under the prefix.
            rel_paths (List[str]): Store-relative paths to fetch.
            dest (Path): Local directory to mirror them into.

        Returns:
            Path: The local partial-store root.

        Raises:
            ArchiveError: If a required object is missing.
        """
        dest = Path(dest)
        for rel in rel_paths:
            key = self._key(store_name, rel)
            try:
                self._with_retries(
                    f"download {rel}",
                    lambda k=key, r=rel: self.bucket.blob(
                        k
                    ).download_to_filename(str(dest / r)),
                )
            except FileNotFoundError as e:
                raise ArchiveError(str(e))
        return dest

    def list_store(self, store_name: str) -> List[str]:
        """List a store's object keys, store-relative.

        Args:
            store_name (str): Store name under the prefix.

        Returns:
            List[str]: Relative paths of every object.
        """
        base = self._key(store_name, "")
        blobs = self.bucket.list_blobs(prefix=base)
        return [b.name[len(base):].lstrip("/") for b in blobs]
