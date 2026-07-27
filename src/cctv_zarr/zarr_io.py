"""Minimal, dependency-free Zarr v2 array read/write.

Writes the standard on-disk Zarr v2 format (one ``.zarray`` JSON
document per array, C-order chunk files keyed ``t.c.z.y.x``, zlib
compression) so the stores this package produces are readable by
zarr-python, ome-zarr-py, napari, and any other conforming client -
without this package needing any of them installed.

Only what cctv_zarr uses is implemented: uint8/float64/int64 arrays,
zlib or raw chunks, full-chunk writes with fill-value padding on the
final partial chunk, and whole-chunk reads.
"""

import json
import zlib
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

_DTYPE_MAP = {
    "uint8": "|u1",
    "float64": "<f8",
    "int64": "<i8",
}
_DTYPE_BACK = {v: k for k, v in _DTYPE_MAP.items()}


def _chunk_key(indices: Sequence[int]) -> str:
    """Zarr v2 chunk key with the default '.' separator."""
    return ".".join(str(i) for i in indices)


class ZarrArray:
    """One Zarr v2 array rooted at a directory."""

    def __init__(self, path: Path, meta: dict):
        self.path = Path(path)
        self.meta = meta
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        self.dtype = np.dtype(_DTYPE_BACK[meta["dtype"]])
        self._compressed = meta["compressor"] is not None
        self._level = (
            meta["compressor"]["level"] if self._compressed else 0
        )

    @classmethod
    def create(
        cls,
        path: Path,
        shape: Sequence[int],
        chunks: Sequence[int],
        dtype: str,
        compression_level: int,
        fill_value=0,
    ) -> "ZarrArray":
        """Create an array directory and write its .zarray document.

        Args:
            path (Path): Array directory (created).
            shape (Sequence[int]): Array shape.
            chunks (Sequence[int]): Chunk shape (same rank as shape).
            dtype (str): One of 'uint8', 'float64', 'int64'.
            compression_level (int): zlib level; 0 stores raw chunks.
            fill_value: Fill for unwritten/padded regions.

        Returns:
            ZarrArray: The created array.

        Raises:
            ValueError: On rank mismatch or unsupported dtype.
        """
        if len(shape) != len(chunks):
            raise ValueError("shape and chunks must have equal rank")
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"unsupported dtype '{dtype}'")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        compressor = (
            {"id": "zlib", "level": int(compression_level)}
            if compression_level > 0 else None
        )
        meta = {
            "zarr_format": 2,
            "shape": list(int(s) for s in shape),
            "chunks": list(int(c) for c in chunks),
            "dtype": _DTYPE_MAP[dtype],
            "compressor": compressor,
            "fill_value": fill_value,
            "order": "C",
            "filters": None,
            "dimension_separator": ".",
        }
        (path / ".zarray").write_text(json.dumps(meta, indent=2))
        return cls(path, meta)

    @classmethod
    def open(cls, path: Path) -> "ZarrArray":
        """Open an existing array from its .zarray document.

        Args:
            path (Path): Array directory.

        Returns:
            ZarrArray: The opened array.

        Raises:
            FileNotFoundError: If .zarray is absent.
        """
        meta = json.loads((Path(path) / ".zarray").read_text())
        return cls(path, meta)

    def grid_shape(self) -> Tuple[int, ...]:
        """Number of chunks along each axis."""
        return tuple(
            -(-s // c) for s, c in zip(self.shape, self.chunks)
        )

    def write_chunk(self, indices: Sequence[int], data: np.ndarray):
        """Write one chunk, padding a partial edge chunk with fill.

        Args:
            indices (Sequence[int]): Chunk grid indices.
            data (np.ndarray): The chunk's valid region; may be
                smaller than the chunk shape on trailing edges.

        Raises:
            ValueError: On rank or dtype mismatch, or oversized data.
        """
        if data.ndim != len(self.chunks):
            raise ValueError("chunk data rank mismatch")
        if data.dtype != self.dtype:
            raise ValueError(
                f"chunk dtype {data.dtype} != array dtype {self.dtype}"
            )
        if any(d > c for d, c in zip(data.shape, self.chunks)):
            raise ValueError("chunk data exceeds chunk shape")
        if tuple(data.shape) != self.chunks:
            padded = np.full(
                self.chunks, self.meta["fill_value"], dtype=self.dtype,
            )
            padded[tuple(slice(0, s) for s in data.shape)] = data
            data = padded
        raw = np.ascontiguousarray(data).tobytes()
        if self._compressed:
            raw = zlib.compress(raw, self._level)
        (self.path / _chunk_key(indices)).write_bytes(raw)

    def read_chunk(self, indices: Sequence[int]) -> np.ndarray:
        """Read one chunk (full chunk shape, fill where unwritten).

        Args:
            indices (Sequence[int]): Chunk grid indices.

        Returns:
            np.ndarray: The chunk, chunk-shaped.
        """
        path = self.path / _chunk_key(indices)
        if not path.exists():
            return np.full(
                self.chunks, self.meta["fill_value"], dtype=self.dtype,
            )
        raw = path.read_bytes()
        if self._compressed:
            raw = zlib.decompress(raw)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.chunks).copy()

    def read_full(self) -> np.ndarray:
        """Materialise the whole array (small arrays only).

        Returns:
            np.ndarray: The array trimmed to its true shape.
        """
        grid = self.grid_shape()
        out = np.full(
            tuple(g * c for g, c in zip(grid, self.chunks)),
            self.meta["fill_value"], dtype=self.dtype,
        )
        for flat in range(int(np.prod(grid))):
            indices = np.unravel_index(flat, grid)
            region = tuple(
                slice(i * c, (i + 1) * c)
                for i, c in zip(indices, self.chunks)
            )
            out[region] = self.read_chunk(indices)
        return out[tuple(slice(0, s) for s in self.shape)]


def write_group(path: Path, attrs: Optional[dict] = None):
    """Write a Zarr v2 group document (and optional attributes).

    Args:
        path (Path): Group directory (created).
        attrs (Optional[dict]): Contents for .zattrs.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}, indent=2))
    if attrs is not None:
        (path / ".zattrs").write_text(json.dumps(attrs, indent=2))


def read_attrs(path: Path) -> dict:
    """Read a group's .zattrs, tolerating absence.

    Args:
        path (Path): Group directory.

    Returns:
        dict: Attributes; {} when absent.
    """
    attrs_path = Path(path) / ".zattrs"
    if not attrs_path.exists():
        return {}
    return json.loads(attrs_path.read_text())
