"""Zarr v2 format round-trip and spec-shape tests. Fully offline."""

import json

import numpy as np
import pytest

from cctv_zarr import zarr_io


class TestZarrArray:
    def test_roundtrip_uint8(self, tmp_path):
        arr = zarr_io.ZarrArray.create(
            tmp_path / "a", shape=(10, 3, 1, 8, 8),
            chunks=(4, 3, 1, 8, 8), dtype="uint8", compression_level=4,
        )
        rng = np.random.default_rng(0)
        data = rng.integers(0, 255, (4, 3, 1, 8, 8), dtype=np.uint8)
        arr.write_chunk((0, 0, 0, 0, 0), data)
        out = zarr_io.ZarrArray.open(tmp_path / "a").read_chunk(
            (0, 0, 0, 0, 0)
        )
        assert np.array_equal(out, data)

    def test_partial_edge_chunk_padded_and_trimmed(self, tmp_path):
        arr = zarr_io.ZarrArray.create(
            tmp_path / "a", shape=(10,), chunks=(4,),
            dtype="float64", compression_level=4,
        )
        for i, size in enumerate((4, 4, 2)):
            arr.write_chunk((i,), np.arange(size, dtype=np.float64))
        full = arr.read_full()
        assert full.shape == (10,)
        assert np.array_equal(full[8:], [0.0, 1.0])

    def test_zarray_document_is_spec_shaped(self, tmp_path):
        zarr_io.ZarrArray.create(
            tmp_path / "a", shape=(6,), chunks=(3,),
            dtype="uint8", compression_level=2,
        )
        meta = json.loads((tmp_path / "a" / ".zarray").read_text())
        assert meta["zarr_format"] == 2
        assert meta["dtype"] == "|u1"
        assert meta["compressor"] == {"id": "zlib", "level": 2}
        assert meta["order"] == "C"
        assert meta["dimension_separator"] == "."

    def test_uncompressed_mode(self, tmp_path):
        arr = zarr_io.ZarrArray.create(
            tmp_path / "a", shape=(4,), chunks=(4,),
            dtype="int64", compression_level=0,
        )
        data = np.arange(4, dtype=np.int64)
        arr.write_chunk((0,), data)
        raw = (tmp_path / "a" / "0").read_bytes()
        assert raw == data.tobytes()

    def test_missing_chunk_reads_fill(self, tmp_path):
        arr = zarr_io.ZarrArray.create(
            tmp_path / "a", shape=(8,), chunks=(4,),
            dtype="uint8", compression_level=4, fill_value=0,
        )
        assert np.all(arr.read_chunk((1,)) == 0)

    def test_dtype_mismatch_raises(self, tmp_path):
        arr = zarr_io.ZarrArray.create(
            tmp_path / "a", shape=(4,), chunks=(4,),
            dtype="uint8", compression_level=4,
        )
        with pytest.raises(ValueError):
            arr.write_chunk((0,), np.zeros(4, dtype=np.float64))

    def test_oversized_chunk_raises(self, tmp_path):
        arr = zarr_io.ZarrArray.create(
            tmp_path / "a", shape=(4,), chunks=(4,),
            dtype="uint8", compression_level=4,
        )
        with pytest.raises(ValueError):
            arr.write_chunk((0,), np.zeros(5, dtype=np.uint8))

    def test_group_and_attrs(self, tmp_path):
        zarr_io.write_group(tmp_path / "g", {"hello": 1})
        assert json.loads(
            (tmp_path / "g" / ".zgroup").read_text()
        ) == {"zarr_format": 2}
        assert zarr_io.read_attrs(tmp_path / "g") == {"hello": 1}
        assert zarr_io.read_attrs(tmp_path / "nowhere") == {}
