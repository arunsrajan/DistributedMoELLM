"""Minimal, dependency-free safetensors reader/writer on top of numpy.

Tensors are memory-mapped, so opening a 100 GB checkpoint costs nothing until
a tensor is actually touched. This is what lets a node read only the experts it
owns, the same idea Colibri uses to stream routed experts from disk.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

# safetensors dtype tag -> (numpy storage dtype, needs bf16 widening)
_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,  # widened to float32 on read
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}
_TAGS = {np.dtype(v).str: k for k, v in _DTYPES.items() if k != "BF16"}


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """Widen raw bfloat16 bits (stored as uint16) to float32."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bits (round-to-nearest-even), as uint16."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounding = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding) >> 16).astype(np.uint16)


class SafeTensorFile:
    """One .safetensors file, memory-mapped."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        self.metadata = header.pop("__metadata__", {}) or {}
        self.entries = header
        self._data_offset = 8 + hlen
        self._mm = None

    def _map(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._mm

    def keys(self):
        return self.entries.keys()

    def dtype_tag(self, name: str) -> str:
        return self.entries[name]["dtype"]

    def nbytes(self, name: str) -> int:
        a, b = self.entries[name]["data_offsets"]
        return b - a

    def raw(self, name: str) -> tuple[np.ndarray, str, list[int]]:
        """Return (raw bytes view, dtype tag, shape) without conversion."""
        e = self.entries[name]
        a, b = e["data_offsets"]
        mm = self._map()
        return mm[self._data_offset + a:self._data_offset + b], e["dtype"], e["shape"]

    def get(self, name: str, dtype=np.float32) -> np.ndarray:
        buf, tag, shape = self.raw(name)
        return decode_tensor(buf, tag, shape, dtype)


def decode_tensor(buf, tag: str, shape, dtype=np.float32) -> np.ndarray:
    """Decode raw little-endian bytes of a safetensors tensor."""
    store = _DTYPES[tag]
    arr = np.frombuffer(buf, dtype=store).reshape(shape)
    if tag == "BF16":
        arr = bf16_to_f32(arr)
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    elif dtype is not None:
        arr = np.array(arr, copy=True)  # detach from the memmap
    return arr


class SafeTensorDir:
    """A HF-style checkpoint directory: one file or an index of shards."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        self.files: dict[str, SafeTensorFile] = {}
        self.where: dict[str, str] = {}
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            for name, fname in weight_map.items():
                self.where[name] = fname
        else:
            for p in sorted(self.root.glob("*.safetensors")):
                for name in SafeTensorFile(p).keys():
                    self.where[name] = p.name
        if not self.where:
            raise FileNotFoundError(f"no safetensors weights in {self.root}")

    def _file(self, name: str) -> SafeTensorFile:
        fname = self.where[name]
        f = self.files.get(fname)
        if f is None:
            f = self.files[fname] = SafeTensorFile(self.root / fname)
        return f

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def keys(self):
        return self.where.keys()

    def get(self, name: str, dtype=np.float32) -> np.ndarray:
        return self._file(name).get(name, dtype)

    def raw(self, name: str):
        return self._file(name).raw(name)

    def nbytes(self, name: str) -> int:
        return self._file(name).nbytes(name)


def save_file(tensors: dict[str, np.ndarray], path: str | os.PathLike,
              metadata: dict | None = None, bf16: set[str] | None = None) -> None:
    """Write a .safetensors file. Names in ``bf16`` are stored as bfloat16."""
    bf16 = bf16 or set()
    header: dict = {}
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        if name in bf16:
            data, tag = f32_to_bf16(arr), "BF16"
        else:
            data = np.ascontiguousarray(arr)
            tag = _TAGS[data.dtype.str]
        b = data.tobytes()
        header[name] = {"dtype": tag, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    hjson = json.dumps(header, separators=(",", ":")).encode()
    hjson += b" " * (-len(hjson) % 8)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)
