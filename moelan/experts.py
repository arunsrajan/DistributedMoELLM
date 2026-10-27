"""Routed-expert storage and compute.

An :class:`ExpertStore` holds some experts resident in RAM ("pinned": the ones
this node owns) and can stream any other expert from disk through a small LRU,
like Colibri's expert cache. The same class runs on workers and, as the
fallback path, on the coordinator.

Experts can be resident in three formats, a speed/RAM trade-off (numpy, one
768x2048 matrix-vector product on a laptop CPU):

  ======  ===================  =========================  ===========
  format  RAM per Qwen3-30B    how it computes            time/matrix
          expert (3 matrices)
  ======  ===================  =========================  ===========
  int4    2.7 MB               LUT unpack, scale, matmul  ~10 ms
  int8    5.0 MB               fused dequant, matmul      ~2.7 ms
  f32     18.9 MB              BLAS matmul                ~0.25 ms
  ======  ===================  =========================  ===========

So spreading experts over more machines buys speed twice: more RAM lets each
node keep a faster format, and every node computes its experts in parallel.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import kernels
from .config import ModelConfig
from .safetensors_io import SafeTensorDir, SafeTensorFile, decode_tensor, save_file

ExpertKey = tuple[int, int]  # (layer, expert)
FORMATS = ("f32", "int8", "int4")

ROWS = 64  # rows dequantized per step: the f32 scratch (64 x 2048 x 4 = 512 KiB) stays in L2

_tls = threading.local()


def _scratch(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-thread (uint8, f32) scratch blocks, so the hot loop never allocates."""
    b = getattr(_tls, "b", None)
    if b is None or b[0].size < rows * cols:
        b = _tls.b = (np.empty(rows * cols, np.uint8), np.empty(rows * cols, np.float32))
    return b[0][:rows * cols].reshape(rows, cols), b[1][:rows * cols].reshape(rows, cols)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Matrix representations: all expose ``matmul_t(x) == x @ W.T`` and ``nbytes``.
# Quantized ones dequantize ROWS output rows at a time into cache-sized scratch,
# so main memory only ever sees the compact weights.
# ---------------------------------------------------------------------------

class F32Matrix:
    __slots__ = ("w",)

    def __init__(self, w: np.ndarray):
        self.w = np.ascontiguousarray(w, dtype=np.float32)

    def matmul_t(self, x):
        if kernels.ENABLED:
            return kernels.f32_matmul_t(x, self.w)
        return x @ self.w.T

    def dequant(self):
        return self.w

    @property
    def nbytes(self):
        return self.w.nbytes


class Half16:
    """A matrix kept in its stored 16-bit form (IEEE half or bfloat16 bits).
    The kernels widen each row to f32 exactly while computing, so results equal
    the f32 path with half the memory traffic."""

    __slots__ = ("bits", "bf16")

    def __init__(self, bits: np.ndarray, bf16: bool):
        self.bits, self.bf16 = np.ascontiguousarray(bits, np.uint16), bf16

    def dequant(self) -> np.ndarray:
        if self.bf16:
            return (self.bits.astype(np.uint32) << 16).view(np.float32)
        return self.bits.view(np.float16).astype(np.float32)

    def matmul_t(self, x):
        if kernels.ENABLED and x.shape[0] <= 32:   # long prompts: widen once, then BLAS
            return kernels.h16_matmul_t(x, self.bits, self.bf16)
        return x @ self.dequant().T

    @property
    def nbytes(self):
        return self.bits.nbytes


class GroupQuant:
    """int8 values with one f32 scale per ``gs`` inputs of each output row."""

    __slots__ = ("q", "s", "gs")

    def __init__(self, q: np.ndarray, s: np.ndarray, gs: int):
        self.q, self.s, self.gs = q, s, gs

    @classmethod
    def quantize(cls, w: np.ndarray, gs: int = 64) -> "GroupQuant":
        O, I = w.shape
        if I % gs:
            gs = I
        g = w.reshape(O, I // gs, gs)
        amax = np.abs(g).max(axis=2)
        s = np.where(amax == 0, 1.0, amax / 127.0).astype(np.float32)
        q = np.clip(np.rint(g / s[:, :, None]), -127, 127).astype(np.int8).reshape(O, I)
        return cls(q, s, gs)

    def dequant(self) -> np.ndarray:
        O, I = self.q.shape
        return (self.q.reshape(O, I // self.gs, self.gs) * self.s[:, :, None]).reshape(O, I)

    def matmul_t(self, x):
        if kernels.ENABLED and x.shape[0] <= kernels.MAX_ROWS:
            return kernels.q8_matmul_t(x, self.q, self.s, self.gs)
        O, I = self.q.shape
        ng, gs = I // self.gs, self.gs
        y = np.empty((x.shape[0], O), np.float32)
        for r in range(0, O, ROWS):
            n = min(ROWS, O - r)
            _, f = _scratch(n, I)
            np.multiply(self.q[r:r + n].reshape(n, ng, gs), self.s[r:r + n, :, None],
                        out=f.reshape(n, ng, gs))
            y[:, r:r + n] = x @ f.T
        return y

    @property
    def nbytes(self):
        return self.q.nbytes + self.s.nbytes


class Packed4:
    """Colibri's int4: two signed 4-bit values per byte (low nibble = even
    column), one f32 scale per ``gs`` inputs.

    Kernel: XOR-ing a byte with 0x88 turns each two's-complement nibble v into
    v + 8 (0..15), so W = (u - 8) * s and

        x @ W.T = x @ (u * s).T - 8 * (group sums of x) @ s.T

    which needs only and/shift on bytes plus one multiply, no table lookups.
    Resident copies are XOR-ed once at load (``offset=True``)."""

    __slots__ = ("raw", "s", "gs", "shape", "offset")

    def __init__(self, raw: np.ndarray, s: np.ndarray, gs: int, shape: tuple[int, int],
                 offset: bool = False):
        self.raw, self.s, self.gs, self.shape, self.offset = raw, s, gs, shape, offset

    def resident(self) -> "Packed4":
        """An in-RAM copy in offset form (off the memory map)."""
        raw = self.raw if self.offset else np.bitwise_xor(self.raw, 0x88)
        return Packed4(np.ascontiguousarray(raw), np.array(self.s), self.gs, self.shape, True)

    def _unpack(self, r: int, n: int) -> np.ndarray:
        """Rows r..r+n as unsigned nibbles u = v + 8, uint8 (n, I)."""
        I = self.shape[1]
        rr = self.raw[r:r + n]
        if not self.offset:
            rr = np.bitwise_xor(rr, 0x88)
        u, _ = _scratch(n, I)
        u3 = u.reshape(n, I // 2, 2)
        np.bitwise_and(rr, 15, out=u3[:, :, 0])
        np.right_shift(rr, 4, out=u3[:, :, 1])
        return u

    def to_int8(self) -> GroupQuant:
        O, I = self.shape
        q = np.empty((O, I), np.int8)
        for r in range(0, O, ROWS):
            n = min(ROWS, O - r)
            np.subtract(self._unpack(r, n), 8, out=q[r:r + n].view(np.uint8))  # wraps to int8
        return GroupQuant(q, np.array(self.s), self.gs)

    def dequant(self) -> np.ndarray:
        return self.to_int8().dequant()

    def matmul_t(self, x):
        if kernels.ENABLED and x.shape[0] <= kernels.MAX_ROWS:
            return kernels.q4_matmul_t(x, self.raw, self.s, self.gs, self.offset)
        O, I = self.shape
        ng, gs = I // self.gs, self.gs
        y = np.empty((x.shape[0], O), np.float32)
        for r in range(0, O, ROWS):
            n = min(ROWS, O - r)
            u = self._unpack(r, n)
            _, f = _scratch(n, I)
            np.multiply(u.reshape(n, ng, gs), self.s[r:r + n, :, None], out=f.reshape(n, ng, gs))
            y[:, r:r + n] = x @ f.T
        y -= 8.0 * (x.reshape(-1, ng, gs).sum(2) @ self.s.T)
        return y

    @property
    def nbytes(self):
        return self.raw.nbytes + self.s.nbytes


class QuantMatrix(GroupQuant):
    """Per-row int8 (one group spanning the whole row); kept for API compatibility."""

    def __init__(self, w: np.ndarray):
        g = GroupQuant.quantize(w, gs=w.shape[1])
        super().__init__(g.q, g.s, g.gs)


class Expert:
    """A SwiGLU expert with gate and up fused into one (2*inter, hidden) matrix."""

    __slots__ = ("gu", "down", "inter")

    def __init__(self, gu, down, inter: int):
        self.gu, self.down, self.inter = gu, down, inter

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = self.gu.matmul_t(x)
        g, u = h[:, :self.inter], h[:, self.inter:]
        return self.down.matmul_t(silu(g) * u)

    @property
    def nbytes(self):
        return self.gu.nbytes + self.down.nbytes


def expert_ffn(x: np.ndarray, gate, up, down) -> np.ndarray:
    """SwiGLU with plain f32 matrices (dense MLP layers, shared experts)."""
    return (silu(x @ gate.T) * (x @ up.T)) @ down.T


def to_format(e: Expert, fmt: str, gs: int = 64) -> Expert:
    """Convert an expert to a resident format."""
    def conv(m):
        if fmt == "f32":
            return m if isinstance(m, F32Matrix) else F32Matrix(m.dequant())
        if fmt == "int8":
            if isinstance(m, Packed4):
                return m.to_int8()
            if isinstance(m, GroupQuant):
                return m
            return GroupQuant.quantize(m.w, gs)
        if fmt == "int4":
            if isinstance(m, Packed4):
                if kernels.ENABLED and not m.offset:
                    # The kernels read colibri's bytes as they are: keep the
                    # memory-mapped view (no second copy next to the OS page
                    # cache) and fault its pages in now.
                    np.bitwise_or.reduce(m.raw.reshape(-1)[::4096])
                    return m
                return m.resident()
            raise ValueError("int4 residency needs an int4 checkpoint (e.g. a colibri container)")
        raise ValueError(f"unknown expert format {fmt!r}")
    return Expert(conv(e.gu), conv(e.down), e.inter)


def default_format(cfg: ModelConfig) -> str:
    """For int4 checkpoints: int4 as stored when the compiled kernels are there
    (memory-mapped, no copy), else int8 (numpy is ~3x faster on it). f32 otherwise."""
    if cfg.expert_layout == "colibri_int4":
        return "int4" if kernels.ENABLED else "int8"
    return "f32"


def resident_bytes(cfg: ModelConfig, fmt: str) -> int:
    """RAM one routed expert takes in ``fmt``."""
    n = 3 * cfg.hidden_size * cfg.moe_intermediate_size
    gs = cfg.expert_gs or 64
    if fmt == "f32":
        return 4 * n
    if fmt == "int8":
        return n + 4 * n // gs
    return n // 2 + 4 * n // gs


# ---------------------------------------------------------------------------
# Where expert weights come from. Each source yields the expert's tensors as
# named raw blobs (to ship over the network) and as an :class:`Expert`.
# ---------------------------------------------------------------------------

def build_expert(cfg: ModelConfig, get) -> Expert:
    """Assemble an Expert from named tensors; ``get(name, raw=False)``."""
    H, I = cfg.hidden_size, cfg.moe_intermediate_size
    if cfg.expert_layout == "colibri_int4":
        raw = get("w")                     # uint8: gate | up | down, row-major, packed
        s = get("qs")                      # f32 scales, same order
        gs = cfg.expert_gs
        ngu = 2 * I * H // 2
        nsu = 2 * I * (H // gs)
        gu = Packed4(raw[:ngu].reshape(2 * I, H // 2), s[:nsu].reshape(2 * I, H // gs), gs, (2 * I, H))
        down = Packed4(raw[ngu:].reshape(H, I // 2), s[nsu:].reshape(H, I // gs), gs, (H, I))
        return Expert(gu, down, I)
    g, u, d = get("gate"), get("up"), get("down")
    return Expert(F32Matrix(np.concatenate([g, u], 0)), F32Matrix(d), I)


def _part_names(cfg: ModelConfig) -> tuple[str, ...]:
    return ("w", "qs") if cfg.expert_layout == "colibri_int4" else ("gate", "up", "down")


class CheckpointSource:
    """Experts read straight out of the original checkpoint (memory-mapped)."""

    def __init__(self, model_dir: str | Path, cfg: ModelConfig):
        self.cfg = cfg
        self.st = SafeTensorDir(model_dir)

    def _names(self, key: ExpertKey) -> dict[str, str]:
        return dict(zip(_part_names(self.cfg), self.cfg.expert_names(*key)))

    def has(self, key: ExpertKey) -> bool:
        return self.cfg.expert_names(*key)[0] in self.st

    def raw(self, key: ExpertKey) -> dict[str, tuple[bytes, str, list[int]]]:
        out = {}
        for part, name in self._names(key).items():
            buf, tag, shape = self.st.raw(name)
            out[part] = (bytes(buf), tag, shape)
        return out

    def load(self, key: ExpertKey) -> Expert:
        names = self._names(key)

        def get(part):
            buf, tag, shape = self.st.raw(names[part])
            if tag == "U8":   # packed int4: a view of the memory map, no copy
                return buf
            if tag == "F32":  # scales: also a view
                return np.frombuffer(buf, np.float32).reshape(shape)
            return self.st.get(names[part])
        return build_expert(self.cfg, get)

    def nbytes(self, key: ExpertKey) -> int:
        return sum(self.st.nbytes(n) for n in self.cfg.expert_names(*key))

class CacheSource:
    """Experts a coordinator pushed over the network, one small file each."""

    def __init__(self, root: str | Path, cfg: ModelConfig | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

    def _path(self, key: ExpertKey) -> Path:
        return self.root / f"L{key[0]:03d}_E{key[1]:04d}.safetensors"

    def has(self, key: ExpertKey) -> bool:
        return self._path(key).exists()

    def keys(self) -> list[ExpertKey]:
        out = []
        for p in self.root.glob("L*_E*.safetensors"):
            l, e = p.stem[1:].split("_E")
            out.append((int(l), int(e)))
        return sorted(out)

    def put(self, key: ExpertKey, raw: dict[str, tuple[bytes, str, list[int]]]) -> None:
        # Keep the original dtype on disk.
        arrays, bf16 = {}, set()
        for part, (buf, tag, shape) in raw.items():
            if tag == "BF16":
                bf16.add(part)
            arrays[part] = decode_tensor(buf, tag, shape, dtype=None)
        save_file(arrays, self._path(key), bf16=bf16)

    def load(self, key: ExpertKey) -> Expert:
        f = SafeTensorFile(self._path(key))  # memory-mapped, like the checkpoint

        def get(part):
            buf, tag, shape = f.raw(part)
            if tag == "U8":
                return buf
            if tag == "F32":
                return np.frombuffer(buf, np.float32).reshape(shape)
            return decode_tensor(buf, tag, shape, np.float32)
        return build_expert(self.cfg, get)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

def default_threads() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))


class ExpertStore:
    def __init__(self, sources: list, fmt: str = "f32", lru_size: int = 0,
                 threads: int | None = None, gs: int = 64):
        if fmt not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}")
        self.sources = [s for s in sources if s is not None]
        self.fmt = fmt
        self.gs = gs
        self.lru_size = lru_size
        self.pinned: dict[ExpertKey, Expert] = {}
        self.lru: OrderedDict[ExpertKey, Expert] = OrderedDict()
        self.mapped: dict[ExpertKey, Expert] = {}  # zero-copy views, OS page cache does the caching
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self.threads = threads or default_threads()
        self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="moelan-expert") \
            if self.threads > 1 else None

    @property
    def quant(self) -> str:  # older name
        return self.fmt

    def _read(self, key: ExpertKey, fmt: str | None = None) -> Expert:
        for src in self.sources:
            if src.has(key):
                e = src.load(key)
                return to_format(e, fmt, self.gs) if fmt else e
        raise KeyError(f"expert {key} is not available on this node")

    def can_serve(self, key: ExpertKey) -> bool:
        return key in self.pinned or any(s.has(key) for s in self.sources)

    def pin(self, keys) -> None:
        keys = [tuple(k) for k in keys if tuple(k) not in self.pinned]

        def load(k):
            return k, self.lru.pop(k, None) or self._read(k, self.fmt)
        it = self._pool.map(load, keys) if self._pool else map(load, keys)
        for k, e in it:
            self.pinned[k] = e

    def unpin_all_except(self, keep) -> None:
        keep = {tuple(k) for k in keep}
        for key in list(self.pinned):
            if key not in keep:
                del self.pinned[key]

    def get(self, key: ExpertKey) -> Expert:
        w = self.pinned.get(key)
        if w is not None:
            self.hits += 1
            return w
        with self._lock:
            w = self.lru.get(key)
            if w is not None:
                self.lru.move_to_end(key)
                self.hits += 1
                return w
        w = self.mapped.get(key)
        if w is not None:
            self.misses += 1
            return w
        self.misses += 1
        # Not resident: compute straight from the (memory-mapped) source format.
        w = self._read(key)
        if isinstance(w.gu, Packed4) and not w.gu.offset:
            self.mapped[key] = w  # views of the memory map cost no RAM: keep them all
        elif self.lru_size > 0:
            with self._lock:
                self.lru[key] = w
                while len(self.lru) > self.lru_size:
                    self.lru.popitem(last=False)
        return w

    def resident_bytes(self) -> int:
        return sum(e.nbytes for e in self.pinned.values()) + \
            sum(e.nbytes for e in self.lru.values())

    def compute(self, layer: int, x: np.ndarray, groups) -> np.ndarray:
        """Weighted sum of the routed experts' outputs for the given rows.

        ``groups`` is an iterable of ``(expert, rows, weights)``: the expert id,
        the row indices of ``x`` routed to it, and their gate weights. Experts
        run in parallel threads (numpy releases the GIL in its kernels).
        """
        def one(g):
            expert, rows, weights = g
            e = self.get((layer, int(expert)))
            return rows, e(x[rows]) * weights[:, None]

        groups = list(groups)
        out = np.zeros_like(x, dtype=np.float32)
        # numba kernels parallelize inside each matmul (and their thread pool is
        # not re-entrant), so with them the experts run one after another.
        parallel = self._pool and len(groups) > 1 and not kernels.ENABLED
        it = self._pool.map(one, groups) if parallel else map(one, groups)
        for rows, y in it:
            out[rows] += y  # a token picks an expert at most once
        return out
