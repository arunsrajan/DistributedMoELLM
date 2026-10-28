"""Optional compiled kernels (numba). Used automatically when numba imports;
``MOELAN_KERNELS=0`` turns them off. Without them moelan falls back to the
pure-numpy blocked dequantization in :mod:`moelan.experts`.

The kernels compute ``x @ W.T`` for group-scaled int4/int8 weights straight
from the compact bytes, never writing f32 weights to memory, parallel over
output rows. For one to three input rows (decode) a weight byte is decoded inside the dot
product; for more rows (prefill) each weight row is decoded once into a small
f32 buffer and reused for every input row.
"""

from __future__ import annotations

import os

import numpy as np

ENABLED = False
WHY_DISABLED = ""
MAX_ROWS = 1 << 30  # all sizes go through the kernels
ROW_DECODE_FROM = 4  # from this many input rows, decode each weight row once
BLAS_FROM = 8        # from this many rows (prompt reading), widen the matrix once and use BLAS

if os.environ.get("MOELAN_KERNELS", "1") != "0":
    try:
        from numba import njit, prange
        ENABLED = True
    except ImportError:  # pragma: no cover - depends on the environment
        WHY_DISABLED = "numba is not installed for this Python"
else:
    WHY_DISABLED = "MOELAN_KERNELS=0"


def warn_if_slow(stream=None) -> bool:
    """Print a loud notice when the compiled kernels are off. Returns True if so."""
    if ENABLED:
        return False
    import sys
    stream = stream or sys.stderr
    v = "%d.%d" % sys.version_info[:2]
    bar = "!" * 78
    print(f"""{bar}
 moelan is running WITHOUT its compiled kernels ({WHY_DISABLED}, Python {v}).
 The pure-numpy fallback is about 10x slower: roughly 0.3 tokens/s on
 Qwen3-30B-A3B instead of 3-5. Use a Python that numba supports (3.10 - 3.13):
     py -3.12 -m venv .venv312
     .venv312\\Scripts\\python -m pip install -e ".[fast,hf]"
     .venv312\\Scripts\\python -m moelan ...
{bar}""", file=stream, flush=True)
    return True

if ENABLED:
    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _q4(xe, xo, raw, s, gs, xor):
        # xe / xo: the even / odd input columns, so both nibble streams read
        # contiguous memory and the loop vectorizes. Nibbles are taken unsigned
        # (u = v + 8); the caller subtracts 8 * sum_g s[o, g] * sum(x in g).
        n = xe.shape[0]
        O = raw.shape[0]
        half = gs // 2
        ng = raw.shape[1] // half
        y = np.empty((n, O), np.float32)
        for o in prange(O):
            for r in range(n):
                acc = np.float32(0.0)
                for g in range(ng):
                    part = np.float32(0.0)
                    b0 = g * half
                    for k in range(half):
                        b = np.int32(raw[o, b0 + k] ^ xor)  # int32: the u8 -> f32 widening vectorizes
                        part += np.float32(b & np.int32(15)) * xe[r, b0 + k] + np.float32(b >> np.int32(4)) * xo[r, b0 + k]
                    acc += part * s[o, g]
                y[r, o] = acc
        return y

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _q4_rows(xp, raw, s, gs, xor):
        # xp: inputs permuted to [even columns | odd columns]; each weight row is
        # decoded once, planar, into w, then dotted with every input row.
        n, I = xp.shape
        O = raw.shape[0]
        half = gs // 2
        ng = I // gs
        H = I // 2
        y = np.empty((n, O), np.float32)
        for o in prange(O):
            w = np.empty(I, np.float32)
            for g in range(ng):
                sc = s[o, g]
                for k in range(half):
                    b = np.int32(raw[o, g * half + k] ^ xor)
                    w[g * half + k] = np.float32((b & np.int32(15)) - np.int32(8)) * sc
                    w[H + g * half + k] = np.float32((b >> np.int32(4)) - np.int32(8)) * sc
            for r in range(n):
                acc = np.float32(0.0)
                for i in range(I):
                    acc += w[i] * xp[r, i]
                y[r, o] = acc
        return y

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _q8_rows(x, q, s, gs):
        n, I = x.shape
        O = q.shape[0]
        ng = I // gs
        y = np.empty((n, O), np.float32)
        for o in prange(O):
            w = np.empty(I, np.float32)
            for g in range(ng):
                sc = s[o, g]
                for k in range(gs):
                    w[g * gs + k] = np.float32(q[o, g * gs + k]) * sc
            for r in range(n):
                acc = np.float32(0.0)
                for i in range(I):
                    acc += w[i] * x[r, i]
                y[r, o] = acc
        return y

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _q8(x, q, s, gs):
        n, I = x.shape
        O = q.shape[0]
        ng = I // gs
        y = np.empty((n, O), np.float32)
        for o in prange(O):
            for r in range(n):
                acc = np.float32(0.0)
                for g in range(ng):
                    part = np.float32(0.0)
                    x0 = g * gs
                    for k in range(gs):
                        part += np.float32(q[o, x0 + k]) * x[r, x0 + k]
                    acc += part * s[o, g]
                y[r, o] = acc
        return y


    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _f32(x, w):
        n, I = x.shape
        O = w.shape[0]
        y = np.empty((n, O), np.float32)
        for o in prange(O):
            for r in range(n):
                acc = np.float32(0.0)
                for i in range(I):
                    acc += w[o, i] * x[r, i]
                y[r, o] = acc
        return y


    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _h16(x, bits, bf16):
        # 16-bit weights as stored, widened to f32 one row at a time (exactly)
        # and dotted with every input row. IEEE half: put exponent+mantissa in
        # the f32 position and multiply by 2^112, which rebiases normals and
        # turns half subnormals into the right normal f32 - branch-free, so
        # the loops vectorize. bfloat16 is just the top half of an f32.
        n, I = x.shape
        O = bits.shape[0]
        y = np.empty((n, O), np.float32)
        rebias = np.float32(5.192296858534828e33)  # 2^112
        for o in prange(O):
            wf = np.empty(I, np.float32)
            wu = wf.view(np.uint32)
            if bf16:
                for i in range(I):
                    wu[i] = np.uint32(bits[o, i]) << 16
            else:
                for i in range(I):
                    h = np.uint32(bits[o, i])
                    wu[i] = ((h & 0x7FFF) << 13) | ((h & 0x8000) << 16)
                for i in range(I):
                    wf[i] = wf[i] * rebias
            for r in range(n):
                acc = np.float32(0.0)
                for i in range(I):
                    acc += wf[i] * x[r, i]
                y[r, o] = acc
        return y

    @njit(fastmath=True, cache=True, nogil=True)
    def _q4_dequant(raw, s, gs, xor, out):
        # Single-threaded and GIL-free, so several experts can be widened at
        # once from Python threads (the parallel kernels are not re-entrant).
        O = raw.shape[0]
        half = gs // 2
        ng = out.shape[1] // gs
        for o in range(O):
            for g in range(ng):
                sc = s[o, g]
                for j in range(half):
                    b = np.int32(raw[o, g * half + j] ^ xor)
                    out[o, g * gs + 2 * j] = np.float32((b & np.int32(15)) - np.int32(8)) * sc
                    out[o, g * gs + 2 * j + 1] = np.float32((b >> np.int32(4)) - np.int32(8)) * sc

    @njit(fastmath=True, cache=True, nogil=True)
    def _q8_dequant(q, s, gs, out):
        O, I = q.shape
        for o in range(O):
            for g in range(I // gs):
                sc = s[o, g]
                for j in range(gs):
                    out[o, g * gs + j] = np.float32(q[o, g * gs + j]) * sc


def q4_dequant(raw, s, gs: int, offset: bool, out: np.ndarray) -> np.ndarray:
    _q4_dequant(raw, s, gs, np.uint8(0 if offset else 0x88), out)
    return out


def q8_dequant(q, s, gs: int, out: np.ndarray) -> np.ndarray:
    _q8_dequant(q, s, gs, out)
    return out


def h16_matmul_t(x: np.ndarray, bits: np.ndarray, bf16: bool) -> np.ndarray:
    """x @ W.T for W stored as 16-bit floats (uint16 bits), decoded exactly."""
    return _h16(np.ascontiguousarray(x, np.float32), bits, bf16)


def f32_matmul_t(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x @ W.T for f32 W, parallel without OpenBLAS threads; big batches use BLAS."""
    if x.shape[0] > 16:
        return x @ w.T
    return _f32(np.ascontiguousarray(x, np.float32), w)


def q4_matmul_t(x: np.ndarray, raw: np.ndarray, s: np.ndarray, gs: int, offset: bool) -> np.ndarray:
    """x @ W.T for colibri int4 bytes; ``offset`` means raw was XOR-ed with 0x88."""
    x = np.ascontiguousarray(x, np.float32)
    xor = np.uint8(0 if offset else 0x88)
    if x.shape[0] >= ROW_DECODE_FROM:
        return _q4_rows(np.concatenate([x[:, 0::2], x[:, 1::2]], 1), raw, s, gs, xor)
    xe, xo = np.ascontiguousarray(x[:, 0::2]), np.ascontiguousarray(x[:, 1::2])
    y = _q4(xe, xo, raw, s, gs, xor)
    y -= 8.0 * (x.reshape(x.shape[0], -1, gs).sum(2) @ s.T)
    return y


def q8_matmul_t(x: np.ndarray, q: np.ndarray, s: np.ndarray, gs: int) -> np.ndarray:
    x = np.ascontiguousarray(x, np.float32)
    return (_q8_rows if x.shape[0] >= ROW_DECODE_FROM else _q8)(x, q, s, gs)
