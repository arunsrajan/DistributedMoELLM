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
MAX_ROWS = 1 << 30  # all sizes go through the kernels
ROW_DECODE_FROM = 4  # from this many input rows, decode each weight row once

if os.environ.get("MOELAN_KERNELS", "1") != "0":
    try:
        from numba import njit, prange
        ENABLED = True
    except ImportError:  # pragma: no cover - depends on the environment
        pass

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
                        b = raw[o, b0 + k] ^ xor
                        part += np.float32(b & 15) * xe[r, b0 + k] + np.float32(b >> 4) * xo[r, b0 + k]
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
                    b = raw[o, g * half + k] ^ xor
                    w[g * half + k] = np.float32(np.int32(b & 15) - 8) * sc
                    w[H + g * half + k] = np.float32(np.int32(b >> 4) - 8) * sc
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
