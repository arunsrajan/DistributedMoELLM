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

import math
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


    # ---- GGUF (llama.cpp) block formats --------------------------------------
    # Each _dq_* decodes one row of blocks into f32, following llama.cpp's
    # dequantize_row_* (ggml-quants.c); checked against gguf-py in the tests.

    @njit(inline="always")
    def _f16(lo, hi):
        h = np.int32(lo) | (np.int32(hi) << 8)
        e = (h >> 10) & 0x1F
        m = h & 0x3FF
        if e == 0:
            v = m * 5.9604644775390625e-08
        elif e == 31:
            v = np.inf
        else:
            v = math.ldexp(1024 + m, e - 25)
        return np.float32(-v if h & 0x8000 else v)

    @njit(inline="always")
    def _i8(b):
        v = np.int32(b)
        return v - 256 if v > 127 else v

    @njit(inline="always")
    def _dq_q8_0(row, out):
        for b in range(out.shape[0] // 32):
            p = b * 34
            d = _f16(row[p], row[p + 1])
            for j in range(32):
                out[b * 32 + j] = d * np.float32(_i8(row[p + 2 + j]))

    @njit(inline="always")
    def _dq_q4_0(row, out):
        for b in range(out.shape[0] // 32):
            p = b * 18
            d = _f16(row[p], row[p + 1])
            for j in range(16):
                q = np.int32(row[p + 2 + j])
                out[b * 32 + j] = d * np.float32((q & 15) - 8)
                out[b * 32 + j + 16] = d * np.float32((q >> 4) - 8)

    @njit(inline="always")
    def _dq_q4_1(row, out):
        for b in range(out.shape[0] // 32):
            p = b * 20
            d = _f16(row[p], row[p + 1])
            mn = _f16(row[p + 2], row[p + 3])
            for j in range(16):
                q = np.int32(row[p + 4 + j])
                out[b * 32 + j] = d * np.float32(q & 15) + mn
                out[b * 32 + j + 16] = d * np.float32(q >> 4) + mn

    @njit(inline="always")
    def _dq_q5(row, out, with_min):
        size = 24 if with_min else 22
        for b in range(out.shape[0] // 32):
            p = b * size
            d = _f16(row[p], row[p + 1])
            mn = _f16(row[p + 2], row[p + 3]) if with_min else np.float32(0.0)
            h0 = p + (4 if with_min else 2)
            qh = np.int64(row[h0]) | (np.int64(row[h0 + 1]) << 8) | (np.int64(row[h0 + 2]) << 16) \
                | (np.int64(row[h0 + 3]) << 24)
            q0 = h0 + 4
            off = 0 if with_min else 16
            for j in range(16):
                q = np.int64(row[q0 + j])
                x0 = ((q & 15) | (((qh >> j) << 4) & 0x10)) - off
                x1 = ((q >> 4) | ((qh >> (j + 12)) & 0x10)) - off
                out[b * 32 + j] = d * np.float32(x0) + mn
                out[b * 32 + j + 16] = d * np.float32(x1) + mn

    @njit(inline="always")
    def _scale_min_k4(j, row, s0):
        if j < 4:
            return np.int32(row[s0 + j]) & 63, np.int32(row[s0 + j + 4]) & 63
        sc = (np.int32(row[s0 + j + 4]) & 0xF) | ((np.int32(row[s0 + j - 4]) >> 6) << 4)
        mn = (np.int32(row[s0 + j + 4]) >> 4) | ((np.int32(row[s0 + j]) >> 6) << 4)
        return sc, mn

    @njit(inline="always")
    def _dq_q4_k(row, out):
        for b in range(out.shape[0] // 256):
            p = b * 144
            d = _f16(row[p], row[p + 1])
            dmin = _f16(row[p + 2], row[p + 3])
            q = p + 16
            y = b * 256
            for k in range(4):
                sc, m = _scale_min_k4(2 * k, row, p + 4)
                d1, m1 = d * np.float32(sc), dmin * np.float32(m)
                sc, m = _scale_min_k4(2 * k + 1, row, p + 4)
                d2, m2 = d * np.float32(sc), dmin * np.float32(m)
                for l in range(32):
                    v = np.int32(row[q + l])
                    out[y + l] = d1 * np.float32(v & 15) - m1
                    out[y + 32 + l] = d2 * np.float32(v >> 4) - m2
                q += 32
                y += 64

    @njit(inline="always")
    def _dq_q5_k(row, out):
        for b in range(out.shape[0] // 256):
            p = b * 176
            d = _f16(row[p], row[p + 1])
            dmin = _f16(row[p + 2], row[p + 3])
            qh = p + 16
            ql = p + 48
            y = b * 256
            u1, u2 = 1, 2
            for k in range(4):
                sc, m = _scale_min_k4(2 * k, row, p + 4)
                d1, m1 = d * np.float32(sc), dmin * np.float32(m)
                sc, m = _scale_min_k4(2 * k + 1, row, p + 4)
                d2, m2 = d * np.float32(sc), dmin * np.float32(m)
                for l in range(32):
                    v = np.int32(row[ql + l])
                    h = np.int32(row[qh + l])
                    out[y + l] = d1 * np.float32((v & 15) + (16 if h & u1 else 0)) - m1
                    out[y + 32 + l] = d2 * np.float32((v >> 4) + (16 if h & u2 else 0)) - m2
                ql += 32
                y += 64
                u1 <<= 2
                u2 <<= 2

    @njit(inline="always")
    def _dq_q6_k(row, out):
        for b in range(out.shape[0] // 256):
            p = b * 210
            d = _f16(row[p + 208], row[p + 209])
            ql, qh, sc = p, p + 128, p + 192
            y = b * 256
            for n in range(2):
                for l in range(32):
                    s = l // 16
                    a = np.int32(row[ql + l])
                    c = np.int32(row[ql + l + 32])
                    h = np.int32(row[qh + l])
                    q1 = ((a & 15) | (((h >> 0) & 3) << 4)) - 32
                    q2 = ((c & 15) | (((h >> 2) & 3) << 4)) - 32
                    q3 = ((a >> 4) | (((h >> 4) & 3) << 4)) - 32
                    q4 = ((c >> 4) | (((h >> 6) & 3) << 4)) - 32
                    out[y + l] = d * np.float32(_i8(row[sc + s]) * q1)
                    out[y + l + 32] = d * np.float32(_i8(row[sc + s + 2]) * q2)
                    out[y + l + 64] = d * np.float32(_i8(row[sc + s + 4]) * q3)
                    out[y + l + 96] = d * np.float32(_i8(row[sc + s + 6]) * q4)
                y += 128
                ql += 64
                qh += 32
                sc += 8

    @njit(inline="always")
    def _dq_q2_k(row, out):
        for b in range(out.shape[0] // 256):
            p = b * 84
            d = _f16(row[p + 80], row[p + 81])
            dmin = _f16(row[p + 82], row[p + 83])
            q = p + 16
            y = b * 256
            i = 0
            for n in range(2):
                shift = 0
                for j in range(4):
                    s = np.int32(row[p + i])
                    i += 1
                    dl, ml = d * np.float32(s & 15), dmin * np.float32(s >> 4)
                    for l in range(16):
                        out[y] = dl * np.float32((np.int32(row[q + l]) >> shift) & 3) - ml
                        y += 1
                    s = np.int32(row[p + i])
                    i += 1
                    dl, ml = d * np.float32(s & 15), dmin * np.float32(s >> 4)
                    for l in range(16):
                        out[y] = dl * np.float32((np.int32(row[q + l + 16]) >> shift) & 3) - ml
                        y += 1
                    shift += 2
                q += 32

    @njit(inline="always")
    def _dq_q3_k(row, out):
        km1, km2 = np.int64(0x03030303), np.int64(0x0F0F0F0F)
        for b in range(out.shape[0] // 256):
            p = b * 110
            d = _f16(row[p + 108], row[p + 109])
            s0 = p + 96
            a0 = np.int64(row[s0]) | (np.int64(row[s0 + 1]) << 8) | (np.int64(row[s0 + 2]) << 16) | (np.int64(row[s0 + 3]) << 24)
            a1 = np.int64(row[s0 + 4]) | (np.int64(row[s0 + 5]) << 8) | (np.int64(row[s0 + 6]) << 16) | (np.int64(row[s0 + 7]) << 24)
            t = np.int64(row[s0 + 8]) | (np.int64(row[s0 + 9]) << 8) | (np.int64(row[s0 + 10]) << 16) | (np.int64(row[s0 + 11]) << 24)
            words = ((a0 & km2) | (((t >> 0) & km1) << 4), (a1 & km2) | (((t >> 2) & km1) << 4),
                     ((a0 >> 4) & km2) | (((t >> 4) & km1) << 4), ((a1 >> 4) & km2) | (((t >> 6) & km1) << 4))
            scales = np.empty(16, np.int32)
            for w in range(4):
                for k in range(4):
                    scales[w * 4 + k] = np.int32((words[w] >> (8 * k)) & 0xFF) - 32
            hm, q = p, p + 32
            y = b * 256
            m = 1
            i = 0
            for n in range(2):
                shift = 0
                for j in range(4):
                    dl = d * np.float32(scales[i])
                    i += 1
                    for l in range(16):
                        v = ((np.int32(row[q + l]) >> shift) & 3) - (0 if np.int32(row[hm + l]) & m else 4)
                        out[y] = dl * np.float32(v)
                        y += 1
                    dl = d * np.float32(scales[i])
                    i += 1
                    for l in range(16):
                        v = ((np.int32(row[q + l + 16]) >> shift) & 3) - (0 if np.int32(row[hm + l + 16]) & m else 4)
                        out[y] = dl * np.float32(v)
                        y += 1
                    shift += 2
                    m <<= 1
                q += 32

    @njit(inline="always")
    def _gguf_row(row, qtype, out):
        if qtype == 8:
            _dq_q8_0(row, out)
        elif qtype == 12:
            _dq_q4_k(row, out)
        elif qtype == 14:
            _dq_q6_k(row, out)
        elif qtype == 13:
            _dq_q5_k(row, out)
        elif qtype == 2:
            _dq_q4_0(row, out)
        elif qtype == 3:
            _dq_q4_1(row, out)
        elif qtype == 6:
            _dq_q5(row, out, False)
        elif qtype == 7:
            _dq_q5(row, out, True)
        elif qtype == 10:
            _dq_q2_k(row, out)
        elif qtype == 11:
            _dq_q3_k(row, out)

    # Fused GGUF mat-vec for decoding (a few input rows): dot products per
    # 32- or 16-value group straight from the bytes, block scales and minimums
    # applied once per group (Q4_K/Q5_K: d*sc*sum(q*x) - dmin*m*sum(x)).

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _mv_q8_0(x, raw):
        n, cols = x.shape
        O = raw.shape[0]
        y = np.empty((n, O), np.float32)
        for o in prange(O):
            for r in range(n):
                acc = np.float32(0.0)
                for b in range(cols // 32):
                    p = b * 34
                    s = np.float32(0.0)
                    for j in range(32):
                        s += np.float32(_i8(raw[o, p + 2 + j])) * x[r, b * 32 + j]
                    acc += _f16(raw[o, p], raw[o, p + 1]) * s
                y[r, o] = acc
        return y

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _mv_q4_k(x, raw, xs):
        n, cols = x.shape
        O = raw.shape[0]
        y = np.empty((n, O), np.float32)
        for o in prange(O):
            row = raw[o]
            for r in range(n):
                acc = np.float32(0.0)
                for b in range(cols // 256):
                    p = b * 144
                    d = _f16(row[p], row[p + 1])
                    dmin = _f16(row[p + 2], row[p + 3])
                    for k in range(4):
                        sc1, m1 = _scale_min_k4(2 * k, row, p + 4)
                        sc2, m2 = _scale_min_k4(2 * k + 1, row, p + 4)
                        q = p + 16 + k * 32
                        x0 = b * 256 + k * 64
                        s1 = np.float32(0.0)
                        s2 = np.float32(0.0)
                        for l in range(32):
                            v = np.int32(row[q + l])
                            s1 += np.float32(v & np.int32(15)) * x[r, x0 + l]
                            s2 += np.float32(v >> np.int32(4)) * x[r, x0 + 32 + l]
                        g = x0 // 32
                        acc += d * (np.float32(sc1) * s1 + np.float32(sc2) * s2) \
                            - dmin * (np.float32(m1) * xs[r, g] + np.float32(m2) * xs[r, g + 1])
                y[r, o] = acc
        return y

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _mv_q5_k(x, raw, xs):
        n, cols = x.shape
        O = raw.shape[0]
        y = np.empty((n, O), np.float32)
        for o in prange(O):
            row = raw[o]
            for r in range(n):
                acc = np.float32(0.0)
                for b in range(cols // 256):
                    p = b * 176
                    d = _f16(row[p], row[p + 1])
                    dmin = _f16(row[p + 2], row[p + 3])
                    for k in range(4):
                        sc1, m1 = _scale_min_k4(2 * k, row, p + 4)
                        sc2, m2 = _scale_min_k4(2 * k + 1, row, p + 4)
                        ql = p + 48 + k * 32
                        u1 = np.int32(1) << np.int32(2 * k)
                        u2 = np.int32(2) << np.int32(2 * k)
                        x0 = b * 256 + k * 64
                        s1 = np.float32(0.0)
                        s2 = np.float32(0.0)
                        for l in range(32):
                            v = np.int32(row[ql + l])
                            h = np.int32(row[p + 16 + l])
                            s1 += np.float32((v & np.int32(15)) + ((h & u1) != 0) * 16) * x[r, x0 + l]
                            s2 += np.float32((v >> np.int32(4)) + ((h & u2) != 0) * 16) * x[r, x0 + 32 + l]
                        g = x0 // 32
                        acc += d * (np.float32(sc1) * s1 + np.float32(sc2) * s2) \
                            - dmin * (np.float32(m1) * xs[r, g] + np.float32(m2) * xs[r, g + 1])
                y[r, o] = acc
        return y

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _mv_q6_k(x, raw):
        n, cols = x.shape
        O = raw.shape[0]
        y = np.empty((n, O), np.float32)
        for o in prange(O):
            row = raw[o]
            for r in range(n):
                acc = np.float32(0.0)
                for b in range(cols // 256):
                    p = b * 210
                    d = _f16(row[p + 208], row[p + 209])
                    for half in range(2):
                        ql = p + half * 64
                        qh = p + 128 + half * 32
                        sc = p + 192 + half * 8
                        x0 = b * 256 + half * 128
                        for c in range(2):          # 16 values per scale
                            s1 = np.float32(0.0)
                            s2 = np.float32(0.0)
                            s3 = np.float32(0.0)
                            s4 = np.float32(0.0)
                            for l in range(c * 16, c * 16 + 16):
                                a = np.int32(row[ql + l])
                                cc = np.int32(row[ql + l + 32])
                                h = np.int32(row[qh + l])
                                s1 += np.float32(((a & 15) | ((h & 3) << 4)) - 32) * x[r, x0 + l]
                                s2 += np.float32(((cc & 15) | (((h >> 2) & 3) << 4)) - 32) * x[r, x0 + l + 32]
                                s3 += np.float32(((a >> 4) | (((h >> 4) & 3) << 4)) - 32) * x[r, x0 + l + 64]
                                s4 += np.float32(((cc >> 4) | (((h >> 6) & 3) << 4)) - 32) * x[r, x0 + l + 96]
                            acc += d * (np.float32(_i8(row[sc + c])) * s1 + np.float32(_i8(row[sc + c + 2])) * s2
                                        + np.float32(_i8(row[sc + c + 4])) * s3 + np.float32(_i8(row[sc + c + 6])) * s4)
                y[r, o] = acc
        return y

    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _gguf_mm(x, raw, qtype, cols):
        n = x.shape[0]
        O = raw.shape[0]
        y = np.empty((n, O), np.float32)
        for o in prange(O):
            w = np.empty(cols, np.float32)
            _gguf_row(raw[o], qtype, w)
            for r in range(n):
                acc = np.float32(0.0)
                for i in range(cols):
                    acc += w[i] * x[r, i]
                y[r, o] = acc
        return y

    @njit(cache=True, nogil=True)
    def _gguf_dequant(raw, qtype, out):
        # single-threaded: called from the expert thread pool (not re-entrant otherwise)
        for o in range(raw.shape[0]):
            _gguf_row(raw[o], qtype, out[o])


def q4_dequant(raw, s, gs: int, offset: bool, out: np.ndarray) -> np.ndarray:
    _q4_dequant(raw, s, gs, np.uint8(0 if offset else 0x88), out)
    return out


def q8_dequant(q, s, gs: int, out: np.ndarray) -> np.ndarray:
    _q8_dequant(q, s, gs, out)
    return out


def gguf_matmul_t(x: np.ndarray, raw: np.ndarray, qtype: int, cols: int) -> np.ndarray:
    """x @ W.T for a GGUF block-quantized W given as (rows, row bytes). A few
    rows (decoding) use the fused kernels where there is one, otherwise each
    weight row is decoded once and dotted with every input row."""
    x = np.ascontiguousarray(x, np.float32)
    if x.shape[0] < ROW_DECODE_FROM:
        if qtype == 12 or qtype == 13:
            xs = x.reshape(x.shape[0], -1, 32).sum(axis=2)
            return (_mv_q4_k if qtype == 12 else _mv_q5_k)(x, raw, xs)
        if qtype == 14:
            return _mv_q6_k(x, raw)
        if qtype == 8:
            return _mv_q8_0(x, raw)
    return _gguf_mm(x, raw, qtype, cols)


def gguf_dequant(raw: np.ndarray, qtype: int, out: np.ndarray) -> np.ndarray:
    _gguf_dequant(raw, qtype, out)
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
