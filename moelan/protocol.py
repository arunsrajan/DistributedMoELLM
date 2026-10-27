"""Wire protocol: length-prefixed frames of a JSON header plus raw array bytes.

Frame layout::

    b"MOEL" | u32 header_len | u64 payload_len | header (JSON utf-8) | payload

``header["arrays"]`` describes the numpy arrays packed back to back in the
payload (name, dtype, shape, offset). Raw expert bytes (``header["blobs"]``)
travel the same way. No pickle, so a worker never executes data it receives.
"""

from __future__ import annotations

import json
import socket
import struct

import numpy as np

MAGIC = b"MOEL"
PROTOCOL_VERSION = 1
_PREFIX = struct.Struct("<4sIQ")
MAX_HEADER = 64 * 1024 * 1024
MAX_PAYLOAD = 16 * 1024 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


def _recv_exact(sock: socket.socket, n: int) -> memoryview:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], min(n - got, 1 << 22))
        if k == 0:
            raise ConnectionError("peer closed the connection")
        got += k
    return view


def send_msg(sock: socket.socket, header: dict, arrays: dict[str, np.ndarray] | None = None,
             blobs: dict[str, bytes] | None = None) -> None:
    parts: list = []
    descr = []
    offset = 0
    for name, arr in (arrays or {}).items():
        arr = np.ascontiguousarray(arr)
        descr.append({"name": name, "dtype": arr.dtype.str, "shape": list(arr.shape),
                      "offset": offset, "nbytes": arr.nbytes})
        parts.append(memoryview(arr).cast("B"))
        offset += arr.nbytes
    bdescr = []
    for name, b in (blobs or {}).items():
        bdescr.append({"name": name, "offset": offset, "nbytes": len(b)})
        parts.append(memoryview(b))
        offset += len(b)
    header = dict(header, arrays=descr, blobs=bdescr)
    hbytes = json.dumps(header, separators=(",", ":")).encode()
    sock.sendall(_PREFIX.pack(MAGIC, len(hbytes), offset) + hbytes)
    for p in parts:
        sock.sendall(p)


def recv_msg(sock: socket.socket) -> tuple[dict, dict[str, np.ndarray], dict[str, memoryview]]:
    magic, hlen, plen = _PREFIX.unpack(_recv_exact(sock, _PREFIX.size))
    if magic != MAGIC:
        raise ProtocolError("bad magic: not a moelan peer")
    if hlen > MAX_HEADER or plen > MAX_PAYLOAD:
        raise ProtocolError("frame too large")
    header = json.loads(bytes(_recv_exact(sock, hlen)))
    payload = _recv_exact(sock, plen) if plen else memoryview(b"")
    arrays = {}
    for d in header.pop("arrays", []):
        end = d["offset"] + d["nbytes"]
        if end > plen:
            raise ProtocolError("array outside payload")
        dt = np.dtype(d["dtype"])
        if dt.kind not in "fiub":
            raise ProtocolError(f"refusing array dtype {dt}")
        arrays[d["name"]] = np.frombuffer(payload[d["offset"]:end], dtype=dt).reshape(d["shape"])
    blobs = {}
    for d in header.pop("blobs", []):
        end = d["offset"] + d["nbytes"]
        if end > plen:
            raise ProtocolError("blob outside payload")
        blobs[d["name"]] = payload[d["offset"]:end]
    return header, arrays, blobs


def tune_socket(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, 4 * 1024 * 1024)
        except OSError:
            pass
