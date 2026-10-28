"""Expert worker: a laptop/desktop on the LAN that holds some routed experts in
RAM and runs their FFNs for the coordinator.

A worker does not need the model. When the coordinator assigns experts the
worker does not have, the coordinator pushes just those expert tensors and the
worker caches them on disk (``~/.moelan/cache/<fingerprint>/``), so the next
start loads them locally with no transfer.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import socketserver
import threading
import time
from pathlib import Path

import numpy as np

from . import __version__
from .config import load_config, model_fingerprint
from .discovery import Beacon, local_ip
from .experts import CacheSource, CheckpointSource, ExpertStore
from .protocol import PROTOCOL_VERSION, recv_msg, send_msg, tune_socket

log = logging.getLogger("moelan.worker")


def total_ram_bytes() -> int:
    try:
        import psutil  # optional
        return int(psutil.virtual_memory().total)
    except ImportError:
        pass
    if hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    if os.name == "nt":
        import ctypes

        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.ullTotalPhys)
    return 8 * 2**30


def parse_size(s: str | int | None, default: int) -> int:
    if s is None:
        return default
    if isinstance(s, int):
        return s
    s = s.strip().upper()
    mult = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}
    if s.endswith("B"):
        s = s[:-1]
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(float(s))


class WorkerState:
    def __init__(self, node_id: str, budget: int, cache_root: Path,
                 model_dir: str | None = None, lru_size: int = 0, threads: int | None = None,
                 gpu: str | None = None, gpu_budget: int = 0, gpu_mode: str = "auto"):
        self.node_id = node_id
        self.threads = threads
        self.gpu, self.gpu_budget, self.gpu_mode = gpu, gpu_budget, gpu_mode
        self.heat: dict = {}
        self.budget = budget
        self.cache_root = Path(cache_root)
        self.lru_size = lru_size
        self.model_dir = model_dir
        self.local_fp = model_fingerprint(model_dir) if model_dir else None
        self.stores: dict[str, ExpertStore] = {}
        self.assigned: dict[str, list[tuple[int, int]]] = {}
        self.requests = 0
        self.busy_s = 0.0
        self.lock = threading.Lock()
        self.connections: set[socket.socket] = set()

    def cache_dir(self, fp: str) -> Path:
        return self.cache_root / fp

    def store_for(self, fp: str, cfg_raw: dict | None, meta: dict | None, fmt: str) -> ExpertStore:
        with self.lock:
            st = self.stores.get(fp)
            if st is not None and st.fmt == fmt:
                return st
            d = self.cache_dir(fp)
            d.mkdir(parents=True, exist_ok=True)
            if cfg_raw is not None:
                (d / "config.json").write_text(json.dumps(cfg_raw))
            if meta is not None:
                (d / "qwen36_meta.json").write_text(json.dumps(meta))
            cfg = load_config(d)
            sources = [CacheSource(d, cfg)]
            if self.local_fp == fp:
                sources.insert(0, CheckpointSource(self.model_dir, load_config(self.model_dir)))
            st = self.stores[fp] = ExpertStore(sources, fmt=fmt, lru_size=self.lru_size,
                                               threads=self.threads, gs=cfg.expert_gs or 64)
            return st

    def hello(self) -> dict:
        cached = {}
        if self.cache_root.exists():
            for d in self.cache_root.iterdir():
                if d.is_dir():
                    cached[d.name] = sum(1 for _ in d.glob("L*_E*.safetensors"))
        from . import kernels
        return {"node_id": self.node_id, "version": __version__, "protocol": PROTOCOL_VERSION,
                "kernels": "numba" if kernels.ENABLED else "numpy", "gpu": self.gpu,
                "budget_bytes": self.budget, "cpus": os.cpu_count(), "cached": cached,
                "local_model": self.local_fp}


class _Handler(socketserver.BaseRequestHandler):
    state: WorkerState

    def handle(self) -> None:
        sock: socket.socket = self.request
        tune_socket(sock)
        peer = "%s:%d" % self.client_address[:2]
        log.info("coordinator connected from %s", peer)
        self.state.connections.add(sock)
        try:
            while True:
                header, arrays, blobs = recv_msg(sock)
                try:
                    reply, out = self.dispatch(header, arrays, blobs)
                except Exception as exc:  # report, keep the connection
                    log.exception("request %s failed", header.get("op"))
                    reply, out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, None
                send_msg(sock, reply, out)
                if header.get("op") == "bye":
                    return
        except (ConnectionError, OSError):
            log.info("coordinator %s disconnected", peer)
        finally:
            self.state.connections.discard(sock)

    def dispatch(self, h: dict, arrays: dict, blobs: dict):
        st = self.state
        op = h.get("op")
        if op == "hello":
            return dict(st.hello(), ok=True), None
        if op == "ping":
            return {"ok": True, "t": time.time()}, None
        if op == "bye":
            return {"ok": True}, None
        if op == "assign":
            store = st.store_for(h["fingerprint"], h.get("config"), h.get("meta"), h.get("format", "int8"))
            keys = [tuple(k) for k in h["experts"]]
            st.assigned[h["fingerprint"]] = keys
            st.heat[h["fingerprint"]] = h.get("heat") or []
            missing = [list(k) for k in keys if not store.can_serve(k)]
            return {"ok": True, "missing": missing}, None
        if op == "put":
            store = st.stores[h["fingerprint"]]
            cache = next(s for s in store.sources if isinstance(s, CacheSource))
            raw = {p: (bytes(blobs[p]), h["tags"][p], h["shapes"][p]) for p in h["tags"]}
            cache.put((h["layer"], h["expert"]), raw)
            return {"ok": True}, None
        if op == "load":
            fp = h["fingerprint"]
            store = st.stores[fp]
            keys = st.assigned.get(fp, [])
            store.unpin_all_except(keys)
            t0 = time.perf_counter()
            store.pin(keys)
            gpu = {}
            if st.gpu and store.gpu is None:   # hybrid: hottest assigned experts to the GPU
                heat = dict(zip(keys, st.heat.get(fp, [])))
                gpu = store.setup_gpu(st.gpu, st.gpu_budget, keys, heat=heat, mode=st.gpu_mode,
                                      progress=log.info)
            elif store.gpu_info is not None:
                gpu = store.gpu.stats() if store.gpu else store.gpu_info
            return {"ok": True, "resident_bytes": store.resident_bytes(),
                    "load_s": time.perf_counter() - t0, "gpu": gpu}, None
        if op == "moe":
            t0 = time.perf_counter()
            store = st.stores[h["fingerprint"]]
            x = arrays["x"].astype(np.float32, copy=False)
            experts, offsets = arrays["experts"], arrays["offsets"]
            rows, weights = arrays["rows"], arrays["weights"].astype(np.float32, copy=False)
            groups = [(int(experts[i]), rows[offsets[i]:offsets[i + 1]],
                       weights[offsets[i]:offsets[i + 1]]) for i in range(len(experts))]
            y = store.compute(h["layer"], x, groups)
            touched = np.unique(rows)
            dt = time.perf_counter() - t0
            st.requests += 1
            st.busy_s += dt
            return ({"ok": True, "compute_s": dt},
                    {"rows": touched.astype(np.int32),
                     "y": y[touched].astype(np.float16 if h.get("wire") == "f16" else np.float32)})
        if op == "stats":
            return {"ok": True, "requests": st.requests, "busy_s": st.busy_s,
                    "stores": {fp: {"pinned": len(s.pinned), "hits": s.hits, "misses": s.misses,
                                    "resident_bytes": s.resident_bytes()}
                               for fp, s in st.stores.items()},
                    "gpu": {fp: s.gpu.stats() for fp, s in st.stores.items() if s.gpu}}, None
        raise ValueError(f"unknown op {op!r}")


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(host: str = "0.0.0.0", port: int = 9100, budget: int | None = None,
          cache_root: str | None = None, model_dir: str | None = None,
          node_id: str | None = None, advertise: bool = True, lru_size: int = 0,
          threads: int | None = None, gpu: str | None = None, gpu_budget: int = 0,
          gpu_mode: str = "auto",
          ready: threading.Event | None = None, stop: threading.Event | None = None) -> None:
    budget = budget if budget is not None else int(total_ram_bytes() * 0.6)
    cache_root = Path(cache_root or Path.home() / ".moelan" / "cache")
    node_id = node_id or f"{socket.gethostname()}:{port}"
    state = WorkerState(node_id, budget, cache_root, model_dir, lru_size, threads,
                        gpu=gpu, gpu_budget=gpu_budget, gpu_mode=gpu_mode)
    handler = type("Handler", (_Handler,), {"state": state})
    server = _Server((host, port), handler)
    port = server.server_address[1]
    beacon = None
    if advertise:
        beacon = Beacon({"node_id": node_id, "host": local_ip() if host == "0.0.0.0" else host,
                         "port": port, "budget_bytes": budget, "version": __version__})
        beacon.start()
    log.info("worker %s listening on %s:%d, RAM budget %.1f GiB, cache %s",
             node_id, host, port, budget / 2**30, cache_root)
    if ready is not None:
        ready.port = port  # type: ignore[attr-defined]
        ready.set()
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        while not (stop and stop.is_set()):
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if beacon:
            beacon.stop()
        server.shutdown()
        server.server_close()
        for sock in list(state.connections):  # drop live coordinators too
            try:
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
            except OSError:
                pass
