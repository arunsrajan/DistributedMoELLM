"""Expert worker: a laptop/desktop on the LAN that holds some routed experts in
RAM and runs their FFNs for the coordinator.

A worker does not need the model. When the coordinator assigns experts the
worker does not have, the coordinator pushes just those expert tensors and the
worker caches them on disk (``~/.moelan/cache/<fingerprint>/``), so the next
start loads them locally with no transfer.

Two tiers: the hottest assigned experts are kept in RAM (up to ``--budget``);
with the coordinator's ``--ssd`` modes a worker is assigned more than that, and
computes the rest straight from its SSD through memory maps, like Colibri
streams experts on one machine (``--disk`` caps the SSD space they take).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import socket
import socketserver
import threading
import time
from pathlib import Path

import numpy as np

from . import __version__
from .config import load_config, model_fingerprint
from .discovery import Beacon, local_ip
from .experts import CacheSource, ExpertStore, open_source, resident_bytes
from .protocol import PROTOCOL_VERSION, decode_rows, encode_rows, recv_msg, send_msg, tune_socket

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


def raise_fd_limit() -> int | None:
    """Let this process open many files: every expert computed from the SSD
    cache keeps its own memory map, which holds a file descriptor on Linux
    and macOS. Returns the soft limit (None where there is none, Windows)."""
    try:
        import resource
    except ImportError:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = 65536 if hard == resource.RLIM_INFINITY else min(hard, 65536)
        for target in (want, 10240):            # macOS refuses more than OPEN_MAX
            if soft >= target:
                break
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
                soft = target
                break
            except (ValueError, OSError):
                continue
        return soft
    except (ValueError, OSError):
        return None


def map_size_for(fd_limit: int | None) -> int:
    """How many memory-mapped experts the store keeps open."""
    return 4096 if fd_limit is None else max(64, min(4096, fd_limit - 512))


def default_disk_budget(cache_root: Path, cached: int) -> int:
    """Half the free space on the cache's drive, plus what the cache already holds."""
    try:
        return int(shutil.disk_usage(cache_root).free * 0.5) + cached
    except OSError:
        return cached


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
                 gpu: str | None = None, gpu_budget: int = 0, gpu_mode: str = "auto",
                 disk_budget: int | None = None, map_size: int = 4096):
        self.node_id = node_id
        self.threads = threads
        self.gpu, self.gpu_budget, self.gpu_mode = gpu, gpu_budget, gpu_mode
        self.heat: dict = {}
        self.extra: dict[str, list[tuple[int, int]]] = {}   # held on the SSD for other nodes' experts
        self.extra_heat: dict[str, list[int]] = {}
        self.budget = budget
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.disk_budget = disk_budget if disk_budget is not None else \
            default_disk_budget(self.cache_root, sum(self.cached_bytes().values()))
        self.map_size = map_size
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
                sources.insert(0, open_source(self.model_dir, load_config(self.model_dir)))
            st = self.stores[fp] = ExpertStore(sources, fmt=fmt, lru_size=self.lru_size,
                                               threads=self.threads, gs=cfg.expert_gs or 64,
                                               map_size=self.map_size)
            st.cfg = cfg
            return st

    def _cache_dirs(self):
        if self.cache_root.exists():
            for d in self.cache_root.iterdir():
                if d.is_dir():
                    yield d

    def cached_bytes(self) -> dict[str, int]:
        """SSD space each model's pushed experts take, by fingerprint."""
        out = {}
        for d in self._cache_dirs():
            n = 0
            for f in d.glob("L*_E*.safetensors"):
                try:
                    n += f.stat().st_size
                except OSError:
                    pass
            out[d.name] = n
        return out

    def hello(self) -> dict:
        cached = {d.name: sum(1 for _ in d.glob("L*_E*.safetensors")) for d in self._cache_dirs()}
        from . import kernels
        return {"node_id": self.node_id, "version": __version__, "protocol": PROTOCOL_VERSION,
                "kernels": "numba" if kernels.ENABLED else "numpy", "gpu": self.gpu,
                "host": socket.gethostname(),
                "budget_bytes": self.budget, "cpus": os.cpu_count(), "cached": cached,
                "disk_budget_bytes": self.disk_budget, "cached_bytes": self.cached_bytes(),
                "local_model": self.local_fp}

    def make_room(self, store: ExpertStore, keep: set, need: int) -> int:
        """Delete this model's cached experts that are no longer assigned here
        until ``need`` more bytes fit in the disk budget. Returns bytes freed."""
        cache = next((s for s in store.sources if isinstance(s, CacheSource)), None)
        over = sum(self.cached_bytes().values()) + need - self.disk_budget
        if cache is None or over <= 0:
            return 0
        stale = [k for k in cache.keys() if k not in keep]
        store.forget(stale)
        gc.collect()          # close their memory maps, or Windows will not delete the files
        freed = 0
        for k in stale:
            if freed >= over:
                break
            n = cache.nbytes(k)
            if cache.remove(k):
                freed += n
        if freed < over:
            log.warning("disk budget: %.1f GiB more than --disk allows after removing unassigned experts",
                        (over - freed) / 2**30)
        return freed

    def ram_slots(self, store: ExpertStore) -> int:
        """How many experts the RAM budget holds in the store's resident format."""
        try:
            per = resident_bytes(store.cfg, store.fmt)
        except (KeyError, AttributeError, TypeError):
            return 1 << 30
        return self.budget // per if per > 0 else 1 << 30


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
        if op == "echo":     # round trip with a realistic payload, for speed-aware placement
            return {"ok": True}, dict(arrays)
        if op == "bench":    # ms per expert on this machine, before it holds any experts
            from .experts import bench_expert
            out = {"ok": True, "ms": bench_expert(h["hidden"], h["inter"], h["format"], h.get("gs", 64))}
            if st.gpu:       # and on its GPU, so placement counts the GPU too
                from .accel import bench_gpu
                try:
                    out.update(bench_gpu(st.gpu, st.gpu_budget, h["hidden"], h["inter"], h["format"],
                                         h.get("gs", 64)))
                except RuntimeError as exc:
                    out["gpu_error"] = str(exc)
            return out, None
        if op == "bye":
            return {"ok": True}, None
        if op == "assign":
            fp = h["fingerprint"]
            store = st.store_for(fp, h.get("config"), h.get("meta"), h.get("format", "int8"))
            keys = [tuple(k) for k in h["experts"]]
            heat = h.get("heat") or []
            extra = [tuple(k) for k in h.get("extra") or []]
            if (st.assigned.get(fp), st.heat.get(fp)) != (keys, heat):
                store.gpu = None     # a new plan: "load" chooses the GPU's experts again
            st.assigned[fp] = keys
            st.heat[fp] = heat
            st.extra[fp] = extra
            st.extra_heat[fp] = h.get("extra_heat") or []
            missing = [list(k) for k in keys + extra if not store.can_serve(k)]
            freed = 0
            if missing and h.get("disk_bytes"):    # SSD modes: keep the cache within --disk
                freed = st.make_room(store, set(keys) | set(extra), len(missing) * int(h["disk_bytes"]))
            return {"ok": True, "missing": missing, "freed_bytes": freed}, None
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
            extra = st.extra.get(fp, [])
            # RAM takes this node's own experts hottest first, then (with room
            # left) copies it holds for others; the rest it computes from the SSD
            def hottest(ks, hs):
                hs = list(hs) + [0] * (len(ks) - len(hs))
                return [k for _, _, k in sorted(zip((-v for v in hs), range(len(ks)), ks))]
            order = hottest(keys, st.heat.get(fp, [])) + hottest(extra, st.extra_heat.get(fp, []))
            ram = order[:st.ram_slots(store)]
            store.unpin_all_except(ram)
            t0 = time.perf_counter()
            store.pin(ram)
            gpu = {}
            if st.gpu and store.gpu is None:   # hybrid: hottest assigned experts to the GPU
                heat = dict(zip(keys, st.heat.get(fp, [])))
                gpu = store.setup_gpu(st.gpu, st.gpu_budget, keys, heat=heat, mode=st.gpu_mode,
                                      progress=log.info)
            elif store.gpu_info is not None:
                gpu = store.gpu.stats() if store.gpu else store.gpu_info
            held = len(set(keys) | set(extra))
            return {"ok": True, "resident_bytes": store.resident_bytes(),
                    "load_s": time.perf_counter() - t0, "gpu": gpu,
                    "pinned": [list(k) for k in ram], "ram_experts": len(ram),
                    "ssd_experts": held - len(ram)}, None
        if op == "moe":
            t0 = time.perf_counter()
            store = st.stores[h["fingerprint"]]
            x = decode_rows(arrays, "x")
            experts, offsets = arrays["experts"], arrays["offsets"]
            rows, weights = arrays["rows"], arrays["weights"].astype(np.float32, copy=False)
            groups = [(int(experts[i]), rows[offsets[i]:offsets[i + 1]],
                       weights[offsets[i]:offsets[i + 1]]) for i in range(len(experts))]
            y = store.compute(h["layer"], x, groups)
            touched = np.unique(rows)
            dt = time.perf_counter() - t0
            st.requests += 1
            st.busy_s += dt
            return ({"ok": True, "compute_s": dt, "ssd_reads": store.misses},
                    dict(encode_rows(y[touched], h.get("wire", "f32"), "y"), rows=touched.astype(np.int32)))
        if op == "stats":
            return {"ok": True, "requests": st.requests, "busy_s": st.busy_s,
                    "stores": {fp: {"pinned": len(s.pinned), "hits": s.hits, "misses": s.misses,
                                    "resident_bytes": s.resident_bytes(),
                                    "ssd_experts": len(set(st.assigned.get(fp, [])) | set(st.extra.get(fp, [])))
                                    - len(s.pinned)}
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
          gpu_mode: str = "auto", disk_budget: int | None = None,
          ready: threading.Event | None = None, stop: threading.Event | None = None) -> None:
    budget = budget if budget is not None else int(total_ram_bytes() * 0.6)
    cache_root = Path(cache_root or Path.home() / ".moelan" / "cache")
    node_id = node_id or f"{socket.gethostname()}:{port}"
    fd_limit = raise_fd_limit()
    if fd_limit is not None and fd_limit < 2048:
        log.warning("only %d open files allowed: experts computed from the SSD cache are kept "
                    "open %d at a time (raise it with ulimit -n)", fd_limit, map_size_for(fd_limit))
    state = WorkerState(node_id, budget, cache_root, model_dir, lru_size, threads,
                        gpu=gpu, gpu_budget=gpu_budget, gpu_mode=gpu_mode,
                        disk_budget=disk_budget, map_size=map_size_for(fd_limit))
    handler = type("Handler", (_Handler,), {"state": state})
    server = _Server((host, port), handler)
    port = server.server_address[1]
    beacon = None
    if advertise:
        beacon = Beacon({"node_id": node_id, "host": local_ip() if host == "0.0.0.0" else host,
                         "port": port, "budget_bytes": budget, "version": __version__})
        beacon.start()
    log.info("worker %s listening on %s:%d, RAM budget %.1f GiB, SSD budget %.1f GiB, cache %s",
             node_id, host, port, budget / 2**30, state.disk_budget / 2**30, cache_root)
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
