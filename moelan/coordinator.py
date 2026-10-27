"""Coordinator side: plan expert placement, ship experts to workers, and run the
routed FFNs of every MoE layer across the LAN.

Per layer and per forward pass, each worker gets exactly one request holding
the rows routed to its experts (the batch union, as in Colibri's cluster mode),
so a token costs one round trip per worker per layer, all workers in flight at
once. The coordinator computes its own experts while the workers compute
theirs. If a worker fails, its share is recomputed locally from disk and the
worker is retried later.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ModelConfig, model_fingerprint
from .experts import CheckpointSource, ExpertStore, default_format, resident_bytes
from .model import ExpertBackend, group_by_expert
from .planner import LOCAL, Plan, make_plan
from .protocol import PROTOCOL_VERSION, recv_msg, send_msg, tune_socket

log = logging.getLogger("moelan.coordinator")
RETRY_AFTER_S = 30.0


@dataclass
class WorkerConn:
    address: str
    timeout: float = 120.0
    sock: socket.socket | None = None
    info: dict = field(default_factory=dict)
    alive: bool = False
    failed_at: float = 0.0
    requests: int = 0
    bytes_out: int = 0
    wait_s: float = 0.0       # coordinator blocked on this worker's reply
    remote_s: float = 0.0     # of which the worker was computing
    resident_bytes: int = 0

    @property
    def node_id(self) -> str:
        return self.info.get("node_id", self.address)

    def connect(self, connect_timeout: float = 5.0) -> dict:
        host, port = self.address.rsplit(":", 1)
        s = socket.create_connection((host, int(port)), timeout=connect_timeout)
        tune_socket(s)
        s.settimeout(self.timeout)
        self.sock = s
        self.info = self.call({"op": "hello"})[0]
        if self.info.get("protocol") != PROTOCOL_VERSION:
            raise RuntimeError(f"{self.address}: protocol {self.info.get('protocol')} != {PROTOCOL_VERSION}")
        self.alive = True
        return self.info

    def send(self, header, arrays=None, blobs=None) -> None:
        send_msg(self.sock, header, arrays, blobs)

    def recv(self):
        h, arrays, _ = recv_msg(self.sock)
        if not h.get("ok", False):
            raise RuntimeError(f"{self.address}: {h.get('error', 'request failed')}")
        return h, arrays

    def call(self, header, arrays=None, blobs=None):
        self.send(header, arrays, blobs)
        return self.recv()

    def mark_failed(self, exc: Exception) -> None:
        log.warning("worker %s failed: %s; its experts fall back to the coordinator", self.address, exc)
        self.alive = False
        self.failed_at = time.monotonic()
        try:
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        self.sock = None

    def close(self) -> None:
        if self.sock:
            try:
                self.call({"op": "bye"})
            except (OSError, RuntimeError, ConnectionError):
                pass
            self.sock.close()
            self.sock = None


class DistributedBackend(ExpertBackend):
    def __init__(self, model_dir: str, cfg: ModelConfig, workers: list[str],
                 local_budget: int = 0, heat=None, fmt: str | None = None,
                 wire: str = "f32", replicate: bool = True, lru_size: int = 0,
                 progress=None):
        super().__init__()
        if wire not in ("f32", "f16"):
            raise ValueError("wire must be f32 or f16")
        self.cfg = cfg
        self.wire = wire
        self.fmt = fmt = fmt or default_format(cfg)
        self.fingerprint = model_fingerprint(model_dir)
        self.source = CheckpointSource(model_dir, cfg)
        self.local = ExpertStore([self.source], fmt=fmt, lru_size=lru_size, gs=cfg.expert_gs or 64)
        meta = Path(model_dir) / "qwen36_meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else None
        self.progress = progress or (lambda msg: log.info(msg))
        self._lock = threading.Lock()

        self.conns: dict[str, WorkerConn] = {}
        for addr in workers:
            c = WorkerConn(addr)
            try:
                c.connect()
                self.conns[c.node_id] = c
                self.progress(f"connected {addr} ({c.node_id}, budget "
                              f"{c.info['budget_bytes'] / 2**30:.1f} GiB, {c.info.get('cpus')} cpus)")
            except (OSError, RuntimeError) as exc:
                self.progress(f"skipping {addr}: {exc}")

        self.expert_ram = resident_bytes(cfg, fmt)
        budgets = {nid: c.info["budget_bytes"] for nid, c in self.conns.items()}
        if local_budget > 0:
            budgets[LOCAL] = local_budget
        self.plan: Plan = make_plan(cfg.moe_layers(), cfg.num_experts, self.expert_ram,
                                    budgets, heat=heat, replicate=replicate)
        self.progress("placement plan:\n" + self.plan.summary(self.expert_ram))
        self._setup_all()

    # ---- setup -----------------------------------------------------------

    def _setup_worker(self, c: WorkerConn) -> None:
        keys = self.plan.per_node.get(c.node_id, [])
        h, _ = c.call({"op": "assign", "fingerprint": self.fingerprint, "config": self.cfg.raw,
                       "meta": self.meta, "experts": [list(k) for k in keys], "format": self.fmt})
        missing = [tuple(k) for k in h["missing"]]
        if missing:
            mb = len(missing) * self.source.nbytes(missing[0]) / 2**20
            self.progress(f"{c.node_id}: sending {len(missing)} experts ({mb:.0f} MiB)")
        t0 = time.perf_counter()
        window = 16  # puts in flight; the worker answers in order
        sent = acked = nbytes = 0
        while acked < len(missing):
            while sent < len(missing) and sent - acked < window:
                key = missing[sent]
                raw = self.source.raw(key)
                c.send({"op": "put", "fingerprint": self.fingerprint, "layer": key[0], "expert": key[1],
                        "tags": {p: r[1] for p, r in raw.items()},
                        "shapes": {p: r[2] for p, r in raw.items()}},
                       blobs={p: r[0] for p, r in raw.items()})
                nbytes += sum(len(r[0]) for r in raw.values())
                sent += 1
            c.recv()
            acked += 1
            if acked % 256 == 0:
                rate = nbytes / (time.perf_counter() - t0) / 2**20
                self.progress(f"{c.node_id}: {acked}/{len(missing)} experts sent, {rate:.0f} MiB/s")
        h, _ = c.call({"op": "load", "fingerprint": self.fingerprint})
        c.resident_bytes = h["resident_bytes"]
        self.progress(f"{c.node_id}: {len(keys)} experts resident "
                      f"({h['resident_bytes'] / 2**30:.2f} GiB, loaded in {h['load_s']:.1f}s)")

    def _setup_all(self) -> None:
        conns = list(self.conns.values())
        if conns:
            with ThreadPoolExecutor(len(conns)) as ex:
                for c, fut in [(c, ex.submit(self._setup_worker, c)) for c in conns]:
                    try:
                        fut.result()
                    except (OSError, RuntimeError, ConnectionError) as exc:
                        c.mark_failed(exc)
        self.local.pin(self.plan.per_node.get(LOCAL, []))

    def _maybe_reconnect(self) -> None:
        now = time.monotonic()
        for c in self.conns.values():
            if not c.alive and now - c.failed_at > RETRY_AFTER_S:
                try:
                    c.connect(connect_timeout=1.0)
                    self._setup_worker(c)
                    log.info("worker %s is back", c.address)
                except (OSError, RuntimeError, ConnectionError) as exc:
                    c.mark_failed(exc)

    # ---- the hot path ------------------------------------------------------
    def _owner(self, layer: int, expert: int, load: dict[str, int]) -> str:
        owners = self.plan.owners.get((layer, expert))
        if not owners:
            return LOCAL
        best = None
        for nid in owners:
            if nid == LOCAL or (nid in self.conns and self.conns[nid].alive):
                if best is None or load.get(nid, 0) < load.get(best, 0):
                    best = nid
        return best or LOCAL

    def _moe(self, layer, x, idx, w):
        self._maybe_reconnect()
        experts, offsets, rows, weights = group_by_expert(idx, w)
        out = np.zeros_like(x, dtype=np.float32)

        def groups_of(sel):
            return [(int(experts[i]), rows[offsets[i]:offsets[i + 1]], weights[offsets[i]:offsets[i + 1]])
                    for i in sel]

        pending = list(range(len(experts)))
        while pending:  # a failed worker's share is re-routed (replica, else local)
            per_node: dict[str, list[int]] = {}
            load: dict[str, int] = {}
            for i in pending:
                nid = self._owner(layer, int(experts[i]), load)
                per_node.setdefault(nid, []).append(i)
                load[nid] = load.get(nid, 0) + int(offsets[i + 1] - offsets[i])
            pending = []

            # 1. fire one request per worker (only the rows that worker needs)
            inflight = []
            for nid, sel in per_node.items():
                if nid == LOCAL:
                    continue
                c = self.conns[nid]
                sub_rows = np.concatenate([rows[offsets[i]:offsets[i + 1]] for i in sel])
                uniq, remap = np.unique(sub_rows, return_inverse=True)
                sub_w = np.concatenate([weights[offsets[i]:offsets[i + 1]] for i in sel])
                sub_off = np.concatenate([[0], np.cumsum([offsets[i + 1] - offsets[i] for i in sel])])
                xs = x[uniq].astype(np.float16 if self.wire == "f16" else np.float32)
                try:
                    c.send({"op": "moe", "fingerprint": self.fingerprint, "layer": layer, "wire": self.wire},
                           {"x": xs, "experts": experts[sel], "offsets": sub_off.astype(np.int32),
                            "rows": remap.astype(np.int32), "weights": sub_w})
                    c.requests += 1
                    c.bytes_out += xs.nbytes
                    inflight.append((c, sel, uniq))
                except (OSError, ConnectionError) as exc:
                    c.mark_failed(exc)
                    pending.extend(sel)

            # 2. compute the coordinator's own experts meanwhile
            if LOCAL in per_node:
                out += self.local.compute(layer, x, groups_of(per_node[LOCAL]))

            # 3. collect the workers' partial sums
            for c, sel, uniq in inflight:
                t0 = time.perf_counter()
                try:
                    h, arrays = c.recv()
                    c.remote_s += h.get("compute_s", 0.0)
                    out[uniq[arrays["rows"]]] += arrays["y"].astype(np.float32)
                except (OSError, RuntimeError, ConnectionError) as exc:
                    c.mark_failed(exc)
                    pending.extend(sel)
                c.wait_s += time.perf_counter() - t0
        return out

    def stats(self) -> list[dict]:
        out = []
        for c in self.conns.values():
            row = {"node": c.node_id, "address": c.address, "alive": c.alive,
                   "experts": len(self.plan.per_node.get(c.node_id, [])),
                   "requests": c.requests, "mib_out": c.bytes_out / 2**20, "wait_s": c.wait_s,
                   "compute_s": c.remote_s, "budget_bytes": c.info.get("budget_bytes"),
                   "cpus": c.info.get("cpus"), "resident_bytes": c.resident_bytes}
            out.append(row)
        out.append({"node": LOCAL, "address": "-", "alive": True,
                    "experts": len(self.plan.per_node.get(LOCAL, [])),
                    "disk_reads": self.local.misses, "hits": self.local.hits,
                    "resident_bytes": self.local.resident_bytes()})
        return out

    def close(self) -> None:
        for c in self.conns.values():
            c.close()
