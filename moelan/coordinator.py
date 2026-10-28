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
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ModelConfig, model_fingerprint
from .experts import CheckpointSource, ExpertStore, bench_expert, default_format, resident_bytes
from .model import ExpertBackend, group_by_expert
from .planner import LOCAL, Plan, make_plan, node_ms, speed_shares
from .protocol import (PROTOCOL_VERSION, WIRE_FORMATS, decode_rows, encode_rows, recv_msg, send_msg,
                       tune_socket)

log = logging.getLogger("moelan.coordinator")
RETRY_AFTER_S = 30.0

# MOELAN_SIM_NET="rtt_ms,mbit_per_s" adds half a round trip plus the transfer
# time to every message to and from workers, to measure wire formats and LAN
# speeds on one machine. Off unless set.
_SIM_NET = None
if os.environ.get("MOELAN_SIM_NET"):
    _rtt, _mbit = (float(v) for v in os.environ["MOELAN_SIM_NET"].split(","))
    _SIM_NET = (_rtt / 2000.0, _mbit * 1e6 / 8)


def _this_host() -> str:
    return socket.gethostname()


def _sim_delay(nbytes: int) -> None:
    half_rtt, bytes_per_s = _SIM_NET
    time.sleep(half_rtt + nbytes / bytes_per_s)


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
    bytes_in: int = 0
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
        if _SIM_NET:   # benchmarking a LAN on one machine: latency + bandwidth per message
            _sim_delay(sum(a.nbytes for a in (arrays or {}).values()) + sum(len(b) for b in (blobs or {}).values()))
        send_msg(self.sock, header, arrays, blobs)

    def recv(self):
        h, arrays, _ = recv_msg(self.sock)
        if _SIM_NET:
            _sim_delay(sum(a.nbytes for a in arrays.values()))
        if not h.get("ok", False):
            raise RuntimeError(f"{self.address}: {h.get('error', 'request failed')}")
        return h, arrays

    def call(self, header, arrays=None, blobs=None, timeout: float | None = None):
        if timeout is None:
            self.send(header, arrays, blobs)
            return self.recv()
        self.sock.settimeout(timeout)   # e.g. a worker filling a large GPU budget
        try:
            self.send(header, arrays, blobs)
            return self.recv()
        finally:
            self.sock.settimeout(self.timeout)

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
                 progress=None, gpu: str | None = None, gpu_budget: int = 0, gpu_mode: str = "auto",
                 placement: str = "speed"):
        super().__init__()
        if placement not in ("speed", "ram"):
            raise ValueError("placement must be speed or ram")
        self.heat_in = dict(heat or {})
        self.gpu_spec, self.gpu_budget, self.gpu_mode = gpu, gpu_budget, gpu_mode
        if wire not in WIRE_FORMATS:
            raise ValueError(f"wire must be one of {WIRE_FORMATS}")
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
        self.local_budget = local_budget
        self.speed: dict = {}
        budgets = {nid: c.info["budget_bytes"] for nid, c in self.conns.items()}
        shares = None
        if placement == "speed" and self.conns:
            shares = self._measure_speeds(budgets)
            # the coordinator computes its share from its own checkpoint: no RAM cap
            budgets[LOCAL] = 2 * self.expert_ram * len(cfg.moe_layers()) * cfg.num_experts
            replicate = False   # a replica on a slower node would only attract work it is bad at
        elif local_budget > 0:
            budgets[LOCAL] = local_budget
        self.plan: Plan = make_plan(cfg.moe_layers(), cfg.num_experts, self.expert_ram,
                                    budgets, heat=heat, replicate=replicate, shares=shares)
        self.progress("placement plan:\n" + self.plan.summary(self.expert_ram))
        self._setup_all()

    # ---- speed-aware placement ------------------------------------------------
    def _measure_speeds(self, budgets: dict[str, int]) -> dict[str, float]:
        """Benchmark this machine and every worker, measure round trips, and
        return the shares of the routed work that balance a layer's time."""
        cfg = self.cfg
        dims = {"hidden": cfg.hidden_size, "inter": cfg.moe_intermediate_size, "format": self.fmt,
                "gs": cfg.expert_gs or 64}
        local_ms = bench_expert(dims["hidden"], dims["inter"], self.fmt, dims["gs"])
        local_gpu = {}
        if self.gpu_spec:   # this machine's GPU counts too
            from .accel import bench_gpu
            try:
                local_gpu = bench_gpu(self.gpu_spec, self.gpu_budget, dims["hidden"], dims["inter"],
                                      self.fmt, dims["gs"])
            except RuntimeError as exc:
                self.progress(f"this machine's GPU: {exc}")
        total = self.expert_ram * len(cfg.moe_layers()) * cfg.num_experts
        n_experts = len(cfg.moe_layers()) * cfg.num_experts
        workers = {}
        gpus = {}
        row = encode_rows(np.zeros((1, cfg.hidden_size), np.float32), self.wire, "x")
        for nid, c in self.conns.items():
            if c.info.get("host") == _this_host():
                # same CPU and memory as this process: its compute competes with
                # ours instead of adding to it, so it can never speed a token up
                self.progress(f"{nid}: on this machine, shares its CPU -> not used for speed")
                self.speed.setdefault("same_host", []).append(nid)
                continue
            try:
                bench = c.call({"op": "bench", **dims}, timeout=600)[0]
                ms = bench["ms"]
                if bench.get("gpu_ms"):
                    gpus[nid] = bench
                elif bench.get("gpu_error"):
                    self.progress(f"{nid}: GPU not usable ({bench['gpu_error']})")
                for _ in range(3):
                    c.call({"op": "echo"}, row)
                rtts = []
                for _ in range(15):
                    t0 = time.perf_counter()
                    c.call({"op": "echo"}, row)
                    rtts.append((time.perf_counter() - t0) * 1000)
                rtt = float(np.median(rtts))
            except (OSError, RuntimeError, ConnectionError, KeyError) as exc:
                self.progress(f"{nid}: could not measure ({exc}); not used")
                continue
            workers[nid] = (ms, rtt, min(1.0, budgets.get(nid, 0) / total))
        # Solve with each GPU at full speed, then again with its share checked
        # against how many experts its memory holds.
        cpu_only = dict(workers)

        def effective(shares):
            eff = {}
            for nid, (ms, rtt, cap) in cpu_only.items():
                g = gpus.get(nid, {})
                held = shares.get(nid, 0.0) * n_experts if shares else 0.0
                eff[nid] = (node_ms(ms, g.get("gpu_ms"), g.get("gpu_experts", 0), held, cfg.top_k), rtt, cap)
            held = shares.get(LOCAL, 1.0) * n_experts if shares else 0.0
            local = node_ms(local_ms, local_gpu.get("gpu_ms"), local_gpu.get("gpu_experts", 0), held, cfg.top_k)
            return local, eff
        local_eff, workers = effective(None)
        shares, _, _ = speed_shares(cfg.top_k, local_eff, workers)
        local_eff, workers = effective(shares)
        shares, layer_ms, _ = speed_shares(cfg.top_k, local_eff, workers)
        alone_ms = cfg.top_k * local_eff       # this machine on its own, with its GPU if any
        same_host = self.speed.get("same_host", [])
        self.speed = {"local_ms": local_eff, "local_cpu_ms": local_ms, "local_gpu": local_gpu or None,
                      "layer_ms": layer_ms, "layer_ms_alone": alone_ms,
                      "same_host": same_host,
                      "nodes": {nid: {"expert_ms": w[0], "cpu_ms": cpu_only[nid][0], "rtt_ms": w[1],
                                      "share": shares.get(nid, 0.0),
                                      "gpu": ({k: gpus[nid][k] for k in ("gpu_ms", "gpu_experts", "gpu_name")}
                                              if nid in gpus else None)}
                                for nid, w in workers.items()}}

        def describe(cpu, eff, g):
            if not g:
                return f"{cpu:.2f} ms per expert"
            used = "" if eff < cpu - 1e-9 else " (GPU too slow to help)"
            return (f"CPU {cpu:.2f} + {g['gpu_name']} {g['gpu_ms']:.2f} ms per expert "
                    f"({g['gpu_experts']} fit) -> {eff:.2f} ms{used}")
        self.progress(f"this machine: {describe(local_ms, local_eff, local_gpu)} -> "
                      f"{alone_ms:.1f} ms per layer alone")
        for nid, (ms, rtt, cap) in workers.items():
            sh = shares.get(nid, 0.0)
            what = describe(cpu_only[nid][0], ms, gpus.get(nid))
            if sh > 1e-9:
                self.progress(f"{nid}: {what}, round trip {rtt:.2f} ms -> {sh:.0%} of the routed work")
            else:
                self.progress(f"{nid}: {what}, round trip {rtt:.2f} ms -> not used: "
                              f"waiting for it costs more than computing its share here")
        gain = alone_ms / layer_ms if layer_ms > 0 else 1.0
        self.progress(f"expected {layer_ms:.1f} ms per layer for the experts "
                      f"({gain:.2f}x vs this machine alone); this machine takes {shares[LOCAL]:.0%}")
        for nid in self.conns:
            if nid not in workers:
                shares[nid] = 0.0
        return shares

    # ---- setup -----------------------------------------------------------

    def _setup_worker(self, c: WorkerConn) -> None:
        keys = self.plan.per_node.get(c.node_id, [])
        h, _ = c.call({"op": "assign", "fingerprint": self.fingerprint, "config": self.cfg.raw,
                       "meta": self.meta, "experts": [list(k) for k in keys], "format": self.fmt,
                       "heat": [int(self.heat_in.get(k, 0)) for k in keys]})
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
        h, _ = c.call({"op": "load", "fingerprint": self.fingerprint}, timeout=900)
        c.resident_bytes = h["resident_bytes"]
        c.gpu = h.get("gpu") or {}
        if c.gpu.get("experts"):
            self.progress(f"{c.node_id}: {c.gpu['experts']} experts on {c.gpu['device']} "
                          f"(GPU share {c.gpu['share']:.0%})")
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
        mine = self.plan.per_node.get(LOCAL, [])
        if self.speed:   # speed placement: computed from the checkpoint, pinned only within --local-budget
            mine = mine[:self.local_budget // self.expert_ram] if self.local_budget else []
        self.local.pin(mine)
        if self.gpu_spec:   # the coordinator computes its own experts and the unplaced ones
            mine = list(self.plan.per_node.get(LOCAL, [])) + list(self.plan.unplaced)
            self.local.setup_gpu(self.gpu_spec, self.gpu_budget, mine, heat=self.heat_in,
                                 mode=self.gpu_mode, progress=self.progress)

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
                xs = encode_rows(x[uniq], self.wire, "x")
                try:
                    c.send({"op": "moe", "fingerprint": self.fingerprint, "layer": layer, "wire": self.wire},
                           dict(xs, experts=experts[sel], offsets=sub_off.astype(np.int32),
                                rows=remap.astype(np.int32), weights=sub_w))
                    c.requests += 1
                    c.bytes_out += sum(a.nbytes for a in xs.values())
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
                    c.bytes_in += sum(a.nbytes for a in arrays.values())
                    out[uniq[arrays["rows"]]] += decode_rows(arrays, "y")
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
                   "requests": c.requests, "mib_out": c.bytes_out / 2**20, "mib_in": c.bytes_in / 2**20,
                   "wait_s": c.wait_s,
                   "compute_s": c.remote_s, "budget_bytes": c.info.get("budget_bytes"),
                   "cpus": c.info.get("cpus"), "resident_bytes": c.resident_bytes,
                   "kernels": c.info.get("kernels"), "gpu": getattr(c, "gpu", None) or None,
                   "speed": ({"share": 0.0, "same_host": True} if c.node_id in self.speed.get("same_host", [])
                             else self.speed.get("nodes", {}).get(c.node_id))}
            out.append(row)
        out.append({"node": LOCAL, "address": "-", "alive": True,
                    "experts": len(self.plan.per_node.get(LOCAL, [])),
                    "disk_reads": self.local.misses, "hits": self.local.hits,
                    "resident_bytes": self.local.resident_bytes(),
                    "gpu": self.local.gpu.stats() if self.local.gpu else self.local.gpu_info,
                    "speed": ({"expert_ms": self.speed["local_ms"], "cpu_ms": self.speed.get("local_cpu_ms"),
                               "gpu": self.speed.get("local_gpu"), "share": 1.0 - sum(
                        n["share"] for n in self.speed["nodes"].values()),
                        "layer_ms": self.speed["layer_ms"], "layer_ms_alone": self.speed["layer_ms_alone"]}
                        if self.speed else None)})
        return out

    def close(self) -> None:
        for c in self.conns.values():
            c.close()
