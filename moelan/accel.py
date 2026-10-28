"""GPU expert tier: hybrid CPU + GPU expert compute inside one node.

Each node (the coordinator and every worker) can keep some of its experts on
a GPU, dequantized to fp16, through PyTorch: CUDA (NVIDIA), MPS (Apple),
DirectML (AMD / Intel / any DirectX 12 GPU on Windows, ``torch-directml``)
or ``torch-cpu`` (for testing the path without a GPU).

For every MoE layer the GPU's experts are launched first, the CPU computes
its own experts meanwhile, and the results are added. How many experts go to
the GPU is decided by measurement: at start-up the node times one expert on
each device, and the GPU gets the hottest experts (by routing statistics)
until it is expected to carry its fair share of the routed work, so that
both devices finish a layer at about the same time. A fast discrete GPU ends
up taking as much as its memory budget allows; a slow integrated one only a
small hot set, or nothing.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger("moelan.accel")

DEVICES = ("auto", "cuda", "mps", "dml", "torch-cpu")


def open_device(spec: str):
    """(torch, device, human name) for ``spec``, or raise RuntimeError."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the GPU tier needs PyTorch (pip install torch, or torch-directml "
                           "for AMD/Intel GPUs on Windows)") from exc
    if spec == "auto":
        for cand in ("cuda", "mps", "dml"):
            try:
                return open_device(cand)
            except RuntimeError:
                continue
        raise RuntimeError("no GPU found (tried CUDA, MPS, DirectML)")
    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to this PyTorch")
        dev = torch.device(spec)
        return torch, dev, f"cuda: {torch.cuda.get_device_name(dev)}"
    if spec == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("Apple MPS is not available")
        return torch, torch.device("mps"), "mps: Apple GPU"
    if spec == "dml":
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("DirectML needs torch-directml (pip install torch-directml)") from exc
        if torch_directml.device_count() < 1:
            raise RuntimeError("no DirectML device")
        return torch, torch_directml.device(), f"dml: {torch_directml.device_name(0).strip()}"
    if spec == "torch-cpu":
        return torch, torch.device("cpu"), "torch-cpu (test device)"
    raise RuntimeError(f"unknown device {spec!r}; choose from {DEVICES}")


class GpuTier:
    """Experts resident on one GPU as dense fp16 matrices (fp32 on torch-cpu)."""

    def __init__(self, spec: str, budget_bytes: int):
        self.torch, self.device, self.name = open_device(spec)
        self.spec = spec
        self.dtype = self.torch.float32 if spec == "torch-cpu" else self.torch.float16
        self.budget = int(budget_bytes)
        self.experts: dict = {}            # (layer, expert) -> (gate_up, down) on the device
        self.used = 0
        self.lock = threading.Lock()
        self.calls = 0
        self.busy_s = 0.0
        self.share = 0.0                   # target share of routed work, from calibration
        self.gpu_ms = self.cpu_ms = None

    def expert_bytes(self, inter: int, hidden: int) -> int:
        return 3 * inter * hidden * (2 if self.dtype == self.torch.float16 else 4)

    def __contains__(self, key) -> bool:
        return key in self.experts

    def add(self, key, expert) -> bool:
        """Upload one :class:`~moelan.experts.Expert` (any format). False if full."""
        gu, dn = expert.gu.dequant(), expert.down.dequant()
        size = (gu.size + dn.size) * (2 if self.dtype == self.torch.float16 else 4)
        if self.used + size > self.budget:
            return False
        t = self.torch
        pair = (t.as_tensor(np.ascontiguousarray(gu, np.float32)).to(self.device, self.dtype),
                t.as_tensor(np.ascontiguousarray(dn, np.float32)).to(self.device, self.dtype))
        with self.lock:
            self.experts[key] = pair
            self.used += size
        return True

    def launch(self, layer: int, x: np.ndarray, groups):
        """Queue the given (expert, rows, weights) on the GPU; returns a handle.
        CUDA / DirectML run it asynchronously while the CPU does its share."""
        t = self.torch
        t0 = time.perf_counter()
        with self.lock, t.no_grad():
            xd = t.as_tensor(np.ascontiguousarray(x, np.float32)).to(self.device, self.dtype)
            out = []
            for expert, rows, weights in groups:
                gu, dn = self.experts[(layer, int(expert))]
                inter = dn.shape[1]
                xr = xd[t.as_tensor(np.asarray(rows, np.int64)).to(self.device)] if len(rows) != x.shape[0] else xd
                h = xr @ gu.T
                y = (t.nn.functional.silu(h[:, :inter]) * h[:, inter:]) @ dn.T
                w = t.as_tensor(np.asarray(weights, np.float32)).to(self.device, self.dtype)
                out.append((rows, y * w[:, None]))
            self.calls += 1
        return out, t0

    def collect(self, handle, out: np.ndarray) -> None:
        parts, t0 = handle
        for rows, y in parts:
            out[rows] += y.float().cpu().numpy()
        self.busy_s += time.perf_counter() - t0

    def calibrate(self, sample_expert, cpu_expert_fn, hidden: int, repeats: int = 12) -> float:
        """Time one expert on each device (1 input row, as in decoding) and set
        ``share``, the fraction of routed work the GPU should carry so both
        devices finish together: share = cpu / (gpu + cpu)."""
        self.add((-1, 0), sample_expert)            # layer -1: never routed to
        x = np.random.default_rng(0).standard_normal((1, hidden)).astype(np.float32)
        groups = [(0, np.array([0]), np.ones(1, np.float32))]

        def gpu_once():
            o = np.zeros_like(x)
            self.collect(self.launch(-1, x, groups), o)
        for _ in range(3):
            gpu_once()
            cpu_expert_fn(x)
        t0 = time.perf_counter()
        for _ in range(repeats):
            gpu_once()
        g = (time.perf_counter() - t0) / repeats
        t0 = time.perf_counter()
        for _ in range(repeats):
            cpu_expert_fn(x)
        c = (time.perf_counter() - t0) / repeats
        with self.lock:
            gu, dn = self.experts.pop((-1, 0))
            self.used -= (gu.numel() + dn.numel()) * (2 if self.dtype == self.torch.float16 else 4)
        self.calls = 0
        self.busy_s = 0.0
        self.gpu_ms, self.cpu_ms = g * 1000, c * 1000
        self.share = c / (g + c)
        return self.share

    def stats(self) -> dict:
        return {"device": self.name, "experts": len(self.experts), "used_bytes": self.used,
                "budget_bytes": self.budget, "share": round(self.share, 3),
                "gpu_ms_per_expert": self.gpu_ms, "cpu_ms_per_expert": self.cpu_ms,
                "calls": self.calls, "busy_s": round(self.busy_s, 3)}


def bench_gpu(spec: str, budget: int, hidden: int, inter: int, fmt: str, gs: int = 64) -> dict:
    """GPU side of a node's speed benchmark: ms per expert on the GPU (one input
    row, transfers included) and how many experts fit in ``budget``. Raises
    RuntimeError if the device cannot be opened."""
    from .experts import synthetic_expert
    tier = GpuTier(spec, budget)
    e = synthetic_expert(hidden, inter, fmt, gs)
    tier.calibrate(e, lambda x: e(x), hidden)
    out = {"gpu_ms": tier.gpu_ms, "gpu_experts": int(budget // tier.expert_bytes(inter, hidden)),
           "gpu_name": tier.name}
    tier.experts.clear()
    return out


def choose_gpu_experts(candidates, heat, share: float, per_layer_cap: int | None = None):
    """Hottest experts per layer until they cover ``share`` of that layer's
    routed picks (by ``heat`` counts; uniform when there are none)."""
    by_layer: dict[int, list] = {}
    for key in candidates:
        by_layer.setdefault(key[0], []).append(key)
    chosen = []
    for layer, keys in by_layer.items():
        keys.sort(key=lambda k: -(heat or {}).get(k, 0))
        total = sum((heat or {}).get(k, 0) for k in keys)
        if total <= 0:
            take = int(round(share * len(keys)))
        else:
            acc, take = 0.0, 0
            for k in keys:
                if acc >= share * total:
                    break
                acc += (heat or {}).get(k, 0)
                take += 1
        if per_layer_cap is not None:
            take = min(take, per_layer_cap)
        chosen.append(keys[:take])
    # interleave layers so a budget that runs out still covers every layer
    out = []
    for i in range(max((len(c) for c in chosen), default=0)):
        out += [c[i] for c in chosen if i < len(c)]
    return out
