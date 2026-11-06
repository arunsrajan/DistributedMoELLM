"""A GPU that makes layers slower in real use is dropped (--gpu-mode auto).

The start-up calibration times one expert at a time in a tight loop; on a
laptop RTX 4050 it said 0.46 ms per expert against 0.36 ms on the CPU, and
the GPU got 43% of the routed work, but generating, layers that touched the
GPU took 7.7 ms against 2.3 ms on the CPU alone. These tests slow the
``torch-cpu`` test device down (or the CPU) and check what the node does."""

import functools
import threading
import time

import pytest

from moelan import accel
from moelan.accel import GpuGuard
from moelan.config import load_config
from moelan.coordinator import DistributedBackend
from moelan.experts import ExpertStore
from moelan.model import MoEModel
from moelan.tiny import make_tiny
from moelan.worker import serve


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("tiny"), layers=3, experts=8, top_k=2)


def greedy(model, prompt, n):
    return list(model.generate(prompt, max_new_tokens=n, stop_ids=[]))


PROMPT = [256] + list(b"gpu or not")


class Worker:
    def __init__(self, tmp_path, name, budget, **extra):
        self.ready, self.stop = threading.Event(), threading.Event()
        self.t = threading.Thread(target=serve, daemon=True, kwargs=dict(
            host="127.0.0.1", port=0, budget=budget, cache_root=tmp_path / name, node_id=name,
            advertise=False, ready=self.ready, stop=self.stop, **extra))
        self.t.start()
        assert self.ready.wait(10)
        self.address = f"127.0.0.1:{self.ready.port}"

    def kill(self):
        self.stop.set()
        self.t.join(10)


# --- the comparison itself ----------------------------------------------------------

def run_until_verdict(guard, gpu_s, cpu_s):
    while True:
        arm = guard.choose(1)
        verdict = guard.record(arm, gpu_s if arm == "gpu" else cpu_s)
        if verdict is not None:
            return verdict


def test_guard_drops_a_gpu_that_is_not_faster():
    keep, g, c = run_until_verdict(GpuGuard(samples=4, seed=1), gpu_s=0.0077, cpu_s=0.0023)
    assert not keep and g == pytest.approx(7.7) and c == pytest.approx(2.3)
    keep, _, _ = run_until_verdict(GpuGuard(samples=4), gpu_s=0.002, cpu_s=0.002)
    assert not keep                                     # no faster: not worth its memory either


def test_guard_keeps_a_faster_gpu_and_checks_again_later():
    guard = GpuGuard(samples=4, recheck=10)
    assert run_until_verdict(guard, gpu_s=0.001, cpu_s=0.003) == (True, pytest.approx(1.0), pytest.approx(3.0))
    assert [guard.choose(1) for _ in range(10)] == ["on"] * 10      # used, untimed
    assert guard.choose(1) in ("gpu", "cpu")                         # then compared again
    assert guard.stats() == {"checks": 1, "layer_ms_with_gpu": 1.0, "layer_ms_cpu_alone": 3.0}


def test_guard_times_generating_layers_only_and_mostly_on_the_gpu():
    guard = GpuGuard()
    assert guard.choose(GpuGuard.ROWS + 1) == "on"                   # reading a prompt: always the GPU
    assert guard.record("on", 1.0) is None
    arms = [guard.choose(1) for _ in range(4000)]
    assert set(arms) == {"gpu", "cpu"}
    assert arms.count("cpu") / len(arms) == pytest.approx(0.25, abs=0.03)


# --- a node in real use ---------------------------------------------------------

@pytest.fixture
def hybrid(monkeypatch):
    """--gpu-mode auto with a fixed calibration (the GPU takes half the routed
    work) and comparisons after 8 layers of each kind."""
    pytest.importorskip("torch")

    def calibrate(self, sample_expert, cpu_expert_fn, hidden, repeats=12):
        self.gpu_ms = self.cpu_ms = 0.1
        self.share = 0.5
        return self.share
    monkeypatch.setattr(accel.GpuTier, "calibrate", calibrate)
    monkeypatch.setattr(accel, "GpuGuard", functools.partial(GpuGuard, samples=8))


@pytest.fixture
def slow_gpu(hybrid, monkeypatch):
    collect = accel.GpuTier.collect

    def slow(self, handle, out):
        time.sleep(0.005)               # every layer that touches the GPU waits 5 ms more
        collect(self, handle, out)
    monkeypatch.setattr(accel.GpuTier, "collect", slow)


def test_a_gpu_that_slows_layers_down_is_dropped(tiny, slow_gpu):
    ref = greedy(MoEModel(tiny), PROMPT, 40)
    msgs = []
    m = MoEModel(tiny, gpu="torch-cpu", gpu_budget=2**30, gpu_mode="auto", progress=msgs.append)
    store = m.backend.store
    tier = store.gpu
    assert tier is not None and tier.experts and store.gpu_guard is not None
    assert greedy(m, PROMPT, 40) == ref                  # the same tokens across the switch
    assert store.gpu is None and store.gpu_guard is None
    assert not tier.experts and tier.used == 0           # its memory freed
    info = store.gpu_stats()
    assert info["experts"] == 0 and info["dropped"].startswith("generating, a layer took")
    assert info["layer_ms_with_gpu"] > info["layer_ms_cpu_alone"] + 4
    assert any("GPU dropped" in s for s in msgs)
    calls = tier.calls
    greedy(m, PROMPT, 4)
    assert tier.calls == calls                           # not used any more


def test_max_mode_keeps_a_slow_gpu(tiny, slow_gpu):
    m = MoEModel(tiny, gpu="torch-cpu", gpu_budget=2**30, gpu_mode="max")
    greedy(m, PROMPT, 40)
    assert m.backend.store.gpu is not None and m.backend.store.gpu_guard is None


def test_a_gpu_that_speeds_layers_up_is_kept(tiny, hybrid, monkeypatch):
    cpu = ExpertStore._compute_cpu

    def slow_cpu(self, layer, x, groups, out, one):
        time.sleep(0.003 * len(groups))     # the CPU needs 3 ms per expert
        cpu(self, layer, x, groups, out, one)
    monkeypatch.setattr(ExpertStore, "_compute_cpu", slow_cpu)
    msgs = []
    m = MoEModel(tiny, gpu="torch-cpu", gpu_budget=2**30, gpu_mode="auto", progress=msgs.append)
    greedy(m, PROMPT, 40)
    store = m.backend.store
    assert store.gpu is not None and store.gpu_guard.checks >= 1
    st = store.gpu_stats()
    assert st["experts"] > 0 and st["layer_ms_with_gpu"] < st["layer_ms_cpu_alone"]
    assert any("GPU kept" in s for s in msgs)


def test_a_worker_drops_its_slow_gpu_tells_the_coordinator_and_remembers(tiny, tmp_path, slow_gpu,
                                                                          monkeypatch):
    import moelan.coordinator as coord
    ref = greedy(MoEModel(tiny), PROMPT, 40)
    w = Worker(tmp_path, "g1", 2**22, gpu="torch-cpu", gpu_budget=2**30, gpu_mode="auto")
    try:
        msgs = []
        backend = DistributedBackend(str(tiny), load_config(tiny), [w.address], placement="ram",
                                     progress=msgs.append)
        row = next(r for r in backend.stats() if r["node"] == "g1")
        assert row["gpu"]["experts"] > 0
        assert greedy(MoEModel(tiny, backend=backend), PROMPT, 40) == ref
        row = next(r for r in backend.stats() if r["node"] == "g1")
        assert row["gpu"]["experts"] == 0 and "dropped" in row["gpu"]
        assert any(s.startswith("g1: torch-cpu") and "dropped" in s for s in msgs)
        backend.close()

        # A new coordinator: the worker's benchmark leaves the GPU out, so a
        # speed placement plans for its CPU alone, and the GPU is not set up again.
        monkeypatch.setattr(coord, "_this_host", lambda: "coordinator-box")
        msgs = []
        backend = DistributedBackend(str(tiny), load_config(tiny), [w.address], placement="speed",
                                     progress=msgs.append)
        rows = {r["node"]: r for r in backend.stats()}
        assert rows["g1"]["speed"]["gpu"] is None
        assert rows["g1"]["speed"]["expert_ms"] == rows["g1"]["speed"]["cpu_ms"]
        assert any("GPU not usable (slower than the CPU in real use" in s for s in msgs)
        assert not (rows["g1"]["gpu"] or {}).get("experts")
        assert greedy(MoEModel(tiny, backend=backend), PROMPT, 8) == ref[:8]
        backend.close()
    finally:
        w.kill()
