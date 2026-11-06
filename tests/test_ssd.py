"""Experts on the workers' SSDs: the hottest of each worker's experts in RAM,
the rest computed from its SSD, with the experts either spread over the
cluster's SSDs (distributed) or copied to every worker's SSD (replicated)."""

from types import SimpleNamespace

import pytest

from moelan.cli import _add_cluster_args, main
from moelan.config import load_config, model_fingerprint
from moelan.coordinator import DistributedBackend as _DistributedBackend
from moelan.experts import CheckpointSource, ExpertStore, open_source, resident_bytes
from moelan.model import MoEModel
from moelan.planner import LOCAL, add_ssd_copies, make_plan
from moelan.tiny import make_tiny

from test_moelan import Worker, greedy


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("ssd") / "m", layers=3, experts=8, top_k=2)


def DistributedBackend(*args, **kwargs):
    kwargs.setdefault("placement", "ram")      # in-process workers share this machine
    kwargs.setdefault("progress", lambda m: None)
    return _DistributedBackend(*args, **kwargs)


def ram_for(tiny, n):
    """A RAM budget that holds ``n`` experts in the default (f32) format."""
    cfg = load_config(tiny)
    return n * resident_bytes(cfg, "f32")


def disk_for(tiny, n):
    """An SSD budget that holds ``n`` pushed experts."""
    cfg = load_config(tiny)
    src = open_source(tiny, cfg)
    return n * (max(src.nbytes((l, 0)) for l in cfg.moe_layers()) + 512)


def cached(tmp_path, name, tiny):
    return sorted((tmp_path / name / model_fingerprint(tiny)).glob("L*_E*.safetensors"))


ALL = 3 * 8


# --- planning -----------------------------------------------------------------

def test_replicated_copies_fill_each_ssd_hottest_first():
    heat = {(l, e): 10 * (8 - e) + l for l in (0, 1) for e in range(8)}     # low expert ids are hot
    plan = make_plan([0, 1], 8, 1, {"a": 8, "b": 8}, heat=heat, replicate=False)
    add_ssd_copies(plan, [0, 1], 8, {"a": 16, "b": 12}, heat=heat)
    assert len(plan.per_node["a"]) + len(plan.held["a"]) == 16              # a's SSD holds everything
    assert len(plan.per_node["b"]) + len(plan.held["b"]) == 12              # b's: its own 8 and 4 copies
    others = sorted(set(plan.per_node["a"]), key=lambda k: -heat[k])
    assert plan.held["b"] == others[:4]                                     # the hottest of a's
    for k, own in plan.owners.items():
        assert own[0] in ("a", "b") and len(set(own)) == len(own)
        assert (k in plan.held["b"]) == ("b" in own[1:])
    assert "copies on its SSD" in plan.summary(1)


def test_owner_prefers_a_node_with_the_expert_in_ram():
    key = (0, 3)
    conns = {n: SimpleNamespace(alive=True, in_ram=set()) for n in ("p", "q", "r")}
    me = SimpleNamespace(plan=SimpleNamespace(owners={key: ["p", "q", "r"]}), conns=conns)
    own = lambda load: _DistributedBackend._owner(me, *key, load)              # noqa: E731
    assert own({}) == "p"                                       # the primary, even computing it from its SSD
    conns["r"].in_ram.add(key)
    assert own({"p": 1}) == "r"                                 # another node has it in RAM and is idle
    assert own({"p": 1, "r": 2}) == "p"
    conns["p"].alive = False
    assert own({"r": 5}) == "r"                                 # in RAM beats idle q on its SSD
    conns["r"].alive = False
    assert own({}) == "q"                                       # any holder before the coordinator
    conns["q"].alive = False
    assert own({}) == LOCAL


# --- the worker's two tiers -----------------------------------------------------

def test_store_keeps_a_bounded_number_of_mapped_experts(tmp_path):
    col = make_tiny(tmp_path / "col", layers=2, experts=8, top_k=2, layout="colibri_int4", int4_gs=32)
    cfg = load_config(col)
    store = ExpertStore([CheckpointSource(col, cfg)], fmt="int8", map_size=3)
    for e in range(6):
        store.get((0, e))
    assert len(store.mapped) == 3 and store.misses == 6 and not store.lru
    assert list(store.mapped) == [(0, 3), (0, 4), (0, 5)]          # the most recently used
    store.forget([(0, 4)])
    assert (0, 4) not in store.mapped


def test_distributed_keeps_hottest_in_ram_and_reads_the_rest_from_ssd(tiny, tmp_path):
    cfg = load_config(tiny)
    prompt = [256] + list(b"from the SSD")
    ref = greedy(MoEModel(tiny), prompt)
    heat = {(l, e): 100 - 10 * e for l in cfg.moe_layers() for e in range(8)}
    w1 = Worker(tmp_path, "s1", ram_for(tiny, 3), disk_budget=disk_for(tiny, ALL))
    w2 = Worker(tmp_path, "s2", ram_for(tiny, 3), disk_budget=disk_for(tiny, ALL))
    try:
        b = DistributedBackend(str(tiny), cfg, [w1.address, w2.address], ssd="distributed", heat=heat)
        assert not b.plan.unplaced and all(len(o) == 1 for o in b.plan.owners.values())
        assert sorted(len(b.plan.per_node[n]) for n in ("s1", "s2")) == [12, 12]     # by RAM: half each
        for n in ("s1", "s2"):
            c = b.conns[n]
            assert c.ram_experts == 3 and c.ssd_experts == 9
            mine = b.plan.per_node[n]
            assert c.in_ram == set(sorted(mine, key=lambda k: -heat[k])[:3])         # its hottest in RAM
            assert len(cached(tmp_path, n, tiny)) == 12
        assert greedy(MoEModel(tiny, backend=b), prompt) == ref
        rows = {r["node"]: r for r in b.stats()}
        assert rows["s1"]["ssd_reads"] + rows["s2"]["ssd_reads"] > 0
        assert rows["s1"]["ram_experts"] == 3 and rows["s1"]["ssd_experts"] == 9
        assert b.local.misses == 0                         # nothing fell back to the coordinator
        b.close()
    finally:
        w1.kill(), w2.kill()


def test_distributed_respects_each_workers_disk(tiny, tmp_path):
    cfg = load_config(tiny)
    prompt = [256] + list(b"small disk")
    ref = greedy(MoEModel(tiny), prompt, 6)
    w1 = Worker(tmp_path, "d1", ram_for(tiny, 2), disk_budget=disk_for(tiny, 5))
    w2 = Worker(tmp_path, "d2", ram_for(tiny, 2), disk_budget=disk_for(tiny, 7))
    try:
        msgs = []
        b = DistributedBackend(str(tiny), cfg, [w1.address, w2.address], ssd="distributed", progress=msgs.append)
        assert len(b.plan.per_node["d1"]) == 5 and len(b.plan.per_node["d2"]) == 7
        assert len(b.plan.unplaced) == ALL - 12              # the coordinator computes the rest
        assert any("room for 5 of 24 on its SSD" in m for m in msgs)
        assert greedy(MoEModel(tiny, backend=b), prompt, 6) == ref
        b.close()
    finally:
        w1.kill(), w2.kill()


def test_disk_budget_removes_experts_a_worker_no_longer_holds(tiny, tmp_path):
    cfg = load_config(tiny)
    prompt = [256] + list(b"new plan")
    ref = greedy(MoEModel(tiny), prompt, 6)
    w = Worker(tmp_path, "prune", ram_for(tiny, 2), disk_budget=disk_for(tiny, 12))
    try:
        hot_low = {(l, e): 8 - e for l in cfg.moe_layers() for e in range(8)}
        hot_high = {(l, e): e + 1 for l in cfg.moe_layers() for e in range(8)}
        b = DistributedBackend(str(tiny), cfg, [w.address], ssd="distributed", heat=hot_low)
        first = set(b.plan.per_node["prune"])
        assert len(first) == 12 and len(cached(tmp_path, "prune", tiny)) == 12
        b.close()
        msgs = []
        b = DistributedBackend(str(tiny), cfg, [w.address], ssd="distributed", heat=hot_high, progress=msgs.append)
        second = set(b.plan.per_node["prune"])
        assert second != first
        files = cached(tmp_path, "prune", tiny)
        assert len(files) <= 12 and any("no longer holds" in m for m in msgs)
        assert {(int(f.stem[1:4]), int(f.stem[6:])) for f in files} >= second
        assert greedy(MoEModel(tiny, backend=b), prompt, 6) == ref
        b.close()
    finally:
        w.kill()


# --- replicated -------------------------------------------------------------------

def test_replicated_every_worker_holds_every_expert_and_covers_a_failure(tiny, tmp_path):
    cfg = load_config(tiny)
    prompt = [256] + list(b"replicated SSD")
    ref = greedy(MoEModel(tiny), prompt, 10)
    w1 = Worker(tmp_path, "r1", ram_for(tiny, 4), disk_budget=disk_for(tiny, ALL))
    w2 = Worker(tmp_path, "r2", ram_for(tiny, 4), disk_budget=disk_for(tiny, ALL))
    try:
        msgs = []
        b = DistributedBackend(str(tiny), cfg, [w1.address, w2.address], ssd="replicated", progress=msgs.append)
        assert all(sorted(o) == ["r1", "r2"] for o in b.plan.owners.values())
        assert any("2 of 2 workers hold every expert" in m for m in msgs)
        for n in ("r1", "r2"):
            assert len(cached(tmp_path, n, tiny)) == ALL
            assert b.conns[n].ram_experts == 4 and b.conns[n].ssd_experts == ALL - 4
        model = MoEModel(tiny, backend=b)
        out = []
        for i, t in enumerate(model.generate(prompt, max_new_tokens=10, stop_ids=[])):
            out.append(t)
            if i == 3:
                w2.kill()
        assert out == ref
        assert not b.conns["r2"].alive
        assert b.local.misses == 0                         # r1 took r2's experts, not the coordinator
        b.close()
    finally:
        w1.kill(), w2.kill()


def test_replicated_with_local_model_copies_sends_nothing(tiny, tmp_path):
    cfg = load_config(tiny)
    prompt = [256] + list(b"own copy")
    ref = greedy(MoEModel(tiny), prompt, 6)
    w1 = Worker(tmp_path, "c1", ram_for(tiny, 2), model_dir=str(tiny), disk_budget=0)
    w2 = Worker(tmp_path, "c2", ram_for(tiny, 2), model_dir=str(tiny), disk_budget=0)
    try:
        msgs = []
        b = DistributedBackend(str(tiny), cfg, [w1.address, w2.address], ssd="replicated", progress=msgs.append)
        assert not any("sending" in m for m in msgs)
        assert any("its own copy of the model" in m for m in msgs)
        assert all(len(o) == 2 for o in b.plan.owners.values())
        assert greedy(MoEModel(tiny, backend=b), prompt, 6) == ref
        b.close()
    finally:
        w1.kill(), w2.kill()


def test_ram_only_mode_is_unchanged(tiny, tmp_path):
    """--ssd off: capacity is RAM, the rest stays on the coordinator's disk."""
    cfg = load_config(tiny)
    w = Worker(tmp_path, "ramonly", ram_for(tiny, 5), disk_budget=disk_for(tiny, ALL))
    try:
        b = DistributedBackend(str(tiny), cfg, [w.address])
        assert len(b.plan.per_node["ramonly"]) == 5 and len(b.plan.unplaced) == ALL - 5
        assert b.conns["ramonly"].ram_experts == 5 and b.conns["ramonly"].ssd_experts == 0
        b.close()
    finally:
        w.kill()


def test_worker_reports_its_disk_and_cli_takes_the_modes(tiny, tmp_path, monkeypatch):
    import argparse
    w = Worker(tmp_path, "hello", ram_for(tiny, 2), disk_budget=12345)
    try:
        from moelan.coordinator import WorkerConn
        c = WorkerConn(w.address)
        info = c.connect()
        assert info["disk_budget_bytes"] == 12345 and info["cached_bytes"] == {}
        c.close()
    finally:
        w.kill()
    p = argparse.ArgumentParser()
    _add_cluster_args(p)
    assert p.parse_args(["--model", "m", "--ssd", "replicated"]).ssd == "replicated"
    assert p.parse_args(["--model", "m"]).ssd == "off"
    with pytest.raises(SystemExit):
        p.parse_args(["--model", "m", "--ssd", "sometimes"])
    seen = {}
    import moelan.worker as worker_mod
    monkeypatch.setattr(worker_mod, "serve", lambda **kw: seen.update(kw))
    # --port 0: a real worker may hold the default port on this machine
    main(["worker", "--port", "0", "--disk", "200G", "--no-advertise"])
    assert seen["disk_budget"] == 200 * 2**30
