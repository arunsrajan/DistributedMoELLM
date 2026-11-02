"""Most-picked experts on the fastest GPUs, and the counts behind it: a topic probe
for the first start, then real traffic saved by the server."""

import json
from collections import Counter

import pytest

from moelan.cli import main
from moelan.config import load_config
from moelan.coordinator import DistributedBackend
from moelan.heatfiles import (HALF_LIFE_TOKENS, MIN_TRAFFIC_TOKENS, TrafficRecorder, choose_heat, heat_dir,
                              load_topic_heat, parse_subtopics, run_topic_probe, tokens_of, topic_path, traffic_path)
from moelan.model import MoEModel, load_heat
from moelan.planner import LOCAL, make_plan
from moelan.tiny import make_tiny
from moelan.tokenizer import Tokenizer

from test_moelan import Worker


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("heat") / "m", hidden=64, layers=2, experts=8, top_k=2, bf16=False)


@pytest.fixture(autouse=True)
def _heat_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MOELAN_HEAT_DIR", str(tmp_path / "heat"))


NODES = {"gpu1": 10**6, "gpu2": 10**6, LOCAL: 10**6}
SHARES = {"gpu1": 0.4, "gpu2": 0.3, LOCAL: 0.3}
HEAT = {(l, e): 16 - e for l in (0, 1) for e in range(16)}       # lower numbers are hotter


def test_hottest_experts_go_to_the_gpus_and_the_fastest_gets_the_hottest():
    plan = make_plan([0, 1], 16, 1, NODES, heat=HEAT, replicate=False, shares=SHARES,
                     fast=[("gpu1", 10, 0.05), ("gpu2", 10, 0.10)])
    for layer in (0, 1):
        assert plan.owners[(layer, 0)] == ["gpu1"]
        hot = [plan.owners[(layer, e)][0] for e in range(5)]
        assert LOCAL not in hot and "gpu2" in hot                        # both GPUs carry hot experts
    swapped = make_plan([0, 1], 16, 1, NODES, heat=HEAT, replicate=False, shares=SHARES,
                        fast=[("gpu1", 10, 0.10), ("gpu2", 10, 0.05)])
    assert swapped.owners[(0, 0)] == ["gpu2"]
    assert sorted(plan.owners) == sorted(HEAT) and not plan.unplaced     # everything placed exactly once
    full = make_plan([0, 1], 16, 1, NODES, heat=HEAT, replicate=False, shares=SHARES,
                     fast=[("gpu1", 2, 0.05), ("gpu2", 2, 0.10)])
    assert full.owners[(0, 0)] == ["gpu1"] and full.owners[(0, 1)] == ["gpu2"]   # a full GPU takes no more of them
    assert LOCAL in [full.owners[(0, e)][0] for e in range(2, 16)]
    assert sum(len(v) for v in plan.per_node.values()) == 32


def test_a_gpu_takes_no_more_than_its_share_of_the_picks():
    """A roomy GPU must not take everything: picks that queue on it cost more."""
    shares = {"gpu1": 0.25, LOCAL: 0.75}
    plan = make_plan([0], 16, 1, {"gpu1": 10**6, LOCAL: 10**6}, heat={(0, e): 16 - e for e in range(16)},
                     replicate=False, shares=shares, fast=[("gpu1", 10**4, 0.05)])
    total = sum(16 - e for e in range(16))
    held = sum(16 - e for _, e in plan.per_node["gpu1"])
    assert held / total < 0.35 and len(plan.per_node[LOCAL]) > 8


def test_no_counts_or_unused_gpu_changes_nothing():
    base = make_plan([0], 16, 1, NODES, replicate=False, shares=SHARES)
    assert make_plan([0], 16, 1, NODES, replicate=False, shares=SHARES,
                     fast=[("gpu1", 4, 0.05)]).owners == base.owners
    gone = make_plan([0], 16, 1, NODES, heat=HEAT, replicate=False, shares={"gpu2": 0.5, LOCAL: 0.5, "gpu1": 0.0},
                     fast=[("gpu1", 4, 0.05)])
    assert "gpu1" not in gone.per_node and not gone.unplaced


def test_parse_subtopics():
    reply = "Here you go:\n1. **Testing**\n2) Code review:\n3. Testing\n- Deployment\nA closing remark."
    assert parse_subtopics(reply, 12) == ["Testing", "Code review", "Deployment"]
    assert parse_subtopics(reply, 2) == ["Testing", "Code review"]
    assert parse_subtopics("no list here", 5) == []


def test_real_traffic_replaces_a_topic_probe_once_there_is_enough(tiny, tmp_path):
    cfg = load_config(tiny)
    assert not choose_heat(tiny, cfg, progress=lambda m: None)       # nothing known yet
    topic_path(tiny).parent.mkdir(parents=True)
    topic_path(tiny).write_text(json.dumps({"topic": "t", "heat": {"0:1": 9}}))
    msgs = []
    assert choose_heat(tiny, cfg, progress=msgs.append) == {(0, 1): 9}
    assert "'t'" in msgs[0]
    slots = len(cfg.moe_layers()) * cfg.top_k
    traffic_path(tiny).write_text(json.dumps({"0:2": (MIN_TRAFFIC_TOKENS - 1) * slots}))
    assert (0, 1) in choose_heat(tiny, cfg, progress=lambda m: None)     # not enough yet
    traffic_path(tiny).write_text(json.dumps({"0:2": MIN_TRAFFIC_TOKENS * slots}))
    assert (0, 2) in choose_heat(tiny, cfg, progress=msgs.append) and "real traffic" in msgs[-1]
    explicit = tmp_path / "mine.json"
    explicit.write_text(json.dumps({"1:3": 5}))
    assert choose_heat(tiny, cfg, explicit, progress=lambda m: None) == {(1, 3): 5}


def test_recorder_saves_counts_and_lets_old_traffic_fade(tiny, tmp_path):
    cfg = load_config(tiny)
    path = tmp_path / "traffic.json"
    m = MoEModel(tiny)
    rec = TrafficRecorder(m.backend, cfg, path, every=3600)
    rec.flush()
    assert not path.exists()                                         # nothing served, nothing written
    list(m.generate([1, 2, 3], max_new_tokens=4, stop_ids=[]))
    picks = sum(m.backend.heat.values())
    rec.flush()
    assert sum(load_heat(path).values()) == pytest.approx(picks, abs=1)
    assert set(json.loads(path.read_text())) <= {f"{l}:{e}" for l in cfg.moe_layers() for e in range(8)}  # counts only
    rec.flush()
    assert sum(load_heat(path).values()) == pytest.approx(picks, abs=1)      # counted once, not twice

    class Fake:
        heat = Counter({(0, 7): len(cfg.moe_layers()) * cfg.top_k * HALF_LIFE_TOKENS})
    before = load_heat(path)
    next_run = TrafficRecorder(Fake, cfg, path, every=3600)                  # a restart keeps what was saved
    next_run.flush()
    after = load_heat(path)
    key = next(k for k in before if k != (0, 7))
    assert after[key] == pytest.approx(before[key] / 2, abs=0.1)             # half after one half-life
    assert tokens_of(after, cfg) >= HALF_LIFE_TOKENS


def test_recorder_thread_saves_on_stop(tiny, tmp_path):
    m = MoEModel(tiny)
    rec = TrafficRecorder(m.backend, load_config(tiny), tmp_path / "t.json", every=3600)
    rec.start()
    list(m.generate([1, 2], max_new_tokens=2, stop_ids=[]))
    rec.stop()
    assert sum(load_heat(tmp_path / "t.json").values()) > 0


def test_topic_probe_counts_only_written_tokens_and_continues(tiny, tmp_path):
    cfg = load_config(tiny)
    m, tok = MoEModel(tiny), Tokenizer(tiny)
    path = tmp_path / "topic.json"
    quiet = lambda msg: None
    r = run_topic_probe(m, tok, "t", path, per_subtopic=1, passage_tokens=4, names=["a", "b"], progress=quiet)
    assert r["complete"] and len(r["passages"]) == 2 and m.on_route is None
    # all 4 written tokens go back through the model; the prompt is not counted
    assert r["tokens"] == 8
    slots = len(cfg.moe_layers()) * cfg.top_k
    assert sum(r["heat"].values()) == 8 * slots
    first = dict(r["passages"][0])
    more = run_topic_probe(m, tok, "t", path, per_subtopic=2, passage_tokens=4, progress=quiet)
    assert len(more["passages"]) == 4 and more["passages"][0] == first and more["tokens"] == 16
    assert load_topic_heat(path)["subtopics"] == ["a", "b"]
    other = run_topic_probe(m, tok, "u", path, per_subtopic=1, passage_tokens=4, names=["c", "d"], progress=quiet)
    assert other["topic"] == "u" and len(other["passages"]) == 2         # another topic starts afresh


def test_topic_probe_stops_clearly_on_an_unreadable_reply(tiny, tmp_path):
    m, tok = MoEModel(tiny), Tokenizer(tiny)
    m.generate = lambda *a, **k: iter(tok.encode("no list"))
    with pytest.raises(SystemExit, match="could not read sub-topics"):
        run_topic_probe(m, tok, "t", tmp_path / "x.json", progress=lambda msg: None)


def test_cli_topic_probe_then_placement_uses_it(tiny, capsys):
    path = topic_path(tiny)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"topic": "t", "subtopics": ["a", "b"], "passages": [], "heat": {}, "tokens": 0,
                                "complete": False}))
    args = ["probe", "--model", str(tiny), "--topic", "t", "--passages", "1", "--passage-tokens", "3"]
    main(args)
    doc = load_topic_heat(path)
    assert doc["complete"] and doc["heat"]
    main(args)
    assert "already measured" in capsys.readouterr().out
    assert sum(choose_heat(tiny, load_config(tiny), progress=lambda m: None).values()) == sum(doc["heat"].values())
    with pytest.raises(SystemExit):
        main(args + ["--prompts", "x.json"])
    assert heat_dir() == path.parent


def test_speed_placement_lists_gpus_fastest_first(tiny, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import moelan.coordinator as coord
    monkeypatch.setattr(coord, "_this_host", lambda: "coordinator-box")
    w = Worker(tmp_path, "gpuw", 2**22, gpu="torch-cpu", gpu_budget=2**30, gpu_mode="auto")
    try:
        backend = DistributedBackend(str(tiny), load_config(tiny), [w.address], placement="speed", gpu="torch-cpu",
                                     gpu_budget=2**30, heat={(0, 0): 9}, progress=lambda m: None)
        ms = {"gpuw": backend.speed["nodes"]["gpuw"]["gpu"]["gpu_ms"], LOCAL: backend.speed["local_gpu"]["gpu_ms"]}
        assert all(ms[nid] == t and slots > 0 for nid, slots, t in backend.fast)
        assert [t for _, _, t in backend.fast] == sorted(t for _, _, t in backend.fast)
        backend.close()
    finally:
        w.kill()


def test_worker_chooses_gpu_experts_again_for_a_new_plan(tiny, tmp_path):
    """A worker often outlives the coordinator; it must not keep an old choice."""
    pytest.importorskip("torch")
    w = Worker(tmp_path, "keep", 2**24, gpu="torch-cpu", gpu_budget=2**30, gpu_mode="max")
    try:
        backend = DistributedBackend(str(tiny), load_config(tiny), [w.address], placement="ram",
                                     replicate=False, progress=lambda m: None)
        c = backend.conns["keep"]
        assert c.gpu["experts"] == len(backend.plan.per_node["keep"])
        backend.plan.per_node["keep"] = backend.plan.per_node["keep"][:3]
        backend._setup_worker(c)
        assert c.gpu["experts"] == 3
        backend.close()
    finally:
        w.kill()
