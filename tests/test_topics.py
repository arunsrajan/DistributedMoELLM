"""Expert tokens and topic probes: the statistics on synthetic routing with a
known answer, the probe and API on a tiny model, and tracking being opt-in."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from moelan.cli import main
from moelan.model import MoEModel
from moelan.probe_passages import PASSAGES
from moelan.server import heat_payload, make_server
from moelan.tiny import make_tiny
from moelan.tokenizer import Tokenizer
from moelan.topics import (TokenTracker, affinities, has_signal, load_topics, run_probe, save_topics, summarize,
                           topics_path)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("topics") / "m", hidden=64, layers=2, experts=8, top_k=2, bf16=False)


@pytest.fixture(autouse=True)
def _topics_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MOELAN_TOPICS_DIR", str(tmp_path / "topics"))


def routing(seed, planted=None, passages=5, topics=4, tokens=200, layers=6, experts=16, k=2):
    """Random routing; ``planted`` (topic, layer, expert) also favours one expert in one topic."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(topics), passages)
    counts = np.zeros((len(labels), layers, experts), np.int32)
    for p, t in enumerate(labels):
        for l in range(layers):
            weights = np.ones(experts)
            if planted and planted[:2] == (t, l):
                weights[planted[2]] = 8.0
            counts[p, l] = rng.multinomial(tokens * k, weights / weights.sum())
    return counts, np.full(len(labels), float(tokens * k)), labels


def test_affinity_finds_a_planted_expert():
    counts, n, labels = routing(0, planted=(1, 3, 5))
    found = affinities(counts, n, labels, 1)
    assert (3, 5) in found and found[(3, 5)][0] > 1.5
    assert (3, 5) not in affinities(counts, n, labels, 0)


def test_random_routing_gives_no_signal():
    for seed in range(5):
        counts, n, labels = routing(seed)
        result = summarize(counts, n, labels, ["a", "b", "c", "d"])
        assert not any(has_signal(t) for t in result["topics"])


def test_one_odd_passage_is_not_a_topic():
    counts, n, labels = routing(1)
    counts[0, 2, 7] = 300     # a single passage of topic 0 hammers one expert
    assert (2, 7) not in affinities(counts, n, labels, 0)


def test_token_tracker_prefers_typical_tokens_and_stays_bounded():
    tr = TokenTracker(capacity=8)
    idx = np.array([[0], [1]] * 200)
    common = np.full(400, 7)                        # everywhere: not typical of either expert
    tr.observe(0, common, idx)
    tr.observe(0, np.full(50, 9), np.zeros((50, 1), np.int32))
    assert [row[0] for row in tr.top(0, 0)] == [9]
    for t in range(100, 300):
        tr.observe(0, np.array([t]), np.array([[3]]))
    assert len(tr.experts[(0, 3)]) <= 16


def test_probe_saves_and_labels_on_a_tiny_model(tiny, tmp_path):
    m, tok = MoEModel(tiny), Tokenizer(tiny)
    passages = {name: [t[:60] for t in PASSAGES[name][:3]] for name in ("Code", "Maths")}
    ref = m.forward(list(b"same either way"), m.new_cache())
    result = run_probe(m, tok, passages, progress=lambda s: None)
    assert m.on_route is None
    np.testing.assert_array_equal(m.forward(list(b"same either way"), m.new_cache()), ref)
    assert [t["name"] for t in result["topics"]] == ["Code", "Maths"]
    assert result["tested"] == 16 and result["tokens"]
    assert sum(t["tokens"] for t in result["topics"]) == 2 * sum(len(tok.encode(x)) for v in passages.values() for x in v)
    path = tmp_path / "r.json"
    save_topics(result, path)
    assert load_topics(path)["tested"] == 16


def test_built_in_passages_are_balanced():
    assert len(PASSAGES) == 8 and all(len(v) == 5 for v in PASSAGES.values())
    assert len({text for v in PASSAGES.values() for text in v}) == 40


def test_cli_probe_writes_the_model_file(tiny, tmp_path, capsys):
    prompts = tmp_path / "p.json"
    prompts.write_text(json.dumps({"a": ["hello there world"] * 3, "b": ["other words entirely"] * 3}))
    main(["probe", "--model", str(tiny), "--prompts", str(prompts)])
    assert "saved to" in capsys.readouterr().out
    assert [t["name"] for t in load_topics(topics_path(tiny))["topics"]] == ["a", "b"]


def _serve(model, tok, **kw):
    server = make_server(model, tok, "tiny", port=0, **kw)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.load(r)


def _chat(base):
    body = json.dumps({"prompt": "secret words secret words", "max_tokens": 2}).encode()
    req = urllib.request.Request(base + "/v1/completions", body, {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60).read()


def test_live_tokens_are_off_by_default(tiny):
    server, base = _serve(MoEModel(tiny), Tokenizer(tiny))
    try:
        _chat(base)
        t = _get(base, "/api/topics")
        assert t["available"] and not t["probed"] and not t["tracking"]
        for e in range(8):
            assert _get(base, f"/api/expert?layer=0&expert={e}")["tokens"] == []
    finally:
        server.shutdown()


def test_expert_api_with_tracking_and_probe(tiny):
    m = MoEModel(tiny)
    server, base = _serve(m, Tokenizer(tiny), expert_tokens=True)
    try:
        assert _get(base, "/api/topics")["tracking"]
        _chat(base)
        got = [_get(base, f"/api/expert?layer=0&expert={e}") for e in range(8)]
        assert any(g["source"] == "live" for g in got)
        result = {"topics": [{"name": "Code", "passages": 5, "tokens": 9, "experts": 9, "chance": 1.0, "best": None}],
                  "affinity": {"0:3": [[0, 2.5]]}, "tested": 16, "tokens": {"1:2": [{"text": "x", "count": 3, "lift": 2}]}}
        save_topics(result, topics_path(tiny))
        t = _get(base, "/api/topics")                       # picked up without a restart
        assert t["probed"] and t["topics"][0]["signal"] and t["affinity"]["0:3"] == [[0, 2.5]]
        assert _get(base, "/api/expert?layer=1&expert=2")["source"] in ("probe", "live")
        with pytest.raises(urllib.error.HTTPError):
            _get(base, "/api/expert?layer=x")
    finally:
        server.shutdown()


def test_monitor_mode_has_no_topics():
    server, base = _serve(None, None, monitor=type("M", (), {"snapshot": lambda self: []})())
    try:
        assert _get(base, "/api/topics")["available"] is False
        assert _get(base, "/api/expert?layer=0&expert=0")["tokens"] == []
    finally:
        server.shutdown()


def test_heat_payload_lists_replicas(tiny):
    m = MoEModel(tiny)
    assert heat_payload(m)["replicas"] == {}
    m.backend.plan = type("P", (), {"per_node": {"a": [], "b": []}, "owners": {(0, 1): ["a", "b"], (0, 2): ["a"]}})()
    assert heat_payload(m)["replicas"] == {"0:1": [1, 2]}
