import socket
import threading

import numpy as np
import pytest

from moelan.config import load_config
from moelan.coordinator import DistributedBackend as _DistributedBackend


def DistributedBackend(*args, **kwargs):
    """These tests check the distribution mechanics, so they place by RAM
    unless they ask otherwise (speed placement has its own tests)."""
    kwargs.setdefault("placement", "ram")
    return _DistributedBackend(*args, **kwargs)
from moelan.experts import QuantMatrix
from moelan.model import MoEModel, rms_norm, group_by_expert, route
from moelan.planner import LOCAL, make_plan
from moelan.protocol import ProtocolError, recv_msg, send_msg
from moelan.safetensors_io import SafeTensorFile, bf16_to_f32, f32_to_bf16, save_file
from moelan.tiny import make_tiny
from moelan.worker import serve


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("tiny"), layers=3, experts=8, top_k=2)


class Worker:
    """An in-process worker on an ephemeral port."""

    def __init__(self, tmp_path, name, budget, model_dir=None, **extra):
        self.ready, self.stop = threading.Event(), threading.Event()
        self.t = threading.Thread(target=serve, daemon=True, kwargs=dict(
            host="127.0.0.1", port=0, budget=budget, cache_root=tmp_path / name,
            model_dir=model_dir, node_id=name, advertise=False,
            ready=self.ready, stop=self.stop, **extra))
        self.t.start()
        assert self.ready.wait(10)
        self.address = f"127.0.0.1:{self.ready.port}"

    def kill(self):
        self.stop.set()
        self.t.join(10)


def greedy(model, prompt, n=12):
    return list(model.generate(prompt, max_new_tokens=n, stop_ids=[]))


# --- building blocks -------------------------------------------------------

def test_safetensors_roundtrip_bf16(tmp_path):
    a = np.random.default_rng(0).standard_normal((5, 7)).astype(np.float32)
    save_file({"a": a, "b": a}, tmp_path / "x.safetensors", bf16={"b"})
    f = SafeTensorFile(tmp_path / "x.safetensors")
    assert np.array_equal(f.get("a"), a)
    assert f.dtype_tag("b") == "BF16"
    np.testing.assert_allclose(f.get("b"), a, rtol=1e-2)
    assert np.array_equal(bf16_to_f32(f32_to_bf16(f.get("b"))), f.get("b"))  # lossless re-round


def test_route_and_grouping():
    logits = np.array([[0.0, 3.0, 1.0, 2.0], [5.0, 0.0, 0.0, 4.0]], np.float32)
    idx, w = route(logits, 2, True)
    assert idx.tolist() == [[1, 3], [0, 3]]
    np.testing.assert_allclose(w.sum(1), 1.0, rtol=1e-6)
    experts, offsets, rows, weights = group_by_expert(idx, w)
    assert experts.tolist() == [0, 1, 3]
    assert rows[offsets[2]:offsets[3]].tolist() == [0, 1]


def test_planner_balances_layers_and_respects_budget():
    plan = make_plan([0, 1], 8, 100, {"a": 800, "b": 800}, replicate=False)
    assert not plan.unplaced
    for layer in (0, 1):
        on_a = sum(1 for k in plan.per_node["a"] if k[0] == layer)
        assert on_a == 4  # half of each layer on each node
    small = make_plan([0, 1], 8, 100, {"a": 500}, heat={(1, 7): 10})
    assert len(small.per_node["a"]) == 5 and len(small.unplaced) == 11
    assert (1, 7) in small.per_node["a"]  # hottest placed first


def test_planner_replicates_with_spare_capacity():
    plan = make_plan([0], 4, 100, {"a": 800, "b": 800}, heat={(0, 2): 5})
    assert all(len(o) == 2 for o in plan.owners.values())


def test_protocol_roundtrip_and_bad_magic():
    a, b = socket.socketpair()
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    send_msg(a, {"op": "t"}, {"x": x}, {"blob": b"abc"})
    h, arrays, blobs = recv_msg(b)
    assert h["op"] == "t" and np.array_equal(arrays["x"], x) and bytes(blobs["blob"]) == b"abc"
    a.sendall(b"NOPE" + bytes(12))
    with pytest.raises(ProtocolError):
        recv_msg(b)
    a.close(), b.close()


def test_int8_quant_is_close():
    w = np.random.default_rng(1).standard_normal((32, 16)).astype(np.float32)
    x = np.random.default_rng(2).standard_normal((3, 16)).astype(np.float32)
    ref = x @ w.T
    assert np.abs(QuantMatrix(w).matmul_t(x) - ref).max() < 0.02 * np.abs(ref).max()


# --- the model ---------------------------------------------------------------

def test_kv_cache_matches_full_recompute(tiny):
    m = MoEModel(tiny)
    toks = [256, 72, 101, 108, 108, 111, 33]
    c = m.new_cache()
    m.forward(toks[:3], c)
    for t in toks[3:-1]:
        m.forward([t], c)
    incremental = m.forward(toks[-1:], c)
    full = m.forward(toks, m.new_cache())
    np.testing.assert_allclose(incremental, full, rtol=1e-4, atol=1e-4)


def test_rms_norm():
    x = np.array([[3.0, 4.0]], np.float32)
    np.testing.assert_allclose(rms_norm(x, np.ones(2, np.float32), 0.0), x / np.sqrt(12.5), rtol=1e-6)


# --- distribution ------------------------------------------------------------

def test_distributed_matches_local_and_pushes_experts(tiny, tmp_path):
    cfg = load_config(tiny)
    prompt = [256] + list(b"The quick brown fox")
    ref = greedy(MoEModel(tiny), prompt)
    w1, w2 = Worker(tmp_path, "w1", 2**20), Worker(tmp_path, "w2", 2**20)  # neither has the model
    try:
        backend = DistributedBackend(str(tiny), cfg, [w1.address, w2.address], replicate=False,
                                     progress=lambda m: None)
        assert not backend.plan.unplaced
        model = MoEModel(tiny, backend=backend)
        assert greedy(model, prompt) == ref
        stats = {r["node"]: r for r in backend.stats()}
        assert stats["w1"]["requests"] > 0 and stats["w2"]["requests"] > 0
        assert backend.local.misses == 0  # nothing fell back to the coordinator's disk
        backend.close()

        # second session: the experts are in the workers' caches, nothing to push
        sent = []
        backend = DistributedBackend(str(tiny), cfg, [w1.address, w2.address], replicate=False,
                                     progress=sent.append)
        assert not any("sending" in s for s in sent)
        assert greedy(MoEModel(tiny, backend=backend), prompt) == ref
        backend.close()
    finally:
        w1.kill(), w2.kill()


def test_worker_with_local_model_copy_needs_no_transfer(tiny, tmp_path):
    w = Worker(tmp_path, "local-copy", 2**22, model_dir=str(tiny))
    try:
        sent = []
        backend = DistributedBackend(str(tiny), load_config(tiny), [w.address], progress=sent.append)
        assert not any("sending" in s for s in sent)
        backend.close()
    finally:
        w.kill()


def test_partial_cluster_streams_rest_from_disk(tiny, tmp_path):
    cfg = load_config(tiny)
    prompt = [256] + list(b"abc")
    ref = greedy(MoEModel(tiny), prompt, 6)
    w = Worker(tmp_path, "small", 5 * 4 * 3 * 64 * 96)  # room for 5 experts only
    try:
        backend = DistributedBackend(str(tiny), cfg, [w.address], local_budget=0,
                                     progress=lambda m: None)
        assert len(backend.plan.per_node["small"]) == 5 and backend.plan.unplaced
        assert greedy(MoEModel(tiny, backend=backend), prompt, 6) == ref
        backend.close()
    finally:
        w.kill()


def test_failover_when_a_worker_dies(tiny, tmp_path):
    cfg = load_config(tiny)
    prompt = [256] + list(b"failover")
    ref = greedy(MoEModel(tiny), prompt, 10)
    w1, w2 = Worker(tmp_path, "f1", 2**20), Worker(tmp_path, "f2", 2**20)
    try:
        backend = DistributedBackend(str(tiny), cfg, [w1.address, w2.address], replicate=False,
                                     progress=lambda m: None)
        model = MoEModel(tiny, backend=backend)
        out = []
        for i, t in enumerate(model.generate(prompt, max_new_tokens=10, stop_ids=[])):
            out.append(t)
            if i == 3:
                w2.kill()
        assert out == ref
        assert not backend.conns["f2"].alive
        backend.close()
    finally:
        w1.kill(), w2.kill()


def test_replicas_take_over_and_f16_wire_stays_close(tiny, tmp_path):
    cfg = load_config(tiny)
    m_ref = MoEModel(tiny)
    toks = [256] + list(b"replica")
    ref = m_ref.forward(toks, m_ref.new_cache())
    w1, w2 = Worker(tmp_path, "r1", 2**22), Worker(tmp_path, "r2", 2**22)
    try:
        backend = DistributedBackend(str(tiny), cfg, [w1.address, w2.address], wire="f16",
                                     progress=lambda m: None)
        assert all(len(o) == 2 for o in backend.plan.owners.values())
        model = MoEModel(tiny, backend=backend)
        np.testing.assert_allclose(model.forward(toks, model.new_cache()), ref, atol=0.05)
        w1.kill()
        np.testing.assert_allclose(model.forward(toks, model.new_cache()), ref, atol=0.05)
        assert backend.local.misses == 0  # the replica on r2 served r1's experts
        backend.close()
    finally:
        w1.kill(), w2.kill()


def test_plan_uses_coordinator_ram(tiny):
    cfg = load_config(tiny)
    plan = make_plan(cfg.moe_layers(), cfg.num_experts, 1, {LOCAL: 10**9})
    assert len(plan.per_node[LOCAL]) == cfg.num_layers * cfg.num_experts


# --- colibri int4 containers and the kernels -----------------------------------

from moelan.experts import Packed4
from moelan.model import load_heat


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    """The same int4 expert values as a colibri container and as an f32 HF checkpoint."""
    kw = dict(layers=3, experts=8, top_k=2, int4_gs=32, seed=7)
    col = make_tiny(tmp_path_factory.mktemp("col"), layout="colibri_int4", **kw)
    hf = make_tiny(tmp_path_factory.mktemp("hf"), **kw)
    return col, hf


@pytest.mark.parametrize("rows", [1, 3, 4, 17])
def test_int4_matmul_matches_reference(rows):
    rng = np.random.default_rng(rows)
    O, I, gs = 96, 128, 32
    v = rng.integers(-8, 8, (O, I))
    s = rng.random((O, I // gs)).astype(np.float32)
    nib = (v & 15).astype(np.uint8)
    raw = nib[:, 0::2] | (nib[:, 1::2] << 4)
    W = (v.reshape(O, -1, gs) * s[:, :, None]).reshape(O, I).astype(np.float32)
    x = rng.standard_normal((rows, I)).astype(np.float32)
    p = Packed4(raw, s, gs, (O, I))
    for m in (p, p.resident(), p.to_int8()):
        np.testing.assert_allclose(m.matmul_t(x), x @ W.T, atol=1e-3)
    np.testing.assert_allclose(p.dequant(), W, atol=1e-6)


def test_colibri_container_matches_hf_twin(twins):
    col, hf = twins
    assert load_config(col).expert_layout == "colibri_int4"
    prompt = [256] + list(b"colibri")
    m_hf = MoEModel(hf)
    ref = m_hf.forward(prompt, m_hf.new_cache())
    for fmt in ("int4", "int8", "f32"):
        m = MoEModel(col, expert_format=fmt, pin_budget=10**9)
        np.testing.assert_allclose(m.forward(prompt, m.new_cache()), ref, atol=1e-3)
    m = MoEModel(col)  # nothing pinned: computed from the memory map
    assert greedy(m, prompt) == greedy(m_hf, prompt)


def test_distributed_colibri_container(twins, tmp_path):
    col, _ = twins
    prompt = [256] + list(b"int4 over the wire")
    ref = greedy(MoEModel(col), prompt)
    w1, w2 = Worker(tmp_path, "c1", 2**22), Worker(tmp_path, "c2", 2**22)
    try:
        for fmt in ("int4", "int8"):
            backend = DistributedBackend(str(col), load_config(col), [w1.address, w2.address],
                                         fmt=fmt, replicate=False, progress=lambda m: None)
            assert greedy(MoEModel(col, backend=backend), prompt) == ref
            assert backend.local.misses == 0
            backend.close()
    finally:
        w1.kill(), w2.kill()


def test_int8_dense_is_close(tiny):
    toks = [256] + list(b"dense")
    a = MoEModel(tiny)
    b = MoEModel(tiny, dense_format="int8")
    la, lb = a.forward(toks, a.new_cache()), b.forward(toks, b.new_cache())
    assert np.abs(la - lb).max() < 0.05 * np.abs(la).max()


def test_load_colibri_usage(tmp_path):
    p = tmp_path / ".coli_usage"
    p.write_text("-1 48 128\n-2 1 999\n0 5 17\n3 2 4\n")
    assert load_heat(p) == {(0, 5): 17, (3, 2): 4}


# --- monitoring dashboard ---------------------------------------------------------

import json as _json
import time as _time
import urllib.request

from moelan.server import Telemetry, WorkerMonitor, make_server


def _http(server, path, body=None):
    url = f"http://127.0.0.1:{server.server_address[1]}{path}"
    req = urllib.request.Request(url, data=None if body is None else _json.dumps(body).encode(),
                                 method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def _serve(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return t


def test_telemetry_tracks_requests():
    tel = Telemetry()
    tel.begin("r1", 5)
    tel.begin("r2", 3)
    snap = tel.snapshot()
    assert snap["busy"] and snap["running"] == 2
    assert [a["phase"] for a in snap["active"]] == ["waiting", "waiting"]
    tel.admit("r1")
    assert tel.snapshot()["active"][0]["phase"] == "prefill"
    for _ in range(4):
        tel.token("r1")
    assert tel.snapshot()["active"][0]["phase"] == "decode"
    tel.end("r1")
    snap = tel.snapshot()
    assert snap["running"] == 1 and snap["tokens_out"] == 4 and snap["tokens_in"] == 8
    assert snap["recent"][0]["completion_tokens"] == 4 and snap["rate_tok_s"] > 0
    tel.end("r2")
    assert not tel.snapshot()["busy"]


def test_dashboard_and_status_api(tiny, tmp_path):
    from moelan.tokenizer import Tokenizer
    cfg = load_config(tiny)
    w = Worker(tmp_path, "dash", 2**20)
    try:
        backend = DistributedBackend(str(tiny), cfg, [w.address], progress=lambda m: None)
        server = make_server(MoEModel(tiny, backend=backend), Tokenizer(tiny), "tiny", port=0)
        _serve(server)
        code, ctype, body = _http(server, "/")
        assert code == 200 and ctype.startswith("text/html") and b"moelan" in body
        code, _, body = _http(server, "/v1/completions", {"prompt": "hi", "max_tokens": 5})
        assert code == 200
        status = _json.loads(_http(server, "/api/status")[2])
        assert status["mode"] == "serve" and status["model"]["backend"] == "distributed"
        assert status["gen"]["requests"] == 1 and status["gen"]["tokens_out"] == 5
        node = next(n for n in status["nodes"] if n["node"] == "dash")
        assert node["alive"] and node["requests"] > 0 and node["budget_bytes"] == 2**20
        heat = _json.loads(_http(server, "/api/heat")[2])
        assert heat["layers"] == cfg.num_layers and heat["nodes"][0] == "local" and "dash" in heat["nodes"]
        assert sum(map(sum, heat["counts"])) > 0
        assert all(len(v) == cfg.top_k for v in heat["last"].values())
        server.shutdown()
        backend.close()
    finally:
        w.kill()


def test_monitor_mode_sees_workers_go_down(tmp_path):
    w = Worker(tmp_path, "mon", 2**20)
    mon = WorkerMonitor([w.address], interval=0.1)
    mon.start()
    server = make_server(None, None, "monitor", port=0, monitor=mon)
    _serve(server)
    try:
        deadline = _time.time() + 10
        while _time.time() < deadline and not mon.snapshot():
            _time.sleep(0.05)
        status = _json.loads(_http(server, "/api/status")[2])
        assert status["mode"] == "monitor" and status["nodes"][0]["alive"]
        assert _http(server, "/v1/models")[0] == 404
        w.kill()
        deadline = _time.time() + 10
        while _time.time() < deadline and mon.snapshot()[0]["alive"]:
            _time.sleep(0.05)
        assert not mon.snapshot()[0]["alive"]
    finally:
        server.shutdown()


# --- DeepSeek V2/V3: MLA, shared experts, grouped routing ----------------------------

import math as _math

from moelan.safetensors_io import SafeTensorDir
from moelan.tiny import make_tiny_deepseek


def _naive_deepseek_logits(model_dir, tokens):
    """Straight transcription of HF modeling_deepseek (V2/V3) for a whole
    sequence: expanded per-head keys/values, no cache, a per-token router."""
    cfg = load_config(model_dir)
    raw = cfg.raw
    st = SafeTensorDir(model_dir)
    g = st.get
    H, nope, rd, vd, r = cfg.num_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    eps = cfg.rms_norm_eps

    def norm(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

    def silu(x):
        return x / (1 + np.exp(-x))

    def mlp(x, p):
        return (silu(x @ g(p + "gate_proj.weight").T) * (x @ g(p + "up_proj.weight").T)) @ g(p + "down_proj.weight").T

    # rotary tables as in DeepseekV2YarnRotaryEmbedding
    rs, base, dim = raw.get("rope_scaling"), raw["rope_theta"], rd
    freq_extra = 1.0 / base ** (np.arange(0, dim, 2) / dim)
    inv, msc = freq_extra, 1.0
    scale = (nope + rd) ** -0.5
    if rs:
        f, orig = rs["factor"], rs["original_max_position_embeddings"]
        freq_inter = 1.0 / (f * base ** (np.arange(0, dim, 2) / dim))
        cd = lambda rot: dim * _math.log(orig / (rot * 2 * _math.pi)) / (2 * _math.log(base))
        lo, hi = max(_math.floor(cd(rs["beta_fast"])), 0), min(_math.ceil(cd(rs["beta_slow"])), dim - 1)
        ramp = np.clip((np.arange(dim // 2) - lo) / ((hi + 0.001 if hi == lo else hi) - lo), 0, 1)
        mask = 1 - ramp
        inv = freq_inter * (1 - mask) + freq_extra * mask
        ms = lambda s, m: 1.0 if s <= 1 else 0.1 * m * _math.log(s) + 1.0
        msc = ms(f, rs["mscale"]) / ms(f, rs["mscale_all_dim"])
        scale *= ms(f, rs["mscale_all_dim"]) ** 2
    T = len(tokens)
    emb = np.concatenate([np.outer(np.arange(T), inv)] * 2, -1)
    cos, sin = np.cos(emb) * msc, np.sin(emb) * msc

    def rope(x):  # x (T, h, d): de-interleave, then rotate_half
        x = x.reshape(*x.shape[:-1], dim // 2, 2).swapaxes(-1, -2).reshape(x.shape)
        rot = np.concatenate([-x[..., dim // 2:], x[..., :dim // 2]], -1)
        return x * cos[:, None] + rot * sin[:, None]

    x = g("model.embed_tokens.weight")[tokens]
    for i in range(cfg.num_layers):
        p = f"model.layers.{i}."
        a = p + "self_attn."
        h = norm(x, g(p + "input_layernorm.weight"))
        if cfg.q_lora_rank:
            q = norm(h @ g(a + "q_a_proj.weight").T, g(a + "q_a_layernorm.weight")) @ g(a + "q_b_proj.weight").T
        else:
            q = h @ g(a + "q_proj.weight").T
        q = q.reshape(T, H, nope + rd)
        ckv = h @ g(a + "kv_a_proj_with_mqa.weight").T
        kv = (norm(ckv[:, :r], g(a + "kv_a_layernorm.weight")) @ g(a + "kv_b_proj.weight").T).reshape(T, H, nope + vd)
        k_pe = rope(ckv[:, None, r:])
        Q = np.concatenate([q[..., :nope], rope(q[..., nope:])], -1)
        K = np.concatenate([kv[..., :nope], np.repeat(k_pe, H, 1)], -1)
        s = np.einsum("thd,shd->hts", Q, K) * scale
        s = np.where(np.triu(np.ones((T, T), bool), 1)[None], -np.inf, s)
        pr = np.exp(s - s.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", pr, kv[..., nope:]).reshape(T, H * vd)
        x = x + o @ g(a + "o_proj.weight").T
        h = norm(x, g(p + "post_attention_layernorm.weight"))
        if not cfg.is_moe_layer(i):
            x = x + mlp(h, p + "mlp.")
            continue
        logits = h @ g(p + "mlp.gate.weight").T
        y = mlp(h, p + "mlp.shared_experts.")
        E, G = cfg.num_experts, cfg.n_group
        for t in range(T):
            if cfg.scoring_func == "sigmoid":
                sc = 1 / (1 + np.exp(-logits[t]))
                ch = sc + g(p + "mlp.gate.e_score_correction_bias")
                gsc = np.array([np.sort(c)[-2:].sum() for c in ch.reshape(G, -1)])
            else:
                sc = np.exp(logits[t] - logits[t].max())
                sc /= sc.sum()
                ch = sc
                gsc = ch.reshape(G, -1).max(-1)
            keep = np.argsort(-gsc)[:cfg.topk_group]
            ch = np.where(np.isin(np.arange(E) // (E // G), keep), ch, 0.0)
            top = np.argsort(-ch)[:cfg.top_k]
            wt = sc[top]
            if cfg.norm_topk_prob:
                wt = wt / (wt.sum() + 1e-20)
            if cfg.scoring_func == "sigmoid" or not cfg.norm_topk_prob:
                wt = wt * cfg.routed_scaling_factor
            for e, ww in zip(top, wt):
                y[t] += ww * mlp(h[t:t + 1], p + f"mlp.experts.{e}.")[0]
        x = x + y
    return norm(x, g("model.norm.weight")) @ g("lm_head.weight").T


@pytest.fixture(scope="module", params=[(3, 48, True), (2, None, True), (3, None, False)],
                ids=["v3-qlora-yarn", "v2-yarn", "v3-plain-rope"])
def deepseek(request, tmp_path_factory):
    v, ql, yarn = request.param
    return make_tiny_deepseek(tmp_path_factory.mktemp(f"ds{v}"), version=v, q_lora_rank=ql, yarn=yarn, seed=v)


def test_deepseek_config(deepseek):
    cfg = load_config(deepseek)
    assert cfg.attention == "mla" and cfg.dense_layers == [0] and cfg.moe_layers() == [1, 2]
    assert cfg.kv_cache_dims() == (1, cfg.kv_lora_rank, cfg.qk_rope_head_dim)
    assert cfg.shared_expert_intermediate_size == cfg.moe_intermediate_size


def test_deepseek_matches_naive_reference(deepseek):
    toks = [256] + list(b"DeepSeek MLA")
    ref = _naive_deepseek_logits(deepseek, toks)
    m = MoEModel(deepseek)
    np.testing.assert_allclose(m.forward(toks, m.new_cache()), ref[-1], atol=2e-4, rtol=1e-4)
    c = m.new_cache()            # latent KV cache, one token at a time
    m.forward(toks[:4], c)
    for t in toks[4:-1]:
        m.forward([t], c)
    np.testing.assert_allclose(m.forward(toks[-1:], c), ref[-1], atol=2e-4, rtol=1e-4)


def test_deepseek_distributed(deepseek, tmp_path):
    prompt = [256] + list(b"experts over the LAN")
    ref = greedy(MoEModel(deepseek), prompt)
    w1, w2 = Worker(tmp_path, "d1", 2**22), Worker(tmp_path, "d2", 2**22)
    try:
        backend = DistributedBackend(str(deepseek), load_config(deepseek), [w1.address, w2.address],
                                     replicate=False, progress=lambda m: None)
        assert set(k[0] for k in backend.plan.owners) == {1, 2}  # dense layer 0 has no experts
        assert greedy(MoEModel(deepseek, backend=backend), prompt) == ref
        assert backend.local.misses == 0
        backend.close()
    finally:
        w1.kill(), w2.kill()


# --- native 16-bit dense weights and the fused expert layer ------------------------

from moelan import kernels as _kernels
from moelan.experts import Half16


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", [1, 5, 40])
def test_half16_matmul_is_exact(bf16, rows):
    rng = np.random.default_rng(rows)
    w = rng.standard_normal((70, 96)).astype(np.float32)
    if bf16:
        from moelan.safetensors_io import f32_to_bf16
        bits = f32_to_bf16(w)
    else:
        w[0, :4] = [6e-8, -3e-6, 0.0, -0.0]          # half subnormals and signed zeros
        bits = w.astype(np.float16).view(np.uint16)
    m = Half16(bits, bf16)
    W = m.dequant()
    x = rng.standard_normal((rows, 96)).astype(np.float32)
    np.testing.assert_allclose(m.matmul_t(x), x @ W.T, rtol=1e-5, atol=1e-5)
    if not bf16:
        np.testing.assert_array_equal(W, bits.view(np.float16).astype(np.float32))


def test_dense_formats_agree(tiny):
    toks = [256] + list(b"formats")
    ref = None
    for fmt in ("f32", "f16"):
        m = MoEModel(tiny, dense_format=fmt)
        out = m.forward(toks, m.new_cache())
        if ref is None:
            ref = out
        np.testing.assert_allclose(out, ref, atol=1e-4)  # bf16-stored weights: f16 path is exact


# --- batched decoding --------------------------------------------------------------

from moelan.batching import Batcher, GenRequest

_PROMPTS = [[256] + list(b"one"), [256] + list(b"a longer second prompt"), [256, 72], [256] + list(b"fourth!")]


def _separate_logits(model, prompts, steps):
    """Each prompt alone: prefill, then ``steps`` greedy single-token steps."""
    out = []
    for p in prompts:
        c = model.new_cache()
        lg = [model.forward(p, c)]
        for _ in range(steps):
            lg.append(model.forward([int(np.argmax(lg[-1]))], c))
        out.append(lg)
    return out


def _batched_logits(model, prompts, steps):
    caches = [model.new_cache() for _ in prompts]
    lg = [list(model.forward_batch(prompts, caches))]
    for _ in range(steps):
        lg.append(list(model.forward_batch([[int(np.argmax(v))] for v in lg[-1]], caches)))
    return [[lg[t][i] for t in range(steps + 1)] for i in range(len(prompts))]


def test_forward_batch_matches_separate(tiny):
    m = MoEModel(tiny)
    for sep, bat in zip(_separate_logits(m, _PROMPTS, 4), _batched_logits(m, _PROMPTS, 4)):
        for a, b in zip(sep, bat):
            np.testing.assert_allclose(b, a, atol=1e-4, rtol=1e-4)


def test_forward_batch_matches_separate_deepseek(deepseek):
    m = MoEModel(deepseek)
    for sep, bat in zip(_separate_logits(m, _PROMPTS, 3), _batched_logits(m, _PROMPTS, 3)):
        for a, b in zip(sep, bat):
            np.testing.assert_allclose(b, a, atol=2e-4, rtol=1e-4)


def test_forward_batch_distributed_one_request_per_layer(tiny, tmp_path):
    w1, w2 = Worker(tmp_path, "b1", 2**20), Worker(tmp_path, "b2", 2**20)
    try:
        backend = DistributedBackend(str(tiny), load_config(tiny), [w1.address, w2.address],
                                     replicate=False, progress=lambda m: None)
        m = MoEModel(tiny, backend=backend)
        ref = _separate_logits(MoEModel(tiny), _PROMPTS, 2)
        caches = [m.new_cache() for _ in _PROMPTS]
        m.forward_batch(_PROMPTS, caches)
        before = sum(c.requests for c in backend.conns.values())
        step = m.forward_batch([[int(np.argmax(r[0]))] for r in ref], caches)
        sent = sum(c.requests for c in backend.conns.values()) - before
        assert sent <= 2 * len(m.cfg.moe_layers())   # at most one per worker per layer, whole batch
        for i, r in enumerate(ref):
            np.testing.assert_allclose(step[i], r[1], atol=1e-4, rtol=1e-4)
        backend.close()
    finally:
        w1.kill(), w2.kill()


def test_batcher_gives_same_text_as_one_at_a_time(tiny):
    m = MoEModel(tiny)
    want = [list(m.generate(p, max_new_tokens=12, stop_ids=[])) for p in _PROMPTS]
    b = Batcher(m, max_batch=8)
    reqs = [GenRequest(f"r{i}", p, max_new_tokens=12) for i, p in enumerate(_PROMPTS)]
    for r in reqs:
        b.submit(r)            # all queued before the scheduler starts: one batch
    b.start()
    got = [list(r.tokens()) for r in reqs]
    assert got == want
    st = b.stats()
    assert st["avg_batch"] > 3 and st["running"] == 0


def test_server_batches_concurrent_requests(tiny):
    from concurrent.futures import ThreadPoolExecutor
    from moelan.tokenizer import Tokenizer
    m = MoEModel(tiny)
    server = make_server(m, Tokenizer(tiny), "tiny", port=0, max_batch=4)
    _serve(server)
    try:
        prompts = ["alpha", "beta beta", "gamma", "delta delta delta"]

        def ask(p):
            return _json.loads(_http(server, "/v1/completions", {"prompt": p, "max_tokens": 16})[2])

        alone = [ask(p)["choices"][0]["text"] for p in prompts]
        with ThreadPoolExecutor(4) as ex:
            together = [r["choices"][0]["text"] for r in ex.map(ask, prompts)]
        assert together == alone
        status = _json.loads(_http(server, "/api/status")[2])
        assert status["batch"]["max_batch"] == 4 and status["gen"]["requests"] == 8
        assert status["batch"]["avg_batch"] > 1  # the concurrent ones shared steps
    finally:
        server.shutdown()


# --- tool calling -----------------------------------------------------------------

from moelan.tools import normalize_messages, parse_tool_calls, safe_prefix_len

json = _json

WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
        "where": {"type": "object"}}, "required": ["city"]}}}]


def test_parse_qwen3_coder_xml_with_schema_types():
    text = ("Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n"
            "<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n"
            "<parameter=where>\n{\"lat\": 40.7}\n</parameter>\n</function>\n</tool_call>")
    content, calls = parse_tool_calls(text, WEATHER)
    assert content == "Let me check." and len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "get_weather" and calls[0]["type"] == "function" and calls[0]["id"].startswith("call_")
    assert json.loads(fn["arguments"]) == {"city": "New York", "days": 3, "metric": True, "where": {"lat": 40.7}}


def test_parse_hermes_json_and_multiple_calls():
    text = ('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": "{\\"city\\": \\"Rome\\"}"}\n</tool_call>')
    content, calls = parse_tool_calls(text, WEATHER)
    assert content is None
    assert [json.loads(c["function"]["arguments"])["city"] for c in calls] == ["Oslo", "Rome"]


def test_parse_deepseek_v3_format():
    text = ("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
            "```json\n{\"city\": \"Lima\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
    _, calls = parse_tool_calls(text, WEATHER)
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Lima"}


def test_plain_text_and_broken_calls_stay_text():
    assert parse_tool_calls("Just an answer.", WEATHER) == ("Just an answer.", [])
    broken = "<tool_call>\nnot json and no function\n</tool_call>"
    assert parse_tool_calls(broken, WEATHER) == (broken, [])


def test_streaming_never_leaks_a_partial_tool_tag():
    assert safe_prefix_len("Hello <tool_c") == len("Hello ")
    assert safe_prefix_len("Hello <tool_call>\n<function=x>") == len("Hello ")
    assert safe_prefix_len("a < b") == len("a < b")
    assert safe_prefix_len("price <") == len("price ")


def test_normalize_messages_for_templates():
    msgs = normalize_messages([{"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert msgs[0]["content"] == "" and msgs[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    assert msgs[1]["content"] == "hi"


class _ScriptedBatcher:
    """Stands in for the scheduler: answers every request with fixed text."""

    def __init__(self, text, tok):
        self.ids = tok.encode(text)

    def submit(self, gen):
        for t in self.ids:
            gen.out.put(t)
        gen.out.put(None)

    def stats(self):
        return {}


def _tool_server(tiny, text):
    from moelan.tokenizer import Tokenizer
    tok = Tokenizer(tiny)
    server = make_server(MoEModel(tiny), tok, "tiny", port=0)
    server.RequestHandlerClass.batcher = _ScriptedBatcher(text, tok)
    _serve(server)
    return server


def test_server_returns_tool_calls(tiny):
    reply = "Checking.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    server = _tool_server(tiny, reply)
    try:
        body = {"messages": [{"role": "user", "content": "weather in Paris?"}], "tools": WEATHER}
        r = _json.loads(_http(server, "/v1/chat/completions", body)[2])
        ch = r["choices"][0]
        assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] == "Checking."
        assert _json.loads(ch["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
        # without tools the same text is returned as plain content
        r = _json.loads(_http(server, "/v1/chat/completions", {"messages": body["messages"]})[2])
        assert r["choices"][0]["finish_reason"] == "stop" and "<tool_call>" in r["choices"][0]["message"]["content"]
    finally:
        server.shutdown()


def test_server_streams_tool_calls(tiny):
    reply = "Sure <tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>"
    server = _tool_server(tiny, reply)
    try:
        body = {"messages": [{"role": "user", "content": "x"}], "tools": WEATHER, "stream": True}
        raw = _http(server, "/v1/chat/completions", body)[2].decode()
        events = [_json.loads(l[6:]) for l in raw.split("\n\n") if l.startswith("data: {")]
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        assert text == "Sure " and "<tool" not in text
        calls = [c for e in events for c in e["choices"][0]["delta"].get("tool_calls", [])]
        assert calls[0]["index"] == 0 and _json.loads(calls[0]["function"]["arguments"]) == {"city": "Rome"}
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    finally:
        server.shutdown()


def test_prompt_cache_reuses_shared_prefix(tiny):
    m = MoEModel(tiny)
    base = [256] + list(b"You are a helpful assistant with many tools. ")
    p1, p2 = base + list(b"First question?"), base + list(b"Second one!")
    want = list(m.generate(p2, max_new_tokens=8, stop_ids=[]))
    b = Batcher(m, max_batch=1, prompt_cache_tokens=4096)
    b.start()
    r1 = b.submit(GenRequest("a", p1, max_new_tokens=8))
    list(r1.tokens())
    r2 = b.submit(GenRequest("b", p2, max_new_tokens=8))
    assert list(r2.tokens()) == want            # same output as computing everything
    assert r1.cached == 0 and r2.cached == len(base)
    st = b.stats()["prompt_cache"]
    assert st["hits"] == 1 and st["tokens_saved"] == len(base)


# --- hybrid GPU / CPU experts ---------------------------------------------------

from moelan.accel import choose_gpu_experts


def test_choose_gpu_experts_takes_hot_share_per_layer():
    cands = [(l, e) for l in (0, 1) for e in range(4)]
    heat = {(0, 0): 70, (0, 1): 20, (0, 2): 5, (0, 3): 5, (1, 3): 50, (1, 2): 50}
    got = choose_gpu_experts(cands, heat, share=0.6)
    assert set(got) == {(0, 0), (1, 3), (1, 2)} or set(got) == {(0, 0), (1, 2), (1, 3)}
    assert got[:2] in ([(0, 0), (1, 2)], [(0, 0), (1, 3)])          # layers interleaved
    assert len(choose_gpu_experts(cands, None, share=0.5)) == 4        # no heat: by count


def test_gpu_tier_matches_cpu(tiny):
    pytest.importorskip("torch")
    toks = [256] + list(b"hybrid experts")
    cpu = MoEModel(tiny)
    ref = cpu.forward(toks, cpu.new_cache())
    hyb = MoEModel(tiny, gpu="torch-cpu", gpu_budget=2**30, gpu_mode="max")
    tier = hyb.backend.store.gpu
    assert tier is not None and len(tier.experts) == len(hyb.cfg.moe_layers()) * hyb.cfg.num_experts
    np.testing.assert_allclose(hyb.forward(toks, hyb.new_cache()), ref, atol=1e-4, rtol=1e-4)
    assert greedy(hyb, toks, 8) == greedy(cpu, toks, 8)
    assert tier.calls > 0 and 0 < tier.share < 1


def test_gpu_tier_auto_and_budget(tiny):
    pytest.importorskip("torch")
    m = MoEModel(tiny, gpu="torch-cpu", gpu_budget=3 * 4 * 3 * 64 * 96, gpu_mode="max")
    assert len(m.backend.store.gpu.experts) == 3          # budget for exactly three experts
    a = MoEModel(tiny, gpu="torch-cpu", gpu_budget=2**30, gpu_mode="auto")
    tier = a.backend.store.gpu
    n = len(tier.experts) if tier else 0
    total = len(a.cfg.moe_layers()) * a.cfg.num_experts
    assert n == 0 or abs(n / total - tier.share) < 0.2   # auto: about its measured share


def test_worker_hybrid_gpu(tiny, tmp_path):
    pytest.importorskip("torch")
    prompt = [256] + list(b"worker on a gpu")
    ref = greedy(MoEModel(tiny), prompt)
    w = Worker(tmp_path, "g1", 2**22, gpu="torch-cpu", gpu_budget=2**30, gpu_mode="max")
    try:
        backend = DistributedBackend(str(tiny), load_config(tiny), [w.address], progress=lambda m: None)
        assert greedy(MoEModel(tiny, backend=backend), prompt) == ref
        row = next(r for r in backend.stats() if r["node"] == "g1")
        assert row["gpu"]["experts"] > 0 and row["gpu"]["device"].startswith("torch-cpu")
        backend.close()
    finally:
        w.kill()


# --- long prompts --------------------------------------------------------------------

def test_chunked_prefill_matches_one_pass(tiny, monkeypatch):
    m = MoEModel(tiny)
    toks = [256] + list(np.random.default_rng(3).integers(0, 256, 299))
    one = m.forward(toks, m.new_cache())                    # 300 rows: threaded attention blocks
    c = m.new_cache()
    for t in toks[:-1]:                                     # serial path, one token at a time
        m.forward([t], c)
    np.testing.assert_allclose(m.forward(toks[-1:], c), one, atol=1e-4, rtol=1e-4)
    monkeypatch.setattr(MoEModel, "PREFILL_CHUNK", 37)
    seen = []
    c = m.new_cache()
    chunked = m.forward_batch([toks], [c], on_progress=seen.append)[0]
    np.testing.assert_allclose(chunked, one, atol=1e-4, rtol=1e-4)
    reads = [d[0] for d in seen]                           # reported after every layer
    assert reads == sorted(reads) and reads[-1] == 300 and c.length == 300
    assert {37, 74} <= set(reads)                          # whole passes show up exactly


def test_abandoned_prefill_is_kept_for_the_retry(tiny, monkeypatch):
    monkeypatch.setattr(MoEModel, "PREFILL_CHUNK", 16)
    m = MoEModel(tiny)
    toks = [256] + list(np.random.default_rng(4).integers(0, 256, 99))
    first = GenRequest("a", toks, max_new_tokens=4)

    class Tel:   # the client goes away once two passes (32 tokens) are read
        def admit(self, *a, **k): pass
        def token(self, *a): pass
        def end(self, *a): pass
        def progress(self, rid, read):
            if read >= 32:
                first.cancelled = True
    b = Batcher(m, Tel(), max_batch=1)
    b._prefill([first])
    assert first.done and first.cache is None          # stopped within a layer, not after the prompt
    retry = GenRequest("b", toks, max_new_tokens=4)
    b._prefill([retry])
    assert 16 <= retry.cached < len(toks)              # resumed from the passes completed before the stop
    b._finish(retry)


def test_prompt_cache_survives_restart_on_disk(tiny, tmp_path):
    from moelan.batching import PromptCache
    m = MoEModel(tiny)
    toks = [256] + list(np.random.default_rng(5).integers(0, 256, 63))
    c = m.new_cache()
    m.forward(toks, c)
    pc = PromptCache(4096, disk_dir=tmp_path, disk_max_bytes=2**30, new_cache=m.new_cache, disk_min=8)
    pc.store(toks, c)
    pc._writer.shutdown(wait=True)
    fresh = PromptCache(4096, disk_dir=tmp_path, disk_max_bytes=2**30, new_cache=m.new_cache, disk_min=8)
    assert len(fresh.disk) == 1 and not fresh.entries          # nothing in memory: a restart
    longer = toks + [65, 66, 67]
    cache, hit = fresh.lookup(longer)
    assert hit == len(toks) and fresh.disk_hits == 1
    want = m.forward(longer, m.new_cache())
    np.testing.assert_allclose(m.forward(longer[hit:], cache), want, atol=1e-5)


# --- robustness against clients that hang up ------------------------------------

import struct as _struct


def _rst(sock):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, _struct.pack("ii", 1, 0))  # close = RST
    sock.close()


def _post(path, body):
    data = _json.dumps(body).encode()
    return (f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(data)}\r\n\r\n").encode() + data


def test_server_survives_resets_and_cancels_abandoned_work(tiny):
    from moelan.tokenizer import Tokenizer
    server = make_server(MoEModel(tiny), Tokenizer(tiny), "tiny", port=0)
    _serve(server)
    port = server.server_address[1]
    try:
        acts = [lambda s: None,
                lambda s: s.sendall(b"POST /v1/comp"),
                lambda s: s.sendall(_post("/v1/completions", {"prompt": "x"})[:60]),
                lambda s: (s.sendall(_post("/v1/completions", {"prompt": "hi", "max_tokens": 50, "stream": True})),
                           s.recv(100)),
                lambda s: (s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n"), s.recv(4096))]
        for act in acts:
            s = socket.create_connection(("127.0.0.1", port))
            act(s)
            _rst(s)
            assert _http(server, "/health")[0] == 200              # still serving
        # a non-streamed request whose client hangs up is cancelled, not run to the end
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(_post("/v1/completions", {"prompt": "go on", "max_tokens": 100000}))
        _time.sleep(0.3)
        _rst(s)
        deadline = _time.time() + 15
        while _time.time() < deadline and _json.loads(_http(server, "/api/status")[2])["gen"]["busy"]:
            _time.sleep(0.2)
        status = _json.loads(_http(server, "/api/status")[2])
        assert not status["gen"]["busy"]
        assert status["gen"]["recent"][0]["completion_tokens"] < 100000
        assert _http(server, "/v1/completions", {"prompt": "after", "max_tokens": 3})[0] == 200
    finally:
        server.shutdown()


def test_scheduler_survives_a_failing_step(tiny, monkeypatch):
    from moelan.tokenizer import Tokenizer
    m = MoEModel(tiny)
    server = make_server(m, Tokenizer(tiny), "tiny", port=0)
    _serve(server)
    real = m.forward_batch
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise MemoryError("simulated")
        return real(*a, **k)
    monkeypatch.setattr(m, "forward_batch", flaky)
    try:
        assert _http(server, "/v1/completions", {"prompt": "a", "max_tokens": 3})[0] == 500
        assert _http(server, "/v1/completions", {"prompt": "b", "max_tokens": 3})[0] == 200
    finally:
        server.shutdown()


# --- wire formats ---------------------------------------------------------------------

from moelan.protocol import decode_rows, encode_rows


def test_wire_encodings_size_and_error():
    a = np.random.default_rng(6).standard_normal((5, 2048)).astype(np.float32) * 3
    a[:, 7] *= 40                                                    # an outlier channel, as in real activations
    for wire, ratio, tol in (("f32", 1.0, 0.0), ("f16", 0.5, 1e-3), ("int8", 0.27, 6e-3)):
        enc = encode_rows(a, wire, "x")
        size = sum(v.nbytes for v in enc.values())
        assert size <= a.nbytes * ratio                             # int8: + one f32 scale per 64 values
        err = np.abs(decode_rows(enc, "x") - a)[:, 64:].max(1) / np.abs(a[:, 64:]).max(1)
        assert err.max() <= tol                                     # the outlier does not spoil other groups
    z = encode_rows(np.zeros((2, 8), np.float32), "int8", "x")       # all-zero rows stay zero
    assert not decode_rows(z, "x").any()


@pytest.mark.parametrize("wire,tol", [("f16", 0.02), ("int8", 0.08)])
def test_distributed_wire_formats_stay_close(tiny, tmp_path, wire, tol):
    toks = [256] + list(b"compressed activations")
    ref_m = MoEModel(tiny)
    ref = ref_m.forward(toks, ref_m.new_cache())
    w = Worker(tmp_path, f"w-{wire}", 2**22)
    try:
        backend = DistributedBackend(str(tiny), load_config(tiny), [w.address], wire=wire,
                                     progress=lambda m: None)
        m = MoEModel(tiny, backend=backend)
        out = m.forward(toks, m.new_cache())
        assert np.abs(out - ref).max() <= tol * np.abs(ref).max()
        row = next(r for r in backend.stats() if r["node"] == f"w-{wire}")
        assert row["mib_in"] > 0 and row["mib_out"] > 0
        backend.close()
    finally:
        w.kill()


# --- speed-aware placement ------------------------------------------------------------

from moelan.planner import speed_shares


def test_speed_shares_balance_and_drop():
    # an equally fast worker 0.3 ms away: takes a large share, layer gets faster
    sh, t, alone = speed_shares(8, 0.5, {"near": (0.5, 0.3, 1.0)})
    assert abs(sum(sh.values()) - 1) < 1e-9 and 0.25 < sh["near"] < 0.6 and t < 0.8 * alone
    # 10 ms away: the whole layer takes 4 ms here, so waiting for it can never pay
    sh, t, alone = speed_shares(8, 0.5, {"far": (0.5, 10.0, 1.0)})
    assert sh["far"] == 0 and sh[LOCAL] == 1 and t == pytest.approx(alone)
    # 1.5 ms away: a small share would still cost a round trip in most layers
    sh, _, _ = speed_shares(8, 0.45, {"meh": (0.45, 1.5, 1.0)})
    assert sh["meh"] == 0 or sh["meh"] > 0.2
    # a slow worker gets less than a fast one; a RAM cap is respected
    sh, _, _ = speed_shares(8, 0.5, {"fast": (0.5, 0.3, 1.0), "slow": (2.0, 0.3, 1.0), "tiny": (0.5, 0.3, 0.05)})
    assert sh["slow"] <= sh["fast"] and sh["tiny"] <= 0.05 + 1e-9


def test_make_plan_follows_shares():
    heat = {(0, e): (30 if e < 4 else 10) for e in range(16)}     # a few hot experts, the rest warm
    plan = make_plan([0], 16, 1, {"w": 10**6, LOCAL: 10**6, "off": 10**6}, heat=heat,
                     replicate=False, shares={"w": 0.25, LOCAL: 0.75, "off": 0.0})
    assert "off" not in plan.per_node
    total = sum(heat.values())
    w_heat = sum(heat[k] for k in plan.per_node["w"]) / total
    assert 0.18 < w_heat < 0.32         # about a quarter of the routing heat, not a quarter of experts
    assert len(plan.per_node["w"]) + len(plan.per_node[LOCAL]) == 16


def test_speed_placement_drops_a_worker_too_far_away(tiny, tmp_path, monkeypatch):
    import moelan.coordinator as coord
    prompt = [256] + list(b"far away")
    ref = greedy(MoEModel(tiny), prompt)
    w = Worker(tmp_path, "far", 2**22)
    monkeypatch.setattr(coord, "_SIM_NET", (0.025, 1e9))      # 50 ms round trip
    monkeypatch.setattr(coord, "_this_host", lambda: "coordinator-box")   # the worker counts as another machine
    try:
        msgs = []
        backend = _DistributedBackend(str(tiny), load_config(tiny), [w.address], placement="speed",
                                      progress=msgs.append)
        assert not backend.plan.per_node.get("far")             # nothing placed on it
        assert any("not used" in m for m in msgs)
        row = next(r for r in backend.stats() if r["node"] == "far")
        assert row["speed"]["share"] == 0 and row["speed"]["rtt_ms"] > 40
        monkeypatch.setattr(coord, "_SIM_NET", None)
        assert greedy(MoEModel(tiny, backend=backend), prompt) == ref
        assert row["requests"] == 0
        backend.close()
    finally:
        w.kill()


def test_speed_placement_ignores_a_worker_on_the_same_machine(tiny, tmp_path):
    w = Worker(tmp_path, "twin", 2**22)          # in-process: same host name as the coordinator
    try:
        msgs = []
        backend = _DistributedBackend(str(tiny), load_config(tiny), [w.address], placement="speed",
                                      progress=msgs.append)
        assert not backend.plan.per_node.get("twin") and any("on this machine" in m for m in msgs)
        row = next(r for r in backend.stats() if r["node"] == "twin")
        assert row["speed"] == {"share": 0.0, "same_host": True}
        backend.close()
    finally:
        w.kill()


from moelan.planner import node_ms


def test_node_ms_combines_cpu_and_gpu():
    assert node_ms(1.0) == 1.0                                         # no GPU
    assert node_ms(1.0, 0.25, 10**6) == pytest.approx(0.2)              # both busy: 1*0.25/(1.25)
    capped = node_ms(1.0, 0.25, gpu_experts=100, node_experts=1000)     # GPU holds only 10% of them
    assert capped == pytest.approx(0.9) and capped < 1.0
    assert node_ms(0.5, 7.6, 10**6, top_k=8) == 0.5                    # slower than a whole CPU layer: ignored


def test_speed_placement_counts_gpus(tiny, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import moelan.coordinator as coord
    monkeypatch.setattr(coord, "_this_host", lambda: "coordinator-box")
    w = Worker(tmp_path, "gpuw", 2**22, gpu="torch-cpu", gpu_budget=2**30, gpu_mode="auto")
    try:
        msgs = []
        backend = _DistributedBackend(str(tiny), load_config(tiny), [w.address], placement="speed",
                                      gpu="torch-cpu", gpu_budget=2**30, progress=msgs.append)
        rows = {r["node"]: r for r in backend.stats()}
        sp = rows["gpuw"]["speed"]
        assert sp["gpu"]["gpu_name"].startswith("torch-cpu") and sp["gpu"]["gpu_experts"] > 0
        assert sp["expert_ms"] <= sp["cpu_ms"] + 1e-9                    # the GPU can only help
        local = rows["local"]["speed"]
        assert local["gpu"]["gpu_name"].startswith("torch-cpu")         # the coordinator's GPU counted too
        assert any("torch-cpu" in m and "per expert" in m for m in msgs)
        backend.close()
    finally:
        w.kill()
