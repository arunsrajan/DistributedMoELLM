import socket
import threading

import numpy as np
import pytest

from moelan.config import load_config
from moelan.coordinator import DistributedBackend
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

    def __init__(self, tmp_path, name, budget, model_dir=None):
        self.ready, self.stop = threading.Event(), threading.Event()
        self.t = threading.Thread(target=serve, daemon=True, kwargs=dict(
            host="127.0.0.1", port=0, budget=budget, cache_root=tmp_path / name,
            model_dir=model_dir, node_id=name, advertise=False,
            ready=self.ready, stop=self.stop))
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
