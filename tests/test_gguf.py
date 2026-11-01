"""GGUF support: block decoders against llama.cpp's gguf-py, a tiny Qwen3-MoE
GGUF (written with gguf-py's writer, K-quant experts) against its safetensors
twin, the tokenizer, distribution, and - when present - a real llama.cpp file."""

import json
from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")

from moelan import kernels  # noqa: E402
from moelan.config import load_config  # noqa: E402
from moelan.gguf import QTYPE_IDS, QTYPES, QUANTIZED, dequantize, random_blocks  # noqa: E402
from moelan.model import MoEModel  # noqa: E402
from moelan.safetensors_io import SafeTensorDir, save_file  # noqa: E402
from moelan.tiny import make_tiny  # noqa: E402

from test_moelan import Worker, _DistributedBackend, greedy  # noqa: E402

Q = gguf.GGMLQuantizationType


@pytest.mark.parametrize("qtype", sorted(QUANTIZED), ids=[QTYPES[q][0] for q in sorted(QUANTIZED)])
def test_block_decoders_match_gguf_py(qtype):
    rng = np.random.default_rng(qtype)
    cols = 512
    raw = rng.integers(0, 256, (5, cols // QTYPES[qtype][1] * QTYPES[qtype][2]), dtype=np.uint8)
    want = gguf.quants.dequantize(raw.copy(), Q(qtype)).astype(np.float32).reshape(5, cols)
    got = dequantize(raw, qtype, cols)
    ok = np.isfinite(want)
    assert np.array_equal(np.isfinite(got), ok) and np.array_equal(got[ok], want[ok])
    if kernels.ENABLED:   # and the matmul kernel on well-behaved blocks
        raw = random_blocks(5, cols, qtype, rng)
        w = gguf.quants.dequantize(raw.copy(), Q(qtype)).astype(np.float32).reshape(5, cols)
        for rows in (1, 2, 5):   # fused decode kernels and the row-decoding one
            x = rng.standard_normal((rows, cols)).astype(np.float32)
            np.testing.assert_allclose(kernels.gguf_matmul_t(x, raw, qtype, cols), x @ w.T, rtol=1e-4, atol=1e-4)


# --- a tiny Qwen3-MoE GGUF and its safetensors twin ---------------------------------

def _byte_vocab():
    """GPT-2 style byte-level vocabulary: 256 byte tokens + 2 specials (ids 256, 257)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    by_byte = dict(zip(bs, (chr(c) for c in cs)))
    return [by_byte[b] for b in range(256)] + ["<|endoftext|>", "<|im_end|>"]


def write_tiny_gguf(tmp: Path, expert_types=("Q4_K", "Q5_K", "Q6_K")):
    """A 2-layer Qwen3-MoE as GGUF (experts in K-quants, attention Q8_0) plus
    a safetensors twin holding exactly the values the GGUF decodes to."""
    hf = make_tiny(tmp / "src", hidden=256, layers=2, experts=8, top_k=2, moe_inter=256, heads=4,
                   kv_heads=2, seed=3, bf16=False, shards=1)
    st = SafeTensorDir(hf)
    cfg = load_config(hf)
    rng = np.random.default_rng(1)
    twin = {n: st.get(n) for n in st.keys()}
    w = gguf.GGUFWriter(str(tmp / "tiny-qwen3moe.gguf"), "qwen3moe")
    w.add_block_count(cfg.num_layers)
    w.add_context_length(2048)
    w.add_embedding_length(cfg.hidden_size)
    w.add_feed_forward_length(cfg.moe_intermediate_size)
    w.add_expert_feed_forward_length(cfg.moe_intermediate_size)
    w.add_head_count(cfg.num_heads)
    w.add_head_count_kv(cfg.num_kv_heads)
    w.add_key_length(cfg.head_dim)
    w.add_value_length(cfg.head_dim)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_expert_count(cfg.num_experts)
    w.add_expert_used_count(cfg.top_k)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(_byte_vocab())
    w.add_token_types([1] * 256 + [3, 3])
    w.add_token_merges([])
    w.add_bos_token_id(256)
    w.add_eos_token_id(257)
    w.add_chat_template("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                        "{% endfor %}{{ '<|im_start|>assistant\\n' }}")   # an expression keeps the final newline

    def put(name, arr, qtype="F32"):
        if qtype == "F32":
            w.add_tensor(name, arr.astype(np.float32))
            return arr.astype(np.float32)
        if qtype == "F16":
            w.add_tensor(name, arr.astype(np.float16))
            return arr.astype(np.float16).astype(np.float32)
        q = gguf.quants.quantize(arr.astype(np.float32), Q[qtype])
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=Q[qtype])
        return gguf.quants.dequantize(q, Q[qtype]).astype(np.float32).reshape(arr.shape)

    twin["model.embed_tokens.weight"] = put("token_embd.weight", twin["model.embed_tokens.weight"], "F16")
    twin["lm_head.weight"] = put("output.weight", twin["lm_head.weight"], "Q8_0")
    put("output_norm.weight", twin["model.norm.weight"])
    for i in range(cfg.num_layers):
        p, b = f"model.layers.{i}", f"blk.{i}"
        put(f"{b}.attn_norm.weight", twin[f"{p}.input_layernorm.weight"])
        put(f"{b}.ffn_norm.weight", twin[f"{p}.post_attention_layernorm.weight"])
        for hf_name, g_name in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"),
                                ("o_proj", "attn_output")):
            twin[f"{p}.self_attn.{hf_name}.weight"] = put(f"{b}.{g_name}.weight",
                                                          twin[f"{p}.self_attn.{hf_name}.weight"], "Q8_0")
        put(f"{b}.attn_q_norm.weight", twin[f"{p}.self_attn.q_norm.weight"])
        put(f"{b}.attn_k_norm.weight", twin[f"{p}.self_attn.k_norm.weight"])
        put(f"{b}.ffn_gate_inp.weight", twin[f"{p}.mlp.gate.weight"])
        for part, qname in zip(("gate", "up", "down"), expert_types):
            q = QTYPE_IDS[qname]
            rows, cols = (cfg.moe_intermediate_size, cfg.hidden_size) if part != "down" else \
                (cfg.hidden_size, cfg.moe_intermediate_size)
            raw = np.stack([random_blocks(rows, cols, q, rng, scale=0.004) for _ in range(cfg.num_experts)])
            w.add_tensor(f"{b}.ffn_{part}_exps.weight", raw, raw_shape=raw.shape, raw_dtype=Q[qname])
            vals = gguf.quants.dequantize(raw, Q[qname]).astype(np.float32).reshape(cfg.num_experts, rows, cols)
            for e in range(cfg.num_experts):
                twin[f"{p}.mlp.experts.{e}.{part}_proj.weight"] = vals[e]
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    tw = tmp / "twin"
    tw.mkdir()
    save_file(twin, tw / "model.safetensors")
    (tw / "config.json").write_text((hf / "config.json").read_text())
    return tmp / "tiny-qwen3moe.gguf", tw


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    return write_tiny_gguf(tmp_path_factory.mktemp("gguf"))


def test_gguf_config(tiny_gguf):
    path, twin = tiny_gguf
    a, b = load_config(path), load_config(twin)
    assert a.expert_layout == "gguf" and not a.rope_interleaved
    for f in ("family", "hidden_size", "num_layers", "num_heads", "num_kv_heads", "head_dim", "num_experts",
              "top_k", "moe_intermediate_size", "norm_topk_prob", "vocab_size"):
        assert getattr(a, f) == getattr(b, f), f
    assert a.raw["moelan_gguf"]["expert_types"] == {"ffn_gate_exps": "Q4_K", "ffn_up_exps": "Q5_K",
                                                    "ffn_down_exps": "Q6_K"}


def test_gguf_model_matches_its_twin(tiny_gguf):
    path, twin = tiny_gguf
    toks = [256] + list(b"gguf experts in k-quants")
    g, t = MoEModel(path), MoEModel(twin)
    a, b = g.forward(toks, g.new_cache()), t.forward(toks, t.new_cache())
    np.testing.assert_allclose(a, b, atol=2e-3 * np.abs(b).max())
    assert greedy(g, toks, 10) == greedy(t, toks, 10)
    assert g.dense_format == "native" and g.backend.store.fmt == "native"


def test_gguf_tokenizer_byte_level_and_template(tiny_gguf):
    from moelan.tokenizer import Tokenizer
    tok = Tokenizer(tiny_gguf[0])
    text = "Hello, wörld!  \n  ok"
    ids = tok.encode(text)
    assert ids == list(text.encode("utf-8")) and tok.decode(ids) == text
    assert tok.encode("<|im_end|>") == [257]                          # special token stays whole
    assert tok.chat_prompt([{"role": "user", "content": "hi"}]) == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert 257 in load_config(tiny_gguf[0]).eos_token_ids


def test_gguf_experts_distributed(tiny_gguf, tmp_path):
    path, twin = tiny_gguf
    toks = [256] + list(b"spread over workers")
    ref = greedy(MoEModel(path), toks, 8)
    w1, w2 = Worker(tmp_path, "g1", 2**24), Worker(tmp_path, "g2", 2**24)       # no model: experts pushed
    try:
        b = _DistributedBackend(str(path), load_config(path), [w1.address, w2.address], placement="ram",
                                replicate=False, progress=lambda m: None)
        assert all(b.plan.per_node.get(n) for n in ("g1", "g2"))
        assert greedy(MoEModel(path, backend=b), toks, 8) == ref
        assert b.local.misses == 0
        b.close()
        sent = []                                                      # second start: from the cache
        b = _DistributedBackend(str(path), load_config(path), [w1.address, w2.address], placement="ram",
                                replicate=False, progress=sent.append)
        assert not any("sending" in s for s in sent)
        assert greedy(MoEModel(path, backend=b), toks, 8) == ref
        b.close()
    finally:
        w1.kill(), w2.kill()


STORIES = Path(__file__).parent / "data" / "stories15M_MOE"


@pytest.mark.skipif(not (STORIES / "stories15M_MOE-Q8_0.gguf").exists(),
                    reason="download ggml-org/stories15M_MOE into tests/data/stories15M_MOE to run")
def test_real_llama_cpp_gguf_matches_safetensors():
    from moelan.tokenizer import Tokenizer
    hf = MoEModel(STORIES)
    ids = [1] + Tokenizer(STORIES).encode("Once upon a time, there was a little girl named Lily. She")
    ref = greedy(hf, ids, 24)
    for name in ("stories15M_MOE-F16.gguf", "stories15M_MOE-Q8_0.gguf"):
        m = MoEModel(STORIES / name)
        assert greedy(m, ids, 24) == ref, name
        tok = Tokenizer(STORIES / name)                                 # rebuilt from the GGUF vocabulary
        assert tok.encode("  Hello  world!\n    def f(x):") == Tokenizer(STORIES).encode("  Hello  world!\n    def f(x):")


def write_tiny_phimoe_gguf(tmp: Path):
    """A PhiMoE GGUF the way llama.cpp lays it out (LayerNorm and projection
    biases, rope_factors tensors, K-quant experts) plus its safetensors twin."""
    from moelan.tiny import make_tiny_phimoe
    hf = make_tiny_phimoe(tmp / "src", hidden=256, inter=256, layers=2, experts=4, seed=5, bf16=False)
    st, cfg = SafeTensorDir(hf), load_config(hf)
    twin = {n: st.get(n) for n in st.keys()}
    rs, rng = cfg.rope_scaling, np.random.default_rng(2)
    w = gguf.GGUFWriter(str(tmp / "tiny-phimoe.gguf"), "phimoe")
    w.add_block_count(cfg.num_layers)
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size)
    w.add_feed_forward_length(cfg.moe_intermediate_size)
    w.add_head_count(cfg.num_heads)
    w.add_head_count_kv(cfg.num_kv_heads)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_rope_scaling_attn_factors(rs["short_mscale"])
    w.add_rope_scaling_orig_ctx_len(rs["original_max_position_embeddings"])
    w.add_expert_count(cfg.num_experts)
    w.add_expert_used_count(cfg.top_k)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(_byte_vocab()[:256] + ["<|endoftext|>", "<|end|>"])
    w.add_token_types([1] * 256 + [3, 3])
    w.add_token_merges([])
    w.add_bos_token_id(256)
    w.add_eos_token_id(256)

    def put(name, arr, qtype="F32"):
        if qtype == "F32":
            w.add_tensor(name, arr.astype(np.float32))
            return arr.astype(np.float32)
        q = gguf.quants.quantize(arr.astype(np.float32), Q[qtype])
        w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=Q[qtype])
        return gguf.quants.dequantize(q, Q[qtype]).astype(np.float32).reshape(arr.shape)

    put("token_embd.weight", twin["model.embed_tokens.weight"])
    put("rope_factors_short.weight", np.array(rs["short_factor"]))
    put("rope_factors_long.weight", np.array(rs["long_factor"]))
    twin["lm_head.weight"] = put("output.weight", twin["lm_head.weight"], "Q8_0")
    for n, g in (("lm_head.bias", "output.bias"), ("model.norm.weight", "output_norm.weight"),
                 ("model.norm.bias", "output_norm.bias")):
        put(g, twin[n])
    for i in range(cfg.num_layers):
        p, b = f"model.layers.{i}", f"blk.{i}"
        for hf_name, g_name in (("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")):
            put(f"{b}.{g_name}.weight", twin[f"{p}.{hf_name}.weight"])
            put(f"{b}.{g_name}.bias", twin[f"{p}.{hf_name}.bias"])
        for hf_name, g_name in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"),
                                ("o_proj", "attn_output")):
            a = f"{p}.self_attn.{hf_name}"
            twin[f"{a}.weight"] = put(f"{b}.{g_name}.weight", twin[f"{a}.weight"], "Q8_0")
            put(f"{b}.{g_name}.bias", twin[f"{a}.bias"])
        m = f"{p}.block_sparse_moe"
        put(f"{b}.ffn_gate_inp.weight", twin[f"{m}.gate.weight"])
        for part, hf_w, qname in (("gate", "w1", "Q4_K"), ("up", "w3", "Q5_K"), ("down", "w2", "Q6_K")):
            rows, cols = twin[f"{m}.experts.0.{hf_w}.weight"].shape     # gguf-py cannot write K-quants
            raw = np.stack([random_blocks(rows, cols, QTYPE_IDS[qname], rng, scale=0.004)
                            for _ in range(cfg.num_experts)])
            w.add_tensor(f"{b}.ffn_{part}_exps.weight", raw, raw_shape=raw.shape, raw_dtype=Q[qname])
            vals = gguf.quants.dequantize(raw, Q[qname]).astype(np.float32).reshape(-1, rows, cols)
            for e in range(cfg.num_experts):
                twin[f"{m}.experts.{e}.{hf_w}.weight"] = vals[e]
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    tw = tmp / "twin"
    tw.mkdir()
    save_file(twin, tw / "model.safetensors")
    (tw / "config.json").write_text((hf / "config.json").read_text())
    return tmp / "tiny-phimoe.gguf", tw


def test_phimoe_gguf_matches_its_twin(tmp_path):
    path, twin = write_tiny_phimoe_gguf(tmp_path)
    a, b = load_config(path), load_config(twin)
    assert a.family == b.family == "phimoe" and a.norm_type == "layer" and a.topk_method == "sparsemixer"
    assert a.rope_scaling["short_factor"] == pytest.approx(b.rope_scaling["short_factor"])
    assert 257 in a.eos_token_ids                                       # <|end|>
    toks = [256] + list(b"phi experts in k-quants")
    g, t = MoEModel(path), MoEModel(twin)
    x, y = g.forward(toks, g.new_cache()), t.forward(toks, t.new_cache())
    np.testing.assert_allclose(x, y, atol=2e-3 * np.abs(y).max())
    assert greedy(g, toks, 10) == greedy(t, toks, 10)
