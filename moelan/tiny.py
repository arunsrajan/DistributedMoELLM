"""Write a small random MoE checkpoint (Qwen3-MoE layout) for tests and demos.

It exercises every code path - safetensors shards, routing, GQA, q/k norm,
distribution - without downloading gigabytes. Byte-level vocabulary (256 + 2).

``int4_gs`` rounds the experts to group-scaled int4 values; with
``layout="colibri_int4"`` they are written in colibri's container format
(merged_weight + qs + qwen36_meta.json). An HF twin made with the same seed
and ``int4_gs`` holds exactly the same expert values in f32.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .safetensors_io import save_file


def make_tiny(out_dir: str | Path, hidden: int = 64, layers: int = 4, experts: int = 16,
              top_k: int = 4, moe_inter: int = 96, heads: int = 4, kv_heads: int = 2,
              seed: int = 0, bf16: bool = True, shards: int = 2, int4_gs: int = 0,
              layout: str = "hf") -> Path:
    if layout == "colibri_int4" and not int4_gs:
        raise ValueError("a colibri container needs int4_gs")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vocab = 258
    hd = hidden // heads

    def w(*shape, scale=None):
        s = scale if scale is not None else 1.0 / np.sqrt(shape[-1])
        return (rng.standard_normal(shape) * s).astype(np.float32)

    tensors = {"model.embed_tokens.weight": w(vocab, hidden, scale=0.5),
               "model.norm.weight": np.ones(hidden, np.float32),
               "lm_head.weight": w(vocab, hidden)}
    for i in range(layers):
        p = f"model.layers.{i}"
        tensors.update({
            f"{p}.input_layernorm.weight": (1 + 0.1 * rng.standard_normal(hidden)).astype(np.float32),
            f"{p}.post_attention_layernorm.weight": (1 + 0.1 * rng.standard_normal(hidden)).astype(np.float32),
            f"{p}.self_attn.q_proj.weight": w(heads * hd, hidden),
            f"{p}.self_attn.k_proj.weight": w(kv_heads * hd, hidden),
            f"{p}.self_attn.v_proj.weight": w(kv_heads * hd, hidden),
            f"{p}.self_attn.o_proj.weight": w(hidden, heads * hd),
            f"{p}.self_attn.q_norm.weight": np.ones(hd, np.float32),
            f"{p}.self_attn.k_norm.weight": np.ones(hd, np.float32),
            f"{p}.mlp.gate.weight": w(experts, hidden, scale=1.0),
        })
        for e in range(experts):
            q = f"{p}.mlp.experts.{e}"
            mats = [w(moe_inter, hidden), w(moe_inter, hidden), w(hidden, moe_inter)]
            if int4_gs:
                quant = [_int4(m, int4_gs) for m in mats]
                mats = [v * np.repeat(s, int4_gs, 1) for v, s in quant]
            if layout == "colibri_int4":
                tensors[f"{q}.merged_weight"] = np.concatenate([_pack(v) for v, _ in quant])
                tensors[f"{q}.qs"] = np.concatenate([s.reshape(-1) for _, s in quant])
            else:
                for n, m in zip(("gate", "up", "down"), mats):
                    tensors[f"{q}.{n}_proj.weight"] = m.astype(np.float32)

    names = list(tensors)
    weight_map = {}
    per = (len(names) + shards - 1) // shards
    for s in range(shards):
        chunk = names[s * per:(s + 1) * per]
        fname = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        save_file({n: tensors[n] for n in chunk}, out / fname,
                  bf16={n for n in chunk if n.endswith("proj.weight") and not
                        (int4_gs and ".experts." in n)} if bf16 else None)
        weight_map.update({n: fname for n in chunk})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (out / "config.json").write_text(json.dumps({
        "architectures": ["MoelanTinyForCausalLM"], "model_type": "qwen3_moe",
        "vocab_size": vocab, "hidden_size": hidden, "num_hidden_layers": layers,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": hd,
        "num_experts": experts, "num_experts_per_tok": top_k, "moe_intermediate_size": moe_inter,
        "norm_topk_prob": True, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "bos_token_id": 256, "eos_token_id": 257, "tie_word_embeddings": False,
        "max_position_embeddings": 2048,
    }, indent=2))
    if layout == "colibri_int4":
        (out / "qwen36_meta.json").write_text(json.dumps({
            "model_type": "qwen3_moe", "ebits": 4, "expert_gs": int4_gs,
            "num_experts": experts, "topk": top_k, "moe_inter": moe_inter}))
    return out


def _int4(m: np.ndarray, gs: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int4 per group of ``gs`` inputs: values in -8..7, f32 scales."""
    O, I = m.shape
    g = m.reshape(O, I // gs, gs)
    s = (np.abs(g).max(2) / 7.0).astype(np.float32)
    v = np.clip(np.rint(g / s[:, :, None]), -8, 7).astype(np.int8).reshape(O, I)
    return v, s


def _pack(v: np.ndarray) -> np.ndarray:
    """Two's-complement nibbles, low nibble = even column (colibri's pack_int4)."""
    n = (v & 15).astype(np.uint8)
    return (n[:, 0::2] | (n[:, 1::2] << 4)).reshape(-1)


def make_tiny_deepseek(out_dir: str | Path, version: int = 3, hidden: int = 64, layers: int = 3,
                       experts: int = 16, top_k: int = 4, n_group: int = 4, topk_group: int = 2,
                       heads: int = 4, q_lora_rank: int | None = 48, kv_lora_rank: int = 32,
                       qk_nope: int = 16, qk_rope: int = 8, v_dim: int = 16, moe_inter: int = 32,
                       inter: int = 96, n_shared: int = 1, first_dense: int = 1,
                       yarn: bool = True, seed: int = 0, bf16: bool = True) -> Path:
    """A random DeepSeek-V2/V3-layout checkpoint (MLA, shared experts, grouped
    routing, dense first layers) with HF tensor names and config keys."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vocab = 258

    def w(*shape, scale=None):
        s = scale if scale is not None else 1.0 / np.sqrt(shape[-1])
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def norm(n):
        return (1 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    t = {"model.embed_tokens.weight": w(vocab, hidden, scale=0.5), "model.norm.weight": norm(hidden),
         "lm_head.weight": w(vocab, hidden)}
    for i in range(layers):
        p, a = f"model.layers.{i}", f"model.layers.{i}.self_attn"
        t[f"{p}.input_layernorm.weight"] = norm(hidden)
        t[f"{p}.post_attention_layernorm.weight"] = norm(hidden)
        if q_lora_rank:
            t[f"{a}.q_a_proj.weight"] = w(q_lora_rank, hidden)
            t[f"{a}.q_a_layernorm.weight"] = norm(q_lora_rank)
            t[f"{a}.q_b_proj.weight"] = w(heads * (qk_nope + qk_rope), q_lora_rank)
        else:
            t[f"{a}.q_proj.weight"] = w(heads * (qk_nope + qk_rope), hidden)
        t[f"{a}.kv_a_proj_with_mqa.weight"] = w(kv_lora_rank + qk_rope, hidden)
        t[f"{a}.kv_a_layernorm.weight"] = norm(kv_lora_rank)
        t[f"{a}.kv_b_proj.weight"] = w(heads * (qk_nope + v_dim), kv_lora_rank)
        t[f"{a}.o_proj.weight"] = w(hidden, heads * v_dim)
        m = f"{p}.mlp"
        if i < first_dense:
            t[f"{m}.gate_proj.weight"], t[f"{m}.up_proj.weight"] = w(inter, hidden), w(inter, hidden)
            t[f"{m}.down_proj.weight"] = w(hidden, inter)
            continue
        t[f"{m}.gate.weight"] = w(experts, hidden, scale=1.0)
        if version == 3:
            t[f"{m}.gate.e_score_correction_bias"] = (0.05 * rng.standard_normal(experts)).astype(np.float32)
        for e in range(experts):
            q = f"{m}.experts.{e}"
            t[f"{q}.gate_proj.weight"], t[f"{q}.up_proj.weight"] = w(moe_inter, hidden), w(moe_inter, hidden)
            t[f"{q}.down_proj.weight"] = w(hidden, moe_inter)
        if n_shared:
            s = f"{m}.shared_experts"
            si = moe_inter * n_shared
            t[f"{s}.gate_proj.weight"], t[f"{s}.up_proj.weight"] = w(si, hidden), w(si, hidden)
            t[f"{s}.down_proj.weight"] = w(hidden, si)
    save_file(t, out / "model.safetensors",
              bf16={n for n in t if n.endswith("proj.weight") or n.endswith("mqa.weight")} if bf16 else None)
    cfg = {
        "architectures": [f"DeepseekV{version}ForCausalLM"], "model_type": f"deepseek_v{version}",
        "vocab_size": vocab, "hidden_size": hidden, "intermediate_size": inter,
        "moe_intermediate_size": moe_inter, "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": heads, "n_shared_experts": n_shared, "n_routed_experts": experts,
        "routed_scaling_factor": 2.5, "kv_lora_rank": kv_lora_rank, "q_lora_rank": q_lora_rank,
        "qk_rope_head_dim": qk_rope, "v_head_dim": v_dim, "qk_nope_head_dim": qk_nope,
        "head_dim": qk_rope, "n_group": n_group, "topk_group": topk_group, "num_experts_per_tok": top_k,
        "moe_layer_freq": 1, "first_k_dense_replace": first_dense, "hidden_act": "silu",
        "max_position_embeddings": 2048, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "attention_bias": False, "attention_dropout": 0.0, "tie_word_embeddings": False,
        "bos_token_id": 256, "eos_token_id": 257, "torch_dtype": "float32",
        "rope_scaling": {"type": "yarn", "rope_type": "yarn", "factor": 4.0,
                         "original_max_position_embeddings": 512, "beta_fast": 32, "beta_slow": 1,
                         "mscale": 1.0, "mscale_all_dim": 1.0} if yarn else None,
    }
    if version == 3:
        cfg.update(topk_method="noaux_tc", scoring_func="sigmoid", norm_topk_prob=True)
    else:
        cfg.update(topk_method="group_limited_greedy", scoring_func="softmax", norm_topk_prob=False,
                   aux_loss_alpha=0.001, seq_aux=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return out
