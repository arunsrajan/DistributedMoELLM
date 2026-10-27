"""Model configuration and tensor naming for the supported MoE families.

Supported (HF transformers layouts, safetensors weights):
  * Mixtral          (``MixtralForCausalLM``)        block_sparse_moe.experts.{e}.w1/w2/w3
  * Qwen2-MoE        (``Qwen2MoeForCausalLM``)       mlp.experts.{e}.* + shared_expert
  * Qwen3-MoE        (``Qwen3MoeForCausalLM``)       mlp.experts.{e}.*, per-head q/k norm
  * OLMoE            (``OlmoeForCausalLM``)          mlp.experts.{e}.*, full-width q/k norm
  * DeepSeek V2      (``DeepseekV2ForCausalLM``)     MLA attention, shared experts, softmax
                                                     (group-limited) routing, dense first layers
  * DeepSeek V3      (``DeepseekV3ForCausalLM``)     as V2 with sigmoid routing + correction bias,
                                                     node-limited groups (also Kimi K2, BF16 only)
  * colibri int4 containers of the above (``qwen36_meta.json`` + ``merged_weight``/``qs``)
  * moelan-tiny      (random test model written by ``moelan make-tiny``; Qwen3-MoE layout)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    family: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    intermediate_size: int = 0          # dense MLP layers, if any
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False
    shared_expert_intermediate_size: int = 0
    dense_layers: list[int] = field(default_factory=list)  # layers with a plain MLP
    max_position_embeddings: int = 4096
    bos_token_id: int | None = None
    eos_token_ids: list[int] = field(default_factory=list)
    expert_layout: str = "hf"           # "hf" or "colibri_int4"
    expert_gs: int = 0                  # int4 scale group size (colibri)
    # routing (defaults = plain softmax top-k)
    scoring_func: str = "softmax"       # "softmax" or "sigmoid"
    topk_method: str = "greedy"         # "greedy", "group_limited_greedy" (V2), "noaux_tc" (V3)
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    # multi-head latent attention (DeepSeek); attention == "gqa" for the others
    attention: str = "gqa"
    q_lora_rank: int | None = None
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    rope_scaling: dict | None = None
    raw: dict = field(default_factory=dict, repr=False)

    # ---- tensor names -------------------------------------------------
    def _mlp(self, layer: int) -> str:
        if self.family == "mixtral":
            return f"model.layers.{layer}.block_sparse_moe"
        return f"model.layers.{layer}.mlp"

    def router_name(self, layer: int) -> str:
        return f"{self._mlp(layer)}.gate.weight"

    def expert_names(self, layer: int, expert: int) -> tuple[str, ...]:
        """Tensor names of one routed expert: (gate, up, down), or for a colibri
        int4 container (merged gate|up|down weight, scales)."""
        p = f"{self._mlp(layer)}.experts.{expert}"
        if self.expert_layout == "colibri_int4":
            return f"{p}.merged_weight", f"{p}.qs"
        if self.family == "mixtral":
            return f"{p}.w1.weight", f"{p}.w3.weight", f"{p}.w2.weight"
        return f"{p}.gate_proj.weight", f"{p}.up_proj.weight", f"{p}.down_proj.weight"

    def dense_mlp_names(self, layer: int) -> tuple[str, str, str]:
        p = f"model.layers.{layer}.mlp"
        return f"{p}.gate_proj.weight", f"{p}.up_proj.weight", f"{p}.down_proj.weight"

    def shared_expert_names(self, layer: int) -> tuple[str, str, str, str]:
        if self.family.startswith("deepseek"):  # plural, and no sigmoid gate
            p = f"model.layers.{layer}.mlp.shared_experts"
            return (f"{p}.gate_proj.weight", f"{p}.up_proj.weight", f"{p}.down_proj.weight",
                    f"model.layers.{layer}.mlp.shared_experts_gate.none")
        p = f"model.layers.{layer}.mlp.shared_expert"
        return (f"{p}.gate_proj.weight", f"{p}.up_proj.weight", f"{p}.down_proj.weight",
                f"model.layers.{layer}.mlp.shared_expert_gate.weight")

    def router_bias_name(self, layer: int) -> str:
        return f"{self._mlp(layer)}.gate.e_score_correction_bias"

    def kv_cache_dims(self) -> tuple[int, int, int]:
        """(heads, key dim, value dim) cached per token and layer. MLA caches the
        compressed latent and the shared rotary key: one 'head'."""
        if self.attention == "mla":
            return 1, self.kv_lora_rank, self.qk_rope_head_dim
        return self.num_kv_heads, self.head_dim, self.head_dim

    def is_moe_layer(self, layer: int) -> bool:
        return layer not in self.dense_layers

    def moe_layers(self) -> list[int]:
        return [i for i in range(self.num_layers) if self.is_moe_layer(i)]


_FAMILIES = {
    "MixtralForCausalLM": "mixtral",
    "Qwen2MoeForCausalLM": "qwen2_moe",
    "Qwen3MoeForCausalLM": "qwen3_moe",
    "OlmoeForCausalLM": "olmoe",
    "MoelanTinyForCausalLM": "qwen3_moe",
    "DeepseekV2ForCausalLM": "deepseek_v2",
    "DeepseekV3ForCausalLM": "deepseek_v3",
}


def load_config(model_dir: str | Path) -> ModelConfig:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    arch = (raw.get("architectures") or ["?"])[0]
    family = _FAMILIES.get(arch)
    if family is None:
        raise ValueError(f"unsupported architecture {arch!r}; supported: {sorted(_FAMILIES)}")

    hidden = raw["hidden_size"]
    heads = raw["num_attention_heads"]
    layers = raw["num_hidden_layers"]
    extra: dict = {}
    if family == "mixtral":
        n_exp, top_k = raw["num_local_experts"], raw["num_experts_per_tok"]
        moe_inter = raw["intermediate_size"]
        norm_topk = True
    elif family.startswith("deepseek"):
        v3 = family == "deepseek_v3"
        n_exp, top_k = raw["n_routed_experts"], raw["num_experts_per_tok"]
        moe_inter = raw["moe_intermediate_size"]
        norm_topk = raw.get("norm_topk_prob", v3)
        method = raw.get("topk_method", "noaux_tc" if v3 else "greedy")
        extra = dict(
            scoring_func=raw.get("scoring_func", "sigmoid" if v3 else "softmax"),
            topk_method="greedy" if method == "gready" else method,  # V2's config default has the typo
            n_group=raw.get("n_group") or 1, topk_group=raw.get("topk_group") or 1,
            routed_scaling_factor=float(raw.get("routed_scaling_factor", 1.0)),
            attention="mla", q_lora_rank=raw.get("q_lora_rank"), kv_lora_rank=raw["kv_lora_rank"],
            qk_nope_head_dim=raw["qk_nope_head_dim"], qk_rope_head_dim=raw["qk_rope_head_dim"],
            v_head_dim=raw["v_head_dim"],
            shared_expert_intermediate_size=moe_inter * (raw.get("n_shared_experts") or 0))
    else:
        n_exp, top_k = raw["num_experts"], raw["num_experts_per_tok"]
        moe_inter = raw.get("moe_intermediate_size", raw.get("intermediate_size"))
        norm_topk = raw.get("norm_topk_prob", family != "qwen2_moe")

    dense_layers: list[int] = list(raw.get("mlp_only_layers", []) or [])
    step = raw.get("decoder_sparse_step", 1) or 1
    if step > 1:
        dense_layers += [i for i in range(layers) if (i + 1) % step != 0]
    if family.startswith("deepseek"):
        first, freq = raw.get("first_k_dense_replace", 0), raw.get("moe_layer_freq", 1) or 1
        dense_layers += [i for i in range(layers) if i < first or i % freq != 0]
    quant = raw.get("quantization_config") or {}
    if quant.get("quant_method") == "fp8":
        raise ValueError("FP8 checkpoints (e.g. the official DeepSeek-V3 release) are not supported; "
                         "use a BF16 conversion")

    layout, gs = "hf", 0
    meta = Path(model_dir) / "qwen36_meta.json"
    if meta.exists():  # colibri container: experts packed int4, group-scaled
        m = json.loads(meta.read_text())
        if m.get("ebits") != 4:
            raise ValueError(f"colibri container with ebits={m.get('ebits')} is not supported yet (int4 only)")
        layout, gs = "colibri_int4", int(m.get("expert_gs") or 0)
        if not gs:
            raise ValueError("colibri container without group scales (expert_gs) is not supported yet")
    gen = Path(model_dir) / "generation_config.json"
    eos = raw.get("eos_token_id")
    if gen.exists():
        eos = json.loads(gen.read_text()).get("eos_token_id", eos)
    eos_ids = eos if isinstance(eos, list) else ([eos] if eos is not None else [])
    cfg = ModelConfig(
        family=family,
        vocab_size=raw["vocab_size"],
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        num_kv_heads=raw.get("num_key_value_heads") or heads,
        head_dim=raw.get("head_dim") or hidden // heads,
        num_experts=n_exp,
        top_k=top_k,
        moe_intermediate_size=moe_inter,
        intermediate_size=raw.get("intermediate_size", 0),
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        rope_theta=raw.get("rope_theta", 10000.0),
        norm_topk_prob=bool(norm_topk),
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
        shared_expert_intermediate_size=raw.get("shared_expert_intermediate_size", 0) or 0,
        rope_scaling=raw.get("rope_scaling"),
        dense_layers=sorted(set(dense_layers)),
        max_position_embeddings=raw.get("max_position_embeddings", 4096),
        bos_token_id=raw.get("bos_token_id"),
        eos_token_ids=eos_ids,
        expert_layout=layout,
        expert_gs=gs,
        raw=raw,
    )
    for k, v in extra.items():
        setattr(cfg, k, v)
    if cfg.attention == "mla":
        cfg.num_kv_heads = heads
        cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim  # query/key head size
    return cfg


def model_fingerprint(model_dir: str | Path) -> str:
    """Stable id of a checkpoint: config + tensor layout. Workers key caches by it."""
    root = Path(model_dir)
    h = hashlib.sha256((root / "config.json").read_bytes())
    index = root / "model.safetensors.index.json"
    if index.exists():
        h.update(index.read_bytes())
    else:
        for p in sorted(root.glob("*.safetensors")):
            with open(p, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                h.update(f.read(n))
    return h.hexdigest()[:16]
