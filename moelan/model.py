"""The dense part of an MoE transformer (attention, norms, router, embeddings),
in numpy. Routed experts are delegated to an *expert backend*: local (this
machine, streaming from disk) or distributed (workers on the LAN).

The dense part is what Colibri keeps resident in RAM; it is small next to the
experts (for Qwen3-30B-A3B about 1.5 B of the 30 B parameters).
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import kernels
from .config import ModelConfig, load_config
from .experts import (CheckpointSource, ExpertStore, GroupQuant, default_format, expert_ffn,
                      resident_bytes)
from .safetensors_io import SafeTensorDir


def _mm(x: np.ndarray, w) -> np.ndarray:
    """x @ W.T for a plain f32 matrix or a quantized one."""
    if isinstance(w, np.ndarray):
        return kernels.f32_matmul_t(x, w) if kernels.ENABLED and w.dtype == np.float32 else x @ w.T
    return w.matmul_t(x)


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * w


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


class Rotary:
    """Rotary position embedding, HF ``rotate_half`` convention.

    ``interleaved``: the weights store each rotated pair side by side (DeepSeek
    MLA); the vector is de-interleaved first, as in DeepSeek's
    ``apply_rotary_pos_emb``. ``rope_scaling`` of type ``yarn`` follows
    DeepSeek's YaRN: blended frequencies and a cos/sin magnitude scale."""

    def __init__(self, dim: int, theta: float, scaling: dict | None = None, interleaved: bool = False):
        self.dim, self.interleaved = dim, interleaved
        exps = np.arange(0, dim, 2, dtype=np.float64) / dim
        inv = 1.0 / theta ** exps
        self.mscale = 1.0
        kind = (scaling or {}).get("type") or (scaling or {}).get("rope_type")
        if kind == "yarn":
            f = float(scaling["factor"])
            orig = scaling.get("original_max_position_embeddings", 4096)
            fast, slow = scaling.get("beta_fast", 32), scaling.get("beta_slow", 1)

            def corr_dim(rot):
                return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(theta))
            low = max(math.floor(corr_dim(fast)), 0)
            high = min(math.ceil(corr_dim(slow)), dim - 1)
            if low == high:
                high += 0.001
            ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
            extra_mask = 1.0 - ramp
            inv = inv / f * (1 - extra_mask) + inv * extra_mask
            self.mscale = _yarn_mscale(f, scaling.get("mscale", 1)) /                 _yarn_mscale(f, scaling.get("mscale_all_dim", 0))
        self.inv = inv

    def __call__(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """x: (n, heads, dim)."""
        if self.interleaved:
            x = np.concatenate([x[..., 0::2], x[..., 1::2]], axis=-1)
        ang = positions[:, None].astype(np.float64) * self.inv[None, :]
        cos = (np.concatenate([np.cos(ang)] * 2, -1) * self.mscale).astype(np.float32)[:, None, :]
        sin = (np.concatenate([np.sin(ang)] * 2, -1) * self.mscale).astype(np.float32)[:, None, :]
        half = self.dim // 2
        rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        return x * cos + rot * sin


def rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """Plain rotary embedding (kept for callers that only have a theta)."""
    return Rotary(x.shape[-1], theta)(x, positions)


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest per row, largest first (ties: lower index first)."""
    idx = np.argsort(-scores, axis=-1, kind="stable")[:, :k]
    return idx


def route(logits: np.ndarray, top_k: int, normalize: bool) -> tuple[np.ndarray, np.ndarray]:
    """Softmax router, top-k selection. Returns (expert ids (n,k), weights (n,k))."""
    probs = softmax(logits.astype(np.float32), axis=-1)
    idx = _top_k(probs, top_k)
    w = np.take_along_axis(probs, idx, axis=-1)
    if normalize:
        w = w / w.sum(-1, keepdims=True)
    return idx.astype(np.int32), w.astype(np.float32)


def route_grouped(logits: np.ndarray, cfg: ModelConfig, bias: np.ndarray | None):
    """DeepSeek routers.

    V2: softmax scores; ``greedy`` top-k, or ``group_limited_greedy`` (only
    experts in the ``topk_group`` groups with the best single score).
    V3 (``noaux_tc``): sigmoid scores; groups ranked by the sum of their two
    best bias-corrected scores; experts picked by bias-corrected score but
    weighted by the raw sigmoid score. Then normalize and/or scale.
    """
    n, E = logits.shape
    logits = logits.astype(np.float32)
    scores = 1.0 / (1.0 + np.exp(-logits)) if cfg.scoring_func == "sigmoid" else softmax(logits)
    choice = scores + bias[None, :] if bias is not None else scores
    if cfg.topk_method in ("group_limited_greedy", "noaux_tc") and cfg.n_group > 1:
        g = choice.reshape(n, cfg.n_group, E // cfg.n_group)
        if cfg.topk_method == "noaux_tc":
            group_score = -np.sort(-g, axis=-1)[..., :2].sum(-1)
        else:
            group_score = g.max(-1)
        keep = _top_k(group_score, cfg.topk_group)
        mask = np.zeros((n, cfg.n_group), bool)
        np.put_along_axis(mask, keep, True, axis=-1)
        choice = np.where(np.repeat(mask, E // cfg.n_group, axis=1), choice, 0.0)
    idx = _top_k(choice, cfg.top_k)
    w = np.take_along_axis(scores, idx, axis=-1)
    if cfg.top_k > 1 and cfg.norm_topk_prob:
        w = w / (w.sum(-1, keepdims=True) + 1e-20)
        if cfg.topk_method == "noaux_tc":
            w = w * cfg.routed_scaling_factor
    else:
        w = w * cfg.routed_scaling_factor
    return idx.astype(np.int32), w.astype(np.float32)


def group_by_expert(idx: np.ndarray, w: np.ndarray):
    """(n,k) routing -> experts, offsets, rows, weights (rows grouped per expert)."""
    k = idx.shape[1]
    flat = idx.reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_e = flat[order]
    experts, starts = np.unique(sorted_e, return_index=True)
    offsets = np.append(starts, len(flat)).astype(np.int32)
    rows = (order // k).astype(np.int32)
    weights = w.reshape(-1)[order].astype(np.float32)
    return experts.astype(np.int32), offsets, rows, weights


# ---------------------------------------------------------------------------
# Expert backends
# ---------------------------------------------------------------------------

class ExpertBackend:
    """Computes sum_k w_k * expert_k(x) for routed rows. Records routing heat."""

    def __init__(self):
        self.heat: Counter = Counter()
        self.last: dict[int, list[int]] = {}  # layer -> experts the latest token picked
        self.moe_time = 0.0

    def moe(self, layer: int, x: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        for e in idx.reshape(-1):
            self.heat[(layer, int(e))] += 1
        self.last[layer] = idx[-1].tolist()
        y = self._moe(layer, x, idx, w)
        self.moe_time += time.perf_counter() - t0
        return y

    def _moe(self, layer, x, idx, w):  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalBackend(ExpertBackend):
    """All experts on this machine: resident or streamed from disk (Colibri-style)."""

    def __init__(self, store: ExpertStore):
        super().__init__()
        self.store = store

    def _moe(self, layer, x, idx, w):
        experts, offsets, rows, weights = group_by_expert(idx, w)
        groups = [(experts[i], rows[offsets[i]:offsets[i + 1]], weights[offsets[i]:offsets[i + 1]])
                  for i in range(len(experts))]
        return self.store.compute(layer, x, groups)


def save_heat(heat: Counter, path: str | Path) -> None:
    p = Path(path)
    old = load_heat(p)
    old.update(heat)
    p.write_text(json.dumps({f"{l}:{e}": c for (l, e), c in old.items()}))


def load_heat(path: str | Path) -> Counter:
    """Routing counts: moelan's JSON, or colibri's ``.coli_usage`` text file."""
    p = Path(path)
    if not p.exists():
        return Counter()
    text = p.read_text()
    if not text.lstrip().startswith("{"):  # colibri: "layer expert count" lines
        heat = Counter()
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 3 and int(parts[0]) >= 0:
                heat[(int(parts[0]), int(parts[1]))] = int(parts[2])
        return heat
    raw = json.loads(text)
    return Counter({tuple(int(v) for v in k.split(":")): c for k, c in raw.items()})


# ---------------------------------------------------------------------------
# KV cache and the model
# ---------------------------------------------------------------------------

class KVCache:
    def __init__(self, cfg: ModelConfig, capacity: int = 256):
        self.cfg = cfg
        self.length = 0
        self.cap = capacity
        heads, kd, vd = cfg.kv_cache_dims()
        self.k = [np.zeros((heads, capacity, kd), np.float32) for _ in range(cfg.num_layers)]
        self.v = [np.zeros((heads, capacity, vd), np.float32) for _ in range(cfg.num_layers)]

    def _grow(self, need: int) -> None:
        cap = self.cap
        while cap < need:
            cap *= 2
        if cap == self.cap:
            return
        for lst in (self.k, self.v):
            for i, a in enumerate(lst):
                b = np.zeros((a.shape[0], cap, a.shape[2]), np.float32)
                b[:, :self.cap] = a
                lst[i] = b
        self.cap = cap

    def append(self, layer: int, k: np.ndarray, v: np.ndarray, start: int):
        n = k.shape[1]
        self._grow(start + n)
        self.k[layer][:, start:start + n] = k
        self.v[layer][:, start:start + n] = v
        return self.k[layer][:, :start + n], self.v[layer][:, :start + n]


class MoEModel:
    def __init__(self, model_dir: str | Path, backend: ExpertBackend | None = None,
                 lru_size: int = 0, expert_format: str | None = None, pin_budget: int = 0,
                 heat: Counter | None = None, progress=None, dense_format: str = "f32"):
        self.model_dir = Path(model_dir)
        self.cfg = cfg = load_config(model_dir)
        self.dense_format = dense_format
        st = self.st = SafeTensorDir(model_dir)

        def opt(name):
            return st.get(name) if name in st else None

        self.embed = st.get("model.embed_tokens.weight", dtype=None)  # only rows are read: keep as stored
        self.norm = st.get("model.norm.weight")
        self.lm_head = st.get("model.embed_tokens.weight") \
            if cfg.tie_word_embeddings or "lm_head.weight" not in st else st.get("lm_head.weight")
        if cfg.attention == "mla":
            self.rotary = Rotary(cfg.qk_rope_head_dim, cfg.rope_theta, cfg.rope_scaling, interleaved=True)
            self.softmax_scale = (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5
            rs = cfg.rope_scaling or {}
            if rs.get("mscale_all_dim"):
                m = _yarn_mscale(float(rs["factor"]), rs["mscale_all_dim"])
                self.softmax_scale *= m * m
        else:
            self.rotary = Rotary(cfg.head_dim, cfg.rope_theta)
        self.layers = []
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}"
            a = f"{p}.self_attn"
            L = {
                "in_norm": st.get(f"{p}.input_layernorm.weight"),
                "post_norm": st.get(f"{p}.post_attention_layernorm.weight"),
                "o": st.get(f"{a}.o_proj.weight"),
            }
            if cfg.attention == "mla":
                if cfg.q_lora_rank:
                    L.update(q_a=st.get(f"{a}.q_a_proj.weight"), q_a_norm=st.get(f"{a}.q_a_layernorm.weight"),
                             q_b=st.get(f"{a}.q_b_proj.weight"))
                else:
                    L["q"] = st.get(f"{a}.q_proj.weight")
                L["kv_a"] = st.get(f"{a}.kv_a_proj_with_mqa.weight")
                L["kv_a_norm"] = st.get(f"{a}.kv_a_layernorm.weight")
                # kv_b expands the latent into per-head keys and values; it is
                # absorbed into the query and output instead (see _mla).
                kv_b = st.get(f"{a}.kv_b_proj.weight").reshape(
                    cfg.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim, cfg.kv_lora_rank)
                L["w_uk"] = np.ascontiguousarray(kv_b[:, :cfg.qk_nope_head_dim])                 # (H, nope, r)
                L["w_uv_t"] = np.ascontiguousarray(kv_b[:, cfg.qk_nope_head_dim:].transpose(0, 2, 1))  # (H, r, v)
            else:
                L.update({
                    "q": st.get(f"{a}.q_proj.weight"), "k": st.get(f"{a}.k_proj.weight"),
                    "v": st.get(f"{a}.v_proj.weight"),
                    "qb": opt(f"{a}.q_proj.bias"), "kb": opt(f"{a}.k_proj.bias"), "vb": opt(f"{a}.v_proj.bias"),
                    "q_norm": opt(f"{a}.q_norm.weight"), "k_norm": opt(f"{a}.k_norm.weight"),
                })
            if cfg.is_moe_layer(i):
                L["router"] = st.get(cfg.router_name(i))
                L["router_bias"] = opt(cfg.router_bias_name(i))
                sg, su, sd, sgate = cfg.shared_expert_names(i)
                if sg in st:
                    L["shared"] = (st.get(sg), st.get(su), st.get(sd))
                    L["shared_gate"] = opt(sgate)
            else:
                L["mlp"] = tuple(st.get(n) for n in cfg.dense_mlp_names(i))
            if dense_format == "int8":
                for name in ("q", "k", "v", "o", "q_a", "q_b", "kv_a"):
                    if name in L:
                        L[name] = GroupQuant.quantize(L[name], 64)
            self.layers.append(L)
        if dense_format == "int8":  # colibri's default: about 4x less memory traffic per token
            self.lm_head = GroupQuant.quantize(self.lm_head, 64)
        elif dense_format != "f32":
            raise ValueError("dense_format must be f32 or int8")
        if backend is None:
            fmt = expert_format or default_format(cfg)
            store = ExpertStore([CheckpointSource(model_dir, cfg)], fmt=fmt, lru_size=lru_size,
                                gs=cfg.expert_gs or 64)
            if pin_budget > 0:  # keep the hottest experts resident, stream the rest
                n = pin_budget // resident_bytes(cfg, fmt)
                order = sorted(((l, e) for l in cfg.moe_layers() for e in range(cfg.num_experts)),
                               key=lambda k: -(heat or {}).get(k, 0))
                if progress:
                    progress(f"pinning {min(n, len(order))} of {len(order)} experts as {fmt}")
                store.pin(order[:n])
            backend = LocalBackend(store)
        self.backend = backend

    def dense_bytes(self) -> int:
        n = self.embed.nbytes + self.norm.nbytes + self.lm_head.nbytes
        for L in self.layers:
            for v in L.values():
                if isinstance(v, tuple):
                    n += sum(m.nbytes for m in v)
                elif v is not None:
                    n += v.nbytes
        return n

    def new_cache(self) -> KVCache:
        return KVCache(self.cfg)

    def _mla(self, L, h, positions, cache: KVCache, layer: int) -> np.ndarray:
        """DeepSeek multi-head latent attention with the latent KV cache.

        Keys/values are never expanded: with W_kv_b = [W_uk; W_uv] per head,
        q_nope . (W_uk c) = (W_uk^T q_nope) . c, and sum_t p_t W_uv c_t =
        W_uv (sum_t p_t c_t). So the cache holds only the normalized latent c
        (kv_lora_rank floats) and the shared rotary key per token."""
        cfg = self.cfg
        n, H = h.shape[0], cfg.num_heads
        nope, rd, eps = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.rms_norm_eps
        if "q" in L:
            q = _mm(h, L["q"])
        else:
            q = _mm(rms_norm(_mm(h, L["q_a"]), L["q_a_norm"], eps), L["q_b"])
        q = q.reshape(n, H, nope + rd)
        q_nope, q_pe = q[..., :nope], self.rotary(q[..., nope:], positions)
        ckv = _mm(h, L["kv_a"])
        c = rms_norm(ckv[:, :cfg.kv_lora_rank], L["kv_a_norm"], eps)
        k_pe = self.rotary(ckv[:, None, cfg.kv_lora_rank:], positions)[:, 0]
        C, KPE = cache.append(layer, c[None], k_pe[None], int(positions[0]))
        C, KPE = C[0], KPE[0]                                            # (T, r), (T, rd)
        q_lat = np.matmul(q_nope.transpose(1, 0, 2), L["w_uk"])         # (H, n, r)
        scores = (q_lat @ C.T + q_pe.transpose(1, 0, 2) @ KPE.T) * self.softmax_scale
        mask = np.arange(C.shape[0])[None, :] > positions[:, None]
        probs = softmax(np.where(mask[None], -np.inf, scores), axis=-1)  # (H, n, T)
        out = np.matmul(probs @ C, L["w_uv_t"])                          # (H, n, v)
        return _mm(out.transpose(1, 0, 2).reshape(n, H * cfg.v_head_dim), L["o"])

    def _attention(self, L, h, positions, cache: KVCache, layer: int) -> np.ndarray:
        cfg = self.cfg
        if cfg.attention == "mla":
            return self._mla(L, h, positions, cache, layer)
        n = h.shape[0]
        H, KV, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        q, k, v = _mm(h, L["q"]), _mm(h, L["k"]), _mm(h, L["v"])
        if L["qb"] is not None:
            q, k, v = q + L["qb"], k + L["kb"], v + L["vb"]
        qn, kn = L["q_norm"], L["k_norm"]
        if qn is not None and qn.shape[0] == H * hd:   # OLMoE: norm over the full projection
            q = rms_norm(q, qn, cfg.rms_norm_eps)
            k = rms_norm(k, kn, cfg.rms_norm_eps)
        q = q.reshape(n, H, hd)
        k = k.reshape(n, KV, hd)
        v = v.reshape(n, KV, hd)
        if qn is not None and qn.shape[0] == hd:       # Qwen3: per-head norm
            q = rms_norm(q, qn, cfg.rms_norm_eps)
            k = rms_norm(k, kn, cfg.rms_norm_eps)
        q = self.rotary(q, positions)
        k = self.rotary(k, positions)
        K, V = cache.append(layer, k.transpose(1, 0, 2), v.transpose(1, 0, 2), int(positions[0]))
        T = K.shape[1]
        g = H // KV
        qg = q.transpose(1, 0, 2).reshape(KV, g, n, hd)
        scores = np.einsum("kgnd,ktd->kgnt", qg, K) * (1.0 / math.sqrt(hd))  # python float: stays f32
        key_pos = np.arange(T)
        mask = key_pos[None, :] > positions[:, None]          # (n, T) causal
        scores = np.where(mask[None, None], -np.inf, scores)
        probs = softmax(scores, axis=-1)
        out = np.einsum("kgnt,ktd->kgnd", probs, V)           # (KV, g, n, hd)
        out = out.reshape(H, n, hd).transpose(1, 0, 2).reshape(n, H * hd)
        return _mm(out, L["o"])

    def _ffn(self, L, h, layer: int) -> np.ndarray:
        cfg = self.cfg
        if "mlp" in L:
            return expert_ffn(h, *L["mlp"])
        logits = h @ L["router"].T
        if cfg.family.startswith("deepseek"):
            idx, w = route_grouped(logits, cfg, L.get("router_bias"))
        else:
            idx, w = route(logits, cfg.top_k, cfg.norm_topk_prob)
        y = self.backend.moe(layer, h, idx, w)
        if "shared" in L:
            s = expert_ffn(h, *L["shared"])
            if L.get("shared_gate") is not None:
                s = s * (1.0 / (1.0 + np.exp(-(h @ L["shared_gate"].T))))
            y = y + s
        return y

    def forward(self, tokens, cache: KVCache) -> np.ndarray:
        """Run ``tokens`` (appended after the cache) and return last-position logits."""
        cfg = self.cfg
        tokens = np.asarray(tokens, dtype=np.int64)
        positions = np.arange(cache.length, cache.length + len(tokens))
        x = self.embed[tokens].astype(np.float32)
        for i, L in enumerate(self.layers):
            h = rms_norm(x, L["in_norm"], cfg.rms_norm_eps)
            x = x + self._attention(L, h, positions, cache, i)
            h = rms_norm(x, L["post_norm"], cfg.rms_norm_eps)
            x = x + self._ffn(L, h, i)
        cache.length += len(tokens)
        x = rms_norm(x[-1:], self.norm, cfg.rms_norm_eps)
        return _mm(x, self.lm_head)[0]

    def generate(self, prompt_ids, max_new_tokens: int = 64, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0, seed: int | None = None,
                 stop_ids=None, on_token=None):
        """Yield generated token ids. Greedy when temperature == 0."""
        rng = np.random.default_rng(seed)
        stop = set(stop_ids if stop_ids is not None else self.cfg.eos_token_ids)
        cache = self.new_cache()
        logits = self.forward(prompt_ids, cache)
        for _ in range(max_new_tokens):
            tok = sample(logits, temperature, top_p, top_k, rng)
            if on_token:
                on_token(tok)
            yield tok
            if tok in stop:
                return
            logits = self.forward([tok], cache)


def sample(logits: np.ndarray, temperature: float, top_p: float, top_k: int,
           rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits.astype(np.float64) / temperature
    if top_k > 0:
        kth = np.partition(z, -top_k)[-top_k]
        z = np.where(z < kth, -np.inf, z)
    p = np.exp(z - z.max())
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        cut = order[cum - p[order] > top_p]
        p[cut] = 0
        p /= p.sum()
    return int(rng.choice(len(p), p=p))
