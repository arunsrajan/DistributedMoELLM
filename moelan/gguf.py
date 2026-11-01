"""GGUF (llama.cpp) model files.

A GGUF file holds the model's settings, tokenizer, chat template and all
weights in one file. This module reads it memory-mapped and maps it onto
moelan's own structures:

* the metadata becomes an HF-style config dict (``config_dict``), so the rest
  of moelan parses it like a safetensors checkpoint's ``config.json``;
* tensors keep their stored quantization: :class:`QuantMatrix` multiplies
  straight from the GGUF bytes (compiled kernels, or gguf-py without numba),
  and a layer's stacked experts (``blk.N.ffn_gate_exps``) are sliced per
  expert without copying, which is also what gets sent to workers;
* the tokenizer is rebuilt from the vocabulary in the file (byte-level BPE for
  Qwen / DeepSeek, SentencePiece for llama / Mixtral), and the chat template
  is the file's ``tokenizer.chat_template``.

Architectures: ``qwen3moe``, ``qwen2moe``, ``phimoe`` (Phi-3.5-MoE) and
``llama`` with experts (Mixtral). For llama, llama.cpp stores the query / key rows reordered for its
interleaved rotary layout; moelan applies interleaved rotary instead of
reordering quantized rows. Quantizations: F32, F16, BF16, Q8_0, Q4_0, Q4_1,
Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.
"""

from __future__ import annotations

import hashlib
import mmap
import struct
from pathlib import Path

import numpy as np

from . import kernels

# ggml type id -> (name, elements per block, bytes per block)
QTYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
          6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84),
          11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
          14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2)}
QTYPE_IDS = {v[0]: k for k, v in QTYPES.items()}
QUANTIZED = {k for k, v in QTYPES.items() if v[1] > 1}

_GGUF_MAGIC = b"GGUF"
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def is_gguf(path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    with open(p, "rb") as f:
        return f.read(4) == _GGUF_MAGIC


class Tensor:
    __slots__ = ("name", "qtype", "shape", "offset", "nbytes", "file")

    def __init__(self, name, qtype, shape, offset, nbytes, file):
        self.name, self.qtype, self.shape, self.offset, self.nbytes, self.file = name, qtype, shape, offset, nbytes, file

    @property
    def type_name(self) -> str:
        return QTYPES[self.qtype][0]

    def raw(self) -> np.ndarray:
        """Bytes as (rows..., bytes per row) uint8, a view of the memory map."""
        _, block, size = QTYPES[self.qtype]
        row_bytes = self.shape[-1] // block * size
        buf = np.frombuffer(self.file.mm, np.uint8, self.nbytes, self.file.data_start + self.offset)
        return buf.reshape(*self.shape[:-1], row_bytes)

    def array(self, dtype=np.float32) -> np.ndarray:
        """The whole tensor as floats (small tensors: norms, router, biases)."""
        if self.qtype == 0:
            a = self.raw().view(np.float32).reshape(self.shape)
        elif self.qtype == 1:
            a = self.raw().view(np.float16).reshape(self.shape)
        elif self.qtype == 30:
            a = (self.raw().view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(self.shape)
        else:
            r = self.raw()
            a = dequantize(r.reshape(-1, r.shape[-1]), self.qtype, self.shape[-1]).reshape(self.shape)
        return np.array(a, dtype=dtype)


class GGUFFile:
    def __init__(self, path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        self.mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        pos = 0

        def take(fmt):
            nonlocal pos
            v = struct.unpack_from(fmt, self.mm, pos)
            pos += struct.calcsize(fmt)
            return v[0]

        def string():
            nonlocal pos
            n = take("<Q")
            s = bytes(self.mm[pos:pos + n]).decode("utf-8", errors="replace")
            pos += n
            return s

        def value(t):
            nonlocal pos
            if t == 8:
                return string()
            if t == 9:
                it, n = take("<I"), take("<Q")
                if it in _SCALARS and it != 7:                      # numeric arrays in one go
                    dt = np.dtype(_SCALARS[it])
                    a = np.frombuffer(self.mm, dt, n, pos).copy()
                    pos += n * dt.itemsize
                    return a
                return [value(it) for _ in range(n)]
            return take(_SCALARS[t])

        if take("4s") != _GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        self.version = take("<I")
        if self.version < 2:
            raise ValueError(f"{path}: GGUF version {self.version} is too old")
        n_tensors, n_kv = take("<Q"), take("<Q")
        self.meta = {}
        for _ in range(n_kv):
            key = string()
            self.meta[key] = value(take("<I"))
        self.tensors: dict[str, Tensor] = {}
        infos = []
        for _ in range(n_tensors):
            name = string()
            dims = [take("<Q") for _ in range(take("<I"))]
            qtype, offset = take("<I"), take("<Q")
            infos.append((name, dims, qtype, offset))
        align = int(self.meta.get("general.alignment", 32))
        self.data_start = (pos + align - 1) // align * align
        self.header_bytes = pos
        for name, dims, qtype, offset in infos:
            if qtype not in QTYPES:
                raise ValueError(f"{name}: GGUF tensor type {qtype} is not supported yet "
                                 f"(supported: {', '.join(v[0] for v in QTYPES.values())})")
            shape = tuple(int(d) for d in reversed(dims))             # numpy order: rows first
            _, block, size = QTYPES[qtype]
            nbytes = int(np.prod(shape[:-1], dtype=np.int64)) * (shape[-1] // block) * size
            self.tensors[name] = Tensor(name, qtype, shape, offset, nbytes, self)

    def __contains__(self, name):
        return name in self.tensors

    def __getitem__(self, name) -> Tensor:
        return self.tensors[name]

    def fingerprint(self) -> str:
        h = hashlib.sha256(bytes(self.mm[:self.header_bytes]))
        h.update(str(self.path.stat().st_size).encode())
        return h.hexdigest()[:16]


_OPEN: dict[str, GGUFFile] = {}


def open_gguf(path) -> GGUFFile:
    key = str(Path(path).resolve())
    f = _OPEN.get(key)
    if f is None:
        f = _OPEN[key] = GGUFFile(path)
    return f


# ---------------------------------------------------------------------------
# Metadata -> HF-style config
# ---------------------------------------------------------------------------

_ARCHS = {"qwen3moe": "Qwen3MoeForCausalLM", "qwen2moe": "Qwen2MoeForCausalLM", "llama": "MixtralForCausalLM",
          "phimoe": "PhimoeForCausalLM"}


def config_dict(path) -> dict:
    """A config.json-like dict from the GGUF metadata (with a ``moelan_gguf``
    section for what config.json cannot say)."""
    g = open_gguf(path)
    m = g.meta
    arch = m.get("general.architecture")
    if arch not in _ARCHS:
        raise ValueError(f"GGUF architecture {arch!r} is not supported yet; supported: {sorted(_ARCHS)}")

    def k(name, default=None):
        return m.get(f"{arch}.{name}", default)
    if not k("expert_count"):
        raise ValueError(f"{path}: not a mixture-of-experts model (no {arch}.expert_count)")
    tokens = m.get("tokenizer.ggml.tokens") or []
    heads = int(k("attention.head_count"))
    hidden = int(k("embedding_length"))
    gate = "blk.0.ffn_gate_exps.weight"
    moe_inter = int(k("expert_feed_forward_length") or (g[gate].shape[1] if gate in g else k("feed_forward_length")))
    cfg = {
        "architectures": [_ARCHS[arch]], "model_type": arch,
        "vocab_size": int(k("vocab_size") or len(tokens) or g["token_embd.weight"].shape[0]),
        "hidden_size": hidden, "num_hidden_layers": int(k("block_count")),
        "num_attention_heads": heads, "num_key_value_heads": int(k("attention.head_count_kv", heads)),
        "head_dim": int(k("attention.key_length") or hidden // heads),
        "rms_norm_eps": float(k("attention.layer_norm_rms_epsilon", k("attention.layer_norm_epsilon", 1e-6))),
        "rope_theta": float(k("rope.freq_base", 10000.0)),
        "max_position_embeddings": int(k("context_length", 4096)),
        "bos_token_id": m.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": m.get("tokenizer.ggml.eos_token_id"),
        "tie_word_embeddings": "output.weight" not in g,
        "intermediate_size": moe_inter if arch in ("llama", "phimoe") else int(k("feed_forward_length", moe_inter)),
    }
    n_exp, used = int(k("expert_count")), int(k("expert_used_count"))
    if arch in ("llama", "phimoe"):
        cfg.update(num_local_experts=n_exp, num_experts_per_tok=used)
    if arch == "phimoe":
        # not in the file: Phi-3.5-MoE's router_jitter_noise
        cfg["router_jitter_noise"] = 0.01
        if "rope_factors_short.weight" in g:   # LongRoPE as llama.cpp stores it
            cfg["rope_scaling"] = {
                "type": "longrope", "short_factor": g["rope_factors_short.weight"].array().tolist(),
                "long_factor": g["rope_factors_long.weight"].array().tolist()
                if "rope_factors_long.weight" in g else None,
                "attention_factor": k("rope.scaling.attn_factor"),
                "original_max_position_embeddings": k("rope.scaling.original_context_length")}
    else:
        cfg.update(num_experts=n_exp, num_experts_per_tok=used, moe_intermediate_size=moe_inter,
                   norm_topk_prob=bool(k("expert_weights_norm", arch == "qwen3moe")))
    if arch == "qwen2moe":
        sh = "blk.0.ffn_gate_shexp.weight"
        cfg["shared_expert_intermediate_size"] = int(k("expert_shared_feed_forward_length")
                                                     or (g[sh].shape[0] if sh in g else 0))
    # extra eos ids: Qwen ends chat turns with <|im_end|>, also an "end of generation" token
    eos = [cfg["eos_token_id"]] if cfg["eos_token_id"] is not None else []
    for key in ("tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"):
        if m.get(key) is not None and m[key] not in eos:
            eos.append(m[key])
    for end in ("<|im_end|>", "<|end|>"):     # Qwen's and Phi's
        if end in tokens:
            i = tokens.index(end)
            if i not in eos:
                eos.append(i)
    cfg["eos_token_id"] = eos
    layer0 = [t for t in ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight")
              if t in g]
    cfg["moelan_gguf"] = {
        "file": g.path.name, "rope_interleaved": arch == "llama",
        "expert_types": {t.split(".")[2]: g[t].type_name for t in layer0},
        "expert_bytes": int(sum(g[t].nbytes for t in layer0) // n_exp) if layer0 else 0,
    }
    return cfg


# ---------------------------------------------------------------------------
# Quantized matrices
# ---------------------------------------------------------------------------

def dequantize(raw: np.ndarray, qtype: int, cols: int, out: np.ndarray | None = None) -> np.ndarray:
    """(rows, row bytes) quantized -> (rows, cols) float32."""
    if kernels.ENABLED:
        if out is None:
            out = np.empty((raw.shape[0], cols), np.float32)
        kernels.gguf_dequant(raw, qtype, out)
        return out
    try:
        from gguf.quants import dequantize as ref
        from gguf.constants import GGMLQuantizationType
    except ImportError as exc:
        raise RuntimeError("quantized GGUF weights need numba (pip install numba) or gguf "
                           "(pip install gguf)") from exc
    a = ref(np.ascontiguousarray(raw), GGMLQuantizationType(qtype)).astype(np.float32).reshape(raw.shape[0], cols)
    if out is not None:
        out[:] = a
        return out
    return a


class QuantMatrix:
    """A GGUF-quantized (rows, cols) weight, multiplied straight from its bytes."""

    __slots__ = ("raw", "qtype", "shape")

    def __init__(self, raw: np.ndarray, qtype: int, cols: int):
        self.raw, self.qtype = raw, qtype
        self.shape = (raw.shape[0], cols)

    def matmul_t(self, x: np.ndarray) -> np.ndarray:
        if kernels.ENABLED and x.shape[0] < kernels.BLAS_FROM:
            return kernels.gguf_matmul_t(x, self.raw, self.qtype, self.shape[1])
        from .experts import _scratch
        _, f = _scratch(*self.shape)
        return x @ dequantize(self.raw, self.qtype, self.shape[1], f).T

    def dequant(self) -> np.ndarray:
        return dequantize(self.raw, self.qtype, self.shape[1])

    def rows(self, idx) -> np.ndarray:
        idx = np.asarray(idx)
        return dequantize(self.raw[idx.reshape(-1)], self.qtype, self.shape[1]).reshape(*idx.shape, self.shape[1])

    @property
    def nbytes(self) -> int:
        return self.raw.nbytes


class Stacked:
    """Gate and up projections side by side: (gate rows + up rows, cols)."""

    __slots__ = ("parts",)

    def __init__(self, *parts):
        self.parts = parts

    def matmul_t(self, x):
        return np.concatenate([p.matmul_t(x) for p in self.parts], axis=1)

    def dequant(self):
        return np.concatenate([p.dequant() for p in self.parts], axis=0)

    @property
    def nbytes(self):
        return sum(p.nbytes for p in self.parts)


def matrix(t: Tensor, index=None):
    """A GGUF tensor (or one slice of a stacked one) as a moelan matrix."""
    from .experts import F32Matrix, Half16
    raw = t.raw() if index is None else t.raw()[index]
    cols = t.shape[-1]
    if t.qtype in QUANTIZED:
        return QuantMatrix(raw, t.qtype, cols)
    if t.qtype == 0:
        return F32Matrix(raw.view(np.float32).reshape(-1, cols))
    return Half16(raw.view(np.uint16).reshape(-1, cols), bf16=(t.qtype == 30))


class Embedding:
    """Token embedding lookup straight from a (possibly quantized) GGUF tensor."""

    def __init__(self, t: Tensor):
        self.t, self.m = t, matrix(t)

    def __getitem__(self, tokens):
        tokens = np.asarray(tokens)
        if isinstance(self.m, QuantMatrix):
            return self.m.rows(tokens)
        return self.m.dequant()[tokens] if not hasattr(self.m, "bits") else \
            self.m.__class__(self.m.bits[tokens.reshape(-1)], self.m.bf16).dequant().reshape(*tokens.shape, -1)

    @property
    def nbytes(self):
        return self.t.nbytes


# ---------------------------------------------------------------------------
# Experts of a GGUF file
# ---------------------------------------------------------------------------

PARTS = ("gate", "up", "down")


class GGUFSource:
    """Experts read straight out of a GGUF file (memory-mapped, not copied)."""

    def __init__(self, path, cfg):
        self.cfg = cfg
        self.g = open_gguf(path)

    def _tensor(self, layer, part):
        return self.g.tensors.get(f"blk.{layer}.ffn_{part}_exps.weight")

    def has(self, key) -> bool:
        return self._tensor(key[0], "gate") is not None and key[1] < self.cfg.num_experts

    def raw(self, key):
        out = {}
        for part in PARTS:
            t = self._tensor(key[0], part)
            r = t.raw()[key[1]]
            out[part] = (bytes(r), "GGUF:" + t.type_name, [int(r.shape[0]), int(r.shape[1])])
        return out

    def load(self, key):
        from .experts import Expert
        gate, up, down = (matrix(self._tensor(key[0], p), key[1]) for p in PARTS)
        return Expert(Stacked(gate, up), down, self.cfg.moe_intermediate_size)

    def nbytes(self, key) -> int:
        return sum(self._tensor(key[0], p).nbytes // self.cfg.num_experts for p in PARTS)


def expert_from_parts(cfg, parts: dict) -> "object":
    """An Expert from raw per-part bytes and GGUF type names (worker cache)."""
    from .experts import Expert
    H, I = cfg.hidden_size, cfg.moe_intermediate_size
    cols = {"gate": H, "up": H, "down": I}

    def mk(part):
        raw, tname = parts[part]
        q = QTYPE_IDS[tname]
        if q in QUANTIZED:
            return QuantMatrix(raw, q, cols[part])
        from .experts import F32Matrix, Half16
        if q == 0:
            return F32Matrix(raw.view(np.float32).reshape(-1, cols[part]))
        return Half16(raw.view(np.uint16).reshape(-1, cols[part]), bf16=(q == 30))
    return Expert(Stacked(mk("gate"), mk("up")), mk("down"), I)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_QWEN2_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")
_LLAMA3_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                 r"|\s*[\r\n]+|\s+(?!\S)|\s+")


def build_tokenizer(path):
    """An HF ``tokenizers`` object from the vocabulary inside a GGUF file."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    m = open_gguf(path).meta
    kind = m.get("tokenizer.ggml.model")
    tokens = list(m["tokenizer.ggml.tokens"])
    types = m.get("tokenizer.ggml.token_type")
    types = list(types) if types is not None else [1] * len(tokens)
    vocab = {t: i for i, t in enumerate(tokens)}
    if kind == "gpt2":
        merges = [tuple(x.split(" ", 1)) for x in m.get("tokenizer.ggml.merges", [])]
        tok = Tokenizer(models.BPE(vocab, merges, ignore_merges=False))
        pattern = _LLAMA3_SPLIT if m.get("tokenizer.ggml.pre") in ("llama-bpe", "llama3") else _QWEN2_SPLIT
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tok.decoder = decoders.ByteLevel()
    elif kind == "llama":
        scores = m.get("tokenizer.ggml.scores")
        scores = [float(v) for v in scores] if scores is not None else []
        if len(set(scores)) <= 1:
            # Placeholder scores (some converters write one value for all): use
            # SentencePiece's own order - by id, whitespace runs last.
            scores = [-1e9 if len(t) > 1 and set(t) == {"▁"} else -float(i) for i, t in enumerate(tokens)]
        merges = []
        for piece, score in zip(tokens, scores):
            # SentencePiece BPE as HF converts it: every split of a piece into two
            # known pieces is a merge, ranked by the merged piece's score, ties by
            # the parts' ids
            merges.extend((score, vocab[piece[:i]], vocab[piece[i:]], piece[:i], piece[i:])
                          for i in range(1, len(piece)) if piece[:i] in vocab and piece[i:] in vocab)
        merges.sort(key=lambda t: (-t[0], t[1], t[2]))
        merges = [(-t[0], t[3], t[4]) for t in merges]
        tok = Tokenizer(models.BPE(vocab, [(l, r) for _, l, r in merges], unk_token=tokens[0],
                                   byte_fallback=True, fuse_unk=True))
        tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
        tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(),
                                         decoders.Fuse(), decoders.Strip(" ", 1, 0)])
    else:
        raise ValueError(f"GGUF tokenizer type {kind!r} is not supported yet; pass --tokenizer with a "
                         "tokenizer.json")
    special = [AddedToken(t, special=True, normalized=False) for t, ty in zip(tokens, types) if ty in (3, 4)]
    if special:
        tok.add_special_tokens(special)
    return tok


def chat_template(path) -> str | None:
    return open_gguf(path).meta.get("tokenizer.chat_template")


# ---------------------------------------------------------------------------
# HF tensor names on top of a GGUF file (so the model loader is shared)
# ---------------------------------------------------------------------------

_LAYER_NAMES = {
    "input_layernorm.weight": "attn_norm.weight", "input_layernorm.bias": "attn_norm.bias",
    "post_attention_layernorm.weight": "ffn_norm.weight", "post_attention_layernorm.bias": "ffn_norm.bias",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_proj.bias": "attn_q.bias", "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.bias": "attn_v.bias", "self_attn.o_proj.bias": "attn_output.bias",
    "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight", "block_sparse_moe.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
_GLOBAL_NAMES = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
                 "model.norm.bias": "output_norm.bias", "lm_head.weight": "output.weight",
                 "lm_head.bias": "output.bias"}


class GGUFNames:
    """The few ``SafeTensorDir`` methods the model loader uses, answering HF
    tensor names from a GGUF file."""

    def __init__(self, path):
        self.g = open_gguf(path)

    def _name(self, hf: str) -> str | None:
        if hf in _GLOBAL_NAMES:
            return _GLOBAL_NAMES[hf]
        if hf.startswith("model.layers."):
            layer, rest = hf[len("model.layers."):].split(".", 1)
            if rest in _LAYER_NAMES:
                return f"blk.{layer}.{_LAYER_NAMES[rest]}"
        return None

    def __contains__(self, hf: str) -> bool:
        n = self._name(hf)
        return n is not None and n in self.g

    def tensor(self, hf: str) -> Tensor:
        n = self._name(hf)
        if n is None or n not in self.g:
            raise KeyError(f"{hf} (GGUF: {n}) is not in {self.g.path.name}")
        return self.g[n]

    def get(self, hf: str, dtype=np.float32) -> np.ndarray:
        return self.tensor(hf).array(dtype or np.float32)

    def matrix(self, hf: str):
        return matrix(self.tensor(hf))


# offsets of the f16 scale fields in each block (set to small values in random test blocks)
_SCALE_FIELDS = {2: (0,), 3: (0, 2), 6: (0,), 7: (0, 2), 8: (0,), 10: (80, 82), 11: (108,),
                 12: (0, 2), 13: (0, 2), 14: (208,)}


def random_blocks(rows: int, cols: int, qtype: int, rng=None, scale: float = 0.01) -> np.ndarray:
    """Random but well-behaved quantized rows (benchmarks and tests)."""
    rng = rng or np.random.default_rng(0)
    _, block, size = QTYPES[qtype]
    raw = rng.integers(0, 256, (rows, cols // block, size), dtype=np.uint8)
    half = np.frombuffer(np.float16(scale).tobytes(), np.uint8)
    for off in _SCALE_FIELDS.get(qtype, ()):
        raw[:, :, off:off + 2] = half
    return raw.reshape(rows, cols // block * size)


def synthetic_expert(hidden: int, inter: int, types: list[str], rng=None):
    """A random expert in the given GGUF types (gate, up, down)."""
    from .experts import Expert
    rng = rng or np.random.default_rng(0)
    g, u, d = (QTYPE_IDS[t] for t in types)
    mk = lambda q, o, i: QuantMatrix(random_blocks(o, i, q, rng, scale=0.001), q, i) if q in QUANTIZED else None
    return Expert(Stacked(mk(g, inter, hidden), mk(u, inter, hidden)), mk(d, hidden, inter), inter)
