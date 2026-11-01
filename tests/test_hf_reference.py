"""Cross-check moelan against Hugging Face transformers' own implementations on
tiny random checkpoints. Skipped unless torch and transformers are installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from moelan.model import MoEModel, route_sparsemixer  # noqa: E402
from moelan.tiny import make_tiny, make_tiny_deepseek, make_tiny_phimoe  # noqa: E402

TOKENS = [256] + list(b"Hello, MoE over a LAN!")


def hf_logits(model_dir):
    m = transformers.AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor([TOKENS])).logits[0, -1].numpy()


def ours(model_dir):
    m = MoEModel(model_dir)
    c = m.new_cache()
    m.forward(TOKENS[:5], c)          # through the KV cache, one token at a time after
    for t in TOKENS[5:-1]:
        m.forward([t], c)
    return m.forward(TOKENS[-1:], c)


@pytest.mark.parametrize("version,q_lora,yarn", [(3, 48, True), (3, None, False), (2, None, True), (2, 48, False)])
def test_deepseek_matches_transformers(tmp_path, version, q_lora, yarn):
    d = make_tiny_deepseek(tmp_path, version=version, q_lora_rank=q_lora, yarn=yarn, seed=version)
    ref = hf_logits(d)
    got = ours(d)
    np.testing.assert_allclose(got, ref, atol=3e-4, rtol=1e-4)
    assert int(np.argmax(got)) == int(np.argmax(ref))


def test_qwen3_moe_matches_transformers(tmp_path):
    d = make_tiny(tmp_path, layers=3, experts=8, top_k=2)
    import json
    cfg = json.loads((d / "config.json").read_text())
    cfg["architectures"] = ["Qwen3MoeForCausalLM"]  # the tiny model is a Qwen3-MoE layout
    (d / "config.json").write_text(json.dumps(cfg))
    np.testing.assert_allclose(ours(d), hf_logits(d), atol=3e-4, rtol=1e-4)


@pytest.mark.parametrize("bf16", [False, True])
def test_phimoe_matches_transformers(tmp_path, bf16):
    d = make_tiny_phimoe(tmp_path, bf16=bf16, seed=4)
    ref = hf_logits(d)
    got = ours(d)
    np.testing.assert_allclose(got, ref, atol=3e-4, rtol=1e-4)
    assert int(np.argmax(got)) == int(np.argmax(ref))


def test_sparsemixer_matches_transformers():
    from transformers.models.phimoe.modeling_phimoe import sparsemixer
    rng = np.random.default_rng(0)
    base = rng.standard_normal((64, 16)).astype(np.float32)
    # near-ties within the 2% band, negative maxima and exact zeros
    base[::3, 1:4] = base[::3, :1] * np.float32(1 - 0.01 * rng.random((22, 3)))
    base[1::4] = -np.abs(base[1::4])
    base[2::5, :5] = 0.0
    for eps in (0.01, 0.1):
        w, idx = sparsemixer(torch.from_numpy(base), jitter_eps=eps, training=False)
        my_idx, my_w = route_sparsemixer(base, eps)
        np.testing.assert_array_equal(my_idx, idx.numpy())
        np.testing.assert_allclose(my_w, w.numpy(), rtol=1e-6, atol=1e-7)
        assert (my_w < 0.999).any()          # the band really kept several experts
