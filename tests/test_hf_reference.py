"""Cross-check moelan against Hugging Face transformers' own implementations on
tiny random checkpoints. Skipped unless torch and transformers are installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from moelan.model import MoEModel  # noqa: E402
from moelan.tiny import make_tiny, make_tiny_deepseek  # noqa: E402

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
