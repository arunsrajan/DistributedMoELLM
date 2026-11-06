"""Reading prompts on a GPU (--gpu on serve / run).

On a CPU most of a long prompt's reading time is attention: an 18,169-token
agent prompt took about 21 minutes on a two-laptop cluster, 78% of each layer
in the coordinator's attention and its projections. A pass of many rows now
does those on the GPU; generating stays on the CPU. These tests use PyTorch's
``torch-cpu`` test device (f32), so results must match the CPU closely."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from moelan import accel                                    # noqa: E402
from moelan.accel import PromptGpu                          # noqa: E402
from moelan.experts import GroupQuant                       # noqa: E402
from moelan.model import MoEModel, _gqa_attention, _mm     # noqa: E402
from moelan.tiny import make_tiny, make_tiny_deepseek      # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("tiny"), layers=3, experts=8, top_k=2)


@pytest.fixture(scope="module")
def cpu_model(tiny):
    return MoEModel(tiny)


PROMPT = [256] + [int(t) for t in np.random.default_rng(1).integers(0, 256, 299)]


def greedy(model, prompt, n):
    return list(model.generate(prompt, max_new_tokens=n, stop_ids=[]))


def gpu_model(tiny, **kw):
    msgs = []
    m = MoEModel(tiny, prompt_gpu="torch-cpu", gpu_mode=kw.pop("gpu_mode", "max"), progress=msgs.append, **kw)
    return m, msgs


# --- the attention step ----------------------------------------------------------------

def attention_case(m, T, H=8, KV=2, hd=16, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((m, H, hd)).astype(np.float32)
    K = rng.standard_normal((KV, T + 5, hd)).astype(np.float32)     # a cache with room to spare
    V = rng.standard_normal((KV, T + 5, hd)).astype(np.float32)
    return q, K, V, np.arange(T - m, T)


@pytest.mark.parametrize("fused", [True, False], ids=["sdpa", "blocks"])
@pytest.mark.parametrize("m,T", [(40, 40), (37, 300), (200, 1000)], ids=["whole", "later-pass", "long"])
def test_attention_matches_the_cpu(fused, m, T, monkeypatch):
    pg = PromptGpu("torch-cpu")
    if not fused:
        pg._sdpa = False
        monkeypatch.setattr(PromptGpu, "SCORES", 2000)           # several blocks of query rows
    q, K, V, pos = attention_case(m, T)
    want, got = np.empty_like(q), np.empty_like(q)
    _gqa_attention(q, K[:, :T], V[:, :T], pos, want)
    assert pg.attention(q, K[:, :T], V[:, :T], pos, got)
    np.testing.assert_allclose(got, want, atol=2e-5)
    assert pg.calls == 1 and pg.rows == m


def test_rows_at_scattered_positions_go_to_the_cpu():
    pg = PromptGpu("torch-cpu")
    q, K, V, _ = attention_case(40, 100)
    assert not pg.attention(q, K, V, np.arange(40) * 2 + 20, np.empty_like(q))
    assert pg.calls == 0


def test_projections_match_the_cpu_for_every_weight_kind():
    pg = PromptGpu("torch-cpu")
    rng = np.random.default_rng(2)
    x = rng.standard_normal((64, 96)).astype(np.float32)
    w = (rng.standard_normal((48, 96)) * 0.05).astype(np.float32)
    for key, weight in (("f32", w), ("int8", GroupQuant.quantize(w, 32))):
        np.testing.assert_allclose(pg.matmul(key, x, weight), _mm(x, weight), atol=1e-4)
    assert set(pg.weights) == {"f32", "int8"} and pg.used == 2 * 48 * 96 * 4


def test_a_weight_that_cannot_be_uploaded_stays_on_the_cpu():
    pg = PromptGpu("torch-cpu")
    assert pg.matmul("x", np.ones((40, 4), np.float32), object()) is None
    assert pg.weights["x"] is None and "x" in pg.failed


# --- the model ------------------------------------------------------------------------

def test_prompts_read_on_the_gpu_give_the_same_output(tiny, cpu_model):
    m, msgs = gpu_model(tiny)
    pg = m.prompt_gpu
    assert pg is not None and "reads prompts" in msgs[-1]
    np.testing.assert_allclose(m.forward(PROMPT, m.new_cache()), cpu_model.forward(PROMPT, cpu_model.new_cache()),
                               atol=1e-4, rtol=1e-4)
    layers = len(m.layers)
    assert pg.calls == layers and pg.rows == layers * len(PROMPT)
    assert len(pg.weights) == 4 * layers                  # q, k, v, o of every layer, uploaded once
    assert greedy(m, PROMPT, 8) == greedy(cpu_model, PROMPT, 8)
    calls = pg.calls
    c = m.new_cache()
    m.forward(PROMPT[:20], c)                              # a short prompt and generating: the CPU
    m.forward([65], c)
    assert pg.calls == calls


def test_later_passes_of_a_long_prompt_too(tiny, cpu_model, monkeypatch):
    monkeypatch.setattr(MoEModel, "PREFILL_CHUNK", 37)     # passes 2.. start after earlier tokens
    m, _ = gpu_model(tiny)
    np.testing.assert_allclose(m.forward(PROMPT, m.new_cache()), cpu_model.forward(PROMPT, cpu_model.new_cache()),
                               atol=1e-4, rtol=1e-4)
    assert m.prompt_gpu.calls == len(m.layers) * 8         # 8 passes of 37 rows; the last (4 rows): CPU


def test_a_failing_gpu_step_falls_back_to_the_cpu(tiny, cpu_model, monkeypatch):
    m, _ = gpu_model(tiny)

    def broken(self, *a):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(PromptGpu, "_attend", broken)
    np.testing.assert_allclose(m.forward(PROMPT, m.new_cache()), cpu_model.forward(PROMPT, cpu_model.new_cache()),
                               atol=1e-4, rtol=1e-4)
    assert m.prompt_gpu.calls == 0 and "out of memory" in m.prompt_gpu.failed


def test_auto_mode_leaves_out_a_gpu_that_is_not_faster(tiny, monkeypatch):
    monkeypatch.setattr(PromptGpu, "calibrate", lambda self, *a, **k: (setattr(self, "gpu_ms", 9.0),
                                                                      setattr(self, "cpu_ms", 10.0), 1.1)[-1])
    m, msgs = gpu_model(tiny, gpu_mode="auto")
    assert m.prompt_gpu is None and "not used for reading prompts" in msgs[-1]


def test_no_gpu_found_is_said_not_fatal(tiny, monkeypatch):
    def none(spec):
        raise RuntimeError("CUDA is not available to this PyTorch")
    monkeypatch.setattr(accel, "open_device", none)
    m, msgs = gpu_model(tiny)
    assert m.prompt_gpu is None and "not available" in msgs[-1]


def test_mla_attention_stays_on_the_cpu(tmp_path):
    ds = make_tiny_deepseek(tmp_path / "ds", version=3, q_lora_rank=48, yarn=True, seed=3)
    msgs = []
    m = MoEModel(ds, prompt_gpu="torch-cpu", gpu_mode="max", progress=msgs.append)
    assert m.prompt_gpu is None and "MLA" in msgs[-1]


def test_serve_runs_prompts_on_the_gpu_it_was_given(tiny, monkeypatch):
    """--gpu reaches the model, for the coordinator's own attention too."""
    import argparse
    from moelan import cli
    from moelan.server import model_info
    p = argparse.ArgumentParser()
    cli._add_cluster_args(p)
    args = p.parse_args(["--model", str(tiny), "--gpu", "torch-cpu", "--gpu-mode", "max"])
    m = cli.build_model(args)
    assert m.prompt_gpu is not None
    assert model_info(m, "tiny")["prompt_gpu"]["device"].startswith("torch-cpu")
