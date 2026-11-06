"""One compiled call per MoE layer for int4 experts while generating: it must
give what one call per expert gives."""

import numpy as np
import pytest

from moelan import kernels
from moelan.config import load_config
from moelan.experts import CheckpointSource, Expert, ExpertStore
from moelan.model import MoEModel
from moelan.tiny import make_tiny

pytestmark = pytest.mark.skipif(not kernels.ENABLED, reason="needs the numba kernels")


@pytest.fixture(scope="module")
def col(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("fused") / "c", layers=3, experts=8, top_k=2,
                     layout="colibri_int4", int4_gs=32)


@pytest.fixture
def fused_off(monkeypatch):
    def off(flag=False):
        monkeypatch.setattr(kernels, "FUSED", flag)
    return off


def groups_for(rng, n_rows, picks):
    """(expert, rows, weights) as the model hands them over, rows ascending."""
    out = []
    for e, rows in picks:
        out.append((e, np.array(rows), rng.random(len(rows)).astype(np.float32)))
    return out


def test_layer_call_matches_one_call_per_expert(col, fused_off):
    cfg = load_config(col)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, cfg.hidden_size)).astype(np.float32)
    # one row, several rows, rows shared between experts, and one expert with
    # enough rows (5) that it takes the per-expert path next to the fused ones
    groups = groups_for(rng, 6, [(0, [0]), (2, [0, 3]), (4, [1, 2, 3]), (5, [0, 1, 2, 4, 5]), (7, [5])])
    got = {}
    for flag in (False, True):
        fused_off(flag)
        store = ExpertStore([CheckpointSource(col, cfg)], fmt="int4")
        got[flag] = store.compute(1, x, groups)
        if flag:
            few = [e for e, rows, _ in groups if len(rows) < kernels.ROW_DECODE_FROM]
            assert all(store.get((1, e)).fused for e in few)              # addresses cached on the expert
            assert store.get((1, 5)).fused is None                        # 5 rows: its own path
    np.testing.assert_allclose(got[True], got[False], rtol=1e-5, atol=1e-5)
    assert np.abs(got[False]).max() > 0.1


def test_layer_call_reads_resident_copies_too(col):
    """Experts copied into RAM in offset form (XOR-ed once) and memory-mapped ones together."""
    cfg = load_config(col)
    src = CheckpointSource(col, cfg)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, cfg.hidden_size)).astype(np.float32)
    mapped = src.load((0, 3))
    copy = src.load((0, 6))
    copy = Expert(copy.gu.resident(), copy.down.resident(), copy.inter)
    assert copy.gu.offset and not mapped.gu.offset
    w = np.array([0.25, 0.75], np.float32)
    out = np.zeros_like(x)
    items = [(kernels.q4_expert_ptrs(mapped), np.array([0, 1]), w), (kernels.q4_expert_ptrs(copy), np.array([1]), w[:1])]
    kernels.moe_q4(x, items, cfg.hidden_size, cfg.moe_intermediate_size, mapped.gu.gs, out)
    want = np.zeros_like(x)
    want[[0, 1]] += mapped(x[[0, 1]]) * w[:, None]
    want[[1]] += copy(x[[1]]) * w[:1, None]
    np.testing.assert_allclose(out, want, rtol=1e-5, atol=1e-5)


def test_generation_is_the_same_with_and_without(col, fused_off):
    prompt = [256] + list(b"fused experts")
    runs = {}
    for flag in (False, True):
        fused_off(flag)
        m = MoEModel(col, expert_format="int4")
        logits = m.forward(prompt, m.new_cache())
        runs[flag] = (list(m.generate(prompt, max_new_tokens=10, stop_ids=[])), logits)
    assert runs[True][0] == runs[False][0]
    np.testing.assert_allclose(runs[True][1], runs[False][1], rtol=1e-4, atol=1e-4)


@pytest.fixture(scope="module")
def col64(tmp_path_factory):
    """Groups of 64, the size the int8-activation kernel takes (as Qwen3-Coder's)."""
    return make_tiny(tmp_path_factory.mktemp("fused64") / "c", hidden=128, moe_inter=128, layers=3,
                     experts=8, top_k=2, layout="colibri_int4", int4_gs=64)


@pytest.fixture
def int8_act(monkeypatch):
    def set_(flag):
        monkeypatch.setattr(kernels, "INT8_ACT", flag)
    return set_


needs_avx2 = pytest.mark.skipif(not kernels.AVX2, reason="the int8-activation kernel needs AVX2")


def assert_close_to_float(want, got):
    """Rounding activations to int8 costs ~1-2% (rms) on random weights, ~0.8% on real ones."""
    assert np.linalg.norm(got - want) < 3e-2 * np.linalg.norm(want)
    assert float((want * got).sum() / np.sqrt((want * want).sum() * (got * got).sum())) > 0.999
    for r in range(want.shape[0]):           # every row, not just the total
        if np.abs(want[r]).max() > 0:
            assert np.linalg.norm(got[r] - want[r]) < 5e-2 * np.linalg.norm(want[r])


@needs_avx2
def test_int8_activations_stay_close_to_float(col64, int8_act):
    cfg = load_config(col64)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, cfg.hidden_size)).astype(np.float32)
    groups = groups_for(rng, 6, [(0, [0]), (2, [0, 3]), (4, [1, 2, 3]), (5, [0, 1, 2, 4, 5]), (7, [5])])
    assert kernels.int8_act_for(cfg.hidden_size, cfg.moe_intermediate_size, 64)
    got = {}
    for flag in (False, True):
        int8_act(flag)
        got[flag] = ExpertStore([CheckpointSource(col64, cfg)], fmt="int4").compute(1, x, groups)
    assert_close_to_float(got[False], got[True])


@needs_avx2
def test_int8_activations_with_many_rows(col64, int8_act):
    """Eight rows, each with its own experts: the call's int8 buffers are sized
    by rows, and must live until the kernel has read them."""
    cfg = load_config(col64)
    src = CheckpointSource(col64, cfg)
    rng = np.random.default_rng(6)
    x = rng.standard_normal((8, cfg.hidden_size)).astype(np.float32)
    experts = {i: src.load((2, i)) for i in range(8)}
    items = [(kernels.q4_expert_ptrs(experts[i]), np.array(sorted({i, (i + 3) % 8})), rng.random(2).astype(np.float32))
             for i in range(8)]
    got = {}
    for flag in (False, True):
        int8_act(flag)
        got[flag] = np.zeros_like(x)
        kernels.moe_q4(x, items, cfg.hidden_size, cfg.moe_intermediate_size, 64, got[flag])
    assert_close_to_float(got[False], got[True])


@needs_avx2
def test_int8_activations_read_resident_copies_too(col64, int8_act):
    cfg = load_config(col64)
    src = CheckpointSource(col64, cfg)
    x = np.random.default_rng(4).standard_normal((2, cfg.hidden_size)).astype(np.float32)
    mapped, copy = src.load((0, 3)), src.load((0, 6))
    copy = Expert(copy.gu.resident(), copy.down.resident(), copy.inter)
    assert copy.gu.offset and not mapped.gu.offset
    w = np.array([0.25, 0.75], np.float32)
    items = [(kernels.q4_expert_ptrs(mapped), np.array([0, 1]), w), (kernels.q4_expert_ptrs(copy), np.array([1]), w[:1])]
    got = {}
    for flag in (False, True):
        int8_act(flag)
        got[flag] = np.zeros_like(x)
        kernels.moe_q4(x, items, cfg.hidden_size, cfg.moe_intermediate_size, 64, got[flag])
    assert_close_to_float(got[False], got[True])


def test_int8_activations_off_is_exact(col64, fused_off, int8_act):
    """MOELAN_INT8_ACT=0: the layer call gives what one call per expert gives."""
    cfg = load_config(col64)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, cfg.hidden_size)).astype(np.float32)
    groups = groups_for(rng, 3, [(1, [0]), (3, [0, 2]), (6, [1])])
    int8_act(False)
    got = {}
    for flag in (False, True):
        fused_off(flag)
        got[flag] = ExpertStore([CheckpointSource(col64, cfg)], fmt="int4").compute(2, x, groups)
    np.testing.assert_allclose(got[True], got[False], rtol=1e-5, atol=1e-5)


def test_groups_other_than_64s_keep_float_activations(col):
    """The int8-activation kernel takes groups of 64 (or multiples); the
    group-32 fixture stays on the float kernel, exact (as the first test shows)."""
    cfg = load_config(col)
    assert not kernels.int8_act_for(cfg.hidden_size, cfg.moe_intermediate_size, 32)


@needs_avx2
def test_generation_with_int8_activations_tracks_float(col64, int8_act):
    prompt = [256] + list(b"int8 activations")
    logits = {}
    for flag in (False, True):
        int8_act(flag)
        m = MoEModel(col64, expert_format="int4")
        logits[flag] = m.forward(prompt, m.new_cache())
    a, b = logits[False].ravel(), logits[True].ravel()
    assert float(a @ b / np.sqrt((a @ a) * (b @ b))) > 0.999


def test_other_formats_keep_their_own_path(col, fused_off):
    cfg = load_config(col)
    store = ExpertStore([CheckpointSource(col, cfg)], fmt="int8")
    x = np.random.default_rng(2).standard_normal((1, cfg.hidden_size)).astype(np.float32)
    y = store.compute(0, x, [(1, np.array([0]), np.array([1.0], np.float32))])
    assert store.get((0, 1)).fused is None and np.abs(y).max() > 0
