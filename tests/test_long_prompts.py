"""Agent prompts: long, sent again whole at every step, and read slowly.

A coding agent's prompt ran to 18,169 tokens and took about 20 minutes to
read on a two-laptop cluster. The prompt cache skipped conversations longer
than --prompt-cache-tokens altogether (memory and disk), so every agent step
read it all again; and while it was read, a request that was generating got
no tokens at all."""

import threading

import numpy as np
import pytest

from moelan.batching import Batcher, GenRequest, PromptCache
from moelan.model import MoEModel
from moelan.tiny import make_tiny


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("tiny"), layers=3, experts=8, top_k=2)


@pytest.fixture(scope="module")
def model(tiny):
    return MoEModel(tiny)


def greedy(model, prompt, n):
    return list(model.generate(prompt, max_new_tokens=n, stop_ids=[]))


def conversation(m, n, seed):
    toks = [256] + [int(t) for t in np.random.default_rng(seed).integers(0, 256, n - 1)]
    cache = m.new_cache()
    m.forward(toks, cache)
    return toks, cache


def assert_continues_like_fresh(m, longer, cache, hit):
    want = m.forward(longer, m.new_cache())
    np.testing.assert_allclose(m.forward(longer[hit:], cache), want, atol=1e-5)


# --- the prompt cache ----------------------------------------------------------------

def test_a_conversation_longer_than_memory_keeps_its_start_there(model):
    toks, c = conversation(model, 100, 1)
    pc = PromptCache(64, new_cache=model.new_cache)
    pc.store(toks, c)
    assert [len(t) for t, _ in pc.entries] == [64]
    longer = toks + [65, 66]
    cache, hit = pc.lookup(longer)
    assert hit == 64                          # only the rest is read again
    assert_continues_like_fresh(model, longer, cache, hit)


def test_a_long_conversation_goes_to_disk_whole(model, tmp_path):
    toks, c = conversation(model, 100, 2)
    pc = PromptCache(64, disk_dir=tmp_path, disk_max_bytes=2**30, new_cache=model.new_cache, disk_min=8)
    pc.store(toks, c)
    pc._writer.shutdown(wait=True)
    assert [len(t) for t, _ in pc.disk] == [100] and not pc.writing
    longer = toks + [70, 71, 72]
    cache, hit = pc.lookup(longer)
    assert hit == 100 and pc.disk_hits == 1
    assert_continues_like_fresh(model, longer, cache, hit)


def test_the_next_step_does_not_wait_for_the_disk(model, tmp_path, monkeypatch):
    """An agent sends its next step a moment after the last one ended, often
    before a long conversation is on disk: lookups copy it from memory."""
    gate = threading.Event()
    save = PromptCache._save

    def slow_save(self, toks, cache):
        gate.wait(10)
        save(self, toks, cache)
    monkeypatch.setattr(PromptCache, "_save", slow_save)
    toks, c = conversation(model, 100, 3)
    pc = PromptCache(64, disk_dir=tmp_path, disk_max_bytes=2**30, new_cache=model.new_cache, disk_min=8)
    pc.store(toks, c)
    assert len(pc.writing) == 1 and not pc.disk
    longer = toks + [80]
    cache, hit = pc.lookup(longer)
    assert hit == 100 and pc.disk_hits == 0
    assert_continues_like_fresh(model, longer, cache, hit)
    gate.set()
    pc._writer.shutdown(wait=True)
    assert not pc.writing and [len(t) for t, _ in pc.disk] == [100]


def test_a_conversation_bigger_than_the_disk_budget_stays_off_disk(model, tmp_path):
    toks, c = conversation(model, 100, 4)
    per_token = sum(a.shape[0] * a.shape[2] * a.itemsize for a in (*c.k, *c.v))
    pc = PromptCache(64, disk_dir=tmp_path, disk_max_bytes=50 * per_token, new_cache=model.new_cache,
                     disk_min=8)
    pc.store(toks, c)
    pc._writer.shutdown(wait=True)
    assert not pc.disk and not pc.writing
    assert [len(t) for t, _ in pc.entries] == [64]       # its start is still kept in memory


# --- the scheduler ---------------------------------------------------------------------

SHORT = [256] + list(b"hello there")
LONG = [256] + [int(t) for t in np.random.default_rng(6).integers(0, 256, 299)]


class Log:
    """Telemetry that records the order of events."""

    def __init__(self):
        self.events = []

    def admit(self, rid, cached=0):
        self.events.append(("admit", rid))

    def token(self, rid):
        self.events.append(("token", rid))

    def progress(self, rid, read):
        pass

    def end(self, rid):
        self.events.append(("end", rid))


@pytest.fixture
def passes(monkeypatch):
    """Prompts read in passes of 37 tokens (9 passes of 3 layers for LONG),
    and every layer counts as long enough to let others run."""
    monkeypatch.setattr(MoEModel, "PREFILL_CHUNK", 37)
    monkeypatch.setattr(Batcher, "SLICE_S", 0.0)


def run_out(b):
    while b.running:
        b._step()


def test_a_request_keeps_generating_while_a_long_prompt_is_read(model, passes):
    want_a, want_b = greedy(model, SHORT, 60), greedy(model, LONG, 6)
    tel = Log()
    b = Batcher(model, tel, max_batch=4)
    a = b.submit(GenRequest("a", SHORT, max_new_tokens=60))
    b._step()                                         # a is generating
    long_req = b.submit(GenRequest("b", LONG, max_new_tokens=6))
    b._step()                                         # b's prompt is read
    ev = tel.events
    during = ev[ev.index(("admit", "b")):ev.index(("token", "b"))]
    assert during.count(("token", "a")) >= 9          # between the layers of b's passes
    run_out(b)
    assert list(a.tokens()) == want_a and list(long_req.tokens()) == want_b


def test_a_short_request_is_answered_while_a_long_prompt_is_read(model, passes):
    want_b, want_c = greedy(model, LONG, 4), greedy(model, SHORT, 8)
    c = GenRequest("c", SHORT, max_new_tokens=8)

    class Arrives(Log):        # c comes in once b's prompt is being read
        def progress(self, rid, read):
            if rid == "b" and not self.events.count(("sent", "c")):
                self.events.append(("sent", "c"))
                b.submit(c)
    tel = Arrives()
    b = Batcher(model, tel, max_batch=4)
    long_req = b.submit(GenRequest("b", LONG, max_new_tokens=4))
    b._step()
    ev = tel.events
    assert ev.index(("end", "c")) < ev.index(("token", "b"))     # c done before b's first token
    run_out(b)
    assert list(long_req.tokens()) == want_b and list(c.tokens()) == want_c


def test_a_full_batch_admits_nobody_meanwhile(model, passes):
    c = GenRequest("c", SHORT, max_new_tokens=3)

    class Arrives(Log):
        def progress(self, rid, read):
            if rid == "b" and not self.events.count(("sent", "c")):
                self.events.append(("sent", "c"))
                b.submit(c)
    tel = Arrives()
    b = Batcher(model, tel, max_batch=1)
    long_req = b.submit(GenRequest("b", LONG, max_new_tokens=2))
    b._step()
    assert ("admit", "c") not in tel.events
    run_out(b)
    b._step()                                         # then c, in its own step
    run_out(b)
    assert len(list(long_req.tokens())) == 2 and list(c.tokens()) == greedy(model, SHORT, 3)
