"""Continuous batching: several generations advance together.

One scheduler thread owns the model. Each loop it admits waiting requests
(their prompts go through one batched forward pass), then advances every
running request by one token in a single batched decode step. In a cluster
that step is one request per worker per MoE layer for the whole batch, so the
48 network round trips per step are shared by all running requests, and an
expert that several requests picked decodes its weights once for all of them.

Requests join and leave between steps; a long prompt being read delays the
other requests' next token by that prefill.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np

from .model import MoEModel, sample

log = logging.getLogger("moelan.batching")


class GenRequest:
    """One generation. The scheduler puts token ids on ``out``, then ``None``
    when finished, or the exception if the model failed."""

    def __init__(self, rid: str, prompt_ids, max_new_tokens: int = 256, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0, seed=None, stop_ids=()):
        if not len(prompt_ids):
            raise ValueError("empty prompt")
        self.rid = rid
        self.ids = list(prompt_ids)
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.rng = np.random.default_rng(seed)
        self.stop = set(stop_ids)
        self.out: queue.Queue = queue.Queue()
        self.cache = None
        self.last = None
        self.generated = 0
        self.done = False
        self.cancelled = False   # set by the HTTP side when the client goes away

    def tokens(self):
        """Yield generated ids until the request finishes; re-raise model errors."""
        while True:
            item = self.out.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


class Batcher(threading.Thread):
    def __init__(self, model: MoEModel, telemetry=None, max_batch: int = 4):
        super().__init__(daemon=True, name="moelan-batcher")
        self.model = model
        self.tel = telemetry
        self.max_batch = max(1, int(max_batch))
        self.pending: queue.Queue = queue.Queue()
        self.running: list[GenRequest] = []
        self.steps = 0
        self.batch_sizes = 0  # sum over steps, for the average batch size

    def submit(self, req: GenRequest) -> GenRequest:
        self.pending.put(req)
        return req

    # ---- scheduler loop ---------------------------------------------------
    def run(self) -> None:
        while True:
            new = []
            if not self.running:
                new.append(self.pending.get())        # idle: wait for work
            while len(self.running) + len(new) < self.max_batch:
                try:
                    new.append(self.pending.get_nowait())
                except queue.Empty:
                    break
            new = [r for r in new if not r.cancelled]
            try:
                if new:
                    self._prefill(new)
                self.running = [r for r in self.running + new if not r.done]
                if self.running:
                    self._decode(self.running)
                self.running = [r for r in self.running if not r.done]
            except Exception as exc:  # fail the affected requests, keep serving
                log.exception("batched step failed")
                for r in self.running + new:
                    if not r.done:
                        self._finish(r, exc)
                self.running = []

    def _prefill(self, reqs: list[GenRequest]) -> None:
        for r in reqs:
            r.cache = self.model.new_cache()
            if self.tel:
                self.tel.admit(r.rid)
        logits = self.model.forward_batch([r.ids for r in reqs], [r.cache for r in reqs])
        for r, lg in zip(reqs, logits):
            self._emit(r, lg)

    def _decode(self, reqs: list[GenRequest]) -> None:
        live = [r for r in reqs if not r.done]
        for r in live:
            if r.cancelled:
                self._finish(r)
        live = [r for r in live if not r.done]
        if not live:
            return
        logits = self.model.forward_batch([[r.last] for r in live], [r.cache for r in live])
        self.steps += 1
        self.batch_sizes += len(live)
        for r, lg in zip(live, logits):
            self._emit(r, lg)

    def _emit(self, r: GenRequest, logits: np.ndarray) -> None:
        tok = sample(logits, r.temperature, r.top_p, r.top_k, r.rng)
        r.generated += 1
        r.last = tok
        if self.tel:
            self.tel.token(r.rid)
        r.out.put(tok)
        if tok in r.stop or r.generated >= r.max_new_tokens:
            self._finish(r)

    def _finish(self, r: GenRequest, error: BaseException | None = None) -> None:
        r.done = True
        r.cache = None  # free the KV cache now, not when the client reads the end
        r.out.put(error)
        if self.tel:
            self.tel.end(r.rid)

    def stats(self) -> dict:
        return {"max_batch": self.max_batch, "running": len(self.running),
                "waiting": self.pending.qsize(), "steps": self.steps,
                "avg_batch": self.batch_sizes / self.steps if self.steps else 0.0}
