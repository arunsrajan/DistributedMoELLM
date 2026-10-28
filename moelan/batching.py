"""Continuous batching: several generations advance together.

One scheduler thread owns the model. Each loop it admits waiting requests
(their prompts go through one batched forward pass), then advances every
running request by one token in a single batched decode step. In a cluster
that step is one request per worker per MoE layer for the whole batch, so the
48 network round trips per step are shared by all running requests, and an
expert that several requests picked decodes its weights once for all of them.

Requests join and leave between steps; a long prompt being read delays the
other requests' next token by that prefill.

A prompt cache keeps the KV caches of recent conversations. A new prompt that
starts like one of them (the same system prompt and tool list, or the next
turn of the same chat) copies the shared part and only reads the rest, which
matters for agents whose tool definitions run to hundreds of tokens.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from .model import Interrupted, MoEModel, sample

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
        self.cached = 0       # prompt tokens reused from the prompt cache
        self.fed: list[int] = []
        self.last = None
        self.generated = 0
        self.done = False
        self.cancelled = False   # set by the HTTP side when the client goes away

    def tokens(self, idle=None, every: float = 10.0):
        """Yield generated ids until the request finishes; re-raise model errors.
        ``idle()`` is called every ``every`` seconds without a token (a long
        prompt being read), e.g. to keep a streaming connection alive."""
        while True:
            try:
                item = self.out.get(timeout=every if idle else None)
            except queue.Empty:
                idle()
                continue
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


class PromptCache:
    """KV caches of finished conversations, keyed by the tokens they hold.

    In memory, least recently used entries go first once ``max_tokens`` is
    exceeded. With ``disk_dir``, long entries (``disk_min`` tokens and up, such
    as a coding agent's system prompt) are also written to disk in the
    background and found again after a restart, so a prompt that takes minutes
    to read is read once, not once per server start. Disk use is capped at
    ``disk_max_bytes``, oldest entries deleted first."""

    def __init__(self, max_tokens: int = 16384, min_match: int = 8, disk_dir=None,
                 disk_max_bytes: int = 0, new_cache=None, disk_min: int = 512):
        self.max_tokens = max_tokens
        self.min_match = min_match
        self.entries: list[tuple[np.ndarray, object]] = []   # most recently used last
        self.lock = threading.Lock()
        self.hits = 0
        self.disk_hits = 0
        self.lookups = 0
        self.tokens_saved = 0
        self.new_cache = new_cache
        self.disk_min = disk_min
        self.disk_max = disk_max_bytes
        self.disk_dir = Path(disk_dir) if disk_dir and disk_max_bytes > 0 and new_cache else None
        self.disk: list[tuple[np.ndarray, Path]] = []
        self._writer = None
        if self.disk_dir is not None:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            from concurrent.futures import ThreadPoolExecutor
            self._writer = ThreadPoolExecutor(1, thread_name_prefix="moelan-kvdisk")
            for d in sorted(self.disk_dir.iterdir(), key=lambda d: d.stat().st_mtime):
                try:
                    self.disk.append((np.load(d / "tokens.npy"), d))
                except (OSError, ValueError):
                    shutil.rmtree(d, ignore_errors=True)     # half-written entry

    @staticmethod
    def _common(a: np.ndarray, toks: np.ndarray, limit: int) -> int:
        n = min(limit, len(toks))
        diff = np.flatnonzero(a[:n] != toks[:n])
        return int(diff[0]) if diff.size else n

    def lookup(self, ids):
        """(cache holding the longest cached prefix of ``ids``, its length) or
        (None, 0). At least one prompt token is always left to compute."""
        a = np.asarray(ids, dtype=np.int64)
        with self.lock:
            self.lookups += 1
            best, best_len, where = None, 0, None
            for i, (toks, _) in enumerate(self.entries):
                common = self._common(a, toks, len(a) - 1)
                if common > best_len:
                    best, best_len, where = i, common, "mem"
            for i, (toks, _) in enumerate(self.disk):
                common = self._common(a, toks, len(a) - 1)
                if common > best_len:
                    best, best_len, where = i, common, "disk"
            if best is None or best_len < self.min_match:
                return None, 0
            if where == "mem":
                entry = self.entries.pop(best)
                self.entries.append(entry)
                cache = entry[1].clone(best_len)
            else:
                cache = self._load(self.disk[best][1], best_len)
                if cache is None:
                    return None, 0
                os.utime(self.disk[best][1])            # most recently used
                self.disk_hits += 1
            self.hits += 1
            self.tokens_saved += best_len
            return cache, best_len

    def store(self, tokens, cache) -> None:
        if self.max_tokens <= 0:
            return
        toks = np.asarray(tokens, dtype=np.int64)[:cache.length]
        if len(toks) < self.min_match or len(toks) > self.max_tokens:
            return
        compact = cache.clone(len(toks))
        with self.lock:
            # an entry that is a prefix of the new one is now redundant
            self.entries = [(t, c) for t, c in self.entries
                            if not (len(t) <= len(toks) and np.array_equal(toks[:len(t)], t))]
            self.entries.append((toks, compact))
            while sum(len(t) for t, _ in self.entries) > self.max_tokens:
                self.entries.pop(0)
        if self._writer is not None and len(toks) >= self.disk_min:
            self._writer.submit(self._save, toks, compact)

    # ---- disk -----------------------------------------------------------
    def _save(self, toks: np.ndarray, cache) -> None:
        try:
            d = self.disk_dir / hashlib.sha1(toks.tobytes()).hexdigest()[:20]
            if d.exists():
                return
            tmp = d.with_suffix(".part")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            n = len(toks)
            for name, arrs in (("k", cache.k), ("v", cache.v)):
                h, _, dim = arrs[0].shape
                mm = np.lib.format.open_memmap(tmp / f"{name}.npy", mode="w+", dtype=np.float32,
                                               shape=(len(arrs), h, n, dim))
                for i, layer in enumerate(arrs):
                    mm[i] = layer[:, :n]
                mm.flush()
                del mm
            np.save(tmp / "tokens.npy", toks)           # written last: marks the entry complete
            tmp.rename(d)
            with self.lock:
                # an entry that is a prefix of the new one is now redundant
                for t, path in [e for e in self.disk if len(e[0]) <= n and np.array_equal(toks[:len(e[0])], e[0])]:
                    shutil.rmtree(path, ignore_errors=True)
                self.disk = [e for e in self.disk if e[1].exists()]
                self.disk.append((toks, d))
                while len(self.disk) > 1 and self._disk_bytes() > self.disk_max:
                    shutil.rmtree(self.disk.pop(0)[1], ignore_errors=True)
        except OSError:
            log.exception("could not write the prompt cache to disk")

    def _disk_bytes(self) -> int:
        return sum(f.stat().st_size for _, d in self.disk for f in d.glob("*.npy") if f.exists())

    def _load(self, d: Path, length: int):
        try:
            k = np.load(d / "k.npy", mmap_mode="r")
            v = np.load(d / "v.npy", mmap_mode="r")
        except (OSError, ValueError):
            return None
        c = self.new_cache()
        c.reserve(length)
        for i in range(len(c.k)):
            c.k[i][:, :length] = k[i, :, :length]
            c.v[i][:, :length] = v[i, :, :length]
        c.length = length
        return c

    def stats(self) -> dict:
        with self.lock:
            return {"entries": len(self.entries), "tokens": int(sum(len(t) for t, _ in self.entries)),
                    "max_tokens": self.max_tokens, "hits": self.hits, "lookups": self.lookups,
                    "tokens_saved": self.tokens_saved, "disk_entries": len(self.disk),
                    "disk_hits": self.disk_hits,
                    "disk_bytes": self._disk_bytes() if self.disk_dir else 0}


class Batcher(threading.Thread):
    def __init__(self, model: MoEModel, telemetry=None, max_batch: int = 4,
                 prompt_cache_tokens: int = 16384, prompt_cache_dir=None,
                 prompt_cache_disk: int = 0):
        super().__init__(daemon=True, name="moelan-batcher")
        self.model = model
        self.tel = telemetry
        self.max_batch = max(1, int(max_batch))
        self.pending: queue.Queue = queue.Queue()
        self.running: list[GenRequest] = []
        self.steps = 0
        self.batch_sizes = 0  # sum over steps, for the average batch size
        disk = None
        if prompt_cache_disk > 0:
            from .config import model_fingerprint
            root = Path(prompt_cache_dir or Path.home() / ".moelan" / "kvcache")
            disk = root / f"{model_fingerprint(model.model_dir)}-{model.dense_format}"
        self.pcache = PromptCache(prompt_cache_tokens, disk_dir=disk, disk_max_bytes=prompt_cache_disk,
                                  new_cache=model.new_cache)

    def submit(self, req: GenRequest) -> GenRequest:
        self.pending.put(req)
        return req

    # ---- scheduler loop ---------------------------------------------------
    def run(self) -> None:
        # The scheduler must outlive any single failure: if this thread died the
        # server would keep accepting connections and never answer them.
        while True:
            try:
                self._step()
            except BaseException:   # noqa: BLE001 - log, drop the batch, keep scheduling
                log.exception("scheduler step failed; continuing")
                for r in self.running:
                    if not r.done:
                        self._finish(r, RuntimeError("internal error in the scheduler"))
                self.running = []
                time.sleep(0.1)

    def _step(self) -> None:
        new = []
        if not self.running:
            new.append(self.pending.get())        # idle: wait for work
        while len(self.running) + len(new) < self.max_batch:
            try:
                new.append(self.pending.get_nowait())
            except queue.Empty:
                break
        for r in new:
            if r.cancelled:     # the client left while queued
                self._finish(r)
        new = [r for r in new if not r.done]
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
        todo = []
        for r in reqs:
            cache, hit = self.pcache.lookup(r.ids)
            r.cache = cache if cache is not None else self.model.new_cache()
            r.cached = hit
            r.fed = list(r.ids)
            todo.append(r.ids[hit:])
            if self.tel:
                self.tel.admit(r.rid, cached=hit)
        def progress(done):
            if self.tel:
                for r, d in zip(reqs, done):
                    self.tel.progress(r.rid, r.cached + d)
        try:
            logits = self.model.forward_batch(
                todo, [r.cache for r in reqs], on_progress=progress,
                should_stop=lambda: all(r.cancelled for r in reqs))   # checked after every layer
        except Interrupted:
            # Every client went away. Keep the passes already read: a client
            # that retries resumes from there instead of starting over.
            for r in reqs:
                if r.cache is not None and r.cache.length > r.cached:
                    self.pcache.store(r.ids[:r.cache.length], r.cache)
                r.cache = None
                self._finish(r)
            return
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
        for r in live:
            r.fed.append(r.last)
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
        if error is None and r.cache is not None:
            try:
                self.pcache.store(r.fed, r.cache)
            except Exception:  # the cache is an optimisation; never fail a request over it
                log.exception("prompt cache store failed")
        r.cache = None  # free the KV cache now, not when the client reads the end
        r.out.put(error)
        if self.tel:
            self.tel.end(r.rid)

    def stats(self) -> dict:
        return {"max_batch": self.max_batch, "running": len(self.running),
                "waiting": self.pending.qsize(), "steps": self.steps,
                "avg_batch": self.batch_sizes / self.steps if self.steps else 0.0,
                "prompt_cache": self.pcache.stats()}
