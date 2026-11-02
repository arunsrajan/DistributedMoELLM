"""What experts respond to, measured from routing.

Model files say nothing about what an expert "knows": experts are trained with
the router, unlabelled, and picked per token and per layer. So everything here
is observed behaviour, never a claim about training:

* :class:`TokenTracker` counts which tokens were routed to each expert. On a
  server it sees real prompts, so it only runs when asked (``--expert-tokens``).
* :func:`run_probe` reads labelled passages (``moelan probe``) and
  :func:`affinities` keeps the experts picked clearly more often for one topic
  than for the others. A passage's tokens are not independent samples, so an
  expert must also lean that way in most of the topic's passages, and the same
  test with shuffled labels gives the count expected by chance.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from threading import Lock

import numpy as np

from .config import model_fingerprint

MIN_LIFT = 1.5          # picked at least this much more often than for other topics
MIN_Z = 4.0             # two-proportion z: thousands of experts are tested
MIN_COUNT = 5           # routed tokens within the topic
MIN_CONSISTENT = 0.8    # share of the topic's passages that lean the same way
SHUFFLES = 5            # label shuffles behind the chance estimate
LIFT_CAP = 20.0         # beyond this the ratio only says the expert barely fires elsewhere


class TokenTracker:
    """Per-expert counts of routed tokens, bounded: an expert's rarest tokens are
    dropped once it holds twice ``capacity``, so its common ones survive."""

    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.experts: dict[tuple[int, int], dict[int, int]] = {}
        self.slots: dict[tuple[int, int], int] = {}
        self.totals: dict[int, int] = {}
        self.total = 0
        self._lock = Lock()

    def observe(self, layer: int, tokens: np.ndarray, idx: np.ndarray) -> None:
        """``tokens`` (n,) went to the experts ``idx`` (n, k) in this layer."""
        k = idx.shape[1]
        keys = (idx.reshape(-1).astype(np.int64) << 32) | np.repeat(np.asarray(tokens, np.int64), k)
        pairs, counts = np.unique(keys, return_counts=True)
        with self._lock:
            for key, c in zip(pairs.tolist(), counts.tolist()):
                expert, tok = key >> 32, key & 0xFFFFFFFF
                d = self.experts.setdefault((layer, expert), {})
                d[tok] = d.get(tok, 0) + c
                self.slots[(layer, expert)] = self.slots.get((layer, expert), 0) + c
                self.totals[tok] = self.totals.get(tok, 0) + c
                self.total += c
                if len(d) > 2 * self.capacity:
                    keep = sorted(d, key=d.get, reverse=True)[:self.capacity]
                    self.experts[(layer, expert)] = {t: d[t] for t in keep}

    def top(self, layer: int, expert: int, n: int = 5) -> list[tuple[int, int, float]]:
        """(token, count, lift) of the tokens most typical of this expert: seen
        often, and more often here than across all experts."""
        with self._lock:
            slots = self.slots.get((layer, expert), 0)
            rows = []
            for tok, c in self.experts.get((layer, expert), {}).items():
                lift = (c / slots) / (self.totals[tok] / self.total)
                if c >= 2 and lift > 1.0:
                    rows.append((c * math.log(lift), tok, c, lift))
        rows.sort(reverse=True)
        return [(tok, c, lift) for _, tok, c, lift in rows[:n]]


def token_rows(tracker: TokenTracker, tok, layer: int, expert: int, n: int = 5) -> list[dict]:
    """An expert's typical tokens as text; pieces of multi-byte characters are left out."""
    rows = []
    for t, c, lift in tracker.top(layer, expert, 3 * n):
        s = tok.decode([int(t)], keep_special=True)
        if s and "�" not in s:
            rows.append({"text": s, "count": c, "lift": round(lift, 2)})
    return rows[:n]


def affinities(counts: np.ndarray, tokens: np.ndarray, labels: np.ndarray, topic: int) -> dict:
    """Experts leaning to one topic. ``counts`` (passages, layers, experts) of
    routed tokens, ``tokens`` (passages) of routing slots per layer, ``labels``
    (passages) the topic of each passage. Returns {(layer, expert): (lift, z)}."""
    inside = labels == topic
    if not inside.any() or inside.all():
        return {}
    c_in, c_out = counts[inside].sum(0, dtype=np.float64), counts[~inside].sum(0, dtype=np.float64)
    n_in, n_out = float(tokens[inside].sum()), float(tokens[~inside].sum())
    p_in, p_out = c_in / n_in, c_out / n_out
    pool = (c_in + c_out) / (n_in + n_out)
    se = np.sqrt(pool * (1 - pool) * (1 / n_in + 1 / n_out))
    z = np.divide(p_in - p_out, se, out=np.zeros_like(se), where=se > 0)
    lift = (c_in + 0.5) / (n_in + 0.5) / ((c_out + 0.5) / (n_out + 0.5))
    per_passage = counts[inside] / tokens[inside][:, None, None]
    steady = (per_passage > p_out).mean(0) >= MIN_CONSISTENT
    keep = (z >= MIN_Z) & (lift >= MIN_LIFT) & (c_in >= MIN_COUNT) & steady
    return {(int(l), int(e)): (min(float(lift[l, e]), LIFT_CAP), float(z[l, e])) for l, e in zip(*np.nonzero(keep))}


def summarize(counts: np.ndarray, tokens: np.ndarray, labels: np.ndarray, names: list[str],
              seed: int = 0) -> dict:
    """The per-topic result, with the number of experts shuffled labels pass."""
    rng = np.random.default_rng(seed)
    shuffles = [rng.permutation(labels) for _ in range(SHUFFLES)]
    topics, table = [], {}
    for t, name in enumerate(names):
        found = affinities(counts, tokens, labels, t)
        for key, (lift, _) in found.items():
            table.setdefault(key, []).append((t, lift))
        best = max(found.items(), key=lambda kv: kv[1][1], default=None)     # the strongest evidence
        chance = np.mean([len(affinities(counts, tokens, s, t)) for s in shuffles])
        topics.append({"name": name, "passages": int((labels == t).sum()),
                       "tokens": int(tokens[labels == t].sum()), "experts": len(found),
                       "chance": round(float(chance), 1),
                       "best": {"layer": best[0][0], "expert": best[0][1], "lift": round(best[1][0], 2)} if best else None})
    return {"topics": topics,
            "affinity": {f"{l}:{e}": [[t, round(v, 2)] for t, v in sorted(rows, key=lambda r: -r[1])]
                         for (l, e), rows in table.items()}}


def run_probe(model, tok, passages: dict[str, list[str]], progress=print) -> dict:
    """Read every passage, record its routing, and summarize. The passages are
    fed as plain text, so no chat template is counted as part of a topic."""
    cfg = model.cfg
    names = list(passages)
    items = [(t, text) for t, name in enumerate(names) for text in passages[name]]
    counts = np.zeros((len(items), cfg.num_layers, cfg.num_experts), np.int32)
    tokens = np.zeros(len(items), np.float64)
    labels = np.array([t for t, _ in items], np.int32)
    tracker = TokenTracker()
    current = 0

    def on_route(layer, toks, idx):
        np.add.at(counts[current, layer], idx.reshape(-1), 1)
        tracker.observe(layer, toks, idx)

    before, model.on_route = model.on_route, on_route
    t0, done = time.perf_counter(), 0
    try:
        for current, (t, text) in enumerate(items):
            ids = tok.encode(text)
            model.forward(ids, model.new_cache())
            tokens[current] = len(ids) * cfg.top_k
            done += len(ids)
            spent = time.perf_counter() - t0
            left = spent / (current + 1) * (len(items) - current - 1)
            progress(f"[{current + 1}/{len(items)}] {names[t]}: {len(ids)} tokens, "
                     f"{done / spent:.1f} tok/s, about {left / 60:.0f} min left")
    finally:
        model.on_route = before
    out = summarize(counts, tokens, labels, names)
    rows = {}
    for layer, expert in sorted(tracker.experts):
        r = token_rows(tracker, tok, layer, expert)
        if r:
            rows[f"{layer}:{expert}"] = r
    out.update(tested=len(cfg.moe_layers()) * cfg.num_experts, tokens=rows,
               seconds=round(time.perf_counter() - t0, 1),
               thresholds={"lift": MIN_LIFT, "z": MIN_Z, "consistent": MIN_CONSISTENT})
    return out


def has_signal(topic: dict) -> bool:
    """More experts lean to this topic than shuffled labels would give."""
    return topic["experts"] >= max(3, 3 * topic["chance"])


def topics_path(model_dir, root=None) -> Path:
    base = Path(root) if root else Path(os.environ.get("MOELAN_TOPICS_DIR") or Path.home() / ".moelan" / "topics")
    return base / f"{model_fingerprint(model_dir)}.json"


def save_topics(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, path)


def load_topics(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
