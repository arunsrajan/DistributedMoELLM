"""Where placement's routing statistics come from, and keeping them.

Placement wants to know how often each expert will be picked. Three sources, best
first: real traffic saved by ``serve`` (:class:`TrafficRecorder`), what a topic
probe measured (:func:`run_topic_probe`, for the first start), and colibri's
``.coli_usage``. Only counts per expert are stored, never prompt text.

Experts answer to the form of the text as much as to its subject, so a topic
probe mixes prose, code, how-tos and conversation, and real traffic replaces it
as soon as there is enough.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

from .config import model_fingerprint
from .model import load_heat

log = logging.getLogger("moelan.heat")

MIN_TRAFFIC_TOKENS = 200     # below this a topic probe, if any, still decides
HALF_LIFE_TOKENS = 20000     # older traffic counts half after this many new tokens
SAVE_EVERY_S = 60.0

KINDS = ["Write a short explanatory passage about {sub} in {topic}.",
         "Write a short realistic code example with brief comments that illustrates {sub} in {topic}.",
         "Write a short practical how-to with concrete steps for {sub} in {topic}.",
         "Write a short technical conversation between two engineers about {sub} in {topic}."]


def heat_dir(root=None) -> Path:
    return Path(root) if root else Path(os.environ.get("MOELAN_HEAT_DIR") or Path.home() / ".moelan" / "heat")


def traffic_path(model_dir, root=None) -> Path:
    return heat_dir(root) / f"{model_fingerprint(model_dir)}.traffic.json"


def topic_path(model_dir, root=None) -> Path:
    return heat_dir(root) / f"{model_fingerprint(model_dir)}.topic.json"


def tokens_of(heat, cfg) -> int:
    """Tokens behind routing counts: each token makes top_k picks in every MoE layer."""
    slots = len(cfg.moe_layers()) * cfg.top_k
    return int(sum(heat.values()) / slots) if slots else 0


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, path)


def load_topic_heat(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _parse_heat(raw: dict) -> Counter:
    return Counter({tuple(int(v) for v in k.split(":")): c for k, c in raw.items()})


def choose_heat(model_dir, cfg, explicit=None, progress=print) -> Counter:
    """The routing counts placement should use, saying which it chose."""
    if explicit:
        return load_heat(explicit)
    traffic = load_heat(traffic_path(model_dir))
    n = tokens_of(traffic, cfg)
    if n >= MIN_TRAFFIC_TOKENS:
        progress(f"placing experts by real traffic (about {n} tokens)")
        return traffic
    doc = load_topic_heat(topic_path(model_dir))
    if doc and doc.get("heat"):
        progress(f"placing experts by the probe of {doc['topic']!r} ({n} of {MIN_TRAFFIC_TOKENS} tokens "
                 "of real traffic needed to replace it)")
        return _parse_heat(doc["heat"])
    return load_heat(Path(model_dir) / ".coli_usage")


class TrafficRecorder:
    """Saves the backend's routing counts now and then, giving recent traffic more
    weight, so the next start can place experts for what is really being asked."""

    def __init__(self, backend, cfg, path, every: float = SAVE_EVERY_S):
        self.backend, self.cfg, self.path, self.every = backend, cfg, Path(path), every
        self.saved = load_heat(self.path)
        self.seen: Counter = Counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="moelan-traffic", daemon=True)

    def flush(self) -> None:
        now = Counter(dict(list(self.backend.heat.items())))
        new = now - self.seen
        if not new:
            return
        keep = 0.5 ** (tokens_of(new, self.cfg) / HALF_LIFE_TOKENS)
        merged = Counter({k: v * keep for k, v in self.saved.items()})
        merged.update(new)
        try:
            _write_json(self.path, {f"{l}:{e}": round(c, 1) for (l, e), c in merged.items() if c >= 0.05})
        except OSError as exc:
            log.warning("could not save routing counts to %s: %s", self.path, exc)
            return
        self.saved, self.seen = merged, now

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            self.flush()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self.flush()


def parse_subtopics(reply: str, limit: int) -> list[str]:
    """Sub-topics from the model's numbered or bulleted list, without repeats."""
    found = re.findall(r"^\s*(?:\d+[.)]|[-*•])\s+(.+?)\s*$", reply, re.M)
    names = [re.sub(r"[*_`]+", "", s).strip(" :.") for s in found]
    return list(dict.fromkeys(s for s in names if s))[:limit]


def run_topic_probe(model, tok, topic: str, path, subtopics: int = 12, per_subtopic: int = 4,
                    passage_tokens: int = 140, names: list[str] | None = None, progress=print) -> dict:
    """The model chooses sub-topics of ``topic`` and writes passages, and the
    experts picked while writing are counted. Saved after every passage: called
    again with the same topic it continues where it stopped."""
    path = Path(path)

    def ask(text, n, temperature=0.0, seed=None, record=None):
        ids = tok.encode(tok.chat_prompt([{"role": "user", "content": text}]))
        started = []        # the prompt's own routing is the instruction's, so it is not counted

        def on_route(layer, tokens, idx):
            if started and record is not None:
                record.update((layer, int(e)) for e in idx.reshape(-1))
        before, model.on_route = model.on_route, on_route
        try:
            out = list(model.generate(ids, max_new_tokens=n, temperature=temperature, top_p=0.95, seed=seed,
                                      on_token=lambda t: started.append(t)))
        finally:
            model.on_route = before
        return tok.decode(out)

    doc = load_topic_heat(path)
    if not doc or doc.get("topic") != topic:
        doc = {"topic": topic, "subtopics": names, "passages": [], "heat": {}, "tokens": 0, "complete": False}
    if not doc["subtopics"]:
        reply = ask(f"List {subtopics} distinct sub-topics of {topic}. Reply with a numbered list only, "
                    "a few words each, no explanations.", 15 * subtopics)
        doc["subtopics"] = parse_subtopics(reply, subtopics)
        if len(doc["subtopics"]) < 2:
            raise SystemExit(f"could not read sub-topics from the model's reply:\n{reply}")
        _write_json(path, doc)
    progress("sub-topics: " + ", ".join(doc["subtopics"]))
    heat = _parse_heat(doc["heat"])
    finished = {(p["sub"], p["kind"]) for p in doc["passages"]}
    total = len(doc["subtopics"]) * per_subtopic
    t0 = time.perf_counter()
    made = 0
    for i, sub in enumerate(doc["subtopics"]):
        for j in range(per_subtopic):
            if (sub, j) in finished:
                continue
            counted: Counter = Counter()
            prompt = KINDS[j % len(KINDS)].format(sub=sub, topic=topic) + " Start directly, no preamble."
            text = ask(prompt, passage_tokens, 0.7, 100 * i + j, record=counted)
            heat.update(counted)
            doc["passages"].append({"sub": sub, "kind": j, "text": text})
            doc["tokens"] += tokens_of(counted, model.cfg)
            doc["heat"] = {f"{l}:{e}": c for (l, e), c in heat.items()}
            _write_json(path, doc)
            made += 1
            spent = time.perf_counter() - t0
            left = spent / made * (total - len(doc["passages"]))
            progress(f"[{len(doc['passages'])}/{total}] {sub}, kind {j}: about {left / 60:.0f} min left")
    doc["complete"] = True
    _write_json(path, doc)
    return doc
