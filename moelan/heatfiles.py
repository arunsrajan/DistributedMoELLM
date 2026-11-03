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

import numpy as np

from .config import model_fingerprint
from .model import load_heat
from .topics import has_signal, summarize

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


def heat_choice(model_dir, cfg, explicit=None, progress=print) -> tuple[Counter, str | None]:
    """The routing counts placement should use, and where they come from."""
    if explicit:
        return load_heat(explicit), "the --heat file"
    traffic = load_heat(traffic_path(model_dir))
    n = tokens_of(traffic, cfg)
    if n >= MIN_TRAFFIC_TOKENS:
        progress(f"placing experts by real traffic (about {n} tokens)")
        return traffic, f"real traffic, about {n} tokens"
    doc = load_topic_heat(topic_path(model_dir))
    if doc and doc.get("heat"):
        progress(f"placing experts by the probe of {doc['topic']!r} ({n} of {MIN_TRAFFIC_TOKENS} tokens "
                 "of real traffic needed to replace it)")
        return _parse_heat(doc["heat"]), f"the probe of \"{doc['topic']}\""
    usage = load_heat(Path(model_dir) / ".coli_usage")
    return usage, "colibri's .coli_usage" if usage else None


def choose_heat(model_dir, cfg, explicit=None, progress=print) -> Counter:
    return heat_choice(model_dir, cfg, explicit, progress)[0]


def device_rows(model, heat) -> list[dict]:
    """Every device of the cluster, CPU and GPU, with how many experts it holds and
    the share of the counted picks those experts had. A worker reports only how many
    experts its GPU took, and it takes its hottest, so that split is worked out here."""
    total = sum(heat.values())
    backend = model.backend
    plan = getattr(backend, "plan", None)
    local_store = getattr(backend, "local", None) or getattr(backend, "store", None)
    speed = getattr(backend, "speed", None) or {}

    def row(node, device, keys, ms):
        picks = sum(heat.get(k, 0) for k in keys) / total if total else None
        return {"node": node, "device": device, "experts": len(keys), "picks": picks, "ms": ms}

    def split(node, keys, gpu_keys, gpu_name, cpu_ms, gpu_ms):
        rows = []
        if gpu_keys:
            rows.append(row(node, gpu_name, gpu_keys, gpu_ms))
        gpu_keys = set(gpu_keys)
        rest = [k for k in keys if k not in gpu_keys]
        if rest or not rows:
            rows.append(row(node, "CPU", rest, cpu_ms))
        return rows

    local_gpu = getattr(local_store, "gpu", None)
    local_gpu_keys = [k for k in (local_gpu.experts if local_gpu else {}) if k[0] >= 0]
    local_gpu_name = local_gpu.name if local_gpu else None
    if plan is None:
        cfg = model.cfg
        keys = [(l, e) for l in cfg.moe_layers() for e in range(cfg.num_experts)]
        return split("local", keys, local_gpu_keys, local_gpu_name, None, local_gpu.gpu_ms if local_gpu else None)
    rows = []
    for nid, conn in backend.conns.items():
        keys = plan.per_node.get(nid, [])
        if not keys:
            continue
        info = (speed.get("nodes") or {}).get(nid) or {}
        gpu = getattr(conn, "gpu", None) or {}
        hottest = sorted(keys, key=lambda k: -heat.get(k, 0))[:gpu.get("experts", 0)]
        rows += split(nid, keys, hottest, gpu.get("device"), info.get("cpu_ms"), (info.get("gpu") or {}).get("gpu_ms"))
    mine = list(plan.per_node.get("local", [])) + list(plan.unplaced)
    if mine:
        rows += split("local", mine, local_gpu_keys, local_gpu_name, speed.get("local_cpu_ms"),
                      (speed.get("local_gpu") or {}).get("gpu_ms"))
    return rows


def _summary(doc: dict, cfg) -> dict | None:
    """Which experts lean to which sub-topic, from the passages counted so far."""
    done = [p for p in doc["passages"] if p.get("counts")]
    names = [s for s in doc["subtopics"] if any(p["sub"] == s for p in done)]
    if len(names) < 2:
        return None
    counts = np.zeros((len(done), cfg.num_layers, cfg.num_experts), np.int32)
    for i, p in enumerate(done):
        for key, c in p["counts"].items():
            l, e = key.split(":")
            counts[i, int(l), int(e)] = c
    tokens = np.array([p["tokens"] * cfg.top_k for p in done], np.float64)
    labels = np.array([names.index(p["sub"]) for p in done], np.int32)
    return summarize(counts, tokens, labels, names)


def topic_status(doc: dict | None, cfg) -> dict | None:
    """A topic probe as the dashboard shows it, finished or still running."""
    if not doc:
        return None
    summary = doc.get("summary") or {"topics": [], "affinity": {}}
    by_name = {t["name"]: t for t in summary["topics"]}
    rows = []
    for name in doc.get("subtopics") or []:
        t = by_name.get(name)
        passages = sum(1 for p in doc["passages"] if p["sub"] == name)
        rows.append(dict(t, signal=has_signal(t)) if t else
                    {"name": name, "passages": passages, "tokens": 0, "experts": 0, "chance": 0, "best": None,
                     "signal": False})
        rows[-1]["passages"] = passages
    index = {t["name"]: i for i, t in enumerate(rows)}
    order = [t["name"] for t in summary["topics"]]
    affinity = {k: [[index[order[t]], v] for t, v in r] for k, r in summary["affinity"].items()}
    total = len(doc.get("subtopics") or []) * doc.get("per_subtopic", 0)
    return {"topic": doc["topic"], "complete": bool(doc.get("complete")), "done": len(doc["passages"]),
            "total": total, "tokens": doc.get("tokens", 0), "subtopics": rows, "affinity": affinity,
            "tested": len(cfg.moe_layers()) * cfg.num_experts}


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
    doc["per_subtopic"], doc["complete"] = per_subtopic, False
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
            written = tokens_of(counted, model.cfg)
            doc["passages"].append({"sub": sub, "kind": j, "text": text, "tokens": written,
                                    "counts": {f"{l}:{e}": c for (l, e), c in counted.items()}})
            doc["tokens"] += written
            doc["heat"] = {f"{l}:{e}": c for (l, e), c in heat.items()}
            doc["summary"] = _summary(doc, model.cfg)
            _write_json(path, doc)
            made += 1
            spent = time.perf_counter() - t0
            left = spent / made * (total - len(doc["passages"]))
            progress(f"[{len(doc['passages'])}/{total}] {sub}, kind {j}: about {left / 60:.0f} min left")
    doc["complete"] = True
    _write_json(path, doc)
    return doc
