"""HTTP side: an OpenAI-compatible API (``/v1/chat/completions``,
``/v1/completions``, ``/v1/models``) and the monitoring dashboard (``/``) with
its JSON feeds (``/api/status``, ``/api/heat``; ``/cluster`` is kept).
One generation at a time; requests queue on a lock, monitoring never waits.

``moelan monitor`` serves the same dashboard without a model: it polls the
workers directly and shows their health.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from .model import MoEModel
from .tokenizer import Tokenizer

MAX_BODY = 4 * 1024 * 1024
RATE_WINDOW_S = 3.0


def dashboard_html() -> bytes:
    return resources.files(__package__).joinpath("dashboard.html").read_bytes()


class Telemetry:
    """Generation counters for the dashboard. Thread-safe, cheap per token."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.active: dict | None = None
        self.recent: deque = deque(maxlen=20)
        self.token_times: deque = deque(maxlen=4096)

    def begin(self, rid: str, prompt_tokens: int) -> None:
        with self.lock:
            self.requests += 1
            self.tokens_in += prompt_tokens
            self.active = {"id": rid, "prompt_tokens": prompt_tokens, "generated": 0,
                           "started": time.time(), "first_token": None}

    def token(self) -> None:
        now = time.time()
        with self.lock:
            self.tokens_out += 1
            self.token_times.append(now)
            if self.active is not None:
                self.active["generated"] += 1
                if self.active["first_token"] is None:
                    self.active["first_token"] = now

    def end(self) -> None:
        with self.lock:
            a, self.active = self.active, None
        if a is None:
            return
        now = time.time()
        first = a["first_token"] or now
        decode_n = max(a["generated"] - 1, 0)
        self.recent.appendleft({
            "id": a["id"], "prompt_tokens": a["prompt_tokens"], "completion_tokens": a["generated"],
            "prefill_s": round(first - a["started"], 3), "total_s": round(now - a["started"], 3),
            "decode_tok_s": round(decode_n / (now - first), 3) if decode_n and now > first else None,
            "ended": now})

    def snapshot(self) -> dict:
        now = time.time()
        with self.lock:
            recent_tokens = sum(1 for t in self.token_times if now - t <= RATE_WINDOW_S)
            active = dict(self.active) if self.active else None
            out = {"uptime_s": now - self.started, "requests": self.requests,
                   "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                   "rate_tok_s": recent_tokens / RATE_WINDOW_S, "busy": active is not None,
                   "recent": list(self.recent)}
        if active:
            ft = active["first_token"]
            active["elapsed_s"] = now - active["started"]
            active["phase"] = "decode" if ft else "prefill"
            n = active["generated"] - 1
            active["decode_tok_s"] = n / (now - ft) if ft and n > 0 and now > ft else None
            out["active"] = active
        return out


def model_info(model: MoEModel, name: str) -> dict:
    cfg = model.cfg
    b = model.backend
    store = getattr(b, "store", None) or getattr(b, "local", None)
    return {"name": name, "family": cfg.family, "layout": cfg.expert_layout,
            "layers": cfg.num_layers, "moe_layers": len(cfg.moe_layers()), "experts": cfg.num_experts,
            "top_k": cfg.top_k, "hidden": cfg.hidden_size, "dense_bytes": model.dense_bytes(),
            "expert_format": getattr(b, "fmt", None) or (store.fmt if store else None),
            "dense_format": getattr(model, "dense_format", "f32"),
            "backend": "distributed" if hasattr(b, "conns") else "local"}


def node_stats(model: MoEModel) -> list[dict]:
    b = model.backend
    if hasattr(b, "stats"):
        return b.stats()
    st = b.store
    return [{"node": "local", "address": "-", "alive": True, "experts": len(st.pinned),
             "hits": st.hits, "disk_reads": st.misses, "resident_bytes": st.resident_bytes()}]


def heat_payload(model: MoEModel) -> dict:
    """Routing counts per (layer, expert), each expert's primary owner, and the
    experts the most recent token picked."""
    cfg = model.cfg
    b = model.backend
    L, E = cfg.num_layers, cfg.num_experts
    counts = [[0] * E for _ in range(L)]
    for (l, e), c in list(b.heat.items()):
        if l < L and e < E:
            counts[l][e] = c
    nodes = ["local"]
    owners = [[0] * E for _ in range(L)]
    plan = getattr(b, "plan", None)
    if plan is not None:
        nodes += [n for n in plan.per_node if n != "local"]
        index = {n: i for i, n in enumerate(nodes)}
        for (l, e), own in plan.owners.items():
            owners[l][e] = index.get(own[0], 0)
    last = {str(l): list(map(int, ex)) for l, ex in list(b.last.items())}
    return {"layers": L, "experts": E, "counts": counts, "owners": owners, "nodes": nodes, "last": last}


class _Handler(BaseHTTPRequestHandler):
    model: MoEModel | None
    tok: Tokenizer | None
    name: str
    lock: threading.Lock
    tel: Telemetry
    monitor = None  # WorkerMonitor in monitor-only mode
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/dashboard"):
            return self._send(200, dashboard_html(), "text/html; charset=utf-8")
        if path == "/api/status":
            return self._json(200, self.status())
        if path == "/api/heat":
            if self.model is None:
                return self._json(200, {"layers": 0, "experts": 0, "counts": [], "owners": [],
                                        "nodes": [], "last": {}})
            return self._json(200, heat_payload(self.model))
        if path == "/health":
            return self._json(200, {"ok": True, "model": self.name})
        if self.model is None:
            return self._json(404, {"error": "monitor-only mode: no model loaded"})
        if path == "/v1/models":
            return self._json(200, {"object": "list", "data": [{"id": self.name, "object": "model"}]})
        if path == "/cluster":
            return self._json(200, {"nodes": node_stats(self.model), "moe_time_s": self.model.backend.moe_time})
        self._json(404, {"error": "not found"})

    def status(self) -> dict:
        if self.model is None:
            return {"mode": "monitor", "model": None, "nodes": self.monitor.snapshot(),
                    "gen": self.tel.snapshot(), "moe_time_s": 0.0}
        return {"mode": "serve", "model": model_info(self.model, self.name),
                "nodes": node_stats(self.model), "gen": self.tel.snapshot(),
                "moe_time_s": self.model.backend.moe_time}

    def do_POST(self):
        if self.model is None:
            return self._json(404, {"error": "monitor-only mode: no model loaded"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._json(413, {"error": "body too large"})
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json(400, {"error": "invalid JSON"})
        if self.path == "/v1/chat/completions":
            prompt = self.tok.chat_prompt(req.get("messages", []))
            chat = True
        elif self.path == "/v1/completions":
            prompt = req.get("prompt", "")
            chat = False
        else:
            return self._json(404, {"error": "not found"})
        ids = self.tok.encode(prompt)
        kw = dict(max_new_tokens=int(req.get("max_tokens", 256)),
                  temperature=float(req.get("temperature", 0.0)),
                  top_p=float(req.get("top_p", 1.0)), top_k=int(req.get("top_k", 0)),
                  seed=req.get("seed"))
        rid = f"moelan-{uuid.uuid4().hex[:12]}"
        with self.lock:
            self.tel.begin(rid, len(ids))
            try:
                if req.get("stream"):
                    return self._stream(rid, ids, kw, chat)
                out = []
                t0 = time.perf_counter()
                for t in self.model.generate(ids, **kw):
                    self.tel.token()
                    out.append(t)
                text = self.tok.decode([t for t in out if t not in self.model.cfg.eos_token_ids])
                usage = {"prompt_tokens": len(ids), "completion_tokens": len(out),
                         "total_tokens": len(ids) + len(out),
                         "tokens_per_second": len(out) / max(time.perf_counter() - t0, 1e-9)}
            finally:
                self.tel.end()
        choice = ({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                  if chat else {"index": 0, "text": text, "finish_reason": "stop"})
        self._json(200, {"id": rid, "object": "chat.completion" if chat else "text_completion",
                         "created": int(time.time()), "model": self.name,
                         "choices": [choice], "usage": usage})

    def _stream(self, rid, ids, kw, chat):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        out, sent = [], ""
        eos = set(self.model.cfg.eos_token_ids)
        for t in self.model.generate(ids, **kw):
            self.tel.token()
            if t in eos:
                break
            out.append(t)
            text = self.tok.decode(out)
            if text.endswith("�"):
                continue  # incomplete UTF-8 sequence, wait for the next token
            delta, sent = text[len(sent):], text
            chunk = ({"index": 0, "delta": {"content": delta}} if chat else {"index": 0, "text": delta})
            self._event({"id": rid, "object": "chat.completion.chunk" if chat else "text_completion",
                         "model": self.name, "choices": [chunk]})
        end = ({"index": 0, "delta": {}, "finish_reason": "stop"} if chat
               else {"index": 0, "text": "", "finish_reason": "stop"})
        self._event({"id": rid, "model": self.name, "choices": [end]})
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True

    def _event(self, obj) -> None:
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()


class WorkerMonitor(threading.Thread):
    """Monitor-only mode: polls each worker's ``stats`` over its own connection."""

    def __init__(self, addresses: list[str], interval: float = 2.0, discover_s: float = 0.0):
        super().__init__(daemon=True, name="moelan-monitor")
        self.addresses = list(addresses)
        self.interval = interval
        self.discover_s = discover_s
        self.rows: dict[str, dict] = {}
        self.lock = threading.Lock()
        self._conns: dict = {}

    def _poll(self, addr: str) -> dict:
        from .coordinator import WorkerConn
        c = self._conns.get(addr)
        try:
            if c is None or c.sock is None:
                c = self._conns[addr] = WorkerConn(addr, timeout=5.0)
                c.connect(connect_timeout=2.0)
            h, _ = c.call({"op": "stats"})
            stores = h.get("stores", {})
            return {"node": c.node_id, "address": addr, "alive": True,
                    "experts": sum(s["pinned"] for s in stores.values()),
                    "requests": h.get("requests", 0), "compute_s": h.get("busy_s", 0.0),
                    "budget_bytes": c.info.get("budget_bytes"), "cpus": c.info.get("cpus"),
                    "resident_bytes": sum(s["resident_bytes"] for s in stores.values()),
                    "version": c.info.get("version")}
        except (OSError, RuntimeError, ConnectionError) as exc:
            if c is not None and c.sock is not None:
                c.sock.close()
                c.sock = None
            prev = self.rows.get(addr, {})
            return dict(prev, address=addr, alive=False, node=prev.get("node", addr), error=str(exc))

    def run(self) -> None:
        from .discovery import discover
        while True:
            if self.discover_s:
                for info in discover(self.discover_s):
                    a = f"{info['host']}:{info['port']}"
                    if a not in self.addresses:
                        self.addresses.append(a)
            for a in list(self.addresses):
                row = self._poll(a)
                with self.lock:
                    self.rows[a] = row
            time.sleep(self.interval)

    def snapshot(self) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.rows.values()]


def make_server(model: MoEModel | None, tok: Tokenizer | None, name: str,
                host: str = "127.0.0.1", port: int = 8000, monitor: WorkerMonitor | None = None):
    handler = type("H", (_Handler,), {"model": model, "tok": tok, "name": name,
                                      "lock": threading.Lock(), "tel": Telemetry(), "monitor": monitor})
    return ThreadingHTTPServer((host, port), handler)


def serve_http(model: MoEModel | None, tok: Tokenizer | None, name: str, host: str = "127.0.0.1",
               port: int = 8000, monitor: WorkerMonitor | None = None):
    httpd = make_server(model, tok, name, host, port, monitor)
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    print(f"dashboard:  http://{shown}:{port}/")
    if model is not None:
        print(f"OpenAI API: http://{shown}:{port}/v1")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
