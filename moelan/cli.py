"""Command line: ``python -m moelan <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__


def _add_gpu_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu", choices=["none", "auto", "cuda", "mps", "dml", "torch-cpu"], default="none",
                   help="hybrid mode: also compute experts on this GPU (needs PyTorch; dml = "
                        "torch-directml for AMD/Intel GPUs on Windows)")
    p.add_argument("--gpu-budget", default="4G", help="GPU memory for experts (fp16), e.g. 6G")
    p.add_argument("--gpu-mode", choices=["auto", "max"], default="auto",
                   help="auto: measure GPU vs CPU and give the GPU its fair share of the hot "
                        "experts; max: fill the budget")


def _add_cluster_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF checkpoint directory (safetensors), colibri container, or a .gguf file")
    p.add_argument("--tokenizer", default=None,
                   help="tokenizer.json to use instead of the model's own (e.g. for a GGUF whose "
                        "vocabulary type moelan cannot rebuild)")
    _add_gpu_args(p)
    p.add_argument("--workers", default="", help="comma-separated host:port list")
    p.add_argument("--discover", type=float, default=0.0, metavar="SECONDS",
                   help="find workers on the LAN by UDP broadcast for this long")
    p.add_argument("--local-budget", default="0",
                   help="RAM the coordinator itself gives to experts, e.g. 4G (default 0: "
                        "experts not placed on workers stream from disk)")
    p.add_argument("--expert-format", choices=["f32", "int8", "int4"], default=None,
                   help="how resident experts are kept in RAM: f32 (fastest, most RAM), int8, "
                        "int4 (least RAM, slowest). Default: int8 for int4 checkpoints, f32 otherwise")
    p.add_argument("--dense-format", choices=["auto", "f32", "f16", "int8"], default="auto",
                   help="attention/output-head weights on the coordinator: f16 = as stored, exact "
                        "(default with numba), f32 (default without), int8 = group-quantized like "
                        "colibri's default (fastest, not exact)")
    p.add_argument("--wire", choices=["f32", "f16", "int8"], default="f32",
                   help="activations on the network: f32 (exact), f16 (half the bytes, ~0.02%% "
                        "error), int8 (a quarter, ~0.4%% error); matters for long prompts on slow links")
    p.add_argument("--no-replicate", action="store_true", help="do not replicate hot experts")
    p.add_argument("--placement", choices=["speed", "ram"], default="speed",
                   help="speed: measure every node and its round trip and share the experts so all "
                        "finish a layer together, dropping workers that would only slow it down; "
                        "ram: split by RAM budget only")
    p.add_argument("--heat", default=None,
                   help="routing statistics file: read to place hot experts first, updated after a run. "
                        "Default: what serve saved from real traffic, else what probe --topic measured, "
                        "else the model's colibri .coli_usage")
    p.add_argument("--lru", type=int, default=64, help="experts kept in the coordinator's disk-stream LRU")


def _gpu_kw(args) -> dict:
    from .worker import parse_size
    if args.gpu == "none":
        return {}
    return {"gpu": args.gpu, "gpu_budget": parse_size(args.gpu_budget, 0), "gpu_mode": args.gpu_mode}


def build_model(args):
    from . import kernels
    from .config import load_config
    from .coordinator import DistributedBackend
    from .discovery import discover
    from .heatfiles import choose_heat
    from .model import MoEModel
    from .worker import parse_size

    kernels.warn_if_slow()
    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    if args.discover:
        print(f"discovering workers for {args.discover:.0f}s ...")
        for info in discover(args.discover):
            addr = f"{info['host']}:{info['port']}"
            if addr not in workers:
                workers.append(addr)
    cfg = load_config(args.model)
    heat = choose_heat(args.model, cfg, args.heat)
    budget = parse_size(args.local_budget, 0)
    if not workers:
        print("no workers: every expert runs on this machine "
              f"({'hottest pinned in RAM, ' if budget else ''}the rest computed from the memory-mapped checkpoint)")
        model = MoEModel(args.model, lru_size=args.lru, expert_format=args.expert_format,
                         pin_budget=budget, heat=heat, progress=print,
                         dense_format=args.dense_format, **_gpu_kw(args))
    else:
        backend = DistributedBackend(args.model, cfg, workers,
                                     local_budget=budget, heat=heat,
                                     fmt=args.expert_format, wire=args.wire,
                                     replicate=not args.no_replicate, lru_size=args.lru,
                                     progress=print, placement=args.placement, **_gpu_kw(args))
        model = MoEModel(args.model, backend=backend, dense_format=args.dense_format)
    return model


def cmd_worker(args) -> None:
    from . import kernels
    from .worker import parse_size, serve
    kernels.warn_if_slow()
    serve(host=args.host, port=args.port, budget=parse_size(args.budget, None) if args.budget else None,
          cache_root=args.cache, model_dir=args.model, node_id=args.name,
          advertise=not args.no_advertise, lru_size=args.lru, threads=args.threads,
          **_gpu_kw(args))


def cmd_discover(args) -> None:
    from .discovery import discover
    found = discover(args.timeout)
    if not found:
        print("no workers found (same subnet? UDP 47474 allowed through the firewall?)")
    for i in found:
        print(f"{i['host']}:{i['port']:<6} {i.get('node_id', '?'):<30} "
              f"budget {i.get('budget_bytes', 0) / 2**30:6.1f} GiB  v{i.get('version', '?')}")


def cmd_inspect(args) -> None:
    from .config import load_config
    from .experts import open_source, resident_bytes
    cfg = load_config(args.model)
    src = open_source(args.model, cfg)
    layers = cfg.moe_layers()
    disk = src.nbytes((layers[0], 0)) if layers else 0
    n = len(layers) * cfg.num_experts
    print(f"family            {cfg.family} ({cfg.expert_layout} experts)")
    print(f"layers            {cfg.num_layers} ({len(layers)} MoE)")
    print(f"experts           {cfg.num_experts} per layer, top-{cfg.top_k}, {n} routed in total")
    print(f"one expert        {disk / 2**20:.2f} MiB on disk")
    if cfg.expert_layout == "gguf":
        types = cfg.raw["moelan_gguf"]["expert_types"]
        print(f"GGUF experts      gate {types.get('ffn_gate_exps')}, up {types.get('ffn_up_exps')}, "
              f"down {types.get('ffn_down_exps')} - kept as stored, memory-mapped")
    fmts = ("native",) if cfg.expert_layout == "gguf" else         ("f32", "int8", "int4") if cfg.expert_layout == "colibri_int4" else ("f32", "int8")
    for f in fmts:
        r = resident_bytes(cfg, f)
        print(f"  as {f:<5}         {r / 2**20:6.2f} MiB each, all experts {n * r / 2**30:7.2f} GiB, "
              f"touched per token {len(layers) * cfg.top_k * r / 2**30:.2f} GiB")


def cmd_plan(args) -> None:
    from .config import load_config
    from .discovery import discover
    from .experts import default_format, resident_bytes
    from .model import load_heat
    from .planner import LOCAL, make_plan
    from .worker import parse_size
    cfg = load_config(args.model)
    nodes = {}
    for spec in filter(None, args.nodes.split(",")):
        name, size = spec.split("=")
        nodes[name] = parse_size(size, 0)
    if args.discover:
        for i in discover(args.discover):
            nodes[i.get("node_id", f"{i['host']}:{i['port']}")] = int(i.get("budget_bytes", 0))
    if parse_size(args.local_budget, 0):
        nodes[LOCAL] = parse_size(args.local_budget, 0)
    if not nodes:
        sys.exit("give --nodes name=SIZE,... or --discover SECONDS")
    ram = resident_bytes(cfg, args.expert_format or default_format(cfg))
    plan = make_plan(cfg.moe_layers(), cfg.num_experts, ram, nodes,
                     heat=load_heat(args.heat) if args.heat else None)
    print(plan.summary(ram))


def cmd_run(args) -> None:
    from .model import save_heat
    from .tokenizer import Tokenizer
    model = build_model(args)
    tok = Tokenizer(args.model, args.tokenizer)
    text = tok.chat_prompt([{"role": "user", "content": args.prompt}]) if args.chat else args.prompt
    ids = tok.encode(text)
    if model.cfg.bos_token_id is not None and not args.chat and args.bos:
        ids = [model.cfg.bos_token_id] + ids
    print(f"prompt: {len(ids)} tokens\n---")
    sys.stdout.write(args.prompt if not args.chat else "")
    out, shown = [], ""
    t0 = time.perf_counter()
    t_first = None
    for t in model.generate(ids, max_new_tokens=args.max_tokens, temperature=args.temperature,
                            top_p=args.top_p, top_k=args.top_k, seed=args.seed):
        if t_first is None:
            t_first = time.perf_counter()
        if t in model.cfg.eos_token_ids:
            break
        out.append(t)
        s = tok.decode(out)
        if not s.endswith("�"):
            sys.stdout.write(s[len(shown):])
            sys.stdout.flush()
            shown = s
    dt = time.perf_counter() - (t_first or t0)
    print(f"\n---\nprefill {((t_first or t0) - t0):.2f}s, decode {max(len(out) - 1, 0) / max(dt, 1e-9):.2f} tok/s, "
          f"MoE time {model.backend.moe_time:.2f}s")
    stats = getattr(model.backend, "stats", None)
    if stats:
        for row in stats():
            print("  " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in row.items()))
    if args.heat:
        save_heat(model.backend.heat, args.heat)
    model.backend.close()


def cmd_serve(args) -> None:
    from .heatfiles import traffic_path
    from .server import serve_http
    from .tokenizer import Tokenizer
    from .worker import parse_size
    model = build_model(args)
    serve_http(model, Tokenizer(args.model, args.tokenizer), Path(args.model).name, args.host, args.port,
               max_batch=args.max_batch, prompt_cache_tokens=args.prompt_cache_tokens,
               prompt_cache_dir=args.prompt_cache_dir,
               prompt_cache_disk=parse_size(args.prompt_cache_disk, 0), expert_tokens=args.expert_tokens,
               traffic_file=None if args.no_save_traffic else traffic_path(args.model))
    model.backend.close()


def cmd_probe_topic(args) -> None:
    from .heatfiles import load_topic_heat, run_topic_probe, topic_path
    from .tokenizer import Tokenizer
    path = Path(args.out) if args.out else topic_path(args.model)
    per = args.passages or 4
    done = load_topic_heat(path)
    if done and done.get("complete") and done["topic"] == args.topic and not args.fresh:
        print(f"already measured {args.topic!r}: {path} (--fresh measures it again)")
        return
    if args.fresh:
        path.unlink(missing_ok=True)
    model = build_model(args)
    tok = Tokenizer(args.model, args.tokenizer)
    print(f"{args.topic!r}: the model chooses {args.subtopics} sub-topics and writes {per} passages on each. "
          "Ctrl+C keeps the finished passages; the same command continues.")
    try:
        result = run_topic_probe(model, tok, args.topic, path, subtopics=args.subtopics, per_subtopic=per,
                                 passage_tokens=args.passage_tokens)
    finally:
        model.backend.close()
    print(f"saved {len(result['passages'])} passages, {result['tokens']} tokens, to {path}. The next start uses "
          "these experts until serve has saved enough real traffic.")


def cmd_probe(args) -> None:
    import json
    if args.topic:
        if args.prompts:
            sys.exit("give --topic or --prompts, not both")
        return cmd_probe_topic(args)
    from .probe_passages import PASSAGES
    from .tokenizer import Tokenizer
    from .topics import has_signal, run_probe, save_topics, topics_path
    passages = json.loads(Path(args.prompts).read_text(encoding="utf-8")) if args.prompts else PASSAGES
    passages = {name: texts[:args.passages] for name, texts in passages.items()}
    if len(passages) < 2:
        sys.exit("a probe compares topics: give at least two")
    model = build_model(args)
    tok = Tokenizer(args.model, args.tokenizer)
    n = sum(len(v) for v in passages.values())
    print(f"reading {n} passages from {len(passages)} topics (Ctrl+C stops without saving)")
    try:
        result = run_probe(model, tok, passages)
    finally:
        model.backend.close()
    path = Path(args.out) if args.out else topics_path(args.model)
    save_topics(result, path)
    print(f"\nexperts leaning to each topic, of {result['tested']} tested:")
    for t in result["topics"]:
        best = f"strongest L{t['best']['layer']}.E{t['best']['expert']} x{t['best']['lift']:g}" if t["best"] else ""
        note = "" if has_signal(t) else "  (no clear signal)"
        print(f"  {t['name']:<12} {t['experts']:>5}  (about {t['chance']:g} by chance)  {best}{note}")
    print(f"saved to {path}; a running server picks it up by itself")


def cmd_monitor(args) -> None:
    from .server import WorkerMonitor, serve_http
    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    if not workers and not args.discover:
        sys.exit("give --workers host:port,... and/or --discover SECONDS")
    mon = WorkerMonitor(workers, interval=args.interval, discover_s=args.discover)
    mon.start()
    serve_http(None, None, "monitor", args.host, args.port, monitor=mon)


def cmd_make_tiny(args) -> None:
    from .tiny import make_tiny, make_tiny_deepseek
    if args.arch.startswith("deepseek"):
        out = make_tiny_deepseek(args.out, version=int(args.arch[-1]), layers=args.layers,
                                 experts=args.experts, hidden=args.hidden, top_k=args.top_k, seed=args.seed)
    else:
        out = make_tiny(args.out, layers=args.layers, experts=args.experts, hidden=args.hidden,
                        top_k=args.top_k, seed=args.seed)
    print(f"wrote tiny MoE checkpoint to {out}")


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):  # a cp1252 console must not crash on model output
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="moelan",
                                 description="Distribute Mixture-of-Experts LLM inference over a LAN.")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("worker", help="run an expert worker on this machine")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--budget", default=None, help="RAM for experts, e.g. 12G (default 60%% of RAM)")
    p.add_argument("--cache", default=None, help="expert cache dir (default ~/.moelan/cache)")
    p.add_argument("--model", default=None, help="optional local copy of the checkpoint (skips transfer)")
    p.add_argument("--name", default=None, help="node id (default hostname:port)")
    p.add_argument("--no-advertise", action="store_true", help="do not broadcast discovery beacons")
    p.add_argument("--lru", type=int, default=0, help="extra non-assigned experts cached in RAM")
    p.add_argument("--threads", type=int, default=None, help="expert compute threads")
    _add_gpu_args(p)
    p.set_defaults(fn=cmd_worker)

    p = sub.add_parser("discover", help="list workers broadcasting on the LAN")
    p.add_argument("--timeout", type=float, default=3.0)
    p.set_defaults(fn=cmd_discover)

    p = sub.add_parser("inspect", help="show a checkpoint's MoE layout and memory needs")
    p.add_argument("--model", required=True)
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("plan", help="preview expert placement for given node budgets")
    p.add_argument("--model", required=True)
    p.add_argument("--nodes", default="", help="e.g. laptop1=8G,desktop=24G")
    p.add_argument("--discover", type=float, default=0.0)
    p.add_argument("--local-budget", default="0")
    p.add_argument("--expert-format", choices=["f32", "int8", "int4"], default=None)
    p.add_argument("--heat", default=None)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="generate text from a prompt")
    _add_cluster_args(p)
    p.add_argument("--prompt", required=True)
    p.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat template")
    p.add_argument("--bos", action="store_true", help="prepend the BOS token")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("serve", help="OpenAI-compatible HTTP API and monitoring dashboard")
    _add_cluster_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-batch", type=int, default=4,
                   help="requests decoded together (1 = one at a time)")
    p.add_argument("--prompt-cache-tokens", type=int, default=16384,
                   help="tokens of recent conversations kept to reuse shared prompt starts "
                        "(about 0.2 MB each for Qwen3-30B; 0 = off)")
    p.add_argument("--prompt-cache-dir", default=None,
                   help="where long prompt caches are kept across restarts (default ~/.moelan/kvcache)")
    p.add_argument("--prompt-cache-disk", default="8G",
                   help="disk space for them, e.g. 20G (0 = memory only)")
    p.add_argument("--expert-tokens", action="store_true",
                   help="count which tokens reach each expert, for the dashboard's tooltip. Fragments of "
                        "real prompts become visible to anyone who can open the dashboard")
    p.add_argument("--no-save-traffic", action="store_true",
                   help="do not save how often each expert is picked (counts only, no text), which the "
                        "next start uses to place the most-picked experts on the fastest GPUs")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("probe", help="measure which topics experts lean to, for the dashboard")
    _add_cluster_args(p)
    p.add_argument("--prompts", default=None,
                   help='JSON file {"topic": ["passage", ...], ...} with at least 3 passages per topic '
                        "(default: 8 built-in topics, about 4,000 tokens)")
    p.add_argument("--passages", type=int, default=None,
                   help="use only this many passages per topic (with --topic: per sub-topic, default 4)")
    p.add_argument("--out", default=None,
                   help="result file (default ~/.moelan/topics/<model id>.json; with --topic "
                        "~/.moelan/heat/<model id>.topic.json)")
    p.add_argument("--topic", default=None,
                   help="for placement, before any real traffic: the model chooses sub-topics of this topic "
                        "and writes passages, and their most-picked experts go to the fastest GPUs at the next start")
    p.add_argument("--subtopics", type=int, default=12, help="sub-topics to choose with --topic")
    p.add_argument("--passage-tokens", type=int, default=140, help="length of each passage with --topic")
    p.add_argument("--fresh", action="store_true", help="with --topic: start again, dropping an earlier result")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("monitor", help="dashboard of worker health, without loading a model")
    p.add_argument("--workers", default="", help="comma-separated host:port list")
    p.add_argument("--discover", type=float, default=0.0, metavar="SECONDS",
                   help="also find workers by UDP broadcast (repeated every poll)")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between polls")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.set_defaults(fn=cmd_monitor)

    p = sub.add_parser("make-tiny", help="write a small random MoE checkpoint for testing")
    p.add_argument("out")
    p.add_argument("--arch", choices=["qwen3-moe", "deepseek-v3", "deepseek-v2"], default="qwen3-moe")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_make_tiny)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose or args.cmd == "worker" else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
