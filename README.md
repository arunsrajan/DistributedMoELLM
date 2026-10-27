# moelan: Mixture-of-Experts inference spread over the laptops on your LAN

`moelan` runs one MoE language model across several ordinary machines on a
local network. It is **expert parallel**: each machine keeps a different
slice of the routed experts in RAM, and the coordinator keeps the small dense
part (attention, norms, router, embeddings). It is pure Python plus numpy, and
needs no GPU, no CUDA and no compiler.

```
            ┌──────── coordinator (has the checkpoint) ────────┐
 tokens ──► │ embeddings · attention · router · KV cache       │
            │ own experts (RAM) · remaining experts (disk)     │
            └───┬───────────────────────┬──────────────────────┘
   one request  │ per layer, per worker │ (only the rows routed there)
                ▼                       ▼
        laptop A: experts 0-41     laptop B: experts 42-127   ...
```

## 1. Research: what already exists (October 2026)

| Project | Language | How it splits the model | MoE expert sharding? | Notes |
|---|---|---|---|---|
| **Colibri** (JustVugg/colibri) | C | Single machine; streams routed experts from disk. Has a *local cluster mode* (`coli cluster worker`) that runs routed experts on other machines | Partly: each worker needs **its own full copy of the converted model** | Main inspiration: dense part in RAM, experts streamed, batch-union of one TCP request per layer |
| exo | Python (MLX / tinygrad) | Pipeline: contiguous **layer ranges** per device | No | Works across LAN, mostly Apple silicon |
| Petals | Python (PyTorch) | Pipeline of transformer **blocks** over the internet | No | Swarm design, high latency per hop |
| llama.cpp RPC | C++ | Layers/tensors offloaded to `rpc-server` hosts | No | Sends weights at startup; layer split |
| distributed-llama | C++ | Tensor parallel (each node a slice of every matrix) | No (slices every expert) | Needs 2^n nodes, sync every layer |
| prima.cpp | C++ | Piped-ring layer split across home devices | No | Heterogeneous devices |
| DeepSpeed-MoE / Megatron / vLLM / SGLang EP | Python + CUDA | Expert parallel | Yes | Datacenter GPUs, NCCL, InfiniBand; not laptops over Wi-Fi |

**Conclusion.** Expert parallelism over a LAN is feasible and fits MoE well:
a token activates only top-k experts per layer, so only a few hidden-state
rows (a few KiB per token per layer) have to cross the network, while the
large weights never move after setup. What did not exist is a *Python* tool
that does this for consumer machines where **each worker downloads only its
own experts**. That is `moelan`.

The cost to keep in mind: decode needs one network round trip per MoE layer
(all workers in parallel). On wired gigabit at about 0.2 ms RTT, a 48-layer
model pays roughly 10 ms per token in latency; on Wi-Fi (2-5 ms RTT) it is
100-250 ms per token. So the win is **fitting a model no single machine can
hold in RAM**, and prefill throughput, rather than faster decode on a model
that already fits.

## 2. What it does

* **Expert placement planner.** Spreads each layer's experts across nodes in
  proportion to their RAM budgets, so a token's top-k experts land on
  different machines and run in parallel. Hot experts (from recorded routing
  statistics, `--heat`) are placed first; spare capacity replicates hot
  experts on a second node.
* **Ships only what each worker needs.** Workers need no model. The
  coordinator pushes each worker's experts once (pipelined); they are cached
  in `~/.moelan/cache/<fingerprint>/` so later runs start with no transfer. A
  worker that already has the checkpoint (`--model`) reads it locally.
* **One round trip per layer.** For each MoE layer each worker gets one
  request carrying only the rows routed to its experts, all workers in
  flight at once, while the coordinator computes its own experts.
* **Colibri-style fallback.** Experts that fit nowhere are streamed from the
  coordinator's disk (memory-mapped safetensors, LRU cache).
* **Fault tolerance.** If a worker drops out, its requests move to a replica,
  else to the coordinator's disk, and generation continues with the same
  output. Dead workers are retried every 30 s.
* **Zero-config discovery.** Workers broadcast UDP beacons (port 47474);
  `--discover 3` finds them. `--workers host:port,...` works across subnets.
* **Faithful by default.** f32 activations, the router's decisions unchanged.
  Expert residency `--expert-format f32|int8|int4`, `--dense-format f16|int8|f32`
  (colibri's default, ~1.5x faster decode), `--wire f16` (half the traffic).
* **Reads colibri int4 containers directly** (`merged_weight` + `qs`,
  `qwen36_meta.json`), and uses colibri's `.coli_usage` routing statistics to
  place hot experts first.
* **Compiled kernels when numba is installed** (`pip install -e ".[fast]"`):
  int4/int8 group-scaled mat-vec straight from the packed bytes, ~13x faster
  than the pure-numpy fallback.
* **OpenAI-compatible API** (`moelan serve`): `/v1/chat/completions` (with
  streaming), `/v1/completions`, `/v1/models`.
* **Monitoring dashboard** at `http://<host>:<port>/` of `moelan serve` (see
  below), and `moelan monitor` for watching workers without loading a model.
* **Safe wire format.** JSON header + raw numpy bytes; no pickle.

Supported checkpoints (HF safetensors, F32/F16/BF16): **Mixtral**,
**Qwen2-MoE**, **Qwen3-MoE** (Qwen3-30B-A3B, Qwen3-Coder-30B-A3B, ...),
**OLMoE**, **DeepSeek V2 / V2-Lite / V2.5** and **DeepSeek V3 / R1 layouts**
(`DeepseekV3ForCausalLM`, which Kimi K2 also uses), plus colibri int4
containers of Qwen3-MoE (e.g. `Justvugg/Qwen3-Coder-30B-A3B-colibri-int4`).

DeepSeek specifics: multi-head latent attention with the compressed KV cache
(the latent and the shared rotary key per token: 576 floats for V3 instead
of 32k), weight absorption so per-head keys and values are never built,
DeepSeek's YaRN RoPE and softmax scale, shared experts (kept on the
coordinator as part of the dense model), the dense first layers
(`first_k_dense_replace`), and both routers: V2 softmax `greedy` /
`group_limited_greedy`, and V3 `noaux_tc` (sigmoid scores plus
`e_score_correction_bias`, top groups by the sum of their two best experts,
`routed_scaling_factor`). Only the routed experts are distributed.

Not yet: the official FP8 DeepSeek-V3 / Kimi K2 releases (use a BF16
conversion), V3's multi-token-prediction layer (ignored), GLM, Qwen3-Next,
Llama 4, gpt-oss, quantized GGUF/GPTQ/AWQ checkpoints, and YaRN for
non-DeepSeek families.

## 3. Quick start

```bash
pip install -e .            # numpy only
pip install -e ".[hf]"      # + tokenizers and jinja2 for real HF models
```

On every helper machine:

```bash
python -m moelan worker --budget 12G
```

On the machine that has the checkpoint:

```bash
python -m moelan discover
python -m moelan inspect --model ~/models/Qwen3-30B-A3B
python -m moelan run   --model ~/models/Qwen3-30B-A3B --discover 3 --local-budget 8G --chat --prompt "Hello!"
python -m moelan serve --model ~/models/Qwen3-30B-A3B --discover 3 --local-budget 8G --host 0.0.0.0
```

Try it without downloading anything:

```bash
python -m moelan make-tiny ./tiny
python -m moelan worker --port 9101 --budget 40M --cache ./c1   # terminal 1
python -m moelan worker --port 9102 --budget 60M --cache ./c2   # terminal 2
python -m moelan run --model ./tiny --discover 3 --prompt "Once upon" --max-tokens 24
```

### Monitoring dashboard

`moelan serve` serves a dashboard at its root URL (`http://127.0.0.1:8000/`).
It is a single self-contained page (no internet needed), refreshed every
second:

* **Speed:** live tokens/s, last request's decode speed and prefill time,
  totals, a 2-minute throughput chart, and the request in progress.
* **Nodes:** up/down, experts held, RAM used against budget, CPUs, requests,
  data sent, worker compute time and how long the coordinator waited on it.
  A worker that drops out turns red, and its experts move to a replica or the
  coordinator's disk.
* **Experts:** a layers x experts map, either routing heat (how often each
  expert ran) or placement (which node holds it), with the experts the latest
  token used outlined; hover for details.
* **Recent requests** and a **Try it** chat box that streams from the model.

JSON feeds behind it: `/api/status`, `/api/heat` (`/cluster` still works).
To watch the workers without loading a model:

```bash
python -m moelan monitor --workers 192.168.1.20:9100,192.168.1.21:9100 --port 8090
python -m moelan monitor --discover 2
```

Use `--host 0.0.0.0` to open the dashboard from another machine; it has no
login, so only do that on a trusted network.

Preview a placement without connecting anything:

```bash
python -m moelan plan --model ./tiny --nodes laptop=8G,desktop=24G --local-budget 4G
```

Firewall: allow TCP 9100 (worker) and UDP 47474 (discovery) on the LAN.
Workers execute no code they receive, but the protocol is unauthenticated
and unencrypted: run it on a network you trust.

## 4. Measured speed (Qwen3-Coder-30B-A3B, colibri int4)

Laptop: Ryzen 5 5500U (6 cores / 12 threads), 32 GB, Wi-Fi. Decode, greedy,
60 tokens of a chat answer. Output text is identical in every distributed run.

| Setup | tok/s |
|---|---|
| first version (numpy; a float64 promotion bug) | 0.04 |
| pure numpy, no numba | 0.25-0.30 |
| numba kernels, f32 attention/head (previous default) | 2.1-2.4 |
| **numba kernels, attention/head as stored 16-bit (default, exact)** | **3.1-3.3** |
| **numba kernels, `--dense-format int8`** | **3.5-3.9** |
| coordinator + 1 worker process, same laptop | 2.5 |
| coordinator + 2 worker processes, same laptop | 3.0 |
| colibri (C/SIMD) on a similar CPU, its README | 8.5-13 |

Measured with the model files in the OS file cache and nothing else holding
RAM. The 15 GB of experts are read through that cache, so on a 32 GB machine
another large process (a second moelan server, a browser with many tabs)
pushes experts out to disk and speed can halve.

Where one token's ~0.3 s goes on one machine: routed experts about 0.17 s (384
expert evaluations, numba's int4 kernels at roughly 8-12 GB/s), attention +
output head about 0.13 s as 16-bit or 0.09 s as int8. Loopback round
trip 0.08 ms; the full per-layer request/reply overhead is about 0.4 ms.

**Does the LAN make it faster?** The workers on one laptop share the same six
cores, so that test cannot show a speedup, only correctness and overhead.
With separate machines, decode time per token is roughly

    dense (coordinator)  +  experts / N machines  +  48 layers x (RTT + 0.3 ms)

| N identical laptops | wired gigabit (RTT ~0.3 ms) | Wi-Fi (RTT ~3 ms) |
|---|---|---|
| 1 | 3.8 tok/s | 3.8 tok/s |
| 2 | about 4.9 tok/s | about 3.0 tok/s |
| 4 | about 6.2 tok/s | about 3.4 tok/s |

These are estimates from the measured parts, not measurements. The ceiling is
the coordinator's dense part (~11 tok/s). So: on wired Ethernet more machines
help moderately; on Wi-Fi the 48 round trips per token eat the gain. The
reliable win is capacity: a model whose experts do not fit in one machine's
RAM (Qwen3-235B-A22B, ~120 GB int4) runs from RAM across several machines
instead of from disk.

## 5. Layout

| File | Role |
|---|---|
| `moelan/safetensors_io.py` | memory-mapped safetensors reader/writer, bf16 |
| `moelan/config.py` | config parsing, per-family tensor names, checkpoint fingerprint |
| `moelan/model.py` | numpy transformer: RMSNorm, RoPE, GQA, q/k norm, router, KV cache, sampling |
| `moelan/experts.py` | expert store and formats (f32 / int8 / colibri int4), memory-mapped streaming |
| `moelan/kernels.py` | optional numba kernels for group-scaled int4/int8/f32 mat-vec |
| `moelan/planner.py` | expert placement and replication |
| `moelan/protocol.py` | framed JSON + array wire protocol |
| `moelan/worker.py` | expert worker server |
| `moelan/coordinator.py` | distributed backend: setup, transfer, per-layer fan-out, failover |
| `moelan/discovery.py` | UDP broadcast discovery |
| `moelan/server.py` | OpenAI-compatible HTTP API, dashboard feeds, worker monitor |
| `moelan/dashboard.html` | the monitoring dashboard (one self-contained page) |
| `moelan/tiny.py` | random tiny checkpoint for tests |

Tests (`python -m pytest tests`) check that distributed generation is token
for token identical to single-machine generation, with expert transfer,
cache reuse, partial clusters, a worker killed mid-generation, and replica
takeover. With `pip install -e ".[ref]"` (torch + transformers),
`tests/test_hf_reference.py` also compares moelan's logits with transformers'
own DeepSeek V2, DeepSeek V3 and Qwen3-MoE implementations on tiny random
checkpoints (they agree to about 1e-5).

## 6. Roadmap

* Verify logits against HF transformers / colibri's reference on real weights
  (so far: coherent output on Qwen3-Coder-30B and exact agreement with an HF
  twin on synthetic int4 models).
* Hide the per-layer round trip: speculative routing / prefetch of the next
  layer's experts, and pipelining several sequences.
* Router-lookahead prefetch and speculative routing to hide the per-layer round trip.
* Pipelining several sequences (batch decoding) to fill the network gaps.
* Optional faster kernels (int4, numba/torch when present) on workers.
* Run DeepSeek-V2-Lite (16B, BF16) end to end on real weights; FP8 block
  dequantization for the official DeepSeek-V3 / Kimi K2 files.
* GLM / Qwen3-Next / gpt-oss adapters; GGUF input.
* Authentication (shared secret) and TLS for untrusted networks.

## License

Apache License 2.0, see [LICENSE](LICENSE).
