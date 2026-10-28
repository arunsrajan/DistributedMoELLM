"""Expert placement: which node keeps which routed experts in RAM.

Rules, in order:
  1. Within each layer, spread experts across nodes in proportion to their RAM
     budget, so the experts a token picks in one layer are likely to sit on
     different machines and run in parallel.
  2. Hot experts (from routing statistics, when available) are placed first,
     so if the cluster cannot hold everything it is the cold ones that stay
     on the coordinator's disk.
  3. Leftover capacity replicates the hottest experts on a second node, which
     gives both failover and a choice of replica to balance load.
Experts that fit nowhere are served by the coordinator, streamed from disk.

Speed-aware placement (the default when the coordinator can measure its
workers): instead of RAM alone, each node's share of the routed work follows
its measured speed. For one generated token a layer takes, on a worker,
``rtt + picks * expert_ms`` (it gets one request per layer, if any of its
experts were picked), and on the coordinator ``picks * expert_ms`` (it
computes its own experts while waiting). :func:`speed_shares` picks the
shares with the lowest expected layer time; a worker that cannot make a layer
clearly faster gets no experts at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ExpertKey = tuple[int, int]
LOCAL = "local"


@dataclass
class Node:
    node_id: str
    budget_bytes: int
    used_bytes: int = 0
    experts: list[ExpertKey] = field(default_factory=list)

    @property
    def free(self) -> int:
        return self.budget_bytes - self.used_bytes


@dataclass
class Plan:
    owners: dict[ExpertKey, list[str]]          # expert -> node ids (primary first)
    per_node: dict[str, list[ExpertKey]]
    unplaced: list[ExpertKey]                     # coordinator streams these from disk

    def summary(self, expert_bytes: int) -> str:
        lines = []
        for nid, keys in sorted(self.per_node.items()):
            lines.append(f"  {nid:<28} {len(keys):6d} experts  {len(keys) * expert_bytes / 2**30:7.2f} GiB")
        if self.unplaced:
            lines.append(f"  {'(disk on coordinator)':<28} {len(self.unplaced):6d} experts")
        replicas = sum(1 for o in self.owners.values() if len(o) > 1)
        lines.append(f"  replicated experts: {replicas}")
        return "\n".join(lines)


def speed_shares(top_k: int, local_ms: float, workers: dict[str, tuple[float, float, float]],
                 min_gain: float = 1.10, samples: int = 4000):
    """Shares of the routed work that make a generated token fastest.

    ``workers``: node id -> (ms per expert, round-trip ms, largest share its
    RAM allows). A layer routes ``top_k`` picks; with shares p each node gets a
    multinomial number of them. The coordinator computes its picks while the
    workers compute theirs, and a worker that got any picks costs its round
    trip, so a layer takes max(local picks x local ms, rtt_w + picks_w x ms_w).
    Shares are chosen to minimise the *expected* layer time (sampled), which
    accounts for a small share still costing a full round trip in most layers.
    Workers are used only if they speed a layer up by ``min_gain`` or more.
    Returns (shares by node id incl. ``LOCAL``, expected ms per layer, ms per
    layer with everything on the coordinator)."""
    import numpy as np
    alone = top_k * local_ms
    ids = list(workers)
    if not ids:
        return {LOCAL: 1.0}, alone, alone
    ms = np.array([workers[i][0] for i in ids])
    rtt = np.array([workers[i][1] for i in ids])
    caps = np.array([workers[i][2] for i in ids])

    def expected(p):
        probs = np.concatenate([[max(0.0, 1.0 - p.sum())], p])
        picks = np.random.default_rng(0).multinomial(top_k, probs / probs.sum(), size=samples)
        local = picks[:, 0] * local_ms
        remote = np.where(picks[:, 1:] > 0, rtt + picks[:, 1:] * ms, 0.0)
        return float(np.maximum(local, remote.max(axis=1)).mean())

    p = np.zeros(len(ids))
    best = expected(p)
    for _ in range(3):                                   # coordinate search over a share grid
        for j in range(len(ids)):
            room = min(caps[j], 1.0 - (p.sum() - p[j]))
            for cand in np.linspace(0.0, max(room, 0.0), 41):
                q = p.copy()
                q[j] = cand
                t = expected(q)
                if t < best - 1e-9:
                    best, p = t, q
    if best * min_gain > alone:                          # not worth the risk: keep it all here
        p, best = np.zeros(len(ids)), alone
    shares = {i: float(v) for i, v in zip(ids, p)}
    shares[LOCAL] = 1.0 - float(p.sum())
    return shares, best, alone


def make_plan(layers: list[int], num_experts: int, expert_bytes: int,
              nodes: dict[str, int], heat: dict[ExpertKey, float] | None = None,
              replicate: bool = True, shares: dict[str, float] | None = None) -> Plan:
    """Place ``layers x num_experts`` experts of ``expert_bytes`` each on ``nodes``
    (node id -> RAM budget in bytes). With ``shares`` (node id -> fraction of
    the routed work, e.g. from :func:`speed_shares`) each layer's routing heat
    is split in those proportions instead of by RAM; nodes with a zero share get
    nothing."""
    heat = heat or {}
    pool = {nid: Node(nid, b) for nid, b in nodes.items() if b > 0}
    if shares is not None:
        pool = {nid: n for nid, n in pool.items() if shares.get(nid, 0) > 1e-9}
    owners: dict[ExpertKey, list[str]] = {}
    unplaced: list[ExpertKey] = []
    total_budget = sum(n.budget_bytes for n in pool.values()) or 1

    # Visit experts hottest first; ties keep a layer-major order.
    order = sorted(((l, e) for l in layers for e in range(num_experts)),
                   key=lambda k: (-heat.get(k, 0.0), k[0], k[1]))
    per_layer_count: dict[tuple[int, str], float] = {}
    # with shares, balance routing heat (what actually costs time), smoothed so
    # experts never seen still count a little
    mean_heat = (sum(heat.values()) / len(heat)) if heat else 0.0

    def weight(key) -> float:
        return heat.get(key, 0.0) + 0.05 * mean_heat + (1.0 if not heat else 0.0)
    layer_mean = {l: sum(weight((l, e)) for e in range(num_experts)) / num_experts for l in layers}

    for key in order:
        layer = key[0]
        best, best_score = None, None
        for n in pool.values():
            if n.free < expert_bytes:
                continue
            # how far above its fair share of this layer the node would be with this expert
            share = (shares[n.node_id] if shares is not None else n.budget_bytes / total_budget) * num_experts
            add = weight(key) / layer_mean[layer] if shares is not None else 0.0
            score = (per_layer_count.get((layer, n.node_id), 0) + add) / share
            if best is None or score < best_score - 1e-9 or (
                    abs(score - best_score) <= 1e-9 and n.free > best.free):
                best, best_score = n, score
        if best is None:
            unplaced.append(key)
            continue
        best.used_bytes += expert_bytes
        best.experts.append(key)
        per_layer_count[(layer, best.node_id)] = per_layer_count.get((layer, best.node_id), 0) + (
            weight(key) / layer_mean[layer] if shares is not None else 1)
        owners[key] = [best.node_id]

    if replicate and len(pool) > 1 and not unplaced:
        for key in order:  # hottest first
            prim = owners[key][0]
            cands = [n for n in pool.values() if n.node_id != prim and n.free >= expert_bytes]
            if not cands:
                continue
            n = max(cands, key=lambda c: c.free)
            n.used_bytes += expert_bytes
            n.experts.append(key)
            owners[key].append(n.node_id)

    return Plan(owners=owners,
                per_node={n.node_id: n.experts for n in pool.values()},
                unplaced=unplaced)
