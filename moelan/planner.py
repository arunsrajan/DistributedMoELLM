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


def make_plan(layers: list[int], num_experts: int, expert_bytes: int,
              nodes: dict[str, int], heat: dict[ExpertKey, float] | None = None,
              replicate: bool = True) -> Plan:
    """Place ``layers x num_experts`` experts of ``expert_bytes`` each on ``nodes``
    (node id -> RAM budget in bytes)."""
    heat = heat or {}
    pool = {nid: Node(nid, b) for nid, b in nodes.items() if b > 0}
    owners: dict[ExpertKey, list[str]] = {}
    unplaced: list[ExpertKey] = []
    total_budget = sum(n.budget_bytes for n in pool.values()) or 1

    # Visit experts hottest first; ties keep a layer-major order.
    order = sorted(((l, e) for l in layers for e in range(num_experts)),
                   key=lambda k: (-heat.get(k, 0.0), k[0], k[1]))
    per_layer_count: dict[tuple[int, str], int] = {}

    for key in order:
        layer = key[0]
        best, best_score = None, None
        for n in pool.values():
            if n.free < expert_bytes:
                continue
            # how far this node is above its fair share of this layer
            share = n.budget_bytes / total_budget * num_experts
            score = per_layer_count.get((layer, n.node_id), 0) / share
            if best is None or score < best_score - 1e-9 or (
                    abs(score - best_score) <= 1e-9 and n.free > best.free):
                best, best_score = n, score
        if best is None:
            unplaced.append(key)
            continue
        best.used_bytes += expert_bytes
        best.experts.append(key)
        per_layer_count[(layer, best.node_id)] = per_layer_count.get((layer, best.node_id), 0) + 1
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
