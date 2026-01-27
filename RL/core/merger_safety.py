from __future__ import annotations

from typing import Dict, List, Tuple, Set, Optional, DefaultDict
from collections import defaultdict
from dataclasses import dataclass
from .data_structures import DeltaE, EdgeEdit


def _uf_find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def _uf_union(p, r, a, b):
    ra, rb = _uf_find(p, a), _uf_find(p, b)
    if ra == rb:
        return False
    if r[ra] < r[rb]:
        ra, rb = rb, ra
    p[rb] = ra
    if r[ra] == r[rb]:
        r[ra] += 1
    return True


def _connected(n: int, edges: Set[Tuple[int, int]]):
    if n <= 1:
        return True
    p = list(range(n))
    r = [0] * n
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            _uf_union(p, r, u, v)
    root = _uf_find(p, 0)
    return all(_uf_find(p, i) == root for i in range(n))


@dataclass
class SafetyConfig:
    min_cross_edges_per_neighbor: int
    deltaE_budget_per_step: int
    cooldown_steps: int
    enforce_cross_edge_quota: bool = False


class MergerSafety:
    """Merge actions across agents and enforce safety constraints.

    - Budget: |DeltaE| per step limit
    - Cooldown: prohibit repeated edits on the same edge within T_c
    - Connectivity: do not allow deletions that disconnect the graph
    - Simple AND for deletions: any agent veto blocks deletion (stub)
    """

    def __init__(self, cfg: SafetyConfig, cross_policy: str = 'and', score_thresh: float = 0.2):
        self.cfg = cfg
        self._cooldown: Dict[Tuple[int, int], int] = {}
        self.cross_policy = cross_policy
        self.score_thresh = float(score_thresh)

    def reset(self):
        self._cooldown.clear()

    def _apply_cooldown_decay(self):
        rem = {}
        for e, t in self._cooldown.items():
            if t > 1:
                rem[e] = t - 1
        self._cooldown = rem

    def merge(self, actions_by_agent: Dict[int, object], graph: Dict[str, object]) -> DeltaE:
        nodes: List[int] = list(graph.get('nodes', []))
        edges: Set[Tuple[int, int]] = set(tuple(sorted(e)) for e in graph.get('edges', set()))
        partitions: Dict[int, int] = graph.get('partitions', {}) or {}
        n = len(nodes)

        proposals_add: List[Tuple[int, int, float]] = []
        proposals_del: List[Tuple[int, int, float]] = []
        cross_proposals: DefaultDict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)

        # Collect proposals
        for aid, act in actions_by_agent.items():
            internal = getattr(act, 'internal_edges', None) if not isinstance(act, dict) else act.get('internal_edges')
            cross = getattr(act, 'cross_edge_scores', None) if not isinstance(act, dict) else act.get('cross_edge_scores')
            if internal:
                for e in internal:
                    if isinstance(e, dict):
                        i, j, op, score = e.get('i'), e.get('j'), e.get('op'), float(e.get('score', 0.0))
                    else:
                        i, j, op, score = e.i, e.j, e.op, float(e.score)
                    if i is None or j is None or i == j:
                        continue
                    u, v = sorted((int(i), int(j)))
                    if op == 'add':
                        proposals_add.append((u, v, score))
                    elif op == 'del':
                        proposals_del.append((u, v, score))
            if cross:
                for e in cross:
                    if isinstance(e, dict):
                        i, j, op, score = e.get('i'), e.get('j'), e.get('op', 'add'), float(e.get('score', 0.0))
                    else:
                        i, j, op, score = e.i, e.j, getattr(e, 'op', 'add'), float(e.score)
                    if i is None or j is None or i == j:
                        continue
                    u, v = sorted((int(i), int(j)))
                    # only cross if in different partitions
                    if partitions and partitions.get(u) == partitions.get(v):
                        continue
                    # For DELETE operations on cross edges: treat as regular delete (no AND policy)
                    # This allows any agent with an endpoint to delete a cross edge
                    if op == 'del':
                        proposals_del.append((u, v, score))
                    else:
                        # For ADD: still use AND policy
                        cross_proposals[(u, v)].append((int(aid), float(score)))

        # Sort by score descending for adds, ascending for dels (prefer safer changes)
        proposals_add.sort(key=lambda x: x[2], reverse=True)
        proposals_del.sort(key=lambda x: x[2])

        add_out: List[Tuple[int, int]] = []
        del_out: List[Tuple[int, int]] = []
        rejected: List[Dict[str, object]] = []

        budget = int(self.cfg.deltaE_budget_per_step)

        def can_edit(e: Tuple[int, int]):
            return self._cooldown.get(e, 0) == 0

        # Apply deletions with safety
        # NOTE: Unlike additions, deletions do NOT require AND policy for cross-edges.
        # Any agent with an endpoint in the edge can propose deletion.
        # This is necessary because most edges are cross-partition.
        for u, v, _score in proposals_del:
            if budget <= 0:
                break
            e = (u, v)
            if e not in edges:
                continue
            if not can_edit(e):
                rejected.append({'edge': e, 'reason': 'cooldown'})
                continue
            # simulate deletion
            edges.remove(e)
            if not _connected(n, edges):
                # revert and reject
                edges.add(e)
                rejected.append({'edge': e, 'reason': 'disconnect'})
                continue
            del_out.append(e)
            self._cooldown[e] = self.cfg.cooldown_steps
            budget -= 1

        # Apply additions (internal first)
        for u, v, _score in proposals_add:
            if budget <= 0:
                break
            e = (u, v)
            if e in edges:
                continue
            if not can_edit(e):
                rejected.append({'edge': e, 'reason': 'cooldown'})
                continue
            edges.add(e)
            add_out.append(e)
            self._cooldown[e] = self.cfg.cooldown_steps
            budget -= 1

        # Merge cross edges according to policy
        cross_candidates_sorted: List[Tuple[int, int, float]] = []
        for (u, v), lst in cross_proposals.items():
            if self.cross_policy == 'and':
                # require two different agents to propose same edge
                agent_ids = set(a for a, _s in lst)
                if len(agent_ids) >= 2:
                    min_score = min(s for _a, s in lst)
                    if min_score >= self.score_thresh:
                        cross_candidates_sorted.append((u, v, min_score))
                else:
                    # mark as rejected due to AND policy
                    rejected.append({'edge': (u, v), 'reason': 'policy_and'})
            else:  # 'min'
                if len(lst) >= 2:
                    min_score = min(s for _a, s in lst)
                    if min_score >= self.score_thresh:
                        cross_candidates_sorted.append((u, v, min_score))
        cross_candidates_sorted.sort(key=lambda x: x[2], reverse=True)

        for u, v, _score in cross_candidates_sorted:
            if budget <= 0:
                break
            e = (u, v)
            if e in edges:
                continue
            if not can_edit(e):
                rejected.append({'edge': e, 'reason': 'cooldown'})
                continue
            edges.add(e)
            add_out.append(e)
            self._cooldown[e] = self.cfg.cooldown_steps
            budget -= 1

        # Neighbor pair cross-edge quota: ensure >= b per adjacent partitions
        if partitions and bool(getattr(self.cfg, 'enforce_cross_edge_quota', False)) and int(self.cfg.min_cross_edges_per_neighbor) > 0:
            # Count current cross edges per (pa, pb)
            cross_count: DefaultDict[Tuple[int, int], int] = defaultdict(int)
            for (u, v) in edges:
                a, b = partitions.get(u), partitions.get(v)
                if a is None or b is None or a == b:
                    continue
                key = (min(a, b), max(a, b))
                cross_count[key] += 1

            # Build candidate pool for quota fill: prefer pairs with high degree endpoints
            degs: DefaultDict[int, int] = defaultdict(int)
            for (u, v) in edges:
                degs[u] += 1
                degs[v] += 1

            neighbor_pairs = set(cross_count.keys())
            # Augment neighbor pairs by centroid KNN (k=2)
            node_pos = graph.get('node_pos', {}) or {}
            agents = sorted(set(partitions.values()))
            # compute centroids
            cent = {}
            count = {a: 0 for a in agents}
            for n, a in partitions.items():
                x, y = node_pos.get(n, (0.0, 0.0))
                cx, cy = cent.get(a, (0.0, 0.0))
                cent[a] = (cx + x, cy + y)
                count[a] += 1
            for a in agents:
                if count[a] > 0:
                    cx, cy = cent.get(a, (0.0, 0.0))
                    cent[a] = (cx / count[a], cy / count[a])
                else:
                    cent[a] = (0.0, 0.0)
            # KNN neighbor pairs
            for ai in agents:
                others = [(aj, (cent[ai][0] - cent[aj][0]) ** 2 + (cent[ai][1] - cent[aj][1]) ** 2) for aj in agents if aj != ai]
                others.sort(key=lambda t: t[1])
                for aj, _d in others[:2]:
                    key = (min(ai, aj), max(ai, aj))
                    neighbor_pairs.add(key)

            # For each pair with deficit, add edges up to b (respecting budget)
            for (pa, pb) in sorted(neighbor_pairs):
                need = max(0, self.cfg.min_cross_edges_per_neighbor - cross_count.get((pa, pb), 0))
                if need <= 0:
                    continue
                if budget <= 0:
                    break
                # Candidate node sets
                A_nodes = [n for n, p in partitions.items() if p == pa]
                B_nodes = [n for n, p in partitions.items() if p == pb]
                # Sort by degree descending
                A_nodes.sort(key=lambda x: degs.get(x, 0), reverse=True)
                B_nodes.sort(key=lambda x: degs.get(x, 0), reverse=True)
                # Greedy fill
                filled = 0
                for ua in A_nodes:
                    if budget <= 0 or filled >= need:
                        break
                    for vb in B_nodes:
                        if budget <= 0 or filled >= need:
                            break
                        e = tuple(sorted((ua, vb)))
                        if e in edges or not can_edit(e):
                            continue
                        edges.add(e)
                        add_out.append(e)
                        self._cooldown[e] = self.cfg.cooldown_steps
                        budget -= 1
                        filled += 1
                        # NOTE: quota-fill is an automatic safety action, not an agent rejection.

        self._apply_cooldown_decay()
        return DeltaE(add=add_out, delete=del_out, rejected=rejected)
