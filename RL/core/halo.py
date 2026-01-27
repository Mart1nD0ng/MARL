from __future__ import annotations

from typing import Dict, List, Tuple, Set, Optional
import math

from .data_structures import HaloSummary


class HaloBuilder:
    """Build fixed-dimension halo summaries via r-hop or spatial radius.

    - mode: 'rhop' or 'radius'
    - r: hop count for rhop mode
    - radius: spatial radius for radius mode (requires node_pos)
    """

    def __init__(self, buckets: int = 8, topk: int = 4, mode: str = 'rhop', r: int = 1, radius: float = 1.0):
        self.buckets = buckets
        self.topk = topk
        self.mode = mode
        self.r = max(1, int(r))
        self.radius = float(radius)

    def _degree_hist(self, degrees: List[int]) -> List[float]:
        hist = [0.0] * self.buckets
        for d in degrees:
            idx = min(max(int(d), 0), self.buckets - 1)
            hist[idx] += 1.0
        total = sum(hist) if sum(hist) > 0 else 1.0
        return [h / total for h in hist]

    def _snr_stats_from_weights(self, weights: List[float]) -> Tuple[float, float, float]:
        if not weights:
            return (0.0, 0.0, 0.0)
        xs = sorted(weights)
        n = len(xs)
        mean = sum(xs) / n
        p10 = xs[max(0, int(0.10 * (n - 1)))]
        p90 = xs[min(n - 1, int(0.90 * (n - 1)))]
        return (float(mean), float(p10), float(p90))

    def _build_adj(self, edge_list: List[Tuple[int, int, float]]) -> Dict[int, List[Tuple[int, float]]]:
        adj: Dict[int, List[Tuple[int, float]]] = {}
        for u, v, w in edge_list:
            adj.setdefault(int(u), []).append((int(v), float(w)))
            adj.setdefault(int(v), []).append((int(u), float(w)))
        return adj

    def _rhop_nodes(self, nodes_local: Set[int], adj: Dict[int, List[Tuple[int, float]]], hops: int) -> Set[int]:
        frontier = set(nodes_local)
        visited = set(nodes_local)
        for _ in range(hops):
            nxt = set()
            for u in frontier:
                for v, _w in adj.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        return visited - nodes_local

    def _radius_nodes(self, nodes_local: Set[int], node_pos: Dict[int, Tuple[float, float]]) -> Set[int]:
        if not node_pos:
            return set()
        halo = set()
        # Compute local centroid to reduce O(n^2) checks
        cx = sum(node_pos[n][0] for n in nodes_local) / max(1, len(nodes_local))
        cy = sum(node_pos[n][1] for n in nodes_local) / max(1, len(nodes_local))
        r2 = self.radius * self.radius
        for nid, (x, y) in node_pos.items():
            if nid in nodes_local:
                continue
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy <= r2:
                halo.add(nid)
        return halo

    def build_for_partition(
        self,
        edge_list: List[Tuple[int, int, float]],
        nodes_local: List[int],
        nodes_halo_hint: Optional[List[int]] = None,
        node_pos: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> HaloSummary:
        # Build adjacency and local/halo sets
        adj = self._build_adj(edge_list)
        local_set = set(int(n) for n in nodes_local)

        if self.mode == 'radius' and node_pos is not None:
            halo_set = self._radius_nodes(local_set, node_pos)
        else:
            # r-hop halo
            halo_set = self._rhop_nodes(local_set, adj, self.r)
        # Intersect with hint if provided
        if nodes_halo_hint is not None:
            halo_set &= set(int(n) for n in nodes_halo_hint)

        # Degree on induced subgraph of local+halo
        sub_nodes = local_set | halo_set
        degrees = []
        local_degrees = []
        snr_weights = []
        for u in sub_nodes:
            deg_u = 0
            for v, w in adj.get(u, []):
                if v in sub_nodes:
                    deg_u += 1
                    # Only count weights on edges that touch local for snr stats
                    if u in local_set or v in local_set:
                        snr_weights.append(w)
            degrees.append(deg_u)
            if u in local_set:
                local_degrees.append(deg_u)

        hist = self._degree_hist(local_degrees if local_degrees else degrees)
        mean, p10, p90 = self._snr_stats_from_weights(snr_weights if snr_weights else local_degrees)
        min_degree = min(local_degrees) if local_degrees else (min(degrees) if degrees else 0)

        # cross candidates = edges with one endpoint local and the other in halo
        cross_candidates = 0
        for u in local_set:
            for v, _w in adj.get(u, []):
                if v in halo_set:
                    cross_candidates += 1

        # top-k anchors from halo: prefer high-degree (or high weight sum) and close distance
        anchors: List[Tuple[float, float, float, float]] = []  # (dx,dy,deg,snr)
        halo_scores = []
        for h in halo_set:
            deg_h = sum(1 for v, _ in adj.get(h, []) if v in sub_nodes)
            snr_h = sum(w for v, w in adj.get(h, []) if v in local_set)
            if node_pos and h in node_pos and nodes_local:
                # relative to local centroid
                cx = sum(node_pos[n][0] for n in nodes_local if n in node_pos) / max(1, len(nodes_local))
                cy = sum(node_pos[n][1] for n in nodes_local if n in node_pos) / max(1, len(nodes_local))
                dx = node_pos[h][0] - cx
                dy = node_pos[h][1] - cy
            else:
                dx = 0.0
                dy = 0.0
            halo_scores.append((h, deg_h, snr_h, dx, dy))
        # sort by combined score: degree then snr
        halo_scores.sort(key=lambda t: (t[1], t[2]), reverse=True)
        for _, deg_h, snr_h, dx, dy in halo_scores[: self.topk]:
            anchors.append((float(dx), float(dy), float(deg_h), float(snr_h)))
        while len(anchors) < self.topk:
            anchors.append((0.0, 0.0, 0.0, 0.0))

        return HaloSummary(
            deg_hist=hist,
            snr_stats=(float(mean), float(p10), float(p90)),
            min_degree=int(min_degree),
            cross_candidates=int(cross_candidates),
            topk_anchor=anchors,
        )

    # Backward-compatible simple builder
    def build(self, local_adj: List[Tuple[int, int, float]], node_count: int) -> HaloSummary:
        deg = [0] * node_count
        for i, j, _ in local_adj:
            if 0 <= i < node_count:
                deg[i] += 1
            if 0 <= j < node_count:
                deg[j] += 1
        hist = self._degree_hist(deg)
        mean, p10, p90 = self._snr_stats_from_weights(deg)
        min_degree = min(deg) if deg else 0
        return HaloSummary(hist, (mean, p10, p90), int(min_degree), 0, [(0.0, 0.0, 0.0, 0.0) for _ in range(self.topk)])
