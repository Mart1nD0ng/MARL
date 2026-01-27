from __future__ import annotations

from typing import Dict, List, Tuple
import hashlib
import math


class Partitioner:
    """Consistent-hash or geometric-honeycomb partitioner.

    Provides node->agent assignment and simple local rebalance hooks.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        pcfg = cfg.get('partition', {}) or {}
        self.scheme = pcfg.get('scheme', 'consistent_hash')
        self.cell_size = float(pcfg.get('cell_size', 50.0))
        self._assign: Dict[int, int] = {}
        self.anchors: Dict[int, Tuple[float, float]] = {}  # Fixed anchors for agents
        self._m = 1

    def init(self, m: int, nodes: List[int], node_pos: Dict[int, Tuple[float, float]] | None = None) -> Dict[int, int]:
        self._m = max(1, int(m))
        # Choose strategy
        # Prioritize grid-based partitioning for spatial non-overlapping territories
        if node_pos and len(nodes) > 0:
            self._assign = self.assign_grid_partition(self._m, nodes, node_pos)
        elif self.scheme == 'geometric' and node_pos:
            self._assign = self.assign_geo(nodes, node_pos, self._m)
        else:
            self._assign = self._assign_consistent_hash(nodes, self._m)
        return dict(self._assign)

    # Autoscaling helpers
    def estimate_m_for_nodes(self, num_nodes: int, target_slots_per_agent: int, min_agents: int, max_agents: int) -> int:
        if target_slots_per_agent <= 0:
            target_slots_per_agent = 10
        m = (num_nodes + target_slots_per_agent - 1) // target_slots_per_agent
        m = max(min_agents, min(max_agents, m))
        return max(1, int(m))

    def resize_agents(self, m: int, nodes: List[int], node_pos: Dict[int, Tuple[float, float]] | None = None) -> Dict[int, int]:
        """Recompute node->agent mapping for new m.

        Keep consistent hashing or geometric assignment according to scheme.
        """
        return self.init(m, nodes, node_pos)

    def assign_grid_partition(self, m: int, nodes: List[int], node_pos: Dict[int, Tuple[float, float]]) -> Dict[int, int]:
        """Balanced K-Means Partitioning for uniform node distribution.
        
        Uses K-Means clustering with balanced assignment to ensure each agent
        gets approximately equal number of nodes while maintaining spatial locality.
        """
        if m >= len(nodes):
            return {n: i for i, n in enumerate(nodes)}
        
        if not nodes or not node_pos:
            return {}
        
        # Use balanced K-Means: cluster then redistribute
        assignment = self._balanced_kmeans(m, nodes, node_pos)
        
        # Store centroids as anchors for consistency
        self._compute_centroids_from_assignment(assignment, node_pos, m)
        
        # Validate and log
        agent_counts: Dict[int, int] = {}
        for n, a in assignment.items():
            agent_counts[a] = agent_counts.get(a, 0) + 1
        
        active_agents = [c for c in agent_counts.values() if c > 0]
        avg_nodes = sum(active_agents) / max(1, len(active_agents)) if active_agents else 0.0
        print(f"[BalancedKMeans] {m} agents, avg_nodes={avg_nodes:.1f}, "
              f"range=[{min(agent_counts.values()) if agent_counts else 0}, "
              f"{max(agent_counts.values()) if agent_counts else 0}]")
        
        return assignment
    
    def _balanced_kmeans(self, m: int, nodes: List[int], node_pos: Dict[int, Tuple[float, float]], iterations: int = 20) -> Dict[int, int]:
        """K-Means with balanced assignment constraint."""
        import random
        random.seed(42)  # Reproducible
        
        # Init centroids using K-Means++
        points = [(n, node_pos[n]) for n in nodes if n in node_pos]
        if not points:
            return {}
        
        centroids = [random.choice(points)[1]]
        while len(centroids) < m:
            # Pick point furthest from existing centroids
            best = None
            best_dist = -1
            for n, p in points:
                min_d = min(self._dist2(p, c) for c in centroids)
                if min_d > best_dist:
                    best_dist = min_d
                    best = p
            if best:
                centroids.append(best)
            else:
                break
        
        assignment: Dict[int, int] = {}
        target = len(nodes) // m
        max_per_cluster = target + 2  # Allow small imbalance
        
        for _ in range(iterations):
            # Assign with capacity constraint
            cluster_counts = [0] * m
            assignment = {}
            
            # Sort points by distance to their nearest centroid (farthest first for better balancing)
            sorted_points = sorted(points, key=lambda np: min(self._dist2(np[1], c) for c in centroids), reverse=True)
            
            for n, p in sorted_points:
                # Find nearest centroid with capacity
                dists = [(self._dist2(p, centroids[k]), k) for k in range(m)]
                dists.sort()
                
                for d, k in dists:
                    if cluster_counts[k] < max_per_cluster:
                        assignment[n] = k
                        cluster_counts[k] += 1
                        break
                else:
                    # All full, assign to least loaded
                    k = cluster_counts.index(min(cluster_counts))
                    assignment[n] = k
                    cluster_counts[k] += 1
            
            # Update centroids
            new_centroids = []
            for k in range(m):
                cluster_points = [node_pos[n] for n, a in assignment.items() if a == k and n in node_pos]
                if cluster_points:
                    cx = sum(p[0] for p in cluster_points) / len(cluster_points)
                    cy = sum(p[1] for p in cluster_points) / len(cluster_points)
                    new_centroids.append((cx, cy))
                else:
                    new_centroids.append(centroids[k])
            centroids = new_centroids
        
        return assignment
    
    def _compute_centroids_from_assignment(self, assignment: Dict[int, int], node_pos: Dict[int, Tuple[float, float]], m: int):
        """Compute centroids from current assignment for anchor consistency."""
        self.anchors = {}
        for k in range(m):
            cluster_points = [node_pos[n] for n, a in assignment.items() if a == k and n in node_pos]
            if cluster_points:
                cx = sum(p[0] for p in cluster_points) / len(cluster_points)
                cy = sum(p[1] for p in cluster_points) / len(cluster_points)
                self.anchors[k] = (cx, cy)
            else:
                # Empty cluster - use region center
                region_bbox = (self.cfg.get('region') or {}).get('bbox', [-100, -100, 100, 100])
                cx = (region_bbox[0] + region_bbox[2]) / 2
                cy = (region_bbox[1] + region_bbox[3]) / 2
                self.anchors[k] = (cx, cy)
    
    def _place_seeds_uniform(self, m: int, min_x: float, max_x: float, min_y: float, max_y: float) -> List[Tuple[float, float]]:
        """Place m seeds uniformly across the spatial area.
        
        Uses grid layout for uniform distribution.
        """
        # Determine grid dimensions for seeds
        cols = int(math.ceil(math.sqrt(m)))
        rows = int(math.ceil(m / cols))
        
        width = max_x - min_x
        height = max_y - min_y
        
        seeds: List[Tuple[float, float]] = []
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= m:
                    break
                # Place seed at grid position (centered in cell)
                x = min_x + (col + 0.5) * width / cols
                y = min_y + (row + 0.5) * height / rows
                seeds.append((x, y))
                idx += 1
            if idx >= m:
                break
        
        return seeds

    def assign_kmeans(self, m: int, nodes: List[int], node_pos: Dict[int, Tuple[float, float]], iterations: int = 10) -> Dict[int, int]:
        """K-Means clustering for spatial partitioning with validation."""
        if m >= len(nodes):
            return {n: i for i, n in enumerate(nodes)}
        
        # Init centroids randomly from points
        points = [node_pos[n] for n in nodes]
        if not points: return {}
        
        import random
        centroids = random.sample(points, m)
        
        assignment = {}
        
        for _ in range(iterations):
            # Assign
            clusters = [[] for _ in range(m)]
            assignment = {}
            for n in nodes:
                p = node_pos[n]
                # find nearest centroid
                best_k = 0
                best_d = 1e9
                for k in range(m):
                    dx = p[0] - centroids[k][0]
                    dy = p[1] - centroids[k][1]
                    d = dx*dx + dy*dy
                    if d < best_d:
                        best_d = d
                        best_k = k
                assignment[n] = best_k
                clusters[best_k].append(p)
            
            # Update
            new_centroids = []
            for k in range(m):
                pts = clusters[k]
                if pts:
                    cx = sum(pt[0] for pt in pts) / len(pts)
                    cy = sum(pt[1] for pt in pts) / len(pts)
                    new_centroids.append((cx, cy))
                else:
                    # re-init empty cluster
                    new_centroids.append(random.choice(points))
            centroids = new_centroids
        
        # Validation: compute cluster compactness metrics
        cluster_metrics = []
        for k in range(m):
            nodes_in_cluster = [n for n in nodes if assignment.get(n) == k]
            if nodes_in_cluster:
                positions = [node_pos[n] for n in nodes_in_cluster]
                centroid = centroids[k]
                
                # Compute max distance from centroid (cluster radius)
                max_dist = max(math.sqrt((p[0]-centroid[0])**2 + (p[1]-centroid[1])**2) 
                              for p in positions)
                
                # Compute average distance
                avg_dist = sum(math.sqrt((p[0]-centroid[0])**2 + (p[1]-centroid[1])**2) 
                              for p in positions) / len(positions)
                
                cluster_metrics.append({
                    'agent': k,
                    'nodes': len(nodes_in_cluster),
                    'max_radius': max_dist,
                    'avg_radius': avg_dist,
                    'centroid': centroid
                })
        
        # Log clustering quality (can be viewed in verbose mode)
        if cluster_metrics:
            avg_max_radius = sum(c['max_radius'] for c in cluster_metrics) / len(cluster_metrics)
            print(f"[K-Means] {m} clusters, avg_max_radius={avg_max_radius:.1f}, "
                  f"nodes_per_cluster={len(nodes)/m:.1f}")
            
            # Validation check: warn if clusters are too spread out
            for cm in cluster_metrics:
                if cm['max_radius'] > 100.0:  # threshold for 3x3 map
                    print(f"  WARNING: Agent {cm['agent']} has large cluster radius: {cm['max_radius']:.1f}")
        
        return assignment

    def update_on_join_leave(self, partitions: Dict[int, int], added_nodes: List[int], removed_nodes: List[int], node_pos: Dict[int, Tuple[float, float]] | None = None) -> Dict[int, int]:
        """Incrementally update mapping on join/leave events.

        - For added nodes: assign using current scheme (geo_honeycomb if pos given, else consistent hash)
        - For removed nodes: drop from mapping
        - For mobility: recompute assignment every tick using geofence anchors to avoid overlap
        """
        if partitions is None:
            partitions = {}
        # keep internal map in sync
        if not self._assign:
            self._assign = dict(partitions)
        # refresh m from existing mapping
        m = max(self._assign.values()) + 1 if self._assign else max(partitions.values()) + 1 if partitions else self._m
        self._m = max(1, m)
        # Remove nodes
        for n in removed_nodes:
            self._assign.pop(n, None)
            partitions.pop(n, None)
        # Ensure anchors exist
        if node_pos:
            self._ensure_anchors(self._m, node_pos)
        # Add nodes using Fixed Anchors
        if added_nodes:
            if self.anchors and node_pos:
                # Assign to nearest ANCHOR (Home Base)
                for n in added_nodes:
                    if n in node_pos:
                        x, y = node_pos[n]
                        # Find closest anchor
                        best_agent = min(self.anchors.keys(),
                                       key=lambda a: (x - self.anchors[a][0])**2 + (y - self.anchors[a][1])**2)
                        self._assign[n] = best_agent
                        partitions[n] = best_agent
                    else:
                        # Fallback: hash
                        key = hashlib.md5(str(n).encode()).hexdigest()
                        bucket = int(key[:8], 16) % self._m
                        self._assign[n] = bucket
                        partitions[n] = bucket
            else:
                # No anchors set yet? Should not happen if init called.
                # Fallback to hash
                for n in added_nodes:
                    key = hashlib.md5(str(n).encode()).hexdigest()
                    bucket = int(key[:8], 16) % self._m
                    self._assign[n] = bucket
                    partitions[n] = bucket

        # Recompute full assignment each tick to allow nodes to migrate between agents
        if node_pos:
            nodes_all = [n for n in node_pos.keys()]
            target_slots = self._target_slots_per_agent()
            overload_tol = self._overload_tol()
            self._assign = self._balanced_anchor_assignment(nodes_all, node_pos, target_slots, overload_tol)
            partitions = dict(self._assign)
        else:
            # Fallback: ensure dict is in sync
            partitions = dict(self._assign)
        return dict(self._assign)

    def assign_roles(self, m: int) -> Dict[int, int]:
        # simple role id = agent id
        return {aid: aid for aid in range(m)}

    def get_partition(self, agent_id: int) -> List[int]:
        return [n for n, aid in self._assign.items() if aid == agent_id]

    def node_to_agent(self, node_id: int) -> int:
        return int(self._assign.get(node_id, 0))

    def local_rebalance(self, partitions: Dict[int, int], node_pos: Dict[int, Tuple[float, float]], rho_high: float, rho_low: float) -> Dict[str, List[Tuple[int, int, int]]]:
        """Greedy local rebalance between neighboring partitions.
        
        DISABLED for Fixed Anchor strategy to ensure zero overlap.
        """
        # [FixedAnchor] strict spatial mode: changing partition violates "nearest anchor" property.
        # So we disable load balancing here.
        return {'moves': []}

    def assign_geo(self, nodes: List[int], node_pos: Dict[int, Tuple[float, float]], m: int) -> Dict[int, int]:
        """Assign nodes to agents by geometric grid with contiguous regions (DEPRECATED).

        NOTE: This method can create overlapping territories. Use assign_grid_partition instead.
        Kept for backward compatibility only.
        """
        print("[WARNING] assign_geo is deprecated. Use assign_grid_partition for non-overlapping territories.")
        # Map cell -> list of nodes
        cell_map: Dict[Tuple[int, int], List[int]] = {}
        for n in nodes:
            x, y = node_pos.get(n, (0.0, 0.0))
            cx = int(math.floor(x / self.cell_size))
            cy = int(math.floor(y / self.cell_size))
            cell_map.setdefault((cx, cy), []).append(n)
        # Order cells by space-filling curve (row-major)
        ordered_cells = sorted(cell_map.keys(), key=lambda c: (c[1], c[0]))
        assign: Dict[int, int] = {}
        for idx, cell in enumerate(ordered_cells):
            agent = idx % max(1, m)
            for n in cell_map[cell]:
                assign[n] = agent
        return assign

    @staticmethod
    def _assign_consistent_hash(nodes: List[int], m: int) -> Dict[int, int]:
        out = {}
        for n in nodes:
            key = hashlib.md5(str(n).encode()).hexdigest()
            bucket = int(key[:8], 16) % m
            out[n] = bucket
        return out

    @staticmethod
    def _assign_round_robin(nodes: List[int], m: int) -> Dict[int, int]:
        out = {}
        for idx, n in enumerate(sorted(nodes)):
            out[n] = idx % m
        return out

    @staticmethod
    def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def _target_slots_per_agent(self) -> int:
        acfg = (self.cfg.get('autoscale') or {})
        model_cfg = (self.cfg.get('model') or {})
        return int(acfg.get('target_slots_per_agent', model_cfg.get('slots', 10)))

    def _overload_tol(self) -> float:
        pcfg = (self.cfg.get('partition') or {})
        return float(pcfg.get('overload_tol', 1.15))

    def _ensure_anchors(self, m: int, node_pos: Dict[int, Tuple[float, float]]):
        """Place anchors on road intersections for better alignment with vehicle positions."""
        # If anchors already match m, keep them
        if self.anchors and len(self.anchors) == m:
            return
        
        # Determine bbox and grid
        region_bbox = (self.cfg.get('region') or {}).get('bbox')
        if region_bbox and len(region_bbox) == 4:
            min_x, min_y, max_x, max_y = map(float, region_bbox)
        else:
            xs = [p[0] for p in node_pos.values()] if node_pos else []
            ys = [p[1] for p in node_pos.values()] if node_pos else []
            if not xs or not ys:
                min_x = min_y = 0.0
                max_x = max_y = float(self.cell_size)
            else:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
        
        # Use road grid for anchors (align with city grid)
        grid_stride = float((self.cfg.get('city') or {}).get('grid_stride', 100.0))
        
        # Generate grid intersection points within bbox
        grid_x = []
        grid_y = []
        curr = math.ceil(min_x / grid_stride) * grid_stride
        while curr <= max_x:
            grid_x.append(curr)
            curr += grid_stride
        curr = math.ceil(min_y / grid_stride) * grid_stride
        while curr <= max_y:
            grid_y.append(curr)
            curr += grid_stride
        
        # All intersection points (road crossings)
        intersections = [(x, y) for x in grid_x for y in grid_y]
        
        if len(intersections) >= m:
            # K-Means++ like selection: spread out anchors on intersections
            import random
            random.seed(42)  # Reproducible
            selected = [random.choice(intersections)]
            while len(selected) < m:
                # Pick intersection furthest from all selected
                best = None
                best_dist = -1
                for p in intersections:
                    if p in selected:
                        continue
                    min_d = min(self._dist2(p, s) for s in selected)
                    if min_d > best_dist:
                        best_dist = min_d
                        best = p
                if best:
                    selected.append(best)
                else:
                    break
            self.anchors = {aid: pos for aid, pos in enumerate(selected)}
        else:
            # Fallback: uniform grid
            seeds = self._place_seeds_uniform(m, min_x, max_x, min_y, max_y)
            self.anchors = {aid: pos for aid, pos in enumerate(seeds)}
        
        print(f"[FixedAnchor] Initialized {m} anchors on road intersections.")

    def _balanced_anchor_assignment(self, nodes: List[int], node_pos: Dict[int, Tuple[float, float]], target_slots: int, overload_tol: float) -> Dict[int, int]:
        """Assign nodes to nearest anchor with a hard cap; no force-fill of underloaded agents."""
        if not nodes or not node_pos or not self.anchors:
            return self._assign_consistent_hash(nodes, max(1, self._m))
        cap_high = max(1, int(math.ceil(target_slots * overload_tol)))
        assignment: Dict[int, int] = {}
        load = {a: 0 for a in self.anchors.keys()}

        # Pre-sort anchors for each node by distance
        dist_rank: Dict[int, List[int]] = {}
        for n in nodes:
            if n not in node_pos:
                continue
            dist_rank[n] = sorted(self.anchors.keys(), key=lambda a: self._dist2(node_pos[n], self.anchors[a]))

        # Greedy pass: pick nearest anchor that still has room; otherwise nearest overall
        nodes_sorted = sorted([n for n in nodes if n in dist_rank], key=lambda nid: self._dist2(node_pos[nid], self.anchors[dist_rank[nid][0]]))
        for n in nodes_sorted:
            chosen = None
            for a in dist_rank[n]:
                if load[a] < cap_high:
                    chosen = a
                    break
            if chosen is None:
                chosen = dist_rank[n][0]
            assignment[n] = chosen
            load[chosen] += 1

        # Spill from overloaded anchors: move nodes that are naturally closer to another anchor with capacity
        for a_over, c_over in sorted(load.items(), key=lambda kv: kv[1], reverse=True):
            while load[a_over] > cap_high:
                nodes_over = [n for n, ag in assignment.items() if ag == a_over and n in dist_rank]
                if not nodes_over:
                    break
                # Rank nodes by how close they are to some other anchor (best improvement first)
                best_move = None
                best_gain = None
                for n in nodes_over:
                    for dest in dist_rank[n]:
                        if dest == a_over or load.get(dest, 0) >= cap_high:
                            continue
                        gain = self._dist2(node_pos[n], self.anchors[a_over]) - self._dist2(node_pos[n], self.anchors[dest])
                        if best_gain is None or gain > best_gain:
                            best_gain = gain
                            best_move = (n, dest)
                        break  # only consider nearest acceptable dest
                if best_move is None:
                    break
                n_move, dest = best_move
                assignment[n_move] = dest
                load[a_over] -= 1
                load[dest] += 1

        # Assign any nodes missing position to the least loaded anchor to keep mapping total
        for n in nodes:
            if n not in assignment:
                a_best = min(load.keys(), key=load.get)
                assignment[n] = a_best
                load[a_best] += 1

        return assignment
