from __future__ import annotations

from typing import Dict, List, Tuple, Set, Optional
import math
import random
from collections import deque

from .data_structures import Observation, Slot, HaloSummary, DeltaE
from .halo import HaloBuilder

# Physical/link defaults (override via cfg['physics'])
DEFAULT_PHYSICS = {
    # Reference distance (meters-like in our coordinate system)
    'D0': 200.0,
    # Message size in bits (for delay/energy accounting)
    'MSG_BITS': 4000.0,
    # Node spatial density gamma for 2-D Poisson point process (paper)
    'GAMMA': 0.0,
    # Epsilon for -log(success) failure shaping
    'EPS_SUCCESS': 1e-12,
    'P_NI': 0.2,
    # Interference coupling coefficient (effective noise grows with edge density)
    'INTERF_K': 0.0,
    'PT': 1.0,
    'ALPHA': 1.7,
    'Z': 4.0,
    'B': 800.0,
    # Communication radius (same unit as node_pos); 0 disables radius limit
    'R_A': 0.0,
    # NLOS attenuation multiplier when blocked by buildings (legacy, superseded by CI model)
    'NLOS_ATTEN': 0.25,
    'MAX_DELAY': 1.0,
    'MAX_ENERGY': 200.0,
    # ========== CI Path Loss Model for Urban Street Canyon mmWave ==========
    # Carrier frequency in GHz (28 GHz is typical mmWave for urban micro-cell)
    'FC_GHZ': 28.0,
    # Path loss exponent for LOS (2.2 limits range to ~100-150m)
    'PL_N_LOS': 2.2,
    # Path loss exponent for NLOS (2.8 allows short-range NLOS ~30-40m)
    'PL_N_NLOS': 2.8,
    # Shadow fading std dev (dB) for LOS
    'PL_SIGMA_LOS': 3.0,
    # Shadow fading std dev (dB) for NLOS
    'PL_SIGMA_NLOS': 8.0,
    # Corner loss per 90-degree turn (dB) - diffraction/reflection penalty
    # Reduced to 6dB to allow some NLOS connectivity at short range
    'CORNER_LOSS_DB': 6.0,
    # Path loss cutoff (dB) - links with PL > this are considered outage
    # Tuned for urban street canyon:
    # - LOS: effective up to ~80m (with MAX_LINK_DIST=100m hard cap)
    # - NLOS (1 corner): effective up to ~30m
    'PL_CUTOFF_DB': 108.0,
    # Hard distance limit (m) - links beyond this are always outage regardless of PL
    # This prevents unrealistic long LOS links even with low path loss
    'MAX_LINK_DIST': 100.0,
    # Noise power in linear (Watts), used to compute SINR from gain
    # -90 dBm = 1e-12 W is typical receiver noise floor for mmWave
    'NOISE_POWER_LIN': 1e-12,
    # TX power in linear (Watts) for SINR calculation
    # 23 dBm = 0.2 W is typical for mmWave small cell
    'PT_LIN': 0.2,
    # Shadowing correlation distance (m) for LOS
    'SHADOW_DCOR_LOS': 10.0,
    # Shadowing correlation distance (m) for NLOS
    'SHADOW_DCOR_NLOS': 13.0,
    # Enable new CI path loss model (set False to use legacy model)
    'USE_CI_MODEL': True,
}


class CoreEnv:
    """Core environment managing mobility/link/consensus/reward and applying 螖E.

    This is a lightweight skeleton to support the new training loop while
    preserving existing legacy entry points.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.slots_n = int(((cfg.get('model') or {}).get('slots') or 10))
        self.msg_dim = int(((cfg.get('model') or {}).get('msg_dim') or 32))
        self.nodes: List[int] = []
        self.edges: Set[Tuple[int, int]] = set()
        self.time_step = 0
        halo_cfg = (cfg.get('halo') or {})
        self.halo = HaloBuilder(
            buckets=int(halo_cfg.get('buckets', 8)),
            topk=int(halo_cfg.get('topk', 4)),
            mode=str(halo_cfg.get('mode', 'rhop')),
            r=int(halo_cfg.get('r', 1)),
            radius=float(halo_cfg.get('radius', 1.0)),
        )
        # Optional coordinates for radius halo & geometric partitioning
        self.node_pos: Dict[int, Tuple[float, float]] = {}
        self._prev_node_pos: Dict[int, Tuple[float, float]] = {}
        self.partitions: Dict[int, int] = {}
        # Mobility and churn
        self.mobility = CityGridMobility(cfg)
        self._offnet_joiners: List[Dict[str, object]] = []  # candidates moving toward bbox
        self._next_node_id = 0
        phys_cfg = (cfg.get('physics') or {})
        self._phys = {k: float(phys_cfg.get(k, v)) if not isinstance(v, bool) else bool(phys_cfg.get(k, v)) for k, v in DEFAULT_PHYSICS.items()}
        # Shadowing cache: (u,v) -> (S_dB, is_los) for spatially correlated shadow fading
        self._shadow_cache: Dict[Tuple[int, int], Tuple[float, bool]] = {}
        # Link geometry cache: (u,v) -> (d_euc, is_los, d_path, n_corner)
        self._link_geo_cache: Dict[Tuple[int, int], Tuple[float, bool, float, int]] = {}
        # Position hash for cache invalidation
        self._pos_hash: int = 0

    @property
    def graph(self) -> Dict[str, object]:
        return {
            'nodes': list(self.nodes),
            'edges': set(self.edges),
            'partitions': dict(self.partitions),
            'node_pos': dict(self.node_pos),
            'buildings': list(self.mobility.buildings),
            'grid_stride': self.mobility.grid_stride,
            'road_width': self.mobility.road_width,
            'bbox': self.mobility.bbox
        }

    def reset(self, partitions: Dict[int, int]) -> Dict[int, Observation]:
        # Build initial graph using k-nearest (more realistic than a ring and avoids huge multi-hop paths).
        self.nodes = sorted(set(partitions.keys()))
        self.edges = set()
        self.partitions = dict(partitions)
        # Init coordinates (random in a square), deterministic per node for stability
        self.mobility.reset(self.nodes, self.node_pos)
        self._prev_node_pos = dict(self.node_pos)
        # Clear link caches for new episode
        self._shadow_cache.clear()
        self._link_geo_cache.clear()
        self._pos_hash = 0
        if len(self.nodes) >= 2:
            topo_init = (self.cfg.get('topology_init') or {})
            init_mode = str(topo_init.get('init_mode', 'knn'))
            k = int(topo_init.get('k_nearest', 3))
            snr_th = float(topo_init.get('snr_thresh', 0.0))
            R_A = self._R_a_for_n(len(self.nodes))
            # Helper: check direct wireless feasibility (within R_a, not blocked, not outage)
            use_ci = self._phys.get('USE_CI_MODEL', True)
            def _ok_edge(u: int, v: int) -> bool:
                ux, uy = self.node_pos.get(u, (0.0, 0.0))
                vx, vy = self.node_pos.get(v, (0.0, 0.0))
                d = math.hypot(ux - vx, uy - vy)
                if R_A > 0.0 and d > R_A:
                    return False
                # Check outage using CI model
                if use_ci:
                    pl_db, is_outage = self._path_loss_db(u, v)
                    if is_outage:
                        return False
                else:
                    # Legacy: just check blocking
                    if self.mobility.is_blocked((ux, uy), (vx, vy)):
                        return False
                if snr_th > 0.0:
                    sinr = self._link_snr(u, v)
                    if (sinr / (1.0 + sinr)) < snr_th:
                        return False
                return True

            if init_mode == 'clique':
                # Dense initial topology: prioritize internal edges, then short cross edges
                # First pass: create all internal edges (within same partition)
                for i in range(len(self.nodes)):
                    u = self.nodes[i]
                    for j in range(i + 1, len(self.nodes)):
                        v = self.nodes[j]
                        if partitions.get(u) == partitions.get(v):  # Same partition
                            if _ok_edge(u, v):
                                self.edges.add(tuple(sorted((u, v))))
                
                # Second pass: add cross edges only for short distances (< 50m)
                # This ensures inter-partition connectivity but limits long edges
                cross_edge_max_dist = 50.0  # Only allow short cross edges initially
                for i in range(len(self.nodes)):
                    u = self.nodes[i]
                    for j in range(i + 1, len(self.nodes)):
                        v = self.nodes[j]
                        if partitions.get(u) != partitions.get(v):  # Different partition
                            ux, uy = self.node_pos.get(u, (0.0, 0.0))
                            vx, vy = self.node_pos.get(v, (0.0, 0.0))
                            d = math.hypot(ux - vx, uy - vy)
                            if d <= cross_edge_max_dist and _ok_edge(u, v):
                                self.edges.add(tuple(sorted((u, v))))
            elif init_mode == 'knn_internal':
                # KNN but prioritize internal neighbors, ensure connectivity
                # Phase 1: Build internal edges within each partition
                for u in self.nodes:
                    ux, uy = self.node_pos.get(u, (0.0, 0.0))
                    u_part = partitions.get(u, -1)
                    internal_cand: List[Tuple[float, int]] = []
                    cross_cand: List[Tuple[float, int]] = []
                    for v in self.nodes:
                        if v == u:
                            continue
                        if not _ok_edge(u, v):
                            continue
                        vx, vy = self.node_pos.get(v, (0.0, 0.0))
                        d = math.hypot(ux - vx, uy - vy)
                        if partitions.get(v, -1) == u_part:
                            internal_cand.append((d, v))
                        else:
                            cross_cand.append((d, v))
                    # Prioritize internal, then add cross if needed
                    internal_cand.sort(key=lambda t: t[0])
                    cross_cand.sort(key=lambda t: t[0])
                    # Take k internal neighbors (more to ensure connectivity)
                    for _d, v in internal_cand[:max(2, k)]:
                        self.edges.add(tuple(sorted((u, v))))
                    # Add 2 shortest cross neighbors to ensure inter-partition connectivity
                    for _d, v in cross_cand[:2]:
                        self.edges.add(tuple(sorted((u, v))))
                
                # Phase 2: Ensure global connectivity with a greedy approach
                # If graph is still disconnected, add shortest edges to connect components
                node_set = set(self.nodes)
                while True:
                    # Find connected components
                    adj: Dict[int, Set[int]] = {n: set() for n in self.nodes}
                    for u, v in self.edges:
                        adj[u].add(v)
                        adj[v].add(u)
                    visited = set()
                    components = []
                    for start in self.nodes:
                        if start in visited:
                            continue
                        comp = set()
                        stack = [start]
                        while stack:
                            n = stack.pop()
                            if n in comp:
                                continue
                            comp.add(n)
                            for nb in adj[n]:
                                if nb not in comp:
                                    stack.append(nb)
                        visited |= comp
                        components.append(comp)
                    
                    if len(components) <= 1:
                        break  # Connected
                    
                    # Find shortest edge between any two components
                    best_edge = None
                    best_dist = float('inf')
                    for i, comp_a in enumerate(components):
                        for comp_b in components[i+1:]:
                            for u in comp_a:
                                for v in comp_b:
                                    if _ok_edge(u, v):
                                        ux, uy = self.node_pos.get(u, (0.0, 0.0))
                                        vx, vy = self.node_pos.get(v, (0.0, 0.0))
                                        d = math.hypot(ux - vx, uy - vy)
                                        if d < best_dist:
                                            best_dist = d
                                            best_edge = (u, v)
                    
                    if best_edge is None:
                        # No feasible edge found between components
                        # This means some nodes are physically unreachable
                        # Don't add infeasible edges - just break
                        break
                    else:
                        self.edges.add(tuple(sorted(best_edge)))
            else:
                # KNN edges (undirected) with hard distance limit
                # Prefer short edges, but never exceed MAX_LINK_DIST
                # IMPORTANT: Use higher k to ensure enough redundancy for delete operations
                max_link_dist = float(self._phys.get('MAX_LINK_DIST', 150.0))
                preferred_dist = max_link_dist * 0.6  # Prefer shorter edges (90m for 150m limit)
                
                # Ensure sufficient edge redundancy: at least k edges per node
                # This prevents all edges from being bridge edges
                effective_k = max(k, 4)  # Minimum 4 neighbors to ensure redundancy
                
                for u in self.nodes:
                    ux, uy = self.node_pos.get(u, (0.0, 0.0))
                    cand: List[Tuple[float, int]] = []
                    for v in self.nodes:
                        if v == u:
                            continue
                        vx, vy = self.node_pos.get(v, (0.0, 0.0))
                        d = math.hypot(ux - vx, uy - vy)
                        # Hard limit: never add edges longer than MAX_LINK_DIST
                        if d <= max_link_dist:
                            cand.append((d, v))
                    cand.sort(key=lambda t: t[0])
                    
                    # Take effective_k nearest neighbors within distance limit
                    for d, v in cand[:effective_k]:
                        self.edges.add(tuple(sorted((u, v))))
                
                # Ensure graph is 2-edge-connected by adding redundant edges
                # This prevents any single edge deletion from disconnecting the graph
                import networkx as nx
                G = nx.Graph()
                G.add_nodes_from(self.nodes)
                G.add_edges_from(self.edges)
                
                # Add edges to eliminate bridges (edges whose removal disconnects the graph)
                max_bridge_fix_attempts = len(self.nodes) * 2
                attempts = 0
                while attempts < max_bridge_fix_attempts:
                    bridges = list(nx.bridges(G))
                    if len(bridges) == 0:
                        break  # No more bridges, graph is 2-edge-connected
                    
                    # For each bridge, try to add an alternative path
                    fixed_any = False
                    for u, v in bridges:
                        # Find shortest alternative edge that would provide redundancy
                        # Either endpoint can connect to another node in the other's component
                        ux, uy = self.node_pos.get(u, (0.0, 0.0))
                        vx, vy = self.node_pos.get(v, (0.0, 0.0))
                        
                        best_edge = None
                        best_dist = float('inf')
                        
                        # Try to find an edge from u's neighbors (excluding v) to v's side
                        G.remove_edge(u, v)
                        if not nx.is_connected(G):
                            # Find the two components
                            comps = list(nx.connected_components(G))
                            comp_u = next(c for c in comps if u in c)
                            comp_v = next(c for c in comps if v in c)
                            
                            # Find shortest edge between components (excluding u-v)
                            for a in comp_u:
                                ax, ay = self.node_pos.get(a, (0.0, 0.0))
                                for b in comp_v:
                                    if (a, b) == (u, v) or (b, a) == (u, v):
                                        continue
                                    if tuple(sorted((a, b))) in self.edges:
                                        continue  # Already exists
                                    bx, by = self.node_pos.get(b, (0.0, 0.0))
                                    d = math.hypot(ax - bx, ay - by)
                                    if d <= max_link_dist and d < best_dist:
                                        best_dist = d
                                        best_edge = (a, b)
                        G.add_edge(u, v)  # Restore the bridge
                        
                        if best_edge:
                            self.edges.add(tuple(sorted(best_edge)))
                            G.add_edge(*best_edge)
                            fixed_any = True
                            break  # Re-check bridges after adding
                    
                    if not fixed_any:
                        break  # No more bridges can be fixed
                    attempts += 1
            # Fallback: if graph is disconnected or too sparse, add more edges
            # 1. First ensure connectivity by adding shortest edges between components
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(self.nodes)
            G.add_edges_from(self.edges)
            
            # Connect isolated nodes first
            for n in self.nodes:
                if G.degree(n) == 0:
                    # Find nearest neighbor regardless of distance
                    nx_pos = self.node_pos.get(n, (0.0, 0.0))
                    best_dist = float('inf')
                    best_v = None
                    for v in self.nodes:
                        if v == n:
                            continue
                        vx, vy = self.node_pos.get(v, (0.0, 0.0))
                        d = math.hypot(nx_pos[0] - vx, nx_pos[1] - vy)
                        if d < best_dist:
                            best_dist = d
                            best_v = v
                    if best_v is not None:
                        self.edges.add(tuple(sorted((n, best_v))))
                        G.add_edge(n, best_v)
            
            # Ensure graph is connected
            while not nx.is_connected(G):
                comps = list(nx.connected_components(G))
                if len(comps) <= 1:
                    break
                # Find shortest edge between any two components
                best_edge = None
                best_dist = float('inf')
                for i, comp_a in enumerate(comps):
                    for comp_b in comps[i+1:]:
                        for u in comp_a:
                            ux, uy = self.node_pos.get(u, (0.0, 0.0))
                            for v in comp_b:
                                vx, vy = self.node_pos.get(v, (0.0, 0.0))
                                d = math.hypot(ux - vx, uy - vy)
                                if d < best_dist:
                                    best_dist = d
                                    best_edge = (u, v)
                if best_edge:
                    self.edges.add(tuple(sorted(best_edge)))
                    G.add_edge(*best_edge)
                else:
                    break  # No edge found
            
            # 2. Ensure minimum degree of 2 for all nodes (to allow edge deletion)
            for n in self.nodes:
                while G.degree(n) < 2:
                    nx_pos = self.node_pos.get(n, (0.0, 0.0))
                    # Find nearest non-neighbor
                    cand = []
                    for v in self.nodes:
                        if v == n or G.has_edge(n, v):
                            continue
                        vx, vy = self.node_pos.get(v, (0.0, 0.0))
                        d = math.hypot(nx_pos[0] - vx, nx_pos[1] - vy)
                        cand.append((d, v))
                    if not cand:
                        break
                    cand.sort(key=lambda x: x[0])
                    _, v = cand[0]
                    self.edges.add(tuple(sorted((n, v))))
                    G.add_edge(n, v)
        self.time_step = 0
        self._offnet_joiners.clear()
        self._next_node_id = (max(self.nodes) + 1) if self.nodes else 0
        return self._build_observations(partitions)

    def step(self, deltaE: DeltaE, partitions: Dict[int, int]) -> Tuple[Dict[int, Observation], float, Dict]:
        # Apply deletions
        for u, v in deltaE.delete:
            e = tuple(sorted((int(u), int(v))))
            if e in self.edges:
                self.edges.remove(e)
        # Apply additions
        for u, v in deltaE.add:
            u, v = int(u), int(v)
            # Only accept edges between existing nodes
            if u != v and u in self.nodes and v in self.nodes:
                # Check wireless feasibility using CI path loss model
                if u in self.node_pos and v in self.node_pos:
                    # Use path loss model to determine if link is feasible
                    use_ci = self._phys.get('USE_CI_MODEL', True)
                    if use_ci:
                        # CI model: check outage based on path loss
                        pl_db, is_outage = self._path_loss_db(u, v)
                        if is_outage:
                            continue  # Link is in outage, reject
                    else:
                        # Legacy model: check R_a and blocking
                        ra = self._R_a_for_n(len(self.nodes))
                        x1, y1 = self.node_pos[u]
                        x2, y2 = self.node_pos[v]
                        d = math.hypot(x1 - x2, y1 - y2)
                        if ra > 0.0 and d > ra:
                            continue
                        if self.mobility.is_blocked((x1, y1), (x2, y2)):
                            continue
                self.edges.add(tuple(sorted((u, v))))
        # Sanitize edges in case any stale endpoints remain
        self.edges = {e for e in self.edges if e[0] in self.nodes and e[1] in self.nodes}

        # Enforce connectivity via a minimal repair if needed (stub)
        self._ensure_connectivity()

        # Advance time
        self.time_step += 1

        # ==============================
        # Reward (learnable by edge edits)
        # ==============================
        # Actions only modify edges (螖E); partitions are external. Avoid non-learnable territory terms.
        consensus_success = self._consensus_success()
        # Use -log(success) as failure cost to avoid vanishing signal when success is close to 1.
        # This is still a "higher worse" cost and stays well-behaved with an epsilon floor.
        eps_succ = float(self._phys.get('EPS_SUCCESS', 1e-12))
        failure_cost = -math.log(max(eps_succ, float(consensus_success)))
        latency_cost = self._latency_cost()
        energy_cost = self._energy_cost()
        is_conn = self._is_connected()
        connectivity_bonus = 1.0 if is_conn else 0.0

        n = len(self.nodes)
        e = len(self.edges)
        # Edge density penalty: target average degree ~= 3 (edges ~= 1.5*n)
        e_target = max(1.0, 1.5 * float(n))
        edge_over = max(0.0, (float(e) - e_target) / e_target)
        edge_cost = float(min(1.0, edge_over))

        # Long-edge penalty (encourage local connectivity)
        if self.edges and self.node_pos:
            d0 = max(1e-6, float(self._phys.get('D0', 200.0)))
            d_mean = 0.0
            cnt = 0
            for (u, v) in self.edges:
                if u in self.node_pos and v in self.node_pos:
                    x1, y1 = self.node_pos[u]
                    x2, y2 = self.node_pos[v]
                    d_mean += (math.hypot(x1 - x2, y1 - y2) / d0)
                    cnt += 1
            d_mean = d_mean / max(1, cnt)
            long_edge_cost = float(min(1.0, max(0.0, (d_mean - 0.7) / 1.3)))
        else:
            long_edge_cost = 0.0

        delta_sz = float(len(deltaE.add) + len(deltaE.delete))
        edit_cost = min(1.0, delta_sz / 20.0)

        reward = (
            # Use failure rate as a cost so that (failure/delay/energy) are all "higher worse".
            # Note: -3*(1-success) differs from +3*success by a constant -3, which does not
            # change optimal policies under advantage-based methods; it mainly improves interpretability.
            - 3.0 * failure_cost
            + 0.3 * connectivity_bonus
            - 1.2 * latency_cost
            - 0.8 * energy_cost
            - 1.2 * edge_cost
            - 0.6 * long_edge_cost
            - 0.15 * edit_cost
        )

        info = {
            # Training loop expects these proxy keys
            'success_proxy': float(consensus_success),
            'failure_proxy': float(failure_cost),
            'delay_proxy': float(latency_cost),
            'energy_proxy': float(energy_cost),
            # Extra diagnostics
            'consensus_success': float(consensus_success),
            'consensus_failure': float(failure_cost),
            'latency_cost': float(latency_cost),
            'energy_cost': float(energy_cost),
            'connectivity_bonus': float(connectivity_bonus),
            'edge_cost': float(edge_cost),
            'long_edge_cost': float(long_edge_cost),
            'edit_cost': float(edit_cost),
            'edges': len(self.edges),
            'is_connected': bool(is_conn),
        }
        obs = self._build_observations(partitions)
        return obs, reward, info

    # ------------------------------
    # Mobility and churn (Task A)
    # ------------------------------

    def tick_mobility(self):
        """Advance city-grid mobility for in-network nodes and off-net joiners."""
        if not self.mobility.enable:
            return
        # Snapshot positions to compute velocity features
        self._prev_node_pos = dict(self.node_pos)
        # Update in-network nodes
        self.mobility.step_nodes(self.nodes, self.node_pos)
        # Update off-net joiners
        if self._offnet_joiners:
            self.mobility.step_joiners(self._offnet_joiners)
        # Invalidate link geometry cache (positions changed)
        self._invalidate_link_caches()

    def apply_join_leave(self, partitioner, partitions: Dict[int, int]) -> Tuple[Dict[int, int], Dict[str, List[int]]]:
        """Apply geofence-based join/leave and initialize topology for new nodes.

        Returns updated partitions and an events dict.
        """
        events = {'added': [], 'removed': []}
        # 1) Generate join candidates by rate
        churn = (self.cfg.get('churn') or {})
        region = (self.cfg.get('region') or {})
        topo_init = (self.cfg.get('topology_init') or {})
        max_nodes = int(churn.get('max_nodes', 2000))
        min_nodes = int(churn.get('min_nodes', 50))
        join_rate = float(churn.get('join_rate', 0.02))
        # Maintain pool size heuristics
        current_n = len(self.nodes)
        if current_n < max_nodes:
            # proportional candidates per step
            cand = max(0, int(round(join_rate * max(1, current_n))))
            # If below min_nodes, force at least 1 joiner
            if current_n < min_nodes:
                cand = max(1, cand)
            for _ in range(cand):
                self._offnet_joiners.append(self.mobility.sample_offnet_joiner())

        # 2) Check off-net joiners entering bbox -> add nodes
        added_nodes: List[int] = []
        bbox = tuple(region.get('bbox', (-120.0, -120.0, 120.0, 120.0)))
        inside = lambda p: (bbox[0] <= p[0] <= bbox[2]) and (bbox[1] <= p[1] <= bbox[3])
        remain_joiners: List[Dict[str, object]] = []
        for j in self._offnet_joiners:
            pos = j['pos']  # type: ignore
            if inside(pos):
                nid = self._next_node_id
                self._next_node_id += 1
                self.nodes.append(nid)
                self.node_pos[nid] = (float(pos[0]), float(pos[1]))
                # assign heading into mobility state
                self.mobility.add_node(nid, pos, j.get('heading', 'E'))
                added_nodes.append(nid)
                events['added'].append(nid)
            else:
                remain_joiners.append(j)
        self._offnet_joiners = remain_joiners

        # 3) Remove nodes that left bbox
        removed_nodes: List[int] = []
        leave_margin = float(region.get('leave_margin', 0.0))
        lbbox = (bbox[0] - leave_margin, bbox[1] - leave_margin, bbox[2] + leave_margin, bbox[3] + leave_margin)
        def inside_leave(p):
            return (lbbox[0] <= p[0] <= lbbox[2]) and (lbbox[1] <= p[1] <= lbbox[3])
        stay_nodes: List[int] = []
        for nid in list(self.nodes):
            p = self.node_pos.get(nid, (0.0, 0.0))
            if inside_leave(p):
                stay_nodes.append(nid)
            else:
                # remove
                removed_nodes.append(nid)
                events['removed'].append(nid)
                # cleanup edges
                self.edges = {e for e in self.edges if nid not in e}
                self.node_pos.pop(nid, None)
                self.mobility.remove_node(nid)
                partitions.pop(nid, None)
        self.nodes = stay_nodes

        # 4) Initialize topology for new nodes
        if added_nodes:
            k = int(topo_init.get('k_nearest', 3))
            snr_th = float(topo_init.get('snr_thresh', 0.0))
            for n in added_nodes:
                # distances to current nodes
                dists: List[Tuple[float, int]] = []
                px, py = self.node_pos.get(n, (0.0, 0.0))
                for m in self.nodes:
                    if m == n:
                        continue
                    qx, qy = self.node_pos.get(m, (0.0, 0.0))
                    # Check occlusion
                    if self.mobility.is_blocked((px, py), (qx, qy)):
                        continue
                    d = math.hypot(px - qx, py - qy)
                    # Simple LOS/NLOS model? For now just block
                    snr = 1.0 / (1.0 + d)
                    if snr >= snr_th:
                        dists.append((d, m))
                dists.sort(key=lambda t: t[0])
                for _, m in dists[:k]:
                    if n != m:
                        self.edges.add(tuple(sorted((n, m))))

        # 5) Update partitioner mapping incrementally (always, so nodes can migrate across regions)
        if hasattr(partitioner, 'update_on_join_leave'):
            partitions = partitioner.update_on_join_leave(partitions, added_nodes, removed_nodes, self.node_pos)
            self.partitions = partitions

        return partitions, events

    # ------------------------------
    # Enhanced Reward Components
    # ------------------------------

    def _consensus_success(self) -> float:
        """PBFT success probability (paper-aligned P-L model; assume P_n=1).

        We align to the paper's PBFT reliability structure:
        - Per committee size n_g: f=floor((n_g-1)/3), quorum threshold 2f.
        - Use paper Eq.(7) to compute average direct transmission success P_l (=P_s).
        - Use binomial tails (paper Eq.(6)) to compute quorum/commit/system success.

        In this codebase, a "consensus system" corresponds to each governance partition (agent region),
        so we compute PBFT success per partition and aggregate (size-weighted).
        """
        part_map = dict(self.partitions) if self.partitions else {}
        if not part_map:
            # fallback: treat whole network as one group
            part_map = {n: 0 for n in self.nodes}

        # Binomial tail helper (stable enough for n<=200)
        def _binom_tail(trials: int, p: float, k0: int) -> float:
            if trials <= 0:
                return 1.0 if k0 <= 0 else 0.0
            p = float(max(0.0, min(1.0, p)))
            k0 = int(max(0, min(trials, k0)))
            if k0 == 0:
                return 1.0
            if p == 0.0:
                return 0.0
            if p == 1.0:
                return 1.0 if k0 <= trials else 0.0
            s = 0.0
            for k in range(k0, trials + 1):
                s += math.comb(trials, k) * (p ** k) * ((1.0 - p) ** (trials - k))
            return float(max(0.0, min(1.0, s)))

        # Group nodes by partition/agent id
        groups: Dict[int, List[int]] = {}
        for n in self.nodes:
            groups.setdefault(int(part_map.get(n, 0)), []).append(int(n))

        total_weight = 0.0
        total_p = 0.0
        for _gid, gnodes in groups.items():
            n = len(gnodes)
            if n <= 1:
                total_p += float(n) * 1.0
                total_weight += float(n)
                continue

            # PBFT tolerance for this committee size
            f = max(0, (n - 1) // 3)
            k_min = 2 * f
            need_commit = max(0, n - f)

            # Paper Eq.(7): average direct transmission success probability P_l (=P_s)
            p_bar = self._paper_avg_link_success(n)

            # Quorum -> node commit -> system commit
            p_quorum = _binom_tail(n - 1, p_bar, k_min)
            p_node_commit = float(max(0.0, min(1.0, p_quorum * p_quorum)))
            p_system = _binom_tail(n, p_node_commit, need_commit)

            total_p += float(n) * float(p_system)
            total_weight += float(n)

        if total_weight <= 0:
            return 0.0
        return float(max(0.0, min(1.0, total_p / total_weight)))

    # ------------------------------
    # Paper-aligned wireless helpers
    # ------------------------------

    def _R_a_for_n(self, n: int) -> float:
        """Compute R_a from gamma if configured; else use fixed R_A.

        Paper: R_a = sqrt(n / (pi * gamma))
        """
        n = int(max(0, n))
        ra_cfg = float(self._phys.get('R_A', 0.0))
        if ra_cfg and ra_cfg > 0.0:
            return float(ra_cfg)
        gamma = float(self._phys.get('GAMMA', 0.0))
        if gamma <= 0.0 or n <= 0:
            return 0.0
        return float(math.sqrt(float(n) / (math.pi * gamma)))

    def _paper_avg_link_success(self, n: int) -> float:
        """Paper Eq.(7): average success probability of transmission under PPP.

        P_s = (2*pi*gamma/n) * ∫_0^{R_a} exp(-(P_N+P_I) * (r/D0)^alpha * z / P_T) * r dr
        """
        n = int(max(1, n))
        ra = self._R_a_for_n(n)
        if ra <= 0.0:
            return 0.0

        gamma = float(self._phys.get('GAMMA', 0.0))
        D0 = max(1e-6, float(self._phys.get('D0', 200.0)))
        alpha = float(self._phys.get('ALPHA', 1.7))
        P_NI = max(1e-12, self._effective_P_NI())
        PT = max(1e-12, float(self._phys.get('PT', 1.0)))
        z = max(1e-12, float(self._phys.get('Z', 1.0)))
        c = (P_NI * z) / PT

        steps = 256
        dr = ra / float(steps)
        integ = 0.0
        for i in range(steps + 1):
            r = dr * float(i)
            w = 0.5 if (i == 0 or i == steps) else 1.0
            rn = max(0.0, r / D0)
            p = math.exp(-c * (rn ** alpha))
            integ += w * p * r
        integ *= dr

        # Use distance PDF f(r)=2r/Ra^2 (paper Eq.4). This avoids double-counting gamma/n.
        pref = 2.0 / max(1e-12, ra * ra)
        return float(max(0.0, min(1.0, pref * integ)))

    def _paper_avg_link_delay(self, n: int) -> float:
        """Paper-aligned expected one-hop transmission delay under f(r)=2r/R_a^2.

        Use mean SINR (E[h]=1):
          SINR_mean(r) = P_T * (r/D0)^(-alpha) / (P_N+P_I)
        Rate: C = B * log2(1+SINR)
        Delay: t = MSG_BITS / C
        """
        n = int(max(1, n))
        ra = self._R_a_for_n(n)
        if ra <= 0.0:
            return float(self._phys.get('MAX_DELAY', 1.0))

        D0 = max(1e-6, float(self._phys.get('D0', 200.0)))
        alpha = float(self._phys.get('ALPHA', 1.7))
        P_NI = max(1e-12, self._effective_P_NI())
        PT = max(1e-12, float(self._phys.get('PT', 1.0)))
        B = max(1e-12, float(self._phys.get('B', 1.0)))
        msg_bits = max(1e-12, float(self._phys.get('MSG_BITS', 4000.0)))

        steps = 256
        dr = ra / float(steps)
        integ = 0.0
        for i in range(steps + 1):
            r = dr * float(i)
            w = 0.5 if (i == 0 or i == steps) else 1.0
            # f(r)=2r/R_a^2
            fr = (2.0 * r) / max(1e-12, ra * ra)
            rn = max(1e-6, r / D0)
            sinr = (PT * (rn ** (-alpha))) / P_NI
            cap = B * math.log2(1.0 + max(0.0, sinr))
            t = msg_bits / max(1e-12, cap)
            integ += w * t * fr
        integ *= dr
        # Clamp to avoid pathological huge values if parameters are extreme
        return float(max(1e-6, min(float(self._phys.get('MAX_DELAY', 1.0)), integ)))

    
    def _latency_cost(self) -> float:
        """Latency proxy.

        Paper-aligned mode (when GAMMA>0 and R_A is auto-derived):
        - PBFT over wireless direct transmissions within radius R_a.
        - Use expected one-hop transmission delay under distance distribution.

        Fallback mode:
        - Use max shortest-path delay over the current graph (legacy skeleton).
        """
        n = len(self.nodes)
        if n <= 1:
            return 0.0

        gamma = float(self._phys.get('GAMMA', 0.0))
        ra_cfg = float(self._phys.get('R_A', 0.0))
        use_paper = (gamma > 0.0 and (ra_cfg <= 0.0))
        if use_paper:
            stage_delay = self._paper_avg_link_delay(n)
            total_delay = 3.0 * stage_delay
            norm = total_delay / max(1e-6, self._phys['MAX_DELAY'])
            return float(min(1.0, norm))

        # Fallback: topology-based max path delay
        if not self._is_connected():
            return 1.0
        adj = self._adjacency()
        rng = random.Random(1337 + int(self.time_step))
        nodes = list(self.nodes)
        if len(nodes) > 10:
            sample_src = nodes[:5]
            sample_dst = rng.sample(nodes, 5)
            pairs = [(s, d) for s in sample_src for d in sample_dst if s != d]
        else:
            pairs = [(s, d) for s in nodes for d in nodes if s != d]

        max_delay = 0.0
        for src, dst in pairs:
            path = self._shortest_path(src, dst, adj)
            if not path:
                continue
            max_delay = max(max_delay, self._route_delay(path))
        total_delay = 3.0 * float(max_delay)
        norm = total_delay / max(1e-6, self._phys['MAX_DELAY'])
        return float(min(1.0, norm))
    
    def _energy_cost(self) -> float:
        """Energy proxy using TX power * link time for each active edge (bidirectional)."""
        n = len(self.nodes)
        if n <= 1:
            return 0.0
        total_energy = 0.0
        for (u, v) in self.edges:
            delay = self._link_delay(u, v)
            total_energy += 2.0 * delay * self._phys['PT']
        norm = total_energy / max(1e-6, self._phys['MAX_ENERGY'])
        return float(min(1.0, norm))

    def _connectivity_bonus(self) -> float:
        """Binary bonus for maintaining a connected graph.
        
        Returns 1.0 if connected, 0.0 otherwise.
        """
        return 1.0 if self._is_connected() else 0.0
    
    def _is_connected(self) -> bool:
        """Check if the graph is connected using BFS."""
        if len(self.nodes) <= 1:
            return True
        
        if not self.edges:
            return len(self.nodes) <= 1
        
        # Build adjacency list
        adj: Dict[int, List[int]] = {n: [] for n in self.nodes}
        for u, v in self.edges:
            if u not in adj or v not in adj:
                continue
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from first node
        visited = set()
        queue = [self.nodes[0]]
        visited.add(self.nodes[0])
        
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(self.nodes)

    # ------------------------------
    # Link-layer helpers (PBFT-aware)
    # ------------------------------

    def _adjacency(self) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {n: [] for n in self.nodes}
        for u, v in self.edges:
            if u in adj and v in adj:
                adj[u].append(v)
                adj[v].append(u)
        return adj

    def _shortest_path(self, src: int, dst: int, adj: Dict[int, List[int]]) -> List[int]:
        if src not in adj or dst not in adj:
            return []
        q: deque = deque([src])
        prev: Dict[int, int] = {src: -1}
        while q:
            cur = q.popleft()
            if cur == dst:
                break
            for nb in adj.get(cur, []):
                if nb not in prev:
                    prev[nb] = cur
                    q.append(nb)
        if dst not in prev:
            return []
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def _route_success(self, path: List[int]) -> float:
        p = 1.0
        for i in range(len(path) - 1):
            p *= self._link_success(path[i], path[i + 1])
        return float(max(0.0, min(1.0, p)))

    def _route_delay(self, path: List[int]) -> float:
        return float(sum(self._link_delay(path[i], path[i + 1]) for i in range(len(path) - 1)))

    def _link_success(self, u: int, v: int) -> float:
        """Link success probability for urban street canyon mmWave.
        
        Uses the CI path loss model with corner loss. The success probability is modeled
        as the probability that SINR exceeds a threshold z under Rayleigh fading:
            P(success) = P(SINR > z) = exp(-z / meanSINR)
        
        For outage links (PL > cutoff), returns a very small value (not exactly 0 to avoid
        numerical issues in some downstream code).
        
        Falls back to legacy model if USE_CI_MODEL is False.
        """
        use_ci = self._phys.get('USE_CI_MODEL', True)
        if not use_ci:
            return self._link_success_legacy(u, v)
        
        if u not in self.node_pos or v not in self.node_pos:
            return 0.0
        
        # Compute path loss and check outage
        pl_db, is_outage = self._path_loss_db(u, v)
        
        if is_outage:
            # Return very small value for outage (not exactly 0)
            return 1e-12
        
        # Convert PL to SINR
        sinr = self._pl_to_sinr(pl_db)
        
        if sinr <= 0.0:
            return 1e-12
        
        # Rayleigh fading success: P(h > z/sinr) = exp(-z/sinr)
        z = float(self._phys.get('Z', 4.0))
        expo = -z / max(1e-12, sinr)
        expo = max(-80.0, min(0.0, expo))
        
        return float(math.exp(expo))

    def _link_success_legacy(self, u: int, v: int) -> float:
        """Legacy link success model (for backward compatibility).
        
        Rayleigh fading RF model with simple distance-based path loss.
        """
        if u not in self.node_pos or v not in self.node_pos:
            return 0.0
        x1, y1 = self.node_pos[u]
        x2, y2 = self.node_pos[v]
        d = math.hypot(x1 - x2, y1 - y2)

        D0 = max(1e-6, float(self._phys.get('D0', 200.0)))
        alpha = float(self._phys.get('ALPHA', 1.7))
        P_NI = max(1e-12, self._effective_P_NI())
        PT = max(1e-12, float(self._phys.get('PT', 1.0)))
        z = max(1e-12, float(self._phys.get('Z', 1.0)))
        nlos_atten = float(self._phys.get('NLOS_ATTEN', 0.25))

        r = max(1e-6, d / D0)
        PT_eff = PT
        if self.mobility.is_blocked((x1, y1), (x2, y2)):
            PT_eff = max(1e-12, PT * max(0.0, min(1.0, nlos_atten)))

        expo = - (P_NI * (r ** alpha) * z) / PT_eff
        expo = max(-80.0, min(0.0, float(expo)))
        return float(math.exp(expo))

    def _link_delay(self, u: int, v: int) -> float:
        """One-way transmission delay (seconds) from Shannon capacity.
        
        Uses CI path loss model to compute SINR, then:
            Rate = B * log2(1 + SINR)
            Delay = MSG_BITS / Rate
        
        For outage links, returns MAX_DELAY (or a large value).
        Falls back to legacy model if USE_CI_MODEL is False.
        """
        use_ci = self._phys.get('USE_CI_MODEL', True)
        if not use_ci:
            return self._link_delay_legacy(u, v)
        
        B = float(self._phys.get('B', 800.0))
        msg_bits = float(self._phys.get('MSG_BITS', 4000.0))
        max_delay = float(self._phys.get('MAX_DELAY', 1.0))
        
        if B <= 0.0 or msg_bits <= 0.0:
            return 1e-3
        
        if u not in self.node_pos or v not in self.node_pos:
            return max_delay
        
        # Compute path loss and check outage
        pl_db, is_outage = self._path_loss_db(u, v)
        
        if is_outage:
            return max_delay
        
        # Convert PL to SINR
        sinr = self._pl_to_sinr(pl_db)
        
        # Shannon capacity: C = B * log2(1 + SINR)
        cap = B * math.log2(1.0 + max(0.0, sinr))
        
        # Ensure minimum rate to avoid division issues
        cap = max(1.0, cap)  # at least 1 bit/s
        
        # Delay = bits / rate
        t = msg_bits / cap
        
        return float(max(1e-6, min(max_delay, t)))

    def _link_delay_legacy(self, u: int, v: int) -> float:
        """Legacy link delay model (for backward compatibility)."""
        snr = self._link_snr_legacy(u, v)
        B = float(self._phys.get('B', 800.0))
        msg_bits = float(self._phys.get('MSG_BITS', 4000.0))
        if B <= 0.0 or msg_bits <= 0.0:
            return 1e-3
        cap = B * math.log2(1.0 + max(0.0, snr))
        if cap <= 1e-9:
            return float(self._phys.get('MAX_DELAY', 1.0))
        t = msg_bits / cap
        return float(max(1e-6, min(float(self._phys.get('MAX_DELAY', 1.0)), t)))

    def _link_snr(self, u: int, v: int) -> float:
        """Mean SINR for link (u, v) using CI path loss model.
        
        For outage links, returns 0.
        Falls back to legacy model if USE_CI_MODEL is False.
        """
        use_ci = self._phys.get('USE_CI_MODEL', True)
        if not use_ci:
            return self._link_snr_legacy(u, v)
        
        if u not in self.node_pos or v not in self.node_pos:
            return 0.0
        
        # Compute path loss and check outage
        pl_db, is_outage = self._path_loss_db(u, v)
        
        if is_outage:
            return 0.0
        
        # Convert PL to SINR
        sinr = self._pl_to_sinr(pl_db)
        return float(max(0.0, sinr))

    def _link_snr_legacy(self, u: int, v: int) -> float:
        """Legacy SNR model (for backward compatibility).
        
        Simple distance-based path loss with NLOS attenuation.
        """
        if u not in self.node_pos or v not in self.node_pos:
            return 0.0
        x1, y1 = self.node_pos[u]
        x2, y2 = self.node_pos[v]
        d = math.hypot(x1 - x2, y1 - y2)

        D0 = max(1e-6, float(self._phys.get('D0', 200.0)))
        alpha = float(self._phys.get('ALPHA', 1.7))
        P_NI = max(1e-12, self._effective_P_NI())
        PT = max(1e-12, float(self._phys.get('PT', 1.0)))
        nlos_atten = float(self._phys.get('NLOS_ATTEN', 0.25))

        r = max(1e-6, d / D0)
        PT_eff = PT
        if self.mobility.is_blocked((x1, y1), (x2, y2)):
            PT_eff = max(1e-12, PT * max(0.0, min(1.0, nlos_atten)))
        sinr = PT_eff * (r ** (-alpha)) / P_NI
        return float(max(0.0, sinr))

    def _effective_P_NI(self) -> float:
        """Effective (P_N + P_I) that grows with topology density to make optimization meaningful.

        We approximate interference as increasing with mean degree / edge density:
            edge_density = 2|E| / (n * max(1, k_ref))
        where k_ref is the target mean degree (~3).
        """
        base = float(self._phys.get('P_NI', 0.1))
        k = float(self._phys.get('INTERF_K', 0.0))
        if k <= 0.0:
            return max(1e-12, base)
        n = max(1, len(self.nodes))
        e = len(self.edges)
        # target mean degree ≈ 3 => target edges ≈ 1.5n
        e_target = max(1.0, 1.5 * float(n))
        density = float(max(0.0, e / e_target))
        return max(1e-12, base * (1.0 + k * density))

    # ==========================================================================
    # Urban Street Canyon mmWave Link Model (CI Path Loss + Corner Loss + Shadowing)
    # ==========================================================================

    def _invalidate_link_caches(self):
        """Invalidate link geometry and shadowing caches when positions change significantly."""
        # Compute a simple hash of node positions
        if not self.node_pos:
            new_hash = 0
        else:
            # Use sum of rounded positions as a cheap hash
            s = sum(int(x * 10) + int(y * 10) for x, y in self.node_pos.values())
            new_hash = hash(s) & 0xFFFFFFFF
        if new_hash != self._pos_hash:
            self._link_geo_cache.clear()
            self._pos_hash = new_hash
            # Note: shadowing cache is NOT cleared - it uses AR(1) update instead

    def _compute_link_geometry(self, u: int, v: int) -> Tuple[float, bool, float, int]:
        """Compute geometric features for link (u, v).
        
        Returns:
            d_euc: Euclidean distance (meters)
            is_los: True if line-of-sight (not blocked by buildings)
            d_path: Street canyon path distance (Manhattan distance for grid cities)
            n_corner: Number of corners in shortest street path (0 or 1 for simple model)
        
        This function is cached for performance.
        """
        key = tuple(sorted((u, v)))
        if key in self._link_geo_cache:
            return self._link_geo_cache[key]
        
        if u not in self.node_pos or v not in self.node_pos:
            result = (float('inf'), False, float('inf'), 99)
            return result
        
        x1, y1 = self.node_pos[u]
        x2, y2 = self.node_pos[v]
        
        # Euclidean distance
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        d_euc = math.hypot(dx, dy)
        
        # LOS check using existing building intersection
        is_los = not self.mobility.is_blocked((x1, y1), (x2, y2))
        
        # Manhattan distance (street canyon path) - appropriate for grid cities
        d_path = dx + dy
        
        # Corner count:
        # In a grid city, if both dx and dy are non-zero, the path must turn at least once.
        # More refined: check if aligned with grid (on same road segment)
        grid_stride = self.mobility.grid_stride
        eps = 2.0  # tolerance for "on same grid line"
        
        # Check if both nodes are on the same horizontal or vertical road
        same_h_road = abs(y1 - y2) < eps  # same horizontal road (same y)
        same_v_road = abs(x1 - x2) < eps  # same vertical road (same x)
        
        if same_h_road or same_v_road:
            # Direct path along one road - no corners
            n_corner = 0
            # For LOS cases on same road, use Euclidean (wave-guiding effect)
            if is_los:
                d_path = d_euc
        else:
            # Must turn at least once
            n_corner = 1
            # In complex scenarios, could count more corners, but 1 is a good approximation
        
        # For NLOS with corners, d_path is Manhattan distance
        # d_path should be at least d_euc
        d_path = max(d_euc, d_path)
        
        result = (d_euc, is_los, d_path, n_corner)
        self._link_geo_cache[key] = result
        return result

    def _get_shadowing_db(self, u: int, v: int, is_los: bool) -> float:
        """Get spatially correlated shadow fading value (dB) for link (u, v).
        
        Uses AR(1) process to maintain spatial correlation:
            S_new = rho * S_old + sqrt(1 - rho^2) * N(0, sigma)
        where rho = exp(-delta_move / d_cor)
        
        Args:
            u, v: Node IDs
            is_los: Whether the link is LOS (affects sigma and d_cor)
        
        Returns:
            Shadow fading value in dB (can be positive or negative)
        """
        key = tuple(sorted((u, v)))
        sigma = float(self._phys.get('PL_SIGMA_LOS' if is_los else 'PL_SIGMA_NLOS', 3.0 if is_los else 8.0))
        d_cor = float(self._phys.get('SHADOW_DCOR_LOS' if is_los else 'SHADOW_DCOR_NLOS', 10.0 if is_los else 13.0))
        
        if key in self._shadow_cache:
            old_s, old_los = self._shadow_cache[key]
            # If LOS state changed, reset shadowing
            if old_los != is_los:
                s_new = random.gauss(0.0, sigma)
            else:
                # Compute movement delta from previous positions
                delta_move = 0.0
                for nid in [u, v]:
                    if nid in self.node_pos and nid in self._prev_node_pos:
                        px, py = self._prev_node_pos[nid]
                        cx, cy = self.node_pos[nid]
                        delta_move += math.hypot(cx - px, cy - py)
                delta_move = delta_move / 2.0  # average of both nodes
                
                # AR(1) update
                rho = math.exp(-delta_move / max(1e-6, d_cor))
                innovation_std = sigma * math.sqrt(max(0.0, 1.0 - rho * rho))
                s_new = rho * old_s + random.gauss(0.0, innovation_std)
        else:
            # Initialize with random sample
            s_new = random.gauss(0.0, sigma)
        
        # Clamp to reasonable range (±3 sigma)
        s_new = max(-3.0 * sigma, min(3.0 * sigma, s_new))
        self._shadow_cache[key] = (s_new, is_los)
        return s_new

    def _path_loss_db(self, u: int, v: int) -> Tuple[float, bool]:
        """Compute path loss (dB) using CI model with corner loss for urban street canyon.
        
        CI (Close-In) model:
            PL(d) = FSPL(1m, fc) + 10 * n * log10(d) + X_sigma
        
        where:
            - FSPL(1m) = 32.4 + 20*log10(fc_GHz) for d=1m reference
            - n = path loss exponent (different for LOS/NLOS)
            - X_sigma = shadow fading (spatially correlated)
        
        For NLOS:
            - Use d_path (Manhattan/street distance) instead of d_euc
            - Add corner loss: n_corner * CORNER_LOSS_DB
        
        Key insight for urban mmWave:
            - LOS links on same road (n_corner=0): good wave-guiding, use PL_N_LOS
            - LOS links across roads (n_corner>0): still "LOS" but weaker, add corner penalty
            - NLOS links: much worse, use PL_N_NLOS + corner penalty
        
        Returns:
            (PL_dB, is_outage): Path loss in dB and whether link is in outage
        """
        d_euc, is_los, d_path, n_corner = self._compute_link_geometry(u, v)
        
        # Handle invalid/infinite distance
        if d_euc <= 0.0 or not math.isfinite(d_euc):
            return (float(self._phys.get('PL_CUTOFF_DB', 160.0)) + 10.0, True)
        
        # Hard distance limit - always outage beyond this regardless of path loss
        max_link_dist = float(self._phys.get('MAX_LINK_DIST', 0.0))
        if max_link_dist > 0.0 and d_euc > max_link_dist:
            return (float(self._phys.get('PL_CUTOFF_DB', 160.0)) + 10.0, True)
        
        fc_ghz = float(self._phys.get('FC_GHZ', 28.0))
        pl_cutoff = float(self._phys.get('PL_CUTOFF_DB', 105.0))
        corner_loss = float(self._phys.get('CORNER_LOSS_DB', 12.0))
        
        # FSPL at 1m reference: 32.4 + 20*log10(fc_GHz)
        # (derived from FSPL = 20*log10(4*pi*d*f/c) at d=1m)
        fspl_1m = 32.4 + 20.0 * math.log10(max(1e-3, fc_ghz))
        
        if is_los:
            # LOS: use Euclidean distance with LOS path loss exponent
            n = float(self._phys.get('PL_N_LOS', 1.8))
            d_use = max(1.0, d_euc)  # min 1m to avoid log(0)
            pl_db = fspl_1m + 10.0 * n * math.log10(d_use)
            
            # For LOS links that cross roads (n_corner > 0), add a smaller corner penalty
            # This models the fact that crossing perpendicular streets weakens the signal
            if n_corner > 0:
                pl_db += n_corner * corner_loss * 0.5  # 50% of NLOS corner loss
        else:
            # NLOS: use Manhattan/street path distance + full corner loss
            n = float(self._phys.get('PL_N_NLOS', 2.9))
            d_use = max(1.0, d_path)  # min 1m
            pl_db = fspl_1m + 10.0 * n * math.log10(d_use) + n_corner * corner_loss
        
        # Add spatially correlated shadow fading
        shadow_db = self._get_shadowing_db(u, v, is_los)
        pl_db += shadow_db
        
        # Clamp to physical range
        pl_db = max(0.0, pl_db)
        
        # Check outage
        is_outage = (pl_db > pl_cutoff)
        
        return (pl_db, is_outage)

    def _pl_to_sinr(self, pl_db: float) -> float:
        """Convert path loss (dB) to linear SINR.
        
        SINR = (PT_lin * gain_lin) / noise_lin
        where gain_lin = 10^(-PL_dB / 10)
        
        Args:
            pl_db: Path loss in dB
        
        Returns:
            Linear SINR value
        """
        pt_lin = float(self._phys.get('PT_LIN', 0.1))
        noise_lin = float(self._phys.get('NOISE_POWER_LIN', 1e-10))
        
        # Add interference contribution from edge density
        interf_k = float(self._phys.get('INTERF_K', 0.0))
        if interf_k > 0.0:
            n = max(1, len(self.nodes))
            e = len(self.edges)
            e_target = max(1.0, 1.5 * float(n))
            density = float(max(0.0, e / e_target))
            noise_lin = noise_lin * (1.0 + interf_k * density)
        
        noise_lin = max(1e-15, noise_lin)
        
        # Path loss to linear gain
        gain_lin = 10.0 ** (-pl_db / 10.0)
        
        # SINR
        sinr = (pt_lin * gain_lin) / noise_lin
        return max(0.0, sinr)
    
    def _territory_compactness(self, partitions: Dict[int, int]) -> float:
        """Measure territory spatial compactness (0-1, higher better).
        
        Compact = nodes close to their agent's centroid.
        """
        if not self.node_pos or not partitions:
            return 0.0
        
        # Compute centroids per agent
        agent_centroids: Dict[int, List[float]] = {}
        agent_counts: Dict[int, int] = {}
        for n, a in partitions.items():
            if n in self.node_pos:
                x, y = self.node_pos[n]
                if a not in agent_centroids:
                    agent_centroids[a] = [0.0, 0.0]
                    agent_counts[a] = 0
                agent_centroids[a][0] += x
                agent_centroids[a][1] += y
                agent_counts[a] += 1
        
        for a in agent_centroids:
            if agent_counts[a] > 0:
                agent_centroids[a][0] /= agent_counts[a]
                agent_centroids[a][1] /= agent_counts[a]
        
        # Compute average distance to centroid (normalized)
        total_dist = 0.0
        for n, a in partitions.items():
            if n in self.node_pos and a in agent_centroids:
                x, y = self.node_pos[n]
                cx, cy = agent_centroids[a]
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                total_dist += dist
        
        avg_dist = total_dist / max(1, len(partitions))
        # Normalize: assume max reasonable distance is 100 units
        compactness = max(0.0, 1.0 - avg_dist / 100.0)
        return float(compactness)
    
    def _territory_overlap_penalty(self, partitions: Dict[int, int]) -> float:
        """Penalty for overlapping convex hulls (0-1, higher worse).
        
        Uses convex hull intersection detection.
        """
        if not self.node_pos or not partitions:
            return 0.0
        
        try:
            ## SciPy removed: ConvexHull
            from matplotlib.path import Path  # type: ignore
            import numpy as np  # type: ignore
            
            agents = sorted(set(partitions.values()))
            if len(agents) <= 1:
                return 0.0
            
            # Build convex hulls
            agent_hulls: Dict[int, object] = {}
            for a in agents:
                nodes_a = [n for n, ag in partitions.items() if ag == a]
                if len(nodes_a) >= 3:
                    points = np.array([self.node_pos[n] for n in nodes_a if n in self.node_pos])
                    if len(points) >= 3:
                        try:
                            ## SciPy removed: hull = ConvexHull(points)
                            agent_hulls[a] = points[hull.vertices]
                        except:  # noqa: E722
                            pass
            
            # Count overlaps
            overlap_count = 0
            total_pairs = 0
            agent_ids = sorted(agent_hulls.keys())
            for i, a1 in enumerate(agent_ids):
                for a2 in agent_ids[i+1:]:
                    total_pairs += 1
                    hull1 = agent_hulls[a1]
                    hull2 = agent_hulls[a2]
                    
                    path1 = Path(hull1)
                    path2 = Path(hull2)
                    
                    if any(path1.contains_points(hull2)) or any(path2.contains_points(hull1)):  # type: ignore
                        overlap_count += 1
            
            if total_pairs == 0:
                return 0.0
            
            overlap_ratio = overlap_count / total_pairs
            return float(overlap_ratio)
            
        except ImportError:
            # Fallback: use simpler metric based on centroid distances
            return self._territory_overlap_penalty_simple(partitions)
    
    def _territory_overlap_penalty_simple(self, partitions: Dict[int, int]) -> float:
        """Simplified overlap penalty using centroid distances."""
        # Compute centroids
        agent_centroids: Dict[int, List[float]] = {}
        agent_counts: Dict[int, int] = {}
        for n, a in partitions.items():
            if n in self.node_pos:
                x, y = self.node_pos[n]
                if a not in agent_centroids:
                    agent_centroids[a] = [0.0, 0.0]
                    agent_counts[a] = 0
                agent_centroids[a][0] += x
                agent_centroids[a][1] += y
                agent_counts[a] += 1
        
        for a in agent_centroids:
            if agent_counts[a] > 0:
                agent_centroids[a][0] /= agent_counts[a]
                agent_centroids[a][1] /= agent_counts[a]
        
        agents = list(agent_centroids.keys())
        if len(agents) <= 1:
            return 0.0
        
        # Penalty based on how close centroids are (closer = more overlap likely)
        min_dist = float('inf')
        for i, a1 in enumerate(agents):
            for a2 in agents[i+1:]:
                cx1, cy1 = agent_centroids[a1]
                cx2, cy2 = agent_centroids[a2]
                dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                min_dist = min(min_dist, dist)
        
        # Normalize: penalty high if centroids within 50 units
        threshold = 50.0
        penalty = max(0.0, 1.0 - min_dist / threshold)
        return float(penalty)

    # ------------------------------
    # Legacy proxies (deprecated, kept for compatibility)
    # ------------------------------

    def _success_proxy(self) -> float:
        """Legacy success proxy. Use _consensus_success instead."""
        return self._consensus_success()

    def _delay_proxy(self) -> float:
        """Legacy delay proxy. Use _latency_cost instead."""
        return self._latency_cost()

    def _energy_proxy(self) -> float:
        """Legacy energy proxy. Use _energy_cost instead."""
        return self._energy_cost() * 100  # Scale to edge count for backward compat

    def _ensure_connectivity(self):
        # Simple repair: if no edges and >=2 nodes, reconnect ring
        if not self.edges and len(self.nodes) >= 2:
            for i in range(len(self.nodes)):
                u = self.nodes[i]
                v = self.nodes[(i + 1) % len(self.nodes)]
                self.edges.add(tuple(sorted((u, v))))
        # Remove any edges that reference missing nodes (safety)
        self.edges = {e for e in self.edges if e[0] in self.nodes and e[1] in self.nodes}

    def _build_observations(self, partitions: Dict[int, int]) -> Dict[int, Observation]:
        # Build fixed-length per-agent observation with zero-padded slots
        obs_by_agent: Dict[int, Observation] = {}
        node_to_agent = partitions

        # Precompute adjacency for neighbor lookup
        adj_map: Dict[int, List[int]] = {n: [] for n in self.nodes}
        for (u, v) in self.edges:
            if u in adj_map:
                adj_map[u].append(v)
            if v in adj_map:
                adj_map[v].append(u)

        # Edge weights for halo/gnn: use normalized SNR in [0,1]
        def w_snr(u: int, v: int) -> float:
            s = self._link_snr(u, v)
            return float(s / (1.0 + s))

        edge_list: List[Tuple[int, int, float]] = [(u, v, w_snr(u, v)) for (u, v) in sorted(self.edges)]

        # Centroids for delta_pos normalization (per agent)
        agent_centroids: Dict[int, Tuple[float, float]] = {}
        agent_counts: Dict[int, int] = {}
        for n, a in node_to_agent.items():
            if n in self.node_pos:
                x, y = self.node_pos[n]
                cx, cy = agent_centroids.get(a, (0.0, 0.0))
                agent_centroids[a] = (cx + x, cy + y)
                agent_counts[a] = agent_counts.get(a, 0) + 1
        for a, (sx, sy) in list(agent_centroids.items()):
            c = agent_counts.get(a, 0)
            if c > 0:
                agent_centroids[a] = (sx / c, sy / c)
            else:
                agent_centroids[a] = (0.0, 0.0)

        bbox = tuple((self.cfg.get('region') or {}).get('bbox', (-120.0, -120.0, 120.0, 120.0)))
        span = max(1e-6, max(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])))
        pos_scale = 0.5 * span

        for aid in set(node_to_agent.values()):
            # Collect nodes in this partition
            nodes_local = [n for n, a in node_to_agent.items() if a == aid]
            nodes_local_set = set(nodes_local)
            
            # Halo candidate nodes = 1-hop neighbors outside local
            neighbors_out_1hop: Set[int] = set()
            boundary_nodes_set: Set[int] = set()
            for u in nodes_local:
                for v in adj_map.get(u, []):
                    if v not in nodes_local_set:
                        neighbors_out_1hop.add(v)
                        boundary_nodes_set.add(u)  # u is a boundary node
            
            # 2-hop halo = neighbors of 1-hop halo that are still outside local
            neighbors_out_2hop: Set[int] = set()
            for h in neighbors_out_1hop:
                for v in adj_map.get(h, []):
                    if v not in nodes_local_set and v not in neighbors_out_1hop:
                        neighbors_out_2hop.add(v)
            
            halo_nodes = list(neighbors_out_1hop | neighbors_out_2hop)
            nodes_halo_hint = list(neighbors_out_1hop)  # For HaloSummary (1-hop only)
            
            # Compute existing edges for deletion candidates
            existing_internal_edges: List[Tuple[int, int]] = []
            existing_cross_edges: List[Tuple[int, int]] = []
            for (u, v) in self.edges:
                u_in_local = u in nodes_local_set
                v_in_local = v in nodes_local_set
                if u_in_local and v_in_local:
                    # Both endpoints in this partition - internal edge
                    existing_internal_edges.append((u, v))
                elif u_in_local or v_in_local:
                    # One endpoint in this partition - cross edge (this agent can propose deletion)
                    existing_cross_edges.append((u, v))
            # Build slots (fixed N)
            slots: List[Slot] = []
            cx, cy = agent_centroids.get(aid, (0.0, 0.0))
            for idx in range(self.slots_n):
                if idx < len(nodes_local):
                    nid = nodes_local[idx]
                    x, y = self.node_pos.get(nid, (0.0, 0.0))
                    px, py = self._prev_node_pos.get(nid, (x, y))
                    vx, vy = (x - px), (y - py)
                    # Simple per-node summary from its current graph neighbors
                    nbs = adj_map.get(nid, [])
                    if nbs:
                        snrs = [self._link_snr(nid, nb) for nb in nbs[:6]]
                        ds = [self._link_delay(nid, nb) for nb in nbs[:6]]
                        snr_mean = sum(snrs) / max(1, len(snrs))
                        d_mean = sum(ds) / max(1, len(ds))
                    else:
                        snr_mean = 0.0
                        d_mean = float(self._phys['MAX_DELAY'])
                    snr_norm = float(snr_mean / (1.0 + snr_mean))
                    rtt = float(min(self._phys['MAX_DELAY'], 2.0 * d_mean))
                    loss = float(1.0 - (1.0 - math.exp(-max(0.0, snr_mean))))
                    slots.append(Slot(
                        node_id=int(nid),
                        mask=1,
                        delta_pos=((x - cx) / pos_scale, (y - cy) / pos_scale),
                        delta_vel=(vx / max(1e-6, self.mobility.speed_mean), vy / max(1e-6, self.mobility.speed_mean)),
                        battery=1.0,
                        link_snr=snr_norm,
                        link_loss=loss,
                        link_rtt=rtt,
                        role_bit=0,
                    ))
                else:
                    slots.append(Slot(
                        node_id=-1,
                        mask=0,
                        delta_pos=(0.0, 0.0),
                        delta_vel=(0.0, 0.0),
                        battery=0.0,
                        link_snr=0.0,
                        link_loss=0.0,
                        link_rtt=0.0,
                        role_bit=0,
                    ))

            # Build halo summary using configured mode
            halo_summary: HaloSummary = self.halo.build_for_partition(
                edge_list=edge_list,
                nodes_local=nodes_local,
                nodes_halo_hint=nodes_halo_hint,
                node_pos=self.node_pos if self.halo.mode == 'radius' else None,
            )
            
            # Build normalized positions for all relevant nodes (internal + boundary + halo)
            # This enables Actor to properly sort cross-edge candidates by actual distance
            node_positions: Dict[int, Tuple[float, float]] = {}
            for nid in nodes_local:
                x, y = self.node_pos.get(nid, (0.0, 0.0))
                node_positions[nid] = ((x - cx) / pos_scale, (y - cy) / pos_scale)
            for nid in halo_nodes:
                x, y = self.node_pos.get(nid, (0.0, 0.0))
                node_positions[nid] = ((x - cx) / pos_scale, (y - cy) / pos_scale)
            
            obs_by_agent[int(aid)] = Observation(
                slots=slots,
                # Provide local edges among visible slots only; Actor will ignore edges whose endpoints are not in slots.
                adj_local=edge_list,
                halo_summary=halo_summary,
                time_feat=(float(self.time_step % 60) / 60.0, float(len(nodes_local)) / max(1.0, len(self.nodes))),
                msg_in=[0.0] * self.msg_dim,
                role_id=int(aid),
                # NEW: For hierarchical action space
                internal_nodes=nodes_local,
                boundary_nodes=list(boundary_nodes_set),
                halo_nodes=halo_nodes,
                existing_internal_edges=existing_internal_edges,
                existing_cross_edges=existing_cross_edges,
                node_positions=node_positions,
            )

        return obs_by_agent

    def _init_positions(self):
        # Delegate to mobility to place nodes on roads
        self.mobility.reset(self.nodes, self.node_pos)


class CityGridMobility:
    """City-grid mobility model with occlusion physics.

    - Generates roads (grid) and buildings (in-between).
    - Enforces vehicles stay on roads.
    - Provides occlusion checks.
    """

    HEADINGS = ['N', 'E', 'S', 'W']
    DIR_VEC = {
        'N': (0.0, 1.0),
        'E': (1.0, 0.0),
        'S': (0.0, -1.0),
        'W': (-1.0, 0.0),
    }

    def __init__(self, cfg: dict):
        ccfg = (cfg.get('city') or {})
        self.enable = bool(ccfg.get('enable', True))
        self.grid_stride = float(ccfg.get('grid_stride', 50.0))
        self.road_width = float(ccfg.get('road_width', 22.0))
        self.speed_mean = float(ccfg.get('speed_mean', 3.0))
        self.speed_std = float(ccfg.get('speed_std', 0.5))
        self.turn_prob = float(ccfg.get('turn_prob', 0.2))
        self.wrap = bool(ccfg.get('wrap', True))
        region = (cfg.get('region') or {})
        self.bbox = tuple(region.get('bbox', (-75.0, -75.0, 75.0, 75.0)))  # 3x3 grid at stride=50
        # state
        self.heading: Dict[int, str] = {}
        self.buildings: List[Tuple[float, float, float, float]] = [] # x,y,w,h
        self._generate_buildings()

    def _generate_buildings(self):
        """Generate dense urban street canyon buildings.
        
        This creates a realistic urban mmWave environment where:
        - Buildings fill each city block, leaving only street corridors
        - Streets form a grid at regular intervals (grid_stride)
        - Street width uses city.road_width
        - Buildings occupy the interior of each block
        
        This produces the "street canyon" effect seen in mmWave propagation studies:
        - LOS only along the same street corridor
        - NLOS/outage when crossing between parallel streets
        """
        self.buildings.clear()
        xmin, ymin, xmax, ymax = self.bbox
        stride = self.grid_stride
        
        # Street width (m), default 22m for 100m blocks
        road_half_width = max(1.0, self.road_width / 2.0)
        
        # Calculate grid range
        grid_x_min = int(math.floor(xmin / stride))
        grid_x_max = int(math.ceil(xmax / stride))
        grid_y_min = int(math.floor(ymin / stride))
        grid_y_max = int(math.ceil(ymax / stride))
        
        rng = random.Random(12345)  # Static seed for consistent map
        
        # For each city block (between intersections), create a large building
        for ix in range(grid_x_min, grid_x_max):
            for iy in range(grid_y_min, grid_y_max):
                # Block boundaries (inside the street corridors)
                # Roads are AT grid lines; buildings are between them
                block_x_min = ix * stride + road_half_width
                block_x_max = (ix + 1) * stride - road_half_width
                block_y_min = iy * stride + road_half_width
                block_y_max = (iy + 1) * stride - road_half_width
                
                block_w = block_x_max - block_x_min
                block_h = block_y_max - block_y_min
                
                if block_w <= 0 or block_h <= 0:
                    continue
                
                # Skip some blocks randomly for variety (parks, plazas)
                if rng.random() < 0.1:  # 10% chance to skip (empty block)
                    continue
                
                # Option 1: Single large building filling most of the block (70% chance)
                # Option 2: Multiple smaller buildings with gaps (30% chance)
                if rng.random() < 0.7:
                    # Single building with small setback
                    setback = rng.uniform(1.0, 3.0)
                    bx = block_x_min + setback
                    by = block_y_min + setback
                    bw = block_w - 2 * setback
                    bh = block_h - 2 * setback
                    if bw > 2.0 and bh > 2.0:
                        self.buildings.append((bx, by, bw, bh))
                else:
                    # Multiple buildings with internal courtyard/alley
                    # Split block into 2-4 sub-buildings
                    n_split = rng.choice([2, 2, 3, 4])
                    gap = rng.uniform(2.0, 4.0)
                    
                    if n_split == 2:
                        # Split horizontally or vertically
                        if rng.random() < 0.5:
                            # Horizontal split
                            h1 = (block_h - gap) * rng.uniform(0.4, 0.6)
                            h2 = block_h - gap - h1
                            self.buildings.append((block_x_min + 1, block_y_min + 1, block_w - 2, h1 - 1))
                            self.buildings.append((block_x_min + 1, block_y_min + h1 + gap, block_w - 2, h2 - 1))
                        else:
                            # Vertical split
                            w1 = (block_w - gap) * rng.uniform(0.4, 0.6)
                            w2 = block_w - gap - w1
                            self.buildings.append((block_x_min + 1, block_y_min + 1, w1 - 1, block_h - 2))
                            self.buildings.append((block_x_min + w1 + gap, block_y_min + 1, w2 - 1, block_h - 2))
                    else:
                        # 4-quadrant split
                        gap_h = gap / 2
                        w1 = (block_w - gap) / 2
                        h1 = (block_h - gap) / 2
                        quads = [
                            (block_x_min + 1, block_y_min + 1, w1 - 1, h1 - 1),
                            (block_x_min + w1 + gap, block_y_min + 1, w1 - 1, h1 - 1),
                            (block_x_min + 1, block_y_min + h1 + gap, w1 - 1, h1 - 1),
                            (block_x_min + w1 + gap, block_y_min + h1 + gap, w1 - 1, h1 - 1),
                        ]
                        # Randomly skip one quadrant for L-shape or U-shape
                        skip_idx = rng.randint(0, 3) if rng.random() < 0.3 else -1
                        for i, (bx, by, bw, bh) in enumerate(quads):
                            if i != skip_idx and bw > 1 and bh > 1:
                                self.buildings.append((bx, by, bw, bh))

    def reset(self, nodes: List[int], node_pos: Dict[int, Tuple[float, float]]):
        self.heading.clear()
        # Place nodes on roads WITHIN the bbox
        grid_lines_x = []
        grid_lines_y = []
        stride = self.grid_stride
        xmin, ymin, xmax, ymax = self.bbox
        
        # Build valid grid lines strictly within bbox
        curr = math.ceil(xmin / stride) * stride  # Start from first grid line inside bbox
        while curr <= xmax:  # Don't exceed bbox
            grid_lines_x.append(curr)
            curr += stride
        curr = math.ceil(ymin / stride) * stride
        while curr <= ymax:
            grid_lines_y.append(curr)
            curr += stride
        
        # Ensure we have at least one grid line
        if not grid_lines_x:
            grid_lines_x = [(xmin + xmax) / 2]
        if not grid_lines_y:
            grid_lines_y = [(ymin + ymax) / 2]

        for n in nodes:
            # Pick a road (either H or V)
            if random.random() < 0.5:
                # Horizontal road: fixed y, random x along the road
                y = random.choice(grid_lines_y)
                x = random.uniform(xmin, xmax)
                heading = random.choice(['E', 'W'])
            else:
                # Vertical road: fixed x, random y along the road
                x = random.choice(grid_lines_x)
                y = random.uniform(ymin, ymax)
                heading = random.choice(['N', 'S'])
            node_pos[n] = (x, y)
            self.heading[n] = heading

    def add_node(self, nid: int, pos: Tuple[float, float], heading: object | None = None):
        self.heading[nid] = (heading if isinstance(heading, str) and heading in self.DIR_VEC else random.choice(self.HEADINGS))  # type: ignore

    def remove_node(self, nid: int):
        self.heading.pop(nid, None)

    def is_blocked(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if line segment p1-p2 intersects any building."""
        x1, y1 = p1
        x2, y2 = p2
        for (bx, by, w, h) in self.buildings:
            # Simple AABB intersection with line segment? 
            # Or line-AABB. Cohen-Sutherland or Liang-Barsky is better but simple sampling is fine for RL env.
            # Building is [bx, bx+w] x [by, by+h]
            # Check bounding box filter first
            rx_min, rx_max = min(x1, x2), max(x1, x2)
            ry_min, ry_max = min(y1, y2), max(y1, y2)
            if rx_max < bx or rx_min > bx+w or ry_max < by or ry_min > by+h:
                continue
            
            # Liang-Barsky / Intersection test
            # Top
            if self._segment_intersects_rect(p1, p2, bx, by, w, h):
                return True
        return False

    def _segment_intersects_rect(self, p1, p2, x, y, w, h):
        # Check intersection with 4 lines of rect
        lines = [
            ((x, y), (x+w, y)),
            ((x, y), (x, y+h)),
            ((x+w, y), (x+w, y+h)),
            ((x, y+h), (x+w, y+h))
        ]
        for l in lines:
            if self._ccw(p1, l[0], l[1]) != self._ccw(p2, l[0], l[1]) and \
               self._ccw(p1, p2, l[0]) != self._ccw(p1, p2, l[1]):
                return True
        # Also check if p1 or p2 is INSIDE
        if (x <= p1[0] <= x+w and y <= p1[1] <= y+h): return True
        if (x <= p2[0] <= x+w and y <= p2[1] <= y+h): return True
        return False

    def _ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def update_config(self, cfg: dict):
        ccfg = (cfg.get('city') or {})
        self.enable = bool(ccfg.get('enable', self.enable))
        self.grid_stride = float(ccfg.get('grid_stride', self.grid_stride))
        self.speed_mean = float(ccfg.get('speed_mean', self.speed_mean))
        self.speed_std = float(ccfg.get('speed_std', self.speed_std))
        self.turn_prob = float(ccfg.get('turn_prob', self.turn_prob))
        self.wrap = bool(ccfg.get('wrap', self.wrap))
        region = (cfg.get('region') or {})
        bbox = region.get('bbox')
        if bbox is not None:
            self.bbox = tuple(bbox)


    def step_nodes(self, nodes: List[int], node_pos: Dict[int, Tuple[float, float]]):
        stride = self.grid_stride
        for n in nodes:
            p = node_pos.get(n)
            if p is None:
                continue
            h = self.heading.get(n, 'E')
            
            # Correction to exact grid line drift
            # If heading is E/W, y should be snapped
            if h in ('E', 'W'):
                p = (p[0], self._snap(p[1], stride))
            else:
                p = (self._snap(p[0], stride), p[1])
            
            np = self._step_point(p, h)
            node_pos[n] = np
            self.heading[n] = self._maybe_turn(np, h)

    def step_joiners(self, joiners: List[Dict[str, object]]):
        for j in joiners:
            p = j['pos']  # type: ignore
            h = j.get('heading', 'E')  # type: ignore
            np = self._step_point(p, h)
            j['pos'] = np
            # update heading at intersections
            j['heading'] = self._maybe_turn(np, h)

    def sample_offnet_joiner(self) -> Dict[str, object]:
        # Spawn on a random road segment outside bbox with a heading pointing inward
        stride = self.grid_stride
        xmin, ymin, xmax, ymax = self.bbox
        # choose a side
        side = random.choice(['left', 'right', 'bottom', 'top'])
        if side == 'left':
            x = xmin - stride
            y = self._snap(random.uniform(ymin, ymax), stride)
            heading = 'E'
        elif side == 'right':
            x = xmax + stride
            y = self._snap(random.uniform(ymin, ymax), stride)
            heading = 'W'
        elif side == 'bottom':
            x = self._snap(random.uniform(xmin, xmax), stride)
            y = ymin - stride
            heading = 'N'
        else:
            x = self._snap(random.uniform(xmin, xmax), stride)
            y = ymax + stride
            heading = 'S'
        return {'pos': (x, y), 'heading': heading}

    def _step_point(self, pos: Tuple[float, float], heading: str) -> Tuple[float, float]:
        # speed truncated normal
        v = random.gauss(self.speed_mean, self.speed_std)
        v = max(0.1, min(self.speed_mean + 3 * self.speed_std, v)) # Ensure non-zero speed
        dx, dy = self.DIR_VEC.get(heading, (1.0, 0.0))
        nx = pos[0] + dx * v
        ny = pos[1] + dy * v
        # at intersection, maybe turn
        # heading2 = self._maybe_turn((nx, ny), heading) # This was removed, _maybe_turn is called after
        # wrap or not
        if self.wrap:
            nx, ny = self._wrap((nx, ny))
        # update heading state if associated with a node id later by caller
        # return new position; caller updates heading separately
        # But we can store heading by reverse lookup; simpler to return pos and let caller set heading
        # For nodes we own heading map; update it
        # Note: heading2 computed from snapped intersection coords
        return (nx, ny)

    def _maybe_turn(self, pos: Tuple[float, float], heading: str) -> str:
        # check if near an intersection (within half stride)
        stride = self.grid_stride
        # Check intersection proximity
        on_x = abs(pos[0] - self._snap(pos[0], stride)) < 2.0 # tolerance
        on_y = abs(pos[1] - self._snap(pos[1], stride)) < 2.0
        
        if on_x and on_y and random.random() < self.turn_prob:
            # choose a new orthogonal direction (including straight)
            if heading in ('N', 'S'):
                return random.choice(['N', 'E', 'W'])
            else:
                return random.choice(['E', 'N', 'S'])
        return heading

    def _wrap(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bbox
        x, y = pos
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0: return pos
        while x < xmin:
            x += w
        while x > xmax:
            x -= w
        while y < ymin:
            y += h
        while y > ymax:
            y -= h
        return (x, y)

    @staticmethod
    def _snap(x: float, stride: float) -> float:
        if stride <= 0: return x
        return round(x / stride) * stride


