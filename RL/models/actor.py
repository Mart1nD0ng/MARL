from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.data_structures import Observation, Action, EdgeEdit


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # x: [N, F], adj: [N, N] (symmetric, with self-loops)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = adj / deg
        h = norm_adj @ x
        return self.lin(h)


class Actor(nn.Module):
    """GNN + LSTM Actor.

    - Builds slot-level node features from Observation
    - Applies 2-layer GCN with mean pooling
    - Concats halo + time + role + msg to pooled feature
    - LSTM to maintain temporal context
    - Heads: edge-candidate logits, op logits, message, value
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg.get('model', {}) or {}
        self.slots = int(model_cfg.get('slots', 10))
        self.msg_dim = int(model_cfg.get('msg_dim', 32))
        self.lstm_hidden = int(model_cfg.get('lstm_hidden', 128))

        # Node feature: mask, battery, snr, loss, rtt, role_bit, dpos(2), dvel(2)
        self.node_fdim = 1 + 1 + 1 + 1 + 1 + 1 + 2 + 2
        gnn_hidden = 64
        self.gcn1 = SimpleGCNLayer(self.node_fdim, gnn_hidden)
        self.gcn2 = SimpleGCNLayer(gnn_hidden, gnn_hidden)

        halo_dim = 8 + 3 + 1 + 1 + 4 * 4
        time_dim = 2
        role_dim = 1
        post_dim = gnn_hidden + halo_dim + time_dim + self.msg_dim + role_dim

        self.post = nn.Sequential(
            nn.Linear(post_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.lstm = nn.LSTM(128, self.lstm_hidden, batch_first=True)

        # Policy heads - Simplified Hierarchical Action Space
        # === 动作空间对称性设计 ===
        # ADD和DEL候选数量相同，确保策略不会偏向某一类型
        self.max_internal_candidates = 4   # Internal edges within partition
        self.max_cross_candidates = 4      # Cross-partition edges
        self.max_candidates = self.max_internal_candidates + self.max_cross_candidates
        
        # Operation type head: 4 choices (internal_add, internal_del, cross_add, cross_del)
        self.head_op_type = nn.Linear(self.lstm_hidden, 4)
        # Edge index head: select from candidates (max of internal or cross)
        self.head_edge_idx = nn.Linear(self.lstm_hidden, max(self.max_internal_candidates, self.max_cross_candidates))
        
        # Legacy heads for backward compatibility
        self.head_internal = nn.Linear(self.lstm_hidden, self.max_internal_candidates)
        self.head_cross = nn.Linear(self.lstm_hidden, self.max_cross_candidates)
        self.head_op = nn.Linear(self.lstm_hidden, 2)
        # Message and value heads
        self.head_msg = nn.Linear(self.lstm_hidden, self.msg_dim)
        self.head_value = nn.Linear(self.lstm_hidden, 1)
        
        # Legacy compatibility
        self.head_edge = self.head_internal

        # IMPORTANT: RNN state must be per-agent. Using a single shared hidden state across
        # different agents breaks CTDE/DEX and makes learning unstable.
        self._lstm_state: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        # Distance thresholds for candidate filtering (meters)
        self.max_internal_dist = 60.0  # Internal edges should be short
        self.max_cross_dist = 80.0     # Cross edges can be slightly longer
        
        # === 条件熵正则化系数 ===
        # 用于计算分层动作空间的条件熵
        self.entropy_coef_op = 1.0      # 操作类型的熵系数
        self.entropy_coef_edge = 1.5    # 边选择的熵系数 (更高以鼓励探索)

    def reset_rnn(self):
        """Clear all per-agent recurrent states."""
        self._lstm_state.clear()

    def _init_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        h0 = torch.zeros(1, 1, self.lstm_hidden, device=device)
        c0 = torch.zeros(1, 1, self.lstm_hidden, device=device)
        return h0, c0

    def _node_features(self, obs: Observation) -> torch.Tensor:
        feats: List[List[float]] = []
        for s in obs.slots[: self.slots]:
            feats.append([
                float(s.mask),
                float(s.battery),
                float(s.link_snr),
                float(s.link_loss),
                float(s.link_rtt),
                float(s.role_bit),
                float(s.delta_pos[0]), float(s.delta_pos[1]),
                float(s.delta_vel[0]), float(s.delta_vel[1]),
            ])
        while len(feats) < self.slots:
            feats.append([0.0] * (self.node_fdim))
        x = torch.tensor(feats, dtype=torch.float32)
        return x

    def _adjacency_from_obs(self, obs: Observation) -> torch.Tensor:
        # Map node_id -> slot index
        id_to_idx = {s.node_id: i for i, s in enumerate(obs.slots[: self.slots]) if s.mask > 0}
        N = self.slots
        A = torch.zeros((N, N), dtype=torch.float32)
        for (i, j, w) in obs.adj_local:
            if i in id_to_idx and j in id_to_idx:
                u, v = id_to_idx[i], id_to_idx[j]
                A[u, v] = max(A[u, v].item(), float(w))
                A[v, u] = max(A[v, u].item(), float(w))
        A += torch.eye(N, dtype=torch.float32)
        return A

    def _halo_vector(self, obs: Observation) -> torch.Tensor:
        hs = obs.halo_summary
        vec: List[float] = []
        vec.extend([float(x) for x in hs.deg_hist])
        vec.extend([float(x) for x in hs.snr_stats])
        vec.append(float(hs.min_degree))
        vec.append(float(hs.cross_candidates))
        for a in hs.topk_anchor:
            vec.extend([float(a[0]), float(a[1]), float(a[2]), float(a[3])])
        return torch.tensor(vec, dtype=torch.float32)

    def _post_embed(self, obs: Observation) -> torch.Tensor:
        x = self._node_features(obs)
        A = self._adjacency_from_obs(obs)
        h1 = F.relu(self.gcn1(x, A))
        h2 = F.relu(self.gcn2(h1, A))
        pooled = h2.mean(dim=0)

        halo_v = self._halo_vector(obs)
        time_v = torch.tensor([float(obs.time_feat[0]), float(obs.time_feat[1])], dtype=torch.float32)
        msg_in = torch.tensor([float(v) for v in obs.msg_in], dtype=torch.float32)
        role = torch.tensor([float(obs.role_id)], dtype=torch.float32)

        z = torch.cat([pooled, halo_v, time_v, msg_in, role], dim=0)
        z = self.post(z)
        return z

    def _lstm_forward(self, z: torch.Tensor, role_id: int, mutate_state: bool = True) -> torch.Tensor:
        y_in = z.view(1, 1, -1)
        if mutate_state:
            st = self._lstm_state.get(int(role_id))
            if st is None:
                st = self._init_state()
            y, st2 = self.lstm(y_in, st)
            self._lstm_state[int(role_id)] = st2
        else:
            # Evaluate with zeroed hidden to avoid mutating state during PPO update
            h0, c0 = self._init_state()
            y, _ = self.lstm(y_in, (h0, c0))
        return y[:, -1, :].squeeze(0)

    def _candidates(self, obs: Observation) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Generate hierarchical candidate edges for ADD and DEL operations separately.
        
        Simplified action space design:
        - Internal actions: Add short edges, delete long edges within partition
        - Cross actions: Add edges to HALO, delete long cross-partition edges
        
        NOTE: Uses normalized positions from obs.node_positions which includes
        both internal nodes AND halo nodes for proper cross-edge candidate sorting.
        
        Returns:
            (internal_add_candidates, cross_add_candidates, 
             internal_del_candidates, cross_del_candidates)
        """
        # Use node_positions from Observation (includes internal + halo nodes)
        node_pos: Dict[int, Tuple[float, float]] = getattr(obs, 'node_positions', None) or {}
        
        # Fallback to slots if node_positions not available
        if not node_pos:
            for s in obs.slots:
                if s.mask > 0:
                    node_pos[s.node_id] = s.delta_pos
        
        # Get node sets and existing edges
        internal_nodes = getattr(obs, 'internal_nodes', None)
        boundary_nodes = getattr(obs, 'boundary_nodes', None)
        halo_nodes = getattr(obs, 'halo_nodes', None)
        existing_internal = set(tuple(sorted(e)) for e in getattr(obs, 'existing_internal_edges', []))
        existing_cross = set(tuple(sorted(e)) for e in getattr(obs, 'existing_cross_edges', []))
        
        if internal_nodes is None:
            # Fallback to old behavior
            valid_nodes = [s.node_id for s in obs.slots if s.mask > 0]
            pairs = [(int(u), int(v)) for u, v in itertools.combinations(valid_nodes[:6], 2)]
            return pairs[:self.max_internal_candidates], [], pairs[:4], []
        
        internal_set = set(internal_nodes)
        boundary_set = set(boundary_nodes) if boundary_nodes else set()
        halo_set = set(halo_nodes) if halo_nodes else set()
        
        # ===== ADD CANDIDATES: Non-existing edges, prefer short distance =====
        internal_add_cands: List[Tuple[float, int, int]] = []
        for u, v in itertools.combinations(internal_nodes, 2):
            e = tuple(sorted((u, v)))
            if e in existing_internal:
                continue  # Skip existing edges
            if u in node_pos and v in node_pos:
                dx = node_pos[u][0] - node_pos[v][0]
                dy = node_pos[u][1] - node_pos[v][1]
                d_norm = math.sqrt(dx*dx + dy*dy)
                internal_add_cands.append((d_norm, int(u), int(v)))
        internal_add_cands.sort(key=lambda x: x[0])  # Prefer shorter edges
        internal_add_out = [(u, v) for _, u, v in internal_add_cands[:self.max_internal_candidates]]
        
        # Cross-partition ADD candidates: boundary -> halo
        cross_add_cands: List[Tuple[float, int, int]] = []
        for u in boundary_nodes or []:
            for v in halo_nodes or []:
                e = tuple(sorted((u, v)))
                if e in existing_cross:
                    continue  # Skip existing
                # Compute distance for cross edges too
                if u in node_pos and v in node_pos:
                    dx = node_pos[u][0] - node_pos[v][0]
                    dy = node_pos[u][1] - node_pos[v][1]
                    d_norm = math.sqrt(dx*dx + dy*dy)
                else:
                    d_norm = 0.5  # Default medium distance
                cross_add_cands.append((d_norm, int(u), int(v)))
        cross_add_cands.sort(key=lambda x: x[0])  # Prefer shorter cross edges
        cross_add_out = [(u, v) for _, u, v in cross_add_cands[:self.max_cross_candidates]]
        
        # ===== DEL CANDIDATES: All existing edges, sorted by distance =====
        # Internal DEL: ALL existing internal edges, longest first (to reduce internal redundancy)
        internal_del_cands: List[Tuple[float, int, int]] = []
        for u, v in existing_internal:
            if u in node_pos and v in node_pos:
                dx = node_pos[u][0] - node_pos[v][0]
                dy = node_pos[u][1] - node_pos[v][1]
                d_norm = math.sqrt(dx*dx + dy*dy)
            else:
                d_norm = 0.3  # Default if no position info
            internal_del_cands.append((d_norm, int(u), int(v)))
        internal_del_cands.sort(key=lambda x: -x[0])  # Longest first - reduce redundancy
        internal_del_out = [(u, v) for _, u, v in internal_del_cands[:self.max_internal_candidates]]
        
        # Cross-partition DEL candidates: All existing cross edges, longest first
        cross_del_cands: List[Tuple[float, int, int]] = []
        for u, v in existing_cross:
            u_in = u in internal_set
            v_in = v in internal_set
            local_node = u if u_in else v
            remote_node = v if u_in else u
            # Compute distance
            if local_node in node_pos and remote_node in node_pos:
                dx = node_pos[local_node][0] - node_pos[remote_node][0]
                dy = node_pos[local_node][1] - node_pos[remote_node][1]
                d_norm = math.sqrt(dx*dx + dy*dy)
            else:
                d_norm = 0.5  # Default high priority for unknown distance (likely long)
            cross_del_cands.append((d_norm, int(u), int(v)))
        cross_del_cands.sort(key=lambda x: -x[0])  # Delete longest cross edges first
        cross_del_out = [(u, v) for _, u, v in cross_del_cands[:self.max_cross_candidates]]
        
        return internal_add_out, cross_add_out, internal_del_out, cross_del_out
    
    def _legacy_candidates(self, obs: Observation) -> List[Tuple[int, int]]:
        """Legacy candidate generation for backward compatibility."""
        internal_add, cross_add, _, _ = self._candidates(obs)
        return (internal_add + cross_add)[:self.max_candidates]

    def compute_logits(self, obs: Observation):
        """Compute policy logits with simplified hierarchical action space.
        
        Action Space (total 16 effective combinations):
        - Operation type: 4 choices
          [0] Internal ADD, [1] Internal DEL, [2] Cross ADD, [3] Cross DEL
        - Edge index: 4 choices (from relevant candidate list)
        """
        z = self._post_embed(obs)
        y = self._lstm_forward(z, role_id=int(obs.role_id), mutate_state=True)
        
        # Simplified hierarchical logits
        logits_op_type = self.head_op_type(y)  # 4 operation types
        logits_edge_idx = self.head_edge_idx(y)  # Edge selection
        msg_out = torch.tanh(self.head_msg(y))
        value = self.head_value(y).squeeze(-1)
        
        # Get all candidate lists
        internal_add, cross_add, internal_del, cross_del = self._candidates(obs)
        
        # Store in hier_info for action selection
        hier_info = {
            'internal_add': internal_add,
            'cross_add': cross_add,
            'internal_del': internal_del,
            'cross_del': cross_del,
            'logits_op_type': logits_op_type,
            'logits_edge_idx': logits_edge_idx,
        }
        
        # Legacy compatibility
        logits_internal = self.head_internal(y)
        logits_cross = self.head_cross(y)
        logits_op = self.head_op(y)
        cands = internal_add + cross_add
        
        return logits_internal, logits_cross, logits_op, msg_out, value, cands, hier_info

    @torch.no_grad()
    def act(self, obs: Observation):
        """Sample action using simplified hierarchical action space.
        
        Action Space:
        - op_type: 4 choices [0=internal_add, 1=internal_del, 2=cross_add, 3=cross_del]
        - edge_idx: 4 choices (from relevant candidate list)
        
        === 条件熵计算 ===
        H(π) = H(π_op) + E_op[H(π_edge|op)]
        这确保每种操作类型内部都有探索，避免策略退化到"不行动"
        """
        logits_internal, logits_cross, logits_op, msg_out, value, cands, hier_info = self.compute_logits(obs)
        
        # Get the new hierarchical logits
        logits_op_type = hier_info['logits_op_type']  # [4]
        logits_edge_idx = hier_info['logits_edge_idx']  # [4]
        device = logits_op_type.device
        
        # Map op_type -> (is_add, is_internal)
        # [0] Internal ADD, [1] Internal DEL, [2] Cross ADD, [3] Cross DEL
        op_type_map = {
            0: ('add', 'internal', hier_info['internal_add']),
            1: ('del', 'internal', hier_info['internal_del']),
            2: ('add', 'cross', hier_info['cross_add']),
            3: ('del', 'cross', hier_info['cross_del']),
        }
        
        # === 动作空间对称性：确保ADD和DEL都有候选 ===
        # 统计各类型候选数量
        n_add_cands = len(hier_info['internal_add']) + len(hier_info['cross_add'])
        n_del_cands = len(hier_info['internal_del']) + len(hier_info['cross_del'])
        
        # Apply masking to op_type based on available candidates
        masked_logits_op = logits_op_type.clone()
        for op_idx, (_, _, cand_list) in op_type_map.items():
            if len(cand_list) == 0:
                masked_logits_op[op_idx] = float('-inf')
        
        # === 添加操作平衡偏置 ===
        # 当DEL候选较少时，给DEL操作添加正偏置以鼓励探索
        if n_del_cands > 0 and n_add_cands > 0:
            balance_ratio = n_add_cands / max(1, n_del_cands)
            if balance_ratio > 2.0:  # ADD候选多于DEL
                # 给DEL操作添加偏置
                del_boost = min(1.0, math.log(balance_ratio))
                masked_logits_op[1] = masked_logits_op[1] + del_boost  # internal_del
                masked_logits_op[3] = masked_logits_op[3] + del_boost  # cross_del
        
        # Sample operation type
        probs_op = torch.softmax(masked_logits_op, dim=-1)
        dist_op = torch.distributions.Categorical(probs_op)
        op_type_idx = dist_op.sample()
        logp_op = dist_op.log_prob(op_type_idx)
        entropy_op = dist_op.entropy()
        
        op_str, edge_type, selected_cands = op_type_map[op_type_idx.item()]
        is_add = (op_str == 'add')
        is_internal = (edge_type == 'internal')
        
        # Apply masking to edge_idx based on selected candidates
        max_cands = self.max_internal_candidates if is_internal else self.max_cross_candidates
        masked_logits_edge = logits_edge_idx[:max_cands].clone()
        
        if len(selected_cands) == 0:
            selected_cands = [(-1, -1)]
            mask = torch.full((max_cands,), float('-inf'), device=device)
            mask[0] = 0.0
            masked_logits_edge = masked_logits_edge + mask
        elif len(selected_cands) < max_cands:
            mask = torch.full((max_cands,), float('-inf'), device=device)
            mask[:len(selected_cands)] = 0.0
            masked_logits_edge = masked_logits_edge + mask
        
        # Sample edge index
        probs_edge = torch.softmax(masked_logits_edge, dim=-1)
        dist_edge = torch.distributions.Categorical(probs_edge)
        edge_idx = dist_edge.sample()
        logp_edge = dist_edge.log_prob(edge_idx)
        entropy_edge = dist_edge.entropy()
        
        logprob = (logp_op + logp_edge).detach()
        
        # === 条件熵计算 ===
        # H(π) = coef_op * H(π_op) + coef_edge * H(π_edge|op)
        # 使用更高的edge熵系数鼓励边选择的多样性
        entropy = (self.entropy_coef_op * entropy_op + self.entropy_coef_edge * entropy_edge).detach()
        
        # Build Action dataclass
        internal_edges: List[EdgeEdit] = []
        cross_edges: List[EdgeEdit] = []
        
        if edge_idx.item() < len(selected_cands):
            u, v = selected_cands[edge_idx.item()]
            if u >= 0 and v >= 0 and u != v:
                score = float(probs_edge[edge_idx].item())
                edge_edit = EdgeEdit(i=u, j=v, op=op_str, score=score)
                if is_internal:
                    internal_edges.append(edge_edit)
                else:
                    cross_edges.append(edge_edit)
        
        action = Action(
            internal_edges=internal_edges,
            cross_edge_scores=cross_edges,
            msg_out=msg_out.tolist(),
        )
        
        # Store for PPO update
        aux = {
            'idx': int(edge_idx.item()),
            'op': int(op_type_idx.item()),  # Now 0-3 for all 4 types
            'cands': cands,  # Legacy
            'internal_cands': selected_cands if is_internal else [],
            'cross_cands': selected_cands if not is_internal else [],
            'logprob': float(logprob.item()),
            'value': float(value.item()),
            'entropy': float(entropy.item()),
            # 额外的熵统计用于调试
            'entropy_op': float(entropy_op.item()),
            'entropy_edge': float(entropy_edge.item()),
        }
        return action, aux

    def evaluate_logprob_and_value(self, obs: Observation, idx: int, op: int, cands: List[Tuple[int, int]], 
                                    internal_cands: Optional[List[Tuple[int, int]]] = None,
                                    cross_cands: Optional[List[Tuple[int, int]]] = None):
        """Evaluate log probability and value for a given action.
        
        Args:
            obs: Observation
            idx: Selected edge index within the chosen candidate list
            op: Operation type index (0=internal_add, 1=internal_del, 2=cross_add, 3=cross_del)
            cands: Combined candidate list (for legacy compatibility)
            internal_cands: Internal candidates used during action
            cross_cands: Cross candidates used during action
            
        === 条件熵计算 ===
        entropy = coef_op * H(π_op) + coef_edge * H(π_edge|op)
        """
        # Use non-mutating LSTM evaluation
        z = self._post_embed(obs)
        y = self._lstm_forward(z, role_id=int(obs.role_id), mutate_state=False)
        
        # Get logits for both heads
        logits_op_type = self.head_op_type(y)  # [4]
        logits_edge_idx = self.head_edge_idx(y)  # [4]
        value = self.head_value(y).squeeze(-1)
        device = logits_op_type.device
        
        # Determine which candidate list was used based on op
        # op: 0=internal_add, 1=internal_del, 2=cross_add, 3=cross_del
        is_internal = (op in [0, 1])
        
        # Get the relevant candidate list
        if is_internal:
            eval_cands = internal_cands if internal_cands is not None else cands[:self.max_internal_candidates]
            max_cands = self.max_internal_candidates
        else:
            eval_cands = cross_cands if cross_cands is not None else []
            max_cands = self.max_cross_candidates
        
        # Regenerate candidates for op masking
        internal_add, cross_add, internal_del, cross_del = self._candidates(obs)
        cand_lists = {
            0: internal_add,
            1: internal_del,
            2: cross_add,
            3: cross_del,
        }
        
        # === 动作空间对称性：添加操作平衡偏置 ===
        n_add_cands = len(internal_add) + len(cross_add)
        n_del_cands = len(internal_del) + len(cross_del)
        
        # Mask op_type logits
        masked_logits_op = logits_op_type.clone()
        for op_idx, c_list in cand_lists.items():
            if len(c_list) == 0:
                masked_logits_op[op_idx] = float('-inf')
        
        # 添加平衡偏置 (与act方法保持一致)
        if n_del_cands > 0 and n_add_cands > 0:
            balance_ratio = n_add_cands / max(1, n_del_cands)
            if balance_ratio > 2.0:
                del_boost = min(1.0, math.log(balance_ratio))
                masked_logits_op[1] = masked_logits_op[1] + del_boost
                masked_logits_op[3] = masked_logits_op[3] + del_boost
        
        probs_op = torch.softmax(masked_logits_op, dim=-1)
        dist_op = torch.distributions.Categorical(probs_op)
        logp_op = dist_op.log_prob(torch.tensor(op, device=device))
        entropy_op = dist_op.entropy()
        
        # Mask edge logits based on candidate availability
        masked_logits_edge = logits_edge_idx[:max_cands].clone()
        if len(eval_cands) == 0:
            eval_cands = [(-1, -1)]
            mask = torch.full((max_cands,), float('-inf'), device=device)
            mask[0] = 0.0
            masked_logits_edge = masked_logits_edge + mask
        elif len(eval_cands) < max_cands:
            mask = torch.full((max_cands,), float('-inf'), device=device)
            mask[:len(eval_cands)] = 0.0
            masked_logits_edge = masked_logits_edge + mask
        
        probs_edge = torch.softmax(masked_logits_edge, dim=-1)
        dist_edge = torch.distributions.Categorical(probs_edge)
        
        # Clamp idx to valid range
        idx_clamped = min(idx, len(eval_cands) - 1) if eval_cands else 0
        logp_edge = dist_edge.log_prob(torch.tensor(idx_clamped, device=device))
        entropy_edge = dist_edge.entropy()
        
        logprob = logp_op + logp_edge
        
        # === 条件熵计算 ===
        # H(π) = coef_op * H(π_op) + coef_edge * H(π_edge|op)
        entropy = self.entropy_coef_op * entropy_op + self.entropy_coef_edge * entropy_edge
        
        return logprob, value, entropy
