from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict


# ------------------------------
# Observation (fixed N slots)
# ------------------------------

@dataclass
class Slot:
    node_id: int
    mask: int  # 0 = dummy
    delta_pos: Tuple[float, float]
    delta_vel: Tuple[float, float]
    battery: float
    link_snr: float
    link_loss: float
    link_rtt: float
    role_bit: int


@dataclass
class HaloSummary:
    deg_hist: List[float]  # B buckets
    snr_stats: Tuple[float, float, float]  # mean, p10, p90
    min_degree: int
    cross_candidates: int
    topk_anchor: List[Tuple[float, float, float, float]]  # K x (dx,dy,deg,snr)


@dataclass
class Observation:
    slots: List[Slot]  # length fixed N
    adj_local: List[Tuple[int, int, float]]  # sparse edges (i,j,w)
    halo_summary: HaloSummary
    time_feat: Tuple[float, float]  # t_mod, rho_local
    msg_in: List[float]  # fixed D dims
    role_id: int
    # NEW: For hierarchical action space
    internal_nodes: List[int]  # Nodes within this agent's partition
    boundary_nodes: List[int]  # Internal nodes with cross-partition neighbors
    halo_nodes: List[int]  # External nodes within 1-2 hops (for cross-edge candidates)
    existing_internal_edges: List[Tuple[int, int]]  # Edges within partition (for del candidates)
    existing_cross_edges: List[Tuple[int, int]]  # Edges crossing partition (for del candidates)
    # Normalized positions for all relevant nodes (internal + halo)
    node_positions: Dict[int, Tuple[float, float]]  # node_id -> (x_norm, y_norm)


# ------------------------------
# Action
# ------------------------------

@dataclass
class EdgeEdit:
    i: int
    j: int
    op: str  # 'add' or 'del'
    score: float


@dataclass
class Action:
    internal_edges: List[EdgeEdit]
    cross_edge_scores: List[EdgeEdit]
    msg_out: List[float]


# ------------------------------
# DeltaE (final applied)
# ------------------------------

@dataclass
class DeltaE:
    add: List[Tuple[int, int]]
    delete: List[Tuple[int, int]]
    rejected: List[Dict[str, object]]  # {'edge': (u,v), 'reason': str}


def pad_or_trim_list(x: List, target_len: int, pad_val):
    if len(x) >= target_len:
        return x[:target_len]
    return x + [pad_val for _ in range(target_len - len(x))]

