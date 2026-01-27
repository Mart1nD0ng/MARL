from __future__ import annotations

import random
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

from core.environment import CoreEnv
from core.messaging import MessageBus


def _compute_partition_neighbors(partitions: Dict[int, int], edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    # Build neighbors based on cross-edges
    nbrs: DefaultDict[int, set] = defaultdict(set)
    for (u, v) in edges:
        a, b = partitions.get(u), partitions.get(v)
        if a is None or b is None or a == b:
            continue
        nbrs[a].add(b)
        nbrs[b].add(a)
    return {k: sorted(list(v)) for k, v in nbrs.items()}


def build_observations(env: CoreEnv, partitions: Dict[int, int], bus: MessageBus, role_ids: Dict[int, int], cfg: dict):
    obs = env._build_observations(partitions)  # internal helper in this skeleton
    # Attach inbound messages
    for aid, ob in obs.items():
        ob.msg_in = bus.get_messages(aid)
    return obs


def maybe_inject_birth_death(partitioner, env: CoreEnv, partitions: Dict[int, int], t: int, cfg: dict):
    # Simple stochastic birth/death to exercise robustness
    if t <= 0:
        return partitions

    birth_p = 0.02
    death_p = 0.01
    nodes = env.nodes

    # Death: remove last node if more than 8 nodes
    if len(nodes) > 8 and random.random() < death_p:
        victim = nodes[-1]
        env.nodes = [n for n in nodes if n != victim]
        env.edges = {e for e in env.edges if victim not in e}
        partitions.pop(victim, None)

    # Birth: add one node
    elif random.random() < birth_p:
        new_id = (max(nodes) + 1) if nodes else 0
        env.nodes.append(new_id)
        # connect to previous to keep connectivity
        if len(env.nodes) > 1:
            u = env.nodes[-2]
            v = env.nodes[-1]
            env.edges.add(tuple(sorted((u, v))))
        # assign to agent via current scheme
        agent_ids = list(set(partitions.values())) or [0]
        chosen_agent = random.choice(agent_ids)
        partitions[new_id] = chosen_agent

    return partitions


def maybe_local_rebalance(partitioner, env: CoreEnv, partitions: Dict[int, int], t: int, cfg: dict) -> Dict[int, int]:
    pcfg = cfg.get('partition', {}) or {}
    if t % max(1, int(pcfg.get('T_high', 120))) != 0:
        return partitions
    rho_high = float(pcfg.get('rho_high', 0.90))
    rho_low = float(pcfg.get('rho_low', 0.58))
    plan = partitioner.local_rebalance(partitions.copy(), env.node_pos, rho_high, rho_low)
    for n, _a, b in plan.get('moves', []):
        partitions[n] = b
    return partitions


def maybe_quantize_noise(vec: List[float], cfg: dict, training: bool = True) -> List[float]:
    mcfg = cfg.get('messaging', {}) or {}
    if not training or not bool(mcfg.get('enable_quant_noise', False)):
        return vec
    Q = int(mcfg.get('quant_levels', 32))
    eps = float(mcfg.get('quant_eps', 0.01))
    out = []
    for x in vec:
        v = round(float(x) * Q) / Q
        v += random.uniform(-eps, eps)
        out.append(max(-1.0, min(1.0, v)))
    return out
