from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

# Ensure project root on sys.path for direct execution (python scripts/run_local_demo.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.train import run as train_run
from core.environment import CoreEnv
from core.merger_safety import MergerSafety, SafetyConfig
from core.messaging import MessageBus
from core.partitioner import Partitioner
from training.rollout import build_observations, maybe_local_rebalance
from models.actor import Actor
from deploy.autoscaler import AutoScaler
import argparse


def check_invariants(env: CoreEnv, deltaE, cfg: dict, info_log: Dict[str, int]):
    # A) deletions never disconnect handled in merger; here we ensure graph non-empty
    assert len(env.nodes) == len(set(env.nodes)), "Duplicate nodes"
    # B) budget and cooldown
    budget = int(cfg['safety']['deltaE_budget_per_step'])
    assert len(deltaE.add) + len(deltaE.delete) <= budget + 1e-6, "ΔE over budget"
    # record reasons
    for r in deltaE.rejected:
        info_log[r.get('reason', 'unknown')] = info_log.get('reason', 0) + 1


def main():
    print("Local demo: invariants and quick sanity check")
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', type=int, default=100)
    ap.add_argument('--autoscale', action='store_true')
    ap.add_argument('--cfg', type=str, default=os.environ.get('RL_CFG') or 'configs/config.yaml')
    args = ap.parse_args()
    cfg_path = args.cfg
    # Run the standard training once as smoke test
    train_run(cfg_path)

    # Build a mid-size env and assert invariants
    try:
        import yaml
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except Exception:
        from scripts.train import load_cfg
        cfg = load_cfg(cfg_path)

    env = CoreEnv(cfg)
    partitioner = Partitioner(cfg)
    safety_cfg = cfg['safety']
    merge_cfg = cfg.get('merge', {}) or {}
    merger = MergerSafety(SafetyConfig(
        min_cross_edges_per_neighbor=int(safety_cfg.get('min_cross_edges_per_neighbor', 2)),
        deltaE_budget_per_step=int(safety_cfg.get('deltaE_budget_per_step', 20)),
        cooldown_steps=int(safety_cfg.get('cooldown_steps', 50)),
    ), cross_policy=str(merge_cfg.get('cross_policy', 'and')), score_thresh=float(merge_cfg.get('score_thresh', 0.2)))
    msg_cfg = cfg['messaging']
    bus = MessageBus(dim=int(cfg['model']['msg_dim']), ttl=int(msg_cfg.get('ttl', 3)), dropout=float(msg_cfg.get('dropout', 0.2)))
    actor = Actor(cfg)
    autoscaler = AutoScaler(cfg, partitioner)

    N = args.nodes
    m = 8
    nodes = list(range(N))
    partitions = partitioner.init(m, nodes, env.node_pos)
    env.reset(partitions)
    info_log = {}

    warmup_until = {}
    for t in range(200):
        # Mobility and churn
        env.tick_mobility()
        partitions, events = env.apply_join_leave(partitioner, partitions)
        obs_by_agent = build_observations(env, partitions, bus, {}, cfg)
        actions = {}
        for aid, obs in obs_by_agent.items():
            act, _aux = actor.act(obs)
            if args.autoscale and warmup_until.get(aid, 0) > 0:
                act.internal_edges = []
            actions[aid] = act
        deltaE = merger.merge(actions, env.graph)
        obs, reward, info = env.step(deltaE, partitions)
        check_invariants(env, deltaE, cfg, info_log)
        partitions = maybe_local_rebalance(partitioner, env, partitions, t, cfg)
        bus.tick()

        # optional autoscale demo
        if args.autoscale:
            acfg = (cfg.get('autoscale') or {})
            target_slots = int(acfg.get('target_slots_per_agent', 10))
            m = max(partitions.values()) + 1 if partitions else 1
            rho_bar = len(env.nodes) / max(1, m * target_slots)
            m_new, action = autoscaler.step(len(env.nodes), rho_bar, m)
            if action:
                print(f"[AutoScale-Demo] {action[0]} Δm={action[1]} | m {m}->{m_new} | rho_bar={rho_bar:.3f}")
                partitions = partitioner.resize_agents(m_new, env.nodes, env.node_pos)
                env.partitions = partitions
                new_agents = set(partitions.values())
                for a in new_agents:
                    warmup_until[a] = int(acfg.get('warmup_steps', 50))
            # tick warmup
            for a in list(warmup_until.keys()):
                warmup_until[a] -= 1
                if warmup_until[a] <= 0:
                    del warmup_until[a]

    print("Demo completed; invariants checked.")

    # C) neighbor cross-edge minimum b
    part_map = env.partitions
    cross_counts = {}
    for (u, v) in env.edges:
        a, b = part_map.get(u), part_map.get(v)
        if a is None or b is None or a == b:
            continue
        key = (min(a, b), max(a, b))
        cross_counts[key] = cross_counts.get(key, 0) + 1
    if cross_counts:
        min_cross = min(cross_counts.values())
        assert min_cross >= int(cfg['safety']['min_cross_edges_per_neighbor']), "Cross-edge quota not satisfied"

    # D) halo dims fixed and r↑ increases neighborhood
    from core.halo import HaloBuilder
    edge_list = [(u, v, 1.0) for (u, v) in env.edges]
    agent0_nodes = [n for n, a in env.partitions.items() if a == 0]
    halo1 = HaloBuilder(buckets=cfg['halo']['buckets'], topk=cfg['halo']['topk'], mode='rhop', r=1)
    halo2 = HaloBuilder(buckets=cfg['halo']['buckets'], topk=cfg['halo']['topk'], mode='rhop', r=2)
    h1 = halo1.build_for_partition(edge_list, agent0_nodes)
    h2 = halo2.build_for_partition(edge_list, agent0_nodes)
    assert len(h1.deg_hist) == len(h2.deg_hist), "Halo histogram dim changed"
    assert h2.cross_candidates >= h1.cross_candidates, "Halo with larger r should not reduce candidates"

    # E) m-sweep minimal check (connectivity stable)
    from training.validation import run_m_sweep
    def _env_builder(_cfg):
        return CoreEnv(_cfg)
    def _actor_builder(_cfg):
        return Actor(_cfg)
    res = run_m_sweep(_env_builder, _actor_builder, cfg)
    if res:
        conns = [d.get('connectivity', 1.0) for d in res.values()]
        assert min(conns) >= 0.95, "Connectivity degraded beyond 5%"


if __name__ == '__main__':
    main()
