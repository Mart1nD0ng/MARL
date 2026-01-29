from __future__ import annotations

import os
import statistics
from typing import Dict, List
import random


def run_m_sweep(env_builder, actor_builder, cfg: dict):
    """Run a light m-sweep with ablation toggles, write summaries.

    This function is intentionally lightweight and uses provided builders when available.
    Returns summary metrics per m.
    """
    ms = cfg.get('training', {}).get('m_grid', [6, 8, 12, 16, 24, 32])
    K = int(cfg.get('training', {}).get('m_repeats', 2))
    out: Dict[int, Dict[str, float]] = {}
    os.makedirs('result_save', exist_ok=True)

    for m in ms:
        success = []
        conn = []
        for _ in range(K):
            # Builders are expected to return minimal env/actor pair
            env = env_builder(cfg)
            actor = actor_builder(cfg)
            # single-step probe
            partitions = {n: n % max(1, m) for n in range(min(100, max(10, m * 3)))}
            env.reset(partitions)
            obs = env._build_observations(partitions)
            # success proxy and connectivity
            conn.append(1.0 if len(env.edges) > 0 else 0.0)
            success.append(1.0)  # placeholder; extend with true metrics
        out[m] = {
            'success_rate': float(statistics.mean(success) if success else 0.0),
            'connectivity': float(statistics.mean(conn) if conn else 0.0),
        }

    with open(os.path.join('result_save', 'm_sweep.txt'), 'w', encoding='utf-8') as f:
        for m, d in out.items():
            f.write(f"m={m}: {d}\n")
    return out


def test_once(env, actor, partitions: Dict[int, int], steps: int, out_dir: str, greedy: bool = True, seed: int = 1337, max_plot_nodes: int = 400):
    """Run a short deterministic rollout and save simple artifacts.

    - Loads best weights if available in out_dir/best.pt
    - Greedy action selection by argmax over candidate logits if supported; otherwise uses actor.act
    - Saves topology_final.png and agent_coverage.png if node count small
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        import torch
        best_pt = os.path.join(out_dir, 'best.pt')
        if os.path.exists(best_pt):
            sd = torch.load(best_pt, map_location='cpu')
            # tolerate actor wrapping
            actor.load_state_dict(sd, strict=False)
    except Exception:
        pass
    # seed
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
    except Exception:
        pass

    env.reset(partitions)
    total_r = 0.0
    for t in range(steps):
        # no mobility here; deterministic
        obs = env._build_observations(partitions)
        actions = {}
        for aid, ob in obs.items():
            if greedy and hasattr(actor, 'compute_logits'):
                le, lo, msg, v, cands = actor.compute_logits(ob)
                # greedy: pick max logits
                idx = int(le.argmax().item())
                op = int(lo.argmax().item())
                # build minimal Action
                act, _ = actor.act(ob)
                if len(cands) > 0 and idx < len(cands):
                    u, vv = cands[idx]
                    if u >= 0 and vv >= 0 and u != vv:
                        act.internal_edges = [type('E', (), {'i': u, 'j': vv, 'op': 'add' if op == 1 else 'del', 'score': 1.0})()]
                actions[aid] = act
            else:
                act, _ = actor.act(ob)
                actions[aid] = act
        deltaE = type('X', (), {'add': [], 'delete': [], 'rejected': []})()
        _next_obs, reward, _info = env.step(deltaE, partitions)
        total_r += reward
    # Save simple metrics json
    with open(os.path.join(out_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        f.write('{"avg_return": %.6f}' % (total_r / max(1, steps)))

    # Save visualizations if not too large
    try:
        from core.visualize import plot_topology, plot_agent_coverage
        if len(env.nodes) <= max_plot_nodes:
            plot_topology(env.graph, env.partitions, os.path.join(out_dir, 'topology_final.png'))
        plot_agent_coverage(env.partitions, env.node_pos, os.path.join(out_dir, 'agent_coverage.png'))
    except Exception:
        pass
