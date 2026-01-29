from __future__ import annotations

import argparse
import os
import sys
import torch
import json
import random
from typing import Dict, List

# Ensure root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import yaml
except ImportError:
    yaml = None

from core.environment import CoreEnv
from core.merger_safety import MergerSafety, SafetyConfig
from core.messaging import MessageBus
from core.partitioner import Partitioner
from deploy.autoscaler import AutoScaler
from training.rollout import build_observations, maybe_local_rebalance
from models.actor import Actor
from core.visualize import plot_topology

from scripts.train import DEFAULT_CFG

def load_cfg(path: str | None) -> Dict:
    if path and os.path.exists(path) and yaml is not None:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    print("Warning: Config file not found, using DEFAULT_CFG")
    return DEFAULT_CFG

def evaluate(args):
    # Load config from run directory if possible
    cfg_path = args.cfg
    if not cfg_path and args.ckpt:
        # Try to infer config from ckpt dir
        run_dir = os.path.dirname(args.ckpt)
        cand = os.path.join(run_dir, 'config_effective.yaml')
        if os.path.exists(cand):
            cfg_path = cand
            print(f"Inferred config from checkpoint: {cfg_path}")
    
    cfg = load_cfg(cfg_path)
    
    # Overrides
    if args.episodes:
        # We handle loop manually
        pass
    
    # Setup Env
    env = CoreEnv(cfg)
    partitioner = Partitioner(cfg)
    
    # Safety
    safety_cfg = cfg.get('safety', {}) or {}
    merge_cfg = cfg.get('merge', {}) or {}
    merger = MergerSafety(SafetyConfig(
        min_cross_edges_per_neighbor=int(safety_cfg.get('min_cross_edges_per_neighbor', 2)),
        deltaE_budget_per_step=int(safety_cfg.get('deltaE_budget_per_step', 20)),
        cooldown_steps=int(safety_cfg.get('cooldown_steps', 50)),
    ), cross_policy=str(merge_cfg.get('cross_policy', 'and')), score_thresh=float(merge_cfg.get('score_thresh', 0.2)))
    
    # Bus
    msg_cfg = cfg.get('messaging', {}) or {}
    bus = MessageBus(dim=int(cfg['model']['msg_dim']), ttl=int(msg_cfg.get('ttl', 3)), dropout=0.0) # No dropout in test
    
    # Model
    actor = Actor(cfg)
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Loading checkpoint: {args.ckpt}")
        try:
            state = torch.load(args.ckpt, map_location='cpu')
            actor.load_state_dict(state)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    else:
        print("Warning: No checkpoint provided or found, using random weights.")

    actor.eval()
    
    # Autoscaler
    autoscaler = AutoScaler(cfg, partitioner)
    
    # Output
    out_dir = args.out_dir or 'result_save/eval_run'
    os.makedirs(out_dir, exist_ok=True)
    
    episodes = args.episodes
    T = args.steps
    
    all_metrics = []
    
    for ep in range(episodes):
        print(f"Running Episode {ep+1}/{episodes}...")
        
        # Init logic similar to train
        # Assume somewhat larger m for test sometimes, or from config
        m_min = int(cfg.get('training', {}).get('m_min', 4))
        m_max = int(cfg.get('training', {}).get('m_max', 8))
        m = (m_min + m_max) // 2
        
        # N nodes
        N = int(cfg['model']['slots']) * 10
        nodes = list(range(N))
        
        partitions = partitioner.init(m, nodes)
        role_ids = partitioner.assign_roles(m)
        env.reset(partitions)
        
        # Autoscaler reset
        autoscaler.reset_state()
        bus.reset(); merger.reset(); actor.reset_rnn(batch_size=1)
        
        ep_ret = 0.0
        success_list = []
        delay_list = []
        energy_list = []
        
        warmup_until = {}
        agents_set = set(partitions.values())

        for t in range(T):
            env.tick_mobility()
            partitions, _ = env.apply_join_leave(partitioner, partitions)
            obs = build_observations(env, partitions, bus, role_ids, cfg)
            
            actions = {}
            for aid, o in obs.items():
                with torch.no_grad():
                    act, _ = actor.act(o)
                if aid in warmup_until and warmup_until[aid] > 0:
                    act.internal_edges = []
                actions[aid] = act
            
            # Msg publish
            actions_as_dict = {aid: {'msg_out': a.msg_out, 'internal_edges': [{'i': e.i, 'j': e.j, 'op': e.op, 'score': e.score} for e in a.internal_edges]} for aid, a in actions.items()}
            neighbors = {} # simple stub for neighbors if needed, bus needs it?
            # actually bus needs neighbors map
            part_map = env.partitions
            edge_pairs = list(env.edges)
            for (u, v) in edge_pairs:
                a, b = part_map.get(u), part_map.get(v)
                if a is None or b is None or a == b: continue
                neighbors.setdefault(a, set()).add(b); neighbors.setdefault(b, set()).add(a)
            neighbors = {k: sorted(list(v)) for k, v in neighbors.items()}
            bus.publish(actions_as_dict, neighbors)
            
            deltaE = merger.merge(actions, env.graph)
            _, rew, info = env.step(deltaE, partitions)
            bus.tick()
            
            ep_ret += rew
            success_list.append(info.get('success_proxy', 0.0))
            delay_list.append(info.get('delay_proxy', 0.0))
            energy_list.append(info.get('energy_proxy', 0.0))
            
            # Autoscale
            acfg = (cfg.get('autoscale') or {})
            target_slots = int(acfg.get('target_slots_per_agent', 10))
            rho_bar = len(env.nodes) / max(1, m * target_slots)
            m_new, action = autoscaler.step(len(env.nodes), rho_bar, m)
            if action:
                m = m_new
                partitions = partitioner.resize_agents(m, env.nodes, env.node_pos)
                env.partitions = partitions
                new_agents = set(partitions.values()) - agents_set
                agents_set = set(partitions.values())
                for na in new_agents:
                    warmup_until[na] = 50
            
            # Warmup decay
            for a in list(warmup_until.keys()):
                warmup_until[a] -= 1
                if warmup_until[a] <= 0: del warmup_until[a]

            partitions = maybe_local_rebalance(partitioner, env, partitions, t, cfg)
            
            # Visualize last step of episode or regular intervals?
            # User wants visual proof. Draw last step.
            if t == T - 1:
                metrics = {
                    'episode': ep+1,
                    'avg_return': ep_ret / (t+1),
                    'success_proxy': sum(success_list)/len(success_list),
                    'delay_proxy': sum(delay_list)/len(delay_list),
                    'energy_proxy': sum(energy_list)/len(energy_list),
                    'm': m,
                    'rho_bar': rho_bar
                }
                fname = os.path.join(out_dir, f"eval_ep{ep+1}_final.png")
                plot_topology(env.graph, env.partitions, fname, metrics=metrics)

        all_metrics.append({
            'episode': ep+1,
            'return': ep_ret / T,
            'success': sum(success_list)/len(success_list)
        })

    # Summary
    print("\nEvaluation Complete.")
    print(json.dumps(all_metrics, indent=2))
    with open(os.path.join(out_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Config path')
    parser.add_argument('--ckpt', type=str, help='Checkpoint path (best.pt)')
    parser.add_argument('--out-dir', type=str, help='Output directory')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--steps', type=int, default=64)
    args = parser.parse_args()
    evaluate(args)
