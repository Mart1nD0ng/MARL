from __future__ import annotations

import os
import sys
import json
import torch
from typing import Dict

# Ensure project root on sys.path when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # fallback

from core.environment import CoreEnv
from core.merger_safety import MergerSafety, SafetyConfig
from core.messaging import MessageBus
from core.partitioner import Partitioner
from deploy.autoscaler import AutoScaler
from training.mappo_trainer import MAPPOTrainer
from training.rollout import build_observations, maybe_local_rebalance
from models.actor import Actor
from models.critic import Critic
from core.visualize import plot_topology, plot_agent_coverage, plot_logical_topology
from training.mappo_trainer import EarlyStopper


DEFAULT_CFG = {
    'model': {
        'slots': 10,
        'gnn_layers': 3,
        'lstm_hidden': 128,
        'msg_dim': 32,
        'role_dim': 16,
    },
    'safety': {
        'min_cross_edges_per_neighbor': 2,
        'deltaE_budget_per_step': 20,
        'cooldown_steps': 50,
    },
    'partition': {
        'scheme': 'consistent_hash',
        'rebalance': 'local_min_cost_flow',
        'rho_high': 0.90,
        'rho_low': 0.58,
        'T_high': 120,
        'T_low': 150,
    },
    'messaging': {
        'period': 1,
        'ttl': 3,
        'dropout': 0.2,
        'max_msgs_per_step': 1,
    },
    'training': {
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'lr': 0.0003,
        'clip': 0.15,
        'target_kl': 0.01,
        'rollout_len': 128,
        'epochs': 1,
        'm_min': 4,
        'm_max': 8,
        'episodes': 2,
        'T': 16,
    }
}



def _prepare_course_schedule(schedule_raw):
    schedule = []
    if not schedule_raw:
        return schedule
    for entry in schedule_raw:
        if not isinstance(entry, dict):
            continue
        start = int(entry.get('start_episode', entry.get('start', entry.get('episode', len(schedule) + 1))))
        if start < 1:
            start = 1
        label = entry.get('name') or entry.get('label')
        updates = {k: v for k, v in entry.items() if k not in {'start_episode', 'start', 'episode', 'name', 'label'}}
        schedule.append({'start': start, 'label': label, 'updates': updates})
    schedule.sort(key=lambda e: e['start'])
    return schedule


def _apply_course_stage(episode_idx: int, phase: Dict[str, object], cfg: Dict, env: CoreEnv, autoscaler: AutoScaler, verbose: bool):
    updates = phase.get('updates', {}) or {}
    for section, payload in updates.items():
        if not isinstance(payload, dict):
            continue
        target = cfg.setdefault(section, {}) if isinstance(cfg.get(section), dict) else {}
        target.update(payload)
        cfg[section] = target
    if 'city' in updates and hasattr(env, 'mobility') and hasattr(env.mobility, 'update_config'):
        env.mobility.update_config(cfg)
    if 'autoscale' in updates and hasattr(autoscaler, 'reset_state'):
        autoscaler.reset_state()
    if verbose:
        label = phase.get('label')
        name_part = f" ({label})" if label else ''
        print(f"[Curriculum] episode {episode_idx}: applying stage starting @ {phase['start']}{name_part}")


def load_cfg(path: str | None) -> Dict:
    # Resolve relative path against project root to avoid CWD-dependent failures.
    if not path:
        return DEFAULT_CFG
    cfg_path = path
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(ROOT, cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('Config file not found: ' + cfg_path)
    if yaml is None:
        raise ImportError('PyYAML is required to load config.yaml. Please install pyyaml.')
    # Use utf-8-sig to tolerate BOM from Windows tools
    with open(cfg_path, 'r', encoding='utf-8-sig') as f:
        loaded = yaml.safe_load(f)  # type: ignore
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError('Invalid config format (expected mapping/dict): ' + cfg_path)
    return loaded


def run(cfg_path: str | None = None, overrides: dict | None = None):
    cfg = load_cfg(cfg_path)
    # Apply overrides if provided (CLI support)
    if overrides:
        # shallow merge for training/autoscale/viz
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    env = CoreEnv(cfg)
    partitioner = Partitioner(cfg)
    safety_cfg = cfg.get('safety', {}) or {}
    merge_cfg = cfg.get('merge', {}) or {}
    merger = MergerSafety(SafetyConfig(
        min_cross_edges_per_neighbor=int(safety_cfg.get('min_cross_edges_per_neighbor', 2)),
        deltaE_budget_per_step=int(safety_cfg.get('deltaE_budget_per_step', 20)),
        cooldown_steps=int(safety_cfg.get('cooldown_steps', 50)),
        enforce_cross_edge_quota=bool(safety_cfg.get('enforce_cross_edge_quota', False)),
    ), cross_policy=str(merge_cfg.get('cross_policy', 'and')), score_thresh=float(merge_cfg.get('score_thresh', 0.2)))
    msg_cfg = cfg.get('messaging', {}) or {}
    bus = MessageBus(dim=int(cfg['model']['msg_dim']), ttl=int(msg_cfg.get('ttl', 3)), dropout=float(msg_cfg.get('dropout', 0.2)))

    actor = Actor(cfg)
    critic = Critic()
    trainer = MAPPOTrainer(actor, critic, cfg)

    tcfg = (cfg.get('training') or {})
    episodes = int(tcfg.get('episodes', 2))
    last_steps_T = int(tcfg.get('max_steps_per_episode', tcfg.get('T', 16)))
    verbose_default = bool(tcfg.get('verbose', True))
    course_schedule = _prepare_course_schedule(tcfg.get('course_schedule'))
    course_stage_idx = -1

    # Simulate nodes labeled 0..N-1 (configurable via cfg['sim']['num_nodes'])
    sim_cfg = (cfg.get('sim') or {})
    N = int(sim_cfg.get('num_nodes', int(cfg['model']['slots']) * 10))
    nodes = list(range(N))

    if verbose_default:
        print(f"[CTDE] episodes={episodes}, steps_T={last_steps_T}, m_range=({int(cfg['training']['m_min'])},{int(cfg['training']['m_max'])})")

    # Autoscaler instance
    autoscaler = AutoScaler(cfg, partitioner)

    # Result directory
    import datetime, csv
    run_name = (tcfg.get('run_name') or datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    run_dir = os.path.join('result_save', run_name)
    os.makedirs(run_dir, exist_ok=True)
    # Save effective config
    try:
        import yaml as _yaml
        with open(os.path.join(run_dir, 'config_effective.yaml'), 'w', encoding='utf-8') as f:
            _yaml.safe_dump(cfg, f)  # type: ignore
    except Exception:
        pass
    csv_path = os.path.join(run_dir, 'metrics.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8') if bool((cfg.get('logging') or {}).get('csv_metrics', True)) else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(['episode','avg_return','success_proxy','delay_proxy','energy_proxy','edges','KL','m','rho_bar','deltaE_size','rejects','loss_pi','loss_v_mse','loss_v_rmse','value_clip_frac','ret_mean','ret_std','lr'])

    # Early stopper
    es_cfg = (tcfg.get('early_stop') or {})
    early_enable = bool(es_cfg.get('enable', False))
    stopper = EarlyStopper(es_cfg.get('metric', 'success_rate'), int(es_cfg.get('patience_episodes', 500)), float(es_cfg.get('min_delta', 0.002))) if early_enable else None
    best_metric = None
    checkpoint_every = int(tcfg.get('checkpoint_interval_episodes', 100))

    # History for quick curves
    hist_actor, hist_critic, hist_ret, hist_succ, hist_delay, hist_energy, hist_kl, hist_m, hist_rho, hist_v_rmse, hist_clip = [], [], [], [], [], [], [], [], [], [], []

    for ep in range(episodes):

        episode_idx = ep + 1
        if course_schedule:
            while (course_stage_idx + 1) < len(course_schedule) and episode_idx >= course_schedule[course_stage_idx + 1]['start']:
                course_stage_idx += 1
                _apply_course_stage(episode_idx, course_schedule[course_stage_idx], cfg, env, autoscaler, bool((cfg.get('training') or {}).get('verbose', verbose_default)))
        tcfg = (cfg.get('training') or {})
        steps_T = int(tcfg.get('max_steps_per_episode', tcfg.get('T', 16)))
        last_steps_T = steps_T
        verbose = bool(tcfg.get('verbose', verbose_default))
        
        # Calculate m: number of agents based on node count and slot capacity
        model_cfg = cfg.get('model', {}) or {}
        slots_per_agent = int(model_cfg.get('slots', 10))
        
        # Get autoscale config
        autoscale_cfg = cfg.get('autoscale', {}) or {}
        target_slots = int(autoscale_cfg.get('target_slots_per_agent', slots_per_agent))
        min_agents = int(autoscale_cfg.get('min_agents', 2))
        max_agents = int(autoscale_cfg.get('max_agents', 128))
        
        # Choose m (number of agents).
        # If autoscale.enable is false, use training.m_fixed.
        import math
        # IMPORTANT: default should be disabled unless explicitly enabled in config.
        if bool((cfg.get('autoscale') or {}).get('enable', False)):
            m = math.ceil(len(nodes) / max(1, target_slots))
            m = max(min_agents, min(max_agents, m))
        else:
            m_fixed = int(tcfg.get('m_fixed', tcfg.get('m_min', 6)))
            m = max(1, m_fixed)
        
        if verbose:
            print(f"[Autoscale] {len(nodes)} nodes / {target_slots} target_slots = {m} agents")

        # 1. Initial dummy partition to bootstrap Env (generates node_pos)
        partitions = partitioner.init(m, nodes, None)
        role_ids = partitioner.assign_roles(m)
        env.reset(partitions)
        
        # 2. Now that we have positions, re-partition using grid-based spatial partitioning
        if env.node_pos:
             partitions = partitioner.init(m, nodes, env.node_pos)
             env.partitions = partitions

        # Generate initial visualization only at ep=0, after environment is initialized
        if ep == 0:
            viz = (cfg.get('viz') or {})
            if bool(viz.get('enable', True)):
                max_nodes_for_plot = int(viz.get('max_graph_nodes_for_plot', 400))
                region_bbox = tuple((cfg.get('region') or {}).get('bbox', [-300, -300, 300, 300]))
                city_stride = float((cfg.get('city') or {}).get('grid_stride', 100.0))
                city_road_w = float((cfg.get('city') or {}).get('road_width', 25.0))
                if bool(viz.get('draw_topology', True)):
                    plot_topology(env.graph, env.partitions, os.path.join(run_dir, "topology_ep0.png"), 
                                 dpi=int(viz.get('dpi', 160)), max_nodes=max_nodes_for_plot)
                    plot_logical_topology(env.nodes, env.edges, env.partitions, 
                                          os.path.join(run_dir, "logical_ep0.png"),
                                          dpi=int(viz.get('dpi', 160)), layout='spring')
                if bool(viz.get('draw_agent_coverage', True)):
                    plot_agent_coverage(env.partitions, env.node_pos, os.path.join(run_dir, "coverage_ep0.png"), 
                                        dpi=int(viz.get('dpi', 160)), max_nodes=max_nodes_for_plot,
                                        bbox=region_bbox, grid_stride=city_stride, road_width=city_road_w)
                if verbose:
                    int_edges_init = sum(1 for u,v in env.edges if env.partitions.get(u)==env.partitions.get(v))
                    print(f"[Init Viz] Initial topology: {len(env.edges)} edges ({int_edges_init} internal, {100*int_edges_init/max(1,len(env.edges)):.1f}%)")

        bus.reset(); merger.reset(); trainer.reset_rnn()
        if verbose:
            print(f"[Episode {ep+1}/{episodes}] m={m} | nodes={len(nodes)} | agents={len(set(partitions.values()))}")
        # Warmup map for new agents this episode
        warmup_until: Dict[int, int] = {}
        agents_set = set(partitions.values())

        trunc_steps = int((cfg.get('training') or {}).get('truncation_steps', 128))
        total_delta_sz = 0
        total_rejects = 0
        avg_return = 0.0
        success_p = 0.0
        delay_p = 0.0
        energy_p = 0.0
        for t in range(steps_T):
            if t > 0 and (t % max(1, trunc_steps) == 0):
                if hasattr(actor, 'reset_rnn'):
                    actor.reset_rnn()
            # Mobility + churn
            env.tick_mobility()
            partitions, mob_events = env.apply_join_leave(partitioner, partitions)
            obs_by_agent = build_observations(env, partitions, bus, role_ids, cfg)
            # Extract agent features for Global State (CTDE)
            # Use Actor's internal embedding method to get [1, F] per agent
            # We must gather them in a fixed order (e.g. sorted by agent_id for consistency, or permute if set transformer)
            active_agents = sorted(obs_by_agent.keys())
            global_feats = []
            
            # Note: We reuse the actor to embed the observation for the global state construction
            # In a stricter impl, we might use a separate encoder or just the raw features.
            # Here we use actor._post_embed(obs) which gives [1, post_dim]
            with torch.no_grad():
                for aid in active_agents:
                    z = actor._post_embed(obs_by_agent[aid]) # [1, Dim]
                    global_feats.append(z)
            
            # Stack -> [1, m, Dim]
            if global_feats:
                global_state_t = torch.stack(global_feats, dim=0).unsqueeze(0)
            else:
                # Fallback if no agents (rare)
                global_state_t = torch.zeros(1, 1, 128) # dummy
            
            # Pad/Mask is handled by Set Transformer implicitly if we just pass [1, m, F]
            # Since Set Transformer is permutation invariant, order doesn't matter much strictly speaking, 
            # Since Set Transformer is permutation invariant, order doesn't matter much strictly speaking, 
            # but stable order helps training stability.

            actions = {}
            aux_store = {}
            for aid, obs in obs_by_agent.items():
                act, aux = actor.act(obs)
                # Honor warmup: new agents don't modify edges
                if aid in warmup_until and warmup_until[aid] > 0:
                    act.internal_edges = []
                actions[aid] = act
                aux_store[aid] = (obs, aux, global_state_t) # Store global state too

            # For messaging, use dict form (bus only needs msg_out)
            actions_as_dict = {aid: {'msg_out': a.msg_out, 'internal_edges': [
                {'i': e.i, 'j': e.j, 'op': e.op, 'score': e.score} for e in a.internal_edges
            ]} for aid, a in actions.items()}
            # Neighbors mapping via current cross-edges
            neighbors = {}
            part_map = env.partitions
            edge_pairs = list(env.edges)
            for (u, v) in edge_pairs:
                a, b = part_map.get(u), part_map.get(v)
                if a is None or b is None or a == b:
                    continue
                neighbors.setdefault(a, set()).add(b)
                neighbors.setdefault(b, set()).add(a)
            neighbors = {k: sorted(list(v)) for k, v in neighbors.items()}
            # Optional quantization/noise on msg_out (training only)
            from training.rollout import maybe_quantize_noise
            for aid in list(actions_as_dict.keys()):
                actions_as_dict[aid]['msg_out'] = maybe_quantize_noise(actions_as_dict[aid]['msg_out'], cfg, training=True)
            bus.publish(actions_as_dict, neighbors)

            deltaE = merger.merge(actions, env.graph)
            next_obs, reward, info = env.step(deltaE, partitions)
            bus.tick()
            total_delta_sz += len(deltaE.add) + len(deltaE.delete)
            total_rejects += len(deltaE.rejected)
            # Aggregate simple proxies
            avg_return += reward
            success_p += float(info.get('success_proxy', 0.0))
            delay_p += float(info.get('delay_proxy', 0.0))
            energy_p += float(info.get('energy_proxy', 0.0))

            # Autoscale step
            acfg = (cfg.get('autoscale') or {})
            target_slots = int(acfg.get('target_slots_per_agent', 10))
            rho_bar = len(env.nodes) / max(1, m * target_slots)
            m_new, action = autoscaler.step(len(env.nodes), rho_bar, m)
            if action:
                if verbose:
                    print(f"    [AutoScale] action={action[0]} 螖m={action[1]} | m {m}->{m_new} | rho_bar={rho_bar:.3f}")
                m = m_new
                partitions = partitioner.resize_agents(m, env.nodes, env.node_pos)
                env.partitions = partitions
                # Update warmup for new agents
                new_agents = set(partitions.values()) - agents_set
                agents_set = set(partitions.values())
                warm_steps = int((cfg.get('autoscale') or {}).get('warmup_steps', 50))
                for na in new_agents:
                    warmup_until[na] = warm_steps

            # Decrement warmup counters
            for a in list(warmup_until.keys()):
                warmup_until[a] -= 1
                if warmup_until[a] <= 0:
                    del warmup_until[a]

            done = (t == steps_T - 1)
            for aid, (obs, aux, global_state_t) in aux_store.items():
                trainer.store_transition(
                    agent_id=aid,
                    obs=obs,
                    global_state=global_state_t,
                    idx=aux['idx'],
                    op=aux['op'],
                    cands=aux['cands'],
                    logprob=aux['logprob'],
                    value=aux['value'],
                    reward=float(reward),
                    done=done,
                    internal_cands=aux.get('internal_cands'),  # Hierarchical action space
                    cross_cands=aux.get('cross_cands'),  # Cross-partition candidates
                )

            partitions = maybe_local_rebalance(partitioner, env, partitions, t, cfg)

        a_loss, c_loss, kl, ret_mean, ret_std, v_rmse, clip_frac, lr = trainer.update()
        if verbose:
            print(f"  [Update] loss_pi={a_loss:.4f} loss_v_mse={c_loss:.4f} loss_v_rmse={v_rmse:.4f} KL={kl:.5f} ret_mean={ret_mean:.2f} ret_std={ret_std:.2f} lr={lr:.2e} | deltaE={total_delta_sz} rejects={total_rejects}")
        # store history
        hist_actor.append(a_loss); hist_critic.append(c_loss); hist_ret.append(avg_return/max(1,steps_T));
        hist_succ.append(success_p/max(1,steps_T)); hist_delay.append(delay_p/max(1,steps_T)); hist_energy.append(energy_p/max(1,steps_T))
        hist_kl.append(kl); hist_m.append(m); hist_rho.append(len(env.nodes)/max(1,m*int((cfg.get('autoscale') or {}).get('target_slots_per_agent',10)))); hist_v_rmse.append(v_rmse); hist_clip.append(clip_frac)
        # Metrics to CSV
        if csv_writer:
            steps = max(1, steps_T)
            csv_writer.writerow([
                ep+1,
                f"{avg_return/steps:.6f}", f"{success_p/steps:.6f}", f"{delay_p/steps:.6f}", f"{energy_p/steps:.6f}",
                len(env.edges), f"{kl:.6f}", m, f"{(len(env.nodes)/max(1,m*int((cfg.get('autoscale') or {}).get('target_slots_per_agent',10)))):.6f}", total_delta_sz, total_rejects,
                f"{a_loss:.6f}", f"{c_loss:.6f}", f"{v_rmse:.6f}", f"{clip_frac:.6f}", f"{ret_mean:.6f}", f"{ret_std:.6f}", f"{lr:.6f}",
            ])

        # Visualization (curves + topology)
        viz = (cfg.get('viz') or {})
        if bool(viz.get('enable', True)) and ((ep + 1) % int(viz.get('save_every_episodes', 50)) == 0):
            # curves
            try:
                import matplotlib.pyplot as _plt
                import numpy as _np
                def _save_curve(vals, name, win: int = 20):
                    _plt.figure(figsize=(6,3))
                    x = _np.asarray(vals, dtype=float)
                    _plt.plot(x, alpha=0.35, label='episode')
                    # moving average
                    if len(x) >= max(3, win):
                        kernel = _np.ones(win) / float(win)
                        ma = _np.convolve(x, kernel, mode='valid')
                        _plt.plot(_np.arange(win-1, win-1+len(ma)), ma, linewidth=2.0, label=f'ma{win}')
                    # cumulative mean
                    cm = _np.cumsum(x) / _np.arange(1, len(x)+1)
                    _plt.plot(cm, linewidth=1.5, label='mean')
                    _plt.legend(fontsize=8)
                    _plt.title(name); _plt.tight_layout(); _plt.savefig(os.path.join(run_dir, f"{name}.png")); _plt.close()
                _save_curve(hist_actor, 'loss_pi')
                _save_curve(hist_critic, 'loss_v_mse')
                _save_curve(hist_v_rmse, 'loss_v_rmse')
                _save_curve(hist_ret, 'return')
                _save_curve(hist_succ, 'success_rate')
                _save_curve(hist_delay, 'delay_proxy')
                _save_curve(hist_energy, 'energy_proxy')
                _save_curve(hist_kl, 'kl')
                _save_curve(hist_m, 'm')
                _save_curve(hist_rho, 'rho_bar')
                _save_curve(hist_clip, 'value_clip_frac')
                # combined m & rho plot for convenience
                _plt.figure(figsize=(6,3))
                _plt.plot(hist_m, label='m')
                _plt.plot(hist_rho, label='rho_bar')
                _plt.legend(); _plt.title('m_rho'); _plt.tight_layout(); _plt.savefig(os.path.join(run_dir, 'm_rho.png')); _plt.close()
            except Exception:
                pass
            max_nodes_for_plot = int(viz.get('max_graph_nodes_for_plot', 400))
            if bool(viz.get('draw_topology', True)):
                plot_topology(env.graph, env.partitions, os.path.join(run_dir, f"topology_ep{ep+1}.png"), dpi=int(viz.get('dpi', 160)), max_nodes=max_nodes_for_plot)
                # Also generate logical topology graph
                plot_logical_topology(env.nodes, env.edges, env.partitions, 
                                      os.path.join(run_dir, f"logical_ep{ep+1}.png"),
                                      dpi=int(viz.get('dpi', 160)), layout='spring')
            if bool(viz.get('draw_agent_coverage', True)):
                region_bbox = tuple((cfg.get('region') or {}).get('bbox', [-300, -300, 300, 300]))
                city_stride = float((cfg.get('city') or {}).get('grid_stride', 100.0))
                city_road_w = float((cfg.get('city') or {}).get('road_width', 25.0))
                plot_agent_coverage(env.partitions, env.node_pos, os.path.join(run_dir, f"coverage_ep{ep+1}.png"), 
                                    dpi=int(viz.get('dpi', 160)), max_nodes=max_nodes_for_plot,
                                    bbox=region_bbox, grid_stride=city_stride, road_width=city_road_w)

        # Checkpointing and early stopping on metric (success rate proxy)
        metric_val = (success_p / max(1, steps_T))
        if best_metric is None or metric_val > best_metric:
            best_metric = metric_val
            try:
                torch.save(Actor(cfg).state_dict() if not hasattr(actor,'state_dict') else actor.state_dict(), os.path.join(run_dir, 'best.pt'))
            except Exception:
                pass
        if checkpoint_every > 0 and ((ep + 1) % checkpoint_every == 0):
            try:
                torch.save(actor.state_dict(), os.path.join(run_dir, f'ckpt_ep{ep+1}.pt'))
            except Exception:
                pass
        if stopper:
            improved = stopper.step({'success_rate': metric_val})
            if not improved and stopper.bad >= stopper.patience:
                if verbose:
                    print(f"[EarlyStop] No improvement in {stopper.patience} episodes. Stopping.")
                break

    if bool((cfg.get('viz') or {}).get('enable', True)):
        max_nodes_for_plot = int((cfg.get('viz') or {}).get('max_graph_nodes_for_plot', 400))
        if bool((cfg.get('viz') or {}).get('draw_topology', True)):
            plot_topology(env.graph, env.partitions, os.path.join(run_dir, f"topology_final.png"), dpi=int((cfg.get('viz') or {}).get('dpi', 160)), max_nodes=max_nodes_for_plot)
        if bool((cfg.get('viz') or {}).get('draw_agent_coverage', True)):
            region_bbox = tuple((cfg.get('region') or {}).get('bbox', [-300, -300, 300, 300]))
            city_stride = float((cfg.get('city') or {}).get('grid_stride', 100.0))
            city_road_w = float((cfg.get('city') or {}).get('road_width', 25.0))
            plot_agent_coverage(env.partitions, env.node_pos, os.path.join(run_dir, f"agent_coverage.png"), 
                                dpi=int((cfg.get('viz') or {}).get('dpi', 160)), max_nodes=max_nodes_for_plot,
                                bbox=region_bbox, grid_stride=city_stride, road_width=city_road_w)
        # Generate logical topology graph (graph-layout based, not physical positions)
        plot_logical_topology(env.nodes, env.edges, env.partitions, 
                              os.path.join(run_dir, "logical_topology.png"),
                              dpi=int((cfg.get('viz') or {}).get('dpi', 160)), layout='spring')

    # Quick test pass to produce test_metrics.json using greedy actor
    try:
        import json as _json
        env.reset(partitions)
        T_test = min(last_steps_T, 64)
        test_ret = 0.0
        for t in range(T_test):
            obs_by_agent = build_observations(env, partitions, bus, role_ids, cfg)
            actions = {aid: actor.act(obs)[0] for aid, obs in obs_by_agent.items()}
            deltaE = merger.merge(actions, env.graph)
            _obs, rew, _info = env.step(deltaE, partitions)
            bus.tick()
            test_ret += rew
        with open(os.path.join(run_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
            _json.dump({'avg_return': float(test_ret/max(1,T_test))}, f)
    except Exception:
        pass

    if csv_file:
        csv_file.close()
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=os.environ.get('RL_CFG'))
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--no-viz', action='store_true')
    parser.add_argument('--run-name', type=str, default=None)
    # Optional overrides for city/churn via CLI
    parser.add_argument('--city-enable', type=int, default=None, help='1/0 to enable city mobility')
    parser.add_argument('--churn-join-rate', type=float, default=None)
    parser.add_argument('--churn-min-nodes', type=int, default=None)
    parser.add_argument('--churn-max-nodes', type=int, default=None)
    args = parser.parse_args()
    overrides = {'training': {}, 'viz': {}}
    if args.episodes is not None:
        overrides['training']['episodes'] = args.episodes
    if args.max_steps is not None:
        overrides['training']['max_steps_per_episode'] = args.max_steps
    if args.run_name is not None:
        overrides['training']['run_name'] = args.run_name
    if args.no_viz:
        overrides['viz']['enable'] = False
    if args.city_enable is not None:
        overrides['city'] = {'enable': bool(args.city_enable)}
    churn_override = {}
    if args.churn_join_rate is not None:
        churn_override['join_rate'] = args.churn_join_rate
    if args.churn_min_nodes is not None:
        churn_override['min_nodes'] = args.churn_min_nodes
    if args.churn_max_nodes is not None:
        churn_override['max_nodes'] = args.churn_max_nodes
    if churn_override:
        overrides['churn'] = churn_override
    run(args.cfg, overrides)



