from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import torch
import torch.nn.functional as F


@dataclass
class Transition:
    agent_id: int
    obs: object
    global_state: torch.Tensor  # New: centralised state for critic
    idx: int
    op: int  # Now 0-3: internal_add, internal_del, cross_add, cross_del
    cands: List[Tuple[int, int]]
    internal_cands: List[Tuple[int, int]]  # For hierarchical action space
    cross_cands: List[Tuple[int, int]]  # For cross-partition actions
    logprob: float
    value: float
    reward: float
    done: bool


class MAPPOTrainer:
    """MAPPO with Centralized Critic (Set Transformer).

    - Actor: Decentralized execution (local obs)
    - Critic: Centralized training (global state)
    """

    def __init__(self, actor, critic, cfg: dict):
        self.actor = actor
        self.critic = critic
        self.cfg = cfg
        self.buffer: List[Transition] = []
        tcfg = cfg.get('training', {}) or {}
        self.gamma = float(tcfg.get('gamma', 0.99))
        self.lmbda = float(tcfg.get('gae_lambda', 0.95))
        self.clip_eps = float(tcfg.get('clip', 0.15))
        self.lr = float(tcfg.get('lr', 3e-4))
        self.min_lr = float(tcfg.get('min_lr', 5e-5))
        self.epochs = int(tcfg.get('epochs', 4))
        self.max_grad_norm = 0.5
        self.entropy_coef = 0.01
        lcfg = cfg.get('loss', {}) or {}
        self.value_coef = float(lcfg.get('value_coef', 0.5))
        self.value_clip = float(lcfg.get('value_clip', 2.0))
        self.normalize_returns = bool(lcfg.get('normalize_returns', True))
        self.log_value_mean = bool(lcfg.get('log_value_mean', True))
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )
        self.target_kl = float(tcfg.get('target_kl', 0.01))
        self.adapt_lr_on_kl = bool(tcfg.get('adapt_lr_on_kl', True))
        self.truncation_steps = int(tcfg.get('truncation_steps', 128))
        self._last_lr = self.lr

    def reset_rnn(self):
        if hasattr(self.actor, 'reset_rnn'):
            self.actor.reset_rnn()

    def store_transition(self, agent_id: int, obs, global_state, idx: int, op: int, cands: List[Tuple[int, int]],
                         logprob: float, value: float, reward: float, done: bool,
                         internal_cands: List[Tuple[int, int]] = None,
                         cross_cands: List[Tuple[int, int]] = None):
        if internal_cands is None:
            internal_cands = cands  # Fallback for backward compatibility
        if cross_cands is None:
            cross_cands = []
        self.buffer.append(Transition(agent_id, obs, global_state, idx, op, cands, internal_cands, cross_cands, logprob, value, reward, done))

    def _compute_gae(self):
        # Group transitions by agent, preserve temporal order
        by_agent: Dict[int, List[Transition]] = {}
        for tr in self.buffer:
            by_agent.setdefault(tr.agent_id, []).append(tr)

        advantages: List[float] = []
        returns: List[float] = []
        order: List[int] = []  # indices back into buffer

        for aid, traj in by_agent.items():
            # ensure time order is preserved (already appended in order)
            last_adv = 0.0
            # Bootstrap with last state value if not done, else 0
            next_value = float(traj[-1].value) if (len(traj) > 0 and not bool(traj[-1].done)) else 0.0
            for t in reversed(range(len(traj))):
                tr = traj[t]
                mask = 0.0 if tr.done else 1.0
                delta = tr.reward + self.gamma * next_value * mask - tr.value
                last_adv = delta + self.gamma * self.lmbda * mask * last_adv
                ret = last_adv + tr.value
                advantages.insert(0, last_adv)
                returns.insert(0, ret)
                order.insert(0, self.buffer.index(tr))
                next_value = tr.value

        # Normalize advantages
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)
        ret_t = torch.tensor(returns, dtype=torch.float32)
        return order, adv_t, ret_t

    def update(self):
        if not self.buffer:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self._last_lr

        order, adv_t, ret_t = self._compute_gae()
        # Stats for logging (pre-normalization)
        ret_mean = float(ret_t.mean().item())
        ret_std = float(ret_t.std(unbiased=False).item())
        # Optionally whiten returns for critic target
        if self.normalize_returns:
            ret_hat = (ret_t - ret_t.mean()) / (ret_t.std(unbiased=False) + 1e-8)
        else:
            ret_hat = ret_t
        old_logps = torch.tensor([self.buffer[i].logprob for i in order], dtype=torch.float32)
        idx_list = [self.buffer[i].idx for i in order]
        op_list = [self.buffer[i].op for i in order]

        # PPO epochs (full-batch)
        observed_kl = 0.0
        lr_changed = False
        for epoch in range(self.epochs):
            new_logps = []
            values = []
            entropies = []
            
            # Re-evaluate actor (local) and critic (global)
            # Optimize: can batch global states if needed, but for now loop is safe
            for i in order:
                tr = self.buffer[i]
                logp, _, entropy = self.actor.evaluate_logprob_and_value(tr.obs, tr.idx, tr.op, tr.cands, tr.internal_cands, tr.cross_cands)
                
                # Centralized Critic call
                # global_state: [1, m, F] -> [1] after critic
                val_pred = self.critic(tr.global_state)
                
                new_logps.append(logp)
                values.append(val_pred)
                entropies.append(entropy)

            new_logps = torch.stack(new_logps).float()
            values = torch.stack(values).float()
            entropies = torch.stack(entropies).float()

            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
            entropy_loss = - self.entropy_coef * torch.mean(entropies)
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            actor_loss = policy_loss + entropy_loss

            # Critic loss with value clipping
            v_pred = values.squeeze(-1)
            target = ret_hat
            raw_delta = v_pred - target
            if self.value_clip and self.value_clip > 0:
                diff = torch.clamp(raw_delta, -self.value_clip, self.value_clip)
            else:
                diff = raw_delta
            critic_loss = torch.mean(diff * diff)
            loss = actor_loss + self.value_coef * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()

            # KL early stop and optional LR adapt
            with torch.no_grad():
                observed_kl = torch.mean(old_logps - new_logps).abs().item()
            # Only reduce LR if KL is VERY high (2x target), and reduce more gently (0.8x instead of 0.5x)
            if self.target_kl and observed_kl > 2.0 * self.target_kl:
                if self.adapt_lr_on_kl:
                    for g in self.optimizer.param_groups:
                        new_lr = max(float(self.min_lr), g['lr'] * 0.8)  # Gentler reduction
                        if new_lr < g['lr']:
                            lr_changed = True
                        g['lr'] = new_lr
                        self._last_lr = g['lr']
                    if lr_changed:
                        try:
                            print(f"[KL EarlyStop] KL={observed_kl:.5f} > {2.0*self.target_kl:.5f}; lowering lr to {self._last_lr:.2e}")
                        except Exception:
                            pass
                break

        # KL estimate
        with torch.no_grad():
            new_logps_final = []
            for i in order:
                tr = self.buffer[i]
                lp, _, _ = self.actor.evaluate_logprob_and_value(tr.obs, tr.idx, tr.op, tr.cands, tr.internal_cands, tr.cross_cands)
                new_logps_final.append(lp)
            new_logps_final = torch.stack(new_logps_final).float()
            kl = torch.mean(old_logps - new_logps_final).abs().item()
        
        # Adaptive LR: increase if KL is in healthy range or too low
        if self.adapt_lr_on_kl and self.target_kl:
            if kl < 1.5 * self.target_kl and self._last_lr < self.lr:
                # KL is acceptable and LR is below initial - gradually recover
                for g in self.optimizer.param_groups:
                    # Recover faster if KL is very low, slower if KL is moderate
                    if kl < 0.3 * self.target_kl:
                        factor = 1.3  # Fast recovery when KL very low
                    elif kl < self.target_kl:
                        factor = 1.15  # Medium recovery
                    else:
                        factor = 1.05  # Slow recovery when KL near target
                    new_lr = min(self.lr, g['lr'] * factor)  # Cap at initial LR
                    if new_lr > g['lr']:
                        g['lr'] = new_lr
                        self._last_lr = g['lr']

        # Clear buffer
        self.buffer.clear()
        loss_v_rmse = float(torch.sqrt(critic_loss).item())
        clip_fraction = 0.0
        if self.value_clip and self.value_clip > 0:
            with torch.no_grad():
                clip_fraction = float((torch.abs(raw_delta) >= self.value_clip - 1e-6).float().mean().item())
        return float(actor_loss.item()), float(critic_loss.item()), float(kl), ret_mean, ret_std, loss_v_rmse, clip_fraction, self._last_lr


class EarlyStopper:
    def __init__(self, metric_name: str, patience: int, min_delta: float):
        self.metric = metric_name
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad = 0

    def step(self, metrics: Dict[str, float]) -> bool:
        val = float(metrics.get(self.metric, 0.0))
        if self.best is None or val > self.best + self.min_delta:
            self.best = val
            self.bad = 0
            return True
        self.bad += 1
        return False
