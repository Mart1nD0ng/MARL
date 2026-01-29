from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AutoScaleState:
    high_cnt: int = 0
    low_cnt: int = 0


class AutoScaler:
    def __init__(self, cfg: dict, partitioner):
        self.cfg = cfg
        self.partitioner = partitioner
        self.state = AutoScaleState()

    def reset_state(self):
        self.state = AutoScaleState()

    def step(self, num_nodes: int, rho_bar: float, current_m: int):
        acfg = (self.cfg.get('autoscale') or {})
        if not bool(acfg.get('enable', False)):
            return current_m, None
        target_slots = int(acfg.get('target_slots_per_agent', 10))
        min_agents = int(acfg.get('min_agents', 2))
        max_agents = int(acfg.get('max_agents', 128))
        m_sug = self.partitioner.estimate_m_for_nodes(num_nodes, target_slots, min_agents, max_agents)
        action = None

        if m_sug > current_m and rho_bar > float(acfg.get('rho_high', 0.9)):
            self.state.high_cnt += 1
            self.state.low_cnt = 0
            if self.state.high_cnt >= int(acfg.get('hysteresis_steps_high', 100)):
                action = ("scale_out", m_sug - current_m)
                self.state.high_cnt = 0
        elif m_sug < current_m and rho_bar < float(acfg.get('rho_low', 0.58)):
            self.state.low_cnt += 1
            self.state.high_cnt = 0
            if self.state.low_cnt >= int(acfg.get('hysteresis_steps_low', 150)):
                action = ("scale_in", current_m - m_sug)
                self.state.low_cnt = 0
        else:
            self.state.high_cnt = 0
            self.state.low_cnt = 0
        return (m_sug if action else current_m), action
