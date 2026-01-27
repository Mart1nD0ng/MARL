from __future__ import annotations

from typing import List, Dict
from models.actor import Actor


class ActorService:
    def __init__(self, cfg: dict):
        self.actor = Actor(cfg)

    def act_batch(self, observations: List[dict]) -> List[dict]:
        # Accepts Observation-like dicts; returns Action-like dicts
        out = []
        for _obs in observations:
            # Placeholder; real service converts dict<->dataclass and runs actor
            out.append({'internal_edges': [], 'cross_edge_scores': [], 'msg_out': []})
        return out

