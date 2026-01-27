from __future__ import annotations

import random
from typing import Dict, List


class MessageBus:
    """Low-bandwidth adjacent messaging with TTL and dropout.

    Stores fixed-D vectors per agent, decrements TTL, and drops per config.
    """

    def __init__(self, dim: int, ttl: int = 3, dropout: float = 0.2):
        self.dim = dim
        self.ttl = ttl
        self.dropout = dropout
        # agent_id -> list of (vec, ttl)
        self._inbox: Dict[int, List[List[float]]] = {}

    def reset(self):
        self._inbox.clear()

    def get_messages(self, agent_id: int) -> List[float]:
        # Merge messages for agent; simple average
        msgs = self._inbox.get(agent_id, [])
        if not msgs:
            return [0.0] * self.dim
        acc = [0.0] * self.dim
        for vec, _ttl in msgs:
            for k in range(self.dim):
                acc[k] += float(vec[k])
        return [v / max(1, len(msgs)) for v in acc]

    def publish(self, actions_by_agent: Dict[int, dict], neighbors: Dict[int, List[int]] | None = None):
        """Publish outgoing messages to neighbors' inboxes.

        - actions_by_agent[aid] should include 'msg_out' as fixed-D vector.
        - neighbors mapping lists adjacent agent ids per aid; fallback to loopback.
        """
        for aid, act in actions_by_agent.items():
            vec = act.get('msg_out') if isinstance(act, dict) else getattr(act, 'msg_out', None)
            if not vec:
                continue
            if random.random() < self.dropout:
                continue
            targets = neighbors.get(aid, []) if neighbors else []
            if not targets:
                targets = [aid]
            for tid in targets:
                self._inbox.setdefault(tid, []).append((list(vec), self.ttl))

    def tick(self):
        # Decrement TTL and remove expired
        for aid in list(self._inbox.keys()):
            kept = []
            for vec, ttl in self._inbox[aid]:
                ttl -= 1
                if ttl > 0:
                    kept.append((vec, ttl))
            if kept:
                self._inbox[aid] = kept
            else:
                del self._inbox[aid]
