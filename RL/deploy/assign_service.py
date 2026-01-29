from __future__ import annotations

from typing import Dict
from core.partitioner import Partitioner


class AssignService:
    def __init__(self, partitioner: Partitioner):
        self.partitioner = partitioner

    def assign_node(self, node_id: int) -> int:
        return self.partitioner.node_to_agent(node_id)

    def get_partition(self, agent_id: int):
        return self.partitioner.get_partition(agent_id)

    def rebalance(self):
        self.partitioner.local_rebalance()

