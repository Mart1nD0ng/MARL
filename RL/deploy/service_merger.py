from __future__ import annotations

from typing import Dict
from core.merger_safety import MergerSafety
from core.data_structures import DeltaE


class MergerService:
    def __init__(self, merger: MergerSafety):
        self.merger = merger

    def apply_actions(self, actions_by_agent: Dict[int, dict], graph: dict) -> DeltaE:
        return self.merger.merge(actions_by_agent, graph)

