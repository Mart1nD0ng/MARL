from __future__ import annotations

class RegistryBus:
    def __init__(self):
        self.registry = {}

    def register(self, agent_id: int, endpoint: str):
        self.registry[agent_id] = endpoint

    def get(self, agent_id: int) -> str:
        return self.registry.get(agent_id, "")

