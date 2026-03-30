from __future__ import annotations

from typing import Iterable

from .tool_types import AssistantTool, ToolSpec


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, AssistantTool] = {}

    def register(self, tool: AssistantTool) -> None:
        name = str(tool.spec.name).strip()
        if not name:
            raise ValueError("Tool spec name cannot be empty.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> AssistantTool | None:
        return self._tools.get(str(name or "").strip())

    def names(self) -> set[str]:
        return set(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def extend(self, tools: Iterable[AssistantTool]) -> None:
        for tool in tools:
            self.register(tool)


TOOL_REGISTRY = ToolRegistry()
