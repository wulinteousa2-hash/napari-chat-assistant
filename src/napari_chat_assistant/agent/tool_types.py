from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


ExecutionMode = Literal["immediate", "worker"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    description: str = ""
    required: bool = False
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    enum: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    display_name: str
    category: str
    description: str
    execution_mode: ExecutionMode
    supported_layer_types: tuple[str, ...] = ()
    parameter_schema: tuple[ParamSpec, ...] = ()
    output_type: str = "message"
    supports_batch: bool = False
    ui_metadata: dict[str, Any] = field(default_factory=dict)
    provenance_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    viewer: Any
    layer_context: dict[str, Any]
    selected_layer_profile: dict[str, Any] | None = None
    session_memory: dict[str, Any] | None = None


@dataclass
class PreparedJob:
    tool_name: str
    kind: str
    mode: ExecutionMode
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"tool_name": self.tool_name, "kind": self.kind, **self.payload}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, default_tool_name: str | None = None) -> "PreparedJob":
        tool_name = str(data.get("tool_name") or default_tool_name or "").strip()
        return cls(
            tool_name=tool_name,
            kind=str(data["kind"]),
            mode="worker",
            payload={k: v for k, v in data.items() if k not in {"tool_name"}},
        )


@dataclass
class ToolResult:
    tool_name: str
    kind: str
    payload: dict[str, Any]
    message: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {"tool_name": self.tool_name, "kind": self.kind, **self.payload}
        if self.message:
            data["message"] = self.message
        if self.provenance:
            data["provenance"] = self.provenance
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, default_tool_name: str | None = None) -> "ToolResult":
        tool_name = str(data.get("tool_name") or default_tool_name or "").strip()
        return cls(
            tool_name=tool_name,
            kind=str(data["kind"]),
            payload={k: v for k, v in data.items() if k not in {"tool_name", "message", "provenance"}},
            message=str(data.get("message") or ""),
            provenance=dict(data.get("provenance") or {}),
        )


class AssistantTool(Protocol):
    spec: ToolSpec

    def prepare(self, ctx: ToolContext, arguments: dict[str, Any]) -> PreparedJob | str:
        ...

    def execute(self, job: PreparedJob) -> ToolResult:
        ...

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        ...
