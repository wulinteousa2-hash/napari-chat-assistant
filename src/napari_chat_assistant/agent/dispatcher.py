from __future__ import annotations

import napari

from .context import layer_context_json
from .tool_registry import TOOL_REGISTRY
from .tool_types import PreparedJob, ToolContext, ToolResult
from .tools_builtin import builtin_tools


def _ensure_builtin_registry() -> None:
    for tool in builtin_tools():
        if TOOL_REGISTRY.get(tool.spec.name) is None:
            TOOL_REGISTRY.register(tool)


def _tool_context(viewer: napari.Viewer) -> ToolContext:
    payload = layer_context_json(viewer)
    return ToolContext(
        viewer=viewer,
        layer_context=payload,
        selected_layer_profile=payload.get("selected_layer_profile"),
    )


def prepare_tool_job(viewer: napari.Viewer, tool_name: str, arguments: dict) -> dict:
    _ensure_builtin_registry()
    registry_tool = TOOL_REGISTRY.get(tool_name)
    if registry_tool is None:
        return {"mode": "immediate", "message": f"Unsupported tool: {tool_name}"}
    prepared = registry_tool.prepare(_tool_context(viewer), arguments or {})
    if isinstance(prepared, str):
        return {"mode": "immediate", "message": prepared}
    if prepared.mode == "immediate":
        result = registry_tool.execute(prepared)
        message = registry_tool.apply(_tool_context(viewer), result)
        return {"mode": "immediate", "message": message}
    return {"mode": prepared.mode, "job": prepared.to_dict()}


def run_tool_job(job: dict) -> dict:
    _ensure_builtin_registry()
    registry_tool = TOOL_REGISTRY.get(job.get("tool_name", ""))
    if registry_tool is None:
        raise ValueError(f"Unsupported tool job: {job.get('tool_name', '') or job.get('kind', '')}")
    prepared_job = PreparedJob.from_dict(job)
    return registry_tool.execute(prepared_job).to_dict()


def apply_tool_job_result(viewer: napari.Viewer, result: dict) -> str:
    _ensure_builtin_registry()
    registry_tool = TOOL_REGISTRY.get(result.get("tool_name", ""))
    if registry_tool is None:
        return f"Unsupported tool result: {result.get('kind', '')}"
    return registry_tool.apply(_tool_context(viewer), ToolResult.from_dict(result))
