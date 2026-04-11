"""
Intent telemetry module for napari-chat-assistant.

This module captures what users are trying to accomplish, independent of chat agent implementation.
Used for structural improvement analysis and understanding user workflows.

Example usage:

>>> event = IntentEvent(
...     intent_category="analysis",
...     intent_description="User wants to measure ROI intensity",
...     layer_context=build_layer_context(selected_profile),
...     workspace_state="loaded",
...     success=True,
...     duration_ms=2500,
...     feedback="helpful"
... )
>>> record_intent(event)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from napari_chat_assistant.telemetry.logging_utils import append_telemetry_event


@dataclass
class IntentEvent:
    """
    Captures user intent independent of implementation.

    Used to understand what users are trying to accomplish for structural improvement.
    Not coupled to chat agent behavior.
    """

    intent_category: str
    intent_description: str
    layer_context: dict[str, Any]
    workspace_state: str
    success: bool
    duration_ms: int
    feedback: str | None = None
    metadata: dict[str, Any] | None = None


def record_intent(event: IntentEvent) -> None:
    """
    Record an intent event for structural improvement analysis.

    This function helps understand user workflows without coupling to chat agent implementation.
    Data is stored in the telemetry log for later analysis.

    Args:
        event: IntentEvent with intent details
    """

    _ = datetime.now(timezone.utc)
    payload = {
        "intent_category": event.intent_category,
        "intent_description": event.intent_description,
        "layer_context": event.layer_context,
        "workspace_state": event.workspace_state,
        "success": event.success,
        "duration_ms": event.duration_ms,
        "feedback": event.feedback,
        "metadata": event.metadata or {},
    }
    append_telemetry_event("intent_captured", payload)


def build_layer_context(selected_layer_profile: dict[str, Any] | None) -> dict[str, Any]:
    """Build layer context dict from selected layer profile."""
    if not isinstance(selected_layer_profile, dict):
        return {"layer_count": 0, "layer_types": []}

    return {
        "layer_name": str(selected_layer_profile.get("layer_name", "")).strip(),
        "layer_type": str(selected_layer_profile.get("layer_type", "")).strip(),
        "shape": selected_layer_profile.get("shape"),
    }


def categorize_intent(prompt_text: str) -> str:
    """
    Infer intent category from prompt text.

    Returns one of: "analysis", "data_prep", "visualization", "workflow", "tool_usage", "unknown"
    """
    source = " ".join(str(prompt_text or "").strip().lower().split())
    if not source:
        return "unknown"

    if any(word in source for word in ("measure", "analyze", "threshold", "histogram", "statistics", "roi", "intensity")):
        return "analysis"

    if any(word in source for word in ("clahe", "gaussian", "denoise", "smooth", "filter", "enhance", "normalize")):
        return "data_prep"

    if any(word in source for word in ("display", "overlay", "color", "scale", "grid", "montage", "zoom")):
        return "visualization"

    if any(word in source for word in ("save", "load", "workspace", "session", "project", "restore")):
        return "workflow"

    if any(word in source for word in ("tool", "action", "run", "execute", "apply")):
        return "tool_usage"

    return "unknown"
