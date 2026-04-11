from __future__ import annotations

from typing import Any

from napari_chat_assistant.telemetry.logging_utils import append_telemetry_event


ACTIVITY_WORKSPACE = "workspace_operation"
ACTIVITY_TOOL = "tool_executed"
ACTIVITY_MODEL = "model_responded"
ACTIVITY_CODE = "code_executed"
ACTIVITY_USER = "user_feedback"
ACTIVITY_ERROR = "error_occurred"
ACTIVITY_DIAGNOSTIC = "diagnostic_event"

VALID_ACTIVITY_TYPES = {
    ACTIVITY_WORKSPACE,
    ACTIVITY_TOOL,
    ACTIVITY_MODEL,
    ACTIVITY_CODE,
    ACTIVITY_USER,
    ACTIVITY_ERROR,
    ACTIVITY_DIAGNOSTIC,
}


def log_activity(
    activity_type: str,
    details: dict[str, Any],
    source: str = "assistant",
) -> None:
    """
    Log an activity event with structured details.

    Args:
        activity_type: One of ACTIVITY_* constants
        details: Dict with activity-specific details
        source: Who/what triggered this activity (e.g., "assistant", "user", "system")

    Example:
        log_activity(
            ACTIVITY_WORKSPACE,
            {"operation": "save", "layer_count": 5, "file_size_mb": 12.5},
            source="user"
        )
    """
    if activity_type not in VALID_ACTIVITY_TYPES:
        activity_type = "unknown_activity"

    payload = {
        "activity_type": activity_type,
        "source": str(source or "unknown").strip(),
        **{str(k): v for k, v in dict(details or {}).items()},
    }
    append_telemetry_event("activity", payload)


def log_workspace_operation(
    operation: str,
    layer_count: int = 0,
    file_size_mb: float = 0.0,
    success: bool = True,
    error: str | None = None,
) -> None:
    """Log a workspace operation."""
    log_activity(
        ACTIVITY_WORKSPACE,
        {
            "operation": operation,
            "layer_count": layer_count,
            "file_size_mb": file_size_mb,
            "success": success,
            "error": error,
        },
        source="user",
    )


def log_tool_execution(
    tool_name: str,
    success: bool,
    duration_ms: int = 0,
    error: str | None = None,
    layer_involved: str | None = None,
) -> None:
    """Log a tool execution."""
    log_activity(
        ACTIVITY_TOOL,
        {
            "tool_name": tool_name,
            "success": success,
            "duration_ms": duration_ms,
            "error": error,
            "layer_involved": layer_involved,
        },
        source="assistant",
    )


def log_model_response(
    model_name: str,
    response_action: str,
    latency_ms: int = 0,
    turn_number: int = 0,
) -> None:
    """Log a model response completion."""
    log_activity(
        ACTIVITY_MODEL,
        {
            "model_name": model_name,
            "response_action": response_action,
            "latency_ms": latency_ms,
            "turn_number": turn_number,
        },
        source="model",
    )


def log_user_feedback(
    feedback_type: str,
    context: str = "",
) -> None:
    """Log user feedback."""
    log_activity(
        ACTIVITY_USER,
        {
            "feedback_type": feedback_type,
            "context": context,
        },
        source="user",
    )


def log_error(
    error_type: str,
    error_message: str,
    context: dict[str, Any] | None = None,
) -> None:
    """Log an error event."""
    log_activity(
        ACTIVITY_ERROR,
        {
            "error_type": error_type,
            "error_message": error_message,
            **(context or {}),
        },
        source="system",
    )
