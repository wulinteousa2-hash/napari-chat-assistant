from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable


MAX_RECENT_ACTIONS = 5


@dataclass
class RecentAction:
    tool_name: str
    action_kind: str
    turn_id: str = ""
    input_layers: list[str] = field(default_factory=list)
    output_layers: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)
    explanation_hints: dict[str, Any] = field(default_factory=dict)
    message: str = ""


def empty_recent_action_state(max_items: int = MAX_RECENT_ACTIONS) -> dict[str, Any]:
    return {
        "max_items": max(1, int(max_items or MAX_RECENT_ACTIONS)),
        "items": [],
    }


def normalize_recent_action(item: dict[str, Any] | RecentAction | None) -> dict[str, Any]:
    if isinstance(item, RecentAction):
        item = {
            "tool_name": item.tool_name,
            "action_kind": item.action_kind,
            "turn_id": item.turn_id,
            "input_layers": item.input_layers,
            "output_layers": item.output_layers,
            "parameters": item.parameters,
            "result_summary": item.result_summary,
            "explanation_hints": item.explanation_hints,
            "message": item.message,
        }
    if not isinstance(item, dict):
        return {}
    tool_name = str(item.get("tool_name", "")).strip()
    action_kind = str(item.get("action_kind", "")).strip()
    if not tool_name:
        return {}
    return {
        "tool_name": tool_name,
        "action_kind": action_kind or tool_name,
        "turn_id": str(item.get("turn_id", "")).strip(),
        "input_layers": [str(name).strip() for name in item.get("input_layers", []) if str(name).strip()],
        "output_layers": [str(name).strip() for name in item.get("output_layers", []) if str(name).strip()],
        "parameters": dict(item.get("parameters", {}) or {}),
        "result_summary": dict(item.get("result_summary", {}) or {}),
        "explanation_hints": dict(item.get("explanation_hints", {}) or {}),
        "message": " ".join(str(item.get("message", "")).split()).strip(),
    }


def normalize_recent_action_state(data: dict[str, Any] | None, max_items: int = MAX_RECENT_ACTIONS) -> dict[str, Any]:
    state = empty_recent_action_state(max_items=max_items)
    if not isinstance(data, dict):
        return state
    state["max_items"] = max(1, int(data.get("max_items", max_items) or max_items))
    items = [normalize_recent_action(item) for item in data.get("items", [])]
    state["items"] = [item for item in items if item][: state["max_items"]]
    return state


def record_recent_action(
    data: dict[str, Any] | None,
    item: dict[str, Any] | RecentAction,
    *,
    max_items: int = MAX_RECENT_ACTIONS,
) -> dict[str, Any]:
    state = normalize_recent_action_state(data, max_items=max_items)
    normalized = normalize_recent_action(item)
    if not normalized:
        return state
    items = [existing for existing in state["items"] if not _same_action(existing, normalized)]
    items.insert(0, normalized)
    state["items"] = items[: state["max_items"]]
    return state


def latest_recent_action(
    data: dict[str, Any] | None,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    state = normalize_recent_action_state(data)
    if predicate is None:
        return dict(state["items"][0]) if state["items"] else {}
    for item in state["items"]:
        if predicate(item):
            return dict(item)
    return {}


def threshold_adjustment_direction(text: str) -> str:
    source = " ".join(str(text or "").strip().lower().split())
    if not source:
        return ""
    stricter_phrases = (
        "stricter",
        "more strict",
        "less area",
        "smaller mask",
        "too much background",
        "reduce background",
        "less background",
        "reduce the area",
        "shrink the mask",
        "increase threshold",
        "higher threshold",
    )
    looser_phrases = (
        "less strict",
        "looser",
        "more area",
        "larger mask",
        "include more",
        "more foreground",
        "grow the mask",
        "decrease threshold",
        "lower threshold",
        "expand the mask",
    )
    if any(phrase in source for phrase in stricter_phrases):
        return "stricter"
    if any(phrase in source for phrase in looser_phrases):
        return "looser"
    return ""


def refine_threshold_action(action: dict[str, Any] | None, direction: str) -> dict[str, Any]:
    item = normalize_recent_action(action)
    if item.get("action_kind") != "threshold":
        return {}
    direction_name = str(direction or "").strip().lower()
    if direction_name not in {"stricter", "looser"}:
        return {}

    parameters = dict(item.get("parameters", {}) or {})
    threshold_value = parameters.get("threshold_value")
    if threshold_value is None:
        return {}
    try:
        base_threshold = float(threshold_value)
    except Exception:
        return {}
    if not math.isfinite(base_threshold):
        return {}

    resolved_mode = str(parameters.get("foreground_mode_resolved") or parameters.get("foreground_mode") or "").strip().lower()
    if resolved_mode not in {"bright", "dim"}:
        return {}

    input_layers = list(item.get("input_layers", []) or [])
    layer_name = input_layers[0] if input_layers else ""
    if not layer_name:
        return {}

    image_min = parameters.get("image_min")
    image_max = parameters.get("image_max")
    try:
        image_min_value = float(image_min)
        image_max_value = float(image_max)
    except Exception:
        image_min_value = base_threshold - max(abs(base_threshold) * 0.5, 0.5)
        image_max_value = base_threshold + max(abs(base_threshold) * 0.5, 0.5)
    if not math.isfinite(image_min_value) or not math.isfinite(image_max_value) or image_max_value <= image_min_value:
        image_min_value = min(image_min_value, base_threshold) if math.isfinite(image_min_value) else base_threshold - 0.5
        image_max_value = max(image_max_value, base_threshold) if math.isfinite(image_max_value) else base_threshold + 0.5
        if image_max_value <= image_min_value:
            image_max_value = image_min_value + 1.0

    step = max((image_max_value - image_min_value) * 0.1, 1e-6)
    if resolved_mode == "bright":
        refined_threshold = base_threshold + step if direction_name == "stricter" else base_threshold - step
    else:
        refined_threshold = base_threshold - step if direction_name == "stricter" else base_threshold + step
    refined_threshold = min(image_max_value, max(image_min_value, refined_threshold))

    if abs(refined_threshold - base_threshold) < 1e-12:
        return {}

    direction_text = "more selective" if direction_name == "stricter" else "more inclusive"
    mode_text = "brighter regions" if resolved_mode == "bright" else "dimmer regions"
    return {
        "tool": "apply_threshold",
        "arguments": {
            "layer_name": layer_name,
            "polarity": resolved_mode,
            "threshold_value": refined_threshold,
        },
        "message": (
            f"Adjusting the last threshold on [{layer_name}] to be {direction_text} "
            f"while still keeping {mode_text} as foreground."
        ),
    }


def route_recent_action_followup(
    text: str,
    data: dict[str, Any] | None,
    *,
    selected_layer_name: str = "",
    selected_layer_type: str = "",
) -> dict[str, Any]:
    source = " ".join(str(text or "").strip().lower().split())
    if not source:
        return {}

    selected_name = str(selected_layer_name or "").strip()
    selected_type = str(selected_layer_type or "").strip().lower()

    direction = threshold_adjustment_direction(source)
    if direction:
        action = latest_recent_action(data, lambda item: item.get("action_kind") == "threshold")
        if action:
            return refine_threshold_action(action, direction)

    if _looks_like_same_settings_followup(source):
        action = latest_recent_action(data, lambda item: item.get("action_kind") == "threshold")
        if action:
            route = _rerun_same_threshold_settings(action, source, selected_layer_name=selected_name, selected_layer_type=selected_type)
            if route:
                return route

    if _looks_like_roi_handoff_followup(source):
        action = latest_recent_action(data)
        image_name = _resolve_recent_image_target(
            action,
            selected_layer_name=selected_name,
            selected_layer_type=selected_type,
            prefer_selected=_prefers_selected_image(source),
        )
        if image_name:
            return {
                "tool": "open_intensity_metrics_table",
                "arguments": {"layer_name": image_name},
                "message": f"Opening ROI Intensity Analysis for [{image_name}].",
            }

    if _looks_like_histogram_handoff_followup(source):
        action = latest_recent_action(data)
        image_name = _resolve_recent_image_target(
            action,
            selected_layer_name=selected_name,
            selected_layer_type=selected_type,
            prefer_selected=_prefers_selected_image(source),
        )
        if image_name:
            return {
                "tool": "plot_histogram",
                "arguments": {"layer_name": image_name},
                "message": f"Opening the intensity histogram for [{image_name}].",
            }

    return {}


def _looks_like_same_settings_followup(source: str) -> bool:
    phrases = (
        "same setting",
        "same settings",
        "same threshold",
        "use that setting again",
        "use those settings again",
        "do that again",
        "apply that to this image",
        "apply that to the selected image",
        "apply that to the current image",
        "apply the same thing",
        "use the same thing",
    )
    return any(phrase in source for phrase in phrases)


def _looks_like_roi_handoff_followup(source: str) -> bool:
    phrases = (
        "open roi intensity analysis",
        "open the roi intensity analysis",
        "open roi widget",
        "open the roi widget",
        "open roi intensity widget",
        "roi intensity analysis for that image",
        "roi intensity analysis for this image",
        "open the widget instead",
    )
    return any(phrase in source for phrase in phrases)


def _looks_like_histogram_handoff_followup(source: str) -> bool:
    phrases = (
        "show histogram for that image",
        "show histogram for this image",
        "show histogram for that result",
        "plot histogram for that image",
        "plot histogram for this image",
        "show histogram of that image",
        "show histogram of this image",
        "open histogram",
    )
    return any(phrase in source for phrase in phrases)


def _prefers_selected_image(source: str) -> bool:
    return any(token in source for token in ("this image", "selected image", "current image"))


def _resolve_recent_image_target(
    action: dict[str, Any] | None,
    *,
    selected_layer_name: str = "",
    selected_layer_type: str = "",
    prefer_selected: bool = False,
) -> str:
    selected_name = str(selected_layer_name or "").strip()
    if prefer_selected and str(selected_layer_type or "").strip().lower() == "image" and selected_name:
        return selected_name
    item = normalize_recent_action(action)
    inputs = list(item.get("input_layers", []) or [])
    if inputs:
        return inputs[0]
    if str(selected_layer_type or "").strip().lower() == "image" and selected_name:
        return selected_name
    return ""


def _rerun_same_threshold_settings(
    action: dict[str, Any] | None,
    source: str,
    *,
    selected_layer_name: str = "",
    selected_layer_type: str = "",
) -> dict[str, Any]:
    item = normalize_recent_action(action)
    if item.get("action_kind") != "threshold":
        return {}
    parameters = dict(item.get("parameters", {}) or {})
    threshold_value = parameters.get("threshold_value")
    try:
        threshold_value = float(threshold_value)
    except Exception:
        return {}
    mode = str(parameters.get("foreground_mode_resolved") or parameters.get("foreground_mode") or "").strip().lower()
    if mode not in {"bright", "dim"}:
        return {}

    recent_input_layers = list(item.get("input_layers", []) or [])
    recent_image_name = recent_input_layers[0] if recent_input_layers else ""
    target_image_name = _resolve_recent_image_target(
        item,
        selected_layer_name=selected_layer_name,
        selected_layer_type=selected_layer_type,
        prefer_selected=_prefers_selected_image(source),
    )
    if not target_image_name:
        return {}

    if target_image_name != recent_image_name:
        intro = f"Reusing the last threshold settings on [{target_image_name}]"
    else:
        intro = f"Reapplying the last threshold settings to [{target_image_name}]"
    mode_text = "brighter regions" if mode == "bright" else "dimmer regions"
    return {
        "tool": "apply_threshold",
        "arguments": {
            "layer_name": target_image_name,
            "polarity": mode,
            "threshold_value": threshold_value,
        },
        "message": f"{intro} with the same cutoff and {mode_text} as foreground.",
    }


def _same_action(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        str(a.get("tool_name", "")).strip() == str(b.get("tool_name", "")).strip()
        and str(a.get("turn_id", "")).strip() == str(b.get("turn_id", "")).strip()
        and str(a.get("message", "")).strip() == str(b.get("message", "")).strip()
    )
