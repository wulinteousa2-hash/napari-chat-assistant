from __future__ import annotations

from typing import Any

from .followup_semantics import parse_followup_constraint


def empty_intent_state() -> dict[str, Any]:
    return {
        "goal": "",
        "mode_preference": "",
        "blocked_tools": [],
        "target_layer_types": [],
        "negative_constraints": [],
        "reason": "",
        "carry_forward": False,
    }


def empty_failed_tool_state() -> dict[str, Any]:
    return {
        "tool_name": "",
        "reason": "",
        "supported_layer_types": [],
    }


def extract_turn_intent(text: str, *, last_failed_tool_state: dict[str, Any] | None = None) -> dict[str, Any]:
    source = " ".join(str(text or "").strip().lower().split())
    state = empty_intent_state()
    if not source:
        return state

    followup = parse_followup_constraint(text)
    state["goal"] = str(text or "").strip()[:240]
    state["mode_preference"] = str(followup.get("requested_mode", "")).strip() or _infer_mode_preference(source)
    state["target_layer_types"] = _merge_unique(
        _infer_target_layer_types(source),
        [str(followup.get("change_target", "")).strip()] if str(followup.get("change_target", "")).strip() else [],
    )
    state["negative_constraints"] = _merge_unique(
        _infer_negative_constraints(source),
        [str(item).strip() for item in followup.get("negations", []) if str(item).strip()],
    )
    # Do not carry forward prior intent just because a fresh request mentions
    # the selected/current image or a target layer type. Reserve carry-forward
    # for actual follow-up language that reuses or edits a prior plan.
    state["carry_forward"] = bool(followup.get("reuse_previous")) or _looks_like_followup_constraint(source)

    failed_tool = dict(last_failed_tool_state or {})
    failed_tool_name = str(failed_tool.get("tool_name", "")).strip()
    failed_reason = str(failed_tool.get("reason", "")).strip()

    if failed_reason:
        state["reason"] = failed_reason

    if failed_tool_name and _should_block_failed_tool(source, failed_tool_name, state):
        state["blocked_tools"] = [failed_tool_name]
        if not state["reason"]:
            state["reason"] = failed_reason or f"Previous tool [{failed_tool_name}] did not fit the request."

    return state


def merge_intent_state(previous: dict[str, Any] | None, current: dict[str, Any] | None) -> dict[str, Any]:
    prior = dict(previous or empty_intent_state())
    new = dict(current or empty_intent_state())
    merged = empty_intent_state()

    carry_forward = bool(new.get("carry_forward"))
    for key in ("goal", "mode_preference", "reason"):
        merged[key] = str(new.get(key) or "").strip() or (str(prior.get(key) or "").strip() if carry_forward else "")

    for key in ("blocked_tools", "target_layer_types", "negative_constraints"):
        current_values = [str(value).strip() for value in new.get(key, []) if str(value).strip()]
        prior_values = [str(value).strip() for value in prior.get(key, []) if str(value).strip()] if carry_forward else []
        merged[key] = _merge_unique(prior_values, current_values)

    merged["carry_forward"] = carry_forward
    return merged


def should_skip_local_workflow_route(intent_state: dict[str, Any] | None) -> bool:
    state = dict(intent_state or {})
    mode = str(state.get("mode_preference", "")).strip().lower()
    negatives = {str(value).strip().lower() for value in state.get("negative_constraints", []) if str(value).strip()}
    return mode in {"code", "explain"} or "no_builtin_tools" in negatives


def should_block_tool(intent_state: dict[str, Any] | None, tool_name: str) -> bool:
    token = str(tool_name or "").strip()
    if not token:
        return False
    state = dict(intent_state or {})
    blocked = {str(value).strip() for value in state.get("blocked_tools", []) if str(value).strip()}
    return token in blocked


def remember_failed_tool(tool_name: str, message: str) -> dict[str, Any]:
    token = str(tool_name or "").strip()
    text = str(message or "").strip()
    state = empty_failed_tool_state()
    if not token or not text:
        return state
    state["tool_name"] = token
    state["reason"] = text
    lowered = text.lower()
    if "grayscale 2d image layers" in lowered or "usable 2d image layers" in lowered:
        state["supported_layer_types"] = ["image"]
    return state


def _infer_mode_preference(source: str) -> str:
    code_signals = (
        "generate code",
        "write code",
        "give me code",
        "return code",
        "custom code",
        "python code",
    )
    explain_signals = ("why does this fail", "why did this fail", "why", "explain")
    tool_block_signals = ("don't use built-in", "dont use built-in", "not the built-in", "not image-only")
    if any(signal in source for signal in code_signals):
        return "code"
    if any(signal in source for signal in explain_signals) and not any(signal in source for signal in code_signals):
        return "explain"
    if any(signal in source for signal in tool_block_signals):
        return "code"
    return ""


def _infer_target_layer_types(source: str) -> list[str]:
    pairs = (
        ("image", ("image", "images", "grayscale image", "rgb image")),
        ("labels", ("labels", "label layer", "label layers", "segmentation")),
        ("shapes", ("shapes", "shape layer", "shape layers", "roi", "rois")),
        ("points", ("points", "point layer", "point layers")),
    )
    found: list[str] = []
    for label, signals in pairs:
        if any(signal in source for signal in signals):
            found.append(label)
    return found


def _infer_negative_constraints(source: str) -> list[str]:
    constraints: list[str] = []
    if any(signal in source for signal in ("not image-only", "not image only", "my current is not image")):
        constraints.append("not_image_only")
    if any(signal in source for signal in ("don't use built-in", "dont use built-in", "not the built-in", "not use built in")):
        constraints.append("no_builtin_tools")
    return constraints


def _looks_like_followup_constraint(source: str) -> bool:
    return any(
        signal in source
        for signal in (
            "similar to that",
            "similar to",
            "same as",
            "same workflow",
            "but for",
            "instead",
            "not image-only",
            "dont use",
            "don't use",
        )
    )


def _should_block_failed_tool(source: str, failed_tool_name: str, state: dict[str, Any]) -> bool:
    if str(state.get("mode_preference", "")).strip().lower() == "code":
        return True
    targets = {str(value).strip().lower() for value in state.get("target_layer_types", []) if str(value).strip()}
    negatives = {str(value).strip().lower() for value in state.get("negative_constraints", []) if str(value).strip()}
    if failed_tool_name == "create_analysis_montage":
        if targets.intersection({"labels", "shapes", "points"}):
            return True
        if "not_image_only" in negatives or "no_builtin_tools" in negatives:
            return True
    return False


def _merge_unique(first: list[str], second: list[str]) -> list[str]:
    merged: list[str] = []
    for value in first + second:
        token = str(value).strip()
        if token and token not in merged:
            merged.append(token)
    return merged
