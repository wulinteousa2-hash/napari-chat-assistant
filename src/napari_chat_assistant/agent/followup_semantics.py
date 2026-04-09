from __future__ import annotations

from typing import Any


def empty_followup_constraint() -> dict[str, Any]:
    return {
        "is_followup": False,
        "reuse_previous": False,
        "change_target": "",
        "change_scope": "",
        "avoid_tools": [],
        "prefer_tools": [],
        "modifiers": [],
        "negations": [],
        "target_layer_reference": "",
        "requested_mode": "",
        "confidence": 0.0,
    }


def detect_followup_constraint(text: str) -> bool:
    return bool(parse_followup_constraint(text).get("is_followup"))


def parse_followup_constraint(text: str) -> dict[str, Any]:
    source = " ".join(str(text or "").strip().lower().split())
    result = empty_followup_constraint()
    if not source:
        return result

    reuse_previous = _has_any(
        source,
        (
            "same as before",
            "same workflow",
            "same idea",
            "same as",
            "similar to that",
            "similar to",
            "like before",
            "do that again",
            "do it again",
            "that again",
            "same thing",
            "do that",
        ),
    )
    target_layer_reference = _infer_target_layer_reference(source)
    change_target = _infer_target_layer_type(source)
    change_scope = _infer_scope_change(source)
    avoid_tools = _infer_avoid_tools(source)
    prefer_tools = _infer_prefer_tools(source)
    modifiers = _infer_modifiers(source)
    negations = _infer_negations(source)
    requested_mode = _infer_requested_mode(source)

    is_followup = any(
        (
            reuse_previous,
            bool(target_layer_reference),
            bool(change_target),
            bool(change_scope),
            bool(avoid_tools),
            bool(prefer_tools),
            bool(modifiers),
            bool(negations),
            bool(requested_mode),
        )
    )
    confidence = 0.0
    if is_followup:
        confidence = 0.45
        if reuse_previous:
            confidence += 0.2
        if target_layer_reference or change_target:
            confidence += 0.15
        if negations or avoid_tools:
            confidence += 0.1
        if requested_mode:
            confidence += 0.1

    result.update(
        {
            "is_followup": is_followup,
            "reuse_previous": reuse_previous,
            "change_target": change_target,
            "change_scope": change_scope,
            "avoid_tools": avoid_tools,
            "prefer_tools": prefer_tools,
            "modifiers": modifiers,
            "negations": negations,
            "target_layer_reference": target_layer_reference,
            "requested_mode": requested_mode,
            "confidence": min(0.99, confidence),
        }
    )
    return result


def _infer_target_layer_reference(source: str) -> str:
    if _has_any(source, ("selected one", "selected layer", "selected image", "current selected one", "use the selected one")):
        return "selected_layer"
    if _has_any(source, ("this layer", "this image", "current layer", "current image", "for this layer")):
        return "selected_layer"
    if _has_any(source, ("other layer", "other image", "use the other image", "that other one")):
        return "other_layer"
    if _has_any(source, ("that one", "that layer", "that image", "do that again")):
        return "recent_target"
    return ""


def _infer_target_layer_type(source: str) -> str:
    if _has_any(source, ("labels", "label layer", "label layers", "segmentation")):
        return "labels"
    if _has_any(source, ("shapes", "shape layer", "shape layers", "roi", "rois")):
        return "shapes"
    if _has_any(source, ("points", "point layer", "point layers")):
        return "points"
    if _has_any(source, ("image", "images")):
        return "image"
    return ""


def _infer_scope_change(source: str) -> str:
    if _has_any(source, ("3d instead", "instead 3d", "use 3d", "make it 3d")):
        return "3d"
    if _has_any(source, ("2d instead", "instead 2d", "use 2d", "make it 2d")):
        return "2d"
    if _has_any(source, ("rgb instead", "instead rgb", "use rgb", "make it rgb")):
        return "rgb"
    if "spectral" in source:
        return "spectral"
    return ""


def _infer_avoid_tools(source: str) -> list[str]:
    avoid: list[str] = []
    if _has_any(source, ("don't use widget", "dont use widget", "without widget", "skip widget", "avoid widget")):
        avoid.append("widget")
    if _has_any(source, ("without histogram", "skip histogram", "no histogram", "avoid histogram")):
        avoid.append("histogram")
    if _has_any(source, ("skip preview", "without preview", "no preview", "avoid preview")):
        avoid.append("preview")
    if _has_any(source, ("don't use code", "dont use code", "no code")):
        avoid.append("code")
    return avoid


def _infer_prefer_tools(source: str) -> list[str]:
    prefer: list[str] = []
    if _has_any(source, ("only tool", "tool only", "use tool", "use built-in")):
        prefer.append("tool")
    if _has_any(source, ("just explain", "only explain", "explain only")):
        prefer.append("reply")
    if _has_any(source, ("generate code", "write code", "custom code", "python code")):
        prefer.append("code")
    return prefer


def _infer_modifiers(source: str) -> list[str]:
    modifiers: list[str] = []
    if "faster" in source:
        modifiers.append("faster")
    if "smaller" in source:
        modifiers.append("smaller")
    if "stricter" in source or "more strict" in source:
        modifiers.append("stricter")
    if "looser" in source or "less strict" in source:
        modifiers.append("looser")
    return modifiers


def _infer_negations(source: str) -> list[str]:
    negations: list[str] = []
    if _has_any(source, ("not image-only", "not image only", "my current is not image")):
        negations.append("image_only")
    if _has_any(source, ("don't use", "dont use", "do not use", "without", "skip", "avoid", "leave out", "except")):
        negations.append("explicit_exclusion")
    return negations


def _infer_requested_mode(source: str) -> str:
    if _has_any(source, ("just explain", "only explain", "explain only", "why does this fail")):
        return "reply"
    if _has_any(source, ("generate code", "write code", "custom code", "python code")):
        return "code"
    if _has_any(source, ("tool only", "only tool", "use built-in")):
        return "tool"
    return ""


def _has_any(source: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in source for phrase in phrases)
