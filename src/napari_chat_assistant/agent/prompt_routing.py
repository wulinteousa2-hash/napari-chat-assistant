from __future__ import annotations


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def looks_like_multistep_segmentation_workflow(text: str) -> bool:
    source = _normalize_text(text)
    if not source:
        return False
    step_signals = (
        "1.",
        "2.",
        "3.",
        "4.",
        "5.",
        "step 1",
        "step 2",
        "then ",
    )
    workflow_ops = (
        "gaussian",
        "smoothing",
        "threshold",
        "closing",
        "remove tiny",
        "remove small",
        "clear border",
        "border-touch",
        "filter labeled",
        "area/shape",
        "area shape",
    )
    score = sum(1 for op in workflow_ops if op in source)
    return _has_any(source, step_signals) or score >= 3


def route_local_workflow_prompt(text: str, selected_layer_profile: dict | None = None) -> dict | None:
    source = _normalize_text(text)
    if not source:
        return None

    profile = selected_layer_profile if isinstance(selected_layer_profile, dict) else {}
    layer_type = str(profile.get("layer_type", "")).strip().lower()
    layer_name = str(profile.get("layer_name", "")).strip()

    if layer_type and layer_type != "image":
        return None

    em_axon_signals = (
        "axon interior",
        "axon interiors",
        "candidate axon",
        "myelin ring",
        "myelin rings",
        "dark myelin",
        "dark ring",
        "dark rings",
        "enclosed interior",
        "enclosed interiors",
    )
    if not _has_any(source, em_axon_signals):
        return None

    if not looks_like_multistep_segmentation_workflow(source) and "extract" not in source:
        return None

    arguments: dict[str, object] = {}
    if layer_name:
        arguments["image_layer"] = layer_name
    if _has_any(source, ("gaussian", "smoothing", "smooth")):
        arguments["sigma"] = 1.0
    if "closing" in source:
        arguments["closing_radius"] = 2
    if _has_any(source, ("remove tiny", "remove small")):
        arguments["min_area"] = 64
    if _has_any(source, ("clear border", "border-touch")):
        arguments["clear_border"] = True

    return {
        "action": "tool",
        "tool": "extract_axon_interiors",
        "arguments": arguments,
        "message": "Running the built-in axon-interior extraction workflow on the selected EM image.",
    }
