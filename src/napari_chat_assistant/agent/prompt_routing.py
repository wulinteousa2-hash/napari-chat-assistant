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


def _looks_like_roi_measurement_request(text: str) -> bool:
    source = _normalize_text(text)
    if not source:
        return False
    roi_terms = (
        "roi",
        "shape",
        "shapes",
        "polygon",
        "rectangle",
        "ellipse",
        "circle",
        "path",
        "line",
    )
    measurement_terms = (
        "measure",
        "measurement",
        "stat",
        "stats",
        "statistics",
        "quantify",
        "quantification",
    )
    return _has_any(source, roi_terms) and _has_any(source, measurement_terms)


def _prefers_interactive_roi_widget(text: str) -> bool:
    source = _normalize_text(text)
    if not source:
        return False
    text_only_terms = (
        "just tell",
        "just give",
        "print",
        "summary",
        "summarize",
        "in chat",
        "text only",
        "without widget",
        "no widget",
    )
    if _has_any(source, text_only_terms):
        return False
    widget_terms = (
        "widget",
        "table",
        "interactive",
        "popup",
        "pop up",
        "open",
        "histogram",
        "csv",
        "export",
        "rename",
        "percent",
        "normalized",
    )
    return _has_any(source, widget_terms) or True


def _looks_like_group_comparison_request(text: str) -> bool:
    source = _normalize_text(text)
    if not source:
        return False
    group_terms = ("group a", "group b", "wt", "mutant", "compare two groups", "3 vs 3", "versus", "vs")
    stats_terms = ("t-test", "welch", "mann-whitney", "normality", "variance", "comparison", "descriptive stats")
    return _has_any(source, group_terms) and _has_any(source, stats_terms)


def _prefers_interactive_group_widget(text: str) -> bool:
    source = _normalize_text(text)
    if not source:
        return False
    interactive_terms = (
        "widget",
        "interactive",
        "table",
        "plot",
        "box plot",
        "bar plot",
        "bar chart",
        "popup",
        "export",
    )
    return _has_any(source, interactive_terms)


def route_local_workflow_prompt(text: str, selected_layer_profile: dict | None = None) -> dict | None:
    source = _normalize_text(text)
    if not source:
        return None

    profile = selected_layer_profile if isinstance(selected_layer_profile, dict) else {}
    layer_type = str(profile.get("layer_type", "")).strip().lower()
    layer_name = str(profile.get("layer_name", "")).strip()
    semantic_type = str(profile.get("semantic_type", "")).strip().lower()

    if _looks_like_roi_measurement_request(source):
        roi_layer_name = layer_name if layer_type == "shapes" or semantic_type == "roi_shapes" else ""
        if _prefers_interactive_roi_widget(source):
            arguments: dict[str, object] = {}
            if layer_type == "image" and layer_name:
                arguments["layer_name"] = layer_name
            return {
                "action": "tool",
                "tool": "open_intensity_metrics_table",
                "arguments": arguments,
                "message": "Opening the ROI Intensity Metrics widget for interactive ROI measurement with a live histogram, table, and absolute versus normalized views.",
            }
        arguments = {}
        if roi_layer_name:
            arguments["roi_layer"] = roi_layer_name
        return {
            "action": "tool",
            "tool": "measure_shapes_roi_stats",
            "arguments": arguments,
            "message": "Measuring ROI shape statistics for the selected Shapes layer.",
        }

    if _looks_like_group_comparison_request(source) and _prefers_interactive_group_widget(source):
        return {
            "action": "tool",
            "tool": "open_group_comparison_widget",
            "arguments": {},
            "message": "Opening the Group Comparison Stats widget with per-sample table, descriptive statistics, and a switchable box-versus-bar plot view.",
        }

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
