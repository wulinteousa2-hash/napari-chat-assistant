from __future__ import annotations

import re

from .workflow_planner import (
    looks_like_conservative_binary_segmentation_request,
    plan_conservative_binary_segmentation,
)


LOCAL_WORKFLOW_PROMPT_LIBRARY = (
    {
        "id": "conservative_masking",
        "keywords": (
            "mask",
            "masking",
            "binary",
            "threshold",
            "segmentation",
            "segment",
            "foreground",
        ),
        "title": "Conservative Binary Masking",
        "prompts": (
            "Build a conservative binary mask for the selected image. Inspect it first, preview threshold before applying it, choose polarity correctly, and preserve faint real structures.",
            "Create a conservative binary mask from the selected image and explain each workflow step before applying it.",
            "Plan a conservative binary segmentation workflow for the selected layer, then run it with minimal cleanup.",
        ),
        "note": "These prompts are phrased to favor the local masking workflow.",
    },
    {
        "id": "image_review",
        "keywords": (
            "review",
            "viewer",
            "grid",
            "tile",
            "scale bar",
            "axes",
            "image review",
        ),
        "title": "Image Review Setup",
        "prompts": (
            "Prepare this viewer for image review: hide labels and points, show only image layers, turn axes labels on, hide the scale bar, tile the images in grid view, and fit the view.",
            "Set up the current viewer for image review with only image layers visible, grid view enabled, and the camera fit to visible data.",
        ),
        "note": "These prompts are phrased to favor safe quick-control workflows.",
    },
    {
        "id": "roi_measurement",
        "keywords": (
            "roi",
            "measure",
            "measurement",
            "intensity",
            "shapes",
            "polygon",
            "statistics",
        ),
        "title": "ROI Measurement",
        "prompts": (
            "Open ROI Intensity Analysis for the current viewer.",
            "Measure ROI intensity statistics for the selected Shapes ROI layer.",
            "Open the ROI measurement widget with histogram and table for the current image.",
        ),
        "note": "These prompts are phrased to favor local ROI measurement tools.",
    },
)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _looks_like_local_prompt_suggestion_request(source: str) -> bool:
    if not source:
        return False
    prompt_terms = (
        "write me a prompt",
        "give me a prompt",
        "make me a prompt",
        "help me write a prompt",
        "suggest a prompt",
        "prompt to trigger",
        "prompt that trigger",
        "prompt that runs",
        "how should i ask",
        "how do i ask",
        "how should i phrase",
        "how do i phrase",
    )
    workflow_terms = (
        "local workflow",
        "workflow",
        "mask",
        "masking",
        "threshold",
        "segment",
        "segmentation",
        "roi",
        "measure",
        "review",
        "viewer",
    )
    return _has_any(source, prompt_terms) and _has_any(source, workflow_terms)


def _match_local_prompt_suggestion(source: str) -> dict | None:
    best_match = None
    best_score = 0
    for record in LOCAL_WORKFLOW_PROMPT_LIBRARY:
        score = sum(1 for keyword in record["keywords"] if keyword in source)
        if score > best_score:
            best_score = score
            best_match = record
    return best_match if best_score > 0 else None


def _local_prompt_suggestion_message(source: str) -> str:
    match = _match_local_prompt_suggestion(source)
    if match is None:
        examples = [
            "Build a conservative binary mask for the selected image.",
            "Open ROI Intensity Analysis for the current viewer.",
            "Prepare this viewer for image review with only image layers visible.",
        ]
        rendered = "\n".join(f"- `{prompt}`" for prompt in examples)
        return (
            "Here are a few prompts phrased to favor local workflows:\n"
            f"{rendered}\n\n"
            "Tell me your goal, such as masking, ROI measurement, or viewer setup, and I can make it more specific."
        )
    rendered = "\n".join(f"- `{prompt}`" for prompt in match["prompts"])
    return (
        f"Try one of these prompts for **{match['title']}**:\n"
        f"{rendered}\n\n"
        f"{match['note']}"
    )


def infer_tool_clarification_request(message: str) -> dict | None:
    source = _normalize_text(message)
    if not source:
        return None
    if not _has_any(source, ("which image layer", "which layer", "what layer")):
        return None

    tool_patterns = (
        (
            ("gaussian denoising", "gaussian smoothing", "gaussian blur", "gaussian filter"),
            {
                "tool": "gaussian_denoise",
                "arguments": {"sigma": 1.0},
                "layer_argument": "layer_name",
                "layer_scope": "image",
                "operation_label": "Gaussian denoising",
            },
        ),
        (
            ("clahe", "adaptive histogram equalization", "local contrast enhancement"),
            {
                "tool": "apply_clahe",
                "arguments": {},
                "layer_argument": "layer_name",
                "layer_scope": "image",
                "operation_label": "CLAHE",
            },
        ),
        (
            ("threshold", "segmentation by threshold", "binary mask"),
            {
                "tool": "preview_threshold",
                "arguments": {"polarity": "auto"},
                "layer_argument": "layer_name",
                "layer_scope": "image",
                "operation_label": "threshold preview",
            },
        ),
    )
    for needles, payload in tool_patterns:
        if _has_any(source, needles):
            result = dict(payload)
            result["options"] = extract_layer_options_from_clarification(message)
            return result
    return None


def extract_layer_options_from_clarification(message: str) -> list[str]:
    text = str(message or "")
    matches = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", text)
    candidates: list[str] = []
    for match in matches:
        clean = str(match).strip("[](){}<>,.:;!?\"'`").strip()
        normalized = clean.lower()
        if not clean or "_" not in clean:
            continue
        if normalized in {"gaussian_denoising", "gaussian_smoothing", "binary_mask"}:
            continue
        if clean not in candidates:
            candidates.append(clean)
    return candidates


def resolve_followup_choice_index(text: str, options: list[str] | tuple[str, ...]) -> str:
    source = _normalize_text(text)
    if not source:
        return ""
    normalized_options = [str(option or "").strip() for option in options if str(option or "").strip()]
    if not normalized_options:
        return ""

    ordinal_pairs = (
        ("third", 2),
        ("three", 2),
        ("second", 1),
        ("two", 1),
        ("first", 0),
        ("one", 0),
        ("fourth", 3),
        ("four", 3),
        ("fifth", 4),
        ("five", 4),
        ("1", 0),
        ("2", 1),
        ("3", 2),
        ("4", 3),
        ("5", 4),
    )
    for token, index in ordinal_pairs:
        if re.search(rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", source) and index < len(normalized_options):
            return normalized_options[index]
    return ""


def resolve_followup_layer_reference(
    text: str,
    *,
    selected_layer_name: str = "",
    available_layer_names: list[str] | tuple[str, ...] = (),
) -> list[str]:
    source = _normalize_text(text)
    if not source:
        return []

    matches: list[str] = []
    selected_clean = str(selected_layer_name or "").strip()
    if selected_clean and _has_any(
        source,
        (
            "selected layer",
            "selected one",
            "current selected",
            "current one",
            "the selected one",
            "the current one",
            "the selected layer",
            "the current selected one",
        ),
    ):
        matches.append(selected_clean)

    normalized_names = {
        " ".join(str(name or "").strip().lower().split()): str(name or "").strip()
        for name in available_layer_names
        if str(name or "").strip()
    }
    for normalized_name, original_name in normalized_names.items():
        if normalized_name and normalized_name in source and original_name not in matches:
            matches.append(original_name)
    return matches


def is_affirmative_followup(text: str) -> bool:
    source = _normalize_text(text)
    if not source:
        return False
    return _has_any(
        source,
        (
            "yes",
            "yeah",
            "yep",
            "ok",
            "okay",
            "oky",
            "oki",
            "go ahead",
            "go",
            "do it",
            "apply it",
            "run it",
            "continue",
            "perform it",
        ),
    )


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


def _looks_like_demo_onboarding_request(text: str) -> bool:
    source = _normalize_text(text)
    if not source:
        return False
    no_image_terms = (
        "no image",
        "no images",
        "dont have any image",
        "don't have any image",
        "dont have image",
        "don't have image",
        "dont have an image",
        "don't have an image",
        "do not have any image",
        "do not have image",
        "do not have an image",
        "without image",
        "without any image",
        "no microscopy image",
        "no microscopy images",
        "i dont bring image",
        "i don't bring image",
        "i have no image",
        "i have no images",
    )
    demo_terms = (
        "how can i test",
        "how do i test",
        "test your ability",
        "test your power",
        "test the plugin",
        "show me a demo",
        "demo",
        "what can i test",
        "what should i do",
        "not sure how can i test",
        "not sure how to test",
    )
    return _has_any(source, no_image_terms) and _has_any(source, demo_terms)


def _demo_onboarding_message() -> str:
    return (
        "No problem. You do not need a real microscopy image to test the plugin.\n\n"
        "The fastest options are:\n"
        "- Use the Library `Templates` tab, open the `Data` category, and double-click a synthetic dataset template to run it immediately.\n"
        "- If you prefer, ask me to generate a synthetic image here in chat, such as a 2D grayscale image, a 3D grayscale volume, or an RGB sample.\n\n"
        "If you are not comfortable with Python, the `Templates` -> `Data` route is the easiest because it creates demo layers for you directly. "
        "You can also load a template first to inspect the code before running it.\n\n"
        "Once the demo image is loaded, I can show ROI measurement, histogram plotting, thresholding, segmentation, and widget-based analysis."
    )


def _looks_like_getting_started_request(text: str) -> bool:
    source = _normalize_text(text)
    if not source:
        return False
    start_terms = (
        "how do i start",
        "how should i start",
        "where do i start",
        "how can i start",
        "how do i begin",
        "where do i begin",
        "getting started",
        "get started",
        "what should i do first",
        "how to start",
    )
    return _has_any(source, start_terms)


def _getting_started_message(has_selected_layer: bool) -> str:
    if has_selected_layer:
        return (
            "Start with a simple action on the current viewer. A good first step is to inspect the selected layer, "
            "plot a histogram, try denoising, preview a threshold, or open a measurement widget depending on what you want to learn.\n\n"
            "If you want, tell me your goal and I can suggest the best first operation."
        )
    return (
        "The easiest start is to load or generate a test image, then try one simple workflow on it.\n\n"
        "You can use the Library `Templates` tab, open the `Data` category, and run a synthetic dataset, or I can generate a synthetic 2D grayscale, 3D grayscale, 2D RGB, or 3D RGB demo image here in chat.\n\n"
        "Once an image is loaded, good first things to try are histogram plotting, denoising, threshold preview, ROI measurement, or opening an analysis widget."
    )


def _infer_synthetic_demo_variant(text: str) -> str:
    source = _normalize_text(text)
    if not source:
        return ""
    if _has_any(source, ("3d rgb", "rgb 3d", "3-dimensional rgb", "3d color", "3d colour")):
        return "3d_rgb"
    if _has_any(source, ("2d rgb", "rgb 2d", "2-dimensional rgb", "2d color", "2d colour", "rgb image")):
        return "2d_rgb"
    if _has_any(source, ("3d grayscale", "3d gray", "3d grey", "grayscale 3d", "gray 3d", "grey 3d", "3d volume")):
        return "3d_gray"
    if _has_any(source, ("2d grayscale", "2d gray", "2d grey", "grayscale 2d", "gray 2d", "grey 2d")):
        return "2d_gray"
    if source in {"rgb", "color", "colour"}:
        return "2d_rgb"
    if source in {"3d", "3d grayscale", "3d gray", "3d grey"}:
        return "3d_gray"
    if source in {"2d", "2d grayscale", "2d gray", "2d grey", "grayscale", "gray", "grey"}:
        return "2d_gray"
    return ""


def _looks_like_synthetic_demo_request(text: str) -> bool:
    source = _normalize_text(text)
    if not source:
        return False
    if _has_any(source, ("synthetic image", "synthetic demo", "demo image", "test image", "sample image")):
        return True
    creation_terms = ("create", "generate", "make", "add")
    variant = _infer_synthetic_demo_variant(source)
    if variant and _has_any(source, creation_terms):
        return True
    if variant and len(source.split()) <= 4:
        return True
    return False


def _synthetic_demo_choice_message() -> str:
    return "Sure. I can generate a synthetic 2D grayscale image, 3D grayscale volume, 2D RGB image, or 3D RGB volume. Which one would you like?"


def _looks_like_image_review_setup_request(source: str) -> bool:
    if not _has_any(source, ("prepare", "set up", "setup", "configure", "make this viewer ready")):
        return False
    if not _has_any(source, ("image review", "review images", "image inspection", "viewer for image", "images for review")):
        return False
    return _has_any(source, ("hide labels", "hide points", "show only image", "image layers", "grid view", "tile"))


def _image_review_setup_steps(source: str) -> list[dict]:
    steps: list[dict] = []
    if _has_any(source, ("hide labels", "hide label", "without labels")):
        steps.append({"tool": "hide_layers_by_type", "arguments": {"layer_type": "labels"}, "on_error": "skip"})
    if _has_any(source, ("hide points", "hide point", "hide labels and points", "hide labels and point", "without points")):
        steps.append({"tool": "hide_layers_by_type", "arguments": {"layer_type": "points"}, "on_error": "skip"})
    if _has_any(source, ("show only image", "only image layers", "keep only image", "show image layers")):
        steps.append({"tool": "show_only_layer_type", "arguments": {"layer_type": "image"}, "on_error": "stop"})
    if _has_any(source, ("axes labels on", "turn axes labels on", "show axes labels", "axis labels on")):
        steps.append({"tool": "set_axes_labels", "arguments": {"enabled": True}, "on_error": "skip"})
    if _has_any(source, ("hide scale bar", "hide the scale bar", "scale bar off", "turn the scale bar off")):
        steps.append({"tool": "set_scale_bar_visible", "arguments": {"visible": False}, "on_error": "skip"})
    elif _has_any(source, ("show scale bar", "scale bar on", "turn the scale bar on")):
        steps.append({"tool": "set_scale_bar_visible", "arguments": {"visible": True}, "on_error": "skip"})
    if _has_any(source, ("grid view", "tile", "tiled", "side by side", "side-by-side")):
        steps.append({"tool": "show_image_layers_in_grid", "arguments": {"spacing": 2}, "on_error": "skip"})
    if _has_any(source, ("fit view", "fit the view", "fit viewer", "fit visible", "center")):
        steps.append({"tool": "fit_view", "arguments": {}, "on_error": "skip"})
    if not steps:
        steps = [
            {"tool": "hide_layers_by_type", "arguments": {"layer_type": "labels"}, "on_error": "skip"},
            {"tool": "hide_layers_by_type", "arguments": {"layer_type": "points"}, "on_error": "skip"},
            {"tool": "show_only_layer_type", "arguments": {"layer_type": "image"}, "on_error": "stop"},
            {"tool": "set_axes_labels", "arguments": {"enabled": True}, "on_error": "skip"},
            {"tool": "set_scale_bar_visible", "arguments": {"visible": False}, "on_error": "skip"},
            {"tool": "show_image_layers_in_grid", "arguments": {"spacing": 2}, "on_error": "skip"},
            {"tool": "fit_view", "arguments": {}, "on_error": "skip"},
        ]
    return steps


def _quick_control_bool_from_clause(clause: str, *, default: bool = True) -> bool:
    if _has_any(clause, (" off", " hide", " hidden", " disable", " disabled", " without ")):
        return False
    if _has_any(clause, (" on", " show", " visible", " enable", " enabled", " turn ")):
        return True
    return default


def _quick_control_workflow_clauses(source: str) -> list[str]:
    text = str(source or "").strip()
    if not text:
        return []

    numbered = list(re.finditer(r"(?:^|\s)(?:\d+[\.\)]|step\s+\d+[\.\):\-]?)\s*", text))
    if numbered:
        clauses: list[str] = []
        for index, match in enumerate(numbered):
            start = match.end()
            end = numbered[index + 1].start() if index + 1 < len(numbered) else len(text)
            clause = text[start:end].strip(" ,.;:-")
            if clause:
                clauses.append(clause)
        return clauses

    separators = r"\s*(?:,|;|\bthen\b|\band then\b)\s+"
    return [clause.strip(" ,.;:-") for clause in re.split(separators, text) if clause.strip(" ,.;:-")]


def _quick_control_step_from_clause(clause: str) -> dict | None:
    source = _normalize_text(clause)
    if not source:
        return None

    if _has_any(source, ("show only image layers", "only image layers", "keep only image layers", "show only images")):
        return {"tool": "show_only_layer_type", "arguments": {"layer_type": "image"}, "on_error": "stop"}
    if _has_any(source, ("show only labels layers", "only labels layers", "keep only labels layers")):
        return {"tool": "show_only_layer_type", "arguments": {"layer_type": "labels"}, "on_error": "stop"}
    if _has_any(source, ("show only points layers", "only points layers", "keep only points layers")):
        return {"tool": "show_only_layer_type", "arguments": {"layer_type": "points"}, "on_error": "stop"}
    if _has_any(source, ("show only shapes layers", "only shapes layers", "keep only shapes layers")):
        return {"tool": "show_only_layer_type", "arguments": {"layer_type": "shapes"}, "on_error": "stop"}

    if _has_any(source, ("hide labels layers", "hide label layers", "hide labels", "without labels")):
        return {"tool": "hide_layers_by_type", "arguments": {"layer_type": "labels"}, "on_error": "skip"}
    if _has_any(source, ("show labels layers", "show label layers", "show labels")):
        return {"tool": "show_layers_by_type", "arguments": {"layer_type": "labels"}, "on_error": "skip"}
    if _has_any(source, ("hide points layers", "hide point layers", "hide points", "without points")):
        return {"tool": "hide_layers_by_type", "arguments": {"layer_type": "points"}, "on_error": "skip"}
    if _has_any(source, ("show points layers", "show point layers", "show points")):
        return {"tool": "show_layers_by_type", "arguments": {"layer_type": "points"}, "on_error": "skip"}

    if _has_any(
        source,
        (
            "fit view",
            "fit the view",
            "fit viewer",
            "fit visible",
            "fit current visible",
            "fit the current visible",
            "fit layers in view",
            "fit the layers in view",
            "center view",
        ),
    ):
        return {"tool": "fit_view", "arguments": {}, "on_error": "skip"}
    if _has_any(source, ("zoom in", "camera in")):
        return {"tool": "zoom_in_view", "arguments": {}, "on_error": "skip"}
    if _has_any(source, ("zoom out", "camera out")):
        return {"tool": "zoom_out_view", "arguments": {}, "on_error": "skip"}
    if _has_any(source, ("toggle 2d", "toggle 3d", "switch 2d", "switch 3d", "2d and 3d", "2d/3d")):
        return {"tool": "toggle_2d_3d_camera", "arguments": {}, "on_error": "skip"}

    if _has_any(source, ("grid view", "tile images", "tile the images", "tiled view", "side by side", "side-by-side")):
        if _quick_control_bool_from_clause(source, default=True):
            return {"tool": "show_image_layers_in_grid", "arguments": {"spacing": 2}, "on_error": "skip"}
        return {"tool": "hide_image_grid_view", "arguments": {}, "on_error": "skip"}

    if "tooltip" in source:
        return {"tool": "set_layer_tooltips_visible", "arguments": {"visible": _quick_control_bool_from_clause(source)}, "on_error": "skip"}

    if "axes labels" in source or "axis labels" in source:
        return {"tool": "set_axes_labels", "arguments": {"enabled": _quick_control_bool_from_clause(source)}, "on_error": "skip"}
    if "axes arrows" in source or "axis arrows" in source:
        return {"tool": "set_axes_arrows", "arguments": {"enabled": _quick_control_bool_from_clause(source)}, "on_error": "skip"}
    if "colored axes" in source or "colour axes" in source or "axes colored" in source or "axes colour" in source:
        return {"tool": "set_axes_colored", "arguments": {"enabled": _quick_control_bool_from_clause(source)}, "on_error": "skip"}
    if "dashed axes" in source or "axes dashed" in source:
        return {"tool": "set_axes_dashed", "arguments": {"enabled": _quick_control_bool_from_clause(source)}, "on_error": "skip"}
    if "axes" in source or "axis" in source:
        return {"tool": "set_axes_visible", "arguments": {"visible": _quick_control_bool_from_clause(source)}, "on_error": "skip"}

    if "scale bar box" in source:
        return {"tool": "set_scale_bar_box", "arguments": {"enabled": _quick_control_bool_from_clause(source)}, "on_error": "skip"}
    if "scale bar ticks" in source:
        return {"tool": "set_scale_bar_ticks", "arguments": {"enabled": _quick_control_bool_from_clause(source)}, "on_error": "skip"}
    if "colored scale bar" in source or "colour scale bar" in source or "scale bar colored" in source or "scale bar colour" in source:
        return {"tool": "set_scale_bar_colored", "arguments": {"enabled": _quick_control_bool_from_clause(source)}, "on_error": "skip"}
    if "scale bar" in source:
        return {"tool": "set_scale_bar_visible", "arguments": {"visible": _quick_control_bool_from_clause(source)}, "on_error": "skip"}

    if "bounding box" in source:
        return {"tool": "set_selected_layer_bounding_box_visible", "arguments": {"visible": _quick_control_bool_from_clause(source)}, "on_error": "skip"}
    if "name overlay" in source or "layer name" in source:
        return {"tool": "set_selected_layer_name_overlay_visible", "arguments": {"visible": _quick_control_bool_from_clause(source)}, "on_error": "skip"}

    return None


def _quick_control_sequence_steps(source: str) -> list[dict]:
    steps: list[dict] = []
    for clause in _quick_control_workflow_clauses(source):
        step = _quick_control_step_from_clause(clause)
        if step is not None:
            steps.append(step)
    return steps


def _looks_like_restore_last_workflow_request(source: str) -> bool:
    return _has_any(
        source,
        (
            "undo last workflow",
            "undo the last workflow",
            "undo quick workflow",
            "restore before workflow",
            "restore viewer before workflow",
            "reset quick controls",
            "restore quick controls",
        ),
    )


def _workflow_report_mode_request(source: str) -> str:
    if _has_any(source, ("show debug", "show full debug", "debug workflow", "full debug trace")):
        return "debug"
    if _has_any(source, ("show details", "show full steps", "show step details", "show workflow details")):
        return "details"
    if _has_any(source, ("show plan", "show workflow plan", "what was the plan", "full plan")):
        return "plan"
    if _has_any(source, ("show summary", "compact summary", "short summary")):
        return "compact"
    return ""


def _replace_label_value_request(source: str) -> dict | None:
    if not source:
        return None
    if not _has_any(source, ("label", "labels", "mask", "foreground")):
        return None

    patterns = (
        r"(?:change|replace|relabel|set)\s+(?:the\s+)?(?:label\s+)?value\s+(\d+)\s+(?:to|with|as)\s+(\d+)",
        r"(?:replace|relabel)\s+(?:label\s+)?(\d+)\s+(?:with|to|as)\s+(\d+)",
        r"(?:change|replace|relabel|set)\s+(?:the\s+)?(?:selected\s+)?(?:mask\s+)?(?:foreground\s+)?(?:label\s+)?(?:value\s+)?from\s+(\d+)\s+(?:to|with|as)\s+(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, source)
        if match:
            return {"source_value": int(match.group(1)), "target_value": int(match.group(2))}

    default_source_patterns = (
        r"(?:change|replace|relabel|set)\s+(?:the\s+)?(?:selected\s+)?(?:current\s+)?(?:mask\s+)?foreground(?:\s+label)?(?:\s+value)?\s+(?:to|as)\s+(\d+)",
        r"(?:change|replace|relabel|set)\s+(?:the\s+)?(?:selected\s+)?(?:current\s+)?(?:mask\s+)?(?:label\s+)?value\s+(?:to|as)\s+(\d+)",
        r"(?:change|replace|relabel|set)\s+(?:the\s+)?current value\s+(?:to|as)\s+(\d+)",
    )
    for pattern in default_source_patterns:
        match = re.search(pattern, source)
        if match:
            return {"source_value": 1, "target_value": int(match.group(1))}
    return None


def route_local_workflow_prompt(text: str, selected_layer_profile: dict | None = None) -> dict | None:
    source = _normalize_text(text)
    if not source:
        return None

    profile = selected_layer_profile if isinstance(selected_layer_profile, dict) else {}
    layer_type = str(profile.get("layer_type", "")).strip().lower()
    layer_name = str(profile.get("layer_name", "")).strip()
    semantic_type = str(profile.get("semantic_type", "")).strip().lower()
    has_selected_layer = bool(layer_type or layer_name or semantic_type)

    if not has_selected_layer and _looks_like_demo_onboarding_request(source):
        return {
            "action": "reply",
            "message": _demo_onboarding_message(),
        }

    if _looks_like_getting_started_request(source):
        return {
            "action": "reply",
            "message": _getting_started_message(has_selected_layer),
        }

    if _looks_like_local_prompt_suggestion_request(source):
        return {
            "action": "reply",
            "message": _local_prompt_suggestion_message(source),
        }

    if _looks_like_restore_last_workflow_request(source):
        return {
            "action": "restore_tool_sequence",
            "message": "Restoring viewer controls from before the last quick-control workflow.",
        }

    workflow_report_mode = _workflow_report_mode_request(source)
    if workflow_report_mode:
        return {
            "action": "workflow_report",
            "mode": workflow_report_mode,
            "message": f"Showing the last workflow {workflow_report_mode} view.",
        }

    if layer_type == "labels":
        replace_request = _replace_label_value_request(source)
        if replace_request is not None:
            arguments = dict(replace_request)
            if layer_name:
                arguments["layer_name"] = layer_name
            return {
                "action": "tool",
                "tool": "replace_label_value",
                "arguments": arguments,
                "message": (
                    f"Replacing label value {arguments['source_value']} with {arguments['target_value']} "
                    "in the selected labels layer."
                ),
            }

    if (not layer_type or layer_type == "image") and looks_like_conservative_binary_segmentation_request(source):
        plan = plan_conservative_binary_segmentation(text, selected_layer_profile=profile)
        if plan is not None:
            return {
                "action": "workflow_execute",
                "plan": plan.to_dict(),
                "message": "Running a conservative binary segmentation workflow planned from your prompt.",
            }

    if _looks_like_image_review_setup_request(source):
        return {
            "action": "tool_sequence",
            "steps": _image_review_setup_steps(source),
            "message": "Preparing the viewer for image review with a safe quick-control workflow.",
        }

    quick_control_steps = _quick_control_sequence_steps(source)
    if len(quick_control_steps) >= 2:
        return {
            "action": "tool_sequence",
            "steps": quick_control_steps,
            "message": "Running a safe quick-control workflow in the order you requested.",
        }

    if _looks_like_synthetic_demo_request(source):
        variant = _infer_synthetic_demo_variant(source)
        if not variant:
            return {
                "action": "reply",
                "message": _synthetic_demo_choice_message(),
            }
        return {
            "action": "tool",
            "tool": "create_synthetic_demo_image",
            "arguments": {"variant": variant},
            "message": f"Creating a synthetic {variant.replace('_', ' ')} demo image.",
        }

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
                "message": "Opening the ROI Intensity Analysis widget for interactive ROI measurement with a live histogram, table, and absolute versus normalized views.",
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
