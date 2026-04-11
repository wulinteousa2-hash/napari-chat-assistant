from __future__ import annotations

from napari_chat_assistant.agent.workflow_planner import (
    looks_like_conservative_binary_segmentation_request,
    plan_conservative_binary_segmentation,
    workflow_plan_to_markdown,
)


SEGMENTATION_PROMPT = """
Inspect the selected image first and describe its likely signal, background, and noise pattern.

Then build the cleanest possible binary mask step by step for the main foreground objects using a conservative workflow:

1. Apply light denoising only if needed, and explain why.
2. Preview threshold first instead of applying it immediately.
3. Choose the threshold polarity correctly for bright or dim foreground.
4. After thresholding, clean the mask with the minimum necessary morphology:
   - remove tiny specks
   - fill small holes
   - smooth jagged edges
   - avoid over-merging nearby objects
5. Measure or summarize the mask quality after each major step.
6. If the result is too loose or too strict, refine it iteratively.
7. Stop when the mask is clean, biologically plausible, and low noise.

Prefer built-in tools over code when possible.
At each step, tell me what you are changing and why.
Do not destroy faint real structures just to make the mask look cleaner.
"""

SHORT_SEGMENTATION_PROMPT = """
Build a conservative binary mask for the selected image. Inspect it first, preview threshold before applying it, decide whether the
objects are brighter or dimmer than the background, clean the mask minimally, measure quality after each major step, and preserve
faint real structures.
"""


def test_detects_conservative_binary_segmentation_request():
    assert looks_like_conservative_binary_segmentation_request(SEGMENTATION_PROMPT) is True
    assert looks_like_conservative_binary_segmentation_request(SHORT_SEGMENTATION_PROMPT) is True


def test_plans_conservative_binary_segmentation_workflow():
    plan = plan_conservative_binary_segmentation(
        SEGMENTATION_PROMPT,
        selected_layer_profile={"layer_type": "image", "layer_name": "image_a", "semantic_type": "2d_intensity"},
    )

    assert plan is not None
    payload = plan.to_dict()
    assert payload["workflow_type"] == "conservative_binary_segmentation"
    assert payload["target_layer"] == "image_a"
    assert any(item["id"] == "prefer_builtin_tools" for item in payload["constraints"])
    assert any(step["kind"] == "analysis" for step in payload["steps"])
    assert any(step["kind"] == "decision" for step in payload["steps"])
    assert any(step["kind"] == "stop_check" for step in payload["steps"])
    tool_names = [step["tool"] for step in payload["steps"] if step["kind"] == "tool"]
    assert "gaussian_denoise" in tool_names
    assert "preview_threshold" in tool_names
    assert "apply_threshold" in tool_names
    assert "measure_mask" in tool_names
    assert "remove_small_objects" in tool_names
    assert "fill_mask_holes" in tool_names
    assert "run_mask_op" in tool_names


def test_plans_short_conservative_binary_segmentation_prompt():
    plan = plan_conservative_binary_segmentation(
        SHORT_SEGMENTATION_PROMPT,
        selected_layer_profile={"layer_type": "image", "layer_name": "image_a", "semantic_type": "2d_intensity"},
    )

    assert plan is not None
    payload = plan.to_dict()
    assert payload["workflow_type"] == "conservative_binary_segmentation"
    assert any(
        "brighter than the background" in step["condition"] or "dimmer than the background" in step["condition"]
        for step in payload["steps"]
        if step["kind"] == "decision"
    )


def test_workflow_plan_to_markdown_renders_useful_summary():
    plan = plan_conservative_binary_segmentation(
        SEGMENTATION_PROMPT,
        selected_layer_profile={"layer_type": "image", "layer_name": "image_a"},
    )

    rendered = workflow_plan_to_markdown(plan)

    assert "Planned workflow: `conservative_binary_segmentation`" in rendered
    assert "Target layer: [image_a]" in rendered
    assert "using `preview_threshold`" in rendered
    assert "Stop when:" in rendered
