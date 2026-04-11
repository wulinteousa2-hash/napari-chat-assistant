from __future__ import annotations

import numpy as np

from napari_chat_assistant.agent.workflow_executor import execute_workflow_plan
from napari_chat_assistant.agent.workflow_planner import plan_conservative_binary_segmentation


SEGMENTATION_PROMPT = """
Inspect the selected image first and describe its likely signal, background, and noise pattern.
Then build the cleanest possible binary mask step by step for the main foreground objects using a conservative workflow:
1. Apply light denoising only if needed, and explain why.
2. Preview threshold first instead of applying it immediately.
3. Choose the threshold polarity correctly for bright or dim foreground.
4. After thresholding, clean the mask with the minimum necessary morphology.
5. Measure or summarize the mask quality after each major step.
6. If the result is too loose or too strict, refine it iteratively.
7. Stop when the mask is clean, biologically plausible, and low noise.
Prefer built-in tools over code when possible.
At each step, tell me what you are changing and why.
Do not destroy faint real structures just to make the mask look cleaner.
"""


def test_execute_workflow_plan_runs_conservative_binary_segmentation(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = np.zeros((64, 64), dtype=np.float32)
    image[16:28, 18:30] = 1.0
    image[36:50, 34:48] = 0.85
    image += 0.08 * np.random.default_rng(3).normal(size=image.shape).astype(np.float32)
    viewer.add_image(image, name="image_a")

    plan = plan_conservative_binary_segmentation(
        SEGMENTATION_PROMPT,
        selected_layer_profile={"layer_type": "image", "layer_name": "image_a", "semantic_type": "2d_intensity"},
    )

    result = execute_workflow_plan(viewer, plan)

    assert result["ok"] is True
    assert result["workflow_type"] == "conservative_binary_segmentation"
    assert "__assistant_threshold_preview__" in viewer.layers
    assert result["final_mask_layer"] in viewer.layers
    assert "Inspection on [image_a]" in result["message"]
    assert "Stopped with [" in result["message"]


def test_execute_workflow_plan_rejects_unknown_workflow(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((8, 8), dtype=np.float32), name="image_a")

    result = execute_workflow_plan(viewer, {"workflow_type": "unknown_workflow"})

    assert result["ok"] is False
    assert "Unsupported workflow type" in result["message"]


def test_execute_workflow_plan_records_refinement_cycles(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = np.zeros((64, 64), dtype=np.float32)
    image[8:56, 8:56] = 0.35
    image[20:44, 20:44] = 0.75
    viewer.add_image(image, name="image_a")

    plan = plan_conservative_binary_segmentation(
        SEGMENTATION_PROMPT,
        selected_layer_profile={"layer_type": "image", "layer_name": "image_a", "semantic_type": "2d_intensity"},
    )

    result = execute_workflow_plan(viewer, plan)

    assert result["ok"] is True
    assert result["cycle_count"] >= 1
    assert any(step.get("id", "").startswith("judge_mask_quality_cycle_") for step in result["executed_steps"])
