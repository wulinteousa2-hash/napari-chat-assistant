from __future__ import annotations

from napari_chat_assistant.agent.workflow_executor import (
    workflow_execution_to_compact_markdown,
    workflow_execution_to_debug_markdown,
)
from napari_chat_assistant.agent.workflow_planner import workflow_plan_to_markdown


def test_workflow_execution_to_compact_markdown_is_shorter_than_debug():
    result = {
        "final_mask_layer": "image_a_labels_clean",
        "message": "1. Long debug line\n2. Another debug line\n3. Final debug line",
        "executed_steps": [
            {"id": "inspect_selected_image", "message": "Inspection on [image_a]: likely bright foreground signal."},
            {"id": "preview_threshold", "message": "Preview threshold updated for [image_a] at 0.25."},
            {"id": "apply_threshold", "message": "Applied threshold to [image_a] as [image_a_labels]."},
            {"id": "remove_tiny_specks_cycle_1", "message": "Removed small objects from [image_a_labels] into [image_a_labels_clean]."},
            {"id": "stop_when_conservative_quality_is_met", "message": "Stopped with [image_a_labels_clean] because the mask is in a conservative operating range."},
        ],
    }

    compact = workflow_execution_to_compact_markdown(result)
    debug = workflow_execution_to_debug_markdown(result)

    assert "Built a conservative mask for [image_a_labels_clean]." in compact
    assert "Summary:" in compact
    assert "show details" in compact
    assert "1. Long debug line" in debug
    assert "Long debug line" not in compact


def test_workflow_plan_to_markdown_still_renders_full_plan():
    plan = {
        "workflow_type": "conservative_binary_segmentation",
        "intent": "Inspect the selected image and build a conservative binary foreground mask.",
        "target_layer": "image_a",
        "constraints": [{"description": "Prefer registered built-in tools over generated code."}],
        "steps": [{"title": "Inspect the selected image first", "kind": "analysis", "rationale": "Estimate signal and noise."}],
        "stop_conditions": ["The mask is clean enough for downstream use."],
        "planner_notes": ["Planner note."],
    }

    rendered = workflow_plan_to_markdown(plan)

    assert "Planned workflow: `conservative_binary_segmentation`" in rendered
    assert "Constraints:" in rendered
    assert "Planner notes:" in rendered
