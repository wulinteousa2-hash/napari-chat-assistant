from __future__ import annotations

from napari_chat_assistant.agent.tool_registry import ToolRegistry
from napari_chat_assistant.agent.tools_builtin import builtin_tools


def test_builtin_registry_includes_workbench_scaffold_tools():
    registry = ToolRegistry()
    registry.extend(builtin_tools())

    expected = {
        "apply_clahe",
        "gaussian_denoise",
        "remove_small_objects",
        "fill_mask_holes",
        "keep_largest_component",
        "label_connected_components",
        "measure_labels_table",
        "project_max_intensity",
        "crop_to_layer_bbox",
        "show_image_layers_in_grid",
        "hide_image_grid_view",
        "arrange_layers_for_presentation",
        "inspect_roi_context",
        "extract_roi_values",
        "sam_segment_from_box",
        "sam_segment_from_points",
        "sam_refine_mask",
        "sam_auto_segment",
        "recommend_next_step",
        "record_workflow_step",
    }

    assert expected.issubset(registry.names())
