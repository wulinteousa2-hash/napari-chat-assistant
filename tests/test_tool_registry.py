from __future__ import annotations

from napari_chat_assistant.agent.tool_registry import ToolRegistry
from napari_chat_assistant.agent.tools_builtin import builtin_tools


def test_builtin_registry_includes_workbench_scaffold_tools():
    registry = ToolRegistry()
    registry.extend(builtin_tools())

    expected = {
        "apply_clahe",
        "apply_clahe_batch",
        "apply_threshold",
        "apply_threshold_batch",
        "compare_image_layers_ttest",
        "inspect_layer",
        "inspect_selected_layer",
        "list_layers",
        "measure_mask",
        "measure_mask_batch",
        "measure_shapes_roi_stats",
        "gaussian_denoise",
        "open_group_comparison_widget",
        "open_intensity_metrics_table",
        "open_nd2_converter",
        "open_spectral_analysis",
        "open_spectral_viewer",
        "plot_histogram",
        "preview_threshold",
        "preview_threshold_batch",
        "remove_small_objects",
        "run_mask_op",
        "run_mask_op_batch",
        "fill_mask_holes",
        "edit_mask_in_roi",
        "keep_largest_component",
        "label_connected_components",
        "measure_labels_table",
        "project_max_intensity",
        "crop_to_layer_bbox",
        "show_image_layers_in_grid",
        "hide_image_grid_view",
        "show_layers",
        "hide_layers",
        "show_only_layers",
        "show_all_layers",
        "arrange_layers_for_presentation",
        "create_analysis_montage",
        "create_synthetic_demo_image",
        "split_montage_annotations_to_sources",
        "inspect_roi_context",
        "extract_roi_values",
        "summarize_intensity",
        "sam_segment_from_box",
        "sam_segment_from_points",
        "sam_refine_mask",
        "sam_auto_segment",
        "recommend_next_step",
        "record_workflow_step",
    }

    assert expected.issubset(registry.names())
