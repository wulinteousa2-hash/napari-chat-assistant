# Developer Tool Inventory

## Scope

Current assistant-exposed tool surface in `napari-chat-assistant`.

## Runtime Model

- `registry-backed`
  - all assistant-visible tools now resolve through `ToolRegistry`
  - specs live on tool classes
  - both `immediate` and `worker` tools use the same `prepare/execute/apply` flow
- `placeholder`
  - registered name exists
  - not implemented yet

## Registry-Backed Tools

### Inspection / ROI / Measurement

- `inspect_roi_context`
- `measure_labels_table`
- `measure_shapes_roi_area`
- `extract_roi_values`
- `compare_roi_groups`
- `compare_image_groups`

### Enhancement / Image Processing

- `apply_clahe`
- `gaussian_denoise`
- `project_max_intensity`
- `crop_to_layer_bbox`

### Segmentation / Mask Cleanup

- `remove_small_objects`
- `fill_mask_holes`
- `edit_mask_in_roi`
- `keep_largest_component`
- `label_connected_components`
- `extract_axon_interiors`

### Viewer Layout / Visibility

- `show_image_layers_in_grid`
- `hide_image_grid_view`
- `show_layers`
- `hide_layers`
- `delete_layers`
- `show_only_layers`
- `show_all_layers`
- `arrange_layers_for_presentation`
- `create_analysis_montage`

### SAM / Prompted Segmentation

- `sam_segment_from_box`
- `sam_segment_from_points`
- `sam_propagate_points_3d`

### Workflow

- `recommend_next_step` (`placeholder`)
- `record_workflow_step` (`placeholder`)

### SAM Placeholders

- `sam_refine_mask` (`placeholder`)
- `sam_auto_segment` (`placeholder`)

## Registry-Backed Tools Migrated From The Legacy Dispatcher

### Session / Layer Inspection

- `list_layers`
- `inspect_selected_layer`
- `inspect_layer`

### ND2 / Spectral Integration

- `open_nd2_converter`
- `open_spectral_viewer`
- `open_spectral_analysis`

### Thresholding

- `preview_threshold`
- `apply_threshold`
- `preview_threshold_batch`
- `apply_threshold_batch`

### Mask Summary / Batch Mask Ops

- `measure_mask`
- `measure_mask_batch`
- `run_mask_op`
- `run_mask_op_batch`

### ROI Stats / Widgets / Stats

- `measure_shapes_roi_stats`
- `open_intensity_metrics_table`
- `open_group_comparison_widget`
- `summarize_intensity`
- `plot_histogram`
- `compare_image_layers_ttest`

### Batch Enhancement

- `apply_clahe_batch`

## Newer Workflow-Oriented Additions

- `edit_mask_in_roi`
  - local Labels edit inside Shapes/Labels ROI only
- `delete_layers`
  - explicit deletion by name or type
- `create_analysis_montage`
  - composite analysis canvas with reversible tile metadata
- `arrange_layers_for_presentation`
  - presentation layout, distinct from viewer grid
- `show_image_layers_in_grid`
  - napari grid compare mode

## Recommended Direction

- add new tools through `ToolRegistry`
- keep assistant-visible names in sync with `assistant_system_prompt()`
- keep `dispatcher.py` as a thin adapter only, not a second implementation surface
