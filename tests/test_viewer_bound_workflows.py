from __future__ import annotations

import numpy as np

from napari_chat_assistant.agent.context import layer_summary
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job


def test_layer_summary_reports_selected_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.arange(16, dtype=np.float32).reshape(4, 4), name="image_a")
    viewer.add_labels((np.arange(16).reshape(4, 4) > 8).astype(np.uint8), name="mask_a")

    summary = layer_summary(viewer)

    assert "Layers: 2" in summary
    assert "- image_a [Image]" in summary
    assert "- mask_a [Labels]" in summary
    assert "Selected layer: mask_a" in summary


def test_prepare_tool_job_prefers_selected_image_and_avoids_name_collisions(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((8, 8), dtype=np.float32), name="image_a")
    selected = viewer.add_image(np.ones((8, 8), dtype=np.float32), name="image_b", scale=(2.0, 3.0), translate=(5.0, 7.0))
    viewer.add_image(np.ones((8, 8), dtype=np.float32), name="image_b_clahe")
    viewer.layers.selection.active = selected

    prepared = prepare_tool_job(viewer, "apply_clahe", {"kernel_size": 4})

    assert prepared["mode"] == "worker"
    job = prepared["job"]
    assert job["tool_name"] == "apply_clahe"
    assert job["layer_name"] == "image_b"
    assert job["output_name"] == "image_b_clahe_01"
    assert job["kernel_size"] == (4, 4)
    assert job["scale"] == (2.0, 3.0)
    assert job["translate"] == (5.0, 7.0)


def test_preview_threshold_reuses_single_preview_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.linspace(0.0, 1.0, 25, dtype=np.float32).reshape(5, 5), name="image_a")

    first = run_tool_job(prepare_tool_job(viewer, "preview_threshold", {})["job"])
    apply_tool_job_result(viewer, first)
    preview = viewer.layers["__assistant_threshold_preview__"]
    first_data = np.asarray(preview.data).copy()

    image.data = np.flipud(np.asarray(image.data))

    second = run_tool_job(prepare_tool_job(viewer, "preview_threshold", {})["job"])
    apply_tool_job_result(viewer, second)

    assert "__assistant_threshold_preview__" in viewer.layers
    assert sum(layer.name == "__assistant_threshold_preview__" for layer in viewer.layers) == 1
    assert not np.array_equal(np.asarray(preview.data), first_data)


def test_apply_threshold_adds_labels_layer_with_image_transform(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(
        np.array([[0.0, 0.1, 0.8], [0.2, 0.9, 1.0], [0.1, 0.7, 0.95]], dtype=np.float32),
        name="image_a",
        scale=(1.5, 2.5),
        translate=(10.0, 20.0),
    )

    result = run_tool_job(prepare_tool_job(viewer, "apply_threshold", {"polarity": "bright"})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "image_a_labels" in viewer.layers
    labels = viewer.layers["image_a_labels"]
    assert tuple(labels.scale) == (1.5, 2.5)
    assert tuple(labels.translate) == (10.0, 20.0)
    assert labels.data.shape == (3, 3)
    assert "Applied threshold to [image_a] as [image_a_labels]" in message


def test_run_mask_op_creates_snapshot_before_mutation(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    original = np.zeros((9, 9), dtype=np.uint8)
    original[4, 4] = 1
    labels = viewer.add_labels(original.copy(), name="mask_a", scale=(1.0, 2.0), translate=(3.0, 4.0))

    result = run_tool_job(prepare_tool_job(viewer, "run_mask_op", {"op": "dilate", "radius": 1})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "mask_a_assistant_snapshot_01" in viewer.layers
    snapshot = viewer.layers["mask_a_assistant_snapshot_01"]
    assert np.array_equal(np.asarray(snapshot.data), original)
    assert tuple(snapshot.scale) == tuple(labels.scale)
    assert tuple(snapshot.translate) == tuple(labels.translate)
    assert np.count_nonzero(np.asarray(labels.data)) > np.count_nonzero(original)
    assert "Saved snapshot [mask_a_assistant_snapshot_01]" in message


def test_preview_threshold_batch_creates_named_preview_per_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.arange(16, dtype=np.float32).reshape(4, 4), name="image_a")
    viewer.add_image(np.arange(16, 32, dtype=np.float32).reshape(4, 4), name="image_b")

    result = run_tool_job(prepare_tool_job(viewer, "preview_threshold_batch", {"polarity": "bright"})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "__assistant_threshold_preview__::image_a" in viewer.layers
    assert "__assistant_threshold_preview__::image_b" in viewer.layers
    assert "Updated preview masks for 2 image layers with polarity=bright." in message


def test_gaussian_denoise_adds_output_image_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.eye(7, dtype=np.float32), name="image_a", scale=(2.0, 3.0), translate=(4.0, 5.0))

    result = run_tool_job(prepare_tool_job(viewer, "gaussian_denoise", {"sigma": 1.25})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "image_a_gaussian" in viewer.layers
    layer = viewer.layers["image_a_gaussian"]
    assert tuple(layer.scale) == (2.0, 3.0)
    assert tuple(layer.translate) == (4.0, 5.0)
    assert layer.data.shape == (7, 7)
    assert "Applied Gaussian denoising to [image_a] as [image_a_gaussian]" in message


def test_remove_small_objects_adds_cleaned_labels_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.zeros((8, 8), dtype=np.uint8)
    data[1, 1] = 1
    data[4:7, 4:7] = 1
    viewer.add_labels(data, name="mask_a", scale=(1.5, 2.5), translate=(10.0, 20.0))

    result = run_tool_job(prepare_tool_job(viewer, "remove_small_objects", {"min_size": 4})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "mask_a_clean" in viewer.layers
    layer = viewer.layers["mask_a_clean"]
    assert tuple(layer.scale) == (1.5, 2.5)
    assert tuple(layer.translate) == (10.0, 20.0)
    assert int(np.count_nonzero(np.asarray(layer.data))) == 9
    assert "Removed small objects from [mask_a] into [mask_a_clean]" in message


def test_fill_mask_holes_adds_filled_labels_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.ones((5, 5), dtype=np.uint8)
    data[2, 2] = 0
    viewer.add_labels(data, name="mask_a")

    result = run_tool_job(prepare_tool_job(viewer, "fill_mask_holes", {})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "mask_a_filled" in viewer.layers
    layer = viewer.layers["mask_a_filled"]
    assert int(np.asarray(layer.data)[2, 2]) == 1
    assert "Filled mask holes in [mask_a] into [mask_a_filled]" in message


def test_project_max_intensity_drops_projected_axis_transform(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    viewer.add_image(data, name="volume_a", scale=(1.0, 2.0, 3.0), translate=(10.0, 20.0, 30.0))

    result = run_tool_job(prepare_tool_job(viewer, "project_max_intensity", {"axis": 0})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "volume_a_mip" in viewer.layers
    layer = viewer.layers["volume_a_mip"]
    assert layer.data.shape == (3, 4)
    assert tuple(layer.scale) == (2.0, 3.0)
    assert tuple(layer.translate) == (20.0, 30.0)
    assert "Created max-intensity projection for [volume_a] as [volume_a_mip] along axis=0." in message


def test_keep_largest_component_adds_labels_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.zeros((8, 8), dtype=np.uint8)
    data[1, 1] = 1
    data[4:7, 4:7] = 1
    viewer.add_labels(data, name="mask_a", scale=(1.5, 2.5), translate=(10.0, 20.0))

    result = run_tool_job(prepare_tool_job(viewer, "keep_largest_component", {})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "mask_a_largest" in viewer.layers
    layer = viewer.layers["mask_a_largest"]
    assert tuple(layer.scale) == (1.5, 2.5)
    assert tuple(layer.translate) == (10.0, 20.0)
    assert int(np.count_nonzero(np.asarray(layer.data))) == 9
    assert "Kept the largest connected component from [mask_a] in [mask_a_largest]." in message


def test_label_connected_components_adds_instance_labels(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.zeros((6, 6), dtype=np.uint8)
    data[1:3, 1:3] = 1
    data[4:6, 4:6] = 1
    viewer.add_labels(data, name="mask_a")

    result = run_tool_job(prepare_tool_job(viewer, "label_connected_components", {"connectivity": 1})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "mask_a_instances" in viewer.layers
    layer = viewer.layers["mask_a_instances"]
    assert int(np.max(np.asarray(layer.data))) == 2
    assert "objects=2." in message


def test_measure_labels_table_returns_summary(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    labels = np.zeros((5, 5), dtype=np.uint8)
    labels[1:3, 1:3] = 1
    labels[3:5, 3:5] = 2
    intensities = np.arange(25, dtype=np.float32).reshape(5, 5)
    viewer.add_labels(labels, name="mask_a")
    viewer.add_image(intensities, name="image_a")

    result = run_tool_job(
        prepare_tool_job(
            viewer,
            "measure_labels_table",
            {"layer_name": "mask_a", "intensity_layer": "image_a", "properties": ["label", "area", "mean_intensity"]},
        )["job"]
    )
    message = apply_tool_job_result(viewer, result)

    assert "Measured 2 labeled object(s) in [mask_a]." in message
    assert "label=1" in message
    assert "mean_intensity=" in message


def test_crop_to_layer_bbox_adds_cropped_image_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 1
    viewer.add_image(image, name="image_a", scale=(2.0, 3.0), translate=(10.0, 20.0))
    viewer.add_labels(mask, name="mask_a")

    result = run_tool_job(
        prepare_tool_job(
            viewer,
            "crop_to_layer_bbox",
            {"source_layer": "image_a", "reference_layer": "mask_a", "padding": 1},
        )["job"]
    )
    message = apply_tool_job_result(viewer, result)

    assert "image_a_crop" in viewer.layers
    layer = viewer.layers["image_a_crop"]
    assert layer.data.shape == (6, 7)
    assert tuple(layer.scale) == (2.0, 3.0)
    assert tuple(layer.translate) == (12.0, 26.0)
    assert "Cropped [image_a] to the bounding box of [mask_a] as [image_a_crop]." in message


def test_show_image_layers_in_grid_enables_grid_with_auto_shape(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((4, 5), dtype=np.float32), name="image_a")
    viewer.add_image(np.ones((4, 5), dtype=np.float32), name="image_b")
    viewer.add_image(np.full((4, 5), 2.0, dtype=np.float32), name="image_c")
    viewer.add_image(np.full((4, 5), 3.0, dtype=np.float32), name="image_d")
    viewer.add_image(np.full((4, 5), 4.0, dtype=np.float32), name="image_e")
    viewer.add_image(np.full((4, 5), 5.0, dtype=np.float32), name="image_f")
    mask = viewer.add_labels(np.ones((4, 5), dtype=np.uint8), name="mask_a")
    mask.visible = True

    if not hasattr(viewer, "grid"):
        class _Grid:
            enabled = False
            shape = (-1, -1)
            spacing = 0.0

        viewer.grid = _Grid()
    if not hasattr(viewer, "reset_view"):
        viewer.reset_view = lambda: None

    prepared = prepare_tool_job(
        viewer,
        "show_image_layers_in_grid",
        {"spacing": 12},
    )
    assert prepared["mode"] == "immediate"

    result = run_tool_job(prepared["job"])
    message = apply_tool_job_result(viewer, result)

    assert viewer.grid.enabled is True
    assert tuple(viewer.grid.shape) == (2, 3)
    assert float(viewer.grid.spacing) == 12.0
    assert mask.visible is False
    assert "shape=(2, 3)" in message


def test_hide_image_grid_view_disables_grid_and_restores_non_image_layers(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((4, 4), dtype=np.float32), name="image_a")
    mask = viewer.add_labels(np.ones((4, 4), dtype=np.uint8), name="mask_a")
    if not hasattr(viewer, "grid"):
        class _Grid:
            enabled = False
            shape = (-1, -1)
            spacing = 0.0

        viewer.grid = _Grid()
    if not hasattr(viewer, "reset_view"):
        viewer.reset_view = lambda: None
    viewer.grid.enabled = True
    setattr(viewer, "_assistant_grid_hidden_non_image_layers", ["mask_a"])
    mask.visible = False

    prepared = prepare_tool_job(viewer, "hide_image_grid_view", {})
    result = run_tool_job(prepared["job"])
    message = apply_tool_job_result(viewer, result)

    assert viewer.grid.enabled is False
    assert mask.visible is True
    assert "Disabled image grid view." == message


def test_inspect_roi_context_reports_labels_roi(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 1
    viewer.add_labels(mask, name="mask_a")

    prepared = prepare_tool_job(viewer, "inspect_roi_context", {"roi_layer": "mask_a"})

    assert prepared["mode"] == "immediate"
    assert "Labels ROI" in prepared["message"]
    assert "foreground=20" in prepared["message"]


def test_inspect_roi_context_reports_shapes_roi(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(
        data=[np.array([[2, 2], [2, 7], [7, 7], [7, 2]], dtype=float)],
        shape_type="polygon",
        name="roi_shapes",
    )

    prepared = prepare_tool_job(viewer, "inspect_roi_context", {"roi_layer": "roi_shapes"})

    assert prepared["mode"] == "immediate"
    assert "Shapes ROI" in prepared["message"]


def test_measure_shapes_roi_area_reports_total_and_stores_metadata(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_shapes(
        data=[np.array([[2, 2], [2, 7], [7, 7], [7, 2]], dtype=float)],
        shape_type="polygon",
        name="roi_shapes",
    )

    prepared = prepare_tool_job(viewer, "measure_shapes_roi_area", {"roi_layer": "roi_shapes"})

    assert prepared["mode"] == "immediate"
    assert 'Selected Shapes layer: "roi_shapes"' in prepared["message"]
    assert "Measured 1 shape(s)." in prepared["message"]
    assert "Total measured area:" in prepared["message"]
    assert "roi_shape_areas" in layer.metadata
    assert layer.metadata["roi_shape_areas"]["layer_name"] == "roi_shapes"
    assert len(layer.metadata["roi_shape_areas"]["areas"]) == 1


def test_extract_roi_values_from_labels_roi(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 1
    viewer.add_image(image, name="image_a")
    viewer.add_labels(mask, name="mask_a")

    result = run_tool_job(
        prepare_tool_job(viewer, "extract_roi_values", {"image_layer": "image_a", "roi_layer": "mask_a"})["job"]
    )
    message = apply_tool_job_result(viewer, result)

    assert "Extracted ROI values from [image_a] using [mask_a]." in message
    assert "roi_voxels=20" in message
    assert "mean=" in message


def test_extract_roi_values_from_shapes_roi(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    viewer.add_image(image, name="image_a")
    viewer.add_shapes(
        data=[np.array([[2, 2], [2, 7], [7, 7], [7, 2]], dtype=float)],
        shape_type="polygon",
        name="roi_shapes",
    )

    result = run_tool_job(
        prepare_tool_job(viewer, "extract_roi_values", {"image_layer": "image_a", "roi_layer": "roi_shapes"})["job"]
    )
    message = apply_tool_job_result(viewer, result)

    assert "Extracted ROI values from [image_a] using [roi_shapes]." in message
    assert "roi_voxels=" in message


def test_sam_segment_from_box_reports_missing_backend_cleanly(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.arange(100, dtype=np.float32).reshape(10, 10), name="image_a")
    viewer.add_shapes(
        data=[np.array([[2, 2], [2, 7], [7, 7], [7, 2]], dtype=float)],
        shape_type="polygon",
        name="roi_shapes",
    )

    prepared = prepare_tool_job(viewer, "sam_segment_from_box", {"image_layer": "image_a", "roi_layer": "roi_shapes"})

    assert prepared["mode"] == "immediate"
    assert "SAM2 backend is not configured." in prepared["message"]


def test_sam_segment_from_points_reports_missing_backend_cleanly(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.arange(100, dtype=np.float32).reshape(10, 10), name="image_a")
    viewer.add_points(np.array([[3, 3], [6, 6]], dtype=float), name="points_a")

    prepared = prepare_tool_job(viewer, "sam_segment_from_points", {"image_layer": "image_a", "points_layer": "points_a"})

    assert prepared["mode"] == "immediate"
    assert "SAM2 backend is not configured." in prepared["message"]
