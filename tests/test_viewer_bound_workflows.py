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
    assert "Selected: mask_a" in summary


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


def test_run_mask_op_convert_to_mask_replaces_multilabel_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    labels = np.array([[0, 2, 0], [3, 0, 4], [0, 0, 0]], dtype=np.int32)
    viewer.add_labels(labels.copy(), name="mask_a")

    result = run_tool_job(prepare_tool_job(viewer, "run_mask_op", {"op": "convert_to_mask"})["job"])
    message = apply_tool_job_result(viewer, result)

    converted = np.asarray(viewer.layers["mask_a"].data)
    assert np.array_equal(converted, np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.int32))
    assert "applied convert_to_mask to [mask_a]" in message.lower()


def test_run_mask_op_distance_map_adds_image_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.zeros((7, 7), dtype=np.uint8)
    data[2:5, 2:5] = 1
    viewer.add_labels(data, name="mask_a", scale=(1.0, 2.0), translate=(3.0, 4.0))

    result = run_tool_job(prepare_tool_job(viewer, "run_mask_op", {"op": "distance_map"})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "mask_a_distance_map" in viewer.layers
    layer = viewer.layers["mask_a_distance_map"]
    assert layer.data.shape == (7, 7)
    assert tuple(layer.scale) == (1.0, 2.0)
    assert tuple(layer.translate) == (3.0, 4.0)
    assert "Created [mask_a_distance_map] from [mask_a] using distance_map." in message


def test_run_mask_op_watershed_adds_labels_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.zeros((9, 9), dtype=np.uint8)
    data[2:4, 2:4] = 1
    data[2:4, 5:7] = 1
    data[4:6, 3:6] = 1
    viewer.add_labels(data, name="mask_a")

    result = run_tool_job(prepare_tool_job(viewer, "run_mask_op", {"op": "watershed"})["job"])
    message = apply_tool_job_result(viewer, result)

    assert "mask_a_watershed" in viewer.layers
    watershed_layer = viewer.layers["mask_a_watershed"]
    assert int(np.max(np.asarray(watershed_layer.data))) >= 1
    assert "Created [mask_a_watershed] from [mask_a] using watershed." in message


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


def test_delete_layers_removes_named_shape_layers(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes([np.array([[0.0, 0.0], [0.0, 5.0], [5.0, 5.0]])], shape_type="polygon", name="shape_a")
    viewer.add_shapes([np.array([[1.0, 1.0], [1.0, 6.0], [6.0, 6.0]])], shape_type="polygon", name="shape_b")
    viewer.add_image(np.zeros((8, 8), dtype=np.float32), name="image_a")

    prepared = prepare_tool_job(viewer, "delete_layers", {"layer_names": ["shape_a", "shape_b"]})
    message = apply_tool_job_result(viewer, run_tool_job(prepared["job"]))

    assert "shape_a" not in viewer.layers
    assert "shape_b" not in viewer.layers
    assert "image_a" in viewer.layers
    assert "Deleted 2 layer(s): [shape_a], [shape_b]." == message


def test_delete_layers_removes_all_shapes_layers_by_type(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_shapes(
        [np.array([[0.0, 0.0], [0.0, 5.0], [5.0, 5.0]])], shape_type="polygon", name="template_profile_line"
    )
    viewer.add_shapes(
        [np.array([[1.0, 1.0], [1.0, 6.0], [6.0, 6.0]])], shape_type="polygon", name="template_intensity_roi"
    )
    viewer.add_image(np.zeros((8, 8), dtype=np.float32), name="em_2d_snr_mid")
    viewer.add_labels(np.zeros((8, 8), dtype=np.uint8), name="em_2d_mask")

    prepared = prepare_tool_job(viewer, "delete_layers", {"layer_type": "shapes"})
    message = apply_tool_job_result(viewer, run_tool_job(prepared["job"]))

    assert "template_profile_line" not in viewer.layers
    assert "template_intensity_roi" not in viewer.layers
    assert "em_2d_snr_mid" in viewer.layers
    assert "em_2d_mask" in viewer.layers
    assert "Deleted 2 [shapes] layer(s): [template_profile_line], [template_intensity_roi]." == message


def test_delete_all_layers_removes_everything(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((8, 8), dtype=np.float32), name="image_a")
    viewer.add_labels(np.zeros((8, 8), dtype=np.uint8), name="mask_a")
    viewer.add_shapes([np.array([[0.0, 0.0], [0.0, 5.0], [5.0, 5.0]])], shape_type="polygon", name="roi_a")

    message = apply_tool_job_result(viewer, run_tool_job(prepare_tool_job(viewer, "delete_all_layers", {})["job"]))

    assert len(viewer.layers) == 0
    assert message == "Deleted all 3 layer(s): [image_a], [mask_a], [roi_a]."


def test_hide_all_layers_turns_every_layer_invisible(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((8, 8), dtype=np.float32), name="image_a")
    viewer.add_labels(np.zeros((8, 8), dtype=np.uint8), name="mask_a")

    message = apply_tool_job_result(viewer, run_tool_job(prepare_tool_job(viewer, "hide_all_layers", {})["job"]))

    assert all(not bool(layer.visible) for layer in viewer.layers)
    assert message == "Hid all 2 layer(s)."


def test_show_all_except_layers_hides_named_layers_and_shows_others(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.zeros((8, 8), dtype=np.float32), name="image_a")
    mask = viewer.add_labels(np.zeros((8, 8), dtype=np.uint8), name="mask_a")
    roi = viewer.add_shapes([np.array([[0.0, 0.0], [0.0, 5.0], [5.0, 5.0]])], shape_type="polygon", name="roi_a")
    image.visible = False
    mask.visible = False
    roi.visible = False

    message = apply_tool_job_result(
        viewer,
        run_tool_job(prepare_tool_job(viewer, "show_all_except_layers", {"layer_names": ["mask_a"]})["job"]),
    )

    assert bool(image.visible) is True
    assert bool(mask.visible) is False
    assert bool(roi.visible) is True
    assert message == "Showed all layers except 1 layer(s): [mask_a]."


def test_show_only_layers_defaults_to_selected_layer_when_no_names_provided(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.zeros((8, 8), dtype=np.float32), name="image_a")
    mask = viewer.add_labels(np.zeros((8, 8), dtype=np.uint8), name="mask_a")
    viewer.layers.selection.active = image

    message = apply_tool_job_result(viewer, run_tool_job(prepare_tool_job(viewer, "show_only_layers", {})["job"]))

    assert bool(image.visible) is True
    assert bool(mask.visible) is False
    assert message == "Showing only 1 layer(s): [image_a]."


def test_set_layer_scale_applies_scalar_to_selected_2d_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.zeros((8, 8), dtype=np.float32), name="image_a", scale=(2.0, 3.0))
    viewer.layers.selection.active = image

    prepared = prepare_tool_job(viewer, "set_layer_scale", {"scale": 0.1})
    message = apply_tool_job_result(viewer, run_tool_job(prepared["job"]))

    assert tuple(image.scale) == (0.1, 0.1)
    assert message == "Set scale for [image_a] to (0.1, 0.1)."


def test_set_layer_scale_resets_named_3d_layer(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((4, 5, 6), dtype=np.float32), name="volume_a", scale=(2.0, 0.5, 0.25))

    prepared = prepare_tool_job(viewer, "set_layer_scale", {"layer_name": "volume_a", "scale": 1.0})
    message = apply_tool_job_result(viewer, run_tool_job(prepared["job"]))

    assert tuple(viewer.layers["volume_a"].scale) == (1.0, 1.0, 1.0)
    assert message == "Set scale for [volume_a] to (1.0, 1.0, 1.0)."


def test_create_analysis_montage_builds_composite_image_mask_and_tile_boxes(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.ones((4, 6), dtype=np.float32), name="img_a")
    viewer.add_image(np.ones((2, 4), dtype=np.float32) * 2.0, name="img_b")
    viewer.add_image(np.ones((3, 5), dtype=np.float32) * 3.0, name="img_c")
    viewer.add_image(np.ones((4, 4, 4), dtype=np.float32), name="volume_a")

    prepared = prepare_tool_job(
        viewer,
        "create_analysis_montage",
        {"layer_names": ["img_a", "img_b", "img_c"], "rows": 2, "columns": 2, "spacing": 2, "show_tile_boxes": True},
    )
    message = apply_tool_job_result(viewer, run_tool_job(prepared["job"]))

    assert "analysis_montage" in viewer.layers
    assert "analysis_montage_mask" in viewer.layers
    assert "analysis_montage_tiles" in viewer.layers
    montage = viewer.layers["analysis_montage"]
    mask = viewer.layers["analysis_montage_mask"]
    boxes = viewer.layers["analysis_montage_tiles"]
    assert montage.data.shape == (10, 14)
    assert mask.data.shape == (10, 14)
    assert len(boxes.data) == 3
    metadata = montage.metadata["montage_canvas"]
    assert metadata["purpose"] == "analysis"
    assert metadata["layout"]["rows"] == 2
    assert metadata["layout"]["columns"] == 2
    assert metadata["layout"]["spacing"] == 2
    assert metadata["layout"]["tile_size"] == [4, 6]
    assert [item["source_layer"] for item in metadata["created_from"]] == ["img_a", "img_b", "img_c"]
    assert metadata["linked_outputs"]["montage_labels_layer"] == "analysis_montage_mask"
    assert mask.metadata["montage_canvas"]["role"] == "mask"
    assert boxes.metadata["montage_canvas"]["role"] == "tile_boxes"
    assert "Created analysis montage [analysis_montage] from 3 image layer(s) with grid=2x2 spacing=2." in message


def test_create_analysis_montage_uses_selected_2d_images_by_default(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image_a = viewer.add_image(np.arange(9, dtype=np.float32).reshape(3, 3), name="img_a")
    image_b = viewer.add_image(np.arange(16, dtype=np.float32).reshape(4, 4), name="img_b")
    viewer.layers.selection.add(image_a)
    viewer.layers.selection.add(image_b)

    prepared = prepare_tool_job(viewer, "create_analysis_montage", {"columns": 2, "create_mask_layer": False, "show_tile_boxes": False})
    message = apply_tool_job_result(viewer, run_tool_job(prepared["job"]))

    assert "analysis_montage" in viewer.layers
    assert "analysis_montage_mask" not in viewer.layers
    assert "analysis_montage_tiles" not in viewer.layers
    montage = viewer.layers["analysis_montage"]
    assert montage.metadata["montage_canvas"]["created_from"][0]["source_layer"] == "img_a"
    assert montage.metadata["montage_canvas"]["created_from"][1]["source_layer"] == "img_b"
    assert montage.data.shape == (4, 8)
    assert "mask_layer=false tile_boxes=false." in message


def test_split_montage_annotations_to_sources_exports_labels_per_image(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.ones((2, 2), dtype=np.float32), name="img_a", scale=(1.5, 2.5), translate=(10.0, 20.0))
    viewer.add_image(np.ones((2, 2), dtype=np.float32) * 2.0, name="img_b", scale=(3.0, 4.0), translate=(30.0, 40.0))

    message = apply_tool_job_result(
        viewer,
        run_tool_job(
            prepare_tool_job(
                viewer,
                "create_analysis_montage",
                {"layer_names": ["img_a", "img_b"], "rows": 1, "columns": 2, "spacing": 1},
            )["job"]
        ),
    )

    assert "Created analysis montage [analysis_montage]" in message
    montage_mask = viewer.layers["analysis_montage_mask"]
    montage_mask.data[0:2, 0:2] = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    montage_mask.data[0:2, 3:5] = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    split_message = apply_tool_job_result(
        viewer,
        run_tool_job(
            prepare_tool_job(
                viewer,
                "split_montage_annotations_to_sources",
                {"annotation_layer": "analysis_montage_mask", "montage_layer": "analysis_montage"},
            )["job"]
        ),
    )

    assert "img_a_analysis_montage_mask" in viewer.layers
    assert "img_b_analysis_montage_mask" in viewer.layers
    split_a = viewer.layers["img_a_analysis_montage_mask"]
    split_b = viewer.layers["img_b_analysis_montage_mask"]
    assert tuple(split_a.scale) == (1.5, 2.5)
    assert tuple(split_a.translate) == (10.0, 20.0)
    assert tuple(split_b.scale) == (3.0, 4.0)
    assert tuple(split_b.translate) == (30.0, 40.0)
    assert np.array_equal(np.asarray(split_a.data), np.array([[1, 0], [0, 1]], dtype=np.uint8))
    assert np.array_equal(np.asarray(split_b.data), np.array([[0, 1], [1, 0]], dtype=np.uint8))
    assert split_a.metadata["montage_split"]["source_layer"] == "img_a"
    assert "Split montage labels [analysis_montage_mask] into 2 per-source layer(s)" in split_message


def test_split_montage_annotations_to_sources_exports_points_per_image(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.ones((2, 2), dtype=np.float32), name="img_a", scale=(1.0, 1.0), translate=(5.0, 6.0))
    viewer.add_image(np.ones((2, 2), dtype=np.float32) * 2.0, name="img_b", scale=(2.0, 3.0), translate=(7.0, 8.0))

    apply_tool_job_result(
        viewer,
        run_tool_job(
            prepare_tool_job(
                viewer,
                "create_analysis_montage",
                {"layer_names": ["img_a", "img_b"], "rows": 1, "columns": 2, "spacing": 1, "create_mask_layer": False},
            )["job"]
        ),
    )
    viewer.add_points(np.array([[0.5, 0.5], [1.5, 3.5]], dtype=np.float32), name="montage_points")

    split_message = apply_tool_job_result(
        viewer,
        run_tool_job(
            prepare_tool_job(
                viewer,
                "split_montage_annotations_to_sources",
                {"annotation_layer": "montage_points", "montage_layer": "analysis_montage"},
            )["job"]
        ),
    )

    assert "img_a_montage_points" in viewer.layers
    assert "img_b_montage_points" in viewer.layers
    points_a = viewer.layers["img_a_montage_points"]
    points_b = viewer.layers["img_b_montage_points"]
    assert tuple(points_a.scale) == (1.0, 1.0)
    assert tuple(points_a.translate) == (5.0, 6.0)
    assert tuple(points_b.scale) == (2.0, 3.0)
    assert tuple(points_b.translate) == (7.0, 8.0)
    assert np.allclose(np.asarray(points_a.data), np.array([[0.5, 0.5]], dtype=np.float32))
    assert np.allclose(np.asarray(points_b.data), np.array([[1.5, 0.5]], dtype=np.float32))
    assert points_b.metadata["montage_split"]["source_kind"] == "points"
    assert "with 2 point(s)" in split_message


def test_edit_mask_in_roi_applies_operation_only_inside_shapes_roi(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.zeros((8, 8), dtype=np.uint8)
    data[2:6, 2:6] = 1
    data[3, 4] = 0
    viewer.add_labels(data, name="mask_a", scale=(1.5, 2.5), translate=(10.0, 20.0))
    viewer.add_shapes(
        [np.array([[2.0, 2.0], [2.0, 6.0], [6.0, 6.0], [6.0, 2.0]], dtype=np.float32)],
        shape_type="rectangle",
        name="roi_a",
    )

    prepared = prepare_tool_job(
        viewer,
        "edit_mask_in_roi",
        {"mask_layer": "mask_a", "roi_layer": "roi_a", "op": "fill_holes"},
    )
    message = apply_tool_job_result(viewer, run_tool_job(prepared["job"]))

    assert "mask_a_fill_holes_roi" in viewer.layers
    edited = viewer.layers["mask_a_fill_holes_roi"]
    assert tuple(edited.scale) == (1.5, 2.5)
    assert tuple(edited.translate) == (10.0, 20.0)
    assert int(np.asarray(edited.data)[3, 4]) == 1
    assert np.array_equal(np.asarray(edited.data)[:2, :], data[:2, :])
    assert "Applied [fill_holes] to [mask_a] only inside ROI [roi_a]" in message


def test_edit_mask_in_roi_keeps_global_mask_unchanged_outside_roi(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.zeros((9, 9), dtype=np.uint8)
    data[1, 1] = 1
    data[4:7, 4:7] = 1
    viewer.add_labels(data, name="mask_a")
    viewer.add_shapes(
        [np.array([[3.0, 3.0], [3.0, 8.0], [8.0, 8.0], [8.0, 3.0]], dtype=np.float32)],
        shape_type="rectangle",
        name="roi_big",
    )

    prepared = prepare_tool_job(
        viewer,
        "edit_mask_in_roi",
        {"mask_layer": "mask_a", "roi_layer": "roi_big", "op": "remove_small", "min_size": 4},
    )
    message = apply_tool_job_result(viewer, run_tool_job(prepared["job"]))

    edited = viewer.layers["mask_a_remove_small_roi"]
    edited_data = np.asarray(edited.data)
    assert int(edited_data[1, 1]) == 1
    assert int(np.sum(edited_data[4:7, 4:7])) == 9
    assert "min_size=4" in message


def test_extract_axon_interiors_adds_labels_layer_from_dark_ring_enclosure(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = np.ones((32, 32), dtype=np.float32) * 0.8
    image[8:24, 8:24] = 0.1
    image[11:21, 11:21] = 0.85
    viewer.add_image(image, name="em_a", scale=(2.0, 2.0), translate=(5.0, 7.0))

    result = run_tool_job(
        prepare_tool_job(
            viewer,
            "extract_axon_interiors",
            {"image_layer": "em_a", "sigma": 0.0, "dark_quantile": 0.2, "closing_radius": 1, "min_area": 20},
        )["job"]
    )
    message = apply_tool_job_result(viewer, result)

    assert "em_a_axon_interiors" in viewer.layers
    labels = viewer.layers["em_a_axon_interiors"]
    assert tuple(labels.scale) == (2.0, 2.0)
    assert tuple(labels.translate) == (5.0, 7.0)
    assert int(np.max(np.asarray(labels.data))) >= 1
    assert "Extracted candidate axon interiors from [em_a]" in message
    assert "axon_interior_extraction" in labels.metadata


def test_extract_axon_interiors_ignores_black_padded_background(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = np.zeros((64, 64), dtype=np.float32)
    image[8:56, 8:56] = 0.8
    image[20:44, 20:44] = 0.1
    image[25:39, 25:39] = 0.85
    viewer.add_image(image, name="em_padded")

    result = run_tool_job(
        prepare_tool_job(
            viewer,
            "extract_axon_interiors",
            {"image_layer": "em_padded", "sigma": 0.0, "closing_radius": 1, "min_area": 20},
        )["job"]
    )
    message = apply_tool_job_result(viewer, result)

    labels = viewer.layers["em_padded_axon_interiors"]
    assert int(np.max(np.asarray(labels.data))) >= 1
    metadata = labels.metadata["axon_interior_extraction"]
    assert float(metadata["threshold"]) > 0.0
    assert "Extracted candidate axon interiors from [em_padded]" in message


def test_sam_propagate_points_3d_adds_volume_labels_layer(make_napari_viewer_proxy, monkeypatch):
    import napari_chat_assistant.agent.tools_builtin.workbench as wb

    monkeypatch.setattr(wb, "get_sam2_backend_status", lambda: (True, "ok"))

    def fake_propagate(volume, *, seed_frame_idx, point_coords_xy, point_labels, model_name=None, config=None):
        result = np.zeros_like(volume, dtype=np.int32)
        result[:, 2:5, 3:7] = 1
        return result, f"fake backend seed_slice={seed_frame_idx}"

    monkeypatch.setattr(wb, "propagate_volume_from_points", fake_propagate)

    viewer = make_napari_viewer_proxy()
    volume = np.zeros((5, 12, 14), dtype=np.float32)
    viewer.add_image(volume, name="vol_a", scale=(1.0, 2.0, 3.0), translate=(10.0, 20.0, 30.0))
    viewer.add_points(
        np.asarray([[2.0, 4.0, 5.0], [2.0, 6.0, 8.0]], dtype=np.float32),
        name="pts3d",
        features={"sam_label": np.asarray([1, 0], dtype=np.int32)},
    )

    result = run_tool_job(
        prepare_tool_job(
            viewer,
            "sam_propagate_points_3d",
            {"image_layer": "vol_a", "points_layer": "pts3d"},
        )["job"]
    )
    message = apply_tool_job_result(viewer, result)

    assert "vol_a_sam2_propagated" in viewer.layers
    labels = viewer.layers["vol_a_sam2_propagated"]
    assert tuple(labels.scale) == (1.0, 2.0, 3.0)
    assert tuple(labels.translate) == (10.0, 20.0, 30.0)
    assert labels.data.shape == (5, 12, 14)
    assert int(np.max(np.asarray(labels.data))) == 1
    assert "seed_slice=2" in message
    assert "tracked_slices=5" in message


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


def test_show_layers_makes_named_layers_visible(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image_a = viewer.add_image(np.zeros((4, 4), dtype=np.float32), name="image_a")
    mask_a = viewer.add_labels(np.ones((4, 4), dtype=np.uint8), name="mask_a")
    image_a.visible = False
    mask_a.visible = False

    prepared = prepare_tool_job(viewer, "show_layers", {"layer_names": ["image_a"]})
    result = run_tool_job(prepared["job"])
    message = apply_tool_job_result(viewer, result)

    assert image_a.visible is True
    assert mask_a.visible is False
    assert "Showed 1 layer(s): [image_a]." == message


def test_hide_layers_hides_named_layers(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image_a = viewer.add_image(np.zeros((4, 4), dtype=np.float32), name="image_a")
    mask_a = viewer.add_labels(np.ones((4, 4), dtype=np.uint8), name="mask_a")
    image_a.visible = True
    mask_a.visible = True

    prepared = prepare_tool_job(viewer, "hide_layers", {"layer_names": ["mask_a"]})
    result = run_tool_job(prepared["job"])
    message = apply_tool_job_result(viewer, result)

    assert image_a.visible is True
    assert mask_a.visible is False
    assert "Hid 1 layer(s): [mask_a]." == message


def test_show_only_layers_hides_everything_else(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image_a = viewer.add_image(np.zeros((4, 4), dtype=np.float32), name="image_a")
    image_b = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="image_b")
    mask_a = viewer.add_labels(np.ones((4, 4), dtype=np.uint8), name="mask_a")

    prepared = prepare_tool_job(viewer, "show_only_layers", {"layer_names": ["image_b"]})
    result = run_tool_job(prepared["job"])
    message = apply_tool_job_result(viewer, result)

    assert image_a.visible is False
    assert image_b.visible is True
    assert mask_a.visible is False
    assert "Showing only 1 layer(s): [image_b]." == message


def test_show_all_layers_restores_all_visibility(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image_a = viewer.add_image(np.zeros((4, 4), dtype=np.float32), name="image_a")
    image_b = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="image_b")
    image_a.visible = False
    image_b.visible = False

    prepared = prepare_tool_job(viewer, "show_all_layers", {})
    result = run_tool_job(prepared["job"])
    message = apply_tool_job_result(viewer, result)

    assert image_a.visible is True
    assert image_b.visible is True
    assert "Showed all 2 layer(s)." == message


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

    assert prepared["mode"] == "worker"
    assert prepared["job"]["tool_name"] == "sam_segment_from_box"
    assert prepared["job"]["image_layer_name"] == "image_a"
    assert prepared["job"]["roi_layer_name"] == "roi_shapes"


def test_sam_segment_from_points_reports_missing_backend_cleanly(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.arange(100, dtype=np.float32).reshape(10, 10), name="image_a")
    viewer.add_points(np.array([[3, 3], [6, 6]], dtype=float), name="points_a")

    prepared = prepare_tool_job(viewer, "sam_segment_from_points", {"image_layer": "image_a", "points_layer": "points_a"})

    assert prepared["mode"] == "worker"
    assert prepared["job"]["tool_name"] == "sam_segment_from_points"
    assert prepared["job"]["image_layer_name"] == "image_a"
    assert prepared["job"]["points_layer_name"] == "points_a"


def test_sam_auto_segment_adds_labels_layer_when_backend_returns_mask(make_napari_viewer_proxy, monkeypatch):
    from napari_chat_assistant.agent.tools_builtin import workbench as wb

    viewer = make_napari_viewer_proxy()
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    viewer.add_image(image, name="image_a")

    def fake_segment_image_auto(data, *, model_name=None):
        del model_name
        return (np.asarray(data) > 50).astype(np.int32), "fake-auto"

    monkeypatch.setattr(wb, "segment_image_auto", fake_segment_image_auto)

    prepared = prepare_tool_job(viewer, "sam_auto_segment", {"image_layer": "image_a"})
    result = run_tool_job(prepared["job"])
    message = apply_tool_job_result(viewer, result)

    assert "Auto-segmented [image_a]" in message
    assert "fake-auto" in message
    assert "image_a_sam2_auto" in viewer.layers


def test_sam_refine_mask_adds_refined_labels_layer(make_napari_viewer_proxy, monkeypatch):
    from napari_chat_assistant.agent.tools_builtin import workbench as wb

    viewer = make_napari_viewer_proxy()
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2:7, 2:7] = 1
    viewer.add_image(image, name="image_a")
    viewer.add_labels(mask, name="mask_a")

    def fake_refine_mask_from_mask(data, *, mask, roi_mask=None, model_name=None):
        del data, roi_mask, model_name
        refined = np.asarray(mask, dtype=np.int32).copy()
        refined[1:8, 1:8] = np.maximum(refined[1:8, 1:8], 1)
        return refined, "fake-refine"

    monkeypatch.setattr(wb, "refine_mask_from_mask", fake_refine_mask_from_mask)

    prepared = prepare_tool_job(viewer, "sam_refine_mask", {"image_layer": "image_a", "mask_layer": "mask_a"})
    result = run_tool_job(prepared["job"])
    message = apply_tool_job_result(viewer, result)

    assert "Refined [mask_a] on [image_a]" in message
    assert "fake-refine" in message
    assert "mask_a_sam2_refined" in viewer.layers
