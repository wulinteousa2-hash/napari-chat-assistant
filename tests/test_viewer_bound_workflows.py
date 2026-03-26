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
