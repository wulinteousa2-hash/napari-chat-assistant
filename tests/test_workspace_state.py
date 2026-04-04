from __future__ import annotations

import json
import numpy as np

from napari_chat_assistant.agent.workspace_state import load_workspace_manifest, save_workspace_manifest


def test_workspace_manifest_round_trips_generated_image_via_ome_zarr(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        name="generated_image",
        scale=(2.0, 3.0),
        translate=(4.0, 5.0),
    )
    image.contrast_limits = (1.0, 12.0)
    image.gamma = 0.8

    destination = tmp_path / "workspace.json"
    save_result = save_workspace_manifest(viewer, destination)

    assert save_result["saved_layers"] == 1
    assert save_result["skipped_layers"] == []
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    record = payload["layers"][0]
    assert record["asset_format"] == "ome-zarr"
    assert record["asset_path"].endswith(".ome.zarr")
    assert (tmp_path / "workspace_assets" / record["asset_path"] / "0").exists()

    restored_viewer = make_napari_viewer_proxy()
    load_result = load_workspace_manifest(restored_viewer, destination)

    assert load_result["skipped_layers"] == []
    assert "generated_image" in restored_viewer.layers
    restored = restored_viewer.layers["generated_image"]
    assert np.array_equal(np.asarray(restored.data), np.arange(16, dtype=np.float32).reshape(4, 4))
    assert tuple(restored.scale) == (2.0, 3.0)
    assert tuple(restored.translate) == (4.0, 5.0)
    assert tuple(restored.contrast_limits) == (1.0, 12.0)
    assert float(restored.gamma) == 0.8


def test_workspace_manifest_load_handles_scale_lists_without_truthiness_errors(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(
        np.ones((3, 3), dtype=np.float32),
        name="image_a",
        scale=(1.0, 1.0),
        translate=(0.0, 0.0),
    )
    destination = tmp_path / "workspace.json"
    save_workspace_manifest(viewer, destination)

    restored_viewer = make_napari_viewer_proxy()
    load_result = load_workspace_manifest(restored_viewer, destination)

    assert load_result["skipped_layers"] == []
    assert tuple(restored_viewer.layers["image_a"].scale) == (1.0, 1.0)


def test_workspace_manifest_round_trips_generated_labels_via_ome_zarr(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    labels = np.zeros((6, 6), dtype=np.uint16)
    labels[1:4, 1:4] = 7
    viewer.add_labels(labels, name="segmentation")

    destination = tmp_path / "workspace_labels.json"
    save_result = save_workspace_manifest(viewer, destination)

    assert save_result["saved_layers"] == 1
    assert save_result["skipped_layers"] == []
    payload = json.loads(destination.read_text(encoding="utf-8"))
    record = payload["layers"][0]
    assert record["asset_dataset"] == "labels/labels/0"
    assert (tmp_path / "workspace_labels_assets" / record["asset_path"] / "labels" / "labels" / "0").exists()

    restored_viewer = make_napari_viewer_proxy()
    load_result = load_workspace_manifest(restored_viewer, destination)

    assert load_result["skipped_layers"] == []
    restored = restored_viewer.layers["segmentation"]
    assert np.array_equal(np.asarray(restored.data), labels)


def test_workspace_manifest_round_trips_rgb_generated_image_via_ome_zarr(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    data = np.zeros((5, 5, 3), dtype=np.float32)
    data[..., 0] = 0.25
    data[..., 1] = 0.5
    data[..., 2] = 0.75
    viewer.add_image(data, name="rgb_image", rgb=True, scale=(0.2, 0.2))

    destination = tmp_path / "workspace_rgb.json"
    save_workspace_manifest(viewer, destination)

    restored_viewer = make_napari_viewer_proxy()
    load_result = load_workspace_manifest(restored_viewer, destination)

    assert load_result["skipped_layers"] == []
    restored = restored_viewer.layers["rgb_image"]
    assert bool(restored.rgb) is True
    assert np.allclose(np.asarray(restored.data), data)


def test_workspace_manifest_saves_shapes_with_per_shape_edge_width(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    shapes = viewer.add_shapes(
        [
            np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 4.0]], dtype=float),
            np.array([[5.0, 5.0], [5.0, 8.0], [8.0, 8.0]], dtype=float),
        ],
        shape_type=["polygon", "polygon"],
        name="roi_shapes",
        edge_width=[1.0, 2.0],
    )
    shapes.features = {"label": ["roi_a", "roi_b"]}

    destination = tmp_path / "workspace_shapes.json"
    save_result = save_workspace_manifest(viewer, destination)

    assert save_result["saved_layers"] == 1
    assert save_result["skipped_layers"] == []

    restored_viewer = make_napari_viewer_proxy()
    load_result = load_workspace_manifest(restored_viewer, destination)

    assert load_result["skipped_layers"] == []
    assert "roi_shapes" in restored_viewer.layers
    restored = restored_viewer.layers["roi_shapes"]
    assert len(restored.data) == 2
    assert list(np.asarray(restored.edge_width, dtype=float)) == [1.0, 2.0]


def test_workspace_manifest_round_trips_large_generated_image_via_asset_file(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    data = np.ones((1100, 1100), dtype=np.float32)
    viewer.add_image(data, name="large_generated_image")

    destination = tmp_path / "workspace_large.json"
    save_result = save_workspace_manifest(viewer, destination)

    assert save_result["saved_layers"] == 1
    assert save_result["skipped_layers"] == []
    assert (tmp_path / "workspace_large_assets").exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    record = payload["layers"][0]
    assert record["asset_format"] == "ome-zarr"
    assert (tmp_path / "workspace_large_assets" / record["asset_path"] / "0").exists()

    restored_viewer = make_napari_viewer_proxy()
    load_result = load_workspace_manifest(restored_viewer, destination)

    assert load_result["skipped_layers"] == []
    assert "large_generated_image" in restored_viewer.layers
    restored = restored_viewer.layers["large_generated_image"]
    assert restored.data.shape == (1100, 1100)
    assert np.allclose(np.asarray(restored.data), data)


def test_workspace_manifest_preserves_transparent_shape_faces(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    shapes = viewer.add_shapes(
        [np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 4.0]], dtype=float)],
        shape_type=["polygon"],
        name="roi_shapes",
        edge_color=["yellow"],
        face_color=["transparent"],
    )
    shapes.features = {"label": ["roi_a"]}

    destination = tmp_path / "workspace_transparent_shapes.json"
    save_workspace_manifest(viewer, destination)

    restored_viewer = make_napari_viewer_proxy()
    load_result = load_workspace_manifest(restored_viewer, destination)

    assert load_result["skipped_layers"] == []
    restored = restored_viewer.layers["roi_shapes"]
    assert len(restored.data) == 1
    assert np.asarray(restored.face_color).shape[0] == 1


def test_workspace_manifest_round_trips_points_with_features(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    points = viewer.add_points(
        np.array([[5.0, 6.0], [10.0, 12.0]], dtype=float),
        name="sam2_prompts",
        size=[8, 12],
        symbol=["o", "square"],
        face_color=["green", "red"],
        border_color=["white", "yellow"],
        shown=[True, False],
        out_of_slice_display=True,
        features={"sam_label": ["positive", "negative"]},
    )
    points.scale = (0.5, 0.5)

    destination = tmp_path / "workspace_points.json"
    save_result = save_workspace_manifest(viewer, destination)

    assert save_result["saved_layers"] == 1
    assert save_result["skipped_layers"] == []

    restored_viewer = make_napari_viewer_proxy()
    load_result = load_workspace_manifest(restored_viewer, destination)

    assert load_result["skipped_layers"] == []
    assert "sam2_prompts" in restored_viewer.layers
    restored = restored_viewer.layers["sam2_prompts"]
    assert np.allclose(np.asarray(restored.data), np.array([[5.0, 6.0], [10.0, 12.0]], dtype=float))
    assert list(restored.features["sam_label"]) == ["positive", "negative"]
    assert list(np.asarray(restored.shown, dtype=bool)) == [True, False]
    assert tuple(restored.scale) == (0.5, 0.5)


def test_workspace_manifest_repeated_save_replaces_assets_cleanly(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    destination = tmp_path / "workspace_repeat.json"

    viewer.add_image(np.ones((8, 8), dtype=np.float32), name="image_a")
    save_workspace_manifest(viewer, destination)

    viewer.layers.clear()
    viewer.add_image(np.full((8, 8), 2.0, dtype=np.float32), name="image_b")
    save_workspace_manifest(viewer, destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert [record["name"] for record in payload["layers"]] == ["image_b"]

    restored_viewer = make_napari_viewer_proxy()
    load_result = load_workspace_manifest(restored_viewer, destination)

    assert load_result["skipped_layers"] == []
    assert "image_b" in restored_viewer.layers
    assert "image_a" not in restored_viewer.layers
    assert np.allclose(np.asarray(restored_viewer.layers["image_b"].data), 2.0)
