from __future__ import annotations

import json
import types
import numpy as np

from napari_chat_assistant.agent.workspace_state import (
    apply_workspace_viewer_state,
    load_workspace_manifest,
    read_workspace_manifest,
    restore_workspace_layer,
    save_workspace_manifest,
    workspace_record_loading_kind,
)


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


def test_workspace_restore_helpers_support_staged_ui_loading(make_napari_viewer_proxy, tmp_path):
    manifest_path = tmp_path / "workspace_inline.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "viewer": {"dims_current_step": [0, 0], "selected_layer_name": "inline_image"},
                "layers": [
                    {
                        "layer_type": "Image",
                        "name": "inline_image",
                        "inline_data": [[1, 2], [3, 4]],
                        "dtype": "float32",
                        "shape": [2, 2],
                        "visible": True,
                        "opacity": 0.4,
                        "blending": "translucent",
                        "scale": [1.5, 2.5],
                        "translate": [10.0, 20.0],
                        "contrast_limits": [1.0, 4.0],
                        "colormap": "gray",
                        "gamma": 0.7,
                    }
                ],
                "skipped_layers": [],
            }
        ),
        encoding="utf-8",
    )

    restored_viewer = make_napari_viewer_proxy()
    path, payload = read_workspace_manifest(manifest_path)
    layer = restore_workspace_layer(restored_viewer, payload["layers"][0], manifest_path=path)
    apply_workspace_viewer_state(restored_viewer, payload)

    assert layer is not None
    restored = restored_viewer.layers["inline_image"]
    assert np.array_equal(np.asarray(restored.data), np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert tuple(restored.scale) == (1.5, 2.5)
    assert tuple(restored.translate) == (10.0, 20.0)
    assert tuple(restored.contrast_limits) == (1.0, 4.0)
    assert float(restored.gamma) == 0.7
    assert restored_viewer.layers.selection.active.name == restored.name
    assert tuple(restored_viewer.dims.current_step) == (0, 0)


def test_workspace_record_loading_kind_prioritizes_source_types():
    assert workspace_record_loading_kind({"inline_kind": "Shapes"}) == "inline"
    assert workspace_record_loading_kind({"asset_path": "layer_000.ome.zarr"}) == "asset"
    assert workspace_record_loading_kind({"source": {"path": "/data/sample.ome.zarr"}}) == "source"
    assert (
        workspace_record_loading_kind(
            {"source_recipe": {"kind": "spectral_view", "source_path": "/data/sample.ome.zarr"}}
        )
        == "recipe"
    )


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


def test_workspace_manifest_sanitizes_non_json_metadata(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="metadata_image")
    image.metadata = {
        "source_path": "/tmp/example.ome.zarr",
        "non_json_value": object(),
    }

    destination = tmp_path / "workspace_metadata.json"
    save_result = save_workspace_manifest(viewer, destination)

    assert save_result["saved_layers"] == 1
    payload = json.loads(destination.read_text(encoding="utf-8"))
    metadata = payload["layers"][0]["microscopy_metadata"]["source_metadata"]
    assert metadata["source_path"] == "/tmp/example.ome.zarr"
    assert metadata["non_json_value"]["python_type"] == "builtins.object"


def test_workspace_manifest_saves_spectral_derived_layer_as_recipe(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.ones((4, 4, 3), dtype=np.uint8), name="sample truecolor preview", rgb=True)
    image.metadata = {
        "source_path": "/data/sample.ome.zarr",
        "dataset_metadata": {"is_spectral": True},
        "wavelengths_nm": [450.0, 550.0, 650.0],
        "spectral_cube": np.ones((3, 4, 4), dtype=np.float32),
    }

    destination = tmp_path / "workspace_spectral.json"
    save_workspace_manifest(viewer, destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    record = payload["layers"][0]
    assert record["source_recipe"]["kind"] == "spectral_view"
    assert record["source_recipe"]["source_path"] == "/data/sample.ome.zarr"
    assert record["source_recipe"]["view_type"] == "truecolor"
    assert "asset_path" not in record


def test_workspace_manifest_restores_spectral_recipe_via_reader_builder(make_napari_viewer_proxy, tmp_path, monkeypatch):
    viewer = make_napari_viewer_proxy()
    manifest_path = tmp_path / "workspace_recipe.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "viewer": {"dims_current_step": [0, 0], "selected_layer_name": "sample visible sum preview"},
                "layers": [
                    {
                        "layer_type": "Image",
                        "name": "sample visible sum preview",
                        "visible": True,
                        "opacity": 1.0,
                        "blending": "translucent",
                        "scale": [1.0, 1.0],
                        "translate": [0.0, 0.0],
                        "source": {"path": None, "reader_plugin": None},
                        "source_recipe": {
                            "kind": "spectral_view",
                            "source_path": "/data/sample.ome.zarr",
                            "reader_plugin": "napari-nd2-spectral-ome-zarr",
                            "view_type": "visible_sum",
                            "zarr_use_preview": True,
                        },
                    }
                ],
                "skipped_layers": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_import_module(name: str):
        if name != "napari_nd2_spectral_ome_zarr._reader":
            raise AssertionError(name)

        def build_layer_data(source_path, **kwargs):
            assert source_path == "/data/sample.ome.zarr"
            assert kwargs["include_visible_layer"] is True
            assert kwargs["include_truecolor_layer"] is False
            assert kwargs["include_raw_layer"] is False
            return [
                (
                    np.full((3, 3), 7, dtype=np.float32),
                    {"name": "rebuilt visible sum", "metadata": {"source_path": source_path}},
                    "image",
                )
            ]

        return types.SimpleNamespace(build_layer_data=build_layer_data)

    monkeypatch.setattr("napari_chat_assistant.agent.workspace_state.importlib.import_module", fake_import_module)

    result = load_workspace_manifest(viewer, manifest_path)

    assert result["skipped_layers"] == []
    assert "sample visible sum preview" in viewer.layers
    restored = viewer.layers["sample visible sum preview"]
    assert np.array_equal(np.asarray(restored.data), np.full((3, 3), 7, dtype=np.float32))


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


def test_workspace_manifest_round_trips_points_text_annotations(make_napari_viewer_proxy, tmp_path):
    viewer = make_napari_viewer_proxy()
    points = viewer.add_points(
        np.array([[5.0, 6.0], [10.0, 12.0]], dtype=float),
        name="image_a_text_annotations",
        features={"label": ["cell A", "cell B"]},
        size=6,
        face_color="transparent",
        border_color="yellow",
        border_width=1,
    )
    points.feature_defaults = {"label": "cell C"}
    points.text = {
        "string": "{label}",
        "size": 12,
        "color": "yellow",
        "anchor": "upper_left",
        "translation": [0.0, -6.0],
        "blending": "translucent",
        "visible": True,
        "scaling": False,
        "rotation": 0.0,
    }
    points.metadata = {
        "text_annotation_text_style": {
            "string": "{label}",
            "size": 12,
            "color": "yellow",
            "anchor": "upper_left",
            "translation": [0.0, -6.0],
            "blending": "translucent",
            "visible": True,
            "scaling": False,
            "rotation": 0.0,
        }
    }

    destination = tmp_path / "workspace_text_annotations.json"
    save_workspace_manifest(viewer, destination)

    restored_viewer = make_napari_viewer_proxy()
    load_result = load_workspace_manifest(restored_viewer, destination)

    assert load_result["skipped_layers"] == []
    restored = restored_viewer.layers["image_a_text_annotations"]
    assert list(restored.features["label"]) == ["cell A", "cell B"]
    assert getattr(restored.text.string, "format", None) == "{label}"
    assert str(restored.text.anchor) == "upper_left"


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
