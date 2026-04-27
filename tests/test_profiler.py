from __future__ import annotations

import numpy as np

from napari_chat_assistant.agent.context import layer_context_for_model, layer_context_json
from napari_chat_assistant.agent.profiler import profile_layer


def test_profile_labels_layer_as_high_confidence_mask(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_labels((np.arange(25).reshape(5, 5) > 10).astype(np.uint8), name="mask_a")

    profile = profile_layer(layer)

    assert profile["semantic_type"] == "label_mask"
    assert profile["confidence"] == "high"
    assert profile["source_kind"] == "napari_labels_layer"
    assert profile["axes_detected"] == "YX"
    assert "mask_measurement" in profile["recommended_operation_classes"]
    assert "semantic_napari" in profile["evidence"]
    assert any("Labels" in reason for reason in profile["evidence"]["semantic_napari"])


def test_profile_rgb_layer_from_napari_rgb_semantics(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    rgb = np.zeros((6, 7, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    layer = viewer.add_image(rgb, name="rgb_a", rgb=True)

    profile = profile_layer(layer)

    assert profile["semantic_type"] == "rgb"
    assert profile["confidence"] == "high"
    assert profile["axes_detected"] == "YXC"
    assert "visual_review" in profile["recommended_operation_classes"]


def test_profile_probability_map_from_bounded_float_image(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)
    layer = viewer.add_image(data, name="prob_a")

    profile = profile_layer(layer)

    assert profile["semantic_type"] == "probability_map"
    assert profile["confidence"] in {"medium", "high"}
    assert profile["dtype"] == "float32"
    assert "thresholding" in profile["recommended_operation_classes"]


def test_profile_multichannel_fluorescence_from_channel_metadata(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    data = np.zeros((3, 8, 8), dtype=np.float32)
    layer = viewer.add_image(
        data,
        name="fluor_a",
        metadata={"axes": "CYX", "channel_names": ["DAPI", "FITC", "TRITC"]},
    )

    profile = profile_layer(layer)

    assert profile["semantic_type"] == "multichannel_fluorescence"
    assert profile["channel_metadata_present"] is True
    assert profile["axes_detected"] == "CYX"
    assert "channel_selection" in profile["recommended_operation_classes"]
    assert any("Channel metadata present" in reason for reason in profile["evidence"]["format_metadata"])


def test_layer_context_json_exposes_selected_layer_profile(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.random.default_rng(0).random((4, 5), dtype=np.float32), name="image_a")
    selected = viewer.add_image(
        np.zeros((4, 3, 5, 5), dtype=np.float32),
        name="time_a",
        metadata={"axes": "TCYX", "frame_interval": 2.5},
    )
    viewer.layers.selection.active = selected

    payload = layer_context_json(viewer)

    assert payload["selected_layer"] == "time_a"
    assert payload["selected_layer_profile"]["semantic_type"] == "time_series"
    assert payload["selected_layer_profile"]["time_calibration_present"] is True
    assert "evidence" in payload["selected_layer_profile"]
    assert len(payload["layers"]) == 2
    assert all("profile" in item for item in payload["layers"])


def test_layer_context_for_model_uses_compact_profiles(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.random.default_rng(0).random((4, 5), dtype=np.float32), name="image_a")
    selected = viewer.add_image(
        np.zeros((4, 3, 5, 5), dtype=np.float32),
        name="time_a",
        metadata={"axes": "TCYX", "frame_interval": 2.5},
    )
    viewer.layers.selection.active = selected

    payload = layer_context_for_model(viewer)

    assert payload["selected_layer"] == "time_a"
    assert payload["selected_layer_profile"]["name"] == "time_a"
    assert payload["selected_layer_profile"]["semantic_type"] == "time_series"
    assert payload["selected_layer_profile"]["axes_detected"] == "TCYX"
    assert payload["selected_layer_profile"]["time_calibration_present"] is True
    assert "evidence" not in payload["selected_layer_profile"]
    assert "reasons" not in payload["selected_layer_profile"]
    assert len(payload["layers"]) == 2
    assert all("profile" not in item for item in payload["layers"])
    assert all("recommended_operation_classes" in item for item in payload["layers"])
