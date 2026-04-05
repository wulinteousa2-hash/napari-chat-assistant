from __future__ import annotations

from napari_chat_assistant.agent.recent_action_state import (
    empty_recent_action_state,
    latest_recent_action,
    refine_threshold_action,
    record_recent_action,
    route_recent_action_followup,
    threshold_adjustment_direction,
)


def test_record_recent_action_keeps_latest_first():
    state = empty_recent_action_state(max_items=3)
    state = record_recent_action(
        state,
        {
            "tool_name": "gaussian_denoise",
            "action_kind": "enhancement",
            "turn_id": "turn-1",
            "message": "Applied gaussian denoising.",
        },
    )
    state = record_recent_action(
        state,
        {
            "tool_name": "apply_threshold",
            "action_kind": "threshold",
            "turn_id": "turn-2",
            "message": "Applied threshold.",
            "parameters": {"threshold_value": 0.33, "foreground_mode": "bright"},
        },
    )

    latest = latest_recent_action(state)

    assert latest["tool_name"] == "apply_threshold"
    assert latest["action_kind"] == "threshold"
    assert latest["parameters"]["foreground_mode"] == "bright"


def test_record_recent_action_prunes_to_max_items():
    state = empty_recent_action_state(max_items=2)
    state = record_recent_action(state, {"tool_name": "tool_a", "action_kind": "a", "turn_id": "1", "message": "a"})
    state = record_recent_action(state, {"tool_name": "tool_b", "action_kind": "b", "turn_id": "2", "message": "b"})
    state = record_recent_action(state, {"tool_name": "tool_c", "action_kind": "c", "turn_id": "3", "message": "c"})

    assert [item["tool_name"] for item in state["items"]] == ["tool_c", "tool_b"]


def test_latest_recent_action_can_filter_by_kind():
    state = empty_recent_action_state(max_items=4)
    state = record_recent_action(state, {"tool_name": "create_synthetic_demo_image", "action_kind": "demo", "turn_id": "1", "message": "demo"})
    state = record_recent_action(state, {"tool_name": "apply_threshold", "action_kind": "threshold", "turn_id": "2", "message": "threshold"})

    latest_threshold = latest_recent_action(state, lambda item: item.get("action_kind") == "threshold")

    assert latest_threshold["tool_name"] == "apply_threshold"


def test_threshold_adjustment_direction_detects_stricter_and_looser_language():
    assert threshold_adjustment_direction("make it stricter") == "stricter"
    assert threshold_adjustment_direction("include more area") == "looser"


def test_refine_threshold_action_adjusts_cutoff_for_bright_foreground():
    route = refine_threshold_action(
        {
            "tool_name": "apply_threshold",
            "action_kind": "threshold",
            "turn_id": "2",
            "input_layers": ["image_a"],
            "parameters": {
                "threshold_value": 0.4,
                "foreground_mode": "bright",
                "foreground_mode_resolved": "bright",
                "image_min": 0.0,
                "image_max": 1.0,
            },
            "message": "Applied threshold.",
        },
        "stricter",
    )

    assert route["tool"] == "apply_threshold"
    assert route["arguments"]["layer_name"] == "image_a"
    assert route["arguments"]["polarity"] == "bright"
    assert route["arguments"]["threshold_value"] > 0.4


def test_refine_threshold_action_adjusts_cutoff_for_dim_foreground():
    route = refine_threshold_action(
        {
            "tool_name": "apply_threshold",
            "action_kind": "threshold",
            "turn_id": "2",
            "input_layers": ["image_a"],
            "parameters": {
                "threshold_value": 0.4,
                "foreground_mode": "dim",
                "foreground_mode_resolved": "dim",
                "image_min": 0.0,
                "image_max": 1.0,
            },
            "message": "Applied threshold.",
        },
        "looser",
    )

    assert route["tool"] == "apply_threshold"
    assert route["arguments"]["polarity"] == "dim"
    assert route["arguments"]["threshold_value"] > 0.4


def test_route_recent_action_followup_reuses_same_threshold_on_selected_image():
    state = empty_recent_action_state()
    state = record_recent_action(
        state,
        {
            "tool_name": "apply_threshold",
            "action_kind": "threshold",
            "turn_id": "2",
            "input_layers": ["image_a"],
            "parameters": {
                "threshold_value": 0.4,
                "foreground_mode": "bright",
                "foreground_mode_resolved": "bright",
                "image_min": 0.0,
                "image_max": 1.0,
            },
            "message": "Applied threshold.",
        },
    )

    route = route_recent_action_followup(
        "apply that to this image",
        state,
        selected_layer_name="image_b",
        selected_layer_type="image",
    )

    assert route["tool"] == "apply_threshold"
    assert route["arguments"]["layer_name"] == "image_b"
    assert route["arguments"]["threshold_value"] == 0.4


def test_route_recent_action_followup_hands_off_to_histogram_for_recent_image():
    state = empty_recent_action_state()
    state = record_recent_action(
        state,
        {
            "tool_name": "apply_threshold",
            "action_kind": "threshold",
            "turn_id": "2",
            "input_layers": ["image_a"],
            "output_layers": ["image_a_labels"],
            "parameters": {"threshold_value": 0.4, "foreground_mode": "bright", "foreground_mode_resolved": "bright"},
            "message": "Applied threshold.",
        },
    )

    route = route_recent_action_followup("show histogram for that result", state)

    assert route["tool"] == "plot_histogram"
    assert route["arguments"]["layer_name"] == "image_a"


def test_route_recent_action_followup_hands_off_to_roi_widget_for_recent_image():
    state = empty_recent_action_state()
    state = record_recent_action(
        state,
        {
            "tool_name": "summarize_intensity",
            "action_kind": "measurement",
            "turn_id": "2",
            "input_layers": ["image_a"],
            "message": "Intensity Summary",
        },
    )

    route = route_recent_action_followup("open ROI intensity analysis for that image", state)

    assert route["tool"] == "open_intensity_metrics_table"
    assert route["arguments"]["layer_name"] == "image_a"
