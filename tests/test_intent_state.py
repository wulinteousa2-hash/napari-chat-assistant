from __future__ import annotations

from napari_chat_assistant.agent.intent_state import (
    empty_failed_tool_state,
    empty_intent_state,
    extract_turn_intent,
    merge_intent_state,
    remember_failed_tool,
    should_block_tool,
    should_skip_local_workflow_route,
)


def test_extract_turn_intent_prefers_code_for_explicit_montage_override():
    failed = remember_failed_tool(
        "create_analysis_montage",
        "Need at least 2 grayscale 2D image layers to build an analysis montage.",
    )

    state = extract_turn_intent(
        "generate code similar to create analysis montage, but for labels and shapes because the built-in only takes images",
        last_failed_tool_state=failed,
    )

    assert state["mode_preference"] == "code"
    assert "labels" in state["target_layer_types"]
    assert "shapes" in state["target_layer_types"]
    assert "create_analysis_montage" in state["blocked_tools"]


def test_extract_turn_intent_detects_explain_mode():
    state = extract_turn_intent("Why does this code fail?")

    assert state["mode_preference"] == "explain"


def test_merge_intent_state_carries_forward_constraints_on_followup():
    previous = {
        **empty_intent_state(),
        "mode_preference": "code",
        "blocked_tools": ["create_analysis_montage"],
        "target_layer_types": ["labels", "shapes"],
        "reason": "built-in montage is image-only",
    }
    current = extract_turn_intent("same workflow, but for points instead")

    merged = merge_intent_state(previous, current)

    assert merged["mode_preference"] == "code"
    assert "create_analysis_montage" in merged["blocked_tools"]
    assert "points" in merged["target_layer_types"]


def test_should_skip_local_workflow_route_for_code_preference():
    state = {**empty_intent_state(), "mode_preference": "code"}

    assert should_skip_local_workflow_route(state) is True


def test_should_block_tool_for_blocked_montage():
    state = {**empty_intent_state(), "blocked_tools": ["create_analysis_montage"]}

    assert should_block_tool(state, "create_analysis_montage") is True
    assert should_block_tool(state, "gaussian_denoise") is False


def test_remember_failed_tool_detects_image_only_montage_failure():
    state = remember_failed_tool(
        "create_analysis_montage",
        "No usable 2D image layers were available to build the analysis montage.",
    )

    assert state["tool_name"] == "create_analysis_montage"
    assert state["supported_layer_types"] == ["image"]


def test_empty_failed_tool_state_is_blank():
    assert empty_failed_tool_state()["tool_name"] == ""
