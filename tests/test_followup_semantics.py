from __future__ import annotations

from napari_chat_assistant.agent.followup_semantics import (
    detect_followup_constraint,
    parse_followup_constraint,
)


def test_detect_followup_constraint_for_same_as_before():
    assert detect_followup_constraint("same as before") is True


def test_parse_followup_constraint_for_labels_target_switch():
    parsed = parse_followup_constraint("same as before but for labels")

    assert parsed["is_followup"] is True
    assert parsed["reuse_previous"] is True
    assert parsed["change_target"] == "labels"


def test_parse_followup_constraint_for_selected_layer_reference():
    parsed = parse_followup_constraint("use the selected one")

    assert parsed["target_layer_reference"] == "selected_layer"


def test_parse_followup_constraint_for_widget_negation():
    parsed = parse_followup_constraint("don't use widget")

    assert "widget" in parsed["avoid_tools"]
    assert "explicit_exclusion" in parsed["negations"]


def test_parse_followup_constraint_for_speed_modifier():
    parsed = parse_followup_constraint("same workflow but faster")

    assert parsed["reuse_previous"] is True
    assert "faster" in parsed["modifiers"]


def test_parse_followup_constraint_for_scope_change():
    parsed = parse_followup_constraint("same as before, but RGB instead")

    assert parsed["change_scope"] == "rgb"


def test_parse_followup_constraint_for_explain_only():
    parsed = parse_followup_constraint("just explain")

    assert parsed["requested_mode"] == "reply"
