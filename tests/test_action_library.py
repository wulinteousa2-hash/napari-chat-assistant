from __future__ import annotations

from napari_chat_assistant.agent.action_library import action_library_payload


def test_action_library_payload_has_categories_and_actions():
    payload = action_library_payload()

    assert "Widgets" in payload["categories"]
    assert "Workspace" in payload["categories"]
    assert "Annotation" in payload["categories"]
    assert "Enhance" in payload["categories"]
    assert "Montage" in payload["categories"]
    assert "Segmentation" in payload["categories"]
    assert "actions" in payload
    assert payload["actions"]


def test_action_library_contains_function_and_tool_actions():
    payload = action_library_payload()
    actions = payload["actions"]

    kinds = {str(action.get("execution", {}).get("kind", "")).strip() for action in actions}
    titles = {str(action.get("title", "")).strip() for action in actions}
    groups = {str(action.get("group", "")).strip() for action in actions if str(action.get("group", "")).strip()}

    assert "function" in kinds
    assert "tool" in kinds
    assert "ROI Intensity Analysis" in titles
    assert "Line Profile Analysis" in titles
    assert "Group Comparison Statistics" in titles
    assert "Delete All" in titles
    assert "Save Workspace" in titles
    assert "Save Workspace As" in titles
    assert "Load Workspace" in titles
    assert "Restore Last Workspace" in titles
    assert "Add Text Annotation" in titles
    assert "List Text Annotations" in titles
    assert "Numbered Callouts" in titles
    assert "Title Label" in titles
    assert "Rename Text Annotation" in titles
    assert "Delete Text Annotation" in titles
    assert "Hide All" in titles
    assert "Isolate Selected" in titles
    assert "Set Uniform Scale 0.1" in titles
    assert "Reset Uniform Scale" in titles
    assert "Gaussian Blur" in titles
    assert "Threshold Preview" in titles
    assert "Spectral Viewer" in titles
    assert "Make Analysis Montage" in titles
    assert "Auto Segment" in titles
    assert "Segment Anything 2 Setup" in titles
    assert "Segment Anything 2 Live" in titles
    assert "Initialize Point Prompts" in titles
    assert "Segment From Points" in titles
    assert "Propagate Through Z" in titles
    assert "SAM2 Points" in groups
    assert "SAM2 Session" in groups
