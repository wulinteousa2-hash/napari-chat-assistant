from __future__ import annotations

from napari_chat_assistant.agent.code_validation import build_code_repair_context
from napari_chat_assistant.agent.ui_help import answer_ui_question


def test_answer_ui_question_does_not_match_log_inside_biological():
    reply = answer_ui_question("what is the image about? asking biological question")

    assert reply is None


def test_answer_ui_question_matches_telemetry_log_phrase():
    reply = answer_ui_question("What is telemetry log?")

    assert reply is not None
    assert "**Log**" in reply
    assert "raw local telemetry records" in reply


def test_answer_ui_question_matches_actions_tab():
    reply = answer_ui_question("What is Actions tab?")

    assert reply is not None
    assert "**Actions Tab**" in reply
    assert "deterministic built-in functions" in reply


def test_answer_ui_question_matches_shortcuts():
    reply = answer_ui_question("What are shortcuts?")

    assert reply is not None
    assert "**Shortcuts**" in reply
    assert "one-click action buttons" in reply


def test_answer_ui_question_matches_roi_intensity_analysis():
    reply = answer_ui_question("What is ROI intensity analysis?")

    assert reply is not None
    assert "**ROI Intensity Analysis**" in reply
    assert "area-ROI measurement" in reply


def test_answer_ui_question_matches_report_bug():
    reply = answer_ui_question("What is report bug?")

    assert reply is not None
    assert "**Report Bug**" in reply
    assert "support email" in reply


def test_answer_ui_question_does_not_hijack_polite_action_request():
    reply = answer_ui_question("Can you apply spacing 2 in grid view?")

    assert reply is None


def test_answer_ui_question_does_not_hijack_add_annotation_grid_request():
    reply = answer_ui_question("add annotation to the grid view with low, mid, high SNR?")

    assert reply is None


def test_answer_ui_question_does_not_match_bare_montage_alias():
    reply = answer_ui_question("What is montage?")

    assert reply is None


def test_answer_ui_question_does_not_match_structured_multiline_helpish_text():
    text = """
Can you explain this?

Grid Spacing
Purpose: Controls the visual gap between tiles in image grid view.
Tip: Ask for spacing 0.
"""

    assert answer_ui_question(text) is None


def test_code_refine_prompt_with_ui_help_alias_is_not_ui_help(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    text = """
Refine this code so it runs in the current napari plugin environment.
Preserve the original intent, explain the main fix briefly, and return corrected runnable Python.
If the code contains placeholder or template layer names, replace them with the best matching current viewer layers automatically.

Grid Spacing
Purpose: Controls the visual gap between tiles in image grid view.

Code:
```python
prepared = prepare_tool_job(
    viewer,
    "show_image_layers_in_grid",
    {"layer_names": ["img_a", "img_b"], "spacing": 2},
)
```
    """
    assert build_code_repair_context(text, viewer=viewer) is not None
    assert answer_ui_question(text) is None
