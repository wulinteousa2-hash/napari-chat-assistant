from __future__ import annotations

from napari_chat_assistant.agent.ui_help import answer_ui_question


def test_answer_ui_question_does_not_match_log_inside_biological():
    reply = answer_ui_question("what is the image about? asking biological question")

    assert reply is None


def test_answer_ui_question_matches_telemetry_log_phrase():
    reply = answer_ui_question("What is telemetry log?")

    assert reply is not None
    assert "**Log**" in reply
    assert "raw local telemetry records" in reply
