from __future__ import annotations

from napari_chat_assistant.agent.telemetry_report import format_markdown_telemetry_report
from napari_chat_assistant.agent.telemetry_summary import summarize_telemetry_events


def test_format_markdown_telemetry_report_renders_publishable_table():
    events = [
        {
            "timestamp": "2026-03-30T10:00:01+00:00",
            "event_type": "turn_completed",
            "model": "m1",
            "response_action": "reply",
            "latency_ms": 120,
        },
        {
            "timestamp": "2026-03-30T10:01:01+00:00",
            "event_type": "turn_completed",
            "model": "m1",
            "response_action": "tool",
            "latency_ms": 350,
        },
        {
            "timestamp": "2026-03-30T10:02:01+00:00",
            "event_type": "code_execution",
            "model": "m1",
            "success": True,
        },
        {
            "timestamp": "2026-03-30T10:03:01+00:00",
            "event_type": "turn_completed",
            "model": "m2",
            "response_action": "error",
            "latency_ms": 900,
            "error": "request boom",
        },
        {
            "timestamp": "2026-03-30T10:04:01+00:00",
            "event_type": "turn_feedback",
            "model": "m1",
            "feedback": "reject",
        },
    ]

    summary = summarize_telemetry_events(events, invalid_lines=1)
    rendered = format_markdown_telemetry_report(summary, title="My Results")

    assert "# My Results" in rendered
    assert "not from a controlled benchmark suite" in rendered
    assert "- Records: 5" in rendered
    assert "- Completed turns: 3" in rendered
    assert "## Response Mix" in rendered
    assert "`reply` (1), `tool` (1), `error` (1)" in rendered
    assert "## Per-Model Summary" in rendered
    assert "| `m1` | 2 | 235 | 350 | 1 | 1 | 0 | 0 | 1 | 50% | 100% |" in rendered
    assert "| `m2` | 1 | 900 | 900 | 0 | 0 | 0 | 1 | 0 | 0% | - |" in rendered
    assert "## Quick Takeaways" in rendered
    assert "Fastest median latency: `m1` at 235 ms" in rendered
    assert "Most used: `m1` with 2 completed turns" in rendered
    assert "## Recent Errors" in rendered
    assert "- request boom" in rendered
