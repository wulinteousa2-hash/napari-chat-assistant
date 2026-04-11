from __future__ import annotations

from napari_chat_assistant.telemetry import format_markdown_telemetry_report, summarize_telemetry_events


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
        {
            "timestamp": "2026-03-30T10:04:10+00:00",
            "event_type": "intent_captured",
            "intent_category": "analysis",
            "intent_description": "measure roi intensity on image layer",
            "success": True,
            "duration_ms": 2400,
            "feedback": "helpful",
        },
        {
            "timestamp": "2026-03-30T10:04:20+00:00",
            "event_type": "intent_captured",
            "intent_category": "workflow",
            "intent_description": "save workspace session",
            "success": False,
            "duration_ms": 1800,
            "feedback": "failed",
        },
        {
            "timestamp": "2026-03-30T10:04:30+00:00",
            "event_type": "turn_cancelled",
            "model": "m1",
            "cancel_bucket": "long",
            "latency_ms": 18000,
        },
    ]

    summary = summarize_telemetry_events(events, invalid_lines=1)
    rendered = format_markdown_telemetry_report(summary, title="My Results")

    assert "# My Results" in rendered
    assert "not from a controlled benchmark suite" in rendered
    assert "- Records: 8" in rendered
    assert "- Completed turns: 3" in rendered
    assert "- Abandonment: `long` (1)" in rendered
    assert "## Response Mix" in rendered
    assert "`reply` (1), `tool` (1), `error` (1)" in rendered
    assert "## Intent Mix" in rendered
    assert "`analysis` (1), `workflow` (1)" in rendered
    assert "## Per-Model Summary" in rendered
    assert "| `m1` | 2 | 235 | 350 | 1 | 1 | 0 | 0 | 1 | 50% | 100% |" in rendered
    assert "| `m2` | 1 | 900 | 900 | 0 | 0 | 0 | 1 | 0 | 0% | - |" in rendered
    assert "## Intent Routing Signals" in rendered
    assert "| `analysis` | 1 | 50% | 100% | 2400 | helpful:1 | `measure`, `intensity`" in rendered
    assert "| `workflow` | 1 | 50% | 0% | 1800 | failed:1 | `save`, `workspace`, `session`" in rendered
    assert "## Quick Takeaways" in rendered
    assert "Fastest median latency: `m1` at 235 ms" in rendered
    assert "Most used: `m1` with 2 completed turns" in rendered
    assert "Most triggered intent: `analysis` at 50% of captured intent events" in rendered
    assert "Strongest routing fit: `analysis` at 100% success over 1 captures" in rendered
    assert "Weakest routing fit: `workflow` at 0% success over 1 captures" in rendered
    assert "## Recent Errors" in rendered
    assert "- request boom" in rendered
