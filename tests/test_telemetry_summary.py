from __future__ import annotations

from napari_chat_assistant.telemetry import (
    format_telemetry_summary,
    load_telemetry_events,
    read_telemetry_tail,
    summarize_telemetry_events,
)


def test_load_telemetry_events_skips_invalid_json(tmp_path):
    telemetry_path = tmp_path / "model_telemetry.jsonl"
    telemetry_path.write_text(
        '{"timestamp":"2026-03-30T10:00:00+00:00","event_type":"turn_started","model":"m1"}\n'
        'not json\n'
        '["wrong-shape"]\n'
        '{"timestamp":"2026-03-30T10:00:01+00:00","event_type":"turn_completed","model":"m1","response_action":"reply","latency_ms":120}\n',
        encoding="utf-8",
    )

    events, invalid_lines = load_telemetry_events(telemetry_path)

    assert len(events) == 2
    assert invalid_lines == 2
    assert events[0]["_line_number"] == 1
    assert events[1]["event_type"] == "turn_completed"


def test_summarize_and_format_telemetry_events_reports_core_metrics():
    events = [
        {"timestamp": "2026-03-30T10:00:00+00:00", "event_type": "turn_started", "model": "m1"},
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
            "tool_success": False,
            "error": "tool boom",
        },
        {
            "timestamp": "2026-03-30T10:02:01+00:00",
            "event_type": "code_execution",
            "model": "m1",
            "success": True,
        },
        {
            "timestamp": "2026-03-30T10:03:01+00:00",
            "event_type": "code_execution",
            "model": "m1",
            "success": False,
            "error": "exec boom",
        },
        {
            "timestamp": "2026-03-30T10:03:30+00:00",
            "event_type": "turn_completed",
            "model": "m2",
            "response_action": "reply",
            "latency_ms": 900,
        },
        {
            "timestamp": "2026-03-30T10:03:40+00:00",
            "event_type": "code_execution",
            "model": "m2",
            "success": True,
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
            "intent_category": "analysis",
            "intent_description": "analyze histogram intensity distribution",
            "success": False,
            "duration_ms": 3200,
            "feedback": "failed",
        },
        {
            "timestamp": "2026-03-30T10:04:30+00:00",
            "event_type": "intent_captured",
            "intent_category": "workflow",
            "intent_description": "save workspace session",
            "success": True,
            "duration_ms": 1800,
            "feedback": "helpful",
        },
        {
            "timestamp": "2026-03-30T10:04:40+00:00",
            "event_type": "turn_cancelled",
            "model": "m1",
            "cancel_bucket": "quick",
            "latency_ms": 4000,
        },
        {
            "timestamp": "2026-03-30T10:04:50+00:00",
            "event_type": "turn_cancelled",
            "model": "m2",
            "cancel_bucket": "long",
            "latency_ms": 25000,
        },
    ]

    summary = summarize_telemetry_events(events, invalid_lines=1)
    rendered = format_telemetry_summary(summary)

    assert summary["total_events"] == 13
    assert summary["turn_completed"] == 3
    assert summary["latency_median_ms"] == 350
    assert summary["code_success"] == 2
    assert summary["code_failure"] == 1
    assert summary["reject_feedback"] == 1
    assert summary["tool_failures"] == 1
    assert summary["cancel_total"] == 2
    assert summary["cancel_counts"]["quick"] == 1
    assert summary["cancel_counts"]["long"] == 1
    assert summary["intent_total"] == 3
    assert summary["per_intent"][0]["intent_category"] == "analysis"
    assert summary["per_intent"][0]["count"] == 2
    assert summary["per_intent"][0]["duration_median_ms"] == 2800
    assert summary["per_intent"][0]["success_rate"] == 0.5
    assert "intensity" in summary["per_intent"][0]["top_triggers"]
    assert "measure" in summary["per_intent"][0]["top_triggers"]
    assert "histogram" in summary["per_intent"][0]["top_triggers"]
    assert summary["intent_ranking"]["most_triggered"]["intent_category"] == "analysis"
    assert summary["intent_ranking"]["most_successful"]["intent_category"] == "workflow"
    assert summary["intent_ranking"]["least_successful"]["intent_category"] == "analysis"
    assert summary["per_model"][0]["model"] == "m1"
    assert summary["per_model"][0]["latency_median_ms"] == 235
    assert summary["per_model"][0]["reject_rate"] == 0.5
    assert summary["per_model"][1]["model"] == "m2"
    assert summary["per_model"][1]["latency_median_ms"] == 900
    assert summary["ranking"]["fastest"]["model"] == "m1"
    assert summary["ranking"]["slowest"]["model"] == "m2"
    assert summary["ranking"]["best_code_success"]["model"] == "m2"
    assert summary["ranking"]["worst_code_success"]["model"] == "m1"
    assert "Records: 13" in rendered
    assert "Completed turns: 3 via `reply` (2), `tool` (1)" in rendered
    assert "Intent events: 3 via `analysis` (2), `workflow` (1)" in rendered
    assert "Fastest: `m1` by median latency (235 ms across 2 completed turns)" in rendered
    assert "Slowest: `m2` by median latency (900 ms across 1 completed turns)" in rendered
    assert "Best code-run signal: `m2` (100% success over 1 executions)" in rendered
    assert "Weakest code-run signal: `m1` (50% success over 2 executions)" in rendered
    assert "`m1`: 2 completed turns; median 235 ms, max 350 ms; 50% reject; 50% code-run success; actions: `reply` (1), `tool` (1)" in rendered
    assert "`m2`: 1 completed turns; median 900 ms, max 900 ms; 0% reject; 100% code-run success; actions: `reply` (1)" in rendered
    assert "Most triggered intent: `analysis` (67% of captured intent events)" in rendered
    assert "Strongest routing fit: `workflow` (100% success over 1 captures)" in rendered
    assert "Weakest routing fit: `analysis` (50% success over 2 captures)" in rendered
    assert "`analysis`: 2 captures (67% trigger share); 50% success; median 2800 ms, max 3200 ms;" in rendered
    assert "feedback: `helpful` (1), `failed` (1);" in rendered
    assert "likely triggers:" in rendered
    assert "`workflow`: 1 captures (33% trigger share); 100% success; median 1800 ms, max 1800 ms; feedback: `helpful` (1);" in rendered
    assert "`save`" in rendered
    assert "`workspace`" in rendered
    assert "`session`" in rendered
    assert "Code execution: 2 succeeded, 1 failed" in rendered
    assert "Abandonment: `quick` (1), `long` (1)" in rendered
    assert "Invalid JSONL lines skipped: 1" in rendered
    assert "tool boom" in rendered
    assert "exec boom" in rendered


def test_read_telemetry_tail_returns_last_lines(tmp_path):
    telemetry_path = tmp_path / "model_telemetry.jsonl"
    telemetry_path.write_text("1\n2\n3\n4\n", encoding="utf-8")

    tail = read_telemetry_tail(telemetry_path, max_lines=2)

    assert tail == "3\n4"
