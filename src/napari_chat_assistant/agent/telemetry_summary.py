from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import median

from napari_chat_assistant.agent.logging_utils import TELEMETRY_LOG_PATH


def load_telemetry_events(path: Path | None = None) -> tuple[list[dict], int]:
    source = path or TELEMETRY_LOG_PATH
    if not source.exists():
        return [], 0

    events: list[dict] = []
    invalid_lines = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            if not isinstance(parsed, dict):
                invalid_lines += 1
                continue
            parsed["_line_number"] = line_number
            events.append(parsed)
    return events, invalid_lines


def read_telemetry_tail(path: Path | None = None, *, max_lines: int = 200) -> str:
    source = path or TELEMETRY_LOG_PATH
    if not source.exists():
        return ""

    recent_lines: deque[str] = deque(maxlen=max(1, int(max_lines)))
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            recent_lines.append(raw_line.rstrip("\n"))
    return "\n".join(recent_lines)


def summarize_telemetry_events(events: list[dict], invalid_lines: int = 0) -> dict:
    models = Counter()
    actions = Counter()
    latencies: list[int] = []
    model_turns = Counter()
    model_latencies: dict[str, list[int]] = defaultdict(list)
    model_actions: dict[str, Counter] = defaultdict(Counter)
    model_rejects = Counter()
    model_code_success = Counter()
    model_code_failure = Counter()
    tool_failures = 0
    code_success = 0
    code_failure = 0
    reject_feedback = 0
    latest_errors: list[str] = []
    first_timestamp = ""
    last_timestamp = ""

    for event in events:
        timestamp = str(event.get("timestamp", "")).strip()
        if timestamp and not first_timestamp:
            first_timestamp = timestamp
        if timestamp:
            last_timestamp = timestamp

        model = str(event.get("model", "")).strip()
        if model:
            models[model] += 1

        event_type = str(event.get("event_type", "")).strip()
        if event_type == "turn_completed":
            response_action = str(event.get("response_action", "")).strip() or "unknown"
            actions[response_action] += 1
            if model:
                model_turns[model] += 1
                model_actions[model][response_action] += 1
            latency_ms = event.get("latency_ms")
            if isinstance(latency_ms, (int, float)):
                latency_value = int(latency_ms)
                latencies.append(latency_value)
                if model:
                    model_latencies[model].append(latency_value)
            if response_action == "error":
                error_text = str(event.get("error", "")).strip()
                if error_text:
                    latest_errors.append(error_text)

            if event.get("tool_success") is False:
                tool_failures += 1
                error_text = str(event.get("error", "")).strip()
                if error_text:
                    latest_errors.append(error_text)

        if event_type == "turn_feedback" and str(event.get("feedback", "")).strip() == "reject":
            reject_feedback += 1
            if model:
                model_rejects[model] += 1

        if event_type == "code_execution":
            if event.get("success") is True:
                code_success += 1
                if model:
                    model_code_success[model] += 1
            elif event.get("success") is False:
                code_failure += 1
                if model:
                    model_code_failure[model] += 1
                error_text = str(event.get("error", "")).strip()
                if error_text:
                    latest_errors.append(error_text)

    latest_errors = [text for text in latest_errors if text][-3:]
    per_model: list[dict] = []
    for model_name, turn_count in model_turns.most_common():
        latencies_for_model = model_latencies.get(model_name, [])
        per_model.append(
            {
                "model": model_name,
                "turns": turn_count,
                "actions": model_actions.get(model_name, Counter()),
                "latency_count": len(latencies_for_model),
                "latency_median_ms": int(median(latencies_for_model)) if latencies_for_model else None,
                "latency_max_ms": max(latencies_for_model) if latencies_for_model else None,
                "rejects": model_rejects.get(model_name, 0),
                "reject_rate": model_rejects.get(model_name, 0) / turn_count if turn_count else None,
                "code_success": model_code_success.get(model_name, 0),
                "code_failure": model_code_failure.get(model_name, 0),
            }
        )

    ranking = build_model_ranking(per_model)

    return {
        "total_events": len(events),
        "invalid_lines": int(invalid_lines),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "models": models,
        "actions": actions,
        "turn_completed": sum(actions.values()),
        "latency_count": len(latencies),
        "latency_median_ms": int(median(latencies)) if latencies else None,
        "latency_max_ms": max(latencies) if latencies else None,
        "tool_failures": tool_failures,
        "code_success": code_success,
        "code_failure": code_failure,
        "reject_feedback": reject_feedback,
        "latest_errors": latest_errors,
        "per_model": per_model,
        "ranking": ranking,
    }


def build_model_ranking(per_model: list[dict]) -> dict:
    ranked_latency = [item for item in per_model if item.get("latency_count")]
    ranked_rejects = [item for item in per_model if item.get("turns")]
    ranked_code = [
        item
        for item in per_model
        if (item.get("code_success", 0) + item.get("code_failure", 0)) > 0
    ]

    ranking: dict[str, dict | None] = {
        "fastest": None,
        "slowest": None,
        "most_used": per_model[0] if per_model else None,
        "lowest_reject_rate": None,
        "highest_reject_rate": None,
        "best_code_success": None,
        "worst_code_success": None,
    }
    if ranked_latency:
        ranking["fastest"] = min(
            ranked_latency,
            key=lambda item: (item.get("latency_median_ms", float("inf")), -item.get("turns", 0)),
        )
        ranking["slowest"] = max(
            ranked_latency,
            key=lambda item: (item.get("latency_median_ms", float("-inf")), item.get("turns", 0)),
        )
    if ranked_rejects:
        ranking["lowest_reject_rate"] = min(
            ranked_rejects,
            key=lambda item: (item.get("reject_rate", 0.0), -item.get("turns", 0)),
        )
        ranking["highest_reject_rate"] = max(
            ranked_rejects,
            key=lambda item: (item.get("reject_rate", 0.0), item.get("turns", 0)),
        )
    if ranked_code:
        ranking["best_code_success"] = max(
            ranked_code,
            key=lambda item: (
                item.get("code_success", 0)
                / max(1, item.get("code_success", 0) + item.get("code_failure", 0)),
                item.get("code_success", 0) + item.get("code_failure", 0),
            ),
        )
        ranking["worst_code_success"] = min(
            ranked_code,
            key=lambda item: (
                item.get("code_success", 0)
                / max(1, item.get("code_success", 0) + item.get("code_failure", 0)),
                -(item.get("code_success", 0) + item.get("code_failure", 0)),
            ),
        )
    return ranking


def format_telemetry_summary(summary: dict) -> str:
    if not summary.get("total_events"):
        return (
            "**Telemetry Summary**\n"
            "- No telemetry events recorded yet.\n"
            f"- Path: `{TELEMETRY_LOG_PATH}`"
        )

    model_bits = [
        f"`{name}` ({count})"
        for name, count in summary.get("models", {}).most_common(5)
    ]
    action_bits = [
        f"`{name}` ({count})"
        for name, count in summary.get("actions", {}).most_common()
    ]

    lines = [
        "**Telemetry Summary**",
        f"- Path: `{TELEMETRY_LOG_PATH}`",
        f"- Records: {summary.get('total_events', 0)}",
    ]
    if summary.get("first_timestamp") or summary.get("last_timestamp"):
        lines.append(
            f"- Time range: `{summary.get('first_timestamp', '')}` to `{summary.get('last_timestamp', '')}`"
        )
    if model_bits:
        lines.append(f"- Models: {', '.join(model_bits)}")
    if action_bits:
        lines.append(f"- Completed turns: {summary.get('turn_completed', 0)} via {', '.join(action_bits)}")
    if summary.get("latency_count"):
        lines.append(
            "- Latency: median "
            f"{summary.get('latency_median_ms')} ms, max {summary.get('latency_max_ms')} ms"
        )
    ranking = summary.get("ranking", {})
    if ranking:
        lines.append("**Quick Ranking**")
        fastest = ranking.get("fastest")
        if fastest:
            lines.append(
                f"- Fastest: `{fastest.get('model', '')}` by median latency "
                f"({fastest.get('latency_median_ms')} ms across {fastest.get('turns', 0)} completed turns)"
            )
        slowest = ranking.get("slowest")
        if slowest:
            lines.append(
                f"- Slowest: `{slowest.get('model', '')}` by median latency "
                f"({slowest.get('latency_median_ms')} ms across {slowest.get('turns', 0)} completed turns)"
            )
        most_used = ranking.get("most_used")
        if most_used:
            lines.append(
                f"- Most used: `{most_used.get('model', '')}` "
                f"({most_used.get('turns', 0)} completed turns)"
            )
        lowest_reject_rate = ranking.get("lowest_reject_rate")
        if lowest_reject_rate:
            lines.append(
                f"- Best feedback signal: `{lowest_reject_rate.get('model', '')}` "
                f"({int(round(100 * (lowest_reject_rate.get('reject_rate') or 0.0)))}% reject rate)"
            )
        highest_reject_rate = ranking.get("highest_reject_rate")
        if highest_reject_rate:
            lines.append(
                f"- Weakest feedback signal: `{highest_reject_rate.get('model', '')}` "
                f"({int(round(100 * (highest_reject_rate.get('reject_rate') or 0.0)))}% reject rate)"
            )
        best_code_success = ranking.get("best_code_success")
        if best_code_success:
            best_total = best_code_success.get("code_success", 0) + best_code_success.get("code_failure", 0)
            best_rate = int(round(100 * best_code_success.get("code_success", 0) / max(1, best_total)))
            lines.append(
                f"- Best code-run signal: `{best_code_success.get('model', '')}` "
                f"({best_rate}% success over {best_total} executions)"
            )
        worst_code_success = ranking.get("worst_code_success")
        if worst_code_success:
            worst_total = worst_code_success.get("code_success", 0) + worst_code_success.get("code_failure", 0)
            worst_rate = int(round(100 * worst_code_success.get("code_success", 0) / max(1, worst_total)))
            lines.append(
                f"- Weakest code-run signal: `{worst_code_success.get('model', '')}` "
                f"({worst_rate}% success over {worst_total} executions)"
            )
    per_model = summary.get("per_model", [])
    if per_model:
        lines.append("**Per-Model Turn Summary**")
        for item in per_model:
            action_counts = [
                f"`{name}` ({count})"
                for name, count in item.get("actions", Counter()).most_common()
            ]
            latency_text = "no latency recorded"
            if item.get("latency_count"):
                latency_text = (
                    f"median {item.get('latency_median_ms')} ms, max {item.get('latency_max_ms')} ms"
                )
            reject_text = f"{int(round(100 * (item.get('reject_rate') or 0.0)))}% reject"
            code_total = item.get("code_success", 0) + item.get("code_failure", 0)
            code_text = "no code runs"
            if code_total:
                code_text = f"{int(round(100 * item.get('code_success', 0) / max(1, code_total)))}% code-run success"
            lines.append(
                f"- `{item.get('model', '')}`: {item.get('turns', 0)} completed turns; "
                f"{latency_text}; {reject_text}; {code_text}; actions: "
                f"{', '.join(action_counts) if action_counts else 'none'}"
            )
    lines.append(
        "- Code execution: "
        f"{summary.get('code_success', 0)} succeeded, {summary.get('code_failure', 0)} failed"
    )
    lines.append(f"- Reject feedback: {summary.get('reject_feedback', 0)}")
    if summary.get("tool_failures"):
        lines.append(f"- Tool failures: {summary.get('tool_failures', 0)}")
    if summary.get("invalid_lines"):
        lines.append(f"- Invalid JSONL lines skipped: {summary.get('invalid_lines', 0)}")
    if summary.get("latest_errors"):
        lines.append("**Recent Errors**")
        lines.extend(f"- {text}" for text in summary["latest_errors"])
    return "\n".join(lines)
