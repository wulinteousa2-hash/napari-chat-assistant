from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import median

from napari_chat_assistant.telemetry.logging_utils import TELEMETRY_LOG_PATH


_INTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "apply",
    "can",
    "current",
    "for",
    "from",
    "how",
    "i",
    "image",
    "in",
    "is",
    "it",
    "layer",
    "make",
    "me",
    "my",
    "of",
    "on",
    "please",
    "show",
    "the",
    "this",
    "to",
    "use",
    "want",
    "with",
}


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


def _extract_trigger_terms(text: str) -> list[str]:
    tokens = []
    for raw_token in str(text or "").lower().split():
        token = "".join(ch for ch in raw_token if ch.isalnum() or ch == "_")
        if len(token) < 4 or token.isdigit() or token in _INTENT_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def build_intent_ranking(per_intent: list[dict]) -> dict:
    ranked_success = [item for item in per_intent if item.get("count")]
    ranking: dict[str, dict | None] = {
        "most_triggered": per_intent[0] if per_intent else None,
        "least_successful": None,
        "most_successful": None,
    }
    if ranked_success:
        ranking["most_successful"] = max(
            ranked_success,
            key=lambda item: (item.get("success_rate", 0.0), item.get("count", 0)),
        )
        ranking["least_successful"] = min(
            ranked_success,
            key=lambda item: (item.get("success_rate", 0.0), -item.get("count", 0)),
        )
    return ranking


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
    feedback_counts = Counter()
    cancel_counts = Counter()
    model_cancels = Counter()
    latest_errors: list[str] = []
    first_timestamp = ""
    last_timestamp = ""
    intent_categories = Counter()
    intent_durations: dict[str, list[int]] = defaultdict(list)
    intent_success = Counter()
    intent_failure = Counter()
    intent_feedback: dict[str, Counter] = defaultdict(Counter)
    intent_trigger_terms: dict[str, Counter] = defaultdict(Counter)

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

        if event_type == "turn_feedback":
            feedback = str(event.get("feedback", "")).strip().lower()
            if feedback:
                feedback_counts[feedback] += 1
            if feedback in {"reject", "wrong_answer"}:
                reject_feedback += 1
                if model:
                    model_rejects[model] += 1

        if event_type == "turn_cancelled":
            cancel_bucket = str(event.get("cancel_bucket", "")).strip().lower() or "unknown"
            cancel_counts[cancel_bucket] += 1
            if model:
                model_cancels[model] += 1

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

        if event_type == "intent_captured":
            intent_category = str(event.get("intent_category", "")).strip() or "unknown"
            intent_categories[intent_category] += 1
            duration_ms = event.get("duration_ms")
            if isinstance(duration_ms, (int, float)) and int(duration_ms) > 0:
                intent_durations[intent_category].append(int(duration_ms))
            if event.get("success") is True:
                intent_success[intent_category] += 1
            elif event.get("success") is False:
                intent_failure[intent_category] += 1
            feedback = str(event.get("feedback", "")).strip().lower()
            if feedback:
                intent_feedback[intent_category][feedback] += 1
            for token in _extract_trigger_terms(str(event.get("intent_description", ""))):
                intent_trigger_terms[intent_category][token] += 1

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
                "cancels": model_cancels.get(model_name, 0),
            }
        )

    ranking = build_model_ranking(per_model)
    intent_total = sum(intent_categories.values())
    per_intent: list[dict] = []
    for intent_name, count in intent_categories.most_common():
        durations = intent_durations.get(intent_name, [])
        success_count = intent_success.get(intent_name, 0)
        failure_count = intent_failure.get(intent_name, 0)
        feedback_counts = intent_feedback.get(intent_name, Counter())
        per_intent.append(
            {
                "intent_category": intent_name,
                "count": count,
                "share": count / intent_total if intent_total else 0.0,
                "success": success_count,
                "failure": failure_count,
                "success_rate": success_count / max(1, success_count + failure_count),
                "duration_count": len(durations),
                "duration_median_ms": int(median(durations)) if durations else None,
                "duration_max_ms": max(durations) if durations else None,
                "feedback": feedback_counts,
                "top_triggers": [token for token, _ in intent_trigger_terms.get(intent_name, Counter()).most_common(5)],
            }
        )
    intent_ranking = build_intent_ranking(per_intent)

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
        "feedback_counts": feedback_counts,
        "cancel_counts": cancel_counts,
        "cancel_total": sum(cancel_counts.values()),
        "latest_errors": latest_errors,
        "per_model": per_model,
        "ranking": ranking,
        "intent_total": intent_total,
        "intent_categories": intent_categories,
        "per_intent": per_intent,
        "intent_ranking": intent_ranking,
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
    intent_bits = [
        f"`{name}` ({count})"
        for name, count in summary.get("intent_categories", {}).most_common(5)
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
    if intent_bits:
        lines.append(f"- Intent events: {summary.get('intent_total', 0)} via {', '.join(intent_bits)}")
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
    per_intent = summary.get("per_intent", [])
    if per_intent:
        lines.append("**Intent Routing Signals**")
        intent_ranking = summary.get("intent_ranking", {})
        most_triggered = intent_ranking.get("most_triggered")
        if most_triggered:
            lines.append(
                f"- Most triggered intent: `{most_triggered.get('intent_category', '')}` "
                f"({int(round(100 * (most_triggered.get('share') or 0.0)))}% of captured intent events)"
            )
        most_successful = intent_ranking.get("most_successful")
        if most_successful:
            lines.append(
                f"- Strongest routing fit: `{most_successful.get('intent_category', '')}` "
                f"({int(round(100 * (most_successful.get('success_rate') or 0.0)))}% success over {most_successful.get('count', 0)} captures)"
            )
        least_successful = intent_ranking.get("least_successful")
        if least_successful:
            lines.append(
                f"- Weakest routing fit: `{least_successful.get('intent_category', '')}` "
                f"({int(round(100 * (least_successful.get('success_rate') or 0.0)))}% success over {least_successful.get('count', 0)} captures)"
            )
        for item in per_intent:
            feedback_counts = item.get("feedback", Counter())
            feedback_text = ", ".join(
                f"`{name}` ({count})" for name, count in feedback_counts.most_common()
            ) or "none"
            duration_text = "no duration recorded"
            if item.get("duration_count"):
                duration_text = (
                    f"median {item.get('duration_median_ms')} ms, max {item.get('duration_max_ms')} ms"
                )
            trigger_text = ", ".join(f"`{token}`" for token in item.get("top_triggers", [])) or "none"
            lines.append(
                f"- `{item.get('intent_category', '')}`: {item.get('count', 0)} captures "
                f"({int(round(100 * (item.get('share') or 0.0)))}% trigger share); "
                f"{int(round(100 * (item.get('success_rate') or 0.0)))}% success; "
                f"{duration_text}; feedback: {feedback_text}; likely triggers: {trigger_text}"
            )
    lines.append(
        "- Code execution: "
        f"{summary.get('code_success', 0)} succeeded, {summary.get('code_failure', 0)} failed"
    )
    lines.append(f"- Reject feedback: {summary.get('reject_feedback', 0)}")
    if summary.get("tool_failures"):
        lines.append(f"- Tool failures: {summary.get('tool_failures', 0)}")
    if summary.get("feedback_counts"):
        feedback_bits = [
            f"`{name}` ({count})"
            for name, count in summary.get("feedback_counts", {}).most_common()
        ]
        lines.append(f"- Feedback: {', '.join(feedback_bits)}")
    if summary.get("cancel_counts"):
        cancel_bits = [
            f"`{name}` ({count})"
            for name, count in summary.get("cancel_counts", {}).most_common()
        ]
        lines.append(f"- Abandonment: {', '.join(cancel_bits)}")
    if summary.get("invalid_lines"):
        lines.append(f"- Invalid JSONL lines skipped: {summary.get('invalid_lines', 0)}")
    if summary.get("latest_errors"):
        lines.append("**Recent Errors**")
        lines.extend(f"- {text}" for text in summary["latest_errors"])
    return "\n".join(lines)
