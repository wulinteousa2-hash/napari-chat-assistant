from __future__ import annotations

import argparse
from pathlib import Path

from napari_chat_assistant.telemetry.logging_utils import TELEMETRY_LOG_PATH
from napari_chat_assistant.telemetry.summary import (
    format_telemetry_summary,
    load_telemetry_events,
    summarize_telemetry_events,
)


def _format_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "-"
    return f"{round((numerator / denominator) * 100):d}%"


def _response_count(item: dict, action: str) -> int:
    actions = item.get("actions", {})
    return int(actions.get(action, 0))


def _feedback_summary(feedback: dict) -> str:
    counts = dict(feedback or {})
    if not counts:
        return "-"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{name}:{count}" for name, count in ordered)


def format_markdown_telemetry_report(
    summary: dict,
    *,
    path: Path | None = None,
    title: str = "Telemetry Results",
) -> str:
    source_path = path or TELEMETRY_LOG_PATH
    if not summary.get("total_events"):
        return "\n".join(
            [
                f"# {title}",
                "",
                f"No telemetry events were found in `{source_path}`.",
            ]
        )

    lines: list[str] = [
        f"# {title}",
        "",
        "> These results come from local telemetry collected during real plugin usage, not from a controlled benchmark suite.",
        "> Performance depends on hardware, model quantization, dataset size, and workflow type.",
        "",
        "## Overview",
        "",
        f"- Source: `{source_path}`",
        f"- Records: {summary.get('total_events', 0)}",
        f"- Completed turns: {summary.get('turn_completed', 0)}",
    ]
    if summary.get("first_timestamp") or summary.get("last_timestamp"):
        lines.append(
            f"- Time range: `{summary.get('first_timestamp', '')}` to `{summary.get('last_timestamp', '')}`"
        )
    if summary.get("latency_count"):
        lines.append(
            f"- Overall latency: median {summary.get('latency_median_ms')} ms, max {summary.get('latency_max_ms')} ms"
        )
    lines.extend(
        [
            f"- Tool failures: {summary.get('tool_failures', 0)}",
            f"- Code execution: {summary.get('code_success', 0)} succeeded, {summary.get('code_failure', 0)} failed",
            f"- Reject feedback: {summary.get('reject_feedback', 0)}",
        ]
    )
    if summary.get("feedback_counts"):
        feedback_bits = [
            f"`{name}` ({count})"
            for name, count in summary.get("feedback_counts", {}).most_common()
        ]
        lines.append(f"- Feedback mix: {', '.join(feedback_bits)}")
    if summary.get("cancel_counts"):
        cancel_bits = [
            f"`{name}` ({count})"
            for name, count in summary.get("cancel_counts", {}).most_common()
        ]
        lines.append(f"- Abandonment: {', '.join(cancel_bits)}")
    if summary.get("invalid_lines"):
        lines.append(f"- Invalid JSONL lines skipped: {summary.get('invalid_lines', 0)}")

    action_bits = []
    for action_name, count in summary.get("actions", {}).most_common():
        action_bits.append(f"`{action_name}` ({count})")
    if action_bits:
        lines.extend(["", "## Response Mix", "", f"- {', '.join(action_bits)}"])

    intent_bits = []
    for intent_name, count in summary.get("intent_categories", {}).most_common():
        intent_bits.append(f"`{intent_name}` ({count})")
    if intent_bits:
        lines.extend(["", "## Intent Mix", "", f"- {', '.join(intent_bits)}"])

    lines.extend(
        [
            "",
            "## Per-Model Summary",
            "",
            "| Model | Turns | Median Latency (ms) | Max Latency (ms) | Reply | Tool | Code | Error | Rejects | Reject Rate | Code Success |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for item in summary.get("per_model", []):
        code_total = int(item.get("code_success", 0)) + int(item.get("code_failure", 0))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.get('model', '')}`",
                    str(item.get("turns", 0)),
                    str(item.get("latency_median_ms", "-") if item.get("latency_count") else "-"),
                    str(item.get("latency_max_ms", "-") if item.get("latency_count") else "-"),
                    str(_response_count(item, "reply")),
                    str(_response_count(item, "tool")),
                    str(_response_count(item, "code")),
                    str(_response_count(item, "error")),
                    str(item.get("rejects", 0)),
                    _format_percent(int(item.get("rejects", 0)), int(item.get("turns", 0))),
                    _format_percent(int(item.get("code_success", 0)), code_total),
                ]
            )
            + " |"
        )

    per_intent = summary.get("per_intent", [])
    if per_intent:
        lines.extend(
            [
                "",
                "## Intent Routing Signals",
                "",
                "| Intent | Captures | Trigger Share | Success | Median Duration (ms) | Feedback | Likely Trigger Terms |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for item in per_intent:
            trigger_terms = ", ".join(f"`{term}`" for term in item.get("top_triggers", [])) or "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{item.get('intent_category', '')}`",
                        str(item.get("count", 0)),
                        _format_percent(int(item.get("count", 0)), int(summary.get("intent_total", 0))),
                        _format_percent(int(item.get("success", 0)), int(item.get("success", 0)) + int(item.get("failure", 0))),
                        str(item.get("duration_median_ms", "-") if item.get("duration_count") else "-"),
                        _feedback_summary(item.get("feedback", {})),
                        trigger_terms,
                    ]
                )
                + " |"
            )

    ranking = summary.get("ranking", {})
    ranking_lines: list[str] = []
    fastest = ranking.get("fastest")
    if fastest:
        ranking_lines.append(
            f"- Fastest median latency: `{fastest.get('model', '')}` at {fastest.get('latency_median_ms')} ms"
        )
    slowest = ranking.get("slowest")
    if slowest:
        ranking_lines.append(
            f"- Slowest median latency: `{slowest.get('model', '')}` at {slowest.get('latency_median_ms')} ms"
        )
    most_used = ranking.get("most_used")
    if most_used:
        ranking_lines.append(
            f"- Most used: `{most_used.get('model', '')}` with {most_used.get('turns', 0)} completed turns"
        )
    best_code = ranking.get("best_code_success")
    if best_code:
        code_total = int(best_code.get("code_success", 0)) + int(best_code.get("code_failure", 0))
        ranking_lines.append(
            f"- Best code-run signal: `{best_code.get('model', '')}` at {_format_percent(int(best_code.get('code_success', 0)), code_total)} over {code_total} executions"
        )
    worst_code = ranking.get("worst_code_success")
    if worst_code:
        code_total = int(worst_code.get("code_success", 0)) + int(worst_code.get("code_failure", 0))
        ranking_lines.append(
            f"- Weakest code-run signal: `{worst_code.get('model', '')}` at {_format_percent(int(worst_code.get('code_success', 0)), code_total)} over {code_total} executions"
        )
    intent_ranking = summary.get("intent_ranking", {})
    most_triggered = intent_ranking.get("most_triggered")
    if most_triggered:
        ranking_lines.append(
            f"- Most triggered intent: `{most_triggered.get('intent_category', '')}` at {_format_percent(int(most_triggered.get('count', 0)), int(summary.get('intent_total', 0)))} of captured intent events"
        )
    most_successful = intent_ranking.get("most_successful")
    if most_successful:
        ranking_lines.append(
            f"- Strongest routing fit: `{most_successful.get('intent_category', '')}` at {_format_percent(int(most_successful.get('success', 0)), int(most_successful.get('success', 0)) + int(most_successful.get('failure', 0)))} success over {most_successful.get('count', 0)} captures"
        )
    least_successful = intent_ranking.get("least_successful")
    if least_successful:
        ranking_lines.append(
            f"- Weakest routing fit: `{least_successful.get('intent_category', '')}` at {_format_percent(int(least_successful.get('success', 0)), int(least_successful.get('success', 0)) + int(least_successful.get('failure', 0)))} success over {least_successful.get('count', 0)} captures"
        )

    if ranking_lines:
        lines.extend(["", "## Quick Takeaways", ""])
        lines.extend(ranking_lines)

    latest_errors = summary.get("latest_errors", [])
    if latest_errors:
        lines.extend(["", "## Recent Errors", ""])
        for error_text in latest_errors:
            lines.append(f"- {error_text}")

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize napari-chat-assistant telemetry into chat text or publishable Markdown."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=TELEMETRY_LOG_PATH,
        help=f"Path to telemetry JSONL. Default: {TELEMETRY_LOG_PATH}",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "chat"),
        default="markdown",
        help="Output format. 'markdown' is suitable for README/issues/discussions. 'chat' matches the in-app summary style.",
    )
    parser.add_argument(
        "--title",
        default="Telemetry Results",
        help="Markdown report title.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. If omitted, prints to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    events, invalid_lines = load_telemetry_events(args.input)
    summary = summarize_telemetry_events(events, invalid_lines)
    if args.format == "chat":
        rendered = format_telemetry_summary(summary)
    else:
        rendered = format_markdown_telemetry_report(summary, path=args.input, title=args.title)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
