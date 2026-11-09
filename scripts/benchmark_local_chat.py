"""Measure local chat latency without executing any model-generated actions.

Run in the plugin's Python environment. Results include model loading, prompt
evaluation, generation, and hidden thinking; no conversation logs are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from urllib import request


def post(base_url, route, payload):
    req = request.Request(
        base_url.rstrip("/") + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--think", choices=("default", "on", "off"), default="default")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or (args.context is not None and args.context < 1):
        parser.error("repeats and context must be positive")

    from napari_chat_assistant.agent.tools import assistant_system_prompt

    system = assistant_system_prompt()
    prompts = [
        "What is the difference between Gaussian and median denoising? Keep the answer brief.",
        "How should I choose a threshold for a fluorescence image with uneven illumination? Keep the answer brief.",
    ]
    options = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "repeat_penalty": 1.5, "seed": 42}
    if args.context is not None:
        options["num_ctx"] = args.context
    rows = []
    report = {"model": args.model, "context": args.context, "think": args.think,
              "system_prompt_chars": len(system), "options": options, "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for index, prompt in enumerate(prompts):
            payload = {
                "model": args.model, "stream": False, "options": options,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps({
                        "viewer_context": {"layers": [], "selected_layer": None, "selected_layer_profile": None},
                        "session_memory": {}, "intent_state": {}, "last_failed_tool_state": {},
                        "followup_constraint": None, "code_repair_context": None,
                        "user_message": prompt,
                    })},
                ],
            }
            if args.think != "default":
                payload["think"] = args.think == "on"
            started = time.perf_counter()
            result = post(args.base_url, "/api/chat", payload)
            wall = time.perf_counter() - started
            msg = result.get("message", {})
            content = msg.get("content", "")
            try:
                parsed = json.loads(content)
                valid = isinstance(parsed, dict) and parsed.get("action") in {"reply", "tool", "code"}
            except (ValueError, TypeError):
                valid = False
            row = {"repeat": repeat, "prompt_index": index, "wall_s": round(wall, 3),
                   "content_chars": len(content), "thinking_chars": len(msg.get("thinking", "")),
                   "valid_action_json": valid, "done_reason": result.get("done_reason")}
            for field in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
                value = result.get(field)
                row[field + "_s"] = round(value / 1e9, 4) if value is not None else None
            for field in ("prompt_eval_count", "eval_count"):
                row[field] = result.get(field)
            rows.append(row)
            report["median_wall_s"] = statistics.median(r["wall_s"] for r in rows)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    print("Median seconds:", report["median_wall_s"], flush=True)


if __name__ == "__main__":
    main()
