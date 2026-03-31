# Telemetry Results

> These results come from local telemetry collected during real plugin usage, not from a controlled benchmark suite.
> Performance depends on hardware, model quantization, dataset size, and workflow type.

## Overview

- Source: `/home/wteox2/.napari-chat-assistant/model_telemetry.jsonl`
- Records: 390
- Completed turns: 158
- Time range: `2026-03-29T16:09:16.008582+00:00` to `2026-03-31T20:08:57.378387+00:00`
- Overall latency: median 11108 ms, max 181283 ms
- Tool failures: 0
- Code execution: 47 succeeded, 17 failed
- Reject feedback: 7

## Response Mix

- `reply` (74), `tool` (46), `code` (38)

## Per-Model Summary

| Model | Turns | Median Latency (ms) | Max Latency (ms) | Reply | Tool | Code | Error | Rejects | Reject Rate | Code Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nemotron-cascade-2:30b` | 76 | 12772 | 91056 | 23 | 31 | 22 | 0 | 0 | 0% | 87% |
| `gpt-oss:120b` | 41 | 6451 | 33472 | 21 | 13 | 7 | 0 | 0 | 0% | 40% |
| `qwen2.5:7b` | 12 | 2661 | 6448 | 11 | 1 | 0 | 0 | 3 | 25% | - |
| `qwen3.5:35b` | 6 | 12612 | 19195 | 4 | 0 | 2 | 0 | 2 | 33% | 25% |
| `olmo-3.1:32b` | 6 | 68245 | 104226 | 5 | 0 | 1 | 0 | 0 | 0% | 100% |
| `qwen3-coder-next:latest` | 5 | 15650 | 32636 | 2 | 0 | 3 | 0 | 0 | 0% | 25% |
| `qwen3.5:latest` | 5 | 13105 | 24358 | 2 | 1 | 2 | 0 | 1 | 20% | 50% |
| `local_ui_help` | 4 | - | - | 4 | 0 | 0 | 0 | 0 | 0% | - |
| `deepseek-r1:70b` | 3 | 90808 | 181283 | 2 | 0 | 1 | 0 | 1 | 33% | 0% |

## Quick Takeaways

- Fastest median latency: `qwen2.5:7b` at 2661 ms
- Slowest median latency: `deepseek-r1:70b` at 90808 ms
- Most used: `nemotron-cascade-2:30b` with 76 completed turns
- Best code-run signal: `olmo-3.1:32b` at 100% over 1 executions
- Weakest code-run signal: `deepseek-r1:70b` at 0% over 1 executions

## Recent Errors

- `'Image' object has no attribute 'kind'`
- `'Labels' object has no attribute 'kind'`
- `'LayerList' object has no attribute 'get'`
