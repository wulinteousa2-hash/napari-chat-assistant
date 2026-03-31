# Tested Models

This document tracks local Ollama models that have been tested with `napari-chat-assistant` during development.

It is not a formal benchmark. The goal is to record what has been tried, what felt useful in practice, and where telemetry can later help compare models under real plugin usage.

## How To Read This

- `Recommended` means a model is a reasonable starting point for most users.
- `Tested` means the model was exercised in development, but not necessarily promoted as a default recommendation.
- results may vary by hardware, memory, Ollama version, prompt style, and dataset size

## Recommended Starting Models

These are the models currently recommended in the main README:

| Model | Why use it |
| --- | --- |
| `nemotron-cascade-2:30b` | Current default. Strong general choice for local interactive imaging workflows. |
| `gpt-oss:120b` | Large model that can still feel relatively fast on high-memory systems. Useful when you want a stronger model and have the hardware for it. |
| `qwen3-coder-next:latest` | Good candidate when code generation quality matters most. Heavier than the default options. |
| `qwen3.5` | Solid general-purpose alternative. |
| `qwen2.5:7b` | Lighter option for smaller-memory systems. |

## Models Tested During Development

The following models have been exercised during development. This table is meant to be more practical than a benchmark leaderboard: it captures the tags, rough model footprint, and why each model was tried.

| Model | Architecture | Type | Quantization | Size | Status | Notes |
| --- | --- | --- | --- | ---: | --- | --- |
| `olmo3-1:32b` | `olmo3` | Dense | `Q4_K_M` | 32.2B | Tested | Large general model option. |
| `gpt-oss:120b` | `gptoss` | Dense | `MXFP4` | 116.8B | Recommended | Large but still relatively fast on strong hardware; useful when you want a higher-capacity model. |
| `deepseek-r1:70b` | `llama` | Dense | `Q4_K_M` | 70.6B | Tested | Reasoning-oriented alternative; heavy and slower in local use. |
| `qwen3-vl:30b` | `qwen3vlmoe` | MoE | `Q4_K_M` | 31.1B | Tested | Multimodal-capable model, but multimodal image prompting is not yet a primary plugin path. |
| `qwen3.5:122b` | `qwen35moe` | MoE | `Q4_K_M` | 125.1B | Tested | Very large general model option. |
| `qwen3-next:80b` | `qwen3next` | Dense | `Q4_K_M` | 79.7B | Tested | Larger model in the Qwen family. |
| `nemotron-cascade-2:30b` | `nemotron_h_moe` | MoE | `Q4_K_M` | 31.6B | Recommended | Current default for the plugin. |
| `qwen3-coder-next:latest` | `qwen3next` | Dense | `Q4_K_M` | 79.7B | Recommended | Strong candidate for Python and napari code generation. |
| `qwen3.5:35b` | `qwen35moe` | MoE | `Q4_K_M` | 36.0B | Tested | Mid-range alternative in the Qwen family. |
| `qwen3.5:latest` | `qwen35` | Dense | `Q4_K_M` | 9.7B | Recommended | General-purpose alternative with a smaller footprint than the very large tags. |
| `qwen2.5:7b` | `qwen2` | Dense | `Q4_K_M` | 7.6B | Recommended | Lighter local option for smaller-memory systems. |

## Current Telemetry Snapshot

For a generated markdown summary of current real-world plugin telemetry, see [docs/telemetry_results.md](telemetry_results.md).

## Posting Telemetry Results

Yes, you can post telemetry results, but they should be framed as:
- real-world plugin usage observations
- hardware-specific measurements
- early comparative data rather than a controlled benchmark

That is the strongest honest framing unless you standardize prompts, datasets, and repeated runs across models.

You can now generate a publishable markdown report directly from the local telemetry log:

```bash
napari-chat-assistant-telemetry-report --format markdown
```

To save it to a file:

```bash
napari-chat-assistant-telemetry-report --format markdown --output docs/telemetry_results.md
```

To point at a specific telemetry file:

```bash
napari-chat-assistant-telemetry-report --input ~/.napari-chat-assistant/model_telemetry.jsonl --format markdown
```

## Suggested Telemetry Summary Format

If you want to publish results from `~/.napari-chat-assistant/model_telemetry.jsonl`, summarize them with:

| Field | What to report |
| --- | --- |
| Hardware | CPU, GPU, RAM, VRAM, workstation name if relevant |
| Software | OS, Python, napari version, plugin version, Ollama version |
| Model | Exact model tag |
| Workload | Example prompts or workflow categories used |
| Dataset context | 2D/3D, approximate size, microscopy or other modality |
| Request count | Number of recorded interactions |
| Latency | Median and range if possible |
| Response mix | Counts of `reply`, `tool`, `code`, and `error` |
| Notes | Where the model felt strong or weak in practice |

## Suggested Language For Sharing Results

Use wording like:

> These results come from local telemetry collected during real plugin usage, not from a controlled benchmark suite.

> Performance depends on hardware, model quantization, dataset size, and workflow type.

> The goal of sharing these numbers is to help users choose a reasonable starting model for local napari workflows.

## What To Avoid

Avoid claiming:
- one model is universally best
- telemetry from one machine will generalize to all systems
- interactive latency alone reflects overall workflow quality

For this project, model quality also depends on:
- tool-call reliability
- code generation quality
- response usefulness in imaging workflows
- stability across repeated real tasks
