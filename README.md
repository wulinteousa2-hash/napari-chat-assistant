# napari-chat-assistant

[![License MIT](https://img.shields.io/pypi/l/napari-chat-assistant.svg?color=green)](https://github.com/wulinteousa2-hash/napari-chat-assistant/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-chat-assistant.svg?color=green)](https://pypi.org/project/napari-chat-assistant)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-chat-assistant.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-chat-assistant)](https://napari-hub.org/plugins/napari-chat-assistant)

---

Local Ollama-powered assistant for napari image-analysis workflows.

`napari-chat-assistant` adds a dock widget inside napari that understands the active viewer session, runs built-in image-analysis actions, and generates executable napari Python code when a request goes beyond the current toolset.

It is designed for local interactive work, repeatable workflows, and gradual automation rather than cloud chat or fully opaque “one-click AI”.

The current direction is a deterministic, layer-aware assistant: the plugin profiles loaded napari layers first, then uses that structured context to guide tool choice and generated code.

## Overview

Current capabilities include:
- connect to a local Ollama server
- discover and unload local models from the plugin UI
- inspect layers and selected-layer properties
- profile layers with a deterministic Phase 1 dataset profiler
- apply built-in image tools from chat
- automate batch actions across multiple layers
- generate napari Python code when no built-in tool fits
- copy or run generated code from the assistant UI
- paste and run your own Python directly from the Prompt box with `Run My Code`, without opening QtConsole
- save, pin, and reuse prompts through a local Prompt Library
- delete selected built-in, recent, or saved prompts from the Prompt Library
- clear unpinned recent and built-in prompts while keeping saved and pinned items
- keep bounded session memory from approved prior turns
- reject the last assistant outcome from session memory with a thumbs-down control
- optionally open ND2 conversion, spectral viewer, and spectral analysis widgets from `napari-nd2-spectral-ome-zarr`

The current default model is:
- `nemotron-cascade-2:30b`

## Why This Plugin

Most chat interfaces are detached from the actual napari session. This plugin keeps the assistant inside the viewer and grounds its responses in:
- loaded layers
- the selected layer
- shape and dtype
- semantic layer profiling
- labels statistics
- local tool execution
- local Python code generation
- bounded session memory

### Local-first by design

The assistant runs on local open-weight models through Ollama:

- no API key required
- no internet dependency
- no cloud services
- no data leaves your workstation

This makes it suitable for research workflows where the user wants interactive help, repeatable prompts, and local control over data and models.

## Current Features

### Session-aware tools

The assistant currently supports built-in tools for:
- listing all layers
- inspecting the selected layer
- inspecting a specific named layer
- CLAHE contrast enhancement for grayscale 2D and 3D images
- batch CLAHE across multiple image layers
- threshold preview
- threshold apply
- batch threshold preview and apply
- mask measurement
- batch mask measurement
- mask morphology operations

Layer inspection is now backed by a deterministic profile object that includes:
- `semantic_type`
- `confidence`
- `axes_detected`
- `source_kind`
- metadata flags such as multiscale, lazy/chunked, channel metadata, and wavelength metadata
- recommended and discouraged operation classes
- evidence buckets for debugging and future adapter work

Supported mask operations:
- `dilate`
- `erode`
- `open`
- `close`
- `fill_holes`
- `remove_small`
- `keep_largest`

### Code generation workflows

When a request is not covered by a built-in tool, the assistant can return napari Python code instead of guessing.

Generated code can be:
- copied to the clipboard
- executed from the plugin after user review

You can also paste your own Python directly into the Prompt box and run it from the plugin with `Run My Code`, without switching to QtConsole.

This is useful when you want a reusable script, need to adjust code manually, test a small viewer-bound snippet quickly, or prefer explicit code over hidden automation.

### Optional ND2 and spectral integration

If `napari-nd2-spectral-ome-zarr` is installed, the assistant can open:
- the ND2-to-OME-Zarr export widget
- the Spectral Viewer widget
- the Spectral Analysis widget

This lets chat act as an entry point for Nikon ND2 conversion and spectral workflows without rebuilding those UIs inside this plugin.

Install links:
- GitHub: `https://github.com/wulinteousa2-hash/napari-nd2-spectral-ome-zarr`
- napari Hub: `https://napari-hub.org/plugins/napari-nd2-spectral-ome-zarr.html`

### Selective Session Memory

The assistant now includes bounded session memory with three states:
- `provisional`
- `approved`
- `rejected`

Behavior:
- new assistant outcomes start as provisional
- successful follow-up actions can promote them to approved
- only approved items are sent back to the model as `session_memory`
- current viewer context and current layer profiles always override memory
- `Thumbs Down Last Answer` rejects the most recent memory candidate for the current session

This is intentionally not full transcript memory. The model is still grounded primarily in the current napari viewer state.

### Prompt Library

The assistant includes a persistent Prompt Library for repeatable workflows:
- built-in starter prompts
- recent prompts captured automatically
- saved prompts for reusable tasks
- pinned prompts for high-frequency workflows

Interaction:
- single click loads a prompt into the editor
- double click sends it directly
- multi-select supports Shift/Ctrl selection for batch actions
- `Delete Selected` can remove saved prompts, recent prompts, or hide built-in prompts
- `Clear Non-Saved` removes unpinned recent and built-in prompts while keeping saved and pinned items

Logic:
- `saved` means a user-managed prompt you want to keep as your own reusable entry
- `pinned` means keep this prompt surfaced at the top of the library
- a prompt can be pinned without being saved
- built-in prompts are shipped examples; deleting them hides them from the current local library view

This is designed for users who want repeatable automation without committing everything to full scripting.

## Requirements
- Python 3.9+
- napari
- Ollama installed locally
- a local Ollama model such as `nemotron-cascade-2:30b`

Tested during development on an NVIDIA DGX Spark workstation.

The plugin does not bundle the Ollama server or model weights.

## Installation

### 1. Install Ollama

Install Ollama on the same machine that runs napari, then start the local server:

```bash
ollama serve
```

Pull a model before using the plugin:

```bash
ollama pull nemotron-cascade-2:30b
```

Optional alternatives:

```bash
ollama pull qwen3-coder-next:latest
ollama pull qwen3.5
ollama pull qwen2.5:7b
```

### 2. Install the plugin

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/wulinteousa2-hash/napari-chat-assistant.git
cd napari-chat-assistant
pip install -e .
```

## Release

This package is published to PyPI so napari Hub can discover it.
For maintainer release instructions and PyPI publishing setup, see [RELEASING.md](RELEASING.md)
## Usage

1. Start napari.
2. Open `Plugins -> Chat Assistant`.
3. Leave `Base URL` as `http://127.0.0.1:11434` unless your Ollama server is elsewhere.
4. Choose a model from the `Model` dropdown or type a model tag manually.
5. Click `Test Connection`.
6. Start chatting, or use the Prompt Library for repeatable tasks.

If you already have Python code you want to try, paste it into the Prompt box and click `Run My Code`. This runs viewer-bound code directly inside napari without opening QtConsole.

The assistant works best when prompts describe a concrete action.

Examples:

Layer inspection:
- `list all layers in the current viewer`
- `inspect the selected layer properties`
- `inspect layer LV-nerve and report its shape and dtype`

EM contrast enhancement:
- `apply CLAHE to the selected EM image`
- `apply CLAHE to the selected image with kernel_size 32, clip_limit 0.01, nbins 256`
- `apply CLAHE to all open EM images with kernel_size 64, clip_limit 0.02, nbins 512`

Thresholding and masks:
- `preview a threshold mask for the selected image layer`
- `apply a threshold optimized for dim objects on the selected image`
- `measure connected components in the current mask layer`

Code generation:
- `write napari code to duplicate the selected layer`
- `generate QtConsole code to print the selected layer shape`
- `create a synthetic noisy image in the current viewer and generate napari code for it`
- `create a docked histogram widget for the selected image and report mean, noise SD, and simple SNR`

Profile-aware prompts:
- `show every loaded layer with semantic type, confidence, axes, shape, and dtype`
- `inspect the selected layer and explain what kind of dataset it is and why`
- `tell me which operation classes are recommended or discouraged for the selected layer`
- `decide if CLAHE is appropriate for the selected layer before using it`

Demo and education prompts:
- `create a synthetic noisy image in the current viewer for teaching image noise`
- `generate a docked histogram and simple SNR widget for the selected image`
- `create two synthetic images with low noise and high noise and compare their histograms`
- `simulate low-SNR and high-SNR examples for teaching imaging quality`
- `generate napari code that shows how noise level changes histogram width and simple SNR`
- `create a demo image with bright spots on dark background and vary the noise step by step`

## UI Overview

### Model Connection

- local Ollama base URL
- model picker with discovered local models
- test connection
- use selected model
- `Ollama Setup` help with install, `ollama serve`, and model pull examples
- unload model

### Prompt Library

- built-in prompts
- recent prompts
- saved prompts
- pinned prompts
- `saved` keeps your own reusable copy
- `pinned` keeps a prompt at the top regardless of whether it is built-in, recent, or saved
- single click to load
- double click to send
- Shift/Ctrl multi-select for batch actions
- `Delete Selected` works on built-in, recent, and saved prompts
- `Clear Non-Saved` keeps saved and pinned items and clears unpinned recent and built-in items
- `A-` and `A+` adjust prompt-library font size in small steps

### Chat

- multi-line prompt box
- Enter to send
- Shift/Ctrl/Alt+Enter for newline
- transcript showing user messages, assistant replies, tool results, and generated code

### Code Actions

- `Reject`
- `Run Code`
- `Run My Code`
- `Copy Code`
- `Help`

`Run Code` is for assistant-generated code that has been staged in the chat.

`Run My Code` is for your own pasted Python from the Prompt box when you want to test or iterate directly inside napari without opening QtConsole.

### Current Context

- current layer summary from the active napari viewer
- shortened layer names and a compact per-layer summary to avoid over-stretching the left column

### Action Log

- local status updates
- model connection messages
- tool execution messages
- code execution and copy actions
- color-highlighted path entries for assistant log, crash log, telemetry log, prompt library, and session memory
- `Enable Telemetry` switch for advanced users
- `Performance Summary`, `Telemetry Log`, and `Reset Log` only when telemetry is enabled

## How It Works

The assistant is designed to operate within constrained napari workflows rather than as a general-purpose chatbot.

The current strategy is:
1. collect structured napari viewer context
2. build deterministic per-layer profile objects from the current viewer state
3. add bounded approved session memory when available
4. send that context and the user request to a local Ollama model
5. the model returns a structured JSON response that specifies either:
   - a normal reply
   - a built-in tool call
   - generated Python code
6. run the selected tool or expose the generated code through the UI
7. update session memory from explicit user feedback or successful follow-up behavior

This keeps the assistant more grounded than a plain chat interface and makes common operations more reliable.

## Recommended Models

Good starting choices:
- `nemotron-cascade-2:30b`
- `qwen3-coder-next:latest`
- `qwen3.5`
- `qwen2.5:7b`

Selection guidance:
- `nemotron-cascade-2:30b` is the current default and a strong general model for this workflow.
- `qwen3-coder-next:latest` is a better candidate for Python and napari code generation, but it is significantly heavier.
- `qwen3.5` remains a useful alternative general model.
- `qwen2.5:7b` is lighter and may fit smaller-memory systems more easily.

Memory note:
- Larger tags require more RAM or VRAM.
- On the DGX Spark setup used during development, `qwen3-coder-next:latest` may need around 100 GB of available memory to run comfortably.

## Current Limitations

- the dataset profiler is still Phase 1 and currently strongest on already-loaded napari layers rather than file-format-specific readers
- TIFF vs OME-Zarr adapter behavior is not implemented yet
- ND2 and Zeiss adapters are not implemented yet
- session memory is selective and bounded; it is not full conversation memory
- model output can still be inconsistent, especially for generated code
- not all requests map cleanly to built-in tools yet
- generated code can still fail if the model invents incorrect napari APIs
- no multi-step task planning yet (complex workflows may require several prompts)
- no image attachment or multimodal input pipeline yet
- performance optimization for very large 2D/3D datasets is still in progress
- hard native crashes in Qt/C-extension code may not be captured cleanly by the plugin crash log even when normal plugin errors are logged

Most reliable current workflow:
- use built-in tools for common layer inspection and mask/image actions
- trust current viewer context and current layer profiles over any remembered prior turn
- use the Prompt Library for repeated tasks
- use generated code when you want explicit review and control
- use `Run My Code` when you already have working Python and want to test it directly inside napari

For demo and education workflows:
- ask for code that uses the current napari `viewer`
- avoid prompts that create a second `napari.Viewer()` or call `napari.run()`
- prefer docked widgets over unmanaged popup windows for histogram or SNR teaching tools

## Troubleshooting

### Ollama not running

If `Test Connection` fails after restarting your computer, Ollama is usually not running yet.

Start it in a terminal:

```bash
ollama serve
```

Then return to the plugin and click `Test Connection` again.

### Pulling a model

Model downloads are intentionally handled outside the plugin.

To try a different model:
- browse tags at `https://ollama.com/search`
- type the tag into the plugin `Model` field if needed
- pull it in a terminal, for example:

```bash
ollama pull nemotron-cascade-2:30b
```

Then use `Test Connection` to refresh the plugin state.

### Logs and crash logs

The plugin writes two local log files:
- `~/.napari-chat-assistant/assistant.log`
- `~/.napari-chat-assistant/crash.log`

Use these together with the terminal traceback when diagnosing crashes or unclear UI failures.

### Local model telemetry

The plugin also writes lightweight local telemetry to:
- `~/.napari-chat-assistant/model_telemetry.jsonl`

This records real usage events such as:
- request start and completion
- selected model and prompt hash
- total latency
- response type (`reply`, `tool`, `code`, or `error`)
- reject feedback from `👎 Reject`
- approved code execution success or failure

Telemetry is now opt-in from the Action Log through `Enable Telemetry`.

The goal is passive model tracking during actual work rather than separate benchmarking runs.

For advanced users, the Action Log includes:
- `Performance Summary` to generate a quick in-app summary of recent model speed and behavior
- `Telemetry Log` to inspect the latest raw JSONL records together with the summary
- `Reset Log` to clear the local telemetry file and start fresh from the next request

This keeps the append-only log intact while making it easier to review without leaving napari.

Generated code is also preflight-validated before execution for common dtype mistakes, unsupported napari imports, and unavailable `viewer.*` APIs. When validation blocks execution, the code remains visible and copyable for review or regeneration.

## Development

Editable install:

```bash
pip install -e .
```

Build a release artifact:

```bash
python -m build
```

## License

MIT. 
