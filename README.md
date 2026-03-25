# napari-chat-assistant

Local Ollama-powered assistant for napari image-analysis workflows.

`napari-chat-assistant` adds a dock widget inside napari that understands the active viewer session, runs built-in image-analysis actions, and generates executable napari Python code when a request goes beyond the current toolset.

It is designed for local interactive work, repeatable workflows, and gradual automation rather than cloud chat or fully opaque “one-click AI”.

## Overview

Current capabilities include:
- connect to a local Ollama server
- discover, pull, and unload local models from the plugin UI
- inspect layers and selected-layer properties
- apply built-in image tools from chat
- automate batch actions across multiple layers
- generate napari Python code when no built-in tool fits
- copy or run generated code from the assistant UI
- save, pin, and reuse prompts through a local Prompt Library

The current default model is:
- `qwen3.5`

## Why This Plugin

Most chat interfaces are detached from the actual napari session. This plugin keeps the assistant inside the viewer and grounds its responses in:
- loaded layers
- the selected layer
- shape and dtype
- labels statistics
- local tool execution
- local Python code generation

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

This is useful when you want a reusable script, need to adjust code manually, or prefer explicit code over hidden automation.

### Prompt Library

The assistant includes a persistent Prompt Library for repeatable workflows:
- built-in starter prompts
- recent prompts captured automatically
- saved prompts for reusable tasks
- pinned prompts for high-frequency workflows

Interaction:
- single click loads a prompt into the editor
- double click sends it directly

This is designed for users who want repeatable automation without committing everything to full scripting.

## Requirements
- Python 3.9+
- napari
- Ollama installed locally
- a local Ollama model such as `qwen3.5`

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
ollama pull qwen3.5
```

Optional stronger model:

```bash
ollama pull qwen3.5:27b
```

### 2. Install the plugin

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/wulinteousa2-hash/napari-chat-assistant.git
cd napari-chat-assistant
pip install -e .
```

## Usage

1. Start napari.
2. Open `Plugins -> Chat Assistant`.
3. Leave `Base URL` as `http://127.0.0.1:11434` unless your Ollama server is elsewhere.
4. Choose a model from the `Model` dropdown or type a model tag manually.
5. Click `Test Connection`.
6. Start chatting, or use the Prompt Library for repeatable tasks.

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

## UI Overview

### Model Connection

- local Ollama base URL
- model picker with discovered local models
- test connection
- save current settings
- pull model
- unload model

### Prompt Library

- built-in prompts
- recent prompts
- saved prompts
- pinned prompts
- single click to load
- double click to send

### Chat

- multi-line prompt box
- Enter to send
- Shift/Ctrl/Alt+Enter for newline
- transcript showing user messages, assistant replies, tool results, and generated code

### Code Actions

- `Run Pending Code`
- `Copy Pending Code`
- `Discard Pending Code`

### Current Context

- current layer summary from the active napari viewer

### Action Log

- local status updates
- model connection messages
- tool execution messages
- code execution and copy actions

## How It Works

The assistant is designed to operate within constrained napari workflows rather than as a general-purpose chatbot.

The current strategy is:
1. collect structured napari viewer context
2. send that context and the user request to a local Ollama model
3. the model returns a structured JSON response that specifies either:
   - a normal reply
   - a built-in tool call
   - generated Python code
4. run the selected tool or expose the generated code through the UI

This keeps the assistant more grounded than a plain chat interface and makes common operations more reliable.

## Recommended Models

Good starting choices:
- `qwen3.5`
- `qwen3.5:27b`

`qwen3.5` is the current default because it has performed well in this workflow.

## Current Limitations

- model output can still be inconsistent, especially for generated code
- not all requests map cleanly to built-in tools yet
- generated code can still fail if the model invents incorrect napari APIs
- no multi-step task planning yet (complex workflows may require several prompts)
- no image attachment or multimodal input pipeline yet
- performance optimization for very large 2D/3D datasets is still in progress

Most reliable current workflow:
- use built-in tools for common layer inspection and mask/image actions
- use the Prompt Library for repeated tasks
- use generated code when you want explicit review and control

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
