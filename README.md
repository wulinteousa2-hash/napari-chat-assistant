# napari-chat-assistant

Local Ollama-powered chat assistant for napari image-analysis workflows.

`napari-chat-assistant` adds a dock widget to napari that stays aware of the current viewer, selected layer, and available layers. It can answer questions about the session, run a controlled set of built-in image-analysis actions, and generate Python code for napari when needed.

## Overview

This plugin is designed for local interactive work inside napari.

Current capabilities:
- connect to a local Ollama server
- discover, pull, and unload local models from the plugin UI
- inspect napari layers and selected-layer properties
- run built-in thresholding and mask tools
- generate Python code for napari workflows
- copy generated code into the napari QtConsole
- optionally run generated code from the plugin

The current default model is:
- `qwen3.5`

## Why This Plugin

Most chat interfaces are detached from the napari viewer. This plugin keeps the assistant inside napari and grounds responses in the current session.

That means the assistant can:
- see the current layers
- see which layer is selected
- distinguish common napari layer kinds such as `Image`, `Labels`, `Points`, and `Shapes`
- report layer dimensions, dtype, visibility, and related properties
- run explicit image-analysis actions rather than only talking about them

## Current Features

### Built-in Tool Actions

The assistant currently supports built-in tools for:
- listing all layers
- inspecting the selected layer
- inspecting a specific named layer
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

### Code Workflows

When the request is not covered by a built-in tool, the assistant can generate napari Python code.

There are two ways to use that code:
- `Run Pending Code`
  Executes the pending code inside the plugin.
- `Copy Pending Code`
  Copies the pending code to the clipboard so you can paste it into the napari QtConsole.

This is useful when you want:
- a script you can inspect first
- code you want to modify manually
- code to run in the napari QtConsole instead of through the plugin

## Requirements

- Python 3.9+
- napari
- Ollama installed on the machine
- a local Ollama model such as `qwen3.5`

The plugin package is self-contained for napari, but it does not bundle the Ollama server or model weights.

## Installation

### Step 1. Install Ollama Provider

Install Ollama on the machine first.

Start the local Ollama server:

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

### Step 2. Install The napari Plugin

Clone the repository and install the plugin:

```bash
git clone https://github.com/wteox2/napari-chat-assistant.git
cd napari-chat-assistant
pip install -e .
```

## Usage

1. Start napari.
2. Open `Plugins -> Chat Assistant`.
3. Leave `Base URL` as `http://127.0.0.1:11434` unless your Ollama server is elsewhere.
4. Choose a model from the `Model` dropdown or type a model tag manually.
5. Click `Test Connection`.
6. Start chatting.

You do not need to rely only on free-form prompts. The plugin is strongest when you ask for concrete napari actions.

Examples:
- `show me my layers`
- `inspect the selected layer`
- `inspect layer LV-nerve`
- `preview threshold for the selected image`
- `apply threshold for dim objects`
- `measure the current mask`
- `write napari code to duplicate the selected layer`
- `give me QtConsole code to print the selected layer shape`

## UI Overview

### Model Connection

- local Ollama base URL
- model picker with discovered local models
- test connection
- save current settings
- pull model
- unload model

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

The assistant is not a fully unrestricted chatbot.

The current strategy is:
1. collect structured napari viewer context
2. send that context and the user request to a local Ollama model
3. let the model return one JSON response
4. interpret that response as either:
   - a normal reply
   - a built-in tool call
   - generated Python code
5. run the selected tool or expose the generated code through the UI

This keeps the assistant more grounded than a plain chat interface and makes common operations more reliable.

## Recommended Models

Good starting choices:
- `qwen3.5`
- `qwen3.5:27b`

`qwen3.5` is the current default because it has performed better in this plugin than the previous `qwen2.5` default.

## Current Limitations

- still an early-stage plugin
- model output can still be inconsistent, especially for generated code
- not all requests map cleanly to built-in tools yet
- generated code can still fail if the model invents incorrect napari APIs
- no multi-step planning loop yet
- no image attachment or multimodal input pipeline yet
- large 2D/3D volumes are not heavily optimized yet

For now, the most reliable path is:
- use built-in tools for common layer inspection and mask/image actions
- use `Copy Pending Code` when you want to inspect or adjust Python yourself

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

MIT. See [LICENSE](/home/wteox2/Projects/napari/napari-chat-assistant/LICENSE).
