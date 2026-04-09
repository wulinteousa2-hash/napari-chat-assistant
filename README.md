# napari-chat-assistant

[![License MIT](https://img.shields.io/pypi/l/napari-chat-assistant.svg?color=green)](https://github.com/wulinteousa2-hash/napari-chat-assistant/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-chat-assistant.svg?color=green)](https://pypi.org/project/napari-chat-assistant)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-chat-assistant.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-chat-assistant)](https://napari-hub.org/plugins/napari-chat-assistant)

Local, Ollama-powered AI and deterministic workbench for napari image-analysis workflows.

`napari-chat-assistant` adds a dock widget inside napari that understands the active viewer session, runs built-in image-analysis actions, generates executable napari Python code when a request goes beyond the current toolset, and lets users promote repeatable tasks into one-click shortcuts.

The goal is not to bolt a generic chatbot onto a viewer. The goal is to turn napari into a more practical analysis workspace for people who work with microscopy and other large multidimensional imaging datasets, especially users who want local AI help, reproducible workflows, direct control over their data, and fewer clicks per task.

## What's New In 2.0.0

Version `2.0.0` focuses on stronger built-in library content rather than another major interface redesign.

- Reorganized `Templates` into clearer `Prompt Templates`, `Code Templates`, and `Learning` sections.
- Added a much larger `Learning` library for microscopy, EM, biophotonics, image formation, quantitative imaging, statistics, academic prompting, and language support.
- Moved many common examples out of `Prompt Tips` into reusable prompt templates so users can load and run them directly.
- Improved template browsing with clearer structure, preserved tree state, and color-coded sections.
- Added chat transcript `A-` / `A+` controls so users can adjust reading size directly in the dock.
- Fixed prompt-library stability issues, including pinned recent items surviving `Clear`, and increased recent history capacity.
- Added `Atlas Stitch` as an advanced workflow for specialized stitching and export tasks.

For complete release history, see [CHANGELOG.md](CHANGELOG.md).

## Who It Is For

This plugin is built for:
- imaging core facility users
- researchers, staff scientists, and students working with imaging data
- teachers and educators running imaging demos or training sessions
- It is also designed for users who prefer to describe analysis goals in natural language instead of memorizing commands or writing scripts.

It is especially useful when you:
- inspect large 2D or 3D image data in napari
- move between interactive viewing, measurement, and Python-based analysis
- want a local open-weight model instead of a cloud service
- need to save, reuse, restore, and teach common imaging workflows
- want fast deterministic actions and one-click shortcuts for repeated tasks

## Why It Is Different

This plugin is built from practical imaging workflow needs, not from the idea of putting a generic chatbot beside a viewer.

It is designed around how work actually happens in napari:
- start from the data already open in the viewer
- inspect layers, objects, and regions of interest
- run the next analysis step through chat, actions, templates, shortcuts, or code
- review the result directly in the same session
- save useful workflows for later reuse

The assistant is grounded in the live napari session. It can inspect loaded layers, use ROI context, run built-in analysis actions, generate or refine viewer-bound Python when needed, and support deterministic one-click workflows through `Actions` and `Shortcuts`. The result is closer to an imaging workbench than a chat panel.

## Interaction Model

The plugin now supports a deliberate spectrum of interaction styles:
- `Prompt`: AI-first natural language requests
- `Code`: direct viewer-bound Python for users who want exact control
- `Templates`: reusable examples and built-in starting points
- `Actions`: deterministic built-in functions that can be run directly
- `Shortcuts`: user-defined one-click action buttons for repeated daily work

This is now a core design principle of the plugin: reduce how many clicks and how much time it takes for a user to complete a task.

## Interface Overview

![Interface overview](docs/interface-overview.png)

The main dock is organized into a small number of practical work areas:

1. `Connection and model controls`
   Select the local model, monitor status, and manage the backend connection.

2. `Layer Context`
   Review the active workspace layers and insert exact layer names into prompts or code.

3. `Library`
   Browse reusable `Prompts`, `Code`, `Templates`, and deterministic `Actions`.

4. `Shortcuts`
   Keep your most-used actions as one-click buttons for repeated work.

5. `Session`
   Save or restore workspace state and access activity, telemetry, and diagnostics.

6. `Chat`
   See the assistant transcript, generated code, and direct action feedback in one place.

7. `Prompt`
   Enter natural language requests or paste Python for `Run My Code` and `Refine My Code`.

## What You Can Do

Current workflows include:
- inspect the selected layer or named layers with structured summaries
- review live layer context and insert exact layer names into prompts or code
- run built-in tools for enhancement, thresholding, binary mask cleanup, measurement, projection, cropping, montage, presentation, and layer visibility control
- add non-destructive annotation overlays, including free text, particle labels, callout labels with leader lines, and boxed titles above the image
- use deterministic `Actions` for common workflows without depending on prompt phrasing
- build and save your own `Shortcuts` layouts for repeated one-click work
- inspect ROI context and measure or extract values from `Labels`, `Shapes`, and line-based workflows
- use interactive analysis widgets such as `ROI Intensity Analysis`, `Line Profile Analysis`, and `Group Comparison Statistics`
- access SAM2 setup, live preview, box prompting, points prompting, and mask refinement from the same workbench
- browse built-in prompt templates, code templates, and learning templates for plugin workflows, teaching, and model testing
- generate napari Python code when no built-in tool is the right fit
- paste and run your own viewer-bound Python from the prompt box with `Run My Code`
- repair or explain broken pasted Python with `Refine My Code`
- save, pin, tag, rename, and reuse prompts and code from the local Library
- use built-in synthetic data generators for repeatable testing, teaching, and workflow development
- learn from built-in content for microscopy, electron microscopy, imaging physics, quantitative imaging, statistics, academic prompting, and language support
- save and restore workspace state with a JSON manifest plus OME-Zarr assets for generated image and labels data
- use `Atlas Stitch` from the advanced menu for specialized stitching and export workflows

Example requests:
- `Inspect the selected layer`
- `Preview threshold on the selected image`
- `Apply gaussian blur to the selected image with sigma 1.2`
- `Remove small objects from the selected mask with min_size 64`
- `Run watershed on the selected mask`
- `Measure labels table for the selected labels layer`
- `Annotate template_blob_labels with particle 1 to 4`
- `Annotate template_blob_labels using publication-style callout labels with leader lines.`
- `Add title WT Group N=10 above the image on the left`
- `Inspect the current ROI`
- `Extract ROI values from the selected image using the current ROI`
- `Open ROI intensity analysis`
- `Initialize a SAM2 points prompt layer for the selected image`
- `Write napari code to plot object area by condition`

## Local-First By Design

The assistant runs on local open-weight models through Ollama:
- no API key required
- no cloud dependency
- no internet requirement during normal use
- no image data leaves your workstation

This makes it a better fit for research and facility environments where users want privacy, controllability, and local reproducibility.

## Quick Start

1. Install Ollama and pull a local model.
2. `pip install napari-chat-assistant`
3. Open `Plugins -> Chat Assistant` in napari and start with a concrete prompt such as `Inspect the selected layer`.

## Requirements

- Python 3.9+
- napari
- Ollama installed locally and running on the same machine
- at least one local Ollama model such as `nemotron-cascade-2:30b`

Core Python dependencies used by the plugin are installed with the package itself.

Optional:
- `napari-nd2-spectral-ome-zarr` for ND2 export, spectral viewer, and spectral analysis integration
- external SAM2 project, weights, and config if you want the experimental SAM2 workflow

Notes:
- The plugin does not bundle the Ollama server or model weights.
- Model memory requirements vary substantially by model tag.
- Larger local models may require significant RAM or VRAM.

Tested during development on an NVIDIA DGX Spark workstation.

## Installation

### 1. Install Ollama

macOS and Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

Windows:
- download from `https://ollama.com/download/windows`
- install Ollama
- start the Ollama service or application

Pull at least one model before using the plugin:

```bash
ollama pull nemotron-cascade-2:30b
```

Optional alternatives:

```bash
ollama pull gpt-oss:120b
ollama pull qwen3-coder-next:latest
ollama pull qwen3.5
ollama pull qwen2.5:7b
```

### 2. Install the plugin

For normal use:

```bash
pip install napari-chat-assistant
```

For development:

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
5. Use `Load` if you want to warm the selected model before the first request.
6. Start chatting, or use the Library for repeatable tasks and reusable code.

If you already have Python code you want to try, paste it into the Prompt box and click `Run My Code`. This runs viewer-bound code directly inside the plugin runtime without opening QtConsole.

If your pasted code fails or needs adaptation to the current viewer session, click `Refine My Code` to send it back through the assistant with the current napari context and local validation feedback.

The assistant works best when prompts describe a concrete action. Natural language is fine.

Examples:
- `Inspect the selected layer`
- `Preview threshold on em_2d_snr_mid`
- `Apply gaussian denoise to em_2d_snr_low with sigma 1.2`
- `Fill holes in mask_messy_2d`
- `Remove small objects from mask_messy_2d with min_size 64`
- `Keep only the largest connected component in mask_messy_2d`
- `Measure labels table for rgb_cells_2d_labels`
- `Annotate template_blob_labels with publication-style callout labels`
- `Move the particle labels outside with boxes and lines`
- `Place title Control outside top right`
- `Create a max intensity projection from em_3d_snr_mid along axis 0`
- `Crop em_2d_snr_high to the bounding box of em_2d_mask with padding 8`
- `Inspect the current ROI`
- `Extract ROI values from em_2d_snr_mid using em_2d_mask`

## Typical Workflow

1. Open your image or volume in napari.
2. Use `Layer Context` if you want to copy or insert exact layer summaries into the Prompt box.
3. Ask the assistant to inspect the layer and suggest the next step, or browse `Actions` if you already know the function you want.
4. Run a built-in tool for denoising, thresholding, cleanup, measurement, layout, layer visibility, or workspace restore.
5. Add repeated actions to `Shortcuts` so common tasks become one-click operations.
6. Select an ROI or object in the viewer if you want local analysis.
7. Ask for code when you need a custom plot, statistics, or reusable script.
8. Use `Run My Code` for your own Python and `Refine My Code` when pasted code fails or needs repair for this plugin environment.
9. Save useful prompts, code snippets, templates, workspaces, and shortcut setups for later reuse.

This is the core value of the plugin: users can stay in the viewer, interact with the data, ask questions, run analysis, and keep the resulting workflow close to the image session.

## Synthetic Data Templates

Use the Library `Templates > Data` area or built-in code snippets to load repeatable synthetic datasets.

Current built-in synthetic generators include:
- Synthetic 2D SNR Sweep Gray
- Synthetic 3D SNR Sweep Gray
- Synthetic 2D SNR Sweep RGB
- Synthetic 3D SNR Sweep RGB
- messy masks 2D/3D

These create named layers so you can test built-in tools quickly without hunting for sample data. Labels layers from these synthetic datasets can also be used as ROIs for ROI inspection and value extraction.

Example pipeline:
1. Load an RGB image or run `Synthetic 2D SNR Sweep RGB`.
2. Ask the assistant to split the RGB image into separate grayscale channels.
3. Create an analysis montage from the split channel images for side-by-side review.
4. If you want ROI-based intensity measurements, open `ROI Intensity Analysis` and draw area ROIs.
5. If you want line-based measurements, open `Line Profile Analysis` and draw line ROIs.
6. Use deterministic `Actions` for thresholding, mask cleanup, watershed, connected components, or ROI-based inspection as needed.
7. Save the session with `Save Workspace` so the manifest and generated image or labels data are stored together using JSON plus OME-Zarr assets.
8. If no built-in tool fits the task, ask the assistant to generate code.
9. If the generated code does not work in the current plugin runtime, use `Refine My Code` and run it again.

## Current Features

### Session-aware tools

The workbench currently supports built-in tools for:
- listing all layers
- inspecting the selected layer
- inspecting a specific named layer
- CLAHE contrast enhancement for grayscale 2D and 3D images
- batch CLAHE across multiple image layers
- Gaussian denoising for grayscale image layers
- threshold preview
- threshold apply
- batch threshold preview and apply
- mask measurement
- batch mask measurement
- mask cleanup operations such as hole filling, small-object removal, and largest-component selection
- connected-component labeling for binary masks
- per-object measurement table summaries for labels layers
- max-intensity projection for 3D grayscale images
- cropping one layer to the bounding box of another layer
- showing image layers in a comparison grid
- hiding the image grid view and restoring hidden non-image layers
- arranging layers for presentation in rows, columns, grids, or repeated groups
- showing, hiding, isolating, and restoring layer visibility directly from chat
- ROI inspection and grayscale value extraction from labels or shapes regions
- registry-backed tool execution as the foundation for future workflow and pipeline expansion
- deterministic `Actions` and reusable `Shortcuts` for repeated one-click work

Interactive analysis widgets now built into the workflow include:
- `ROI Intensity Analysis` for shape-based fluorescence or grayscale ROI measurements with tables, histograms, normalization views, CSV export, and on-canvas ROI labeling
- `Line Profile Analysis` for profile-based measurements along user-defined lines, including Gaussian fitting
- `Group Comparison Statistics` for two-group comparison workflows on image-level or ROI-derived measurements, with descriptive statistics, assumption checks, and test summaries
- `SAM2 Setup` and `SAM2 Live` entry points so segmentation workflows can be opened directly from the same workbench

Layer inspection is backed by a deterministic profile object that includes:
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

Additional built-in workflow tools currently exposed through chat include:
- `gaussian_denoise`
- `remove_small_objects`
- `fill_mask_holes`
- `keep_largest_component`
- `label_connected_components`
- `measure_labels_table`
- `project_max_intensity`
- `crop_to_layer_bbox`
- `inspect_roi_context`
- `extract_roi_values`

### Code generation workflows

When a request is not covered by a built-in tool, the assistant can return napari Python code instead of forcing the wrong tool.

Generated code can be:
- copied to the clipboard
- reviewed in chat
- executed from the plugin
- repaired or explained in place when you use `Refine My Code` on pasted or failed user code

You can also paste your own Python directly into the Prompt box and run it from the plugin with `Run My Code`, without switching to QtConsole.

Use assistant-generated code when you want a reusable script or need custom logic beyond the current built-in tools.

Use `Run My Code` when you already have Python you want to test quickly inside the current napari session.

Use `Refine My Code` when your own code fails validation, errors at runtime, or needs adjustment to the current napari viewer state.

### Selective session memory

The assistant includes bounded session memory with three states:
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

### Library

The plugin includes a persistent Library for repeatable workflows and reusable code:
- built-in starter prompts
- built-in synthetic data generators and reusable code examples in the `Code` tab
- built-in categorized starter templates in the `Templates` tab
- deterministic built-in functions in the `Actions` tab
- recent prompts captured automatically
- saved prompts for reusable tasks
- pinned prompts for high-frequency workflows
- recent and saved code snippets in a separate `Code` tab

Interaction:
- single click loads a prompt or code snippet into the editor
- double click sends a prompt directly or runs a code snippet
- templates can be previewed, loaded into the Prompt box, or run directly
- actions can be previewed, added to `Shortcuts`, or run directly
- right click can rename or edit tags for saved and recent prompt/code items
- multi-select supports Shift/Ctrl selection for batch actions
- `Delete` can remove saved prompts, recent prompts, code snippets, or hide built-in prompts
- `Clear` removes unpinned recent prompt/code items while keeping saved and pinned items

Logic:
- `saved` means a user-managed prompt you want to keep as your own reusable entry
- `pinned` means keep this prompt surfaced at the top of the library
- a prompt can be pinned without being saved
- built-in prompts are shipped examples; deleting them hides them from the current local library view
- built-in code entries include synthetic data generators and starter `Run My Code` examples
- built-in code entries remain visible even if the same snippet also appears in recent history
- code snippets can be tagged and renamed so they are easier to reference later in workflows

This is designed for users who want repeatable automation without committing everything to full scripting.

### Actions and Shortcuts

`Actions` is the deterministic side of the plugin. It exposes built-in functions by category so users can browse what the plugin can do without depending on exact prompt phrasing.

`Shortcuts` is the user-defined one-click layer above `Actions`:
- add your most-used actions from the `Actions` catalog
- grow the button layout when you need more space
- save and load shortcut setups for different tasks or teaching sessions
- remove repeated browsing when you already know the workflow you want

This is now part of the core product direction: keep AI available, but let mature workflows become fast, button-driven, and reusable.

### Optional ND2 and spectral integration

If `napari-nd2-spectral-ome-zarr` is installed, the assistant can open:
- the ND2-to-OME-Zarr export widget
- the Spectral Viewer widget
- the Spectral Analysis widget

This lets chat act as an entry point for Nikon ND2 conversion and spectral workflows without rebuilding those UIs inside this plugin.

Install links:
- GitHub: `https://github.com/wulinteousa2-hash/napari-nd2-spectral-ome-zarr`
- napari Hub: `https://napari-hub.org/plugins/napari-nd2-spectral-ome-zarr.html`

### Experimental SAM2 integration

Behavior:
- SAM2 is accessed from `Advanced`, not from the main toolbar
- `SAM2 Setup` is always available from `Advanced`
- `SAM2 Live` stays disabled until the backend is configured and passes readiness checks
- the rest of the assistant remains usable even if SAM2 is not configured

Current setup expects:
- a working Python environment that already includes the dependencies required by SAM2
- an external SAM2 project path
- a valid checkpoint path
- a valid config path

`napari-chat-assistant` now ships its own bundled SAM2 adapter in
`napari_chat_assistant.integrations.sam2_adapter`, so users only need the SAM2 repo,
checkpoint, and config files in the normal places.

The `SAM2 Setup` dialog now includes:
- `Auto Detect` to scan common local clone locations and fill likely project, checkpoint, and config paths
- `Setup Help` for short setup commands and field tips

Minimal install:

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```

Typical setup flow:
1. Start napari from the environment that contains your SAM2 dependencies.
2. Open `Plugins -> Chat Assistant`.
3. Open `Advanced -> SAM2 Setup`.
4. Click `Auto Detect` first.
5. Confirm or edit the SAM2 project path, checkpoint path, config path, and device.
6. Click `Test`.
7. Save the settings.
8. Open `Advanced -> SAM2 Live` when the backend reports ready.

## UI Overview

### Model connection

- local Ollama base URL
- model picker with discovered local models
- `Test`
- `Load`
- `Setup` help with install, `ollama serve`, and model pull examples
- `Unload`

### Library

- built-in prompts
- built-in synthetic data generators and starter code in the `Code` tab
- built-in templates in the `Templates` tab
- deterministic built-in functions in the `Actions` tab
- recent prompts
- saved prompts
- recent and saved code snippets in the `Code` tab
- pinned prompts
- `saved` keeps your own reusable copy
- `pinned` keeps a prompt at the top regardless of whether it is built-in, recent, or saved
- single click to load
- double click to send from `Prompts` or run from `Code`
- right click to rename or edit tags for saved/recent prompt and code items
- Shift/Ctrl multi-select for batch actions
- `Delete` works on built-in, recent, saved, and code items
- `Clear` keeps saved and pinned items and clears unpinned recent items
- `A-` and `A+` adjust library font size in small steps

### Shortcuts

- one-click user-defined action buttons
- start with a 3x2 layout and add or remove rows as needed
- save and load shortcut setups
- remove single shortcuts from the button grid without resetting the whole layout

### Chat

- multi-line prompt box
- Enter to send
- Shift/Ctrl/Alt+Enter for newline
- transcript showing user messages, assistant replies, tool results, and generated code

### Code actions

- `Reject`
- `Run Code`
- `Run My Code`
- `Copy Code`
- `Advanced`
- `Help`

`Run Code` is for assistant-generated code that has been staged in the chat.

`Run My Code` is for your own pasted Python from the Prompt box when you want to test or iterate directly inside the plugin runtime without opening QtConsole.

`Advanced` contains optional integrations such as experimental SAM2 setup and live preview.

### Current context

- current layer summary from the active napari viewer
- shortened layer names and a compact per-layer summary to avoid over-stretching the left column

### Session

- `Activity` tab shows local status updates, model connection messages, tool execution messages, and code execution/copy actions
- `Telemetry` tab contains the optional telemetry controls
- `Diagnostics` tab provides access to the app log and crash log
- color-highlighted path entries for assistant log, crash log, telemetry log, prompt library, and session memory
- `Enable Telemetry` switch for advanced users
- `Summary`, `Log`, and `Reset` only when telemetry is enabled

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
6. run the selected registry-backed tool or expose the generated code through the UI
7. update session memory from explicit user feedback or successful follow-up behavior

This keeps the assistant more grounded than a plain chat interface and makes common operations more reliable.

## Design Direction

The intended architecture is:

1. natural language at the user surface
2. registry-backed tools underneath
3. explicit scope resolution for full-layer and ROI/subregion workflows

This means users should be able to ask for operations in normal language, while the plugin resolves those requests into deterministic tool calls with structured parameters.

### Tool model

Registered tools are the common execution model for:
- chat-triggered actions
- reusable UI actions
- future workflow and pipeline steps
- future plugin-contributed extensions

Each tool is moving toward a shared definition with:
- stable name
- parameter schema
- supported layer types
- prepare/execute/apply lifecycle
- UI metadata
- provenance metadata

### Scope model

For imaging analysis, operations may target:
- the full layer
- a labels mask
- a specific labels object
- a shapes ROI
- a bounding-box crop

Natural language can express these requests, but the plugin still needs deterministic binding rules underneath.

The preferred resolution order is:
1. explicit user binding such as `image_a using roi_shapes`
2. current viewer selection when there is only one clear match
3. a short clarification question when multiple bindings are plausible

Session memory should remain secondary context. Current viewer state and explicit user clarification should remain the primary source of truth.

## Recommended Models

For a broader list of models tested during development, see [docs/tested_models.md](docs/tested_models.md).

Good starting choices:
- `nemotron-cascade-2:30b`
- `gpt-oss:120b`
- `qwen3-coder-next:latest`
- `qwen3.5`
- `qwen2.5:7b`

Selection guidance:
- `nemotron-cascade-2:30b` is the current default and a strong general model for this workflow.
- `gpt-oss:120b` is a large model that can still feel relatively fast in practice on high-memory systems; it is a good option when you want stronger reasoning without moving to a smaller lightweight tag.
- `qwen3-coder-next:latest` is a better candidate for Python and napari code generation, but it is significantly heavier.
- `qwen3.5` remains a useful alternative general model.
- `qwen2.5:7b` is lighter and may fit smaller-memory systems more easily.

Memory note:
- Larger tags require more RAM or VRAM.
- On the DGX Spark setup used during development, `qwen3-coder-next:latest` may need around 100 GB of available memory to run comfortably.

## Current Limitations

- the dataset profiler is still Phase 1 and remains strongest on already-loaded napari layers rather than reader- or file-format-specific workflows
- TIFF vs OME-Zarr adapter behavior is not implemented yet
- ND2 and Zeiss reader-aware adapters are not implemented in this plugin
- the tool registry is in progress; some tools are now registry-backed, but the migration is not complete yet
- session memory is selective and bounded; it is not full conversation memory
- model output can still be inconsistent, especially when falling back to generated code
- some requests still miss built-in tools and fall through to code generation when a stronger built-in workflow would be preferable
- generated code can still fail if the model invents incorrect napari APIs or unsupported imports
- multi-step workflow planning and replay are not implemented yet
- no image attachment or multimodal input pipeline yet
- performance optimization for very large 2D/3D datasets is still in progress
- hard native crashes in Qt/C-extension code may not be captured cleanly by the plugin crash log even when normal plugin errors are logged

Most reliable current workflow:
- use built-in tools for common layer inspection and mask/image actions
- trust current viewer context and current layer profiles over any remembered prior turn
- use the Library for repeated prompts, demo packs, and reusable code
- use generated code when you want explicit review and control
- use `Run My Code` when you already have working Python and want to test it directly inside napari

For demo and education workflows:
- ask for code that uses the current napari `viewer`
- avoid prompts that create a second `napari.Viewer()` or call `napari.run()`
- prefer docked widgets over unmanaged popup windows for histogram or SNR teaching tools

## Troubleshooting

### Ollama not running

If `Test` fails after restarting your computer, Ollama is usually not running yet.

Start it in a terminal:

```bash
ollama serve
```

Then return to the plugin and click `Test` again.

### Pulling a model

Model downloads are intentionally handled outside the plugin.

To try a different model:
- browse tags at `https://ollama.com/search`
- type the tag into the plugin `Model` field if needed
- pull it in a terminal, for example:

```bash
ollama pull nemotron-cascade-2:30b
```

Then use `Test` to refresh the plugin state.

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
- reject feedback from `Reject`
- approved code execution success or failure

Telemetry is opt-in from the `Session -> Telemetry` tab through `Enable Telemetry`.

For advanced users, the `Session -> Telemetry` tab includes:
- `Summary` to generate a quick in-app summary of recent model speed and behavior
- `Log` to inspect the latest raw JSONL records together with the summary
- `Reset` to clear the local telemetry file and start fresh from the next request

Generated code is also preflight-validated before execution for common dtype mistakes, unsupported napari imports, and unavailable `viewer.*` APIs. When validation blocks execution, the code remains visible and copyable for review or regeneration.

## Release

This package is published to PyPI so napari Hub can discover it.

For maintainer release instructions and PyPI publishing setup, see [RELEASING.md](RELEASING.md).

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
