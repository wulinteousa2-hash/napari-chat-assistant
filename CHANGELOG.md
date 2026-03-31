# Changelog

All notable user-facing changes to `napari-chat-assistant` should be documented in this file.

## 1.4.3

- Added built-in image grid view for side-by-side comparison of currently loaded image layers.
- Added a built-in action to turn image grid view off and restore any non-image layers hidden for comparison.
- Added built-in plugin UI help so users can ask what controls do and get short usage tips directly in chat.
- Expanded UI help coverage for Library, Prompt, code actions, model controls, telemetry, diagnostics, status, and prompt-writing tips.
- Kept experimental SAM2 under `Advanced` and kept the clearer `Load`, `Unload`, `Test`, `Setup` model flow.

## 1.4.2

- Changed `Use` to `Load` and made it warm the selected Ollama model instead of only saving the selection.
- Reordered model controls to `Load`, `Unload`, `Test`, `Setup` and added clearer tooltips.
- Streamlined status messages for loading, replies, tools, and generated code.
- Strengthened napari-specific code validation and added narrow automatic repair for common generated-code mistakes.
- Blocked generated code from creating a new napari `Viewer` instead of using the current session viewer.
- Improved chat code-block rendering with a more editor-like dark style and Python token coloring.
- Kept experimental SAM2 under `Advanced` rather than the default toolbar.

## 1.4.1

- Added experimental SAM2 integration as an optional advanced workflow.
- Moved SAM2 access out of the default toolbar and into `Advanced`.
- Improved dock resizing and splitter behavior on larger displays.
- Added ROI-aware inspection and grayscale value extraction support for `Labels` and `Shapes` layers.

## 1.4.0

- Introduced a new tool-registry foundation and migrated the first built-in tools to registry-backed execution.
- Added new built-in workflow tools including Gaussian denoising, mask cleanup, connected-component labeling, labels-table measurement, max-intensity projection, and bbox crop.
- Added built-in demo packs for EM-style grayscale data, fluorescent RGB cell data, SNR sweeps, and messy mask cleanup tests.
- Updated the Help panel and README with shorter natural-language guidance, demo-pack examples, and clearer workflow-oriented usage.
- Improved Library behavior so built-in demo code entries remain visible with stable built-in titles even after use.

## 1.3.1

- Renamed `Prompt Library` to `Library`.
- Added a `Code` tab alongside `Prompts` for reusable runnable snippets.
- Added built-in background-execution demo code snippets to the Code tab.
- Added right-click rename and tag editing for prompt and code items.
- Reorganized session information into `Activity`, `Telemetry`, and `Diagnostics` tabs.
- Shortened several UI button labels and moved detail into tooltips to reduce layout pressure.
- Added `run_in_background(...)` to the code runtime for heavy work that should not block the napari UI.

## 1.3.0

- Added `Run My Code` so you can paste Python into the Prompt box and run it directly without opening QtConsole.
- Kept `Run Code` focused on assistant-generated code after review.
- Improved generated-code safety with better local validation and clearer `scikit-image` error messages.
- Reformatted intensity summaries into a more readable block layout.
- Fixed UI stability issues, including left-panel width shifting from long model/status text.
- Prevented crashes caused by deleted Analysis controls being refreshed after code execution.
- Changed the waiting indicator to a simpler sequential dot animation.
- Made telemetry opt-in with `Enable Telemetry`, hidden by default for average users.
- Updated the welcome message and README to reflect the current workflow.
