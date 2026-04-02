# Changelog

All notable user-facing changes to `napari-chat-assistant` should be documented in this file.

## 1.6.0

- Added `Refine My Code` as a first-class recovery workflow beside `Run My Code`, so users can repair pasted or failed viewer-bound Python for the current napari plugin environment without leaving the dock.
- Added structured code-repair context for pasted code, including local validation errors, warnings, repair notes, and current viewer context to improve explanation and rewrite quality.
- Added direct layer visibility controls with built-in tools for `show_layers`, `hide_layers`, `show_only_layers`, and `show_all_layers`.
- Refactored the main dock layout around a compact top model/status bar and reduced the visual weight of configuration controls by moving `Base URL`, `Test`, and `Setup` behind a `Connection` toggle.
- Reworked the old context area into `Layer Context`, with a copyable summary view plus a per-layer quick-action view that supports `Insert` and `Copy` actions for prompt building.
- Made `Layer Context` update live from napari layer and selection events instead of only after plugin-triggered refreshes.
- Made the `Session` section collapsible and closed by default so `Library`, `Chat`, and `Prompt` stay visually primary.
- Improved library usability by collapsing the Templates tree by default, adding tab tooltips for `Prompts`, `Code`, and `Templates`, and darkening the `Prompts` and `Code` list backgrounds to better match the rest of the interface.

## 1.5.0

- Added a new built-in `Templates` tab with a category tree, read-only preview, `Load Template`, and double-click-to-run behavior through `Run My Code`.
- Organized starter templates around a broader napari workbench model with categories for `Data`, `Inspect`, `Process`, `Segment`, `Measure`, `Visualize`, `Compare`, `Workbench`, and `Background Jobs`.
- Added plugin-native starter code designed for this runtime, including templates that use `viewer`, `selected_layer`, and `run_in_background(...)` where appropriate.
- Brought the built-in demo packs into `Templates > Data` so users can browse and launch EM, RGB cell, and messy-mask demo generators from the new template library.
- Added a measurement-focused `Line Profile Gaussian Fit` template that generates synthetic data, adds a line ROI, fits a Gaussian profile, and reports quantitative outputs such as sigma and FWHM.

## 1.4.7

- Clarified README workflow wording so it no longer implies the text-only assistant directly sees image pixels, and instead describes work starting from the data already open in the napari viewer.

## 1.4.6

- Expanded SAM2 into the main release focus with bundled adapter support, setup auto-detect, checkpoint/config discovery, a live-model selector, and a more practical SAM2-managed points workflow for preview and propagation inside napari.
- Improved SAM2 Live prompt interaction with managed prompt-layer initialization, polarity toggling on the active points layer, better point coloring, and clearer live status messages during preview, propagation, and save steps.
- Refactored local Python guardrails into dual-mode validation: strict blocking for assistant-generated code and permissive execution with warnings for user-pasted `Run My Code` workflows.
- Preserved protections against clearly bad napari hallucinations and known dtype hazards while separating hard errors, soft warnings, and repair notes in validation results.
- Continued the broader 1.4.x workflow polish already visible in the current worktree, including SAM2-related UI/help updates and related formatting/documentation refreshes included in this release.

## 1.4.5

- Added local prompt routing for compound imaging requests and a built-in axon-interior extraction workflow for dark-ring EM patterns.
- Strengthened napari code validation to reject invented layer attributes such as `.type` and `._type` before execution.
- Improved `Run My Code` and prompt-library handling so saved code keeps its original formatting more reliably.
- Added a telemetry report CLI and supporting documentation for tested models and telemetry result snapshots.
- Improved experimental SAM2 Live behavior with a non-modal dialog, status/progress feedback, 3D propagation support, and a simpler SAM2-managed points workflow.

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
