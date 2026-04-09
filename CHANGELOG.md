# Changelog

All notable user-facing changes to `napari-chat-assistant` should be documented in this file.

## 2.0.1

- Reduced dock startup load by lazy-opening heavy advanced and analysis widgets such as Atlas Stitch, ROI Intensity Analysis, and Line Profile Analysis only when the user selects them.
- Upgraded assistant message rendering from a small regex-based Markdown subset to a real Markdown parser path with broader support, while keeping the existing styled code blocks and a fallback renderer.

## 2.0.0

- Reframed this release around richer built-in library content rather than another major dock redesign, with expanded reusable prompt, code, and learning material for plugin workflows and model testing.
- Reorganized `Templates` into clearer read-only sections for `Prompt Templates`, `Code Templates`, and `Learning`, replacing the older code-only template framing with a more library-like structure.
- Added a broad new `Learning` library with graduate- and university-level prompt starters across `Microscopy`, `Electron Microscopy`, `Biophotonics`, `Image Formation`, `Quantitative Imaging`, `Statistics`, `Academic Prompting`, and `Language Support`.
- Promoted many everyday prompt examples out of `Prompt Tips` and into reusable `Prompt Templates`, so common tasks such as inspection, threshold preview, morphology cleanup, ROI extraction, projection, and measurement are easier to load and reuse.
- Improved template browsing by preserving tree expansion/selection state during interaction and by adding section-based color accents for stronger visual separation.
- Fixed prompt-library stability issues so clearing the library no longer drops pinned recent custom prompts, and increased the recent prompt/code history cap from 20 to 100 entries.
- Continued wording cleanup across the action library, including clearer names for max intensity projection, SAM segmentation, SAM point initialization, and numbered callouts.
- Added an advanced `Atlas Stitch` workflow to the plugin for specialized stitching/export work. This is available through the advanced surface rather than the main everyday workflow.
- Began a low-risk internal UI refactor by extracting major chat sections into dedicated modules, making the dock code easier to maintain without changing the overall workflow model.
- Improved the chat code-control area with a cleaner two-row layout, transcript `A-` / `A+` font-size controls, and a less prominent placement for advanced reject-feedback actions under `Help`.

## 1.9.0

- Added a new annotation workflow focus with non-destructive text overlays, deterministic promptable annotation tools, and Action-tab entries for common annotation tasks.
- Added automatic labels-to-text annotation so users can annotate objects in a `Labels` layer by centroid with prompts such as `annotate template_blob_labels with particle 1 to 4`.
- Added publication-style callout annotations with external label boxes and leader lines for 2D label layers, allowing segmentation results to be presented as figure-style callouts instead of plain text markers.
- Added boxed title labeling above 2D images with `outside_top` placement and `left`, `center`, or `right` alignment for prompts such as `add title WT Group N=10 above the image on the left`.
- Added a dedicated text annotation editor dialog under `Advanced`, plus prompt routing examples so annotation requests map more reliably to deterministic tools.
- Expanded workspace persistence for managed annotation layers so text-overlay state round-trips more cleanly with saved workspaces.
- Improved workspace loading responsiveness with staged restore, a lightweight progress popup, deferred heavy source-backed layer loading, and final saved-layer-order preservation during restore.
- Improved dock resizing so the main assistant widget expands with the napari dock instead of staying stuck at a compact initial height.
- Fixed ROI intensity refresh behavior so newly drawn Shapes ROIs update more reliably without requiring visibility-mode toggles.
- Added RGB support to ROI intensity measurement by reducing truecolor layers to a luminance plane for histogram and ROI summary workflows.

## 1.8.4

- Added a spectral-aware workspace path so layers derived from `napari-nd2-spectral-ome-zarr` are saved as source-plus-view recipes instead of being re-exported as standalone OME-Zarr image assets.
- Workspace restore can now rebuild spectral-derived `visible sum`, `truecolor`, and raw spectral views from the original source `.ome.zarr` through the spectral reader when that plugin is installed.
- Fixed workspace save failures caused by non-JSON runtime metadata attached to spectral-derived napari layers, such as dask-backed `spectral_cube` objects.
- Added regression coverage for sanitized non-JSON metadata and spectral source-recipe save/load behavior in the workspace test suite.

## 1.8.3

- Improved chat onboarding so broad questions such as `who are you?`, `how do I start?`, and no-image demo requests are routed more naturally instead of falling into analysis-specific clarification loops.
- Added built-in synthetic demo image generation for quick 2D grayscale, 3D grayscale, and RGB testing workflows directly from chat.
- Unified more multi-turn behavior around recent-action state so the assistant can explain the last threshold result, distinguish histogram versus intensity-summary workflows more clearly, and keep follow-up questions on topic.
- Added local follow-up handling for threshold refinement requests such as making a mask stricter or including more area, with reuse of the last threshold settings on the current or recent image.
- Improved recent-action handoff so users can more naturally continue into histogram views or ROI Intensity Analysis from the image they were just working on.
- Replaced threshold jargon like `polarity=bright` in chat-facing replies with more imaging-friendly wording such as keeping brighter or dimmer regions as foreground.

## 1.8.2

- Added an explicit `matplotlib` runtime dependency because the ROI intensity, line-profile, and group-comparison widgets import Matplotlib plot canvases directly.
- Kept the `1.8.1` workspace dependency fallback so fresh installs continue to load the main dock even when OME-Zarr-related packages are incomplete.

## 1.8.1

- Fixed a plugin startup failure caused by importing workspace persistence during dock initialization when the OME-Zarr stack was incomplete in the active napari environment.
- Added an explicit `numcodecs` dependency so workspace persistence installs with the required codec package more reliably.
- Changed workspace save/load wiring to import lazily and show a clear install-repair message when workspace dependencies are unavailable, allowing the rest of the plugin to keep loading normally.

## 1.8.0

- Switched workspace asset storage to a `workspace.json` manifest plus OME-Zarr sidecar assets for generated `Image` and `Labels` layers, making saved workspaces more scalable for large derived data such as montages and presentation outputs.
- Added safer workspace overwrite behavior by writing new manifests and asset folders to temporary paths first, then replacing the old workspace only after the new save completes.
- Expanded workspace persistence to include `Points` layers so SAM2 prompt layers now round-trip with coordinates, features, colors, symbols, shown-state, and out-of-slice display.
- Added a broad set of binary-mask operations to the deterministic `Masks` action library, including `Convert To Mask`, `Erode`, `Dilate`, `Open`, `Close`, `Median`, `Outline`, `Skeletonize`, `Distance Map`, `Ultimate Points`, `Watershed`, and `Voronoi`.
- Split binary-mask action behavior more cleanly between destructive cleanup operations that snapshot and replace the current mask, and derived-result operations that create new image or labels layers.
- Improved `Actions -> Masks` preview text so users can see parameter hints and prompt examples for fine adjustment, such as `radius`, `min_size`, `connectivity`, and polarity choices.
- Simplified plugin help behavior by moving UI help into the `Help` menu as a persistent toggle, defaulting it off for expert workflows, and reducing accidental local-help interception of normal requests.
- Expanded the `Help` menu into a small plugin information hub with `Prompt Tips`, `What's New`, `About`, `Report Bug`, and `UI Help Enabled`.
- Updated `Layer Context -> Layers` and related prompt-building workflows so users can insert exact layer names more naturally when editing prompts or code.
- Added hover and pressed styling for category-colored shortcut buttons so the deterministic shortcut surface feels more responsive without changing the workflow model.
- Updated README workflow framing to better match the mature dock layout and the plugin’s current hybrid design of chat, code, templates, actions, shortcuts, and workspace state.

## 1.7.0

- Reframed the plugin as a hybrid napari workbench rather than a chat-only dock, with a clearer progression from AI-guided prompting to deterministic one-click execution.
- Added a full `Actions` catalog so users can browse built-in functions by category and run them directly without routing through the model.
- Replaced the earlier split `Pinned Actions` and `Quick Actions` concept with a unified `Shortcuts` system for user-defined one-click buttons.
- Added customizable `Shortcuts` layouts with add-row, remove-row, clear, save-setup, and load-setup support so users can build their own button-driven workflow surfaces.
- Added `Workspace` as a first-class `Actions` category with deterministic buttons for `Save Workspace`, `Save Workspace As`, `Load Workspace`, and `Restore Last Workspace`.
- Expanded deterministic layer-control workflows with `Delete All`, `Isolate Selected`, `Hide All`, `Show All`, and improved layer-scale shortcuts.
- Brought interactive analysis widgets more clearly into the workbench model by exposing `ROI Intensity Analysis`, `Line Profile Analysis`, `Group Comparison Statistics`, `SAM2 Setup`, and `SAM2 Live` through the deterministic `Actions` surface.
- Continued refining microscopy- and statistics-facing wording so ROI measurement, line-profile analysis, and group comparison read more like imaging software workflows than generic widget terminology.
- Added a dedicated `delete_all_layers` backend tool so full-viewer cleanup is available from both actions and prompt routing.
- Improved workspace manifest saving for mixed image and Shapes sessions by fixing Shapes serialization for list-like per-shape display values such as `edge_width`.
- Improved `Layer Context -> Layers` insertion with a second `Inline` mode so layer names can be inserted at the current cursor position without forcing a new line.
- Promoted synthetic grayscale and RGB SNR sweeps into stable `Templates > Data` entries for repeatable testing and demos.
- Clarified plugin-runtime execution in code-facing UI so `Run My Code` and `Refine My Code` better communicate the difference between this plugin environment and generic napari QtConsole scripting.
- Strengthened the overall design direction around minimizing click count and time-to-task, making deterministic execution and reusable user-defined controls part of the core product rather than an add-on.

## 1.6.1

- Added interactive ROI measurement tools with a new `ROI Intensity Metrics` widget for live shape-based measurement, histogram and table views, renameable ROI labels, chat insertion, and CSV export.
- Added an interactive `Line Profile Gaussian Fit` widget for line-based measurements with live profile plotting, Gaussian fitting, renameable line labels, chat insertion, and CSV export.
- Added a new `Group Comparison Stats` widget for whole-image and ROI-based group comparisons with descriptive statistics, assumption checks, test selection, plots, and CSV export.
- Added workspace manifest save/load support so users can restore layer order and recoverable display state from a lightweight workspace file.
- Added routing and dispatcher support so chat can open the new measurement and statistics widgets directly from user requests.
- Added new built-in ROI-group and image-group comparison tools in the workbench tool registry.
- Improved layer profiling and context handling for `Shapes` layers so ROI geometry is recognized more explicitly in assistant workflows.
- Expanded template support for widget-style measurement workflows and clarified the separation between `Code` snippets and runnable `Templates`.
- Updated chat UI, help text, and saved UI state handling to support workspace persistence and the newer measurement/statistics workflows.

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
