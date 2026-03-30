# Changelog

All notable user-facing changes to `napari-chat-assistant` should be documented in this file.

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
