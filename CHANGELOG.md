# Changelog

All notable user-facing changes to `napari-chat-assistant` should be documented in this file.

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
