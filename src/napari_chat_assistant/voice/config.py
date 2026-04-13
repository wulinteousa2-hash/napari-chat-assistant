from __future__ import annotations

VOICE_INPUT_ACTION_LABEL = "Voice Input"
VOICE_INPUT_DIALOG_TITLE = "Voice Input"

VOICE_INPUT_HELP_TEXT = (
    "Voice input is optional.\n"
    "This feature requires `faster-whisper` installed in the same Python environment as napari.\n"
    "Example install:\n"
    "pip install faster-whisper"
)

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNEL_COUNT = 1
DEFAULT_MODEL_SIZE = "base"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_DEVICE = "auto"
