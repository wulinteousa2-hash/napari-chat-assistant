from __future__ import annotations

import json
from pathlib import Path


def ui_state_path() -> Path:
    return Path.home() / ".napari-chat-assistant" / "ui_state.json"


def load_ui_state() -> dict:
    path = ui_state_path()
    if not path.exists():
        return {"welcome_dismissed": False, "telemetry_enabled": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"welcome_dismissed": False, "telemetry_enabled": False}
    return {
        "welcome_dismissed": bool(payload.get("welcome_dismissed", False)),
        "telemetry_enabled": bool(payload.get("telemetry_enabled", False)),
    }


def save_ui_state(data: dict) -> None:
    path = ui_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "welcome_dismissed": bool(data.get("welcome_dismissed", False)),
        "telemetry_enabled": bool(data.get("telemetry_enabled", False)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
