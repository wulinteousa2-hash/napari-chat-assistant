from __future__ import annotations

import json
from pathlib import Path


DEFAULT_UI_STATE = {
    "welcome_dismissed": False,
    "telemetry_enabled": False,
    "assistant_splitter_ratio": 0.45,
    "last_workspace_path": "",
    "sam2_project_path": str(Path.home() / "Projects" / "napari" / "Sam2"),
    "sam2_checkpoint_path": "checkpoints/sam2.1_hiera_large.pt",
    "sam2_config_path": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_device": "cuda",
}


def ui_state_path() -> Path:
    return Path.home() / ".napari-chat-assistant" / "ui_state.json"


def load_ui_state() -> dict:
    path = ui_state_path()
    if not path.exists():
        return dict(DEFAULT_UI_STATE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_UI_STATE)
    return {
        "welcome_dismissed": bool(payload.get("welcome_dismissed", DEFAULT_UI_STATE["welcome_dismissed"])),
        "telemetry_enabled": bool(payload.get("telemetry_enabled", DEFAULT_UI_STATE["telemetry_enabled"])),
        "assistant_splitter_ratio": float(
            payload.get("assistant_splitter_ratio", DEFAULT_UI_STATE["assistant_splitter_ratio"])
        ),
        "last_workspace_path": str(payload.get("last_workspace_path", DEFAULT_UI_STATE["last_workspace_path"])).strip(),
        "sam2_project_path": str(payload.get("sam2_project_path", DEFAULT_UI_STATE["sam2_project_path"])).strip(),
        "sam2_checkpoint_path": str(payload.get("sam2_checkpoint_path", DEFAULT_UI_STATE["sam2_checkpoint_path"])).strip(),
        "sam2_config_path": str(payload.get("sam2_config_path", DEFAULT_UI_STATE["sam2_config_path"])).strip(),
        "sam2_device": str(payload.get("sam2_device", DEFAULT_UI_STATE["sam2_device"])).strip() or "cuda",
    }


def save_ui_state(data: dict) -> None:
    path = ui_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "welcome_dismissed": bool(data.get("welcome_dismissed", DEFAULT_UI_STATE["welcome_dismissed"])),
        "telemetry_enabled": bool(data.get("telemetry_enabled", DEFAULT_UI_STATE["telemetry_enabled"])),
        "assistant_splitter_ratio": float(
            data.get("assistant_splitter_ratio", DEFAULT_UI_STATE["assistant_splitter_ratio"])
        ),
        "last_workspace_path": str(data.get("last_workspace_path", DEFAULT_UI_STATE["last_workspace_path"])).strip(),
        "sam2_project_path": str(data.get("sam2_project_path", DEFAULT_UI_STATE["sam2_project_path"])).strip(),
        "sam2_checkpoint_path": str(data.get("sam2_checkpoint_path", DEFAULT_UI_STATE["sam2_checkpoint_path"])).strip(),
        "sam2_config_path": str(data.get("sam2_config_path", DEFAULT_UI_STATE["sam2_config_path"])).strip(),
        "sam2_device": str(data.get("sam2_device", DEFAULT_UI_STATE["sam2_device"])).strip() or "cuda",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
