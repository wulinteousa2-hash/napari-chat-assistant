from __future__ import annotations

import atexit
import faulthandler
import json
import logging
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path.home() / ".napari-chat-assistant"
APP_LOG_PATH = LOG_DIR / "assistant.log"
CRASH_LOG_PATH = LOG_DIR / "crash.log"
TELEMETRY_LOG_PATH = LOG_DIR / "model_telemetry.jsonl"

_LOGGER_READY = False
_FAULT_HANDLE = None


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_plugin_logger() -> logging.Logger:
    global _LOGGER_READY
    _ensure_log_dir()
    logger = logging.getLogger("napari_chat_assistant")
    if _LOGGER_READY:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    _LOGGER_READY = True
    return logger


def enable_fault_logging() -> Path:
    global _FAULT_HANDLE
    _ensure_log_dir()
    if _FAULT_HANDLE is None:
        _FAULT_HANDLE = open(CRASH_LOG_PATH, "a", encoding="utf-8")
        faulthandler.enable(_FAULT_HANDLE)
        atexit.register(_close_fault_handle)
    return CRASH_LOG_PATH


def _close_fault_handle() -> None:
    global _FAULT_HANDLE
    if _FAULT_HANDLE is None:
        return
    try:
        _FAULT_HANDLE.flush()
        _FAULT_HANDLE.close()
    finally:
        _FAULT_HANDLE = None


def append_telemetry_event(event_type: str, payload: dict) -> Path:
    _ensure_log_dir()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": str(event_type or "").strip() or "unknown",
        **{str(k): v for k, v in dict(payload or {}).items()},
    }
    with TELEMETRY_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return TELEMETRY_LOG_PATH
