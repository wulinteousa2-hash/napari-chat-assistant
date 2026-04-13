from __future__ import annotations

import importlib

from qtpy.QtWidgets import QApplication

from napari_chat_assistant.voice.backends.faster_whisper_backend import backend_status
from napari_chat_assistant.widgets.chat_sections.pending_code_panel import PendingCodePanel


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_backend_status_reports_missing_faster_whisper(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "faster_whisper":
            raise ModuleNotFoundError("No module named 'faster_whisper'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    status = backend_status()

    assert status.available is False
    assert "same Python environment" in status.detail


def test_pending_code_panel_exposes_voice_input_action():
    _app()
    panel = PendingCodePanel()

    assert panel.voice_input_action.text() == "Voice Input"
