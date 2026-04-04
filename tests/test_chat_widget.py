from __future__ import annotations

import importlib

import pytest

from napari_chat_assistant.widgets.chat_widget import _workspace_state_functions


def test_workspace_state_functions_reports_missing_dependency(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "napari_chat_assistant.agent.workspace_state":
            raise ModuleNotFoundError("No module named 'numcodecs'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="Workspace save/load is not available"):
        _workspace_state_functions()
