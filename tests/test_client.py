from __future__ import annotations

import pytest

from napari_chat_assistant.agent import client
from napari_chat_assistant.agent import ui_state


@pytest.mark.parametrize(
    "model,info,expected",
    [
        ("nemotron-cascade-2:30b", {"capabilities": ["completion", "thinking"]}, False),
        ("custom", {"thinking": {"values": [True, False]}}, False),
        ("custom", {"thinking": {"values": ["low", "medium", "high"]}}, "low"),
        ("gpt-oss:20b", {"capabilities": ["thinking"]}, "low"),
        ("qwen2.5:7b", {"capabilities": ["completion"]}, None),
        ("custom", {"thinking": {"values": [True]}}, None),
    ],
)
def test_fast_mode_uses_supported_model_control(monkeypatch, model, info, expected):
    client._fast_reasoning_setting.cache_clear()
    calls = []

    def fake_http(url, payload=None, **kwargs):
        calls.append((url, payload))
        if url.endswith("/api/show"):
            return info
        return {"message": {"content": '{"action":"reply","message":"Ready"}'}}

    monkeypatch.setattr(client, "http_json", fake_http)
    metadata = {}
    for _ in range(2):
        client.chat_ollama(
            "http://localhost:11434", model, system_prompt="test", user_payload={},
            options={}, fast_mode=True, request_metadata_holder=metadata,
        )
    requests = [payload for url, payload in calls if url.endswith("/api/chat")]
    assert len([url for url, _ in calls if url.endswith("/api/show")]) == 1
    for payload in requests:
        if expected is None:
            assert "think" not in payload
        else:
            assert payload["think"] == expected
        assert "think" not in payload["options"]
    assert metadata["fast_mode"] is True


def test_fast_mode_off_restores_model_defaults_without_discovery(monkeypatch):
    client._fast_reasoning_setting.cache_clear()
    calls = []

    def fake_http(url, payload=None, **kwargs):
        calls.append((url, payload))
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(client, "http_json", fake_http)
    client.chat_ollama("http://localhost", "custom", system_prompt="test", user_payload={}, options={})
    assert len(calls) == 1
    assert calls[0][0].endswith("/api/chat")
    assert "think" not in calls[0][1]


def test_fast_mode_keeps_chat_available_without_model_metadata(monkeypatch):
    client._fast_reasoning_setting.cache_clear()

    def fake_http(url, payload=None, **kwargs):
        if url.endswith("/api/show"):
            raise RuntimeError("HTTP 404")
        assert "think" not in payload
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(client, "http_json", fake_http)
    assert client.chat_ollama(
        "http://localhost", "custom", system_prompt="test", user_payload={}, options={}, fast_mode=True,
    ) == "ok"


def test_fast_mode_preference_survives_reload(monkeypatch, tmp_path):
    monkeypatch.setattr(ui_state, "ui_state_path", lambda: tmp_path / "ui_state.json")
    assert ui_state.load_ui_state()["fast_mode"] is False
    for enabled in (True, False):
        state = ui_state.load_ui_state()
        state["fast_mode"] = enabled
        ui_state.save_ui_state(state)
        assert ui_state.load_ui_state()["fast_mode"] is enabled
