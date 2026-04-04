from __future__ import annotations

from napari_chat_assistant.agent.tools import assistant_system_prompt


def test_assistant_system_prompt_includes_response_examples():
    prompt = assistant_system_prompt()

    assert "Response examples:" in prompt
    assert '"tool":"delete_layers"' in prompt
    assert '"tool":"create_analysis_montage"' in prompt
    assert "layer_binding_hints" in prompt
    assert "img_a" in prompt
    assert "em_2d_snr_low" in prompt
    assert "selected one" in prompt
    assert '"2" or "second"' in prompt
    assert '"ok", "okay", "go", or "continue"' in prompt
    assert "do not use `run_in_background(...)`" in prompt
    assert "setting layer.scale" in prompt
