from __future__ import annotations

from napari_chat_assistant.agent.template_library import template_library_payload


def test_template_library_includes_stats_category_and_group_comparison_templates():
    payload = template_library_payload()

    assert "Stats" in payload["categories"]
    titles = {str(record.get("title", "")).strip() for record in payload["templates"]}
    assert "Compare Two ROI Groups" in titles
    assert "Compare Two Image Groups" in titles
    assert "Open Group Comparison Widget" in titles


def test_template_library_includes_synthetic_snr_data_templates():
    payload = template_library_payload()

    titles = {str(record.get("title", "")).strip() for record in payload["templates"]}
    assert "Synthetic 2D SNR Sweep Gray" in titles
    assert "Synthetic 3D SNR Sweep Gray" in titles
    assert "Synthetic 2D SNR Sweep RGB" in titles
    assert "Synthetic 3D SNR Sweep RGB" in titles
