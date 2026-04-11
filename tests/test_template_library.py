from __future__ import annotations

from napari_chat_assistant.agent.template_library import template_library_payload
from napari_chat_assistant.library.template_catalog import template_button_labels


def test_template_library_uses_sections_and_learning_branch():
    payload = template_library_payload()

    section_labels = [str(section.get("label", "")).strip() for section in payload["sections"]]
    assert "Prompt Templates" in section_labels
    assert "Code Templates" in section_labels
    assert "Learning" in section_labels
    titles = {str(record.get("title", "")).strip() for record in payload["templates"]}
    assert "Fit View" in titles
    assert "Axes Toggle" in titles
    assert "Scale Bar Toggle" in titles
    assert "Bounding Box Toggle" in titles
    assert "Layer Name Toggle" in titles
    assert "Show Only Image Layers" in titles
    assert "Conservative Binary Mask Workflow" in titles
    assert "Prepare Image Review Workflow" in titles
    assert "Undo Last Workflow" in titles
    assert "Labels Layers Visibility" in titles
    assert "Delete ROI Layers" in titles
    assert "Reset Selected Scale" in titles
    assert "Compare Two ROI Groups" in titles
    assert "Compare Two Image Groups" in titles
    assert "Apply Gaussian Blur" in titles
    assert "Extract ROI Values" in titles
    assert "Reply In Markdown" in titles
    assert "Microscopy Concepts Students Misuse" in titles
    assert "Cautious EM Interpretation" in titles
    assert "Fluorescence Signal Formation" in titles
    assert "Intensity As Measurement" in titles
    assert "Effect Size Vs Significance" in titles
    assert "Diffraction, PSF, And Resolution" in titles
    assert "Upgrade A Science Question" in titles
    assert "Hungarian Onboarding Prompt" in titles
    assert "Chinese Onboarding Prompt" in titles
    assert "Spanish Onboarding Prompt" in titles

    learning_section = next(section for section in payload["sections"] if section["label"] == "Learning")
    learning_categories = [str(category.get("name", "")).strip() for category in learning_section["categories"]]
    assert "Academic Prompting" in learning_categories
    assert "Biophotonics" in learning_categories
    assert "Image Formation" in learning_categories
    assert "Language Support" in learning_categories
    assert "Quantitative Imaging" in learning_categories
    assert "Statistics" in learning_categories

    prompt_section = next(section for section in payload["sections"] if section["label"] == "Prompt Templates")
    prompt_categories = [str(category.get("name", "")).strip() for category in prompt_section["categories"]]
    assert "Quick Controls" in prompt_categories
    assert "Workflow" in prompt_categories


def test_template_library_moves_widget_launchers_out_of_templates():
    payload = template_library_payload()

    titles = {str(record.get("title", "")).strip() for record in payload["templates"]}
    assert "ROI Intensity Analysis" not in titles
    assert "Line Profile Gaussian Fit" not in titles
    assert "Open Group Comparison Widget" not in titles


def test_template_library_includes_synthetic_snr_data_templates():
    payload = template_library_payload()

    titles = {str(record.get("title", "")).strip() for record in payload["templates"]}
    assert "Synthetic 2D SNR Sweep Gray" in titles
    assert "Synthetic 3D SNR Sweep Gray" in titles
    assert "Synthetic 2D SNR Sweep RGB" in titles
    assert "Synthetic 3D SNR Sweep RGB" in titles


def test_learning_and_code_templates_expose_different_button_labels():
    payload = template_library_payload()
    learning_record = next(record for record in payload["templates"] if record["title"] == "Cautious EM Interpretation")
    code_record = next(record for record in payload["templates"] if record["title"] == "Synthetic Blob Image")

    assert template_button_labels(learning_record) == ("Load Prompt", "Ask Now")
    assert template_button_labels(code_record) == ("Load Code", "Run Code")
