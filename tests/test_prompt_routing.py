from __future__ import annotations

from napari_chat_assistant.agent.prompt_routing import (
    extract_layer_options_from_clarification,
    infer_tool_clarification_request,
    is_affirmative_followup,
    looks_like_multistep_segmentation_workflow,
    resolve_followup_choice_index,
    resolve_followup_layer_reference,
    route_local_workflow_prompt,
)


def test_looks_like_multistep_segmentation_workflow_detects_numbered_steps():
    text = (
        "1. add light Gaussian smoothing before threshold\n"
        "2. use morphological closing on the dark myelin mask\n"
        "3. remove tiny regions\n"
        "4. optionally clear border-touching objects\n"
        "5. filter labeled regions by area/shape"
    )
    assert looks_like_multistep_segmentation_workflow(text) is True


def test_route_local_workflow_prompt_routes_compound_axon_prompt_to_tool():
    text = (
        "Extract candidate axon interiors from dark myelin rings.\n"
        "1. add light Gaussian smoothing before threshold\n"
        "2. use morphological closing on the dark myelin mask\n"
        "3. remove tiny regions\n"
        "4. optionally clear border-touching objects\n"
        "5. filter labeled regions by area/shape"
    )
    route = route_local_workflow_prompt(text, {"layer_type": "image", "layer_name": "em_a"})
    assert route is not None
    assert route["action"] == "tool"
    assert route["tool"] == "extract_axon_interiors"
    assert route["arguments"]["image_layer"] == "em_a"
    assert route["arguments"]["sigma"] == 1.0
    assert route["arguments"]["closing_radius"] == 2
    assert route["arguments"]["min_area"] == 64
    assert route["arguments"]["clear_border"] is True


def test_route_local_workflow_prompt_skips_non_image_selected_layer():
    route = route_local_workflow_prompt(
        "Extract candidate axon interiors from dark myelin rings",
        {"layer_type": "labels", "layer_name": "mask_a"},
    )
    assert route is None


def test_infer_tool_clarification_request_detects_gaussian_layer_question():
    payload = infer_tool_clarification_request(
        "Which image layer would you like to apply Gaussian denoising to "
        "(e.g., em_2d_snr_low, em_2d_snr_mid, or em_2d_snr_high)?"
    )

    assert payload is not None
    assert payload["tool"] == "gaussian_denoise"
    assert payload["arguments"]["sigma"] == 1.0
    assert payload["layer_argument"] == "layer_name"
    assert payload["layer_scope"] == "image"
    assert payload["options"] == ["em_2d_snr_low", "em_2d_snr_mid", "em_2d_snr_high"]


def test_extract_layer_options_from_clarification_reads_named_examples():
    options = extract_layer_options_from_clarification(
        "Which image layer would you like to apply Gaussian denoising to "
        "(e.g., em_2d_snr_low, em_2d_snr_mid, or em_2d_snr_high)?"
    )

    assert options == ["em_2d_snr_low", "em_2d_snr_mid", "em_2d_snr_high"]


def test_resolve_followup_layer_reference_uses_selected_layer_aliases():
    matches = resolve_followup_layer_reference(
        "current selected one",
        selected_layer_name="em_2d_snr_mid",
        available_layer_names=("em_2d_snr_low", "em_2d_snr_mid", "em_2d_snr_high"),
    )

    assert matches == ["em_2d_snr_mid"]


def test_resolve_followup_layer_reference_matches_named_layer_mentions():
    matches = resolve_followup_layer_reference(
        "use em_2d_snr_high for this",
        selected_layer_name="em_2d_snr_mid",
        available_layer_names=("em_2d_snr_low", "em_2d_snr_mid", "em_2d_snr_high"),
    )

    assert matches == ["em_2d_snr_high"]


def test_is_affirmative_followup_detects_simple_confirmation():
    assert is_affirmative_followup("yes. i want to perform gaussian denoising") is True


def test_is_affirmative_followup_detects_short_ok_variants():
    assert is_affirmative_followup("ok") is True
    assert is_affirmative_followup("okay") is True
    assert is_affirmative_followup("oky") is True
    assert is_affirmative_followup("go") is True


def test_resolve_followup_choice_index_supports_numbers():
    options = ["em_2d_snr_low", "em_2d_snr_mid", "em_2d_snr_high"]

    assert resolve_followup_choice_index("2", options) == "em_2d_snr_mid"
    assert resolve_followup_choice_index("3", options) == "em_2d_snr_high"


def test_resolve_followup_choice_index_supports_ordinals():
    options = ["em_2d_snr_low", "em_2d_snr_mid", "em_2d_snr_high"]

    assert resolve_followup_choice_index("first", options) == "em_2d_snr_low"
    assert resolve_followup_choice_index("third one", options) == "em_2d_snr_high"
