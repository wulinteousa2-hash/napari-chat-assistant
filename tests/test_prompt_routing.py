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


def test_route_local_workflow_prompt_handles_no_image_demo_onboarding():
    route = route_local_workflow_prompt(
        "I dont have any image with me. Not sure how can I test your ability.",
        None,
    )

    assert route is not None
    assert route["action"] == "reply"
    assert "Templates" in route["message"]
    assert "Data" in route["message"]
    assert "synthetic image" in route["message"]


def test_route_local_workflow_prompt_handles_no_image_who_are_you_demo_case():
    route = route_local_workflow_prompt(
        "Who are you? I don't have an image and just want to test the plugin.",
        {},
    )

    assert route is not None
    assert route["action"] == "reply"
    assert "do not need a real microscopy image" in route["message"]


def test_route_local_workflow_prompt_does_not_route_plain_identity_question():
    route = route_local_workflow_prompt(
        "Howdy, who are you?",
        {},
    )

    assert route is None


def test_route_local_workflow_prompt_routes_direct_synthetic_variant_request():
    route = route_local_workflow_prompt(
        "create 2D grayscale",
        {},
    )

    assert route is not None
    assert route["action"] == "tool"
    assert route["tool"] == "create_synthetic_demo_image"
    assert route["arguments"]["variant"] == "2d_gray"


def test_route_local_workflow_prompt_routes_image_review_setup_to_tool_sequence():
    route = route_local_workflow_prompt(
        "Prepare this viewer for image review: hide labels and points layers, show only image layers, "
        "turn axes labels on, hide the scale bar, tile the images in grid view, and fit the view.",
        {"layer_type": "image", "layer_name": "image_a"},
    )

    assert route is not None
    assert route["action"] == "tool_sequence"
    tools = [step["tool"] for step in route["steps"]]
    assert tools == [
        "hide_layers_by_type",
        "hide_layers_by_type",
        "show_only_layer_type",
        "set_axes_labels",
        "set_scale_bar_visible",
        "show_image_layers_in_grid",
        "fit_view",
    ]
    assert route["steps"][2]["on_error"] == "stop"


def test_route_local_workflow_prompt_routes_numbered_quick_controls_to_sequence():
    route = route_local_workflow_prompt(
        """
        1. Show only image layers.
        2. Turn axes on.
        3. Turn axes labels on.
        4. Turn scale bar on.
        5. Turn selected layer bounding box on.
        6. Fit the view.
        """,
        {"layer_type": "image", "layer_name": "image_a"},
    )

    assert route is not None
    assert route["action"] == "tool_sequence"
    assert [step["tool"] for step in route["steps"]] == [
        "show_only_layer_type",
        "set_axes_visible",
        "set_axes_labels",
        "set_scale_bar_visible",
        "set_selected_layer_bounding_box_visible",
        "fit_view",
    ]
    assert route["steps"][0]["arguments"] == {"layer_type": "image"}
    assert route["steps"][3]["arguments"] == {"visible": True}
    assert route["steps"][4]["arguments"] == {"visible": True}


def test_route_local_workflow_prompt_routes_compact_numbered_quick_controls_to_sequence():
    route = route_local_workflow_prompt(
        """
        1.Fit the current visible layers in view.
        2.Turn the viewer axes on.
        3.Turn the scale bar on.
        4.Turn the selected layer bounding box on.
        5.Turn the selected layer name overlay on.
        """,
        {"layer_type": "image", "layer_name": "image_a"},
    )

    assert route is not None
    assert route["action"] == "tool_sequence"
    assert [step["tool"] for step in route["steps"]] == [
        "fit_view",
        "set_axes_visible",
        "set_scale_bar_visible",
        "set_selected_layer_bounding_box_visible",
        "set_selected_layer_name_overlay_visible",
    ]


def test_route_local_workflow_prompt_routes_undo_last_workflow():
    route = route_local_workflow_prompt(
        "undo last workflow",
        {"layer_type": "image", "layer_name": "image_a"},
    )

    assert route is not None
    assert route["action"] == "restore_tool_sequence"
    assert "Restoring viewer controls" in route["message"]


def test_route_local_workflow_prompt_replies_when_synthetic_variant_is_unspecified():
    route = route_local_workflow_prompt(
        "can you generate a synthetic image?",
        {},
    )

    assert route is not None
    assert route["action"] == "reply"
    assert "2D grayscale" in route["message"]


def test_route_local_workflow_prompt_handles_getting_started_without_layers():
    route = route_local_workflow_prompt(
        "how do i start?",
        {},
    )

    assert route is not None
    assert route["action"] == "reply"
    assert "load or generate a test image" in route["message"]
    assert "Templates" in route["message"]


def test_route_local_workflow_prompt_handles_getting_started_with_selected_layer():
    route = route_local_workflow_prompt(
        "how do i start?",
        {"layer_type": "image", "layer_name": "image_a"},
    )

    assert route is not None
    assert route["action"] == "reply"
    assert "inspect the selected layer" in route["message"]


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
