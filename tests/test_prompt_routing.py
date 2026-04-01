from __future__ import annotations

from napari_chat_assistant.agent.prompt_routing import (
    looks_like_multistep_segmentation_workflow,
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
