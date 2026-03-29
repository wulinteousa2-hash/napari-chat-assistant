from __future__ import annotations

import napari
import numpy as np

from .image_ops import (
    close_binary_mask,
    dilate_binary_mask,
    erode_binary_mask,
    fill_holes,
    keep_largest_component,
    open_binary_mask,
    remove_small_components,
)


TOOL_OPS = {
    "dilate": lambda data, args: dilate_binary_mask(data, radius=int(args.get("radius", 1))),
    "erode": lambda data, args: erode_binary_mask(data, radius=int(args.get("radius", 1))),
    "open": lambda data, args: open_binary_mask(data, radius=int(args.get("radius", 1))),
    "close": lambda data, args: close_binary_mask(data, radius=int(args.get("radius", 1))),
    "fill_holes": lambda data, args: fill_holes(data),
    "remove_small": lambda data, args: remove_small_components(data, min_size=int(args.get("min_size", 64))),
    "keep_largest": lambda data, args: keep_largest_component(data),
}

ASSISTANT_TOOL_NAMES = {
    "list_layers",
    "inspect_selected_layer",
    "inspect_layer",
    "open_nd2_converter",
    "open_spectral_viewer",
    "open_spectral_analysis",
    "apply_clahe",
    "apply_clahe_batch",
    "preview_threshold",
    "apply_threshold",
    "preview_threshold_batch",
    "apply_threshold_batch",
    "measure_mask",
    "measure_mask_batch",
    "run_mask_op",
    "run_mask_op_batch",
}


def normalize_polarity(value) -> str:
    pol = str(value or "auto").strip().lower()
    if pol in {"bright", "bright objects", "foreground bright"}:
        return "bright"
    if pol in {"dim", "dark", "dim objects", "foreground dim"}:
        return "dim"
    return "auto"


def normalize_int(value, default: int, minimum: int = 1, maximum: int = 1_000_000) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(int(minimum), min(int(maximum), out))


def normalize_float(value, default: float, minimum: float = 0.0, maximum: float = 1_000_000.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    return max(float(minimum), min(float(maximum), out))


def normalize_kernel_size(value, ndim: int | None = None):
    if isinstance(value, (list, tuple)):
        values = [normalize_int(v, 32, minimum=1, maximum=4096) for v in value]
    else:
        values = [normalize_int(value, 32, minimum=1, maximum=4096)]
    if ndim is None:
        return values[0] if len(values) == 1 else values
    if len(values) == 1:
        return tuple([values[0]] * ndim)
    if len(values) < ndim:
        return tuple((values + [values[-1]] * ndim)[:ndim])
    return tuple(values[:ndim])


def next_snapshot_name(viewer: napari.Viewer, base_name: str) -> str:
    index = 1
    while True:
        name = f"{base_name}_assistant_snapshot_{index:02d}"
        if name not in viewer.layers:
            return name
        index += 1


def save_mask_snapshot(viewer: napari.Viewer, layer: napari.layers.Labels) -> str:
    snapshot_name = next_snapshot_name(viewer, layer.name)
    viewer.add_labels(
        (np.asarray(layer.data) > 0).astype(np.uint8).copy(),
        name=snapshot_name,
        scale=layer.scale,
        translate=layer.translate,
    )
    return snapshot_name


def next_output_name(viewer: napari.Viewer, base_name: str) -> str:
    if base_name not in viewer.layers:
        return base_name
    suffix = 1
    while f"{base_name}_{suffix:02d}" in viewer.layers:
        suffix += 1
    return f"{base_name}_{suffix:02d}"


def assistant_system_prompt() -> str:
    return (
        "You are a specialized local napari image-analysis assistant, not a general chatbot. "
        "Focus on practical image-analysis actions inside napari. "
        "Prefer concrete tool use over broad discussion. "
        "If the request is ambiguous about which layer to use, ask a short clarification question. "
        "You must respond with exactly one JSON object and no extra text. "
        "Allowed response forms are: "
        '{"action":"reply","message":"..."} '
        'or {"action":"tool","tool":"<tool_name>","arguments":{...},"message":"..."} '
        'or {"action":"code","message":"...","code":"<python code>"}.\n'
        "Allowed tools:\n"
        "- list_layers: {}\n"
        '- inspect_selected_layer: {}\n'
        '- inspect_layer: {"layer_name": string}\n'
        '- open_nd2_converter: {}\n'
        '- open_spectral_viewer: {}\n'
        '- open_spectral_analysis: {}\n'
        '- apply_clahe: {"layer_name": optional string, "kernel_size": optional int or list, "clip_limit": optional float, "nbins": optional int}\n'
        '- apply_clahe_batch: {"kernel_size": optional int or list, "clip_limit": optional float, "nbins": optional int}\n'
        '- preview_threshold: {"layer_name": optional string, "polarity": "auto|bright|dim"}\n'
        '- apply_threshold: {"layer_name": optional string, "polarity": "auto|bright|dim"}\n'
        '- preview_threshold_batch: {"polarity": "auto|bright|dim"}\n'
        '- apply_threshold_batch: {"polarity": "auto|bright|dim"}\n'
        '- measure_mask: {"layer_name": optional string}\n'
        '- measure_mask_batch: {}\n'
        '- run_mask_op: {"layer_name": optional string, "op": "dilate|erode|open|close|fill_holes|remove_small|keep_largest", "radius": optional int, "min_size": optional int}\n'
        '- run_mask_op_batch: {"op": "dilate|erode|open|close|fill_holes|remove_small|keep_largest", "radius": optional int, "min_size": optional int}\n'
        "- If the user is asking what exists or what is selected, use list_layers.\n"
        "- If the user asks about the selected layer's kind, properties, dimensions, dtype, or statistics, use inspect_selected_layer.\n"
        "- If the user asks about a specific named layer's kind, properties, dimensions, dtype, or statistics, use inspect_layer.\n"
        "- If the user asks to convert ND2, Nikon microscopy files, Nikon proprietary files, or Nikon spectral files to OME-Zarr, use open_nd2_converter.\n"
        "- If the user asks to open the ND2 converter, ND2 exporter, batch exporter, or OME-Zarr converter, use open_nd2_converter.\n"
        "- If the user asks for a spectral viewer, spectral rendering, pseudocolor viewer, or visible/truecolor spectral display, use open_spectral_viewer.\n"
        "- If the user asks for PCA, spectral ratio analysis, Welch t-test, ANOVA, or spectral statistics, use open_spectral_analysis.\n"
        "- The viewer_context includes deterministic per-layer dataset profiles with semantic_type, confidence, axes_detected, and recommendation classes. Use those fields instead of guessing from the prompt.\n"
        "- The payload may include session_memory with approved prior decisions. Use it only as secondary context. If session_memory conflicts with the current selected_layer_profile or viewer_context, follow the current viewer data.\n"
        "- If the user asks for CLAHE, adaptive histogram equalization, local contrast enhancement, or EM contrast enhancement, use apply_clahe or apply_clahe_batch.\n"
        "- CLAHE parameters are kernel_size, clip_limit, and nbins.\n"
        "- If the user asks for all images, every image, batch, or multiple open images, prefer the batch tools.\n"
        "- If the user asks for thresholding, segmentation by threshold, binary mask creation, converting an image into labels, converting an image into a mask, Otsu-style thresholding, or creating a labels layer from an image, use preview_threshold or apply_threshold instead of action=code.\n"
        "- For thresholding or mask creation, use preview_threshold first unless they explicitly ask to apply.\n"
        "- If the user explicitly wants the threshold result added as a labels layer, use apply_threshold. The built-in threshold tools already create labels output layers and are preferred over generated code.\n"
        "- If a profile indicates label_mask or probability_map, avoid treating that layer as a generic intensity image unless the user explicitly asks.\n"
        "- If a profile indicates rgb, avoid grayscale-only operations such as CLAHE unless the user asks for a conversion workflow.\n"
        "- If the user asks for measurement, area, volume, or object count on a mask, use measure_mask.\n"
        "- Use action=code only when the request needs custom napari/python logic that is not covered by the built-in tools.\n"
        "- Do not generate code for tasks already covered by built-in tools, especially threshold preview/apply, mask measurement, mask morphology, or CLAHE.\n"
        "- Generated code must assume it will be shown to the user for approval before execution.\n"
        "- Generated code must be pure Python only, with no Markdown fences, no JSON fragments, and no explanatory text outside Python comments.\n"
        "- Generated code must use the provided existing `viewer` object.\n"
        "- Generated code must not create a new napari viewer.\n"
        "- Generated code must not close the viewer.\n"
        "- Generated code must not import or reference `ViewerViewerContext`.\n"
        "- Prefer no imports unless absolutely required. `napari`, `np`, `numpy`, `viewer`, and `selected_layer` are already available.\n"
        "- The selected layer is available as `viewer.layers.selection.active`.\n"
        "- Layer array shape should usually be read from `layer.data.shape`.\n"
        "- Generated code may use the provided helper variable `selected_layer`.\n"
        "- Generated code must check that `selected_layer` is not None before using it.\n"
        "- Do not hardcode a layer lookup by name unless you first verify the layer exists.\n"
        "- When generated code creates napari layers, prefer minimal core napari API calls with minimal keyword arguments. Do not guess optional styling kwargs for Labels or other layer types.\n"
        "- Do not invent napari imports or viewer methods. If you are not sure an API exists, do not use it.\n"
        "- Be concise and operational.\n"
        "- For EM or volume data, mention whether the operation is acting on the current 2D/3D labels data as-is.\n"
        "- Prefer explicit layer_name values when the user names a specific layer.\n"
        "- Mask operations automatically create a snapshot before editing.\n"
        "- If no tool is needed, use action=reply.\n"
    )
