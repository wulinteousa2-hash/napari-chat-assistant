from __future__ import annotations

import napari

from .context import layer_context_json
from .tool_registry import TOOL_REGISTRY
from .tool_types import PreparedJob, ToolContext, ToolResult
from .tools_builtin import builtin_tools


SAFE_SEQUENCE_TOOLS = {
    "fit_view",
    "hide_image_grid_view",
    "hide_layers",
    "hide_layers_by_type",
    "list_layers",
    "set_axes_arrows",
    "set_axes_colored",
    "set_axes_dashed",
    "set_axes_labels",
    "set_axes_visible",
    "set_layer_tooltips_visible",
    "set_scale_bar_box",
    "set_scale_bar_colored",
    "set_scale_bar_ticks",
    "set_scale_bar_visible",
    "set_selected_layer_bounding_box_visible",
    "set_selected_layer_name_overlay_visible",
    "show_all_layers",
    "show_image_layers_in_grid",
    "show_layers",
    "show_layers_by_type",
    "show_only_layer_type",
    "show_only_layers",
    "toggle_2d_3d_camera",
    "zoom_in_view",
    "zoom_out_view",
}


def _ensure_builtin_registry() -> None:
    for tool in builtin_tools():
        if TOOL_REGISTRY.get(tool.spec.name) is None:
            TOOL_REGISTRY.register(tool)


def _tool_context(viewer: napari.Viewer) -> ToolContext:
    payload = layer_context_json(viewer)
    return ToolContext(
        viewer=viewer,
        layer_context=payload,
        selected_layer_profile=payload.get("selected_layer_profile"),
    )


def prepare_tool_job(viewer: napari.Viewer, tool_name: str, arguments: dict) -> dict:
    _ensure_builtin_registry()
    registry_tool = TOOL_REGISTRY.get(tool_name)
    if registry_tool is None:
        return {"mode": "immediate", "message": f"Unsupported tool: {tool_name}"}
    prepared = registry_tool.prepare(_tool_context(viewer), arguments or {})
    if isinstance(prepared, str):
        return {"mode": "immediate", "message": prepared}
    if prepared.mode == "immediate":
        result = registry_tool.execute(prepared)
        message = registry_tool.apply(_tool_context(viewer), result)
        return {"mode": "immediate", "message": message}
    return {"mode": prepared.mode, "job": prepared.to_dict()}


def capture_viewer_control_snapshot(viewer: napari.Viewer) -> dict:
    layers: dict[str, dict] = {}
    for layer in getattr(viewer, "layers", []) or []:
        name = str(getattr(layer, "name", "")).strip()
        if not name:
            continue
        layer_record = {"visible": bool(getattr(layer, "visible", True))}
        bounding_box = getattr(layer, "bounding_box", None)
        if bounding_box is not None and hasattr(bounding_box, "visible"):
            layer_record["bounding_box_visible"] = bool(getattr(bounding_box, "visible", False))
        name_overlay = getattr(layer, "name_overlay", None)
        if name_overlay is not None and hasattr(name_overlay, "visible"):
            layer_record["name_overlay_visible"] = bool(getattr(name_overlay, "visible", False))
        layers[name] = layer_record

    viewer_state: dict[str, object] = {}
    grid = getattr(viewer, "grid", None)
    if grid is not None:
        viewer_state["grid_enabled"] = bool(getattr(grid, "enabled", False))
        viewer_state["grid_spacing"] = float(getattr(grid, "spacing", 0.0) or 0.0)
        try:
            viewer_state["grid_shape"] = list(getattr(grid, "shape", ()) or ())
        except Exception:
            viewer_state["grid_shape"] = []
    axes = getattr(viewer, "axes", None)
    if axes is not None:
        for attr in ("visible", "colored", "labels", "dashed", "arrows"):
            if hasattr(axes, attr):
                viewer_state[f"axes_{attr}"] = bool(getattr(axes, attr))
    scale_bar = getattr(viewer, "scale_bar", None)
    if scale_bar is not None:
        for attr in ("visible", "box", "colored", "ticks"):
            if hasattr(scale_bar, attr):
                viewer_state[f"scale_bar_{attr}"] = bool(getattr(scale_bar, attr))
    tooltip = getattr(viewer, "tooltip", None)
    if tooltip is not None and hasattr(tooltip, "visible"):
        viewer_state["tooltip_visible"] = bool(getattr(tooltip, "visible"))
    dims = getattr(viewer, "dims", None)
    if dims is not None and hasattr(dims, "ndisplay"):
        viewer_state["ndisplay"] = int(getattr(dims, "ndisplay", 2) or 2)
    camera = getattr(viewer, "camera", None)
    if camera is not None:
        if hasattr(camera, "zoom"):
            viewer_state["camera_zoom"] = float(getattr(camera, "zoom", 1.0) or 1.0)
        if hasattr(camera, "center"):
            try:
                viewer_state["camera_center"] = list(getattr(camera, "center") or [])
            except Exception:
                pass

    return {"version": 1, "layers": layers, "viewer": viewer_state}


def restore_viewer_control_snapshot(viewer: napari.Viewer, snapshot: dict) -> str:
    payload = snapshot if isinstance(snapshot, dict) else {}
    layer_records = payload.get("layers", {})
    restored_layers = 0
    missing_layers = 0
    if isinstance(layer_records, dict):
        for name, record in layer_records.items():
            layer = None
            try:
                layer = viewer.layers[str(name)]
            except Exception:
                layer = None
            if layer is None:
                missing_layers += 1
                continue
            layer_payload = record if isinstance(record, dict) else {}
            if "visible" in layer_payload:
                layer.visible = bool(layer_payload["visible"])
            bounding_box = getattr(layer, "bounding_box", None)
            if bounding_box is not None and "bounding_box_visible" in layer_payload:
                bounding_box.visible = bool(layer_payload["bounding_box_visible"])
            name_overlay = getattr(layer, "name_overlay", None)
            if name_overlay is not None and "name_overlay_visible" in layer_payload:
                name_overlay.visible = bool(layer_payload["name_overlay_visible"])
            restored_layers += 1

    viewer_state = payload.get("viewer", {})
    if not isinstance(viewer_state, dict):
        viewer_state = {}
    grid = getattr(viewer, "grid", None)
    if grid is not None:
        if "grid_enabled" in viewer_state:
            grid.enabled = bool(viewer_state["grid_enabled"])
        if "grid_spacing" in viewer_state:
            grid.spacing = float(viewer_state["grid_spacing"])
        if "grid_shape" in viewer_state:
            try:
                grid.shape = tuple(viewer_state["grid_shape"])
            except Exception:
                pass
    axes = getattr(viewer, "axes", None)
    if axes is not None:
        for attr in ("visible", "colored", "labels", "dashed", "arrows"):
            key = f"axes_{attr}"
            if key in viewer_state and hasattr(axes, attr):
                setattr(axes, attr, bool(viewer_state[key]))
    scale_bar = getattr(viewer, "scale_bar", None)
    if scale_bar is not None:
        for attr in ("visible", "box", "colored", "ticks"):
            key = f"scale_bar_{attr}"
            if key in viewer_state and hasattr(scale_bar, attr):
                setattr(scale_bar, attr, bool(viewer_state[key]))
    tooltip = getattr(viewer, "tooltip", None)
    if tooltip is not None and "tooltip_visible" in viewer_state:
        tooltip.visible = bool(viewer_state["tooltip_visible"])
    dims = getattr(viewer, "dims", None)
    if dims is not None and "ndisplay" in viewer_state:
        dims.ndisplay = int(viewer_state["ndisplay"])
    camera = getattr(viewer, "camera", None)
    if camera is not None:
        if "camera_zoom" in viewer_state and hasattr(camera, "zoom"):
            camera.zoom = float(viewer_state["camera_zoom"])
        if "camera_center" in viewer_state and hasattr(camera, "center"):
            try:
                camera.center = tuple(viewer_state["camera_center"])
            except Exception:
                pass

    missing_text = f" Missing {missing_layers} layer(s) that no longer exist." if missing_layers else ""
    return f"Restored viewer controls for {restored_layers} layer(s).{missing_text}"


def run_tool_sequence(viewer: napari.Viewer, steps: list[dict], *, capture_undo: bool = True) -> dict:
    messages: list[str] = []
    completed = 0
    skipped = 0
    stopped = False
    undo_snapshot = capture_viewer_control_snapshot(viewer) if capture_undo else None

    for index, raw_step in enumerate(steps or [], start=1):
        step = raw_step if isinstance(raw_step, dict) else {}
        tool_name = str(step.get("tool", "")).strip()
        arguments = step.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        on_error = str(step.get("on_error", "skip") or "skip").strip().lower()
        if on_error not in {"skip", "stop"}:
            on_error = "skip"

        if not tool_name:
            skipped += 1
            messages.append(f"{index}. Skipped empty workflow step.")
            if on_error == "stop":
                stopped = True
                break
            continue
        if tool_name not in SAFE_SEQUENCE_TOOLS:
            skipped += 1
            messages.append(f"{index}. Skipped unsafe or unsupported workflow step [{tool_name}].")
            if on_error == "stop":
                stopped = True
                break
            continue

        prepared = prepare_tool_job(viewer, tool_name, arguments)
        if prepared.get("mode") != "immediate":
            skipped += 1
            messages.append(f"{index}. Skipped [{tool_name}] because sequence execution only supports immediate tools for now.")
            if on_error == "stop":
                stopped = True
                break
            continue

        message = str(prepared.get("message", "")).strip() or f"No result from [{tool_name}]."
        if _tool_sequence_message_indicates_failure(message):
            skipped += 1
            messages.append(f"{index}. Skipped [{tool_name}]: {message}")
            if on_error == "stop":
                stopped = True
                break
            continue

        completed += 1
        messages.append(f"{index}. {message}")

    summary = f"Completed {completed} of {len(steps or [])} workflow step(s)."
    if skipped:
        summary += f" Skipped {skipped}."
    if stopped:
        summary += " Stopped early."
    return {
        "mode": "immediate",
        "completed": completed,
        "skipped": skipped,
        "stopped": stopped,
        "message": "\n".join([summary, *messages]).strip(),
        "undo_snapshot": undo_snapshot,
    }


def _tool_sequence_message_indicates_failure(message: str) -> bool:
    text = str(message or "").strip().lower()
    failure_prefixes = (
        "could not ",
        "need ",
        "no ",
        "provide ",
        "unsupported ",
    )
    unavailable_fragments = (
        " is unavailable",
        " was unavailable",
        " were unavailable",
        " failed",
    )
    return text.startswith(failure_prefixes) or any(fragment in text for fragment in unavailable_fragments)


def run_tool_job(job: dict) -> dict:
    _ensure_builtin_registry()
    registry_tool = TOOL_REGISTRY.get(job.get("tool_name", ""))
    if registry_tool is None:
        raise ValueError(f"Unsupported tool job: {job.get('tool_name', '') or job.get('kind', '')}")
    prepared_job = PreparedJob.from_dict(job)
    return registry_tool.execute(prepared_job).to_dict()


def apply_tool_job_result(viewer: napari.Viewer, result: dict) -> str:
    _ensure_builtin_registry()
    registry_tool = TOOL_REGISTRY.get(result.get("tool_name", ""))
    if registry_tool is None:
        return f"Unsupported tool result: {result.get('kind', '')}"
    return registry_tool.apply(_tool_context(viewer), ToolResult.from_dict(result))
