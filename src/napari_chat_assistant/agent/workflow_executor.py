from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import napari
import numpy as np
from scipy import ndimage as ndi

from .context import find_image_layer, find_labels_layer
from .dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job
from .image_ops import fill_holes, intensity_statistics, mask_statistics, median_binary_mask
from .recent_action_state import refine_threshold_action


MAX_REFINEMENT_CYCLES = 3


@dataclass
class WorkflowExecutionState:
    target_layer: str
    aliases: dict[str, str] = field(default_factory=dict)
    threshold_state: dict[str, Any] = field(default_factory=dict)
    executed_steps: list[dict[str, Any]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    cycle_count: int = 0

    def add_line(self, message: str) -> None:
        self.lines.append(f"{len(self.lines) + 1}. {message}")

    def add_step(self, **payload: Any) -> None:
        self.executed_steps.append(dict(payload))


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _run_registered_tool(viewer: napari.Viewer, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    prepared = prepare_tool_job(viewer, tool_name, arguments if isinstance(arguments, dict) else {})
    if prepared.get("mode") == "immediate" and "job" not in prepared:
        message = str(prepared.get("message", "")).strip()
        return {
            "ok": not _looks_like_error_message(message),
            "tool": tool_name,
            "message": message,
            "result": {},
        }
    if "job" not in prepared:
        message = str(prepared.get("message", "")).strip() or f"Tool [{tool_name}] could not be prepared."
        return {"ok": False, "tool": tool_name, "message": message, "result": {}}
    result = run_tool_job(prepared["job"])
    message = apply_tool_job_result(viewer, result)
    return {"ok": True, "tool": tool_name, "message": str(message or "").strip(), "result": dict(result)}


def _looks_like_error_message(message: str) -> bool:
    source = _normalize_text(message)
    if not source:
        return True
    return source.startswith(("no ", "unsupported ", "could not ", "need ", "provide "))


def _resolve_argument_placeholders(arguments: dict[str, Any], aliases: dict[str, str], selected_layer: str) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if isinstance(value, str) and value.startswith("$"):
            token = value.strip().lower()
            if token == "$working_image_or_selected":
                resolved[key] = aliases.get("working_image", "") or selected_layer
            elif token == "$base_mask":
                resolved[key] = aliases.get("base_mask", "")
            elif token == "$latest_mask":
                resolved[key] = aliases.get("latest_mask", "") or aliases.get("base_mask", "")
            else:
                resolved[key] = aliases.get(value[1:], "")
        else:
            resolved[key] = value
    return resolved


def _maybe_record_output_alias(step: dict[str, Any], tool_result: dict[str, Any], aliases: dict[str, str]) -> None:
    output_alias = str(step.get("output_alias", "")).strip()
    payload = tool_result if isinstance(tool_result, dict) else {}
    tool_name = str(payload.get("tool_name") or step.get("tool") or "").strip()
    output_name = ""
    if tool_name == "preview_threshold":
        output_name = "__assistant_threshold_preview__"
    else:
        output_name = str(payload.get("output_name", "")).strip()
    if output_alias and output_name:
        aliases[output_alias] = output_name
    if output_name and tool_name in {"apply_threshold", "remove_small_objects", "fill_mask_holes", "run_mask_op"}:
        aliases["latest_mask"] = output_name
    if tool_name == "apply_threshold" and output_name:
        aliases["base_mask"] = output_name
        aliases["latest_mask"] = output_name
    if tool_name == "gaussian_denoise" and output_name:
        aliases["working_image"] = output_name


def _inspect_image(viewer: napari.Viewer, layer_name: str) -> dict[str, Any]:
    image_layer = find_image_layer(viewer, layer_name)
    if image_layer is None:
        return {
            "message": "No valid image layer was available for inspection.",
            "noise_ratio": 0.0,
            "denoise_needed": False,
            "polarity": "auto",
        }
    data = np.asarray(image_layer.data, dtype=np.float32)
    stats = intensity_statistics(data)
    q05, q25, q50, q75, q95 = [float(x) for x in np.percentile(data[np.isfinite(data)], [5, 25, 50, 75, 95])]
    residual = data - ndi.gaussian_filter(data, sigma=1.0)
    residual_std = float(np.std(residual[np.isfinite(residual)])) if np.isfinite(residual).any() else 0.0
    total_std = max(float(stats["std"]), 1e-8)
    noise_ratio = residual_std / total_std
    upper_span = q95 - q50
    lower_span = q50 - q05
    polarity = "bright" if upper_span >= lower_span else "dim"
    denoise_needed = noise_ratio >= 0.55
    if noise_ratio >= 0.7:
        noise_text = "noticeable fine-grained noise"
    elif noise_ratio >= 0.45:
        noise_text = "moderate fine-grained noise"
    else:
        noise_text = "relatively low fine-grained noise"
    background_text = "broad low-frequency background variation" if abs(q75 - q25) > total_std * 0.8 else "fairly compact background range"
    signal_text = (
        "likely bright foreground signal on a darker background"
        if polarity == "bright"
        else "likely dim foreground signal on a brighter background"
    )
    return {
        "message": (
            f"Inspection on [{image_layer.name}]: {signal_text}, {background_text}, and {noise_text}. "
            f"Intensity stats mean={stats['mean']:.4g}, std={stats['std']:.4g}, median={stats['median']:.4g}, "
            f"p05={q05:.4g}, p95={q95:.4g}. "
            f"{'Light denoising is justified before thresholding.' if denoise_needed else 'Light denoising is not clearly needed, so the raw image should remain the primary reference.'}"
        ),
        "noise_ratio": noise_ratio,
        "denoise_needed": denoise_needed,
        "polarity": polarity,
        "image_min": float(stats["min"]),
        "image_max": float(stats["max"]),
    }


def _assess_mask_quality(viewer: napari.Viewer, layer_name: str) -> dict[str, Any]:
    labels_layer = find_labels_layer(viewer, layer_name)
    if labels_layer is None:
        return {
            "message": f"Mask layer [{layer_name}] is not available for review.",
            "quality": "unknown",
            "fg_fraction": 0.0,
            "object_count": 0,
            "largest_fraction": 0.0,
        }
    data = np.asarray(labels_layer.data)
    stats = mask_statistics(data)
    total_pixels = max(1, int(np.asarray(data).size))
    fg_fraction = float(stats["foreground_pixels"]) / float(total_pixels)
    largest_fraction = (
        float(stats["largest_object"]) / float(stats["foreground_pixels"])
        if int(stats["foreground_pixels"]) > 0
        else 0.0
    )
    if stats["foreground_pixels"] == 0 or fg_fraction < 0.003:
        quality = "too_strict"
        summary = "The mask is too strict and is missing most plausible foreground."
    elif fg_fraction > 0.55 or (fg_fraction > 0.3 and largest_fraction > 0.9):
        quality = "too_loose"
        summary = "The mask is too loose and likely includes too much background."
    else:
        quality = "acceptable"
        summary = "The mask is in a conservative range for further cleanup."
    return {
        "message": (
            f"Mask review on [{labels_layer.name}]: {summary} "
            f"foreground={stats['foreground_pixels']} px ({fg_fraction:.2%}), "
            f"objects={stats['object_count']}, largest={stats['largest_object']} px."
        ),
        "quality": quality,
        "fg_fraction": fg_fraction,
        "object_count": int(stats["object_count"]),
        "largest_fraction": largest_fraction,
        "stats": stats,
    }


def _should_remove_small_specks(mask_quality: dict[str, Any]) -> bool:
    fg_fraction = float(mask_quality.get("fg_fraction", 0.0) or 0.0)
    object_count = int(mask_quality.get("object_count", 0) or 0)
    largest_fraction = float(mask_quality.get("largest_fraction", 0.0) or 0.0)
    return object_count > 20 and fg_fraction < 0.2 and largest_fraction < 0.8


def _should_fill_small_holes(viewer: napari.Viewer, layer_name: str) -> bool:
    labels_layer = find_labels_layer(viewer, layer_name)
    if labels_layer is None:
        return False
    data = np.asarray(labels_layer.data)
    original_fg = int(mask_statistics(data)["foreground_pixels"])
    filled_fg = int(mask_statistics(fill_holes(data))["foreground_pixels"])
    if original_fg <= 0:
        return False
    fraction_gain = float(filled_fg - original_fg) / float(original_fg)
    return 0.01 <= fraction_gain <= 0.15


def _should_smooth_edges(viewer: napari.Viewer, layer_name: str) -> bool:
    labels_layer = find_labels_layer(viewer, layer_name)
    if labels_layer is None:
        return False
    data = np.asarray(labels_layer.data)
    mask = data > 0
    fg_pixels = int(mask.sum())
    if fg_pixels <= 0:
        return False
    boundary = mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(mask.ndim, 1))
    boundary_ratio = float(boundary.sum()) / float(fg_pixels)
    smoothed = median_binary_mask(data, radius=1)
    smoothed_fg = int(mask_statistics(smoothed)["foreground_pixels"])
    fraction_change = abs(smoothed_fg - fg_pixels) / float(fg_pixels)
    return boundary_ratio >= 0.35 and fraction_change <= 0.12


def _refine_threshold_arguments(mask_quality: dict[str, Any], threshold_state: dict[str, Any]) -> dict[str, Any]:
    quality = str(mask_quality.get("quality", "")).strip().lower()
    if quality not in {"too_loose", "too_strict"}:
        return {}
    direction = "stricter" if quality == "too_loose" else "looser"
    recent_like = {
        "tool_name": "apply_threshold",
        "action_kind": "threshold",
        "input_layers": [str(threshold_state.get("layer_name", "")).strip()],
        "parameters": {
            "threshold_value": threshold_state.get("threshold_value"),
            "foreground_mode": threshold_state.get("polarity"),
            "foreground_mode_resolved": threshold_state.get("polarity"),
            "image_min": threshold_state.get("image_min"),
            "image_max": threshold_state.get("image_max"),
        },
    }
    return refine_threshold_action(recent_like, direction)


def _judge_mask_quality(mask_quality: dict[str, Any]) -> dict[str, Any]:
    quality = str(mask_quality.get("quality", "")).strip().lower() or "unknown"
    issues: list[str] = []
    next_action = "stop"
    if quality == "too_loose":
        issues.append("too much background is included")
        next_action = "refine_threshold_stricter"
    elif quality == "too_strict":
        issues.append("too little plausible foreground is included")
        next_action = "refine_threshold_looser"
    if _should_remove_small_specks(mask_quality):
        issues.append("many small disconnected specks remain")
        if next_action == "stop":
            next_action = "remove_small_objects"
    return {"quality": quality, "issues": issues, "next_action": next_action}


def _attempt_cleanup_actions(viewer: napari.Viewer, state: WorkflowExecutionState, mask_quality: dict[str, Any]) -> dict[str, Any]:
    latest_mask = state.aliases.get("latest_mask", "") or state.aliases.get("base_mask", "")
    cleanup_applied = False
    if latest_mask and _should_remove_small_specks(mask_quality):
        remove_result = _run_registered_tool(viewer, "remove_small_objects", {"layer_name": latest_mask, "min_size": 64})
        state.add_step(id=f"remove_tiny_specks_cycle_{state.cycle_count}", kind="tool", **remove_result)
        if remove_result["ok"]:
            _maybe_record_output_alias({"tool": "remove_small_objects", "output_alias": "latest_mask"}, remove_result["result"], state.aliases)
            latest_mask = state.aliases.get("latest_mask", latest_mask)
            state.add_line(f"{remove_result['message']} This was justified because the mask still contained many small disconnected objects.")
            cleanup_applied = True

    if latest_mask and _should_fill_small_holes(viewer, latest_mask):
        fill_result = _run_registered_tool(viewer, "fill_mask_holes", {"layer_name": latest_mask})
        state.add_step(id=f"fill_small_holes_cycle_{state.cycle_count}", kind="tool", **fill_result)
        if fill_result["ok"]:
            _maybe_record_output_alias({"tool": "fill_mask_holes", "output_alias": "latest_mask"}, fill_result["result"], state.aliases)
            latest_mask = state.aliases.get("latest_mask", latest_mask)
            state.add_line(f"{fill_result['message']} This was justified because the mask still had small internal holes.")
            cleanup_applied = True

    if latest_mask and _should_smooth_edges(viewer, latest_mask):
        smooth_result = _run_registered_tool(viewer, "run_mask_op", {"layer_name": latest_mask, "op": "median", "radius": 1})
        state.add_step(id=f"smooth_jagged_edges_cycle_{state.cycle_count}", kind="tool", **smooth_result)
        if smooth_result["ok"]:
            _maybe_record_output_alias({"tool": "run_mask_op", "output_alias": "latest_mask"}, smooth_result["result"], state.aliases)
            state.add_line(f"{smooth_result['message']} This was limited to a radius-1 median cleanup to avoid over-merging nearby objects.")
            cleanup_applied = True

    latest_mask = state.aliases.get("latest_mask", latest_mask)
    reviewed = _assess_mask_quality(viewer, latest_mask)
    state.add_step(
        id=f"measure_mask_after_cleanup_cycle_{state.cycle_count}",
        kind="analysis",
        message=reviewed["message"],
        quality=reviewed["quality"],
    )
    if cleanup_applied:
        state.add_line(reviewed["message"])
    return reviewed


def _run_refinement_loop(viewer: napari.Viewer, state: WorkflowExecutionState, initial_mask_quality: dict[str, Any]) -> dict[str, Any]:
    mask_quality = dict(initial_mask_quality)
    while state.cycle_count < MAX_REFINEMENT_CYCLES:
        state.cycle_count += 1
        latest_mask = state.aliases.get("latest_mask", "") or state.aliases.get("base_mask", "")
        judgment = _judge_mask_quality(mask_quality)
        state.add_step(
            id=f"judge_mask_quality_cycle_{state.cycle_count}",
            kind="decision",
            message=f"Cycle {state.cycle_count}: quality={judgment['quality']} next_action={judgment['next_action']}",
            quality=judgment["quality"],
            next_action=judgment["next_action"],
            issues=list(judgment["issues"]),
        )
        if judgment["next_action"] in {"refine_threshold_stricter", "refine_threshold_looser"}:
            refine_route = _refine_threshold_arguments(mask_quality, state.threshold_state)
            if not refine_route:
                state.add_line("No safe threshold refinement could be derived, so the current mask was kept.")
                break
            direction_text = "stricter" if judgment["next_action"].endswith("stricter") else "looser"
            state.add_line(f"Refinement cycle {state.cycle_count}: the mask looked {judgment['quality'].replace('_', ' ')}, so the threshold was adjusted to be {direction_text}.")
            refine_result = _run_registered_tool(viewer, "apply_threshold", refine_route.get("arguments", {}))
            state.add_step(id=f"refine_threshold_cycle_{state.cycle_count}", kind="tool", **refine_result)
            if not refine_result["ok"]:
                state.add_line(f"Threshold refinement failed: {refine_result['message']}")
                break
            _maybe_record_output_alias({"tool": "apply_threshold", "output_alias": "base_mask"}, refine_result["result"], state.aliases)
            state.add_line(refine_result["message"])
            refine_payload = refine_result.get("result", {})
            state.threshold_state["threshold_value"] = refine_payload.get(
                "threshold_value",
                state.threshold_state.get("threshold_value"),
            )
            latest_mask = state.aliases.get("base_mask", latest_mask)
            mask_quality = _assess_mask_quality(viewer, latest_mask)
            state.add_step(
                id=f"measure_refined_mask_cycle_{state.cycle_count}",
                kind="analysis",
                message=mask_quality["message"],
                quality=mask_quality["quality"],
            )
            state.add_line(mask_quality["message"])
            mask_quality = _attempt_cleanup_actions(viewer, state, mask_quality)
            if mask_quality["quality"] == "acceptable":
                break
            continue

        mask_quality = _attempt_cleanup_actions(viewer, state, mask_quality)
        if mask_quality["quality"] == "acceptable":
            break
        if judgment["next_action"] == "stop":
            break
    return mask_quality


def execute_workflow_plan(viewer: napari.Viewer, plan: dict[str, Any] | Any) -> dict[str, Any]:
    payload = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan or {})
    workflow_type = str(payload.get("workflow_type", "")).strip().lower()
    if workflow_type != "conservative_binary_segmentation":
        return {
            "ok": False,
            "workflow_type": workflow_type or "unknown",
            "message": f"Unsupported workflow type: {workflow_type or 'unknown'}.",
            "executed_steps": [],
        }
    return _execute_conservative_binary_segmentation(viewer, payload)


def workflow_execution_to_compact_markdown(result: dict[str, Any] | Any) -> str:
    payload = dict(result or {})
    final_mask = str(payload.get("final_mask_layer", "")).strip()
    executed_steps = payload.get("executed_steps", [])
    summary_lines: list[str] = []
    for step in executed_steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id", "")).strip()
        message = str(step.get("message", "")).strip()
        if not message:
            continue
        if step_id == "inspect_selected_image":
            summary_lines.append(message)
        elif step_id == "gaussian_denoise_if_needed":
            summary_lines.append(message)
        elif step_id == "preview_threshold":
            summary_lines.append(message)
        elif step_id == "apply_threshold":
            summary_lines.append(message)
        elif "remove_tiny_specks" in step_id or "fill_small_holes" in step_id or "smooth_jagged_edges" in step_id:
            summary_lines.append(message)
        elif step_id == "stop_when_conservative_quality_is_met":
            summary_lines.append(message)
    deduped: list[str] = []
    for line in summary_lines:
        if line not in deduped:
            deduped.append(line)
    lines = []
    if final_mask:
        lines.append(f"Built a conservative mask for [{final_mask}].")
    else:
        lines.append("Built a conservative mask.")
    if deduped:
        lines.append("")
        lines.append("Summary:")
        for line in deduped[:6]:
            lines.append(f"- {line}")
    lines.append("")
    lines.append("Ask `show details`, `show plan`, or `show debug` if you want the full workflow trace.")
    return "\n".join(lines).strip()


def workflow_execution_to_debug_markdown(result: dict[str, Any] | Any) -> str:
    payload = dict(result or {})
    return str(payload.get("message", "")).strip() or "No workflow debug trace is available."


def _execute_conservative_binary_segmentation(viewer: napari.Viewer, plan: dict[str, Any]) -> dict[str, Any]:
    target_layer = str(plan.get("target_layer", "")).strip()
    state = WorkflowExecutionState(
        target_layer=target_layer,
        aliases={"selected_image": target_layer, "working_image": target_layer},
    )

    inspection = _inspect_image(viewer, target_layer)
    state.add_step(id="inspect_selected_image", kind="analysis", message=inspection["message"])
    state.add_line(inspection["message"])

    polarity = str(inspection.get("polarity", "auto") or "auto")
    state.add_line(
        (
            "Applying light denoising because the fine-grained noise estimate is high enough to interfere with thresholding."
            if inspection.get("denoise_needed")
            else "Skipping denoising because the current noise estimate does not justify smoothing faint structures."
        )
    )
    if inspection.get("denoise_needed"):
        denoise_step = {
            "tool": "gaussian_denoise",
            "arguments": {"layer_name": target_layer, "sigma": 1.0, "preserve_range": True},
            "output_alias": "working_image",
        }
        denoise_result = _run_registered_tool(viewer, "gaussian_denoise", denoise_step["arguments"])
        state.add_step(id="gaussian_denoise_if_needed", kind="tool", **denoise_result)
        if denoise_result["ok"]:
            _maybe_record_output_alias(denoise_step, denoise_result["result"], state.aliases)
            state.add_line(denoise_result["message"])
        else:
            state.add_line(f"Denoising was requested but failed: {denoise_result['message']}")

    state.add_line(
        f"Chose {'bright-foreground' if polarity == 'bright' else 'dim-foreground'} thresholding because the intensity spread suggests the foreground is more likely "
        f"{'brighter' if polarity == 'bright' else 'dimmer'} than the background."
    )

    preview_args = _resolve_argument_placeholders(
        {"layer_name": "$working_image_or_selected", "polarity": polarity},
        state.aliases,
        target_layer,
    )
    preview_step = {"tool": "preview_threshold", "output_alias": "threshold_preview"}
    preview_result = _run_registered_tool(viewer, "preview_threshold", preview_args)
    state.add_step(id="preview_threshold", kind="tool", **preview_result)
    if not preview_result["ok"]:
        state.add_line(f"Threshold preview failed: {preview_result['message']}")
        return {
            "ok": False,
            "workflow_type": "conservative_binary_segmentation",
            "message": "\n".join(state.lines),
            "executed_steps": state.executed_steps,
            "aliases": state.aliases,
        }
    _maybe_record_output_alias(preview_step, preview_result["result"], state.aliases)
    preview_payload = preview_result.get("result", {})
    state.threshold_state = {
        "layer_name": str(preview_args.get("layer_name", "")).strip() or target_layer,
        "polarity": polarity,
        "threshold_value": preview_payload.get("threshold_value"),
        "image_min": inspection.get("image_min"),
        "image_max": inspection.get("image_max"),
    }
    state.add_line(preview_result["message"])

    preview_quality = _assess_mask_quality(viewer, state.aliases.get("threshold_preview", "__assistant_threshold_preview__"))
    state.add_step(id="review_threshold_preview", kind="analysis", message=preview_quality["message"], quality=preview_quality["quality"])
    state.add_line(preview_quality["message"])

    apply_args = _resolve_argument_placeholders(
        {"layer_name": "$working_image_or_selected", "polarity": polarity},
        state.aliases,
        target_layer,
    )
    apply_step = {"tool": "apply_threshold", "output_alias": "base_mask"}
    apply_result = _run_registered_tool(viewer, "apply_threshold", apply_args)
    state.add_step(id="apply_threshold", kind="tool", **apply_result)
    if not apply_result["ok"]:
        state.add_line(f"Threshold application failed: {apply_result['message']}")
        return {
            "ok": False,
            "workflow_type": "conservative_binary_segmentation",
            "message": "\n".join(state.lines),
            "executed_steps": state.executed_steps,
            "aliases": state.aliases,
        }
    _maybe_record_output_alias(apply_step, apply_result["result"], state.aliases)
    apply_payload = apply_result.get("result", {})
    state.threshold_state["threshold_value"] = apply_payload.get("threshold_value", state.threshold_state["threshold_value"])
    state.add_line(apply_result["message"])

    base_mask_name = state.aliases.get("base_mask", "")
    base_quality = _assess_mask_quality(viewer, base_mask_name)
    state.add_step(id="measure_base_mask", kind="analysis", message=base_quality["message"], quality=base_quality["quality"])
    state.add_line(base_quality["message"])

    final_quality = _run_refinement_loop(viewer, state, base_quality)
    final_mask_name = state.aliases.get("latest_mask", base_mask_name)

    stop_message = (
        f"Stopped with [{final_mask_name}] because the mask is in a conservative operating range."
        if final_quality["quality"] == "acceptable"
        else f"Stopped with [{final_mask_name}] but the result still looks {final_quality['quality'].replace('_', ' ')} and may need manual follow-up."
    )
    state.add_step(id="stop_when_conservative_quality_is_met", kind="stop_check", message=stop_message)
    state.add_line(stop_message)

    return {
        "ok": True,
        "workflow_type": "conservative_binary_segmentation",
        "message": "\n".join(state.lines).strip(),
        "executed_steps": state.executed_steps,
        "aliases": state.aliases,
        "final_mask_layer": final_mask_name,
        "cycle_count": state.cycle_count,
    }
