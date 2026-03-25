from __future__ import annotations

import napari
import numpy as np

from .context import (
    find_any_layer,
    find_all_image_layers,
    find_all_labels_layers,
    find_image_layer,
    find_labels_layer,
    layer_detail_summary,
    layer_summary,
    mask_measurement_summary,
)
from .image_ops import auto_threshold_mask, mask_statistics
from .tools import TOOL_OPS, next_output_name, next_snapshot_name, normalize_int, normalize_polarity, save_mask_snapshot


def prepare_tool_job(viewer: napari.Viewer, tool_name: str, arguments: dict) -> dict:
    args = arguments or {}
    if tool_name == "list_layers":
        return {"mode": "immediate", "message": layer_summary(viewer)}

    if tool_name == "inspect_selected_layer":
        layer = find_any_layer(viewer)
        return {"mode": "immediate", "message": layer_detail_summary(layer)}

    if tool_name == "inspect_layer":
        layer = find_any_layer(viewer, args.get("layer_name"))
        if layer is None:
            return {"mode": "immediate", "message": f"No layer found with name [{args.get('layer_name', '')}]."}
        return {"mode": "immediate", "message": layer_detail_summary(layer)}

    if tool_name in ("preview_threshold", "apply_threshold"):
        image_layer = find_image_layer(viewer, args.get("layer_name"))
        if image_layer is None:
            return {"mode": "immediate", "message": "No valid image layer available."}
        polarity = normalize_polarity(args.get("polarity", "auto"))
        if tool_name == "preview_threshold":
            return {
                "mode": "worker",
                "job": {
                    "kind": "preview_threshold",
                    "layer_name": image_layer.name,
                    "polarity": polarity,
                    "data": np.asarray(image_layer.data).copy(),
                    "scale": tuple(image_layer.scale),
                    "translate": tuple(image_layer.translate),
                },
            }

        return {
            "mode": "worker",
            "job": {
                "kind": "apply_threshold",
                "layer_name": image_layer.name,
                "output_name": next_output_name(viewer, f"{image_layer.name}_labels"),
                "polarity": polarity,
                "data": np.asarray(image_layer.data).copy(),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
            },
        }

    if tool_name in ("preview_threshold_batch", "apply_threshold_batch"):
        image_layers = find_all_image_layers(viewer)
        if not image_layers:
            return {"mode": "immediate", "message": "No image layers available for batch thresholding."}
        polarity = normalize_polarity(args.get("polarity", "auto"))
        items = []
        for layer in image_layers:
            item = {
                "layer_name": layer.name,
                "polarity": polarity,
                "data": np.asarray(layer.data).copy(),
                "scale": tuple(layer.scale),
                "translate": tuple(layer.translate),
            }
            if tool_name == "apply_threshold_batch":
                item["output_name"] = next_output_name(viewer, f"{layer.name}_labels")
            items.append(item)
        return {"mode": "worker", "job": {"kind": tool_name, "polarity": polarity, "items": items}}

    if tool_name == "measure_mask":
        labels_layer = find_labels_layer(viewer, args.get("layer_name"))
        if labels_layer is None:
            return {"mode": "immediate", "message": "No valid labels layer available."}
        return {"mode": "immediate", "message": mask_measurement_summary(labels_layer)}

    if tool_name == "measure_mask_batch":
        labels_layers = find_all_labels_layers(viewer)
        if not labels_layers:
            return {"mode": "immediate", "message": "No labels layers available for batch measurement."}
        return {"mode": "immediate", "message": "\n".join(mask_measurement_summary(layer) for layer in labels_layers)}

    if tool_name == "run_mask_op":
        labels_layer = find_labels_layer(viewer, args.get("layer_name"))
        if labels_layer is None:
            return {"mode": "immediate", "message": "No valid labels layer available."}
        op_name = str(args.get("op", "")).strip().lower()
        if op_name not in TOOL_OPS:
            return {"mode": "immediate", "message": f"Unsupported mask operation: {op_name}"}
        safe_args = dict(args)
        safe_args["radius"] = normalize_int(args.get("radius", 1), default=1, minimum=1, maximum=20)
        safe_args["min_size"] = normalize_int(args.get("min_size", 64), default=64, minimum=1, maximum=10_000_000)
        return {
            "mode": "worker",
            "job": {
                "kind": "run_mask_op",
                "layer_name": labels_layer.name,
                "snapshot_name": next_snapshot_name(viewer, labels_layer.name),
                "op_name": op_name,
                "args": safe_args,
                "data": np.asarray(labels_layer.data).copy(),
                "scale": tuple(labels_layer.scale),
                "translate": tuple(labels_layer.translate),
            },
        }

    if tool_name == "run_mask_op_batch":
        labels_layers = find_all_labels_layers(viewer)
        if not labels_layers:
            return {"mode": "immediate", "message": "No labels layers available for batch mask operations."}
        op_name = str(args.get("op", "")).strip().lower()
        if op_name not in TOOL_OPS:
            return {"mode": "immediate", "message": f"Unsupported mask operation: {op_name}"}
        safe_args = dict(args)
        safe_args["radius"] = normalize_int(args.get("radius", 1), default=1, minimum=1, maximum=20)
        safe_args["min_size"] = normalize_int(args.get("min_size", 64), default=64, minimum=1, maximum=10_000_000)
        items = []
        for layer in labels_layers:
            items.append(
                {
                    "layer_name": layer.name,
                    "snapshot_name": next_snapshot_name(viewer, layer.name),
                    "op_name": op_name,
                    "args": safe_args,
                    "data": np.asarray(layer.data).copy(),
                    "scale": tuple(layer.scale),
                    "translate": tuple(layer.translate),
                }
            )
        return {"mode": "worker", "job": {"kind": "run_mask_op_batch", "op_name": op_name, "items": items}}

    return {"mode": "immediate", "message": f"Unsupported tool: {tool_name}"}


def run_tool_job(job: dict) -> dict:
    kind = job["kind"]
    if kind in {"preview_threshold", "apply_threshold"}:
        threshold_value, labels = auto_threshold_mask(job["data"], polarity=job["polarity"])
        return {**job, "kind": kind, "threshold_value": threshold_value, "labels": labels, "stats": mask_statistics(labels)}

    if kind in {"preview_threshold_batch", "apply_threshold_batch"}:
        results = []
        for item in job["items"]:
            threshold_value, labels = auto_threshold_mask(item["data"], polarity=item["polarity"])
            results.append({**item, "threshold_value": threshold_value, "labels": labels, "stats": mask_statistics(labels)})
        return {"kind": kind, "items": results, "polarity": job["polarity"]}

    if kind == "run_mask_op":
        op = TOOL_OPS[job["op_name"]]
        result = op(job["data"], job["args"])
        return {**job, "kind": kind, "result": result, "stats": mask_statistics(result)}

    if kind == "run_mask_op_batch":
        results = []
        for item in job["items"]:
            op = TOOL_OPS[item["op_name"]]
            result = op(item["data"], item["args"])
            results.append({**item, "result": result, "stats": mask_statistics(result)})
        return {"kind": kind, "op_name": job["op_name"], "items": results}

    raise ValueError(f"Unsupported worker job kind: {kind}")


def apply_tool_job_result(viewer: napari.Viewer, result: dict) -> str:
    kind = result["kind"]
    if kind == "preview_threshold":
        preview_name = "__assistant_threshold_preview__"
        labels = result["labels"]
        if preview_name in viewer.layers and isinstance(viewer.layers[preview_name], napari.layers.Labels):
            preview = viewer.layers[preview_name]
            preview.data = labels
            preview.scale = result["scale"]
            preview.translate = result["translate"]
        else:
            viewer.add_labels(labels, name=preview_name, scale=result["scale"], translate=result["translate"])
        stats = result["stats"]
        return (
            f"Preview threshold updated for [{result['layer_name']}] at {result['threshold_value']:.6g} "
            f"with polarity={result['polarity']}. objects={stats['object_count']} fg={stats['foreground_pixels']} px."
        )

    if kind == "apply_threshold":
        viewer.add_labels(result["labels"], name=result["output_name"], scale=result["scale"], translate=result["translate"])
        stats = result["stats"]
        return (
            f"Applied threshold to [{result['layer_name']}] as [{result['output_name']}] at {result['threshold_value']:.6g} "
            f"with polarity={result['polarity']}. objects={stats['object_count']} fg={stats['foreground_pixels']} px."
        )

    if kind == "preview_threshold_batch":
        lines = []
        for item in result["items"]:
            preview_name = f"__assistant_threshold_preview__::{item['layer_name']}"
            if preview_name in viewer.layers and isinstance(viewer.layers[preview_name], napari.layers.Labels):
                preview = viewer.layers[preview_name]
                preview.data = item["labels"]
                preview.scale = item["scale"]
                preview.translate = item["translate"]
            else:
                viewer.add_labels(item["labels"], name=preview_name, scale=item["scale"], translate=item["translate"])
            stats = item["stats"]
            lines.append(
                f"[{item['layer_name']}] preview at {item['threshold_value']:.6g} "
                f"objects={stats['object_count']} fg={stats['foreground_pixels']} px."
            )
        return f"Updated preview masks for {len(result['items'])} image layers with polarity={result['polarity']}.\n" + "\n".join(lines)

    if kind == "apply_threshold_batch":
        lines = []
        for item in result["items"]:
            viewer.add_labels(item["labels"], name=item["output_name"], scale=item["scale"], translate=item["translate"])
            stats = item["stats"]
            lines.append(
                f"[{item['layer_name']}] -> [{item['output_name']}] at {item['threshold_value']:.6g} "
                f"objects={stats['object_count']} fg={stats['foreground_pixels']} px."
            )
        return f"Applied threshold to {len(result['items'])} image layers with polarity={result['polarity']}.\n" + "\n".join(lines)

    if kind == "run_mask_op":
        labels_layer = find_labels_layer(viewer, result["layer_name"])
        if labels_layer is None:
            return f"Labels layer [{result['layer_name']}] is no longer available."
        snapshot_name = save_mask_snapshot(viewer, labels_layer)
        labels_layer.data = result["result"]
        stats = result["stats"]
        return (
            f"Saved snapshot [{snapshot_name}] and applied {result['op_name']} to [{labels_layer.name}]. "
            f"objects={stats['object_count']} fg={stats['foreground_pixels']} px largest={stats['largest_object']} px."
        )

    if kind == "run_mask_op_batch":
        lines = []
        applied = 0
        for item in result["items"]:
            labels_layer = find_labels_layer(viewer, item["layer_name"])
            if labels_layer is None:
                lines.append(f"[{item['layer_name']}] skipped because the layer is no longer available.")
                continue
            snapshot_name = save_mask_snapshot(viewer, labels_layer)
            labels_layer.data = item["result"]
            stats = item["stats"]
            applied += 1
            lines.append(
                f"[{labels_layer.name}] snapshot [{snapshot_name}] {item['op_name']} "
                f"objects={stats['object_count']} fg={stats['foreground_pixels']} px largest={stats['largest_object']} px."
            )
        return f"Applied {result['op_name']} to {applied} labels layers.\n" + "\n".join(lines)

    return f"Unsupported tool result: {kind}"
