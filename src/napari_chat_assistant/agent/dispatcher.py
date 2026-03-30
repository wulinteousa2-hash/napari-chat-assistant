from __future__ import annotations

import importlib

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
    layer_context_json,
)
from .image_ops import apply_clahe, auto_threshold_mask, mask_statistics
from .image_ops import compare_intensity_populations, intensity_histogram, intensity_statistics
from .tool_registry import TOOL_REGISTRY
from .tool_types import PreparedJob, ToolContext, ToolResult
from .tools import (
    TOOL_OPS,
    next_output_name,
    next_snapshot_name,
    normalize_float,
    normalize_int,
    normalize_kernel_size,
    normalize_polarity,
    save_mask_snapshot,
)
from .tools_builtin import builtin_tools


_ND2_INTEGRATION_MESSAGE = (
    "ND2 conversion and spectral-analysis integration is not available in this environment.\n\n"
    "To enable Nikon ND2 to OME-Zarr workflows, install `napari-nd2-spectral-ome-zarr`.\n\n"
    "GitHub:\n"
    "https://github.com/wulinteousa2-hash/napari-nd2-spectral-ome-zarr\n\n"
    "napari Hub:\n"
    "https://napari-hub.org/plugins/napari-nd2-spectral-ome-zarr.html"
)


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


def _format_intensity_summary(layer_name: str, stats: dict) -> str:
    return (
        f"Intensity Summary\n"
        f"Layer: [{layer_name}]\n"
        f"Pixels: {stats['count']}\n"
        f"Mean: {stats['mean']:.6g}\n"
        f"Std Dev: {stats['std']:.6g}\n"
        f"Median: {stats['median']:.6g}\n"
        f"Min: {stats['min']:.6g}\n"
        f"Max: {stats['max']:.6g}"
    )


def _load_optional_widget(module_name: str, class_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, class_name, None)


def _dock_optional_widget(viewer: napari.Viewer, widget_cls, display_name: str) -> str:
    if widget_cls is None:
        return _ND2_INTEGRATION_MESSAGE
    widget = widget_cls(viewer)
    viewer.window.add_dock_widget(widget, name=display_name)
    return f"Opened [{display_name}] from napari-nd2-spectral-ome-zarr."


def prepare_tool_job(viewer: napari.Viewer, tool_name: str, arguments: dict) -> dict:
    _ensure_builtin_registry()
    args = arguments or {}
    registry_tool = TOOL_REGISTRY.get(tool_name)
    if registry_tool is not None:
        prepared = registry_tool.prepare(_tool_context(viewer), args)
        if isinstance(prepared, str):
            return {"mode": "immediate", "message": prepared}
        return {"mode": prepared.mode, "job": prepared.to_dict()}

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

    if tool_name == "open_nd2_converter":
        widget_cls = _load_optional_widget("napari_nd2_spectral_ome_zarr._widget", "Nd2SpectralWidget")
        return {"mode": "immediate", "message": _dock_optional_widget(viewer, widget_cls, "ND2 Spectral Export")}

    if tool_name == "open_spectral_viewer":
        widget_cls = _load_optional_widget("napari_nd2_spectral_ome_zarr._spectral_viewer", "SpectralViewerWidget")
        return {"mode": "immediate", "message": _dock_optional_widget(viewer, widget_cls, "Spectral Viewer")}

    if tool_name == "open_spectral_analysis":
        widget_cls = _load_optional_widget("napari_nd2_spectral_ome_zarr._spectral_analysis", "SpectralAnalysisWidget")
        return {"mode": "immediate", "message": _dock_optional_widget(viewer, widget_cls, "Spectral Analysis")}

    if tool_name == "apply_clahe_batch":
        image_layers = find_all_image_layers(viewer)
        if not image_layers:
            return {"mode": "immediate", "message": "No image layers available for batch CLAHE."}
        items = []
        for layer in image_layers:
            if getattr(layer, "rgb", False):
                continue
            layer_data = np.asarray(layer.data)
            items.append(
                {
                    "layer_name": layer.name,
                    "output_name": next_output_name(viewer, f"{layer.name}_clahe"),
                    "kernel_size": normalize_kernel_size(args.get("kernel_size", 32), ndim=layer_data.ndim),
                    "clip_limit": normalize_float(args.get("clip_limit", 0.01), default=0.01, minimum=1e-6, maximum=10.0),
                    "nbins": normalize_int(args.get("nbins", 256), default=256, minimum=2, maximum=65536),
                    "data": layer_data.copy(),
                    "scale": tuple(layer.scale),
                    "translate": tuple(layer.translate),
                }
            )
        if not items:
            return {"mode": "immediate", "message": "No grayscale 2D/3D image layers available for batch CLAHE."}
        return {"mode": "worker", "job": {"kind": "apply_clahe_batch", "items": items}}

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

    if tool_name == "summarize_intensity":
        image_layer = find_image_layer(viewer, args.get("layer_name"))
        if image_layer is None:
            return {"mode": "immediate", "message": "No valid image layer available for intensity summary."}
        stats = intensity_statistics(np.asarray(image_layer.data))
        return {
            "mode": "immediate",
            "message": _format_intensity_summary(image_layer.name, stats),
        }

    if tool_name == "plot_histogram":
        image_layer = find_image_layer(viewer, args.get("layer_name"))
        if image_layer is None:
            return {"mode": "immediate", "message": "No valid image layer available for histogram plotting."}
        return {
            "mode": "worker",
            "job": {
                "kind": "plot_histogram",
                "layer_name": image_layer.name,
                "bins": normalize_int(args.get("bins", 64), default=64, minimum=2, maximum=512),
                "data": np.asarray(image_layer.data).copy(),
            },
        }

    if tool_name == "compare_image_layers_ttest":
        image_layers = find_all_image_layers(viewer)
        if len(image_layers) < 2:
            return {"mode": "immediate", "message": "At least 2 image layers are required for a t-test comparison."}
        pair = _resolve_image_layer_pair(viewer, args)
        if pair is None:
            return {
                "mode": "immediate",
                "message": "Could not resolve 2 image layers for comparison. Specify layer_name_a and layer_name_b.",
            }
        layer_a, layer_b = pair
        return {
            "mode": "worker",
            "job": {
                "kind": "compare_image_layers_ttest",
                "layer_name_a": layer_a.name,
                "layer_name_b": layer_b.name,
                "equal_var": bool(args.get("equal_var", True)),
                "data_a": np.asarray(layer_a.data).copy(),
                "data_b": np.asarray(layer_b.data).copy(),
            },
        }

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
    _ensure_builtin_registry()
    registry_tool = TOOL_REGISTRY.get(job.get("tool_name", ""))
    if registry_tool is not None:
        prepared_job = PreparedJob.from_dict(job)
        return registry_tool.execute(prepared_job).to_dict()

    kind = job["kind"]
    if kind == "apply_clahe_batch":
        results = []
        for item in job["items"]:
            result = apply_clahe(
                item["data"],
                kernel_size=item["kernel_size"],
                clip_limit=item["clip_limit"],
                nbins=item["nbins"],
            )
            results.append({**item, "result": result})
        return {"kind": kind, "items": results}

    if kind in {"preview_threshold", "apply_threshold"}:
        threshold_value, labels = auto_threshold_mask(job["data"], polarity=job["polarity"])
        return {**job, "kind": kind, "threshold_value": threshold_value, "labels": labels, "stats": mask_statistics(labels)}

    if kind == "plot_histogram":
        histogram = intensity_histogram(job["data"], bins=job["bins"])
        return {**job, "kind": kind, "histogram": histogram}

    if kind == "compare_image_layers_ttest":
        comparison = compare_intensity_populations(job["data_a"], job["data_b"], equal_var=job["equal_var"])
        return {**job, "kind": kind, "comparison": comparison}

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
    _ensure_builtin_registry()
    registry_tool = TOOL_REGISTRY.get(result.get("tool_name", ""))
    if registry_tool is not None:
        return registry_tool.apply(_tool_context(viewer), ToolResult.from_dict(result))

    kind = result["kind"]
    if kind == "apply_clahe_batch":
        lines = []
        for item in result["items"]:
            viewer.add_image(
                item["result"],
                name=item["output_name"],
                scale=item["scale"],
                translate=item["translate"],
            )
            lines.append(
                f"[{item['layer_name']}] -> [{item['output_name']}] "
                f"kernel_size={item['kernel_size']} clip_limit={item['clip_limit']:.6g} nbins={item['nbins']}."
            )
        return f"Applied CLAHE to {len(result['items'])} image layers.\n" + "\n".join(lines)

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

    if kind == "plot_histogram":
        histogram = result["histogram"]
        _show_histogram_popup(result["layer_name"], histogram)
        stats = histogram["stats"]
        return (
            f"Opened histogram for [{result['layer_name']}] with {histogram['bins']} bins. "
            f"n={stats['count']} mean={stats['mean']:.6g} std={stats['std']:.6g}."
        )

    if kind == "compare_image_layers_ttest":
        comparison = result["comparison"]
        return (
            f"{comparison['test_name']} for [{result['layer_name_a']}] vs [{result['layer_name_b']}]: "
            f"t={comparison['statistic']:.6g} p={comparison['pvalue']:.6g} "
            f"mean_a={comparison['mean_a']:.6g} mean_b={comparison['mean_b']:.6g} "
            f"delta={comparison['delta_mean']:.6g} n_a={comparison['count_a']} n_b={comparison['count_b']}."
        )

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


def _resolve_image_layer_pair(viewer: napari.Viewer, args: dict):
    name_a = str(args.get("layer_name_a") or "").strip()
    name_b = str(args.get("layer_name_b") or "").strip()
    layer_a = find_image_layer(viewer, name_a) if name_a else None
    layer_b = find_image_layer(viewer, name_b) if name_b else None
    if layer_a is not None and layer_b is not None and layer_a is not layer_b:
        return layer_a, layer_b
    image_layers = find_all_image_layers(viewer)
    if len(image_layers) == 2:
        return image_layers[0], image_layers[1]
    selected = viewer.layers.selection.active if viewer is not None else None
    if isinstance(selected, napari.layers.Image):
        others = [layer for layer in image_layers if layer is not selected]
        if len(others) == 1:
            return selected, others[0]
    return None


def _show_histogram_popup(layer_name: str, histogram: dict) -> None:
    import matplotlib.pyplot as plt

    counts = np.asarray(histogram["counts"])
    bin_edges = np.asarray(histogram["bin_edges"])
    stats = histogram["stats"]

    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        align="edge",
        color="steelblue",
        edgecolor="black",
    )
    ax.axvline(stats["mean"], color="green", linestyle="--", linewidth=2, label="Mean")
    ax.axvline(stats["median"], color="darkorange", linestyle=":", linewidth=2, label="Median")
    ax.set_title(f"Intensity Histogram: {layer_name}")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    plt.show(block=False)
