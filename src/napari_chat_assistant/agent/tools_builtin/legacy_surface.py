from __future__ import annotations

import importlib

import napari
import numpy as np

from ..context import (
    find_all_image_layers,
    find_all_labels_layers,
    find_any_layer,
    find_image_layer,
    find_labels_layer,
    find_shapes_layer,
    layer_detail_summary,
    layer_summary,
    mask_measurement_summary,
)
from ..image_ops import (
    apply_clahe,
    auto_threshold_mask,
    compare_intensity_populations,
    intensity_histogram,
    intensity_statistics,
    mask_statistics,
)
from ..tool_types import ParamSpec, PreparedJob, ToolContext, ToolResult, ToolSpec
from ..tools import (
    TOOL_OPS,
    TOOL_OP_OUTPUTS,
    next_output_name,
    next_snapshot_name,
    normalize_float,
    normalize_int,
    normalize_kernel_size,
    normalize_polarity,
    save_mask_snapshot,
)
from ...widgets.group_comparison_widget import open_group_comparison_widget
from ...widgets.intensity_metrics_widget import open_intensity_metrics_widget


_ND2_INTEGRATION_MESSAGE = (
    "ND2 conversion and spectral-analysis integration is not available in this environment.\n\n"
    "To enable Nikon ND2 to OME-Zarr workflows, install `napari-nd2-spectral-ome-zarr`.\n\n"
    "GitHub:\n"
    "https://github.com/wulinteousa2-hash/napari-nd2-spectral-ome-zarr\n\n"
    "napari Hub:\n"
    "https://napari-hub.org/plugins/napari-nd2-spectral-ome-zarr.html"
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


def _selected_image_plane(viewer: napari.Viewer, layer: napari.layers.Image) -> np.ndarray:
    data = np.asarray(layer.data, dtype=np.float32)
    if data.ndim < 2:
        raise ValueError(f"Layer [{layer.name}] must have at least 2 dimensions.")
    if data.ndim == 2:
        return data
    current_step = tuple(int(step) for step in viewer.dims.current_step[: data.ndim])
    leading_shape = data.shape[:-2]
    leading_indices = []
    for axis, axis_size in enumerate(leading_shape):
        step_index = current_step[axis] if axis < len(current_step) else 0
        leading_indices.append(int(np.clip(step_index, 0, axis_size - 1)))
    plane = data[tuple(leading_indices) + (slice(None), slice(None))]
    return np.asarray(plane, dtype=np.float32)


def _shape_label(layer: napari.layers.Shapes, index: int) -> str:
    features = getattr(layer, "features", None)
    if features is not None and "label" in features:
        try:
            values = list(features["label"])
            if index < len(values):
                label = str(values[index]).strip()
                if label:
                    return label
        except Exception:
            pass
    return f"Shape {index + 1}"


def _format_centroid(point_yx: np.ndarray | None) -> str:
    if point_yx is None or len(point_yx) < 2:
        return "n/a"
    return f"(y={float(point_yx[0]):.2f}, x={float(point_yx[1]):.2f})"


def _format_bbox_yx(mins: np.ndarray | None, maxs: np.ndarray | None) -> str:
    if mins is None or maxs is None or len(mins) < 2 or len(maxs) < 2:
        return "n/a"
    return f"y=[{float(mins[0]):.2f}, {float(maxs[0]):.2f}] x=[{float(mins[1]):.2f}, {float(maxs[1]):.2f}]"


def _format_roi_intensity_stats(values: np.ndarray) -> str:
    if values.size == 0:
        return "intensity: no finite pixels"
    return (
        f"intensity: pixels={int(values.size)} mean={float(np.mean(values)):.2f} "
        f"std={float(np.std(values)):.2f} median={float(np.median(values)):.2f} "
        f"min={float(np.min(values)):.2f} max={float(np.max(values)):.2f}"
    )


def _load_optional_widget(module_name: str, class_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, class_name, None)


def _optional_threshold_value(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except Exception:
        return None


def _dock_optional_widget(viewer: napari.Viewer, widget_cls, display_name: str) -> str:
    if widget_cls is None:
        return _ND2_INTEGRATION_MESSAGE
    widget = widget_cls(viewer)
    viewer.window.add_dock_widget(widget, name=display_name)
    return f"Opened [{display_name}] from napari-nd2-spectral-ome-zarr."


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


def _threshold_mode_text(mode: str) -> str:
    normalized = str(mode or "auto").strip().lower()
    if normalized == "bright":
        return "keeping brighter regions"
    if normalized == "dim":
        return "keeping dimmer regions"
    return "using automatic foreground selection"


class ListLayersTool:
    spec = ToolSpec(
        name="list_layers",
        display_name="List Layers",
        category="inspection",
        description="List layers in the current viewer and report the active selection.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        output_type="message",
        ui_metadata={"panel_group": "Inspection"},
        provenance_metadata={"algorithm": "layer_summary", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        return layer_summary(ctx.viewer)

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class InspectSelectedLayerTool:
    spec = ToolSpec(
        name="inspect_selected_layer",
        display_name="Inspect Selected Layer",
        category="inspection",
        description="Summarize the currently selected layer.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        output_type="message",
        ui_metadata={"panel_group": "Inspection"},
        provenance_metadata={"algorithm": "layer_detail_summary", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        layer = find_any_layer(ctx.viewer)
        return layer_detail_summary(layer)

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class InspectLayerTool:
    spec = ToolSpec(
        name="inspect_layer",
        display_name="Inspect Layer",
        category="inspection",
        description="Summarize a named layer.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        parameter_schema=(ParamSpec("layer_name", "string", description="Layer name to inspect."),),
        output_type="message",
        ui_metadata={"panel_group": "Inspection"},
        provenance_metadata={"algorithm": "layer_detail_summary", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        layer_name = (arguments or {}).get("layer_name")
        layer = find_any_layer(ctx.viewer, layer_name)
        if layer is None:
            return f"No layer found with name [{layer_name or ''}]."
        return layer_detail_summary(layer)

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class OpenND2ConverterTool:
    spec = ToolSpec(
        name="open_nd2_converter",
        display_name="Open ND2 Converter",
        category="integration",
        description="Open the ND2 conversion widget when the optional integration is installed.",
        execution_mode="immediate",
        output_type="message",
        ui_metadata={"panel_group": "Integration"},
        provenance_metadata={"algorithm": "optional_widget_launch", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        widget_cls = _load_optional_widget("napari_nd2_spectral_ome_zarr._widget", "Nd2SpectralWidget")
        return _dock_optional_widget(ctx.viewer, widget_cls, "ND2 Spectral Export")

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class OpenSpectralViewerTool:
    spec = ToolSpec(
        name="open_spectral_viewer",
        display_name="Open Spectral Viewer",
        category="integration",
        description="Open the spectral viewer widget when the optional integration is installed.",
        execution_mode="immediate",
        output_type="message",
        ui_metadata={"panel_group": "Integration"},
        provenance_metadata={"algorithm": "optional_widget_launch", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        widget_cls = _load_optional_widget("napari_nd2_spectral_ome_zarr._spectral_viewer", "SpectralViewerWidget")
        return _dock_optional_widget(ctx.viewer, widget_cls, "Spectral Viewer")

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class OpenSpectralAnalysisTool:
    spec = ToolSpec(
        name="open_spectral_analysis",
        display_name="Open Spectral Analysis",
        category="integration",
        description="Open the spectral analysis widget when the optional integration is installed.",
        execution_mode="immediate",
        output_type="message",
        ui_metadata={"panel_group": "Integration"},
        provenance_metadata={"algorithm": "optional_widget_launch", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        widget_cls = _load_optional_widget("napari_nd2_spectral_ome_zarr._spectral_analysis", "SpectralAnalysisWidget")
        return _dock_optional_widget(ctx.viewer, widget_cls, "Spectral Analysis")

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class ApplyClaheBatchTool:
    spec = ToolSpec(
        name="apply_clahe_batch",
        display_name="Apply CLAHE Batch",
        category="enhancement",
        description="Apply CLAHE to all grayscale image layers.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("kernel_size", "int_or_int_list", description="CLAHE kernel size.", default=32),
            ParamSpec("clip_limit", "float", description="CLAHE clip limit.", default=0.01),
            ParamSpec("nbins", "int", description="Histogram bin count.", default=256),
        ),
        output_type="layer",
        supports_batch=True,
        ui_metadata={"panel_group": "Enhancement"},
        provenance_metadata={"algorithm": "clahe", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layers = find_all_image_layers(ctx.viewer)
        if not image_layers:
            return "No image layers available for batch CLAHE."
        items = []
        for layer in image_layers:
            if getattr(layer, "rgb", False):
                continue
            layer_data = np.asarray(layer.data)
            items.append(
                {
                    "layer_name": layer.name,
                    "output_name": next_output_name(ctx.viewer, f"{layer.name}_clahe"),
                    "kernel_size": normalize_kernel_size(args.get("kernel_size", 32), ndim=layer_data.ndim),
                    "clip_limit": normalize_float(args.get("clip_limit", 0.01), default=0.01, minimum=1e-6, maximum=10.0),
                    "nbins": normalize_int(args.get("nbins", 256), default=256, minimum=2, maximum=65536),
                    "data": layer_data.copy(),
                    "scale": tuple(layer.scale),
                    "translate": tuple(layer.translate),
                }
            )
        if not items:
            return "No grayscale 2D/3D image layers available for batch CLAHE."
        return PreparedJob(tool_name=self.spec.name, kind=self.spec.name, mode="worker", payload={"kind": self.spec.name, "items": items})

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        results = []
        for item in payload["items"]:
            result = apply_clahe(
                item["data"],
                kernel_size=item["kernel_size"],
                clip_limit=item["clip_limit"],
                nbins=item["nbins"],
            )
            results.append({**item, "result": result})
        payload["items"] = results
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        lines = []
        for item in result.payload["items"]:
            ctx.viewer.add_image(
                item["result"],
                name=item["output_name"],
                scale=item["scale"],
                translate=item["translate"],
            )
            lines.append(
                f"[{item['layer_name']}] -> [{item['output_name']}] "
                f"kernel_size={item['kernel_size']} clip_limit={item['clip_limit']:.6g} nbins={item['nbins']}."
            )
        return f"Applied CLAHE to {len(result.payload['items'])} image layers.\n" + "\n".join(lines)


class PreviewThresholdTool:
    spec = ToolSpec(
        name="preview_threshold",
        display_name="Preview Threshold",
        category="segmentation",
        description="Preview an automatic threshold as a temporary labels layer.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="Optional image layer.", required=False),
            ParamSpec("polarity", "string", description="Threshold polarity.", default="auto", enum=("auto", "bright", "dim")),
            ParamSpec("threshold_value", "float", description="Optional manual threshold cutoff.", required=False),
        ),
        output_type="layer",
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "auto_threshold", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("layer_name"))
        if image_layer is None:
            return "No valid image layer available."
        polarity = normalize_polarity(args.get("polarity", "auto"))
        threshold_value = _optional_threshold_value(args.get("threshold_value"))
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": image_layer.name,
                "polarity": polarity,
                "threshold_value": threshold_value,
                "data": np.asarray(image_layer.data).copy(),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        threshold_value, labels = auto_threshold_mask(
            payload["data"],
            polarity=payload["polarity"],
            threshold_value=payload.get("threshold_value"),
        )
        payload["threshold_value"] = threshold_value
        payload["labels"] = labels
        payload["stats"] = mask_statistics(labels)
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        preview_name = "__assistant_threshold_preview__"
        labels = payload["labels"]
        if preview_name in ctx.viewer.layers and isinstance(ctx.viewer.layers[preview_name], napari.layers.Labels):
            preview = ctx.viewer.layers[preview_name]
            preview.data = labels
            preview.scale = payload["scale"]
            preview.translate = payload["translate"]
        else:
            ctx.viewer.add_labels(labels, name=preview_name, scale=payload["scale"], translate=payload["translate"])
        stats = payload["stats"]
        return (
            f"Preview threshold updated for [{payload['layer_name']}] at {payload['threshold_value']:.6g} "
            f"({_threshold_mode_text(payload['polarity'])}). objects={stats['object_count']} fg={stats['foreground_pixels']} px."
        )


class ApplyThresholdTool:
    spec = ToolSpec(
        name="apply_threshold",
        display_name="Apply Threshold",
        category="segmentation",
        description="Apply an automatic threshold and create a labels layer.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="Optional image layer.", required=False),
            ParamSpec("polarity", "string", description="Threshold polarity.", default="auto", enum=("auto", "bright", "dim")),
            ParamSpec("threshold_value", "float", description="Optional manual threshold cutoff.", required=False),
        ),
        output_type="layer",
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "auto_threshold", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("layer_name"))
        if image_layer is None:
            return "No valid image layer available."
        polarity = normalize_polarity(args.get("polarity", "auto"))
        threshold_value = _optional_threshold_value(args.get("threshold_value"))
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": image_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{image_layer.name}_labels"),
                "polarity": polarity,
                "threshold_value": threshold_value,
                "data": np.asarray(image_layer.data).copy(),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        threshold_value, labels = auto_threshold_mask(
            payload["data"],
            polarity=payload["polarity"],
            threshold_value=payload.get("threshold_value"),
        )
        payload["threshold_value"] = threshold_value
        payload["labels"] = labels
        payload["stats"] = mask_statistics(labels)
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        ctx.viewer.add_labels(payload["labels"], name=payload["output_name"], scale=payload["scale"], translate=payload["translate"])
        stats = payload["stats"]
        return (
            f"Applied threshold to [{payload['layer_name']}] as [{payload['output_name']}] at {payload['threshold_value']:.6g} "
            f"({_threshold_mode_text(payload['polarity'])}). objects={stats['object_count']} fg={stats['foreground_pixels']} px."
        )


class PreviewThresholdBatchTool:
    spec = ToolSpec(
        name="preview_threshold_batch",
        display_name="Preview Threshold Batch",
        category="segmentation",
        description="Preview automatic thresholds for all image layers.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(ParamSpec("polarity", "string", description="Threshold polarity.", default="auto", enum=("auto", "bright", "dim")),),
        output_type="layer",
        supports_batch=True,
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "auto_threshold", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layers = find_all_image_layers(ctx.viewer)
        if not image_layers:
            return "No image layers available for batch thresholding."
        polarity = normalize_polarity(args.get("polarity", "auto"))
        items = []
        for layer in image_layers:
            items.append(
                {
                    "layer_name": layer.name,
                    "polarity": polarity,
                    "data": np.asarray(layer.data).copy(),
                    "scale": tuple(layer.scale),
                    "translate": tuple(layer.translate),
                }
            )
        return PreparedJob(tool_name=self.spec.name, kind=self.spec.name, mode="worker", payload={"kind": self.spec.name, "polarity": polarity, "items": items})

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        results = []
        for item in payload["items"]:
            threshold_value, labels = auto_threshold_mask(item["data"], polarity=item["polarity"])
            results.append({**item, "threshold_value": threshold_value, "labels": labels, "stats": mask_statistics(labels)})
        payload["items"] = results
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        lines = []
        for item in result.payload["items"]:
            preview_name = f"__assistant_threshold_preview__::{item['layer_name']}"
            if preview_name in ctx.viewer.layers and isinstance(ctx.viewer.layers[preview_name], napari.layers.Labels):
                preview = ctx.viewer.layers[preview_name]
                preview.data = item["labels"]
                preview.scale = item["scale"]
                preview.translate = item["translate"]
            else:
                ctx.viewer.add_labels(item["labels"], name=preview_name, scale=item["scale"], translate=item["translate"])
            stats = item["stats"]
            lines.append(
                f"[{item['layer_name']}] preview at {item['threshold_value']:.6g} "
                f"objects={stats['object_count']} fg={stats['foreground_pixels']} px."
            )
        return f"Updated preview masks for {len(result.payload['items'])} image layers ({_threshold_mode_text(result.payload['polarity'])}).\n" + "\n".join(lines)


class ApplyThresholdBatchTool:
    spec = ToolSpec(
        name="apply_threshold_batch",
        display_name="Apply Threshold Batch",
        category="segmentation",
        description="Apply automatic thresholds to all image layers.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(ParamSpec("polarity", "string", description="Threshold polarity.", default="auto", enum=("auto", "bright", "dim")),),
        output_type="layer",
        supports_batch=True,
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "auto_threshold", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layers = find_all_image_layers(ctx.viewer)
        if not image_layers:
            return "No image layers available for batch thresholding."
        polarity = normalize_polarity(args.get("polarity", "auto"))
        items = []
        for layer in image_layers:
            items.append(
                {
                    "layer_name": layer.name,
                    "output_name": next_output_name(ctx.viewer, f"{layer.name}_labels"),
                    "polarity": polarity,
                    "data": np.asarray(layer.data).copy(),
                    "scale": tuple(layer.scale),
                    "translate": tuple(layer.translate),
                }
            )
        return PreparedJob(tool_name=self.spec.name, kind=self.spec.name, mode="worker", payload={"kind": self.spec.name, "polarity": polarity, "items": items})

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        results = []
        for item in payload["items"]:
            threshold_value, labels = auto_threshold_mask(item["data"], polarity=item["polarity"])
            results.append({**item, "threshold_value": threshold_value, "labels": labels, "stats": mask_statistics(labels)})
        payload["items"] = results
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        lines = []
        for item in result.payload["items"]:
            ctx.viewer.add_labels(item["labels"], name=item["output_name"], scale=item["scale"], translate=item["translate"])
            stats = item["stats"]
            lines.append(
                f"[{item['layer_name']}] -> [{item['output_name']}] at {item['threshold_value']:.6g} "
                f"objects={stats['object_count']} fg={stats['foreground_pixels']} px."
            )
        return f"Applied threshold to {len(result.payload['items'])} image layers ({_threshold_mode_text(result.payload['polarity'])}).\n" + "\n".join(lines)


class MeasureMaskTool:
    spec = ToolSpec(
        name="measure_mask",
        display_name="Measure Mask",
        category="measurement",
        description="Summarize one labels layer as a mask.",
        execution_mode="immediate",
        supported_layer_types=("labels",),
        parameter_schema=(ParamSpec("layer_name", "string", description="Optional labels layer."),),
        output_type="message",
        ui_metadata={"panel_group": "Measurement"},
        provenance_metadata={"algorithm": "mask_summary", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        labels_layer = find_labels_layer(ctx.viewer, (arguments or {}).get("layer_name"))
        if labels_layer is None:
            return "No valid labels layer available."
        return mask_measurement_summary(labels_layer)

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class MeasureMaskBatchTool:
    spec = ToolSpec(
        name="measure_mask_batch",
        display_name="Measure Mask Batch",
        category="measurement",
        description="Summarize all labels layers as masks.",
        execution_mode="immediate",
        supported_layer_types=("labels",),
        output_type="message",
        supports_batch=True,
        ui_metadata={"panel_group": "Measurement"},
        provenance_metadata={"algorithm": "mask_summary", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        labels_layers = find_all_labels_layers(ctx.viewer)
        if not labels_layers:
            return "No labels layers available for batch measurement."
        return "\n".join(mask_measurement_summary(layer) for layer in labels_layers)

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class RunMaskOpTool:
    spec = ToolSpec(
        name="run_mask_op",
        display_name="Run Mask Operation",
        category="segmentation",
        description="Run a morphology or transform operation on a labels layer.",
        execution_mode="worker",
        supported_layer_types=("labels",),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="Optional labels layer."),
            ParamSpec("op", "string", description="Mask operation to run.", required=True),
            ParamSpec("radius", "int", description="Radius for morphology operations.", default=1),
            ParamSpec("min_size", "int", description="Minimum object size.", default=64),
        ),
        output_type="layer",
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "mask_operation", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        labels_layer = find_labels_layer(ctx.viewer, args.get("layer_name"))
        if labels_layer is None:
            return "No valid labels layer available."
        op_name = str(args.get("op", "")).strip().lower()
        if op_name not in TOOL_OPS:
            return f"Unsupported mask operation: {op_name}"
        safe_args = dict(args)
        safe_args["radius"] = normalize_int(args.get("radius", 1), default=1, minimum=1, maximum=20)
        safe_args["min_size"] = normalize_int(args.get("min_size", 64), default=64, minimum=1, maximum=10_000_000)
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": labels_layer.name,
                "snapshot_name": next_snapshot_name(ctx.viewer, labels_layer.name),
                "output_kind": TOOL_OP_OUTPUTS[op_name],
                "output_name": f"{labels_layer.name}_{op_name}",
                "op_name": op_name,
                "args": safe_args,
                "data": np.asarray(labels_layer.data).copy(),
                "scale": tuple(labels_layer.scale),
                "translate": tuple(labels_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        op = TOOL_OPS[payload["op_name"]]
        result = op(payload["data"], payload["args"])
        output_kind = payload.get("output_kind", "replace_labels")
        payload["result"] = result
        payload["stats"] = mask_statistics(result) if output_kind in {"replace_labels", "new_labels"} else intensity_statistics(result)
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        labels_layer = find_labels_layer(ctx.viewer, payload["layer_name"])
        if labels_layer is None:
            return f"Labels layer [{payload['layer_name']}] is no longer available."
        output_kind = payload.get("output_kind", "replace_labels")
        if output_kind == "replace_labels":
            snapshot_name = save_mask_snapshot(ctx.viewer, labels_layer)
            labels_layer.data = payload["result"]
            stats = payload["stats"]
            return (
                f"Saved snapshot [{snapshot_name}] and applied {payload['op_name']} to [{labels_layer.name}]. "
                f"objects={stats['object_count']} fg={stats['foreground_pixels']} px largest={stats['largest_object']} px."
            )
        if output_kind == "new_labels":
            output_name = next_output_name(ctx.viewer, payload.get("output_name") or f"{labels_layer.name}_{payload['op_name']}")
            ctx.viewer.add_labels(payload["result"], name=output_name, scale=payload["scale"], translate=payload["translate"])
            stats = payload["stats"]
            return (
                f"Created [{output_name}] from [{labels_layer.name}] using {payload['op_name']}. "
                f"objects={stats['object_count']} fg={stats['foreground_pixels']} px largest={stats['largest_object']} px."
            )
        output_name = next_output_name(ctx.viewer, payload.get("output_name") or f"{labels_layer.name}_{payload['op_name']}")
        ctx.viewer.add_image(payload["result"], name=output_name, scale=payload["scale"], translate=payload["translate"])
        stats = payload["stats"]
        return (
            f"Created [{output_name}] from [{labels_layer.name}] using {payload['op_name']}. "
            f"mean={stats['mean']:.6g} std={stats['std']:.6g} min={stats['min']:.6g} max={stats['max']:.6g}."
        )


class RunMaskOpBatchTool:
    spec = ToolSpec(
        name="run_mask_op_batch",
        display_name="Run Mask Operation Batch",
        category="segmentation",
        description="Run a morphology or transform operation on all labels layers.",
        execution_mode="worker",
        supported_layer_types=("labels",),
        parameter_schema=(
            ParamSpec("op", "string", description="Mask operation to run.", required=True),
            ParamSpec("radius", "int", description="Radius for morphology operations.", default=1),
            ParamSpec("min_size", "int", description="Minimum object size.", default=64),
        ),
        output_type="layer",
        supports_batch=True,
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "mask_operation", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        labels_layers = find_all_labels_layers(ctx.viewer)
        if not labels_layers:
            return "No labels layers available for batch mask operations."
        op_name = str(args.get("op", "")).strip().lower()
        if op_name not in TOOL_OPS:
            return f"Unsupported mask operation: {op_name}"
        safe_args = dict(args)
        safe_args["radius"] = normalize_int(args.get("radius", 1), default=1, minimum=1, maximum=20)
        safe_args["min_size"] = normalize_int(args.get("min_size", 64), default=64, minimum=1, maximum=10_000_000)
        items = []
        for layer in labels_layers:
            items.append(
                {
                    "layer_name": layer.name,
                    "snapshot_name": next_snapshot_name(ctx.viewer, layer.name),
                    "output_kind": TOOL_OP_OUTPUTS[op_name],
                    "output_name": f"{layer.name}_{op_name}",
                    "op_name": op_name,
                    "args": safe_args,
                    "data": np.asarray(layer.data).copy(),
                    "scale": tuple(layer.scale),
                    "translate": tuple(layer.translate),
                }
            )
        return PreparedJob(tool_name=self.spec.name, kind=self.spec.name, mode="worker", payload={"kind": self.spec.name, "op_name": op_name, "items": items})

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        results = []
        for item in payload["items"]:
            op = TOOL_OPS[item["op_name"]]
            result = op(item["data"], item["args"])
            output_kind = item.get("output_kind", "replace_labels")
            stats = mask_statistics(result) if output_kind in {"replace_labels", "new_labels"} else intensity_statistics(result)
            results.append({**item, "result": result, "stats": stats})
        payload["items"] = results
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        lines = []
        applied = 0
        for item in result.payload["items"]:
            labels_layer = find_labels_layer(ctx.viewer, item["layer_name"])
            if labels_layer is None:
                lines.append(f"[{item['layer_name']}] skipped because the layer is no longer available.")
                continue
            output_kind = item.get("output_kind", "replace_labels")
            stats = item["stats"]
            applied += 1
            if output_kind == "replace_labels":
                snapshot_name = save_mask_snapshot(ctx.viewer, labels_layer)
                labels_layer.data = item["result"]
                lines.append(
                    f"[{labels_layer.name}] snapshot [{snapshot_name}] {item['op_name']} "
                    f"objects={stats['object_count']} fg={stats['foreground_pixels']} px largest={stats['largest_object']} px."
                )
            elif output_kind == "new_labels":
                output_name = next_output_name(ctx.viewer, item.get("output_name") or f"{labels_layer.name}_{item['op_name']}")
                ctx.viewer.add_labels(item["result"], name=output_name, scale=item["scale"], translate=item["translate"])
                lines.append(
                    f"[{labels_layer.name}] -> [{output_name}] {item['op_name']} "
                    f"objects={stats['object_count']} fg={stats['foreground_pixels']} px largest={stats['largest_object']} px."
                )
            else:
                output_name = next_output_name(ctx.viewer, item.get("output_name") or f"{labels_layer.name}_{item['op_name']}")
                ctx.viewer.add_image(item["result"], name=output_name, scale=item["scale"], translate=item["translate"])
                lines.append(
                    f"[{labels_layer.name}] -> [{output_name}] {item['op_name']} "
                    f"mean={stats['mean']:.6g} std={stats['std']:.6g} min={stats['min']:.6g} max={stats['max']:.6g}."
                )
        return f"Applied {result.payload['op_name']} to {applied} labels layers.\n" + "\n".join(lines)


class MeasureShapesROIStatsTool:
    spec = ToolSpec(
        name="measure_shapes_roi_stats",
        display_name="Measure Shapes ROI Stats",
        category="roi",
        description="Summarize shape geometry and sampled image intensity for a Shapes ROI layer.",
        execution_mode="immediate",
        supported_layer_types=("shapes", "image"),
        parameter_schema=(
            ParamSpec("roi_layer", "string", description="Shapes ROI layer.", required=False),
            ParamSpec("image_layer", "string", description="Optional image layer to sample.", required=False),
        ),
        output_type="message",
        ui_metadata={"panel_group": "ROI"},
        provenance_metadata={"algorithm": "roi_shape_summary", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        roi_layer = find_shapes_layer(ctx.viewer, args.get("roi_layer"))
        if roi_layer is None:
            return "No valid Shapes ROI layer available. Select or name a Shapes layer."
        total_shapes = len(getattr(roi_layer, "data", []))
        if total_shapes == 0:
            return f"Shapes layer [{roi_layer.name}] contains no shapes."
        selected_indices = sorted(int(i) for i in getattr(roi_layer, "selected_data", set()) if 0 <= int(i) < total_shapes)
        indices = selected_indices or list(range(total_shapes))
        image_layer = find_image_layer(ctx.viewer, args.get("image_layer"))
        image_plane = None
        if image_layer is not None:
            try:
                image_plane = _selected_image_plane(ctx.viewer, image_layer)
            except Exception:
                image_plane = None
        masks = roi_layer.to_masks(mask_shape=None if image_plane is None else image_plane.shape)
        lines = [
            f'Shapes ROI stats for "{roi_layer.name}"',
            f"Measured {len(indices)} shape(s)" + (" (selected only)." if selected_indices else "."),
        ]
        if image_layer is not None:
            lines.append(f'Intensity source: "{image_layer.name}"')
        lines.append("")
        shape_types = list(getattr(roi_layer, "shape_type", []))
        for index in indices:
            verts = np.asarray(roi_layer.data[index], dtype=float)
            mins = np.min(verts, axis=0) if verts.ndim == 2 and len(verts) else None
            maxs = np.max(verts, axis=0) if verts.ndim == 2 and len(verts) else None
            centroid = np.mean(verts, axis=0) if verts.ndim == 2 and len(verts) else None
            shape_type = str(shape_types[index] if index < len(shape_types) else "polygon")
            line = [f"- {_shape_label(roi_layer, index)}"]
            line.append(f"type={shape_type}")
            line.append(f"centroid={_format_centroid(centroid)}")
            line.append(f"bbox={_format_bbox_yx(mins, maxs)}")
            area = None
            try:
                mask = np.asarray(masks[index], dtype=bool)
                area = int(np.sum(mask))
            except Exception:
                mask = None
            if area is not None:
                line.append(f"area={area} px^2")
            if image_plane is not None and mask is not None and mask.shape == image_plane.shape:
                values = image_plane[mask]
                values = values[np.isfinite(values)]
                line.append(_format_roi_intensity_stats(values))
            lines.append(" | ".join(line))
        lines.extend(
            [
                "",
                "If you want an interactive table with renameable ROIs, histogram, percent view, chat insertion, and CSV export, open the ROI Intensity Analysis widget.",
            ]
        )
        return "\n".join(lines)

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class OpenIntensityMetricsTableTool:
    spec = ToolSpec(
        name="open_intensity_metrics_table",
        display_name="Open ROI Intensity Analysis",
        category="widget",
        description="Open the interactive ROI Intensity Analysis widget.",
        execution_mode="immediate",
        supported_layer_types=("image",),
        parameter_schema=(ParamSpec("layer_name", "string", description="Optional image layer."),),
        output_type="message",
        ui_metadata={"panel_group": "Widget"},
        provenance_metadata={"algorithm": "widget_launch", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        image_layer = find_image_layer(ctx.viewer, (arguments or {}).get("layer_name"))
        if image_layer is None:
            image_layers = find_all_image_layers(ctx.viewer)
            image_layer = image_layers[-1] if image_layers else None
        if image_layer is None:
            return "No valid image layer available for ROI intensity measurement."
        open_intensity_metrics_widget(ctx.viewer)
        return (
            f"Opened ROI Intensity Analysis for [{image_layer.name}].\n\n"
            "Use the Shapes ROI layer to draw or edit one or more regions, then the widget updates the histogram and table live.\n"
            "Absolute view shows raw measurements such as pixels, mean, std, median, min, max, and sum.\n"
            "Percent or Normalized view rescales those values relative to the image intensity range or image totals so different ROIs are easier to compare.\n"
            "You can rename ROIs, copy rows to chat, and export CSV from the same widget."
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class OpenGroupComparisonWidgetTool:
    spec = ToolSpec(
        name="open_group_comparison_widget",
        display_name="Open Group Comparison Widget",
        category="widget",
        description="Open the interactive group comparison widget.",
        execution_mode="immediate",
        output_type="message",
        ui_metadata={"panel_group": "Widget"},
        provenance_metadata={"algorithm": "widget_launch", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        open_group_comparison_widget(ctx.viewer)
        return (
            "Opened Group Comparison Stats.\n\n"
            "Set the two group prefixes, choose Whole Image or ROI-based analysis, then run the comparison.\n"
            "The widget shows the per-sample table, descriptive statistics, assumption checks, and the selected test.\n"
            "For the plot, choose Box + Points for a more scientific distribution view or Bar + Error for a compact summary view.\n"
            "You can also insert the stats summary into chat or export the dataset to CSV."
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class SummarizeIntensityTool:
    spec = ToolSpec(
        name="summarize_intensity",
        display_name="Summarize Intensity",
        category="measurement",
        description="Summarize image intensities.",
        execution_mode="immediate",
        supported_layer_types=("image",),
        parameter_schema=(ParamSpec("layer_name", "string", description="Optional image layer."),),
        output_type="message",
        ui_metadata={"panel_group": "Measurement"},
        provenance_metadata={"algorithm": "intensity_summary", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        image_layer = find_image_layer(ctx.viewer, (arguments or {}).get("layer_name"))
        if image_layer is None:
            return "No valid image layer available for intensity summary."
        stats = intensity_statistics(np.asarray(image_layer.data))
        return _format_intensity_summary(image_layer.name, stats)

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class PlotHistogramTool:
    spec = ToolSpec(
        name="plot_histogram",
        display_name="Plot Histogram",
        category="measurement",
        description="Plot an intensity histogram for an image layer.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="Optional image layer."),
            ParamSpec("bins", "int", description="Histogram bins.", default=64),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Measurement"},
        provenance_metadata={"algorithm": "histogram", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("layer_name"))
        if image_layer is None:
            return "No valid image layer available for histogram plotting."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": image_layer.name,
                "bins": normalize_int(args.get("bins", 64), default=64, minimum=2, maximum=512),
                "data": np.asarray(image_layer.data).copy(),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        payload["histogram"] = intensity_histogram(payload["data"], bins=payload["bins"])
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        histogram = result.payload["histogram"]
        _show_histogram_popup(result.payload["layer_name"], histogram)
        stats = histogram["stats"]
        return (
            f"Opened histogram for [{result.payload['layer_name']}] with {histogram['bins']} bins. "
            f"n={stats['count']} mean={stats['mean']:.6g} std={stats['std']:.6g}."
        )


class CompareImageLayersTTestTool:
    spec = ToolSpec(
        name="compare_image_layers_ttest",
        display_name="Compare Image Layers T-Test",
        category="statistics",
        description="Compare the pixel populations of two image layers with a t-test.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("layer_name_a", "string", description="First image layer.", required=False),
            ParamSpec("layer_name_b", "string", description="Second image layer.", required=False),
            ParamSpec("equal_var", "boolean", description="Use equal-variance Student t-test.", default=True),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Statistics"},
        provenance_metadata={"algorithm": "ttest", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        image_layers = find_all_image_layers(ctx.viewer)
        if len(image_layers) < 2:
            return "At least 2 image layers are required for a t-test comparison."
        pair = _resolve_image_layer_pair(ctx.viewer, arguments or {})
        if pair is None:
            return "Could not resolve 2 image layers for comparison. Specify layer_name_a and layer_name_b."
        layer_a, layer_b = pair
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name_a": layer_a.name,
                "layer_name_b": layer_b.name,
                "equal_var": bool((arguments or {}).get("equal_var", True)),
                "data_a": np.asarray(layer_a.data).copy(),
                "data_b": np.asarray(layer_b.data).copy(),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        payload["comparison"] = compare_intensity_populations(
            payload["data_a"],
            payload["data_b"],
            equal_var=payload["equal_var"],
        )
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        comparison = result.payload["comparison"]
        return (
            f"{comparison['test_name']} for [{result.payload['layer_name_a']}] vs [{result.payload['layer_name_b']}]: "
            f"t={comparison['statistic']:.6g} p={comparison['pvalue']:.6g} "
            f"mean_a={comparison['mean_a']:.6g} mean_b={comparison['mean_b']:.6g} "
            f"delta={comparison['delta_mean']:.6g} n_a={comparison['count_a']} n_b={comparison['count_b']}."
        )


def legacy_surface_tools():
    return [
        ListLayersTool(),
        InspectSelectedLayerTool(),
        InspectLayerTool(),
        OpenND2ConverterTool(),
        OpenSpectralViewerTool(),
        OpenSpectralAnalysisTool(),
        ApplyClaheBatchTool(),
        PreviewThresholdTool(),
        ApplyThresholdTool(),
        PreviewThresholdBatchTool(),
        ApplyThresholdBatchTool(),
        MeasureMaskTool(),
        MeasureMaskBatchTool(),
        RunMaskOpTool(),
        RunMaskOpBatchTool(),
        MeasureShapesROIStatsTool(),
        OpenIntensityMetricsTableTool(),
        OpenGroupComparisonWidgetTool(),
        SummarizeIntensityTool(),
        PlotHistogramTool(),
        CompareImageLayersTTestTool(),
    ]
