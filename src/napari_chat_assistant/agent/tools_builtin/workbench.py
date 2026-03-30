from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as sk_label
from skimage.measure import regionprops_table

from napari_chat_assistant.agent.context import find_any_layer, find_image_layer, find_labels_layer
from napari_chat_assistant.agent.image_ops import fill_holes, keep_largest_component, remove_small_components
from napari_chat_assistant.agent.tool_types import ParamSpec, PreparedJob, ToolContext, ToolResult, ToolSpec
from napari_chat_assistant.agent.tools import next_output_name, normalize_float, normalize_int


class PlaceholderTool:
    def __init__(self, spec: ToolSpec):
        self.spec = spec

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        return (
            f"Tool [{self.spec.name}] is registered in the workbench registry but not implemented yet. "
            f"Category={self.spec.category}."
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message or f"Tool [{self.spec.name}] is not implemented yet."


class GaussianDenoiseTool:
    spec = ToolSpec(
        name="gaussian_denoise",
        display_name="Gaussian Denoise",
        category="preprocessing",
        description="Apply Gaussian smoothing to a grayscale image layer.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="Optional image layer name."),
            ParamSpec("sigma", "float_or_list", description="Gaussian sigma.", default=1.0, minimum=0.0),
            ParamSpec("preserve_range", "bool", description="Preserve original intensity range.", default=True),
        ),
        output_type="image_layer",
        ui_metadata={"panel_group": "Preprocessing"},
        provenance_metadata={"algorithm": "gaussian_filter", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("layer_name"))
        if image_layer is None:
            return "No valid image layer available for Gaussian denoising."
        if getattr(image_layer, "rgb", False):
            return "Gaussian denoising currently supports grayscale image layers, not RGB layers."
        sigma = args.get("sigma", 1.0)
        if isinstance(sigma, (list, tuple)):
            sigma_value = tuple(normalize_float(value, default=1.0, minimum=0.0, maximum=100.0) for value in sigma)
        else:
            sigma_value = normalize_float(sigma, default=1.0, minimum=0.0, maximum=100.0)
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": image_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{image_layer.name}_gaussian"),
                "sigma": sigma_value,
                "preserve_range": bool(args.get("preserve_range", True)),
                "data": np.asarray(image_layer.data).copy(),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        filtered = ndi.gaussian_filter(np.asarray(payload["data"], dtype=np.float32), sigma=payload["sigma"])
        payload["result"] = filtered.astype(np.float32, copy=False)
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "parameters": {"sigma": payload["sigma"], "preserve_range": payload["preserve_range"]},
                "input_layer": payload["layer_name"],
                "output_layer": payload["output_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        ctx.viewer.add_image(
            payload["result"],
            name=payload["output_name"],
            scale=payload["scale"],
            translate=payload["translate"],
        )
        return f"Applied Gaussian denoising to [{payload['layer_name']}] as [{payload['output_name']}] with sigma={payload['sigma']}."


class RemoveSmallObjectsTool:
    spec = ToolSpec(
        name="remove_small_objects",
        display_name="Remove Small Objects",
        category="segmentation_cleanup",
        description="Remove connected foreground components smaller than a minimum size.",
        execution_mode="worker",
        supported_layer_types=("labels",),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="Optional labels layer name."),
            ParamSpec("min_size", "int", description="Minimum object size to keep.", default=64, minimum=1),
        ),
        output_type="labels_layer",
        ui_metadata={"panel_group": "Segmentation Cleanup"},
        provenance_metadata={"algorithm": "remove_small_objects", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        labels_layer = find_labels_layer(ctx.viewer, args.get("layer_name"))
        if labels_layer is None:
            return "No valid labels layer available for removing small objects."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": labels_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{labels_layer.name}_clean"),
                "min_size": normalize_int(args.get("min_size", 64), default=64, minimum=1, maximum=10_000_000),
                "data": np.asarray(labels_layer.data).copy(),
                "scale": tuple(labels_layer.scale),
                "translate": tuple(labels_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        payload["result"] = remove_small_components(payload["data"], min_size=payload["min_size"])
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "parameters": {"min_size": payload["min_size"]},
                "input_layer": payload["layer_name"],
                "output_layer": payload["output_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        ctx.viewer.add_labels(
            payload["result"],
            name=payload["output_name"],
            scale=payload["scale"],
            translate=payload["translate"],
        )
        return f"Removed small objects from [{payload['layer_name']}] into [{payload['output_name']}] with min_size={payload['min_size']}."


class FillMaskHolesTool:
    spec = ToolSpec(
        name="fill_mask_holes",
        display_name="Fill Mask Holes",
        category="segmentation_cleanup",
        description="Fill internal holes in a binary mask or labels foreground.",
        execution_mode="worker",
        supported_layer_types=("labels",),
        parameter_schema=(ParamSpec("layer_name", "string", description="Optional labels layer name."),),
        output_type="labels_layer",
        ui_metadata={"panel_group": "Segmentation Cleanup"},
        provenance_metadata={"algorithm": "binary_fill_holes", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        labels_layer = find_labels_layer(ctx.viewer, (arguments or {}).get("layer_name"))
        if labels_layer is None:
            return "No valid labels layer available for filling mask holes."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": labels_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{labels_layer.name}_filled"),
                "data": np.asarray(labels_layer.data).copy(),
                "scale": tuple(labels_layer.scale),
                "translate": tuple(labels_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        payload["result"] = fill_holes(payload["data"])
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "input_layer": payload["layer_name"],
                "output_layer": payload["output_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        ctx.viewer.add_labels(
            payload["result"],
            name=payload["output_name"],
            scale=payload["scale"],
            translate=payload["translate"],
        )
        return f"Filled mask holes in [{payload['layer_name']}] into [{payload['output_name']}]."


class ProjectMaxIntensityTool:
    spec = ToolSpec(
        name="project_max_intensity",
        display_name="Max Intensity Projection",
        category="visualization",
        description="Create a max-intensity projection from a 3D image layer.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="3D image layer name."),
            ParamSpec("axis", "int", description="Projection axis.", default=0, minimum=0),
        ),
        output_type="image_layer",
        ui_metadata={"panel_group": "Visualization"},
        provenance_metadata={"algorithm": "max_projection", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("layer_name"))
        if image_layer is None:
            return "No valid image layer available for max-intensity projection."
        if getattr(image_layer, "rgb", False):
            return "Max-intensity projection currently supports grayscale image layers, not RGB layers."
        data = np.asarray(image_layer.data)
        if data.ndim < 3:
            return f"Max-intensity projection requires at least 3 dimensions. Got ndim={data.ndim}."
        axis = normalize_int(args.get("axis", 0), default=0, minimum=0, maximum=max(0, data.ndim - 1))
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": image_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{image_layer.name}_mip"),
                "axis": axis,
                "data": data.copy(),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        payload["result"] = np.max(payload["data"], axis=payload["axis"])
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "parameters": {"axis": payload["axis"]},
                "input_layer": payload["layer_name"],
                "output_layer": payload["output_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        axis = int(payload["axis"])
        scale = tuple(value for index, value in enumerate(payload["scale"]) if index != axis)
        translate = tuple(value for index, value in enumerate(payload["translate"]) if index != axis)
        ctx.viewer.add_image(
            payload["result"],
            name=payload["output_name"],
            scale=scale,
            translate=translate,
        )
        return f"Created max-intensity projection for [{payload['layer_name']}] as [{payload['output_name']}] along axis={axis}."


class KeepLargestComponentTool:
    spec = ToolSpec(
        name="keep_largest_component",
        display_name="Keep Largest Component",
        category="segmentation_cleanup",
        description="Keep only the largest connected foreground object.",
        execution_mode="worker",
        supported_layer_types=("labels",),
        parameter_schema=(ParamSpec("layer_name", "string", description="Optional labels layer name."),),
        output_type="labels_layer",
        ui_metadata={"panel_group": "Segmentation Cleanup"},
        provenance_metadata={"algorithm": "largest_component", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        labels_layer = find_labels_layer(ctx.viewer, (arguments or {}).get("layer_name"))
        if labels_layer is None:
            return "No valid labels layer available for keeping the largest component."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": labels_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{labels_layer.name}_largest"),
                "data": np.asarray(labels_layer.data).copy(),
                "scale": tuple(labels_layer.scale),
                "translate": tuple(labels_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        payload["result"] = keep_largest_component(payload["data"])
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "input_layer": payload["layer_name"],
                "output_layer": payload["output_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        ctx.viewer.add_labels(
            payload["result"],
            name=payload["output_name"],
            scale=payload["scale"],
            translate=payload["translate"],
        )
        return f"Kept the largest connected component from [{payload['layer_name']}] in [{payload['output_name']}]."


class LabelConnectedComponentsTool:
    spec = ToolSpec(
        name="label_connected_components",
        display_name="Label Connected Components",
        category="segmentation",
        description="Convert a binary mask into connected-component instance labels.",
        execution_mode="worker",
        supported_layer_types=("labels", "image"),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="Optional source layer name."),
            ParamSpec("connectivity", "int", description="Neighborhood connectivity.", default=1, minimum=1, maximum=3),
        ),
        output_type="labels_layer",
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "connected_components", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        layer = find_any_layer(ctx.viewer, args.get("layer_name"))
        if layer is None:
            return "No valid source layer available for connected-component labeling."
        data = np.asarray(getattr(layer, "data", None))
        if data.size == 0:
            return "Connected-component labeling requires a non-empty image or labels layer."
        connectivity = normalize_int(args.get("connectivity", 1), default=1, minimum=1, maximum=min(3, data.ndim))
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": layer.name,
                "output_name": next_output_name(ctx.viewer, f"{layer.name}_instances"),
                "connectivity": connectivity,
                "data": data.copy(),
                "scale": tuple(getattr(layer, "scale", ())),
                "translate": tuple(getattr(layer, "translate", ())),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        binary = np.asarray(payload["data"]) > 0
        payload["result"] = sk_label(binary, connectivity=payload["connectivity"]).astype(np.int32, copy=False)
        payload["object_count"] = int(np.max(payload["result"])) if payload["result"].size else 0
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "parameters": {"connectivity": payload["connectivity"]},
                "input_layer": payload["layer_name"],
                "output_layer": payload["output_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        ctx.viewer.add_labels(
            payload["result"],
            name=payload["output_name"],
            scale=payload["scale"],
            translate=payload["translate"],
        )
        return (
            f"Labeled connected components from [{payload['layer_name']}] into [{payload['output_name']}]. "
            f"objects={payload['object_count']}."
        )


class MeasureLabelsTableTool:
    spec = ToolSpec(
        name="measure_labels_table",
        display_name="Measure Labels Table",
        category="measurement",
        description="Measure labeled objects and return a structured per-object table.",
        execution_mode="worker",
        supported_layer_types=("labels",),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="Labels layer to measure."),
            ParamSpec("intensity_layer", "string", description="Optional paired intensity image layer."),
            ParamSpec(
                "properties",
                "string_list",
                description="Requested region properties.",
                default=("label", "area", "centroid", "bbox", "mean_intensity"),
            ),
        ),
        output_type="table",
        ui_metadata={"panel_group": "Measurement"},
        provenance_metadata={"algorithm": "regionprops_table", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        labels_layer = find_labels_layer(ctx.viewer, args.get("layer_name"))
        if labels_layer is None:
            return "No valid labels layer available for measurement."
        intensity_layer = None
        intensity_name = str(args.get("intensity_layer") or "").strip()
        if intensity_name:
            intensity_layer = find_image_layer(ctx.viewer, intensity_name)
            if intensity_layer is None:
                return f"No valid intensity image layer found with name [{intensity_name}]."
        properties = args.get("properties", ("label", "area", "centroid", "bbox", "mean_intensity"))
        if isinstance(properties, str):
            properties = [part.strip() for part in properties.split(",") if part.strip()]
        property_names = tuple(str(value).strip() for value in properties if str(value).strip())
        if not property_names:
            property_names = ("label", "area", "centroid", "bbox")
        if intensity_layer is None:
            property_names = tuple(name for name in property_names if name != "mean_intensity") or ("label", "area")
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": labels_layer.name,
                "intensity_layer_name": None if intensity_layer is None else intensity_layer.name,
                "properties": property_names,
                "labels_data": np.asarray(labels_layer.data).copy(),
                "intensity_data": None if intensity_layer is None else np.asarray(intensity_layer.data).copy(),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        table = regionprops_table(
            np.asarray(payload["labels_data"]),
            intensity_image=payload["intensity_data"],
            properties=payload["properties"],
        )
        rows: list[dict[str, object]] = []
        columns = list(table.keys())
        row_count = len(table[columns[0]]) if columns else 0
        for row_index in range(row_count):
            row: dict[str, object] = {}
            for column in columns:
                value = table[column][row_index]
                if isinstance(value, np.generic):
                    value = value.item()
                row[column] = value
            rows.append(row)
        payload["rows"] = rows
        payload["row_count"] = row_count
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "parameters": {"properties": payload["properties"]},
                "input_layer": payload["layer_name"],
                "intensity_layer": payload["intensity_layer_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        if not payload["rows"]:
            return f"No labeled objects found in [{payload['layer_name']}]."
        columns = list(payload["rows"][0].keys())
        preview_lines = []
        for row in payload["rows"][:5]:
            preview_lines.append(", ".join(f"{column}={row[column]}" for column in columns))
        return (
            f"Measured {payload['row_count']} labeled object(s) in [{payload['layer_name']}].\n"
            + "\n".join(preview_lines)
        )


class CropToLayerBBoxTool:
    spec = ToolSpec(
        name="crop_to_layer_bbox",
        display_name="Crop To Layer Bounding Box",
        category="roi",
        description="Crop a source layer to the foreground bounding box of a reference layer.",
        execution_mode="worker",
        supported_layer_types=("image", "labels"),
        parameter_schema=(
            ParamSpec("source_layer", "string", description="Layer to crop.", required=True),
            ParamSpec("reference_layer", "string", description="Layer that defines the bounding box.", required=True),
            ParamSpec("padding", "int_or_list", description="Padding around the bounding box.", default=0, minimum=0),
        ),
        output_type="layer",
        ui_metadata={"panel_group": "ROI"},
        provenance_metadata={"algorithm": "bbox_crop", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        source_name = str(args.get("source_layer") or "").strip()
        reference_name = str(args.get("reference_layer") or "").strip()
        if not source_name or not reference_name:
            return "Crop to layer bounding box requires source_layer and reference_layer."
        source_layer = find_any_layer(ctx.viewer, source_name)
        reference_layer = find_any_layer(ctx.viewer, reference_name)
        if source_layer is None or reference_layer is None:
            return "Could not resolve source_layer and reference_layer for bbox crop."
        source_data = np.asarray(getattr(source_layer, "data", None))
        reference_data = np.asarray(getattr(reference_layer, "data", None))
        if source_data.size == 0 or reference_data.size == 0:
            return "Crop to layer bounding box requires non-empty source and reference layers."
        padding = args.get("padding", 0)
        if isinstance(padding, (list, tuple)):
            padding_values = tuple(normalize_int(value, default=0, minimum=0, maximum=100000) for value in padding)
        else:
            pad = normalize_int(padding, default=0, minimum=0, maximum=100000)
            padding_values = tuple([pad] * reference_data.ndim)
        if len(padding_values) < reference_data.ndim:
            padding_values = tuple((list(padding_values) + [padding_values[-1]] * reference_data.ndim)[: reference_data.ndim])
        elif len(padding_values) > reference_data.ndim:
            padding_values = tuple(padding_values[: reference_data.ndim])
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "source_layer_name": source_layer.name,
                "reference_layer_name": reference_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{source_layer.name}_crop"),
                "source_is_labels": source_layer.__class__.__name__ == "Labels",
                "source_data": source_data.copy(),
                "reference_data": reference_data.copy(),
                "padding": padding_values,
                "scale": tuple(getattr(source_layer, "scale", ())),
                "translate": tuple(getattr(source_layer, "translate", ())),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        reference_binary = np.asarray(payload["reference_data"]) > 0
        if not np.any(reference_binary):
            raise ValueError("Reference layer has no foreground for bounding-box crop.")
        coords = np.argwhere(reference_binary)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        slices = []
        new_translate = []
        source_shape = np.asarray(payload["source_data"]).shape
        for axis, (start, stop, pad, scale_value, translate_value) in enumerate(
            zip(mins, maxs, payload["padding"], payload["scale"], payload["translate"])
        ):
            lo = max(0, int(start) - int(pad))
            hi = min(int(source_shape[axis]), int(stop) + int(pad))
            slices.append(slice(lo, hi))
            new_translate.append(float(translate_value) + float(scale_value) * lo)
        payload["bbox_slices"] = tuple(slices)
        payload["result"] = np.asarray(payload["source_data"])[tuple(slices)].copy()
        payload["cropped_translate"] = tuple(new_translate)
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "input_layer": payload["source_layer_name"],
                "reference_layer": payload["reference_layer_name"],
                "output_layer": payload["output_name"],
                "padding": payload["padding"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        if payload["source_is_labels"]:
            ctx.viewer.add_labels(
                payload["result"],
                name=payload["output_name"],
                scale=payload["scale"],
                translate=payload["cropped_translate"],
            )
        else:
            ctx.viewer.add_image(
                payload["result"],
                name=payload["output_name"],
                scale=payload["scale"],
                translate=payload["cropped_translate"],
            )
        return (
            f"Cropped [{payload['source_layer_name']}] to the bounding box of [{payload['reference_layer_name']}] "
            f"as [{payload['output_name']}]."
        )


def workbench_scaffold_tools():
    return [
        GaussianDenoiseTool(),
        RemoveSmallObjectsTool(),
        FillMaskHolesTool(),
        KeepLargestComponentTool(),
        LabelConnectedComponentsTool(),
        MeasureLabelsTableTool(),
        ProjectMaxIntensityTool(),
        CropToLayerBBoxTool(),
        PlaceholderTool(
            ToolSpec(
                name="recommend_next_step",
                display_name="Recommend Next Step",
                category="workflow",
                description="Recommend the next analysis action using dataset profile and current workflow state.",
                execution_mode="immediate",
                supported_layer_types=("image", "labels", "points", "shapes"),
                parameter_schema=(
                    ParamSpec("layer_name", "string", description="Optional layer to focus on."),
                    ParamSpec("goal", "string", description="Optional analysis goal."),
                ),
                output_type="message",
                ui_metadata={"panel_group": "Workflow"},
                provenance_metadata={"algorithm": "rule_based_recommendation", "deterministic": True},
            )
        ),
        PlaceholderTool(
            ToolSpec(
                name="record_workflow_step",
                display_name="Record Workflow Step",
                category="workflow",
                description="Record a tool invocation and bindings as a reusable workflow step.",
                execution_mode="immediate",
                supported_layer_types=(),
                parameter_schema=(
                    ParamSpec("tool_name", "string", description="Tool to record.", required=True),
                    ParamSpec("arguments", "object", description="Tool arguments.", required=True),
                    ParamSpec("input_bindings", "object", description="Workflow input bindings.", default={}),
                ),
                output_type="workflow_step",
                ui_metadata={"panel_group": "Workflow"},
                provenance_metadata={"algorithm": "workflow_recording", "deterministic": True},
            )
        ),
    ]
