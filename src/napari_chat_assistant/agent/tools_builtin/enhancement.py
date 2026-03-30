from __future__ import annotations

import numpy as np

from napari_chat_assistant.agent.context import find_image_layer
from napari_chat_assistant.agent.image_ops import apply_clahe
from napari_chat_assistant.agent.tool_types import ParamSpec, PreparedJob, ToolContext, ToolResult, ToolSpec
from napari_chat_assistant.agent.tools import next_output_name, normalize_float, normalize_int, normalize_kernel_size


class ApplyClaheTool:
    spec = ToolSpec(
        name="apply_clahe",
        display_name="Apply CLAHE",
        category="enhancement",
        description="Apply contrast-limited adaptive histogram equalization to a grayscale image layer.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="Optional image layer name."),
            ParamSpec("kernel_size", "int_or_list", description="Kernel size for local histogram windows.", default=32),
            ParamSpec("clip_limit", "float", description="CLAHE clip limit.", default=0.01, minimum=1e-6, maximum=10.0),
            ParamSpec("nbins", "int", description="Histogram bin count.", default=256, minimum=2, maximum=65536),
        ),
        output_type="image_layer",
        ui_metadata={"panel_group": "Enhancement"},
        provenance_metadata={"algorithm": "clahe", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("layer_name"))
        if image_layer is None:
            return "No valid image layer available for CLAHE."
        if getattr(image_layer, "rgb", False):
            return "CLAHE currently supports grayscale 2D/3D image layers, not RGB layers."
        layer_data = np.asarray(image_layer.data)
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "layer_name": image_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{image_layer.name}_clahe"),
                "kernel_size": normalize_kernel_size(args.get("kernel_size", 32), ndim=layer_data.ndim),
                "clip_limit": normalize_float(args.get("clip_limit", 0.01), default=0.01, minimum=1e-6, maximum=10.0),
                "nbins": normalize_int(args.get("nbins", 256), default=256, minimum=2, maximum=65536),
                "data": layer_data.copy(),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
                "input_profile": ctx.selected_layer_profile,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        payload["result"] = apply_clahe(
            payload["data"],
            kernel_size=payload["kernel_size"],
            clip_limit=payload["clip_limit"],
            nbins=payload["nbins"],
        )
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "parameters": {
                    "kernel_size": payload["kernel_size"],
                    "clip_limit": payload["clip_limit"],
                    "nbins": payload["nbins"],
                },
                "input_layer": payload["layer_name"],
                "output_layer": payload["output_name"],
                "input_profile": payload.get("input_profile"),
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
        return (
            f"Applied CLAHE to [{payload['layer_name']}] as [{payload['output_name']}]. "
            f"kernel_size={payload['kernel_size']} clip_limit={payload['clip_limit']:.6g} nbins={payload['nbins']}."
        )
