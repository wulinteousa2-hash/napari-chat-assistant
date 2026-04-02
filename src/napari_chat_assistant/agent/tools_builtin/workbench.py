from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import stats
import napari
from napari.layers import Labels, Shapes, Image, Points
from skimage.measure import label as sk_label
from skimage.measure import regionprops, regionprops_table
from skimage.draw import polygon
from skimage.filters import threshold_otsu
from skimage.morphology import disk, remove_small_objects
from skimage.segmentation import clear_border

from napari_chat_assistant.agent.context import find_any_layer, find_image_layer, find_labels_layer
from napari_chat_assistant.agent.image_ops import fill_holes, keep_largest_component, remove_small_components
from napari_chat_assistant.agent.sam2_backend import (
    get_sam2_backend_status,
    propagate_volume_from_points,
    segment_image_from_box,
    segment_image_from_points,
)
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


def _resolve_roi_layer(viewer, roi_layer_name: object | None = None):
    name = str(roi_layer_name or "").strip()
    if name:
        layer = find_any_layer(viewer, name)
        if isinstance(layer, (Labels, Shapes)):
            return layer
        return None
    selected = viewer.layers.selection.active if viewer is not None else None
    if isinstance(selected, (Labels, Shapes)):
        return selected
    for layer in viewer.layers if viewer is not None else []:
        if isinstance(layer, (Labels, Shapes)):
            return layer
    return None


def _labels_bbox(binary: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(binary)
    if coords.size == 0:
        return tuple()
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return tuple((int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def _shape_indices(layer: Shapes) -> list[int]:
    selected = getattr(layer, "selected_data", set()) or set()
    if selected:
        return sorted(int(i) for i in selected)
    return list(range(int(getattr(layer, "nshapes", len(getattr(layer, "data", []))))))


def _rasterize_shapes_roi(layer: Shapes, image_shape: tuple[int, ...]) -> np.ndarray:
    if len(image_shape) != 2:
        raise ValueError("Shapes ROI extraction currently supports 2D image layers only.")
    mask = np.zeros(image_shape, dtype=bool)
    indices = _shape_indices(layer)
    shape_types = list(getattr(layer, "shape_type", []))
    for index in indices:
        if index >= len(layer.data):
            continue
        vertices = np.asarray(layer.data[index], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] < 2 or len(vertices) < 3:
            continue
        shape_type = shape_types[index] if index < len(shape_types) else "polygon"
        if str(shape_type) in {"line", "path"}:
            continue
        rr, cc = polygon(vertices[:, 0], vertices[:, 1], shape=image_shape)
        mask[rr, cc] = True
    return mask


def _shape_vertices_yx(vertices: object) -> np.ndarray | None:
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[0] < 2:
        return None
    if verts.shape[1] < 2:
        return None
    if verts.shape[1] > 2:
        return verts[:, -2:]
    return verts


def _polygon_area_yx(vertices_yx: np.ndarray) -> float:
    y = vertices_yx[:, 0]
    x = vertices_yx[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _measure_shape_area(layer: Shapes, index: int) -> tuple[str, float | None, str | None]:
    shape_types = list(getattr(layer, "shape_type", []))
    shape_type = str(shape_types[index] if index < len(shape_types) else "polygon").lower()
    if index >= len(layer.data):
        return shape_type, None, "shape index out of range"

    verts_yx = _shape_vertices_yx(layer.data[index])
    if verts_yx is None:
        return shape_type, None, "invalid vertex array"

    if shape_type in {"polygon", "rectangle"}:
        if verts_yx.shape[0] < 3:
            return shape_type, None, "fewer than 3 vertices"
        return shape_type, float(_polygon_area_yx(verts_yx)), None
    if shape_type == "ellipse":
        mins = verts_yx.min(axis=0)
        maxs = verts_yx.max(axis=0)
        ry = (maxs[0] - mins[0]) / 2.0
        rx = (maxs[1] - mins[1]) / 2.0
        return shape_type, float(np.pi * ry * rx), None
    if shape_type == "path":
        if verts_yx.shape[0] < 3:
            return shape_type, None, "fewer than 3 vertices"
        return shape_type, float(_polygon_area_yx(verts_yx)), None
    return shape_type, None, "unsupported shape type for area"


def _resolve_shapes_layer(viewer, roi_layer_name: object | None = None):
    name = str(roi_layer_name or "").strip()
    if name:
        layer = find_any_layer(viewer, name)
        return layer if isinstance(layer, Shapes) else None
    selected = viewer.layers.selection.active if viewer is not None else None
    if isinstance(selected, Shapes):
        return selected
    for layer in viewer.layers if viewer is not None else []:
        if isinstance(layer, Shapes):
            return layer
    return None


def _resolve_points_layer(viewer, points_layer_name: object | None = None):
    name = str(points_layer_name or "").strip()
    if name:
        layer = find_any_layer(viewer, name)
        return layer if isinstance(layer, Points) else None
    selected = viewer.layers.selection.active if viewer is not None else None
    if isinstance(selected, Points):
        return selected
    for layer in viewer.layers if viewer is not None else []:
        if isinstance(layer, Points):
            return layer
    return None


def _shape_bbox_xyxy(layer: Shapes, shape_index: int | None = None) -> tuple[tuple[float, float, float, float], int]:
    indices = _shape_indices(layer)
    if not indices:
        raise ValueError(f"Shapes layer [{layer.name}] does not contain a usable ROI prompt.")
    selected_index = int(shape_index) if shape_index is not None else indices[0]
    if selected_index < 0 or selected_index >= len(layer.data):
        raise ValueError(f"Shape index {selected_index} is out of range for [{layer.name}].")
    vertices = np.asarray(layer.data[selected_index], dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] < 2 or len(vertices) < 2:
        raise ValueError(f"Shape index {selected_index} in [{layer.name}] does not define a valid ROI.")
    min_y = float(np.min(vertices[:, 0]))
    max_y = float(np.max(vertices[:, 0]))
    min_x = float(np.min(vertices[:, 1]))
    max_x = float(np.max(vertices[:, 1]))
    return (min_x, min_y, max_x, max_y), selected_index


def _feature_value(layer: Points, column_name: str, row_index: int):
    features = getattr(layer, "features", None)
    if features is None:
        return None
    try:
        column = features[column_name]
    except Exception:
        return None
    try:
        return column.iloc[row_index]
    except Exception:
        try:
            return column[row_index]
        except Exception:
            return None


def _coerce_point_label(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "pos", "positive", "fg", "foreground", "true"}:
            return 1
        if text in {"0", "neg", "negative", "bg", "background", "false"}:
            return 0
        return None
    if isinstance(value, (bool, np.bool_)):
        return 1 if bool(value) else 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1 if float(value) > 0 else 0
    return None


def _points_prompt_xy_and_labels(layer: Points) -> tuple[np.ndarray, np.ndarray]:
    indices = sorted(int(i) for i in (getattr(layer, "selected_data", set()) or set()))
    if not indices:
        indices = list(range(len(layer.data)))
    if not indices:
        raise ValueError(f"Points layer [{layer.name}] does not contain any prompt points.")

    coords_xy: list[list[float]] = []
    labels: list[int] = []
    feature_candidates = ("sam_label", "point_label", "prompt_label", "label", "is_positive", "positive")
    for index in indices:
        point = np.asarray(layer.data[index], dtype=float)
        if point.shape[0] < 2:
            continue
        coords_xy.append([float(point[1]), float(point[0])])
        resolved_label = None
        for candidate in feature_candidates:
            resolved_label = _coerce_point_label(_feature_value(layer, candidate, index))
            if resolved_label is not None:
                break
        labels.append(1 if resolved_label is None else resolved_label)
    if not coords_xy:
        raise ValueError(f"Points layer [{layer.name}] does not contain usable 2D point prompts.")
    return np.asarray(coords_xy, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _points_prompt_slice_xy_and_labels(layer: Points) -> tuple[int, np.ndarray, np.ndarray]:
    indices = sorted(int(i) for i in (getattr(layer, "selected_data", set()) or set()))
    if not indices:
        indices = list(range(len(layer.data)))
    if not indices:
        raise ValueError(f"Points layer [{layer.name}] does not contain any prompt points.")

    coords_xy: list[list[float]] = []
    labels: list[int] = []
    z_indices: list[int] = []
    feature_candidates = ("sam_label", "point_label", "prompt_label", "label", "is_positive", "positive")
    for index in indices:
        point = np.asarray(layer.data[index], dtype=float)
        if point.shape[0] < 3:
            continue
        z_indices.append(int(round(float(point[0]))))
        coords_xy.append([float(point[2]), float(point[1])])
        resolved_label = None
        for candidate in feature_candidates:
            resolved_label = _coerce_point_label(_feature_value(layer, candidate, index))
            if resolved_label is not None:
                break
        labels.append(1 if resolved_label is None else resolved_label)
    if not coords_xy:
        raise ValueError(f"Points layer [{layer.name}] does not contain usable 3D point prompts.")
    unique_z = sorted(set(z_indices))
    if len(unique_z) != 1:
        raise ValueError(
            f"Points layer [{layer.name}] must place selected prompts on a single z slice for 3D propagation. "
            f"Found slices {unique_z}."
        )
    return unique_z[0], np.asarray(coords_xy, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _roi_mask_from_layer(roi_layer, target_shape: tuple[int, ...]) -> np.ndarray:
    if isinstance(roi_layer, Labels):
        data = np.asarray(roi_layer.data) > 0
        if data.shape != target_shape:
            raise ValueError(
                f"ROI labels shape {data.shape} does not match target image shape {target_shape}."
            )
        return data
    if isinstance(roi_layer, Shapes):
        return _rasterize_shapes_roi(roi_layer, target_shape)
    raise ValueError("ROI layer must be a Labels or Shapes layer.")


def _format_bbox(bbox: tuple[tuple[int, int], ...]) -> str:
    if not bbox:
        return "none"
    return ", ".join(f"dim{axis}=[{lo}:{hi}]" for axis, (lo, hi) in enumerate(bbox))


def _normalize_prefix(text: object) -> str:
    return str(text or "").strip()


def _strip_prefix_token(name: str, prefix: str) -> str:
    base = str(name or "").strip()
    token = str(prefix or "").strip()
    if token and base.startswith(token):
        base = base[len(token):]
    return base.lstrip(" _-.")


def _current_image_plane(layer: Image, viewer) -> np.ndarray:
    data = np.asarray(layer.data)
    if data.ndim < 2:
        raise ValueError(f"Image layer [{layer.name}] must have at least 2 dimensions.")
    if data.ndim == 2:
        return data
    current_step = tuple(int(step) for step in viewer.dims.current_step[: data.ndim])
    leading_shape = data.shape[:-2]
    leading_indices = []
    for axis, axis_size in enumerate(leading_shape):
        step_index = current_step[axis] if axis < len(current_step) else 0
        leading_indices.append(int(np.clip(step_index, 0, axis_size - 1)))
    return np.asarray(data[tuple(leading_indices) + (slice(None), slice(None))])


def _resolve_group_image_layers(viewer, prefix: str) -> list[Image]:
    token = _normalize_prefix(prefix)
    if not token:
        return []
    return [layer for layer in viewer.layers if isinstance(layer, Image) and str(layer.name).startswith(token)]


def _resolve_matching_roi_layer(viewer, image_layer: Image, roi_kind: str = "auto"):
    requested = str(roi_kind or "auto").strip().lower()
    candidates = []
    for layer in viewer.layers:
        if layer is image_layer:
            continue
        name = str(getattr(layer, "name", ""))
        if not name.startswith(str(image_layer.name)):
            continue
        if requested in {"auto", "labels"} and isinstance(layer, Labels):
            candidates.append(layer)
        if requested in {"auto", "shapes"} and isinstance(layer, Shapes):
            candidates.append(layer)
    if requested == "auto":
        for layer in candidates:
            if isinstance(layer, Shapes):
                return layer
        for layer in candidates:
            if isinstance(layer, Labels):
                return layer
    return candidates[0] if candidates else None


def _metric_value(values: np.ndarray, metric: str) -> float:
    metric_name = str(metric or "mean").strip().lower()
    if metric_name == "mean":
        return float(np.mean(values))
    if metric_name == "median":
        return float(np.median(values))
    if metric_name == "sum":
        return float(np.sum(values))
    if metric_name == "std":
        return float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if metric_name == "area":
        return float(values.size)
    raise ValueError(f"Unsupported ROI metric [{metric_name}].")


def _group_descriptive(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _format_group_descriptive(name: str, stats_dict: dict[str, float | int]) -> str:
    return (
        f"{name}: n={stats_dict['n']} mean={stats_dict['mean']:.4g} std={stats_dict['std']:.4g} "
        f"median={stats_dict['median']:.4g} min={stats_dict['min']:.4g} max={stats_dict['max']:.4g}"
    )


def _resolve_presentation_layers(viewer, layer_names: object | None = None) -> list[object]:
    resolved: list[object] = []
    if isinstance(layer_names, str):
        names = [part.strip() for part in layer_names.split(",") if part.strip()]
    elif isinstance(layer_names, (list, tuple)):
        names = [str(value).strip() for value in layer_names if str(value).strip()]
    else:
        names = []

    if names:
        for name in names:
            layer = find_any_layer(viewer, name)
            if isinstance(layer, (Image, Labels)):
                resolved.append(layer)
        return resolved

    selected_layers = list(getattr(viewer.layers.selection, "_selected", []) or [])
    selected_layers = [layer for layer in selected_layers if isinstance(layer, (Image, Labels))]
    if len(selected_layers) >= 2:
        return selected_layers

    return [layer for layer in viewer.layers if isinstance(layer, (Image, Labels))]


def _normalized_layout_shape(layout: object) -> str:
    value = str(layout or "row").strip().lower()
    if value in {"row", "rows", "horizontal"}:
        return "row"
    if value in {"column", "col", "vertical"}:
        return "column"
    if value in {"grid"}:
        return "grid"
    if value in {"pairs", "pair", "image_mask_pairs"}:
        return "pairs"
    return "row"


def _layer_display_extent(layer) -> tuple[float, float]:
    data = np.asarray(layer.data)
    spatial_ndim = int(getattr(layer, "ndim", data.ndim))
    if spatial_ndim < 2:
        return 1.0, 1.0
    scale_values = getattr(layer, "scale", None)
    scale = tuple(float(value) for value in scale_values) if scale_values is not None else ()
    if len(scale) < spatial_ndim:
        scale = (1.0,) * (spatial_ndim - len(scale)) + scale
    height = max(1.0, float(data.shape[-2]) * abs(float(scale[-2])))
    width = max(1.0, float(data.shape[-1]) * abs(float(scale[-1])))
    return height, width


def _arranged_translate(layer, *, offset_y: float, offset_x: float, match_origin: bool) -> tuple[float, ...]:
    data = np.asarray(layer.data)
    ndim = int(getattr(layer, "ndim", data.ndim))
    translate_values = getattr(layer, "translate", None)
    existing = tuple(float(value) for value in translate_values) if translate_values is not None else ()
    if len(existing) < ndim:
        existing = (0.0,) * (ndim - len(existing)) + existing
    base = [0.0] * ndim if match_origin else list(existing)
    if ndim >= 2:
        base[-2] = float(offset_y) if match_origin else float(base[-2] + offset_y)
        base[-1] = float(offset_x) if match_origin else float(base[-1] + offset_x)
    return tuple(base)


def _clone_layer_for_presentation(ctx: ToolContext, layer, *, output_name: str, translate: tuple[float, ...]):
    data = np.asarray(layer.data).copy()
    scale_values = getattr(layer, "scale", None)
    scale = tuple(float(value) for value in scale_values) if scale_values is not None else ()
    if isinstance(layer, Image):
        ctx.viewer.add_image(data, name=output_name, scale=scale, translate=translate)
    elif isinstance(layer, Labels):
        ctx.viewer.add_labels(data, name=output_name, scale=scale, translate=translate)


def _placements_for_row(layers: list[object], extents: list[tuple[float, float]], spacing: float) -> list[tuple[object, float, float]]:
    placements: list[tuple[object, float, float]] = []
    current_x = 0.0
    for layer, (_height, width) in zip(layers, extents):
        placements.append((layer, 0.0, current_x))
        current_x += width + spacing
    return placements


def _placements_for_column(layers: list[object], extents: list[tuple[float, float]], spacing: float) -> list[tuple[object, float, float]]:
    placements: list[tuple[object, float, float]] = []
    current_y = 0.0
    for layer, (height, _width) in zip(layers, extents):
        placements.append((layer, current_y, 0.0))
        current_y += height + spacing
    return placements


def _placements_for_grid(
    layers: list[object],
    extents: list[tuple[float, float]],
    spacing: float,
    columns: int,
) -> list[tuple[object, float, float]]:
    columns = max(1, columns)
    row_heights: list[float] = []
    col_widths: list[float] = [0.0] * columns
    grouped: list[list[tuple[object, tuple[float, float]]]] = []

    for start in range(0, len(layers), columns):
        chunk = list(zip(layers[start : start + columns], extents[start : start + columns]))
        grouped.append(chunk)
        row_heights.append(max((height for _layer, (height, _width) in chunk), default=0.0))
        for col_index, (_layer, (_height, width)) in enumerate(chunk):
            col_widths[col_index] = max(col_widths[col_index], width)

    row_offsets: list[float] = []
    current_y = 0.0
    for height in row_heights:
        row_offsets.append(current_y)
        current_y += height + spacing

    col_offsets: list[float] = []
    current_x = 0.0
    for width in col_widths:
        col_offsets.append(current_x)
        current_x += width + spacing

    placements: list[tuple[object, float, float]] = []
    for row_index, chunk in enumerate(grouped):
        for col_index, (layer, _extent) in enumerate(chunk):
            placements.append((layer, row_offsets[row_index], col_offsets[col_index]))
    return placements


def _auto_grid_shape(count: int) -> tuple[int, int]:
    count = max(1, int(count))
    cols = max(1, int(np.ceil(np.sqrt(count))))
    rows = max(1, int(np.ceil(count / cols)))
    return rows, cols


def _hide_non_image_layers(viewer) -> list[str]:
    hidden: list[str] = []
    for layer in viewer.layers:
        if isinstance(layer, Image):
            continue
        if bool(getattr(layer, "visible", True)):
            layer.visible = False
            hidden.append(layer.name)
    return hidden


def _parse_layer_names_argument(value: object) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _resolve_existing_layers(viewer, names: list[str]) -> list[object]:
    resolved: list[object] = []
    seen: set[str] = set()
    for name in names:
        layer = find_any_layer(viewer, name)
        if layer is None or layer.name in seen:
            continue
        resolved.append(layer)
        seen.add(layer.name)
    return resolved


class ShowImageLayersInGridTool:
    spec = ToolSpec(
        name="show_image_layers_in_grid",
        display_name="Show Image Layers In Grid",
        category="presentation",
        description="Show open image layers in napari grid view for side-by-side comparison.",
        execution_mode="immediate",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("layer_names", "string_list", description="Optional ordered image layer names to show."),
            ParamSpec("spacing", "float", description="Optional grid spacing.", default=0.0, minimum=0.0),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Presentation"},
        provenance_metadata={"algorithm": "napari_grid_view", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        if isinstance(args.get("layer_names"), str):
            names = [part.strip() for part in str(args.get("layer_names", "")).split(",") if part.strip()]
        elif isinstance(args.get("layer_names"), (list, tuple)):
            names = [str(value).strip() for value in args.get("layer_names", []) if str(value).strip()]
        else:
            names = []

        image_layers: list[Image] = []
        if names:
            for name in names:
                layer = find_any_layer(ctx.viewer, name)
                if isinstance(layer, Image):
                    image_layers.append(layer)
        else:
            image_layers = [layer for layer in ctx.viewer.layers if isinstance(layer, Image)]

        if len(image_layers) < 2:
            return "Need at least 2 image layers to show a side-by-side grid."

        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "layer_names": [layer.name for layer in image_layers],
                "spacing": normalize_float(args.get("spacing", 0.0), default=0.0, minimum=0.0, maximum=1500.0),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        layer_names = [str(name).strip() for name in payload.get("layer_names", []) if str(name).strip()]
        image_layers = [find_any_layer(ctx.viewer, name) for name in layer_names]
        image_layers = [layer for layer in image_layers if isinstance(layer, Image)]
        if len(image_layers) < 2:
            return "No usable image layers were available for grid view."

        selected_names = {layer.name for layer in image_layers}
        for layer in ctx.viewer.layers:
            if isinstance(layer, Image):
                layer.visible = layer.name in selected_names
        hidden_non_image = _hide_non_image_layers(ctx.viewer)
        rows, cols = _auto_grid_shape(len(image_layers))
        ctx.viewer.grid.shape = (rows, cols)
        ctx.viewer.grid.spacing = float(payload.get("spacing", 0.0))
        ctx.viewer.grid.enabled = True
        setattr(ctx.viewer, "_assistant_grid_hidden_non_image_layers", hidden_non_image)
        if hasattr(ctx.viewer, "reset_view"):
            ctx.viewer.reset_view()
        return (
            f"Enabled image grid view for {len(image_layers)} image layer(s) with shape=({rows}, {cols}). "
            f"Hidden {len(hidden_non_image)} non-image layer(s)."
        )


class HideImageGridViewTool:
    spec = ToolSpec(
        name="hide_image_grid_view",
        display_name="Hide Image Grid View",
        category="presentation",
        description="Turn off napari grid view and restore hidden non-image layers.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "points", "shapes"),
        parameter_schema=(),
        output_type="message",
        ui_metadata={"panel_group": "Presentation"},
        provenance_metadata={"algorithm": "napari_grid_view", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        return PreparedJob(tool_name=self.spec.name, kind=self.spec.name, mode="immediate", payload={"kind": self.spec.name})

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        hidden_non_image = list(getattr(ctx.viewer, "_assistant_grid_hidden_non_image_layers", []) or [])
        for name in hidden_non_image:
            layer = find_any_layer(ctx.viewer, name)
            if layer is not None:
                layer.visible = True
        setattr(ctx.viewer, "_assistant_grid_hidden_non_image_layers", [])
        ctx.viewer.grid.enabled = False
        if hasattr(ctx.viewer, "reset_view"):
            ctx.viewer.reset_view()
        return "Disabled image grid view."


class ShowLayersTool:
    spec = ToolSpec(
        name="show_layers",
        display_name="Show Layers",
        category="visualization",
        description="Show specific layers without changing the visibility of other layers.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        parameter_schema=(
            ParamSpec("layer_names", "string_list", description="Layer names to show."),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Visualization"},
        provenance_metadata={"algorithm": "layer_visibility", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        names = _parse_layer_names_argument((arguments or {}).get("layer_names"))
        if not names:
            return "Provide at least one layer name to show."
        layers = _resolve_existing_layers(ctx.viewer, names)
        if not layers:
            return "No matching layers were found to show."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={"kind": self.spec.name, "layer_names": [layer.name for layer in layers]},
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        layers = _resolve_existing_layers(ctx.viewer, _parse_layer_names_argument(result.payload.get("layer_names", [])))
        if not layers:
            return "No usable layers were available to show."
        for layer in layers:
            layer.visible = True
        shown = ", ".join(f"[{layer.name}]" for layer in layers)
        return f"Showed {len(layers)} layer(s): {shown}."


class HideLayersTool:
    spec = ToolSpec(
        name="hide_layers",
        display_name="Hide Layers",
        category="visualization",
        description="Hide specific layers without changing the visibility of other layers.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        parameter_schema=(
            ParamSpec("layer_names", "string_list", description="Layer names to hide."),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Visualization"},
        provenance_metadata={"algorithm": "layer_visibility", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        names = _parse_layer_names_argument((arguments or {}).get("layer_names"))
        if not names:
            return "Provide at least one layer name to hide."
        layers = _resolve_existing_layers(ctx.viewer, names)
        if not layers:
            return "No matching layers were found to hide."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={"kind": self.spec.name, "layer_names": [layer.name for layer in layers]},
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        layers = _resolve_existing_layers(ctx.viewer, _parse_layer_names_argument(result.payload.get("layer_names", [])))
        if not layers:
            return "No usable layers were available to hide."
        for layer in layers:
            layer.visible = False
        hidden = ", ".join(f"[{layer.name}]" for layer in layers)
        return f"Hid {len(layers)} layer(s): {hidden}."


class ShowOnlyLayersTool:
    spec = ToolSpec(
        name="show_only_layers",
        display_name="Show Only Layers",
        category="visualization",
        description="Show only the specified layers and hide all other layers.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        parameter_schema=(
            ParamSpec("layer_names", "string_list", description="Layer names to keep visible."),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Visualization"},
        provenance_metadata={"algorithm": "layer_visibility", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        names = _parse_layer_names_argument((arguments or {}).get("layer_names"))
        if not names:
            return "Provide at least one layer name to show exclusively."
        layers = _resolve_existing_layers(ctx.viewer, names)
        if not layers:
            return "No matching layers were found to show exclusively."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={"kind": self.spec.name, "layer_names": [layer.name for layer in layers]},
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        layers = _resolve_existing_layers(ctx.viewer, _parse_layer_names_argument(result.payload.get("layer_names", [])))
        if not layers:
            return "No usable layers were available to show exclusively."
        selected = {layer.name for layer in layers}
        for layer in ctx.viewer.layers:
            layer.visible = layer.name in selected
        shown = ", ".join(f"[{layer.name}]" for layer in layers)
        return f"Showing only {len(layers)} layer(s): {shown}."


class ShowAllLayersTool:
    spec = ToolSpec(
        name="show_all_layers",
        display_name="Show All Layers",
        category="visualization",
        description="Show every layer in the current viewer.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        parameter_schema=(),
        output_type="message",
        ui_metadata={"panel_group": "Visualization"},
        provenance_metadata={"algorithm": "layer_visibility", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        if len(ctx.viewer.layers) == 0:
            return "No layers are open to show."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={"kind": self.spec.name},
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        for layer in ctx.viewer.layers:
            layer.visible = True
        return f"Showed all {len(ctx.viewer.layers)} layer(s)."


class ArrangeLayersForPresentationTool:
    spec = ToolSpec(
        name="arrange_layers_for_presentation",
        display_name="Arrange Layers For Presentation",
        category="presentation",
        description="Arrange image and labels layers into a row, column, grid, or repeated image-mask pairs for presentation.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels"),
        parameter_schema=(
            ParamSpec("layer_names", "string_list", description="Optional ordered layer names to arrange."),
            ParamSpec("layout", "string", description="row, column, grid, or pairs.", default="row"),
            ParamSpec("spacing", "float", description="World-coordinate spacing between arranged layers.", default=20.0, minimum=0.0),
            ParamSpec("columns", "int", description="Optional number of columns for grid layout.", default=0, minimum=0),
            ParamSpec("group_size", "int", description="Group size for pairs layout.", default=2, minimum=1),
            ParamSpec("use_copies", "bool", description="Create display copies instead of moving the original layers.", default=True),
            ParamSpec("match_origin", "bool", description="Reset arranged layers to a shared top-left origin before layout offsets are applied.", default=True),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Presentation"},
        provenance_metadata={"algorithm": "layer_presentation_layout", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        layers = _resolve_presentation_layers(ctx.viewer, args.get("layer_names"))
        if len(layers) < 2:
            return "Need at least 2 image or labels layers to arrange for presentation."
        layer_names = [layer.name for layer in layers]
        layout = _normalized_layout_shape(args.get("layout", "row"))
        spacing = normalize_float(args.get("spacing", 20.0), default=20.0, minimum=0.0, maximum=10_000.0)
        columns = normalize_int(args.get("columns", 0), default=0, minimum=0, maximum=64)
        group_size = normalize_int(args.get("group_size", 2), default=2, minimum=1, maximum=32)
        use_copies = bool(args.get("use_copies", True))
        match_origin = bool(args.get("match_origin", True))
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "layer_names": layer_names,
                "layout": layout,
                "spacing": spacing,
                "columns": columns,
                "group_size": group_size,
                "use_copies": use_copies,
                "match_origin": match_origin,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        layer_names = [str(name).strip() for name in payload.get("layer_names", []) if str(name).strip()]
        layers = [find_any_layer(ctx.viewer, name) for name in layer_names]
        layers = [layer for layer in layers if isinstance(layer, (Image, Labels))]
        if len(layers) < 2:
            return "No usable image or labels layers were available to arrange."

        spacing = float(payload.get("spacing", 20.0))
        layout = _normalized_layout_shape(payload.get("layout", "row"))
        columns = int(payload.get("columns", 0) or 0)
        group_size = int(payload.get("group_size", 2) or 2)
        use_copies = bool(payload.get("use_copies", True))
        match_origin = bool(payload.get("match_origin", True))
        extents = [_layer_display_extent(layer) for layer in layers]
        if layout == "column":
            placements = _placements_for_column(layers, extents, spacing)
        elif layout == "grid":
            if columns <= 0:
                columns = max(1, int(np.ceil(np.sqrt(len(layers)))))
            placements = _placements_for_grid(layers, extents, spacing, columns)
        elif layout == "pairs":
            group_size = max(1, group_size)
            placements = _placements_for_grid(layers, extents, spacing, group_size)
        else:
            placements = _placements_for_row(layers, extents, spacing)

        arranged_names: list[str] = []
        for layer, offset_y, offset_x in placements:
            translate = _arranged_translate(layer, offset_y=offset_y, offset_x=offset_x, match_origin=match_origin)
            if use_copies:
                output_name = next_output_name(ctx.viewer, f"{layer.name}_present")
                _clone_layer_for_presentation(ctx, layer, output_name=output_name, translate=translate)
                arranged_names.append(output_name)
            else:
                layer.translate = translate
                arranged_names.append(layer.name)

        arranged_label = ", ".join(f"[{name}]" for name in arranged_names[:6])
        if len(arranged_names) > 6:
            arranged_label += f" and {len(arranged_names) - 6} more"
        action = "Created presentation copies for" if use_copies else "Arranged"
        return (
            f"{action} {len(arranged_names)} layer(s) with layout={layout} spacing={spacing:.6g}. "
            f"match_origin={str(match_origin).lower()} use_copies={str(use_copies).lower()}. {arranged_label}."
        )


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


class ExtractAxonInteriorsTool:
    spec = ToolSpec(
        name="extract_axon_interiors",
        display_name="Extract Axon Interiors",
        category="segmentation",
        description="Extract candidate axon interiors from a 2D grayscale EM image using dark-ring enclosure and region filtering.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("image_layer", "string", description="Optional source EM image layer."),
            ParamSpec("sigma", "float", description="Gaussian smoothing sigma before dark-ring thresholding.", default=1.0, minimum=0.0, maximum=5.0),
            ParamSpec("dark_quantile", "float", description="Quantile used to threshold dark myelin-like signal.", default=0.2, minimum=0.01, maximum=0.5),
            ParamSpec("closing_radius", "int", description="Morphological closing radius for dark rings.", default=2, minimum=0, maximum=12),
            ParamSpec("min_area", "int", description="Minimum enclosed interior area to keep.", default=100, minimum=1),
            ParamSpec("max_area", "int", description="Maximum enclosed interior area to keep.", default=500000, minimum=1),
            ParamSpec("clear_border", "bool", description="Drop candidate interiors that touch the image border.", default=True),
            ParamSpec("min_solidity", "float", description="Minimum solidity required for a kept candidate.", default=0.2, minimum=0.0, maximum=1.0),
            ParamSpec("max_eccentricity", "float", description="Maximum eccentricity allowed for a kept candidate.", default=0.995, minimum=0.0, maximum=1.0),
        ),
        output_type="labels_layer",
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "dark_ring_enclosed_interiors", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("image_layer"))
        if image_layer is None:
            return "No valid image layer available for axon-interior extraction."
        if getattr(image_layer, "rgb", False):
            return "Axon-interior extraction currently supports grayscale 2D image layers, not RGB layers."
        image_data = np.asarray(image_layer.data)
        if image_data.ndim != 2:
            return f"Axon-interior extraction currently supports 2D image layers only. Got ndim={image_data.ndim}."
        sigma = normalize_float(args.get("sigma", 1.0), default=1.0, minimum=0.0, maximum=5.0)
        dark_quantile = normalize_float(args.get("dark_quantile", 0.2), default=0.2, minimum=0.01, maximum=0.5)
        closing_radius = normalize_int(args.get("closing_radius", 2), default=2, minimum=0, maximum=12)
        min_area = normalize_int(args.get("min_area", 100), default=100, minimum=1)
        max_area = normalize_int(args.get("max_area", 500000), default=500000, minimum=min_area)
        clear_border_flag = bool(args.get("clear_border", True))
        min_solidity = normalize_float(args.get("min_solidity", 0.2), default=0.2, minimum=0.0, maximum=1.0)
        max_eccentricity = normalize_float(args.get("max_eccentricity", 0.995), default=0.995, minimum=0.0, maximum=1.0)
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "image_layer_name": image_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{image_layer.name}_axon_interiors"),
                "data": image_data.astype(np.float32, copy=True),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
                "sigma": sigma,
                "dark_quantile": dark_quantile,
                "closing_radius": closing_radius,
                "min_area": min_area,
                "max_area": max_area,
                "clear_border": clear_border_flag,
                "min_solidity": min_solidity,
                "max_eccentricity": max_eccentricity,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        image = np.asarray(payload["data"], dtype=np.float32)
        finite = np.isfinite(image)
        if not np.any(finite):
            payload["result"] = np.zeros_like(image, dtype=np.int32)
            payload["object_count"] = 0
            payload["candidate_count"] = 0
            payload["threshold"] = None
            return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

        image = np.where(finite, image, np.nanmedian(image[finite]))
        image_min = float(image.min())
        image_max = float(image.max())
        if image_max > image_min:
            normalized = (image - image_min) / (image_max - image_min)
        else:
            normalized = np.zeros_like(image, dtype=np.float32)

        sigma = float(payload["sigma"])
        smoothed = ndi.gaussian_filter(normalized, sigma=sigma) if sigma > 0 else normalized
        support_epsilon = max(1e-6, (image_max - image_min) * 1e-6)
        support_mask = finite & (image > (image_min + support_epsilon))
        support_values = smoothed[support_mask]
        if support_values.size == 0:
            support_values = smoothed[finite]
            support_mask = finite
        quantile_threshold = float(np.quantile(support_values, float(payload["dark_quantile"])))
        try:
            otsu_threshold = float(threshold_otsu(support_values))
        except Exception:
            otsu_threshold = quantile_threshold
        dark_threshold = otsu_threshold
        dark_mask = support_mask & (smoothed <= dark_threshold)

        closing_radius = int(payload["closing_radius"])
        if closing_radius > 0:
            dark_mask = ndi.binary_closing(dark_mask, structure=disk(closing_radius))

        enclosed = ndi.binary_fill_holes(dark_mask) & ~dark_mask
        enclosed = remove_small_objects(enclosed.astype(bool), min_size=max(1, int(payload["min_area"])))
        if bool(payload["clear_border"]):
            enclosed = clear_border(enclosed)

        labeled = sk_label(enclosed, connectivity=1).astype(np.int32, copy=False)
        kept = np.zeros_like(labeled, dtype=np.int32)
        kept_count = 0
        candidate_count = 0
        min_area = int(payload["min_area"])
        max_area = int(payload["max_area"])
        min_solidity = float(payload["min_solidity"])
        max_eccentricity = float(payload["max_eccentricity"])
        for prop in regionprops(labeled):
            candidate_count += 1
            if prop.area < min_area or prop.area > max_area:
                continue
            solidity = getattr(prop, "solidity", None)
            solidity_value = 0.0 if solidity is None else float(solidity)
            if solidity_value < min_solidity:
                continue
            eccentricity = getattr(prop, "eccentricity", None)
            eccentricity_value = 1.0 if eccentricity is None else float(eccentricity)
            if eccentricity_value > max_eccentricity:
                continue
            kept_count += 1
            kept[labeled == prop.label] = kept_count

        payload["result"] = kept
        payload["object_count"] = kept_count
        payload["candidate_count"] = candidate_count
        payload["threshold"] = dark_threshold
        payload["foreground_pixels"] = int(np.count_nonzero(kept))
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "parameters": {
                    "sigma": payload["sigma"],
                    "dark_quantile": payload["dark_quantile"],
                    "closing_radius": payload["closing_radius"],
                    "min_area": payload["min_area"],
                    "max_area": payload["max_area"],
                    "clear_border": payload["clear_border"],
                    "min_solidity": payload["min_solidity"],
                    "max_eccentricity": payload["max_eccentricity"],
                },
                "input_layer": payload["image_layer_name"],
                "output_layer": payload["output_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        labels_layer = ctx.viewer.add_labels(
            payload["result"],
            name=payload["output_name"],
            scale=payload["scale"],
            translate=payload["translate"],
        )
        labels_layer.metadata = dict(getattr(labels_layer, "metadata", {}) or {})
        labels_layer.metadata["axon_interior_extraction"] = {
            "source_layer": payload["image_layer_name"],
            "object_count": int(payload["object_count"]),
            "candidate_count": int(payload["candidate_count"]),
            "threshold": payload["threshold"],
            "sigma": payload["sigma"],
            "dark_quantile": payload["dark_quantile"],
            "closing_radius": payload["closing_radius"],
            "min_area": payload["min_area"],
            "max_area": payload["max_area"],
            "clear_border": payload["clear_border"],
            "min_solidity": payload["min_solidity"],
            "max_eccentricity": payload["max_eccentricity"],
        }
        return (
            f"Extracted candidate axon interiors from [{payload['image_layer_name']}] into [{payload['output_name']}]. "
            f"kept={payload['object_count']} candidates={payload['candidate_count']} "
            f"foreground_pixels={payload['foreground_pixels']} dark_threshold={payload['threshold']:.6g}."
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


class InspectROIContextTool:
    spec = ToolSpec(
        name="inspect_roi_context",
        display_name="Inspect ROI Context",
        category="roi",
        description="Inspect a labels or shapes ROI layer and summarize its spatial scope.",
        execution_mode="immediate",
        supported_layer_types=("labels", "shapes"),
        parameter_schema=(ParamSpec("roi_layer", "string", description="Optional ROI layer name."),),
        output_type="message",
        ui_metadata={"panel_group": "ROI"},
        provenance_metadata={"algorithm": "roi_summary", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        roi_layer = _resolve_roi_layer(ctx.viewer, (arguments or {}).get("roi_layer"))
        if roi_layer is None:
            return "No valid ROI layer available. Select or name a Labels or Shapes layer."
        if isinstance(roi_layer, Labels):
            binary = np.asarray(roi_layer.data) > 0
            bbox = _labels_bbox(binary)
            return (
                f"ROI layer [{roi_layer.name}] is a Labels ROI. "
                f"ndim={binary.ndim} foreground={int(binary.sum())} bbox={_format_bbox(bbox)}."
            )
        indices = _shape_indices(roi_layer)
        bbox = tuple()
        if indices:
            vertices = np.concatenate([np.asarray(roi_layer.data[i], dtype=float) for i in indices if i < len(roi_layer.data)], axis=0)
            if vertices.size:
                mins = vertices.min(axis=0)
                maxs = vertices.max(axis=0)
                bbox = tuple((int(np.floor(lo)), int(np.ceil(hi))) for lo, hi in zip(mins, maxs))
        selection_text = f"selected_shapes={len(indices)}" if getattr(roi_layer, "selected_data", set()) else f"shapes={len(indices)}"
        return (
            f"ROI layer [{roi_layer.name}] is a Shapes ROI. "
            f"ndim={getattr(roi_layer, 'ndim', 'n/a')} {selection_text} bbox={_format_bbox(bbox)}."
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class MeasureShapesROIAreaTool:
    spec = ToolSpec(
        name="measure_shapes_roi_area",
        display_name="Measure Shapes ROI Area",
        category="roi",
        description="Measure total area and per-shape areas for the selected or named Shapes ROI layer.",
        execution_mode="immediate",
        supported_layer_types=("shapes",),
        parameter_schema=(ParamSpec("roi_layer", "string", description="Optional Shapes ROI layer name."),),
        output_type="message",
        ui_metadata={"panel_group": "ROI"},
        provenance_metadata={"algorithm": "shapes_roi_area", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        roi_layer = _resolve_shapes_layer(ctx.viewer, (arguments or {}).get("roi_layer"))
        if roi_layer is None:
            return "No valid Shapes ROI layer available. Select or name a Shapes layer."

        indices = _shape_indices(roi_layer)
        if not indices:
            return f"Shapes layer [{roi_layer.name}] contains no shapes."

        measured: list[tuple[int, str, float]] = []
        skipped: list[tuple[int, str, str]] = []
        for index in indices:
            shape_type, area, reason = _measure_shape_area(roi_layer, index)
            display_index = int(index) + 1
            if area is None:
                skipped.append((display_index, shape_type, reason or "unknown reason"))
                continue
            measured.append((display_index, shape_type, float(area)))

        total_area = float(sum(area for _, _, area in measured))
        metadata = dict(getattr(roi_layer, "metadata", {}) or {})
        metadata["roi_shape_areas"] = {
            "layer_name": roi_layer.name,
            "areas": measured,
            "total_area": total_area,
            "skipped": skipped,
        }
        roi_layer.metadata = metadata

        lines = [
            f'Selected Shapes layer: "{roi_layer.name}"',
            f"Measured {len(measured)} shape(s).",
        ]
        for display_index, shape_type, area in measured:
            lines.append(f"  Shape {display_index} ({shape_type}): area = {area:.2f} px^2")
        lines.append(f"Total measured area: {total_area:.2f} px^2")
        if skipped:
            lines.append("")
            lines.append("Skipped shapes:")
            for display_index, shape_type, reason in skipped:
                lines.append(f"  Shape {display_index} ({shape_type}): {reason}")
        return "\n".join(lines)

    def execute(self, job: PreparedJob) -> ToolResult:
        raise RuntimeError("Immediate tool does not execute worker jobs.")

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        return result.message


class ExtractROIValuesTool:
    spec = ToolSpec(
        name="extract_roi_values",
        display_name="Extract ROI Values",
        category="roi",
        description="Extract image intensity values from a labels or shapes ROI and summarize them.",
        execution_mode="worker",
        supported_layer_types=("image", "labels", "shapes"),
        parameter_schema=(
            ParamSpec("image_layer", "string", description="Image layer to sample.", required=False),
            ParamSpec("roi_layer", "string", description="Labels or Shapes ROI layer.", required=False),
        ),
        output_type="message",
        ui_metadata={"panel_group": "ROI"},
        provenance_metadata={"algorithm": "roi_value_summary", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("image_layer"))
        if image_layer is None:
            return "No valid image layer available for ROI value extraction."
        if getattr(image_layer, "rgb", False):
            return "ROI value extraction currently supports grayscale image layers, not RGB layers."
        roi_layer = _resolve_roi_layer(ctx.viewer, args.get("roi_layer"))
        if roi_layer is None:
            return "No valid ROI layer available. Select or name a Labels or Shapes layer."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "image_layer_name": image_layer.name,
                "roi_layer_name": roi_layer.name,
                "roi_kind": roi_layer.__class__.__name__,
                "image_data": np.asarray(image_layer.data).copy(),
                "roi_data": np.asarray(getattr(roi_layer, "data", None), dtype=object if isinstance(roi_layer, Shapes) else None).copy()
                if isinstance(roi_layer, Labels)
                else [np.asarray(shape, dtype=float).copy() for shape in roi_layer.data],
                "roi_shape_type": list(getattr(roi_layer, "shape_type", [])) if isinstance(roi_layer, Shapes) else None,
                "roi_selected": sorted(int(i) for i in getattr(roi_layer, "selected_data", set())) if isinstance(roi_layer, Shapes) else None,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        image_data = np.asarray(payload["image_data"])
        if payload["roi_kind"] == "Labels":
            roi_layer = Labels(np.asarray(payload["roi_data"]).astype(np.int32, copy=False))
        else:
            roi_layer = Shapes(
                data=list(payload["roi_data"]),
                shape_type=payload["roi_shape_type"] or "polygon",
            )
            if payload["roi_selected"]:
                roi_layer.selected_data = set(payload["roi_selected"])
        roi_mask = _roi_mask_from_layer(roi_layer, image_data.shape)
        if not np.any(roi_mask):
            payload["roi_voxels"] = 0
            payload["bbox"] = tuple()
            payload["stats"] = {}
            return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)
        values = image_data[roi_mask]
        bbox = _labels_bbox(roi_mask)
        payload["roi_voxels"] = int(roi_mask.sum())
        payload["bbox"] = bbox
        payload["stats"] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "image_layer": payload["image_layer_name"],
                "roi_layer": payload["roi_layer_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        if int(payload.get("roi_voxels", 0)) <= 0:
            return (
                f"ROI [{payload['roi_layer_name']}] did not produce any pixels/voxels inside "
                f"[{payload['image_layer_name']}]."
            )
        stats = payload["stats"]
        return (
            f"Extracted ROI values from [{payload['image_layer_name']}] using [{payload['roi_layer_name']}]. "
            f"roi_voxels={payload['roi_voxels']} bbox={_format_bbox(payload['bbox'])} "
            f"mean={stats['mean']:.6g} std={stats['std']:.6g} median={stats['median']:.6g} "
            f"min={stats['min']:.6g} max={stats['max']:.6g}."
        )


class CompareROIGroupsTool:
    spec = ToolSpec(
        name="compare_roi_groups",
        display_name="Compare ROI Groups",
        category="statistics",
        description="Extract one ROI metric per image for two groups, run descriptive stats and assumption checks, then compare the groups.",
        execution_mode="worker",
        supported_layer_types=("image", "labels", "shapes"),
        parameter_schema=(
            ParamSpec("group_a_prefix", "string", description="Prefix for group A image layers.", required=True),
            ParamSpec("group_b_prefix", "string", description="Prefix for group B image layers.", required=True),
            ParamSpec("metric", "string", description="ROI metric to extract: mean, median, sum, std, or area.", default="mean"),
            ParamSpec("roi_kind", "string", description="ROI layer type: auto, shapes, or labels.", default="auto"),
            ParamSpec("pair_mode", "string", description="paired_suffix or unpaired.", default="paired_suffix"),
            ParamSpec("alpha", "float", description="Significance threshold.", default=0.05, minimum=1e-6, maximum=0.5),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Statistics"},
        provenance_metadata={"algorithm": "roi_group_compare", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        prefix_a = _normalize_prefix(args.get("group_a_prefix"))
        prefix_b = _normalize_prefix(args.get("group_b_prefix"))
        if not prefix_a or not prefix_b:
            return "ROI group comparison requires group_a_prefix and group_b_prefix."
        metric = str(args.get("metric", "mean")).strip().lower() or "mean"
        if metric not in {"mean", "median", "sum", "std", "area"}:
            return f"Unsupported ROI metric [{metric}]. Use mean, median, sum, std, or area."
        roi_kind = str(args.get("roi_kind", "auto")).strip().lower() or "auto"
        if roi_kind not in {"auto", "shapes", "labels"}:
            return f"Unsupported roi_kind [{roi_kind}]. Use auto, shapes, or labels."
        pair_mode = str(args.get("pair_mode", "paired_suffix")).strip().lower() or "paired_suffix"
        if pair_mode not in {"paired_suffix", "unpaired"}:
            return f"Unsupported pair_mode [{pair_mode}]. Use paired_suffix or unpaired."

        images_a = _resolve_group_image_layers(ctx.viewer, prefix_a)
        images_b = _resolve_group_image_layers(ctx.viewer, prefix_b)
        if not images_a or not images_b:
            return (
                f"Could not resolve both image groups. group_a={len(images_a)} images for prefix [{prefix_a}], "
                f"group_b={len(images_b)} images for prefix [{prefix_b}]."
            )

        items: list[dict[str, object]] = []
        missing_roi: list[str] = []
        for group_name, prefix, layers in (("A", prefix_a, images_a), ("B", prefix_b, images_b)):
            for image_layer in layers:
                if getattr(image_layer, "rgb", False):
                    continue
                roi_layer = _resolve_matching_roi_layer(ctx.viewer, image_layer, roi_kind=roi_kind)
                if roi_layer is None:
                    missing_roi.append(image_layer.name)
                    continue
                try:
                    image_plane = _current_image_plane(image_layer, ctx.viewer)
                except Exception:
                    continue
                pair_key = _strip_prefix_token(image_layer.name, prefix)
                item = {
                    "group": group_name,
                    "pair_key": pair_key or image_layer.name,
                    "image_layer_name": image_layer.name,
                    "roi_layer_name": roi_layer.name,
                    "roi_kind": roi_layer.__class__.__name__,
                    "image_data": np.asarray(image_plane).copy(),
                    "roi_data": np.asarray(getattr(roi_layer, "data", None), dtype=object if isinstance(roi_layer, Labels) else None).copy()
                    if isinstance(roi_layer, Labels)
                    else [np.asarray(shape, dtype=float).copy() for shape in roi_layer.data],
                    "roi_shape_type": list(getattr(roi_layer, "shape_type", [])) if isinstance(roi_layer, Shapes) else None,
                    "roi_selected": sorted(int(i) for i in getattr(roi_layer, "selected_data", set())) if isinstance(roi_layer, Shapes) else None,
                }
                items.append(item)

        if missing_roi:
            return (
                "Could not find matching ROI layers for some images: "
                + ", ".join(f"[{name}]" for name in missing_roi[:8])
                + ("..." if len(missing_roi) > 8 else "")
            )
        if len(items) < 2:
            return "Not enough image/ROI pairs were resolved for ROI group comparison."

        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "group_a_prefix": prefix_a,
                "group_b_prefix": prefix_b,
                "metric": metric,
                "roi_kind": roi_kind,
                "pair_mode": pair_mode,
                "alpha": normalize_float(args.get("alpha", 0.05), 0.05, minimum=1e-6, maximum=0.5),
                "items": items,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        metric = str(payload["metric"])
        rows: list[dict[str, object]] = []
        for item in payload["items"]:
            image_data = np.asarray(item["image_data"])
            if item["roi_kind"] == "Labels":
                roi_layer = Labels(np.asarray(item["roi_data"]).astype(np.int32, copy=False))
            else:
                roi_layer = Shapes(data=list(item["roi_data"]), shape_type=item["roi_shape_type"] or "polygon")
                if item["roi_selected"]:
                    roi_layer.selected_data = set(item["roi_selected"])
            roi_mask = _roi_mask_from_layer(roi_layer, image_data.shape)
            values = image_data[roi_mask]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            rows.append(
                {
                    "group": item["group"],
                    "pair_key": item["pair_key"],
                    "image_layer_name": item["image_layer_name"],
                    "roi_layer_name": item["roi_layer_name"],
                    "metric": metric,
                    "value": _metric_value(values, metric),
                    "roi_pixels": int(values.size),
                }
            )

        values_a = [float(row["value"]) for row in rows if row["group"] == "A"]
        values_b = [float(row["value"]) for row in rows if row["group"] == "B"]
        if len(values_a) < 2 or len(values_b) < 2:
            return ToolResult(
                tool_name=self.spec.name,
                kind=job.kind,
                payload={**payload, "rows": rows, "comparison_error": "Each group needs at least 2 valid ROI samples."},
            )

        group_stats = {"A": _group_descriptive(values_a), "B": _group_descriptive(values_b)}
        pair_mode = str(payload["pair_mode"])
        normality = {}
        variance = {}
        selected_test = "Welch t-test"
        result_stats: dict[str, object] = {}

        arr_a = np.asarray(values_a, dtype=float)
        arr_b = np.asarray(values_b, dtype=float)

        if pair_mode == "paired_suffix":
            rows_a = {str(row["pair_key"]): float(row["value"]) for row in rows if row["group"] == "A"}
            rows_b = {str(row["pair_key"]): float(row["value"]) for row in rows if row["group"] == "B"}
            shared = sorted(set(rows_a) & set(rows_b))
            if len(shared) < 2:
                return ToolResult(
                    tool_name=self.spec.name,
                    kind=job.kind,
                    payload={**payload, "rows": rows, "comparison_error": "Paired comparison needs at least 2 matched suffix pairs."},
                )
            paired_a = np.asarray([rows_a[key] for key in shared], dtype=float)
            paired_b = np.asarray([rows_b[key] for key in shared], dtype=float)
            diffs = paired_a - paired_b
            if diffs.size >= 3:
                shapiro = stats.shapiro(diffs)
                normality["paired_differences"] = {"statistic": float(shapiro.statistic), "pvalue": float(shapiro.pvalue)}
            if diffs.size >= 3 and normality["paired_differences"]["pvalue"] < float(payload["alpha"]):
                selected_test = "Wilcoxon signed-rank"
                test = stats.wilcoxon(paired_a, paired_b, zero_method="wilcox", correction=False)
            else:
                selected_test = "Paired t-test"
                test = stats.ttest_rel(paired_a, paired_b, nan_policy="omit")
            result_stats = {
                "matched_pairs": int(len(shared)),
                "statistic": float(test.statistic),
                "pvalue": float(test.pvalue),
                "delta_mean": float(np.mean(paired_a) - np.mean(paired_b)),
            }
        else:
            if arr_a.size >= 3:
                shapiro_a = stats.shapiro(arr_a)
                normality["A"] = {"statistic": float(shapiro_a.statistic), "pvalue": float(shapiro_a.pvalue)}
            if arr_b.size >= 3:
                shapiro_b = stats.shapiro(arr_b)
                normality["B"] = {"statistic": float(shapiro_b.statistic), "pvalue": float(shapiro_b.pvalue)}
            if arr_a.size >= 2 and arr_b.size >= 2:
                lev = stats.levene(arr_a, arr_b, center="median")
                variance = {"statistic": float(lev.statistic), "pvalue": float(lev.pvalue)}
            normal_pass = all(info.get("pvalue", 1.0) >= float(payload["alpha"]) for info in normality.values()) if normality else False
            equal_var = variance.get("pvalue", 0.0) >= float(payload["alpha"]) if variance else False
            if normal_pass:
                if equal_var:
                    selected_test = "Student t-test"
                    test = stats.ttest_ind(arr_a, arr_b, equal_var=True, nan_policy="omit")
                else:
                    selected_test = "Welch t-test"
                    test = stats.ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit")
            else:
                selected_test = "Mann-Whitney U"
                test = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
            result_stats = {
                "statistic": float(test.statistic),
                "pvalue": float(test.pvalue),
                "delta_mean": float(np.mean(arr_a) - np.mean(arr_b)),
            }

        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload={
                **payload,
                "rows": rows,
                "group_stats": group_stats,
                "normality": normality,
                "variance": variance,
                "selected_test": selected_test,
                "result_stats": result_stats,
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        if payload.get("comparison_error"):
            return str(payload["comparison_error"])
        metric = str(payload["metric"])
        lines = [
            f"ROI group comparison using metric [{metric}]",
            f"Group A prefix: [{payload['group_a_prefix']}]",
            f"Group B prefix: [{payload['group_b_prefix']}]",
            _format_group_descriptive("Group A", payload["group_stats"]["A"]),
            _format_group_descriptive("Group B", payload["group_stats"]["B"]),
        ]
        if payload["normality"]:
            lines.append("")
            lines.append("Normality checks:")
            for name, info in payload["normality"].items():
                lines.append(f"- {name}: Shapiro-Wilk W={info['statistic']:.4g}, p={info['pvalue']:.4g}")
        if payload["variance"]:
            lines.append(f"- Variance check: Levene statistic={payload['variance']['statistic']:.4g}, p={payload['variance']['pvalue']:.4g}")
        lines.extend(
            [
                "",
                f"Selected test: {payload['selected_test']}",
                f"Statistic={payload['result_stats']['statistic']:.4g}, p={payload['result_stats']['pvalue']:.4g}, "
                f"delta_mean={payload['result_stats']['delta_mean']:.4g}",
            ]
        )
        if payload["selected_test"] in {"Paired t-test", "Wilcoxon signed-rank"}:
            lines.append(f"Matched pairs={payload['result_stats']['matched_pairs']}")
        lines.extend(
            [
                "",
                "Each sample is one per-image ROI summary, which is the correct unit for group comparison.",
            ]
        )
        return "\n".join(lines)


class CompareImageGroupsTool:
    spec = ToolSpec(
        name="compare_image_groups",
        display_name="Compare Image Groups",
        category="statistics",
        description="Extract one whole-image metric per image for two groups, run descriptive stats and assumption checks, then compare the groups.",
        execution_mode="worker",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("group_a_prefix", "string", description="Prefix for group A image layers.", required=True),
            ParamSpec("group_b_prefix", "string", description="Prefix for group B image layers.", required=True),
            ParamSpec("metric", "string", description="Whole-image metric: mean, median, sum, std.", default="mean"),
            ParamSpec("pair_mode", "string", description="paired_suffix or unpaired.", default="paired_suffix"),
            ParamSpec("alpha", "float", description="Significance threshold.", default=0.05, minimum=1e-6, maximum=0.5),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Statistics"},
        provenance_metadata={"algorithm": "image_group_compare", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        prefix_a = _normalize_prefix(args.get("group_a_prefix"))
        prefix_b = _normalize_prefix(args.get("group_b_prefix"))
        if not prefix_a or not prefix_b:
            return "Image group comparison requires group_a_prefix and group_b_prefix."
        metric = str(args.get("metric", "mean")).strip().lower() or "mean"
        if metric not in {"mean", "median", "sum", "std"}:
            return f"Unsupported image metric [{metric}]. Use mean, median, sum, or std."
        pair_mode = str(args.get("pair_mode", "paired_suffix")).strip().lower() or "paired_suffix"
        if pair_mode not in {"paired_suffix", "unpaired"}:
            return f"Unsupported pair_mode [{pair_mode}]. Use paired_suffix or unpaired."

        images_a = _resolve_group_image_layers(ctx.viewer, prefix_a)
        images_b = _resolve_group_image_layers(ctx.viewer, prefix_b)
        if not images_a or not images_b:
            return (
                f"Could not resolve both image groups. group_a={len(images_a)} images for prefix [{prefix_a}], "
                f"group_b={len(images_b)} images for prefix [{prefix_b}]."
            )

        items: list[dict[str, object]] = []
        for group_name, prefix, layers in (("A", prefix_a, images_a), ("B", prefix_b, images_b)):
            for image_layer in layers:
                if getattr(image_layer, "rgb", False):
                    continue
                try:
                    image_plane = _current_image_plane(image_layer, ctx.viewer)
                except Exception:
                    continue
                items.append(
                    {
                        "group": group_name,
                        "pair_key": _strip_prefix_token(image_layer.name, prefix) or image_layer.name,
                        "image_layer_name": image_layer.name,
                        "image_data": np.asarray(image_plane).copy(),
                    }
                )
        if len(items) < 2:
            return "Not enough valid image layers were resolved for image group comparison."

        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "group_a_prefix": prefix_a,
                "group_b_prefix": prefix_b,
                "metric": metric,
                "pair_mode": pair_mode,
                "alpha": normalize_float(args.get("alpha", 0.05), 0.05, minimum=1e-6, maximum=0.5),
                "items": items,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        metric = str(payload["metric"])
        rows: list[dict[str, object]] = []
        for item in payload["items"]:
            values = np.asarray(item["image_data"], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            rows.append(
                {
                    "group": item["group"],
                    "pair_key": item["pair_key"],
                    "image_layer_name": item["image_layer_name"],
                    "metric": metric,
                    "value": _metric_value(values, metric),
                    "pixels": int(values.size),
                }
            )

        values_a = [float(row["value"]) for row in rows if row["group"] == "A"]
        values_b = [float(row["value"]) for row in rows if row["group"] == "B"]
        if len(values_a) < 2 or len(values_b) < 2:
            return ToolResult(
                tool_name=self.spec.name,
                kind=job.kind,
                payload={**payload, "rows": rows, "comparison_error": "Each group needs at least 2 valid image samples."},
            )

        group_stats = {"A": _group_descriptive(values_a), "B": _group_descriptive(values_b)}
        pair_mode = str(payload["pair_mode"])
        normality = {}
        variance = {}
        selected_test = "Welch t-test"
        result_stats: dict[str, object] = {}
        arr_a = np.asarray(values_a, dtype=float)
        arr_b = np.asarray(values_b, dtype=float)

        if pair_mode == "paired_suffix":
            rows_a = {str(row["pair_key"]): float(row["value"]) for row in rows if row["group"] == "A"}
            rows_b = {str(row["pair_key"]): float(row["value"]) for row in rows if row["group"] == "B"}
            shared = sorted(set(rows_a) & set(rows_b))
            if len(shared) < 2:
                return ToolResult(
                    tool_name=self.spec.name,
                    kind=job.kind,
                    payload={**payload, "rows": rows, "comparison_error": "Paired comparison needs at least 2 matched suffix pairs."},
                )
            paired_a = np.asarray([rows_a[key] for key in shared], dtype=float)
            paired_b = np.asarray([rows_b[key] for key in shared], dtype=float)
            diffs = paired_a - paired_b
            if diffs.size >= 3:
                shapiro = stats.shapiro(diffs)
                normality["paired_differences"] = {"statistic": float(shapiro.statistic), "pvalue": float(shapiro.pvalue)}
            if diffs.size >= 3 and normality["paired_differences"]["pvalue"] < float(payload["alpha"]):
                selected_test = "Wilcoxon signed-rank"
                test = stats.wilcoxon(paired_a, paired_b, zero_method="wilcox", correction=False)
            else:
                selected_test = "Paired t-test"
                test = stats.ttest_rel(paired_a, paired_b, nan_policy="omit")
            result_stats = {
                "matched_pairs": int(len(shared)),
                "statistic": float(test.statistic),
                "pvalue": float(test.pvalue),
                "delta_mean": float(np.mean(paired_a) - np.mean(paired_b)),
            }
        else:
            if arr_a.size >= 3:
                shapiro_a = stats.shapiro(arr_a)
                normality["A"] = {"statistic": float(shapiro_a.statistic), "pvalue": float(shapiro_a.pvalue)}
            if arr_b.size >= 3:
                shapiro_b = stats.shapiro(arr_b)
                normality["B"] = {"statistic": float(shapiro_b.statistic), "pvalue": float(shapiro_b.pvalue)}
            if arr_a.size >= 2 and arr_b.size >= 2:
                lev = stats.levene(arr_a, arr_b, center="median")
                variance = {"statistic": float(lev.statistic), "pvalue": float(lev.pvalue)}
            normal_pass = all(info.get("pvalue", 1.0) >= float(payload["alpha"]) for info in normality.values()) if normality else False
            equal_var = variance.get("pvalue", 0.0) >= float(payload["alpha"]) if variance else False
            if normal_pass:
                if equal_var:
                    selected_test = "Student t-test"
                    test = stats.ttest_ind(arr_a, arr_b, equal_var=True, nan_policy="omit")
                else:
                    selected_test = "Welch t-test"
                    test = stats.ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit")
            else:
                selected_test = "Mann-Whitney U"
                test = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
            result_stats = {
                "statistic": float(test.statistic),
                "pvalue": float(test.pvalue),
                "delta_mean": float(np.mean(arr_a) - np.mean(arr_b)),
            }

        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload={
                **payload,
                "rows": rows,
                "group_stats": group_stats,
                "normality": normality,
                "variance": variance,
                "selected_test": selected_test,
                "result_stats": result_stats,
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        if payload.get("comparison_error"):
            return str(payload["comparison_error"])
        metric = str(payload["metric"])
        lines = [
            f"Whole-image group comparison using metric [{metric}]",
            f"Group A prefix: [{payload['group_a_prefix']}]",
            f"Group B prefix: [{payload['group_b_prefix']}]",
            _format_group_descriptive("Group A", payload["group_stats"]["A"]),
            _format_group_descriptive("Group B", payload["group_stats"]["B"]),
        ]
        if payload["normality"]:
            lines.append("")
            lines.append("Normality checks:")
            for name, info in payload["normality"].items():
                lines.append(f"- {name}: Shapiro-Wilk W={info['statistic']:.4g}, p={info['pvalue']:.4g}")
        if payload["variance"]:
            lines.append(f"- Variance check: Levene statistic={payload['variance']['statistic']:.4g}, p={payload['variance']['pvalue']:.4g}")
        lines.extend(
            [
                "",
                f"Selected test: {payload['selected_test']}",
                f"Statistic={payload['result_stats']['statistic']:.4g}, p={payload['result_stats']['pvalue']:.4g}, "
                f"delta_mean={payload['result_stats']['delta_mean']:.4g}",
            ]
        )
        if payload["selected_test"] in {"Paired t-test", "Wilcoxon signed-rank"}:
            lines.append(f"Matched pairs={payload['result_stats']['matched_pairs']}")
        lines.extend(
            [
                "",
                "Each sample is one whole-image summary, not all pixels pooled together.",
            ]
        )
        return "\n".join(lines)


class SAMSegmentFromBoxTool:
    spec = ToolSpec(
        name="sam_segment_from_box",
        display_name="SAM Segment From Box",
        category="segmentation",
        description="Run SAM2 using a Shapes ROI prompt on a 2D grayscale image.",
        execution_mode="worker",
        supported_layer_types=("image", "shapes"),
        parameter_schema=(
            ParamSpec("image_layer", "string", description="Target 2D grayscale image layer."),
            ParamSpec("roi_layer", "string", description="Shapes ROI layer used as the prompt."),
            ParamSpec("shape_index", "int", description="Optional shape index.", default=0, minimum=0),
            ParamSpec("multimask_output", "bool", description="Reserved for future SAM2 multimask support.", default=False),
            ParamSpec("model_name", "string", description="Optional SAM2 wrapper/model identifier."),
        ),
        output_type="labels_layer",
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "sam2", "deterministic": False},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("image_layer"))
        if image_layer is None:
            return "No valid image layer available for SAM2 box segmentation."
        if getattr(image_layer, "rgb", False):
            return "SAM2 box segmentation currently supports grayscale 2D image layers, not RGB layers."
        image_data = np.asarray(image_layer.data)
        if image_data.ndim != 2:
            return f"SAM2 box segmentation currently supports 2D image layers only. Got ndim={image_data.ndim}."
        roi_layer = _resolve_shapes_layer(ctx.viewer, args.get("roi_layer"))
        if roi_layer is None:
            return "No valid Shapes ROI layer available for SAM2. Select or name a Shapes layer."
        requested_shape_index = args.get("shape_index")
        normalized_shape_index = None
        if requested_shape_index is not None:
            normalized_shape_index = normalize_int(
                requested_shape_index,
                default=0,
                minimum=0,
                maximum=max(0, len(getattr(roi_layer, "data", [])) - 1),
            )
        try:
            box_xyxy, used_shape_index = _shape_bbox_xyxy(roi_layer, normalized_shape_index)
        except Exception as exc:
            return str(exc)

        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "image_layer_name": image_layer.name,
                "roi_layer_name": roi_layer.name,
                "shape_index": used_shape_index,
                "box_xyxy": tuple(float(value) for value in box_xyxy),
                "output_name": next_output_name(ctx.viewer, f"{image_layer.name}_sam2"),
                "model_name": str(args.get("model_name") or "").strip() or None,
                "data": image_data.copy(),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        try:
            mask, backend_message = segment_image_from_box(
                np.asarray(payload["data"]),
                box_xyxy=tuple(float(value) for value in payload["box_xyxy"]),
                model_name=payload.get("model_name"),
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.spec.name,
                kind=job.kind,
                payload=payload,
                message=str(exc),
            )
        payload["result"] = mask
        payload["backend_message"] = backend_message
        payload["foreground_pixels"] = int(np.count_nonzero(mask))
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "parameters": {
                    "shape_index": payload["shape_index"],
                    "box_xyxy": payload["box_xyxy"],
                    "model_name": payload.get("model_name"),
                },
                "input_layer": payload["image_layer_name"],
                "roi_layer": payload["roi_layer_name"],
                "output_layer": payload["output_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        if "result" not in payload:
            return result.message or "SAM2 box segmentation did not produce a result."
        ctx.viewer.add_labels(
            payload["result"],
            name=payload["output_name"],
            scale=payload["scale"],
            translate=payload["translate"],
        )
        suffix = f" {payload['backend_message']}" if payload.get("backend_message") else ""
        return (
            f"Segmented [{payload['image_layer_name']}] with SAM2 using [{payload['roi_layer_name']}] "
            f"shape_index={payload['shape_index']} as [{payload['output_name']}]. "
            f"box_xyxy={payload['box_xyxy']} foreground_pixels={payload['foreground_pixels']}.{suffix}"
        )


class SAMSegmentFromPointsTool:
    spec = ToolSpec(
        name="sam_segment_from_points",
        display_name="SAM Segment From Points",
        category="segmentation",
        description="Run SAM2 using positive and negative point prompts on a 2D grayscale image.",
        execution_mode="worker",
        supported_layer_types=("image", "points"),
        parameter_schema=(
            ParamSpec("image_layer", "string", description="Target 2D grayscale image layer."),
            ParamSpec("points_layer", "string", description="Points prompt layer."),
            ParamSpec("multimask_output", "bool", description="Reserved for future SAM2 multimask support.", default=False),
            ParamSpec("model_name", "string", description="Optional SAM2 wrapper/model identifier."),
        ),
        output_type="labels_layer",
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "sam2", "deterministic": False},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("image_layer"))
        if image_layer is None:
            return "No valid image layer available for SAM2 point segmentation."
        if getattr(image_layer, "rgb", False):
            return "SAM2 point segmentation currently supports grayscale 2D image layers, not RGB layers."
        image_data = np.asarray(image_layer.data)
        if image_data.ndim != 2:
            return f"SAM2 point segmentation currently supports 2D image layers only. Got ndim={image_data.ndim}."
        points_layer = _resolve_points_layer(ctx.viewer, args.get("points_layer"))
        if points_layer is None:
            return "No valid Points prompt layer available for SAM2. Select or name a Points layer."
        try:
            point_coords_xy, point_labels = _points_prompt_xy_and_labels(points_layer)
        except Exception as exc:
            return str(exc)

        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "image_layer_name": image_layer.name,
                "points_layer_name": points_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{image_layer.name}_sam2_points"),
                "model_name": str(args.get("model_name") or "").strip() or None,
                "data": image_data.copy(),
                "point_coords_xy": point_coords_xy,
                "point_labels": point_labels,
                "positive_points": int(np.count_nonzero(point_labels == 1)),
                "negative_points": int(np.count_nonzero(point_labels == 0)),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        try:
            mask, backend_message = segment_image_from_points(
                np.asarray(payload["data"]),
                point_coords_xy=np.asarray(payload["point_coords_xy"], dtype=np.float32),
                point_labels=np.asarray(payload["point_labels"], dtype=np.int32),
                model_name=payload.get("model_name"),
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.spec.name,
                kind=job.kind,
                payload=payload,
                message=str(exc),
            )
        payload["result"] = mask
        payload["backend_message"] = backend_message
        payload["foreground_pixels"] = int(np.count_nonzero(mask))
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "parameters": {
                    "point_count": int(len(payload["point_labels"])),
                    "positive_points": payload["positive_points"],
                    "negative_points": payload["negative_points"],
                    "model_name": payload.get("model_name"),
                },
                "input_layer": payload["image_layer_name"],
                "points_layer": payload["points_layer_name"],
                "output_layer": payload["output_name"],
            },
        )

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        if "result" not in payload:
            return result.message or "SAM2 point segmentation did not produce a result."
        ctx.viewer.add_labels(
            payload["result"],
            name=payload["output_name"],
            scale=payload["scale"],
            translate=payload["translate"],
        )
        suffix = f" {payload['backend_message']}" if payload.get("backend_message") else ""
        return (
            f"Segmented [{payload['image_layer_name']}] with SAM2 using [{payload['points_layer_name']}] "
            f"positive_points={payload['positive_points']} negative_points={payload['negative_points']} "
            f"as [{payload['output_name']}]. foreground_pixels={payload['foreground_pixels']}.{suffix}"
        )


class SAMPropagatePoints3DTool:
    spec = ToolSpec(
        name="sam_propagate_points_3d",
        display_name="SAM Propagate Points 3D",
        category="segmentation",
        description="Run SAM2 propagation through a 3D grayscale volume from positive and negative points placed on one seed slice.",
        execution_mode="worker",
        supported_layer_types=("image", "points"),
        parameter_schema=(
            ParamSpec("image_layer", "string", description="Target 3D grayscale image layer."),
            ParamSpec("points_layer", "string", description="3D Points prompt layer placed on one seed slice."),
            ParamSpec("model_name", "string", description="Optional SAM2 wrapper/model identifier."),
        ),
        output_type="labels_layer",
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "sam2_video_propagation", "deterministic": False},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("image_layer"))
        if image_layer is None:
            return "No valid image layer available for 3D SAM2 propagation."
        if getattr(image_layer, "rgb", False):
            return "3D SAM2 propagation currently supports grayscale 3D image layers, not RGB layers."
        image_data = np.asarray(image_layer.data)
        if image_data.ndim != 3:
            return f"3D SAM2 propagation currently supports 3D image layers only. Got ndim={image_data.ndim}."
        points_layer = _resolve_points_layer(ctx.viewer, args.get("points_layer"))
        if points_layer is None:
            return "No valid Points prompt layer available for 3D SAM2 propagation."
        try:
            seed_slice, point_coords_xy, point_labels = _points_prompt_slice_xy_and_labels(points_layer)
        except Exception as exc:
            return str(exc)
        if seed_slice < 0 or seed_slice >= image_data.shape[0]:
            return f"Prompt seed slice {seed_slice} is out of range for image depth {image_data.shape[0]}."

        ready, status_message = get_sam2_backend_status()
        if not ready:
            return status_message

        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "image_layer_name": image_layer.name,
                "points_layer_name": points_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{image_layer.name}_sam2_propagated"),
                "model_name": str(args.get("model_name") or "").strip() or None,
                "data": image_data.copy(),
                "seed_slice": int(seed_slice),
                "point_coords_xy": point_coords_xy,
                "point_labels": point_labels,
                "positive_points": int(np.count_nonzero(point_labels == 1)),
                "negative_points": int(np.count_nonzero(point_labels == 0)),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        mask_volume, backend_message = propagate_volume_from_points(
            np.asarray(payload["data"]),
            seed_frame_idx=int(payload["seed_slice"]),
            point_coords_xy=np.asarray(payload["point_coords_xy"], dtype=np.float32),
            point_labels=np.asarray(payload["point_labels"], dtype=np.int32),
            model_name=payload.get("model_name"),
        )
        payload["result"] = mask_volume
        payload["backend_message"] = backend_message
        payload["foreground_voxels"] = int(np.count_nonzero(mask_volume))
        payload["tracked_slices"] = int(np.count_nonzero(np.any(mask_volume > 0, axis=(1, 2))))
        return ToolResult(
            tool_name=self.spec.name,
            kind=job.kind,
            payload=payload,
            provenance={
                "tool_name": self.spec.name,
                "parameters": {
                    "seed_slice": int(payload["seed_slice"]),
                    "point_count": int(len(payload["point_labels"])),
                    "positive_points": payload["positive_points"],
                    "negative_points": payload["negative_points"],
                    "model_name": payload.get("model_name"),
                },
                "input_layer": payload["image_layer_name"],
                "points_layer": payload["points_layer_name"],
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
        suffix = f" {payload['backend_message']}" if payload.get("backend_message") else ""
        return (
            f"Propagated SAM2 through 3D image [{payload['image_layer_name']}] using [{payload['points_layer_name']}] "
            f"seed_slice={payload['seed_slice']} positive_points={payload['positive_points']} "
            f"negative_points={payload['negative_points']} as [{payload['output_name']}]. "
            f"tracked_slices={payload['tracked_slices']} foreground_voxels={payload['foreground_voxels']}.{suffix}"
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
        ExtractAxonInteriorsTool(),
        CropToLayerBBoxTool(),
        ShowImageLayersInGridTool(),
        HideImageGridViewTool(),
        ShowLayersTool(),
        HideLayersTool(),
        ShowOnlyLayersTool(),
        ShowAllLayersTool(),
        ArrangeLayersForPresentationTool(),
        InspectROIContextTool(),
        MeasureShapesROIAreaTool(),
        ExtractROIValuesTool(),
        CompareROIGroupsTool(),
        CompareImageGroupsTool(),
        SAMSegmentFromBoxTool(),
        SAMSegmentFromPointsTool(),
        SAMPropagatePoints3DTool(),
        PlaceholderTool(
            ToolSpec(
                name="sam_refine_mask",
                display_name="SAM Refine Mask",
                category="segmentation",
                description="Refine an existing mask using Segment Anything.",
                execution_mode="worker",
                supported_layer_types=("image", "labels"),
                parameter_schema=(
                    ParamSpec("image_layer", "string", description="Target image layer."),
                    ParamSpec("mask_layer", "string", description="Existing mask or labels layer."),
                    ParamSpec("roi_layer", "string", description="Optional additional ROI prompt."),
                    ParamSpec("model_name", "string", description="Optional SAM backend/model identifier."),
                ),
                output_type="labels_layer",
                ui_metadata={"panel_group": "Segmentation"},
                provenance_metadata={"algorithm": "sam", "deterministic": False},
            )
        ),
        PlaceholderTool(
            ToolSpec(
                name="sam_auto_segment",
                display_name="SAM Auto Segment",
                category="segmentation",
                description="Run automatic Segment Anything mask generation on an image.",
                execution_mode="worker",
                supported_layer_types=("image",),
                parameter_schema=(
                    ParamSpec("image_layer", "string", description="Target image layer."),
                    ParamSpec("model_name", "string", description="Optional SAM backend/model identifier."),
                ),
                output_type="labels_layer",
                ui_metadata={"panel_group": "Segmentation"},
                provenance_metadata={"algorithm": "sam", "deterministic": False},
            )
        ),
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
