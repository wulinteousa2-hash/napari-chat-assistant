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
from skimage.segmentation import clear_border, watershed

from napari_chat_assistant.agent.context import find_any_layer, find_image_layer, find_labels_layer
from napari_chat_assistant.agent.image_ops import fill_holes, keep_largest_component, remove_small_components
from napari_chat_assistant.agent.sam2_backend import (
    get_sam2_backend_status,
    refine_mask_from_mask,
    propagate_volume_from_points,
    segment_image_auto,
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


def _default_text_annotation_style() -> dict[str, object]:
    return {
        "string": "{label}",
        "size": 12,
        "color": "yellow",
        "anchor": "upper_left",
        "translation": [0.0, -6.0],
        "blending": "translucent",
        "visible": True,
        "scaling": False,
        "rotation": 0.0,
    }


def _text_annotation_style_from_layer(layer: Points) -> dict[str, object]:
    metadata = dict(getattr(layer, "metadata", {}) or {})
    style = _default_text_annotation_style()
    saved = metadata.get("text_annotation_text_style")
    if isinstance(saved, dict):
        style.update(saved)
    return style


def _configure_text_annotation_layer(
    layer: Points,
    *,
    source_layer_name: str,
    current_text: str | None = None,
    style_updates: dict[str, object] | None = None,
) -> None:
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata["text_annotations_managed"] = True
    metadata["text_annotation_source_layer"] = source_layer_name
    style = _text_annotation_style_from_layer(layer)
    if style_updates:
        style.update(style_updates)
    metadata["text_annotation_text_style"] = style
    layer.metadata = metadata
    try:
        layer.text = dict(style)
    except Exception:
        pass
    if current_text is not None:
        try:
            layer.feature_defaults = {"label": str(current_text)}
        except Exception:
            pass
        try:
            layer.current_properties = {"label": np.asarray([str(current_text)], dtype=object)}
        except Exception:
            pass


def _text_annotation_labels(layer: Points) -> list[str]:
    features = getattr(layer, "features", None)
    if features is None:
        return []
    try:
        if "label" in features:
            return [str(value or "").strip() for value in list(features["label"])]
    except Exception:
        pass
    return []


def _text_annotation_layer_name(source_layer_name: str) -> str:
    base = str(source_layer_name or "").strip()
    return f"{base}_text_annotations" if base else "text_annotations"


def _callout_group_names(source_layer_name: str) -> dict[str, str]:
    base = str(source_layer_name or "").strip() or "annotations"
    return {
        "text": f"{base}_callout_text",
        "boxes": f"{base}_callout_boxes",
        "leaders": f"{base}_callout_lines",
    }


def _title_group_names(source_layer_name: str) -> dict[str, str]:
    base = str(source_layer_name or "").strip() or "annotations"
    return {
        "text": f"{base}_title_text",
        "box": f"{base}_title_box",
    }


def _default_callout_text_style() -> dict[str, object]:
    return {
        "string": "{label}",
        "size": 12,
        "color": "#f4f7fb",
        "anchor": "center",
        "translation": [0.0, 0.0],
        "blending": "translucent",
        "visible": True,
        "scaling": False,
        "rotation": 0.0,
    }


def _callout_text_style_from_layer(layer: Points) -> dict[str, object]:
    metadata = dict(getattr(layer, "metadata", {}) or {})
    style = _default_callout_text_style()
    saved = metadata.get("callout_text_style")
    if isinstance(saved, dict):
        style.update(saved)
    return style


def _configure_callout_text_layer(
    layer: Points,
    *,
    source_layer_name: str,
    style_updates: dict[str, object] | None = None,
) -> None:
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata["callout_annotations_managed"] = True
    metadata["callout_annotation_source_layer"] = source_layer_name
    style = _callout_text_style_from_layer(layer)
    if style_updates:
        style.update(style_updates)
    metadata["callout_text_style"] = style
    layer.metadata = metadata
    try:
        layer.text = dict(style)
    except Exception:
        pass


def _configure_callout_shapes_layer(layer: Shapes, *, source_layer_name: str, role: str) -> None:
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata["callout_annotations_managed"] = True
    metadata["callout_annotation_source_layer"] = source_layer_name
    metadata["callout_annotation_role"] = role
    layer.metadata = metadata


def _resolve_callout_text_layer(viewer, layer_name: object | None = None):
    name = str(layer_name or "").strip()
    if name:
        layer = find_any_layer(viewer, name)
        if isinstance(layer, Points):
            metadata = dict(getattr(layer, "metadata", {}) or {})
            if bool(metadata.get("callout_annotations_managed")):
                return layer
        return None
    selected = viewer.layers.selection.active if viewer is not None else None
    if isinstance(selected, Points) and bool(dict(getattr(selected, "metadata", {}) or {}).get("callout_annotations_managed")):
        return selected
    for layer in viewer.layers if viewer is not None else []:
        metadata = dict(getattr(layer, "metadata", {}) or {})
        if isinstance(layer, Points) and bool(metadata.get("callout_annotations_managed")):
            return layer
    return None


def _ensure_callout_annotation_layers(
    viewer,
    *,
    source_layer_name: object | None = None,
) -> tuple[Points, Shapes, Shapes, object]:
    anchor_layer = _resolve_text_annotation_anchor_layer(viewer, source_layer_name)
    if anchor_layer is None:
        raise ValueError("No usable 2D source layer is available for callout annotations.")
    if int(getattr(anchor_layer, "ndim", 0) or 0) != 2:
        raise ValueError("Callout annotations currently support 2D source layers only.")
    names = _callout_group_names(str(getattr(anchor_layer, "name", "") or ""))

    existing_text = find_any_layer(viewer, names["text"])
    if existing_text is not None and not isinstance(existing_text, Points):
        raise ValueError(f"Layer [{names['text']}] already exists and is not a Points layer.")
    if isinstance(existing_text, Points):
        text_layer = existing_text
    else:
        text_layer = viewer.add_points(
            np.empty((0, 2), dtype=np.float32),
            name=names["text"],
            features={"label": np.empty((0,), dtype=object)},
            size=6,
            face_color="transparent",
            border_color="transparent",
            border_width=0,
        )

    existing_boxes = find_any_layer(viewer, names["boxes"])
    if existing_boxes is not None and not isinstance(existing_boxes, Shapes):
        raise ValueError(f"Layer [{names['boxes']}] already exists and is not a Shapes layer.")
    if isinstance(existing_boxes, Shapes):
        boxes_layer = existing_boxes
    else:
        boxes_layer = viewer.add_shapes(
            [],
            shape_type="rectangle",
            name=names["boxes"],
            edge_width=2.0,
            edge_color="#f4f7fb",
            face_color=[1.0, 1.0, 1.0, 0.12],
        )

    existing_leaders = find_any_layer(viewer, names["leaders"])
    if existing_leaders is not None and not isinstance(existing_leaders, Shapes):
        raise ValueError(f"Layer [{names['leaders']}] already exists and is not a Shapes layer.")
    if isinstance(existing_leaders, Shapes):
        leaders_layer = existing_leaders
    else:
        leaders_layer = viewer.add_shapes(
            [],
            shape_type="line",
            name=names["leaders"],
            edge_width=1.5,
            edge_color="#d9e4f2",
            face_color="transparent",
        )

    source_name = str(getattr(anchor_layer, "name", "") or "")
    _configure_callout_text_layer(text_layer, source_layer_name=source_name)
    _configure_callout_shapes_layer(boxes_layer, source_layer_name=source_name, role="boxes")
    _configure_callout_shapes_layer(leaders_layer, source_layer_name=source_name, role="leaders")
    return text_layer, boxes_layer, leaders_layer, anchor_layer


def _ensure_title_annotation_layers(
    viewer,
    *,
    source_layer_name: object | None = None,
) -> tuple[Points, Shapes, object]:
    anchor_layer = _resolve_text_annotation_anchor_layer(viewer, source_layer_name)
    if anchor_layer is None:
        raise ValueError("No usable 2D source layer is available for title annotations.")
    if int(getattr(anchor_layer, "ndim", 0) or 0) != 2:
        raise ValueError("Title annotations currently support 2D source layers only.")
    names = _title_group_names(str(getattr(anchor_layer, "name", "") or ""))

    existing_text = find_any_layer(viewer, names["text"])
    if existing_text is not None and not isinstance(existing_text, Points):
        raise ValueError(f"Layer [{names['text']}] already exists and is not a Points layer.")
    if isinstance(existing_text, Points):
        text_layer = existing_text
    else:
        text_layer = viewer.add_points(
            np.empty((0, 2), dtype=np.float32),
            name=names["text"],
            features={"label": np.empty((0,), dtype=object)},
            size=6,
            face_color="transparent",
            border_color="transparent",
            border_width=0,
        )

    existing_box = find_any_layer(viewer, names["box"])
    if existing_box is not None and not isinstance(existing_box, Shapes):
        raise ValueError(f"Layer [{names['box']}] already exists and is not a Shapes layer.")
    if isinstance(existing_box, Shapes):
        box_layer = existing_box
    else:
        box_layer = viewer.add_shapes(
            [],
            shape_type="rectangle",
            name=names["box"],
            edge_width=2.2,
            edge_color="#f4f7fb",
            face_color=[1.0, 1.0, 1.0, 0.14],
        )

    source_name = str(getattr(anchor_layer, "name", "") or "")
    _configure_callout_text_layer(text_layer, source_layer_name=source_name)
    _configure_callout_shapes_layer(box_layer, source_layer_name=source_name, role="title_box")
    return text_layer, box_layer, anchor_layer


def _text_box_size(text: str, *, size: float) -> tuple[float, float]:
    cleaned = str(text or "").strip() or "Label"
    box_height = max(18.0, float(size) * 1.8)
    box_width = max(34.0, float(size) * (1.6 + 0.68 * len(cleaned)))
    return float(box_height), float(box_width)


def _rectangle_from_center(center_y: float, center_x: float, *, height: float, width: float) -> np.ndarray:
    half_h = float(height) / 2.0
    half_w = float(width) / 2.0
    return np.asarray(
        [
            [center_y - half_h, center_x - half_w],
            [center_y - half_h, center_x + half_w],
            [center_y + half_h, center_x + half_w],
            [center_y + half_h, center_x - half_w],
        ],
        dtype=np.float32,
    )


def _box_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float], *, margin: float = 6.0) -> bool:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    return not (ax1 + margin < bx0 or bx1 + margin < ax0 or ay1 + margin < by0 or by1 + margin < ay0)


def _fit_callout_box(
    centroid_y: float,
    centroid_x: float,
    *,
    image_shape: tuple[int, int],
    box_height: float,
    box_width: float,
    side: int,
    occupied_boxes: list[tuple[float, float, float, float]],
    base_offset: float,
) -> tuple[float, float, tuple[float, float, float, float]]:
    image_height = float(image_shape[0])
    image_width = float(image_shape[1])
    center_x = centroid_x + float(side) * (float(base_offset) + box_width / 2.0)
    center_x = float(np.clip(center_x, box_width / 2.0 + 2.0, image_width - box_width / 2.0 - 2.0))
    base_center_y = float(np.clip(centroid_y, box_height / 2.0 + 2.0, image_height - box_height / 2.0 - 2.0))
    step = max(8.0, box_height + 6.0)
    candidate_offsets = [0.0]
    for ring in range(1, 24):
        candidate_offsets.append(ring * step)
        candidate_offsets.append(-ring * step)
    for delta_y in candidate_offsets:
        center_y = float(np.clip(base_center_y + delta_y, box_height / 2.0 + 2.0, image_height - box_height / 2.0 - 2.0))
        bbox = (
            center_y - box_height / 2.0,
            center_y + box_height / 2.0,
            center_x - box_width / 2.0,
            center_x + box_width / 2.0,
        )
        if not any(_box_overlap(bbox, existing) for existing in occupied_boxes):
            return center_y, center_x, bbox
    bbox = (
        base_center_y - box_height / 2.0,
        base_center_y + box_height / 2.0,
        center_x - box_width / 2.0,
        center_x + box_width / 2.0,
    )
    return base_center_y, center_x, bbox


def _callout_entry_geometry(
    *,
    text: str,
    centroid_y: float,
    centroid_x: float,
    bbox_y0: float,
    bbox_y1: float,
    bbox_x0: float,
    bbox_x1: float,
    image_shape: tuple[int, int],
    occupied_boxes: list[tuple[float, float, float, float]],
    size: float,
) -> dict[str, object]:
    box_height, box_width = _text_box_size(text, size=size)
    object_width = max(1.0, float(bbox_x1) - float(bbox_x0))
    side = 1 if centroid_x <= float(image_shape[1]) * 0.58 else -1
    base_offset = max(16.0, object_width * 0.8 + 12.0)
    center_y, center_x, bbox = _fit_callout_box(
        centroid_y,
        centroid_x,
        image_shape=image_shape,
        box_height=box_height,
        box_width=box_width,
        side=side,
        occupied_boxes=occupied_boxes,
        base_offset=base_offset,
    )
    occupied_boxes.append(bbox)
    rectangle = _rectangle_from_center(center_y, center_x, height=box_height, width=box_width)
    line_end_x = center_x - side * (box_width / 2.0)
    line = np.asarray([[centroid_y, centroid_x], [center_y, line_end_x]], dtype=np.float32)
    return {
        "text": text,
        "text_position": [center_y, center_x],
        "box_shape": rectangle.tolist(),
        "leader_shape": line.tolist(),
    }

def _resolve_text_annotation_anchor_layer(viewer, source_layer_name: object | None = None):
    name = str(source_layer_name or "").strip()
    if name:
        layer = find_any_layer(viewer, name)
        if layer is not None and int(getattr(layer, "ndim", 0) or 0) in {2, 3}:
            return layer
        return None
    selected = viewer.layers.selection.active if viewer is not None else None
    if selected is not None and int(getattr(selected, "ndim", 0) or 0) in {2, 3}:
        return selected
    for layer in viewer.layers if viewer is not None else []:
        if isinstance(layer, Image) and int(getattr(layer, "ndim", 0) or 0) in {2, 3}:
            return layer
    return None


def _resolve_text_annotation_layer(viewer, annotation_layer: object | None = None):
    name = str(annotation_layer or "").strip()
    if name:
        layer = find_any_layer(viewer, name)
        if isinstance(layer, Points):
            return layer
        return None
    selected = viewer.layers.selection.active if viewer is not None else None
    if isinstance(selected, Points) and bool(dict(getattr(selected, "metadata", {}) or {}).get("text_annotations_managed")):
        return selected
    for layer in viewer.layers if viewer is not None else []:
        metadata = dict(getattr(layer, "metadata", {}) or {})
        if isinstance(layer, Points) and bool(metadata.get("text_annotations_managed")):
            return layer
    return None


def _ensure_text_annotation_layer(
    viewer,
    *,
    source_layer_name: object | None = None,
    annotation_layer_name: object | None = None,
) -> tuple[Points, object]:
    anchor_layer = _resolve_text_annotation_anchor_layer(viewer, source_layer_name)
    if anchor_layer is None:
        raise ValueError("No usable 2D or 3D source layer is available for text annotations.")
    requested_name = str(annotation_layer_name or "").strip()
    layer_name = requested_name or _text_annotation_layer_name(str(getattr(anchor_layer, "name", "") or ""))
    existing = find_any_layer(viewer, layer_name)
    if existing is not None and not isinstance(existing, Points):
        raise ValueError(f"Layer [{layer_name}] already exists and is not a Points layer.")
    if isinstance(existing, Points):
        layer = existing
    else:
        ndim = int(getattr(anchor_layer, "ndim", 2) or 2)
        layer = viewer.add_points(
            np.empty((0, ndim), dtype=np.float32),
            name=layer_name,
            features={"label": np.empty((0,), dtype=object)},
            size=6,
            face_color="transparent",
            border_color="#ffd54f",
            border_width=1,
        )
    _configure_text_annotation_layer(layer, source_layer_name=str(getattr(anchor_layer, "name", "") or ""))
    return layer, anchor_layer


def _normalize_text_annotation_position(value: object, *, ndim: int) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        coords = [float(part) for part in parts]
    elif isinstance(value, (list, tuple, np.ndarray)):
        coords = [float(v) for v in list(value)]
    else:
        coords = []
    if ndim == 2:
        if len(coords) != 2:
            raise ValueError("2D text annotations need position=[x, y].")
        x, y = coords
        return (float(y), float(x))
    if ndim == 3:
        if len(coords) == 3:
            z, y, x = coords
            return (float(z), float(y), float(x))
        raise ValueError("3D text annotations need position=[z, y, x].")
    raise ValueError("Text annotations currently support 2D or 3D layers only.")


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


def _synthetic_2d_gray(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = 256, 256
    yy, xx = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")
    image = np.zeros((y, x), dtype=np.float32)
    for _ in range(12):
        cy = rng.uniform(24, y - 24)
        cx = rng.uniform(24, x - 24)
        ry = rng.uniform(10.0, 28.0)
        rx = rng.uniform(10.0, 28.0)
        amp = rng.uniform(0.4, 1.0)
        blob = np.exp(-(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) * 1.6).astype(np.float32)
        image += amp * blob
    image += 0.10 * rng.normal(size=image.shape).astype(np.float32)
    image = ndi.gaussian_filter(image, sigma=1.1)
    image -= image.min()
    image /= image.max() + 1e-8
    return image.astype(np.float32)


def _synthetic_3d_gray(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z, y, x = 24, 128, 128
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    volume = np.zeros((z, y, x), dtype=np.float32)
    for _ in range(10):
        cz = rng.uniform(3, z - 3)
        cy = rng.uniform(16, y - 16)
        cx = rng.uniform(16, x - 16)
        rz = rng.uniform(1.5, 3.5)
        ry = rng.uniform(8.0, 18.0)
        rx = rng.uniform(8.0, 18.0)
        amp = rng.uniform(0.4, 1.0)
        blob = np.exp(-(((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) * 1.6).astype(np.float32)
        volume += amp * blob
    volume += 0.08 * rng.normal(size=volume.shape).astype(np.float32)
    volume = ndi.gaussian_filter(volume, sigma=(0.6, 1.0, 1.0))
    volume -= volume.min()
    volume /= volume.max() + 1e-8
    return volume.astype(np.float32)


def _synthetic_2d_rgb(seed: int = 7) -> np.ndarray:
    base = _synthetic_2d_gray(seed=seed)
    rgb = np.stack(
        [
            np.clip(base * 0.95, 0.0, 1.0),
            np.clip(np.roll(base, shift=12, axis=0) * 0.80, 0.0, 1.0),
            np.clip(np.roll(base, shift=10, axis=1) * 0.70, 0.0, 1.0),
        ],
        axis=-1,
    )
    return rgb.astype(np.float32)


def _synthetic_3d_rgb(seed: int = 7) -> np.ndarray:
    base = _synthetic_3d_gray(seed=seed)
    rgb = np.stack(
        [
            np.clip(base * 0.95, 0.0, 1.0),
            np.clip(np.roll(base, shift=2, axis=0) * 0.80, 0.0, 1.0),
            np.clip(np.roll(base, shift=6, axis=2) * 0.70, 0.0, 1.0),
        ],
        axis=-1,
    )
    return rgb.astype(np.float32)


def _synthetic_demo_payload(variant: str, seed: int = 7) -> tuple[np.ndarray, str]:
    normalized = str(variant or "2d_gray").strip().lower()
    if normalized == "3d_gray":
        return _synthetic_3d_gray(seed=seed), "synthetic_demo_3d_gray"
    if normalized == "2d_rgb":
        return _synthetic_2d_rgb(seed=seed), "synthetic_demo_2d_rgb"
    if normalized == "3d_rgb":
        return _synthetic_3d_rgb(seed=seed), "synthetic_demo_3d_rgb"
    return _synthetic_2d_gray(seed=seed), "synthetic_demo_2d_gray"


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


def _resolve_montage_image_layers(viewer, layer_names: object | None = None) -> list[Image]:
    resolved: list[Image] = []
    names = _parse_layer_names_argument(layer_names)
    if names:
        for name in names:
            layer = find_any_layer(viewer, name)
            if isinstance(layer, Image) and not getattr(layer, "rgb", False) and np.asarray(layer.data).ndim == 2:
                resolved.append(layer)
        return resolved

    selected_layers = list(getattr(viewer.layers.selection, "_selected", []) or [])
    selected_layers = [
        layer for layer in selected_layers
        if isinstance(layer, Image) and not getattr(layer, "rgb", False) and np.asarray(layer.data).ndim == 2
    ]
    if len(selected_layers) >= 2:
        return selected_layers

    return [
        layer for layer in viewer.layers
        if isinstance(layer, Image) and not getattr(layer, "rgb", False) and np.asarray(layer.data).ndim == 2
    ]


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


def _normalize_layer_type(value: object) -> str:
    layer_type = str(value or "").strip().lower()
    aliases = {
        "shape": "shapes",
        "shapes": "shapes",
        "roi": "shapes",
        "image": "image",
        "images": "image",
        "label": "labels",
        "labels": "labels",
        "mask": "labels",
        "masks": "labels",
        "point": "points",
        "points": "points",
    }
    return aliases.get(layer_type, "")


def _layer_matches_type(layer: object, layer_type: str) -> bool:
    normalized = _normalize_layer_type(layer_type)
    if normalized == "image":
        return isinstance(layer, Image)
    if normalized == "labels":
        return isinstance(layer, Labels)
    if normalized == "shapes":
        return isinstance(layer, Shapes)
    if normalized == "points":
        return isinstance(layer, Points)
    return False


def _resolve_layers_by_type(viewer, layer_type: str) -> list[object]:
    normalized = _normalize_layer_type(layer_type)
    if not normalized or viewer is None:
        return []
    return [layer for layer in viewer.layers if _layer_matches_type(layer, normalized)]


def _apply_mask_operation(data, *, op_name: str, radius: int = 1, min_size: int = 64) -> np.ndarray:
    op = str(op_name or "").strip().lower()
    binary = np.asarray(data) > 0
    if op == "dilate":
        return ndi.binary_dilation(binary, structure=disk(max(1, int(radius)))).astype(np.asarray(data).dtype)
    if op == "erode":
        return ndi.binary_erosion(binary, structure=disk(max(1, int(radius)))).astype(np.asarray(data).dtype)
    if op == "open":
        return ndi.binary_opening(binary, structure=disk(max(1, int(radius)))).astype(np.asarray(data).dtype)
    if op == "close":
        return ndi.binary_closing(binary, structure=disk(max(1, int(radius)))).astype(np.asarray(data).dtype)
    if op == "convert_to_mask":
        return binary.astype(np.asarray(data).dtype)
    if op == "median":
        return (ndi.median_filter(binary.astype(np.uint8), footprint=disk(max(1, int(radius)))) > 0).astype(np.asarray(data).dtype)
    if op == "outline":
        return (binary & ~ndi.binary_erosion(binary, structure=ndi.generate_binary_structure(binary.ndim, 1))).astype(
            np.asarray(data).dtype
        )
    if op == "fill_holes":
        return fill_holes(data)
    if op == "skeletonize":
        from skimage.morphology import skeletonize

        return skeletonize(binary).astype(np.asarray(data).dtype)
    if op == "distance_map":
        return ndi.distance_transform_edt(binary).astype(np.float32, copy=False)
    if op == "ultimate_points":
        from skimage.morphology import local_maxima

        distance = ndi.distance_transform_edt(binary)
        return (local_maxima(distance) & binary).astype(np.int32, copy=False)
    if op == "watershed":
        from skimage.morphology import local_maxima

        distance = ndi.distance_transform_edt(binary)
        maxima = local_maxima(distance) & binary
        markers, _ = ndi.label(maxima)
        if int(np.max(markers)) == 0:
            markers, _ = ndi.label(binary)
        return watershed(-distance, markers=markers, mask=binary).astype(np.int32, copy=False)
    if op == "voronoi":
        markers, object_count = ndi.label(binary)
        if object_count == 0:
            return np.zeros_like(np.asarray(data), dtype=np.int32)
        distance = ndi.distance_transform_edt(~binary)
        return watershed(distance, markers=markers, mask=np.ones_like(binary, dtype=bool)).astype(np.int32, copy=False)
    if op == "remove_small":
        return remove_small_components(data, min_size=max(1, int(min_size)))
    if op == "keep_largest":
        return keep_largest_component(data)
    raise ValueError(f"Unsupported mask operation: {op_name}")


def _normalize_grid_dimensions(count: int, rows: int, columns: int) -> tuple[int, int]:
    count = max(1, int(count))
    rows = max(0, int(rows))
    columns = max(0, int(columns))
    if rows <= 0 and columns <= 0:
        return _auto_grid_shape(count)
    if rows <= 0:
        rows = max(1, int(np.ceil(count / max(1, columns))))
    if columns <= 0:
        columns = max(1, int(np.ceil(count / max(1, rows))))
    if rows * columns < count:
        raise ValueError(f"Grid {rows}x{columns} is too small for {count} image layers.")
    return rows, columns


def _build_montage_canvas(
    layers: list[Image],
    *,
    rows: int,
    columns: int,
    spacing: int,
    background_value: float,
) -> tuple[np.ndarray, list[dict[str, object]], tuple[int, int]]:
    if not layers:
        raise ValueError("Need at least one image layer to build a montage canvas.")
    source_arrays = [np.asarray(layer.data) for layer in layers]
    max_height = max(int(array.shape[0]) for array in source_arrays)
    max_width = max(int(array.shape[1]) for array in source_arrays)
    tile_height = max(1, max_height)
    tile_width = max(1, max_width)
    canvas_height = rows * tile_height + max(0, rows - 1) * spacing
    canvas_width = columns * tile_width + max(0, columns - 1) * spacing
    canvas = np.full((canvas_height, canvas_width), fill_value=float(background_value), dtype=np.float32)
    placements: list[dict[str, object]] = []
    for index, (layer, array) in enumerate(zip(layers, source_arrays)):
        row = index // columns
        col = index % columns
        tile_y0 = row * (tile_height + spacing)
        tile_x0 = col * (tile_width + spacing)
        height = int(array.shape[0])
        width = int(array.shape[1])
        content_y0 = tile_y0 + max(0, (tile_height - height) // 2)
        content_x0 = tile_x0 + max(0, (tile_width - width) // 2)
        content_y1 = content_y0 + height
        content_x1 = content_x0 + width
        canvas[content_y0:content_y1, content_x0:content_x1] = np.asarray(array, dtype=np.float32)
        placements.append(
            {
                "source_layer": layer.name,
                "source_kind": "image",
                "source_shape": [height, width],
                "source_dtype": str(getattr(array, "dtype", "")),
                "tile_index": index,
                "grid_position": [row, col],
                "canvas_bbox": {
                    "y0": int(tile_y0),
                    "y1": int(tile_y0 + tile_height),
                    "x0": int(tile_x0),
                    "x1": int(tile_x0 + tile_width),
                },
                "content_bbox": {
                    "y0": int(content_y0),
                    "y1": int(content_y1),
                    "x0": int(content_x0),
                    "x1": int(content_x1),
                },
                "padding": {
                    "top": int(content_y0 - tile_y0),
                    "bottom": int(tile_y0 + tile_height - content_y1),
                    "left": int(content_x0 - tile_x0),
                    "right": int(tile_x0 + tile_width - content_x1),
                },
            }
        )
    return canvas, placements, (tile_height, tile_width)


def _montage_metadata(
    *,
    montage_id: str,
    purpose: str,
    placements: list[dict[str, object]],
    rows: int,
    columns: int,
    spacing: int,
    tile_size: tuple[int, int],
    canvas_shape: tuple[int, int],
    background_value: float,
    linked_outputs: dict[str, str],
) -> dict[str, object]:
    return {
        "version": 1,
        "montage_id": str(montage_id),
        "purpose": str(purpose),
        "layout": {
            "mode": "grid",
            "rows": int(rows),
            "columns": int(columns),
            "spacing": int(spacing),
            "tile_size": [int(tile_size[0]), int(tile_size[1])],
            "size_mode": "pad_to_max",
            "alignment": "center",
            "fill_order": "row_major",
        },
        "canvas_shape": [int(canvas_shape[0]), int(canvas_shape[1])],
        "canvas_dtype": "float32",
        "background_value": float(background_value),
        "created_from": placements,
        "linked_outputs": dict(linked_outputs),
    }


def _montage_metadata_from_layer(layer) -> dict[str, object] | None:
    metadata = dict(getattr(layer, "metadata", {}) or {})
    montage = metadata.get("montage_canvas")
    return dict(montage) if isinstance(montage, dict) else None


def _resolve_annotation_layer(viewer, annotation_layer_name: object | None = None):
    name = str(annotation_layer_name or "").strip()
    if name:
        layer = find_any_layer(viewer, name)
        return layer if isinstance(layer, (Labels, Points)) else None
    selected = viewer.layers.selection.active if viewer is not None else None
    if isinstance(selected, (Labels, Points)):
        return selected
    for layer in viewer.layers if viewer is not None else []:
        if isinstance(layer, (Labels, Points)):
            return layer
    return None


def _resolve_montage_reference_layer(viewer, montage_layer_name: object | None = None, annotation_layer=None):
    name = str(montage_layer_name or "").strip()
    if name:
        layer = find_any_layer(viewer, name)
        return layer if _montage_metadata_from_layer(layer) is not None else None
    if annotation_layer is not None and _montage_metadata_from_layer(annotation_layer) is not None:
        return annotation_layer
    selected = viewer.layers.selection.active if viewer is not None else None
    if _montage_metadata_from_layer(selected) is not None:
        return selected
    candidates = [layer for layer in viewer.layers if _montage_metadata_from_layer(layer) is not None] if viewer is not None else []
    if not candidates:
        return None
    image_like = [layer for layer in candidates if isinstance(layer, Image)]
    return image_like[0] if image_like else candidates[0]


def _placement_content_bbox(placement: dict[str, object]) -> tuple[int, int, int, int]:
    bbox = dict(placement.get("content_bbox", {}) or placement.get("canvas_bbox", {}) or {})
    return (
        int(bbox.get("y0", 0)),
        int(bbox.get("y1", 0)),
        int(bbox.get("x0", 0)),
        int(bbox.get("x1", 0)),
    )


def _source_layer_transform(viewer, source_layer_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    source_layer = find_any_layer(viewer, source_layer_name)
    if source_layer is None:
        return (), ()
    return tuple(getattr(source_layer, "scale", ())), tuple(getattr(source_layer, "translate", ()))


def _slice_point_features(points_layer: Points, indices: list[int]):
    features = getattr(points_layer, "features", None)
    if features is None:
        return None
    try:
        return features.iloc[indices].reset_index(drop=True)
    except Exception:
        try:
            return {key: np.asarray(value)[indices] for key, value in dict(features).items()}
        except Exception:
            return None


class ShowImageLayersInGridTool:
    spec = ToolSpec(
        name="show_image_layers_in_grid",
        display_name="Quick Compare Grid",
        category="grid_compare",
        description="Use napari grid view to tile image layers for side-by-side comparison without moving layer data.",
        execution_mode="immediate",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("layer_names", "string_list", description="Optional ordered image layer names to show."),
            ParamSpec("spacing", "float", description="Optional grid spacing.", default=0.0, minimum=0.0),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Grid Compare"},
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
        display_name="Turn Off Compare Grid",
        category="grid_compare",
        description="Turn off napari grid view and return to the normal overlapping layer view.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "points", "shapes"),
        parameter_schema=(),
        output_type="message",
        ui_metadata={"panel_group": "Grid Compare"},
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


class HideAllLayersTool:
    spec = ToolSpec(
        name="hide_all_layers",
        display_name="Hide All Layers",
        category="visualization",
        description="Hide every layer in the current viewer.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        parameter_schema=(),
        output_type="message",
        ui_metadata={"panel_group": "Visualization"},
        provenance_metadata={"algorithm": "layer_visibility", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        if len(ctx.viewer.layers) == 0:
            return "No layers are open to hide."
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
            layer.visible = False
        return f"Hid all {len(ctx.viewer.layers)} layer(s)."


class DeleteAllLayersTool:
    spec = ToolSpec(
        name="delete_all_layers",
        display_name="Delete All Layers",
        category="viewer_editing",
        description="Delete every layer in the current viewer.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        parameter_schema=(),
        output_type="message",
        ui_metadata={"panel_group": "Viewer Editing"},
        provenance_metadata={"algorithm": "layer_deletion", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        if len(ctx.viewer.layers) == 0:
            return "No layers are open to delete."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={"kind": self.spec.name, "layer_names": [str(layer.name) for layer in ctx.viewer.layers]},
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        removed_names: list[str] = []
        for layer_name in _parse_layer_names_argument(result.payload.get("layer_names", [])):
            if find_any_layer(ctx.viewer, layer_name) is None:
                continue
            ctx.viewer.layers.remove(ctx.viewer.layers[layer_name])
            removed_names.append(layer_name)
        if not removed_names:
            return "No layers were deleted."
        removed = ", ".join(f"[{name}]" for name in removed_names)
        return f"Deleted all {len(removed_names)} layer(s): {removed}."


class DeleteLayersTool:
    spec = ToolSpec(
        name="delete_layers",
        display_name="Delete Layers",
        category="viewer_editing",
        description="Delete specific layers by name or delete all layers of a given type.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        parameter_schema=(
            ParamSpec("layer_names", "string_list", description="Layer names to delete."),
            ParamSpec("layer_type", "string", description="Optional layer type filter: image, labels, shapes, or points."),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Viewer Editing"},
        provenance_metadata={"algorithm": "layer_deletion", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        names = _parse_layer_names_argument(args.get("layer_names"))
        requested_type = _normalize_layer_type(args.get("layer_type"))
        if names:
            layers = _resolve_existing_layers(ctx.viewer, names)
        elif requested_type:
            layers = _resolve_layers_by_type(ctx.viewer, requested_type)
        else:
            layers = []
        if not layers:
            if requested_type:
                return f"No matching [{requested_type}] layers were found to delete."
            return "Provide one or more layer names, or a layer_type such as shapes, to delete."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "layer_names": [layer.name for layer in layers],
                "layer_type": requested_type,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        layers = _resolve_existing_layers(ctx.viewer, _parse_layer_names_argument(result.payload.get("layer_names", [])))
        if not layers:
            requested_type = _normalize_layer_type(result.payload.get("layer_type"))
            if requested_type:
                return f"No usable [{requested_type}] layers were available to delete."
            return "No usable layers were available to delete."
        removed_names: list[str] = []
        for layer in list(layers):
            layer_name = layer.name
            if find_any_layer(ctx.viewer, layer_name) is None:
                continue
            ctx.viewer.layers.remove(ctx.viewer.layers[layer_name])
            removed_names.append(layer_name)
        if not removed_names:
            return "No layers were deleted."
        removed = ", ".join(f"[{name}]" for name in removed_names)
        requested_type = _normalize_layer_type(result.payload.get("layer_type"))
        if requested_type and len(removed_names) == len(layers):
            return f"Deleted {len(removed_names)} [{requested_type}] layer(s): {removed}."
        return f"Deleted {len(removed_names)} layer(s): {removed}."


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
            selected = ctx.viewer.layers.selection.active if ctx.viewer is not None else None
            if selected is not None:
                names = [str(selected.name)]
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


class ShowAllExceptLayersTool:
    spec = ToolSpec(
        name="show_all_except_layers",
        display_name="Show All Except Layers",
        category="visualization",
        description="Show every layer except the specified layers, which will be hidden.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        parameter_schema=(
            ParamSpec("layer_names", "string_list", description="Layer names to keep hidden while all others are shown."),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Visualization"},
        provenance_metadata={"algorithm": "layer_visibility", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        names = _parse_layer_names_argument((arguments or {}).get("layer_names"))
        if not names:
            return "Provide at least one layer name to keep hidden."
        layers = _resolve_existing_layers(ctx.viewer, names)
        if not layers:
            return "No matching layers were found to keep hidden."
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
            return "No usable layers were available to keep hidden."
        hidden_names = {layer.name for layer in layers}
        for layer in ctx.viewer.layers:
            layer.visible = layer.name not in hidden_names
        hidden = ", ".join(f"[{layer.name}]" for layer in layers)
        return f"Showed all layers except {len(layers)} layer(s): {hidden}."


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


class SetLayerScaleTool:
    spec = ToolSpec(
        name="set_layer_scale",
        display_name="Set Layer Scale",
        category="viewer_editing",
        description="Set a layer scale or pixel size deterministically on the selected or named layer.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels", "shapes", "points"),
        parameter_schema=(
            ParamSpec("layer_name", "string", description="Optional layer name. Falls back to the selected layer."),
            ParamSpec("scale", "float_or_list", description="Scalar or per-dimension scale value.", required=True),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Viewer Editing"},
        provenance_metadata={"algorithm": "set_layer_scale", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        layer_name = str(args.get("layer_name") or "").strip()
        layer = find_any_layer(ctx.viewer, layer_name) if layer_name else None
        if layer is None:
            layer = ctx.viewer.layers.selection.active if ctx.viewer is not None else None
        if layer is None:
            return "No valid layer is selected to set scale."
        if not hasattr(layer, "scale"):
            return f"Layer [{getattr(layer, 'name', 'unknown')}] does not support scale."

        scale_value = args.get("scale")
        if scale_value is None:
            return "Provide a scale value such as 0.1 or [1, 0.1, 0.1]."

        ndim = int(getattr(layer, "ndim", np.asarray(getattr(layer, "data", np.asarray([]))).ndim or 1))
        if isinstance(scale_value, (list, tuple)):
            normalized = [
                normalize_float(value, default=1.0, minimum=1e-9, maximum=1_000_000.0)
                for value in scale_value
            ]
        else:
            normalized = [normalize_float(scale_value, default=1.0, minimum=1e-9, maximum=1_000_000.0)]
        if not normalized:
            normalized = [1.0]
        if len(normalized) == 1:
            resolved_scale = tuple([float(normalized[0])] * ndim)
        elif len(normalized) < ndim:
            resolved_scale = tuple((normalized + [normalized[-1]] * ndim)[:ndim])
        else:
            resolved_scale = tuple(float(value) for value in normalized[:ndim])

        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "layer_name": str(layer.name),
                "scale": resolved_scale,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        layer = find_any_layer(ctx.viewer, payload.get("layer_name"))
        if layer is None:
            return f"Layer [{payload.get('layer_name', 'unknown')}] is no longer available."
        resolved_scale = tuple(float(value) for value in payload.get("scale", ()))
        if not resolved_scale:
            return f"No valid scale was resolved for [{layer.name}]."
        layer.scale = resolved_scale
        return f"Set scale for [{layer.name}] to {resolved_scale}."


class CreateSyntheticDemoImageTool:
    spec = ToolSpec(
        name="create_synthetic_demo_image",
        display_name="Create Synthetic Demo Image",
        category="demo",
        description="Create a small synthetic 2D/3D grayscale or RGB demo image for testing.",
        execution_mode="immediate",
        supported_layer_types=(),
        parameter_schema=(
            ParamSpec(
                "variant",
                "string",
                description="Synthetic demo variant.",
                default="2d_gray",
                enum=("2d_gray", "3d_gray", "2d_rgb", "3d_rgb"),
            ),
            ParamSpec("seed", "int", description="Random seed.", default=7),
        ),
        output_type="image_layer",
        ui_metadata={"panel_group": "Demo"},
        provenance_metadata={"algorithm": "synthetic_demo_generator", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        variant = str(args.get("variant") or "2d_gray").strip().lower() or "2d_gray"
        if variant not in {"2d_gray", "3d_gray", "2d_rgb", "3d_rgb"}:
            return "Unsupported synthetic demo variant. Use 2d_gray, 3d_gray, 2d_rgb, or 3d_rgb."
        seed = normalize_int(args.get("seed", 7), default=7, minimum=0, maximum=1_000_000)
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "variant": variant,
                "seed": seed,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        image, base_name = _synthetic_demo_payload(payload["variant"], seed=int(payload["seed"]))
        payload["image"] = image
        payload["base_name"] = base_name
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        output_name = next_output_name(ctx.viewer, payload["base_name"])
        image = np.asarray(payload["image"], dtype=np.float32)
        is_rgb = bool(image.ndim >= 3 and image.shape[-1] == 3)
        kwargs = {"rgb": True} if is_rgb else {"colormap": "gray"}
        ctx.viewer.add_image(image, name=output_name, **kwargs)
        variant = str(payload["variant"]).replace("_", " ")
        return f"Created [{output_name}] as a synthetic {variant} demo image."


class ArrangeLayersForPresentationTool:
    spec = ToolSpec(
        name="arrange_layers_for_presentation",
        display_name="Create Presentation Layout",
        category="presentation_layout",
        description="Physically arrange image and labels layers into a row, column, montage grid, or image-mask pairs in the same viewer.",
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
        ui_metadata={"panel_group": "Presentation Layout"},
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


class CreateAnalysisMontageTool:
    spec = ToolSpec(
        name="create_analysis_montage",
        display_name="Create Analysis Montage",
        category="analysis_montage",
        description="Create a composite montage canvas from 2D image layers for shared ROI and mask analysis.",
        execution_mode="immediate",
        supported_layer_types=("image",),
        parameter_schema=(
            ParamSpec("layer_names", "string_list", description="Optional ordered 2D image layer names to include."),
            ParamSpec("rows", "int", description="Optional montage row count.", default=0, minimum=0),
            ParamSpec("columns", "int", description="Optional montage column count.", default=0, minimum=0),
            ParamSpec("spacing", "int", description="Pixel spacing between tiles.", default=0, minimum=0),
            ParamSpec("show_tile_boxes", "bool", description="Add a Shapes layer showing tile boundaries.", default=True),
            ParamSpec("create_mask_layer", "bool", description="Create a blank labels layer for montage annotation.", default=True),
            ParamSpec("background_value", "float", description="Background fill value for padded regions.", default=0.0),
        ),
        output_type="image_layer",
        ui_metadata={"panel_group": "Analysis Montage"},
        provenance_metadata={"algorithm": "montage_canvas", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        layers = _resolve_montage_image_layers(ctx.viewer, args.get("layer_names"))
        if len(layers) < 2:
            return "Need at least 2 grayscale 2D image layers to build an analysis montage."
        try:
            rows, columns = _normalize_grid_dimensions(
                len(layers),
                normalize_int(args.get("rows", 0), default=0, minimum=0, maximum=128),
                normalize_int(args.get("columns", 0), default=0, minimum=0, maximum=128),
            )
        except ValueError as exc:
            return str(exc)
        spacing = normalize_int(args.get("spacing", 0), default=0, minimum=0, maximum=4096)
        show_tile_boxes = bool(args.get("show_tile_boxes", True))
        create_mask_layer = bool(args.get("create_mask_layer", True))
        background_value = normalize_float(args.get("background_value", 0.0), default=0.0, minimum=-1_000_000.0, maximum=1_000_000.0)
        montage_id = next_output_name(ctx.viewer, "analysis_montage")
        output_name = montage_id
        mask_name = next_output_name(ctx.viewer, f"{montage_id}_mask") if create_mask_layer else ""
        boxes_name = next_output_name(ctx.viewer, f"{montage_id}_tiles") if show_tile_boxes else ""
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "layer_names": [layer.name for layer in layers],
                "rows": rows,
                "columns": columns,
                "spacing": spacing,
                "show_tile_boxes": show_tile_boxes,
                "create_mask_layer": create_mask_layer,
                "background_value": background_value,
                "montage_id": montage_id,
                "output_name": output_name,
                "mask_name": mask_name,
                "boxes_name": boxes_name,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        layer_names = [str(name).strip() for name in payload.get("layer_names", []) if str(name).strip()]
        layers = [find_any_layer(ctx.viewer, name) for name in layer_names]
        layers = [layer for layer in layers if isinstance(layer, Image) and not getattr(layer, "rgb", False) and np.asarray(layer.data).ndim == 2]
        if len(layers) < 2:
            return "No usable 2D image layers were available to build the analysis montage."

        rows = int(payload.get("rows", 0) or 0)
        columns = int(payload.get("columns", 0) or 0)
        spacing = int(payload.get("spacing", 0) or 0)
        background_value = float(payload.get("background_value", 0.0) or 0.0)
        montage_array, placements, tile_size = _build_montage_canvas(
            layers,
            rows=rows,
            columns=columns,
            spacing=spacing,
            background_value=background_value,
        )
        output_name = str(payload.get("output_name", "")).strip() or next_output_name(ctx.viewer, "analysis_montage")
        mask_name = str(payload.get("mask_name", "")).strip()
        boxes_name = str(payload.get("boxes_name", "")).strip()
        montage_id = str(payload.get("montage_id", output_name)).strip() or output_name
        linked_outputs = {
            "montage_image_layer": output_name,
            "montage_labels_layer": mask_name,
            "tile_boxes_layer": boxes_name,
        }
        metadata = _montage_metadata(
            montage_id=montage_id,
            purpose="analysis",
            placements=placements,
            rows=rows,
            columns=columns,
            spacing=spacing,
            tile_size=tile_size,
            canvas_shape=tuple(int(v) for v in montage_array.shape),
            background_value=background_value,
            linked_outputs=linked_outputs,
        )
        image_layer = ctx.viewer.add_image(montage_array, name=output_name)
        image_layer.metadata = dict(getattr(image_layer, "metadata", {}) or {})
        image_layer.metadata["montage_canvas"] = metadata

        if bool(payload.get("create_mask_layer", True)):
            labels_layer = ctx.viewer.add_labels(np.zeros_like(montage_array, dtype=np.uint8), name=mask_name)
            labels_layer.metadata = dict(getattr(labels_layer, "metadata", {}) or {})
            labels_layer.metadata["montage_canvas"] = {
                **metadata,
                "role": "mask",
                "source_montage_image": output_name,
            }

        if bool(payload.get("show_tile_boxes", True)):
            rectangles = []
            for placement in placements:
                bbox = dict(placement.get("canvas_bbox", {}) or {})
                y0 = float(bbox.get("y0", 0))
                y1 = float(bbox.get("y1", 0))
                x0 = float(bbox.get("x0", 0))
                x1 = float(bbox.get("x1", 0))
                rectangles.append(np.asarray([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=np.float32))
            boxes_layer = ctx.viewer.add_shapes(
                rectangles,
                shape_type="rectangle",
                name=boxes_name,
                edge_width=1.5,
                edge_color="yellow",
                face_color="transparent",
            )
            boxes_layer.metadata = dict(getattr(boxes_layer, "metadata", {}) or {})
            boxes_layer.metadata["montage_canvas"] = {
                **metadata,
                "role": "tile_boxes",
                "source_montage_image": output_name,
            }

        return (
            f"Created analysis montage [{output_name}] from {len(layers)} image layer(s) "
            f"with grid={rows}x{columns} spacing={spacing}. "
            f"mask_layer={str(bool(payload.get('create_mask_layer', True))).lower()} "
            f"tile_boxes={str(bool(payload.get('show_tile_boxes', True))).lower()}."
        )


class SplitMontageAnnotationsTool:
    spec = ToolSpec(
        name="split_montage_annotations_to_sources",
        display_name="Split Montage Annotations To Sources",
        category="analysis_montage",
        description="Split montage-space labels or points back into per-source layers using stored montage metadata.",
        execution_mode="immediate",
        supported_layer_types=("labels", "points", "image"),
        parameter_schema=(
            ParamSpec("annotation_layer", "string", description="Montage-space Labels or Points layer."),
            ParamSpec("montage_layer", "string", description="Optional montage image or montage labels layer."),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Analysis Montage"},
        provenance_metadata={"algorithm": "split_montage_annotations", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        annotation_layer = _resolve_annotation_layer(ctx.viewer, args.get("annotation_layer"))
        if annotation_layer is None:
            return "No valid montage annotation layer available. Select or name a Labels or Points layer."
        if not isinstance(annotation_layer, (Labels, Points)):
            return "Montage annotation splitting currently supports Labels or Points layers only."
        montage_layer = _resolve_montage_reference_layer(ctx.viewer, args.get("montage_layer"), annotation_layer)
        if montage_layer is None:
            return "No montage canvas metadata was found. Select or name the montage image or montage mask layer."
        metadata = _montage_metadata_from_layer(annotation_layer) or _montage_metadata_from_layer(montage_layer)
        if metadata is None:
            return "Selected layers do not contain montage canvas metadata."
        placements = list(metadata.get("created_from", []) or [])
        if not placements:
            return "Montage metadata does not include any source-tile placements."
        if isinstance(annotation_layer, Labels):
            canvas_shape = tuple(int(v) for v in metadata.get("canvas_shape", []))
            if canvas_shape and tuple(np.asarray(annotation_layer.data).shape) != canvas_shape:
                return (
                    f"Annotation layer [{annotation_layer.name}] shape {tuple(np.asarray(annotation_layer.data).shape)} "
                    f"does not match montage canvas shape {canvas_shape}."
                )
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "annotation_layer": annotation_layer.name,
                "montage_layer": montage_layer.name,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        annotation_layer = _resolve_annotation_layer(ctx.viewer, payload.get("annotation_layer"))
        montage_layer = _resolve_montage_reference_layer(ctx.viewer, payload.get("montage_layer"), annotation_layer)
        if annotation_layer is None:
            return "No usable montage annotation layer was available for splitting."
        if montage_layer is None:
            return "No usable montage layer with metadata was available for splitting."

        metadata = _montage_metadata_from_layer(annotation_layer) or _montage_metadata_from_layer(montage_layer)
        if metadata is None:
            return "Montage metadata was not available during split."
        placements = list(metadata.get("created_from", []) or [])
        if not placements:
            return "Montage metadata does not include source placements."

        created_names: list[str] = []
        point_total = 0
        if isinstance(annotation_layer, Labels):
            labels_data = np.asarray(annotation_layer.data)
            for placement in placements:
                source_layer_name = str(placement.get("source_layer", "")).strip()
                if not source_layer_name:
                    continue
                y0, y1, x0, x1 = _placement_content_bbox(placement)
                source_shape = tuple(int(v) for v in placement.get("source_shape", []) or labels_data[y0:y1, x0:x1].shape)
                local_mask = np.asarray(labels_data[y0:y1, x0:x1]).copy()
                if local_mask.shape != source_shape:
                    local_mask = np.asarray(local_mask[: source_shape[0], : source_shape[1]])
                scale, translate = _source_layer_transform(ctx.viewer, source_layer_name)
                output_name = next_output_name(ctx.viewer, f"{source_layer_name}_{annotation_layer.name}")
                split_layer = ctx.viewer.add_labels(local_mask, name=output_name, scale=scale, translate=translate)
                split_layer.metadata = dict(getattr(split_layer, "metadata", {}) or {})
                split_layer.metadata["montage_split"] = {
                    "montage_id": metadata.get("montage_id"),
                    "source_layer": source_layer_name,
                    "annotation_layer": annotation_layer.name,
                    "source_kind": "labels",
                }
                created_names.append(output_name)
            if not created_names:
                return "No per-source labels layers were created from the montage annotation."
            created = ", ".join(f"[{name}]" for name in created_names[:6])
            if len(created_names) > 6:
                created += f" and {len(created_names) - 6} more"
            return (
                f"Split montage labels [{annotation_layer.name}] into {len(created_names)} per-source layer(s): {created}."
            )

        if isinstance(annotation_layer, Points):
            point_data = np.asarray(annotation_layer.data, dtype=float)
            if point_data.ndim != 2 or point_data.shape[1] < 2:
                return f"Points layer [{annotation_layer.name}] does not contain usable 2D coordinates."
            for placement in placements:
                source_layer_name = str(placement.get("source_layer", "")).strip()
                if not source_layer_name:
                    continue
                y0, y1, x0, x1 = _placement_content_bbox(placement)
                inside_indices: list[int] = []
                local_points: list[np.ndarray] = []
                for index, point in enumerate(point_data):
                    py = float(point[-2])
                    px = float(point[-1])
                    if py < y0 or py >= y1 or px < x0 or px >= x1:
                        continue
                    local_point = np.asarray(point, dtype=float).copy()
                    local_point[-2] = py - float(y0)
                    local_point[-1] = px - float(x0)
                    inside_indices.append(index)
                    local_points.append(local_point)
                if not local_points:
                    continue
                scale, translate = _source_layer_transform(ctx.viewer, source_layer_name)
                output_name = next_output_name(ctx.viewer, f"{source_layer_name}_{annotation_layer.name}")
                features = _slice_point_features(annotation_layer, inside_indices)
                split_layer = ctx.viewer.add_points(
                    np.asarray(local_points, dtype=float),
                    name=output_name,
                    scale=scale,
                    translate=translate,
                    features=features,
                )
                split_layer.metadata = dict(getattr(split_layer, "metadata", {}) or {})
                split_layer.metadata["montage_split"] = {
                    "montage_id": metadata.get("montage_id"),
                    "source_layer": source_layer_name,
                    "annotation_layer": annotation_layer.name,
                    "source_kind": "points",
                }
                created_names.append(output_name)
                point_total += len(local_points)
            if not created_names:
                return "No montage points fell inside any source image bounds."
            created = ", ".join(f"[{name}]" for name in created_names[:6])
            if len(created_names) > 6:
                created += f" and {len(created_names) - 6} more"
            return (
                f"Split montage points [{annotation_layer.name}] into {len(created_names)} per-source layer(s) "
                f"with {point_total} point(s): {created}."
            )

        return "Montage annotation splitting currently supports Labels or Points layers only."


class CreateTextAnnotationTool:
    spec = ToolSpec(
        name="create_text_annotation",
        display_name="Create Text Annotation",
        category="annotation",
        description="Create or reuse a managed text-annotation points layer and add one text label at a viewer position.",
        execution_mode="immediate",
        supported_layer_types=("image", "points"),
        parameter_schema=(
            ParamSpec("text", "string", description="Annotation text to place.", required=True),
            ParamSpec("position", "float_or_list", description="2D [x, y] or 3D [z, y, x] position.", required=True),
            ParamSpec("source_layer", "string", description="Optional source image or anchor layer."),
            ParamSpec("annotation_layer", "string", description="Optional managed text-annotation layer name."),
            ParamSpec("size", "float", description="Optional text size.", default=12.0, minimum=6.0, maximum=72.0),
            ParamSpec("color", "string", description="Optional text color.", default="yellow"),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Annotation"},
        provenance_metadata={"algorithm": "text_annotation", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        text = str(args.get("text", "") or "").strip()
        if not text:
            return "Provide annotation text."
        try:
            layer, anchor_layer = _ensure_text_annotation_layer(
                ctx.viewer,
                source_layer_name=args.get("source_layer"),
                annotation_layer_name=args.get("annotation_layer"),
            )
        except Exception as exc:
            return str(exc)
        ndim = int(getattr(anchor_layer, "ndim", 0) or 0)
        try:
            data_position = _normalize_text_annotation_position(args.get("position"), ndim=ndim)
        except Exception as exc:
            return str(exc)
        size = float(args.get("size", 12.0) or 12.0)
        color = str(args.get("color", "yellow") or "yellow").strip() or "yellow"
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "text": text,
                "position": list(data_position),
                "annotation_layer": str(layer.name),
                "source_layer": str(getattr(anchor_layer, "name", "") or ""),
                "size": max(6.0, min(72.0, size)),
                "color": color,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = dict(result.payload)
        try:
            layer, anchor_layer = _ensure_text_annotation_layer(
                ctx.viewer,
                source_layer_name=payload.get("source_layer"),
                annotation_layer_name=payload.get("annotation_layer"),
            )
        except Exception as exc:
            return str(exc)
        text = str(payload.get("text", "") or "").strip()
        position = np.asarray(payload.get("position", []), dtype=float)
        if position.ndim != 1 or position.size != int(getattr(anchor_layer, "ndim", 0) or 0):
            return "Text annotation position is not compatible with the target layer dimensionality."
        current_data = np.asarray(getattr(layer, "data", []), dtype=float)
        if current_data.size == 0:
            current_data = np.empty((0, position.size), dtype=float)
        new_data = np.vstack([current_data, position.reshape(1, -1)])
        labels = _text_annotation_labels(layer)
        labels.append(text)
        layer.data = new_data
        layer.features = {"label": np.asarray(labels, dtype=object)}
        _configure_text_annotation_layer(
            layer,
            source_layer_name=str(getattr(anchor_layer, "name", "") or ""),
            current_text=text,
            style_updates={"size": float(payload.get("size", 12.0) or 12.0), "color": str(payload.get("color", "yellow"))},
        )
        return (
            f"Added text annotation [{text}] to [{layer.name}] at {tuple(float(v) for v in position.tolist())} "
            f"for source layer [{anchor_layer.name}]."
        )


class RenameTextAnnotationTool:
    spec = ToolSpec(
        name="rename_text_annotation",
        display_name="Rename Text Annotation",
        category="annotation",
        description="Rename one text annotation inside a managed text-annotation points layer.",
        execution_mode="immediate",
        supported_layer_types=("points",),
        parameter_schema=(
            ParamSpec("old_text", "string", description="Existing annotation text to rename.", required=True),
            ParamSpec("new_text", "string", description="Replacement annotation text.", required=True),
            ParamSpec("annotation_layer", "string", description="Optional managed text-annotation layer name."),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Annotation"},
        provenance_metadata={"algorithm": "text_annotation", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        old_text = str(args.get("old_text", "") or "").strip()
        new_text = str(args.get("new_text", "") or "").strip()
        if not old_text or not new_text:
            return "Provide both old_text and new_text for renaming."
        layer = _resolve_text_annotation_layer(ctx.viewer, args.get("annotation_layer"))
        if layer is None:
            return "No managed text annotation layer is available to rename."
        labels = _text_annotation_labels(layer)
        if old_text not in labels:
            return f"No annotation text [{old_text}] was found in [{layer.name}]."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "annotation_layer": str(layer.name),
                "old_text": old_text,
                "new_text": new_text,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = dict(result.payload)
        layer = _resolve_text_annotation_layer(ctx.viewer, payload.get("annotation_layer"))
        if layer is None:
            return "No usable managed text annotation layer was available to rename."
        labels = _text_annotation_labels(layer)
        old_text = str(payload.get("old_text", "") or "").strip()
        new_text = str(payload.get("new_text", "") or "").strip()
        try:
            index = labels.index(old_text)
        except ValueError:
            return f"No annotation text [{old_text}] was found in [{layer.name}]."
        labels[index] = new_text
        layer.features = {"label": np.asarray(labels, dtype=object)}
        source_name = str(dict(getattr(layer, "metadata", {}) or {}).get("text_annotation_source_layer", "") or "")
        _configure_text_annotation_layer(layer, source_layer_name=source_name, current_text=new_text)
        return f"Renamed text annotation [{old_text}] to [{new_text}] in [{layer.name}]."


class DeleteTextAnnotationTool:
    spec = ToolSpec(
        name="delete_text_annotation",
        display_name="Delete Text Annotation",
        category="annotation",
        description="Delete one or more text annotations from a managed text-annotation points layer.",
        execution_mode="immediate",
        supported_layer_types=("points",),
        parameter_schema=(
            ParamSpec("text", "string", description="Annotation text to delete."),
            ParamSpec("annotation_layer", "string", description="Optional managed text-annotation layer name."),
            ParamSpec("delete_all", "bool", description="Delete all annotations from the layer.", default=False),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Annotation"},
        provenance_metadata={"algorithm": "text_annotation", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        layer = _resolve_text_annotation_layer(ctx.viewer, args.get("annotation_layer"))
        if layer is None:
            return "No managed text annotation layer is available to delete from."
        delete_all = bool(args.get("delete_all", False))
        text = str(args.get("text", "") or "").strip()
        labels = _text_annotation_labels(layer)
        if not labels:
            return f"Layer [{layer.name}] does not contain any text annotations."
        if not delete_all and not text:
            return "Provide annotation text to delete, or set delete_all=true."
        if not delete_all and text not in labels:
            return f"No annotation text [{text}] was found in [{layer.name}]."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "annotation_layer": str(layer.name),
                "text": text,
                "delete_all": delete_all,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = dict(result.payload)
        layer = _resolve_text_annotation_layer(ctx.viewer, payload.get("annotation_layer"))
        if layer is None:
            return "No usable managed text annotation layer was available to delete from."
        labels = _text_annotation_labels(layer)
        if not labels:
            return f"Layer [{layer.name}] does not contain any text annotations."
        delete_all = bool(payload.get("delete_all", False))
        if delete_all:
            layer.data = np.empty((0, int(getattr(layer, "ndim", 2) or 2)), dtype=float)
            layer.features = {"label": np.empty((0,), dtype=object)}
            return f"Deleted all text annotations from [{layer.name}]."
        text = str(payload.get("text", "") or "").strip()
        keep_indices = [index for index, value in enumerate(labels) if value != text]
        removed_count = len(labels) - len(keep_indices)
        if removed_count <= 0:
            return f"No annotation text [{text}] was found in [{layer.name}]."
        data = np.asarray(getattr(layer, "data", []), dtype=float)
        if keep_indices:
            layer.data = data[keep_indices]
            layer.features = {"label": np.asarray([labels[index] for index in keep_indices], dtype=object)}
        else:
            layer.data = np.empty((0, data.shape[1] if data.ndim == 2 and data.shape[1] > 0 else int(getattr(layer, "ndim", 2) or 2)), dtype=float)
            layer.features = {"label": np.empty((0,), dtype=object)}
        return f"Deleted {removed_count} text annotation(s) matching [{text}] from [{layer.name}]."


class ListTextAnnotationsTool:
    spec = ToolSpec(
        name="list_text_annotations",
        display_name="List Text Annotations",
        category="annotation",
        description="List the current text annotations in a managed text-annotation points layer.",
        execution_mode="immediate",
        supported_layer_types=("points",),
        parameter_schema=(ParamSpec("annotation_layer", "string", description="Optional managed text-annotation layer name."),),
        output_type="message",
        ui_metadata={"panel_group": "Annotation"},
        provenance_metadata={"algorithm": "text_annotation", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        layer = _resolve_text_annotation_layer(ctx.viewer, (arguments or {}).get("annotation_layer"))
        if layer is None:
            return "No managed text annotation layer is available."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={"kind": self.spec.name, "annotation_layer": str(layer.name)},
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        layer = _resolve_text_annotation_layer(ctx.viewer, result.payload.get("annotation_layer"))
        if layer is None:
            return "No usable managed text annotation layer is available."
        labels = _text_annotation_labels(layer)
        if not labels:
            return f"Layer [{layer.name}] does not contain any text annotations."
        coordinates = np.asarray(getattr(layer, "data", []), dtype=float)
        entries: list[str] = []
        for index, label in enumerate(labels):
            if coordinates.ndim == 2 and index < len(coordinates):
                point = ", ".join(f"{value:.1f}" for value in coordinates[index].tolist())
                entries.append(f"{index + 1}. [{label}] at ({point})")
            else:
                entries.append(f"{index + 1}. [{label}]")
        return f"Text annotations in [{layer.name}]: " + "; ".join(entries) + "."


class AnnotateLabelsWithTextTool:
    spec = ToolSpec(
        name="annotate_labels_with_text",
        display_name="Annotate Labels With Text",
        category="annotation",
        description="Create text annotations at labeled-object centroids from a labels layer.",
        execution_mode="immediate",
        supported_layer_types=("labels", "points", "image"),
        parameter_schema=(
            ParamSpec("labels_layer", "string", description="Labels layer to annotate."),
            ParamSpec("source_layer", "string", description="Optional image or anchor layer for the text overlay."),
            ParamSpec("annotation_layer", "string", description="Optional managed text-annotation layer name."),
            ParamSpec("prefix", "string", description="Text prefix such as Particle or Cell.", default="Particle"),
            ParamSpec("start_index", "int", description="Starting display index.", default=1, minimum=0),
            ParamSpec("size", "float", description="Optional text size.", default=12.0, minimum=6.0, maximum=72.0),
            ParamSpec("color", "string", description="Optional text color.", default="yellow"),
            ParamSpec("replace_existing", "bool", description="Clear existing annotations in the managed annotation layer first.", default=False),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Annotation"},
        provenance_metadata={"algorithm": "labels_centroid_annotation", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        labels_layer = find_labels_layer(ctx.viewer, args.get("labels_layer"))
        if labels_layer is None:
            return "No valid labels layer is available for automatic text annotation."
        try:
            layer, anchor_layer = _ensure_text_annotation_layer(
                ctx.viewer,
                source_layer_name=args.get("source_layer") or labels_layer.name,
                annotation_layer_name=args.get("annotation_layer"),
            )
        except Exception as exc:
            return str(exc)
        labeled = np.asarray(labels_layer.data)
        if labeled.ndim < 2:
            return f"Labels layer [{labels_layer.name}] must have at least 2 dimensions."
        props = regionprops(labeled)
        if not props:
            return f"Labels layer [{labels_layer.name}] does not contain any labeled objects."
        prefix = str(args.get("prefix", "Particle") or "Particle").strip() or "Particle"
        start_index = normalize_int(args.get("start_index", 1), default=1, minimum=0, maximum=1_000_000)
        entries: list[dict[str, object]] = []
        ordered = sorted(props, key=lambda prop: int(getattr(prop, "label", 0)))
        for display_offset, prop in enumerate(ordered):
            centroid = tuple(float(v) for v in tuple(getattr(prop, "centroid", ()) or ()))
            if len(centroid) < int(getattr(anchor_layer, "ndim", 0) or 0):
                continue
            entries.append(
                {
                    "label_id": int(getattr(prop, "label", 0)),
                    "text": f"{prefix} {start_index + display_offset}",
                    "position": list(centroid[: int(getattr(anchor_layer, 'ndim', 0) or len(centroid))]),
                }
            )
        if not entries:
            return f"Labels layer [{labels_layer.name}] does not contain usable centroid positions."
        size = float(args.get("size", 12.0) or 12.0)
        color = str(args.get("color", "yellow") or "yellow").strip() or "yellow"
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "labels_layer": labels_layer.name,
                "annotation_layer": str(layer.name),
                "source_layer": str(getattr(anchor_layer, "name", "") or ""),
                "entries": entries,
                "size": max(6.0, min(72.0, size)),
                "color": color,
                "replace_existing": bool(args.get("replace_existing", False)),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = dict(result.payload)
        labels_layer = find_labels_layer(ctx.viewer, payload.get("labels_layer"))
        if labels_layer is None:
            return "No usable labels layer was available for automatic text annotation."
        try:
            layer, anchor_layer = _ensure_text_annotation_layer(
                ctx.viewer,
                source_layer_name=payload.get("source_layer"),
                annotation_layer_name=payload.get("annotation_layer"),
            )
        except Exception as exc:
            return str(exc)
        entries = list(payload.get("entries", []) or [])
        if not entries:
            return f"Labels layer [{labels_layer.name}] does not contain any usable labeled objects."
        current_data = np.asarray(getattr(layer, "data", []), dtype=float)
        if current_data.ndim != 2:
            current_data = np.empty((0, int(getattr(anchor_layer, "ndim", 2) or 2)), dtype=float)
        current_labels = [] if bool(payload.get("replace_existing", False)) else _text_annotation_labels(layer)
        if bool(payload.get("replace_existing", False)):
            current_data = np.empty((0, int(getattr(anchor_layer, "ndim", 2) or 2)), dtype=float)
        appended_positions: list[np.ndarray] = []
        appended_texts: list[str] = []
        for entry in entries:
            position = np.asarray(entry.get("position", []), dtype=float)
            if position.ndim != 1 or position.size != int(getattr(anchor_layer, "ndim", 0) or 0):
                continue
            appended_positions.append(position)
            appended_texts.append(str(entry.get("text", "") or "").strip())
        if not appended_positions:
            return f"Labels layer [{labels_layer.name}] does not contain usable centroid positions."
        new_data = np.vstack([current_data] + [pos.reshape(1, -1) for pos in appended_positions])
        layer.data = new_data
        layer.features = {"label": np.asarray(current_labels + appended_texts, dtype=object)}
        _configure_text_annotation_layer(
            layer,
            source_layer_name=str(getattr(anchor_layer, "name", "") or ""),
            current_text=appended_texts[-1],
            style_updates={"size": float(payload.get("size", 12.0) or 12.0), "color": str(payload.get("color", "yellow"))},
        )
        created = ", ".join(f"[{text}]" for text in appended_texts)
        return (
            f"Added {len(appended_texts)} text annotation(s) from labels layer [{labels_layer.name}] "
            f"to [{layer.name}]: {created}."
        )


class AnnotateLabelsWithCalloutsTool:
    spec = ToolSpec(
        name="annotate_labels_with_callouts",
        display_name="Annotate Labels With Callouts",
        category="annotation",
        description="Create numbered external callout labels with leader lines and label boxes from a Labels layer.",
        execution_mode="immediate",
        supported_layer_types=("labels", "image"),
        parameter_schema=(
            ParamSpec("labels_layer", "string", description="Labels layer to annotate."),
            ParamSpec("source_layer", "string", description="Optional image or anchor layer for the callouts."),
            ParamSpec("prefix", "string", description="Label prefix such as Particle or Cell.", default="Particle"),
            ParamSpec("start_index", "int", description="Starting display index.", default=1, minimum=0),
            ParamSpec("size", "float", description="Optional text size.", default=12.0, minimum=6.0, maximum=72.0),
            ParamSpec("color", "string", description="Optional text color.", default="#f4f7fb"),
            ParamSpec("replace_existing", "bool", description="Rebuild the managed callout group instead of appending.", default=True),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Annotation"},
        provenance_metadata={"algorithm": "labels_callout_annotation", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        labels_layer = find_labels_layer(ctx.viewer, args.get("labels_layer"))
        if labels_layer is None:
            return "No valid labels layer is available for callout annotation."
        try:
            text_layer, boxes_layer, leaders_layer, anchor_layer = _ensure_callout_annotation_layers(
                ctx.viewer,
                source_layer_name=args.get("source_layer") or labels_layer.name,
            )
        except Exception as exc:
            return str(exc)
        labeled = np.asarray(labels_layer.data)
        if labeled.ndim != 2:
            return f"Callout annotation currently supports 2D labels layers only. Got ndim={labeled.ndim}."
        props = regionprops(labeled)
        if not props:
            return f"Labels layer [{labels_layer.name}] does not contain any labeled objects."
        prefix = str(args.get("prefix", "Particle") or "Particle").strip() or "Particle"
        start_index = normalize_int(args.get("start_index", 1), default=1, minimum=0, maximum=1_000_000)
        size = max(6.0, min(72.0, float(args.get("size", 12.0) or 12.0)))
        color = str(args.get("color", "#f4f7fb") or "#f4f7fb").strip() or "#f4f7fb"
        occupied_boxes: list[tuple[float, float, float, float]] = []
        entries: list[dict[str, object]] = []
        ordered = sorted(props, key=lambda prop: int(getattr(prop, "label", 0)))
        for display_offset, prop in enumerate(ordered):
            centroid = tuple(float(v) for v in tuple(getattr(prop, "centroid", ()) or ()))
            bbox = tuple(float(v) for v in tuple(getattr(prop, "bbox", ()) or ()))
            if len(centroid) < 2 or len(bbox) < 4:
                continue
            entries.append(
                _callout_entry_geometry(
                    text=f"{prefix} {start_index + display_offset}",
                    centroid_y=float(centroid[0]),
                    centroid_x=float(centroid[1]),
                    bbox_y0=float(bbox[0]),
                    bbox_y1=float(bbox[2]),
                    bbox_x0=float(bbox[1]),
                    bbox_x1=float(bbox[3]),
                    image_shape=tuple(int(v) for v in labeled.shape),
                    occupied_boxes=occupied_boxes,
                    size=size,
                )
            )
        if not entries:
            return f"Labels layer [{labels_layer.name}] does not contain usable 2D object geometry."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "labels_layer": labels_layer.name,
                "source_layer": str(getattr(anchor_layer, "name", "") or ""),
                "text_layer": text_layer.name,
                "boxes_layer": boxes_layer.name,
                "leaders_layer": leaders_layer.name,
                "entries": entries,
                "size": size,
                "color": color,
                "replace_existing": bool(args.get("replace_existing", True)),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = dict(result.payload)
        labels_layer = find_labels_layer(ctx.viewer, payload.get("labels_layer"))
        if labels_layer is None:
            return "No usable labels layer was available for callout annotation."
        try:
            text_layer, boxes_layer, leaders_layer, anchor_layer = _ensure_callout_annotation_layers(
                ctx.viewer,
                source_layer_name=payload.get("source_layer"),
            )
        except Exception as exc:
            return str(exc)
        entries = list(payload.get("entries", []) or [])
        if not entries:
            return f"Labels layer [{labels_layer.name}] does not contain usable callout geometry."
        replace_existing = bool(payload.get("replace_existing", True))

        text_labels = [str(entry.get("text", "") or "").strip() for entry in entries]
        text_positions = [np.asarray(entry.get("text_position", []), dtype=float) for entry in entries]
        valid_positions = [position for position in text_positions if position.ndim == 1 and position.size == 2]
        if len(valid_positions) != len(text_labels):
            return f"Could not build valid callout text positions for [{labels_layer.name}]."

        if replace_existing:
            existing_text_data = np.empty((0, 2), dtype=float)
            existing_labels: list[str] = []
        else:
            existing_text_data = np.asarray(getattr(text_layer, "data", []), dtype=float)
            if existing_text_data.ndim != 2:
                existing_text_data = np.empty((0, 2), dtype=float)
            existing_labels = _text_annotation_labels(text_layer)
        text_layer.data = np.vstack([existing_text_data] + [position.reshape(1, -1) for position in valid_positions])
        text_layer.features = {"label": np.asarray(existing_labels + text_labels, dtype=object)}
        _configure_callout_text_layer(
            text_layer,
            source_layer_name=str(getattr(anchor_layer, "name", "") or ""),
            style_updates={"size": float(payload.get("size", 12.0) or 12.0), "color": str(payload.get("color", "#f4f7fb"))},
        )

        box_shapes = [np.asarray(entry.get("box_shape", []), dtype=np.float32) for entry in entries]
        leader_shapes = [np.asarray(entry.get("leader_shape", []), dtype=np.float32) for entry in entries]
        boxes_layer.data = box_shapes
        boxes_layer.shape_type = ["rectangle"] * len(box_shapes)
        boxes_layer.features = {"label": np.asarray(text_labels, dtype=object)}
        _configure_callout_shapes_layer(boxes_layer, source_layer_name=str(getattr(anchor_layer, "name", "") or ""), role="boxes")

        leaders_layer.data = leader_shapes
        leaders_layer.shape_type = ["line"] * len(leader_shapes)
        leaders_layer.features = {"label": np.asarray(text_labels, dtype=object)}
        _configure_callout_shapes_layer(leaders_layer, source_layer_name=str(getattr(anchor_layer, "name", "") or ""), role="leaders")

        created = ", ".join(f"[{label}]" for label in text_labels)
        return (
            f"Added {len(text_labels)} Legion-style callout annotation(s) from labels layer [{labels_layer.name}] "
            f"using [{text_layer.name}], [{boxes_layer.name}], and [{leaders_layer.name}]: {created}."
        )


class CreateTitleLabelTool:
    spec = ToolSpec(
        name="create_title_label",
        display_name="Create Title Label",
        category="annotation",
        description="Create a boxed title label aligned left, center, or right above a 2D source layer without modifying image pixels.",
        execution_mode="immediate",
        supported_layer_types=("image", "labels"),
        parameter_schema=(
            ParamSpec("text", "string", description="Title text to place.", required=True),
            ParamSpec("source_layer", "string", description="Optional 2D source image or anchor layer."),
            ParamSpec("placement", "string", description="Title placement, currently outside_top.", default="outside_top"),
            ParamSpec("align", "string", description="Horizontal alignment: left, center, or right.", default="center"),
            ParamSpec("size", "float", description="Optional text size.", default=16.0, minimum=8.0, maximum=96.0),
            ParamSpec("color", "string", description="Optional text color.", default="#f4f7fb"),
            ParamSpec("margin", "float", description="Outer margin from the image edge.", default=8.0, minimum=0.0, maximum=256.0),
            ParamSpec("gap", "float", description="Gap between the image top edge and the title box.", default=10.0, minimum=0.0, maximum=256.0),
        ),
        output_type="message",
        ui_metadata={"panel_group": "Annotation"},
        provenance_metadata={"algorithm": "title_label_annotation", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        text = str(args.get("text", "") or "").strip()
        if not text:
            return "Provide title text."
        try:
            text_layer, box_layer, anchor_layer = _ensure_title_annotation_layers(
                ctx.viewer,
                source_layer_name=args.get("source_layer"),
            )
        except Exception as exc:
            return str(exc)
        data = np.asarray(getattr(anchor_layer, "data", None))
        if data.ndim != 2:
            return f"Title labels currently support 2D source layers only. Got ndim={data.ndim}."
        placement = str(args.get("placement", "outside_top") or "outside_top").strip().lower() or "outside_top"
        if placement not in {"outside_top"}:
            return f"Unsupported title placement [{placement}]. Use outside_top."
        align = str(args.get("align", "center") or "center").strip().lower() or "center"
        if align not in {"left", "center", "right"}:
            return f"Unsupported title alignment [{align}]. Use left, center, or right."
        size = max(8.0, min(96.0, float(args.get("size", 16.0) or 16.0)))
        margin = normalize_float(args.get("margin", 8.0), 8.0, minimum=0.0, maximum=256.0)
        gap = normalize_float(args.get("gap", 10.0), 10.0, minimum=0.0, maximum=256.0)
        box_height, box_width = _text_box_size(text, size=size)
        image_height = float(data.shape[0])
        image_width = float(data.shape[1])
        center_y = -(gap + box_height / 2.0)
        if align == "left":
            center_x = margin + box_width / 2.0
        elif align == "right":
            center_x = image_width - margin - box_width / 2.0
        else:
            center_x = image_width / 2.0
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "text": text,
                "source_layer": str(getattr(anchor_layer, "name", "") or ""),
                "text_layer": text_layer.name,
                "box_layer": box_layer.name,
                "text_position": [center_y, center_x],
                "box_shape": _rectangle_from_center(center_y, center_x, height=box_height, width=box_width).tolist(),
                "placement": placement,
                "align": align,
                "size": size,
                "color": str(args.get("color", "#f4f7fb") or "#f4f7fb").strip() or "#f4f7fb",
                "margin": margin,
                "gap": gap,
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = dict(result.payload)
        try:
            text_layer, box_layer, anchor_layer = _ensure_title_annotation_layers(
                ctx.viewer,
                source_layer_name=payload.get("source_layer"),
            )
        except Exception as exc:
            return str(exc)
        text = str(payload.get("text", "") or "").strip()
        if not text:
            return "Title text is empty."
        text_position = np.asarray(payload.get("text_position", []), dtype=float)
        box_shape = np.asarray(payload.get("box_shape", []), dtype=np.float32)
        if text_position.ndim != 1 or text_position.size != 2:
            return "Title label position is invalid."
        if box_shape.ndim != 2 or box_shape.shape != (4, 2):
            return "Title label box geometry is invalid."
        text_layer.data = np.asarray([text_position], dtype=float)
        text_layer.features = {"label": np.asarray([text], dtype=object)}
        _configure_callout_text_layer(
            text_layer,
            source_layer_name=str(getattr(anchor_layer, "name", "") or ""),
            style_updates={"size": float(payload.get("size", 16.0) or 16.0), "color": str(payload.get("color", "#f4f7fb"))},
        )
        box_layer.data = [box_shape]
        box_layer.shape_type = ["rectangle"]
        box_layer.features = {"label": np.asarray([text], dtype=object)}
        _configure_callout_shapes_layer(box_layer, source_layer_name=str(getattr(anchor_layer, "name", "") or ""), role="title_box")
        return (
            f"Added title label [{text}] to [{anchor_layer.name}] using [{text_layer.name}] and [{box_layer.name}] "
            f"with placement={payload.get('placement', 'outside_top')} align={payload.get('align', 'center')}."
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


class EditMaskInROITool:
    spec = ToolSpec(
        name="edit_mask_in_roi",
        display_name="Edit Mask In ROI",
        category="segmentation_cleanup",
        description="Apply a mask cleanup operation only inside a Labels-or-Shapes ROI, leaving the rest of the mask unchanged.",
        execution_mode="immediate",
        supported_layer_types=("labels", "shapes"),
        parameter_schema=(
            ParamSpec("mask_layer", "string", description="Target labels mask layer."),
            ParamSpec("roi_layer", "string", description="Labels or Shapes ROI layer.", required=True),
            ParamSpec("op", "string", description="convert_to_mask, dilate, erode, open, close, median, outline, fill_holes, skeletonize, distance_map, ultimate_points, watershed, voronoi, remove_small, or keep_largest.", required=True),
            ParamSpec("radius", "int", description="Radius for morphology operations.", default=1, minimum=1),
            ParamSpec("min_size", "int", description="Minimum size for remove_small.", default=64, minimum=1),
        ),
        output_type="labels_layer",
        ui_metadata={"panel_group": "Segmentation Cleanup"},
        provenance_metadata={"algorithm": "roi_constrained_mask_edit", "deterministic": True},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        mask_layer = find_labels_layer(ctx.viewer, args.get("mask_layer"))
        if mask_layer is None:
            return "No valid labels mask layer available for ROI editing."
        roi_layer = _resolve_roi_layer(ctx.viewer, args.get("roi_layer"))
        if roi_layer is None:
            return "No valid ROI layer available. Select or name a Labels or Shapes layer."
        try:
            roi_mask = _roi_mask_from_layer(roi_layer, np.asarray(mask_layer.data).shape)
        except Exception as exc:
            return str(exc)
        if not np.any(roi_mask):
            return f"ROI layer [{roi_layer.name}] does not cover any pixels in [{mask_layer.name}]."
        op_name = str(args.get("op", "")).strip().lower()
        if op_name not in {
            "convert_to_mask",
            "dilate",
            "erode",
            "open",
            "close",
            "median",
            "outline",
            "fill_holes",
            "skeletonize",
            "distance_map",
            "ultimate_points",
            "watershed",
            "voronoi",
            "remove_small",
            "keep_largest",
        }:
            return f"Unsupported ROI mask operation: {op_name}"
        radius = normalize_int(args.get("radius", 1), default=1, minimum=1, maximum=1024)
        min_size = normalize_int(args.get("min_size", 64), default=64, minimum=1, maximum=1_000_000)
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="immediate",
            payload={
                "kind": self.spec.name,
                "mask_layer": mask_layer.name,
                "roi_layer": roi_layer.name,
                "op_name": op_name,
                "radius": radius,
                "min_size": min_size,
                "output_name": next_output_name(ctx.viewer, f"{mask_layer.name}_{op_name}_roi"),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=dict(job.payload))

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        mask_layer = find_labels_layer(ctx.viewer, payload.get("mask_layer"))
        roi_layer = _resolve_roi_layer(ctx.viewer, payload.get("roi_layer"))
        if mask_layer is None:
            return "No usable labels mask layer was available for ROI editing."
        if roi_layer is None:
            return "No usable ROI layer was available for ROI editing."
        try:
            roi_mask = _roi_mask_from_layer(roi_layer, np.asarray(mask_layer.data).shape).astype(bool)
        except Exception as exc:
            return str(exc)
        original = np.asarray(mask_layer.data)
        local_input = np.where(roi_mask, original, 0).astype(original.dtype, copy=False)
        edited_local = _apply_mask_operation(
            local_input,
            op_name=payload.get("op_name", ""),
            radius=payload.get("radius", 1),
            min_size=payload.get("min_size", 64),
        )
        merged = np.asarray(original).copy()
        merged[roi_mask] = np.asarray(edited_local)[roi_mask]
        output_name = str(payload.get("output_name", "")).strip() or next_output_name(ctx.viewer, f"{mask_layer.name}_roi_edit")
        ctx.viewer.add_labels(
            merged,
            name=output_name,
            scale=tuple(getattr(mask_layer, "scale", ())),
            translate=tuple(getattr(mask_layer, "translate", ())),
        )
        return (
            f"Applied [{payload['op_name']}] to [{mask_layer.name}] only inside ROI [{roi_layer.name}] "
            f"as [{output_name}] with radius={payload['radius']} min_size={payload['min_size']}."
        )


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


class SAMRefineMaskTool:
    spec = ToolSpec(
        name="sam_refine_mask",
        display_name="SAM Refine Mask",
        category="segmentation",
        description="Refine an existing mask using Segment Anything.",
        execution_mode="worker",
        supported_layer_types=("image", "labels", "shapes"),
        parameter_schema=(
            ParamSpec("image_layer", "string", description="Target image layer."),
            ParamSpec("mask_layer", "string", description="Existing mask or labels layer."),
            ParamSpec("roi_layer", "string", description="Optional additional ROI prompt."),
            ParamSpec("model_name", "string", description="Optional SAM backend/model identifier."),
        ),
        output_type="labels_layer",
        ui_metadata={"panel_group": "Segmentation"},
        provenance_metadata={"algorithm": "sam2_refine", "deterministic": False},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("image_layer"))
        if image_layer is None:
            return "No valid image layer available for SAM2 mask refinement."
        if getattr(image_layer, "rgb", False):
            return "SAM2 mask refinement currently supports grayscale 2D image layers, not RGB layers."
        image_data = np.asarray(image_layer.data)
        if image_data.ndim != 2:
            return f"SAM2 mask refinement currently supports 2D image layers only. Got ndim={image_data.ndim}."
        mask_layer = find_labels_layer(ctx.viewer, args.get("mask_layer"))
        if mask_layer is None:
            return "No valid mask layer available for SAM2 refinement. Select or name a Labels layer."
        mask_data = (np.asarray(mask_layer.data) > 0).astype(np.uint8, copy=False)
        if mask_data.shape != image_data.shape:
            return f"Mask layer [{mask_layer.name}] shape {mask_data.shape} does not match image layer [{image_layer.name}] shape {image_data.shape}."
        if not np.any(mask_data):
            return f"Mask layer [{mask_layer.name}] does not contain any positive pixels to refine."

        roi_layer = _resolve_roi_layer(ctx.viewer, args.get("roi_layer"))
        roi_mask = None
        roi_layer_name = ""
        if roi_layer is not None:
            roi_layer_name = str(roi_layer.name)
            if isinstance(roi_layer, Labels):
                roi_mask = np.asarray(roi_layer.data) > 0
            else:
                roi_mask = _rasterize_shapes_roi(roi_layer, image_data.shape)
            if np.asarray(roi_mask).shape != image_data.shape:
                return f"ROI layer [{roi_layer.name}] shape does not match image layer [{image_layer.name}]."

        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "image_layer_name": image_layer.name,
                "mask_layer_name": mask_layer.name,
                "roi_layer_name": roi_layer_name,
                "output_name": next_output_name(ctx.viewer, f"{mask_layer.name}_sam2_refined"),
                "model_name": str(args.get("model_name") or "").strip() or None,
                "data": image_data.copy(),
                "mask": mask_data.copy(),
                "roi_mask": None if roi_mask is None else np.asarray(roi_mask, dtype=np.uint8).copy(),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        try:
            refined, backend_message = refine_mask_from_mask(
                np.asarray(payload["data"]),
                mask=np.asarray(payload["mask"]),
                roi_mask=None if payload.get("roi_mask") is None else np.asarray(payload["roi_mask"]),
                model_name=payload.get("model_name"),
            )
        except Exception as exc:
            return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload, message=str(exc))
        payload["result"] = refined
        payload["backend_message"] = backend_message
        payload["foreground_pixels"] = int(np.count_nonzero(refined))
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        if "result" not in payload:
            return result.message or "SAM2 mask refinement did not produce a result."
        ctx.viewer.add_labels(
            payload["result"],
            name=payload["output_name"],
            scale=payload["scale"],
            translate=payload["translate"],
        )
        roi_suffix = f" within ROI [{payload['roi_layer_name']}]" if payload.get("roi_layer_name") else ""
        backend_suffix = f" {payload['backend_message']}" if payload.get("backend_message") else ""
        return (
            f"Refined [{payload['mask_layer_name']}] on [{payload['image_layer_name']}]"
            f"{roi_suffix} as [{payload['output_name']}]. foreground_pixels={payload['foreground_pixels']}.{backend_suffix}"
        )


class SAMAutoSegmentTool:
    spec = ToolSpec(
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
        provenance_metadata={"algorithm": "sam2_auto", "deterministic": False},
    )

    def prepare(self, ctx: ToolContext, arguments: dict[str, object]) -> PreparedJob | str:
        args = arguments or {}
        image_layer = find_image_layer(ctx.viewer, args.get("image_layer"))
        if image_layer is None:
            return "No valid image layer available for SAM2 auto segmentation."
        if getattr(image_layer, "rgb", False):
            return "SAM2 auto segmentation currently supports grayscale 2D image layers, not RGB layers."
        image_data = np.asarray(image_layer.data)
        if image_data.ndim != 2:
            return f"SAM2 auto segmentation currently supports 2D image layers only. Got ndim={image_data.ndim}."
        return PreparedJob(
            tool_name=self.spec.name,
            kind=self.spec.name,
            mode="worker",
            payload={
                "kind": self.spec.name,
                "image_layer_name": image_layer.name,
                "output_name": next_output_name(ctx.viewer, f"{image_layer.name}_sam2_auto"),
                "model_name": str(args.get("model_name") or "").strip() or None,
                "data": image_data.copy(),
                "scale": tuple(image_layer.scale),
                "translate": tuple(image_layer.translate),
            },
        )

    def execute(self, job: PreparedJob) -> ToolResult:
        payload = dict(job.payload)
        try:
            segmented, backend_message = segment_image_auto(
                np.asarray(payload["data"]),
                model_name=payload.get("model_name"),
            )
        except Exception as exc:
            return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload, message=str(exc))
        payload["result"] = segmented
        payload["backend_message"] = backend_message
        payload["foreground_pixels"] = int(np.count_nonzero(segmented))
        return ToolResult(tool_name=self.spec.name, kind=job.kind, payload=payload)

    def apply(self, ctx: ToolContext, result: ToolResult) -> str:
        payload = result.payload
        if "result" not in payload:
            return result.message or "SAM2 auto segmentation did not produce a result."
        ctx.viewer.add_labels(
            payload["result"],
            name=payload["output_name"],
            scale=payload["scale"],
            translate=payload["translate"],
        )
        backend_suffix = f" {payload['backend_message']}" if payload.get("backend_message") else ""
        return (
            f"Auto-segmented [{payload['image_layer_name']}] as [{payload['output_name']}]. "
            f"foreground_pixels={payload['foreground_pixels']}.{backend_suffix}"
        )


def workbench_scaffold_tools():
    return [
        GaussianDenoiseTool(),
        RemoveSmallObjectsTool(),
        FillMaskHolesTool(),
        EditMaskInROITool(),
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
        HideAllLayersTool(),
        DeleteAllLayersTool(),
        DeleteLayersTool(),
        ShowOnlyLayersTool(),
        ShowAllExceptLayersTool(),
        ShowAllLayersTool(),
        SetLayerScaleTool(),
        CreateSyntheticDemoImageTool(),
        ArrangeLayersForPresentationTool(),
        CreateAnalysisMontageTool(),
        SplitMontageAnnotationsTool(),
        CreateTextAnnotationTool(),
        AnnotateLabelsWithTextTool(),
        AnnotateLabelsWithCalloutsTool(),
        CreateTitleLabelTool(),
        RenameTextAnnotationTool(),
        DeleteTextAnnotationTool(),
        ListTextAnnotationsTool(),
        InspectROIContextTool(),
        MeasureShapesROIAreaTool(),
        ExtractROIValuesTool(),
        CompareROIGroupsTool(),
        CompareImageGroupsTool(),
        SAMSegmentFromBoxTool(),
        SAMSegmentFromPointsTool(),
        SAMPropagatePoints3DTool(),
        SAMRefineMaskTool(),
        SAMAutoSegmentTool(),
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
