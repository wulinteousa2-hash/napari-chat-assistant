from __future__ import annotations

import napari
import numpy as np

from .image_ops import mask_statistics


def get_viewer(napari_viewer):
    if napari_viewer is not None:
        return napari_viewer
    try:
        return napari.current_viewer()
    except Exception:
        return None


def layer_summary(viewer: napari.Viewer) -> str:
    if viewer is None:
        return "No active napari viewer."

    lines = []
    if len(viewer.layers) == 0:
        lines.append("Layers: none")
    else:
        lines.append(f"Layers: {len(viewer.layers)}")
        for layer in viewer.layers:
            shape = getattr(getattr(layer, "data", None), "shape", None)
            dtype = getattr(getattr(layer, "data", None), "dtype", None)
            shape_text = tuple(shape) if shape is not None else "n/a"
            dtype_text = str(dtype) if dtype is not None else "n/a"
            lines.append(f"- {layer.name} [{layer.__class__.__name__}] shape={shape_text} dtype={dtype_text}")

    selected = viewer.layers.selection.active
    lines.append(f"Selected layer: {selected.name}" if selected is not None else "Selected layer: none")
    return "\n".join(lines)


def _layer_kind(layer) -> str:
    if isinstance(layer, napari.layers.Image):
        return "image"
    if isinstance(layer, napari.layers.Labels):
        return "labels"
    if isinstance(layer, napari.layers.Points):
        return "points"
    if isinstance(layer, napari.layers.Shapes):
        return "shapes"
    return layer.__class__.__name__.lower()


def _layer_data_shape(layer) -> str:
    data = getattr(layer, "data", None)
    shape = getattr(data, "shape", None)
    if shape is not None:
        return str(tuple(shape))
    if isinstance(layer, napari.layers.Shapes):
        try:
            return str(tuple(np.asarray(layer.data, dtype=object).shape))
        except Exception:
            return "n/a"
    return "n/a"


def _layer_dtype(layer) -> str:
    data = getattr(layer, "data", None)
    dtype = getattr(data, "dtype", None)
    return str(dtype) if dtype is not None else "n/a"


def layer_detail_summary(layer) -> str:
    if layer is None:
        return "No valid layer available."

    lines = [
        f"Layer [{layer.name}]",
        f"- kind: {_layer_kind(layer)}",
        f"- class: {layer.__class__.__name__}",
        f"- visible: {getattr(layer, 'visible', 'n/a')}",
        f"- opacity: {getattr(layer, 'opacity', 'n/a')}",
        f"- ndim: {getattr(layer, 'ndim', 'n/a')}",
        f"- scale: {tuple(getattr(layer, 'scale', ())) or 'n/a'}",
        f"- translate: {tuple(getattr(layer, 'translate', ())) or 'n/a'}",
        f"- data shape: {_layer_data_shape(layer)}",
        f"- data dtype: {_layer_dtype(layer)}",
    ]

    if isinstance(layer, napari.layers.Image):
        data = np.asarray(layer.data)
        lines.extend(
            [
                f"- rgb: {getattr(layer, 'rgb', False)}",
                f"- contrast_limits: {tuple(getattr(layer, 'contrast_limits', ())) or 'n/a'}",
                f"- min intensity: {float(np.min(data)):.6g}",
                f"- max intensity: {float(np.max(data)):.6g}",
                f"- mean intensity: {float(np.mean(data)):.6g}",
            ]
        )
    elif isinstance(layer, napari.layers.Labels):
        stats = mask_statistics(np.asarray(layer.data))
        lines.extend(
            [
                f"- labels count: {len(np.unique(np.asarray(layer.data)))}",
                f"- foreground pixels: {stats['foreground_pixels']}",
                f"- object count: {stats['object_count']}",
                f"- largest object: {stats['largest_object']}",
            ]
        )
    elif isinstance(layer, napari.layers.Points):
        lines.extend(
            [
                f"- points count: {len(layer.data)}",
                f"- edge width: {getattr(layer, 'edge_width', 'n/a')}",
                f"- size: {getattr(layer, 'size', 'n/a')}",
            ]
        )
    elif isinstance(layer, napari.layers.Shapes):
        lines.extend(
            [
                f"- shapes count: {len(layer.data)}",
                f"- shape types: {list(getattr(layer, 'shape_type', []))}",
            ]
        )

    return "\n".join(lines)


def layer_context_json(viewer: napari.Viewer) -> dict:
    if viewer is None:
        return {"layers": [], "selected_layer": None}

    layers = []
    for layer in viewer.layers:
        data = getattr(layer, "data", None)
        shape = getattr(data, "shape", None)
        dtype = getattr(data, "dtype", None)
        entry = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "shape": list(shape) if shape is not None else None,
            "dtype": str(dtype) if dtype is not None else None,
        }
        if isinstance(layer, napari.layers.Labels):
            try:
                entry["mask_stats"] = mask_statistics(np.asarray(layer.data))
            except Exception:
                entry["mask_stats"] = None
        layers.append(entry)

    selected = viewer.layers.selection.active
    return {"layers": layers, "selected_layer": None if selected is None else selected.name}


def mask_measurement_summary(layer: napari.layers.Labels) -> str:
    data = np.asarray(layer.data)
    stats = mask_statistics(data)
    binary = data > 0
    ndim = binary.ndim
    scale = tuple(float(s) for s in getattr(layer, "scale", (1.0,) * ndim))
    if len(scale) < ndim:
        scale = (1.0,) * (ndim - len(scale)) + scale
    voxel_size = 1.0
    for s in scale[-ndim:]:
        voxel_size *= float(s)

    if ndim == 2:
        area = stats["foreground_pixels"] * voxel_size
        unit = "area units^2" if voxel_size != 1.0 else "pixels^2"
        return (
            f"Mask [{layer.name}] measurement: foreground={stats['foreground_pixels']} px, "
            f"objects={stats['object_count']}, largest={stats['largest_object']} px, "
            f"area={area:.6g} {unit}."
        )

    if ndim == 3:
        volume = stats["foreground_pixels"] * voxel_size
        unit = "volume units^3" if voxel_size != 1.0 else "voxels"
        return (
            f"Mask [{layer.name}] measurement: foreground={stats['foreground_pixels']} voxels, "
            f"objects={stats['object_count']}, largest={stats['largest_object']} voxels, "
            f"volume={volume:.6g} {unit}."
        )

    return (
        f"Mask [{layer.name}] measurement: foreground={stats['foreground_pixels']}, "
        f"objects={stats['object_count']}, largest={stats['largest_object']}."
    )


def find_image_layer(viewer: napari.Viewer, name: str | None = None):
    if viewer is None:
        return None
    if name:
        try:
            layer = viewer.layers[name]
            return layer if isinstance(layer, napari.layers.Image) else None
        except KeyError:
            return None
    selected = viewer.layers.selection.active
    if isinstance(selected, napari.layers.Image):
        return selected
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Image):
            return layer
    return None


def find_all_image_layers(viewer: napari.Viewer):
    if viewer is None:
        return []
    return [layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)]


def find_labels_layer(viewer: napari.Viewer, name: str | None = None):
    if viewer is None:
        return None
    if name:
        try:
            layer = viewer.layers[name]
            return layer if isinstance(layer, napari.layers.Labels) else None
        except KeyError:
            return None
    selected = viewer.layers.selection.active
    if isinstance(selected, napari.layers.Labels):
        return selected
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Labels):
            return layer
    return None


def find_all_labels_layers(viewer: napari.Viewer):
    if viewer is None:
        return []
    return [layer for layer in viewer.layers if isinstance(layer, napari.layers.Labels)]


def find_any_layer(viewer: napari.Viewer, name: str | None = None):
    if viewer is None:
        return None
    if name:
        try:
            return viewer.layers[name]
        except KeyError:
            return None
    return viewer.layers.selection.active
