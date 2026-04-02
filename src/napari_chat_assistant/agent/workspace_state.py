from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import napari
import numpy as np


INLINE_MAX_ELEMENTS = 1_000_000


def _json_ready(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _safe_source_info(layer) -> dict[str, Any]:
    source = getattr(layer, "source", None)
    if source is None:
        return {"path": None, "reader_plugin": None}
    path = getattr(source, "path", None)
    plugin = getattr(source, "reader_plugin", None)
    return {
        "path": str(path) if path else None,
        "reader_plugin": str(plugin) if plugin else None,
    }


def _layer_common_state(layer) -> dict[str, Any]:
    return {
        "layer_type": layer.__class__.__name__,
        "name": str(getattr(layer, "name", "")),
        "visible": bool(getattr(layer, "visible", True)),
        "opacity": float(getattr(layer, "opacity", 1.0)),
        "blending": str(getattr(layer, "blending", "translucent")),
        "scale": [float(v) for v in tuple(getattr(layer, "scale", ()) or ())],
        "translate": [float(v) for v in tuple(getattr(layer, "translate", ()) or ())],
        "source": _safe_source_info(layer),
    }


def _serialize_inline_image_like(layer, layer_type: str) -> dict[str, Any] | None:
    data = np.asarray(getattr(layer, "data", None))
    if data.size <= 0 or data.size > INLINE_MAX_ELEMENTS:
        return None
    return {
        "inline_data": data.tolist(),
        "dtype": str(data.dtype),
        "shape": list(data.shape),
        "layer_type": layer_type,
    }


def _serialize_shapes(layer: napari.layers.Shapes) -> dict[str, Any]:
    features = getattr(layer, "features", None)
    labels = None
    if features is not None and "label" in features:
        try:
            labels = [str(value) for value in list(features["label"])]
        except Exception:
            labels = None
    return {
        "inline_kind": "Shapes",
        "data": [[list(map(float, point)) for point in np.asarray(shape, dtype=float)] for shape in layer.data],
        "shape_type": [str(value) for value in list(getattr(layer, "shape_type", []))],
        "features": {"label": labels} if labels is not None else {},
        "edge_width": float(getattr(layer, "edge_width", 1.0)),
    }


def capture_workspace_manifest(viewer: napari.Viewer) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for layer in list(viewer.layers):
        record = _layer_common_state(layer)
        source_path = record["source"].get("path")
        layer_type = str(record["layer_type"])
        if source_path:
            if isinstance(layer, napari.layers.Image):
                record.update(
                    {
                        "contrast_limits": [float(v) for v in tuple(getattr(layer, "contrast_limits", ()) or ())],
                        "colormap": str(getattr(getattr(layer, "colormap", None), "name", "gray") or "gray"),
                        "gamma": float(getattr(layer, "gamma", 1.0)),
                    }
                )
            layers.append(record)
            continue

        if isinstance(layer, napari.layers.Shapes):
            record.update(_serialize_shapes(layer))
            layers.append(record)
            continue

        if isinstance(layer, napari.layers.Labels):
            inline = _serialize_inline_image_like(layer, "Labels")
            if inline is not None:
                record.update(inline)
                layers.append(record)
                continue

        skipped.append({"name": str(record["name"]), "reason": "no recoverable file path or inline serialization"})

    selected = getattr(getattr(viewer.layers, "selection", None), "active", None)
    return {
        "version": 1,
        "viewer": {
            "dims_current_step": [int(v) for v in tuple(getattr(viewer.dims, "current_step", ()) or ())],
            "selected_layer_name": str(getattr(selected, "name", "")) if selected is not None else "",
        },
        "layers": layers,
        "skipped_layers": skipped,
    }


def save_workspace_manifest(viewer: napari.Viewer, destination: str | Path) -> dict[str, Any]:
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = capture_workspace_manifest(viewer)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "path": str(path),
        "saved_layers": len(manifest.get("layers", [])),
        "skipped_layers": manifest.get("skipped_layers", []),
    }


def _apply_common_state(layer, record: dict[str, Any]) -> None:
    try:
        layer.name = str(record.get("name", layer.name))
    except Exception:
        pass
    for attr in ("visible", "opacity", "blending"):
        if attr in record:
            try:
                setattr(layer, attr, record[attr])
            except Exception:
                pass
    for attr in ("scale", "translate"):
        value = record.get(attr)
        if value:
            try:
                setattr(layer, attr, tuple(value))
            except Exception:
                pass
    if isinstance(layer, napari.layers.Image):
        if record.get("contrast_limits"):
            try:
                layer.contrast_limits = tuple(float(v) for v in record["contrast_limits"])
            except Exception:
                pass
        if record.get("colormap"):
            try:
                layer.colormap = str(record["colormap"])
            except Exception:
                pass
        if "gamma" in record:
            try:
                layer.gamma = float(record["gamma"])
            except Exception:
                pass


def _restore_inline_layer(viewer: napari.Viewer, record: dict[str, Any]):
    inline_kind = str(record.get("inline_kind") or record.get("layer_type") or "")
    if inline_kind == "Shapes":
        features = record.get("features") or {}
        return viewer.add_shapes(
            data=[[tuple(point) for point in shape] for shape in record.get("data", [])],
            shape_type=list(record.get("shape_type", [])) or "polygon",
            edge_width=float(record.get("edge_width", 1.0)),
            name=str(record.get("name", "Shapes")),
            features=features,
        )
    if inline_kind in {"Labels", "Image"} and "inline_data" in record:
        data = np.asarray(record["inline_data"], dtype=np.dtype(record.get("dtype") or "float32"))
        if inline_kind == "Labels":
            return viewer.add_labels(data, name=str(record.get("name", "Labels")))
        return viewer.add_image(data, name=str(record.get("name", "Image")))
    return None


def _remove_all_layers(viewer: napari.Viewer) -> None:
    while len(viewer.layers):
        viewer.layers.remove(viewer.layers[0])


def load_workspace_manifest(viewer: napari.Viewer, source: str | Path, *, clear_existing: bool = True) -> dict[str, Any]:
    path = Path(source).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if clear_existing:
        _remove_all_layers(viewer)

    restored_names: list[str] = []
    skipped_layers: list[dict[str, str]] = []
    for record in payload.get("layers", []):
        layer = None
        source_info = record.get("source") or {}
        source_path = source_info.get("path")
        if source_path and Path(source_path).exists():
            try:
                loaded = viewer.open(
                    [source_path],
                    stack=False,
                    plugin=source_info.get("reader_plugin") or None,
                    layer_type=str(record.get("layer_type", "")).lower() or None,
                )
                if loaded:
                    layer = loaded[-1]
            except Exception as exc:
                skipped_layers.append({"name": str(record.get("name", source_path)), "reason": str(exc)})
                continue
        else:
            try:
                layer = _restore_inline_layer(viewer, record)
            except Exception as exc:
                skipped_layers.append({"name": str(record.get("name", "unknown")), "reason": str(exc)})
                continue

        if layer is None:
            skipped_layers.append(
                {"name": str(record.get("name", "unknown")), "reason": "could not restore layer from source or inline data"}
            )
            continue
        _apply_common_state(layer, record)
        restored_names.append(str(getattr(layer, "name", "")))

    selected_name = str(((payload.get("viewer") or {}).get("selected_layer_name")) or "").strip()
    if selected_name:
        for layer in viewer.layers:
            if str(getattr(layer, "name", "")) == selected_name:
                try:
                    viewer.layers.selection.active = layer
                except Exception:
                    pass
                break
    dims_step = list(((payload.get("viewer") or {}).get("dims_current_step")) or [])
    if dims_step:
        try:
            viewer.dims.current_step = tuple(int(v) for v in dims_step)
        except Exception:
            pass

    return {
        "path": str(path),
        "restored_layers": restored_names,
        "skipped_layers": skipped_layers,
    }
