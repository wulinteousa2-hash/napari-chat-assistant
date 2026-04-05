from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import napari
import numpy as np
import zarr
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image, write_labels


INLINE_MAX_ELEMENTS = 1_000_000
WORKSPACE_VERSION = 2
OME_ZARR_LABEL_NAME = "labels"
SPECTRAL_READER_PLUGIN = "napari-nd2-spectral-ome-zarr"


def _sequence_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    try:
        return tuple(value)
    except Exception:
        return ()


def _json_ready(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if hasattr(value, "value"):
        try:
            return _json_ready(value.value)
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        try:
            ready_shape = None if shape is None else [int(v) for v in tuple(shape)]
        except Exception:
            ready_shape = None
        return {
            "python_type": f"{type(value).__module__}.{type(value).__name__}",
            "shape": ready_shape,
            "dtype": None if dtype is None else str(dtype),
        }
    return {"python_type": f"{type(value).__module__}.{type(value).__name__}"}


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


def _spectral_recipe_for_layer(layer) -> dict[str, Any] | None:
    metadata = getattr(layer, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    source_path = str(metadata.get("source_path") or "").strip()
    if not source_path:
        return None
    dataset_metadata = metadata.get("dataset_metadata")
    wavelengths = metadata.get("wavelengths_nm")
    spectral_cube = metadata.get("spectral_cube")
    if spectral_cube is None and wavelengths is None and not isinstance(dataset_metadata, dict):
        return None
    layer_name = str(getattr(layer, "name", "") or "").strip()
    lowered = layer_name.lower()
    view_type = ""
    if "truecolor" in lowered:
        view_type = "truecolor"
    elif "visible sum" in lowered:
        view_type = "visible_sum"
    elif lowered.endswith(" spectral") or " spectral " in lowered:
        view_type = "raw_spectral"
    if not view_type:
        return None
    return {
        "kind": "spectral_view",
        "source_path": source_path,
        "reader_plugin": SPECTRAL_READER_PLUGIN,
        "view_type": view_type,
        "zarr_use_preview": lowered.endswith(" preview"),
        "source_layer_name": str(metadata.get("source_layer_name") or "").strip(),
    }


def _layer_common_state(layer) -> dict[str, Any]:
    record = {
        "layer_type": layer.__class__.__name__,
        "name": str(getattr(layer, "name", "")),
        "visible": bool(getattr(layer, "visible", True)),
        "opacity": float(getattr(layer, "opacity", 1.0)),
        "blending": str(getattr(layer, "blending", "translucent")),
        "scale": [float(v) for v in _sequence_values(getattr(layer, "scale", ()))],
        "translate": [float(v) for v in _sequence_values(getattr(layer, "translate", ()))],
        "source": _safe_source_info(layer),
    }
    spectral_recipe = _spectral_recipe_for_layer(layer)
    if spectral_recipe is not None:
        record["source_recipe"] = spectral_recipe
    return record


def _asset_dir_for_manifest(path: Path) -> Path:
    return path.with_name(f"{path.stem}_assets")


def _data_shape(data_like: Any) -> tuple[int, ...]:
    shape = getattr(data_like, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(v) for v in shape)
    except Exception:
        return ()


def _axis_names_for_layer(layer, data_or_shape: Any) -> list[str]:
    shape = tuple(int(v) for v in data_or_shape) if isinstance(data_or_shape, tuple) else _data_shape(data_or_shape)
    ndim = len(shape)
    rgb = bool(isinstance(layer, napari.layers.Image) and getattr(layer, "rgb", False))
    if rgb:
        mapping = {
            3: ["y", "x", "c"],
            4: ["z", "y", "x", "c"],
            5: ["t", "z", "y", "x", "c"],
        }
        axes = mapping.get(ndim)
        if axes is not None:
            return axes
    mapping = {
        2: ["y", "x"],
        3: ["z", "y", "x"],
        4: ["t", "z", "y", "x"],
        5: ["t", "c", "z", "y", "x"],
    }
    axes = mapping.get(ndim)
    if axes is not None:
        return axes
    return [f"dim_{index}" for index in range(ndim)]


def _ome_axes_metadata(axis_names: list[str]) -> list[dict[str, str]]:
    axes: list[dict[str, str]] = []
    for axis in axis_names:
        if axis == "c":
            axes.append({"name": axis, "type": "unknown"})
        elif axis == "t":
            axes.append({"name": axis, "type": "time", "unit": "second"})
        elif axis in {"x", "y", "z"}:
            axes.append({"name": axis, "type": "space", "unit": "micrometer"})
        else:
            axes.append({"name": axis, "type": "space", "unit": "micrometer"})
    return axes


def _ome_scale_for_layer(layer, axis_names: list[str]) -> list[float]:
    values = [float(v) for v in _sequence_values(getattr(layer, "scale", ()))]
    if not values:
        values = [1.0] * int(getattr(layer, "ndim", len(axis_names)))
    if len(values) == len(axis_names):
        return values
    if axis_names and axis_names[-1] == "c" and len(values) == len(axis_names) - 1:
        return values + [1.0]
    if len(values) < len(axis_names):
        values = [1.0] * (len(axis_names) - len(values)) + values
    return values[: len(axis_names)]


def _suggest_chunks(shape: tuple[int, ...], axis_names: list[str]) -> tuple[int, ...]:
    chunks: list[int] = []
    for size, axis in zip(shape, axis_names):
        size = int(size)
        if axis == "t":
            chunks.append(min(size, 1))
        elif axis == "z":
            chunks.append(min(size, 16))
        elif axis in {"y", "x"}:
            chunks.append(min(size, 256))
        elif axis == "c":
            chunks.append(size)
        else:
            chunks.append(min(size, 64))
    return tuple(chunks)


def _microscopy_metadata(layer, axis_names: list[str]) -> dict[str, Any]:
    layer_metadata = getattr(layer, "metadata", None)
    metadata = dict(layer_metadata) if isinstance(layer_metadata, dict) else {}
    vendor = (
        metadata.get("vendor")
        or metadata.get("manufacturer")
        or metadata.get("microscope_vendor")
        or metadata.get("instrument_vendor")
        or "unknown"
    )
    model = metadata.get("model") or metadata.get("microscope_model") or metadata.get("instrument_model") or "unknown"
    acquisition = (
        metadata.get("software")
        or metadata.get("acquisition_software")
        or metadata.get("application")
        or "unknown"
    )
    scale = _ome_scale_for_layer(layer, axis_names)
    placeholder = {
        "metadata_source": "layer.metadata" if metadata else "placeholder",
        "vendor": str(vendor),
        "instrument_model": str(model),
        "acquisition_software": str(acquisition),
        "axes": list(axis_names),
        "scale": list(scale),
        "units": [
            "micrometer" if axis in {"x", "y", "z"} else "second" if axis == "t" else "channel"
            for axis in axis_names
        ],
    }
    if metadata:
        placeholder["source_metadata"] = _json_ready(metadata)
    return placeholder


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


def _serialize_asset_image_like(layer, layer_type: str, *, asset_dir: Path, asset_name: str) -> dict[str, Any] | None:
    data = np.asarray(getattr(layer, "data", None))
    if data.size <= 0:
        return None
    asset_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{asset_name}.ome.zarr"
    path = asset_dir / filename
    axis_names = _axis_names_for_layer(layer, data)
    axes = _ome_axes_metadata(axis_names)
    chunks = _suggest_chunks(tuple(int(v) for v in data.shape), axis_names)
    transform = [[{"type": "scale", "scale": _ome_scale_for_layer(layer, axis_names)}]]
    root = zarr.open_group(str(path), mode="w")
    microscopy_metadata = _microscopy_metadata(layer, axis_names)
    if layer_type == "Labels":
        write_labels(
            data,
            root,
            name=OME_ZARR_LABEL_NAME,
            scaler=Scaler(max_layer=0),
            axes=axes,
            coordinate_transformations=transform,
            chunks=chunks,
            compute=True,
            label_metadata={"image-label": {"colors": []}},
        )
        dataset_path = f"labels/{OME_ZARR_LABEL_NAME}/0"
    else:
        write_image(
            data,
            root,
            scaler=Scaler(max_layer=0),
            axes=axes,
            coordinate_transformations=transform,
            chunks=chunks,
            compute=True,
            omero={"name": str(getattr(layer, "name", asset_name))},
        )
        dataset_path = "0"
    root.attrs["napari_chat_assistant"] = {
        "workspace_layer_name": str(getattr(layer, "name", asset_name)),
        "microscopy_metadata": microscopy_metadata,
    }
    return {
        "asset_path": filename,
        "asset_format": "ome-zarr",
        "asset_dataset": dataset_path,
        "dtype": str(data.dtype),
        "shape": list(data.shape),
        "layer_type": layer_type,
        "rgb": bool(getattr(layer, "rgb", False)) if isinstance(layer, napari.layers.Image) else False,
        "microscopy_metadata": microscopy_metadata,
    }


def _load_spectral_layer_from_recipe(viewer: napari.Viewer, record: dict[str, Any]):
    recipe = record.get("source_recipe") or {}
    if not isinstance(recipe, dict):
        return None
    if str(recipe.get("kind", "")).strip() != "spectral_view":
        return None
    source_path = str(recipe.get("source_path", "")).strip()
    if not source_path:
        raise ValueError("spectral view recipe is missing source_path")
    try:
        build_layer_data = getattr(
            importlib.import_module("napari_nd2_spectral_ome_zarr._reader"),
            "build_layer_data",
        )
    except Exception as exc:
        raise RuntimeError(
            "Spectral workspace restore requires napari-nd2-spectral-ome-zarr to be installed."
        ) from exc

    view_type = str(recipe.get("view_type", "")).strip().lower()
    include_visible = view_type == "visible_sum"
    include_truecolor = view_type == "truecolor"
    include_raw = view_type == "raw_spectral"
    payloads = build_layer_data(
        source_path,
        use_gpu=False,
        include_visible_layer=include_visible,
        include_truecolor_layer=include_truecolor,
        include_raw_layer=include_raw,
        zarr_use_preview=bool(recipe.get("zarr_use_preview", True)),
    )
    image_payloads = [item for item in payloads if len(item) == 3 and item[2] == "image"]
    if not image_payloads:
        raise ValueError(f"No spectral image payload could be rebuilt from {source_path}")
    if len(image_payloads) != 1:
        raise ValueError(f"Expected one spectral payload for {view_type}, got {len(image_payloads)}")
    data, kwargs, _layer_type = image_payloads[0]
    return viewer.add_image(data, **kwargs)


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
        "edge_width": _json_ready(getattr(layer, "edge_width", 1.0)),
        "edge_color": _json_ready(getattr(layer, "edge_color", None)),
        "face_color": _json_ready(getattr(layer, "face_color", None)),
    }


def _serialize_points(layer: napari.layers.Points) -> dict[str, Any]:
    metadata = dict(getattr(layer, "metadata", {}) or {})
    features = getattr(layer, "features", None)
    feature_payload: dict[str, Any] = {}
    if features is not None:
        try:
            for key in list(features.columns):
                feature_payload[str(key)] = _json_ready(list(features[key]))
        except Exception:
            try:
                for key, value in dict(features).items():
                    feature_payload[str(key)] = _json_ready(value)
            except Exception:
                feature_payload = {}
    feature_defaults_payload: dict[str, Any] = {}
    feature_defaults = getattr(layer, "feature_defaults", None)
    if feature_defaults is not None:
        try:
            for key in list(feature_defaults.columns):
                values = list(feature_defaults[key])
                feature_defaults_payload[str(key)] = _json_ready(values[0] if values else None)
        except Exception:
            feature_defaults_payload = {}
    text_payload = None
    managed_text_style = metadata.get("text_annotation_text_style")
    if isinstance(managed_text_style, dict) and managed_text_style:
        text_payload = _json_ready(dict(managed_text_style))
    else:
        text = getattr(layer, "text", None)
        if text is not None:
            string_value = getattr(getattr(text, "string", None), "format", None)
            if string_value:
                text_payload = {
                    "string": str(string_value),
                    "size": float(getattr(text, "size", 12.0)),
                    "visible": bool(getattr(text, "visible", True)),
                    "anchor": str(getattr(text, "anchor", "center")),
                    "translation": _json_ready(np.asarray(getattr(text, "translation", 0.0), dtype=float)),
                    "rotation": float(getattr(text, "rotation", 0.0)),
                    "scaling": bool(getattr(text, "scaling", False)),
                    "blending": str(
                        getattr(getattr(text, "blending", None), "value", getattr(text, "blending", "translucent"))
                    ),
                }
                color_value = getattr(getattr(text, "color", None), "constant", None)
                if color_value is not None:
                    try:
                        text_payload["color"] = _json_ready(np.asarray(color_value, dtype=float))
                    except Exception:
                        pass
    return {
        "inline_kind": "Points",
        "data": _json_ready(np.asarray(getattr(layer, "data", []), dtype=float)),
        "features": feature_payload,
        "feature_defaults": feature_defaults_payload,
        "size": _json_ready(getattr(layer, "size", 10)),
        "symbol": _json_ready(getattr(layer, "symbol", "o")),
        "face_color": _json_ready(getattr(layer, "face_color", None)),
        "border_color": _json_ready(getattr(layer, "border_color", None)),
        "border_width": _json_ready(getattr(layer, "border_width", None)),
        "shown": _json_ready(getattr(layer, "shown", None)),
        "out_of_slice_display": bool(getattr(layer, "out_of_slice_display", False)),
        "text": text_payload,
    }


def capture_workspace_manifest(viewer: napari.Viewer, *, asset_dir: Path | None = None) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    generated_count = 0
    for layer_index, layer in enumerate(list(viewer.layers)):
        record = _layer_common_state(layer)
        source_path = record["source"].get("path")
        source_recipe = record.get("source_recipe")
        layer_type = str(record["layer_type"])
        if source_path or source_recipe:
            if isinstance(layer, napari.layers.Image):
                record.update(
                    {
                        "contrast_limits": [float(v) for v in _sequence_values(getattr(layer, "contrast_limits", ()))],
                        "colormap": str(getattr(getattr(layer, "colormap", None), "name", "gray") or "gray"),
                        "gamma": float(getattr(layer, "gamma", 1.0)),
                        "rgb": bool(getattr(layer, "rgb", False)),
                        "microscopy_metadata": _microscopy_metadata(
                            layer,
                            _axis_names_for_layer(layer, _data_shape(getattr(layer, "data", None))),
                        ),
                    }
                )
            if isinstance(layer, napari.layers.Labels):
                record["microscopy_metadata"] = _microscopy_metadata(
                    layer,
                    _axis_names_for_layer(layer, _data_shape(getattr(layer, "data", None))),
                )
            layers.append(record)
            continue

        if isinstance(layer, napari.layers.Shapes):
            record.update(_serialize_shapes(layer))
            layers.append(record)
            continue

        if isinstance(layer, napari.layers.Points):
            record.update(_serialize_points(layer))
            layers.append(record)
            continue

        if isinstance(layer, napari.layers.Image):
            if asset_dir is not None:
                asset = _serialize_asset_image_like(
                    layer,
                    "Image",
                    asset_dir=asset_dir,
                    asset_name=f"layer_{layer_index:03d}_{generated_count:03d}_image",
                )
                generated_count += 1
                if asset is not None:
                    record.update(asset)
                    record.update(
                        {
                            "contrast_limits": [float(v) for v in tuple(getattr(layer, "contrast_limits", ()) or ())],
                            "colormap": str(getattr(getattr(layer, "colormap", None), "name", "gray") or "gray"),
                            "gamma": float(getattr(layer, "gamma", 1.0)),
                        }
                    )
                    layers.append(record)
                    continue

        if isinstance(layer, napari.layers.Labels):
            if asset_dir is not None:
                asset = _serialize_asset_image_like(
                    layer,
                    "Labels",
                    asset_dir=asset_dir,
                    asset_name=f"layer_{layer_index:03d}_{generated_count:03d}_labels",
                )
                generated_count += 1
                if asset is not None:
                    record.update(asset)
                    layers.append(record)
                    continue

        skipped.append({"name": str(record["name"]), "reason": "no recoverable file path or inline serialization"})

    selected = getattr(getattr(viewer.layers, "selection", None), "active", None)
    return {
        "version": WORKSPACE_VERSION,
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
    asset_dir = _asset_dir_for_manifest(path)
    temp_asset_dir = path.with_name(f"{path.stem}_assets.__tmp__")
    temp_manifest_path = path.with_name(f"{path.stem}.json.__tmp__")
    if temp_asset_dir.exists():
        shutil.rmtree(temp_asset_dir)
    if temp_manifest_path.exists():
        temp_manifest_path.unlink()

    manifest = capture_workspace_manifest(viewer, asset_dir=temp_asset_dir)
    temp_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    if temp_asset_dir.exists():
        temp_asset_dir.replace(asset_dir)
    temp_manifest_path.replace(path)
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
        if value is not None and len(value) > 0:
            try:
                setattr(layer, attr, tuple(value))
            except Exception:
                pass
    if isinstance(layer, napari.layers.Image):
        contrast_limits = record.get("contrast_limits")
        if contrast_limits is not None and len(contrast_limits) > 0:
            try:
                layer.contrast_limits = tuple(float(v) for v in contrast_limits)
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
    microscopy_metadata = record.get("microscopy_metadata")
    if microscopy_metadata is not None:
        try:
            metadata = dict(getattr(layer, "metadata", {}) or {})
            metadata["microscopy_metadata"] = microscopy_metadata
            layer.metadata = metadata
        except Exception:
            pass


def _restore_inline_layer(viewer: napari.Viewer, record: dict[str, Any]):
    inline_kind = str(record.get("inline_kind") or record.get("layer_type") or "")
    if inline_kind == "Shapes":
        features = record.get("features") or {}
        return viewer.add_shapes(
            data=[[tuple(point) for point in shape] for shape in record.get("data", [])],
            shape_type=list(record.get("shape_type", [])) or "polygon",
            edge_width=record.get("edge_width", 1.0),
            edge_color=record.get("edge_color", None),
            face_color=record.get("face_color", None),
            name=str(record.get("name", "Shapes")),
            features=features,
        )
    if inline_kind == "Points":
        layer = viewer.add_points(
            data=np.asarray(record.get("data", []), dtype=float),
            name=str(record.get("name", "Points")),
            features=record.get("features") or {},
            size=record.get("size", 10),
            symbol=record.get("symbol", "o"),
            face_color=record.get("face_color", None),
            border_color=record.get("border_color", None),
            border_width=record.get("border_width", None),
            shown=record.get("shown", None),
            out_of_slice_display=bool(record.get("out_of_slice_display", False)),
        )
        feature_defaults = record.get("feature_defaults") or {}
        if feature_defaults:
            try:
                layer.feature_defaults = feature_defaults
            except Exception:
                pass
            try:
                layer.current_properties = {
                    str(key): np.asarray([value], dtype=object)
                    for key, value in dict(feature_defaults).items()
                }
            except Exception:
                pass
        text_payload = record.get("text")
        if isinstance(text_payload, dict) and text_payload:
            try:
                layer.text = dict(text_payload)
            except Exception:
                pass
        return layer
    if inline_kind in {"Labels", "Image"} and ("inline_data" in record or "asset_path" in record):
        if "inline_data" in record:
            data = np.asarray(record["inline_data"], dtype=np.dtype(record.get("dtype") or "float32"))
        else:
            raise ValueError("asset-backed restore requires manifest-relative asset resolution")
        if inline_kind == "Labels":
            return viewer.add_labels(data, name=str(record.get("name", "Labels")))
        return viewer.add_image(data, name=str(record.get("name", "Image")), rgb=bool(record.get("rgb", False)))
    return None


def _restore_layer_from_record(viewer: napari.Viewer, record: dict[str, Any], *, manifest_path: Path):
    source_info = record.get("source") or {}
    source_path = source_info.get("path")
    source_recipe = record.get("source_recipe") or {}
    if isinstance(source_recipe, dict) and source_recipe.get("kind") == "spectral_view":
        return _load_spectral_layer_from_recipe(viewer, record)
    if source_path and Path(source_path).exists():
        loaded = viewer.open(
            [source_path],
            stack=False,
            plugin=source_info.get("reader_plugin") or None,
            layer_type=str(record.get("layer_type", "")).lower() or None,
        )
        return loaded[-1] if loaded else None

    if "asset_path" in record:
        asset_dir = _asset_dir_for_manifest(manifest_path)
        asset_path = asset_dir / str(record.get("asset_path", "")).strip()
        if not asset_path.exists():
            raise FileNotFoundError(f"Missing workspace asset: {asset_path}")
        dataset_path = str(record.get("asset_dataset") or "0").strip().strip("/")
        data = zarr.open(str(asset_path / dataset_path), mode="r")
        inline_kind = str(record.get("layer_type") or "")
        if inline_kind == "Labels":
            return viewer.add_labels(data, name=str(record.get("name", "Labels")))
        return viewer.add_image(data, name=str(record.get("name", "Image")), rgb=bool(record.get("rgb", False)))

    return _restore_inline_layer(viewer, record)


def _remove_all_layers(viewer: napari.Viewer) -> None:
    while len(viewer.layers):
        viewer.layers.remove(viewer.layers[0])


def read_workspace_manifest(source: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(source).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload


def clear_workspace_layers(viewer: napari.Viewer) -> None:
    _remove_all_layers(viewer)


def workspace_record_loading_kind(record: dict[str, Any]) -> str:
    source_info = record.get("source") or {}
    source_path = str(source_info.get("path") or "").strip()
    source_recipe = record.get("source_recipe") or {}
    if isinstance(source_recipe, dict) and str(source_recipe.get("kind") or "").strip():
        return "recipe"
    if source_path:
        return "source"
    if "asset_path" in record:
        return "asset"
    return "inline"


def restore_workspace_layer(viewer: napari.Viewer, record: dict[str, Any], *, manifest_path: str | Path):
    path = Path(manifest_path).expanduser()
    layer = _restore_layer_from_record(viewer, record, manifest_path=path)
    if layer is None:
        return None
    _apply_common_state(layer, record)
    return layer


def apply_workspace_viewer_state(viewer: napari.Viewer, payload: dict[str, Any]) -> None:
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


def load_workspace_manifest(viewer: napari.Viewer, source: str | Path, *, clear_existing: bool = True) -> dict[str, Any]:
    path, payload = read_workspace_manifest(source)
    if clear_existing:
        clear_workspace_layers(viewer)

    restored_names: list[str] = []
    skipped_layers: list[dict[str, str]] = []
    for record in payload.get("layers", []):
        try:
            layer = restore_workspace_layer(viewer, record, manifest_path=path)
        except Exception as exc:
            skipped_layers.append({"name": str(record.get("name", "unknown")), "reason": str(exc)})
            continue

        if layer is None:
            skipped_layers.append(
                {"name": str(record.get("name", "unknown")), "reason": "could not restore layer from source or inline data"}
            )
            continue
        restored_names.append(str(getattr(layer, "name", "")))

    apply_workspace_viewer_state(viewer, payload)

    return {
        "path": str(path),
        "restored_layers": restored_names,
        "skipped_layers": skipped_layers,
    }
