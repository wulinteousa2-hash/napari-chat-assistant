from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import napari
import numpy as np


PHASE1_SEMANTIC_TYPES = {
    "2d_intensity",
    "3d_intensity",
    "time_series",
    "multichannel_fluorescence",
    "rgb",
    "spectral",
    "label_mask",
    "probability_map",
    "unknown",
}

EVIDENCE_BUCKETS = (
    "structural",
    "semantic_napari",
    "statistical",
    "format_metadata",
    "reader_specific",
)


@dataclass
class DatasetProfile:
    layer_name: str
    layer_class: str
    semantic_type: str
    confidence: str
    axes_detected: str
    source_kind: str
    shape: list[int] | None
    dtype: str | None
    is_multiscale: bool
    is_lazy_or_chunked: bool
    pixel_or_voxel_scale_present: bool
    time_calibration_present: bool
    channel_metadata_present: bool
    wavelength_metadata_present: bool
    recommended_operation_classes: list[str]
    discouraged_operation_classes: list[str]
    reasons: list[str]
    evidence: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleFacts:
    layer: Any
    layer_name: str
    metadata: dict[str, Any]
    multiscale: bool
    data_obj: Any
    shape: list[int] | None
    dtype: str | None
    axes: str
    scalar_ndim: int | None
    lazy_or_chunked: bool
    scale: tuple[float, ...]
    pixel_scale_present: bool
    time_calibration_present: bool
    channel_names: list[str]
    wavelengths: list[str]
    channel_metadata_present: bool
    wavelength_metadata_present: bool
    sampled: np.ndarray | None
    finite_sample: np.ndarray | None
    value_range_01: bool
    binary_like: bool


@dataclass
class HypothesisState:
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def add(self, score: float, reason: str):
        self.score += float(score)
        self.reasons.append(reason)


def profile_layer(layer) -> dict[str, Any]:
    if layer is None:
        return DatasetProfile(
            layer_name="",
            layer_class="Unknown",
            semantic_type="unknown",
            confidence="low",
            axes_detected="unknown",
            source_kind="unknown",
            shape=None,
            dtype=None,
            is_multiscale=False,
            is_lazy_or_chunked=False,
            pixel_or_voxel_scale_present=False,
            time_calibration_present=False,
            channel_metadata_present=False,
            wavelength_metadata_present=False,
            recommended_operation_classes=[],
            discouraged_operation_classes=[],
            reasons=["No valid napari layer was available."],
            evidence={bucket: [] for bucket in EVIDENCE_BUCKETS},
        ).to_dict()

    facts = _build_rule_facts(layer)
    evidence = _collect_evidence_buckets(facts)
    hypotheses = _score_hypotheses(facts)

    ranked = sorted(hypotheses.items(), key=lambda item: item[1].score, reverse=True)
    best_type, best_state = ranked[0]
    runner_up_score = ranked[1][1].score if len(ranked) > 1 else 0.0
    if best_state.score < 0.45:
        best_type = "unknown"
        best_reasons = ["Evidence was insufficient for a high-confidence Phase 1 classification."]
    else:
        best_reasons = best_state.reasons

    confidence = _score_to_confidence(best_state.score, runner_up_score)
    recommended, discouraged = operation_classes_for_semantic_type(best_type)
    return DatasetProfile(
        layer_name=facts.layer_name,
        layer_class=layer.__class__.__name__,
        semantic_type=best_type,
        confidence=confidence,
        axes_detected=facts.axes or "unknown",
        source_kind=_source_kind(layer),
        shape=facts.shape,
        dtype=facts.dtype,
        is_multiscale=facts.multiscale,
        is_lazy_or_chunked=facts.lazy_or_chunked,
        pixel_or_voxel_scale_present=facts.pixel_scale_present,
        time_calibration_present=facts.time_calibration_present,
        channel_metadata_present=facts.channel_metadata_present,
        wavelength_metadata_present=facts.wavelength_metadata_present,
        recommended_operation_classes=recommended,
        discouraged_operation_classes=discouraged,
        reasons=_unique_preserve_order(_flatten_evidence(evidence) + best_reasons),
        evidence=evidence,
    ).to_dict()


def operation_classes_for_semantic_type(semantic_type: str) -> tuple[list[str], list[str]]:
    recommended_map = {
        "label_mask": ["mask_measurement", "morphology", "roi_qc"],
        "rgb": ["visual_review", "annotation"],
        "2d_intensity": ["contrast_enhancement", "thresholding", "annotation"],
        "3d_intensity": ["contrast_enhancement", "thresholding", "volume_review"],
        "time_series": ["frame_navigation", "temporal_qc", "tracking_preparation"],
        "multichannel_fluorescence": ["channel_selection", "contrast_enhancement", "segmentation_preparation"],
        "spectral": ["spectral_channel_review", "spectral_unmixing_preparation", "metadata_qc"],
        "probability_map": ["thresholding", "calibration_review", "overlay_review"],
        "unknown": ["visual_review", "metadata_inspection"],
    }
    discouraged_map = {
        "label_mask": ["clahe_on_labels", "rgb_interpretation"],
        "rgb": ["single_channel_quantification", "label_morphology"],
        "2d_intensity": ["rgb_interpretation"],
        "3d_intensity": ["rgb_interpretation"],
        "time_series": ["single_frame_assumptions"],
        "multichannel_fluorescence": ["rgb_assumptions"],
        "spectral": ["rgb_assumptions", "naive_channel_collapse"],
        "probability_map": ["label_morphology_before_threshold", "discrete_label_measurement"],
        "unknown": [],
    }
    return (
        list(recommended_map.get(semantic_type, recommended_map["unknown"])),
        list(discouraged_map.get(semantic_type, discouraged_map["unknown"])),
    )


def _build_rule_facts(layer) -> RuleFacts:
    metadata = _coerce_metadata(getattr(layer, "metadata", None))
    multiscale = bool(getattr(layer, "multiscale", False))
    data_obj = _canonical_data_object(getattr(layer, "data", None), multiscale=multiscale)
    shape = _shape_list(data_obj)
    dtype = _dtype_text(data_obj)
    axes = _detect_axes(layer, metadata, shape)
    scalar_ndim = _semantic_ndim(layer, shape)
    lazy_or_chunked = _is_lazy_or_chunked(data_obj)
    scale = _coerce_float_tuple(getattr(layer, "scale", None))
    pixel_scale_present = any(abs(v - 1.0) > 1e-12 for v in scale)
    time_calibration_present = _has_any_key(metadata, {"frame_interval", "time_increment", "dt", "fps", "timestamps"})
    channel_names = _extract_channel_names(metadata)
    wavelengths = _extract_wavelengths(metadata)
    sampled = _sample_numeric_array(data_obj)
    finite_sample = None
    value_range_01 = False
    binary_like = False
    if sampled is not None and sampled.size:
        finite_sample = sampled[np.isfinite(sampled)]
        if finite_sample.size:
            value_range_01 = float(np.min(finite_sample)) >= -1e-6 and float(np.max(finite_sample)) <= 1.0 + 1e-6
            if np.issubdtype(finite_sample.dtype, np.integer) or np.issubdtype(finite_sample.dtype, np.bool_):
                unique = np.unique(finite_sample)
                binary_like = unique.size <= 2 and set(unique.tolist()).issubset({0, 1, False, True})
    return RuleFacts(
        layer=layer,
        layer_name=str(getattr(layer, "name", "")),
        metadata=metadata,
        multiscale=multiscale,
        data_obj=data_obj,
        shape=shape,
        dtype=dtype,
        axes=axes,
        scalar_ndim=scalar_ndim,
        lazy_or_chunked=lazy_or_chunked,
        scale=scale,
        pixel_scale_present=pixel_scale_present,
        time_calibration_present=time_calibration_present,
        channel_names=channel_names,
        wavelengths=wavelengths,
        channel_metadata_present=bool(channel_names),
        wavelength_metadata_present=bool(wavelengths),
        sampled=sampled,
        finite_sample=finite_sample,
        value_range_01=value_range_01,
        binary_like=binary_like,
    )


def _collect_evidence_buckets(facts: RuleFacts) -> dict[str, list[str]]:
    evidence = {bucket: [] for bucket in EVIDENCE_BUCKETS}
    evidence["structural"].append(f"Observed shape {tuple(facts.shape) if facts.shape is not None else 'unknown'}.")
    evidence["structural"].append(f"Detected axes {facts.axes or 'unknown'}.")
    if facts.scalar_ndim is not None:
        evidence["structural"].append(f"Effective semantic ndim is {facts.scalar_ndim}.")
    if facts.multiscale:
        evidence["structural"].append("Layer is multiscale.")
    if facts.lazy_or_chunked:
        evidence["structural"].append("Layer data appears lazy or chunked.")

    evidence["semantic_napari"].append(f"Layer class is {facts.layer.__class__.__name__}.")
    if isinstance(facts.layer, napari.layers.Image):
        evidence["semantic_napari"].append(f"napari rgb flag is {bool(getattr(facts.layer, 'rgb', False))}.")
    if facts.pixel_scale_present:
        evidence["semantic_napari"].append("Non-unit pixel or voxel scale is present.")

    if facts.dtype:
        evidence["statistical"].append(f"Observed dtype {facts.dtype}.")
    if facts.finite_sample is not None and facts.finite_sample.size:
        evidence["statistical"].append(f"Sampled {int(facts.finite_sample.size)} finite values for heuristics.")
        if facts.value_range_01:
            evidence["statistical"].append("Sampled values fit within [0, 1].")
        if facts.binary_like:
            evidence["statistical"].append("Sampled values look binary.")

    if facts.channel_metadata_present:
        evidence["format_metadata"].append(f"Channel metadata present: {facts.channel_names}.")
    if facts.wavelength_metadata_present:
        evidence["format_metadata"].append(f"Wavelength metadata present: {facts.wavelengths}.")
    if facts.time_calibration_present:
        evidence["format_metadata"].append("Time calibration metadata is present.")
    if _first_present_value(facts.metadata, ("ome_axes", "axes", "axis_labels", "dimension_order")) is not None:
        evidence["format_metadata"].append("Axis metadata is present.")

    reader_name = _reader_name_from_metadata(facts.metadata)
    if reader_name:
        evidence["reader_specific"].append(f"Reader/source metadata indicates {reader_name}.")

    return evidence


def _score_hypotheses(facts: RuleFacts) -> dict[str, HypothesisState]:
    states = {name: HypothesisState() for name in PHASE1_SEMANTIC_TYPES if name != "unknown"}
    for rule in (
        _rule_label_mask,
        _rule_rgb,
        _rule_spectral,
        _rule_time_series,
        _rule_multichannel_fluorescence,
        _rule_probability_map,
        _rule_scalar_intensity,
    ):
        rule(facts, states)
    return states


def _rule_label_mask(facts: RuleFacts, states: dict[str, HypothesisState]) -> None:
    if isinstance(facts.layer, napari.layers.Labels):
        states["label_mask"].add(1.2, "napari layer type is Labels, which carries segmentation semantics.")
        if facts.dtype:
            try:
                if "bool" in facts.dtype or np.issubdtype(np.dtype(facts.dtype), np.integer):
                    states["label_mask"].add(0.2, "Labels data is boolean or integer typed.")
            except Exception:
                pass
        if facts.binary_like:
            states["label_mask"].add(0.1, "Sampled labels look binary or ID-like.")


def _rule_rgb(facts: RuleFacts, states: dict[str, HypothesisState]) -> None:
    if not isinstance(facts.layer, napari.layers.Image):
        return
    if bool(getattr(facts.layer, "rgb", False)):
        states["rgb"].add(1.1, "Image layer is marked as rgb by napari.")
        return
    if facts.shape and facts.shape[-1] in (3, 4) and not facts.channel_metadata_present and not facts.wavelength_metadata_present:
        states["rgb"].add(0.45, "Trailing dimension has length 3 or 4 without scientific channel metadata.")


def _rule_spectral(facts: RuleFacts, states: dict[str, HypothesisState]) -> None:
    if not isinstance(facts.layer, napari.layers.Image):
        return
    if facts.wavelength_metadata_present:
        states["spectral"].add(0.95, "Wavelength metadata is present.")
    if _looks_spectral_channel_names(facts.channel_names):
        states["spectral"].add(0.7, "Channel metadata looks wavelength-like.")


def _rule_time_series(facts: RuleFacts, states: dict[str, HypothesisState]) -> None:
    if not isinstance(facts.layer, napari.layers.Image):
        return
    if _axes_has_time(facts.axes):
        states["time_series"].add(0.7, "Detected a time axis in the metadata or inferred axes.")
    if facts.time_calibration_present:
        states["time_series"].add(0.25, "Time calibration metadata is present.")


def _rule_multichannel_fluorescence(facts: RuleFacts, states: dict[str, HypothesisState]) -> None:
    if not isinstance(facts.layer, napari.layers.Image):
        return
    if bool(getattr(facts.layer, "rgb", False)):
        return
    if _axes_has_channel(facts.axes):
        states["multichannel_fluorescence"].add(0.45, "Detected a dedicated channel axis in metadata or inferred axes.")
    if facts.channel_metadata_present and not facts.wavelength_metadata_present:
        states["multichannel_fluorescence"].add(0.5, "Scientific channel metadata is present without rgb semantics.")


def _rule_probability_map(facts: RuleFacts, states: dict[str, HypothesisState]) -> None:
    if not isinstance(facts.layer, napari.layers.Image):
        return
    if facts.finite_sample is None or not facts.finite_sample.size:
        return
    if np.issubdtype(facts.finite_sample.dtype, np.floating) and facts.value_range_01:
        states["probability_map"].add(0.65, "Sampled values are floating point and bounded within [0, 1].")
        if any(token in facts.layer_name.lower() for token in ("prob", "probability", "score", "confidence", "logit")):
            states["probability_map"].add(0.25, "Layer name suggests a score or probability output.")


def _rule_scalar_intensity(facts: RuleFacts, states: dict[str, HypothesisState]) -> None:
    if not isinstance(facts.layer, napari.layers.Image):
        return
    if bool(getattr(facts.layer, "rgb", False)):
        return
    if _axes_has_time(facts.axes) or _axes_has_channel(facts.axes) or facts.wavelength_metadata_present:
        return
    if facts.scalar_ndim == 2:
        states["2d_intensity"].add(0.55, "Scalar image has two effective spatial dimensions.")
        states["2d_intensity"].add(0.15, "No stronger time, channel, spectral, or rgb evidence was found.")
    elif facts.scalar_ndim == 3:
        states["3d_intensity"].add(0.55, "Scalar image has three effective spatial dimensions.")
        states["3d_intensity"].add(0.15, "No stronger time, channel, spectral, or rgb evidence was found.")


def _canonical_data_object(data, *, multiscale: bool):
    if multiscale and isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return data[0] if len(data) > 0 else None
    return data


def _shape_list(data) -> list[int] | None:
    shape = getattr(data, "shape", None)
    if shape is None:
        return None
    return [int(v) for v in shape]


def _dtype_text(data) -> str | None:
    dtype = getattr(data, "dtype", None)
    return str(dtype) if dtype is not None else None


def _semantic_ndim(layer, shape: list[int] | None) -> int | None:
    ndim = getattr(layer, "ndim", None)
    if ndim is not None:
        try:
            return int(ndim)
        except Exception:
            pass
    if shape is None:
        return None
    if bool(getattr(layer, "rgb", False)) and len(shape) >= 1:
        return len(shape) - 1
    return len(shape)


def _source_kind(layer) -> str:
    if isinstance(layer, napari.layers.Image):
        return "napari_image_layer"
    if isinstance(layer, napari.layers.Labels):
        return "napari_labels_layer"
    return "unknown"


def _detect_axes(layer, metadata: dict[str, Any], shape: list[int] | None) -> str:
    explicit = _normalize_axes_value(
        _first_present_value(
            metadata,
            ("axes", "axis_labels", "axes_labels", "dimension_order", "dims", "ome_axes"),
        )
    )
    if explicit:
        return explicit

    ndim = _semantic_ndim(layer, shape)
    if ndim is None:
        return ""
    if isinstance(layer, napari.layers.Labels):
        return {2: "YX", 3: "ZYX", 4: "TZYX"}.get(ndim, "unknown")
    if bool(getattr(layer, "rgb", False)):
        return {2: "YXC", 3: "ZYXC", 4: "TZYXC"}.get(ndim, "unknown")
    return {2: "YX", 3: "ZYX", 4: "TZYX", 5: "TCZYX"}.get(ndim, "unknown")


def _normalize_axes_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        letters = [char.upper() for char in value if char.upper() in {"T", "C", "Z", "Y", "X"}]
        return "".join(letters)
    if isinstance(value, Sequence):
        letters: list[str] = []
        for item in value:
            text = str(item).strip().upper()
            if not text:
                continue
            if text[0] in {"T", "C", "Z", "Y", "X"}:
                letters.append(text[0])
        return "".join(letters)
    return ""


def _has_any_key(metadata: dict[str, Any], keys: set[str]) -> bool:
    normalized_keys = {str(key).lower() for key in metadata}
    return any(key in normalized_keys for key in keys)


def _extract_channel_names(metadata: dict[str, Any]) -> list[str]:
    for key in ("channel_names", "channels", "channel_labels", "channel_metadata"):
        value = metadata.get(key)
        if isinstance(value, dict):
            names = value.get("names")
            if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
                return [str(item) for item in names if str(item).strip()]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(item) for item in value if str(item).strip()]
    return []


def _extract_wavelengths(metadata: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("wavelengths", "emission_wavelengths", "excitation_wavelengths", "channel_wavelengths"):
        value = metadata.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(str(item) for item in value if str(item).strip())
    spectral_axis = metadata.get("spectral_axis")
    if spectral_axis is not None:
        values.append(str(spectral_axis))
    return values


def _looks_spectral_channel_names(channel_names: list[str]) -> bool:
    if not channel_names:
        return False
    hits = 0
    for name in channel_names:
        text = name.lower()
        if "nm" in text or "wavelength" in text or text.strip().isdigit():
            hits += 1
    return hits >= max(2, len(channel_names) // 2)


def _axes_has_time(axes: str) -> bool:
    return "T" in (axes or "")


def _axes_has_channel(axes: str) -> bool:
    return "C" in (axes or "")


def _is_lazy_or_chunked(data) -> bool:
    if data is None:
        return False
    if hasattr(data, "chunks") or hasattr(data, "chunksize"):
        return True
    module = type(data).__module__
    return module.startswith("dask.") or module.startswith("zarr.")


def _sample_numeric_array(data, *, max_points: int = 4096):
    if data is None or not hasattr(data, "shape"):
        return None
    shape = getattr(data, "shape", ())
    if not shape:
        return None

    slices = []
    target = max(1, int(round(max_points ** (1 / max(len(shape), 1)))))
    for size in shape:
        size = int(size)
        if size <= target:
            slices.append(slice(None))
        else:
            step = max(1, size // target)
            slices.append(slice(0, size, step))
    try:
        sampled = np.asarray(data[tuple(slices)])
    except Exception:
        try:
            sampled = np.asarray(data)
        except Exception:
            return None
    if sampled.size > max_points:
        sampled = sampled.reshape(-1)[:max_points]
    return sampled


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _score_to_confidence(best_score: float, runner_up: float) -> str:
    margin = best_score - runner_up
    if best_score >= 0.9 and margin >= 0.25:
        return "high"
    if best_score >= 0.6 and margin >= 0.1:
        return "medium"
    return "low"


def _coerce_metadata(metadata) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    try:
        return dict(metadata)
    except Exception:
        return {}


def _coerce_float_tuple(value) -> tuple[float, ...]:
    if value is None:
        return ()
    try:
        return tuple(float(v) for v in value)
    except Exception:
        return ()


def _first_present_value(metadata: dict[str, Any], keys: Sequence[str]):
    for key in keys:
        if key in metadata:
            value = metadata[key]
            if value is not None:
                return value
    return None


def _reader_name_from_metadata(metadata: dict[str, Any]) -> str:
    for key in ("reader_plugin", "reader", "source_kind", "source_type", "format"):
        value = metadata.get(key)
        if value is not None:
            return str(value)
    return ""


def _flatten_evidence(evidence: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for bucket in EVIDENCE_BUCKETS:
        out.extend(evidence.get(bucket, []))
    return out
