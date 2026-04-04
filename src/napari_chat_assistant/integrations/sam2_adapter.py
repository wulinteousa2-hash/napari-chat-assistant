from __future__ import annotations

from pathlib import Path

import numpy as np


def _pick_device(device: str | None = None) -> str:
    requested = str(device or "cuda").strip().lower()
    try:
        import torch
    except Exception:
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _prepare_image_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"SAM2 image adapter expects a 2D grayscale image, got ndim={array.ndim}.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"SAM2 image adapter expects numeric image data, got dtype={array.dtype}.")
    array = array.astype(np.float32, copy=False)
    if array.size == 0:
        raise ValueError("SAM2 image adapter expects a non-empty image.")
    finite = np.isfinite(array)
    if not np.any(finite):
        raise ValueError("SAM2 image adapter found no finite values in the image.")
    valid = array[finite]
    lo = float(valid.min())
    hi = float(valid.max())
    if hi > lo:
        scaled = (array - lo) / (hi - lo)
    else:
        scaled = np.zeros_like(array, dtype=np.float32)
    rgb = np.stack([scaled, scaled, scaled], axis=-1)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)


def _normalize_config_name(config_path: str) -> str:
    normalized = str(config_path or "").replace("\\", "/")
    marker = "/sam2/configs/"
    if marker in normalized:
        return "configs/" + normalized.split(marker, 1)[1]
    marker = "/configs/"
    if marker in normalized:
        return "configs/" + normalized.split(marker, 1)[1]
    if normalized.startswith("sam2/configs/"):
        return normalized[len("sam2/") :]
    return normalized


def _build_predictor(*, checkpoint_path: str, config_path: str, device: str):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise RuntimeError(
            "Could not import the official SAM2 image predictor. "
            "Install the SAM2 project into the same Python environment as napari."
        ) from exc

    resolved_device = _pick_device(device)
    config_name = _normalize_config_name(str(config_path))
    model = build_sam2(config_name, str(checkpoint_path), device=resolved_device)
    predictor = SAM2ImagePredictor(model)
    return predictor, resolved_device


def _select_mask(masks, scores=None) -> np.ndarray:
    array = np.asarray(masks)
    if array.ndim == 2:
        return (array > 0).astype(np.int32, copy=False)
    if array.ndim != 3:
        raise ValueError(f"SAM2 image predictor returned unexpected mask ndim={array.ndim}.")
    if scores is None:
        index = 0
    else:
        score_array = np.asarray(scores).reshape(-1)
        index = int(np.argmax(score_array)) if score_array.size else 0
    return (array[index] > 0).astype(np.int32, copy=False)


def _predict_with_box_and_points(
    predictor,
    *,
    box_xyxy: tuple[float, float, float, float] | None = None,
    point_coords_xy: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
) -> np.ndarray:
    kwargs: dict[str, object] = {"multimask_output": False}
    if box_xyxy is not None:
        kwargs["box"] = np.asarray(box_xyxy, dtype=np.float32)
    if point_coords_xy is not None and point_labels is not None:
        kwargs["point_coords"] = np.asarray(point_coords_xy, dtype=np.float32)
        kwargs["point_labels"] = np.asarray(point_labels, dtype=np.int32)
    try:
        masks, scores, _logits = predictor.predict(**kwargs)
    except TypeError:
        if "box" in kwargs:
            kwargs["box"] = np.asarray(kwargs["box"], dtype=np.float32)[None, :]
        masks, scores, _logits = predictor.predict(**kwargs)
    return _select_mask(masks, scores)


def _mask_bbox_xyxy(mask: np.ndarray) -> tuple[float, float, float, float]:
    coords = np.argwhere(np.asarray(mask) > 0)
    if coords.size == 0:
        raise ValueError("Mask does not contain any positive pixels.")
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    return float(min_x), float(min_y), float(max_x + 1), float(max_y + 1)


def _sample_prompt_points(mask: np.ndarray, *, max_positive: int = 4) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(mask) > 0
    coords = np.argwhere(binary)
    if coords.size == 0:
        raise ValueError("Mask does not contain any positive pixels.")
    sample_indices = np.linspace(0, len(coords) - 1, num=min(max_positive, len(coords)), dtype=int)
    positive_yx = coords[sample_indices]
    positive_xy = np.stack([positive_yx[:, 1], positive_yx[:, 0]], axis=1).astype(np.float32, copy=False)
    labels = np.ones(len(positive_xy), dtype=np.int32)
    return positive_xy, labels


def segment_image_from_box(
    *,
    image: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    checkpoint_path: str,
    config_path: str,
    device: str,
    model_name: str | None = None,
):
    del model_name
    predictor, resolved_device = _build_predictor(
        checkpoint_path=str(Path(checkpoint_path)),
        config_path=str(Path(config_path)),
        device=device,
    )
    predictor.set_image(_prepare_image_rgb(np.asarray(image)))
    box = np.asarray(box_xyxy, dtype=np.float32)
    try:
        masks, scores, _logits = predictor.predict(box=box, multimask_output=False)
    except TypeError:
        masks, scores, _logits = predictor.predict(box=box[None, :], multimask_output=False)
    return {
        "mask": _select_mask(masks, scores),
        "message": f"adapter=bundled device={resolved_device}",
    }


def segment_image_from_points(
    *,
    image: np.ndarray,
    point_coords_xy: np.ndarray,
    point_labels: np.ndarray,
    checkpoint_path: str,
    config_path: str,
    device: str,
    model_name: str | None = None,
):
    del model_name
    predictor, resolved_device = _build_predictor(
        checkpoint_path=str(Path(checkpoint_path)),
        config_path=str(Path(config_path)),
        device=device,
    )
    predictor.set_image(_prepare_image_rgb(np.asarray(image)))
    coords = np.asarray(point_coords_xy, dtype=np.float32)
    labels = np.asarray(point_labels, dtype=np.int32)
    masks, scores, _logits = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    return {
        "mask": _select_mask(masks, scores),
        "message": f"adapter=bundled device={resolved_device}",
    }


def segment_image_auto(
    *,
    image: np.ndarray,
    checkpoint_path: str,
    config_path: str,
    device: str,
    model_name: str | None = None,
):
    del model_name
    predictor, resolved_device = _build_predictor(
        checkpoint_path=str(Path(checkpoint_path)),
        config_path=str(Path(config_path)),
        device=device,
    )
    image_array = np.asarray(image)
    predictor.set_image(_prepare_image_rgb(image_array))
    height, width = image_array.shape
    box_xyxy = (0.0, 0.0, float(width - 1), float(height - 1))
    mask = _predict_with_box_and_points(predictor, box_xyxy=box_xyxy)
    return {
        "mask": mask,
        "message": f"adapter=bundled device={resolved_device} strategy=full_image_box",
    }


def refine_mask(
    *,
    image: np.ndarray,
    mask: np.ndarray,
    checkpoint_path: str,
    config_path: str,
    device: str,
    roi_mask: np.ndarray | None = None,
    model_name: str | None = None,
):
    del model_name
    predictor, resolved_device = _build_predictor(
        checkpoint_path=str(Path(checkpoint_path)),
        config_path=str(Path(config_path)),
        device=device,
    )
    image_array = np.asarray(image)
    working_mask = np.asarray(mask) > 0
    if roi_mask is not None:
        working_mask = working_mask & (np.asarray(roi_mask) > 0)
    predictor.set_image(_prepare_image_rgb(image_array))
    box_xyxy = _mask_bbox_xyxy(working_mask)
    point_coords_xy, point_labels = _sample_prompt_points(working_mask)
    refined = _predict_with_box_and_points(
        predictor,
        box_xyxy=box_xyxy,
        point_coords_xy=point_coords_xy,
        point_labels=point_labels,
    )
    return {
        "mask": refined,
        "message": f"adapter=bundled device={resolved_device} strategy=mask_bbox_plus_positive_points",
    }
