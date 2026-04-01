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
    model = build_sam2(str(config_path), str(checkpoint_path), device=resolved_device)
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
