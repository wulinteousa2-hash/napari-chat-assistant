from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import ball, disk, remove_small_objects


def _binary_mask(data) -> np.ndarray:
    return np.asarray(data) > 0


def _footprint(ndim: int, radius: int):
    radius = max(1, int(radius))
    if ndim <= 2:
        return disk(radius)
    if ndim == 3:
        return ball(radius)
    return ndi.generate_binary_structure(ndim, 1)


def _preserve_dtype(mask: np.ndarray, template) -> np.ndarray:
    return mask.astype(np.asarray(template).dtype, copy=False)


def mask_statistics(data) -> dict:
    mask = _binary_mask(data)
    labeled, object_count = ndi.label(mask)
    if object_count:
        counts = np.bincount(labeled.ravel())[1:]
        largest = int(counts.max()) if counts.size else 0
    else:
        largest = 0
    return {
        "foreground_pixels": int(mask.sum()),
        "object_count": int(object_count),
        "largest_object": largest,
    }


def auto_threshold_mask(data, polarity: str = "auto") -> tuple[float, np.ndarray]:
    arr = np.asarray(data)
    threshold_value = float(threshold_otsu(arr))
    normalized = str(polarity or "auto").strip().lower()
    if normalized == "dim":
        mask = arr <= threshold_value
    elif normalized == "bright":
        mask = arr >= threshold_value
    else:
        mean_value = float(arr.mean())
        mask = arr >= threshold_value if threshold_value >= mean_value else arr <= threshold_value
    return threshold_value, mask.astype(np.uint8)


def dilate_binary_mask(data, radius: int = 1) -> np.ndarray:
    mask = ndi.binary_dilation(_binary_mask(data), structure=_footprint(np.asarray(data).ndim, radius))
    return _preserve_dtype(mask, data)


def erode_binary_mask(data, radius: int = 1) -> np.ndarray:
    mask = ndi.binary_erosion(_binary_mask(data), structure=_footprint(np.asarray(data).ndim, radius))
    return _preserve_dtype(mask, data)


def open_binary_mask(data, radius: int = 1) -> np.ndarray:
    mask = ndi.binary_opening(_binary_mask(data), structure=_footprint(np.asarray(data).ndim, radius))
    return _preserve_dtype(mask, data)


def close_binary_mask(data, radius: int = 1) -> np.ndarray:
    mask = ndi.binary_closing(_binary_mask(data), structure=_footprint(np.asarray(data).ndim, radius))
    return _preserve_dtype(mask, data)


def fill_holes(data) -> np.ndarray:
    mask = ndi.binary_fill_holes(_binary_mask(data))
    return _preserve_dtype(mask, data)


def remove_small_components(data, min_size: int = 64) -> np.ndarray:
    mask = remove_small_objects(_binary_mask(data), min_size=max(1, int(min_size)))
    return _preserve_dtype(mask, data)


def keep_largest_component(data) -> np.ndarray:
    mask = _binary_mask(data)
    labeled, object_count = ndi.label(mask)
    if object_count == 0:
        return _preserve_dtype(mask, data)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    largest = labeled == largest_label
    return _preserve_dtype(largest, data)
