from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.morphology import ball, disk, local_maxima, remove_small_objects, skeletonize
from skimage.segmentation import watershed


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


def _finite_values(data) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def intensity_statistics(data) -> dict:
    values = _finite_values(data)
    if values.size == 0:
        raise ValueError("Image has no finite intensity values.")
    if values.size == 1:
        std_value = 0.0
    else:
        std_value = float(np.std(values, ddof=1))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": std_value,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def intensity_histogram(data, bins: int = 64) -> dict:
    values = _finite_values(data)
    if values.size == 0:
        raise ValueError("Image has no finite intensity values.")
    safe_bins = max(2, int(bins))
    counts, bin_edges = np.histogram(values, bins=safe_bins)
    return {
        "bins": safe_bins,
        "counts": counts.astype(np.int64, copy=False),
        "bin_edges": bin_edges.astype(np.float64, copy=False),
        "stats": intensity_statistics(values),
    }


def compare_intensity_populations(data_a, data_b, *, equal_var: bool = True) -> dict:
    values_a = _finite_values(data_a)
    values_b = _finite_values(data_b)
    if values_a.size < 2 or values_b.size < 2:
        raise ValueError("Each population must contain at least 2 finite values for a t-test.")
    test = stats.ttest_ind(values_a, values_b, equal_var=bool(equal_var), nan_policy="omit")
    return {
        "test_name": "Student t-test" if equal_var else "Welch t-test",
        "equal_var": bool(equal_var),
        "statistic": float(test.statistic),
        "pvalue": float(test.pvalue),
        "count_a": int(values_a.size),
        "count_b": int(values_b.size),
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "std_a": float(np.std(values_a, ddof=1)) if values_a.size > 1 else 0.0,
        "std_b": float(np.std(values_b, ddof=1)) if values_b.size > 1 else 0.0,
        "delta_mean": float(np.mean(values_a) - np.mean(values_b)),
    }


def auto_threshold_mask(data, polarity: str = "auto", threshold_value: float | None = None) -> tuple[float, np.ndarray]:
    arr = np.asarray(data)
    if threshold_value is None:
        threshold_value = float(threshold_otsu(arr))
    else:
        threshold_value = float(threshold_value)
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


def convert_to_mask(data) -> np.ndarray:
    return _preserve_dtype(_binary_mask(data), data)


def median_binary_mask(data, radius: int = 1) -> np.ndarray:
    filtered = ndi.median_filter(_binary_mask(data).astype(np.uint8), footprint=_footprint(np.asarray(data).ndim, radius)) > 0
    return _preserve_dtype(filtered, data)


def outline_binary_mask(data) -> np.ndarray:
    mask = _binary_mask(data)
    outline = mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(mask.ndim, 1))
    return _preserve_dtype(outline, data)


def skeletonize_binary_mask(data) -> np.ndarray:
    mask = _binary_mask(data)
    if mask.ndim == 3:
        try:
            from skimage.morphology import skeletonize_3d

            skeleton = skeletonize_3d(mask)
        except Exception:
            skeleton = skeletonize(mask)
    else:
        skeleton = skeletonize(mask)
    return _preserve_dtype(skeleton, data)


def distance_map(data) -> np.ndarray:
    return ndi.distance_transform_edt(_binary_mask(data)).astype(np.float32, copy=False)


def ultimate_points(data) -> np.ndarray:
    mask = _binary_mask(data)
    distance = ndi.distance_transform_edt(mask)
    maxima = local_maxima(distance) & mask
    return maxima.astype(np.int32, copy=False)


def watershed_from_mask(data) -> np.ndarray:
    mask = _binary_mask(data)
    distance = ndi.distance_transform_edt(mask)
    maxima = local_maxima(distance) & mask
    markers, _ = ndi.label(maxima)
    if int(np.max(markers)) == 0:
        markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers=markers, mask=mask)
    return labels.astype(np.int32, copy=False)


def voronoi_from_mask(data) -> np.ndarray:
    mask = _binary_mask(data)
    markers, object_count = ndi.label(mask)
    if object_count == 0:
        return np.zeros_like(np.asarray(data), dtype=np.int32)
    distance = ndi.distance_transform_edt(~mask)
    regions = watershed(distance, markers=markers, mask=np.ones_like(mask, dtype=bool))
    return regions.astype(np.int32, copy=False)


def _normalize_image_to_unit_range(data) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(data)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Input image has no finite values.")
    arr_float = arr.astype(np.float32, copy=False)
    min_value = float(arr_float[finite].min())
    max_value = float(arr_float[finite].max())
    if max_value <= min_value:
        return np.zeros_like(arr_float, dtype=np.float32), min_value, max_value
    normalized = (arr_float - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized, min_value, max_value


def _resolve_kernel_size(kernel_size, ndim: int):
    if isinstance(kernel_size, (list, tuple)):
        values = [max(1, int(v)) for v in kernel_size]
        if len(values) == ndim:
            return tuple(values)
        if len(values) == 1:
            return tuple(values * ndim)
        if len(values) < ndim:
            return tuple((values + [values[-1]] * ndim)[:ndim])
        return tuple(values[:ndim])
    value = max(1, int(kernel_size))
    return tuple([value] * ndim)


def apply_clahe(data, *, kernel_size=32, clip_limit: float = 0.01, nbins: int = 256) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim not in (2, 3):
        raise ValueError(f"CLAHE supports 2D and 3D grayscale images. Got ndim={arr.ndim}.")
    normalized, _min_value, _max_value = _normalize_image_to_unit_range(arr)
    kernel = _resolve_kernel_size(kernel_size, arr.ndim)
    clip = max(1e-6, float(clip_limit))
    bins = max(2, int(nbins))
    result = exposure.equalize_adapthist(normalized, kernel_size=kernel, clip_limit=clip, nbins=bins)
    return result.astype(np.float32, copy=False)
