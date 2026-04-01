from __future__ import annotations

from typing import Any

from napari_chat_assistant.agent.prompt_library import DEFAULT_CODE_SNIPPETS


TEMPLATE_LIBRARY_VERSION = 1

TEMPLATE_CATEGORIES = [
    "Data",
    "Inspect",
    "Process",
    "Segment",
    "Measure",
    "Visualize",
    "Compare",
    "Workbench",
    "Background Jobs",
]


TEMPLATE_RECORDS: list[dict[str, Any]] = [
    {
        "id": "data_synthetic_blob_image",
        "title": "Synthetic Blob Image",
        "category": "Data",
        "description": "Create a simple 2D blob image and labels layer for quick testing inside napari.",
        "tags": ["data", "synthetic", "image", "labels", "background"],
        "best_for": "Quick testing of processing, segmentation, and measurement templates.",
        "suggested_followup": "Ask chat to add noise, make it 3D, or create more objects.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": True,
        },
        "code": """
import numpy as np


def compute():
    yy, xx = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    image = np.zeros((256, 256), dtype=np.float32)
    labels = np.zeros((256, 256), dtype=np.int32)
    centers = [(70, 80), (120, 150), (190, 95), (180, 195)]
    radii = [24, 18, 28, 20]
    for index, ((cy, cx), radius) in enumerate(zip(centers, radii), start=1):
        dist = ((yy - cy) ** 2 + (xx - cx) ** 2) ** 0.5
        blob = np.exp(-0.5 * (dist / radius) ** 2).astype(np.float32)
        image += blob
        labels[dist <= radius * 0.8] = index
    image -= image.min()
    image /= image.max() + 1e-8
    return image.astype(np.float32), labels


def apply_result(payload):
    image, labels = payload
    viewer.add_image(image, name="template_blob_image", colormap="gray")
    viewer.add_labels(labels, name="template_blob_labels")
    print("Added template_blob_image and template_blob_labels")


run_in_background(compute, apply_result, label="Generate synthetic blob image")
""".strip(),
    },
    {
        "id": "inspect_selected_layer_summary",
        "title": "Inspect Selected Layer Summary",
        "category": "Inspect",
        "description": "Print a compact summary of the selected layer for quick orientation before analysis.",
        "tags": ["inspect", "selected-layer", "viewer"],
        "best_for": "Understanding the current working layer before processing or plotting.",
        "suggested_followup": "Ask chat to expand this summary into a plotting or measurement workflow.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": True,
            "uses_run_in_background": False,
        },
        "code": """
import numpy as np

layer = selected_layer
if layer is None:
    raise ValueError("Select a layer first.")

data = np.asarray(layer.data)
print(f"name={layer.name}")
print(f"type={layer.__class__.__name__}")
print(f"shape={data.shape}")
print(f"dtype={data.dtype}")
if data.size and np.issubdtype(data.dtype, np.number):
    finite = np.isfinite(data)
    if np.any(finite):
        values = data[finite]
        print(f"min={float(values.min()):.4f}")
        print(f"max={float(values.max()):.4f}")
        print(f"mean={float(values.mean()):.4f}")
        print(f"std={float(values.std()):.4f}")
""".strip(),
    },
    {
        "id": "process_gaussian_selected_layer",
        "title": "Gaussian Process Selected Layer",
        "category": "Process",
        "description": "Apply a simple Gaussian denoise pass to the selected image layer and add the result back to napari.",
        "tags": ["process", "gaussian", "selected-layer", "background"],
        "best_for": "Fast starting point for noise reduction templates.",
        "suggested_followup": "Ask chat to adapt sigma, preserve dtype, or branch for 3D.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": True,
            "uses_run_in_background": True,
        },
        "code": """
import numpy as np
from scipy.ndimage import gaussian_filter

layer = selected_layer
if layer is None:
    raise ValueError("Select an image layer first.")

data = np.asarray(layer.data, dtype=np.float32)


def compute():
    return gaussian_filter(data, sigma=1.2).astype(np.float32)


def apply_result(result):
    viewer.add_image(result, name=f"{layer.name}_gaussian_template", colormap="gray")
    print(f"Added {layer.name}_gaussian_template")


run_in_background(compute, apply_result, label="Gaussian process selected layer")
""".strip(),
    },
    {
        "id": "segment_threshold_selected_layer",
        "title": "Threshold Segment Selected Layer",
        "category": "Segment",
        "description": "Create a simple binary labels result from the selected grayscale image layer.",
        "tags": ["segment", "threshold", "selected-layer"],
        "best_for": "A minimal segmentation starting point that chat can refine.",
        "suggested_followup": "Ask chat to add Otsu thresholding, morphology cleanup, or multi-object labeling.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": True,
            "uses_run_in_background": False,
        },
        "code": """
import numpy as np

layer = selected_layer
if layer is None:
    raise ValueError("Select an image layer first.")

data = np.asarray(layer.data, dtype=np.float32)
threshold = float(data.mean() + 0.5 * data.std())
labels = (data > threshold).astype(np.int32)
viewer.add_labels(labels, name=f"{layer.name}_threshold_template")
print(f"Applied threshold={threshold:.4f}")
""".strip(),
    },
    {
        "id": "measure_label_sizes",
        "title": "Measure Label Sizes",
        "category": "Measure",
        "description": "Measure simple per-label pixel counts from the selected labels layer.",
        "tags": ["measure", "labels", "selected-layer", "stats"],
        "best_for": "Basic object-size measurement before more advanced regionprops workflows.",
        "suggested_followup": "Ask chat to add centroids, mean intensity, or export to a table layer.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": True,
            "uses_run_in_background": False,
        },
        "code": """
import numpy as np

layer = selected_layer
if layer is None:
    raise ValueError("Select a labels layer first.")

labels = np.asarray(layer.data)
values, counts = np.unique(labels[labels > 0], return_counts=True)
if values.size == 0:
    print("No labeled objects found.")
else:
    for value, count in zip(values.tolist(), counts.tolist()):
        print(f"label={value} pixels={count}")
""".strip(),
    },
    {
        "id": "measure_line_profile_gaussian_fit",
        "title": "Line Profile Gaussian Fit",
        "category": "Measure",
        "description": "Generate a synthetic blob image, place a line ROI, sample the intensity profile, and fit a Gaussian to estimate sigma and FWHM.",
        "tags": ["measure", "profile", "gaussian-fit", "roi", "synthetic", "plot"],
        "best_for": "Teaching intensity-profile measurement, Gaussian fitting, and line-based ROI analysis.",
        "suggested_followup": "Ask chat to adapt this from synthetic data to the selected layer or to use an existing line ROI.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": False,
        },
        "code": """
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit


def make_synthetic_image(
    shape=(128, 128),
    center=(60, 70),
    sigma=8.0,
    amplitude=180.0,
    background=20.0,
    noise_sd=6.0,
    seed=42,
):
    rng = np.random.default_rng(seed)
    y, x = np.indices(shape)
    cy, cx = center
    blob = amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    noise = rng.normal(0.0, noise_sd, shape)
    image = background + blob + noise
    return image.astype(np.float32)


def line_profile(image, x0, y0, x1, y1):
    length = np.hypot(x1 - x0, y1 - y0)
    n_samples = int(length) + 1
    x = np.linspace(x0, x1, n_samples)
    y = np.linspace(y0, y1, n_samples)
    profile = map_coordinates(image, [y, x], order=1)
    distance = np.linspace(0.0, length, n_samples)
    return distance, profile


def gaussian(x, baseline, amplitude, mean, sigma):
    return baseline + amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


image = make_synthetic_image()
viewer.add_image(image, name="template_profile_blob", colormap="gray")

x0, y0 = 30, 90
x1, y1 = 95, 35
viewer.add_shapes(
    [[[y0, x0], [y1, x1]]],
    shape_type="line",
    edge_color="red",
    edge_width=3,
    name="template_profile_line",
)

dist, prof = line_profile(image, x0, y0, x1, y1)
p0 = [
    float(np.min(prof)),
    float(np.max(prof) - np.min(prof)),
    float(dist[np.argmax(prof)]),
    5.0,
]
params, _ = curve_fit(gaussian, dist, prof, p0=p0)
baseline, amplitude, mean, sigma = params
fwhm = 2.3548 * sigma

print("Gaussian fit results")
print(f"Baseline: {baseline:.4f}")
print(f"Amplitude: {amplitude:.4f}")
print(f"Mean: {mean:.4f}")
print(f"Sigma: {sigma:.4f}")
print(f"FWHM: {fwhm:.4f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(dist, prof, "o", label="Measured")
ax.plot(dist, gaussian(dist, *params), "-", label="Gaussian fit")
ax.legend()
ax.set_xlabel("Distance (pixels)")
ax.set_ylabel("Intensity")
ax.set_title("Line profile Gaussian fit")
fig.tight_layout()
plt.show()
""".strip(),
    },
    {
        "id": "visualize_histogram_selected_layer",
        "title": "Histogram of Selected Layer",
        "category": "Visualize",
        "description": "Plot a histogram from the selected grayscale image layer.",
        "tags": ["visualize", "plot", "histogram", "selected-layer"],
        "best_for": "Intensity distribution inspection and quick plotting examples.",
        "suggested_followup": "Ask chat to add mean/std lines, log scaling, or compare multiple layers.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": True,
            "uses_run_in_background": False,
        },
        "code": """
import numpy as np
import matplotlib.pyplot as plt

layer = selected_layer
if layer is None:
    raise ValueError("Select an image layer first.")

data = np.asarray(layer.data, dtype=np.float32)
values = data[np.isfinite(data)].ravel()
if values.size == 0:
    raise ValueError("Selected layer contains no finite values.")

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(values, bins=64, color="#4c78a8", alpha=0.85)
ax.set_title(f"Histogram: {layer.name}")
ax.set_xlabel("Intensity")
ax.set_ylabel("Count")
fig.tight_layout()
plt.show()
""".strip(),
    },
    {
        "id": "compare_two_image_layers",
        "title": "Compare Two Image Layers",
        "category": "Compare",
        "description": "Compare mean and standard deviation between the first two image layers in the viewer.",
        "tags": ["compare", "image", "stats", "viewer"],
        "best_for": "Quick before/after or condition-to-condition comparisons.",
        "suggested_followup": "Ask chat to turn this into a bar chart or paired comparison plot.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": False,
        },
        "code": """
import numpy as np

image_layers = [layer for layer in viewer.layers if layer.__class__.__name__ == "Image"]
if len(image_layers) < 2:
    raise ValueError("Load at least two image layers first.")

for layer in image_layers[:2]:
    data = np.asarray(layer.data, dtype=np.float32)
    values = data[np.isfinite(data)]
    print(
        f"{layer.name}: mean={float(values.mean()):.4f} std={float(values.std()):.4f} "
        f"min={float(values.min()):.4f} max={float(values.max()):.4f}"
    )
""".strip(),
    },
    {
        "id": "workbench_selected_layer_scratchpad",
        "title": "Selected Layer Scratchpad",
        "category": "Workbench",
        "description": "A lightweight scratch template for experimenting against the selected layer inside the plugin runtime.",
        "tags": ["workbench", "selected-layer", "viewer", "starter"],
        "best_for": "Loading into the prompt and asking chat to adapt it to a current task.",
        "suggested_followup": "Ask chat to replace the placeholder analysis with the operation you want to test right now.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": True,
            "uses_run_in_background": False,
        },
        "code": """
import numpy as np

layer = selected_layer
if layer is None:
    raise ValueError("Select a layer first.")

data = np.asarray(layer.data)
print(f"Working on layer: {layer.name}")
print(f"Shape: {data.shape}")
print(f"Dtype: {data.dtype}")

# Replace this section with your current experiment.
result = np.asarray(data)
print("Template scratchpad ready. Ask chat to adapt this code for your current workflow.")
""".strip(),
    },
    {
        "id": "background_job_selected_layer_stats",
        "title": "Background Stats Job",
        "category": "Background Jobs",
        "description": "Compute summary statistics for the selected layer in the background and print the result on completion.",
        "tags": ["background", "selected-layer", "stats", "run_in_background"],
        "best_for": "Showing the intended plugin-native background compute/apply pattern.",
        "suggested_followup": "Ask chat to turn this into a plot, table, or per-object background workflow.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": True,
            "uses_run_in_background": True,
        },
        "code": """
import numpy as np

layer = selected_layer
if layer is None:
    raise ValueError("Select a layer first.")

data = np.asarray(layer.data, dtype=np.float32)


def compute():
    values = data[np.isfinite(data)]
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "count": int(values.size),
    }


def apply_result(stats):
    print(f"Background stats for {layer.name}")
    for key, value in stats.items():
        print(f"{key}={value}")


run_in_background(compute, apply_result, label="Compute background stats")
""".strip(),
    },
]


DEMO_PACK_TEMPLATE_METADATA: dict[str, dict[str, Any]] = {
    "Demo Pack: EM 2D SNR Sweep": {
        "id": "data_demo_em_2d_snr_sweep",
        "description": "Generate a 2D electron-microscopy-style grayscale demo pack with low, mid, and high SNR variants plus a mask.",
        "best_for": "Testing denoise, threshold, segmentation, and measurement workflows on grayscale 2D data.",
        "suggested_followup": "Ask chat to adapt this pack to different sizes, SNR spacing, or object density.",
    },
    "Demo Pack: EM 3D SNR Sweep": {
        "id": "data_demo_em_3d_snr_sweep",
        "description": "Generate a 3D electron-microscopy-style grayscale demo pack with low, mid, and high SNR volumes plus a mask.",
        "best_for": "Testing volume processing, 3D segmentation, and SAM-style workflows on grayscale data.",
        "suggested_followup": "Ask chat to reduce the volume size, alter object density, or create a projection workflow.",
    },
    "Demo Pack: RGB Cells 2D SNR Sweep": {
        "id": "data_demo_rgb_cells_2d_snr_sweep",
        "description": "Generate a 2D RGB fluorescent-style cell demo pack with multiple SNR conditions and labels.",
        "best_for": "Testing multi-channel plotting, segmentation, and label measurement on 2D RGB data.",
        "suggested_followup": "Ask chat to isolate channels, add measurements, or convert this into a comparison workflow.",
    },
    "Demo Pack: RGB Cells 3D SNR Sweep": {
        "id": "data_demo_rgb_cells_3d_snr_sweep",
        "description": "Generate a 3D RGB fluorescent-style cell demo pack with multiple SNR conditions and labels.",
        "best_for": "Testing 3D multi-channel workflows, layer comparison, and label analysis.",
        "suggested_followup": "Ask chat to extract one channel, measure labels, or add a projection template on top.",
    },
    "Demo Pack: Messy Masks 2D/3D": {
        "id": "data_demo_messy_masks_2d_3d",
        "description": "Generate clean, messy, and filled-target labels for both 2D and 3D mask-cleanup workflows.",
        "best_for": "Testing morphology, cleanup, hole filling, and label post-processing templates.",
        "suggested_followup": "Ask chat to measure cleanup differences or build a mask repair workflow around these layers.",
    },
}


def _demo_pack_template_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in DEFAULT_CODE_SNIPPETS:
        title = str(item.get("title", "")).strip()
        metadata = DEMO_PACK_TEMPLATE_METADATA.get(title)
        code = str(item.get("code", ""))
        if not metadata or not code.strip():
            continue
        tags = [str(tag).strip() for tag in item.get("tags", []) if str(tag).strip()]
        if "data" not in {tag.lower() for tag in tags}:
            tags.insert(0, "data")
        records.append(
            {
                "id": metadata["id"],
                "title": title,
                "category": "Data",
                "description": metadata["description"],
                "tags": tags,
                "best_for": metadata["best_for"],
                "suggested_followup": metadata["suggested_followup"],
                "runtime": {
                    "plugin_runtime_required": True,
                    "uses_viewer": True,
                    "uses_selected_layer": False,
                    "uses_run_in_background": True,
                },
                "code": code.strip(),
            }
        )
    return records


ALL_TEMPLATE_RECORDS: list[dict[str, Any]] = TEMPLATE_RECORDS + _demo_pack_template_records()


def template_library_payload() -> dict[str, Any]:
    return {
        "version": TEMPLATE_LIBRARY_VERSION,
        "categories": list(TEMPLATE_CATEGORIES),
        "templates": [dict(record) for record in ALL_TEMPLATE_RECORDS],
    }


def template_records() -> list[dict[str, Any]]:
    return [dict(record) for record in ALL_TEMPLATE_RECORDS]
