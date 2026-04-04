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
    "Stats",
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
        "id": "measure_roi_intensity_metrics",
        "title": "ROI Intensity Analysis",
        "category": "Measure",
        "description": "Open a floating ROI measurement widget that tracks one or more editable shapes on the current image layer, shows a live histogram above the table, and updates intensity statistics for the current displayed slice in stacks.",
        "tags": ["measure", "roi", "intensity", "histogram", "widget", "table", "interactive"],
        "best_for": "Interactive ROI intensity measurement with renameable ROI labels, absolute or percent views, copy-to-chat, and CSV export from a floating widget.",
        "suggested_followup": "Ask chat to add more statistics, change the percent normalization logic, or format Insert to Chat as a cleaner summary.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": True,
            "uses_run_in_background": False,
        },
        "ui_mode": "widget",
        "code": """
from napari_chat_assistant.widgets.intensity_metrics_widget import open_intensity_metrics_widget


open_intensity_metrics_widget(viewer)
""".strip(),
    },
    {
        "id": "measure_line_profile_gaussian_fit",
        "title": "Line Profile Gaussian Fit",
        "category": "Measure",
        "description": "Open a floating measurement widget that tracks one or more line ROIs on the current image layer, shows a live profile-plus-fit plot above the table, and updates Gaussian-fit results for the current displayed slice in stacks.",
        "tags": ["measure", "profile", "gaussian-fit", "roi", "interactive", "multi-line", "widget", "table"],
        "best_for": "Interactive line-profile measurement with live Gaussian fitting, absolute or percent readouts, renameable ROI labels, copy-to-chat, and CSV export from a floating widget.",
        "suggested_followup": "Ask chat to add more fit statistics, improve the normalized percent view, or format Insert to Chat as a cleaner narrative summary.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": True,
            "uses_run_in_background": False,
        },
        "ui_mode": "widget",
        "code": """
from napari_chat_assistant.widgets.line_profile_widget import open_line_profile_gaussian_fit_widget


open_line_profile_gaussian_fit_widget(viewer)
""".strip(),
    },
    {
        "id": "stats_compare_two_roi_groups",
        "title": "Compare Two ROI Groups",
        "category": "Stats",
        "description": "Run the built-in two-group ROI comparison workflow for prefixes such as wt vs mutant, using ROI summaries per image.",
        "tags": ["stats", "compare", "roi", "t-test", "welch", "mann-whitney"],
        "best_for": "Comparing two groups of ROI-based measurements with one ROI summary per image.",
        "suggested_followup": "Load this into Prompt and replace the group prefixes with your own layer-name prefixes or ask chat to open the interactive comparison widget.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": False,
        },
        "code": """
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job


prepared = prepare_tool_job(
    viewer,
    "compare_roi_groups",
    {
        "group_a_prefix": "wt",
        "group_b_prefix": "mutant",
        "metric": "mean",
        "roi_kind": "auto",
        "pair_mode": "paired_suffix",
        "alpha": 0.05,
    },
)

if prepared["mode"] == "immediate" and "job" not in prepared:
    print(prepared["message"])
else:
    result = run_tool_job(prepared["job"])
    print(apply_tool_job_result(viewer, result))
""".strip(),
    },
    {
        "id": "stats_compare_two_image_groups",
        "title": "Compare Two Image Groups",
        "category": "Stats",
        "description": "Run the built-in two-group whole-image comparison workflow when you want one summary value per image without ROI.",
        "tags": ["stats", "compare", "image", "t-test", "welch", "group"],
        "best_for": "Comparing two groups of whole-image summaries such as mean or median intensity per image.",
        "suggested_followup": "Load this into Prompt and replace the group prefixes, or open the interactive group comparison widget for a plot-based workflow.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": False,
        },
        "code": """
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job


prepared = prepare_tool_job(
    viewer,
    "compare_image_groups",
    {
        "group_a_prefix": "wt",
        "group_b_prefix": "mutant",
        "metric": "mean",
        "pair_mode": "paired_suffix",
        "alpha": 0.05,
    },
)

if prepared["mode"] == "immediate" and "job" not in prepared:
    print(prepared["message"])
else:
    result = run_tool_job(prepared["job"])
    print(apply_tool_job_result(viewer, result))
""".strip(),
    },
    {
        "id": "stats_open_group_comparison_widget",
        "title": "Open Group Comparison Widget",
        "category": "Stats",
        "description": "Open the interactive group-comparison widget for ROI or whole-image two-group statistics.",
        "tags": ["stats", "compare", "widget", "interactive", "roi", "image"],
        "best_for": "Exploring group comparisons with an interactive widget instead of running a fixed code snippet.",
        "suggested_followup": "Use this when you want plots, descriptive stats, or a guided UI instead of a one-shot console result.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": False,
        },
        "ui_mode": "widget",
        "code": """
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job


prepared = prepare_tool_job(viewer, "open_group_comparison_widget", {})
if prepared["mode"] == "immediate" and "job" not in prepared:
    print(prepared["message"])
else:
    result = run_tool_job(prepared["job"])
    print(apply_tool_job_result(viewer, result))
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
        "id": "visualize_quick_compare_grid",
        "title": "Quick Compare Grid",
        "category": "Visualize",
        "description": "Turn on napari grid view for side-by-side image comparison without moving any layer data.",
        "tags": ["visualize", "grid", "compare", "viewer"],
        "best_for": "Fast visual comparison of multiple image layers while keeping the original layers untouched.",
        "suggested_followup": "Use Turn Off Compare Grid to return to overlap view, or switch to Create Presentation Layout for physically arranged copies.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": False,
        },
        "code": """
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job


prepared = prepare_tool_job(
    viewer,
    "show_image_layers_in_grid",
    {
        "layer_names": ["img_a", "img_b", "img_c"],
        "spacing": 2,
    },
)

if prepared["mode"] == "immediate" and "job" not in prepared:
    print(prepared["message"])
else:
    result = run_tool_job(prepared["job"])
    print(apply_tool_job_result(viewer, result))
""".strip(),
    },
    {
        "id": "visualize_presentation_layout",
        "title": "Create Presentation Layout",
        "category": "Visualize",
        "description": "Arrange image or labels layers as presentation copies in one viewer space, distinct from viewer grid mode.",
        "tags": ["visualize", "presentation", "layout", "viewer"],
        "best_for": "Curated side-by-side layouts for display, figures, or layer-by-layer presentation in one viewer.",
        "suggested_followup": "Use create_analysis_montage instead if you want one real composite canvas for shared ROI or mask analysis.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": False,
        },
        "code": """
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job


prepared = prepare_tool_job(
    viewer,
    "arrange_layers_for_presentation",
    {
        "layer_names": ["img_a", "img_b", "img_c"],
        "layout": "row",
        "spacing": 20,
        "use_copies": True,
        "match_origin": True,
    },
)

if prepared["mode"] == "immediate" and "job" not in prepared:
    print(prepared["message"])
else:
    result = run_tool_job(prepared["job"])
    print(apply_tool_job_result(viewer, result))
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
        "id": "workbench_create_analysis_montage",
        "title": "Create Analysis Montage",
        "category": "Workbench",
        "description": "Build one composite montage canvas from multiple 2D grayscale image layers, with optional tile boxes and a blank mask layer for shared ROI or mask work.",
        "tags": ["workbench", "montage", "analysis", "labels", "viewer"],
        "best_for": "Creating one shared annotation canvas from multiple source images without using chat.",
        "suggested_followup": "After editing the montage mask or montage points, run one of the split-back templates to export per-source layers.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": False,
        },
        "code": """
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job


def run_tool(tool_name, arguments):
    prepared = prepare_tool_job(viewer, tool_name, arguments)
    if prepared["mode"] == "immediate" and "job" not in prepared:
        print(prepared["message"])
        return prepared
    result = run_tool_job(prepared["job"])
    message = apply_tool_job_result(viewer, result)
    print(message)
    return result


run_tool(
    "create_analysis_montage",
    {
        "layer_names": ["img_a", "img_b", "img_c"],
        "rows": 1,
        "columns": 3,
        "spacing": 2,
        "show_tile_boxes": True,
        "create_mask_layer": True,
    },
)

print("Edit analysis_montage_mask or add montage points, then run a split-back template.")
""".strip(),
    },
    {
        "id": "workbench_split_montage_labels_to_sources",
        "title": "Split Montage Labels To Sources",
        "category": "Workbench",
        "description": "Split a montage-space Labels layer back into one labels layer per source image using stored montage tile metadata.",
        "tags": ["workbench", "montage", "labels", "split", "analysis"],
        "best_for": "Exporting per-image masks after editing a single montage mask layer.",
        "suggested_followup": "Use the created per-source labels for measurement, cleanup, export, or downstream spectral ROI extraction.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": False,
        },
        "code": """
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job


prepared = prepare_tool_job(
    viewer,
    "split_montage_annotations_to_sources",
    {
        "annotation_layer": "analysis_montage_mask",
        "montage_layer": "analysis_montage",
    },
)

if prepared["mode"] == "immediate" and "job" not in prepared:
    print(prepared["message"])
else:
    result = run_tool_job(prepared["job"])
    print(apply_tool_job_result(viewer, result))
""".strip(),
    },
    {
        "id": "workbench_split_montage_points_to_sources",
        "title": "Split Montage Points To Sources",
        "category": "Workbench",
        "description": "Split a montage-space Points layer back into one points layer per source image, converting coordinates from montage space to source-local space.",
        "tags": ["workbench", "montage", "points", "split", "analysis"],
        "best_for": "Recovering per-image annotations after adding points on a shared montage canvas.",
        "suggested_followup": "Reuse the output points layers for per-image prompts, tracking, or measurement workflows.",
        "runtime": {
            "plugin_runtime_required": True,
            "uses_viewer": True,
            "uses_selected_layer": False,
            "uses_run_in_background": False,
        },
        "code": """
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job


prepared = prepare_tool_job(
    viewer,
    "split_montage_annotations_to_sources",
    {
        "annotation_layer": "montage_points",
        "montage_layer": "analysis_montage",
    },
)

if prepared["mode"] == "immediate" and "job" not in prepared:
    print(prepared["message"])
else:
    result = run_tool_job(prepared["job"])
    print(apply_tool_job_result(viewer, result))
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
    "Synthetic 2D SNR Sweep Gray": {
        "id": "data_synthetic_2d_snr_sweep_gray",
        "description": "Generate a 2D grayscale synthetic SNR sweep with low, mid, and high noise variants plus a mask.",
        "best_for": "Testing denoise, threshold, segmentation, and measurement workflows on grayscale 2D data.",
        "suggested_followup": "Ask chat to adapt this pack to different sizes, SNR spacing, or object density.",
    },
    "Synthetic 3D SNR Sweep Gray": {
        "id": "data_synthetic_3d_snr_sweep_gray",
        "description": "Generate a 3D grayscale synthetic SNR sweep with low, mid, and high noise volumes plus a mask.",
        "best_for": "Testing volume processing, 3D segmentation, and SAM-style workflows on grayscale data.",
        "suggested_followup": "Ask chat to reduce the volume size, alter object density, or create a projection workflow.",
    },
    "Synthetic 2D SNR Sweep RGB": {
        "id": "data_synthetic_2d_snr_sweep_rgb",
        "description": "Generate a 2D RGB synthetic SNR sweep with multiple noise conditions and labels.",
        "best_for": "Testing multi-channel plotting, segmentation, and label measurement on 2D RGB data.",
        "suggested_followup": "Ask chat to isolate channels, add measurements, or convert this into a comparison workflow.",
    },
    "Synthetic 3D SNR Sweep RGB": {
        "id": "data_synthetic_3d_snr_sweep_rgb",
        "description": "Generate a 3D RGB synthetic SNR sweep with multiple noise conditions and labels.",
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
