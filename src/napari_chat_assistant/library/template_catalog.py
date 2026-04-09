from __future__ import annotations

from typing import Any

from napari_chat_assistant.agent.prompt_library import DEFAULT_CODE_SNIPPETS


TEMPLATE_LIBRARY_VERSION = 2

SECTION_LABELS = {
    "plugin_prompt": "Prompt Templates",
    "plugin_code": "Code Templates",
    "learning": "Learning",
}

SECTION_COLORS = {
    "plugin_prompt": {"section": "#ffd166", "category": "#ffe29a", "item": "#fff4cc"},
    "plugin_code": {"section": "#7bdff2", "category": "#a8ecf7", "item": "#d8f8ff"},
    "learning": {"section": "#95d67b", "category": "#bce8ab", "item": "#e5f8dc"},
}

SECTION_ORDER = ["plugin_prompt", "plugin_code", "learning"]

CATEGORY_ORDER = {
    "plugin_prompt": [
        "Getting Started",
        "Inspect & Plan",
        "Process & Segment",
        "Measure & Compare",
        "Interpret & Report",
    ],
    "plugin_code": [
        "Data Setup",
        "Inspect & Summarize",
        "Enhance & Transform",
        "Segment & Masks",
        "Measure & Quantify",
        "Compare & Statistics",
        "Visualize & Present",
        "Workflow Utilities",
        "Developer Patterns",
    ],
    "learning": [
        "Microscopy",
        "Electron Microscopy",
        "Biophotonics",
        "Image Formation",
        "Quantitative Imaging",
        "Statistics",
        "Academic Prompting",
        "Language Support",
    ],
}

PROMPT_TEMPLATE_RECORDS: list[dict[str, Any]] = [
    {
        "id": "prompt_getting_started_selected_layer",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Inspect Selected Layer And Plan",
        "category": "Getting Started",
        "description": "Inspect the selected layer first, then propose a practical next-step workflow inside the plugin.",
        "best_for": "Starting from an unfamiliar layer without jumping straight into the wrong tool or code.",
        "suggested_followup": "Ask the assistant to execute the recommended first step after it explains the reasoning.",
        "tags": ["prompt", "getting-started", "selected-layer", "workflow"],
        "prompt": "Inspect the selected layer first, summarize what kind of data it likely is, then recommend the best next 3 analysis steps inside this plugin.",
    },
    {
        "id": "prompt_inspect_all_layers_plan",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Inspect All Layers And Plan",
        "category": "Inspect & Plan",
        "description": "Review the full viewer state and build a short analysis plan before doing anything.",
        "best_for": "Orienting a multi-layer viewer where image, labels, ROI, and analysis layers may already be mixed together.",
        "suggested_followup": "Ask the assistant to apply only step 1 after confirming the plan.",
        "tags": ["prompt", "inspect", "viewer", "workflow"],
        "prompt": "Look at all open layers, group them by likely role, point out any layer-selection ambiguity, and suggest a short analysis plan before doing anything.",
    },
    {
        "id": "prompt_threshold_preview_first",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Threshold Preview First",
        "category": "Process & Segment",
        "description": "Preview thresholding before applying a labels result, with reasoning about whether it is appropriate.",
        "best_for": "Users who want segmentation help without committing to a mask too early.",
        "suggested_followup": "Ask the assistant to apply thresholding only if the preview looks appropriate.",
        "tags": ["prompt", "threshold", "preview", "segmentation"],
        "prompt": "Preview threshold first for the selected image, explain whether the preview looks biologically reasonable, and tell me whether I should apply it.",
    },
    {
        "id": "prompt_gaussian_selected_image",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Apply Gaussian Blur",
        "category": "Process & Segment",
        "description": "Apply Gaussian blur to a named or selected image layer with an explicit sigma.",
        "best_for": "Fast denoising or smoothing requests before thresholding or visual inspection.",
        "suggested_followup": "Ask for threshold preview or measurement after smoothing.",
        "tags": ["prompt", "gaussian", "blur", "selected-image"],
        "prompt": "Apply gaussian blur to image_a with sigma 1.2.",
    },
    {
        "id": "prompt_fill_mask_holes",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Fill Mask Holes",
        "category": "Process & Segment",
        "description": "Fill holes in a labels or mask layer.",
        "best_for": "Cleaning up thresholded masks before measurement or later morphology steps.",
        "suggested_followup": "Ask for remove-small-particles or keep-largest after filling.",
        "tags": ["prompt", "mask", "fill-holes", "cleanup"],
        "prompt": "Fill holes in labels_a.",
    },
    {
        "id": "prompt_remove_small_particles",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Remove Small Particles",
        "category": "Process & Segment",
        "description": "Remove small connected components from a mask or labels layer using a size threshold.",
        "best_for": "Cleaning speckle-like segmentation noise before downstream measurement.",
        "suggested_followup": "Ask to keep only the largest connected component if one dominant object is expected.",
        "tags": ["prompt", "mask", "cleanup", "small-objects"],
        "prompt": "Remove small particles from labels_a with min_size 64.",
    },
    {
        "id": "prompt_keep_largest_component",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Keep Largest Component",
        "category": "Process & Segment",
        "description": "Reduce a rough mask to its largest connected object.",
        "best_for": "Workflows where only one dominant foreground object should remain.",
        "suggested_followup": "Ask for a quick mask measurement after cleanup.",
        "tags": ["prompt", "mask", "largest-component", "cleanup"],
        "prompt": "Keep only the largest connected component in labels_a.",
    },
    {
        "id": "prompt_max_intensity_projection",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Max Intensity Projection",
        "category": "Process & Segment",
        "description": "Create a max-intensity projection from a 3D image layer.",
        "best_for": "Fast 2D overview of a 3D grayscale volume.",
        "suggested_followup": "Ask for contrast enhancement or threshold preview on the projected result.",
        "tags": ["prompt", "projection", "mip", "3d-image"],
        "prompt": "Create a max intensity projection from volume_a along axis 0.",
    },
    {
        "id": "prompt_prefer_builtin_tools",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Prefer Built-In Tools",
        "category": "Process & Segment",
        "description": "Push the assistant toward deterministic built-ins first and code only when necessary.",
        "best_for": "Stable workflows where the user wants less fragile custom code.",
        "suggested_followup": "Ask why a tool was chosen instead of generated code.",
        "tags": ["prompt", "tools", "code", "workflow"],
        "prompt": "For the current viewer, prefer built-in tools. If none fit, explain why and then generate safe napari code with the tradeoffs.",
    },
    {
        "id": "prompt_measure_current_mask",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Measure Current Mask",
        "category": "Measure & Compare",
        "description": "Measure the active mask or labels layer and summarize the result in plain language.",
        "best_for": "Fast area and object-count checks without writing code.",
        "suggested_followup": "Ask for a full table if the quick summary looks useful.",
        "tags": ["prompt", "measure", "mask", "labels"],
        "prompt": "Measure the current mask, summarize the result in plain language, and tell me if I should also generate a per-object table.",
    },
    {
        "id": "prompt_analyze_particles_table",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Analyze Particles Table",
        "category": "Measure & Compare",
        "description": "Measure the selected labels layer as a per-object table.",
        "best_for": "Users who want per-object area, centroid, and intensity-style measurements rather than a quick summary.",
        "suggested_followup": "Ask the assistant to explain which columns matter most for your current analysis goal.",
        "tags": ["prompt", "measure", "labels", "table", "particles"],
        "prompt": "Analyze particles table for labels_a.",
    },
    {
        "id": "prompt_inspect_current_roi",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Inspect Current ROI",
        "category": "Measure & Compare",
        "description": "Inspect the currently active ROI context before extracting values or measuring it.",
        "best_for": "Checking which ROI layer is active and what spatial scope it currently covers.",
        "suggested_followup": "Ask for ROI values or a Shapes ROI measurement after inspection.",
        "tags": ["prompt", "roi", "inspect", "selected-roi"],
        "prompt": "Inspect the current ROI.",
    },
    {
        "id": "prompt_extract_roi_values",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Extract ROI Values",
        "category": "Measure & Compare",
        "description": "Extract grayscale image values from a named ROI layer.",
        "best_for": "Sampling intensity values inside Shapes or Labels regions of interest.",
        "suggested_followup": "Ask for a summarized interpretation of the extracted ROI values.",
        "tags": ["prompt", "roi", "values", "image"],
        "prompt": "Extract ROI values from image_a using roi_shapes.",
    },
    {
        "id": "prompt_crop_to_labels_bbox",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Crop To Bounding Box",
        "category": "Measure & Compare",
        "description": "Crop one layer to the bounding box of another layer with explicit padding.",
        "best_for": "Creating focused subimages around a labels or mask-defined foreground region.",
        "suggested_followup": "Ask for measurement or thresholding on the cropped result.",
        "tags": ["prompt", "crop", "bbox", "labels"],
        "prompt": "Crop image_a to the bounding box of labels_a with padding 8.",
    },
    {
        "id": "prompt_compare_two_groups",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Compare Two Groups",
        "category": "Measure & Compare",
        "description": "Ask the assistant to choose the best comparison path for two image or ROI groups already open in the viewer.",
        "best_for": "Users who know they need a group comparison but are unsure whether ROI or whole-image stats make more sense.",
        "suggested_followup": "Ask the assistant to open the dedicated comparison workflow if interactive plots would help.",
        "tags": ["prompt", "compare", "stats", "roi", "image"],
        "prompt": "Inspect the current layers, determine whether ROI-based or whole-image comparison makes more sense for two groups, and then run the best comparison path.",
    },
    {
        "id": "prompt_explain_then_code",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Explain Before Code",
        "category": "Interpret & Report",
        "description": "Force an explanation-first workflow before any generated code is produced.",
        "best_for": "Users who want to understand the reasoning before running custom code.",
        "suggested_followup": "After the explanation, ask the assistant to simplify or optimize the code it proposes.",
        "tags": ["prompt", "explain", "code", "reasoning"],
        "prompt": "Explain the likely workflow for the current viewer first. After that, only generate runnable napari code if no built-in tool is a good fit.",
    },
    {
        "id": "prompt_markdown_answer",
        "branch": "plugin_prompt",
        "template_type": "prompt",
        "title": "Reply In Markdown",
        "category": "Interpret & Report",
        "description": "Request a structured markdown answer with compact sections and bullets.",
        "best_for": "Readable explanations, study notes, or workflow summaries that should stay easy to scan.",
        "suggested_followup": "Ask the assistant to include code only after the explanation if needed.",
        "tags": ["prompt", "markdown", "formatting", "explanation"],
        "prompt": "Reply in markdown. Use bullets and short sections. Explain first, then give runnable plugin code if needed.",
    },
]

CODE_TEMPLATE_RECORDS: list[dict[str, Any]] = [
    {
        "id": "data_synthetic_blob_image",
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Synthetic Blob Image",
        "category": "Data Setup",
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Inspect Selected Layer Summary",
        "category": "Inspect & Summarize",
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Gaussian Process Selected Layer",
        "category": "Enhance & Transform",
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Threshold Segment Selected Layer",
        "category": "Segment & Masks",
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Measure Label Sizes",
        "category": "Measure & Quantify",
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
        "id": "stats_compare_two_roi_groups",
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Compare Two ROI Groups",
        "category": "Compare & Statistics",
        "description": "Run the built-in two-group ROI comparison workflow for prefixes such as wt vs mutant, using ROI summaries per image.",
        "tags": ["stats", "compare", "roi", "t-test", "welch", "mann-whitney"],
        "best_for": "Comparing two groups of ROI-based measurements with one ROI summary per image.",
        "suggested_followup": "Load this into Prompt and replace the group prefixes with your own layer-name prefixes.",
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Compare Two Image Groups",
        "category": "Compare & Statistics",
        "description": "Run the built-in two-group whole-image comparison workflow when you want one summary value per image without ROI.",
        "tags": ["stats", "compare", "image", "t-test", "welch", "group"],
        "best_for": "Comparing two groups of whole-image summaries such as mean or median intensity per image.",
        "suggested_followup": "Load this into Prompt and replace the group prefixes, or ask chat to explain the statistical assumptions.",
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
        "id": "visualize_histogram_selected_layer",
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Histogram of Selected Layer",
        "category": "Visualize & Present",
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Quick Compare Grid",
        "category": "Visualize & Present",
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Create Presentation Layout",
        "category": "Visualize & Present",
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Compare Two Image Layers",
        "category": "Compare & Statistics",
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Create Analysis Montage",
        "category": "Workflow Utilities",
        "description": "Build one composite montage canvas from multiple 2D grayscale image layers, with optional tile boxes and a blank mask layer for shared ROI or mask work.",
        "tags": ["workflow", "montage", "analysis", "labels", "viewer"],
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Split Montage Labels To Sources",
        "category": "Workflow Utilities",
        "description": "Split a montage-space Labels layer back into one labels layer per source image using stored montage tile metadata.",
        "tags": ["workflow", "montage", "labels", "split", "analysis"],
        "best_for": "Exporting per-image masks after editing a single montage mask layer.",
        "suggested_followup": "Use the created per-source labels for measurement, cleanup, export, or downstream analysis.",
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Split Montage Points To Sources",
        "category": "Workflow Utilities",
        "description": "Split a montage-space Points layer back into one points layer per source image, converting coordinates from montage space to source-local space.",
        "tags": ["workflow", "montage", "points", "split", "analysis"],
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Selected Layer Scratchpad",
        "category": "Workflow Utilities",
        "description": "A lightweight scratch template for experimenting against the selected layer inside the plugin runtime.",
        "tags": ["workflow", "selected-layer", "viewer", "starter"],
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
        "branch": "plugin_code",
        "template_type": "code",
        "title": "Background Stats Job",
        "category": "Developer Patterns",
        "description": "Compute summary statistics for the selected layer in the background and print the result on completion.",
        "tags": ["developer", "background", "selected-layer", "stats", "run_in_background"],
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

LEARNING_TEMPLATE_RECORDS: list[dict[str, Any]] = [
    {
        "id": "learning_microscopy_core_misconceptions",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Microscopy Concepts Students Misuse",
        "category": "Microscopy",
        "level": "graduate",
        "description": "Review the microscopy concepts graduate students often think they understand but misapply in practice.",
        "best_for": "Course review, lab onboarding, and correcting weak conceptual foundations before image analysis.",
        "suggested_followup": "Ask for a quiz, checklist, or real-example version tied to your own imaging workflow.",
        "tags": ["learning", "microscopy", "graduate", "misconceptions"],
        "prompt": "Teach me the 10 microscopy concepts graduate students most often misuse in practice, including resolution, contrast, magnification, numerical aperture, sampling, optical sectioning, signal-to-noise, dynamic range, deconvolution, and quantification. Explain each one with examples of how it causes wrong conclusions.",
    },
    {
        "id": "learning_microscopy_trustworthy_quantification",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Trustworthy Microscopy Quantification",
        "category": "Microscopy",
        "level": "graduate",
        "description": "Learn how to judge whether a microscopy image is suitable for quantitative claims.",
        "best_for": "Students moving from pretty images to scientifically defensible measurements.",
        "suggested_followup": "Ask for a lab-meeting checklist or figure-review checklist after the explanation.",
        "tags": ["learning", "microscopy", "quantification", "graduate"],
        "prompt": "Explain how to judge whether a microscopy image is only visually persuasive or actually trustworthy for measurement. Focus on sampling, detector limits, background, display scaling, and acquisition settings.",
    },
    {
        "id": "learning_microscopy_resolution_sampling",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Resolution, Pixel Size, And Sampling",
        "category": "Microscopy",
        "level": "graduate",
        "description": "Understand the practical relationship between optical resolution, pixel size, and sampling.",
        "best_for": "Students who can define these terms but still struggle to apply them correctly to real data.",
        "suggested_followup": "Ask for a worked example with one microscope setup and one bad acquisition setup.",
        "tags": ["learning", "microscopy", "resolution", "sampling"],
        "prompt": "Teach me how resolution, pixel size, and sampling relate in real experiments, including how undersampling and oversampling affect interpretation and quantification.",
    },
    {
        "id": "learning_microscopy_quantitative_vs_pretty",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Beautiful Image Vs Trustworthy Image",
        "category": "Microscopy",
        "level": "graduate",
        "description": "Separate image appearance from scientific validity in microscopy.",
        "best_for": "Students preparing figures, comparing conditions, or learning why aggressive display adjustments can mislead.",
        "suggested_followup": "Ask for examples involving contrast, filtering, deconvolution, and projection.",
        "tags": ["learning", "microscopy", "figures", "quantification"],
        "prompt": "Explain the difference between a beautiful microscopy image and a quantitatively trustworthy microscopy image.",
    },
    {
        "id": "learning_microscopy_intensity_comparison",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Intensity Comparison Pitfalls",
        "category": "Microscopy",
        "level": "graduate",
        "description": "Learn what must be controlled before comparing intensities across images.",
        "best_for": "Graduate students designing quantitative imaging experiments or reviewing figure claims critically.",
        "suggested_followup": "Ask for a checklist tailored to fluorescence microscopy, confocal microscopy, or live-cell imaging.",
        "tags": ["learning", "microscopy", "intensity", "comparison"],
        "prompt": "What should a graduate student check before comparing intensities across microscopy images from different sessions, samples, or microscopes?",
    },
    {
        "id": "learning_em_cautious_interpretation",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Cautious EM Interpretation",
        "category": "Electron Microscopy",
        "level": "graduate",
        "description": "Learn how to interpret EM images cautiously and avoid unjustified structural claims.",
        "best_for": "Graduate students reading EM figures, annotating micrographs, or preparing lab presentations.",
        "suggested_followup": "Ask for a paper-reading checklist or a mentor-style walkthrough after the explanation.",
        "tags": ["learning", "em", "interpretation", "graduate"],
        "prompt": "Teach me how to interpret EM images cautiously at graduate level, focusing on sectioning ambiguity, preparation artifacts, contrast mechanisms, and why a convincing image can still support a weak conclusion.",
    },
    {
        "id": "learning_em_structure_identification_traps",
        "branch": "learning",
        "template_type": "prompt",
        "title": "EM Structure Identification Traps",
        "category": "Electron Microscopy",
        "level": "graduate",
        "description": "Review the structures students most often misidentify in EM images.",
        "best_for": "Students who need to defend structural labels and avoid overconfident annotation.",
        "suggested_followup": "Ask for a quiz version focused on organelle recognition and false confidence traps.",
        "tags": ["learning", "em", "structures", "misconceptions"],
        "prompt": "Explain the 10 most common mistakes students make when identifying structures in EM images, especially membranes, vesicles, mitochondria, myelin, dense bodies, and empty-looking regions.",
    },
    {
        "id": "learning_em_artifact_vs_structure",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Artifact Vs Structure In EM",
        "category": "Electron Microscopy",
        "level": "graduate",
        "description": "Train students to separate likely biological structure from likely preparation artifact.",
        "best_for": "Graduate-level image reading, methods training, and cautious paper interpretation.",
        "suggested_followup": "Ask for a comparison table of common artifacts, their visual appearance, and likely causes.",
        "tags": ["learning", "em", "artifact", "interpretation"],
        "prompt": "Teach me how to separate likely biological structure from likely preparation artifact in electron microscopy, using concrete visual clues and common failure modes.",
    },
    {
        "id": "learning_em_single_section_limits",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Limits Of Single-Section EM",
        "category": "Electron Microscopy",
        "level": "graduate",
        "description": "Learn why single-section EM images are easy to overinterpret and how experts reason more carefully.",
        "best_for": "Students who need to understand what can and cannot be inferred from one section or one plane.",
        "suggested_followup": "Ask for examples where serial sections or complementary evidence would be needed.",
        "tags": ["learning", "em", "sectioning", "critical-reading"],
        "prompt": "Explain why single-section EM images are easy to overinterpret and how an expert would reason more carefully before assigning structure or function.",
    },
    {
        "id": "learning_em_paper_checklist",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Read An EM Figure Critically",
        "category": "Electron Microscopy",
        "level": "graduate",
        "description": "Build a checklist for evaluating how strong an EM figure's claims really are.",
        "best_for": "Journal club, graduate seminars, and critical reading of high-confidence-looking images.",
        "suggested_followup": "Ask the assistant to apply the checklist to a specific figure or paper claim.",
        "tags": ["learning", "em", "papers", "critical-reading"],
        "prompt": "Give me a graduate-level checklist for reading an EM figure critically in a paper, including what claims are justified and what claims need more evidence.",
    },
    {
        "id": "learning_biophotonics_signal_formation",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Fluorescence Signal Formation",
        "category": "Biophotonics",
        "level": "graduate",
        "description": "Learn how excitation, emission, quantum yield, and detector properties act together to determine observed fluorescence signal.",
        "best_for": "Students who need to understand fluorescence as a system-level signal formation process instead of only a molecular property.",
        "suggested_followup": "Ask for a probe-selection example using one fluorophore, one filter set, and one detector.",
        "tags": ["learning", "biophotonics", "fluorescence", "signal-formation"],
        "prompt": "Explain how excitation wavelength, emission wavelength, quantum yield, and detector sensitivity together determine whether a fluorescent probe produces a strong signal in a biological microscopy experiment.",
    },
    {
        "id": "learning_biophotonics_bleaching_vs_toxicity",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Photobleaching Vs Phototoxicity",
        "category": "Biophotonics",
        "level": "graduate",
        "description": "Distinguish probe failure from biological damage in live-cell fluorescence imaging.",
        "best_for": "Students designing live imaging experiments who need to balance signal quality against biological perturbation.",
        "suggested_followup": "Ask for a practical acquisition decision tree for weak live-cell fluorescence imaging.",
        "tags": ["learning", "biophotonics", "photobleaching", "phototoxicity"],
        "prompt": "Photobleaching and phototoxicity both limit live-cell fluorescence imaging. Explain how they arise from light-matter interactions and how imaging parameters should be adjusted differently to minimize each.",
    },
    {
        "id": "learning_biophotonics_imaging_depth",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Imaging Depth In Tissue",
        "category": "Biophotonics",
        "level": "graduate",
        "description": "Explain why scattering and absorption limit deep imaging and why wavelength choice matters in tissue.",
        "best_for": "Students learning why deep-imaging methods and near-infrared strategies are needed in biological tissue.",
        "suggested_followup": "Ask for a comparison between one-photon and multiphoton deep tissue imaging logic.",
        "tags": ["learning", "biophotonics", "tissue-optics", "imaging-depth"],
        "prompt": "Explain how scattering and absorption influence imaging depth in biological tissue and why near-infrared wavelengths are often preferred for deep imaging.",
    },
    {
        "id": "learning_biophotonics_index_mismatch",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Refractive Index Mismatch",
        "category": "Biophotonics",
        "level": "graduate",
        "description": "Understand how refractive index mismatch degrades image quality and distorts biological interpretation.",
        "best_for": "Students troubleshooting poor image quality or depth-dependent degradation in fluorescence microscopy.",
        "suggested_followup": "Ask for an explanation using PSF distortion and spherical aberration in one concrete imaging setup.",
        "tags": ["learning", "biophotonics", "refractive-index", "aberration"],
        "prompt": "How do refractive index mismatches between immersion medium, coverslip, and biological tissue affect image formation quality in fluorescence microscopy?",
    },
    {
        "id": "learning_biophotonics_probe_selection",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Probe Choice As Experimental Design",
        "category": "Biophotonics",
        "level": "graduate",
        "description": "Treat fluorescent probe selection as a full measurement design decision rather than a color choice.",
        "best_for": "Students selecting probes for living tissue experiments where penetration, photostability, and signal origin all matter.",
        "suggested_followup": "Ask for a live-tissue design example comparing two candidate probes and one detector/filter setup.",
        "tags": ["learning", "biophotonics", "probe-selection", "experimental-design"],
        "prompt": "When choosing a fluorescent probe to measure a biological process inside living tissue, how should excitation wavelength, emission spectrum, photostability, tissue scattering properties, and signal origin be considered together?",
    },
    {
        "id": "learning_quant_intensity_proportionality",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Intensity As Measurement",
        "category": "Quantitative Imaging",
        "level": "graduate",
        "description": "Learn when fluorescence pixel intensity can and cannot be treated as proportional to molecular concentration.",
        "best_for": "Students making the transition from visual imaging to quantitative biological measurement.",
        "suggested_followup": "Ask for a checklist of assumptions that must hold before reporting intensity as concentration-related evidence.",
        "tags": ["learning", "quantitative-imaging", "intensity", "measurement"],
        "prompt": "Under what experimental conditions can fluorescence pixel intensity be interpreted as proportional to molecular concentration in a biological sample?",
    },
    {
        "id": "learning_quant_sampling_accuracy",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Sampling And Measurement Accuracy",
        "category": "Quantitative Imaging",
        "level": "graduate",
        "description": "Explain how insufficient spatial sampling produces measurement error even when optics are good.",
        "best_for": "Students who understand resolution conceptually but do not yet see how digital sampling corrupts size measurements.",
        "suggested_followup": "Ask for a numerical example using one optical resolution value and two pixel-size choices.",
        "tags": ["learning", "quantitative-imaging", "sampling", "measurement"],
        "prompt": "How does insufficient spatial sampling affect the accuracy of size measurements in fluorescence microscopy images?",
    },
    {
        "id": "learning_quant_shot_noise",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Shot Noise And Quantification",
        "category": "Quantitative Imaging",
        "level": "graduate",
        "description": "Understand how photon statistics limit quantitative precision and why more light is not always a free improvement.",
        "best_for": "Students learning that quantitative imaging is often noise-limited, bleaching-limited, or toxicity-limited rather than optics-limited.",
        "suggested_followup": "Ask for a comparison between increasing exposure, illumination, and detector gain for one weak fluorescence experiment.",
        "tags": ["learning", "quantitative-imaging", "shot-noise", "precision"],
        "prompt": "Explain how shot noise limits the precision of fluorescence intensity measurements and why increasing illumination intensity does not always improve quantitative accuracy.",
    },
    {
        "id": "learning_quant_index_bias",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Depth Bias From Index Mismatch",
        "category": "Quantitative Imaging",
        "level": "graduate",
        "description": "Learn how refractive index mismatch creates hidden depth-dependent bias in quantitative fluorescence measurements.",
        "best_for": "Graduate students measuring intensity or morphology across different imaging depths in tissue or thick samples.",
        "suggested_followup": "Ask for an example showing how apparent intensity change can arise from optics rather than biology.",
        "tags": ["learning", "quantitative-imaging", "refractive-index", "depth-bias"],
        "prompt": "How do refractive index mismatches between immersion medium and biological tissue affect quantitative fluorescence measurements at increasing imaging depth?",
    },
    {
        "id": "learning_quant_end_to_end_design",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Design A Quantitative Imaging Experiment",
        "category": "Quantitative Imaging",
        "level": "graduate",
        "description": "Design an end-to-end fluorescence measurement workflow while explicitly controlling systematic error sources.",
        "best_for": "Students learning to treat microscopy as measurement engineering rather than only visualization.",
        "suggested_followup": "Ask for the same design problem but focused on fixed samples, thick tissue, or time-lapse live imaging.",
        "tags": ["learning", "quantitative-imaging", "experimental-design", "workflow"],
        "prompt": "Design a fluorescence microscopy experiment that measures relative protein concentration changes inside living cells while minimizing systematic measurement errors. Justify each acquisition decision.",
    },
    {
        "id": "learning_stats_effect_size_vs_significance",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Effect Size Vs Significance",
        "category": "Statistics",
        "level": "graduate",
        "description": "Separate biological importance from p-value significance in imaging-based experiments.",
        "best_for": "Students who report statistical significance without yet understanding effect size, magnitude, and biological relevance.",
        "suggested_followup": "Ask for an imaging-specific example where a tiny effect becomes significant only because sample size is large.",
        "tags": ["learning", "statistics", "effect-size", "significance"],
        "prompt": "A fluorescence experiment comparing two conditions shows a statistically significant difference (p < 0.05), but the mean intensity difference between groups is small. Explain why statistical significance does not necessarily imply biological importance and how effect size should be interpreted in microscopy experiments.",
    },
    {
        "id": "learning_stats_repeated_measures",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Repeated Measures In Time-Lapse",
        "category": "Statistics",
        "level": "graduate",
        "description": "Explain why time-series measurements from the same cells are not independent observations.",
        "best_for": "Students analyzing live-cell imaging data who need to avoid pseudo-replication and incorrect test choice.",
        "suggested_followup": "Ask for examples of what goes wrong if each time point is treated as an independent sample.",
        "tags": ["learning", "statistics", "repeated-measures", "time-lapse"],
        "prompt": "In a live-cell imaging experiment tracking fluorescence intensity over time in the same cells, explain why repeated-measures statistical analysis is more appropriate than treating each time point as an independent observation.",
    },
    {
        "id": "learning_stats_distribution_assumptions",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Distribution Assumptions For Imaging Data",
        "category": "Statistics",
        "level": "graduate",
        "description": "Learn why image-derived measurements often violate textbook distribution assumptions and how that changes test selection.",
        "best_for": "Students who need better judgment when choosing between parametric and nonparametric tests for microscopy data.",
        "suggested_followup": "Ask for examples of skewed fluorescence distributions and how to diagnose them before testing.",
        "tags": ["learning", "statistics", "distribution", "imaging-data"],
        "prompt": "Pixel intensity measurements from fluorescence microscopy often do not follow a normal distribution. Explain why this occurs and how distribution assumptions affect the choice between parametric and nonparametric statistical tests.",
    },
    {
        "id": "learning_stats_measurement_uncertainty",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Measurement Uncertainty In Microscopy",
        "category": "Statistics",
        "level": "graduate",
        "description": "Distinguish measurement uncertainty from biological variability in segmented fluorescence measurements.",
        "best_for": "Graduate students quantifying regions or cells who need to reason more carefully about uncertainty sources.",
        "suggested_followup": "Ask for a table separating segmentation error, detector noise, bleaching drift, and background subtraction uncertainty.",
        "tags": ["learning", "statistics", "uncertainty", "quantification"],
        "prompt": "When measuring fluorescence intensity from segmented cellular regions, what sources of measurement uncertainty should be considered and how can they influence statistical conclusions?",
    },
    {
        "id": "learning_stats_image_evidence_pipeline",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Imaging Data As Biological Evidence",
        "category": "Statistics",
        "level": "graduate",
        "description": "Evaluate whether image-derived differences reflect true biology or a chain of acquisition and analysis bias.",
        "best_for": "Students and staff who need to reason from image measurements to biological claims without skipping validation steps.",
        "suggested_followup": "Ask for a repeatable evidence-check workflow you can apply before writing a figure legend or result statement.",
        "tags": ["learning", "statistics", "evidence", "biological-interpretation"],
        "prompt": "A study reports increased fluorescence intensity in treated cells compared with controls. Describe the steps required to determine whether this difference reflects a true biological change rather than imaging artifacts or analysis bias.",
    },
    {
        "id": "learning_image_formation_diffraction_psf",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Diffraction, PSF, And Resolution",
        "category": "Image Formation",
        "level": "graduate",
        "description": "Explain the central image-formation chain from diffraction to PSF to diffraction-limited resolution.",
        "best_for": "Students who know the words but need the physical chain linking them together correctly.",
        "suggested_followup": "Ask for an intuitive explanation plus one mathematically sharper explanation for graduate students.",
        "tags": ["learning", "image-formation", "diffraction", "psf", "resolution"],
        "prompt": "Explain how diffraction leads to the formation of the point spread function (PSF) in a microscope and how the PSF determines the diffraction-limited resolution of the imaging system.",
    },
    {
        "id": "learning_image_formation_sampling_vs_resolution",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Sampling Vs Optical Resolution",
        "category": "Image Formation",
        "level": "graduate",
        "description": "Separate optical resolution from digital sampling and explain what happens when the detector undersamples the specimen.",
        "best_for": "Students who confuse a sharp objective with a faithfully recorded image.",
        "suggested_followup": "Ask for a Nyquist-based example using specimen-plane pixel size and one resolution value.",
        "tags": ["learning", "image-formation", "sampling", "pixel-size", "nyquist"],
        "prompt": "A microscope objective provides 250 nm optical resolution, but the detector pixel size corresponds to 400 nm in the specimen plane. Explain what happens to image information in this situation and how Nyquist sampling determines the correct pixel size choice.",
    },
    {
        "id": "learning_image_formation_optical_sectioning",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Optical Sectioning Mechanisms",
        "category": "Image Formation",
        "level": "graduate",
        "description": "Compare how different microscopy modes create or reject out-of-focus signal in thick samples.",
        "best_for": "Students who need to understand contrast formation in thick specimens rather than memorizing microscope names.",
        "suggested_followup": "Ask for a direct comparison of widefield, confocal, and multiphoton imaging in one thick tissue example.",
        "tags": ["learning", "image-formation", "optical-sectioning", "confocal", "multiphoton"],
        "prompt": "Compare how widefield fluorescence microscopy, confocal microscopy, and multiphoton microscopy differ in their optical sectioning mechanisms and explain how this affects the final image contrast in thick biological samples.",
    },
    {
        "id": "learning_image_formation_detector_noise",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Detector Noise And SNR",
        "category": "Image Formation",
        "level": "graduate",
        "description": "Explain how shot noise, read noise, and detector sensitivity shape signal-to-noise rather than just image brightness.",
        "best_for": "Students who need to distinguish image brightness from image quality and understand why more light is not always better.",
        "suggested_followup": "Ask for a side-by-side explanation for camera-based detectors and photon-limited imaging.",
        "tags": ["learning", "image-formation", "detector", "noise", "snr"],
        "prompt": "Explain how shot noise, read noise, and detector sensitivity influence the signal-to-noise ratio of a fluorescence microscopy image and why increasing illumination intensity does not always improve image quality.",
    },
    {
        "id": "learning_image_formation_integrated_pipeline",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Integrated Image Formation Pipeline",
        "category": "Image Formation",
        "level": "graduate",
        "description": "Integrate optics, sampling, sectioning, and detector physics into one reasoning chain for resolvability.",
        "best_for": "Students who need system-level imaging intuition rather than isolated definitions.",
        "suggested_followup": "Ask for a worked imaging scenario involving two nearby puncta in a thick sample with weak fluorescence.",
        "tags": ["learning", "image-formation", "integrated", "resolution", "detector"],
        "prompt": "Describe how diffraction, PSF size, numerical aperture, sampling rate, pixel size, optical sectioning method, and detector noise together determine whether two nearby fluorescent structures can be reliably distinguished in a biological microscopy image.",
    },
    {
        "id": "learning_prompting_question_upgrade",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Upgrade A Science Question",
        "category": "Academic Prompting",
        "level": "graduate",
        "description": "Learn how to turn a vague science question into a stronger academic prompt with better scope and rigor.",
        "best_for": "Students who know the topic they want to learn but do not yet know how to ask for a high-quality explanation.",
        "suggested_followup": "Ask for versions tailored to undergraduate study, graduate coursework, or paper-reading support.",
        "tags": ["learning", "academic-prompting", "question-design", "graduate"],
        "prompt": "Turn my vague science question into three stronger academic prompts: one for fast understanding, one for exam study, and one for graduate-level critical reasoning. For each one, explain why it is better.",
    },
    {
        "id": "learning_prompting_assumptions_limits",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Ask For Assumptions And Limits",
        "category": "Academic Prompting",
        "level": "graduate",
        "description": "Teach students how to ask for assumptions, limitations, and alternative interpretations instead of only definitions.",
        "best_for": "Moving from memorization-style questions to prompts that support scientific reasoning.",
        "suggested_followup": "Ask for the same pattern adapted to microscopy, EM, statistics, or paper interpretation.",
        "tags": ["learning", "academic-prompting", "assumptions", "critical-thinking"],
        "prompt": "Teach me how to ask a scientific question so the answer includes assumptions, limitations, uncertainties, and alternative interpretations rather than only a definition.",
    },
    {
        "id": "learning_prompting_weak_vs_strong",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Weak Prompt Vs Strong Prompt",
        "category": "Academic Prompting",
        "level": "graduate",
        "description": "Show the difference between weak, decent, and strong prompts for learning complex scientific topics.",
        "best_for": "Students who need a concrete sense of what makes one academic prompt more useful than another.",
        "suggested_followup": "Ask the assistant to critique one of your own real prompts using the same framework.",
        "tags": ["learning", "academic-prompting", "comparison", "study-skills"],
        "prompt": "Show me the difference between a weak prompt, a decent prompt, and an excellent prompt for learning a complex science topic. Use the same topic across all three versions and explain why the stronger versions produce better learning.",
    },
    {
        "id": "learning_prompting_paper_reading",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Prompting For Paper Reading",
        "category": "Academic Prompting",
        "level": "graduate",
        "description": "Teach students how to use AI prompts to read research papers more critically and efficiently.",
        "best_for": "Students overwhelmed by primary literature who need structure without losing rigor.",
        "suggested_followup": "Ask for a repeatable workflow that separates summary, critique, and follow-up questions.",
        "tags": ["learning", "academic-prompting", "papers", "graduate"],
        "prompt": "Teach me how to ask better AI questions when reading a research paper for the first time. Include prompts for summary, concept clarification, figure critique, method limitations, and follow-up experiments.",
    },
    {
        "id": "learning_prompting_audience_scaling",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Prompt For The Right Academic Level",
        "category": "Academic Prompting",
        "level": "graduate",
        "description": "Show how to request explanations at different academic levels without changing the scientific core.",
        "best_for": "Students, staff, and educators who need the same concept explained for different audiences.",
        "suggested_followup": "Ask for versions aimed at high school, first-year university, graduate coursework, and journal club.",
        "tags": ["learning", "academic-prompting", "audience", "teaching"],
        "prompt": "Show me how to prompt an AI tutor differently for high school, undergraduate, and graduate learning while keeping the scientific content accurate.",
    },
    {
        "id": "learning_language_hungarian_start",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Hungarian Onboarding Prompt",
        "category": "Language Support",
        "level": "university",
        "description": "A Hungarian-language starter prompt for students who want to begin learning image analysis in napari without an image loaded yet.",
        "best_for": "Testing onboarding help and language accessibility for Hungarian-speaking students.",
        "suggested_followup": "Ask the assistant to continue the workflow in Hungarian after synthetic data is loaded.",
        "tags": ["learning", "language", "hungarian", "onboarding"],
        "prompt": "Nem értem az angol nyelvet. Kérlek, magyarul válaszolj.\nSzeretnék tanulni a képfeldolgozásról napari használatával, de még nem töltöttem be képet.\nMit kell először tennem?",
    },
    {
        "id": "learning_language_malay_start",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Malay Onboarding Prompt",
        "category": "Language Support",
        "level": "university",
        "description": "A Malay-language starter prompt for students beginning napari image-analysis learning without loaded data.",
        "best_for": "Testing multilingual onboarding and synthetic-data discovery for Malay-speaking learners.",
        "suggested_followup": "Ask the assistant to continue in Malay with one concrete synthetic-data practice workflow.",
        "tags": ["learning", "language", "malay", "onboarding"],
        "prompt": "Saya tidak fasih berbahasa Inggeris.\nSaya ingin belajar tentang analisis imej dalam napari, tetapi sekarang saya belum memuatkan sebarang imej.\nBolehkah anda membantu saya bermula tanpa imej?\nContohnya, adakah terdapat data contoh atau imej sintetik yang boleh saya gunakan untuk latihan?",
    },
    {
        "id": "learning_language_chinese_start",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Chinese Onboarding Prompt",
        "category": "Language Support",
        "level": "university",
        "description": "A Chinese-language starter prompt for students who want onboarding help in napari without any image loaded.",
        "best_for": "Testing multilingual onboarding for Chinese-speaking users.",
        "suggested_followup": "Ask the assistant to keep all future explanations in Chinese for the session.",
        "tags": ["learning", "language", "chinese", "onboarding"],
        "prompt": "我不会说英语。我想学习使用 napari 做图像分析，但现在还没有加载任何图像。我应该先做什么？请用中文回答。",
    },
    {
        "id": "learning_language_japanese_start",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Japanese Onboarding Prompt",
        "category": "Language Support",
        "level": "university",
        "description": "A Japanese-language starter prompt for first-step napari onboarding without a loaded image.",
        "best_for": "Testing whether the assistant can start image-analysis learning workflows in Japanese.",
        "suggested_followup": "Ask for a Japanese explanation of synthetic data options and the next analysis step.",
        "tags": ["learning", "language", "japanese", "onboarding"],
        "prompt": "英語が分かりませんので、日本語で返信してください。\n私はnapariの画像解析について学びたいですが、まだ画像を読み込んでいません。\n最初に何をすればよいか教えてください。",
    },
    {
        "id": "learning_language_korean_start",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Korean Onboarding Prompt",
        "category": "Language Support",
        "level": "university",
        "description": "A Korean-language onboarding prompt for students beginning napari image analysis without data.",
        "best_for": "Testing accessibility for Korean-speaking learners starting from an empty viewer.",
        "suggested_followup": "Ask for a Korean beginner workflow using one synthetic dataset and one simple measurement task.",
        "tags": ["learning", "language", "korean", "onboarding"],
        "prompt": "저는 영어를 이해하지 못합니다. 한국어로 답변해 주세요.\n저는 napari를 사용한 이미지 분석을 배우고 싶지만 아직 이미지를 불러오지 않았습니다.\n먼저 무엇을 해야 하는지 알려 주세요.",
    },
    {
        "id": "learning_language_french_start",
        "branch": "learning",
        "template_type": "prompt",
        "title": "French Onboarding Prompt",
        "category": "Language Support",
        "level": "university",
        "description": "A French-language prompt for starting napari image-analysis learning without loaded images.",
        "best_for": "Testing multilingual onboarding for French-speaking students.",
        "suggested_followup": "Ask the assistant to continue the lesson in French with a synthetic example dataset.",
        "tags": ["learning", "language", "french", "onboarding"],
        "prompt": "Je ne comprends pas l’anglais. Merci de répondre en français.\nJe voudrais apprendre l’analyse d’images avec napari, mais je n’ai pas encore chargé d’image.\nQue dois-je faire en premier ?",
    },
    {
        "id": "learning_language_spanish_start",
        "branch": "learning",
        "template_type": "prompt",
        "title": "Spanish Onboarding Prompt",
        "category": "Language Support",
        "level": "university",
        "description": "A Spanish-language prompt for beginning image-analysis learning in napari without any image loaded.",
        "best_for": "Testing onboarding help for Spanish-speaking students and staff.",
        "suggested_followup": "Ask for a Spanish beginner pipeline using synthetic data and one threshold preview.",
        "tags": ["learning", "language", "spanish", "onboarding"],
        "prompt": "No entiendo inglés. Por favor, responde en español.\nQuiero aprender análisis de imágenes con napari, pero todavía no he cargado ninguna imagen.\n¿Qué debo hacer primero?",
    },
    {
        "id": "learning_language_german_start",
        "branch": "learning",
        "template_type": "prompt",
        "title": "German Onboarding Prompt",
        "category": "Language Support",
        "level": "university",
        "description": "A German-language starter prompt for learning napari image analysis without a currently loaded image.",
        "best_for": "Testing onboarding and prompt-template accessibility for German-speaking users.",
        "suggested_followup": "Ask for a German explanation of which built-in template to run first.",
        "tags": ["learning", "language", "german", "onboarding"],
        "prompt": "Ich verstehe kein Englisch. Bitte antworten Sie auf Deutsch.\nIch möchte die Bildanalyse mit napari lernen, aber ich habe noch kein Bild geladen.\nWas sollte ich zuerst tun?",
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
                "branch": "plugin_code",
                "template_type": "code",
                "title": title,
                "category": "Data Setup",
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


ALL_TEMPLATE_RECORDS: list[dict[str, Any]] = (
    PROMPT_TEMPLATE_RECORDS
    + CODE_TEMPLATE_RECORDS
    + _demo_pack_template_records()
    + LEARNING_TEMPLATE_RECORDS
)


def template_records() -> list[dict[str, Any]]:
    return [dict(record) for record in ALL_TEMPLATE_RECORDS]


def template_sections() -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for section_key in SECTION_ORDER:
        ordered_categories = CATEGORY_ORDER.get(section_key, [])
        category_entries: list[dict[str, Any]] = []
        for category in ordered_categories:
            templates = [
                dict(record)
                for record in ALL_TEMPLATE_RECORDS
                if str(record.get("branch", "")).strip() == section_key
                and str(record.get("category", "")).strip() == category
            ]
            category_entries.append({"name": category, "templates": templates})
        if category_entries:
            sections.append({"key": section_key, "label": SECTION_LABELS[section_key], "categories": category_entries})
    return sections


def template_library_payload() -> dict[str, Any]:
    return {
        "version": TEMPLATE_LIBRARY_VERSION,
        "sections": template_sections(),
        "templates": template_records(),
    }


def is_template_record(record: Any) -> bool:
    return isinstance(record, dict) and str(record.get("template_type", "")).strip() in {"prompt", "code"}


def template_section_colors(section_key: str) -> dict[str, str]:
    return dict(SECTION_COLORS.get(str(section_key or "").strip(), {"section": "#d6deeb", "category": "#d6deeb", "item": "#d6deeb"}))


def template_load_target(record: dict[str, Any]) -> str:
    return "prompt" if str(record.get("template_type", "")).strip() == "prompt" else "code"


def template_run_target(record: dict[str, Any]) -> str:
    return "send_prompt" if str(record.get("template_type", "")).strip() == "prompt" else "run_code"


def template_body_text(record: dict[str, Any]) -> str:
    if str(record.get("template_type", "")).strip() == "prompt":
        return str(record.get("prompt", "")).strip()
    return str(record.get("code", ""))


def template_button_labels(record: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(record, dict):
        return ("Load Template", "Run Template")
    if str(record.get("template_type", "")).strip() == "prompt":
        if str(record.get("branch", "")).strip() == "learning":
            return ("Load Prompt", "Ask Now")
        return ("Load Prompt", "Run Prompt")
    return ("Load Code", "Run Code")


def template_hint_text(record: dict[str, Any] | None) -> str:
    if not isinstance(record, dict):
        return "Click a template to preview it. Load or run behavior depends on the selected template type."
    if str(record.get("template_type", "")).strip() == "prompt":
        if str(record.get("branch", "")).strip() == "learning":
            return "Click to preview. Load Prompt inserts the study prompt. Ask Now sends it to the assistant."
        return "Click to preview. Load Prompt inserts the workflow prompt. Run Prompt sends it to the assistant."
    return "Click to preview. Load Code inserts the code into the Prompt box. Run Code executes it with Run My Code."


def template_preview_text(record: dict[str, Any]) -> str:
    title = str(record.get("title", "")).strip() or "Untitled Template"
    section_label = SECTION_LABELS.get(str(record.get("branch", "")).strip(), "Templates")
    category = str(record.get("category", "")).strip() or "Templates"
    description = str(record.get("description", "")).strip()
    tags = [str(tag).strip() for tag in record.get("tags", []) if str(tag).strip()]
    best_for = str(record.get("best_for", "")).strip()
    followup = str(record.get("suggested_followup", "")).strip()
    level = str(record.get("level", "")).strip()
    runtime = record.get("runtime", {})
    runtime_flags: list[str] = []
    if isinstance(runtime, dict):
        if runtime.get("uses_viewer"):
            runtime_flags.append("Viewer")
        if runtime.get("uses_selected_layer"):
            runtime_flags.append("Selected Layer")
        if runtime.get("uses_run_in_background"):
            runtime_flags.append("Background")
    kind_label = "Prompt Template" if str(record.get("template_type", "")).strip() == "prompt" else "Code Template"
    lines = [f"{kind_label}: {title}", f"Section: {section_label}", f"Category: {category}"]
    if level:
        lines.append(f"Level: {level.title()}")
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")
    if runtime_flags:
        lines.append(f"Runtime: {', '.join(runtime_flags)}")
    if description:
        lines.extend(["", description])
    if best_for:
        lines.extend(["", f"Best for: {best_for}"])
    if followup:
        lines.extend(["", f"Suggested follow-up: {followup}"])
    if str(record.get("template_type", "")).strip() == "prompt":
        lines.extend(["", "Load to Prompt:", str(record.get("prompt", "")).strip()])
    else:
        lines.extend(
            [
                "",
                "This template is designed to run inside the Chat Assistant plugin runtime in napari.",
                "It may use viewer, selected_layer, and run_in_background(...).",
                "",
                "Code:",
                str(record.get("code", "")).rstrip(),
            ]
        )
    return "\n".join(lines).strip()
