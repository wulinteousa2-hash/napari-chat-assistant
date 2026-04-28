from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from napari_chat_assistant.agent.code_validation import (
    build_code_repair_context,
    compact_code_repair_user_message,
    normalize_generated_code_if_needed,
    validate_generated_code,
)


REPAIR_REGRESSION_CASES = [
    {
        "name": "selected_layer_index_misuse",
        "input_code": """
layer = viewer.layers[selected_layer]
print(layer.name)
""",
        "expected_substrings": ["layer = selected_layer"],
        "forbidden_substrings": ["viewer.layers[selected_layer]"],
        "expected_notes": ["selected_layer"],
    },
    {
        "name": "placeholder_image_lookup_replaced_with_selected_image",
        "input_code": """
from napari import run_in_background
layer = viewer.layers["img_a"]
print(layer.shape)
""",
        "viewer_setup": "selected_image",
        "expected_substrings": ['layer = viewer.layers["sample_a"]', "data = np.asarray(layer.data)", "print(data.shape)"],
        "forbidden_substrings": ['from napari import run_in_background', 'viewer.layers["img_a"]', "print(layer.shape)"],
        "expected_notes": ["Removed `run_in_background` import", "placeholder layer name [img_a]", "layer.shape"],
    },
    {
        "name": "invalid_scipy_symbol_rewritten_to_gaussian_filter",
        "input_code": """
from scipy.ndimage import gaussian_noise

result = gaussian_noise(data, sigma=1.2)
""",
        "expected_substrings": ["from scipy.ndimage import gaussian_filter", "result = gaussian_noise(data, sigma=1.2)"],
        "forbidden_substrings": ["from scipy.ndimage import gaussian_noise"],
        "expected_notes": ["scipy.ndimage", "gaussian_filter"],
    },
]


VALIDATION_REGRESSION_CASES = [
    {
        "name": "viewer_mutation_inside_background_compute",
        "input_code": """
import numpy as np

layer = selected_layer
data = np.asarray(layer.data)

def compute():
    result = data > data.mean()
    viewer.add_labels(result.astype(np.uint8), name="bad_inside_compute")
    return result

run_in_background(compute, lambda x: print("done"))
""",
        "expected_errors": ["contains viewer mutations"],
    },
]


def test_validate_generated_code_rejects_uint8_inplace_randint_add():
    report = validate_generated_code(
        """
import numpy as np
image = np.zeros((32, 32), dtype=np.uint8)
image += np.random.randint(0, 20, size=image.shape)
"""
    )

    assert any("Unsafe in-place arithmetic on uint8 array [image]" in error for error in report.errors)
    assert report.warnings == []


def test_validate_generated_code_rejects_uint8_inplace_add_from_noise_variable():
    report = validate_generated_code(
        """
import numpy as np
image = np.zeros((32, 32), dtype=np.uint8)
noise = np.random.randint(0, 20, size=image.shape)
image += noise
"""
    )

    assert any("using integer noise array [noise]" in error for error in report.errors)


def test_validate_generated_code_allows_uint8_randint_when_dtype_is_explicit():
    report = validate_generated_code(
        """
import numpy as np
image = np.zeros((32, 32), dtype=np.uint8)
noise = np.random.randint(0, 20, size=image.shape, dtype=np.uint8)
image = np.clip(image + noise, 0, 255).astype(np.uint8)
"""
    )

    assert report.errors == []
    assert report.warnings == []


def test_validate_generated_code_rejects_missing_napari_symbol_import():
    report = validate_generated_code("from napari.utils import status\nprint(status)")

    assert any("Unsupported napari import: [status] was not found in [napari.utils]" in warning for warning in report.warnings)
    assert report.errors == []
    assert report.has_blocking_issues("strict")
    assert not report.has_blocking_issues("permissive")


def test_normalize_generated_code_strips_run_in_background_napari_import():
    normalized, report = normalize_generated_code_if_needed(
        """
from napari.utils import run_in_background

def compute():
    return 1

def apply_result(payload):
    print(payload)

run_in_background(compute, apply_result, label="demo")
"""
    )

    assert "from napari.utils import run_in_background" not in normalized
    assert report.errors == []
    assert report.warnings == []


@pytest.mark.parametrize("case", REPAIR_REGRESSION_CASES, ids=lambda case: case["name"])
def test_repair_regression_cases(case, make_napari_viewer_proxy):
    viewer = None
    if case.get("viewer_setup") == "selected_image":
        viewer = make_napari_viewer_proxy()
        image = viewer.add_image(np.asarray([[1, 2], [3, 4]], dtype=np.float32), name="sample_a")
        viewer.layers.selection.active = image

    normalized, report = normalize_generated_code_if_needed(case["input_code"], viewer=viewer)

    for snippet in case.get("expected_substrings", []):
        assert snippet in normalized
    for snippet in case.get("forbidden_substrings", []):
        assert snippet not in normalized
    for note in case.get("expected_notes", []):
        assert any(note in entry for entry in report.notes)


@pytest.mark.parametrize("case", VALIDATION_REGRESSION_CASES, ids=lambda case: case["name"])
def test_validation_regression_cases(case):
    report = validate_generated_code(case["input_code"])

    for error in case.get("expected_errors", []):
        assert any(error in entry for entry in report.errors)


def test_validate_generated_code_rejects_missing_viewer_method():
    viewer = SimpleNamespace(add_image=lambda *args, **kwargs: None)

    report = validate_generated_code("viewer.add_histogram(data)", viewer=viewer)

    assert any("Unsupported viewer API: [viewer.add_histogram]" in warning for warning in report.warnings)
    assert report.errors == []
    assert report.has_blocking_issues("strict")
    assert not report.has_blocking_issues("permissive")


def test_validate_generated_code_allows_existing_viewer_method():
    viewer = SimpleNamespace(add_image=lambda *args, **kwargs: None)

    report = validate_generated_code("viewer.add_image(data)", viewer=viewer)

    assert report.errors == []
    assert report.warnings == []


def test_validate_generated_code_rejects_invalid_layer_type_attribute_check():
    report = validate_generated_code(
        """
selected_layer = viewer.layers.selection.active
if selected_layer is not None and selected_layer._type == "shapes":
    print(selected_layer.name)
"""
    )

    assert any("selected_layer._type" in warning for warning in report.warnings)
    assert report.errors == []


def test_validate_generated_code_rejects_invalid_layer_type_property_check():
    report = validate_generated_code(
        """
selected_layer = viewer.layers.selection.active
if selected_layer is not None and selected_layer.type == "image":
    print(selected_layer.name)
"""
    )

    assert any("selected_layer.type" in warning for warning in report.warnings)
    assert report.errors == []


def test_validate_generated_code_rejects_napari_run_call():
    report = validate_generated_code(
        """
import napari
napari.run()
"""
    )

    assert any("napari.run()" in error for error in report.errors)


def test_validate_generated_code_warns_on_invalid_viewer_keyword():
    viewer = SimpleNamespace(add_image=lambda data, name=None: None)

    report = validate_generated_code("viewer.add_image(data, colormap='gray')", viewer=viewer)

    assert any("Invalid viewer keyword" in warning for warning in report.warnings)
    assert report.errors == []
    assert report.has_blocking_issues("strict")


def test_build_code_repair_context_includes_import_symbol_repair_note():
    context = build_code_repair_context(
        """
fix this code:
```python
from scipy.ndimage import gaussian_noise

result = gaussian_noise(data, sigma=1.2)
```
"""
    )

    assert context is not None
    assert "from scipy.ndimage import gaussian_filter" in context["normalized_code_candidate"]
    notes = context["local_validation"]["notes"]
    assert any("gaussian_filter" in note for note in notes)


def test_build_code_repair_context_extracts_fenced_code_and_detects_repair_intent():
    context = build_code_repair_context(
        """
Please fix this code so it runs here:

```python
selected_layer = viewer.layers.selection.active
print(selected_layer.type)
```
"""
    )

    assert context is not None
    assert context["intent"] == "repair"
    assert "selected_layer = viewer.layers.selection.active" in context["original_code"]
    assert "normalized_code_candidate" not in context
    assert any("selected_layer.type" in warning for warning in context["local_validation"]["warnings"])


def test_build_code_repair_context_detects_explain_only_intent():
    context = build_code_repair_context(
        """
Why does this code not work?
viewer.add_histogram(data)
"""
    )

    assert context is not None
    assert context["intent"] == "explain"
    assert "viewer.add_histogram(data)" in context["original_code"]
    assert "normalized_code_candidate" not in context


def test_compact_code_repair_user_message_omits_duplicate_pasted_code():
    text = """
Please fix this code and keep the output name:
```python
selected_layer = viewer.layers.selection.active
print(selected_layer.type)
```
"""
    context = build_code_repair_context(text)

    message = compact_code_repair_user_message(text, context)

    assert "selected_layer = viewer.layers.selection.active" not in message
    assert "code_repair_context.original_code" in message
    assert "keep the output name" in message


def test_build_code_repair_context_ignores_regular_non_code_requests():
    assert build_code_repair_context("show all images in grid") is None


def test_build_code_repair_context_ignores_plain_english_workflow_prompt():
    text = (
        "Build a conservative binary mask for the selected image using a built-in workflow. "
        "Inspect the selected image first and describe signal, background, and noise. "
        "Preview threshold before applying it. Decide whether the foreground is brighter or dimmer than the background. "
        "Clean the mask with minimum necessary morphology, measure mask quality after each major step, "
        "refine if the mask is too loose or too strict, preserve faint real structures, and prefer built-in tools over code."
    )

    assert build_code_repair_context(text) is None


def test_build_code_repair_context_includes_layer_binding_hints_for_template_images(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image_a = viewer.add_image(np.asarray([[1, 2], [3, 4]], dtype=np.float32), name="sample_a")
    viewer.add_image(np.asarray([[5, 6], [7, 8]], dtype=np.float32), name="sample_b")
    viewer.layers.selection.active = image_a

    context = build_code_repair_context(
        """
Refine this code:
```python
from napari_chat_assistant.agent.dispatcher import prepare_tool_job
prepared = prepare_tool_job(
    viewer,
    "create_analysis_montage",
    {"layer_names": ["img_a", "img_b"], "rows": 1, "columns": 2},
)
```
""",
        viewer=viewer,
    )

    assert context is not None
    hints = context["layer_binding_hints"]
    assert hints["selected_layer_name"] == "sample_a"
    assert hints["layer_candidates"]["image"] == ["sample_a", "sample_b"]
    placeholder = next(item for item in hints["placeholder_bindings"] if item["argument"] == "layer_names")
    assert placeholder["expected_kind"] == "image"
    assert placeholder["suggested_layers"] == ["sample_a", "sample_b"]


def test_build_code_repair_context_includes_binding_hints_for_roi_placeholders(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.asarray([[1, 2], [3, 4]], dtype=np.float32), name="sample_a")
    roi = viewer.add_shapes([], name="sample_a_intensity_roi")
    viewer.layers.selection.active = roi

    context = build_code_repair_context(
        """
Please refine this code:
```python
prepared = prepare_tool_job(
    viewer,
    "extract_roi_values",
    {"image_layer": "img_a", "roi_layer": "roi_a"},
)
```
""",
        viewer=viewer,
    )

    assert context is not None
    hints = context["layer_binding_hints"]
    assert hints["selected_layer_kind"] == "shapes"
    roi_binding = next(item for item in hints["placeholder_bindings"] if item["argument"] == "roi_layer")
    assert roi_binding["expected_kind"] == "shapes"
    assert roi_binding["suggested_layers"] == ["sample_a_intensity_roi"]
