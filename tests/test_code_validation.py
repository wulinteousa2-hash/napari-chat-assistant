from __future__ import annotations

from types import SimpleNamespace

from napari_chat_assistant.agent.code_validation import build_code_repair_context, validate_generated_code


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
    assert not report.has_blocking_issues("permissive")


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


def test_build_code_repair_context_ignores_regular_non_code_requests():
    assert build_code_repair_context("show all images in grid") is None
