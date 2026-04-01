from __future__ import annotations

from types import SimpleNamespace

from napari_chat_assistant.agent.code_validation import validate_generated_code


def test_validate_generated_code_rejects_uint8_inplace_randint_add():
    errors = validate_generated_code(
        """
import numpy as np
image = np.zeros((32, 32), dtype=np.uint8)
image += np.random.randint(0, 20, size=image.shape)
"""
    )

    assert any("Unsafe in-place arithmetic on uint8 array [image]" in error for error in errors)


def test_validate_generated_code_rejects_uint8_inplace_add_from_noise_variable():
    errors = validate_generated_code(
        """
import numpy as np
image = np.zeros((32, 32), dtype=np.uint8)
noise = np.random.randint(0, 20, size=image.shape)
image += noise
"""
    )

    assert any("using integer noise array [noise]" in error for error in errors)


def test_validate_generated_code_allows_uint8_randint_when_dtype_is_explicit():
    errors = validate_generated_code(
        """
import numpy as np
image = np.zeros((32, 32), dtype=np.uint8)
noise = np.random.randint(0, 20, size=image.shape, dtype=np.uint8)
image = np.clip(image + noise, 0, 255).astype(np.uint8)
"""
    )

    assert errors == []


def test_validate_generated_code_rejects_missing_napari_symbol_import():
    errors = validate_generated_code("from napari.utils import status\nprint(status)")

    assert any("Unsupported napari import: [status] was not found in [napari.utils]" in error for error in errors)


def test_validate_generated_code_rejects_missing_viewer_method():
    viewer = SimpleNamespace(add_image=lambda *args, **kwargs: None)

    errors = validate_generated_code("viewer.add_histogram(data)", viewer=viewer)

    assert any("Unsupported viewer API: [viewer.add_histogram]" in error for error in errors)


def test_validate_generated_code_allows_existing_viewer_method():
    viewer = SimpleNamespace(add_image=lambda *args, **kwargs: None)

    errors = validate_generated_code("viewer.add_image(data)", viewer=viewer)

    assert errors == []


def test_validate_generated_code_rejects_invalid_layer_type_attribute_check():
    errors = validate_generated_code(
        """
selected_layer = viewer.layers.selection.active
if selected_layer is not None and selected_layer._type == "shapes":
    print(selected_layer.name)
"""
    )

    assert any("selected_layer._type" in error for error in errors)


def test_validate_generated_code_rejects_invalid_layer_type_property_check():
    errors = validate_generated_code(
        """
selected_layer = viewer.layers.selection.active
if selected_layer is not None and selected_layer.type == "image":
    print(selected_layer.name)
"""
    )

    assert any("selected_layer.type" in error for error in errors)
