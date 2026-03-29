from __future__ import annotations

import ast
import importlib


DISALLOWED_IMPORT_TEXT = (
    "ViewerViewerContext",
    "from napari.viewer import ViewerViewerContext",
)


def validate_generated_code(code_text: str, *, viewer=None) -> list[str]:
    code = str(code_text or "").strip()
    errors: list[str] = []
    if not code:
        return ["Generated code is empty."]
    if code.startswith("```") or "```" in code:
        errors.append("Generated code must not contain Markdown code fences.")
    if '"action"' in code and '"code"' in code and "{" in code and "}" in code:
        errors.append("Generated code appears to contain JSON instead of pure Python.")
    if any(token in code for token in DISALLOWED_IMPORT_TEXT):
        errors.append("Generated code uses an unsupported napari API import.")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        line = f"line {exc.lineno}" if exc.lineno else "unknown line"
        errors.append(f"Generated code is not valid Python: {exc.msg} ({line}).")
        return errors

    errors.extend(_validate_ast(tree, viewer=viewer))
    return errors


def _validate_ast(tree: ast.AST, *, viewer=None) -> list[str]:
    errors: list[str] = []
    uint8_arrays: set[str] = set()
    unsafe_int_arrays: set[str] = set()

    for node in getattr(tree, "body", []):
        _track_assignment_state(node, uint8_arrays, unsafe_int_arrays)
        error = _detect_unsafe_uint8_augassign(node, uint8_arrays, unsafe_int_arrays)
        if error:
            errors.append(error)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module == "napari.viewer" and alias.name == "ViewerViewerContext":
                    errors.append("Do not import ViewerViewerContext from napari.viewer.")
                import_error = _validate_import_from_symbol(module, alias.name)
                if import_error:
                    errors.append(import_error)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ViewerViewerContext":
                    errors.append("Do not import ViewerViewerContext.")
                import_error = _validate_import_module(alias.name)
                if import_error:
                    errors.append(import_error)
        if isinstance(node, ast.Call):
            viewer_error = _validate_viewer_call(node, viewer)
            if viewer_error:
                errors.append(viewer_error)
    return errors


def _track_assignment_state(node: ast.stmt, uint8_arrays: set[str], unsafe_int_arrays: set[str]) -> None:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return
    name = target.id
    if _is_uint8_array_constructor(node.value):
        uint8_arrays.add(name)
        unsafe_int_arrays.discard(name)
        return
    if _is_random_int_array_without_dtype(node.value):
        unsafe_int_arrays.add(name)
        return
    unsafe_int_arrays.discard(name)


def _detect_unsafe_uint8_augassign(node: ast.stmt, uint8_arrays: set[str], unsafe_int_arrays: set[str]) -> str | None:
    if not isinstance(node, ast.AugAssign):
        return None
    if not isinstance(node.target, ast.Name):
        return None
    if node.target.id not in uint8_arrays:
        return None
    if not isinstance(node.op, (ast.Add, ast.Sub)):
        return None
    value = node.value
    if _is_random_int_array_without_dtype(value):
        return (
            f"Unsafe in-place arithmetic on uint8 array [{node.target.id}] with np.random.randint(). "
            "Use dtype=np.uint8 for the noise or cast/clip before assignment."
        )
    if isinstance(value, ast.Name) and value.id in unsafe_int_arrays:
        return (
            f"Unsafe in-place arithmetic on uint8 array [{node.target.id}] using integer noise array "
            f"[{value.id}] with implicit int64 dtype. Cast or clip before assignment."
        )
    return None


def _is_uint8_array_constructor(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    if not isinstance(node.func.value, ast.Name) or node.func.value.id not in {"np", "numpy"}:
        return False
    if node.func.attr not in {"zeros", "ones", "full", "empty"}:
        return False
    return any(kw.arg == "dtype" and _is_uint8_expr(kw.value) for kw in node.keywords)


def _is_random_int_array_without_dtype(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if not isinstance(node.func, ast.Attribute):
        return False
    rand_parent = node.func.value
    if not isinstance(rand_parent, ast.Attribute):
        return False
    if not isinstance(rand_parent.value, ast.Name) or rand_parent.value.id not in {"np", "numpy"}:
        return False
    if rand_parent.attr != "random" or node.func.attr != "randint":
        return False
    return not any(kw.arg == "dtype" for kw in node.keywords)


def _is_uint8_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id in {"np", "numpy"} and node.attr == "uint8"
    if isinstance(node, ast.Name):
        return node.id == "uint8"
    return False


def _validate_import_module(module_name: str) -> str | None:
    module = str(module_name or "").strip()
    if not module.startswith("napari"):
        return None
    try:
        importlib.import_module(module)
    except Exception:
        return f"Unsupported napari import: module [{module}] is not available in the local environment."
    return None


def _validate_import_from_symbol(module_name: str, symbol_name: str) -> str | None:
    module = str(module_name or "").strip()
    symbol = str(symbol_name or "").strip()
    if not module.startswith("napari") or not symbol:
        return None
    try:
        imported = importlib.import_module(module)
    except Exception:
        return f"Unsupported napari import: module [{module}] is not available in the local environment."
    if not hasattr(imported, symbol):
        return f"Unsupported napari import: [{symbol}] was not found in [{module}] in the local environment."
    return None


def _validate_viewer_call(node: ast.Call, viewer) -> str | None:
    if viewer is None:
        return None
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "viewer":
        return None
    method_name = str(func.attr or "").strip()
    if not method_name:
        return None
    if not hasattr(viewer, method_name):
        return f"Unsupported viewer API: [viewer.{method_name}] is not available on the current napari viewer."
    return None
