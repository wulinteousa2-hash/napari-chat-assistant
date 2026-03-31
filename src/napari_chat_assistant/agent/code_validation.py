from __future__ import annotations

import ast
import importlib
import inspect
import re


DISALLOWED_IMPORT_TEXT = (
    "ViewerViewerContext",
    "from napari.viewer import ViewerViewerContext",
)


def normalize_generated_code_if_needed(code_text: str, *, viewer=None) -> tuple[str, list[str], list[str]]:
    code = str(code_text or "").strip()
    raw_errors = validate_generated_code(code, viewer=viewer)
    if not raw_errors:
        return code, [], []
    repaired, repair_notes = _repair_generated_code(code)
    normalized = _normalize_escaped_multiline_code(repaired)
    if normalized == code:
        return code, raw_errors, []
    normalized_errors = validate_generated_code(normalized, viewer=viewer)
    if normalized_errors:
        return code, raw_errors, []
    return normalized, [], repair_notes


def _repair_generated_code(code: str) -> tuple[str, list[str]]:
    text = str(code or "")
    repair_notes: list[str] = []
    repaired = text

    selected_layer_lookup_pattern = r"\bviewer\.layers\[\s*selected_layer\s*\]"
    selected_layer_lookup_repaired = re.sub(selected_layer_lookup_pattern, "selected_layer", repaired)
    if selected_layer_lookup_repaired != repaired:
        repaired = selected_layer_lookup_repaired
        repair_notes.append(
            "Replaced `viewer.layers[selected_layer]` with `selected_layer` because `selected_layer` is already a napari layer object."
        )

    repaired, data_access_notes = _repair_layer_data_access(repaired)
    repair_notes.extend(data_access_notes)

    return repaired, repair_notes


def _repair_layer_data_access(code: str) -> tuple[str, list[str]]:
    text = str(code or "")
    repair_notes: list[str] = []
    repaired = text

    if not re.search(r"(?m)^\s*layer\s*=\s*selected_layer\s*$", repaired):
        return repaired, repair_notes

    data_attr_pattern = r"\blayer\.(shape|dtype|ndim)\b"
    if not re.search(data_attr_pattern, repaired):
        return repaired, repair_notes

    if not re.search(r"(?m)^\s*data\s*=\s*np\.asarray\(layer\.data\)\s*$", repaired):
        layer_assign_pattern = r"(?m)^(?P<indent>\s*)layer\s*=\s*selected_layer\s*$"

        def insert_data_assignment(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return f"{match.group(0)}\n{indent}data = np.asarray(layer.data)"

        repaired = re.sub(layer_assign_pattern, insert_data_assignment, repaired, count=1)

    rewritten = re.sub(data_attr_pattern, lambda m: f"data.{m.group(1)}", repaired)
    if rewritten != repaired:
        repaired = rewritten
        repair_notes.append(
            "Replaced direct napari layer data checks like `layer.shape` and `layer.dtype` with `data = np.asarray(layer.data)` and `data.*` access."
        )

    return repaired, repair_notes


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
    if "equalize_adap_hist" in code:
        errors.append(
            "Generated code uses the wrong scikit-image CLAHE function name [equalize_adap_hist]. "
            "Use [skimage.exposure.equalize_adapthist] instead."
        )

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        line = f"line {exc.lineno}" if exc.lineno else "unknown line"
        errors.append(f"Generated code is not valid Python: {exc.msg} ({line}).")
        return errors

    errors.extend(_validate_ast(tree, viewer=viewer))
    return errors


def _normalize_escaped_multiline_code(code: str) -> str:
    text = str(code or "").strip()
    if not text:
        return text
    if "\n" in text:
        real_newlines = text.count("\n")
    else:
        real_newlines = 0
    escaped_newlines = text.count("\\n")
    if escaped_newlines < 2 or real_newlines > 1:
        return text
    if not _looks_like_escaped_python(text):
        return text
    normalized = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")
    normalized = re.sub(r'(["\'])\n([ \t]+)(["\'])', r"\1\\n\2\3", normalized)
    return normalized.strip()


def _looks_like_escaped_python(text: str) -> bool:
    signals = (
        "if ",
        "for ",
        "while ",
        "def ",
        "class ",
        "import ",
        "from ",
        "return ",
        "raise ",
        "viewer.",
        "selected_layer",
    )
    return any(signal in text for signal in signals)


def _validate_ast(tree: ast.AST, *, viewer=None) -> list[str]:
    errors: list[str] = []
    uint8_arrays: set[str] = set()
    unsafe_int_arrays: set[str] = set()
    stat_value_names: set[str] = set()
    histogram_axes_names: set[str] = set()

    for node in getattr(tree, "body", []):
        _track_assignment_state(node, uint8_arrays, unsafe_int_arrays)
        _track_statistical_values(node, stat_value_names)
        _track_histogram_axes(node, histogram_axes_names)
        error = _detect_unsafe_uint8_augassign(node, uint8_arrays, unsafe_int_arrays)
        if error:
            errors.append(error)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module == "napari.viewer" and alias.name == "ViewerViewerContext":
                    errors.append("Do not import ViewerViewerContext from napari.viewer.")
                if module == "napari" and alias.name == "Viewer":
                    errors.append("Do not import Viewer from napari. Use the existing [viewer] object provided by the plugin.")
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
        if isinstance(node, ast.Assign):
            viewer_assignment_error = _validate_viewer_assignment(node)
            if viewer_assignment_error:
                errors.append(viewer_assignment_error)
        if isinstance(node, ast.Call):
            viewer_creation_error = _validate_new_viewer_call(node)
            if viewer_creation_error:
                errors.append(viewer_creation_error)
            viewer_error = _validate_viewer_call(node, viewer)
            if viewer_error:
                errors.append(viewer_error)
            stats_error = _validate_statistical_coordinate_misuse(node, stat_value_names, histogram_axes_names)
            if stats_error:
                errors.append(stats_error)
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
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "viewer":
        return None
    method_name = str(func.attr or "").strip()
    if not method_name:
        return None
    keyword_error = _validate_viewer_keywords(method_name, node, viewer)
    if keyword_error:
        return keyword_error
    if viewer is None:
        return None
    if not hasattr(viewer, method_name):
        return f"Unsupported viewer API: [viewer.{method_name}] is not available on the current napari viewer."
    return None


def _validate_new_viewer_call(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "Viewer":
        return "Do not create a new napari Viewer. Use the existing [viewer] object provided by the plugin."
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name) and func.value.id == "napari" and func.attr == "Viewer":
            return "Do not create a new napari Viewer. Use the existing [viewer] object provided by the plugin."
    return None


def _validate_viewer_assignment(node: ast.Assign) -> str | None:
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name) or target.id != "viewer":
        return None
    value = node.value
    if not isinstance(value, ast.Call):
        return None
    return _validate_new_viewer_call(value) or "Do not overwrite the provided [viewer] object."


def _validate_viewer_keywords(method_name: str, node: ast.Call, viewer) -> str | None:
    keywords = [kw.arg for kw in node.keywords if kw.arg]
    if method_name == "add_shapes" and "shape" in keywords:
        return "Invalid napari API: [viewer.add_shapes] does not accept keyword [shape]. Use [data] and [shape_type] instead."
    if viewer is None or not hasattr(viewer, method_name):
        return None
    try:
        signature = inspect.signature(getattr(viewer, method_name))
    except Exception:
        return None
    accepts_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_var_keyword:
        return None
    valid_keywords = {
        name
        for name, param in signature.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    for keyword in keywords:
        if keyword not in valid_keywords:
            return (
                f"Invalid viewer keyword: [viewer.{method_name}] does not accept keyword [{keyword}] "
                "on the current napari viewer."
            )
    return None


def _track_statistical_values(node: ast.stmt, stat_value_names: set[str]) -> None:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return
    if _is_statistical_value_expr(node.value):
        stat_value_names.add(target.id)


def _track_histogram_axes(node: ast.stmt, histogram_axes_names: set[str]) -> None:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return
    target = node.targets[0]
    if not isinstance(target, ast.Tuple):
        return
    if not isinstance(node.value, ast.Call):
        return
    func = node.value.func
    if not isinstance(func, ast.Attribute):
        return
    if not isinstance(func.value, ast.Name) or func.value.id != "plt":
        return
    if func.attr != "subplots":
        return
    for elt in target.elts[1:]:
        if isinstance(elt, ast.Name):
            histogram_axes_names.add(elt.id)


def _is_statistical_value_expr(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    return func.attr in {"mean", "std", "median", "var", "min", "max"}


def _validate_statistical_coordinate_misuse(
    node: ast.Call,
    stat_value_names: set[str],
    histogram_axes_names: set[str],
) -> str | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if isinstance(func.value, ast.Name) and func.value.id == "viewer" and func.attr == "add_shapes":
        for keyword in node.keywords:
            if keyword.arg == "data":
                name = _find_statistical_name_in_expr(keyword.value, stat_value_names)
                if name:
                    return (
                        f"Statistical value [{name}] looks like an intensity-domain quantity, not a spatial coordinate. "
                        "Do not use image statistics directly in napari shape coordinates."
                    )
        for arg in node.args:
            name = _find_statistical_name_in_expr(arg, stat_value_names)
            if name:
                return (
                    f"Statistical value [{name}] looks like an intensity-domain quantity, not a spatial coordinate. "
                    "Do not use image statistics directly in napari shape coordinates."
                )
    if (
        isinstance(func.value, ast.Name)
        and func.value.id in histogram_axes_names
        and func.attr in {"axhline", "hlines"}
    ):
        stat_name = None
        if node.args:
            stat_name = _find_statistical_name_in_expr(node.args[0], stat_value_names)
        if stat_name is None:
            for keyword in node.keywords:
                if keyword.arg in {"y", "ymin", "ymax"}:
                    stat_name = _find_statistical_name_in_expr(keyword.value, stat_value_names)
                    if stat_name:
                        break
        if stat_name:
            return (
                f"Histogram annotation misuse: statistical value [{stat_name}] is usually an intensity-axis quantity. "
                "On a standard intensity histogram, use a vertical guide line instead of axhline/hlines."
            )
    return None


def _find_statistical_name_in_expr(node: ast.AST, stat_value_names: set[str]) -> str | None:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in stat_value_names:
            return child.id
    return None
