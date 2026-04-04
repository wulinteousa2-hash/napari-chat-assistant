from __future__ import annotations

import ast
import importlib
import inspect
import re
from dataclasses import dataclass, field
from typing import Literal

import napari


ValidationMode = Literal["strict", "permissive"]


DISALLOWED_IMPORT_TEXT = (
    "ViewerViewerContext",
    "from napari.viewer import ViewerViewerContext",
)


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        return bool(self.errors or self.warnings)

    def has_blocking_issues(self, mode: ValidationMode = "strict") -> bool:
        return bool(self.errors or (mode == "strict" and self.warnings))

    def blocking_messages(self, mode: ValidationMode = "strict") -> list[str]:
        messages = list(self.errors)
        if mode == "strict":
            messages.extend(self.warnings)
        return messages


def normalize_generated_code_if_needed(code_text: str, *, viewer=None) -> tuple[str, ValidationReport]:
    code = str(code_text or "").strip()
    raw_errors = validate_generated_code(code, viewer=viewer)
    if not raw_errors.has_issues():
        return code, raw_errors
    repaired, repair_notes = _repair_generated_code(code)
    normalized = _normalize_escaped_multiline_code(repaired)
    if normalized == code:
        return code, raw_errors
    normalized_errors = validate_generated_code(normalized, viewer=viewer)
    if normalized_errors.errors:
        return code, raw_errors
    if len(normalized_errors.warnings) > len(raw_errors.warnings) and not raw_errors.errors:
        return code, raw_errors
    normalized_errors.notes.extend(repair_notes)
    return normalized, normalized_errors


def build_code_repair_context(user_text: str, *, viewer=None) -> dict | None:
    source = str(user_text or "")
    if not source.strip():
        return None
    code = _extract_code_candidate(source)
    if not code:
        return None
    lowered = " ".join(source.strip().lower().split())
    if not _looks_like_code_help_request(lowered, code):
        return None
    normalized_code, report = normalize_generated_code_if_needed(code, viewer=viewer)
    intent = "repair"
    explain_signals = ("why", "explain", "what is wrong", "why it does not work", "why doesn't it work")
    repair_signals = ("fix", "refine", "repair", "improve", "make it run", "make this run", "debug")
    if any(signal in lowered for signal in explain_signals) and not any(signal in lowered for signal in repair_signals):
        intent = "explain"
    elif not any(signal in lowered for signal in repair_signals):
        intent = "analyze"
    return {
        "intent": intent,
        "original_code": code,
        "normalized_code_candidate": normalized_code,
        "layer_binding_hints": _build_layer_binding_hints(normalized_code, viewer=viewer),
        "local_validation": {
            "errors": list(report.errors),
            "warnings": list(report.warnings),
            "notes": list(report.notes),
        },
    }


def _repair_generated_code(code: str) -> tuple[str, list[str]]:
    text = str(code or "")
    repair_notes: list[str] = []
    repaired = text

    run_in_background_import_pattern = (
        r"(?m)^\s*from\s+napari(?:\.[A-Za-z0-9_]+)*\s+import\s+run_in_background\s*(?:#.*)?\n?"
    )
    repaired_no_bg_import = re.sub(run_in_background_import_pattern, "", repaired)
    if repaired_no_bg_import != repaired:
        repaired = repaired_no_bg_import.lstrip("\n")
        repair_notes.append(
            "Removed `run_in_background` import from `napari...` because the plugin runtime already provides `run_in_background` as a built-in helper."
        )

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


def _extract_code_candidate(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""
    fenced = re.search(r"```(?:python)?\s*(.*?)```", source, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    if _looks_like_python_text(source):
        return source
    marker_patterns = (
        r"(?is)(?:code|python)\s*:\s*(.+)$",
        r"(?is)(?:fix|debug|refine|repair|explain)\s+this\s+code\s*:?\s*(.+)$",
        r"(?is)(?:please\s+)?(?:fix|debug|refine|repair|explain)\s+this\s+code\s*:?\s*(.+)$",
    )
    for pattern in marker_patterns:
        match = re.search(pattern, source)
        if match:
            candidate = str(match.group(1) or "").strip()
            if _looks_like_python_text(candidate):
                return candidate
    return ""


def _looks_like_code_help_request(lowered_text: str, code: str) -> bool:
    if not code.strip():
        return False
    help_signals = (
        "fix",
        "repair",
        "refine",
        "improve",
        "debug",
        "broken",
        "does not work",
        "doesn't work",
        "make it run",
        "make this run",
        "why",
        "explain",
        "error",
        "traceback",
    )
    if any(signal in lowered_text for signal in help_signals):
        return True
    return False


def _looks_like_python_text(text: str) -> bool:
    source = str(text or "").strip()
    if not source:
        return False
    python_signals = (
        "viewer.",
        "selected_layer",
        "import ",
        "from ",
        "def ",
        "for ",
        "if ",
        "while ",
        "np.",
        "napari",
        "run_in_background",
    )
    line_count = len([line for line in source.splitlines() if line.strip()])
    return any(signal in source for signal in python_signals) and line_count >= 1


def _layer_kind(layer) -> str:
    if isinstance(layer, napari.layers.Image):
        return "image"
    if isinstance(layer, napari.layers.Labels):
        return "labels"
    if isinstance(layer, napari.layers.Shapes):
        return "shapes"
    if isinstance(layer, napari.layers.Points):
        return "points"
    return layer.__class__.__name__.lower()


def _semantic_type(layer) -> str:
    try:
        from .profiler import profile_layer

        return str(profile_layer(layer).get("semantic_type", "unknown"))
    except Exception:
        return "unknown"


def _build_layer_binding_hints(code_text: str, *, viewer=None) -> dict:
    layer_records = []
    if viewer is not None:
        for layer in getattr(viewer, "layers", []) or []:
            layer_records.append(
                {
                    "name": str(getattr(layer, "name", "")).strip(),
                    "kind": _layer_kind(layer),
                    "semantic_type": _semantic_type(layer),
                }
            )

    selected_layer = None
    try:
        selected_layer = None if viewer is None else getattr(getattr(viewer.layers, "selection", None), "active", None)
    except Exception:
        selected_layer = None

    by_kind: dict[str, list[str]] = {"image": [], "labels": [], "shapes": [], "points": []}
    for record in layer_records:
        kind = str(record.get("kind", "")).strip()
        name = str(record.get("name", "")).strip()
        if kind in by_kind and name:
            by_kind[kind].append(name)

    return {
        "selected_layer_name": "" if selected_layer is None else str(getattr(selected_layer, "name", "")).strip(),
        "selected_layer_kind": "" if selected_layer is None else _layer_kind(selected_layer),
        "available_layers": layer_records,
        "layer_candidates": by_kind,
        "placeholder_bindings": _extract_code_layer_placeholders(
            code_text,
            selected_layer_name="" if selected_layer is None else str(getattr(selected_layer, "name", "")).strip(),
            selected_layer_kind="" if selected_layer is None else _layer_kind(selected_layer),
            by_kind=by_kind,
        ),
    }


def _extract_code_layer_placeholders(
    code_text: str,
    *,
    selected_layer_name: str,
    selected_layer_kind: str,
    by_kind: dict[str, list[str]],
) -> list[dict]:
    source = str(code_text or "").strip()
    if not source:
        return []
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return []

    key_kind_map = {
        "layer_name": "image",
        "image_layer": "image",
        "source_layer": "image",
        "layer_names": "image",
        "mask_layer": "labels",
        "reference_layer": "labels",
        "intensity_layer": "image",
        "roi_layer": "shapes",
        "points_layer": "points",
        "annotation_layer": "",
        "montage_layer": "image",
    }
    placeholder_signals = {
        "img_a",
        "img_b",
        "img_c",
        "image_a",
        "image_b",
        "image_c",
        "mask_a",
        "mask_b",
        "roi_a",
        "roi_b",
        "shape_a",
        "shape_b",
        "points_a",
        "points_b",
        "analysis_montage",
        "analysis_montage_mask",
        "montage_points",
    }

    bindings: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key_node, value_node in zip(node.keys, node.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            key = str(key_node.value).strip()
            expected_kind = key_kind_map.get(key)
            if expected_kind is None:
                continue
            values: list[str] = []
            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                values = [str(value_node.value).strip()]
            elif isinstance(value_node, (ast.List, ast.Tuple)):
                values = [
                    str(item.value).strip()
                    for item in value_node.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                ]
            if not values:
                continue
            if not any(value in placeholder_signals for value in values):
                continue
            suggested = list(by_kind.get(expected_kind, [])) if expected_kind else []
            if expected_kind and selected_layer_kind == expected_kind and selected_layer_name:
                suggested = [selected_layer_name] + [name for name in suggested if name != selected_layer_name]
            bindings.append(
                {
                    "argument": key,
                    "placeholder_values": values,
                    "expected_kind": expected_kind,
                    "suggested_layers": suggested,
                }
            )
    return bindings


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


def validate_generated_code(code_text: str, *, viewer=None) -> ValidationReport:
    code = str(code_text or "").strip()
    report = ValidationReport()
    if not code:
        report.errors.append("Generated code is empty.")
        return report
    if code.startswith("```") or "```" in code:
        report.errors.append("Generated code must not contain Markdown code fences.")
    if '"action"' in code and '"code"' in code and "{" in code and "}" in code:
        report.errors.append("Generated code appears to contain JSON instead of pure Python.")
    if any(token in code for token in DISALLOWED_IMPORT_TEXT):
        report.errors.append("Generated code uses an unsupported napari API import.")
    if "equalize_adap_hist" in code:
        report.warnings.append(
            "Generated code uses the wrong scikit-image CLAHE function name [equalize_adap_hist]. "
            "Use [skimage.exposure.equalize_adapthist] instead."
        )

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        line = f"line {exc.lineno}" if exc.lineno else "unknown line"
        report.errors.append(f"Generated code is not valid Python: {exc.msg} ({line}).")
        return report

    _validate_ast(tree, viewer=viewer, report=report)
    return report


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


def _validate_ast(tree: ast.AST, *, viewer=None, report: ValidationReport) -> None:
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
            report.errors.append(error)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module == "napari.viewer" and alias.name == "ViewerViewerContext":
                    report.errors.append("Do not import ViewerViewerContext from napari.viewer.")
                if module == "napari" and alias.name == "Viewer":
                    report.errors.append("Do not import Viewer from napari. Use the existing [viewer] object provided by the plugin.")
                import_error = _validate_import_from_symbol(module, alias.name)
                if import_error:
                    report.warnings.append(import_error)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ViewerViewerContext":
                    report.errors.append("Do not import ViewerViewerContext.")
                import_error = _validate_import_module(alias.name)
                if import_error:
                    report.warnings.append(import_error)
        if isinstance(node, ast.Assign):
            viewer_assignment_error = _validate_viewer_assignment(node)
            if viewer_assignment_error:
                report.errors.append(viewer_assignment_error)
        if isinstance(node, ast.Call):
            viewer_creation_error = _validate_new_viewer_call(node)
            if viewer_creation_error:
                report.errors.append(viewer_creation_error)
            run_loop_error = _validate_napari_run_call(node)
            if run_loop_error:
                report.errors.append(run_loop_error)
            viewer_error = _validate_viewer_call(node, viewer)
            if viewer_error:
                report.warnings.append(viewer_error)
            stats_error = _validate_statistical_coordinate_misuse(node, stat_value_names, histogram_axes_names)
            if stats_error:
                report.warnings.append(stats_error)
        if isinstance(node, ast.Attribute):
            layer_attr_error = _validate_layer_attribute_access(node)
            if layer_attr_error:
                report.warnings.append(layer_attr_error)


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


def _validate_layer_attribute_access(node: ast.Attribute) -> str | None:
    attr_name = str(node.attr or "").strip()
    if attr_name not in {"_type", "type"}:
        return None
    if isinstance(node.value, ast.Name):
        target_name = node.value.id
    else:
        target_name = "layer"
    return (
        f"Invalid napari layer attribute access: [{target_name}.{attr_name}] is not a supported napari API. "
        "Use isinstance(..., napari.layers.Shapes/Labels/Image/Points) or inspect the layer class instead."
    )


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


def _validate_napari_run_call(node: ast.Call) -> str | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "napari":
        return None
    if func.attr != "run":
        return None
    return "Do not call [napari.run()] inside plugin-executed code. Use the existing napari event loop."


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
