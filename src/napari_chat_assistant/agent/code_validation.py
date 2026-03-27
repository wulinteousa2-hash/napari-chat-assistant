from __future__ import annotations

import ast


DISALLOWED_IMPORT_TEXT = (
    "ViewerViewerContext",
    "from napari.viewer import ViewerViewerContext",
)


def validate_generated_code(code_text: str) -> list[str]:
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

    errors.extend(_validate_ast(tree))
    return errors


def _validate_ast(tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module == "napari.viewer" and alias.name == "ViewerViewerContext":
                    errors.append("Do not import ViewerViewerContext from napari.viewer.")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ViewerViewerContext":
                    errors.append("Do not import ViewerViewerContext.")
    return errors
