from __future__ import annotations

import builtins
import html
import io
import keyword
import re
import tokenize


_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_ORDERED_LIST_RE = re.compile(r"(\d+)\.\s+(.*)")
_PYTHON_BUILTINS = {name for name in dir(builtins) if not name.startswith("_")}
_PYTHON_KEYWORDS = set(keyword.kwlist)
_PYTHON_SOFT_KEYWORDS = {"match", "case"}


def render_user_message_html(message: str) -> str:
    return _render_plain_text_html(message)


def render_assistant_message_html(message: str) -> str:
    source = str(message or "").replace("\r\n", "\n").strip()
    if not source:
        return "<p></p>"

    segments: list[tuple[str, str, str]] = []

    def store_fence(match: re.Match[str]) -> str:
        language = match.group(1).strip().lower()
        code = match.group(2).strip("\n")
        token = f"__CODE_BLOCK_{len(segments)}__"
        segments.append(("code", language, code))
        return token

    protected = _FENCE_RE.sub(store_fence, source)

    blocks: list[str] = []
    lines = protected.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("__CODE_BLOCK_") and stripped.endswith("__"):
            blocks.append(_restore_code_block(stripped, segments))
            index += 1
            continue
        if _is_bullet_line(stripped):
            items: list[str] = []
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate or not _is_bullet_line(candidate):
                    break
                items.append(candidate[2:].strip())
                index += 1
            items_html = "".join(f"<li>{_render_inline_html(item)}</li>" for item in items)
            blocks.append(f"<ul>{items_html}</ul>")
            continue
        if _is_ordered_list_line(stripped):
            items = []
            while index < len(lines):
                candidate = lines[index].strip()
                match = _ORDERED_LIST_RE.fullmatch(candidate)
                if not candidate or not match:
                    break
                items.append(match.group(2).strip())
                index += 1
            items_html = "".join(f"<li>{_render_inline_html(item)}</li>" for item in items)
            blocks.append(f"<ol>{items_html}</ol>")
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if candidate.startswith("__CODE_BLOCK_") and candidate.endswith("__"):
                break
            if _is_bullet_line(candidate):
                break
            if _is_ordered_list_line(candidate):
                break
            paragraph_lines.append(candidate)
            index += 1
        blocks.append("<p>" + "<br>".join(_render_inline_html(line) for line in paragraph_lines) + "</p>")

    return "".join(blocks) or "<p></p>"


def _restore_code_block(token: str, segments: list[tuple[str, str, str]]) -> str:
    match = re.fullmatch(r"__CODE_BLOCK_(\d+)__", token)
    if not match:
        return f"<p>{_render_inline_html(token)}</p>"
    _, language, code = segments[int(match.group(1))]
    label = html.escape(language) if language else "text"
    rendered_code = _render_code_html(language, code)
    return (
        '<div style="background: #0f1b2d; margin: 12px 0 14px 0; border: 1px solid #2b2d30; border-radius: 10px; overflow: hidden; '
        'box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);">'
        '<div style="background: #252526; color: #cccccc; font-size: 11px; '
        'padding: 7px 12px; letter-spacing: 0.06em; line-height: 1.35; text-transform: uppercase; '
        'border-bottom: 1px solid #2b2d30; font-family: \'Segoe UI\', \'Noto Sans\', sans-serif;">'
        f"{label}</div>"
        '<pre style="margin: 0; padding: 16px 14px; background: #1e1e1e; color: #d4d4d4; line-height: 1.55; '
        'white-space: pre-wrap; font-family: \'Cascadia Code\', \'Fira Code\', \'Consolas\', monospace; '
        'font-size: 13px; tab-size: 4;">'
        f"<code>{rendered_code}</code></pre></div>"
    )


def _render_plain_text_html(message: str) -> str:
    return html.escape(str(message or "")).replace("\n", "<br>")


def _render_inline_html(text: str) -> str:
    escaped = html.escape(str(text or ""))
    escaped = _LINK_RE.sub(lambda m: f'<a href="{m.group(2)}" style="color: #4fc1ff;">{m.group(1)}</a>', escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = _INLINE_CODE_RE.sub(
        r'<code style="background: #2d2d2d; color: #dcdcaa; padding: 2px 5px; border-radius: 4px; '
        r'border: 1px solid #3a3d41; font-family: \'Cascadia Code\', \'Fira Code\', \'Consolas\', monospace; '
        r'font-size: 0.95em;">\1</code>',
        escaped,
    )
    return escaped


def _render_code_html(language: str, code: str) -> str:
    lang = (language or "").strip().lower()
    if lang in {"py", "python"}:
        return _render_python_code_html(code)
    return html.escape(code)


def _render_python_code_html(code: str) -> str:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except Exception:
        return html.escape(code)

    lines = code.splitlines(keepends=True)
    output: list[str] = []
    last_row = 1
    last_col = 0
    expect_def_name = False
    expect_class_name = False

    for token in tokens:
        token_type = token.type
        token_text = token.string
        start_row, start_col = token.start
        end_row, end_col = token.end

        if token_type == tokenize.ENDMARKER:
            break

        output.append(_escaped_segment_between(lines, last_row, last_col, start_row, start_col))
        output.append(_style_python_token(token_type, token_text, expect_def_name, expect_class_name))

        if token_type == tokenize.NAME:
            expect_def_name = token_text == "def"
            expect_class_name = token_text == "class"
        else:
            if token_type not in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}:
                expect_def_name = False
                expect_class_name = False

        last_row, last_col = end_row, end_col

    output.append(_escaped_segment_between(lines, last_row, last_col, len(lines) + 1, 0))
    return "".join(output)


def _escaped_segment_between(
    lines: list[str],
    start_row: int,
    start_col: int,
    end_row: int,
    end_col: int,
) -> str:
    if not lines:
        return ""
    if start_row == end_row:
        if 1 <= start_row <= len(lines):
            return html.escape(lines[start_row - 1][start_col:end_col])
        return ""

    parts: list[str] = []
    if 1 <= start_row <= len(lines):
        parts.append(lines[start_row - 1][start_col:])
    for row in range(start_row + 1, end_row):
        if 1 <= row <= len(lines):
            parts.append(lines[row - 1])
    if 1 <= end_row <= len(lines):
        parts.append(lines[end_row - 1][:end_col])
    return html.escape("".join(parts))


def _style_python_token(
    token_type: int,
    token_text: str,
    expect_def_name: bool,
    expect_class_name: bool,
) -> str:
    escaped = html.escape(token_text)
    color = ""
    extra = ""

    if token_type == tokenize.COMMENT:
        color = "#6a9955"
    elif token_type == tokenize.STRING:
        color = "#ce9178"
    elif token_type == tokenize.NUMBER:
        color = "#b5cea8"
    elif token_type == tokenize.NAME:
        if expect_def_name:
            color = "#dcdcaa"
        elif expect_class_name:
            color = "#4ec9b0"
        elif token_text in _PYTHON_KEYWORDS or token_text in _PYTHON_SOFT_KEYWORDS:
            color = "#c586c0"
        elif token_text in {"True", "False", "None"}:
            color = "#569cd6"
        elif token_text in _PYTHON_BUILTINS:
            color = "#dcdcaa"
    elif token_type == tokenize.OP:
        color = "#d4d4d4"

    if token_text in {"def", "class", "return", "import", "from", "if", "elif", "else", "for", "while", "try", "except", "with"}:
        extra = "font-weight: 600;"

    if not color and not extra:
        return escaped
    return f'<span style="color: {color}; {extra}">{escaped}</span>'


def _is_bullet_line(text: str) -> bool:
    return text.startswith("- ") or text.startswith("* ")


def _is_ordered_list_line(text: str) -> bool:
    return _ORDERED_LIST_RE.fullmatch(text) is not None
