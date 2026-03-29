from __future__ import annotations

import html
import re


_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_ORDERED_LIST_RE = re.compile(r"(\d+)\.\s+(.*)")


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
        blocks.append('<p style="margin: 0 0 10px 0;">' + "<br>".join(_render_inline_html(line) for line in paragraph_lines) + "</p>")

    return "".join(blocks) or "<p></p>"


def _restore_code_block(token: str, segments: list[tuple[str, str, str]]) -> str:
    match = re.fullmatch(r"__CODE_BLOCK_(\d+)__", token)
    if not match:
        return f"<p>{_render_inline_html(token)}</p>"
    _, language, code = segments[int(match.group(1))]
    label = html.escape(language) if language else "text"
    escaped = html.escape(code)
    return (
        '<div style="margin: 10px 0 12px 0; border: 1px solid #294060; border-radius: 8px; overflow: hidden;">'
        '<div style="background: #0f1b2d; color: #93c5fd; font-size: 11px; '
        'padding: 4px 10px; letter-spacing: 0.08em; text-transform: uppercase;">'
        f"{label}</div>"
        '<pre style="margin: 0; padding: 12px; background: #08111f; color: #e5eefc; '
        'white-space: pre-wrap; font-family: monospace;">'
        f"<code>{escaped}</code></pre></div>"
    )


def _render_plain_text_html(message: str) -> str:
    return html.escape(str(message or "")).replace("\n", "<br>")


def _render_inline_html(text: str) -> str:
    escaped = html.escape(str(text or ""))
    escaped = _LINK_RE.sub(lambda m: f'<a href="{m.group(2)}" style="color: #93c5fd;">{m.group(1)}</a>', escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = _INLINE_CODE_RE.sub(
        r'<code style="background: #162033; color: #f8fafc; padding: 1px 4px; border-radius: 4px; font-family: monospace;">\1</code>',
        escaped,
    )
    return escaped


def _is_bullet_line(text: str) -> bool:
    return text.startswith("- ") or text.startswith("* ")


def _is_ordered_list_line(text: str) -> bool:
    return _ORDERED_LIST_RE.fullmatch(text) is not None
