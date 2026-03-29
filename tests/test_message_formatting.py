from __future__ import annotations

from napari_chat_assistant.widgets.message_formatting import render_assistant_message_html, render_user_message_html


def test_render_user_message_html_keeps_plain_text_escaped():
    rendered = render_user_message_html("hello\n<script>alert(1)</script>")

    assert rendered == "hello<br>&lt;script&gt;alert(1)&lt;/script&gt;"


def test_render_assistant_message_html_supports_basic_markdown_subset():
    rendered = render_assistant_message_html(
        "**Plan**\n"
        "- use `threshold`\n"
        "- run [docs](https://example.com)\n\n"
        "1. review the image\n"
        "2. apply threshold\n\n"
        "```bash\n"
        "echo hi\n"
        "```"
    )

    assert "<strong>Plan</strong>" in rendered
    assert "<ul>" in rendered
    assert "<ol>" in rendered
    assert "<li>use <code" in rendered
    assert '<a href="https://example.com"' in rendered
    assert '<div style="background: #0f1b2d;' in rendered
    assert "<pre" in rendered
    assert "<code>echo hi</code>" in rendered
    assert "bash" in rendered


def test_render_assistant_message_html_escapes_html_inside_markdown_content():
    rendered = render_assistant_message_html(
        "Use `<tag>` safely.\n\n```python\nprint('<unsafe>')\n```"
    )

    assert "&lt;tag&gt;" in rendered
    assert "&lt;unsafe&gt;" in rendered
    assert "<tag>" not in rendered
    assert "<unsafe>" not in rendered


def test_render_assistant_message_html_joins_paragraph_lines_with_breaks():
    rendered = render_assistant_message_html("first line\nsecond line")

    assert "<p>first line<br>second line</p>" == rendered
