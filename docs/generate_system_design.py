"""Regenerate the architecture poster; SVG uses only the Python standard library.

Run from any directory: python docs/generate_system_design.py
With qtpy and a Qt binding installed, also writes PNG and PDF.
"""

from html import escape
from pathlib import Path


OUT = Path(__file__).resolve().parent
W, H = 2400, 2080
BG, PANEL, INK = "#171d1b", "#1c2421", "#edf2e8"
GREEN, GOLD, BLUE, RED = "#9dd8b5", "#e8c76d", "#9bcadf", "#e49c8d"
parts = []


def text(x, y, value, size=20, color=INK, weight="normal"):
    parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
                 f'font-weight="{weight}">{escape(value)}</text>')


def rect(x, y, w, h, color=INK, fill=PANEL, dashed=False):
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
                 f'fill="{fill}" stroke="{color}" stroke-width="1.6"{dash}/>')


def arrow(x1, y1, x2, y2, color=GREEN, dashed=False):
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" '
                 f'stroke-width="2" marker-end="url(#{color[1:]})"{dash}/>')


def box(x, y, w, title, lines, color=GREEN, h=112):
    rect(x, y, w, h, color)
    text(x + 15, y + 28, title, 21, color, "bold")
    for i, line in enumerate(lines):
        text(x + 15, y + 54 + i * 23, line, 18)


def panel(col, row, letter, title, subtitle, color=GREEN):
    x, y = 28 + col * 790, 405 + row * 400
    rect(x, y, 764, 376, color, dashed=True)
    text(x + 18, y + 32, f"{letter} / {title}", 23, color, "bold")
    text(x + 18, y + 59, subtitle, 17, "#b5c3b9")
    return x + 18, y + 78


def note(x, y, lines, color=INK):
    for i, line in enumerate(lines):
        text(x, y + i * 25, line, 18, color)


def draw():
    parts.clear()
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                 f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
    parts.append('<title id="title">napari-chat-assistant system design</title>'
                 '<desc id="desc">Architecture poster with request routing, local model, '
                 'tool execution, viewer data, code review, storage, voice, SAM2 and diagnostics.</desc>')
    parts.append('<defs>')
    for color in [GREEN, GOLD, BLUE, RED]:
        parts.append(f'<marker id="{color[1:]}" markerWidth="9" markerHeight="9" '
                     f'refX="8" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="{color}"/></marker>')
    parts.append('</defs>')
    parts.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    parts.append('<g font-family="DejaVu Sans, sans-serif">')
    text(30, 57, "napari-chat-assistant / System Design", 44, INK, "bold")
    text(32, 92, "Local imaging workbench · live viewer context · deterministic actions · Ollama assistance", 23, GREEN)
    text(32, 124, "Implementation map | v2.3.4 | committed source at 2ed1876 | 10 November 2026", 18, "#b5c3b9")
    arrow(1530, 118, 1585, 118)
    text(1598, 124, "request / result", 17, GREEN)
    arrow(1795, 118, 1850, 118, GOLD, True)
    text(1863, 124, "optional / fallback", 17, GOLD)
    text(2100, 124, "red = boundary", 17, RED)

    # The overview gives explicit branches instead of a fictional linear pipeline.
    box(32, 169, 355, "1  User + live viewer", ["Prompt, code, action, shortcut", "Layer names, selection, ROI"], h=109)
    box(444, 169, 355, "2  Dock orchestrator", ["Intent + follow-up state", "Local routing before model"], h=109)
    box(856, 154, 390, "3a  Direct / local route", ["Actions, tools, workflow plans", "No model request for matches"], h=109)
    box(856, 288, 390, "3b  Ollama fallback", ["Compact context + prompt profile", "Reply / tool / pending code"], GOLD, h=94)
    box(1304, 169, 440, "4  Execute or review", ["Tool: prepare → execute → apply", "Code: validate → review → Run"], h=109)
    box(1802, 169, 558, "5  Result in the same workspace", ["Layers, labels, overlays, tables, widgets", "Refresh context + record local diagnostics"], BLUE, h=109)
    arrow(387, 223, 444, 223)
    arrow(799, 208, 856, 208)
    arrow(799, 251, 856, 330, GOLD, True)
    arrow(1246, 208, 1304, 208)
    arrow(1246, 330, 1304, 251, GOLD, True)
    arrow(1744, 223, 1802, 223, BLUE)
    text(1318, 323, "Reply → transcript     Tool → dispatcher     Code → review panel", 18, INK)
    text(1318, 355, "Viewer state feeds the next turn; the model does not own the viewer.", 18, GREEN)

    x, y = panel(0, 0, "A", "CLIENT & WORKBENCH", "napari.yaml → widgets/chat_widget.py")
    box(x, y, 350, "Qt dock widget", ["Connection + model + Fast mode", "Chat transcript + prompt editor"], h=108)
    box(x + 373, y, 350, "Interaction surfaces", ["Library / Templates / Actions", "Shortcuts / Session / Advanced"], h=108)
    note(x, y + 143, ["Prompt → natural language request", "Run My Code → execute user Python after preflight", "Refine My Code → model-assisted repair", "Actions / Shortcuts → direct, repeatable execution", "Optional Voice Input → editable prompt text"])

    x, y = panel(1, 0, "B", "CONTEXT & SESSION STATE", "agent/context.py · profiler.py · session_memory.py")
    box(x, y, 350, "Live viewer context", ["Names, types, shapes, dtype", "Selection, ROI, layer profiles"], BLUE, h=108)
    box(x + 373, y, 350, "Conversation state", ["Intent + follow-up constraints", "Recent action + failure context"], BLUE, h=108)
    note(x, y + 143, ["Compact viewer payload goes to the model.", "Layer profiling uses local image information.", "Session memory binds follow-ups to prior outcomes.", "Pending actions and pending code remain explicit.", "Context is refreshed after tools change the viewer."])

    x, y = panel(2, 0, "C", "ROUTING & WORKFLOW PLANNING", "agent/prompt_routing.py · intent_state.py · workflow_planner.py")
    box(x, y, 350, "Local request handling", ["UI help + recognized operations", "Intent guards + route selection"], h=108)
    box(x + 373, y, 350, "Workflow routes", ["Plan / execute / report", "Safe viewer-control sequences"], h=108)
    note(x, y + 143, ["Matched local routes can avoid Ollama entirely.", "Code / repair intent can skip local workflow routing.", "Conservative binary-segmentation plan is supported.", "Workflow executor assesses and refines mask quality.", "Unmatched requests continue to the model path."])

    x, y = panel(0, 1, "D", "LOCAL MODEL SERVICE", "agent/client.py · tools.py · widgets/chat_widget.py", GOLD)
    box(x, y, 350, "Request builder", ["tool_routing / code_generation", "code_repair prompt profiles"], GOLD, h=108)
    box(x + 373, y, 350, "Ollama HTTP API", ["Configured base URL + model", "POST /api/chat; stream=false"], GOLD, h=108)
    note(x, y + 143, ["Payload: compact context + memory + user request.", "Background request keeps model I/O off the UI thread.", "Fast mode adjusts reasoning for supported models.", "Normalized response: reply, tool, or code.", "Model loading / unloading and pull progress are available."])

    x, y = panel(1, 1, "E", "TOOL EXECUTION CONTRACT", "agent/dispatcher.py · tool_registry.py · tool_types.py")
    bw = 224
    for i, (title, lines) in enumerate([
        ("1  prepare", ["Resolve tool name", "Bind viewer + args"]),
        ("2  execute", ["Compute tool result", "Immediate or worker"]),
        ("3  apply", ["Update viewer / UI", "Return result message"]),
    ]):
        box(x + i * 250, y, bw, title, lines, h=108)
        if i < 2:
            arrow(x + i * 250 + bw, y + 54, x + (i + 1) * 250, y + 54)
    note(x, y + 143, ["ToolSpec describes parameters, layer types and mode.", "PreparedJob → ToolResult carries computation payloads.", "Worker tools compute through napari thread_worker.", "Returned results are applied through the UI callback.", "Direct actions and model-selected tools share dispatch."])

    x, y = panel(2, 1, "F", "IMAGE ANALYSIS & VIEWER TOOLS", "agent/tools_builtin/ · image_ops.py · analysis widgets")
    box(x, y, 350, "Pixels + masks", ["CLAHE / Gaussian / projection", "Threshold / cleanup / labels"], h=108)
    box(x + 373, y, 350, "Measures + layout", ["ROI values / label tables", "Montage / overlays / visibility"], h=108)
    note(x, y + 143, ["NumPy, SciPy, scikit-image and matplotlib support analysis.", "Interactive tools: intensity, line profile, group comparison.", "Layer outputs stay attached to the live viewer.", "Built-ins include inspection and synthetic examples.", "Some registered workflow / SAM names are placeholders."])

    x, y = panel(0, 2, "G", "PYTHON GENERATION & REVIEW", "agent/code_validation.py · pending_code_panel.py", RED)
    box(x, y, 350, "Generate / refine", ["Normalize code + inspect AST", "Layer/API checks + diagnostics"], GOLD, h=108)
    box(x + 373, y, 350, "Review → Run Code", ["Blocked issues prevent running", "Warnings / repair feedback"], RED, h=108)
    note(x, y + 143, ["Execution namespace: viewer, selected_layer, np, napari.", "exec(compile(...)) runs inside the napari process.", "run_in_background(compute, apply) supports heavy work.", "Captured stdout and errors appear in the transcript.", "Validation is a preflight check, not a security sandbox."], RED)

    x, y = panel(1, 2, "H", "WORKSPACE SAVE & RESTORE", "agent/workspace_state.py", BLUE)
    box(x, y, 350, "JSON manifest", ["Layers + viewer display state", "Source references + metadata"], BLUE, h=108)
    box(x + 373, y, 350, "Workspace assets", ["OME-Zarr image / labels data", "Inline records where suitable"], BLUE, h=108)
    note(x, y + 143, ["Capture source information, transforms and layer state.", "Generated image data can be stored beside the manifest.", "Shapes / points and display properties are serialized.", "Restore reopens sources or loads saved layer assets.", "This is workspace persistence, not a model memory DB."])

    x, y = panel(2, 2, "I", "LIBRARY & LOCAL PREFERENCES", "agent/prompt_library.py · ui_state.py · action_library.py", BLUE)
    box(x, y, 350, "Reusable content", ["Saved prompts + code snippets", "Built-in templates + actions"], BLUE, h=108)
    box(x + 373, y, 350, "Local state", ["prompt_library.json", "ui_state.json + shortcuts"], BLUE, h=108)
    note(x, y + 143, ["Default location: ~/.napari-chat-assistant/", "Pin, tag, rename and reuse prompts or code.", "Action definitions describe direct execution targets.", "UI state remembers layouts and selected preferences.", "Templates are curated starting points, not RAG retrieval."])

    x, y = panel(0, 3, "J", "OPTIONAL VOICE INPUT", "voice/controller.py · audio_recorder.py · backends/", GOLD)
    box(x, y, 350, "Record → transcribe", ["Qt multimedia microphone", "Local faster-whisper backend"], GOLD, h=108)
    box(x + 373, y, 350, "Review → submit", ["Editable transcript preview", "Run via shared prompt callback"], GOLD, h=108)
    note(x, y + 143, ["Input-device selection + live recording level meter.", "Transcription runs in a background worker.", "Voice window stays available during viewer work.", "Requires optional dependencies and local model assets.", "Submitted speech follows the ordinary request routes."])

    x, y = panel(1, 3, "K", "OPTIONAL SAM2 SEGMENTATION", "agent/sam2_backend.py · integrations/sam2_adapter.py", GOLD)
    box(x, y, 350, "Prompt geometry", ["Box / foreground-background points", "Interactive preview + 3D tools"], GOLD, h=108)
    box(x + 373, y, 350, "Local SAM2 predictor", ["Configured checkpoint + config", "PyTorch; CUDA when available"], GOLD, h=108)
    note(x, y + 143, ["Adapter prepares 2D numeric grayscale input as RGB.", "SAM2 produces segmentation masks for viewer results.", "Backend setup checks environment and model assets.", "3D propagation is exposed through the tool layer.", "SAM2 is a separate optional model from the chat LLM."])

    x, y = panel(2, 3, "L", "DIAGNOSTICS & SYSTEM LIMITS", "telemetry/ · logging_utils.py · summary.py · report.py", RED)
    box(x, y, 350, "Local observability", ["assistant.log / crash.log", "model_telemetry.jsonl"], BLUE, h=108)
    box(x + 373, y, 350, "Reports + controls", ["Opt-in telemetry UI setting", "Latency / tokens / outcomes"], BLUE, h=108)
    note(x, y + 143, ["Local model by default; base URL is configurable.", "Model payload contains text context, not image uploads.", "Large arrays and synchronous steps can still block UI.", "Undo snapshots cover viewer controls, not all operations.", "Uncommitted remote-execution work is outside this map."], RED)

    text(32, 2038, "Read with system-design.md for request examples, source links, implementation boundaries and regeneration instructions.", 20, GREEN)
    text(32, 2068, "Architecture documentation only · no runtime code changes · SVG remains editable and sharp at any zoom", 17, "#b5c3b9")
    parts.append('</g></svg>')
    target = OUT / "system-design.svg"
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


def render(svg):
    try:
        from qtpy.QtCore import QRectF, QSizeF
        from qtpy.QtGui import QGuiApplication, QImage, QPageSize, QPainter, QPdfWriter
        from qtpy.QtSvg import QSvgRenderer
    except ImportError:
        print("SVG saved. Install qtpy and a Qt binding to also render PNG and PDF.")
        return
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QGuiApplication.instance() or QGuiApplication([])
    renderer = QSvgRenderer(str(svg))
    if not renderer.isValid():
        raise RuntimeError("Invalid architecture SVG")
    image = QImage(W * 2, H * 2, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    if not image.save(str(OUT / "system-design.png")):
        raise RuntimeError("PNG export failed")
    pdf = QPdfWriter(str(OUT / "system-design.pdf"))
    pdf.setTitle("napari-chat-assistant System Design")
    pdf.setPageSize(QPageSize(QSizeF(480, 416), QPageSize.Millimeter))
    painter = QPainter(pdf)
    renderer.render(painter, QRectF(0, 0, pdf.width(), pdf.height()))
    painter.end()
    print(f"Saved SVG, PNG and PDF in {OUT}")
    del app


if __name__ == "__main__":
    render(draw())
