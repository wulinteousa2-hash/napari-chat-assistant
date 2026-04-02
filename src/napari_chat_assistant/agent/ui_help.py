from __future__ import annotations

import re


UI_HELP_ITEMS = [
    {
        "label": "Library",
        "aliases": ("library", "prompt library"),
        "purpose": "Stores reusable prompts and code, plus recent items from your own workflow.",
        "when": "Use it when you want to reload a past prompt, run a saved code snippet, or keep repeatable tasks close to the viewer.",
        "tip": "Single-click loads into the editor. Double-click sends a prompt or runs a code snippet.",
    },
    {
        "label": "Prompts Tab",
        "aliases": ("prompts tab", "prompt tab"),
        "purpose": "Shows reusable prompt entries in the Library.",
        "when": "Use it when you want to reload a natural-language workflow or a repeated assistant request.",
        "tip": "Double-click sends the selected prompt immediately.",
    },
    {
        "label": "Code Tab",
        "aliases": ("code tab",),
        "purpose": "Shows reusable code snippets and demo packs in the Library.",
        "when": "Use it when you want to run a saved snippet or load built-in demo code quickly.",
        "tip": "Double-click runs the selected code entry.",
    },
    {
        "label": "Prompt",
        "aliases": ("prompt", "prompt box", "input box"),
        "purpose": "Main text box for normal requests to the assistant.",
        "when": "Use it when you want to ask about layers, request a tool action, or paste your own Python.",
        "tip": "Press Enter to send. Use Shift, Ctrl, or Alt plus Enter for a newline.",
    },
    {
        "label": "Model",
        "aliases": ("model", "model field", "model picker"),
        "purpose": "Chooses which Ollama model the assistant uses for chat requests.",
        "when": "Use it when you want to switch between a faster smaller model and a slower larger one.",
        "tip": "Smaller models usually respond faster. Larger models may do better on planning and code generation.",
    },
    {
        "label": "Base URL",
        "aliases": ("base url", "ollama url"),
        "purpose": "Points the assistant to the Ollama server.",
        "when": "Use it when Ollama is running somewhere other than the default local address.",
        "tip": "Leave it at `http://127.0.0.1:11434` unless you intentionally changed the Ollama host or port.",
    },
    {
        "label": "Run Code",
        "aliases": ("run code",),
        "purpose": "Runs assistant-generated code after you review and approve it.",
        "when": "Use it when the assistant has already produced code and marked it as pending.",
        "tip": "This is for assistant-generated code, not your own pasted code.",
    },
    {
        "label": "Run My Code",
        "aliases": ("run my code", "prompt code"),
        "purpose": "Runs your own pasted Python directly in the current napari session.",
        "when": "Use it when you already know the code you want to try and do not need the assistant to write it first.",
        "tip": "Paste Python into the Prompt box, then click Run My Code.",
    },
    {
        "label": "Refine My Code",
        "aliases": ("refine my code", "repair my code", "fix my code"),
        "purpose": "Sends pasted or failed user code back through the assistant so it can explain or repair it for the current plugin environment.",
        "when": "Use it when your own code failed, was blocked, or needs adaptation to the current napari session.",
        "tip": "Refine My Code prefers code in the Prompt box and falls back to the last failed Run My Code submission.",
    },
    {
        "label": "Run Code vs Run My Code",
        "aliases": ("run code vs run my code", "difference between run code and run my code"),
        "purpose": "Explains which code button fits assistant-generated code versus your own pasted Python.",
        "when": "Use it when you are not sure which code path to use.",
        "tip": "Run Code is for pending assistant code. Run My Code is for your own pasted Python.",
    },
    {
        "label": "Copy Code",
        "aliases": ("copy code",),
        "purpose": "Copies the current pending assistant-generated code to the clipboard.",
        "when": "Use it when you want to inspect, save, or reuse the generated code outside the plugin before running it.",
        "tip": "Copy Code works on pending assistant code, not on arbitrary text in the Prompt box.",
    },
    {
        "label": "Reject",
        "aliases": ("reject",),
        "purpose": "Rejects the last assistant outcome from session memory.",
        "when": "Use it when the last answer or result was wrong, misleading, or not worth reinforcing.",
        "tip": "Reject helps keep session memory cleaner over the current workflow.",
    },
    {
        "label": "Pending Code",
        "aliases": ("pending code",),
        "purpose": "Shows whether the assistant has prepared code that is waiting for review.",
        "when": "Use it when you want to know whether there is code ready to run, copy, or reject.",
        "tip": "Pending code means prepared, not executed.",
    },
    {
        "label": "Load",
        "aliases": ("load", "load model"),
        "purpose": "Loads the selected Ollama model into memory now.",
        "when": "Use it before chatting when you want to avoid the cold-start delay on the first request.",
        "tip": "Load warms the model. Test only checks reachability and model availability.",
    },
    {
        "label": "Unload",
        "aliases": ("unload", "unload model"),
        "purpose": "Releases the selected Ollama model from memory.",
        "when": "Use it when you want to free RAM or VRAM after a session or before switching models.",
        "tip": "After Unload, the next Load or first prompt will be slower again.",
    },
    {
        "label": "Test",
        "aliases": ("test", "test connection"),
        "purpose": "Checks whether Ollama is reachable and whether the selected model tag is installed locally.",
        "when": "Use it when you want to confirm the server and model are available before chatting.",
        "tip": "Test does not warm the model. Load does.",
    },
    {
        "label": "Setup",
        "aliases": ("setup", "setup button"),
        "purpose": "Shows Ollama setup steps, including how to start Ollama and pull the selected model.",
        "when": "Use it when Ollama is not running yet or the model tag is not installed locally.",
        "tip": "Setup is guidance, not an installer. It shows the commands you need.",
    },
    {
        "label": "Layer Context",
        "aliases": ("layer context", "context", "viewer summary"),
        "purpose": "Shows a copyable summary of the current layers and selection, plus per-layer Insert and Copy actions for prompt building.",
        "when": "Use it when you want to confirm which layers are open, copy exact layer summaries, or insert layer lines into the Prompt box.",
        "tip": "Use the Summary tab for full-session text and the Layers tab for row-by-row Insert or Copy actions.",
    },
    {
        "label": "Advanced",
        "aliases": ("advanced", "advanced menu"),
        "purpose": "Holds optional integrations and advanced controls that do not belong in the main workflow.",
        "when": "Use it when you need optional features such as experimental SAM2 setup or SAM2 Live.",
        "tip": "The default workflow should work without touching Advanced.",
    },
    {
        "label": "Image Grid",
        "aliases": ("image grid", "grid view", "show layers in grid", "tile open images"),
        "purpose": "Places open image layers into napari grid view so they can be compared side by side instead of overlapping.",
        "when": "Use it when several image layers share the same space and you want a quick visual comparison without manual transforms.",
        "tip": "Ask for grid, tile, or side-by-side comparison. You can also ask for tighter spacing or spacing 0.",
    },
    {
        "label": "Grid Spacing",
        "aliases": ("grid spacing", "spacing", "gap", "grid gap"),
        "purpose": "Controls the visual gap between tiles in image grid view.",
        "when": "Use it when the grid layout is too loose or too cramped for comparison.",
        "tip": "Ask for spacing 0, small spacing, or tighter spacing when you want images closer together.",
    },
    {
        "label": "Hide Image Grid",
        "aliases": ("hide image grid", "turn grid off", "disable grid view"),
        "purpose": "Turns image grid view off and restores non-image layers hidden for comparison.",
        "when": "Use it when you want to return to the normal overlapping layer view.",
        "tip": "This only affects the comparison grid state. It does not delete or reorder layers.",
    },
    {
        "label": "Activity",
        "aliases": ("activity", "activity tab"),
        "purpose": "Shows a short local log of status updates, model actions, tool runs, and code actions.",
        "when": "Use it when you want a quick local history of what the assistant just did.",
        "tip": "Activity is the fastest place to check what happened after a request.",
    },
    {
        "label": "Telemetry",
        "aliases": ("telemetry", "telemetry tab"),
        "purpose": "Holds optional local performance tracking and telemetry tools.",
        "when": "Use it when you want to compare latency, inspect usage timing, or debug performance.",
        "tip": "Telemetry is optional and stays local. You need to enable it first.",
    },
    {
        "label": "Summary",
        "aliases": ("summary", "telemetry summary"),
        "purpose": "Shows a short local performance summary from recorded telemetry.",
        "when": "Use it when you want a quick view of latency and usage without reading raw logs.",
        "tip": "Summary is the fastest telemetry view for normal comparison.",
    },
    {
        "label": "Log",
        "aliases": ("log", "telemetry log"),
        "purpose": "Shows the raw local telemetry records.",
        "when": "Use it when you want deeper inspection than the summary view.",
        "tip": "Use Log for debugging or detailed local performance inspection.",
    },
    {
        "label": "Reset",
        "aliases": ("reset", "telemetry reset"),
        "purpose": "Clears the local telemetry history.",
        "when": "Use it when you want to start fresh before a new round of performance testing.",
        "tip": "Reset only affects telemetry records, not prompts, code, or session memory.",
    },
    {
        "label": "Diagnostics",
        "aliases": ("diagnostics", "diagnostics tab"),
        "purpose": "Provides access to the local app log and crash log for troubleshooting.",
        "when": "Use it when the plugin behaves unexpectedly or you want deeper local logs.",
        "tip": "Diagnostics is for troubleshooting, not normal day-to-day chat use.",
    },
    {
        "label": "SAM2 Setup",
        "aliases": ("sam2 setup",),
        "purpose": "Configures the SAM2 repo path, checkpoint, config, and device.",
        "when": "Use it when you want to enable the experimental SAM2 segmentation backend.",
        "tip": "Use Auto Detect after cloning the SAM2 repo, then Test and Save. SAM2 Live stays disabled until the backend passes readiness checks.",
    },
    {
        "label": "SAM2 Live",
        "aliases": ("sam2 live",),
        "purpose": "Opens the interactive SAM2 preview workflow for box or point prompts.",
        "when": "Use it after SAM2 is configured and you want live segmentation previews in the current viewer.",
        "tip": "It is for interactive preview and apply, not basic assistant chat.",
    },
    {
        "label": "Status",
        "aliases": ("status", "status line"),
        "purpose": "Shows short live feedback for connection checks, requests, tool runs, and code state.",
        "when": "Use it when you want the current state at a glance without opening logs.",
        "tip": "Green means success, red means failure, and blue-gray means neutral or in-progress.",
    },
    {
        "label": "How To Ask",
        "aliases": ("how should i ask", "how to ask", "how do i ask", "prompt tips"),
        "purpose": "Gives simple guidance for phrasing requests so the assistant can act more reliably.",
        "when": "Use it when you want better results from prompts, tools, or generated code.",
        "tip": "Name the layer when possible and say whether you want preview, apply, explain, or code.",
    },
]


QUESTION_CUES = (
    "what is",
    "what does",
    "what do",
    "how do i use",
    "how to use",
    "how use",
    "when do i use",
    "when to use",
    "what for",
    "what's",
    "explain",
    "help",
    "difference",
)


def build_ui_help_prompt_block() -> str:
    lines = [
        "UI help facts:",
        "- Load: warms the selected Ollama model now.",
        "- Unload: releases the selected model from memory.",
        "- Test: checks Ollama reachability and whether the selected model tag is installed.",
        "- Setup: shows Ollama start and pull steps.",
        "- Library: stores built-in, recent, saved, pinned prompts and code snippets.",
        "- Prompts tab: reusable natural-language workflows in the Library.",
        "- Code tab: reusable code snippets and demo packs in the Library.",
        "- Prompt: main input box for requests or pasted Python.",
        "- Model: chooses the Ollama model for chat.",
        "- Base URL: points to the Ollama server.",
        "- Run Code: runs assistant-generated pending code after review.",
        "- Run My Code: runs the user's own pasted Python in the current napari session.",
        "- Refine My Code: repairs or explains the user's pasted or failed code for the current plugin environment.",
        "- Copy Code: copies pending assistant-generated code.",
        "- Reject: rejects the last assistant outcome from session memory.",
        "- Pending Code: indicates whether assistant code is waiting for review.",
        "- Layer Context: copyable layer/session summary with per-layer Insert and Copy actions.",
        "- Advanced: optional integrations such as experimental SAM2 controls.",
        "- Image Grid: shows open image layers side by side in napari grid view for quick comparison.",
        "- Grid Spacing: controls the gap between image tiles in grid view.",
        "- Hide Image Grid: turns grid view off and restores non-image layers hidden for comparison.",
        "- Activity: short local action log.",
        "- Telemetry: optional local performance tracking.",
        "- Summary: quick telemetry view.",
        "- Log: raw telemetry records.",
        "- Reset: clears telemetry history.",
        "- Diagnostics: local troubleshooting logs.",
        "- If the user asks what a plugin control does, answer directly and concisely instead of choosing action=tool or action=code.",
        "- If the user asks how to use the assistant better, give short prompt tips and practical examples.",
    ]
    return "\n".join(lines)


def answer_ui_question(text: str) -> str | None:
    lowered = " ".join(str(text or "").strip().lower().split())
    if not lowered:
        return None

    if _looks_like_difference_question(lowered, ("load",), ("test",)):
        return (
            "**Load vs Test**\n"
            "- `Load` warms the selected model into memory so the first real request starts faster.\n"
            "- `Test` only checks whether Ollama is reachable and whether the selected model tag exists locally."
        )
    if _contains_alias(lowered, "run code") and _contains_alias(lowered, "run my code") and ("difference" in lowered or "vs" in lowered):
        return (
            "**Run Code vs Run My Code**\n"
            "- `Run Code` executes assistant-generated code that is already waiting as pending code.\n"
            "- `Run My Code` executes your own pasted Python directly from the Prompt box."
        )
    if any(_contains_alias(lowered, alias) for alias in ("how should i ask", "how to ask", "how do i ask", "prompt tips")):
        return (
            "**Prompt Tips**\n"
            "- Name the layer when possible.\n"
            "- Say whether you want preview, apply, explain, or code.\n"
            "- Mention ROI, box, points, or comparison explicitly when relevant."
        )

    has_question_cue = any(cue in lowered for cue in QUESTION_CUES) or lowered.endswith("?")
    matched_items: list[tuple[int, dict[str, object]]] = []
    for item in UI_HELP_ITEMS:
        matched_aliases = [alias for alias in item["aliases"] if _contains_alias(lowered, alias)]
        if matched_aliases and (has_question_cue or any(lowered == alias for alias in item["aliases"])):
            matched_items.append((max(len(alias) for alias in matched_aliases), item))
    if matched_items:
        _, item = max(matched_items, key=lambda pair: pair[0])
        return (
            f"**{item['label']}**\n"
            f"- Purpose: {item['purpose']}\n"
            f"- Use it when: {item['when']}\n"
            f"- Tip: {item['tip']}"
        )
    return None


def _looks_like_difference_question(text: str, left_aliases: tuple[str, ...], right_aliases: tuple[str, ...]) -> bool:
    return (
        "difference" in text
        and any(_contains_alias(text, alias) for alias in left_aliases)
        and any(_contains_alias(text, alias) for alias in right_aliases)
    )


def _contains_alias(text: str, alias: str) -> bool:
    normalized_text = " ".join(str(text or "").strip().lower().split())
    normalized_alias = " ".join(str(alias or "").strip().lower().split())
    if not normalized_text or not normalized_alias:
        return False
    pattern = r"(?<!\w)" + re.escape(normalized_alias) + r"(?!\w)"
    return re.search(pattern, normalized_text) is not None
