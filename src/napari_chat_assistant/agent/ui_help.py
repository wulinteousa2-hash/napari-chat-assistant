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
        "label": "Templates Tab",
        "aliases": ("templates tab", "template tab"),
        "purpose": "Shows built-in runnable templates grouped by workflow category.",
        "when": "Use it when you want a stable starting point for data generation, processing, measurement, or statistics.",
        "tip": "Preview a template first, then load it into the Prompt box or run it directly.",
    },
    {
        "label": "Actions Tab",
        "aliases": ("actions tab", "actions"),
        "purpose": "Shows deterministic built-in functions grouped by category so they can be run directly without chat interpretation.",
        "when": "Use it when you already know the function you want and do not need the model to choose a tool for you.",
        "tip": "Preview an action, run it directly, or add it to Shortcuts for one-click reuse.",
    },
    {
        "label": "Shortcuts",
        "aliases": ("shortcuts", "shortcut buttons", "shortcuts box"),
        "purpose": "Holds user-defined one-click action buttons built from the Actions catalog.",
        "when": "Use it when you repeat the same deterministic workflows often and want fewer clicks.",
        "tip": "Add actions from the Actions tab, then save and load shortcut setups for different workflows.",
    },
    {
        "label": "Save Setup",
        "aliases": ("save setup", "save shortcuts"),
        "purpose": "Saves the current Shortcuts button layout to a JSON file.",
        "when": "Use it when you want to keep a reusable button layout for a task, project, or teaching session.",
        "tip": "This saves shortcut assignments and slot count, not the current viewer layers.",
    },
    {
        "label": "Load Setup",
        "aliases": ("load setup", "load shortcuts"),
        "purpose": "Loads a saved Shortcuts button layout from a JSON file.",
        "when": "Use it when you want to restore a previous button-driven workflow quickly.",
        "tip": "Load Setup restores the saved shortcut layout, not the viewer state.",
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
        "aliases": ("load model", "load button"),
        "purpose": "Loads the selected Ollama model into memory now.",
        "when": "Use it before chatting when you want to avoid the cold-start delay on the first request.",
        "tip": "Load warms the model. Test only checks reachability and model availability.",
    },
    {
        "label": "Unload",
        "aliases": ("unload model", "unload button"),
        "purpose": "Releases the selected Ollama model from memory.",
        "when": "Use it when you want to free RAM or VRAM after a session or before switching models.",
        "tip": "After Unload, the next Load or first prompt will be slower again.",
    },
    {
        "label": "Test",
        "aliases": ("test connection", "test button"),
        "purpose": "Checks whether Ollama is reachable and whether the selected model tag is installed locally.",
        "when": "Use it when you want to confirm the server and model are available before chatting.",
        "tip": "Test does not warm the model. Load does.",
    },
    {
        "label": "Setup",
        "aliases": ("setup button", "ollama setup"),
        "purpose": "Shows Ollama setup steps, including how to start Ollama and pull the selected model.",
        "when": "Use it when Ollama is not running yet or the model tag is not installed locally.",
        "tip": "Setup is guidance, not an installer. It shows the commands you need.",
    },
    {
        "label": "Layer Context",
        "aliases": ("layer context", "viewer summary", "context panel"),
        "purpose": "Shows a copyable summary of the current layers and selection, plus per-layer Insert and Copy actions for prompt building.",
        "when": "Use it when you want to confirm which layers are open, copy exact layer summaries, or insert layer lines into the Prompt box.",
        "tip": "Use the Summary tab for full-session text and the Layers tab for row-by-row Insert or Copy actions.",
    },
    {
        "label": "Inline",
        "aliases": ("inline", "inline insert"),
        "purpose": "Inserts the exact layer name at the current cursor position in the Prompt box.",
        "when": "Use it when you are editing code or filling a layer-name placeholder without forcing a new line.",
        "tip": "Use Insert for line-based prompt building and Inline for local in-place insertion.",
    },
    {
        "label": "Advanced",
        "aliases": ("advanced menu", "advanced controls"),
        "purpose": "Holds optional integrations and advanced controls that do not belong in the main workflow.",
        "when": "Use it when you need optional features such as experimental SAM2 setup or SAM2 Live.",
        "tip": "The default workflow should work without touching Advanced.",
    },
    {
        "label": "Help",
        "aliases": ("help menu", "help button"),
        "purpose": "Opens UI-help controls, version notes, about text, and bug-report guidance.",
        "when": "Use it when you want help using the plugin, want to see what changed, or need bug-report links.",
        "tip": "UI Help is off by default for expert workflows and can be enabled from the Help menu when you want local UI explanations.",
    },
    {
        "label": "UI Help Enabled",
        "aliases": ("ui help", "ui help enabled"),
        "purpose": "Turns local plugin-control help interception on or off.",
        "when": "Use it when you want short questions about plugin controls answered locally instead of going through the normal tool/chat path.",
        "tip": "Leave it off for expert use. Turn it on when you want to ask how the plugin UI works.",
    },
    {
        "label": "What's New",
        "aliases": ("what's new", "whats new"),
        "purpose": "Shows the short version update summary for the current plugin release.",
        "when": "Use it when you want a brief overview of the latest changes without opening the full changelog.",
        "tip": "Use About for plugin identity and Report Bug for support links.",
    },
    {
        "label": "About",
        "aliases": ("about plugin", "about napari chat assistant"),
        "purpose": "Shows the plugin name, version, and brief license/about text.",
        "when": "Use it when you want a quick identification summary of the plugin you are running.",
        "tip": "About is short by design. Use What's New or the changelog for feature details.",
    },
    {
        "label": "Report Bug",
        "aliases": ("report bug", "bug report"),
        "purpose": "Shows where to send bug reports and what information to include.",
        "when": "Use it when something is broken and you want the project issue link or support email quickly.",
        "tip": "Include the plugin version, what you clicked or asked, and the exact error text if possible.",
    },
    {
        "label": "Quick Compare Grid",
        "aliases": ("image grid", "grid view", "show layers in grid", "tile open images"),
        "purpose": "Places open image layers into napari grid view so they can be compared side by side instead of overlapping, without moving the layer data itself.",
        "when": "Use it when several image layers share the same space and you want a quick visual comparison without manual transforms.",
        "tip": "Ask for grid, tile, or side-by-side comparison. This is viewer grid mode, not a montage layout.",
    },
    {
        "label": "Grid Spacing",
        "aliases": ("grid spacing", "grid gap", "grid tile spacing"),
        "purpose": "Controls the visual gap between tiles in image grid view.",
        "when": "Use it when the grid layout is too loose or too cramped for comparison.",
        "tip": "Ask for spacing 0, small spacing, or tighter spacing when you want images closer together.",
    },
    {
        "label": "Turn Off Compare Grid",
        "aliases": ("hide image grid", "turn grid off", "disable grid view"),
        "purpose": "Turns image grid view off and restores non-image layers hidden for comparison.",
        "when": "Use it when you want to return to the normal overlapping layer view.",
        "tip": "This only affects the comparison grid state. It does not delete or reorder layers.",
    },
    {
        "label": "Presentation Layout",
        "aliases": ("presentation layout", "presentation montage", "arrange layers"),
        "purpose": "Physically arranges image and labels layers in the same viewer as a row, column, montage grid, or image-mask pairs.",
        "when": "Use it when you want a curated presentation layout rather than temporary viewer grid mode.",
        "tip": "This moves the original layers or creates presentation copies in one viewer. It is different from grid view.",
    },
    {
        "label": "ROI Intensity Analysis",
        "aliases": ("roi intensity analysis", "roi intensity"),
        "purpose": "Opens the interactive area-ROI measurement workspace for grayscale or fluorescence signal measurements.",
        "when": "Use it when you want table and histogram measurements from rectangles, ellipses, or polygons on an image.",
        "tip": "This is for area ROIs, not line scans. Use Line Profile Analysis for line-based measurements.",
    },
    {
        "label": "Line Profile Analysis",
        "aliases": ("line profile analysis", "line profile"),
        "purpose": "Opens the interactive line-scan analysis workspace for profile fitting along straight line ROIs.",
        "when": "Use it when you want to measure signal along a line and fit a model such as Gaussian, Lorentzian, or Sech^2.",
        "tip": "This is for straight line ROIs, not area measurements. Use ROI Intensity Analysis for area-based signal measurement.",
    },
    {
        "label": "Group Comparison Statistics",
        "aliases": ("group comparison statistics", "group comparison", "group stats"),
        "purpose": "Opens the interactive statistics workspace for comparing two groups of image-level or ROI-derived measurements.",
        "when": "Use it when you want descriptive statistics, assumption checks, plots, and common two-group test summaries.",
        "tip": "This is for grouped comparisons after measurement, not for drawing ROIs.",
    },
    {
        "label": "Activity",
        "aliases": ("activity tab", "activity panel"),
        "purpose": "Shows a short local log of status updates, model actions, tool runs, and code actions.",
        "when": "Use it when you want a quick local history of what the assistant just did.",
        "tip": "Activity is the fastest place to check what happened after a request.",
    },
    {
        "label": "Workspace",
        "aliases": ("workspace tab", "workspace"),
        "purpose": "Holds lightweight session save and restore controls for the current viewer state.",
        "when": "Use it when you want to save or restore a recoverable napari session without rebuilding the viewer manually.",
        "tip": "Workspace files save recoverable viewer and layer state. They are not full project archives.",
    },
    {
        "label": "Save Workspace",
        "aliases": ("save workspace",),
        "purpose": "Saves a lightweight workspace manifest for the current viewer state.",
        "when": "Use it when you want to save the current recoverable viewer and layer arrangement.",
        "tip": "Generated image layers are saved only when they are small enough for inline recovery.",
    },
    {
        "label": "Load Workspace",
        "aliases": ("load workspace",),
        "purpose": "Loads a saved workspace manifest into the current viewer.",
        "when": "Use it when you want to restore a previously saved recoverable viewer state.",
        "tip": "Use Clear current layers before load when you want a clean restore into the current viewer.",
    },
    {
        "label": "Restore Last Workspace",
        "aliases": ("restore last workspace", "restore last"),
        "purpose": "Reloads the most recently used workspace manifest path.",
        "when": "Use it when you want to jump back to the last saved or loaded workspace quickly.",
        "tip": "This uses the last remembered workspace path from the plugin UI state.",
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
        "aliases": ("telemetry summary", "summary tab"),
        "purpose": "Shows a short local performance summary from recorded telemetry.",
        "when": "Use it when you want a quick view of latency and usage without reading raw logs.",
        "tip": "Summary is the fastest telemetry view for normal comparison.",
    },
    {
        "label": "Log",
        "aliases": ("telemetry log", "log tab"),
        "purpose": "Shows the raw local telemetry records.",
        "when": "Use it when you want deeper inspection than the summary view.",
        "tip": "Use Log for debugging or detailed local performance inspection.",
    },
    {
        "label": "Reset",
        "aliases": ("telemetry reset", "reset telemetry"),
        "purpose": "Clears the local telemetry history.",
        "when": "Use it when you want to start fresh before a new round of performance testing.",
        "tip": "Reset only affects telemetry records, not prompts, code, or session memory.",
    },
    {
        "label": "Diagnostics",
        "aliases": ("diagnostics tab", "diagnostics panel"),
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
        "aliases": ("status line", "connection status"),
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
        "- Code tab: reusable code snippets in the Library.",
        "- Templates tab: built-in runnable templates, including demo packs and measurement widgets.",
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
        "- Quick Compare Grid: shows open image layers side by side in napari grid view for quick comparison without moving the layers.",
        "- Grid Spacing: controls the gap between image tiles in grid view.",
        "- Turn Off Compare Grid: turns grid view off and restores non-image layers hidden for comparison.",
        "- Presentation Layout: physically arranges layers or presentation copies inside the same viewer as a row, column, montage grid, or image-mask pairs.",
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
    raw = str(text or "")
    lowered = " ".join(raw.strip().lower().split())
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

    explicit_question_cue = any(cue in lowered for cue in QUESTION_CUES)
    has_question_cue = explicit_question_cue or lowered.endswith("?")
    if _looks_like_non_help_request(raw, lowered, explicit_question_cue=explicit_question_cue):
        return None
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


def _looks_like_non_help_request(raw: str, lowered: str, *, explicit_question_cue: bool) -> bool:
    if "```" in raw:
        return True
    if "\n" in raw and any(token in raw for token in ("Code:", "Purpose:", "Tip:", "Output:", "```python")):
        return True
    if lowered.endswith("?") and not explicit_question_cue and _starts_like_action_request(lowered):
        return True
    return False


def _starts_like_action_request(lowered: str) -> bool:
    prefixes = (
        "add ",
        "can you ",
        "could you ",
        "would you ",
        "will you ",
        "please ",
        "apply ",
        "run ",
        "show ",
        "hide ",
        "delete ",
        "remove ",
        "create ",
        "make ",
        "arrange ",
        "measure ",
        "perform ",
        "split ",
        "refine ",
        "fix ",
        "label ",
        "annotate ",
    )
    return any(lowered.startswith(prefix) for prefix in prefixes)
