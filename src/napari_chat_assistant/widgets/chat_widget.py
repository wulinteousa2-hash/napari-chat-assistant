from __future__ import annotations

import io
import json
import hashlib
import time
import uuid
from contextlib import redirect_stdout

import napari
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, Signal, QTimer
from qtpy.QtGui import QColor, QTextCursor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_chat_assistant.agent.client import chat_ollama, list_ollama_models, unload_ollama_model
from napari_chat_assistant.agent.code_validation import validate_generated_code
from napari_chat_assistant.agent.context import get_viewer, layer_context_json, layer_summary
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job
from napari_chat_assistant.agent.logging_utils import (
    APP_LOG_PATH,
    CRASH_LOG_PATH,
    TELEMETRY_LOG_PATH,
    append_telemetry_event,
    enable_fault_logging,
    get_plugin_logger,
)
from napari_chat_assistant.agent.prompt_library import (
    clear_code_library,
    clear_prompt_library,
    load_prompt_library,
    merged_code_records,
    merged_prompt_records,
    prompt_title,
    prompt_library_path,
    remove_code_record,
    remove_prompt_record,
    save_prompt_library,
    set_code_pinned,
    set_prompt_pinned,
    update_record_tags,
    update_record_title,
    upsert_recent_code,
    upsert_recent_prompt,
    upsert_saved_code,
    upsert_saved_prompt,
)
from napari_chat_assistant.agent.session_memory import (
    add_memory_item,
    approve_items,
    build_session_memory_payload,
    load_session_memory,
    make_viewer_fingerprint,
    promote_from_user_turn,
    reject_items,
    save_session_memory,
    session_memory_path,
    set_active_dataset_focus,
    update_session_goal,
)
from napari_chat_assistant.agent.telemetry_summary import (
    format_telemetry_summary,
    load_telemetry_events,
    read_telemetry_tail,
    summarize_telemetry_events,
)
from napari_chat_assistant.agent.tools import ASSISTANT_TOOL_NAMES, assistant_system_prompt
from napari_chat_assistant.agent.ui_state import load_ui_state, save_ui_state
from napari_chat_assistant.widgets.message_formatting import render_assistant_message_html, render_user_message_html


class ChatInput(QTextEdit):
    sendRequested = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier):
                return super().keyPressEvent(event)
            self.sendRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


def chat_widget(napari_viewer=None) -> QWidget:
    viewer = get_viewer(napari_viewer)
    logger = get_plugin_logger()
    enable_fault_logging()
    ui_state = load_ui_state()

    root = QWidget()
    layout = QVBoxLayout(root)

    header = QLabel("Local Chat Assistant")
    layout.addWidget(header)

    splitter = QSplitter(Qt.Horizontal)
    layout.addWidget(splitter, 1)

    left_panel = QWidget()
    left_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(0, 0, 0, 0)

    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)

    config_group = QGroupBox("Model Connection")
    config_layout = QFormLayout(config_group)

    provider_combo = QComboBox()
    provider_combo.addItems(["Local (Ollama-style)"])
    provider_combo.setEnabled(False)
    config_layout.addRow("Provider:", provider_combo)

    base_url_edit = QLineEdit("http://127.0.0.1:11434")
    config_layout.addRow("Base URL:", base_url_edit)

    model_combo = QComboBox()
    model_combo.setEditable(True)
    model_combo.setMinimumContentsLength(18)
    model_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    model_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    config_layout.addRow("Model:", model_combo)

    model_hint = QLabel("Type an Ollama model tag or pick one already installed locally.")
    model_hint.setWordWrap(True)
    model_hint.setStyleSheet("QLabel { color: #9fb3c8; padding: 2px 0 6px 0; }")
    config_layout.addRow(model_hint)

    connection_status = QLabel("Status: not connected")
    connection_status.setWordWrap(True)
    connection_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    connection_status.setStyleSheet("QLabel { background: #243447; color: #f5f7fa; padding: 6px; }")
    config_layout.addRow(connection_status)

    config_btn_row = QWidget()
    config_btn_layout = QHBoxLayout(config_btn_row)
    save_btn = QPushButton("Use")
    save_btn.setToolTip("Use the current model selection for chat requests.")
    test_btn = QPushButton("Test")
    test_btn.setToolTip("Test the local Ollama connection and confirm the selected model is available.")
    unload_btn = QPushButton("Unload")
    unload_btn.setToolTip("Unload the selected model from Ollama to free local memory.")
    config_btn_layout.addWidget(save_btn)
    config_btn_layout.addWidget(test_btn)
    config_btn_layout.addWidget(unload_btn)
    pull_btn = QPushButton("Setup")
    pull_btn.setToolTip("Show Ollama setup steps and a pull command for the selected model tag.")
    config_btn_layout.addWidget(pull_btn)
    config_layout.addRow(config_btn_row)
    left_layout.addWidget(config_group)

    context_group = QGroupBox("Current Context")
    context_layout = QVBoxLayout(context_group)
    context_label = QLabel()
    context_label.setTextInteractionFlags(context_label.textInteractionFlags())
    context_label.setWordWrap(True)
    context_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    context_label.setStyleSheet("QLabel { background: #101820; color: #e6edf3; padding: 8px; }")
    context_layout.addWidget(context_label)

    context_btn_row = QWidget()
    context_btn_layout = QHBoxLayout(context_btn_row)
    refresh_btn = QPushButton("Refresh Context")
    context_btn_layout.addWidget(refresh_btn)
    context_layout.addWidget(context_btn_row)
    left_layout.addWidget(context_group, 1)

    prompt_library_group = QGroupBox("Library")
    prompt_library_layout = QVBoxLayout(prompt_library_group)
    prompt_library_hint = QLabel(
        "Click to load. Double-click to send or run. Right-click to rename or edit tags."
    )
    prompt_library_hint.setWordWrap(True)
    prompt_library_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    prompt_library_hint.setStyleSheet("QLabel { color: #cbd5e1; padding: 0 0 4px 0; }")
    library_tabs = QTabWidget()
    library_tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    prompt_library_list = QListWidget()
    prompt_library_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    prompt_library_list.setContextMenuPolicy(Qt.CustomContextMenu)
    prompt_library_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    prompt_library_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    prompt_library_list.setWordWrap(True)
    code_library_list = QListWidget()
    code_library_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    code_library_list.setContextMenuPolicy(Qt.CustomContextMenu)
    code_library_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    code_library_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    code_library_list.setWordWrap(True)
    prompt_library_font = prompt_library_list.font()
    if prompt_library_font.pointSize() > 0:
        prompt_library_font.setPointSize(prompt_library_font.pointSize() + 1)
        prompt_library_list.setFont(prompt_library_font)
        code_library_list.setFont(prompt_library_font)
    library_tabs.addTab(prompt_library_list, "Prompts")
    library_tabs.addTab(code_library_list, "Code")
    prompt_library_btn_row = QWidget()
    prompt_library_btn_layout = QHBoxLayout(prompt_library_btn_row)
    save_prompt_btn = QPushButton("Save")
    pin_prompt_btn = QPushButton("Pin")
    delete_prompt_btn = QPushButton("Delete")
    clear_prompt_btn = QPushButton("Clear")
    prompt_font_down_btn = QPushButton("A-")
    prompt_font_down_btn.setToolTip("Decrease library font size.")
    prompt_font_up_btn = QPushButton("A+")
    prompt_font_up_btn.setToolTip("Increase library font size.")
    prompt_library_btn_layout.addWidget(save_prompt_btn)
    prompt_library_btn_layout.addWidget(pin_prompt_btn)
    prompt_library_btn_layout.addWidget(delete_prompt_btn)
    prompt_library_btn_layout.addWidget(clear_prompt_btn)
    prompt_library_btn_layout.addWidget(prompt_font_down_btn)
    prompt_library_btn_layout.addWidget(prompt_font_up_btn)
    prompt_library_layout.addWidget(prompt_library_hint)
    prompt_library_layout.addWidget(library_tabs)
    prompt_library_layout.addWidget(prompt_library_btn_row)
    left_layout.addWidget(prompt_library_group, 2)

    log_group = QGroupBox("Session")
    log_layout = QVBoxLayout(log_group)
    log_tabs = QTabWidget()
    log_tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    activity_tab = QWidget()
    activity_layout = QVBoxLayout(activity_tab)
    action_log = QListWidget()
    activity_layout.addWidget(action_log, 1)
    telemetry_tab = QWidget()
    telemetry_layout = QVBoxLayout(telemetry_tab)
    log_btn_row = QWidget()
    log_btn_layout = QHBoxLayout(log_btn_row)
    telemetry_toggle = QCheckBox("Enable Telemetry")
    telemetry_toggle.setChecked(bool(ui_state.get("telemetry_enabled", False)))
    telemetry_toggle.setToolTip("Turn on performance telemetry and advanced telemetry tools only when you want them.")
    telemetry_summary_btn = QPushButton("Summary")
    telemetry_summary_btn.setToolTip(
        "Show a quick performance summary from local telemetry only."
    )
    telemetry_view_btn = QPushButton("Log")
    telemetry_view_btn.setToolTip(
        "Open the local append-only telemetry log and summary for advanced inspection of raw usage records."
    )
    telemetry_reset_btn = QPushButton("Reset")
    telemetry_reset_btn.setToolTip(
        "Clear the local telemetry log so Performance Summary starts fresh from the next request."
    )
    log_btn_layout.addWidget(telemetry_toggle)
    log_btn_layout.addWidget(telemetry_summary_btn)
    log_btn_layout.addWidget(telemetry_view_btn)
    log_btn_layout.addWidget(telemetry_reset_btn)
    telemetry_hint = QLabel("Telemetry is optional. Enable it only when you want performance tracking and advanced diagnostics.")
    telemetry_hint.setWordWrap(True)
    telemetry_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    telemetry_hint.setStyleSheet("QLabel { color: #cbd5e1; padding: 0 0 4px 0; }")
    telemetry_layout.addWidget(telemetry_hint)
    telemetry_layout.addWidget(log_btn_row)
    telemetry_layout.addStretch(1)
    diagnostics_tab = QWidget()
    diagnostics_layout = QVBoxLayout(diagnostics_tab)
    diagnostics_hint = QLabel("Advanced local logs for troubleshooting plugin behavior and crashes.")
    diagnostics_hint.setWordWrap(True)
    diagnostics_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    diagnostics_hint.setStyleSheet("QLabel { color: #cbd5e1; padding: 0 0 4px 0; }")
    diagnostics_btn_row = QWidget()
    diagnostics_btn_layout = QHBoxLayout(diagnostics_btn_row)
    app_log_btn = QPushButton("App Log")
    app_log_btn.setToolTip("Open the local application log for detailed plugin activity.")
    crash_log_btn = QPushButton("Crash Log")
    crash_log_btn.setToolTip("Open the local crash log for faults and tracebacks captured by the plugin.")
    diagnostics_btn_layout.addWidget(app_log_btn)
    diagnostics_btn_layout.addWidget(crash_log_btn)
    diagnostics_layout.addWidget(diagnostics_hint)
    diagnostics_layout.addWidget(diagnostics_btn_row)
    diagnostics_layout.addStretch(1)
    log_tabs.addTab(activity_tab, "Activity")
    log_tabs.addTab(telemetry_tab, "Telemetry")
    log_tabs.addTab(diagnostics_tab, "Diagnostics")
    log_layout.addWidget(log_tabs)
    left_layout.addWidget(log_group, 0)

    transcript_group = QGroupBox("Chat")
    transcript_layout = QVBoxLayout(transcript_group)
    transcript = QTextEdit()
    transcript.setReadOnly(True)
    transcript.setPlaceholderText("Conversation will appear here.")
    transcript.setStyleSheet("QTextEdit { background: #0b1021; color: #d6deeb; border: 1px solid #22304a; padding: 10px; }")
    transcript_layout.addWidget(transcript, 1)

    code_btn_row = QWidget()
    code_btn_layout = QHBoxLayout(code_btn_row)
    pending_code_label = QLabel("Pending code: none")
    pending_code_label.setStyleSheet("QLabel { color: #c7d2fe; padding: 2px 0; }")
    reject_memory_btn = QPushButton("👎 Reject")
    reject_memory_btn.setToolTip("Reject the last assistant outcome from session memory.")
    reject_memory_btn.setEnabled(False)
    help_btn = QPushButton("Help")
    help_btn.setToolTip("Show prompt-writing tips and example instruction patterns.")
    run_code_btn = QPushButton("Run Code")
    run_my_code_btn = QPushButton("Run My Code")
    run_my_code_btn.setToolTip("Paste your own Python in the Prompt box and click to run it directly, without opening QtConsole.")
    copy_code_btn = QPushButton("Copy Code")
    run_code_btn.setEnabled(False)
    copy_code_btn.setEnabled(False)
    code_btn_layout.addWidget(pending_code_label, 1)
    code_btn_layout.addWidget(run_code_btn)
    code_btn_layout.addWidget(copy_code_btn)
    code_btn_layout.addWidget(run_my_code_btn)
    code_btn_layout.addWidget(reject_memory_btn)
    code_btn_layout.addWidget(help_btn)
    transcript_layout.addWidget(code_btn_row)

    input_group = QGroupBox("Prompt")
    input_group_layout = QVBoxLayout(input_group)
    prompt = ChatInput()
    prompt.setPlaceholderText("Ask about the current napari session...")
    prompt.setMinimumHeight(120)
    prompt.setStyleSheet("QTextEdit { background: #10182b; color: #d6deeb; border: 1px solid #30415f; padding: 8px; }")
    input_group_layout.addWidget(prompt)

    analysis_group = QGroupBox("Analysis")
    analysis_layout = QFormLayout(analysis_group)
    analysis_hint = QLabel(
        "Use built-in image-layer analysis tools for summary stats, histogram popups, and two-layer t-tests."
    )
    analysis_hint.setWordWrap(True)
    analysis_hint.setStyleSheet("QLabel { color: #cbd5e1; padding: 0 0 4px 0; }")
    analysis_layout.addRow(analysis_hint)

    analysis_layer_combo = QComboBox()
    analysis_layout.addRow("Layer:", analysis_layer_combo)

    histogram_bins_edit = QLineEdit("64")
    analysis_layout.addRow("Bins:", histogram_bins_edit)

    analysis_btn_row = QWidget()
    analysis_btn_layout = QHBoxLayout(analysis_btn_row)
    summary_stats_btn = QPushButton("Summary Stats")
    histogram_btn = QPushButton("Histogram")
    analysis_btn_layout.addWidget(summary_stats_btn)
    analysis_btn_layout.addWidget(histogram_btn)
    analysis_layout.addRow(analysis_btn_row)

    compare_layer_a_combo = QComboBox()
    compare_layer_b_combo = QComboBox()
    compare_test_combo = QComboBox()
    compare_test_combo.addItems(["Student t-test", "Welch t-test"])
    analysis_layout.addRow("Compare A:", compare_layer_a_combo)
    analysis_layout.addRow("Compare B:", compare_layer_b_combo)
    analysis_layout.addRow("Test:", compare_test_combo)

    compare_btn = QPushButton("Compare Layers")
    analysis_layout.addRow(compare_btn)

    right_layout.addWidget(transcript_group, 4)
    right_layout.addWidget(input_group, 1)
    analysis_group.hide()

    splitter.addWidget(left_panel)
    splitter.addWidget(right_panel)
    splitter.setStretchFactor(0, 2)
    splitter.setStretchFactor(1, 3)

    saved_settings = {
        "provider": "Local (Ollama-style)",
        "base_url": "http://127.0.0.1:11434",
        "model": "nemotron-cascade-2:30b",
    }
    active_workers: list[object] = []
    available_models: list[str] = []
    pending_code = {"code": "", "message": "", "turn_id": "", "model": "", "runnable": False}
    last_turn_metrics = {"turn_id": "", "model": "", "action": "", "prompt_hash": ""}
    session_memory_state = load_session_memory()
    last_memory_candidate_ids: list[str] = []
    prompt_library_state = load_prompt_library()
    generation_defaults = {
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "repeat_penalty": 1.5,
    }
    wait_indicator = {
        "active": False,
        "started_at": 0.0,
        "phase": "thinking",
        "style": "QLabel { background: #24472f; color: #f5fff7; padding: 6px; }",
    }
    wait_timer = QTimer(root)
    wait_timer.setInterval(250)

    base_url_edit.setText(saved_settings["base_url"])
    model_combo.addItem(saved_settings["model"])
    model_combo.setCurrentText(saved_settings["model"])

    def append_chat_message(role: str, message: str, *, render_markdown: bool = True):
        if role == "user":
            safe_message = render_user_message_html(message)
            html = (
                '<table width="100%" cellspacing="0" cellpadding="0" style="margin: 8px 0;"><tr><td width="35%"></td>'
                '<td align="right"><div style="display: inline-block; background: #1d2a44; color: #d7ecff; '
                'border: 1px solid #3f5d87; padding: 10px 12px; border-radius: 10px; '
                f'text-align: left; max-width: 100%;">{safe_message}</div></td></tr></table>'
            )
        else:
            safe_message = render_assistant_message_html(message) if render_markdown else render_user_message_html(message)
            html = (
                '<table width="100%" cellspacing="0" cellpadding="0" style="margin: 8px 0;"><tr>'
                '<td align="left"><div style="display: inline-block; background: #111827; color: #d1fae5; '
                'border: 1px solid #2f855a; padding: 10px 12px; border-radius: 10px; '
                f'text-align: left; max-width: 100%;">{safe_message}</div></td><td width="35%"></td></tr></table>'
            )
        transcript.append(html)

    def replace_last_assistant(text_out: str, *, render_markdown: bool = True):
        cursor = transcript.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        append_chat_message("assistant", text_out, render_markdown=render_markdown)

    def wait_indicator_text() -> str:
        frames = (".", "..", "...")
        elapsed = max(0.0, time.perf_counter() - float(wait_indicator["started_at"]))
        frame = frames[int(elapsed * 4) % len(frames)]
        return f"Status: {wait_indicator['phase']} {frame}"

    def start_wait_indicator(*, phase: str = "thinking"):
        wait_indicator["active"] = True
        wait_indicator["started_at"] = time.perf_counter()
        wait_indicator["phase"] = str(phase or "thinking").strip() or "thinking"
        connection_status.setStyleSheet(wait_indicator["style"])
        connection_status.setText(wait_indicator_text())
        wait_timer.start()

    def stop_wait_indicator():
        wait_indicator["active"] = False
        wait_timer.stop()

    def tick_wait_indicator():
        if not wait_indicator["active"]:
            return
        connection_status.setText(wait_indicator_text())

    def widget_is_alive(widget: QWidget | None) -> bool:
        if widget is None:
            return False
        try:
            widget.objectName()
        except RuntimeError:
            return False
        return True

    def refresh_context(*_args):
        if widget_is_alive(context_label):
            context_label.setText(layer_summary(viewer))
        refresh_analysis_controls()

    def image_layer_names() -> list[str]:
        if viewer is None:
            return []
        return [layer.name for layer in viewer.layers if isinstance(layer, napari.layers.Image)]

    def refresh_analysis_controls():
        names = image_layer_names()
        combo_defaults = (
            (analysis_layer_combo, ""),
            (compare_layer_a_combo, ""),
            (compare_layer_b_combo, ""),
        )
        for combo, fallback in combo_defaults:
            if not widget_is_alive(combo):
                continue
            current = combo.currentText().strip()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            elif fallback and fallback in names:
                combo.setCurrentText(fallback)
            combo.blockSignals(False)
        if not all(widget_is_alive(combo) for combo, _fallback in combo_defaults):
            return
        if len(names) >= 2:
            if not compare_layer_a_combo.currentText().strip():
                compare_layer_a_combo.setCurrentText(names[0])
            if not compare_layer_b_combo.currentText().strip() or compare_layer_b_combo.currentText() == compare_layer_a_combo.currentText():
                compare_layer_b_combo.setCurrentText(names[1])
        elif len(names) == 1:
            compare_layer_a_combo.setCurrentText(names[0])
            compare_layer_b_combo.setCurrentText(names[0])

    def run_prepared_tool_request(prepared: dict, *, tool_name: str, tool_message: str = ""):
        nonlocal session_memory_state
        if prepared.get("mode") == "immediate":
            result_message = str(prepared.get("message", ""))
            refresh_context()
            append_chat_message("assistant", f"{tool_message}\n{result_message}" if tool_message else result_message)
            append_log(f"Tool executed from Analysis panel: {tool_name}")
            set_status(f"Status: {tool_name} completed", ok=True)
            remember_assistant_outcome(
                tool_message or result_message,
                target_type="tool_result",
                target_profile=selected_layer_profile(),
                state="approved",
            )
            return
        if prepared.get("mode") != "worker":
            set_status("Status: unsupported tool response", ok=False)
            append_log(f"Analysis panel received unsupported tool response for {tool_name}.")
            return
        try:
            set_status(f"Status: running {tool_name}", ok=None)
            result = run_tool_job(prepared["job"])
            result_message = apply_tool_job_result(viewer, result)
        except Exception as exc:
            logger.exception("Analysis panel tool failed: %s", tool_name)
            append_chat_message("assistant", f"{tool_name} failed:\n{exc}")
            append_log(f"Analysis panel tool failed: {tool_name} | {exc}")
            set_status(f"Status: {tool_name} failed", ok=False)
            return
        refresh_context()
        append_chat_message("assistant", f"{tool_message}\n{result_message}" if tool_message else result_message)
        append_log(f"Tool executed from Analysis panel: {tool_name}")
        set_status(f"Status: {tool_name} completed", ok=True)
        remember_assistant_outcome(
            tool_message or result_message,
            target_type="tool_result",
            target_profile=selected_layer_profile(),
            state="approved",
        )

    def adjust_prompt_library_font(delta: int):
        font = prompt_library_list.font()
        current_size = font.pointSize()
        if current_size <= 0:
            current_size = 10
        font.setPointSize(max(9, min(16, current_size + int(delta))))
        prompt_library_list.setFont(font)
        code_library_list.setFont(font)

    def format_code_block(code_text: str) -> str:
        stripped = str(code_text or "").strip()
        return f"```python\n{stripped}\n```" if stripped else "```python\n# empty\n```"

    def strip_code_fences(code_text: str) -> str:
        source = str(code_text or "").strip()
        if not source.startswith("```"):
            return source
        lines = source.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def extract_manual_code_submission(text: str) -> str:
        source = str(text or "").strip()
        if not source:
            return ""
        lowered = source.lower()
        triggers = ("/code", "!code", "run code", "paste code")
        matched = next((trigger for trigger in triggers if lowered.startswith(trigger)), "")
        if not matched:
            return ""
        payload = source[len(matched):].lstrip(" :\n\t")
        return strip_code_fences(payload)

    def append_log(message: str):
        item = QListWidgetItem(message)
        color_rules = {
            "Assistant log:": "#8ab4f8",
            "Crash log:": "#f28b82",
            "Telemetry log:": "#81c995",
            "Prompt library path:": "#fbc02d",
            "Session memory path:": "#c58af9",
        }
        for prefix, color in color_rules.items():
            if str(message).startswith(prefix):
                item.setForeground(QColor(color))
                break
        action_log.addItem(item)
        action_log.scrollToBottom()
        logger.info(message)

    def telemetry_summary_text() -> str:
        events, invalid_lines = load_telemetry_events()
        summary = summarize_telemetry_events(events, invalid_lines)
        return format_telemetry_summary(summary)

    def show_text_log_dialog(*, title: str, path, empty_message: str, log_prefix: str, status_text: str):
        dialog = QDialog(root)
        dialog.setWindowTitle(title)
        dialog.resize(900, 700)
        dialog_layout = QVBoxLayout(dialog)

        path_label = QLabel(f"{title}: {path}")
        path_label.setTextInteractionFlags(path_label.textInteractionFlags())
        path_label.setWordWrap(True)
        dialog_layout.addWidget(path_label)

        text_box = QTextEdit()
        text_box.setReadOnly(True)
        text_box.setStyleSheet(
            "QTextEdit { background: #0b1021; color: #d6deeb; border: 1px solid #22304a; padding: 8px; }"
        )
        dialog_layout.addWidget(text_box, 1)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        refresh_dialog_btn = QPushButton("Refresh")
        close_dialog_btn = QPushButton("Close")
        button_layout.addWidget(refresh_dialog_btn)
        button_layout.addWidget(close_dialog_btn)
        dialog_layout.addWidget(button_row)

        def refresh_dialog():
            try:
                text_box.setPlainText(path.read_text(encoding="utf-8") or empty_message)
            except Exception as exc:
                text_box.setPlainText(f"Could not read log:\n{exc}")

        refresh_dialog_btn.clicked.connect(refresh_dialog)
        close_dialog_btn.clicked.connect(dialog.accept)
        refresh_dialog()
        append_log(log_prefix)
        set_status(status_text, ok=None)
        dialog.exec_()

    def show_telemetry_summary(*_args):
        append_chat_message("assistant", telemetry_summary_text())
        append_log("Displayed telemetry summary.")
        set_status("Status: telemetry summary ready", ok=True)

    def show_telemetry_viewer(*_args):
        dialog = QDialog(root)
        dialog.setWindowTitle("Telemetry Log")
        dialog.resize(900, 700)
        dialog_layout = QVBoxLayout(dialog)

        path_label = QLabel(f"Telemetry log: {TELEMETRY_LOG_PATH}")
        path_label.setTextInteractionFlags(path_label.textInteractionFlags())
        path_label.setWordWrap(True)
        dialog_layout.addWidget(path_label)

        summary_box = QTextEdit()
        summary_box.setReadOnly(True)
        summary_box.setMinimumHeight(200)
        summary_box.setStyleSheet(
            "QTextEdit { background: #10182b; color: #d6deeb; border: 1px solid #30415f; padding: 8px; }"
        )
        dialog_layout.addWidget(summary_box)

        raw_box = QTextEdit()
        raw_box.setReadOnly(True)
        raw_box.setStyleSheet(
            "QTextEdit { background: #0b1021; color: #d6deeb; border: 1px solid #22304a; padding: 8px; }"
        )
        dialog_layout.addWidget(raw_box, 1)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        refresh_dialog_btn = QPushButton("Refresh")
        close_dialog_btn = QPushButton("Close")
        button_layout.addWidget(refresh_dialog_btn)
        button_layout.addWidget(close_dialog_btn)
        dialog_layout.addWidget(button_row)

        def refresh_dialog():
            summary_box.setMarkdown(telemetry_summary_text())
            raw_box.setPlainText(read_telemetry_tail(max_lines=250) or "[Telemetry log is empty]")

        refresh_dialog_btn.clicked.connect(refresh_dialog)
        close_dialog_btn.clicked.connect(dialog.accept)
        refresh_dialog()
        append_log("Opened telemetry log.")
        set_status("Status: telemetry log opened", ok=None)
        dialog.exec_()

    def show_app_log(*_args):
        show_text_log_dialog(
            title="App Log",
            path=APP_LOG_PATH,
            empty_message="[App log is empty]",
            log_prefix="Opened app log.",
            status_text="Status: app log opened",
        )

    def show_crash_log(*_args):
        show_text_log_dialog(
            title="Crash Log",
            path=CRASH_LOG_PATH,
            empty_message="[Crash log is empty]",
            log_prefix="Opened crash log.",
            status_text="Status: crash log opened",
        )

    def reset_telemetry_log(*_args):
        try:
            TELEMETRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            TELEMETRY_LOG_PATH.write_text("", encoding="utf-8")
        except Exception as exc:
            logger.exception("Failed to reset telemetry log.")
            append_chat_message("assistant", f"Could not reset the telemetry log:\n{exc}")
            append_log(f"Telemetry log reset failed: {exc}")
            set_status("Status: telemetry log reset failed", ok=False)
            return
        append_chat_message(
            "assistant",
            "**Performance Summary**\n"
            "- Telemetry log cleared.\n"
            "- New model usage will be recorded from the next request.\n"
            f"- Path: `{TELEMETRY_LOG_PATH}`",
        )
        append_log("Reset telemetry log.")
        set_status("Status: telemetry log reset", ok=True)

    def prompt_hash(text: str) -> str:
        clean = str(text or "").strip().encode("utf-8")
        return hashlib.sha256(clean).hexdigest()[:16] if clean else ""

    def categorize_prompt(text: str) -> str:
        source = " ".join(str(text or "").strip().lower().split())
        if not source:
            return "unknown"
        if "clahe" in source:
            return "clahe"
        if "threshold" in source and "preview" in source:
            return "threshold_preview"
        if "threshold" in source:
            return "threshold_apply"
        if "mask" in source and any(word in source for word in ("measure", "area", "volume")):
            return "mask_measurement"
        if "mask" in source or "morpholog" in source:
            return "mask_workflow"
        if "histogram" in source or "distribution" in source:
            return "histogram"
        if "t-test" in source or "ttest" in source or "welch" in source:
            return "statistical_test"
        if "mean" in source or "median" in source or "std" in source or "standard deviation" in source:
            return "intensity_summary"
        if "inspect" in source or "layer" in source:
            return "inspection"
        if "code" in source or "python" in source:
            return "code_generation"
        return "general"

    def selected_layer_snapshot() -> dict:
        profile = selected_layer_profile()
        if not isinstance(profile, dict):
            return {"selected_layer_name": "", "selected_layer_type": ""}
        return {
            "selected_layer_name": str(profile.get("layer_name", "")).strip(),
            "selected_layer_type": str(profile.get("layer_type", "")).strip(),
        }

    def record_telemetry(event_type: str, payload: dict):
        if not bool(ui_state.get("telemetry_enabled", False)):
            return
        try:
            append_telemetry_event(event_type, payload)
        except Exception:
            logger.exception("Failed to append telemetry event: %s", event_type)

    def refresh_telemetry_controls():
        enabled = bool(ui_state.get("telemetry_enabled", False))
        telemetry_summary_btn.setVisible(enabled)
        telemetry_view_btn.setVisible(enabled)
        telemetry_reset_btn.setVisible(enabled)

    def toggle_telemetry(enabled: bool):
        ui_state["telemetry_enabled"] = bool(enabled)
        save_ui_state(ui_state)
        refresh_telemetry_controls()
        if enabled:
            append_log("Telemetry enabled.")
            set_status("Status: telemetry enabled", ok=True)
        else:
            append_log("Telemetry disabled.")
            set_status("Status: telemetry disabled", ok=None)

    def show_help_tips(*_args):
        append_chat_message(
            "assistant",
            "**Prompt Tips**\n"
            "- Ask for the exact output style you want.\n"
            "- Ask the assistant to improve your prompt first when the request is vague.\n"
            "- If multiple layers are open, specify the layer name.\n"
            "- Say whether you want a preview, an applied result, or just an explanation.\n\n"
            "**Useful Phrases**\n"
            "- `Use the selected layer`\n"
            "- `Use layer: <layer_name>`\n"
            "- `Preview threshold first`\n"
            "- `Apply threshold now`\n"
            "- `Run My Code` executes Python pasted into the Prompt box\n"
            "- `Inspect the selected layer first`\n"
            "- `Use a built-in tool if possible; otherwise generate napari code`\n\n"
            "**Format Requests**\n"
            "- `Reply in markdown`\n"
            "- `Use bullets and short sections`\n"
            "- `Show a numbered workflow`\n"
            "- `Explain first, then give runnable napari code`\n\n"
            "**Language Tip**\n"
            "- You can prompt in your preferred language.\n"
            "- Results may vary by model and task.\n"
            "- Ask for the final answer in your preferred language if needed.\n\n"
            "**Example Meta-Prompts**\n"
            "- `Improve my prompt first, then answer in markdown.`\n"
            "- `Ask clarifying questions before solving this.`\n"
            "- `Inspect the selected layer first, then recommend the next step.`\n\n"
            "**Heavy Code Tip**\n"
            "- For heavy compute, use `run_in_background(compute_fn, apply_fn, error_fn=None, label=\"...\")` in Run My Code.\n"
            "- Keep `compute_fn` for NumPy/SciPy work and use `apply_fn` for `viewer.add_*` updates.\n"
            "- `Rewrite my request so the result is more precise for napari.`",
        )
        append_log("Opened prompt-writing help.")

    def persist_session_memory():
        save_session_memory(session_memory_state)

    def selected_layer_profile() -> dict | None:
        payload = layer_context_json(viewer)
        profile = payload.get("selected_layer_profile")
        return profile if isinstance(profile, dict) else None

    def set_last_memory_candidates(item_ids: list[str]):
        nonlocal last_memory_candidate_ids
        last_memory_candidate_ids = [item_id for item_id in item_ids if item_id]
        reject_memory_btn.setEnabled(bool(last_memory_candidate_ids))

    def remember_assistant_outcome(summary: str, *, target_type: str, target_profile: dict | None, state: str = "provisional"):
        clean = " ".join(str(summary or "").split()).strip()
        if not clean:
            return
        item_id = None
        nonlocal session_memory_state
        session_memory_state, item_id = add_memory_item(
            session_memory_state,
            target_type=target_type,
            summary=clean,
            target_layer="" if not isinstance(target_profile, dict) else str(target_profile.get("layer_name", "")).strip(),
            viewer_fingerprint=make_viewer_fingerprint(target_profile),
            state=state,
        )
        if item_id:
            set_last_memory_candidates([item_id])
            persist_session_memory()

    def reject_last_memory(*_args):
        nonlocal session_memory_state
        if not last_memory_candidate_ids:
            set_status("Status: no recent answer to reject", ok=False)
            append_log("Reject feedback skipped: no recent assistant memory candidates.")
            return
        session_memory_state = reject_items(session_memory_state, last_memory_candidate_ids)
        persist_session_memory()
        append_chat_message("assistant", "Marked the last assistant outcome as rejected for this session memory.")
        record_telemetry(
            "turn_feedback",
            {
                "turn_id": last_turn_metrics.get("turn_id", ""),
                "model": last_turn_metrics.get("model", ""),
                "feedback": "reject",
                "response_action": last_turn_metrics.get("action", ""),
                "prompt_hash": last_turn_metrics.get("prompt_hash", ""),
            },
        )
        append_log("Rejected last assistant memory candidates.")
        set_status("Status: last answer rejected from session memory", ok=None)
        set_last_memory_candidates([])

    def current_library_kind() -> str:
        return "code" if library_tabs.currentWidget() is code_library_list else "prompt"

    def current_library_list() -> QListWidget:
        return code_library_list if current_library_kind() == "code" else prompt_library_list

    def current_library_item_name() -> str:
        return "code snippet" if current_library_kind() == "code" else "prompt"

    def selected_library_records() -> list[dict]:
        records: list[dict] = []
        for item in current_library_list().selectedItems():
            record = item.data(Qt.UserRole)
            if isinstance(record, dict) and (record.get("prompt") or record.get("code")):
                records.append(record)
        return records

    def format_library_item_label(record: dict) -> tuple[str, str]:
        title = str(record.get("title", "")).strip() or (
            "Untitled Code" if record.get("code") else "Untitled Prompt"
        )
        source = str(record.get("source", "saved"))
        tags = [str(tag).strip() for tag in record.get("tags", []) if str(tag).strip()]
        if record.get("pinned", False):
            badge = "[Pinned]"
            color = "#fbbc05"
        elif source == "saved":
            badge = "[Saved]"
            color = "#34a853"
        elif source == "recent":
            badge = "[Recent]"
            color = "#4285f4"
        else:
            badge = "[Built-in]"
            color = "#9aa0a6"
        tag_suffix = f"  # {' | '.join(tags[:3])}" if tags else ""
        short_title = title if len(title) <= 72 else f"{title[:69].rstrip()}..."
        return f"{badge} {short_title}{tag_suffix}", color

    def format_worker_error(*args) -> str:
        for value in args:
            if isinstance(value, BaseException):
                return str(value)
            if value:
                return str(value)
        return "Unknown worker error."

    def format_code_execution_error(exc: Exception) -> str:
        error_text = str(exc).strip() or exc.__class__.__name__
        lowered = error_text.lower()
        if "equalize_adap_hist" in lowered and "skimage.exposure" in lowered:
            return (
                "Approved code failed because it used the wrong CLAHE function name from scikit-image.\n"
                "Use `skimage.exposure.equalize_adapthist(...)` instead of `skimage.exposure.equalize_adap_hist(...)`.\n"
                "This is usually a code-generation typo, not a missing package."
            )
        if "no module named 'skimage'" in lowered or 'no module named "skimage"' in lowered:
            return (
                "Approved code failed because `scikit-image` is not installed in the napari Python environment.\n"
                "Install it in the same environment that launches napari, then try again.\n"
                "Suggested command: `python -m pip install scikit-image`"
            )
        if "skimage." in lowered and "no attribute" in lowered:
            return (
                "Approved code failed because the requested scikit-image function is not available in the current napari Python environment.\n"
                "This usually means either the function name is wrong or the installed `scikit-image` version does not provide it.\n"
                "If the intent was CLAHE, use `skimage.exposure.equalize_adapthist(...)`.\n"
                f"Original error: {error_text}"
            )
        return error_text

    def extract_json_objects(text: str) -> list[dict]:
        decoder = json.JSONDecoder()
        objects: list[dict] = []
        source = str(text or "")
        index = 0
        while index < len(source):
            brace = source.find("{", index)
            if brace < 0:
                break
            try:
                parsed, end_index = decoder.raw_decode(source[brace:])
            except json.JSONDecodeError:
                index = brace + 1
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)
            index = brace + end_index
        return objects

    def normalize_model_response(reply: str) -> dict:
        objects = extract_json_objects(reply)
        parsed = objects[-1] if objects else None
        if not isinstance(parsed, dict):
            return {"action": "reply", "message": reply}

        action = str(parsed.get("action", "reply")).strip().lower()
        if action in {"reply", "tool", "code"}:
            return parsed
        if action in ASSISTANT_TOOL_NAMES:
            return {
                "action": "tool",
                "tool": action,
                "arguments": parsed.get("arguments", {}),
                "message": str(parsed.get("message", "")).strip(),
            }
        return {"action": "reply", "message": reply}

    def set_status(text: str, *, ok: bool | None = None):
        if ok is True:
            style = "QLabel { background: #24472f; color: #f5fff7; padding: 6px; }"
        elif ok is False:
            style = "QLabel { background: #4a2b2b; color: #fff3f3; padding: 6px; }"
        else:
            style = "QLabel { background: #243447; color: #f5f7fa; padding: 6px; }"
        connection_status.setStyleSheet(style)
        connection_status.setText(text)

    def set_model_controls_enabled(enabled: bool):
        for widget in (base_url_edit, model_combo, test_btn, save_btn, pull_btn, unload_btn, prompt, refresh_btn):
            widget.setEnabled(enabled)

    def set_pending_code(code_text: str = "", *, message: str = "", runnable: bool = True, label: str | None = None):
        pending_code["code"] = str(code_text or "").strip()
        pending_code["message"] = str(message or "").strip()
        pending_code["runnable"] = bool(runnable and pending_code["code"])
        has_code = bool(pending_code["code"])
        if not has_code:
            pending_code["turn_id"] = ""
            pending_code["model"] = ""
            pending_code["runnable"] = False
        pending_code_label.setText(label or ("Pending code: ready to run" if has_code else "Pending code: none"))
        run_code_btn.setEnabled(bool(has_code and pending_code["runnable"]))
        copy_code_btn.setEnabled(has_code)

    def preflight_generated_code(code_text: str) -> list[str]:
        return validate_generated_code(code_text, viewer=viewer)

    def refresh_model_choices(model_names: list[str], preferred: str | None = None):
        nonlocal available_models
        available_models = sorted({m for m in model_names if m})
        current_text = preferred or model_combo.currentText().strip() or saved_settings["model"]
        model_combo.blockSignals(True)
        model_combo.clear()
        for name in available_models:
            model_combo.addItem(name)
        if current_text:
            if current_text not in available_models:
                model_combo.addItem(current_text)
            model_combo.setCurrentText(current_text)
        model_combo.blockSignals(False)

    def refresh_library_controls():
        if current_library_kind() == "code":
            save_prompt_btn.setText("Save")
            pin_prompt_btn.setText("Pin")
            delete_prompt_btn.setText("Delete")
            clear_prompt_btn.setText("Clear")
            save_prompt_btn.setToolTip("Save the current Prompt box content as a reusable code snippet.")
            pin_prompt_btn.setToolTip("Pin or unpin the selected code snippet. Click again to toggle.")
            delete_prompt_btn.setToolTip("Delete the selected code snippet from the Code tab.")
            clear_prompt_btn.setToolTip("Clear recent unpinned code snippets while keeping saved and pinned code.")
            prompt_library_hint.setText(
                "Click to load. Double-click to run with Run My Code. Right-click to rename or edit tags."
            )
            return
        save_prompt_btn.setText("Save")
        pin_prompt_btn.setText("Pin")
        delete_prompt_btn.setText("Delete")
        clear_prompt_btn.setText("Clear")
        save_prompt_btn.setToolTip("Save the current Prompt box content as a reusable prompt.")
        pin_prompt_btn.setToolTip("Pin or unpin the selected prompt. Click again to toggle.")
        delete_prompt_btn.setToolTip("Delete the selected prompt from the Prompts tab.")
        clear_prompt_btn.setToolTip("Clear recent unpinned items in the Prompts tab while keeping saved and pinned prompts.")
        prompt_library_hint.setText(
            "Click to load. Double-click to send. Right-click to rename or edit tags."
        )

    def refresh_prompt_library():
        prompt_library_list.clear()
        for record in merged_prompt_records(prompt_library_state):
            label, color = format_library_item_label(record)
            item = QListWidgetItem()
            item.setText(label)
            item.setData(Qt.UserRole, record)
            item.setForeground(QColor(color))
            prompt_library_list.addItem(item)
        code_library_list.clear()
        for record in merged_code_records(prompt_library_state):
            label, color = format_library_item_label(record)
            item = QListWidgetItem()
            item.setText(label)
            item.setData(Qt.UserRole, record)
            item.setForeground(QColor(color))
            code_library_list.addItem(item)
        refresh_library_controls()

    def persist_prompt_library():
        save_prompt_library(prompt_library_state)

    def save_current_prompt(*_args):
        prompt_text = prompt.toPlainText().strip()
        if not prompt_text:
            set_status("Status: nothing to save", ok=False)
            append_log("Save library item skipped: prompt box is empty.")
            return
        if current_library_kind() == "code":
            code_text = strip_code_fences(prompt_text)
            if not code_text:
                set_status("Status: no code text to save", ok=False)
                append_log("Save code skipped: prompt box does not contain code.")
                return
            upsert_saved_code(prompt_library_state, code_text)
            persist_prompt_library()
            refresh_prompt_library()
            set_status("Status: code saved to library", ok=True)
            append_log(f"Saved code to library: {prompt_title(code_text)}")
            return
        upsert_saved_prompt(prompt_library_state, prompt_text)
        persist_prompt_library()
        refresh_prompt_library()
        set_status("Status: prompt saved to library", ok=True)
        append_log(f"Saved prompt to library: {prompt_text[:80]}")

    def toggle_pin_selected_prompt(*_args):
        records = selected_library_records()
        if not records:
            set_status("Status: no library item selected", ok=False)
            append_log("Pin library item skipped: no selection.")
            return
        should_pin = not all(bool(record.get("pinned", False)) for record in records)
        if current_library_kind() == "code":
            for record in records:
                set_code_pinned(prompt_library_state, record.get("code", ""), should_pin)
        else:
            for record in records:
                set_prompt_pinned(prompt_library_state, record.get("prompt", ""), should_pin)
        persist_prompt_library()
        refresh_prompt_library()
        set_status("Status: library updated", ok=True)
        append_log(f"{'Pinned' if should_pin else 'Unpinned'} {len(records)} {current_library_kind()} item(s).")

    def delete_selected_prompt(*_args):
        records = selected_library_records()
        if not records:
            set_status("Status: no library item selected", ok=False)
            append_log("Delete library item skipped: no selection.")
            return
        deleted_counts = {"saved": 0, "recent": 0, "built_in": 0}
        if current_library_kind() == "code":
            for record in records:
                source = str(record.get("source", "saved")).strip()
                remove_code_record(prompt_library_state, record.get("code", ""), source=source)
                if source in deleted_counts:
                    deleted_counts[source] += 1
        else:
            for record in records:
                source = str(record.get("source", "built_in")).strip()
                remove_prompt_record(prompt_library_state, record.get("prompt", ""), source=source)
                if source in deleted_counts:
                    deleted_counts[source] += 1
        persist_prompt_library()
        refresh_prompt_library()
        set_status(f"Status: deleted {len(records)} {current_library_kind()} item(s)", ok=True)
        append_log(
            f"Deleted {current_library_kind()} selection:"
            f" saved={deleted_counts['saved']}, recent={deleted_counts['recent']}, built-in={deleted_counts['built_in']}."
        )

    def clear_non_saved_prompts(*_args):
        if current_library_kind() == "code":
            clear_code_library(prompt_library_state, keep_saved=True, keep_pinned=True)
            persist_prompt_library()
            refresh_prompt_library()
            set_status("Status: cleared unpinned recent code", ok=True)
            append_log("Cleared code library down to saved and pinned items.")
            return
        clear_prompt_library(prompt_library_state, keep_saved=True, keep_pinned=True)
        persist_prompt_library()
        refresh_prompt_library()
        set_status("Status: cleared unpinned recent and built-in prompts", ok=True)
        append_log("Cleared prompt library down to saved and pinned prompts.")

    def model_is_available(requested_name: str, model_names: list[str]) -> bool:
        requested = str(requested_name or "").strip()
        if not requested:
            return False
        if requested in model_names:
            return True
        if ":" not in requested and f"{requested}:latest" in model_names:
            return True
        return False

    def refresh_models(*_args):
        base_url = base_url_edit.text().strip().rstrip("/")
        model_name = model_combo.currentText().strip()
        if not base_url:
            set_status("Status: missing base URL", ok=False)
            append_log("Refresh models failed: missing base URL.")
            return

        set_status("Status: refreshing local models...", ok=None)
        append_log(f"Refreshing local models from {base_url}")
        try:
            models = list_ollama_models(base_url)
        except Exception as exc:
            logger.exception("Refresh models failed for base_url=%s", base_url)
            set_status("Status: refresh models failed", ok=False)
            append_log(f"Refresh models failed: {exc}")
            return
        refresh_model_choices(models, preferred=model_name)
        saved_settings["base_url"] = base_url
        set_status(f"Status: found {len(models)} local models", ok=True)
        append_log(f"Loaded {len(models)} local models from Ollama")

    def test_connection(*_args):
        provider = "Local (Ollama-style)"
        base_url = base_url_edit.text().strip().rstrip("/")
        model_name = model_combo.currentText().strip()
        if not base_url or not model_name:
            set_status("Status: missing base URL or model name", ok=False)
            append_log("Connection test failed: missing base URL or model name.")
            return
        set_status(f"Status: testing {provider}...", ok=None)
        append_log(f"Testing connection for {provider} | {base_url} | model={model_name}")
        try:
            models = list_ollama_models(base_url)
        except Exception as exc:
            logger.exception("Connection test failed for base_url=%s model=%s", base_url, model_name)
            error_text = str(exc).strip()
            if "Could not connect to Ollama" in error_text:
                user_message = (
                    f"Could not connect to Ollama at {base_url}.\n\n"
                    "This usually means Ollama is not running, often after restarting the computer.\n\n"
                    "Start it in a terminal:\n"
                    "ollama serve\n\n"
                    "Then click Test again."
                )
                set_status("Status: Ollama is not running", ok=False)
                append_chat_message("assistant", user_message)
                append_log("Connection test failed: Ollama is not running.")
            else:
                set_status("Status: connection failed", ok=False)
                append_chat_message("assistant", f"Connection test failed:\n{error_text}")
                append_log(f"Connection test failed: {error_text}")
            return
        is_available = model_is_available(model_name, models)
        refresh_model_choices(models, preferred=model_name)
        saved_settings["base_url"] = base_url
        if is_available:
            saved_settings["model"] = model_name
        message = (
            f"Connected to Ollama. Model {model_name} is available."
            if is_available
            else f"Connected to Ollama, but model {model_name} was not found."
        )
        set_status(f"Status: {message}", ok=True)
        append_log(message)

    def save_settings(*_args):
        provider = provider_combo.currentText()
        base_url = base_url_edit.text().strip()
        model_name = model_combo.currentText().strip()
        if not base_url or not model_name:
            set_status("Status: missing base URL or model name", ok=False)
            append_log("Settings not saved: missing base URL or model name.")
            return
        saved_settings["provider"] = provider
        saved_settings["base_url"] = base_url.rstrip("/")
        saved_settings["model"] = model_name
        set_status(f"Status: ready to use {model_name}", ok=True)
        append_log(f"Saved local model settings for {provider} | {base_url} | model={model_name}")

    def pull_model(*_args):
        base_url = base_url_edit.text().strip().rstrip("/")
        model_name = model_combo.currentText().strip()
        if not model_name:
            set_status("Status: missing model name", ok=False)
            append_log("Model help skipped: missing model name.")
            return
        host_text = base_url or "http://127.0.0.1:11434"
        command = f"ollama pull {model_name}"
        append_chat_message(
            "assistant",
            "Setup\n\n"
            "**Step 1.** Install Ollama from https://ollama.com\n\n"
            "**Step 2.** Start Ollama if it is not already running.\n"
            "Run:\n"
            "```bash\n"
            "ollama serve\n"
            "```\n"
            "**Step 3.** Pull a model tag in a terminal.\n"
            "Example:\n"
            "```bash\n"
            "ollama pull nemotron-cascade-2:30b\n"
            "```\n"
            f"**Step 4.** Enter the tag in the Model field and click Test against {host_text}.\n\n"
            "Examples you can use in the Model field:\n"
            "- nemotron-cascade-2:30b\n"
            "- qwen2.5:7b\n"
            "- qwen3.5\n"
            "- qwen3-coder-next:latest\n\n"
            "Tips:\n"
            "- Smaller models are usually faster.\n"
            "- Larger models usually need more RAM or VRAM.\n"
            "- Use Performance Summary to compare model speed and behavior from your own local usage.\n"
            "- Use Telemetry Log if you want to inspect the raw advanced-user telemetry records.",
        )
        append_log(f"Opened Ollama setup help. Suggested terminal command: {command}")
        set_status("Status: Ollama setup help shown", ok=None)

    def unload_model(*_args):
        base_url = base_url_edit.text().strip().rstrip("/")
        model_name = model_combo.currentText().strip()
        if not base_url or not model_name:
            set_status("Status: missing base URL or model name", ok=False)
            append_log("Unload failed: missing base URL or model name.")
            return

        set_status(f"Status: unloading {model_name}...", ok=None)
        append_log(f"Unloading model {model_name}")
        try:
            unload_ollama_model(base_url, model_name)
        except Exception as exc:
            logger.exception("Unload failed for base_url=%s model=%s", base_url, model_name)
            set_status("Status: unload failed", ok=False)
            append_log(f"Unload failed: {exc}")
            return
        set_status(f"Status: unloaded {model_name}", ok=True)
        append_log(f"Unloaded model {model_name}")

    def run_summary_stats(*_args):
        layer_name = analysis_layer_combo.currentText().strip()
        prepared = prepare_tool_job(viewer, "summarize_intensity", {"layer_name": layer_name})
        run_prepared_tool_request(prepared, tool_name="summarize_intensity")

    def run_histogram(*_args):
        layer_name = analysis_layer_combo.currentText().strip()
        prepared = prepare_tool_job(
            viewer,
            "plot_histogram",
            {"layer_name": layer_name, "bins": histogram_bins_edit.text().strip() or "64"},
        )
        run_prepared_tool_request(prepared, tool_name="plot_histogram")

    def run_layer_comparison(*_args):
        layer_name_a = compare_layer_a_combo.currentText().strip()
        layer_name_b = compare_layer_b_combo.currentText().strip()
        if layer_name_a and layer_name_b and layer_name_a == layer_name_b:
            set_status("Status: choose 2 different image layers", ok=False)
            append_log("Analysis panel comparison skipped: identical image layers selected.")
            return
        equal_var = compare_test_combo.currentText().strip() != "Welch t-test"
        prepared = prepare_tool_job(
            viewer,
            "compare_image_layers_ttest",
            {
                "layer_name_a": layer_name_a,
                "layer_name_b": layer_name_b,
                "equal_var": equal_var,
            },
        )
        run_prepared_tool_request(prepared, tool_name="compare_image_layers_ttest")

    def send_message():
        text = prompt.toPlainText().strip()
        if not text:
            return
        nonlocal session_memory_state
        turn_id = uuid.uuid4().hex
        request_started_at = time.perf_counter()
        current_profile = selected_layer_profile()
        session_memory_state = update_session_goal(session_memory_state, text)
        session_memory_state = set_active_dataset_focus(
            session_memory_state,
            "" if not isinstance(current_profile, dict) else str(current_profile.get("layer_name", "")).strip(),
        )
        session_memory_state, promoted_ids = promote_from_user_turn(session_memory_state, text, current_profile)
        if promoted_ids:
            append_log(f"Promoted {len(promoted_ids)} provisional memory item(s) from user follow-up.")
        persist_session_memory()
        manual_code = extract_manual_code_submission(text)
        if manual_code:
            upsert_recent_code(prompt_library_state, manual_code)
        else:
            upsert_recent_prompt(prompt_library_state, text)
        persist_prompt_library()
        refresh_prompt_library()
        append_chat_message("user", text)
        prompt.clear()
        append_log(f"Queued message: {text}")
        if manual_code:
            validation_errors = preflight_generated_code(manual_code)
            code_message = "Manual code captured from the prompt. Review it, then click Run Code."
            if validation_errors:
                set_pending_code(
                    manual_code,
                    message=code_message,
                    runnable=False,
                    label="Pending code: blocked by validation",
                )
                append_chat_message(
                    "assistant",
                    "Pasted code was rejected by local validation:\n"
                    + "\n".join(f"- {error}" for error in validation_errors)
                    + "\n\nFix the code in the prompt and send it again with `/code` or `run code`.\n\n"
                    + format_code_block(manual_code),
                )
                append_log("Rejected pasted manual code after local validation.")
                set_status("Status: pasted code rejected", ok=False)
                record_telemetry(
                    "turn_completed",
                    {
                        "turn_id": turn_id,
                        "model": "",
                        "base_url": "",
                        "prompt_hash": prompt_hash(text),
                        "prompt_category": "manual_code",
                        "response_action": "manual_code_blocked",
                        "pending_code_generated": True,
                        **selected_layer_snapshot(),
                    },
                )
                return
            set_pending_code(manual_code, message=code_message)
            pending_code["turn_id"] = turn_id
            pending_code["model"] = "manual"
            append_chat_message("assistant", f"{code_message}\n{format_code_block(manual_code)}")
            append_log("Queued pasted manual code; waiting for Run Code approval.")
            set_status("Status: manual code ready to run", ok=None)
            record_telemetry(
                "turn_completed",
                {
                    "turn_id": turn_id,
                    "model": "manual",
                    "base_url": "",
                    "prompt_hash": prompt_hash(text),
                    "prompt_category": "manual_code",
                    "response_action": "manual_code",
                    "pending_code_generated": True,
                    **selected_layer_snapshot(),
                },
            )
            return
        base_url = base_url_edit.text().strip().rstrip("/") or str(saved_settings["base_url"]).rstrip("/")
        model_name = model_combo.currentText().strip() or str(saved_settings["model"]).strip()
        if not base_url or not model_name:
            append_chat_message("assistant", "Model settings are incomplete. Fill in Model Connection first.")
            set_status("Status: missing saved model settings", ok=False)
            return
        saved_settings["base_url"] = base_url
        saved_settings["model"] = model_name
        turn_prompt_hash = prompt_hash(text)
        turn_prompt_category = categorize_prompt(text)
        turn_layer_snapshot = selected_layer_snapshot()
        record_telemetry(
            "turn_started",
            {
                "turn_id": turn_id,
                "model": model_name,
                "base_url": base_url,
                "prompt_hash": turn_prompt_hash,
                "prompt_chars": len(text),
                "prompt_category": turn_prompt_category,
                **turn_layer_snapshot,
            },
        )

        start_wait_indicator(phase="Thinking")
        set_status(f"Status: sending to {model_name}", ok=None)

        @thread_worker(ignore_errors=True)
        def run_chat():
            viewer_payload = layer_context_json(viewer)
            return chat_ollama(
                base_url,
                model_name,
                system_prompt=assistant_system_prompt(),
                user_payload={
                    "viewer_context": viewer_payload,
                    "session_memory": build_session_memory_payload(session_memory_state, viewer_payload.get("selected_layer_profile")),
                    "user_message": text,
                },
                options=dict(generation_defaults),
                timeout=1800,
            )

        worker = run_chat()
        active_workers.append(worker)

        def finish():
            stop_wait_indicator()
            if worker in active_workers:
                active_workers.remove(worker)

        def on_returned(reply):
            try:
                parsed = normalize_model_response(reply)
            except Exception as exc:
                logger.exception("Failed to normalize model response.")
                replace_last_assistant(f"Response parse failed:\n{exc}\n\nRaw reply:\n{reply}")
                set_status("Status: response parse failed", ok=False)
                append_log("Failed to parse model response.")
                finish()
                return

            action = str(parsed.get("action", "reply")).strip().lower()

            if action == "tool":
                tool_name = str(parsed.get("tool", "")).strip()
                arguments = parsed.get("arguments", {})
                tool_message = str(parsed.get("message", "")).strip()
                prepared = prepare_tool_job(viewer, tool_name, arguments if isinstance(arguments, dict) else {})
                if prepared.get("mode") == "immediate":
                    result_message = str(prepared.get("message", ""))
                    refresh_context()
                    replace_last_assistant(f"{tool_message}\n{result_message}" if tool_message else result_message)
                    latency_ms = int((time.perf_counter() - request_started_at) * 1000)
                    last_turn_metrics.update(
                        {"turn_id": turn_id, "model": model_name, "action": "tool", "prompt_hash": turn_prompt_hash}
                    )
                    record_telemetry(
                        "turn_completed",
                        {
                            "turn_id": turn_id,
                            "model": model_name,
                            "base_url": base_url,
                            "prompt_hash": turn_prompt_hash,
                            "prompt_category": turn_prompt_category,
                            "response_action": "tool",
                            "tool_name": tool_name,
                            "tool_success": True,
                            "latency_ms": latency_ms,
                            **turn_layer_snapshot,
                        },
                    )
                    remember_assistant_outcome(
                        tool_message or result_message,
                        target_type="tool_result",
                        target_profile=selected_layer_profile(),
                    )
                    append_log(f"Tool executed: {tool_name}")
                    set_status(f"Status: connected to {model_name}", ok=True)
                    append_log(f"Received response from {model_name}")
                    finish()
                    return

                job = prepared["job"]
                set_status(f"Status: running tool {tool_name}", ok=None)

                @thread_worker(ignore_errors=True)
                def run_backend_tool():
                    return run_tool_job(job)

                tool_worker = run_backend_tool()
                active_workers.append(tool_worker)

                def tool_finish():
                    if tool_worker in active_workers:
                        active_workers.remove(tool_worker)
                    set_status(f"Status: connected to {model_name}", ok=True)
                    append_log(f"Received response from {model_name}")
                    finish()

                def tool_returned(tool_result):
                    nonlocal session_memory_state
                    result_message = apply_tool_job_result(viewer, tool_result)
                    refresh_context()
                    replace_last_assistant(f"{tool_message}\n{result_message}" if tool_message else result_message)
                    latency_ms = int((time.perf_counter() - request_started_at) * 1000)
                    last_turn_metrics.update(
                        {"turn_id": turn_id, "model": model_name, "action": "tool", "prompt_hash": turn_prompt_hash}
                    )
                    record_telemetry(
                        "turn_completed",
                        {
                            "turn_id": turn_id,
                            "model": model_name,
                            "base_url": base_url,
                            "prompt_hash": turn_prompt_hash,
                            "prompt_category": turn_prompt_category,
                            "response_action": "tool",
                            "tool_name": tool_name,
                            "tool_success": True,
                            "latency_ms": latency_ms,
                            **turn_layer_snapshot,
                        },
                    )
                    session_memory_state = approve_items(session_memory_state, last_memory_candidate_ids)
                    persist_session_memory()
                    remember_assistant_outcome(
                        tool_message or result_message,
                        target_type="tool_result",
                        target_profile=selected_layer_profile(),
                        state="approved",
                    )
                    append_log(f"Tool executed: {tool_name}")
                    tool_finish()

                def tool_error(*args):
                    error_text = format_worker_error(*args)
                    logger.exception("Tool execution failed: %s | %s", tool_name, error_text)
                    replace_last_assistant(f"Tool execution failed: {error_text}")
                    latency_ms = int((time.perf_counter() - request_started_at) * 1000)
                    last_turn_metrics.update(
                        {"turn_id": turn_id, "model": model_name, "action": "tool", "prompt_hash": turn_prompt_hash}
                    )
                    record_telemetry(
                        "turn_completed",
                        {
                            "turn_id": turn_id,
                            "model": model_name,
                            "base_url": base_url,
                            "prompt_hash": turn_prompt_hash,
                            "prompt_category": turn_prompt_category,
                            "response_action": "tool",
                            "tool_name": tool_name,
                            "tool_success": False,
                            "latency_ms": latency_ms,
                            "error": error_text,
                            **turn_layer_snapshot,
                        },
                    )
                    append_log(f"Tool execution failed: {tool_name} | {error_text}")
                    set_status("Status: tool execution failed", ok=False)
                    if tool_worker in active_workers:
                        active_workers.remove(tool_worker)
                    finish()

                tool_worker.returned.connect(tool_returned)
                tool_worker.errored.connect(tool_error)
                tool_worker.start()
                return

            if action == "code":
                code_text = str(parsed.get("code", "")).strip()
                code_message = str(parsed.get("message", "")).strip() or "Generated napari code. Review it, then click Run Code."
                validation_errors = preflight_generated_code(code_text)
                if validation_errors:
                    set_pending_code(
                        code_text,
                        message=code_message,
                        runnable=False,
                        label="Pending code: blocked by validation",
                    )
                    pending_code["turn_id"] = turn_id
                    pending_code["model"] = model_name
                    replace_last_assistant(
                        "Generated code was rejected by local validation:\n"
                        + "\n".join(f"- {error}" for error in validation_errors)
                        + "\n\nReview or copy the generated code below, then ask the assistant to regenerate or fix it.\n\n"
                        + format_code_block(code_text)
                    )
                    append_log("Rejected generated code after local validation.")
                    set_status("Status: generated code rejected", ok=False)
                    finish()
                    return
                set_pending_code(code_text, message=code_message)
                pending_code["turn_id"] = turn_id
                pending_code["model"] = model_name
                replace_last_assistant(f"{code_message}\n{format_code_block(code_text)}")
                latency_ms = int((time.perf_counter() - request_started_at) * 1000)
                last_turn_metrics.update(
                    {"turn_id": turn_id, "model": model_name, "action": "code", "prompt_hash": turn_prompt_hash}
                )
                record_telemetry(
                    "turn_completed",
                    {
                        "turn_id": turn_id,
                        "model": model_name,
                        "base_url": base_url,
                        "prompt_hash": turn_prompt_hash,
                        "prompt_category": turn_prompt_category,
                        "response_action": "code",
                        "pending_code_generated": True,
                        "latency_ms": latency_ms,
                        **turn_layer_snapshot,
                    },
                )
                remember_assistant_outcome(code_message, target_type="recommendation", target_profile=selected_layer_profile())
                set_status("Status: code generated, awaiting approval", ok=None)
                append_log("Generated pending napari code; waiting for approval.")
                finish()
                return

            message_text = str(parsed.get("message", reply)).strip() or "[empty response]"
            replace_last_assistant(message_text)
            latency_ms = int((time.perf_counter() - request_started_at) * 1000)
            last_turn_metrics.update(
                {"turn_id": turn_id, "model": model_name, "action": "reply", "prompt_hash": turn_prompt_hash}
            )
            record_telemetry(
                "turn_completed",
                {
                    "turn_id": turn_id,
                    "model": model_name,
                    "base_url": base_url,
                    "prompt_hash": turn_prompt_hash,
                    "prompt_category": turn_prompt_category,
                    "response_action": "reply",
                    "latency_ms": latency_ms,
                    **turn_layer_snapshot,
                },
            )
            remember_assistant_outcome(
                message_text,
                target_type="recommendation",
                target_profile=selected_layer_profile(),
            )
            set_status(f"Status: connected to {model_name}", ok=True)
            append_log(f"Received response from {model_name}")
            finish()

        worker.returned.connect(on_returned)

        def request_failed(*args):
            error_text = format_worker_error(*args)
            logger.exception("Chat request failed: %s", error_text)
            replace_last_assistant(f"Request failed: {error_text}")
            latency_ms = int((time.perf_counter() - request_started_at) * 1000)
            last_turn_metrics.update(
                {"turn_id": turn_id, "model": model_name, "action": "error", "prompt_hash": turn_prompt_hash}
            )
            record_telemetry(
                "turn_completed",
                {
                    "turn_id": turn_id,
                    "model": model_name,
                    "base_url": base_url,
                    "prompt_hash": turn_prompt_hash,
                    "prompt_category": turn_prompt_category,
                    "response_action": "error",
                    "latency_ms": latency_ms,
                    "error": error_text,
                    **turn_layer_snapshot,
                },
            )
            set_status("Status: request failed", ok=False)
            append_log(f"Chat request failed: {error_text}")
            finish()

        worker.errored.connect(request_failed)
        worker.start()

    def copy_pending_code(*_args):
        code_text = pending_code["code"]
        if not code_text:
            set_status("Status: no pending code to copy", ok=False)
            append_log("Copy code skipped: no pending code available.")
            return
        QApplication.clipboard().setText(code_text)
        append_log("Copied pending napari code to clipboard.")
        set_status("Status: pending code copied", ok=True)

    def run_pending_code(*_args):
        nonlocal session_memory_state
        code_text = pending_code["code"]
        if not code_text:
            set_status("Status: no pending code to run", ok=False)
            append_log("Run code skipped: no pending code available.")
            return
        if not run_code_text(
            code_text,
            code_label="approved napari code",
            turn_id=pending_code.get("turn_id", ""),
            model_name=pending_code.get("model", ""),
            code_source="assistant",
            disable_pending_buttons=True,
        ):
            return
        session_memory_state = approve_items(session_memory_state, last_memory_candidate_ids)
        persist_session_memory()
        remember_assistant_outcome(
            pending_code.get("message") or "Approved napari code executed.",
            target_type="code_result",
            target_profile=selected_layer_profile(),
            state="approved",
        )
        set_pending_code()

    def run_code_text(
        code_text: str,
        *,
        code_label: str,
        turn_id: str = "",
        model_name: str = "",
        code_source: str = "assistant",
        disable_pending_buttons: bool = False,
    ) -> bool:
        validation_errors = preflight_generated_code(code_text)
        if validation_errors:
            append_chat_message(
                "assistant",
                f"{code_label.capitalize()} failed local validation:\n" + "\n".join(f"- {error}" for error in validation_errors),
            )
            append_log(f"{code_label.capitalize()} blocked by local validation.")
            set_status(f"Status: {code_label} blocked", ok=False)
            return False

        set_status(f"Status: running {code_label}", ok=None)
        append_log(f"Running {code_label}.")
        if disable_pending_buttons:
            run_code_btn.setEnabled(False)
            copy_code_btn.setEnabled(False)
        stdout_buffer = io.StringIO()
        background_state = {"launched": False, "active_labels": []}

        def run_in_background(compute_fn, apply_fn, error_fn=None, label: str = "Background job"):
            job_label = " ".join(str(label or "Background job").split()).strip() or "Background job"
            if not callable(compute_fn):
                raise TypeError("run_in_background requires a callable compute_fn.")
            if not callable(apply_fn):
                raise TypeError("run_in_background requires a callable apply_fn.")

            @thread_worker(ignore_errors=True)
            def worker_job():
                return compute_fn()

            worker = worker_job()
            active_workers.append(worker)
            background_state["launched"] = True
            background_state["active_labels"].append(job_label)
            append_log(f"Started background job: {job_label}")
            set_status(f"Status: running {job_label}", ok=None)

            def worker_done():
                if worker in active_workers:
                    active_workers.remove(worker)

            def worker_returned(result):
                try:
                    apply_fn(result)
                except Exception as exc:
                    logger.exception("Background apply failed: %s", job_label)
                    error_text = format_code_execution_error(exc)
                    append_chat_message("assistant", f"{job_label} apply step failed:\n{error_text}")
                    append_log(f"Background apply failed: {job_label} | {error_text}")
                    set_status(f"Status: {job_label} apply failed", ok=False)
                    worker_done()
                    return
                append_log(f"Completed background job: {job_label}")
                set_status(f"Status: {job_label} completed", ok=True)
                worker_done()

            def worker_error(*args):
                error = next((value for value in args if isinstance(value, BaseException)), None)
                error_text = format_worker_error(*args)
                logger.exception("Background job failed: %s | %s", job_label, error_text)
                if callable(error_fn):
                    try:
                        error_fn(error or RuntimeError(error_text))
                    except Exception as exc:
                        logger.exception("Background error handler failed: %s", job_label)
                        append_chat_message("assistant", f"{job_label} error handler failed:\n{exc}")
                else:
                    append_chat_message("assistant", f"{job_label} failed:\n{error_text}")
                append_log(f"Background job failed: {job_label} | {error_text}")
                set_status(f"Status: {job_label} failed", ok=False)
                worker_done()

            worker.returned.connect(worker_returned)
            worker.errored.connect(worker_error)
            worker.start()
            return worker

        namespace = {
            "__builtins__": __builtins__,
            "napari": napari,
            "np": np,
            "numpy": np,
            "viewer": viewer,
            "selected_layer": None if viewer is None else viewer.layers.selection.active,
            "run_in_background": run_in_background,
        }
        try:
            with redirect_stdout(stdout_buffer):
                exec(compile(code_text, "<napari-chat-assistant>", "exec"), namespace, namespace)
        except Exception as exc:
            logger.exception("%s failed during execution.", code_label.capitalize())
            error_text = format_code_execution_error(exc)
            append_chat_message("assistant", f"{code_label.capitalize()} failed:\n{error_text}")
            record_telemetry(
                "code_execution",
                {
                    "turn_id": turn_id,
                    "model": model_name,
                    "code_source": code_source,
                    "success": False,
                    "error": error_text,
                },
            )
            append_log(f"{code_label.capitalize()} failed: {error_text}")
            set_status(f"Status: {code_label} failed", ok=False)
            if disable_pending_buttons and pending_code["code"]:
                run_code_btn.setEnabled(True)
                copy_code_btn.setEnabled(True)
            return False

        refresh_context()
        stdout_text = stdout_buffer.getvalue().strip()
        result_message = f"{code_label.capitalize()} executed."
        if background_state["launched"]:
            labels = ", ".join(background_state["active_labels"])
            result_message = f"{code_label.capitalize()} launched background job."
            if labels:
                result_message = f"{result_message}\nJob: {labels}"
        if stdout_text:
            result_message = f"{result_message}\nOutput:\n{stdout_text}"
        append_chat_message("assistant", result_message)
        record_telemetry(
            "code_execution",
            {
                "turn_id": turn_id,
                "model": model_name,
                "code_source": code_source,
                "success": True,
                "stdout_chars": len(stdout_text),
            },
        )
        append_log(f"{code_label.capitalize()} executed successfully.")
        set_status(f"Status: {code_label} executed", ok=True)
        return True

    def run_prompt_code(*_args):
        code_text = strip_code_fences(prompt.toPlainText())
        if not code_text:
            set_status("Status: no prompt code to run", ok=False)
            append_log("Run My Code skipped: prompt box is empty.")
            return
        upsert_recent_code(prompt_library_state, code_text)
        persist_prompt_library()
        refresh_prompt_library()
        append_chat_message("user", f"/my-code\n{format_code_block(code_text)}")
        prompt.clear()
        run_code_text(code_text, code_label="your code", model_name="manual", code_source="user")

    def load_library_prompt(item: QListWidgetItem):
        record = item.data(Qt.UserRole) or {}
        prompt.setPlainText(str(record.get("prompt", "")).strip())
        prompt.setFocus()
        append_log(f"Loaded prompt from library: {record.get('title', 'Untitled Prompt')}")
        set_status("Status: prompt loaded from library", ok=None)

    def send_library_prompt(item: QListWidgetItem):
        record = item.data(Qt.UserRole) or {}
        prompt_text = str(record.get("prompt", "")).strip()
        if not prompt_text:
            set_status("Status: selected prompt is empty", ok=False)
            append_log("Send prompt skipped: selected library prompt is empty.")
            return
        prompt.setPlainText(prompt_text)
        append_log(f"Sending prompt directly from library: {record.get('title', 'Untitled Prompt')}")
        set_status("Status: sending prompt from library", ok=None)
        send_message()

    def load_library_code(item: QListWidgetItem):
        record = item.data(Qt.UserRole) or {}
        code_text = str(record.get("code", "")).strip()
        prompt.setPlainText(code_text)
        prompt.setFocus()
        append_log(f"Loaded code from library: {record.get('title', 'Untitled Code')}")
        set_status("Status: code loaded from library", ok=None)

    def run_library_code(item: QListWidgetItem):
        record = item.data(Qt.UserRole) or {}
        code_text = str(record.get("code", "")).strip()
        if not code_text:
            set_status("Status: selected code is empty", ok=False)
            append_log("Run code skipped: selected library code is empty.")
            return
        prompt.setPlainText(code_text)
        append_log(f"Running code directly from library: {record.get('title', 'Untitled Code')}")
        set_status("Status: running code from library", ok=None)
        run_prompt_code()

    def rename_library_item(record: dict):
        item_id = str(record.get("id", "")).strip()
        if not item_id:
            set_status("Status: item cannot be renamed", ok=False)
            append_log("Rename skipped: selected library item has no stable id.")
            return
        current_title = str(record.get("title", "")).strip() or (
            "Untitled Code" if current_library_kind() == "code" else "Untitled Prompt"
        )
        title, accepted = QInputDialog.getText(
            root,
            "Rename Library Item",
            f"New name for this {current_library_item_name()}:",
            text=current_title,
        )
        if not accepted:
            return
        clean_title = " ".join(str(title or "").strip().split())
        if not clean_title:
            set_status("Status: title cannot be empty", ok=False)
            append_log("Rename skipped: library item title was empty.")
            return
        update_record_title(prompt_library_state, kind=current_library_kind(), item_id=item_id, title=clean_title)
        persist_prompt_library()
        refresh_prompt_library()
        append_log(f"Renamed {current_library_item_name()} to: {clean_title}")
        set_status("Status: library item renamed", ok=True)

    def edit_library_item_tags(record: dict):
        item_id = str(record.get("id", "")).strip()
        if not item_id:
            set_status("Status: tags cannot be edited", ok=False)
            append_log("Edit tags skipped: selected library item has no stable id.")
            return
        current_tags = ", ".join(str(tag).strip() for tag in record.get("tags", []) if str(tag).strip())
        tags_text, accepted = QInputDialog.getText(
            root,
            "Edit Tags",
            f"Comma-separated tags for this {current_library_item_name()}:",
            text=current_tags,
        )
        if not accepted:
            return
        update_record_tags(prompt_library_state, kind=current_library_kind(), item_id=item_id, tags=tags_text)
        persist_prompt_library()
        refresh_prompt_library()
        append_log(f"Updated tags for {current_library_item_name()}: {tags_text or '[none]'}")
        set_status("Status: library item tags updated", ok=True)

    def show_library_context_menu(position):
        item = current_library_list().itemAt(position)
        if item is None:
            return
        record = item.data(Qt.UserRole) or {}
        if not isinstance(record, dict):
            return
        menu = QMenu(current_library_list())
        if str(record.get("source", "")).strip() != "built_in":
            rename_action = menu.addAction("Rename")
            tags_action = menu.addAction("Edit Tags")
        else:
            rename_action = None
            tags_action = None
        chosen = menu.exec_(current_library_list().viewport().mapToGlobal(position))
        if chosen is rename_action:
            rename_library_item(record)
        elif chosen is tags_action:
            edit_library_item_tags(record)

    def cleanup_workers(*_args):
        for worker in list(active_workers):
            cancel = getattr(worker, "_assistant_cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    pass
            try:
                worker.quit()
            except Exception:
                pass
        active_workers.clear()

    test_btn.clicked.connect(test_connection)
    save_btn.clicked.connect(save_settings)
    pull_btn.clicked.connect(pull_model)
    unload_btn.clicked.connect(unload_model)
    app_log_btn.clicked.connect(show_app_log)
    crash_log_btn.clicked.connect(show_crash_log)
    run_code_btn.clicked.connect(run_pending_code)
    run_my_code_btn.clicked.connect(run_prompt_code)
    copy_code_btn.clicked.connect(copy_pending_code)
    reject_memory_btn.clicked.connect(reject_last_memory)
    help_btn.clicked.connect(show_help_tips)
    telemetry_summary_btn.clicked.connect(show_telemetry_summary)
    telemetry_view_btn.clicked.connect(show_telemetry_viewer)
    telemetry_reset_btn.clicked.connect(reset_telemetry_log)
    telemetry_toggle.toggled.connect(toggle_telemetry)
    summary_stats_btn.clicked.connect(run_summary_stats)
    histogram_btn.clicked.connect(run_histogram)
    compare_btn.clicked.connect(run_layer_comparison)
    library_tabs.currentChanged.connect(lambda *_args: refresh_library_controls())
    prompt_library_list.itemClicked.connect(load_library_prompt)
    prompt_library_list.itemDoubleClicked.connect(send_library_prompt)
    prompt_library_list.customContextMenuRequested.connect(show_library_context_menu)
    code_library_list.itemClicked.connect(load_library_code)
    code_library_list.itemDoubleClicked.connect(run_library_code)
    code_library_list.customContextMenuRequested.connect(show_library_context_menu)
    save_prompt_btn.clicked.connect(save_current_prompt)
    pin_prompt_btn.clicked.connect(toggle_pin_selected_prompt)
    delete_prompt_btn.clicked.connect(delete_selected_prompt)
    clear_prompt_btn.clicked.connect(clear_non_saved_prompts)
    prompt_font_down_btn.clicked.connect(lambda *_args: adjust_prompt_library_font(-1))
    prompt_font_up_btn.clicked.connect(lambda *_args: adjust_prompt_library_font(1))
    prompt.sendRequested.connect(send_message)
    refresh_btn.clicked.connect(refresh_context)
    wait_timer.timeout.connect(tick_wait_indicator)
    root.destroyed.connect(cleanup_workers)

    refresh_telemetry_controls()
    refresh_context()
    set_pending_code()
    refresh_models()
    refresh_prompt_library()
    append_log(f"Assistant log: {APP_LOG_PATH}")
    append_log(f"Crash log: {CRASH_LOG_PATH}")
    if ui_state.get("telemetry_enabled", False):
        append_log(f"Telemetry log: {TELEMETRY_LOG_PATH}")
    append_log(f"Prompt library path: {prompt_library_path()}")
    append_log(f"Session memory path: {session_memory_path()}")
    if not ui_state.get("welcome_dismissed", False):
        append_chat_message(
            "assistant",
            "**Welcome**\n"
            "👋 Welcome to Local Chat Assistant.\n"
            "🤖 Connect a local model for chat, tool use, and generated napari code inside the viewer.\n"
            "⌨️ Use the Prompt box for normal requests, or paste your own Python and click `Run My Code` to execute it directly without opening QtConsole.\n"
            "▶️ Use `Run Code` for assistant-generated code after review.\n"
            "🧭 Current Context shows your open layers and the selected layer.\n"
            "📚 Library tabs keep reusable prompts and code close to the workflow.\n"
            "📝 Action Log tracks local actions. Telemetry is optional and stays off unless you enable it.\n\n"
            "Ask about your selected layer, thresholding, CLAHE, measurements, histograms, comparisons, or code.\n\n"
            "Follow for updates: https://x.com/viralvector",
        )
        ui_state["welcome_dismissed"] = True
        save_ui_state(ui_state)
    append_log("Assistant panel initialized.")
    return root
