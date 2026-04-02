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
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_chat_assistant.agent.client import chat_ollama, list_ollama_models, load_ollama_model, unload_ollama_model
from napari_chat_assistant.agent.code_validation import (
    ValidationMode,
    ValidationReport,
    build_code_repair_context,
    normalize_generated_code_if_needed,
    validate_generated_code,
)
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
from napari_chat_assistant.agent.prompt_routing import route_local_workflow_prompt
from napari_chat_assistant.agent.sam2_backend import (
    discover_sam2_setup,
    get_sam2_backend_status,
    list_sam2_checkpoints,
    list_sam2_configs,
    sam2_config_from_ui_state,
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
from napari_chat_assistant.agent.template_library import template_library_payload
from napari_chat_assistant.agent.telemetry_summary import (
    format_telemetry_summary,
    load_telemetry_events,
    read_telemetry_tail,
    summarize_telemetry_events,
)
from napari_chat_assistant.agent.tools import ASSISTANT_TOOL_NAMES, assistant_system_prompt, next_output_name
from napari_chat_assistant.agent.ui_help import answer_ui_question
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

    model_bar = QWidget()
    model_bar_layout = QHBoxLayout(model_bar)
    model_bar_layout.setContentsMargins(0, 0, 0, 0)

    model_label = QLabel("Model:")
    model_bar_layout.addWidget(model_label)

    model_combo = QComboBox()
    model_combo.setEditable(True)
    model_combo.setMinimumContentsLength(18)
    model_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    model_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    model_bar_layout.addWidget(model_combo, 1)

    connection_status = QLabel("Status: not connected")
    connection_status.setWordWrap(True)
    connection_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    connection_status.setStyleSheet("QLabel { background: #243447; color: #f5f7fa; padding: 6px; }")
    model_bar_layout.addWidget(connection_status, 2)

    save_btn = QPushButton("Load")
    save_btn.setToolTip("Load the selected model into Ollama now so the first chat request starts faster.")
    unload_btn = QPushButton("Unload")
    unload_btn.setToolTip("Unload the selected model from Ollama and free RAM or VRAM.")
    connection_toggle_btn = QPushButton("Connection")
    connection_toggle_btn.setCheckable(True)
    connection_toggle_btn.setToolTip("Show or hide connection details such as Base URL, Test, and Setup.")
    model_bar_layout.addWidget(save_btn)
    model_bar_layout.addWidget(unload_btn)
    model_bar_layout.addWidget(connection_toggle_btn)
    layout.addWidget(model_bar)

    connection_details = QGroupBox("Connection Details")
    connection_details.setVisible(False)
    connection_details_layout = QFormLayout(connection_details)
    provider_combo = QComboBox()
    provider_combo.addItems(["Local (Ollama-style)"])
    provider_combo.setEnabled(False)
    connection_details_layout.addRow("Provider:", provider_combo)
    base_url_edit = QLineEdit("http://127.0.0.1:11434")
    connection_details_layout.addRow("Base URL:", base_url_edit)
    model_hint = QLabel("Type an Ollama model tag or pick one already installed locally.")
    model_hint.setWordWrap(True)
    model_hint.setStyleSheet("QLabel { color: #9fb3c8; padding: 2px 0 6px 0; }")
    connection_details_layout.addRow(model_hint)
    config_btn_row = QWidget()
    config_btn_layout = QHBoxLayout(config_btn_row)
    test_btn = QPushButton("Test")
    test_btn.setToolTip("Check that Ollama is reachable and confirm the selected model tag is installed locally.")
    config_btn_layout.addWidget(test_btn)
    pull_btn = QPushButton("Setup")
    pull_btn.setToolTip("Show Ollama setup steps, including how to start Ollama and pull the selected model tag.")
    config_btn_layout.addWidget(pull_btn)
    connection_details_layout.addRow(config_btn_row)
    layout.addWidget(connection_details)

    splitter = QSplitter(Qt.Horizontal)
    layout.addWidget(splitter, 1)

    left_panel = QWidget()
    left_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    left_panel.setMinimumWidth(360)
    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(0, 0, 0, 0)

    right_panel = QWidget()
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)

    context_group = QGroupBox("Layer Context")
    context_layout = QVBoxLayout(context_group)
    context_layout.setContentsMargins(6, 6, 6, 6)
    context_tabs = QTabWidget()
    context_layout.addWidget(context_tabs)

    context_summary_tab = QWidget()
    context_summary_layout = QVBoxLayout(context_summary_tab)
    context_summary_layout.setContentsMargins(0, 0, 0, 0)
    context_summary_box = QTextEdit()
    context_summary_box.setReadOnly(True)
    context_summary_box.setAcceptRichText(False)
    context_summary_box.setPlaceholderText("Copyable layer context will appear here.")
    context_summary_box.setMinimumHeight(120)
    context_summary_box.setMaximumHeight(180)
    context_summary_box.setStyleSheet(
        "QTextEdit { background: #101820; color: #e6edf3; border: 1px solid #22304a; padding: 8px; }"
    )
    context_summary_layout.addWidget(context_summary_box)

    context_layers_tab = QWidget()
    context_layers_layout = QVBoxLayout(context_layers_tab)
    context_layers_layout.setContentsMargins(0, 0, 0, 0)
    context_layers_list = QListWidget()
    context_layers_list.setSelectionMode(QAbstractItemView.NoSelection)
    context_layers_list.setFocusPolicy(Qt.NoFocus)
    context_layers_list.setStyleSheet("QListWidget { background: #101820; color: #d6deeb; border: 1px solid #22304a; }")
    context_layers_layout.addWidget(context_layers_list)

    context_tabs.addTab(context_summary_tab, "Summary")
    context_tabs.addTab(context_layers_tab, "Layers")
    left_layout.addWidget(context_group, 0)

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
    prompt_library_list.setStyleSheet(
        "QListWidget { background: #101820; color: #d6deeb; border: 1px solid #22304a; } "
        "QListWidget::item:selected { background: #1d2a44; color: #e8f1ff; }"
    )
    code_library_list = QListWidget()
    code_library_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    code_library_list.setContextMenuPolicy(Qt.CustomContextMenu)
    code_library_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    code_library_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    code_library_list.setWordWrap(True)
    code_library_list.setStyleSheet(
        "QListWidget { background: #101820; color: #d6deeb; border: 1px solid #22304a; } "
        "QListWidget::item:selected { background: #1d2a44; color: #e8f1ff; }"
    )
    prompt_library_font = prompt_library_list.font()
    if prompt_library_font.pointSize() > 0:
        prompt_library_font.setPointSize(prompt_library_font.pointSize() + 1)
        prompt_library_list.setFont(prompt_library_font)
        code_library_list.setFont(prompt_library_font)
    template_tab = QWidget()
    template_tab_layout = QVBoxLayout(template_tab)
    template_tab_layout.setContentsMargins(0, 0, 0, 0)
    template_splitter = QSplitter(Qt.Horizontal)
    template_tree = QTreeWidget()
    template_tree.setHeaderHidden(True)
    template_tree.setMinimumWidth(180)
    template_tree.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    template_tree.setStyleSheet("QTreeWidget { background: #101820; color: #d6deeb; border: 1px solid #22304a; }")
    template_preview = QTextEdit()
    template_preview.setReadOnly(True)
    template_preview.setAcceptRichText(False)
    template_preview.setPlaceholderText("Select a template to preview its description and code.")
    template_preview.setStyleSheet(
        "QTextEdit { background: #0b1021; color: #d6deeb; border: 1px solid #22304a; padding: 10px; }"
    )
    template_splitter.addWidget(template_tree)
    template_splitter.addWidget(template_preview)
    template_splitter.setStretchFactor(0, 0)
    template_splitter.setStretchFactor(1, 1)
    template_tab_layout.addWidget(template_splitter, 1)
    template_btn_row = QWidget()
    template_btn_layout = QHBoxLayout(template_btn_row)
    template_load_btn = QPushButton("Load Template")
    template_load_btn.setToolTip("Load the selected built-in template into the Prompt box for refinement.")
    template_run_btn = QPushButton("Run Template")
    template_run_btn.setToolTip("Load the selected built-in template and run it immediately with Run My Code.")
    template_btn_layout.addWidget(template_load_btn)
    template_btn_layout.addWidget(template_run_btn)
    template_tab_layout.addWidget(template_btn_row)
    library_tabs.addTab(prompt_library_list, "Prompts")
    library_tabs.addTab(code_library_list, "Code")
    library_tabs.addTab(template_tab, "Templates")
    library_tabs.setTabToolTip(0, "Reusable natural-language prompts you can load, save, pin, and send.")
    library_tabs.setTabToolTip(1, "Saved or recent Python snippets for Run My Code and code refinement.")
    library_tabs.setTabToolTip(2, "Built-in starter templates organized by category for loading or immediate execution.")
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
    log_group.setCheckable(True)
    log_group.setChecked(False)
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
    log_tabs.setVisible(False)
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
    advanced_btn = QPushButton("Advanced")
    advanced_btn.setToolTip("Open advanced and optional integrations.")
    advanced_menu = QMenu(advanced_btn)
    sam2_setup_action = advanced_menu.addAction("SAM2 Setup")
    sam2_live_action = advanced_menu.addAction("SAM2 Live")
    advanced_btn.setMenu(advanced_menu)
    run_code_btn = QPushButton("Run Code")
    run_my_code_btn = QPushButton("Run My Code")
    run_my_code_btn.setToolTip("Paste your own Python in the Prompt box and click to run it directly, without opening QtConsole.")
    refine_my_code_btn = QPushButton("Refine My Code")
    refine_my_code_btn.setToolTip("Ask the assistant to repair prompt-box code or the last failed Run My Code submission for this plugin environment.")
    copy_code_btn = QPushButton("Copy Code")
    run_code_btn.setEnabled(False)
    copy_code_btn.setEnabled(False)
    refine_my_code_btn.setEnabled(False)
    code_btn_layout.addWidget(pending_code_label, 1)
    code_btn_layout.addWidget(run_code_btn)
    code_btn_layout.addWidget(copy_code_btn)
    code_btn_layout.addWidget(run_my_code_btn)
    code_btn_layout.addWidget(refine_my_code_btn)
    code_btn_layout.addWidget(reject_memory_btn)
    code_btn_layout.addWidget(advanced_btn)
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
    splitter.setChildrenCollapsible(False)
    splitter.setStretchFactor(0, 45)
    splitter.setStretchFactor(1, 55)

    def clamp_splitter_ratio(value: float) -> float:
        return min(0.7, max(0.3, float(value)))

    def apply_splitter_ratio() -> None:
        total_width = splitter.size().width()
        if total_width <= 0:
            total_width = max(root.size().width(), 900)
        ratio = clamp_splitter_ratio(ui_state.get("assistant_splitter_ratio", 0.45))
        left_width = max(left_panel.minimumWidth(), int(total_width * ratio))
        right_width = max(1, total_width - left_width)
        splitter.setSizes([left_width, right_width])

    def remember_splitter_ratio(*_args) -> None:
        sizes = splitter.sizes()
        total_width = sum(sizes)
        if total_width <= 0:
            return
        ui_state["assistant_splitter_ratio"] = clamp_splitter_ratio(sizes[0] / total_width)
        save_ui_state(ui_state)

    splitter.splitterMoved.connect(remember_splitter_ratio)
    QTimer.singleShot(0, apply_splitter_ratio)

    saved_settings = {
        "provider": "Local (Ollama-style)",
        "base_url": "http://127.0.0.1:11434",
        "model": "nemotron-cascade-2:30b",
    }
    active_workers: list[object] = []
    available_models: list[str] = []
    pending_code = {
        "code": "",
        "message": "",
        "turn_id": "",
        "model": "",
        "runnable": False,
        "validation_mode": "strict",
        "code_source": "assistant",
    }
    last_user_code_failure = {
        "code": "",
        "error": "",
    }
    last_turn_metrics = {"turn_id": "", "model": "", "action": "", "prompt_hash": ""}
    session_memory_state = load_session_memory()
    last_memory_candidate_ids: list[str] = []
    prompt_library_state = load_prompt_library()
    template_library_state = template_library_payload()
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
        phase = str(wait_indicator["phase"] or "thinking").strip() or "thinking"
        if elapsed >= 2.0:
            return f"Status: {phase}... {int(elapsed)}s"
        frame = frames[int(elapsed * 4) % len(frames)]
        return f"Status: {phase} {frame}"

    def start_wait_indicator(*, phase: str = "thinking"):
        wait_indicator["active"] = True
        wait_indicator["started_at"] = time.perf_counter()
        wait_indicator["phase"] = str(phase or "thinking").strip() or "thinking"
        connection_status.setStyleSheet(wait_indicator["style"])
        connection_status.setText(wait_indicator_text())
        wait_timer.start()

    def set_wait_indicator_phase(phase: str):
        wait_indicator["phase"] = str(phase or wait_indicator.get("phase", "thinking")).strip() or "thinking"
        if wait_indicator["active"]:
            connection_status.setText(wait_indicator_text())

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
        summary_text = layer_summary(viewer)
        if widget_is_alive(context_summary_box):
            context_summary_box.setPlainText(summary_text)
        if widget_is_alive(context_layers_list):
            context_layers_list.clear()
            if viewer is not None:
                for layer in viewer.layers:
                    layer_type = layer.__class__.__name__
                    data = getattr(layer, "data", None)
                    shape = getattr(data, "shape", None)
                    dtype = getattr(data, "dtype", None)
                    semantic = "n/a"
                    try:
                        from napari_chat_assistant.agent.profiler import profile_layer

                        semantic = str(profile_layer(layer).get("semantic_type", "n/a"))
                    except Exception:
                        pass
                    line = (
                        f"- {layer.name} [{layer_type}] "
                        f"shape={tuple(shape) if shape is not None else 'n/a'} "
                        f"dtype={dtype if dtype is not None else 'n/a'} semantic={semantic}"
                    )
                    item = QListWidgetItem()
                    context_layers_list.addItem(item)
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(6, 4, 6, 4)
                    row_layout.setSpacing(8)
                    row_label = QLabel(line)
                    row_label.setWordWrap(True)
                    row_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                    copy_btn = QPushButton("Copy")
                    copy_btn.setMaximumWidth(56)
                    insert_btn = QPushButton("Insert")
                    insert_btn.setMaximumWidth(56)
                    insert_btn.setToolTip("Append this layer summary line to the Prompt box.")
                    copy_btn.setToolTip("Copy this layer summary line to the clipboard.")
                    copy_btn.clicked.connect(lambda _checked=False, text=line: QApplication.clipboard().setText(text))
                    insert_btn.clicked.connect(lambda _checked=False, text=line: append_text_to_prompt(text))
                    row_layout.addWidget(row_label, 1)
                    row_layout.addWidget(insert_btn, 0)
                    row_layout.addWidget(copy_btn, 0)
                    item.setSizeHint(row_widget.sizeHint())
                    context_layers_list.setItemWidget(item, row_widget)
                selected = viewer.layers.selection.active if viewer is not None else None
                if selected is not None:
                    item = QListWidgetItem()
                    context_layers_list.addItem(item)
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(6, 4, 6, 4)
                    row_layout.setSpacing(8)
                    line = f"Selected: {selected.name}"
                    row_label = QLabel(line)
                    row_label.setWordWrap(True)
                    row_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                    copy_btn = QPushButton("Copy")
                    copy_btn.setMaximumWidth(56)
                    insert_btn = QPushButton("Insert")
                    insert_btn.setMaximumWidth(56)
                    insert_btn.setToolTip("Append this selected-layer line to the Prompt box.")
                    copy_btn.setToolTip("Copy this selected-layer line to the clipboard.")
                    copy_btn.clicked.connect(lambda _checked=False, text=line: QApplication.clipboard().setText(text))
                    insert_btn.clicked.connect(lambda _checked=False, text=line: append_text_to_prompt(text))
                    row_layout.addWidget(row_label, 1)
                    row_layout.addWidget(insert_btn, 0)
                    row_layout.addWidget(copy_btn, 0)
                    item.setSizeHint(row_widget.sizeHint())
                    context_layers_list.setItemWidget(item, row_widget)
        refresh_analysis_controls()

    def append_text_to_prompt(text: str):
        content = str(text or "").strip()
        if not content:
            return
        current = prompt.toPlainText().rstrip()
        prompt.setPlainText(f"{current}\n{content}" if current else content)
        prompt.setFocus()

    def connect_viewer_context_events():
        if viewer is None:
            return
        try:
            viewer.layers.events.inserted.connect(refresh_context)
            viewer.layers.events.removed.connect(refresh_context)
            viewer.layers.events.reordered.connect(refresh_context)
            viewer.layers.selection.events.active.connect(refresh_context)
            viewer.layers.selection.events.changed.connect(refresh_context)
        except Exception:
            pass

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
            try:
                if "job" in prepared:
                    tool_result = run_tool_job(prepared["job"])
                    result_message = apply_tool_job_result(viewer, tool_result)
                else:
                    result_message = str(prepared.get("message", ""))
            except Exception as exc:
                logger.exception("Analysis panel immediate tool failed: %s", tool_name)
                append_chat_message("assistant", f"{tool_name} failed:\n{exc}")
                append_log(f"Analysis panel immediate tool failed: {tool_name} | {exc}")
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
        code = str(code_text or "")
        return f"```python\n{code}\n```" if code.strip() else "```python\n# empty\n```"

    def strip_code_fences(code_text: str) -> str:
        source = str(code_text or "")
        if not source.strip():
            return ""
        trimmed = source.strip()
        if not trimmed.startswith("```"):
            return source
        lines = trimmed.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

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

    def looks_like_python_code(text: str) -> bool:
        source = str(text or "").strip()
        if not source:
            return False
        signals = (
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
        return any(signal in source for signal in signals)

    def prompt_code_candidate() -> str:
        code_text = strip_code_fences(prompt.toPlainText()).strip()
        if not code_text:
            return ""
        return code_text if looks_like_python_code(code_text) else ""

    def refresh_code_action_buttons():
        has_prompt_code = bool(prompt_code_candidate())
        has_failed_code = bool(str(last_user_code_failure.get("code", "")).strip())
        run_my_code_btn.setEnabled(has_prompt_code)
        refine_my_code_btn.setEnabled(has_prompt_code or has_failed_code)

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
        if build_code_repair_context(text, viewer=viewer) is not None:
            return "code_repair"
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
            "- Ask for the result you want: preview, apply, explain, or code.\n"
            "- If more than one layer is open, name the layer.\n"
            "- Natural language is fine. The assistant will use the selected layer when it can.\n\n"
            "**Try These**\n"
            "- `Inspect the selected layer`\n"
            "- `Preview threshold on em_2d_snr_mid`\n"
            "- `Apply gaussian denoise to em_2d_snr_low with sigma 1.2`\n"
            "- `Fill holes in mask_messy_2d`\n"
            "- `Remove small objects from mask_messy_2d with min_size 64`\n"
            "- `Keep only the largest connected component in mask_messy_2d`\n"
            "- `Measure labels table for rgb_cells_2d_labels`\n"
            "- `Create a max intensity projection from em_3d_snr_mid along axis 0`\n"
            "- `Crop em_2d_snr_high to the bounding box of em_2d_mask with padding 8`\n"
            "- `Inspect the current ROI`\n"
            "- `Extract ROI values from em_2d_snr_mid using em_2d_mask`\n\n"
            "**ROI Support**\n"
            "- Labels and Shapes layers can be used as regions of interest.\n"
            "- You can inspect ROI context or extract grayscale image values inside an ROI.\n"
            "- Example: `Extract ROI values from image_a using roi_shapes`\n\n"
            "**Demo Data**\n"
            "- Use the Library `Code` tab to load built-in demo packs for testing.\n"
            "- Available demo packs include EM 2D/3D SNR sweeps, RGB cells 2D/3D SNR sweeps, and messy masks 2D/3D.\n"
            "- These demos create named layers so you can test tools quickly and repeatably.\n\n"
            "**Example Pipeline**\n"
            "- `Run the EM 2D SNR Sweep demo pack.`\n"
            "- `Apply gaussian denoise to em_2d_snr_low with sigma 1.0`\n"
            "- `Preview threshold on em_2d_snr_low_gaussian`\n"
            "- `Apply threshold now on em_2d_snr_low_gaussian`\n"
            "- `Fill holes in em_2d_snr_low_gaussian_labels`\n"
            "- `Remove small objects from em_2d_snr_low_gaussian_labels_filled with min_size 64`\n"
            "- `Keep only the largest connected component in em_2d_snr_low_gaussian_labels_filled_clean`\n"
            "- `Measure mask on em_2d_snr_low_gaussian_labels_filled_clean_largest`\n\n"
            "**Run My Code Tip**\n"
            "- Paste Python into the Prompt box and click `Run My Code` to execute it directly inside napari.\n"
            "- For heavy compute, use `run_in_background(compute_fn, apply_fn, error_fn=None, label=\"...\")`.\n"
            "- Keep `compute_fn` for NumPy/SciPy work and use `apply_fn` for `viewer.add_*` updates.\n"
            "- Example: `Write Run My Code for a 3D RGB cell demo using run_in_background.`\n\n"
            "**Formatting**\n"
            "- `Reply in markdown`\n"
            "- `Use bullets and short sections`\n"
            "- `Explain first, then give runnable napari code`\n\n"
            "**Language**\n"
            "- You can prompt in your preferred language.",
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
        current_widget = library_tabs.currentWidget()
        if current_widget is code_library_list:
            return "code"
        if current_widget is template_tab:
            return "template"
        return "prompt"

    def current_library_list() -> QListWidget:
        return code_library_list if current_library_kind() == "code" else prompt_library_list

    def current_library_item_name() -> str:
        if current_library_kind() == "template":
            return "template"
        return "code snippet" if current_library_kind() == "code" else "prompt"

    def selected_library_records() -> list[dict]:
        if current_library_kind() == "template":
            item = template_tree.currentItem()
            if item is None:
                return []
            record = item.data(0, Qt.UserRole)
            return [record] if isinstance(record, dict) and record.get("code") else []
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
        for widget in (
            base_url_edit,
            model_combo,
            test_btn,
            save_btn,
            pull_btn,
            unload_btn,
            prompt,
            connection_toggle_btn,
        ):
            widget.setEnabled(enabled)

    def set_pending_code(
        code_text: str = "",
        *,
        message: str = "",
        runnable: bool = True,
        label: str | None = None,
        validation_mode: ValidationMode = "strict",
        code_source: str = "assistant",
    ):
        pending_code["code"] = str(code_text or "").strip()
        pending_code["message"] = str(message or "").strip()
        pending_code["runnable"] = bool(runnable and pending_code["code"])
        pending_code["validation_mode"] = validation_mode
        pending_code["code_source"] = code_source
        has_code = bool(pending_code["code"])
        if not has_code:
            pending_code["turn_id"] = ""
            pending_code["model"] = ""
            pending_code["runnable"] = False
            pending_code["validation_mode"] = "strict"
            pending_code["code_source"] = "assistant"
        pending_code_label.setText(label or ("Pending code: ready to run" if has_code else "Pending code: none"))
        run_code_btn.setEnabled(bool(has_code and pending_code["runnable"]))
        copy_code_btn.setEnabled(has_code)

    def preflight_generated_code(code_text: str) -> ValidationReport:
        return validate_generated_code(code_text, viewer=viewer)

    def format_validation_report(
        report: ValidationReport,
        *,
        mode: ValidationMode,
        heading: str,
        include_notes: bool = True,
    ) -> str:
        lines = [heading]
        if report.errors:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in report.errors)
        if report.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in report.warnings)
            if mode == "permissive" and not report.errors:
                lines.append("")
                lines.append("Permissive mode allows execution despite warnings.")
        if include_notes and report.notes:
            lines.append("Notes:")
            lines.extend(f"- {note}" for note in report.notes)
        return "\n".join(lines)

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
        if current_library_kind() == "template":
            save_prompt_btn.setText("Save")
            pin_prompt_btn.setText("Pin")
            delete_prompt_btn.setText("Delete")
            clear_prompt_btn.setText("Clear")
            save_prompt_btn.setEnabled(False)
            pin_prompt_btn.setEnabled(False)
            delete_prompt_btn.setEnabled(False)
            clear_prompt_btn.setEnabled(False)
            prompt_library_hint.setText(
                "Click a template to preview it. Double-click to run it with Run My Code."
            )
            template_selected = current_template_record() is not None
            template_load_btn.setEnabled(template_selected)
            template_run_btn.setEnabled(template_selected)
            return
        template_load_btn.setEnabled(False)
        template_run_btn.setEnabled(False)
        save_prompt_btn.setEnabled(True)
        pin_prompt_btn.setEnabled(True)
        delete_prompt_btn.setEnabled(True)
        clear_prompt_btn.setEnabled(True)
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
        template_tree.clear()
        categories = [str(name).strip() for name in template_library_state.get("categories", []) if str(name).strip()]
        category_lookup: dict[str, QTreeWidgetItem] = {}
        for category in categories:
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsSelectable)
            category_item.setFirstColumnSpanned(True)
            category_lookup[category] = category_item
            template_tree.addTopLevelItem(category_item)
        for record in template_library_state.get("templates", []):
            if not isinstance(record, dict) or not str(record.get("code", "")).strip():
                continue
            category = str(record.get("category", "Templates")).strip() or "Templates"
            parent = category_lookup.get(category)
            if parent is None:
                parent = QTreeWidgetItem([category])
                parent.setFlags(parent.flags() & ~Qt.ItemIsSelectable)
                parent.setFirstColumnSpanned(True)
                category_lookup[category] = parent
                template_tree.addTopLevelItem(parent)
            child = QTreeWidgetItem([str(record.get("title", "Untitled Template")).strip() or "Untitled Template"])
            child.setData(0, Qt.UserRole, record)
            parent.addChild(child)
        for index in range(template_tree.topLevelItemCount()):
            template_tree.topLevelItem(index).setExpanded(False)
        template_tree.setCurrentItem(None)
        template_preview.clear()
        refresh_library_controls()

    def current_template_record() -> dict | None:
        item = template_tree.currentItem()
        if item is None:
            return None
        record = item.data(0, Qt.UserRole)
        return record if isinstance(record, dict) and str(record.get("code", "")).strip() else None

    def template_preview_text(record: dict) -> str:
        title = str(record.get("title", "")).strip() or "Untitled Template"
        category = str(record.get("category", "")).strip() or "Templates"
        description = str(record.get("description", "")).strip()
        tags = [str(tag).strip() for tag in record.get("tags", []) if str(tag).strip()]
        best_for = str(record.get("best_for", "")).strip()
        followup = str(record.get("suggested_followup", "")).strip()
        runtime = record.get("runtime", {})
        runtime_flags: list[str] = []
        if isinstance(runtime, dict):
            if runtime.get("uses_viewer"):
                runtime_flags.append("Viewer")
            if runtime.get("uses_selected_layer"):
                runtime_flags.append("Selected Layer")
            if runtime.get("uses_run_in_background"):
                runtime_flags.append("Background")
        lines = [f"Template: {title}", f"Category: {category}"]
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
        if runtime_flags:
            lines.append(f"Runtime: {', '.join(runtime_flags)}")
        if description:
            lines.extend(["", description])
        if best_for:
            lines.extend(["", f"Best for: {best_for}"])
        if followup:
            lines.extend(["", f"Suggested follow-up: {followup}"])
        lines.extend(
            [
                "",
                "This template is designed to run inside the Chat Assistant plugin runtime in napari.",
                "It may use viewer, selected_layer, and run_in_background(...).",
                "",
                "Code:",
                str(record.get("code", "")).rstrip(),
            ]
        )
        return "\n".join(lines).strip()

    def show_template_preview(item: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None = None):
        del _previous
        if item is None:
            template_preview.clear()
            refresh_library_controls()
            return
        record = item.data(0, Qt.UserRole)
        if not isinstance(record, dict) or not str(record.get("code", "")).strip():
            template_preview.clear()
            refresh_library_controls()
            return
        template_preview.setPlainText(template_preview_text(record))
        refresh_library_controls()

    def load_template_record(record: dict | None, *, run_now: bool = False):
        if not isinstance(record, dict):
            set_status("Status: no template selected", ok=False)
            append_log("Template load skipped: no template record selected.")
            return
        code_text = str(record.get("code", ""))
        if not code_text.strip():
            set_status("Status: selected template is empty", ok=False)
            append_log("Template load skipped: selected template is empty.")
            return
        prompt.setPlainText(code_text)
        prompt.setFocus()
        append_log(f"Loaded template: {record.get('title', 'Untitled Template')}")
        set_status("Status: template loaded", ok=None)
        if run_now:
            run_prompt_code()

    def load_selected_template(*_args):
        record = current_template_record()
        if record is None:
            set_status("Status: no template selected", ok=False)
            append_log("Load template skipped: no selection.")
            return
        load_template_record(record)

    def run_selected_template(*_args):
        record = current_template_record()
        if record is None:
            set_status("Status: no template selected", ok=False)
            append_log("Run template skipped: no selection.")
            return
        load_template_record(record, run_now=True)

    def run_template_tree_item(item: QTreeWidgetItem, _column: int):
        del _column
        record = item.data(0, Qt.UserRole)
        if not isinstance(record, dict) or not str(record.get("code", "")).strip():
            return
        load_template_record(record, run_now=True)

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
        set_status("Status: checking connection...", ok=None)
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
            append_log("Model load skipped: missing base URL or model name.")
            return
        saved_settings["provider"] = provider
        saved_settings["base_url"] = base_url.rstrip("/")
        saved_settings["model"] = model_name
        set_status(f"Status: loading {model_name}...", ok=None)
        append_log(f"Loading model for {provider} | {base_url} | model={model_name}")

        @thread_worker(ignore_errors=True)
        def run_model_load():
            return load_ollama_model(saved_settings["base_url"], saved_settings["model"])

        worker = run_model_load()
        active_workers.append(worker)

        def finish_model_load():
            if worker in active_workers:
                active_workers.remove(worker)

        def on_loaded(_result=None):
            finish_model_load()
            set_status(f"Status: model {model_name} loaded and ready", ok=True)
            append_log(f"Loaded model for {provider} | {base_url} | model={model_name}")

        def on_load_error(*args):
            finish_model_load()
            error_text = format_worker_error(*args)
            set_status(f"Status: model load failed", ok=False)
            append_chat_message("assistant", f"Model load failed:\n{error_text}")
            append_log(f"Model load failed: {error_text}")

        worker.returned.connect(on_loaded)
        worker.errored.connect(on_load_error)
        worker.start()

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

    def show_sam2_setup_dialog(*_args):
        sam2_clone_command = "git clone https://github.com/facebookresearch/sam2.git && cd sam2"
        sam2_install_commands = f"{sam2_clone_command}\npip install -e ."
        dialog = QDialog(root)
        dialog.setWindowTitle("SAM2 Setup")
        dialog.resize(760, 430)
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(18, 18, 18, 14)
        dialog_layout.setSpacing(10)

        hint = QLabel("Set up SAM2 for the built-in SAM2 tools.")
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #f3f4f6; font-size: 15px; font-weight: 600; }")
        dialog_layout.addWidget(hint)

        guidance_label = QLabel(
            "Project Path should be the SAM2 repo folder. "
            "Click `Auto Detect` first."
        )
        guidance_label.setWordWrap(True)
        guidance_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        guidance_label.setStyleSheet("QLabel { color: #cbd5e1; line-height: 1.35; }")
        dialog_layout.addWidget(guidance_label)

        install_group = QGroupBox("Quick Install")
        install_group.setStyleSheet(
            "QGroupBox { color: #dbe4f0; font-weight: 600; border: 1px solid #30363d; border-radius: 8px; margin-top: 10px; padding-top: 12px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        install_layout = QVBoxLayout(install_group)
        install_layout.setContentsMargins(12, 10, 12, 12)
        install_layout.setSpacing(8)

        install_hint = QLabel(
            "Use the same virtual environment as napari. Clone SAM2, then run `pip install -e .` from the SAM2 repo."
        )
        install_hint.setWordWrap(True)
        install_hint.setTextInteractionFlags(Qt.TextSelectableByMouse)
        install_hint.setStyleSheet("QLabel { color: #b6c2cf; }")
        install_layout.addWidget(install_hint)

        install_command_box = QTextEdit()
        install_command_box.setReadOnly(True)
        install_command_box.setAcceptRichText(False)
        install_command_box.setLineWrapMode(QTextEdit.NoWrap)
        install_command_box.setMinimumHeight(76)
        install_command_box.setMaximumHeight(76)
        install_command_box.setPlainText(sam2_install_commands)
        install_command_box.setStyleSheet(
            "QTextEdit { background: #0b1021; color: #e6edf3; border: 1px solid #22304a; border-radius: 6px; "
            "padding: 10px 12px; selection-background-color: #264f78; "
            "font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace; font-size: 13px; }"
        )
        install_layout.addWidget(install_command_box)

        copy_install_btn = QPushButton("Copy Commands")
        copy_install_btn.setToolTip("Copy the install commands to the clipboard.")
        install_layout.addWidget(copy_install_btn, 0, Qt.AlignRight)
        dialog_layout.addWidget(install_group)

        settings_group = QGroupBox("Backend Settings")
        settings_group.setStyleSheet(
            "QGroupBox { color: #dbe4f0; font-weight: 600; border: 1px solid #30363d; border-radius: 8px; margin-top: 10px; padding-top: 12px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setContentsMargins(12, 10, 12, 12)
        settings_layout.setSpacing(8)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        project_path_edit = QLineEdit(str(ui_state.get("sam2_project_path", "")))
        checkpoint_path_edit = QComboBox()
        checkpoint_path_edit.setEditable(True)
        config_path_edit = QComboBox()
        config_path_edit.setEditable(True)
        device_combo = QComboBox()
        device_combo.setEditable(True)
        device_combo.addItems(["cuda", "cpu"])
        device_combo.setCurrentText(str(ui_state.get("sam2_device", "cuda")))
        field_style = (
            "QLineEdit, QComboBox { background: #1f2329; color: #e6edf3; border: 1px solid #30363d; "
            "border-radius: 5px; padding: 6px 8px; min-height: 20px; }"
        )
        project_path_edit.setStyleSheet(field_style)
        checkpoint_path_edit.setStyleSheet(field_style)
        config_path_edit.setStyleSheet(field_style)
        device_combo.setStyleSheet(field_style)
        form.addRow("Project Path:", project_path_edit)
        form.addRow("Checkpoint:", checkpoint_path_edit)
        form.addRow("Config:", config_path_edit)
        form.addRow("Device:", device_combo)
        settings_layout.addLayout(form)
        dialog_layout.addWidget(settings_group)

        status_label = QLabel()
        status_label.setWordWrap(True)
        status_label.setStyleSheet(
            "QLabel { background: #101820; color: #e6edf3; border: 1px solid #22304a; border-radius: 6px; padding: 10px 12px; }"
        )
        dialog_layout.addWidget(status_label)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        autodetect_dialog_btn = QPushButton("Detect Setup")
        save_dialog_btn = QPushButton("Save")
        test_dialog_btn = QPushButton("Test")
        close_dialog_btn = QPushButton("Close")
        button_layout.addWidget(autodetect_dialog_btn)
        button_layout.addWidget(save_dialog_btn)
        button_layout.addWidget(test_dialog_btn)
        button_layout.addWidget(close_dialog_btn)
        dialog_layout.addWidget(button_row)

        def current_checkpoint_text() -> str:
            return checkpoint_path_edit.currentText().strip()

        def current_config_text() -> str:
            return config_path_edit.currentText().strip()

        def refresh_checkpoint_and_config_choices():
            project_path = project_path_edit.text().strip()
            selected_checkpoint = current_checkpoint_text() or str(ui_state.get("sam2_checkpoint_path", "")).strip()
            selected_config = current_config_text() or str(ui_state.get("sam2_config_path", "")).strip()
            checkpoint_choices = list_sam2_checkpoints(project_path)
            config_choices = list_sam2_configs(project_path)

            checkpoint_path_edit.blockSignals(True)
            checkpoint_path_edit.clear()
            checkpoint_path_edit.addItems(checkpoint_choices)
            if selected_checkpoint and selected_checkpoint not in checkpoint_choices:
                checkpoint_path_edit.addItem(selected_checkpoint)
            checkpoint_path_edit.setCurrentText(selected_checkpoint)
            checkpoint_path_edit.blockSignals(False)

            config_path_edit.blockSignals(True)
            config_path_edit.clear()
            config_path_edit.addItems(config_choices)
            if selected_config and selected_config not in config_choices:
                config_path_edit.addItem(selected_config)
            config_path_edit.setCurrentText(selected_config)
            config_path_edit.blockSignals(False)

        def refresh_status():
            config = sam2_config_from_ui_state(
                {
                    **ui_state,
                    "sam2_project_path": project_path_edit.text().strip(),
                    "sam2_checkpoint_path": current_checkpoint_text(),
                    "sam2_config_path": current_config_text(),
                    "sam2_device": device_combo.currentText().strip(),
                }
            )
            ok, message = get_sam2_backend_status(config)
            status_label.setText(message)
            status_label.setStyleSheet(
                "QLabel { background: #17341f; color: #e6ffed; padding: 8px; }"
                if ok
                else "QLabel { background: #3a1f1f; color: #ffe4e6; padding: 8px; }"
            )
            return ok, message

        def autodetect_sam2_settings():
            detected, message = discover_sam2_setup(
                {
                    **ui_state,
                    "sam2_project_path": project_path_edit.text().strip(),
                    "sam2_checkpoint_path": current_checkpoint_text(),
                    "sam2_config_path": current_config_text(),
                    "sam2_device": device_combo.currentText().strip(),
                }
            )
            project_path_edit.setText(detected["sam2_project_path"])
            refresh_checkpoint_and_config_choices()
            checkpoint_path_edit.setCurrentText(detected["sam2_checkpoint_path"])
            config_path_edit.setCurrentText(detected["sam2_config_path"])
            device_combo.setCurrentText(detected["sam2_device"])
            guidance_label.setText(message)
            refresh_status()
            append_log("Auto-detected SAM2 setup candidates.")
            set_status("Status: SAM2 auto-detect finished", ok=None)

        def copy_sam2_install_commands():
            QApplication.clipboard().setText(sam2_install_commands)
            append_log("Copied SAM2 install commands.")
            set_status("Status: SAM2 install commands copied", ok=True)

        def save_sam2_settings():
            ui_state["sam2_project_path"] = project_path_edit.text().strip()
            ui_state["sam2_checkpoint_path"] = current_checkpoint_text()
            ui_state["sam2_config_path"] = current_config_text()
            ui_state["sam2_device"] = device_combo.currentText().strip() or "cuda"
            save_ui_state(ui_state)
            ok, _message = refresh_status()
            refresh_sam2_actions()
            append_log("Saved SAM2 settings.")
            set_status("Status: SAM2 settings saved", ok=ok if ok else None)

        def test_sam2_settings():
            ok, message = refresh_status()
            append_chat_message("assistant", message)
            append_log("Tested SAM2 backend configuration.")
            set_status("Status: SAM2 backend ready" if ok else "Status: SAM2 backend not ready", ok=ok)

        autodetect_dialog_btn.clicked.connect(autodetect_sam2_settings)
        copy_install_btn.clicked.connect(copy_sam2_install_commands)
        save_dialog_btn.clicked.connect(save_sam2_settings)
        test_dialog_btn.clicked.connect(test_sam2_settings)
        close_dialog_btn.clicked.connect(dialog.accept)
        project_path_edit.editingFinished.connect(refresh_checkpoint_and_config_choices)
        refresh_checkpoint_and_config_choices()
        refresh_status()
        append_log("Opened SAM2 setup dialog.")
        set_status("Status: SAM2 setup opened", ok=None)
        dialog.exec_()

    def refresh_sam2_actions():
        ok, message = get_sam2_backend_status()
        sam2_live_action.setEnabled(ok)
        advanced_btn.setToolTip(
            "Open advanced and optional integrations."
            if ok
            else f"Open advanced and optional integrations. {message}"
        )

    def show_sam2_live_dialog(*_args):
        dialog = QDialog(root)
        dialog.setWindowTitle("SAM2 Live")
        dialog.resize(760, 420)
        dialog_layout = QVBoxLayout(dialog)

        hint = QLabel(
            "Live SAM2 preview for grayscale images. "
            "Use a Shapes layer for 2D Box mode, or initialize a SAM2-managed points session for points prompting. "
            "For 3D propagation, initialize on the current seed slice and place prompts there before propagating. "
            "When the SAM2 points layer is active, press T to toggle polarity."
        )
        hint.setWordWrap(True)
        dialog_layout.addWidget(hint)

        form = QFormLayout()
        image_combo = QComboBox()
        model_combo = QComboBox()
        model_combo.setEditable(False)
        mode_combo = QComboBox()
        mode_combo.addItems(["Box", "Points"])
        prompt_layer_combo = QComboBox()
        polarity_combo = QComboBox()
        polarity_combo.addItems(["Positive", "Negative"])
        polarity_combo.setToolTip("Point label mode for new prompt points. With the SAM2 points layer active, press T to toggle polarity.")
        auto_update_check = QCheckBox("Auto-update")
        auto_update_check.setChecked(True)
        form.addRow("Image:", image_combo)
        form.addRow("Model:", model_combo)
        form.addRow("Mode:", mode_combo)
        form.addRow("Prompt Layer:", prompt_layer_combo)
        form.addRow("Polarity:", polarity_combo)
        form.addRow("", auto_update_check)
        dialog_layout.addLayout(form)

        status_label = QLabel("SAM2 Live is ready.")
        status_label.setWordWrap(True)
        status_label.setStyleSheet("QLabel { background: #101820; color: #e6edf3; padding: 8px; }")
        dialog_layout.addWidget(status_label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)
        progress_bar.setVisible(False)
        dialog_layout.addWidget(progress_bar)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        init_btn = QPushButton("Initialize")
        preview_btn = QPushButton("Run Preview")
        apply_btn = QPushButton("Apply")
        clear_btn = QPushButton("Clear Preview")
        close_btn = QPushButton("Close")
        button_layout.addWidget(init_btn)
        button_layout.addWidget(preview_btn)
        button_layout.addWidget(apply_btn)
        button_layout.addWidget(clear_btn)
        button_layout.addWidget(close_btn)
        dialog_layout.addWidget(button_row)

        preview_layer_name = "__sam2_preview__"
        combined_points_layer_name = "__sam2_live_points__"
        live_state = {
            "request_id": 0,
            "busy": False,
            "rerun": False,
            "last_payload": None,
            "connections": [],
            "initialized": False,
            "seed_slice": None,
            "managed_points_layer": None,
        }
        debounce_timer = QTimer(dialog)
        debounce_timer.setSingleShot(True)
        debounce_timer.setInterval(250)

        def infer_config_for_checkpoint(checkpoint_path: str, project_path: str) -> str | None:
            checkpoint_name = Path(str(checkpoint_path or "").strip()).name.lower()
            if not checkpoint_name:
                return None
            config_choices = list_sam2_configs(project_path)
            if not config_choices:
                return None
            preferred_tokens = []
            if "large" in checkpoint_name:
                preferred_tokens.extend(["hiera_l", "hiera_large"])
            if "base_plus" in checkpoint_name or "base+" in checkpoint_name or "b+" in checkpoint_name:
                preferred_tokens.extend(["hiera_b+", "hiera_base_plus"])
            if "small" in checkpoint_name:
                preferred_tokens.extend(["hiera_s", "hiera_small"])
            if "tiny" in checkpoint_name:
                preferred_tokens.extend(["hiera_t", "hiera_tiny"])
            lowered_choices = [(choice, choice.lower()) for choice in config_choices]
            for token in preferred_tokens:
                for choice, lowered in lowered_choices:
                    if token in lowered:
                        return choice
            return config_choices[0]

        def refresh_live_model_choices():
            project_path = str(ui_state.get("sam2_project_path", "")).strip()
            selected_checkpoint = str(ui_state.get("sam2_checkpoint_path", "")).strip()
            checkpoint_choices = list_sam2_checkpoints(project_path)
            model_combo.blockSignals(True)
            model_combo.clear()
            model_combo.addItems(checkpoint_choices)
            if selected_checkpoint and selected_checkpoint not in checkpoint_choices:
                model_combo.addItem(selected_checkpoint)
            model_combo.setCurrentText(selected_checkpoint)
            model_combo.blockSignals(False)

        def apply_live_model_selection(*_args):
            selected_checkpoint = model_combo.currentText().strip()
            if not selected_checkpoint:
                return
            ui_state["sam2_checkpoint_path"] = selected_checkpoint
            inferred_config = infer_config_for_checkpoint(selected_checkpoint, str(ui_state.get("sam2_project_path", "")).strip())
            if inferred_config:
                ui_state["sam2_config_path"] = inferred_config
            save_ui_state(ui_state)
            refresh_sam2_actions()
            config_name = str(ui_state.get("sam2_config_path", "")).strip() or "[unknown config]"
            set_live_status(
                f"SAM2 model set to [{selected_checkpoint}] with config [{config_name}].",
                ok=None,
            )

        def set_live_status(message: str, *, ok: bool | None = None):
            status_label.setText(message)
            if ok is True:
                status_label.setStyleSheet("QLabel { background: #17341f; color: #e6ffed; padding: 8px; }")
            elif ok is False:
                status_label.setStyleSheet("QLabel { background: #3a1f1f; color: #ffe4e6; padding: 8px; }")
            else:
                status_label.setStyleSheet("QLabel { background: #101820; color: #e6edf3; padding: 8px; }")

        def set_live_progress(active: bool):
            progress_bar.setVisible(bool(active))

        def set_form_row_visible(field_widget, visible: bool):
            try:
                label_widget = form.labelForField(field_widget)
                if label_widget is not None:
                    label_widget.setVisible(bool(visible))
            except Exception:
                pass
            field_widget.setVisible(bool(visible))

        def current_image_layer():
            image_name = image_combo.currentText().strip()
            if viewer is None or not image_name or image_name not in viewer.layers:
                return None
            layer = viewer.layers[image_name]
            return layer if isinstance(layer, napari.layers.Image) else None

        def current_image_ndim() -> int | None:
            layer = current_image_layer()
            return None if layer is None else int(np.asarray(layer.data).ndim)

        def current_prompt_tool_name() -> str | None:
            image_layer = current_image_layer()
            if image_layer is None:
                return None
            image_data = np.asarray(image_layer.data)
            mode = mode_combo.currentText().strip()
            if image_data.ndim == 2:
                return "sam_segment_from_box" if mode == "Box" else "sam_segment_from_points"
            if image_data.ndim == 3 and mode == "Points":
                return "sam_propagate_points_3d"
            return None

        def image_layer_names_sam2() -> list[str]:
            names = []
            for layer in viewer.layers if viewer is not None else []:
                if isinstance(layer, napari.layers.Image) and not getattr(layer, "rgb", False):
                    data = np.asarray(layer.data)
                    if data.ndim in (2, 3):
                        names.append(layer.name)
            return names

        def prompt_layer_names_for_mode() -> list[str]:
            mode = mode_combo.currentText().strip()
            image_ndim = current_image_ndim()
            names = []
            for layer in viewer.layers if viewer is not None else []:
                if mode == "Box" and image_ndim == 2 and isinstance(layer, napari.layers.Shapes):
                    names.append(layer.name)
            return names

        def point_layer_names_for_ndim(ndim: int | None) -> list[str]:
            names = []
            for layer in viewer.layers if viewer is not None else []:
                if not isinstance(layer, napari.layers.Points):
                    continue
                if layer.name == combined_points_layer_name:
                    continue
                data = np.asarray(layer.data)
                if ndim == 2 and data.ndim == 2 and data.shape[1] >= 2:
                    names.append(layer.name)
                if ndim == 3 and data.ndim == 2 and data.shape[1] >= 3:
                    names.append(layer.name)
            return names

        def current_seed_slice() -> int | None:
            if viewer is None or current_image_ndim() != 3:
                return None
            try:
                return int(round(float(viewer.dims.point[0])))
            except Exception:
                return None

        def managed_points_layer_name() -> str | None:
            image_layer = current_image_layer()
            if image_layer is None:
                return None
            return f"{image_layer.name}_sam2_prompts"

        def active_polarity_label() -> int:
            return 1 if polarity_combo.currentText().strip() == "Positive" else 0

        def active_polarity_name() -> str:
            return polarity_combo.currentText().strip().lower()

        def set_points_mode_polarity(label_value: int):
            polarity_combo.setCurrentText("Positive" if int(label_value) == 1 else "Negative")

        def apply_points_polarity_defaults(layer):
            label_value = active_polarity_label()
            try:
                layer.feature_defaults = {"sam_label": label_value}
            except Exception:
                pass
            try:
                layer.current_properties = {"sam_label": np.asarray([label_value], dtype=np.int32)}
            except Exception:
                pass
            color = "#4caf50" if label_value == 1 else "#ef5350"
            try:
                layer.current_face_color = color
            except Exception:
                pass
            try:
                layer.current_border_color = "white"
            except Exception:
                pass

        def apply_points_layer_colors(layer, labels: np.ndarray | None = None):
            if layer is None:
                return
            if labels is None:
                labels = np.asarray([], dtype=np.int32)
                try:
                    features = getattr(layer, "features", None)
                    if features is not None and "sam_label" in features:
                        column = features["sam_label"]
                        try:
                            labels = np.asarray(column.to_numpy(), dtype=np.int32)
                        except Exception:
                            labels = np.asarray(column, dtype=np.int32)
                except Exception:
                    pass
            colors = np.asarray(["#4caf50" if int(label) == 1 else "#ef5350" for label in np.asarray(labels, dtype=np.int32)], dtype=object)
            try:
                if colors.size:
                    layer.face_color = colors.tolist()
                    layer.border_color = ["white"] * int(colors.size)
            except Exception:
                pass

        def toggle_selected_or_pending_polarity():
            layer_name = str(live_state.get("managed_points_layer") or "").strip()
            if viewer is None or not layer_name or layer_name not in viewer.layers:
                set_live_status("Initialize SAM2 Live first so the managed points layer is available.", ok=False)
                return
            layer = viewer.layers[layer_name]
            if not isinstance(layer, napari.layers.Points):
                set_live_status("Managed SAM2 prompt layer is not a points layer.", ok=False)
                return

            selected = []
            try:
                selected = sorted(int(index) for index in getattr(layer, "selected_data", set()) or set())
            except Exception:
                selected = []

            labels = None
            try:
                features = getattr(layer, "features", None)
                if features is not None and "sam_label" in features:
                    column = features["sam_label"]
                    try:
                        labels = np.asarray(column.to_numpy(), dtype=np.int32)
                    except Exception:
                        labels = np.asarray(column, dtype=np.int32)
            except Exception:
                labels = None
            if labels is None:
                data = np.asarray(getattr(layer, "data", np.empty((0, current_image_ndim() or 2))), dtype=np.float32)
                labels = np.ones((data.shape[0],), dtype=np.int32)

            if selected:
                updated = labels.copy()
                changed = 0
                for index in selected:
                    if 0 <= index < updated.shape[0]:
                        updated[index] = 0 if int(updated[index]) == 1 else 1
                        changed += 1
                try:
                    layer.features = {"sam_label": updated}
                except Exception:
                    try:
                        existing = getattr(layer, "features", None)
                        if existing is not None:
                            existing["sam_label"] = updated
                            layer.features = existing
                    except Exception:
                        pass
                apply_points_layer_colors(layer, updated)
                set_live_status(
                    f"Toggled polarity for {changed} selected SAM2 prompt point(s).",
                    ok=None,
                )
                return

            next_label = 0 if active_polarity_label() == 1 else 1
            set_points_mode_polarity(next_label)
            apply_points_polarity_defaults(layer)
            set_live_status(
                f"SAM2 prompt polarity toggled to {active_polarity_name()}. New points will use that label.",
                ok=None,
            )

        def toggle_sam2_polarity_from_viewer(_viewer=None):
            del _viewer
            active_dialog = getattr(root, "_sam2_live_dialog", None)
            if active_dialog is not dialog or not dialog.isVisible():
                return
            layer_name = str(live_state.get("managed_points_layer") or "").strip()
            if viewer is None or not layer_name or layer_name not in viewer.layers:
                return
            active_layer = getattr(getattr(viewer.layers, "selection", None), "active", None)
            if active_layer is None or getattr(active_layer, "name", "") != layer_name:
                return
            toggle_selected_or_pending_polarity()

        def register_points_toggle_binding(layer):
            if layer is None or not hasattr(layer, "bind_key"):
                return
            if getattr(layer, "_sam2_toggle_registered", False):
                return
            try:
                @layer.bind_key("t", overwrite=True)
                def _toggle_sam2_prompt_polarity(active_layer):
                    del active_layer
                    toggle_selected_or_pending_polarity()
            except TypeError:
                try:
                    layer.bind_key("t", toggle_selected_or_pending_polarity, overwrite=True)
                except Exception:
                    return
            except Exception:
                return
            setattr(layer, "_sam2_toggle_registered", True)

        def register_viewer_toggle_binding():
            if viewer is None or not hasattr(viewer, "bind_key"):
                return
            if getattr(root, "_sam2_viewer_toggle_registered", False):
                return
            try:
                @viewer.bind_key("t", overwrite=True)
                def _toggle_sam2_prompt_polarity(_active_viewer):
                    toggle_sam2_polarity_from_viewer(_active_viewer)
            except TypeError:
                try:
                    viewer.bind_key("t", toggle_sam2_polarity_from_viewer, overwrite=True)
                except Exception:
                    return
            except Exception:
                return
            setattr(root, "_sam2_viewer_toggle_registered", True)

        def ensure_managed_points_layer():
            if viewer is None:
                return None
            image_layer = current_image_layer()
            if image_layer is None:
                set_live_status("Select an image layer first before initializing SAM2 prompts.", ok=False)
                return None
            ndim = current_image_ndim()
            if ndim not in {2, 3}:
                set_live_status("SAM2 prompts currently support 2D or 3D grayscale images only.", ok=False)
                return None
            layer_name = managed_points_layer_name()
            if not layer_name:
                return None
            if layer_name in viewer.layers:
                layer = viewer.layers[layer_name]
                if isinstance(layer, napari.layers.Points):
                    apply_points_polarity_defaults(layer)
                    register_points_toggle_binding(layer)
                    live_state["managed_points_layer"] = layer.name
                    return layer
            empty = np.empty((0, ndim), dtype=np.float32)
            layer = viewer.add_points(
                empty,
                name=layer_name,
                features={"sam_label": np.empty((0,), dtype=np.int32)},
                face_color="#4caf50",
                border_color="white",
                size=10,
            )
            layer.metadata = dict(getattr(layer, "metadata", {}) or {})
            layer.metadata["sam2_managed_points"] = True
            apply_points_polarity_defaults(layer)
            register_points_toggle_binding(layer)
            live_state["managed_points_layer"] = layer.name
            return layer

        def initialize_live_session(*_args):
            image_layer = current_image_layer()
            if image_layer is None:
                set_live_status("Select an image layer before initializing SAM2 Live.", ok=False)
                return
            if mode_combo.currentText().strip() == "Points":
                points_layer = ensure_managed_points_layer()
                if points_layer is None:
                    return
                live_state["initialized"] = True
                live_state["managed_points_layer"] = points_layer.name
                live_state["seed_slice"] = current_seed_slice() if current_image_ndim() == 3 else None
                if current_image_ndim() == 3:
                    set_live_status(
                        f"SAM2 Live initialized for [{image_layer.name}]. "
                        f"Seed slice={live_state['seed_slice'] if live_state['seed_slice'] is not None else '?'}. "
                        f"Add prompt points on [{points_layer.name}], use the Polarity control or press T on the active points layer, then click Propagate.",
                        ok=True,
                    )
                else:
                    set_live_status(
                        f"SAM2 Live initialized for [{image_layer.name}]. "
                        f"Add prompt points on [{points_layer.name}], use the Polarity control or press T on the active points layer, then run preview.",
                        ok=True,
                    )
                try:
                    viewer.layers.selection.active = points_layer
                except Exception:
                    pass
            else:
                live_state["initialized"] = True
                live_state["managed_points_layer"] = None
                live_state["seed_slice"] = None
                set_live_status(
                    f"SAM2 Live initialized for [{image_layer.name}] in Box mode. "
                    "Choose or draw a Shapes prompt, then run preview.",
                    ok=True,
                )

        def sync_combined_points_layer() -> str | None:
            if viewer is None:
                return None
            ndim = current_image_ndim()
            if ndim not in {2, 3}:
                return None
            layer_name = str(live_state.get("managed_points_layer") or "").strip()
            if not layer_name or layer_name not in viewer.layers:
                return None
            layer = viewer.layers[layer_name]
            if not isinstance(layer, napari.layers.Points):
                return None
            data = np.asarray(layer.data, dtype=np.float32)
            if data.ndim != 2 or data.shape[1] < ndim or data.shape[0] == 0:
                return None
            merged_points = data[:, :ndim]
            labels = np.ones((data.shape[0],), dtype=np.int32)
            try:
                features = getattr(layer, "features", None)
                if features is not None and "sam_label" in features:
                    column = features["sam_label"]
                    try:
                        labels = np.asarray(column.to_numpy(), dtype=np.int32)
                    except Exception:
                        labels = np.asarray(column, dtype=np.int32)
            except Exception:
                pass
            if ndim == 3 and live_state.get("seed_slice") is not None:
                seed_slice = int(live_state["seed_slice"])
                z_values = np.rint(merged_points[:, 0]).astype(int)
                keep = z_values == seed_slice
                if not np.any(keep):
                    return None
                merged_points = merged_points[keep]
                labels = labels[keep]
            if combined_points_layer_name in viewer.layers:
                layer = viewer.layers[combined_points_layer_name]
                if isinstance(layer, napari.layers.Points):
                    layer.data = merged_points
                    layer.features = {"sam_label": labels}
                    layer.visible = False
                    return combined_points_layer_name
            layer = viewer.add_points(
                merged_points,
                name=combined_points_layer_name,
                features={"sam_label": labels},
                face_color="transparent",
                border_color="transparent",
                size=1,
                visible=False,
            )
            layer.metadata = dict(getattr(layer, "metadata", {}) or {})
            layer.metadata["sam2_managed_combined"] = True
            return combined_points_layer_name

        def clear_event_connections():
            for emitter, callback in live_state["connections"]:
                try:
                    emitter.disconnect(callback)
                except Exception:
                    pass
            live_state["connections"] = []

        def _connect_if_present(layer, event_name: str, callback):
            events = getattr(layer, "events", None)
            emitter = getattr(events, event_name, None) if events is not None else None
            if emitter is None:
                return
            try:
                emitter.connect(callback)
                live_state["connections"].append((emitter, callback))
            except Exception:
                pass

        def schedule_preview(*_args):
            if auto_update_check.isChecked():
                debounce_timer.start()

        def refresh_prompt_watchers():
            clear_event_connections()
            image_name = image_combo.currentText().strip()
            prompt_name = prompt_layer_combo.currentText().strip()
            image_layer = viewer.layers[image_name] if viewer is not None and image_name in viewer.layers else None
            prompt_layer = viewer.layers[prompt_name] if viewer is not None and prompt_name in viewer.layers else None
            if image_layer is not None:
                _connect_if_present(image_layer, "data", schedule_preview)
            if mode_combo.currentText().strip() == "Box":
                if prompt_layer is not None:
                    _connect_if_present(prompt_layer, "data", schedule_preview)
                    _connect_if_present(prompt_layer, "features", schedule_preview)
            else:
                layer_name = str(live_state.get("managed_points_layer") or "").strip()
                layer = viewer.layers[layer_name] if viewer is not None and layer_name in viewer.layers else None
                if layer is not None:
                    _connect_if_present(layer, "data", schedule_preview)
                    _connect_if_present(layer, "features", schedule_preview)

        def refresh_live_controls(*_args):
            current_image = image_combo.currentText().strip()
            image_names = image_layer_names_sam2()
            image_combo.blockSignals(True)
            image_combo.clear()
            image_combo.addItems(image_names)
            if current_image in image_names:
                image_combo.setCurrentText(current_image)
            image_combo.blockSignals(False)

            managed_name = str(live_state.get("managed_points_layer") or "").strip()
            expected_name = managed_points_layer_name() or ""
            if managed_name and managed_name != expected_name:
                live_state["initialized"] = False
                live_state["managed_points_layer"] = None
                live_state["seed_slice"] = None

            image_layer = current_image_layer()
            image_ndim = np.asarray(image_layer.data).ndim if image_layer is not None else None
            if image_ndim == 3 and mode_combo.currentText().strip() == "Box":
                mode_combo.blockSignals(True)
                mode_combo.setCurrentText("Points")
                mode_combo.blockSignals(False)
            if image_ndim == 2:
                hint.setText(
                    "Live SAM2 preview for 2D grayscale images. Use a Shapes layer for Box mode "
                    "or click Initialize to start a SAM2-managed points session. Press T on the active SAM2 points layer to toggle polarity."
                )
                preview_btn.setText("Run Preview")
                apply_btn.setText("Save Labels")
                auto_update_check.setEnabled(True)
                auto_update_check.setText("Auto-update")
            elif image_ndim == 3:
                hint.setText(
                    "Live SAM2 propagation for 3D grayscale images. Click Initialize on the desired seed slice, "
                    "place positive/negative prompts on that slice, then click Propagate. Press T on the active SAM2 points layer to toggle polarity."
                )
                preview_btn.setText("Propagate")
                apply_btn.setText("Save Labels")
                auto_update_check.setChecked(False)
                auto_update_check.setEnabled(False)
                auto_update_check.setText("Auto-update disabled for 3D")
            else:
                hint.setText(
                    "Live SAM2 preview for grayscale images. Use a Shapes layer for 2D Box mode or initialize a "
                    "SAM2-managed points session for 2D/3D points prompting. Press T on the active SAM2 points layer to toggle polarity."
                )
                preview_btn.setText("Run Preview")
                apply_btn.setText("Save Labels")
                auto_update_check.setEnabled(True)
                auto_update_check.setText("Auto-update")

            current_prompt = prompt_layer_combo.currentText().strip()
            prompt_names = prompt_layer_names_for_mode()
            prompt_layer_combo.blockSignals(True)
            prompt_layer_combo.clear()
            prompt_layer_combo.addItems(prompt_names)
            if current_prompt in prompt_names:
                prompt_layer_combo.setCurrentText(current_prompt)
            prompt_layer_combo.blockSignals(False)

            is_points_mode = mode_combo.currentText().strip() == "Points"
            set_form_row_visible(prompt_layer_combo, not is_points_mode)
            set_form_row_visible(polarity_combo, is_points_mode)
            if is_points_mode and live_state.get("managed_points_layer"):
                layer_name = str(live_state["managed_points_layer"])
                if viewer is not None and layer_name in viewer.layers:
                    layer = viewer.layers[layer_name]
                    if isinstance(layer, napari.layers.Points):
                        apply_points_polarity_defaults(layer)
            refresh_prompt_watchers()

        def on_polarity_changed(*_args):
            layer_name = str(live_state.get("managed_points_layer") or "").strip()
            if viewer is not None and layer_name in viewer.layers:
                layer = viewer.layers[layer_name]
                if isinstance(layer, napari.layers.Points):
                    apply_points_polarity_defaults(layer)
                    try:
                        viewer.layers.selection.active = layer
                    except Exception:
                        pass
            set_live_status(
                f"SAM2 prompt polarity set to {active_polarity_name()}. "
                "Add points now.",
                ok=None,
            )

        def upsert_preview_layer(mask: np.ndarray, *, image_name: str, scale, translate):
            if viewer is None:
                return
            if preview_layer_name in viewer.layers:
                layer = viewer.layers[preview_layer_name]
                if isinstance(layer, napari.layers.Labels):
                    layer.data = np.asarray(mask, dtype=np.int32)
                    layer.scale = scale
                    layer.translate = translate
                    return
            viewer.add_labels(np.asarray(mask, dtype=np.int32), name=preview_layer_name, scale=scale, translate=translate)

        def run_live_preview(*_args):
            if viewer is None:
                return
            image_name = image_combo.currentText().strip()
            tool_name = current_prompt_tool_name()
            if not tool_name:
                set_live_status("Current SAM2 Live mode is not supported for this image dimensionality.", ok=False)
                return
            arguments = {"image_layer": image_name}
            if tool_name == "sam_segment_from_box":
                prompt_name = prompt_layer_combo.currentText().strip()
                if not image_name or not prompt_name:
                    set_live_status("Select both an image layer and a box prompt layer.", ok=False)
                    return
                arguments["roi_layer"] = prompt_name
                prompt_summary = prompt_name
            else:
                if not live_state.get("initialized"):
                    set_live_status("Click Initialize before adding prompts and running SAM2 Live.", ok=False)
                    return
                combined_name = sync_combined_points_layer()
                if not image_name or not combined_name:
                    set_live_status(
                        "Click Initialize, then add prompt points on the managed SAM2 layer before running preview or propagation.",
                        ok=False,
                    )
                    return
                arguments["points_layer"] = combined_name
                prompt_layer_name = str(live_state.get("managed_points_layer") or "-")
                seed_info = ""
                if current_image_ndim() == 3 and live_state.get("seed_slice") is not None:
                    seed_info = f" seed_slice={int(live_state['seed_slice'])}"
                prompt_summary = f"prompts=[{prompt_layer_name}]{seed_info}"
            prepared = prepare_tool_job(viewer, tool_name, arguments)
            if prepared.get("mode") == "immediate":
                set_live_status(str(prepared.get("message", "")), ok=False)
                return
            if live_state["busy"]:
                live_state["rerun"] = True
                return
            live_state["busy"] = True
            set_live_progress(True)
            live_state["request_id"] += 1
            request_id = int(live_state["request_id"])
            action_label = "Propagating" if current_image_ndim() == 3 and tool_name == "sam_propagate_points_3d" else "Running live preview"
            set_live_status(f"{action_label} with {tool_name}...", ok=None)

            @thread_worker(ignore_errors=True)
            def run_live_job():
                return run_tool_job(prepared["job"])

            worker = run_live_job()
            active_workers.append(worker)

            def finish_live_job():
                if worker in active_workers:
                    active_workers.remove(worker)
                live_state["busy"] = False
                set_live_progress(False)
                if live_state["rerun"]:
                    live_state["rerun"] = False
                    run_live_preview()

            def on_returned(result):
                if request_id != live_state["request_id"]:
                    finish_live_job()
                    return
                payload = result.get("result") if isinstance(result, dict) else None
                if not isinstance(payload, np.ndarray):
                    payload = np.asarray(result.get("result")) if isinstance(result, dict) and result.get("result") is not None else None
                if payload is None:
                    set_live_status("SAM2 live preview returned no preview mask.", ok=False)
                    finish_live_job()
                    return
                scale = tuple(result.get("scale", (1.0, 1.0)))
                translate = tuple(result.get("translate", (0.0, 0.0)))
                upsert_preview_layer(payload, image_name=image_name, scale=scale, translate=translate)
                live_state["last_payload"] = {"mask": np.asarray(payload, dtype=np.int32).copy(), "image_name": image_name, "scale": scale, "translate": translate}
                backend_message = str(result.get("backend_message") or "").strip()
                status_prefix = "SAM2 propagation updated" if current_image_ndim() == 3 and tool_name == "sam_propagate_points_3d" else "SAM2 live preview updated"
                set_live_status(
                    f"{status_prefix} for [{image_name}] using {prompt_summary} via [{tool_name}]."
                    + (f" {backend_message}" if backend_message else ""),
                    ok=True,
                )
                finish_live_job()

            def on_error(*args):
                error_text = format_worker_error(*args)
                set_live_status(f"SAM2 live preview failed: {error_text}", ok=False)
                finish_live_job()

            worker.returned.connect(on_returned)
            worker.errored.connect(on_error)
            worker.start()

        def apply_live_preview(*_args):
            payload = live_state.get("last_payload")
            if not payload:
                set_live_status("Run a live preview first before applying a labels layer.", ok=False)
                return
            image_name = str(payload["image_name"])
            output_name = next_output_name(viewer, f"{image_name}_sam2_live")
            viewer.add_labels(
                np.asarray(payload["mask"], dtype=np.int32).copy(),
                name=output_name,
                scale=payload["scale"],
                translate=payload["translate"],
            )
            set_live_status(f"Saved SAM2 result as [{output_name}].", ok=True)
            append_log(f"Saved SAM2 result as {output_name}.")
            refresh_context()

        def clear_live_preview(*_args):
            live_state["last_payload"] = None
            if viewer is not None and preview_layer_name in viewer.layers:
                try:
                    viewer.layers.remove(viewer.layers[preview_layer_name])
                except Exception:
                    pass
            set_live_status("Cleared SAM2 live preview.", ok=None)
            refresh_context()

        def cleanup_live_dialog():
            clear_event_connections()
            debounce_timer.stop()
            set_live_progress(False)

        init_btn.clicked.connect(initialize_live_session)
        debounce_timer.timeout.connect(run_live_preview)
        preview_btn.clicked.connect(run_live_preview)
        apply_btn.clicked.connect(apply_live_preview)
        clear_btn.clicked.connect(clear_live_preview)
        close_btn.clicked.connect(dialog.accept)
        model_combo.currentTextChanged.connect(apply_live_model_selection)
        image_combo.currentTextChanged.connect(refresh_live_controls)
        prompt_layer_combo.currentTextChanged.connect(refresh_prompt_watchers)
        mode_combo.currentTextChanged.connect(refresh_live_controls)
        polarity_combo.currentTextChanged.connect(on_polarity_changed)
        auto_update_check.toggled.connect(
            lambda checked: set_live_status(
                "Auto-update enabled." if checked else ("Auto-update disabled for 3D." if current_image_ndim() == 3 else "Auto-update disabled."),
                ok=None,
            )
        )
        dialog.finished.connect(lambda *_args: cleanup_live_dialog())
        register_viewer_toggle_binding()
        refresh_live_model_choices()
        refresh_live_controls()
        append_log("Opened SAM2 live dialog.")
        set_status("Status: SAM2 live opened", ok=None)
        dialog.setModal(False)
        setattr(root, "_sam2_live_dialog", dialog)
        dialog.destroyed.connect(lambda *_args: setattr(root, "_sam2_live_dialog", None))
        dialog.show()

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
            validation_report = preflight_generated_code(manual_code)
            code_message = "Manual code captured from the prompt. Review it, then click Run Code."
            if validation_report.errors:
                set_pending_code(
                    manual_code,
                    message=code_message,
                    runnable=False,
                    label="Pending code: blocked by validation",
                    validation_mode="permissive",
                    code_source="user",
                )
                append_chat_message(
                    "assistant",
                    format_validation_report(
                        validation_report,
                        mode="permissive",
                        heading="Pasted code was rejected by local validation.",
                        include_notes=False,
                    )
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
            pending_code["validation_mode"] = "permissive"
            pending_code["code_source"] = "user"
            warning_prefix = ""
            if validation_report.warnings:
                warning_prefix = (
                    format_validation_report(
                        validation_report,
                        mode="permissive",
                        heading="Manual code has validation warnings.",
                        include_notes=False,
                    )
                    + "\n\n"
                )
            append_chat_message("assistant", f"{warning_prefix}{code_message}\n{format_code_block(manual_code)}")
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
        ui_help_reply = answer_ui_question(text)
        if ui_help_reply:
            append_chat_message("assistant", ui_help_reply)
            append_log("Answered plugin UI help question locally.")
            set_status("Status: plugin help ready", ok=True)
            record_telemetry(
                "turn_completed",
                {
                    "turn_id": turn_id,
                    "model": "local_ui_help",
                    "base_url": "",
                    "prompt_hash": prompt_hash(text),
                    "prompt_category": "ui_help",
                    "response_action": "reply",
                    "pending_code_generated": False,
                    **selected_layer_snapshot(),
                },
            )
            return
        base_url = base_url_edit.text().strip().rstrip("/") or str(saved_settings["base_url"]).rstrip("/")
        model_name = model_combo.currentText().strip() or str(saved_settings["model"]).strip()
        if not base_url or not model_name:
            append_chat_message("assistant", "Model settings are incomplete. Choose a model and open Connection if you need to adjust the Base URL.")
            set_status("Status: missing saved model settings", ok=False)
            return
        local_workflow_route = route_local_workflow_prompt(text, selected_layer_profile())
        if isinstance(local_workflow_route, dict):
            tool_name = str(local_workflow_route.get("tool", "")).strip()
            arguments = local_workflow_route.get("arguments", {})
            tool_message = str(local_workflow_route.get("message", "")).strip()
            prepared = prepare_tool_job(viewer, tool_name, arguments if isinstance(arguments, dict) else {})
            if prepared.get("mode") == "immediate":
                result_message = str(prepared.get("message", "")).strip() or f"Could not run [{tool_name}]."
                append_chat_message("assistant", f"{tool_message}\n{result_message}" if tool_message else result_message)
                append_log(f"Handled request via local workflow route: {tool_name}")
                set_status(f"Status: {tool_name} completed", ok=True)
                record_telemetry(
                    "turn_completed",
                    {
                        "turn_id": turn_id,
                        "model": "local_workflow_router",
                        "base_url": "",
                        "prompt_hash": prompt_hash(text),
                        "prompt_category": "local_workflow_route",
                        "response_action": "tool",
                        "tool_name": tool_name,
                        "tool_success": True,
                        "pending_code_generated": False,
                        **selected_layer_snapshot(),
                    },
                )
            else:
                job = prepared["job"]
                start_wait_indicator(phase=f"processing {tool_name}")
                set_status(f"Status: running tool {tool_name}", ok=None)

                @thread_worker(ignore_errors=True)
                def run_local_workflow_tool():
                    return run_tool_job(job)

                tool_worker = run_local_workflow_tool()
                active_workers.append(tool_worker)

                def local_tool_finish():
                    stop_wait_indicator()
                    if tool_worker in active_workers:
                        active_workers.remove(tool_worker)

                def local_tool_returned(tool_result):
                    refresh_context()
                    result_message = apply_tool_job_result(viewer, tool_result)
                    append_chat_message("assistant", f"{tool_message}\n{result_message}" if tool_message else result_message)
                    append_log(f"Handled request via local workflow route: {tool_name}")
                    set_status(f"Status: {tool_name} completed", ok=True)
                    record_telemetry(
                        "turn_completed",
                        {
                            "turn_id": turn_id,
                            "model": "local_workflow_router",
                            "base_url": "",
                            "prompt_hash": prompt_hash(text),
                            "prompt_category": "local_workflow_route",
                            "response_action": "tool",
                            "tool_name": tool_name,
                            "tool_success": True,
                            "pending_code_generated": False,
                            **selected_layer_snapshot(),
                        },
                    )
                    local_tool_finish()

                def local_tool_error(*args):
                    error_text = format_worker_error(*args)
                    logger.exception("Local workflow route failed: %s | %s", tool_name, error_text)
                    append_chat_message("assistant", f"{tool_name} failed:\n{error_text}")
                    append_log(f"Local workflow route failed: {tool_name} | {error_text}")
                    set_status(f"Status: {tool_name} failed", ok=False)
                    local_tool_finish()

                tool_worker.returned.connect(local_tool_returned)
                tool_worker.errored.connect(local_tool_error)
                tool_worker.start()
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
            code_repair_context = build_code_repair_context(text, viewer=viewer)
            return chat_ollama(
                base_url,
                model_name,
                system_prompt=assistant_system_prompt(),
                user_payload={
                    "viewer_context": viewer_payload,
                    "session_memory": build_session_memory_payload(session_memory_state, viewer_payload.get("selected_layer_profile")),
                    "code_repair_context": code_repair_context,
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
                    try:
                        if "job" in prepared:
                            tool_result = run_tool_job(prepared["job"])
                            result_message = apply_tool_job_result(viewer, tool_result)
                        else:
                            result_message = str(prepared.get("message", ""))
                    except Exception as exc:
                        logger.exception("Immediate tool failed: %s", tool_name)
                        replace_last_assistant(f"{tool_name} failed:\n{exc}")
                        set_status(f"Status: {tool_name} failed", ok=False)
                        append_log(f"Immediate tool failed: {tool_name} | {exc}")
                        finish()
                        return
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
                    set_status(f"Status: response received from {model_name}", ok=True)
                    append_log(f"Received response from {model_name}")
                    finish()
                    return

                job = prepared["job"]
                set_wait_indicator_phase(f"processing {tool_name}")
                set_status(f"Status: running tool {tool_name}", ok=None)

                @thread_worker(ignore_errors=True)
                def run_backend_tool():
                    return run_tool_job(job)

                tool_worker = run_backend_tool()
                active_workers.append(tool_worker)

                def tool_finish():
                    if tool_worker in active_workers:
                        active_workers.remove(tool_worker)
                    set_status(f"Status: {tool_name} completed", ok=True)
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
                    set_status(f"Status: {tool_name} failed", ok=False)
                    if tool_worker in active_workers:
                        active_workers.remove(tool_worker)
                    finish()

                tool_worker.returned.connect(tool_returned)
                tool_worker.errored.connect(tool_error)
                tool_worker.start()
                return

            if action == "code":
                raw_code_text = str(parsed.get("code", "")).strip()
                code_text, validation_report = normalize_generated_code_if_needed(raw_code_text, viewer=viewer)
                code_message = str(parsed.get("message", "")).strip() or "Generated napari code. Review it, then click Run Code."
                if validation_report.has_blocking_issues("strict"):
                    set_pending_code(
                        code_text,
                        message=code_message,
                        runnable=False,
                        label="Pending code: blocked by validation",
                        validation_mode="strict",
                        code_source="assistant",
                    )
                    pending_code["turn_id"] = turn_id
                    pending_code["model"] = model_name
                    replace_last_assistant(
                        format_validation_report(
                            validation_report,
                            mode="strict",
                            heading="Generated code was rejected by local validation.",
                        )
                        + "\n\nReview or copy the generated code below, then ask the assistant to regenerate or fix it.\n\n"
                        + format_code_block(code_text)
                    )
                    append_log("Rejected generated code after local validation.")
                    set_status("Status: code rejected", ok=False)
                    finish()
                    return
                set_pending_code(code_text, message=code_message, validation_mode="strict", code_source="assistant")
                pending_code["turn_id"] = turn_id
                pending_code["model"] = model_name
                report_prefix = ""
                if validation_report.notes or validation_report.warnings:
                    report_prefix = format_validation_report(
                        validation_report,
                        mode="strict",
                        heading="Local validation summary for generated code.",
                    ) + "\n\n"
                replace_last_assistant(f"{report_prefix}{code_message}\n{format_code_block(code_text)}")
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
                set_status("Status: generated code ready for review", ok=None)
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
            set_status(f"Status: reply ready from {model_name}", ok=True)
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
            code_label="your code" if pending_code.get("code_source") == "user" else "approved napari code",
            validation_mode=pending_code.get("validation_mode", "strict"),
            turn_id=pending_code.get("turn_id", ""),
            model_name=pending_code.get("model", ""),
            code_source=pending_code.get("code_source", "assistant"),
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
        validation_mode: ValidationMode = "strict",
        turn_id: str = "",
        model_name: str = "",
        code_source: str = "assistant",
        disable_pending_buttons: bool = False,
    ) -> bool:
        normalized_code_text = str(code_text or "").strip()
        validation_report = preflight_generated_code(code_text)
        if validation_report.has_blocking_issues(validation_mode):
            if code_source == "user":
                last_user_code_failure["code"] = normalized_code_text
                last_user_code_failure["error"] = format_validation_report(
                    validation_report,
                    mode=validation_mode,
                    heading=f"{code_label.capitalize()} failed local validation.",
                )
                refresh_code_action_buttons()
            append_chat_message(
                "assistant",
                format_validation_report(
                    validation_report,
                    mode=validation_mode,
                    heading=f"{code_label.capitalize()} failed local validation.",
                ),
            )
            append_log(f"{code_label.capitalize()} blocked by local validation.")
            set_status(f"Status: {code_label} blocked", ok=False)
            return False
        if validation_report.warnings:
            append_chat_message(
                "assistant",
                format_validation_report(
                    validation_report,
                    mode=validation_mode,
                    heading=f"{code_label.capitalize()} has validation warnings.",
                ),
            )
            append_log(f"{code_label.capitalize()} allowed with validation warnings.")

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
            if code_source == "user":
                last_user_code_failure["code"] = normalized_code_text
                last_user_code_failure["error"] = error_text
                refresh_code_action_buttons()
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
        if code_source == "user":
            last_user_code_failure["code"] = ""
            last_user_code_failure["error"] = ""
            refresh_code_action_buttons()
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
        set_status(f"Status: {code_label} done", ok=True)
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
        run_code_text(
            code_text,
            code_label="your code",
            validation_mode="permissive",
            model_name="manual",
            code_source="user",
        )

    def refine_prompt_code(*_args):
        code_text = prompt_code_candidate()
        error_text = ""
        if code_text:
            if code_text == str(last_user_code_failure.get("code", "")).strip():
                error_text = str(last_user_code_failure.get("error", "")).strip()
        else:
            code_text = str(last_user_code_failure.get("code", "")).strip()
            error_text = str(last_user_code_failure.get("error", "")).strip()
        if not code_text:
            set_status("Status: no code available to refine", ok=False)
            append_log("Refine My Code skipped: no prompt-box code or failed user code was available.")
            return
        request_lines = [
            "Refine this code so it runs in the current napari plugin environment.",
            "Preserve the original intent, explain the main fix briefly, and return corrected runnable Python.",
        ]
        if error_text:
            request_lines.extend(["", "Latest error or validation failure:", error_text])
        request_lines.extend(["", "Code:", format_code_block(code_text)])
        prompt.setPlainText("\n".join(request_lines))
        append_log("Queued Refine My Code request from prompt-box or failed user code.")
        set_status("Status: refining user code", ok=None)
        send_message()

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
        code_text = str(record.get("code", ""))
        prompt.setPlainText(code_text)
        prompt.setFocus()
        append_log(f"Loaded code from library: {record.get('title', 'Untitled Code')}")
        set_status("Status: code loaded from library", ok=None)

    def run_library_code(item: QListWidgetItem):
        record = item.data(Qt.UserRole) or {}
        code_text = str(record.get("code", ""))
        if not code_text.strip():
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
    refine_my_code_btn.clicked.connect(refine_prompt_code)
    copy_code_btn.clicked.connect(copy_pending_code)
    reject_memory_btn.clicked.connect(reject_last_memory)
    sam2_setup_action.triggered.connect(show_sam2_setup_dialog)
    sam2_live_action.triggered.connect(show_sam2_live_dialog)
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
    prompt.textChanged.connect(refresh_code_action_buttons)
    code_library_list.customContextMenuRequested.connect(show_library_context_menu)
    template_tree.currentItemChanged.connect(show_template_preview)
    template_tree.itemDoubleClicked.connect(run_template_tree_item)
    template_load_btn.clicked.connect(load_selected_template)
    template_run_btn.clicked.connect(run_selected_template)
    save_prompt_btn.clicked.connect(save_current_prompt)
    pin_prompt_btn.clicked.connect(toggle_pin_selected_prompt)
    delete_prompt_btn.clicked.connect(delete_selected_prompt)
    clear_prompt_btn.clicked.connect(clear_non_saved_prompts)
    prompt_font_down_btn.clicked.connect(lambda *_args: adjust_prompt_library_font(-1))
    prompt_font_up_btn.clicked.connect(lambda *_args: adjust_prompt_library_font(1))
    prompt.sendRequested.connect(send_message)
    connection_toggle_btn.toggled.connect(connection_details.setVisible)
    log_group.toggled.connect(log_tabs.setVisible)
    wait_timer.timeout.connect(tick_wait_indicator)
    root.destroyed.connect(cleanup_workers)

    connect_viewer_context_events()
    refresh_telemetry_controls()
    refresh_context()
    set_pending_code()
    refresh_code_action_buttons()
    refresh_models()
    refresh_sam2_actions()
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
            "🧭 Viewer state is read live when you send requests, and the summary strip shows what is currently open.\n"
            "📚 Library tabs keep reusable prompts and code close to the workflow.\n"
            "📝 Action Log tracks local actions. Telemetry is optional and stays off unless you enable it.\n\n"
            "Ask about your selected layer, thresholding, CLAHE, measurements, histograms, comparisons, or code.\n\n"
            "Follow for updates: https://x.com/viralvector",
        )
        ui_state["welcome_dismissed"] = True
        save_ui_state(ui_state)
    append_log("Assistant panel initialized.")
    return root
