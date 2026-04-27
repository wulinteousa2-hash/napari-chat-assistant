from __future__ import annotations

import io
import json
import hashlib
import importlib
import re
import threading
import time
import uuid
from contextlib import redirect_stdout
from pathlib import Path

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
    QFileDialog,
    QFormLayout,
    QGridLayout,
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
    QStackedWidget,
    QTabBar,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_chat_assistant.agent.action_library import action_library_payload
from napari_chat_assistant import __version__
from napari_chat_assistant.agent.client import chat_ollama, list_ollama_models, load_ollama_model, unload_ollama_model
from napari_chat_assistant.agent.code_validation import (
    ValidationMode,
    ValidationReport,
    build_code_repair_context,
    normalize_generated_code_if_needed,
    validate_generated_code,
)
from napari_chat_assistant.agent.context import get_viewer, layer_context_for_model, layer_context_json, layer_summary
from napari_chat_assistant.agent.dispatcher import (
    apply_tool_job_result,
    prepare_tool_job,
    restore_viewer_control_snapshot,
    run_tool_job,
    run_tool_sequence,
)
from napari_chat_assistant.agent.followup_semantics import parse_followup_constraint
from napari_chat_assistant.telemetry import (
    APP_LOG_PATH,
    CRASH_LOG_PATH,
    TELEMETRY_LOG_PATH,
    append_telemetry_event,
    enable_fault_logging,
    get_plugin_logger,
)
from napari_chat_assistant.agent.intent_state import (
    empty_failed_tool_state,
    empty_intent_state,
    extract_turn_intent,
    merge_intent_state,
    remember_failed_tool,
    should_block_tool,
    should_skip_local_workflow_route,
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
from napari_chat_assistant.agent.workflow_executor import execute_workflow_plan
from napari_chat_assistant.agent.workflow_executor import (
    workflow_execution_to_compact_markdown,
    workflow_execution_to_debug_markdown,
)
from napari_chat_assistant.agent.workflow_planner import workflow_plan_to_markdown
from napari_chat_assistant.agent.pending_action import (
    advance_pending_action_turn,
    build_pending_action_from_assistant_message,
    cancel_pending_action,
    complete_pending_action,
    empty_pending_action,
    is_pending_action_cancel_message,
    is_pending_action_waiting,
    normalize_pending_action,
    resolve_pending_action,
)
from napari_chat_assistant.agent.profiler import profile_layer
from napari_chat_assistant.agent.prompt_routing import route_local_workflow_prompt
from napari_chat_assistant.agent.recent_action_state import (
    empty_recent_action_state,
    latest_recent_action,
    record_recent_action,
    route_recent_action_followup,
)
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
from napari_chat_assistant.telemetry import (
    format_markdown_telemetry_report,
    format_telemetry_summary,
    load_telemetry_events,
    read_telemetry_tail,
    summarize_telemetry_events,
)
from napari_chat_assistant.telemetry.intent_tracker import (
    IntentEvent,
    build_layer_context,
    categorize_intent,
    record_intent,
)
from napari_chat_assistant.agent.tools import ASSISTANT_TOOL_NAMES, assistant_system_prompt, next_output_name
from napari_chat_assistant.agent.ui_help import answer_ui_question
from napari_chat_assistant.agent.ui_state import load_ui_state, save_ui_state
from napari_chat_assistant.library.template_catalog import (
    is_template_record,
    template_body_text,
    template_button_labels,
    template_hint_text,
    template_library_payload,
    template_load_target,
    template_preview_text as catalog_template_preview_text,
    template_run_target,
    template_section_colors,
)
from napari_chat_assistant.widgets.chat_sections import LayerContextPanel, LibraryPanel, PendingCodePanel, ShortcutsPanel
from napari_chat_assistant.widgets.message_formatting import render_assistant_message_html, render_user_message_html


_WORKSPACE_DEPENDENCY_MESSAGE = (
    "Workspace save/load is not available in this environment.\n\n"
    "Install or repair the workspace dependencies in the same Python environment as napari:\n"
    "`pip install numcodecs zarr ome-zarr`\n\n"
    "The rest of the plugin can still be used without workspace persistence."
)


def _workspace_state_functions():
    try:
        module = importlib.import_module("napari_chat_assistant.agent.workspace_state")
    except Exception as exc:
        raise RuntimeError(_WORKSPACE_DEPENDENCY_MESSAGE) from exc
    return module.load_workspace_manifest, module.save_workspace_manifest


def _workspace_state_module():
    try:
        return importlib.import_module("napari_chat_assistant.agent.workspace_state")
    except Exception as exc:
        raise RuntimeError(_WORKSPACE_DEPENDENCY_MESSAGE) from exc


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
    root.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
    splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    layout.addWidget(splitter, 1)

    left_panel = QWidget()
    left_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    left_panel.setMinimumWidth(360)
    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(0, 0, 0, 0)

    right_panel = QWidget()
    right_panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)

    layer_context_panel = LayerContextPanel()
    context_group = layer_context_panel
    context_tabs = layer_context_panel.context_tabs
    context_summary_box = layer_context_panel.context_summary_box
    context_layers_list = layer_context_panel.context_layers_list
    context_selected_only_checkbox = layer_context_panel.context_selected_only_checkbox
    left_layout.addWidget(context_group, 0)

    library_panel = LibraryPanel()
    prompt_library_group = library_panel
    prompt_library_hint = library_panel.prompt_library_hint
    library_tabs = library_panel.library_tabs
    actions_tab_btn = library_panel.actions_tab_btn
    library_stack = library_panel.library_stack
    prompt_library_list = library_panel.prompt_library_list
    code_library_list = library_panel.code_library_list
    template_tab = library_panel.template_tab
    template_tree = library_panel.template_tree
    template_preview = library_panel.template_preview
    template_load_btn = library_panel.template_load_btn
    template_run_btn = library_panel.template_run_btn
    action_tab = library_panel.action_tab
    action_tree = library_panel.action_tree
    action_preview = library_panel.action_preview
    action_load_btn = library_panel.action_load_btn
    action_run_btn = library_panel.action_run_btn
    action_add_shortcut_btn = library_panel.action_add_shortcut_btn
    save_prompt_btn = library_panel.save_prompt_btn
    pin_prompt_btn = library_panel.pin_prompt_btn
    delete_prompt_btn = library_panel.delete_prompt_btn
    clear_prompt_btn = library_panel.clear_prompt_btn
    prompt_font_down_btn = library_panel.prompt_font_down_btn
    prompt_font_up_btn = library_panel.prompt_font_up_btn
    left_layout.addWidget(prompt_library_group, 2)

    shortcuts_panel = ShortcutsPanel()
    shortcuts_group = shortcuts_panel
    shortcuts_content = shortcuts_panel.shortcuts_content
    shortcuts_layout = shortcuts_panel.shortcuts_layout
    shortcuts_hint = shortcuts_panel.shortcuts_hint
    shortcuts_grid = shortcuts_panel.shortcuts_grid
    shortcut_buttons: list[QPushButton] = []
    shortcuts_btn_row = shortcuts_panel.shortcuts_btn_row
    shortcuts_add_row_btn = shortcuts_panel.shortcuts_add_row_btn
    shortcuts_remove_row_btn = shortcuts_panel.shortcuts_remove_row_btn
    shortcuts_save_btn = shortcuts_panel.shortcuts_save_btn
    shortcuts_load_btn = shortcuts_panel.shortcuts_load_btn
    shortcuts_clear_btn = shortcuts_panel.shortcuts_clear_btn
    shortcuts_group.setCheckable(True)
    shortcuts_group.setChecked(bool(ui_state.get("shortcuts_group_open", True)))
    shortcuts_content.setVisible(shortcuts_group.isChecked())
    left_layout.addWidget(shortcuts_group, 0)

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
    telemetry_toggle.setToolTip("Turn on telemetry recording. Summary, Copy Report, and Log can inspect existing local telemetry even when recording is off.")
    telemetry_summary_btn = QPushButton("Summary")
    telemetry_summary_btn.setToolTip(
        "Show a quick performance summary from local telemetry only."
    )
    telemetry_report_btn = QPushButton("Copy Report")
    telemetry_report_btn.setToolTip(
        "Copy the full shareable telemetry Markdown report to the clipboard."
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
    log_btn_layout.addWidget(telemetry_report_btn)
    log_btn_layout.addWidget(telemetry_view_btn)
    log_btn_layout.addWidget(telemetry_reset_btn)
    telemetry_hint = QLabel("Telemetry recording is optional. Use Summary, Copy Report, or Log to inspect existing local telemetry.")
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
    workspace_tab = QWidget()
    workspace_layout = QVBoxLayout(workspace_tab)
    workspace_hint = QLabel("Save or restore a lightweight workspace manifest so layer order and display state can be recovered later.")
    workspace_hint.setWordWrap(True)
    workspace_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    workspace_hint.setStyleSheet("QLabel { color: #cbd5e1; padding: 0 0 4px 0; }")
    workspace_layout.addWidget(workspace_hint)
    workspace_path_label = QLabel("Workspace file: [none]")
    workspace_path_label.setWordWrap(True)
    workspace_layout.addWidget(workspace_path_label)
    workspace_clear_checkbox = QCheckBox("Clear current layers before load")
    workspace_clear_checkbox.setChecked(True)
    workspace_layout.addWidget(workspace_clear_checkbox)
    workspace_btn_row = QWidget()
    workspace_btn_layout = QHBoxLayout(workspace_btn_row)
    save_workspace_btn = QPushButton("Save Workspace")
    save_workspace_as_btn = QPushButton("Save As")
    load_workspace_btn = QPushButton("Load Workspace")
    restore_workspace_btn = QPushButton("Restore Last")
    workspace_btn_layout.addWidget(save_workspace_btn)
    workspace_btn_layout.addWidget(save_workspace_as_btn)
    workspace_btn_layout.addWidget(load_workspace_btn)
    workspace_btn_layout.addWidget(restore_workspace_btn)
    workspace_layout.addWidget(workspace_btn_row)
    workspace_status = QLabel("Workspace manifest saves file-backed layers plus recoverable geometry layers such as Shapes.")
    workspace_status.setWordWrap(True)
    workspace_layout.addWidget(workspace_status)
    workspace_layout.addStretch(1)
    log_tabs.addTab(activity_tab, "Activity")
    log_tabs.addTab(workspace_tab, "Workspace")
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

    pending_code_panel = PendingCodePanel(ui_help_enabled=bool(ui_state.get("ui_help_enabled", False)))
    pending_code_label = pending_code_panel.pending_code_label
    help_btn = pending_code_panel.help_btn
    help_whats_new_action = pending_code_panel.help_whats_new_action
    help_about_action = pending_code_panel.help_about_action
    help_report_bug_action = pending_code_panel.help_report_bug_action
    help_ui_toggle_action = pending_code_panel.help_ui_toggle_action
    advanced_btn = pending_code_panel.advanced_btn
    feedback_btn = pending_code_panel.feedback_btn
    feedback_helpful_action = pending_code_panel.feedback_helpful_action
    feedback_wrong_route_action = pending_code_panel.feedback_wrong_route_action
    feedback_wrong_answer_action = pending_code_panel.feedback_wrong_answer_action
    feedback_didnt_work_action = pending_code_panel.feedback_didnt_work_action
    voice_input_action = pending_code_panel.voice_input_action
    sam2_setup_action = pending_code_panel.sam2_setup_action
    sam2_live_action = pending_code_panel.sam2_live_action
    text_annotation_action = pending_code_panel.text_annotation_action
    atlas_stitch_action = pending_code_panel.atlas_stitch_action
    run_code_btn = pending_code_panel.run_code_btn
    run_my_code_btn = pending_code_panel.run_my_code_btn
    refine_my_code_btn = pending_code_panel.refine_my_code_btn
    copy_code_btn = pending_code_panel.copy_code_btn
    stop_btn = pending_code_panel.stop_btn
    chat_font_down_btn = pending_code_panel.chat_font_down_btn
    chat_font_up_btn = pending_code_panel.chat_font_up_btn
    transcript_layout.addWidget(pending_code_panel)

    input_group = QGroupBox("Prompt")
    input_group_layout = QVBoxLayout(input_group)
    prompt = ChatInput()
    prompt.setObjectName("napariChatAssistantPrompt")
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
    workspace_load_state = {
        "active": False,
        "dialog": None,
        "label": None,
        "bar": None,
        "module": None,
        "path": "",
        "payload": None,
        "records": [],
        "index": 0,
        "restored_layers": [],
        "skipped_layers": [],
        "clear_existing": True,
        "phase": "prepare",
        "source_layers": 0,
        "base_layer_count": 0,
        "restored_records": [],
    }
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
    last_turn_metrics = {
        "turn_id": "",
        "model": "",
        "action": "",
        "prompt_hash": "",
        "intent_category": "",
        "predicted_route": "",
        "actual_route": "",
        "outcome_type": "",
        "tool_name": "",
        "success": None,
        "latency_ms": 0,
        "feedback": "",
    }
    pending_action_state = empty_pending_action()
    session_memory_state = load_session_memory()
    recent_action_state = empty_recent_action_state()
    intent_state = empty_intent_state()
    last_failed_tool_state = empty_failed_tool_state()
    last_tool_sequence_undo_snapshot: dict = {}
    last_workflow_plan_payload: dict = {}
    last_workflow_execution_payload: dict = {}
    context_visibility_snapshot: dict[str, bool] | None = None
    last_memory_candidate_ids: list[str] = []
    prompt_library_state = load_prompt_library()
    template_library_state = template_library_payload()
    action_library_state = action_library_payload()
    shortcut_action_ids = list(ui_state.get("shortcuts_action_ids", []) or [])
    shortcut_slot_count = max(6, int(ui_state.get("shortcuts_slot_count", 6) or 6))
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
    local_workflow_marker = "[Local workflow]"

    def refresh_workspace_path_label() -> None:
        current_path = str(ui_state.get("last_workspace_path", "")).strip()
        workspace_path_label.setText(f"Workspace file: {current_path if current_path else '[none]'}")

    def set_workspace_controls_enabled(enabled: bool) -> None:
        for widget in (
            save_workspace_btn,
            save_workspace_as_btn,
            load_workspace_btn,
            restore_workspace_btn,
            workspace_clear_checkbox,
        ):
            widget.setEnabled(enabled)

    def close_workspace_load_dialog() -> None:
        dialog = workspace_load_state.get("dialog")
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        workspace_load_state["dialog"] = None
        workspace_load_state["label"] = None
        workspace_load_state["bar"] = None

    def update_workspace_load_progress() -> None:
        label = workspace_load_state.get("label")
        bar = workspace_load_state.get("bar")
        if label is None or bar is None:
            return
        total = len(workspace_load_state.get("records", []))
        completed = int(workspace_load_state.get("index", 0))
        current_name = ""
        if completed < total:
            record = workspace_load_state["records"][completed]
            current_name = str(record.get("name", "layer")).strip() or "layer"
        phase = str(workspace_load_state.get("phase", "prepare") or "prepare")
        phase_label = "Loading"
        if phase == "deferred_sources":
            phase_label = "Opening source data"
        if current_name:
            label.setText(f"{phase_label} {completed + 1}/{total}: {current_name}")
        else:
            label.setText(f"{phase_label} {completed}/{total} layer(s)")
        bar.setMaximum(max(total, 1))
        bar.setValue(min(completed, max(total, 1)))

    def shortcuts_default_path() -> str:
        current = str(ui_state.get("last_shortcuts_path", "")).strip()
        if current:
            return current
        shortcuts_dir = Path.home() / ".napari-chat-assistant" / "shortcuts"
        shortcuts_dir.mkdir(parents=True, exist_ok=True)
        return str(shortcuts_dir / "shortcuts.json")

    def action_record_by_id(action_id: str) -> dict | None:
        token = str(action_id or "").strip()
        if not token:
            return None
        for record in action_library_state.get("actions", []):
            if isinstance(record, dict) and str(record.get("id", "")).strip() == token:
                return record
        return None

    def quick_action_button_label(record: dict) -> str:
        title = str(record.get("title", "")).strip() or "Action"
        if len(title) <= 18:
            return title
        shortened = (
            title.replace("SAM2 ", "")
            .replace("Create ", "")
            .replace("Open ", "")
            .replace(" Analysis", "")
            .replace(" Comparison", "")
            .strip()
        )
        if 0 < len(shortened) <= 18:
            return shortened
        return f"{title[:15].rstrip()}..."

    def quick_action_button_style(record: dict) -> str:
        category = str(record.get("category", "")).strip().lower()
        palette = {
            "widgets": ("#22314a", "#dbeafe", "#31557f"),
            "layers": ("#2d2f45", "#e5e7eb", "#4b5563"),
            "enhance": ("#3f2b17", "#ffedd5", "#c67b23"),
            "visualize": ("#173b57", "#e0f2fe", "#1d6fa5"),
            "measure": ("#1e4634", "#dcfce7", "#2f855a"),
            "masks": ("#5a3415", "#ffedd5", "#c66a1b"),
            "montage": ("#153f40", "#ccfbf1", "#0f766e"),
            "segmentation": ("#4a2130", "#ffe4e6", "#be3c5d"),
            "spectral": ("#36244f", "#ede9fe", "#7c3aed"),
        }
        bg, fg, border = palette.get(category, ("#243447", "#f5f7fa", "#475569"))
        hover_bg = QColor(bg).lighter(118).name()
        hover_border = QColor(border).lighter(125).name()
        pressed_bg = QColor(bg).darker(112).name()
        return (
            "QPushButton { "
            f"background: {bg}; color: {fg}; border: 1px solid {border}; border-radius: 7px; padding: 6px 10px; "
            "font-weight: 600; } "
            "QPushButton:hover { "
            f"background: {hover_bg}; border: 1px solid {hover_border}; }} "
            "QPushButton:pressed { "
            f"background: {pressed_bg}; border: 1px solid {hover_border}; }} "
            "QPushButton:disabled { background: #1f2937; color: #94a3b8; border: 1px solid #334155; }"
        )

    def save_shortcuts() -> None:
        ui_state["shortcuts_action_ids"] = [str(action_id).strip() for action_id in shortcut_action_ids if str(action_id).strip()]
        ui_state["shortcuts_slot_count"] = max(6, int(shortcut_slot_count))
        save_ui_state(ui_state)

    def toggle_shortcuts_group(checked: bool) -> None:
        shortcuts_content.setVisible(bool(checked))
        ui_state["shortcuts_group_open"] = bool(checked)
        save_ui_state(ui_state)

    def save_shortcuts_layout(*, choose_path: bool) -> None:
        destination = shortcuts_default_path()
        if choose_path or not destination.strip():
            selected, _ = QFileDialog.getSaveFileName(
                root,
                "Save Shortcuts Layout",
                destination or "shortcuts.json",
                "JSON Files (*.json)",
            )
            if not selected:
                return
            destination = selected
        payload = {
            "version": 1,
            "slot_count": max(6, int(shortcut_slot_count)),
            "action_ids": [str(action_id).strip() for action_id in shortcut_action_ids if str(action_id).strip()],
        }
        try:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            append_log(f"Save shortcuts failed: {exc}")
            set_status("Status: save shortcuts failed", ok=False)
            return
        ui_state["last_shortcuts_path"] = str(destination)
        save_ui_state(ui_state)
        append_log(f"Saved shortcuts layout: {destination}")
        set_status("Status: shortcuts saved", ok=True)

    def load_shortcuts_layout(*, choose_path: bool) -> None:
        nonlocal shortcut_slot_count
        source_path = shortcuts_default_path()
        if choose_path or not source_path.strip():
            selected, _ = QFileDialog.getOpenFileName(
                root,
                "Load Shortcuts Layout",
                source_path or "",
                "JSON Files (*.json)",
            )
            if not selected:
                return
            source_path = selected
        if not source_path.strip():
            set_status("Status: no shortcuts file selected", ok=False)
            return
        try:
            payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
        except Exception as exc:
            append_log(f"Load shortcuts failed: {exc}")
            set_status("Status: load shortcuts failed", ok=False)
            return
        slot_count = max(6, int(payload.get("slot_count", 6)))
        raw_ids = [str(value).strip() for value in payload.get("action_ids", []) if str(value).strip()]
        merged_ids: list[str] = []
        for action_id in raw_ids:
            if action_record_by_id(action_id) is not None and action_id not in merged_ids:
                merged_ids.append(action_id)
        shortcut_slot_count = slot_count
        shortcut_action_ids[:] = merged_ids
        while len(shortcut_action_ids) < shortcut_slot_count:
            shortcut_action_ids.append("")
        if len(shortcut_action_ids) > shortcut_slot_count:
            del shortcut_action_ids[shortcut_slot_count:]
        ui_state["last_shortcuts_path"] = str(source_path)
        save_shortcuts()
        refresh_shortcuts()
        append_log(f"Loaded shortcuts layout: {source_path}")
        set_status("Status: shortcuts loaded", ok=True)

    def refresh_action_button_row(buttons: list[QPushButton], action_ids: list[str], *, empty_tooltip: str) -> None:
        for index, button in enumerate(buttons):
            action_id = str(action_ids[index]).strip() if index < len(action_ids) else ""
            record = action_record_by_id(action_id)
            if not isinstance(record, dict):
                button.setText(f"Empty {index + 1}")
                button.setToolTip(empty_tooltip)
                button.setEnabled(False)
                button.setProperty("action_id", "")
                button.setStyleSheet(
                    "QPushButton { background: #1f2937; color: #94a3b8; border: 1px solid #334155; border-radius: 7px; padding: 6px 10px; }"
                )
                continue
            button.setText(quick_action_button_label(record))
            button.setToolTip(
                f"{record.get('title', 'Action')}\n\n{str(record.get('description', '')).strip() or 'Run this action directly.'}"
            )
            button.setEnabled(True)
            button.setProperty("action_id", str(record.get("id", "")).strip())
            button.setStyleSheet(quick_action_button_style(record))

    def rebuild_shortcuts_grid() -> None:
        while shortcuts_grid.count():
            item = shortcuts_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        shortcut_buttons.clear()
        for slot_index in range(max(6, int(shortcut_slot_count))):
            btn = QPushButton(f"Empty {slot_index + 1}")
            btn.setEnabled(False)
            btn.setMinimumHeight(34)
            btn.setToolTip("Add an action from the Actions panel.")
            btn.clicked.connect(lambda _checked=False, idx=slot_index: run_shortcut_button(idx))
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda pos, idx=slot_index, button=btn: show_shortcut_menu(idx, button.mapToGlobal(pos))
            )
            shortcuts_grid.addWidget(btn, slot_index // 3, slot_index % 3)
            shortcut_buttons.append(btn)

    def refresh_shortcuts() -> None:
        while len(shortcut_action_ids) < max(6, int(shortcut_slot_count)):
            shortcut_action_ids.append("")
        if len(shortcut_action_ids) > max(6, int(shortcut_slot_count)):
            del shortcut_action_ids[max(6, int(shortcut_slot_count)) :]
        rebuild_shortcuts_grid()
        refresh_action_button_row(
            shortcut_buttons,
            shortcut_action_ids,
            empty_tooltip="Add an action from the Actions panel.",
        )

    refresh_workspace_path_label()
    refresh_shortcuts()

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

    def refresh_feedback_controls() -> None:
        has_outcome = bool(last_turn_metrics.get("turn_id"))
        feedback_given = bool(last_turn_metrics.get("feedback"))
        feedback_btn.setEnabled(has_outcome)
        enabled = has_outcome and not feedback_given
        feedback_helpful_action.setEnabled(enabled)
        feedback_wrong_route_action.setEnabled(enabled)
        feedback_wrong_answer_action.setEnabled(enabled)
        feedback_didnt_work_action.setEnabled(enabled)

    def refresh_stop_button() -> None:
        stop_btn.setEnabled(
            any(callable(getattr(worker, "_assistant_cancel", None)) for worker in active_workers)
        )

    def stop_active_request(*_args) -> None:
        cancellable_workers = [
            worker
            for worker in list(active_workers)
            if callable(getattr(worker, "_assistant_cancel", None))
        ]
        if not cancellable_workers:
            set_status("Status: no active model request to stop", ok=False)
            append_log("Stop skipped: no active model request.")
            return
        for worker in cancellable_workers:
            cancel = getattr(worker, "_assistant_cancel", None)
            try:
                cancel()
            except Exception:
                pass
        latency_ms = int(last_turn_metrics.get("latency_ms", 0) or 0)
        if not latency_ms and wait_indicator.get("active"):
            latency_ms = int((time.perf_counter() - float(wait_indicator.get("started_at", 0.0))) * 1000)
        cancel_bucket = "quick" if latency_ms < 10000 else "long"
        record_telemetry(
            "turn_cancelled",
            {
                "turn_id": last_turn_metrics.get("turn_id", ""),
                "model": last_turn_metrics.get("model", ""),
                "prompt_hash": last_turn_metrics.get("prompt_hash", ""),
                "intent_category": last_turn_metrics.get("intent_category", ""),
                "predicted_route": last_turn_metrics.get("predicted_route", ""),
                "actual_route": "cancelled",
                "outcome_type": "cancelled",
                "latency_ms": latency_ms,
                "source": "user",
                "phase": "model_generation",
                "cancel_bucket": cancel_bucket,
            },
        )
        last_turn_metrics.update(
            {
                "actual_route": "cancelled",
                "outcome_type": "cancelled",
                "success": False,
                "latency_ms": latency_ms,
            }
        )
        stop_wait_indicator()
        refresh_stop_button()
        append_log("Stopped active model request.")
        set_status("Status: stopped active model request", ok=None)

    def set_latest_outcome(
        *,
        turn_id: str,
        model: str = "",
        action: str = "",
        prompt_hash_value: str = "",
        intent_category: str = "",
        predicted_route: str = "",
        actual_route: str = "",
        outcome_type: str = "",
        tool_name: str = "",
        success: bool | None = None,
        latency_ms: int = 0,
    ) -> None:
        last_turn_metrics.update(
            {
                "turn_id": turn_id,
                "model": model,
                "action": action,
                "prompt_hash": prompt_hash_value,
                "intent_category": intent_category,
                "predicted_route": predicted_route,
                "actual_route": actual_route,
                "outcome_type": outcome_type or action,
                "tool_name": tool_name,
                "success": success,
                "latency_ms": int(latency_ms or 0),
                "feedback": "",
            }
        )
        refresh_feedback_controls()

    def serialize_telemetry_events(events: list[dict], *, empty_message: str, newest_first: bool = True) -> str:
        if not events:
            return empty_message
        ordered_events = list(reversed(events)) if newest_first else list(events)
        return "\n".join(json.dumps(event, ensure_ascii=True) for event in ordered_events)

    def filter_telemetry_events(events: list[dict], kind: str) -> list[dict]:
        if kind == "model":
            return [
                event for event in events
                if str(event.get("event_type", "")).strip() in {"turn_started", "turn_completed", "code_execution"}
            ]
        if kind == "intent":
            return [
                event for event in events
                if str(event.get("event_type", "")).strip() == "intent_captured"
            ]
        if kind == "feedback":
            return [
                event for event in events
                if str(event.get("event_type", "")).strip() == "turn_feedback"
            ]
        if kind == "errors":
            filtered: list[dict] = []
            for event in events:
                event_type = str(event.get("event_type", "")).strip()
                if event_type == "turn_completed" and (
                    str(event.get("response_action", "")).strip() == "error" or event.get("tool_success") is False
                ):
                    filtered.append(event)
                elif event_type == "code_execution" and event.get("success") is False:
                    filtered.append(event)
            return filtered
        return list(events)

    def whats_new_message(version: str) -> str:
        current = str(version or "").strip()
        if current == "2.3.1":
            return (
                f"**What's New In {current}**\n"
                "- Added local-model performance telemetry for Ollama, including prompt-eval tokens, generation tokens, durations, and tokens-per-second metrics.\n"
                "- Telemetry summaries now include tokenization diagnostics so you can see input size, system-prompt share, and local-model bottlenecks.\n"
                "- Model requests now send a compact viewer context instead of verbose per-layer profiler evidence, reducing the dynamic payload sent to local models.\n"
                "- Telemetry Log views now show newest records first.\n"
                "- `Copy Report` copies a full shareable Markdown telemetry report to the clipboard.\n"
                "- Updates: https://github.com/wulinteousa2-hash/napari-chat-assistant/blob/main/CHANGELOG.md"
            )
        if current == "2.3.0":
            return (
                f"**What's New In {current}**\n"
                "- Added optional `Voice Input` under `Advanced` for local microphone recording and speech-to-text prompt entry.\n"
                "- Voice input uses local `faster-whisper` transcription when it is installed in the same napari Python environment.\n"
                "- The voice window now supports microphone-device selection, a live input meter, transcript editing, and direct `Run`.\n"
                "- The selected microphone is remembered, and the voice window stays open without blocking the viewer.\n"
                "- This update follows voice-input feedback from Ron DeSpain on image.sc.\n"
                "- Updates: https://github.com/wulinteousa2-hash/napari-chat-assistant/blob/main/CHANGELOG.md"
            )
        if current == "2.2.0":
            return (
                f"**What's New In {current}**\n"
                "- Added `Rate Result` so you can quickly mark the latest result as `Helpful`, `Wrong Route`, `Wrong Answer`, or `Didn't Work`.\n"
                "- Added a `⏹ Stop` button to cancel long model requests directly from the chat panel.\n"
                "- Improved telemetry with clearer views for `Model Activity`, `Intent Signals`, `Problems`, and `Raw Log`.\n"
                "- The assistant can now suggest prompts that are more likely to trigger supported local workflows such as masking, ROI measurement, and viewer setup.\n"
                "- Runtime messages are clearer when generated code depends on packages that are not included by default.\n"
                "- Updates: https://github.com/wulinteousa2-hash/napari-chat-assistant/blob/main/CHANGELOG.md"
            )
        if current == "2.1.2":
            return (
                f"**What's New In {current}**\n"
                "- Conservative binary-mask workflow prompts now run through a local planner and executor instead of relying on one-off model tool choices.\n"
                "- Workflow replies are compact by default, with `show plan`, `show details`, and `show debug` available when you want the full trace.\n"
                "- A new `Widgets` action, `Relabel Mask Value`, opens a small dock for repeated labels-value changes without using chat.\n"
                "- Routing for plain-English workflow prompts is more reliable because follow-up carry-over and code-repair detection are now stricter.\n"
                "- Updates: https://github.com/wulinteousa2-hash/napari-chat-assistant/blob/main/CHANGELOG.md"
            )
        if current == "2.1.0":
            return (
                f"**What's New In {current}**\n"
                "- `Quick Controls` are now available from both `Templates` and `Actions`, so common viewer setup can be triggered from prompts or one-click actions.\n"
                "- Try: `Hide all layers except the selected layer.`\n"
                "- Safe multi-step viewer workflows can now run several viewer-control steps in order from one numbered prompt.\n"
                "- Try:\n"
                "```text\n"
                "1. Fit visible layers to view.\n"
                "2. Show viewer axes.\n"
                "3. Show scale bar.\n"
                "4. Show selected layer bounding box.\n"
                "5. Show selected layer name overlay.\n"
                "```\n"
                "- Say `undo last workflow` to restore the viewer-control state from before the previous quick-control workflow.\n"
                "- Atlas Stitch source/export options and local code refinement repair were also improved.\n"
                "- Updates: https://github.com/wulinteousa2-hash/napari-chat-assistant/blob/main/CHANGELOG.md"
            )
        if current == "2.0.3":
            return (
                f"**What's New In {current}**\n"
                "- This plugin version was updated.\n"
                "- Follow-up conversation handling is better for short prompts like `same as before`, `but for labels`, and `just explain`.\n"
                "- `Refine My Code` is stricter about placeholder layer mismatches and unsafe viewer mutations.\n"
                "- Added an optics resolution teaching demo and renamed `Mask Cleanup 2D/3D`.\n"
                "- See the changelog for full details.\n"
                "- Updates: https://github.com/wulinteousa2-hash/napari-chat-assistant/blob/main/CHANGELOG.md"
            )
        if current == "2.0.2":
            return (
                f"**What's New In {current}**\n"
                "- This plugin version was updated.\n"
                "- Fixed the `Templates` tab getting stuck after opening `Actions`.\n"
                "- Removed the old `Help` -> `Prompt Tips` entry.\n"
                "- `Shortcuts` can now collapse or expand.\n"
                "- See the changelog for full details.\n"
                "- Updates: https://github.com/wulinteousa2-hash/napari-chat-assistant/blob/main/CHANGELOG.md"
            )
        return (
            f"**What's New In {current}**\n"
            "- This plugin version was updated.\n"
            "- See the changelog for full details.\n"
            "- Updates: https://github.com/wulinteousa2-hash/napari-chat-assistant/blob/main/CHANGELOG.md"
        )

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

    def context_layer_by_name(layer_name: str):
        if viewer is None:
            return None
        try:
            return viewer.layers[str(layer_name)]
        except Exception:
            return None

    def selected_context_layer_names() -> list[str]:
        if viewer is None:
            return []
        try:
            selected_layers = list(viewer.layers.selection)
        except Exception:
            selected_layers = list(getattr(viewer.layers.selection, "_selected", []) or [])
        active = viewer.layers.selection.active
        if active is not None and active not in selected_layers:
            selected_layers.append(active)
        selected_names = {str(layer.name) for layer in selected_layers if layer is not None}
        return [str(layer.name) for layer in viewer.layers if str(layer.name) in selected_names]

    def context_layer_text(layer) -> str:
        profile = profile_layer(layer)
        shape_text = tuple(profile["shape"]) if profile.get("shape") is not None else "n/a"
        dtype_text = profile.get("dtype") or "n/a"
        selected = viewer is not None and viewer.layers.selection.active is layer
        return "\n".join(
            [
                f"Layer [{layer.name}]",
                f"- type: {layer.__class__.__name__}",
                f"- selected: {bool(selected)}",
                f"- visible: {bool(getattr(layer, 'visible', False))}",
                f"- shape: {shape_text}",
                f"- dtype: {dtype_text}",
                f"- semantic_type: {profile.get('semantic_type', 'unknown')}",
            ]
        )

    def copy_context_layer_summary(layer_name: str):
        layer = context_layer_by_name(layer_name)
        if layer is None:
            return
        QApplication.clipboard().setText(context_layer_text(layer))
        append_log(f"Copied layer summary: {layer_name}")
        set_status("Status: layer summary copied", ok=True)

    def select_context_layer_name(layer_name: str):
        layer = context_layer_by_name(layer_name)
        if layer is None or viewer is None:
            return
        viewer.layers.selection.active = layer
        append_log(f"Selected layer from context: {layer_name}")
        set_status("Status: layer selected", ok=True)
        if context_selected_only_checkbox.isChecked():
            apply_context_selected_only_visibility(capture=False, refresh=False)
        refresh_context()

    def apply_context_selected_only_visibility(*, capture: bool, refresh: bool = True):
        nonlocal context_visibility_snapshot
        if viewer is None:
            return
        selected_names = set(selected_context_layer_names())
        if not selected_names:
            set_status("Status: no layer selected to isolate", ok=False)
            append_log("Show selected only skipped: no selected layer.")
            return
        if capture or context_visibility_snapshot is None:
            context_visibility_snapshot = {str(layer.name): bool(getattr(layer, "visible", False)) for layer in viewer.layers}
        for layer in viewer.layers:
            layer.visible = str(layer.name) in selected_names
        append_log(f"Showing selected layer(s) only: {', '.join(selected_names)}")
        set_status("Status: showing selected layer(s) only", ok=True)
        if refresh:
            refresh_context()

    def restore_context_visibility_snapshot():
        nonlocal context_visibility_snapshot
        if viewer is None or context_visibility_snapshot is None:
            return
        for layer in viewer.layers:
            name = str(layer.name)
            if name in context_visibility_snapshot:
                layer.visible = bool(context_visibility_snapshot[name])
        context_visibility_snapshot = None
        append_log("Restored layer visibility from Layer Context.")
        set_status("Status: layer visibility restored", ok=True)
        refresh_context()

    def toggle_context_selected_only(enabled: bool):
        if enabled:
            apply_context_selected_only_visibility(capture=True)
        else:
            restore_context_visibility_snapshot()

    def select_context_layer_item(item: QListWidgetItem):
        data = item.data(Qt.UserRole) or {}
        if isinstance(data, dict):
            select_context_layer_name(str(data.get("layer_name", "")))

    def maybe_note_selected_only_visibility(tool_name: str, result_message: str) -> str:
        tracked_tools = {
            "show_all_layers",
            "show_only_layers",
            "show_only_layer_type",
            "show_all_except_layers",
            "show_layers_by_type",
            "hide_layers_by_type",
        }
        if not widget_is_alive(context_selected_only_checkbox) or not context_selected_only_checkbox.isChecked():
            return result_message
        if str(tool_name or "").strip() not in tracked_tools:
            return result_message
        note = (
            "Note: `Show selected layer(s) only` is still on in `Layer Context` -> `Layers`, "
            "so layer visibility remains constrained there. Turn it off first if you want prompt or action visibility changes to affect all layers."
        )
        return f"{result_message}\n\n{note}" if str(result_message or "").strip() else note

    def refresh_context(*_args):
        if widget_is_alive(context_selected_only_checkbox) and context_selected_only_checkbox.isChecked():
            apply_context_selected_only_visibility(capture=False, refresh=False)
        summary_text = layer_summary(viewer)
        if widget_is_alive(context_summary_box):
            context_summary_box.setPlainText(summary_text)
        if widget_is_alive(context_layers_list):
            context_layers_list.clear()
            if viewer is not None:
                selected = viewer.layers.selection.active if viewer is not None else None
                for layer in viewer.layers:
                    insert_text = str(layer.name)
                    line = insert_text
                    if selected is layer:
                        line = f"{line} [selected]"
                    item = QListWidgetItem()
                    item.setData(Qt.UserRole, {"layer_name": insert_text})
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
                    inline_btn = QPushButton("Inline")
                    inline_btn.setMaximumWidth(56)
                    info_btn = QPushButton("Info")
                    info_btn.setMaximumWidth(56)
                    insert_btn.setToolTip("Insert this exact layer name into the Prompt box.")
                    inline_btn.setToolTip("Insert this exact layer name at the current cursor position.")
                    copy_btn.setToolTip("Copy this exact layer name to the clipboard.")
                    info_btn.setToolTip("Copy a fuller summary for this layer.")
                    copy_btn.clicked.connect(lambda _checked=False, text=insert_text: QApplication.clipboard().setText(text))
                    insert_btn.clicked.connect(lambda _checked=False, text=insert_text: append_text_to_prompt(text))
                    inline_btn.clicked.connect(lambda _checked=False, text=insert_text: insert_text_at_prompt_cursor(text))
                    info_btn.clicked.connect(lambda _checked=False, text=insert_text: copy_context_layer_summary(text))
                    row_layout.addWidget(row_label, 1)
                    row_layout.addWidget(inline_btn, 0)
                    row_layout.addWidget(insert_btn, 0)
                    row_layout.addWidget(copy_btn, 0)
                    row_layout.addWidget(info_btn, 0)
                    item.setSizeHint(row_widget.sizeHint())
                    context_layers_list.setItemWidget(item, row_widget)
                    if selected is layer:
                        context_layers_list.setCurrentItem(item)
        refresh_analysis_controls()

    def append_text_to_prompt(text: str):
        content = str(text or "").strip()
        if not content:
            return
        current = prompt.toPlainText().rstrip()
        prompt.setPlainText(f"{current}\n{content}" if current else content)
        prompt.setFocus()

    def insert_text_at_prompt_cursor(text: str):
        content = str(text or "")
        if not content:
            return
        cursor = prompt.textCursor()
        cursor.insertText(content)
        prompt.setTextCursor(cursor)
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

    def run_prepared_tool_request(prepared: dict, *, tool_name: str, tool_message: str = "", turn_id_value: str = ""):
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
            remember_recent_action(
                tool_name=tool_name,
                turn_id_value=turn_id_value or last_turn_metrics.get("turn_id", ""),
                message=f"{tool_message}\n{result_message}" if tool_message else result_message,
                tool_result=tool_result if "job" in prepared else None,
            )
            remember_assistant_outcome(
                tool_message or result_message,
                target_type="tool_result",
                target_profile=selected_layer_profile(),
                state="approved",
            )
            set_latest_outcome(
                turn_id=turn_id_value or last_turn_metrics.get("turn_id", ""),
                model="local_tool_runner",
                action="tool",
                prompt_hash_value=last_turn_metrics.get("prompt_hash", ""),
                intent_category=last_turn_metrics.get("intent_category", ""),
                predicted_route="local_tool",
                actual_route="tool",
                outcome_type="tool_result",
                tool_name=tool_name,
                success=True,
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
        remember_recent_action(
            tool_name=tool_name,
            turn_id_value=turn_id_value or last_turn_metrics.get("turn_id", ""),
            message=f"{tool_message}\n{result_message}" if tool_message else result_message,
            tool_result=result,
        )
        remember_assistant_outcome(
            tool_message or result_message,
            target_type="tool_result",
            target_profile=selected_layer_profile(),
            state="approved",
        )
        set_latest_outcome(
            turn_id=turn_id_value or last_turn_metrics.get("turn_id", ""),
            model="local_tool_runner",
            action="tool",
            prompt_hash_value=last_turn_metrics.get("prompt_hash", ""),
            intent_category=last_turn_metrics.get("intent_category", ""),
            predicted_route="local_tool",
            actual_route="tool",
            outcome_type="tool_result",
            tool_name=tool_name,
            success=True,
        )

    def adjust_prompt_library_font(delta: int):
        font = prompt_library_list.font()
        current_size = font.pointSize()
        if current_size <= 0:
            current_size = 10
        font.setPointSize(max(9, min(16, current_size + int(delta))))
        prompt_library_list.setFont(font)
        code_library_list.setFont(font)

    def apply_chat_font_size(size: int):
        font = transcript.font()
        font.setPointSize(max(9, min(20, int(size))))
        transcript.setFont(font)
        ui_state["chat_font_size"] = font.pointSize()
        save_ui_state(ui_state)

    def adjust_chat_font(delta: int):
        font = transcript.font()
        current_size = font.pointSize()
        if current_size <= 0:
            current_size = 10
        apply_chat_font_size(current_size + int(delta))

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

    def telemetry_report_text() -> str:
        events, invalid_lines = load_telemetry_events()
        summary = summarize_telemetry_events(events, invalid_lines)
        return format_markdown_telemetry_report(summary, path=TELEMETRY_LOG_PATH)

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

    def show_telemetry_report(*_args):
        report_text = telemetry_report_text()
        QApplication.clipboard().setText(report_text)
        append_log("Copied telemetry report to clipboard.")
        set_status("Status: telemetry report copied", ok=True)

    def show_telemetry_viewer(*_args):
        dialog = QDialog(root)
        dialog.setWindowTitle("Telemetry Log")
        dialog.resize(900, 700)
        dialog_layout = QVBoxLayout(dialog)

        path_label = QLabel(f"Telemetry log: {TELEMETRY_LOG_PATH}")
        path_label.setTextInteractionFlags(path_label.textInteractionFlags())
        path_label.setWordWrap(True)
        dialog_layout.addWidget(path_label)

        viewer_tabs = QTabWidget()
        summary_box = QTextEdit()
        summary_box.setReadOnly(True)
        summary_box.setMinimumHeight(200)
        summary_box.setStyleSheet(
            "QTextEdit { background: #10182b; color: #d6deeb; border: 1px solid #30415f; padding: 8px; }"
        )
        model_box = QTextEdit()
        intent_box = QTextEdit()
        problems_box = QTextEdit()
        raw_box = QTextEdit()
        for box in (model_box, intent_box, problems_box, raw_box):
            box.setReadOnly(True)
            box.setStyleSheet(
                "QTextEdit { background: #0b1021; color: #d6deeb; border: 1px solid #22304a; padding: 8px; }"
            )
        viewer_tabs.addTab(summary_box, "Summary")
        viewer_tabs.addTab(model_box, "Model Activity")
        viewer_tabs.addTab(intent_box, "Intent Signals")
        viewer_tabs.addTab(problems_box, "Problems")
        viewer_tabs.addTab(raw_box, "Raw Log")
        dialog_layout.addWidget(viewer_tabs, 1)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        refresh_dialog_btn = QPushButton("Refresh")
        close_dialog_btn = QPushButton("Close")
        button_layout.addWidget(refresh_dialog_btn)
        button_layout.addWidget(close_dialog_btn)
        dialog_layout.addWidget(button_row)

        def refresh_dialog():
            events, invalid_lines = load_telemetry_events()
            summary_box.setMarkdown(format_telemetry_summary(summarize_telemetry_events(events, invalid_lines)))
            model_box.setPlainText(
                serialize_telemetry_events(
                    filter_telemetry_events(events, "model"),
                    empty_message="[No model telemetry events recorded]",
                )
            )
            intent_box.setPlainText(
                "\n\n".join(
                    part
                    for part in (
                        serialize_telemetry_events(
                            filter_telemetry_events(events, "intent"),
                            empty_message="[No intent telemetry events recorded]",
                        ),
                        serialize_telemetry_events(
                            filter_telemetry_events(events, "feedback"),
                            empty_message="[No feedback events recorded]",
                        ),
                    )
                    if part
                )
            )
            problems_box.setPlainText(
                serialize_telemetry_events(
                    filter_telemetry_events(events, "errors"),
                    empty_message="[No error events recorded]",
                )
            )
            raw_box.setPlainText(read_telemetry_tail(max_lines=250, newest_first=True) or "[Telemetry log is empty]")

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

    def _open_optional_widget(
        *,
        module_name: str,
        function_name: str,
        feature_name: str,
    ) -> bool:
        try:
            module = importlib.import_module(module_name)
            opener = getattr(module, function_name)
        except Exception as exc:
            logger.exception("%s failed to import.", feature_name)
            append_chat_message(
                "assistant",
                f"Could not open {feature_name}.\n{exc}",
                render_markdown=False,
            )
            append_log(f"{feature_name} import failed: {exc}")
            set_status(f"Status: {feature_name} unavailable", ok=False)
            return False
        opener(viewer)
        return True

    def show_intensity_metrics_widget(*_args):
        return _open_optional_widget(
            module_name="napari_chat_assistant.widgets.intensity_metrics_widget",
            function_name="open_intensity_metrics_widget",
            feature_name="ROI Intensity Analysis",
        )

    def show_line_profile_widget(*_args):
        return _open_optional_widget(
            module_name="napari_chat_assistant.widgets.line_profile_widget",
            function_name="open_line_profile_gaussian_fit_widget",
            feature_name="Line Profile Analysis",
        )

    def show_relabel_mask_widget(*_args):
        return _open_optional_widget(
            module_name="napari_chat_assistant.widgets.relabel_mask_widget",
            function_name="open_relabel_mask_widget",
            feature_name="Relabel Mask Values",
        )

    def show_atlas_stitch(*_args):
        _open_optional_widget(
            module_name="napari_chat_assistant.atlas_stitch.widget",
            function_name="open_atlas_stitch_widget",
            feature_name="Atlas Stitch",
        )

    def show_voice_input(*_args):
        def submit_voice_prompt(text: str) -> None:
            content = str(text or "").strip()
            if not content:
                return
            prompt.setPlainText(content)
            append_log("Sending reviewed voice transcript.")
            set_status("Status: sending voice transcript", ok=None)
            send_message()

        try:
            module = importlib.import_module("napari_chat_assistant.voice.controller")
            opener = getattr(module, "open_voice_input_dialog")
        except Exception as exc:
            logger.exception("Voice Input failed to import.")
            append_chat_message(
                "assistant",
                f"Could not open Voice Input.\n{exc}",
                render_markdown=False,
            )
            append_log(f"Voice Input import failed: {exc}")
            set_status("Status: Voice Input unavailable", ok=False)
            return False
        opener(parent=root, submit_callback=submit_voice_prompt)
        return True

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

    def record_intent_telemetry(event: IntentEvent) -> None:
        if not bool(ui_state.get("telemetry_enabled", False)):
            return
        try:
            record_intent(event)
        except Exception:
            logger.exception("Failed to append intent telemetry event.")

    def current_workspace_state() -> str:
        try:
            return "loaded" if len(viewer.layers) else "new"
        except Exception:
            return "loaded"

    def refresh_telemetry_controls():
        telemetry_summary_btn.setVisible(True)
        telemetry_report_btn.setVisible(True)
        telemetry_view_btn.setVisible(True)
        telemetry_reset_btn.setVisible(True)

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

    def show_whats_new(*_args):
        append_chat_message("assistant", whats_new_message(__version__))
        append_log(f"Opened what's new for version {__version__}.")
        set_status("Status: what's new shown", ok=None)

    def show_about_assistant(*_args):
        append_chat_message(
            "assistant",
            f"**napari-chat-assistant {__version__}**\n"
            "- Local AI and deterministic imaging workbench for napari.\n"
            "- Combines Prompt, Code, Templates, Actions, and user-defined Shortcuts.\n"
            "- Designed to reduce click count and keep analysis close to the viewer.\n"
            "- MIT License\n"
            "- Copyright (c) 2026 Wulin Teo",
        )
        append_log(f"Opened about panel for version {__version__}.")
        set_status("Status: about info shown", ok=None)

    def show_report_bug(*_args):
        append_chat_message(
            "assistant",
            "**Report Bug**\n"
            "- GitHub Issues: https://github.com/wulinteousa2-hash/napari-chat-assistant/issues\n"
            "- Email: wulinteo.usa2@gmail.com\n"
            "- Include the plugin version, what you clicked or asked, and any error text you saw.\n"
            f"- Current plugin version: `{__version__}`",
        )
        append_log("Opened bug-report help.")
        set_status("Status: bug-report help shown", ok=None)

    def persist_session_memory():
        save_session_memory(session_memory_state)

    def selected_layer_profile() -> dict | None:
        payload = layer_context_json(viewer)
        profile = payload.get("selected_layer_profile")
        return profile if isinstance(profile, dict) else None

    def set_pending_action(payload: dict | None) -> None:
        nonlocal pending_action_state
        pending_action_state = normalize_pending_action(payload)

    def set_last_memory_candidates(item_ids: list[str]):
        nonlocal last_memory_candidate_ids
        last_memory_candidate_ids = [item_id for item_id in item_ids if item_id]

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

    def classify_recent_action_kind(tool_name: str) -> str:
        token = str(tool_name or "").strip().lower()
        if "threshold" in token:
            return "threshold"
        if "gaussian" in token or "clahe" in token:
            return "enhancement"
        if "roi" in token or "measure" in token or "histogram" in token:
            return "measurement"
        if "widget" in token:
            return "widget"
        if "synthetic" in token or "demo" in token:
            return "demo"
        return token or "tool"

    def build_recent_action_entry(
        *,
        tool_name: str,
        turn_id_value: str,
        message: str,
        tool_result: dict | None = None,
    ) -> dict:
        entry = {
            "tool_name": str(tool_name or "").strip(),
            "action_kind": classify_recent_action_kind(tool_name),
            "turn_id": str(turn_id_value or "").strip(),
            "message": " ".join(str(message or "").split()).strip(),
            "input_layers": [],
            "output_layers": [],
            "parameters": {},
            "result_summary": {},
            "explanation_hints": {},
        }
        payload = dict(tool_result or {})
        if not payload:
            return entry
        for key in ("layer_name", "image_layer_name", "layer_name_a", "layer_name_b", "mask_layer_name", "roi_layer_name"):
            value = str(payload.get(key, "")).strip()
            if value and value not in entry["input_layers"]:
                entry["input_layers"].append(value)
        for key in ("output_name", "layer_name"):
            value = str(payload.get(key, "")).strip()
            if key == "layer_name":
                continue
            if value and value not in entry["output_layers"]:
                entry["output_layers"].append(value)
        if classify_recent_action_kind(tool_name) == "threshold":
            mode = str(payload.get("polarity", "auto")).strip().lower() or "auto"
            threshold_value = payload.get("threshold_value")
            threshold_text = ""
            try:
                threshold_text = f"{float(threshold_value):.6g}"
            except Exception:
                threshold_text = ""
            data = np.asarray(payload.get("data")) if payload.get("data") is not None else np.asarray([], dtype=np.float32)
            finite = data[np.isfinite(data)] if data.size else np.asarray([], dtype=np.float32)
            if mode == "auto":
                try:
                    resolved_mode = "bright" if finite.size and float(threshold_value) >= float(np.mean(finite)) else "dim"
                except Exception:
                    resolved_mode = "bright"
            else:
                resolved_mode = mode
            mode_label = (
                "bright regions"
                if resolved_mode == "bright"
                else "dim regions"
                if resolved_mode == "dim"
                else "automatic foreground selection"
            )
            mode_explanation = (
                "keep pixels brighter than the threshold as foreground"
                if resolved_mode == "bright"
                else "keep pixels dimmer than the threshold as foreground"
                if resolved_mode == "dim"
                else "choose whether brighter or dimmer pixels should be foreground automatically"
            )
            entry["parameters"] = {
                "threshold_value": threshold_value,
                "foreground_mode": mode,
                "foreground_mode_resolved": resolved_mode,
                "image_min": float(np.min(finite)) if finite.size else None,
                "image_max": float(np.max(finite)) if finite.size else None,
            }
            entry["result_summary"] = dict(payload.get("stats", {}) or {})
            entry["explanation_hints"] = {
                "foreground_label": mode_label,
                "foreground_explanation": mode_explanation,
                "threshold_text": threshold_text,
            }
        elif str(tool_name).strip() == "plot_histogram":
            histogram = dict(payload.get("histogram", {}) or {})
            stats = dict(histogram.get("stats", {}) or {})
            bins = histogram.get("bins")
            layer_name = str(payload.get("layer_name", "")).strip()
            if layer_name:
                entry["input_layers"] = [layer_name]
            entry["parameters"] = {"bins": bins}
            entry["result_summary"] = stats
            entry["explanation_hints"] = {
                "kind_label": "global intensity histogram",
                "meaning": "shows how many pixels fall into each intensity range across the whole image",
            }
        elif str(tool_name).strip() == "summarize_intensity":
            compact = " ".join(str(message or "").split())
            layer_match = re.search(r"Layer:\s*\[([^\]]+)\]", compact)
            if layer_match:
                entry["input_layers"] = [layer_match.group(1)]
            stats = {}
            for label, key in (
                ("Pixels", "count"),
                ("Mean", "mean"),
                ("Std Dev", "std"),
                ("Median", "median"),
                ("Min", "min"),
                ("Max", "max"),
            ):
                match = re.search(rf"{re.escape(label)}:\s*([0-9.eE+-]+)", compact)
                if match:
                    try:
                        stats[key] = float(match.group(1))
                    except Exception:
                        pass
            entry["result_summary"] = stats
            entry["explanation_hints"] = {
                "kind_label": "whole-image intensity summary",
                "meaning": "reports simple image-wide statistics in chat rather than opening an interactive widget",
            }
        return entry

    def remember_recent_action(*, tool_name: str, turn_id_value: str, message: str, tool_result: dict | None = None) -> None:
        nonlocal recent_action_state
        recent_action_state = record_recent_action(
            recent_action_state,
            build_recent_action_entry(
                tool_name=tool_name,
                turn_id_value=turn_id_value,
                message=message,
                tool_result=tool_result,
            ),
        )

    def looks_like_threshold_followup(text: str) -> bool:
        source = " ".join(str(text or "").strip().lower().split())
        if not source:
            return False
        phrases = (
            "what is polarity",
            "what is bright",
            "what is dim",
            "what does bright mean",
            "what does dim mean",
            "what parameter",
            "how did you do that",
            "how did you do it",
            "how did you make that mask",
            "how do you adjust",
            "how did you adjust",
            "how did you choose",
            "how did you come out with the mask",
            "what threshold",
            "explain the threshold",
        )
        return any(phrase in source for phrase in phrases)

    def looks_like_histogram_followup(text: str) -> bool:
        source = " ".join(str(text or "").strip().lower().split())
        if not source:
            return False
        phrases = (
            "what does this histogram mean",
            "what does the histogram mean",
            "what is this histogram",
            "what am i looking at",
            "is this bimodal",
            "can i use this for thresholding",
            "what does bimodal mean",
        )
        return any(phrase in source for phrase in phrases)

    def looks_like_intensity_summary_followup(text: str) -> bool:
        source = " ".join(str(text or "").strip().lower().split())
        if not source:
            return False
        phrases = (
            "what does mean mean",
            "what does std mean",
            "what does median mean",
            "is this roi",
            "is this whole image",
            "what do these numbers mean",
            "what does max mean",
            "what does min mean",
        )
        return any(phrase in source for phrase in phrases)

    def reply_for_threshold_followup(text: str) -> str:
        action = latest_recent_action(recent_action_state, lambda item: item.get("action_kind") == "threshold")
        if not action:
            return ""
        source = " ".join(str(text or "").strip().lower().split())
        hints = dict(action.get("explanation_hints", {}) or {})
        threshold_value = str(hints.get("threshold_text", "")).strip()
        mode_name = str(hints.get("foreground_label", "")).strip() or "foreground selection"
        mode_explanation = str(hints.get("foreground_explanation", "")).strip() or "select foreground pixels with thresholding"
        inputs = list(action.get("input_layers", []) or [])
        outputs = list(action.get("output_layers", []) or [])
        image_name = inputs[0] if inputs else ""
        output_name = outputs[0] if outputs else ""

        if any(phrase in source for phrase in ("what is polarity", "what is bright", "what is dim", "what does bright mean", "what does dim mean")):
            detail = f'In this threshold tool, "{mode_name}" means {mode_explanation}.'
            if threshold_value:
                detail += f" The threshold value was {threshold_value}."
            if image_name and output_name:
                detail += f" That is how [{image_name}] became the mask layer [{output_name}]."
            return detail

        detail = "I made the mask with thresholding."
        if image_name and output_name:
            detail += f" I took [{image_name}] and created [{output_name}] from pixels on one side of the threshold."
        detail += f" For this result I used {mode_name}, which means {mode_explanation}."
        if threshold_value:
            detail += f" The cutoff was {threshold_value}, so pixels beyond that cutoff were included in the mask."
        return detail

    def reply_for_histogram_followup(text: str) -> str:
        action = latest_recent_action(recent_action_state, lambda item: item.get("tool_name") == "plot_histogram")
        if not action:
            return ""
        source = " ".join(str(text or "").strip().lower().split())
        hints = dict(action.get("explanation_hints", {}) or {})
        stats = dict(action.get("result_summary", {}) or {})
        layer_name = (action.get("input_layers") or [""])[0]
        base = (
            f"The histogram is a {str(hints.get('kind_label', 'intensity histogram')).strip()} for "
            f"[{layer_name}]. It {str(hints.get('meaning', 'shows intensity distribution across the image')).strip()}."
        )
        if "bimodal" in source:
            return base + " A bimodal histogram means there are two main intensity populations, which can make thresholding easier if one peak is background and the other is foreground."
        if "threshold" in source:
            mean_text = f" The image-wide mean intensity is {float(stats.get('mean', 0.0)):.6g}." if "mean" in stats else ""
            return base + " It is useful for thresholding because separated peaks or a long bright tail can suggest a foreground/background split." + mean_text
        return base + " The x-axis is intensity and the y-axis is the number of pixels in each intensity range."

    def reply_for_intensity_summary_followup(text: str) -> str:
        action = latest_recent_action(recent_action_state, lambda item: item.get("tool_name") == "summarize_intensity")
        if not action:
            return ""
        source = " ".join(str(text or "").strip().lower().split())
        stats = dict(action.get("result_summary", {}) or {})
        layer_name = (action.get("input_layers") or [""])[0]
        if any(phrase in source for phrase in ("is this roi", "is this whole image")):
            return f"This summary is for the whole image layer [{layer_name}], not an ROI. For region-based intensity measurement, use ROI Intensity Analysis or an ROI-specific measurement tool."
        if "std" in source:
            return f"For [{layer_name}], std means how spread out the pixel intensities are around the mean. A larger std usually means broader contrast or more variation in brightness across the image."
        if "median" in source:
            return f"For [{layer_name}], the median is the middle pixel intensity when all intensities are sorted. It is often less sensitive to a few very bright pixels than the mean."
        if "max" in source or "min" in source:
            return f"For [{layer_name}], min and max are the darkest and brightest pixel values in the whole image. They tell you the range of intensities currently present."
        mean_text = f"{float(stats.get('mean', 0.0)):.6g}" if "mean" in stats else "n/a"
        std_text = f"{float(stats.get('std', 0.0)):.6g}" if "std" in stats else "n/a"
        return f"This intensity summary is a quick whole-image readout for [{layer_name}]. Mean={mean_text} gives the average brightness, std={std_text} shows intensity spread, and median/min/max help describe the image without opening an interactive widget."

    def submit_feedback(feedback_label: str) -> None:
        nonlocal session_memory_state
        normalized_feedback = str(feedback_label or "").strip().lower()
        if not last_turn_metrics.get("turn_id"):
            set_status("Status: no recent assistant result to rate", ok=False)
            append_log("Feedback skipped: no recent assistant outcome.")
            return
        if normalized_feedback == "wrong_answer" and last_memory_candidate_ids:
            session_memory_state = reject_items(session_memory_state, last_memory_candidate_ids)
            persist_session_memory()
            set_last_memory_candidates([])
        record_telemetry(
            "turn_feedback",
            {
                "turn_id": last_turn_metrics.get("turn_id", ""),
                "model": last_turn_metrics.get("model", ""),
                "feedback": normalized_feedback,
                "response_action": last_turn_metrics.get("action", ""),
                "prompt_hash": last_turn_metrics.get("prompt_hash", ""),
                "intent_category": last_turn_metrics.get("intent_category", ""),
                "predicted_route": last_turn_metrics.get("predicted_route", ""),
                "actual_route": last_turn_metrics.get("actual_route", ""),
                "outcome_type": last_turn_metrics.get("outcome_type", ""),
                "tool_name": last_turn_metrics.get("tool_name", ""),
                "success": last_turn_metrics.get("success"),
                "latency_ms": last_turn_metrics.get("latency_ms", 0),
            },
        )
        last_turn_metrics["feedback"] = normalized_feedback
        refresh_feedback_controls()
        feedback_display = {
            "helpful": "Helpful",
            "wrong_route": "Wrong Route",
            "wrong_answer": "Wrong Answer",
            "didnt_work": "Didn't Work",
        }.get(normalized_feedback, normalized_feedback or "Feedback")
        append_log(f"Recorded feedback for latest assistant outcome: {feedback_display}")
        set_status(f"Status: feedback saved ({feedback_display})", ok=True)

    apply_chat_font_size(int(ui_state.get("chat_font_size", transcript.font().pointSize() or 10)))

    def current_library_kind() -> str:
        current_widget = library_stack.currentWidget()
        if current_widget is code_library_list:
            return "code"
        if current_widget is template_tab:
            return "template"
        if current_widget is action_tab:
            return "action"
        return "prompt"

    def current_library_list() -> QListWidget:
        return code_library_list if current_library_kind() == "code" else prompt_library_list

    def current_library_item_name() -> str:
        if current_library_kind() == "template":
            return "template"
        if current_library_kind() == "action":
            return "action"
        return "code snippet" if current_library_kind() == "code" else "prompt"

    def show_library_panel(kind: str) -> None:
        target = str(kind or "").strip().lower()
        if target == "code":
            library_stack.setCurrentWidget(code_library_list)
            library_tabs.setCurrentIndex(1)
            actions_tab_btn.setChecked(False)
        elif target == "template":
            library_stack.setCurrentWidget(template_tab)
            library_tabs.setCurrentIndex(2)
            actions_tab_btn.setChecked(False)
        elif target == "action":
            library_stack.setCurrentWidget(action_tab)
            actions_tab_btn.setChecked(True)
        else:
            library_stack.setCurrentWidget(prompt_library_list)
            library_tabs.setCurrentIndex(0)
            actions_tab_btn.setChecked(False)
        refresh_library_controls()

    def show_library_tab_index(index: int) -> None:
        if index == 1:
            show_library_panel("code")
        elif index == 2:
            show_library_panel("template")
        else:
            show_library_panel("prompt")

    def selected_library_records() -> list[dict]:
        if current_library_kind() == "template":
            item = template_tree.currentItem()
            if item is None:
                return []
            record = item.data(0, Qt.UserRole)
            return [record] if isinstance(record, dict) and record.get("code") else []
        if current_library_kind() == "action":
            item = action_tree.currentItem()
            if item is None:
                return []
            record = item.data(0, Qt.UserRole)
            return [record] if isinstance(record, dict) and isinstance(record.get("execution"), dict) else []
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
        if "no module named 'cv2'" in lowered or 'no module named "cv2"' in lowered:
            return (
                "OpenCV is not included by default here.\n"
                "Prefer scipy/skimage, or install `opencv-python-headless` in the napari environment if needed."
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

    def workspace_default_path() -> str:
        current = str(ui_state.get("last_workspace_path", "")).strip()
        if current:
            return current
        workspace_dir = Path.home() / ".napari-chat-assistant" / "workspaces"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return str(workspace_dir / "napari_workspace.json")

    def save_workspace_common(*, choose_path: bool) -> None:
        try:
            _workspace_module = _workspace_state_module()
        except Exception as exc:
            append_log(f"Save workspace unavailable: {exc}")
            set_status("Status: workspace save unavailable", ok=False)
            workspace_status.setText(str(exc))
            return
        destination = workspace_default_path()
        if choose_path or not destination.strip():
            selected, _ = QFileDialog.getSaveFileName(
                root,
                "Save Workspace Manifest",
                destination or "napari_workspace.json",
                "JSON Files (*.json)",
            )
            if not selected:
                return
            destination = selected
        try:
            result = _workspace_module.save_workspace_manifest(viewer, destination)
        except Exception as exc:
            append_log(f"Save workspace failed: {exc}")
            set_status("Status: save workspace failed", ok=False)
            workspace_status.setText(f"Save failed: {exc}")
            return
        ui_state["last_workspace_path"] = str(result["path"])
        save_ui_state(ui_state)
        refresh_workspace_path_label()
        skipped = result.get("skipped_layers", []) or []
        if skipped:
            workspace_status.setText(
                f"Saved {result['saved_layers']} layer(s). Skipped {len(skipped)} layer(s) without recoverable source or inline state."
            )
        else:
            workspace_status.setText(f"Saved {result['saved_layers']} layer(s) to {result['path']}.")
        append_log(
            f"Saved workspace manifest: {result['path']} | saved_layers={result['saved_layers']} skipped={len(skipped)}"
        )
        set_status("Status: workspace saved", ok=True)

    def load_workspace_common(*, choose_path: bool) -> None:
        if workspace_load_state["active"]:
            workspace_status.setText("A workspace load is already in progress.")
            set_status("Status: workspace load already running", ok=None)
            return
        try:
            _workspace_module = _workspace_state_module()
        except Exception as exc:
            append_log(f"Load workspace unavailable: {exc}")
            set_status("Status: workspace load unavailable", ok=False)
            workspace_status.setText(str(exc))
            return
        source_path = workspace_default_path()
        if choose_path or not source_path.strip():
            selected, _ = QFileDialog.getOpenFileName(
                root,
                "Load Workspace Manifest",
                source_path or "",
                "JSON Files (*.json)",
            )
            if not selected:
                return
            source_path = selected
        if not source_path.strip():
            workspace_status.setText("Choose a workspace file first.")
            set_status("Status: no workspace file selected", ok=False)
            return
        try:
            manifest_path, payload = _workspace_module.read_workspace_manifest(source_path)
        except Exception as exc:
            append_log(f"Load workspace failed: {exc}")
            set_status("Status: load workspace failed", ok=False)
            workspace_status.setText(f"Load failed: {exc}")
            return
        records = []
        for saved_index, original_record in enumerate(list(payload.get("layers", []) or [])):
            record = dict(original_record)
            record["__workspace_saved_index__"] = saved_index
            records.append(record)
        source_records = []
        fast_records = []
        for record in records:
            kind = _workspace_module.workspace_record_loading_kind(record)
            if kind in {"source", "recipe"}:
                source_records.append(record)
            else:
                fast_records.append(record)
        ordered_records = fast_records + source_records
        total_layers = len(ordered_records)
        dialog = QDialog(root, Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        dialog.setWindowTitle("Loading Workspace")
        dialog.setModal(False)
        dialog.resize(360, 108)
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(16, 14, 16, 14)
        dialog_layout.setSpacing(8)
        progress_label = QLabel(f"Preparing {total_layers} layer(s)...")
        progress_label.setWordWrap(True)
        progress_bar = QProgressBar()
        progress_bar.setTextVisible(False)
        progress_bar.setRange(0, max(total_layers, 1))
        progress_bar.setValue(0)
        dialog_layout.addWidget(progress_label)
        dialog_layout.addWidget(progress_bar)
        dialog.setStyleSheet(
            "QDialog { background: #f6f7f9; }"
            "QLabel { color: #1d2430; font-size: 12px; }"
            "QProgressBar {"
            " background: #e7ebf0;"
            " border: 1px solid #d6dde6;"
            " border-radius: 5px;"
            " min-height: 12px;"
            " max-height: 12px;"
            " }"
            "QProgressBar::chunk { background: #4f9cf9; border-radius: 4px; }"
        )

        workspace_load_state["active"] = True
        workspace_load_state["dialog"] = dialog
        workspace_load_state["label"] = progress_label
        workspace_load_state["bar"] = progress_bar
        workspace_load_state["module"] = _workspace_module
        workspace_load_state["path"] = str(manifest_path)
        workspace_load_state["payload"] = payload
        workspace_load_state["records"] = ordered_records
        workspace_load_state["index"] = 0
        workspace_load_state["restored_layers"] = []
        workspace_load_state["skipped_layers"] = []
        workspace_load_state["clear_existing"] = bool(workspace_clear_checkbox.isChecked())
        workspace_load_state["phase"] = "fast_layers" if fast_records else "deferred_sources"
        workspace_load_state["source_layers"] = len(source_records)
        workspace_load_state["base_layer_count"] = 0 if bool(workspace_clear_checkbox.isChecked()) else len(viewer.layers)
        workspace_load_state["restored_records"] = []
        set_workspace_controls_enabled(False)
        if source_records:
            workspace_status.setText(
                f"Loading {total_layers} layer(s) from {manifest_path.name}..."
                f" {len(source_records)} source-backed layer(s) will open last."
            )
        else:
            workspace_status.setText(f"Loading {total_layers} layer(s) from {manifest_path.name}...")
        set_status("Status: loading workspace...", ok=None)
        append_log(
            f"Starting workspace load: {manifest_path} | layers={total_layers} source_layers={len(source_records)}"
        )

        def finish_workspace_load() -> None:
            payload = workspace_load_state.get("payload") or {}
            _workspace_module = workspace_load_state.get("module")
            if _workspace_module is not None:
                try:
                    _workspace_module.apply_workspace_viewer_state(viewer, payload)
                except Exception as exc:
                    append_log(f"Workspace viewer-state restore warning: {exc}")
            result = {
                "path": workspace_load_state["path"],
                "restored_layers": list(workspace_load_state["restored_layers"]),
                "skipped_layers": list(workspace_load_state["skipped_layers"]),
            }
            ui_state["last_workspace_path"] = str(result["path"])
            save_ui_state(ui_state)
            refresh_workspace_path_label()
            refresh_context()
            skipped = result.get("skipped_layers", []) or []
            restored_count = len(result.get("restored_layers", []) or [])
            if skipped:
                workspace_status.setText(
                    f"Restored {restored_count} layer(s). Skipped {len(skipped)} layer(s) that could not be reopened."
                )
            else:
                workspace_status.setText(f"Restored {restored_count} layer(s) from {result['path']}.")
            append_log(
                f"Loaded workspace manifest: {result['path']} | restored_layers={restored_count} skipped={len(skipped)}"
            )
            set_status("Status: workspace loaded", ok=True)
            workspace_load_state["active"] = False
            workspace_load_state["module"] = None
            workspace_load_state["payload"] = None
            workspace_load_state["records"] = []
            workspace_load_state["phase"] = "prepare"
            workspace_load_state["source_layers"] = 0
            workspace_load_state["base_layer_count"] = 0
            workspace_load_state["restored_records"] = []
            set_workspace_controls_enabled(True)
            close_workspace_load_dialog()

        def process_next_workspace_layer() -> None:
            if not workspace_load_state["active"]:
                return
            if workspace_load_state["index"] == 0 and workspace_load_state["clear_existing"]:
                try:
                    workspace_load_state["module"].clear_workspace_layers(viewer)
                except Exception as exc:
                    workspace_load_state["active"] = False
                    workspace_load_state["module"] = None
                    workspace_load_state["payload"] = None
                    workspace_load_state["records"] = []
                    workspace_load_state["phase"] = "prepare"
                    workspace_load_state["source_layers"] = 0
                    workspace_load_state["base_layer_count"] = 0
                    workspace_load_state["restored_records"] = []
                    set_workspace_controls_enabled(True)
                    close_workspace_load_dialog()
                    append_log(f"Load workspace failed while clearing layers: {exc}")
                    set_status("Status: load workspace failed", ok=False)
                    workspace_status.setText(f"Load failed: {exc}")
                    return
            records = workspace_load_state["records"]
            if workspace_load_state["index"] >= len(records):
                finish_workspace_load()
                return
            deferred_start = len(records) - int(workspace_load_state.get("source_layers", 0))
            if (
                workspace_load_state["phase"] != "deferred_sources"
                and int(workspace_load_state.get("source_layers", 0)) > 0
                and workspace_load_state["index"] >= deferred_start
            ):
                workspace_load_state["phase"] = "deferred_sources"
                workspace_status.setText(
                    f"Loaded lightweight layers. Opening {workspace_load_state['source_layers']} source-backed layer(s)..."
                )
            update_workspace_load_progress()
            record = records[workspace_load_state["index"]]
            try:
                layer = workspace_load_state["module"].restore_workspace_layer(
                    viewer,
                    record,
                    manifest_path=workspace_load_state["path"],
                )
            except Exception as exc:
                workspace_load_state["skipped_layers"].append(
                    {"name": str(record.get("name", "unknown")), "reason": str(exc)}
                )
            else:
                if layer is None:
                    workspace_load_state["skipped_layers"].append(
                        {
                            "name": str(record.get("name", "unknown")),
                            "reason": "could not restore layer from source or inline data",
                        }
                    )
                else:
                    workspace_load_state["restored_layers"].append(str(getattr(layer, "name", "")))
                    try:
                        target_order = int(record.get("__workspace_saved_index__", workspace_load_state["index"]))
                    except Exception:
                        target_order = int(workspace_load_state["index"])
                    restored_records = list(workspace_load_state.get("restored_records", []))
                    insertion_rank = sum(
                        1
                        for item in restored_records
                        if int(item.get("saved_index", 0)) < target_order
                    )
                    workspace_load_state["restored_records"].append(
                        {"saved_index": target_order, "name": str(getattr(layer, "name", ""))}
                    )
                    try:
                        current_index = len(viewer.layers) - 1
                        desired_index = int(workspace_load_state.get("base_layer_count", 0)) + insertion_rank
                        if current_index != desired_index:
                            viewer.layers.move(current_index, desired_index)
                    except Exception as exc:
                        append_log(
                            f"Workspace layer reorder warning for {getattr(layer, 'name', '')}: {exc}"
                        )
            workspace_load_state["index"] += 1
            progress_bar.setValue(workspace_load_state["index"])
            delay_ms = 15 if workspace_load_state["phase"] == "deferred_sources" else 0
            QTimer.singleShot(delay_ms, process_next_workspace_layer)

        dialog.show()
        dialog.raise_()
        update_workspace_load_progress()
        QTimer.singleShot(0, process_next_workspace_layer)

    save_workspace_btn.clicked.connect(lambda _checked=False: save_workspace_common(choose_path=False))
    save_workspace_as_btn.clicked.connect(lambda _checked=False: save_workspace_common(choose_path=True))
    load_workspace_btn.clicked.connect(lambda _checked=False: load_workspace_common(choose_path=True))
    restore_workspace_btn.clicked.connect(lambda _checked=False: load_workspace_common(choose_path=False))

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
            template_record = current_template_record()
            template_selected = template_record is not None
            load_label, run_label = template_button_labels(template_record)
            template_load_btn.setText(load_label)
            template_run_btn.setText(run_label)
            prompt_library_hint.setText(template_hint_text(template_record))
            template_load_btn.setEnabled(template_selected)
            template_run_btn.setEnabled(template_selected)
            action_load_btn.setEnabled(False)
            action_run_btn.setEnabled(False)
            action_add_shortcut_btn.setEnabled(False)
            return
        if current_library_kind() == "action":
            save_prompt_btn.setText("Save")
            pin_prompt_btn.setText("Pin")
            delete_prompt_btn.setText("Delete")
            clear_prompt_btn.setText("Clear")
            save_prompt_btn.setEnabled(False)
            pin_prompt_btn.setEnabled(False)
            delete_prompt_btn.setEnabled(False)
            clear_prompt_btn.setEnabled(False)
            prompt_library_hint.setText(
                "Click an action to preview its purpose and workflow. Use Load Action to insert the prompt form or Run Action for direct execution."
            )
            template_load_btn.setEnabled(False)
            template_run_btn.setEnabled(False)
            action_selected = current_action_record() is not None
            action_load_btn.setEnabled(action_selected)
            action_run_btn.setEnabled(action_selected)
            action_add_shortcut_btn.setEnabled(action_selected)
            return
        template_load_btn.setEnabled(False)
        template_run_btn.setEnabled(False)
        action_load_btn.setEnabled(False)
        action_run_btn.setEnabled(False)
        action_add_shortcut_btn.setEnabled(False)
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

    def _tree_item_path(item: QTreeWidgetItem | None) -> tuple[str, ...]:
        parts: list[str] = []
        current = item
        while current is not None:
            parts.append(current.text(0))
            current = current.parent()
        return tuple(reversed(parts))

    def _capture_tree_state(tree: QTreeWidget, *, selected_record_id: str = "") -> tuple[set[tuple[str, ...]], str]:
        expanded_paths: set[tuple[str, ...]] = set()

        def visit(item: QTreeWidgetItem):
            if item.isExpanded():
                expanded_paths.add(_tree_item_path(item))
            for child_index in range(item.childCount()):
                visit(item.child(child_index))

        for index in range(tree.topLevelItemCount()):
            visit(tree.topLevelItem(index))

        current_item = tree.currentItem()
        if not selected_record_id and current_item is not None:
            record = current_item.data(0, Qt.UserRole)
            if isinstance(record, dict):
                selected_record_id = str(record.get("id", "")).strip()
        return expanded_paths, selected_record_id

    def _restore_tree_expansion(tree: QTreeWidget, expanded_paths: set[tuple[str, ...]]) -> None:
        def visit(item: QTreeWidgetItem):
            item.setExpanded(_tree_item_path(item) in expanded_paths)
            for child_index in range(item.childCount()):
                visit(item.child(child_index))

        for index in range(tree.topLevelItemCount()):
            visit(tree.topLevelItem(index))

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
        template_expanded_paths, selected_template_id = _capture_tree_state(template_tree)
        template_tree.clear()
        selected_template_item: QTreeWidgetItem | None = None
        for section in template_library_state.get("sections", []):
            if not isinstance(section, dict):
                continue
            section_key = str(section.get("key", "")).strip()
            palette = template_section_colors(section_key)
            section_item = QTreeWidgetItem([str(section.get("label", "Templates")).strip() or "Templates"])
            section_item.setFlags(section_item.flags() & ~Qt.ItemIsSelectable)
            section_item.setFirstColumnSpanned(True)
            section_item.setForeground(0, QColor(palette["section"]))
            template_tree.addTopLevelItem(section_item)
            for category in section.get("categories", []):
                if not isinstance(category, dict):
                    continue
                category_item = QTreeWidgetItem([str(category.get("name", "Templates")).strip() or "Templates"])
                category_item.setFlags(category_item.flags() & ~Qt.ItemIsSelectable)
                category_item.setFirstColumnSpanned(True)
                category_item.setForeground(0, QColor(palette["category"]))
                section_item.addChild(category_item)
                for record in category.get("templates", []):
                    if not is_template_record(record):
                        continue
                    child = QTreeWidgetItem([str(record.get("title", "Untitled Template")).strip() or "Untitled Template"])
                    child.setData(0, Qt.UserRole, record)
                    child.setForeground(0, QColor(palette["item"]))
                    category_item.addChild(child)
                    if str(record.get("id", "")).strip() == selected_template_id:
                        selected_template_item = child
        _restore_tree_expansion(template_tree, template_expanded_paths)
        if selected_template_item is not None:
            template_tree.setCurrentItem(selected_template_item)
        else:
            template_tree.setCurrentItem(None)
            template_preview.clear()
        action_expanded_paths, selected_action_id = _capture_tree_state(action_tree)
        action_tree.clear()
        action_categories = [str(name).strip() for name in action_library_state.get("categories", []) if str(name).strip()]
        action_lookup: dict[str, QTreeWidgetItem] = {}
        selected_action_item: QTreeWidgetItem | None = None
        for category in action_categories:
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(category_item.flags() & ~Qt.ItemIsSelectable)
            category_item.setFirstColumnSpanned(True)
            action_lookup[category] = category_item
            action_tree.addTopLevelItem(category_item)
        for record in action_library_state.get("actions", []):
            if not isinstance(record, dict) or not isinstance(record.get("execution"), dict):
                continue
            category = str(record.get("category", "Actions")).strip() or "Actions"
            parent = action_lookup.get(category)
            if parent is None:
                parent = QTreeWidgetItem([category])
                parent.setFlags(parent.flags() & ~Qt.ItemIsSelectable)
                parent.setFirstColumnSpanned(True)
                action_lookup[category] = parent
                action_tree.addTopLevelItem(parent)
            group = str(record.get("group", "")).strip()
            if group:
                subgroup_key = f"{category}::{group}"
                subgroup = action_lookup.get(subgroup_key)
                if subgroup is None:
                    subgroup = QTreeWidgetItem([group])
                    subgroup.setFlags(subgroup.flags() & ~Qt.ItemIsSelectable)
                    subgroup.setFirstColumnSpanned(True)
                    action_lookup[subgroup_key] = subgroup
                    parent.addChild(subgroup)
                parent = subgroup
            child = QTreeWidgetItem([str(record.get("title", "Untitled Action")).strip() or "Untitled Action"])
            child.setData(0, Qt.UserRole, record)
            parent.addChild(child)
            if str(record.get("id", "")).strip() == selected_action_id:
                selected_action_item = child
        _restore_tree_expansion(action_tree, action_expanded_paths)
        if selected_action_item is not None:
            action_tree.setCurrentItem(selected_action_item)
        else:
            action_tree.setCurrentItem(None)
            action_preview.clear()
        refresh_library_controls()

    def current_template_record() -> dict | None:
        item = template_tree.currentItem()
        if item is None:
            return None
        record = item.data(0, Qt.UserRole)
        return record if is_template_record(record) else None

    def current_action_record() -> dict | None:
        item = action_tree.currentItem()
        if item is None:
            return None
        record = item.data(0, Qt.UserRole)
        return record if isinstance(record, dict) and isinstance(record.get("execution"), dict) else None

    def template_preview_text(record: dict) -> str:
        return catalog_template_preview_text(record)

    def action_preview_text(record: dict) -> str:
        title = str(record.get("title", "")).strip() or "Untitled Action"
        category = str(record.get("category", "")).strip() or "Actions"
        description = str(record.get("description", "")).strip()
        tags = [str(tag).strip() for tag in record.get("tags", []) if str(tag).strip()]
        best_for = str(record.get("best_for", "")).strip()
        expected_input = str(record.get("expected_input", "")).strip()
        load_prompt = str(record.get("load_prompt", "")).strip()
        execution = record.get("execution", {})
        runtime = record.get("runtime", {})
        how_it_works = str(record.get("how_it_works", "")).strip()
        workflow = [str(step).strip() for step in record.get("workflow", []) if str(step).strip()]
        parameter_hints = [str(step).strip() for step in record.get("parameter_hints", []) if str(step).strip()]
        prompt_examples = [str(step).strip() for step in record.get("prompt_examples", []) if str(step).strip()]
        runtime_flags: list[str] = []
        if isinstance(runtime, dict):
            if runtime.get("uses_viewer"):
                runtime_flags.append("Viewer")
            if runtime.get("uses_selected_layer"):
                runtime_flags.append("Selected Layer")
            if runtime.get("uses_run_in_background"):
                runtime_flags.append("Background")
        lines = [f"Action: {title}", f"Category: {category}"]
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
        if runtime_flags:
            lines.append(f"Runtime: {', '.join(runtime_flags)}")
        if description:
            lines.extend(["", description])
        if best_for:
            lines.extend(["", f"Best for: {best_for}"])
        if expected_input:
            lines.extend(["", f"Expected input: {expected_input}"])
        if how_it_works:
            lines.extend(["", f"How it works: {how_it_works}"])
        if workflow:
            lines.extend(["", "Workflow:"])
            lines.extend(f"{index}. {step}" for index, step in enumerate(workflow, start=1))
        if parameter_hints:
            lines.extend(["", "Parameter hints:"])
            lines.extend(f"- {step}" for step in parameter_hints)
        if prompt_examples:
            lines.extend(["", "Prompt examples:"])
            lines.extend(f"- {step}" for step in prompt_examples)
        if load_prompt:
            lines.extend(["", "Load to Prompt:", load_prompt])
        if isinstance(execution, dict):
            kind = str(execution.get("kind", "")).strip()
            target = str(execution.get("target", "")).strip()
            if kind and target:
                lines.extend(["", f"Internal target: {kind}:{target}"])
        return "\n".join(lines).strip()

    def sam2_managed_points_layer_name(image_layer) -> str | None:
        if image_layer is None:
            return None
        return f"{str(getattr(image_layer, 'name', '')).strip()}_sam2_prompts"

    def current_sam2_target_image_layer():
        if viewer is None:
            return None
        active = getattr(getattr(viewer.layers, "selection", None), "active", None)
        if isinstance(active, napari.layers.Image) and not getattr(active, "rgb", False):
            return active
        if isinstance(active, napari.layers.Points):
            source_name = str(getattr(active, "metadata", {}).get("sam2_source_image", "")).strip()
            if source_name and source_name in viewer.layers:
                source_layer = viewer.layers[source_name]
                if isinstance(source_layer, napari.layers.Image) and not getattr(source_layer, "rgb", False):
                    return source_layer
        for layer in viewer.layers if viewer is not None else []:
            if isinstance(layer, napari.layers.Image) and not getattr(layer, "rgb", False):
                return layer
        return None

    def selected_scalable_layer():
        layer = None if viewer is None else getattr(getattr(viewer.layers, "selection", None), "active", None)
        if layer is None:
            raise ValueError("Select a layer first.")
        if not hasattr(layer, "scale"):
            raise ValueError(f"Selected layer [{getattr(layer, 'name', 'layer')}] does not support scale.")
        data = np.asarray(getattr(layer, "data", None))
        ndim = int(getattr(layer, "ndim", data.ndim if data is not None else 0))
        if ndim <= 0:
            raise ValueError(f"Selected layer [{getattr(layer, 'name', 'layer')}] does not have a valid dimensionality.")
        return layer, ndim

    def set_selected_layer_uniform_scale(value: float) -> tuple[str, tuple[float, ...]]:
        layer, ndim = selected_scalable_layer()
        scale = tuple([float(value)] * int(ndim))
        layer.scale = scale
        refresh_context()
        return str(getattr(layer, "name", "layer")), scale

    def set_selected_layer_scale_0_1() -> tuple[str, tuple[float, ...]]:
        return set_selected_layer_uniform_scale(0.1)

    def reset_selected_layer_scale() -> tuple[str, tuple[float, ...]]:
        return set_selected_layer_uniform_scale(1.0)

    def resolve_sam2_points_action_arguments() -> dict[str, object]:
        image_layer = current_sam2_target_image_layer()
        if image_layer is None:
            raise ValueError("Select a grayscale image layer or its SAM2 points layer first.")
        points_name = sam2_managed_points_layer_name(image_layer)
        if not points_name or points_name not in viewer.layers:
            raise ValueError(
                f"No SAM2 points layer found for [{image_layer.name}]. Run Initialize SAM2 Points first."
            )
        points_layer = viewer.layers[points_name]
        if not isinstance(points_layer, napari.layers.Points):
            raise ValueError(f"Layer [{points_name}] exists but is not a SAM2 points layer.")
        return {
            "image_layer": str(image_layer.name),
            "points_layer": str(points_layer.name),
        }

    def set_sam2_points_mode(layer, label_value: int) -> None:
        value = 1 if int(label_value) == 1 else 0
        try:
            layer.feature_defaults = {"sam_label": value}
        except Exception:
            pass
        try:
            layer.current_properties = {"sam_label": np.asarray([value], dtype=np.int32)}
        except Exception:
            pass
        color = "#4caf50" if value == 1 else "#ef5350"
        try:
            layer.current_face_color = color
        except Exception:
            pass
        try:
            layer.current_border_color = "white"
        except Exception:
            pass
        metadata = dict(getattr(layer, "metadata", {}) or {})
        metadata["sam2_current_label"] = value
        layer.metadata = metadata

    def sync_sam2_points_colors(layer) -> None:
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
            labels = np.asarray([], dtype=np.int32)
        if labels.size == 0:
            return
        try:
            layer.face_color = ["#4caf50" if int(label) == 1 else "#ef5350" for label in labels.tolist()]
            layer.border_color = ["white"] * int(labels.size)
        except Exception:
            pass

    def toggle_sam2_points_layer_polarity(layer) -> None:
        selected = []
        try:
            selected = sorted(int(index) for index in (getattr(layer, "selected_data", set()) or set()))
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
            data = np.asarray(getattr(layer, "data", np.empty((0, 2))), dtype=np.float32)
            labels = np.ones((data.shape[0],), dtype=np.int32)

        if selected:
            updated = labels.copy()
            for index in selected:
                if 0 <= index < updated.shape[0]:
                    updated[index] = 0 if int(updated[index]) == 1 else 1
            try:
                layer.features = {"sam_label": updated}
            except Exception:
                try:
                    features = getattr(layer, "features", None)
                    if features is not None:
                        features["sam_label"] = updated
                        layer.features = features
                except Exception:
                    pass
            sync_sam2_points_colors(layer)
            append_log(f"Toggled polarity for {len(selected)} SAM2 prompt point(s) on [{layer.name}].")
            set_status("Status: SAM2 point polarity toggled", ok=True)
            return

        metadata = dict(getattr(layer, "metadata", {}) or {})
        current = 1 if int(metadata.get("sam2_current_label", 1)) == 1 else 0
        next_value = 0 if current == 1 else 1
        set_sam2_points_mode(layer, next_value)
        append_log(
            f"SAM2 point polarity set to [{'positive' if next_value == 1 else 'negative'}] on [{layer.name}]."
        )
        set_status("Status: SAM2 point polarity toggled", ok=True)

    def register_sam2_points_toggle(layer) -> None:
        if layer is None or not hasattr(layer, "bind_key") or getattr(layer, "_sam2_toggle_registered", False):
            return
        try:
            @layer.bind_key("t", overwrite=True)
            def _toggle_sam2_prompt_polarity(active_layer):
                toggle_sam2_points_layer_polarity(active_layer)
        except TypeError:
            try:
                layer.bind_key("t", toggle_sam2_points_layer_polarity, overwrite=True)
            except Exception:
                return
        except Exception:
            return
        setattr(layer, "_sam2_toggle_registered", True)

    def initialize_sam2_points_layer() -> str | None:
        selected = viewer.layers.selection.active if viewer is not None else None
        if not isinstance(selected, napari.layers.Image) or getattr(selected, "rgb", False):
            raise ValueError("Select a 2D or 3D grayscale image layer before initializing SAM2 points.")
        image_data = np.asarray(selected.data)
        if image_data.ndim not in {2, 3}:
            raise ValueError("SAM2 points initialization currently supports 2D or 3D grayscale image layers only.")
        layer_name = sam2_managed_points_layer_name(selected)
        if not layer_name:
            raise ValueError("Could not derive a SAM2 points layer name from the selected image.")
        if layer_name in viewer.layers:
            layer = viewer.layers[layer_name]
            if not isinstance(layer, napari.layers.Points):
                raise ValueError(f"Layer [{layer_name}] already exists and is not a Points layer.")
        else:
            empty = np.empty((0, image_data.ndim), dtype=np.float32)
            layer = viewer.add_points(
                empty,
                name=layer_name,
                features={"sam_label": np.empty((0,), dtype=np.int32)},
                face_color="#4caf50",
                border_color="white",
                size=10,
            )
        metadata = dict(getattr(layer, "metadata", {}) or {})
        metadata["sam2_managed_points"] = True
        metadata["sam2_source_image"] = str(selected.name)
        layer.metadata = metadata
        set_sam2_points_mode(layer, int(metadata.get("sam2_current_label", 1)))
        sync_sam2_points_colors(layer)
        register_sam2_points_toggle(layer)
        try:
            layer.mode = "add"
        except Exception:
            pass
        try:
            viewer.layers.selection.active = layer
        except Exception:
            pass
        append_chat_message(
            "assistant",
            f"SAM2 points initialized for [{selected.name}] as [{layer.name}].\n"
            "Add positive and negative prompt points on that layer. Press `T` on the active SAM2 points layer "
            "to toggle polarity for new points, or to flip selected points.",
        )
        append_log(f"Initialized SAM2 points layer [{layer.name}] for image [{selected.name}].")
        set_status("Status: SAM2 points initialized", ok=True)
        return layer.name

    def annotation_anchor_layer():
        active = viewer.layers.selection.active if viewer is not None else None
        if active is not None and int(getattr(active, "ndim", 0) or 0) in {2, 3}:
            return active
        for layer in viewer.layers if viewer is not None else []:
            if isinstance(layer, napari.layers.Image) and int(getattr(layer, "ndim", 0) or 0) in {2, 3}:
                return layer
        return None

    def text_annotation_layer_name(anchor_layer=None) -> str:
        if anchor_layer is None:
            return "text_annotations"
        base_name = str(getattr(anchor_layer, "name", "") or "").strip()
        return f"{base_name}_text_annotations" if base_name else "text_annotations"

    def default_text_annotation_style() -> dict[str, object]:
        return {
            "string": "{label}",
            "size": 12,
            "color": "yellow",
            "anchor": "upper_left",
            "translation": [0.0, -6.0],
            "blending": "translucent",
            "visible": True,
            "scaling": False,
            "rotation": 0.0,
        }

    def text_annotation_style_from_layer(layer) -> dict[str, object]:
        metadata = dict(getattr(layer, "metadata", {}) or {})
        style = default_text_annotation_style()
        saved = metadata.get("text_annotation_text_style")
        if isinstance(saved, dict):
            style.update(saved)
        return style

    def configure_text_annotation_layer(
        layer,
        *,
        source_layer_name: str,
        current_text: str | None = None,
        style_updates: dict[str, object] | None = None,
    ) -> None:
        metadata = dict(getattr(layer, "metadata", {}) or {})
        metadata["text_annotations_managed"] = True
        metadata["text_annotation_source_layer"] = source_layer_name
        style = text_annotation_style_from_layer(layer)
        if style_updates:
            style.update(style_updates)
        metadata["text_annotation_text_style"] = style
        layer.metadata = metadata
        try:
            layer.text = dict(style)
        except Exception:
            pass
        if current_text is not None:
            try:
                layer.feature_defaults = {"label": str(current_text)}
            except Exception:
                pass
            try:
                layer.current_properties = {"label": np.asarray([str(current_text)], dtype=object)}
            except Exception:
                pass

    def text_annotation_labels(layer) -> list[str]:
        labels: list[str] = []
        features = getattr(layer, "features", None)
        if features is None:
            return labels
        try:
            if "label" in features:
                labels = [str(value or "").strip() for value in list(features["label"])]
        except Exception:
            labels = []
        return labels

    def ensure_text_annotation_layer():
        anchor_layer = annotation_anchor_layer()
        if anchor_layer is None:
            raise ValueError("Select a 2D or 3D layer, or load an image, before adding text annotations.")
        if int(getattr(anchor_layer, "ndim", 0) or 0) not in {2, 3}:
            raise ValueError("Text annotations currently support 2D or 3D layers only.")
        layer_name = text_annotation_layer_name(anchor_layer)
        if layer_name in viewer.layers:
            layer = viewer.layers[layer_name]
            if not isinstance(layer, napari.layers.Points):
                raise ValueError(f"Layer [{layer_name}] already exists and is not a Points layer.")
        else:
            empty = np.empty((0, int(getattr(anchor_layer, "ndim", 2) or 2)), dtype=np.float32)
            layer = viewer.add_points(
                empty,
                name=layer_name,
                features={"label": np.empty((0,), dtype=object)},
                size=6,
                face_color="transparent",
                border_color="#ffd54f",
                border_width=1,
            )
        configure_text_annotation_layer(layer, source_layer_name=str(getattr(anchor_layer, "name", "") or ""))
        return layer, anchor_layer

    def start_text_annotation_mode(*_args):
        try:
            layer, anchor_layer = ensure_text_annotation_layer()
        except Exception as exc:
            append_log(f"Text annotation setup failed: {exc}")
            set_status("Status: text annotation unavailable", ok=False)
            workspace_status.setText(f"Text annotation failed: {exc}")
            return
        existing_defaults = getattr(layer, "feature_defaults", None)
        default_text = ""
        try:
            if existing_defaults is not None and "label" in existing_defaults:
                default_text = str(list(existing_defaults["label"])[0] or "").strip()
        except Exception:
            default_text = ""
        text_value, accepted = QInputDialog.getText(
            root,
            "Text Annotation",
            "Annotation text:",
            text=default_text,
        )
        if not accepted:
            return
        current_text = str(text_value or "").strip()
        if not current_text:
            set_status("Status: annotation text is empty", ok=False)
            append_log("Text annotation start skipped: empty text.")
            return
        configure_text_annotation_layer(
            layer,
            source_layer_name=str(getattr(anchor_layer, "name", "") or ""),
            current_text=current_text,
        )
        try:
            layer.mode = "add"
        except Exception:
            pass
        try:
            viewer.layers.selection.active = layer
        except Exception:
            pass
        append_chat_message(
            "assistant",
            f"Text annotation mode is ready on [{layer.name}] for [{anchor_layer.name}].\n"
            f"Current text: `{current_text}`\n"
            "Click in the viewer to place the text as an overlay. Run Text Annotation again to change the text.",
        )
        append_log(f"Text annotation mode enabled on [{layer.name}] with label [{current_text}].")
        set_status("Status: text annotation mode enabled", ok=True)

    def show_text_annotation_editor(*_args):
        try:
            layer, anchor_layer = ensure_text_annotation_layer()
        except Exception as exc:
            append_log(f"Text annotation editor unavailable: {exc}")
            set_status("Status: text annotation unavailable", ok=False)
            workspace_status.setText(f"Text annotation failed: {exc}")
            return

        dialog = QDialog(root)
        dialog.setWindowTitle("Text Annotation")
        dialog.resize(480, 420)
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(16, 16, 16, 14)
        dialog_layout.setSpacing(10)

        hint = QLabel(
            "Place non-destructive text labels on the viewer. "
            "Annotations are stored in a managed Points layer and do not modify image pixels."
        )
        hint.setWordWrap(True)
        dialog_layout.addWidget(hint)

        form = QFormLayout()
        layer_label = QLabel(f"{layer.name}  |  source: {anchor_layer.name}")
        layer_label.setWordWrap(True)
        text_edit = QLineEdit()
        size_edit = QLineEdit()
        size_edit.setPlaceholderText("12")
        color_combo = QComboBox()
        color_choices = [
            ("Yellow", "yellow"),
            ("White", "white"),
            ("Cyan", "cyan"),
            ("Green", "#66bb6a"),
            ("Red", "#ef5350"),
            ("Orange", "#ffa726"),
        ]
        for label_text, color_value in color_choices:
            color_combo.addItem(label_text, color_value)
        form.addRow("Layer:", layer_label)
        form.addRow("Text:", text_edit)
        form.addRow("Size:", size_edit)
        form.addRow("Color:", color_combo)
        dialog_layout.addLayout(form)

        annotations_list = QListWidget()
        annotations_list.setSelectionMode(QAbstractItemView.SingleSelection)
        annotations_list.setStyleSheet(
            "QListWidget { background: #101820; color: #d6deeb; border: 1px solid #22304a; } "
            "QListWidget::item:selected { background: #1d2a44; color: #e8f1ff; }"
        )
        dialog_layout.addWidget(annotations_list, 1)

        status_label = QLabel("Choose text and click Place Text, then click in the viewer.")
        status_label.setWordWrap(True)
        dialog_layout.addWidget(status_label)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        place_btn = QPushButton("Place Text")
        rename_btn = QPushButton("Rename Selected")
        delete_btn = QPushButton("Delete Selected")
        close_btn = QPushButton("Close")
        button_layout.addWidget(place_btn)
        button_layout.addWidget(rename_btn)
        button_layout.addWidget(delete_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(close_btn)
        dialog_layout.addWidget(button_row)

        def refresh_editor_fields() -> None:
            nonlocal layer, anchor_layer
            try:
                current_layer, current_anchor = ensure_text_annotation_layer()
                layer = current_layer
                anchor_layer = current_anchor
            except Exception:
                pass
            layer_label.setText(f"{layer.name}  |  source: {anchor_layer.name}")
            defaults = getattr(layer, "feature_defaults", None)
            current_text = ""
            try:
                if defaults is not None and "label" in defaults:
                    values = list(defaults["label"])
                    current_text = str(values[0] or "").strip() if values else ""
            except Exception:
                current_text = ""
            if not current_text:
                labels = text_annotation_labels(layer)
                if labels:
                    current_text = labels[-1]
            text_edit.setText(current_text)
            style = text_annotation_style_from_layer(layer)
            size_edit.setText(str(int(float(style.get("size", 12) or 12))))
            color_value = str(style.get("color", "yellow"))
            combo_index = color_combo.findData(color_value)
            color_combo.setCurrentIndex(combo_index if combo_index >= 0 else 0)

        def refresh_annotation_list() -> None:
            annotations_list.clear()
            labels = text_annotation_labels(layer)
            coordinates = np.asarray(getattr(layer, "data", []), dtype=float)
            for index, label_text in enumerate(labels):
                point_text = ""
                if coordinates.ndim == 2 and index < len(coordinates):
                    point_text = ", ".join(f"{value:.1f}" for value in coordinates[index].tolist())
                item = QListWidgetItem(
                    f"{index + 1}. {label_text or '[empty]'}" + (f"  ({point_text})" if point_text else "")
                )
                item.setData(Qt.UserRole, index)
                annotations_list.addItem(item)

        def selected_annotation_index() -> int | None:
            item = annotations_list.currentItem()
            if item is None:
                return None
            data = item.data(Qt.UserRole)
            try:
                return int(data)
            except Exception:
                return None

        def current_style_updates() -> dict[str, object]:
            color_value = color_combo.currentData() or "yellow"
            size_value = 12
            try:
                size_value = max(6, min(72, int(float(size_edit.text().strip() or "12"))))
            except Exception:
                size_value = 12
            size_edit.setText(str(size_value))
            return {"size": size_value, "color": str(color_value)}

        def place_text() -> None:
            current_text = text_edit.text().strip()
            if not current_text:
                status_label.setText("Enter annotation text first.")
                return
            configure_text_annotation_layer(
                layer,
                source_layer_name=str(getattr(anchor_layer, "name", "") or ""),
                current_text=current_text,
                style_updates=current_style_updates(),
            )
            try:
                layer.mode = "add"
            except Exception:
                pass
            try:
                viewer.layers.selection.active = layer
            except Exception:
                pass
            refresh_annotation_list()
            status_label.setText("Placement mode is active. Click in the viewer to place the text.")
            set_status("Status: text annotation placement enabled", ok=True)

        def rename_selected_annotation() -> None:
            index = selected_annotation_index()
            if index is None:
                status_label.setText("Select an annotation to rename.")
                return
            labels = text_annotation_labels(layer)
            if index < 0 or index >= len(labels):
                status_label.setText("Selected annotation is no longer available.")
                refresh_annotation_list()
                return
            new_label, accepted = QInputDialog.getText(
                dialog,
                "Rename Annotation",
                "Annotation text:",
                text=labels[index],
            )
            if not accepted:
                return
            updated = str(new_label or "").strip()
            if not updated:
                status_label.setText("Annotation text cannot be empty.")
                return
            labels[index] = updated
            try:
                layer.features = {"label": np.asarray(labels, dtype=object)}
            except Exception as exc:
                status_label.setText(f"Rename failed: {exc}")
                return
            configure_text_annotation_layer(
                layer,
                source_layer_name=str(getattr(anchor_layer, "name", "") or ""),
                current_text=updated,
                style_updates=current_style_updates(),
            )
            refresh_annotation_list()
            if index < annotations_list.count():
                annotations_list.setCurrentRow(index)
            text_edit.setText(updated)
            status_label.setText("Renamed the selected annotation.")

        def delete_selected_annotation() -> None:
            index = selected_annotation_index()
            if index is None:
                status_label.setText("Select an annotation to delete.")
                return
            try:
                layer.selected_data = {int(index)}
                layer.remove_selected()
                layer.selected_data = set()
            except Exception as exc:
                status_label.setText(f"Delete failed: {exc}")
                return
            refresh_annotation_list()
            status_label.setText("Deleted the selected annotation.")

        def sync_selection_from_list() -> None:
            index = selected_annotation_index()
            if index is None:
                return
            try:
                layer.selected_data = {int(index)}
            except Exception:
                pass
            labels = text_annotation_labels(layer)
            if 0 <= index < len(labels):
                text_edit.setText(labels[index])

        refresh_editor_fields()
        refresh_annotation_list()
        annotations_list.currentItemChanged.connect(lambda *_args: sync_selection_from_list())
        place_btn.clicked.connect(place_text)
        rename_btn.clicked.connect(rename_selected_annotation)
        delete_btn.clicked.connect(delete_selected_annotation)
        close_btn.clicked.connect(dialog.accept)
        dialog.finished.connect(lambda *_args: refresh_context())
        dialog.show()
        dialog.raise_()
        dialog.exec_()

    def launch_widget_template(record: dict) -> bool:
        template_id = str(record.get("id", "")).strip()
        if template_id == "measure_roi_intensity_metrics":
            if show_intensity_metrics_widget():
                append_log(f"Opened widget template: {record.get('title', 'Untitled Template')}")
                set_status("Status: measurement widget opened", ok=True)
                return True
            return False
        if template_id == "measure_line_profile_gaussian_fit":
            if show_line_profile_widget():
                append_log(f"Opened widget template: {record.get('title', 'Untitled Template')}")
                set_status("Status: measurement widget opened", ok=True)
                return True
            return False
        if template_id == "stats_open_group_comparison_widget":
            prepared = prepare_tool_job(viewer, "open_group_comparison_widget", {})
            run_prepared_tool_request(
                prepared,
                tool_name="open_group_comparison_widget",
                tool_message="Opening the group comparison widget.",
            )
            append_log(f"Opened widget template: {record.get('title', 'Untitled Template')}")
            return True
        return False

    def show_template_preview(item: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None = None):
        del _previous
        if item is None:
            template_preview.clear()
            refresh_library_controls()
            return
        record = item.data(0, Qt.UserRole)
        if not is_template_record(record):
            template_preview.clear()
            refresh_library_controls()
            return
        template_preview.setPlainText(template_preview_text(record))
        refresh_library_controls()

    def show_action_preview(item: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None = None):
        del _previous
        if item is None:
            action_preview.clear()
            refresh_library_controls()
            return
        record = item.data(0, Qt.UserRole)
        if not isinstance(record, dict) or not isinstance(record.get("execution"), dict):
            action_preview.clear()
            refresh_library_controls()
            return
        action_preview.setPlainText(action_preview_text(record))
        refresh_library_controls()

    def load_template_record(record: dict | None, *, run_now: bool = False):
        if not isinstance(record, dict):
            set_status("Status: no template selected", ok=False)
            append_log("Template load skipped: no template record selected.")
            return
        body_text = template_body_text(record)
        if not body_text.strip():
            set_status("Status: selected template is empty", ok=False)
            append_log("Template load skipped: selected template is empty.")
            return
        prompt.setPlainText(body_text)
        prompt.setFocus()
        title = str(record.get("title", "Untitled Template")).strip() or "Untitled Template"
        if template_load_target(record) == "prompt":
            append_log(f"Loaded template prompt: {title}")
            set_status("Status: template prompt loaded", ok=None)
        else:
            append_log(f"Loaded template code: {title}")
            set_status("Status: template code loaded", ok=None)
        if run_now:
            if template_run_target(record) == "send_prompt":
                send_message()
            else:
                run_prompt_code()

    def load_selected_template(*_args):
        record = current_template_record()
        if record is None:
            set_status("Status: no template selected", ok=False)
            append_log("Load template skipped: no selection.")
            return
        load_template_record(record)

    def load_action_record(record: dict | None):
        if not isinstance(record, dict):
            set_status("Status: no action selected", ok=False)
            append_log("Load action skipped: no action record selected.")
            return
        prompt_text = str(record.get("load_prompt", "")).strip()
        if not prompt_text:
            set_status("Status: selected action has no prompt form", ok=False)
            append_log("Load action skipped: selected action has no prompt form.")
            return
        prompt.setPlainText(prompt_text)
        prompt.setFocus()
        append_log(f"Loaded action prompt: {record.get('title', 'Untitled Action')}")
        set_status("Status: action loaded to prompt", ok=None)

    def run_action_record(record: dict | None):
        if not isinstance(record, dict):
            set_status("Status: no action selected", ok=False)
            append_log("Run action skipped: no action record selected.")
            return
        execution = record.get("execution", {})
        if not isinstance(execution, dict):
            set_status("Status: selected action is missing execution data", ok=False)
            append_log("Run action skipped: missing execution data.")
            return
        kind = str(execution.get("kind", "")).strip()
        target = str(execution.get("target", "")).strip()
        arguments = dict(execution.get("arguments", {})) if isinstance(execution.get("arguments"), dict) else {}
        title = str(record.get("title", "Untitled Action")).strip() or "Untitled Action"
        load_prompt = str(record.get("load_prompt", "")).strip()
        if load_prompt:
            append_chat_message("user", load_prompt)
        if kind == "function":
            if target == "open_intensity_metrics_widget":
                if show_intensity_metrics_widget():
                    append_chat_message("assistant", f"Opened {title}.")
            elif target == "open_relabel_mask_widget":
                if show_relabel_mask_widget():
                    append_chat_message("assistant", f"Opened {title}.")
            elif target == "open_line_profile_gaussian_fit_widget":
                if show_line_profile_widget():
                    append_chat_message("assistant", f"Opened {title}.")
            elif target == "save_workspace":
                save_workspace_common(choose_path=False)
            elif target == "save_workspace_as":
                save_workspace_common(choose_path=True)
            elif target == "load_workspace":
                load_workspace_common(choose_path=True)
            elif target == "restore_last_workspace":
                load_workspace_common(choose_path=False)
            elif target == "show_sam2_setup_dialog":
                show_sam2_setup_dialog()
                append_chat_message("assistant", f"Opened {title}.")
            elif target == "show_sam2_live_dialog":
                show_sam2_live_dialog()
                append_chat_message("assistant", f"Opened {title}.")
            elif target == "set_selected_layer_scale_0_1":
                layer_name, scale = set_selected_layer_scale_0_1()
                append_chat_message("assistant", f"Set [{layer_name}] scale to {scale}.")
            elif target == "reset_selected_layer_scale":
                layer_name, scale = reset_selected_layer_scale()
                append_chat_message("assistant", f"Reset [{layer_name}] scale to {scale}.")
            elif target == "initialize_sam2_points_layer":
                initialize_sam2_points_layer()
            else:
                set_status("Status: unknown action target", ok=False)
                append_log(f"Run action failed: unknown function target {target}.")
                return
            append_log(f"Ran action: {title}")
            set_status("Status: action completed", ok=True)
            return
        if kind == "tool":
            if target in {"sam_segment_from_points", "sam_propagate_points_3d"}:
                arguments = {**arguments, **resolve_sam2_points_action_arguments()}
            prepared = prepare_tool_job(viewer, target, arguments)
            run_prepared_tool_request(prepared, tool_name=target, tool_message=f"Running action: {title}")
            append_log(f"Ran action: {title}")
            return
        set_status("Status: unsupported action kind", ok=False)
        append_log(f"Run action failed: unsupported action kind {kind}.")

    def run_selected_template(*_args):
        record = current_template_record()
        if record is None:
            set_status("Status: no template selected", ok=False)
            append_log("Run template skipped: no selection.")
            return
        load_template_record(record, run_now=True)

    def load_selected_action(*_args):
        record = current_action_record()
        if record is None:
            set_status("Status: no action selected", ok=False)
            append_log("Load action skipped: no selection.")
            return
        load_action_record(record)

    def add_action_to_shortcuts(record: dict | None):
        if not isinstance(record, dict):
            set_status("Status: no action selected", ok=False)
            append_log("Add shortcut skipped: no action record selected.")
            return
        action_id = str(record.get("id", "")).strip()
        if not action_id:
            set_status("Status: selected action has no id", ok=False)
            append_log("Add shortcut skipped: selected action has no id.")
            return
        if action_id in shortcut_action_ids:
            set_status("Status: action already in shortcuts", ok=None)
            append_log(f"Shortcut already present: {record.get('title', 'Untitled Action')}")
            return
        for index, existing in enumerate(shortcut_action_ids):
            if not str(existing).strip():
                shortcut_action_ids[index] = action_id
                save_shortcuts()
                refresh_shortcuts()
                set_status("Status: added to shortcuts", ok=True)
                append_log(f"Added shortcut: {record.get('title', 'Untitled Action')}")
                return
        set_status("Status: shortcuts full", ok=False)
        append_log("Add shortcut skipped: all shortcut slots are already used.")

    def add_selected_action_to_shortcuts(*_args):
        record = current_action_record()
        if record is None:
            set_status("Status: no action selected", ok=False)
            append_log("Add shortcut skipped: no selection.")
            return
        add_action_to_shortcuts(record)

    def run_selected_action(*_args):
        record = current_action_record()
        if record is None:
            set_status("Status: no action selected", ok=False)
            append_log("Run action skipped: no selection.")
            return
        run_action_record(record)

    def run_action_button_slot(buttons: list[QPushButton], index: int):
        if index < 0 or index >= len(buttons):
            return
        action_id = str(buttons[index].property("action_id") or "").strip()
        record = action_record_by_id(action_id)
        if not isinstance(record, dict):
            set_status("Status: action slot is empty", ok=False)
            append_log(f"Action slot {index + 1} is empty.")
            return
        run_action_record(record)

    def run_shortcut_button(index: int):
        run_action_button_slot(shortcut_buttons, index)

    def clear_shortcuts(*_args):
        for index in range(len(shortcut_action_ids)):
            shortcut_action_ids[index] = ""
        save_shortcuts()
        refresh_shortcuts()
        set_status("Status: shortcuts cleared", ok=True)
        append_log("Cleared shortcuts.")

    def add_shortcut_row(*_args):
        nonlocal shortcut_slot_count
        shortcut_slot_count = max(6, int(shortcut_slot_count) + 3)
        while len(shortcut_action_ids) < shortcut_slot_count:
            shortcut_action_ids.append("")
        save_shortcuts()
        refresh_shortcuts()
        set_status(f"Status: added shortcut row ({shortcut_slot_count} slots)", ok=True)
        append_log(f"Expanded shortcuts to {shortcut_slot_count} slots.")

    def remove_shortcut_row(*_args):
        nonlocal shortcut_slot_count
        if int(shortcut_slot_count) <= 6:
            set_status("Status: shortcuts already at minimum size", ok=False)
            return
        last_row_start = max(0, int(shortcut_slot_count) - 3)
        occupied = [str(value).strip() for value in shortcut_action_ids[last_row_start:int(shortcut_slot_count)] if str(value).strip()]
        if occupied:
            set_status("Status: clear the last shortcut row before removing it", ok=False)
            append_log("Remove shortcut row blocked: last row still contains assigned shortcuts.")
            return
        shortcut_slot_count = max(6, int(shortcut_slot_count) - 3)
        del shortcut_action_ids[shortcut_slot_count:]
        save_shortcuts()
        refresh_shortcuts()
        set_status(f"Status: removed shortcut row ({shortcut_slot_count} slots)", ok=True)
        append_log(f"Reduced shortcuts to {shortcut_slot_count} slots.")

    def remove_shortcut_slot(index: int):
        if index < 0 or index >= len(shortcut_action_ids):
            return
        action_id = str(shortcut_action_ids[index]).strip()
        if not action_id:
            set_status("Status: shortcut slot is already empty", ok=False)
            return
        title = str(action_record_by_id(action_id).get("title", "Action")).strip() if action_record_by_id(action_id) else "Action"
        shortcut_action_ids[index] = ""
        compacted = [value for value in shortcut_action_ids if str(value).strip()]
        while len(compacted) < max(6, int(shortcut_slot_count)):
            compacted.append("")
        shortcut_action_ids[:] = compacted
        save_shortcuts()
        refresh_shortcuts()
        set_status("Status: shortcut removed", ok=True)
        append_log(f"Removed shortcut: {title}")

    def show_shortcut_menu(index: int, global_pos):
        if index < 0 or index >= len(shortcut_buttons):
            return
        action_id = str(shortcut_buttons[index].property("action_id") or "").strip()
        if not action_id:
            return
        menu = QMenu(root)
        remove_action = menu.addAction("Remove")
        chosen = menu.exec_(global_pos)
        if chosen is remove_action:
            remove_shortcut_slot(index)

    def run_template_tree_item(item: QTreeWidgetItem, _column: int):
        del _column
        record = item.data(0, Qt.UserRole)
        if not is_template_record(record):
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

    def toggle_ui_help(enabled: bool):
        ui_state["ui_help_enabled"] = bool(enabled)
        save_ui_state(ui_state)
        if help_ui_toggle_action.isChecked() != bool(enabled):
            help_ui_toggle_action.blockSignals(True)
            help_ui_toggle_action.setChecked(bool(enabled))
            help_ui_toggle_action.blockSignals(False)
        append_chat_message(
            "assistant",
            (
                "UI Help is on. Short questions about plugin controls can now be answered locally."
                if enabled
                else "UI Help is off. Normal requests will go directly to actions, tools, and chat."
            ),
        )
        append_log(f"UI help {'enabled' if enabled else 'disabled'}.")
        set_status(f"Status: UI help {'enabled' if enabled else 'disabled'}", ok=None)

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
        nonlocal session_memory_state, intent_state, last_failed_tool_state
        turn_id = uuid.uuid4().hex
        request_started_at = time.perf_counter()
        current_profile = selected_layer_profile()
        intent_state = merge_intent_state(
            intent_state,
            extract_turn_intent(text, last_failed_tool_state=last_failed_tool_state),
        )
        session_memory_state = update_session_goal(session_memory_state, text)
        session_memory_state = set_active_dataset_focus(
            session_memory_state,
            "" if not isinstance(current_profile, dict) else str(current_profile.get("layer_name", "")).strip(),
        )
        session_memory_state, promoted_ids = promote_from_user_turn(session_memory_state, text, current_profile)
        if promoted_ids:
            append_log(f"Promoted {len(promoted_ids)} provisional memory item(s) from user follow-up.")
        persist_session_memory()
        intent_event = IntentEvent(
            intent_category=categorize_intent(text),
            intent_description=text[:200],
            layer_context=build_layer_context(current_profile),
            workspace_state=current_workspace_state(),
            success=False,
            duration_ms=0,
            metadata={"full_prompt_hash": prompt_hash(text), "turn_id": turn_id},
        )
        record_intent_telemetry(intent_event)
        turn_intent_category = intent_event.intent_category

        def finalize_intent(success: bool, feedback: str, metadata: dict | None = None) -> int:
            latency_ms = int((time.perf_counter() - request_started_at) * 1000)
            intent_event.success = success
            intent_event.duration_ms = latency_ms
            intent_event.feedback = feedback
            if metadata:
                intent_event.metadata = {**(intent_event.metadata or {}), **metadata}
            record_intent_telemetry(intent_event)
            return latency_ms

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
            set_pending_action(None)
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
                set_latest_outcome(
                    turn_id=turn_id,
                    model="manual",
                    action="manual_code_blocked",
                    prompt_hash_value=prompt_hash(text),
                    intent_category=turn_intent_category,
                    predicted_route="manual_code",
                    actual_route="manual_code_blocked",
                    outcome_type="code_result",
                    success=False,
                )
                finalize_intent(False, "failed", {"response_action": "manual_code_blocked"})
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
            set_latest_outcome(
                turn_id=turn_id,
                model="manual",
                action="manual_code",
                prompt_hash_value=prompt_hash(text),
                intent_category=turn_intent_category,
                predicted_route="manual_code",
                actual_route="manual_code",
                outcome_type="code_result",
                success=True,
            )
            finalize_intent(True, "completed", {"response_action": "manual_code"})
            return
        code_repair_context = build_code_repair_context(text, viewer=viewer)
        followup_constraint = parse_followup_constraint(text)
        ui_help_reply = None
        if bool(ui_state.get("ui_help_enabled", False)):
            ui_help_reply = None if code_repair_context is not None else answer_ui_question(text)
        if ui_help_reply:
            set_pending_action(None)
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
            set_latest_outcome(
                turn_id=turn_id,
                model="local_ui_help",
                action="reply",
                prompt_hash_value=prompt_hash(text),
                intent_category=turn_intent_category,
                predicted_route="ui_help",
                actual_route="ui_help",
                outcome_type="reply",
                success=True,
            )
            finalize_intent(True, "completed", {"response_action": "reply", "model": "local_ui_help"})
            return
        recent_action_route = route_recent_action_followup(
            text,
            recent_action_state,
            selected_layer_name=str((selected_layer_profile() or {}).get("layer_name", "")).strip(),
            selected_layer_type=str((selected_layer_profile() or {}).get("layer_type", "")).strip(),
        )
        if isinstance(recent_action_route, dict) and recent_action_route.get("tool"):
            tool_name = str(recent_action_route.get("tool", "")).strip()
            prepared = prepare_tool_job(viewer, tool_name, recent_action_route.get("arguments", {}))
            set_pending_action(None)
            run_prepared_tool_request(
                prepared,
                tool_name=tool_name,
                tool_message=str(recent_action_route.get("message", "")).strip(),
                turn_id_value=turn_id,
            )
            append_log(f"Handled recent-action follow-up locally: {tool_name}")
            set_latest_outcome(
                turn_id=turn_id,
                model="local_recent_action_router",
                action="tool",
                prompt_hash_value=prompt_hash(text),
                intent_category=turn_intent_category,
                predicted_route="recent_action_followup",
                actual_route="tool",
                outcome_type="tool_result",
                tool_name=tool_name,
                success=True,
            )
            finalize_intent(True, "completed", {"response_action": "tool", "tool_name": tool_name})
            return
        threshold_followup_reply = reply_for_threshold_followup(text) if looks_like_threshold_followup(text) else ""
        if threshold_followup_reply:
            set_pending_action(None)
            append_chat_message("assistant", threshold_followup_reply)
            append_log("Answered threshold follow-up locally.")
            set_status("Status: threshold explanation ready", ok=True)
            record_telemetry(
                "turn_completed",
                {
                    "turn_id": turn_id,
                    "model": "local_threshold_explainer",
                    "base_url": "",
                    "prompt_hash": prompt_hash(text),
                    "prompt_category": "local_threshold_explanation",
                    "response_action": "reply",
                    "pending_code_generated": False,
                    **selected_layer_snapshot(),
                },
            )
            set_latest_outcome(
                turn_id=turn_id,
                model="local_threshold_explainer",
                action="reply",
                prompt_hash_value=prompt_hash(text),
                intent_category=turn_intent_category,
                predicted_route="local_explainer",
                actual_route="reply",
                outcome_type="reply",
                success=True,
            )
            finalize_intent(True, "completed", {"response_action": "reply", "model": "local_threshold_explainer"})
            return
        histogram_followup_reply = reply_for_histogram_followup(text) if looks_like_histogram_followup(text) else ""
        if histogram_followup_reply:
            set_pending_action(None)
            append_chat_message("assistant", histogram_followup_reply)
            append_log("Answered histogram follow-up locally.")
            set_status("Status: histogram explanation ready", ok=True)
            record_telemetry(
                "turn_completed",
                {
                    "turn_id": turn_id,
                    "model": "local_histogram_explainer",
                    "base_url": "",
                    "prompt_hash": prompt_hash(text),
                    "prompt_category": "local_histogram_explanation",
                    "response_action": "reply",
                    "pending_code_generated": False,
                    **selected_layer_snapshot(),
                },
            )
            set_latest_outcome(
                turn_id=turn_id,
                model="local_histogram_explainer",
                action="reply",
                prompt_hash_value=prompt_hash(text),
                intent_category=turn_intent_category,
                predicted_route="local_explainer",
                actual_route="reply",
                outcome_type="reply",
                success=True,
            )
            finalize_intent(True, "completed", {"response_action": "reply", "model": "local_histogram_explainer"})
            return
        intensity_summary_followup_reply = reply_for_intensity_summary_followup(text) if looks_like_intensity_summary_followup(text) else ""
        if intensity_summary_followup_reply:
            set_pending_action(None)
            append_chat_message("assistant", intensity_summary_followup_reply)
            append_log("Answered intensity-summary follow-up locally.")
            set_status("Status: intensity summary explanation ready", ok=True)
            record_telemetry(
                "turn_completed",
                {
                    "turn_id": turn_id,
                    "model": "local_intensity_summary_explainer",
                    "base_url": "",
                    "prompt_hash": prompt_hash(text),
                    "prompt_category": "local_intensity_summary_explanation",
                    "response_action": "reply",
                    "pending_code_generated": False,
                    **selected_layer_snapshot(),
                },
            )
            set_latest_outcome(
                turn_id=turn_id,
                model="local_intensity_summary_explainer",
                action="reply",
                prompt_hash_value=prompt_hash(text),
                intent_category=turn_intent_category,
                predicted_route="local_explainer",
                actual_route="reply",
                outcome_type="reply",
                success=True,
            )
            finalize_intent(True, "completed", {"response_action": "reply", "model": "local_intensity_summary_explainer"})
            return
        current_profile = selected_layer_profile() or {}
        selected_layer_name = str(current_profile.get("layer_name", "")).strip()
        available_names = image_layer_names()
        resumed_tool_request = resolve_pending_action(
            pending_action_state,
            user_text=text,
            selected_layer_name=selected_layer_name,
            available_layer_names=available_names,
        )
        if isinstance(resumed_tool_request, dict):
            tool_name = str(resumed_tool_request.get("tool", "")).strip()
            prepared = prepare_tool_job(viewer, tool_name, resumed_tool_request.get("arguments", {}))
            set_pending_action(complete_pending_action(pending_action_state))
            run_prepared_tool_request(
                prepared,
                tool_name=tool_name,
                tool_message=str(resumed_tool_request.get("tool_message", "")).strip(),
            )
            append_log(f"Resolved pending action and resumed tool: {tool_name}")
            return
        if is_pending_action_waiting(pending_action_state) and is_pending_action_cancel_message(text):
            set_pending_action(cancel_pending_action(pending_action_state))
            append_chat_message("assistant", "Cancelled the pending action. Tell me what you want to do next.")
            append_log("Cancelled pending action from user follow-up.")
            set_status("Status: pending action cancelled", ok=None)
            return
        if is_pending_action_waiting(pending_action_state):
            advanced_pending = advance_pending_action_turn(pending_action_state)
            set_pending_action(advanced_pending)
            if advanced_pending.get("status") == "expired":
                append_log("Expired pending action after unresolved follow-up turns.")
        base_url = base_url_edit.text().strip().rstrip("/") or str(saved_settings["base_url"]).rstrip("/")
        model_name = model_combo.currentText().strip() or str(saved_settings["model"]).strip()
        if not base_url or not model_name:
            append_chat_message("assistant", "Model settings are incomplete. Choose a model and open Connection if you need to adjust the Base URL.")
            set_status("Status: missing saved model settings", ok=False)
            finalize_intent(False, "failed", {"response_action": "configuration_error"})
            return
        local_workflow_route = None
        skip_local_workflow = should_skip_local_workflow_route(intent_state)
        if code_repair_context is not None:
            append_log("Local workflow route skipped: code repair context is active for this turn.")
        elif skip_local_workflow:
            append_log(
                "Local workflow route skipped by intent-state guard: "
                f"mode_preference={str(intent_state.get('mode_preference', '')).strip()!r} "
                f"negative_constraints={list(intent_state.get('negative_constraints', []) or [])!r}"
            )
        else:
            local_workflow_route = route_local_workflow_prompt(text, selected_layer_profile())
            if isinstance(local_workflow_route, dict):
                append_log(
                    "Local workflow route matched: "
                    f"action={str(local_workflow_route.get('action', '')).strip().lower()!r}"
                )
            else:
                append_log("Local workflow route did not match this message; falling back to model/tool path.")
        if isinstance(local_workflow_route, dict):
            nonlocal last_workflow_plan_payload, last_workflow_execution_payload
            set_pending_action(None)
            route_action = str(local_workflow_route.get("action", "")).strip().lower()
            if route_action == "reply":
                reply_message = str(local_workflow_route.get("message", "")).strip() or "I could not resolve that request."
                append_chat_message("assistant", f"{local_workflow_marker} {reply_message}")
                append_log("Handled request via local workflow route: reply")
                set_status("Status: local workflow reply completed", ok=True)
                record_telemetry(
                    "turn_completed",
                    {
                        "turn_id": turn_id,
                        "model": "local_workflow_router",
                        "base_url": "",
                        "prompt_hash": prompt_hash(text),
                        "prompt_category": "local_workflow_route",
                        "response_action": "reply",
                        "tool_name": "",
                        "tool_success": True,
                        "pending_code_generated": False,
                        **selected_layer_snapshot(),
                    },
                )
                set_latest_outcome(
                    turn_id=turn_id,
                    model="local_workflow_router",
                    action="reply",
                    prompt_hash_value=prompt_hash(text),
                    intent_category=turn_intent_category,
                    predicted_route="local_workflow",
                    actual_route="reply",
                    outcome_type="reply",
                    success=True,
                )
                finalize_intent(True, "completed", {"response_action": "reply", "model": "local_workflow_router"})
                return
            if route_action == "workflow_plan":
                route_message = str(local_workflow_route.get("message", "")).strip()
                plan_payload = local_workflow_route.get("plan", {})
                last_workflow_plan_payload = dict(plan_payload or {})
                plan_text = workflow_plan_to_markdown(plan_payload)
                message = f"{route_message}\n\n{plan_text}" if route_message else plan_text
                append_chat_message("assistant", f"{local_workflow_marker}\n{message}")
                append_log("Handled request via local workflow route: workflow_plan")
                set_status("Status: workflow plan created", ok=True)
                record_telemetry(
                    "turn_completed",
                    {
                        "turn_id": turn_id,
                        "model": "local_workflow_router",
                        "base_url": "",
                        "prompt_hash": prompt_hash(text),
                        "prompt_category": "local_workflow_route",
                        "response_action": "workflow_plan",
                        "tool_name": "plan_conservative_binary_segmentation",
                        "tool_success": True,
                        "pending_code_generated": False,
                        **selected_layer_snapshot(),
                    },
                )
                set_latest_outcome(
                    turn_id=turn_id,
                    model="local_workflow_router",
                    action="workflow_plan",
                    prompt_hash_value=prompt_hash(text),
                    intent_category=turn_intent_category,
                    predicted_route="local_workflow",
                    actual_route="workflow_plan",
                    outcome_type="workflow_plan",
                    tool_name="plan_conservative_binary_segmentation",
                    success=True,
                )
                finalize_intent(True, "completed", {"response_action": "workflow_plan", "model": "local_workflow_router"})
                return
            if route_action == "workflow_report":
                mode = str(local_workflow_route.get("mode", "")).strip().lower() or "compact"
                if not last_workflow_plan_payload and not last_workflow_execution_payload:
                    append_chat_message("assistant", "No workflow report is available yet.")
                    append_log("Workflow report request skipped: no saved workflow state.")
                    set_status("Status: no workflow report available", ok=False)
                    return
                route_message = str(local_workflow_route.get("message", "")).strip()
                if mode == "plan":
                    report_text = workflow_plan_to_markdown(last_workflow_plan_payload)
                elif mode == "debug":
                    report_text = workflow_execution_to_debug_markdown(last_workflow_execution_payload)
                elif mode == "details":
                    plan_text = workflow_plan_to_markdown(last_workflow_plan_payload)
                    debug_text = workflow_execution_to_debug_markdown(last_workflow_execution_payload)
                    report_text = "\n\n".join(part for part in (plan_text, debug_text) if part)
                else:
                    report_text = workflow_execution_to_compact_markdown(last_workflow_execution_payload)
                message = f"{route_message}\n\n{report_text}" if route_message else report_text
                append_chat_message("assistant", f"{local_workflow_marker}\n{message}")
                append_log(f"Handled request via local workflow route: workflow_report[{mode}]")
                set_status("Status: workflow report shown", ok=True)
                set_latest_outcome(
                    turn_id=turn_id,
                    model="local_workflow_router",
                    action="workflow_report",
                    prompt_hash_value=prompt_hash(text),
                    intent_category=turn_intent_category,
                    predicted_route="local_workflow",
                    actual_route="workflow_report",
                    outcome_type="workflow_report",
                    success=True,
                )
                finalize_intent(True, "completed", {"response_action": "workflow_report", "mode": mode, "model": "local_workflow_router"})
                return
            if route_action == "workflow_execute":
                route_message = str(local_workflow_route.get("message", "")).strip()
                plan_payload = local_workflow_route.get("plan", {})
                last_workflow_plan_payload = dict(plan_payload or {})
                execution_result = execute_workflow_plan(viewer, plan_payload)
                last_workflow_execution_payload = dict(execution_result or {})
                execution_text = workflow_execution_to_compact_markdown(execution_result)
                message = f"{route_message}\n\n{execution_text}" if route_message else execution_text
                append_chat_message("assistant", f"{local_workflow_marker}\n{message}")
                append_log("Handled request via local workflow route: workflow_execute")
                set_status("Status: workflow executed", ok=bool(execution_result.get("ok", False)))
                record_telemetry(
                    "turn_completed",
                    {
                        "turn_id": turn_id,
                        "model": "local_workflow_router",
                        "base_url": "",
                        "prompt_hash": prompt_hash(text),
                        "prompt_category": "local_workflow_route",
                        "response_action": "workflow_execute",
                        "tool_name": "execute_workflow_plan",
                        "tool_success": bool(execution_result.get("ok", False)),
                        "pending_code_generated": False,
                        **selected_layer_snapshot(),
                    },
                )
                set_latest_outcome(
                    turn_id=turn_id,
                    model="local_workflow_router",
                    action="workflow_execute",
                    prompt_hash_value=prompt_hash(text),
                    intent_category=turn_intent_category,
                    predicted_route="local_workflow",
                    actual_route="workflow_execute",
                    outcome_type="workflow_execute",
                    tool_name="execute_workflow_plan",
                    success=bool(execution_result.get("ok", False)),
                )
                finalize_intent(bool(execution_result.get("ok", False)), "completed", {"response_action": "workflow_execute", "model": "local_workflow_router"})
                return
            if route_action == "tool_sequence":
                nonlocal last_tool_sequence_undo_snapshot
                sequence_message = str(local_workflow_route.get("message", "")).strip()
                steps = local_workflow_route.get("steps", [])
                sequence_result = run_tool_sequence(viewer, steps if isinstance(steps, list) else [])
                undo_snapshot = sequence_result.get("undo_snapshot")
                if isinstance(undo_snapshot, dict) and sequence_result.get("completed", 0):
                    last_tool_sequence_undo_snapshot = undo_snapshot
                result_message = str(sequence_result.get("message", "")).strip() or "Workflow sequence did not return a result."
                if sequence_result.get("completed", 0):
                    result_message = f"{result_message}\nYou can say `undo last workflow` to restore the previous viewer controls."
                message = f"{sequence_message}\n{result_message}" if sequence_message else result_message
                append_chat_message("assistant", f"{local_workflow_marker}\n{message}")
                append_log("Handled request via local workflow route: tool_sequence")
                set_status("Status: workflow sequence completed", ok=bool(sequence_result.get("completed", 0)))
                record_telemetry(
                    "turn_completed",
                    {
                        "turn_id": turn_id,
                        "model": "local_workflow_router",
                        "base_url": "",
                        "prompt_hash": prompt_hash(text),
                        "prompt_category": "local_workflow_route",
                        "response_action": "tool_sequence",
                        "tool_name": "run_tool_sequence",
                        "tool_success": True,
                        "pending_code_generated": False,
                        **selected_layer_snapshot(),
                    },
                )
                set_latest_outcome(
                    turn_id=turn_id,
                    model="local_workflow_router",
                    action="tool_sequence",
                    prompt_hash_value=prompt_hash(text),
                    intent_category=turn_intent_category,
                    predicted_route="local_workflow",
                    actual_route="tool_sequence",
                    outcome_type="tool_sequence",
                    tool_name="run_tool_sequence",
                    success=True,
                )
                finalize_intent(True, "completed", {"response_action": "tool_sequence", "model": "local_workflow_router"})
                return
            if route_action == "restore_tool_sequence":
                if not last_tool_sequence_undo_snapshot:
                    append_chat_message("assistant", "No quick-control workflow undo state is available yet.")
                    append_log("Restore quick-control workflow skipped: no undo snapshot.")
                    set_status("Status: no workflow undo state", ok=False)
                    return
                restore_message = restore_viewer_control_snapshot(viewer, last_tool_sequence_undo_snapshot)
                last_tool_sequence_undo_snapshot = {}
                route_message = str(local_workflow_route.get("message", "")).strip()
                message = f"{route_message}\n{restore_message}" if route_message else restore_message
                append_chat_message("assistant", f"{local_workflow_marker}\n{message}")
                append_log("Handled request via local workflow route: restore_tool_sequence")
                set_status("Status: workflow controls restored", ok=True)
                record_telemetry(
                    "turn_completed",
                    {
                        "turn_id": turn_id,
                        "model": "local_workflow_router",
                        "base_url": "",
                        "prompt_hash": prompt_hash(text),
                        "prompt_category": "local_workflow_route",
                        "response_action": "restore_tool_sequence",
                        "tool_name": "restore_viewer_control_snapshot",
                        "tool_success": True,
                        "pending_code_generated": False,
                        **selected_layer_snapshot(),
                    },
                )
                set_latest_outcome(
                    turn_id=turn_id,
                    model="local_workflow_router",
                    action="restore_tool_sequence",
                    prompt_hash_value=prompt_hash(text),
                    intent_category=turn_intent_category,
                    predicted_route="local_workflow",
                    actual_route="restore_tool_sequence",
                    outcome_type="workflow_restore",
                    tool_name="restore_viewer_control_snapshot",
                    success=True,
                )
                finalize_intent(True, "completed", {"response_action": "restore_tool_sequence", "model": "local_workflow_router"})
                return
            tool_name = str(local_workflow_route.get("tool", "")).strip()
            if should_block_tool(intent_state, tool_name):
                local_workflow_route = None
            else:
                arguments = local_workflow_route.get("arguments", {})
                tool_message = str(local_workflow_route.get("message", "")).strip()
                prepared = prepare_tool_job(viewer, tool_name, arguments if isinstance(arguments, dict) else {})
                if prepared.get("mode") == "immediate":
                    result_message = str(prepared.get("message", "")).strip() or f"Could not run [{tool_name}]."
                    append_chat_message("assistant", f"{tool_message}\n{result_message}" if tool_message else result_message)
                    append_log(f"Handled request via local workflow route: {tool_name}")
                    set_status(f"Status: {tool_name} completed", ok=True)
                    remember_recent_action(
                        tool_name=tool_name,
                        turn_id_value=turn_id,
                        message=f"{tool_message}\n{result_message}" if tool_message else result_message,
                    )
                    last_failed_tool_state = remember_failed_tool(tool_name, result_message)
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
                    set_latest_outcome(
                        turn_id=turn_id,
                        model="local_workflow_router",
                        action="tool",
                        prompt_hash_value=prompt_hash(text),
                        intent_category=turn_intent_category,
                        predicted_route="local_workflow",
                        actual_route="tool",
                        outcome_type="tool_result",
                        tool_name=tool_name,
                        success=True,
                    )
                    finalize_intent(True, "completed", {"response_action": "tool", "tool_name": tool_name, "model": "local_workflow_router"})
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
                        nonlocal last_failed_tool_state
                        refresh_context()
                        result_message = apply_tool_job_result(viewer, tool_result)
                        result_message = maybe_note_selected_only_visibility(tool_name, result_message)
                        append_chat_message("assistant", f"{tool_message}\n{result_message}" if tool_message else result_message)
                        append_log(f"Handled request via local workflow route: {tool_name}")
                        set_status(f"Status: {tool_name} completed", ok=True)
                        remember_recent_action(
                            tool_name=tool_name,
                            turn_id_value=turn_id,
                            message=f"{tool_message}\n{result_message}" if tool_message else result_message,
                            tool_result=tool_result,
                        )
                        last_failed_tool_state = remember_failed_tool(tool_name, result_message)
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
                        set_latest_outcome(
                            turn_id=turn_id,
                            model="local_workflow_router",
                            action="tool",
                            prompt_hash_value=prompt_hash(text),
                            intent_category=turn_intent_category,
                            predicted_route="local_workflow",
                            actual_route="tool",
                            outcome_type="tool_result",
                            tool_name=tool_name,
                            success=True,
                        )
                        finalize_intent(True, "completed", {"response_action": "tool", "tool_name": tool_name, "model": "local_workflow_router"})
                        local_tool_finish()

                    def local_tool_error(*args):
                        nonlocal last_failed_tool_state
                        error_text = format_worker_error(*args)
                        logger.exception("Local workflow route failed: %s | %s", tool_name, error_text)
                        append_chat_message("assistant", f"{tool_name} failed:\n{error_text}")
                        append_log(f"Local workflow route failed: {tool_name} | {error_text}")
                        set_status(f"Status: {tool_name} failed", ok=False)
                        last_failed_tool_state = remember_failed_tool(tool_name, error_text)
                        set_latest_outcome(
                            turn_id=turn_id,
                            model="local_workflow_router",
                            action="tool",
                            prompt_hash_value=prompt_hash(text),
                            intent_category=turn_intent_category,
                            predicted_route="local_workflow",
                            actual_route="tool",
                            outcome_type="tool_result",
                            tool_name=tool_name,
                            success=False,
                        )
                        finalize_intent(False, "failed", {"response_action": "tool", "tool_name": tool_name, "error": error_text, "model": "local_workflow_router"})
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
        response_holder: dict = {}
        request_metadata_holder: dict = {}
        response_metadata_holder: dict = {}
        stop_event = threading.Event()

        def model_request_performance_payload(reply_text: str = "") -> dict:
            output_text = str(reply_text or "")
            payload = {
                **dict(request_metadata_holder),
                **dict(response_metadata_holder),
            }
            if output_text:
                payload.update(
                    {
                        "output_chars": len(output_text),
                        "output_bytes": len(output_text.encode("utf-8")),
                        "estimated_output_tokens": max(1, round(len(output_text) / 4)),
                    }
                )
            return payload

        def append_model_performance_log(metrics: dict) -> None:
            prompt_tokens = metrics.get("prompt_eval_count") or metrics.get("estimated_input_tokens")
            prompt_tps = metrics.get("prompt_eval_tokens_per_second")
            generation_tps = metrics.get("generation_tokens_per_second")
            parts = [
                f"input≈{metrics.get('estimated_input_tokens', 0)} tokens",
            ]
            if prompt_tokens:
                parts.append(f"prompt_eval={prompt_tokens} tokens")
            if prompt_tps:
                parts.append(f"prompt_eval={float(prompt_tps):.1f} tok/s")
            if generation_tps:
                parts.append(f"generation={float(generation_tps):.1f} tok/s")
            total_ms = metrics.get("total_duration_ms")
            if total_ms:
                parts.append(f"ollama_total={float(total_ms):.0f} ms")
            append_log("Model telemetry: " + " | ".join(parts))

        @thread_worker(ignore_errors=True)
        def run_chat():
            viewer_payload = layer_context_for_model(viewer)
            return chat_ollama(
                base_url,
                model_name,
                system_prompt=assistant_system_prompt(),
                user_payload={
                    "viewer_context": viewer_payload,
                    "session_memory": build_session_memory_payload(session_memory_state, viewer_payload.get("selected_layer_profile")),
                    "intent_state": intent_state,
                    "last_failed_tool_state": last_failed_tool_state,
                    "followup_constraint": followup_constraint,
                    "code_repair_context": code_repair_context,
                    "user_message": text,
                },
                options=dict(generation_defaults),
                timeout=1800,
                response_holder=response_holder,
                request_metadata_holder=request_metadata_holder,
                response_metadata_holder=response_metadata_holder,
                stop_event=stop_event,
            )

        worker = run_chat()
        def cancel_chat_request() -> None:
            stop_event.set()
            response = response_holder.get("response")
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

        setattr(worker, "_assistant_cancel", cancel_chat_request)
        active_workers.append(worker)
        refresh_stop_button()

        def finish():
            stop_wait_indicator()
            if worker in active_workers:
                active_workers.remove(worker)
            refresh_stop_button()

        def on_returned(reply):
            nonlocal last_failed_tool_state
            if stop_event.is_set():
                finish()
                return
            model_perf_payload = model_request_performance_payload(reply)
            append_model_performance_log(model_perf_payload)
            try:
                parsed = normalize_model_response(reply)
            except Exception as exc:
                logger.exception("Failed to normalize model response.")
                replace_last_assistant(f"Response parse failed:\n{exc}\n\nRaw reply:\n{reply}")
                set_status("Status: response parse failed", ok=False)
                append_log("Failed to parse model response.")
                set_latest_outcome(
                    turn_id=turn_id,
                    model=model_name,
                    action="error",
                    prompt_hash_value=turn_prompt_hash,
                    intent_category=turn_intent_category,
                    predicted_route="model_request",
                    actual_route="error",
                    outcome_type="error",
                    success=False,
                )
                record_telemetry(
                    "turn_completed",
                    {
                        "turn_id": turn_id,
                        "model": model_name,
                        "base_url": base_url,
                        "prompt_hash": turn_prompt_hash,
                        "prompt_category": turn_prompt_category,
                        "response_action": "parse_error",
                        "latency_ms": int((time.perf_counter() - request_started_at) * 1000),
                        "error": str(exc),
                        **model_perf_payload,
                        **turn_layer_snapshot,
                    },
                )
                finish()
                return

            action = str(parsed.get("action", "reply")).strip().lower()

            if action == "tool":
                set_pending_action(None)
                tool_name = str(parsed.get("tool", "")).strip()
                if should_block_tool(intent_state, tool_name):
                    replace_last_assistant(
                        f"Blocked built-in tool [{tool_name}] because it conflicts with your current request constraints. "
                        "Please ask for custom code or restate the request if you want the built-in tool anyway."
                    )
                    set_status("Status: blocked incompatible built-in tool", ok=False)
                    append_log(f"Blocked tool by intent-state guard: {tool_name}")
                    set_latest_outcome(
                        turn_id=turn_id,
                        model=model_name,
                        action="tool_blocked",
                        prompt_hash_value=turn_prompt_hash,
                        intent_category=turn_intent_category,
                        predicted_route="model_tool",
                        actual_route="blocked",
                        outcome_type="tool_result",
                        tool_name=tool_name,
                        success=False,
                    )
                    finalize_intent(False, "failed", {"response_action": "tool_blocked", "tool_name": tool_name})
                    finish()
                    return
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
                        result_message = maybe_note_selected_only_visibility(tool_name, result_message)
                    except Exception as exc:
                        logger.exception("Immediate tool failed: %s", tool_name)
                        replace_last_assistant(f"{tool_name} failed:\n{exc}")
                        set_status(f"Status: {tool_name} failed", ok=False)
                        append_log(f"Immediate tool failed: {tool_name} | {exc}")
                        set_latest_outcome(
                            turn_id=turn_id,
                            model=model_name,
                            action="tool",
                            prompt_hash_value=turn_prompt_hash,
                            intent_category=turn_intent_category,
                            predicted_route="model_tool",
                            actual_route="tool",
                            outcome_type="tool_result",
                            tool_name=tool_name,
                            success=False,
                        )
                        finalize_intent(False, "failed", {"response_action": "tool", "tool_name": tool_name, "error": str(exc), "model": model_name})
                        finish()
                        return
                    refresh_context()
                    replace_last_assistant(f"{tool_message}\n{result_message}" if tool_message else result_message)
                    last_failed_tool_state = remember_failed_tool(tool_name, result_message)
                    latency_ms = finalize_intent(True, "completed", {"response_action": "tool", "tool_name": tool_name, "model": model_name})
                    set_latest_outcome(
                        turn_id=turn_id,
                        model=model_name,
                        action="tool",
                        prompt_hash_value=turn_prompt_hash,
                        intent_category=turn_intent_category,
                        predicted_route="model_tool",
                        actual_route="tool",
                        outcome_type="tool_result",
                        tool_name=tool_name,
                        success=True,
                        latency_ms=latency_ms,
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
                            **model_perf_payload,
                            **turn_layer_snapshot,
                        },
                    )
                    remember_recent_action(
                        tool_name=tool_name,
                        turn_id_value=turn_id,
                        message=f"{tool_message}\n{result_message}" if tool_message else result_message,
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
                    nonlocal session_memory_state, last_failed_tool_state
                    result_message = apply_tool_job_result(viewer, tool_result)
                    result_message = maybe_note_selected_only_visibility(tool_name, result_message)
                    refresh_context()
                    replace_last_assistant(f"{tool_message}\n{result_message}" if tool_message else result_message)
                    last_failed_tool_state = remember_failed_tool(tool_name, result_message)
                    latency_ms = finalize_intent(True, "completed", {"response_action": "tool", "tool_name": tool_name, "model": model_name})
                    set_latest_outcome(
                        turn_id=turn_id,
                        model=model_name,
                        action="tool",
                        prompt_hash_value=turn_prompt_hash,
                        intent_category=turn_intent_category,
                        predicted_route="model_tool",
                        actual_route="tool",
                        outcome_type="tool_result",
                        tool_name=tool_name,
                        success=True,
                        latency_ms=latency_ms,
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
                            **model_perf_payload,
                            **turn_layer_snapshot,
                        },
                    )
                    session_memory_state = approve_items(session_memory_state, last_memory_candidate_ids)
                    persist_session_memory()
                    remember_recent_action(
                        tool_name=tool_name,
                        turn_id_value=turn_id,
                        message=f"{tool_message}\n{result_message}" if tool_message else result_message,
                        tool_result=tool_result,
                    )
                    remember_assistant_outcome(
                        tool_message or result_message,
                        target_type="tool_result",
                        target_profile=selected_layer_profile(),
                        state="approved",
                    )
                    append_log(f"Tool executed: {tool_name}")
                    tool_finish()

                def tool_error(*args):
                    nonlocal last_failed_tool_state
                    error_text = format_worker_error(*args)
                    logger.exception("Tool execution failed: %s | %s", tool_name, error_text)
                    replace_last_assistant(f"Tool execution failed: {error_text}")
                    last_failed_tool_state = remember_failed_tool(tool_name, error_text)
                    latency_ms = finalize_intent(False, "failed", {"response_action": "tool", "tool_name": tool_name, "error": error_text, "model": model_name})
                    set_latest_outcome(
                        turn_id=turn_id,
                        model=model_name,
                        action="tool",
                        prompt_hash_value=turn_prompt_hash,
                        intent_category=turn_intent_category,
                        predicted_route="model_tool",
                        actual_route="tool",
                        outcome_type="tool_result",
                        tool_name=tool_name,
                        success=False,
                        latency_ms=latency_ms,
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
                            **model_perf_payload,
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
                set_pending_action(None)
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
                    set_latest_outcome(
                        turn_id=turn_id,
                        model=model_name,
                        action="code_rejected",
                        prompt_hash_value=turn_prompt_hash,
                        intent_category=turn_intent_category,
                        predicted_route="model_code",
                        actual_route="code_rejected",
                        outcome_type="code_result",
                        success=False,
                    )
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
                latency_ms = finalize_intent(True, "completed", {"response_action": "code", "model": model_name})
                set_latest_outcome(
                    turn_id=turn_id,
                    model=model_name,
                    action="code",
                    prompt_hash_value=turn_prompt_hash,
                    intent_category=turn_intent_category,
                    predicted_route="model_code",
                    actual_route="code",
                    outcome_type="code_result",
                    success=True,
                    latency_ms=latency_ms,
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
                        **model_perf_payload,
                        **turn_layer_snapshot,
                    },
                )
                remember_assistant_outcome(code_message, target_type="recommendation", target_profile=selected_layer_profile())
                set_status("Status: generated code ready for review", ok=None)
                append_log("Generated pending napari code; waiting for approval.")
                finish()
                return

            message_text = str(parsed.get("message", reply)).strip() or "[empty response]"
            set_pending_action(
                build_pending_action_from_assistant_message(
                    message_text,
                    turn_id=turn_id,
                    selected_layer_name=str((selected_layer_profile() or {}).get("layer_name", "")).strip(),
                )
            )
            replace_last_assistant(message_text)
            latency_ms = finalize_intent(True, "completed", {"response_action": "reply", "model": model_name})
            set_latest_outcome(
                turn_id=turn_id,
                model=model_name,
                action="reply",
                prompt_hash_value=turn_prompt_hash,
                intent_category=turn_intent_category,
                predicted_route="model_reply",
                actual_route="reply",
                outcome_type="reply",
                success=True,
                latency_ms=latency_ms,
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
                    **model_perf_payload,
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
            if stop_event.is_set():
                finish()
                return
            error_text = format_worker_error(*args)
            logger.exception("Chat request failed: %s", error_text)
            replace_last_assistant(f"Request failed: {error_text}")
            latency_ms = finalize_intent(False, "failed", {"response_action": "error", "error": error_text, "model": model_name})
            set_latest_outcome(
                turn_id=turn_id,
                model=model_name,
                action="error",
                prompt_hash_value=turn_prompt_hash,
                intent_category=turn_intent_category,
                predicted_route="model_request",
                actual_route="error",
                outcome_type="error",
                success=False,
                latency_ms=latency_ms,
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
                    **model_request_performance_payload(""),
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
            set_latest_outcome(
                turn_id=turn_id or last_turn_metrics.get("turn_id", ""),
                model=model_name or last_turn_metrics.get("model", ""),
                action="code_execution",
                prompt_hash_value=last_turn_metrics.get("prompt_hash", ""),
                intent_category=last_turn_metrics.get("intent_category", ""),
                predicted_route="code_execution",
                actual_route="code_execution",
                outcome_type="code_result",
                success=False,
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
        set_latest_outcome(
            turn_id=turn_id or last_turn_metrics.get("turn_id", ""),
            model=model_name or last_turn_metrics.get("model", ""),
            action="code_execution",
            prompt_hash_value=last_turn_metrics.get("prompt_hash", ""),
            intent_category=last_turn_metrics.get("intent_category", ""),
            predicted_route="code_execution",
            actual_route="code_execution",
            outcome_type="code_result",
            success=True,
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
            "If the code contains placeholder or template layer names, replace them with the best matching current viewer layers automatically.",
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
    feedback_helpful_action.triggered.connect(lambda *_args: submit_feedback("helpful"))
    feedback_wrong_route_action.triggered.connect(lambda *_args: submit_feedback("wrong_route"))
    feedback_wrong_answer_action.triggered.connect(lambda *_args: submit_feedback("wrong_answer"))
    feedback_didnt_work_action.triggered.connect(lambda *_args: submit_feedback("didnt_work"))
    stop_btn.clicked.connect(stop_active_request)
    voice_input_action.triggered.connect(show_voice_input)
    sam2_setup_action.triggered.connect(show_sam2_setup_dialog)
    sam2_live_action.triggered.connect(show_sam2_live_dialog)
    text_annotation_action.triggered.connect(show_text_annotation_editor)
    atlas_stitch_action.triggered.connect(show_atlas_stitch)
    help_whats_new_action.triggered.connect(show_whats_new)
    help_about_action.triggered.connect(show_about_assistant)
    help_report_bug_action.triggered.connect(show_report_bug)
    telemetry_summary_btn.clicked.connect(show_telemetry_summary)
    telemetry_report_btn.clicked.connect(show_telemetry_report)
    telemetry_view_btn.clicked.connect(show_telemetry_viewer)
    telemetry_reset_btn.clicked.connect(reset_telemetry_log)
    telemetry_toggle.toggled.connect(toggle_telemetry)
    summary_stats_btn.clicked.connect(run_summary_stats)
    histogram_btn.clicked.connect(run_histogram)
    compare_btn.clicked.connect(run_layer_comparison)
    library_tabs.currentChanged.connect(show_library_tab_index)
    library_tabs.tabBarClicked.connect(show_library_tab_index)
    actions_tab_btn.clicked.connect(lambda *_args: show_library_panel("action"))
    prompt_library_list.itemClicked.connect(load_library_prompt)
    prompt_library_list.itemDoubleClicked.connect(send_library_prompt)
    prompt_library_list.customContextMenuRequested.connect(show_library_context_menu)
    code_library_list.itemClicked.connect(load_library_code)
    code_library_list.itemDoubleClicked.connect(run_library_code)
    context_layers_list.itemClicked.connect(select_context_layer_item)
    context_selected_only_checkbox.toggled.connect(toggle_context_selected_only)
    prompt.textChanged.connect(refresh_code_action_buttons)
    code_library_list.customContextMenuRequested.connect(show_library_context_menu)
    template_tree.currentItemChanged.connect(show_template_preview)
    template_tree.itemDoubleClicked.connect(run_template_tree_item)
    template_load_btn.clicked.connect(load_selected_template)
    template_run_btn.clicked.connect(run_selected_template)
    action_tree.currentItemChanged.connect(show_action_preview)
    action_load_btn.clicked.connect(load_selected_action)
    action_run_btn.clicked.connect(run_selected_action)
    action_add_shortcut_btn.clicked.connect(add_selected_action_to_shortcuts)
    shortcuts_add_row_btn.clicked.connect(add_shortcut_row)
    shortcuts_remove_row_btn.clicked.connect(remove_shortcut_row)
    shortcuts_save_btn.clicked.connect(lambda _checked=False: save_shortcuts_layout(choose_path=True))
    shortcuts_load_btn.clicked.connect(lambda _checked=False: load_shortcuts_layout(choose_path=True))
    shortcuts_clear_btn.clicked.connect(clear_shortcuts)
    save_prompt_btn.clicked.connect(save_current_prompt)
    pin_prompt_btn.clicked.connect(toggle_pin_selected_prompt)
    delete_prompt_btn.clicked.connect(delete_selected_prompt)
    clear_prompt_btn.clicked.connect(clear_non_saved_prompts)
    prompt_font_down_btn.clicked.connect(lambda *_args: adjust_prompt_library_font(-1))
    prompt_font_up_btn.clicked.connect(lambda *_args: adjust_prompt_library_font(1))
    chat_font_down_btn.clicked.connect(lambda *_args: adjust_chat_font(-1))
    chat_font_up_btn.clicked.connect(lambda *_args: adjust_chat_font(1))
    prompt.sendRequested.connect(send_message)
    connection_toggle_btn.toggled.connect(connection_details.setVisible)
    shortcuts_group.toggled.connect(toggle_shortcuts_group)
    log_group.toggled.connect(log_tabs.setVisible)
    help_ui_toggle_action.toggled.connect(toggle_ui_help)
    wait_timer.timeout.connect(tick_wait_indicator)
    root.destroyed.connect(cleanup_workers)

    connect_viewer_context_events()
    refresh_telemetry_controls()
    refresh_feedback_controls()
    refresh_stop_button()
    refresh_context()
    set_pending_code()
    refresh_code_action_buttons()
    refresh_models()
    refresh_sam2_actions()
    refresh_prompt_library()
    show_library_panel("prompt")
    append_log(f"Assistant log: {APP_LOG_PATH}")
    append_log(f"Crash log: {CRASH_LOG_PATH}")
    if ui_state.get("telemetry_enabled", False):
        append_log(f"Telemetry log: {TELEMETRY_LOG_PATH}")
    append_log(f"Prompt library path: {prompt_library_path()}")
    append_log(f"Session memory path: {session_memory_path()}")
    last_seen_version = str(ui_state.get("last_seen_version", "")).strip()
    if not ui_state.get("welcome_dismissed", False):
        append_chat_message(
            "assistant",
            "**Welcome**\n"
            "Local Chat Assistant connects napari to a local Ollama model, deterministic image-analysis tools, reusable templates, and generated viewer-bound Python.\n\n"
            "**Start here**\n"
            "- Select a model, then click `Load` to warm it before the first request.\n"
            "- Open an image or use `Templates` -> `Data` to create a synthetic test image.\n"
            "- Use `Layer Context` -> `Layers` to select a layer, insert or copy exact layer names, copy layer info, or show only the selected layer.\n"
            "- Use `Templates` and `Actions` for repeatable workflows; add frequent actions to `Shortcuts`.\n"
            "- Paste your own Python into the Prompt box and click `Run My Code`, or use `Refine My Code` when code needs repair for the current viewer.\n\n"
            "**Try these prompts**\n"
            "```text\n"
            "What can you do with my current layers?\n"
            "```\n\n"
            "```text\n"
            "1. Fit visible layers to view.\n"
            "2. Show viewer axes.\n"
            "3. Show scale bar.\n"
            "4. Show selected layer bounding box.\n"
            "5. Show selected layer name overlay.\n"
            "```\n\n"
            "Say `undo last workflow` to restore viewer controls after a quick-control workflow.\n\n"
            "Telemetry is optional and stays off unless you enable it from `Session` -> `Telemetry`.\n\n"
            "Updates: https://github.com/wulinteousa2-hash/napari-chat-assistant/blob/main/CHANGELOG.md\n"
            "Bug reports: https://github.com/wulinteousa2-hash/napari-chat-assistant/issues",
        )
        append_chat_message("assistant", whats_new_message(__version__))
        ui_state["welcome_dismissed"] = True
        ui_state["last_seen_version"] = __version__
        save_ui_state(ui_state)
    elif last_seen_version != __version__:
        append_chat_message("assistant", whats_new_message(__version__))
        ui_state["last_seen_version"] = __version__
        save_ui_state(ui_state)
    append_log("Assistant panel initialized.")
    return root
