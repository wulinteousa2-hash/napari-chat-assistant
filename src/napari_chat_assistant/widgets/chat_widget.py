from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import napari
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QTextCursor
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QListWidgetItem,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_chat_assistant.agent.client import chat_ollama, list_ollama_models, unload_ollama_model
from napari_chat_assistant.agent.context import get_viewer, layer_context_json, layer_summary
from napari_chat_assistant.agent.dispatcher import apply_tool_job_result, prepare_tool_job, run_tool_job
from napari_chat_assistant.agent.prompt_library import (
    load_prompt_library,
    merged_prompt_records,
    prompt_library_path,
    remove_saved_prompt,
    save_prompt_library,
    set_saved_prompt_pinned,
    upsert_recent_prompt,
    upsert_saved_prompt,
)
from napari_chat_assistant.agent.tools import ASSISTANT_TOOL_NAMES, assistant_system_prompt


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

    root = QWidget()
    layout = QVBoxLayout(root)

    header = QLabel("Local Chat Assistant")
    layout.addWidget(header)

    splitter = QSplitter(Qt.Horizontal)
    layout.addWidget(splitter, 1)

    left_panel = QWidget()
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
    config_layout.addRow("Model:", model_combo)

    model_hint = QLabel("Pick from local Ollama models or type a model tag manually. Use Unload Model before switching if you want to free memory.")
    model_hint.setWordWrap(True)
    model_hint.setStyleSheet("QLabel { color: #9fb3c8; padding: 2px 0 6px 0; }")
    config_layout.addRow(model_hint)

    connection_status = QLabel("Status: not connected")
    connection_status.setStyleSheet("QLabel { background: #243447; color: #f5f7fa; padding: 6px; }")
    config_layout.addRow(connection_status)

    config_btn_row = QWidget()
    config_btn_layout = QHBoxLayout(config_btn_row)
    save_btn = QPushButton("Use Selected Model")
    test_btn = QPushButton("Test Connection")
    unload_btn = QPushButton("Unload Model")
    config_btn_layout.addWidget(save_btn)
    config_btn_layout.addWidget(test_btn)
    config_btn_layout.addWidget(unload_btn)
    pull_btn = QPushButton("Model Help")
    config_btn_layout.addWidget(pull_btn)
    config_layout.addRow(config_btn_row)
    left_layout.addWidget(config_group)

    context_group = QGroupBox("Current Context")
    context_layout = QVBoxLayout(context_group)
    context_label = QLabel()
    context_label.setTextInteractionFlags(context_label.textInteractionFlags())
    context_label.setStyleSheet("QLabel { background: #101820; color: #e6edf3; padding: 8px; }")
    context_layout.addWidget(context_label)

    context_btn_row = QWidget()
    context_btn_layout = QHBoxLayout(context_btn_row)
    refresh_btn = QPushButton("Refresh Context")
    context_btn_layout.addWidget(refresh_btn)
    context_layout.addWidget(context_btn_row)
    left_layout.addWidget(context_group, 1)

    prompt_library_group = QGroupBox("Prompt Library")
    prompt_library_layout = QVBoxLayout(prompt_library_group)
    prompt_library_hint = QLabel("Click to load. Double-click to run.")
    prompt_library_hint.setWordWrap(True)
    prompt_library_hint.setStyleSheet("QLabel { color: #cbd5e1; padding: 0 0 4px 0; }")
    prompt_library_list = QListWidget()
    prompt_library_btn_row = QWidget()
    prompt_library_btn_layout = QHBoxLayout(prompt_library_btn_row)
    save_prompt_btn = QPushButton("Save Current")
    pin_prompt_btn = QPushButton("Pin/Unpin")
    delete_prompt_btn = QPushButton("Delete Saved")
    prompt_library_btn_layout.addWidget(save_prompt_btn)
    prompt_library_btn_layout.addWidget(pin_prompt_btn)
    prompt_library_btn_layout.addWidget(delete_prompt_btn)
    prompt_library_layout.addWidget(prompt_library_hint)
    prompt_library_layout.addWidget(prompt_library_list)
    prompt_library_layout.addWidget(prompt_library_btn_row)
    left_layout.addWidget(prompt_library_group, 2)

    log_group = QGroupBox("Action Log")
    log_layout = QVBoxLayout(log_group)
    action_log = QListWidget()
    log_layout.addWidget(action_log)
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
    run_code_btn = QPushButton("Run Pending Code")
    copy_code_btn = QPushButton("Copy Pending Code")
    discard_code_btn = QPushButton("Discard Pending Code")
    run_code_btn.setEnabled(False)
    copy_code_btn.setEnabled(False)
    discard_code_btn.setEnabled(False)
    code_btn_layout.addWidget(pending_code_label, 1)
    code_btn_layout.addWidget(run_code_btn)
    code_btn_layout.addWidget(copy_code_btn)
    code_btn_layout.addWidget(discard_code_btn)
    transcript_layout.addWidget(code_btn_row)

    input_group = QGroupBox("Prompt")
    input_group_layout = QVBoxLayout(input_group)
    prompt = ChatInput()
    prompt.setPlaceholderText("Ask about the current napari session...")
    prompt.setMinimumHeight(120)
    prompt.setStyleSheet("QTextEdit { background: #10182b; color: #d6deeb; border: 1px solid #30415f; padding: 8px; }")
    input_group_layout.addWidget(prompt)

    right_layout.addWidget(transcript_group, 4)
    right_layout.addWidget(input_group, 1)

    splitter.addWidget(left_panel)
    splitter.addWidget(right_panel)
    splitter.setStretchFactor(0, 2)
    splitter.setStretchFactor(1, 3)

    saved_settings = {
        "provider": "Local (Ollama-style)",
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen3.5",
    }
    active_workers: list[object] = []
    available_models: list[str] = []
    pending_code = {"code": "", "message": ""}
    prompt_library_state = load_prompt_library()
    generation_defaults = {
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "repeat_penalty": 1.5,
    }

    base_url_edit.setText(saved_settings["base_url"])
    model_combo.addItem(saved_settings["model"])
    model_combo.setCurrentText(saved_settings["model"])

    def append_chat_message(role: str, message: str):
        safe_message = (
            str(message).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        )
        if role == "user":
            html = (
                '<table width="100%" cellspacing="0" cellpadding="0" style="margin: 8px 0;"><tr><td width="35%"></td>'
                '<td align="right"><div style="display: inline-block; background: #1d2a44; color: #d7ecff; '
                'border: 1px solid #3f5d87; padding: 10px 12px; border-radius: 10px; font-family: monospace; '
                f'text-align: left; max-width: 100%;">{safe_message}</div></td></tr></table>'
            )
        else:
            html = (
                '<table width="100%" cellspacing="0" cellpadding="0" style="margin: 8px 0;"><tr>'
                '<td align="left"><div style="display: inline-block; background: #111827; color: #d1fae5; '
                'border: 1px solid #2f855a; padding: 10px 12px; border-radius: 10px; font-family: monospace; '
                f'text-align: left; max-width: 100%;">{safe_message}</div></td><td width="35%"></td></tr></table>'
            )
        transcript.append(html)

    def refresh_context(*_args):
        context_label.setText(layer_summary(viewer))

    def format_code_block(code_text: str) -> str:
        stripped = str(code_text or "").strip()
        return f"```python\n{stripped}\n```" if stripped else "```python\n# empty\n```"

    def append_log(message: str):
        action_log.addItem(message)
        action_log.scrollToBottom()

    def selected_prompt_record() -> dict | None:
        item = prompt_library_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def format_worker_error(*args) -> str:
        for value in args:
            if isinstance(value, BaseException):
                return str(value)
            if value:
                return str(value)
        return "Unknown worker error."

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

    def set_pending_code(code_text: str = "", *, message: str = ""):
        pending_code["code"] = str(code_text or "").strip()
        pending_code["message"] = str(message or "").strip()
        has_code = bool(pending_code["code"])
        pending_code_label.setText("Pending code: ready to run" if has_code else "Pending code: none")
        run_code_btn.setEnabled(has_code)
        copy_code_btn.setEnabled(has_code)
        discard_code_btn.setEnabled(has_code)

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

    def refresh_prompt_library():
        prompt_library_list.clear()
        for record in merged_prompt_records(prompt_library_state):
            title = str(record.get("title", "")).strip() or "Untitled Prompt"
            source = str(record.get("source", "built_in"))
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
            short_title = title if len(title) <= 72 else f"{title[:69].rstrip()}..."
            label = f"{badge} {short_title}"
            item = QListWidgetItem()
            item.setText(label)
            item.setData(Qt.UserRole, record)
            item.setForeground(QColor(color))
            prompt_library_list.addItem(item)

    def persist_prompt_library():
        save_prompt_library(prompt_library_state)

    def save_current_prompt(*_args):
        prompt_text = prompt.toPlainText().strip()
        if not prompt_text:
            set_status("Status: no prompt text to save", ok=False)
            append_log("Save prompt skipped: prompt box is empty.")
            return
        upsert_saved_prompt(prompt_library_state, prompt_text)
        persist_prompt_library()
        refresh_prompt_library()
        set_status("Status: prompt saved to library", ok=True)
        append_log(f"Saved prompt to library: {prompt_text[:80]}")

    def toggle_pin_selected_prompt(*_args):
        record = selected_prompt_record()
        if not record:
            set_status("Status: no prompt selected", ok=False)
            append_log("Pin prompt skipped: no prompt selected.")
            return
        if record.get("source") == "built_in":
            upsert_saved_prompt(prompt_library_state, record.get("prompt", ""), pin=True)
            append_log("Pinned built-in prompt into saved library.")
        else:
            new_pinned = not bool(record.get("pinned", False))
            upsert_saved_prompt(prompt_library_state, record.get("prompt", ""), pin=new_pinned)
            set_saved_prompt_pinned(prompt_library_state, record.get("prompt", ""), new_pinned)
            append_log(f"{'Pinned' if new_pinned else 'Unpinned'} saved prompt.")
        persist_prompt_library()
        refresh_prompt_library()
        set_status("Status: prompt library updated", ok=True)

    def delete_selected_prompt(*_args):
        record = selected_prompt_record()
        if not record:
            set_status("Status: no prompt selected", ok=False)
            append_log("Delete prompt skipped: no prompt selected.")
            return
        if record.get("source") != "saved":
            set_status("Status: only saved prompts can be deleted", ok=False)
            append_log("Delete prompt skipped: selected prompt is not a saved prompt.")
            return
        remove_saved_prompt(prompt_library_state, record.get("prompt", ""))
        persist_prompt_library()
        refresh_prompt_library()
        set_status("Status: saved prompt deleted", ok=True)
        append_log("Deleted saved prompt from library.")

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

        @thread_worker(ignore_errors=True)
        def run_refresh_models():
            models = list_ollama_models(base_url)
            return {"models": models, "model": model_name}

        worker = run_refresh_models()
        active_workers.append(worker)

        def finish():
            if worker in active_workers:
                active_workers.remove(worker)

        def on_returned(result):
            refresh_model_choices(result.get("models", []), preferred=result.get("model"))
            saved_settings["base_url"] = base_url
            set_status(f"Status: found {len(result.get('models', []))} local models", ok=True)
            append_log(f"Loaded {len(result.get('models', []))} local models from Ollama")
            finish()

        worker.returned.connect(on_returned)
        worker.errored.connect(lambda *args: (set_status("Status: refresh models failed", ok=False), append_log(f"Refresh models failed: {format_worker_error(*args)}"), finish()))
        worker.start()

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

        @thread_worker(ignore_errors=True)
        def run_test():
            models = list_ollama_models(base_url)
            is_available = model_is_available(model_name, models)
            return {
                "models": models,
                "ok": is_available,
                "reachable": True,
                "message": (
                    f"Connected to Ollama. Model {model_name} is available."
                    if is_available
                    else f"Connected to Ollama, but model {model_name} was not found."
                ),
            }

        worker = run_test()
        active_workers.append(worker)

        def finish():
            if worker in active_workers:
                active_workers.remove(worker)

        def on_test_returned(result):
            refresh_model_choices(result.get("models", []), preferred=model_name)
            saved_settings["base_url"] = base_url
            if result.get("ok"):
                saved_settings["model"] = model_name
            set_status(f"Status: {result['message']}", ok=True if result.get("reachable") else bool(result.get("ok")))
            append_log(result["message"])
            finish()

        worker.returned.connect(on_test_returned)
        worker.errored.connect(lambda *args: (set_status("Status: connection failed", ok=False), append_log(f"Connection test failed: {format_worker_error(*args)}"), finish()))
        worker.start()

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
            "To try a different local model:\n"
            "1. Browse available tags at https://ollama.com/search\n"
            "2. Type the model tag into the Model field if needed\n"
            f"3. Pull it in a terminal with:\n{command}\n"
            f"4. Then click Test Connection against {host_text}.",
        )
        append_log(f"Opened model help. Suggested terminal command: {command}")
        set_status("Status: model help shown", ok=None)

    def unload_model(*_args):
        base_url = base_url_edit.text().strip().rstrip("/")
        model_name = model_combo.currentText().strip()
        if not base_url or not model_name:
            set_status("Status: missing base URL or model name", ok=False)
            append_log("Unload failed: missing base URL or model name.")
            return

        set_status(f"Status: unloading {model_name}...", ok=None)
        append_log(f"Unloading model {model_name}")

        @thread_worker(ignore_errors=True)
        def run_unload():
            unload_ollama_model(base_url, model_name)
            return model_name

        worker = run_unload()
        active_workers.append(worker)

        def finish():
            if worker in active_workers:
                active_workers.remove(worker)

        worker.returned.connect(lambda returned_model: (set_status(f"Status: unloaded {returned_model}", ok=True), append_log(f"Unloaded model {returned_model}"), finish()))
        worker.errored.connect(lambda *args: (set_status("Status: unload failed", ok=False), append_log(f"Unload failed: {format_worker_error(*args)}"), finish()))
        worker.start()

    def send_message():
        text = prompt.toPlainText().strip()
        if not text:
            return
        upsert_recent_prompt(prompt_library_state, text)
        persist_prompt_library()
        refresh_prompt_library()
        append_chat_message("user", text)
        prompt.clear()
        append_log(f"Queued message: {text}")
        base_url = base_url_edit.text().strip().rstrip("/") or str(saved_settings["base_url"]).rstrip("/")
        model_name = model_combo.currentText().strip() or str(saved_settings["model"]).strip()
        if not base_url or not model_name:
            append_chat_message("assistant", "Model settings are incomplete. Fill in Model Connection first.")
            set_status("Status: missing saved model settings", ok=False)
            return
        saved_settings["base_url"] = base_url
        saved_settings["model"] = model_name

        append_chat_message("assistant", "...")
        set_status(f"Status: sending to {model_name}", ok=None)

        @thread_worker(ignore_errors=True)
        def run_chat():
            return chat_ollama(
                base_url,
                model_name,
                system_prompt=assistant_system_prompt(),
                user_payload={"viewer_context": layer_context_json(viewer), "user_message": text},
                options=dict(generation_defaults),
                timeout=1800,
            )

        worker = run_chat()
        active_workers.append(worker)

        def finish():
            if worker in active_workers:
                active_workers.remove(worker)

        def replace_last_assistant(text_out: str):
            cursor = transcript.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()
            append_chat_message("assistant", text_out)

        def on_returned(reply):
            parsed = normalize_model_response(reply)

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
                    result_message = apply_tool_job_result(viewer, tool_result)
                    refresh_context()
                    replace_last_assistant(f"{tool_message}\n{result_message}" if tool_message else result_message)
                    append_log(f"Tool executed: {tool_name}")
                    tool_finish()

                def tool_error(*args):
                    error_text = format_worker_error(*args)
                    replace_last_assistant(f"Tool execution failed: {error_text}")
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
                code_message = str(parsed.get("message", "")).strip() or "Generated napari code. Review it, then click Run Pending Code."
                set_pending_code(code_text, message=code_message)
                replace_last_assistant(f"{code_message}\n{format_code_block(code_text)}")
                set_status("Status: code generated, awaiting approval", ok=None)
                append_log("Generated pending napari code; waiting for approval.")
                finish()
                return

            replace_last_assistant(str(parsed.get("message", reply)).strip() or "[empty response]")
            set_status(f"Status: connected to {model_name}", ok=True)
            append_log(f"Received response from {model_name}")
            finish()

        worker.returned.connect(on_returned)
        worker.errored.connect(lambda *args: (replace_last_assistant(f"Request failed: {format_worker_error(*args)}"), set_status("Status: request failed", ok=False), append_log(f"Chat request failed: {format_worker_error(*args)}"), finish()))
        worker.start()

    def discard_pending_code(*_args):
        if not pending_code["code"]:
            return
        set_pending_code()
        append_chat_message("assistant", "Discarded pending napari code.")
        append_log("Discarded pending napari code.")
        set_status("Status: pending code discarded", ok=None)

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
        code_text = pending_code["code"]
        if not code_text:
            set_status("Status: no pending code to run", ok=False)
            append_log("Run code skipped: no pending code available.")
            return

        set_status("Status: running approved code", ok=None)
        append_log("Running approved napari code.")
        run_code_btn.setEnabled(False)
        copy_code_btn.setEnabled(False)
        discard_code_btn.setEnabled(False)
        stdout_buffer = io.StringIO()
        namespace = {
            "__builtins__": __builtins__,
            "napari": napari,
            "np": np,
            "numpy": np,
            "viewer": viewer,
            "selected_layer": None if viewer is None else viewer.layers.selection.active,
        }
        try:
            with redirect_stdout(stdout_buffer):
                exec(compile(code_text, "<napari-chat-assistant>", "exec"), namespace, namespace)
        except Exception as exc:
            error_text = str(exc)
            append_chat_message("assistant", f"Approved code failed:\n{error_text}")
            append_log(f"Approved code failed: {error_text}")
            set_status("Status: approved code failed", ok=False)
            if pending_code["code"]:
                run_code_btn.setEnabled(True)
                copy_code_btn.setEnabled(True)
                discard_code_btn.setEnabled(True)
            return

        refresh_context()
        stdout_text = stdout_buffer.getvalue().strip()
        result_message = "Approved napari code executed."
        if stdout_text:
            result_message = f"{result_message}\nOutput:\n{stdout_text}"
        append_chat_message("assistant", result_message)
        append_log("Approved napari code executed successfully.")
        set_status("Status: approved code executed", ok=True)
        set_pending_code()

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
    run_code_btn.clicked.connect(run_pending_code)
    copy_code_btn.clicked.connect(copy_pending_code)
    discard_code_btn.clicked.connect(discard_pending_code)
    prompt_library_list.itemClicked.connect(load_library_prompt)
    prompt_library_list.itemDoubleClicked.connect(send_library_prompt)
    save_prompt_btn.clicked.connect(save_current_prompt)
    pin_prompt_btn.clicked.connect(toggle_pin_selected_prompt)
    delete_prompt_btn.clicked.connect(delete_selected_prompt)
    prompt.sendRequested.connect(send_message)
    refresh_btn.clicked.connect(refresh_context)
    root.destroyed.connect(cleanup_workers)

    refresh_context()
    set_pending_code()
    refresh_models()
    refresh_prompt_library()
    append_log(f"Prompt library path: {prompt_library_path()}")
    append_log("Assistant panel initialized.")
    return root
