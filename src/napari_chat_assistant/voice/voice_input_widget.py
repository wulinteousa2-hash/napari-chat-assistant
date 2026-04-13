from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_chat_assistant.voice.config import VOICE_INPUT_HELP_TEXT


class VoiceInputWidget(QWidget):
    startRequested = Signal()
    stopRequested = Signal()
    retryRequested = Signal()
    runRequested = Signal(str)
    discardRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro_group = QGroupBox("Optional Local Feature")
        intro_layout = QVBoxLayout(intro_group)
        intro_layout.setContentsMargins(10, 10, 10, 10)
        intro_layout.setSpacing(6)

        self.help_label = QLabel(VOICE_INPUT_HELP_TEXT)
        self.help_label.setWordWrap(True)
        self.help_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.help_label.setStyleSheet("QLabel { color: #cbd5e1; }")
        intro_layout.addWidget(self.help_label)

        self.availability_label = QLabel("")
        self.availability_label.setWordWrap(True)
        self.availability_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        intro_layout.addWidget(self.availability_label)
        layout.addWidget(intro_group)

        controls_group = QGroupBox("Recording")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(8)

        self.instructions_label = QLabel(
            "Start Recording, speak into your microphone, then Stop Recording to transcribe locally."
        )
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setStyleSheet("QLabel { color: #cbd5e1; }")
        controls_layout.addWidget(self.instructions_label)

        self.device_label = QLabel("Input Device")
        self.device_label.setStyleSheet("QLabel { color: #cbd5e1; }")
        controls_layout.addWidget(self.device_label)

        self.device_combo = QComboBox()
        controls_layout.addWidget(self.device_combo)

        button_row = QWidget()
        button_row_layout = QHBoxLayout(button_row)
        button_row_layout.setContentsMargins(0, 0, 0, 0)
        button_row_layout.setSpacing(6)

        self.start_btn = QPushButton("Start Recording")
        self.stop_btn = QPushButton("Stop Recording")
        self.retry_btn = QPushButton("Record Again")
        self.discard_btn = QPushButton("Discard")

        button_row_layout.addWidget(self.start_btn)
        button_row_layout.addWidget(self.stop_btn)
        button_row_layout.addWidget(self.retry_btn)
        button_row_layout.addStretch(1)
        button_row_layout.addWidget(self.discard_btn)
        controls_layout.addWidget(button_row)

        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet("QLabel { color: #cbd5e1; }")
        controls_layout.addWidget(self.progress_label)

        self.level_label = QLabel("Input Level")
        self.level_label.setStyleSheet("QLabel { color: #cbd5e1; }")
        controls_layout.addWidget(self.level_label)

        self.level_meter = QProgressBar()
        self.level_meter.setRange(0, 100)
        self.level_meter.setValue(0)
        self.level_meter.setTextVisible(False)
        self.level_meter.setStyleSheet(
            "QProgressBar { background: #151b2b; border: 1px solid #30415f; min-height: 10px; }"
            "QProgressBar::chunk { background: #22c55e; }"
        )
        controls_layout.addWidget(self.level_meter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        layout.addWidget(controls_group)

        preview_group = QGroupBox("Preview Transcript")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setSpacing(8)

        self.preview_edit = TranscriptPreviewEdit()
        self.preview_edit.setAcceptRichText(False)
        self.preview_edit.setPlaceholderText("Transcript will appear here after local transcription.")
        preview_layout.addWidget(self.preview_edit)

        preview_actions = QWidget()
        preview_actions_layout = QHBoxLayout(preview_actions)
        preview_actions_layout.setContentsMargins(0, 0, 0, 0)
        preview_actions_layout.setSpacing(6)
        self.run_btn = QPushButton("Run")
        preview_actions_layout.addStretch(1)
        preview_actions_layout.addWidget(self.run_btn)
        preview_layout.addWidget(preview_actions)
        layout.addWidget(preview_group, 1)

        self.start_btn.clicked.connect(self.startRequested.emit)
        self.stop_btn.clicked.connect(self.stopRequested.emit)
        self.retry_btn.clicked.connect(self.retryRequested.emit)
        self.discard_btn.clicked.connect(self.discardRequested.emit)
        self.run_btn.clicked.connect(self._emit_run_requested)
        self.preview_edit.textChanged.connect(self._refresh_insert_state)
        self.preview_edit.runRequested.connect(self._emit_run_requested)

        self._feature_enabled = True
        self._disabled_reason = ""
        self.set_recording(False)
        self.set_transcribing(False)
        self.set_retry_available(False)
        self._refresh_insert_state()

    def set_availability(self, lines: list[str], *, enabled: bool, disabled_reason: str = "") -> None:
        self._feature_enabled = bool(enabled)
        self._disabled_reason = str(disabled_reason or "").strip()
        self.availability_label.setText("\n".join(line for line in lines if line).strip())
        self.start_btn.setEnabled(self._feature_enabled)
        self.device_combo.setEnabled(self._feature_enabled)
        self.stop_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        if not self._feature_enabled:
            self.progress_label.setText(
                self._disabled_reason or "Voice input is unavailable in this environment."
            )
            self.start_btn.setToolTip(self._disabled_reason or "Voice input is unavailable in this environment.")
        else:
            self.start_btn.setToolTip("Start local microphone recording.")
        self._refresh_insert_state()

    def set_recording(self, active: bool) -> None:
        self.start_btn.setEnabled(self._feature_enabled and not active and not self.progress_bar.isVisible())
        self.device_combo.setEnabled(self._feature_enabled and not active and not self.progress_bar.isVisible())
        self.stop_btn.setEnabled(self._feature_enabled and active)
        self.level_meter.setValue(0)
        if active:
            self.progress_label.setText("Recording...")
        elif not self.progress_bar.isVisible():
            self.progress_label.setText("Ready to record.")

    def set_transcribing(self, active: bool) -> None:
        self.progress_bar.setVisible(active)
        if active:
            self.start_btn.setEnabled(False)
            self.device_combo.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.retry_btn.setEnabled(False)
            self.run_btn.setEnabled(False)
            self.progress_label.setText("Transcribing locally...")
        else:
            self.set_recording(False)
            self._refresh_insert_state()

    def set_retry_available(self, enabled: bool) -> None:
        self.retry_btn.setEnabled(self._feature_enabled and enabled and not self.progress_bar.isVisible())

    def set_status(self, text: str) -> None:
        self.progress_label.setText(str(text or "").strip())

    def set_input_level(self, level: int) -> None:
        self.level_meter.setValue(max(0, min(100, int(level))))

    def set_input_devices(self, devices: list[str], *, selected: str = "") -> None:
        current = str(selected or "").strip()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for device in devices:
            name = str(device or "").strip()
            if name:
                self.device_combo.addItem(name)
        if self.device_combo.count() > 0:
            index = self.device_combo.findText(current)
            if index < 0:
                index = 0
            self.device_combo.setCurrentIndex(index)
        self.device_combo.blockSignals(False)

    def selected_input_device(self) -> str:
        return self.device_combo.currentText().strip()

    def set_transcript(self, text: str) -> None:
        self.preview_edit.setPlainText(str(text or ""))
        self._refresh_insert_state()

    def clear_transcript(self) -> None:
        self.preview_edit.clear()
        self._refresh_insert_state()

    def transcript_text(self) -> str:
        return self.preview_edit.toPlainText().strip()

    def _emit_run_requested(self) -> None:
        self.runRequested.emit(self.transcript_text())

    def _refresh_insert_state(self) -> None:
        has_text = bool(self.transcript_text())
        self.run_btn.setEnabled(self._feature_enabled and has_text and not self.progress_bar.isVisible())


class TranscriptPreviewEdit(QTextEdit):
    runRequested = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier):
                return super().keyPressEvent(event)
            self.runRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)
