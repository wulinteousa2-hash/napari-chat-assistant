from __future__ import annotations

from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Qt
from qtpy.QtWidgets import QDialog, QVBoxLayout, QWidget

from napari_chat_assistant.agent.ui_state import load_ui_state, save_ui_state
from napari_chat_assistant.voice.audio_recorder import AudioRecorder
from napari_chat_assistant.voice.backends import FasterWhisperBackend
from napari_chat_assistant.voice.config import VOICE_INPUT_DIALOG_TITLE
from napari_chat_assistant.voice.voice_input_widget import VoiceInputWidget


def _format_worker_error(*args) -> str:
    for value in args:
        if isinstance(value, BaseException):
            return str(value)
        if value:
            return str(value)
    return "Unknown worker error."


class VoiceInputController(QObject):
    def __init__(
        self,
        *,
        dialog: QDialog,
        submit_callback=None,
    ) -> None:
        super().__init__(dialog)
        self._dialog = dialog
        self._submit_callback = submit_callback
        self._widget = VoiceInputWidget(dialog)
        self._backend = FasterWhisperBackend()
        self._recorder = AudioRecorder(dialog)
        self._active_worker = None
        self._latest_audio_path = Path()
        self._ui_state = load_ui_state()

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._widget)

        self._widget.startRequested.connect(self.start_recording)
        self._widget.stopRequested.connect(self.stop_recording)
        self._widget.retryRequested.connect(self.retry_recording)
        self._widget.runRequested.connect(self.run_transcript)
        self._widget.discardRequested.connect(self.close_dialog)
        self._widget.device_combo.currentTextChanged.connect(self._remember_selected_device)
        self._recorder.levelChanged.connect(self._widget.set_input_level)
        self._dialog.finished.connect(lambda *_args: self._cleanup())

        self._refresh_availability()

    def start_recording(self) -> None:
        try:
            self._recorder.start(self._widget.selected_input_device())
        except Exception as exc:
            self._widget.set_status(str(exc))
            return
        self._widget.set_recording(True)
        self._widget.set_retry_available(False)
        active_device = self._recorder.active_device_name()
        self._widget.set_status(f"Recording from {active_device}..." if active_device else "Recording...")

    def stop_recording(self) -> None:
        try:
            audio_path = self._recorder.stop()
        except Exception as exc:
            self._widget.set_status(str(exc))
            return
        self._latest_audio_path = Path(audio_path)
        self._widget.set_recording(False)
        self._widget.set_retry_available(True)
        self._widget.set_transcribing(True)
        self._widget.set_status("Transcribing locally...")

        @thread_worker(ignore_errors=True)
        def transcribe_audio():
            return self._backend.transcribe_file(str(self._latest_audio_path))

        worker = transcribe_audio()
        self._active_worker = worker
        worker.returned.connect(self._on_transcription_ready)
        worker.errored.connect(self._on_transcription_error)
        worker.finished.connect(self._on_transcription_finished)
        worker.start()

    def retry_recording(self) -> None:
        self._cleanup_audio_file()
        self._recorder.cleanup()
        self._widget.clear_transcript()
        self._widget.set_retry_available(False)
        self._widget.set_recording(False)
        self._widget.set_transcribing(False)
        self._widget.set_status("Ready to record again.")

    def run_transcript(self, text: str) -> None:
        transcript = str(text or "").strip()
        if not transcript:
            self._widget.set_status("Transcript is empty.")
            return
        if callable(self._submit_callback):
            self._submit_callback(transcript)
        self._widget.set_status("Transcript sent. You can record again or edit and run another prompt.")

    def close_dialog(self) -> None:
        self._dialog.close()

    def _refresh_availability(self) -> None:
        backend = self._backend.status()
        recorder = self._recorder.status()
        selected_device = str(self._ui_state.get("voice_input_device", "")).strip()
        self._widget.set_input_devices(self._recorder.audio_input_names(), selected=selected_device)
        disabled_reason = ""
        if not backend.available:
            disabled_reason = backend.detail or backend.summary
        elif not recorder.available:
            disabled_reason = recorder.detail or recorder.summary
        lines = [
            f"Backend: {backend.summary}",
            backend.detail,
            f"Recorder: {recorder.summary}",
            recorder.detail,
        ]
        self._widget.set_availability(
            lines,
            enabled=backend.available and recorder.available,
            disabled_reason=disabled_reason,
        )
        if backend.available and recorder.available:
            self._widget.set_status("Ready to record.")

    def _remember_selected_device(self, device_name: str) -> None:
        selected = str(device_name or "").strip()
        if selected == str(self._ui_state.get("voice_input_device", "")).strip():
            return
        self._ui_state["voice_input_device"] = selected
        save_ui_state(self._ui_state)

    def _on_transcription_ready(self, text: str) -> None:
        self._widget.set_transcribing(False)
        self._widget.set_transcript(text)
        self._widget.set_retry_available(True)
        self._widget.set_status("Transcript ready. Edit it if needed, then press Run or Enter.")

    def _on_transcription_error(self, *args) -> None:
        self._widget.set_transcribing(False)
        self._widget.set_retry_available(True)
        self._widget.set_status(_format_worker_error(*args))

    def _on_transcription_finished(self) -> None:
        self._active_worker = None
        self._cleanup_audio_file()

    def _cleanup_audio_file(self) -> None:
        if self._latest_audio_path and self._latest_audio_path.exists():
            try:
                self._latest_audio_path.unlink()
            except Exception:
                pass
        self._latest_audio_path = Path()

    def _cleanup(self) -> None:
        self._cleanup_audio_file()
        self._recorder.cleanup()


def open_voice_input_dialog(
    _viewer=None,
    *,
    parent: QWidget | None = None,
    submit_callback=None,
) -> QDialog:
    dialog = None if parent is None else getattr(parent, "_voice_input_dialog", None)
    if dialog is None:
        dialog = QDialog(parent)
        dialog.setWindowTitle(VOICE_INPUT_DIALOG_TITLE)
        dialog.resize(720, 520)
        dialog.setModal(False)
        dialog.setWindowFlag(Qt.Tool, True)
        controller = VoiceInputController(dialog=dialog, submit_callback=submit_callback)
        dialog._voice_input_controller = controller  # type: ignore[attr-defined]
        if parent is not None:
            parent._voice_input_dialog = dialog  # type: ignore[attr-defined]
    else:
        controller = getattr(dialog, "_voice_input_controller", None)
        if controller is not None:
            controller._submit_callback = submit_callback
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
