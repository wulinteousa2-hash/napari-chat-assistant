from __future__ import annotations

import importlib
from dataclasses import dataclass

from napari_chat_assistant.voice.config import DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE, DEFAULT_MODEL_SIZE


@dataclass(frozen=True)
class VoiceBackendStatus:
    available: bool
    summary: str
    detail: str = ""


def backend_status() -> VoiceBackendStatus:
    try:
        importlib.import_module("faster_whisper")
    except Exception:
        return VoiceBackendStatus(
            available=False,
            summary="`faster-whisper` is not installed in this environment.",
            detail=(
                "Install `faster-whisper` in the same Python environment that launches napari "
                "to enable local transcription."
            ),
        )
    return VoiceBackendStatus(
        available=True,
        summary="Local transcription backend is available.",
        detail="Record audio, review the transcript, then insert the edited text into the prompt.",
    )


class FasterWhisperBackend:
    def __init__(
        self,
        *,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ) -> None:
        self._model_size = str(model_size or DEFAULT_MODEL_SIZE).strip() or DEFAULT_MODEL_SIZE
        self._device = str(device or DEFAULT_DEVICE).strip() or DEFAULT_DEVICE
        self._compute_type = str(compute_type or DEFAULT_COMPUTE_TYPE).strip() or DEFAULT_COMPUTE_TYPE
        self._model = None

    def status(self) -> VoiceBackendStatus:
        return backend_status()

    def is_available(self) -> bool:
        return self.status().available

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
                "`faster-whisper` could not be imported from the napari Python environment."
            ) from exc
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        return self._model

    def transcribe_file(self, audio_path: str) -> str:
        model = self._load_model()
        try:
            segments, _info = model.transcribe(audio_path, vad_filter=True)
        except Exception as exc:
            raise RuntimeError(f"Local transcription failed: {exc}") from exc
        transcript = " ".join(str(segment.text or "").strip() for segment in segments).strip()
        if not transcript:
            raise RuntimeError("No speech was detected in the recording.")
        return transcript
