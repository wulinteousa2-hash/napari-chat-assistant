from __future__ import annotations

import audioop
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import QObject, Signal

from napari_chat_assistant.voice.config import DEFAULT_CHANNEL_COUNT, DEFAULT_SAMPLE_RATE


@dataclass(frozen=True)
class RecorderStatus:
    available: bool
    summary: str
    detail: str = ""


def recorder_status() -> RecorderStatus:
    try:
        inputs = _available_audio_inputs()
    except RuntimeError as exc:
        return RecorderStatus(
            available=False,
            summary="Qt multimedia microphone support is not available in this environment.",
            detail=str(exc),
        )
    if not inputs:
        return RecorderStatus(
            available=False,
            summary="No microphone input device was found.",
            detail="Connect a microphone and restart napari if you want to record locally.",
        )
    return RecorderStatus(
        available=True,
        summary="Microphone recording is available.",
        detail="Audio is captured locally and written to a temporary WAV file for transcription.",
    )


class AudioRecorder(QObject):
    recordingStarted = Signal()
    recordingStopped = Signal(str)
    levelChanged = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._audio_source = None
        self._audio_input = None
        self._buffer = bytearray()
        self._recording_path = Path()
        self._format_info = {"sample_rate": DEFAULT_SAMPLE_RATE, "channels": DEFAULT_CHANNEL_COUNT, "sample_width": 2}
        self._active_device_name = ""

    def status(self) -> RecorderStatus:
        return recorder_status()

    def is_available(self) -> bool:
        return self.status().available

    def is_recording(self) -> bool:
        return self._audio_source is not None and self._audio_input is not None

    def audio_input_names(self) -> list[str]:
        names: list[str] = []
        for device in _available_audio_inputs():
            name = _device_name(device)
            if name and name not in names:
                names.append(name)
        return names

    def active_device_name(self) -> str:
        return self._active_device_name

    def start(self, device_name: str = "") -> None:
        if self.is_recording():
            return
        status = self.status()
        if not status.available:
            raise RuntimeError(status.summary)

        self.cleanup()
        self._audio_source, self._audio_input, capture_format, active_device_name = _start_audio_capture(
            parent=self,
            preferred_device_name=device_name,
        )
        if self._audio_input is None:
            self.cleanup()
            raise RuntimeError("Microphone recording could not be started.")
        self._buffer.clear()
        self._format_info = self._describe_audio_format(capture_format)
        self._active_device_name = str(active_device_name or "").strip()
        self._audio_input.readyRead.connect(self._consume_audio)
        self.recordingStarted.emit()

    def stop(self) -> str:
        if not self.is_recording():
            raise RuntimeError("Recording has not started.")
        self._consume_audio()
        self._audio_source.stop()
        path = self._write_wav_file(bytes(self._buffer), self._format_info)
        self._disconnect_audio()
        self._buffer.clear()
        self._recording_path = Path(path)
        self.recordingStopped.emit(path)
        return path

    def cleanup(self) -> None:
        self._disconnect_audio()
        self._buffer.clear()
        if self._recording_path and self._recording_path.exists():
            try:
                self._recording_path.unlink()
            except Exception:
                pass
        self._recording_path = Path()
        self._active_device_name = ""

    def _disconnect_audio(self) -> None:
        if self._audio_input is not None:
            try:
                self._audio_input.readyRead.disconnect(self._consume_audio)
            except Exception:
                pass
        self._audio_input = None
        if self._audio_source is not None:
            try:
                self._audio_source.deleteLater()
            except Exception:
                pass
        self._audio_source = None
        self.levelChanged.emit(0)

    def _consume_audio(self) -> None:
        if self._audio_input is None:
            return
        chunk = self._audio_input.readAll()
        raw_bytes = bytes(chunk)
        if not raw_bytes:
            return
        self._buffer.extend(raw_bytes)
        self.levelChanged.emit(_normalized_level(raw_bytes, int(self._format_info.get("sample_width", 2))))

    @staticmethod
    def _configure_audio_format(audio_format) -> None:
        audio_format.setSampleRate(DEFAULT_SAMPLE_RATE)
        audio_format.setChannelCount(DEFAULT_CHANNEL_COUNT)
        if hasattr(audio_format, "setSampleFormat") and hasattr(type(audio_format), "Int16"):
            audio_format.setSampleFormat(type(audio_format).Int16)
            return
        audio_format.setSampleSize(16)
        audio_format.setCodec("audio/pcm")
        audio_format.setByteOrder(type(audio_format).LittleEndian)
        audio_format.setSampleType(type(audio_format).SignedInt)

    @staticmethod
    def _describe_audio_format(audio_format) -> dict[str, int]:
        channels = int(getattr(audio_format, "channelCount", lambda: DEFAULT_CHANNEL_COUNT)() or DEFAULT_CHANNEL_COUNT)
        sample_rate = int(getattr(audio_format, "sampleRate", lambda: DEFAULT_SAMPLE_RATE)() or DEFAULT_SAMPLE_RATE)
        sample_width = 2
        if hasattr(audio_format, "bytesPerSample"):
            try:
                sample_width = int(audio_format.bytesPerSample())
            except Exception:
                sample_width = 2
        elif hasattr(audio_format, "sampleSize"):
            try:
                sample_width = max(1, int(audio_format.sampleSize()) // 8)
            except Exception:
                sample_width = 2
        return {"sample_rate": sample_rate, "channels": channels, "sample_width": sample_width}

    @staticmethod
    def _write_wav_file(raw_audio: bytes, format_info: dict[str, int]) -> str:
        fd, path = tempfile.mkstemp(prefix="napari-chat-assistant-voice-", suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(max(1, int(format_info.get("channels", DEFAULT_CHANNEL_COUNT))))
            wav_file.setsampwidth(max(1, int(format_info.get("sample_width", 2))))
            wav_file.setframerate(max(1, int(format_info.get("sample_rate", DEFAULT_SAMPLE_RATE))))
            wav_file.writeframes(raw_audio)
        return path


def _available_audio_inputs():
    try:
        from qtpy.QtMultimedia import QMediaDevices

        return list(QMediaDevices.audioInputs())
    except Exception:
        pass
    try:
        from qtpy.QtMultimedia import QAudio, QAudioDeviceInfo

        return list(QAudioDeviceInfo.availableDevices(QAudio.AudioInput))
    except Exception as exc:
        raise RuntimeError(
            "Install a Qt build with multimedia support in the same Python environment as napari."
        ) from exc


def _device_priority(device_name: str) -> tuple[int, str]:
    name = str(device_name or "").strip().lower()
    if name == "default":
        return (0, name)
    if name.startswith("front:"):
        return (1, name)
    if name.startswith("sysdefault"):
        return (2, name)
    if "pipewire" in name:
        return (9, name)
    return (3, name)


def _normalized_level(raw_audio: bytes, sample_width: int) -> int:
    width = max(1, min(int(sample_width or 2), 4))
    try:
        rms = audioop.rms(raw_audio, width)
    except Exception:
        return 0
    max_value = float((1 << (8 * width - 1)) - 1)
    if max_value <= 0:
        return 0
    level = int(max(0.0, min(100.0, (float(rms) / max_value) * 100.0)))
    return level


def _device_name(device) -> str:
    if hasattr(device, "description"):
        try:
            return str(device.description() or "").strip()
        except Exception:
            pass
    if hasattr(device, "deviceName"):
        try:
            return str(device.deviceName() or "").strip()
        except Exception:
            pass
    return ""


def _configured_audio_format_for_device(device):
    from qtpy.QtMultimedia import QAudioFormat

    requested = QAudioFormat()
    AudioRecorder._configure_audio_format(requested)
    if hasattr(device, "isFormatSupported") and hasattr(device, "nearestFormat"):
        try:
            if device.isFormatSupported(requested):
                return requested
        except Exception:
            pass
        try:
            nearest = device.nearestFormat(requested)
            if nearest is not None:
                return nearest
        except Exception:
            pass
    if hasattr(device, "preferredFormat"):
        try:
            preferred = device.preferredFormat()
            if preferred is not None:
                return preferred
        except Exception:
            pass
    return requested


def _ordered_candidate_devices(devices, preferred_device_name: str = ""):
    preferred = str(preferred_device_name or "").strip()
    items = list(devices)
    seen = set()
    ordered = []
    for device in sorted(
        items,
        key=lambda item: (
            0 if preferred and _device_name(item) == preferred else 1,
            _device_priority(_device_name(item)),
        ),
    ):
        name = _device_name(device)
        if name in seen:
            continue
        seen.add(name)
        ordered.append(device)
    return ordered


def _start_audio_capture(*, parent: QObject, preferred_device_name: str = ""):
    try:
        from qtpy.QtMultimedia import QAudioSource, QMediaDevices

        inputs = list(QMediaDevices.audioInputs())
        default_device = QMediaDevices.defaultAudioInput()
        candidates = []
        if not (hasattr(default_device, "isNull") and default_device.isNull()):
            candidates.append(default_device)
        candidates.extend(inputs)
        ordered_candidates = _ordered_candidate_devices(candidates, preferred_device_name)
        if not ordered_candidates:
            raise RuntimeError("No microphone input device was found.")
        errors: list[str] = []
        for device in ordered_candidates:
            audio_format = _configured_audio_format_for_device(device)
            source = QAudioSource(device, audio_format, parent)
            io_device = source.start()
            if io_device is not None:
                return source, io_device, audio_format, _device_name(device)
            errors.append(_device_name(device) or "unknown")
            try:
                source.deleteLater()
            except Exception:
                pass
        raise RuntimeError(
            "Microphone recording could not be started for any detected input device."
            + (f" Tried: {', '.join(errors)}" if errors else "")
        )
    except Exception:
        pass

    from qtpy.QtMultimedia import QAudio, QAudioDeviceInfo, QAudioInput

    inputs = _ordered_candidate_devices(
        list(QAudioDeviceInfo.availableDevices(QAudio.AudioInput)),
        preferred_device_name,
    )
    if not inputs:
        raise RuntimeError("No microphone input device was found.")
    errors: list[str] = []
    for device in inputs:
        audio_format = _configured_audio_format_for_device(device)
        source = QAudioInput(device, audio_format, parent)
        io_device = source.start()
        if io_device is not None:
            return source, io_device, audio_format, _device_name(device)
        errors.append(_device_name(device) or "unknown")
        try:
            source.stop()
            source.deleteLater()
        except Exception:
            pass
    raise RuntimeError(
        "Microphone recording could not be started for any detected input device."
        + (f" Tried: {', '.join(errors)}" if errors else "")
    )
