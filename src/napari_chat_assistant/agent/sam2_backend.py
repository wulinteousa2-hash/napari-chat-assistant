from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

import numpy as np

from napari_chat_assistant.agent.ui_state import DEFAULT_UI_STATE, load_ui_state


@dataclass(frozen=True)
class SAM2BackendConfig:
    project_path: str
    checkpoint_path: str
    config_path: str
    device: str

    @property
    def project_root(self) -> Path:
        return Path(self.project_path).expanduser()

    @property
    def checkpoint_file(self) -> Path:
        path = Path(self.checkpoint_path).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path

    @property
    def config_file(self) -> Path:
        path = Path(self.config_path).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        if not path.exists():
            alt = self.project_root / "sam2" / self.config_path
            if alt.exists():
                return alt
        return path


def sam2_config_from_ui_state(ui_state: dict[str, Any] | None = None) -> SAM2BackendConfig:
    state = ui_state or load_ui_state()
    return SAM2BackendConfig(
        project_path=str(state.get("sam2_project_path", DEFAULT_UI_STATE["sam2_project_path"])).strip(),
        checkpoint_path=str(state.get("sam2_checkpoint_path", DEFAULT_UI_STATE["sam2_checkpoint_path"])).strip(),
        config_path=str(state.get("sam2_config_path", DEFAULT_UI_STATE["sam2_config_path"])).strip(),
        device=str(state.get("sam2_device", DEFAULT_UI_STATE["sam2_device"])).strip() or "cuda",
    )


def _normalize_config_name(config_path: str) -> str:
    normalized = str(config_path).replace("\\", "/")
    marker = "/sam2/configs/"
    if marker in normalized:
        return "configs/" + normalized.split(marker, 1)[1]
    marker = "/configs/"
    if marker in normalized:
        return "configs/" + normalized.split(marker, 1)[1]
    if normalized.startswith("sam2/configs/"):
        return normalized[len("sam2/") :]
    return normalized


def _wrapper_candidates(project_root: Path) -> list[Path]:
    return [
        project_root / "sam2_wrapper.py",
        project_root / "src" / "sam2_wrapper.py",
        project_root / "sam2_wrapper" / "__init__.py",
        project_root / "src" / "sam2_wrapper" / "__init__.py",
    ]


def _load_wrapper_module(config: SAM2BackendConfig) -> ModuleType:
    project_root = config.project_root
    if not project_root.exists():
        raise FileNotFoundError(
            f"SAM2 project path [{project_root}] does not exist. Open SAM2 Setup and update the project path."
        )

    for candidate in _wrapper_candidates(project_root):
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("_napari_chat_assistant_sam2_wrapper", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    raise ImportError(
        f"No SAM2 wrapper module was found under [{project_root}]. "
        "Expected sam2_wrapper.py or sam2_wrapper/__init__.py."
    )


def get_sam2_backend_status(config: SAM2BackendConfig | None = None) -> tuple[bool, str]:
    resolved = config or sam2_config_from_ui_state()
    project_root = resolved.project_root
    if not project_root.exists():
        return False, (
            f"SAM2 backend is not configured. Project path [{project_root}] does not exist. "
            "Open SAM2 Setup to update the path."
        )
    if not resolved.checkpoint_file.exists():
        return False, (
            f"SAM2 backend is not configured. Checkpoint file [{resolved.checkpoint_file}] was not found. "
            "Open SAM2 Setup to update the checkpoint path."
        )
    if not resolved.config_file.exists():
        return False, (
            f"SAM2 backend is not configured. Config file [{resolved.config_file}] was not found. "
            "Open SAM2 Setup to update the config path."
        )
    try:
        wrapper = _load_wrapper_module(resolved)
    except Exception as exc:
        return False, f"SAM2 backend is not configured. {exc}"
    if not hasattr(wrapper, "segment_image_from_box"):
        return False, (
            "SAM2 backend wrapper is missing [segment_image_from_box]. "
            "Expose that function from the external SAM2 wrapper."
        )
    return True, (
        f"SAM2 backend ready. project=[{project_root}] checkpoint=[{resolved.checkpoint_file}] "
        f"config=[{resolved.config_file}] device=[{resolved.device}]."
    )


def segment_image_from_box(
    image: np.ndarray,
    *,
    box_xyxy: tuple[float, float, float, float],
    model_name: str | None = None,
    config: SAM2BackendConfig | None = None,
) -> tuple[np.ndarray, str]:
    resolved = config or sam2_config_from_ui_state()
    ok, status_message = get_sam2_backend_status(resolved)
    if not ok:
        raise RuntimeError(status_message)

    wrapper = _load_wrapper_module(resolved)
    result = wrapper.segment_image_from_box(
        image=np.asarray(image),
        box_xyxy=tuple(float(value) for value in box_xyxy),
        checkpoint_path=str(resolved.checkpoint_file),
        config_path=str(resolved.config_file),
        device=resolved.device,
        model_name=model_name,
    )

    message = ""
    mask = result
    if isinstance(result, dict):
        mask = result.get("mask")
        message = str(result.get("message") or "").strip()

    if mask is None:
        raise RuntimeError("SAM2 wrapper returned no mask for segment_image_from_box.")

    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError(f"SAM2 wrapper returned mask ndim={binary.ndim}; expected a 2D mask.")
    if binary.shape != np.asarray(image).shape:
        raise ValueError(
            f"SAM2 wrapper returned mask shape {binary.shape}, which does not match image shape {np.asarray(image).shape}."
    )
    return (binary > 0).astype(np.int32, copy=False), message


def segment_image_from_points(
    image: np.ndarray,
    *,
    point_coords_xy: np.ndarray,
    point_labels: np.ndarray,
    model_name: str | None = None,
    config: SAM2BackendConfig | None = None,
) -> tuple[np.ndarray, str]:
    resolved = config or sam2_config_from_ui_state()
    ok, status_message = get_sam2_backend_status(resolved)
    if not ok:
        raise RuntimeError(status_message)

    wrapper = _load_wrapper_module(resolved)
    if not hasattr(wrapper, "segment_image_from_points"):
        raise RuntimeError(
            "SAM2 backend wrapper is missing [segment_image_from_points]. "
            "Expose that function from the external SAM2 wrapper."
        )

    result = wrapper.segment_image_from_points(
        image=np.asarray(image),
        point_coords_xy=np.asarray(point_coords_xy, dtype=np.float32),
        point_labels=np.asarray(point_labels, dtype=np.int32),
        checkpoint_path=str(resolved.checkpoint_file),
        config_path=str(resolved.config_file),
        device=resolved.device,
        model_name=model_name,
    )

    message = ""
    mask = result
    if isinstance(result, dict):
        mask = result.get("mask")
        message = str(result.get("message") or "").strip()

    if mask is None:
        raise RuntimeError("SAM2 wrapper returned no mask for segment_image_from_points.")

    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError(f"SAM2 wrapper returned mask ndim={binary.ndim}; expected a 2D mask.")
    if binary.shape != np.asarray(image).shape:
        raise ValueError(
            f"SAM2 wrapper returned mask shape {binary.shape}, which does not match image shape {np.asarray(image).shape}."
    )
    return (binary > 0).astype(np.int32, copy=False), message


def _pick_device_local(device: str | None = None) -> str:
    requested = str(device or "cuda").strip().lower()
    try:
        import torch
    except Exception:
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _prepare_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"_prepare_image expects a 2D grayscale image, got ndim={array.ndim}.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"_prepare_image expects numeric image data, got dtype={array.dtype}.")
    array = array.astype(np.float32, copy=False)
    if array.size == 0:
        raise ValueError("_prepare_image expects a non-empty image.")
    finite = np.isfinite(array)
    if not np.any(finite):
        raise ValueError("_prepare_image image contains no finite values.")
    valid = array[finite]
    lo = float(valid.min())
    hi = float(valid.max())
    if hi > lo:
        scaled = (array - lo) / (hi - lo)
    else:
        scaled = np.zeros_like(array, dtype=np.float32)
    rgb = np.stack([scaled, scaled, scaled], axis=-1)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8, copy=False)


def _mask_from_video_output(video_res_masks) -> np.ndarray:
    masks = video_res_masks
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    array = np.asarray(masks)
    while array.ndim > 3:
        array = np.squeeze(array, axis=0)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"SAM2 video predictor returned unexpected mask ndim={array.ndim}.")
    return (array > 0).astype(np.int32, copy=False)


def propagate_volume_from_points(
    volume: np.ndarray,
    *,
    seed_frame_idx: int,
    point_coords_xy: np.ndarray,
    point_labels: np.ndarray,
    model_name: str | None = None,
    config: SAM2BackendConfig | None = None,
) -> tuple[np.ndarray, str]:
    del model_name
    resolved = config or sam2_config_from_ui_state()
    ok, status_message = get_sam2_backend_status(resolved)
    if not ok:
        raise RuntimeError(status_message)

    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    data = np.asarray(volume)
    if data.ndim != 3:
        raise ValueError(f"propagate_volume_from_points expects a 3D grayscale volume, got ndim={data.ndim}.")
    if seed_frame_idx < 0 or seed_frame_idx >= data.shape[0]:
        raise ValueError(f"seed_frame_idx={seed_frame_idx} is out of range for volume depth={data.shape[0]}.")

    coords = np.asarray(point_coords_xy, dtype=np.float32)
    labels = np.asarray(point_labels, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"point_coords_xy must have shape (N, 2), got {coords.shape}")
    if labels.ndim != 1 or labels.shape[0] != coords.shape[0]:
        raise ValueError(f"point_labels must have shape ({coords.shape[0]},), got {labels.shape}")
    if coords.shape[0] == 0:
        raise ValueError("propagate_volume_from_points requires at least one prompt point.")

    config_name = _normalize_config_name(str(resolved.config_file))
    resolved_device = _pick_device_local(resolved.device)
    predictor = build_sam2_video_predictor(config_name, str(resolved.checkpoint_file), device=resolved_device)
    propagated = np.zeros_like(data, dtype=np.int32)

    with TemporaryDirectory(prefix="sam2_stack_") as tmpdir:
        for frame_idx in range(data.shape[0]):
            rgb_slice = _prepare_image(data[frame_idx])
            Image.fromarray(rgb_slice).save(Path(tmpdir) / f"{frame_idx:05d}.jpg", format="JPEG", quality=95)

        inference_state = predictor.init_state(
            str(tmpdir),
            offload_video_to_cpu=(resolved_device != "cuda"),
            offload_state_to_cpu=(resolved_device != "cuda"),
        )
        predictor.reset_state(inference_state)
        predictor.add_new_points_or_box(
            inference_state,
            frame_idx=int(seed_frame_idx),
            obj_id=1,
            points=coords,
            labels=labels,
            clear_old_points=True,
            normalize_coords=True,
        )

        seen_frames: set[int] = set()
        for reverse in (False, True):
            for frame_idx, _obj_ids, video_res_masks in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=int(seed_frame_idx),
                reverse=reverse,
            ):
                propagated[int(frame_idx)] = _mask_from_video_output(video_res_masks)
                seen_frames.add(int(frame_idx))

    pos_count = int(np.count_nonzero(labels == 1))
    neg_count = int(np.count_nonzero(labels == 0))
    return (
        propagated,
        f"SAM2 propagation used device={resolved_device} seed_slice={seed_frame_idx} "
        f"frames={len(seen_frames)} pos_points={pos_count} neg_points={neg_count}.",
    )
