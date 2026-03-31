from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPTS = [
    "what can you do with my current layers?",
    "inspect the selected layer first, then recommend the next step",
    "what does Library do?",
    "what is the difference between Load and Test?",
    "compare all open images side by side in grid view",
    "show layers in grid with spacing 0",
    "use a built-in tool if possible; otherwise generate napari code",
    "preview threshold first for the selected image, then explain whether I should apply it",
    "measure the current mask",
    "how should I ask for a tool action versus runnable code?",
    "inspect the selected layer",
    "use layer: <layer_name>. inspect it and report its properties and recommendations",
    "apply CLAHE to the selected image with kernel_size 32, clip_limit 0.01, nbins 256",
    "preview threshold for the selected image",
    "apply threshold for dim objects",
    "explain first, then give runnable napari code for the current viewer",
    "generate a synthetic image in the current viewer with chosen dimensions, bit depth, and SNR",
    "for the selected layer, inspect it, explain the likely workflow, then choose the best built-in tool and tell me why",
    "look at all open layers, group them by likely role, then suggest a short analysis plan before doing anything",
    "for the current viewer, prefer built-in tools, but if none fit then generate safe napari code and explain the tradeoff",
    "open SAM2 Setup from Advanced and explain what each field means",
    "improve my prompt first, then answer in markdown with bullets and short sections",
]

DEFAULT_CODE_SNIPPETS = [
    {
        "title": "Demo Pack: EM 2D SNR Sweep",
        "code": """
import numpy as np
from scipy.ndimage import gaussian_filter


def make_em_slice(y=512, x=512, n_objects=26, seed=7):
    rng = np.random.default_rng(seed)
    image = np.zeros((y, x), dtype=np.float32)
    mask = np.zeros((y, x), dtype=np.uint8)

    yy, xx = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")

    background = gaussian_filter(rng.random((y, x)).astype(np.float32), sigma=18.0)
    background = (background - background.min()) / (background.max() - background.min() + 1e-8)
    image += 0.20 * background

    for _ in range(n_objects):
        cy = rng.integers(32, y - 32)
        cx = rng.integers(32, x - 32)
        ry = rng.uniform(12.0, 42.0)
        rx = rng.uniform(12.0, 42.0)
        angle = rng.uniform(0, np.pi)

        x_rot = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        y_rot = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        dist = (x_rot / rx) ** 2 + (y_rot / ry) ** 2

        obj = dist <= 1.0
        shell = np.exp(-((np.sqrt(np.maximum(dist, 1e-8)) - 1.0) ** 2) / 0.03).astype(np.float32)
        core = np.exp(-dist * rng.uniform(1.8, 3.2)).astype(np.float32)

        image += 0.55 * shell + 0.25 * core
        mask[obj] = 1

    image = gaussian_filter(image, sigma=1.2)
    image -= image.min()
    image /= image.max() + 1e-8
    return image.astype(np.float32), mask


def add_noise_for_snr(image, snr_db, seed):
    rng = np.random.default_rng(seed)
    signal_std = float(np.std(image))
    noise_std = signal_std / (10 ** (float(snr_db) / 20.0) + 1e-8)
    noisy = image + rng.normal(0.0, noise_std, image.shape).astype(np.float32)
    noisy = gaussian_filter(noisy, sigma=0.6)
    noisy = np.clip(noisy, 0.0, None)
    noisy -= noisy.min()
    noisy /= noisy.max() + 1e-8
    return noisy.astype(np.float32)


def compute():
    base, mask = make_em_slice()
    return {
        "em_2d_snr_low": add_noise_for_snr(base, 4.0, 11),
        "em_2d_snr_mid": add_noise_for_snr(base, 10.0, 12),
        "em_2d_snr_high": add_noise_for_snr(base, 18.0, 13),
        "em_2d_mask": mask,
    }


def apply_result(payload):
    viewer.add_image(payload["em_2d_snr_low"], name="em_2d_snr_low", colormap="gray")
    viewer.add_image(payload["em_2d_snr_mid"], name="em_2d_snr_mid", colormap="gray")
    viewer.add_image(payload["em_2d_snr_high"], name="em_2d_snr_high", colormap="gray")
    viewer.add_labels(payload["em_2d_mask"], name="em_2d_mask")
    print("Added layers: em_2d_snr_low, em_2d_snr_mid, em_2d_snr_high, em_2d_mask")


run_in_background(compute, apply_result, label="Generate EM 2D SNR sweep")
""".strip(),
        "tags": ["demo", "em", "grayscale", "2d", "snr", "labels"],
    },
    {
        "title": "Demo Pack: EM 3D SNR Sweep",
        "code": """
import numpy as np
from scipy.ndimage import gaussian_filter


def make_em_volume(
    z=30,
    y=256,
    x=256,
    n_objects=22,
    seed=7,
):
    rng = np.random.default_rng(seed)
    volume = np.zeros((z, y, x), dtype=np.float32)
    mask = np.zeros((z, y, x), dtype=np.uint8)

    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    background = gaussian_filter(rng.random((z, y, x)).astype(np.float32), sigma=(1.4, 10.0, 10.0))
    background = (background - background.min()) / (background.max() - background.min() + 1e-8)
    volume += 0.18 * background

    for _ in range(n_objects):
        cz = rng.integers(4, z - 4)
        cy = rng.integers(20, y - 20)
        cx = rng.integers(20, x - 20)
        rz = rng.uniform(2.0, 5.0)
        ry = rng.uniform(8.0, 20.0)
        rx = rng.uniform(8.0, 20.0)

        dist = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        obj = dist <= 1.0
        shell = np.exp(-((np.sqrt(np.maximum(dist, 1e-8)) - 1.0) ** 2) / 0.02).astype(np.float32)
        core = np.exp(-dist * rng.uniform(1.8, 3.0)).astype(np.float32)
        volume += 0.60 * shell + 0.18 * core
        mask[obj] = 1

    volume = gaussian_filter(volume, sigma=(0.8, 1.1, 1.1))
    volume -= volume.min()
    volume /= volume.max() + 1e-8
    return volume.astype(np.float32), mask


def add_noise_for_snr(image, snr_db, seed):
    rng = np.random.default_rng(seed)
    signal_std = float(np.std(image))
    noise_std = signal_std / (10 ** (float(snr_db) / 20.0) + 1e-8)
    noisy = image + rng.normal(0.0, noise_std, image.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, None)
    noisy -= noisy.min()
    noisy /= noisy.max() + 1e-8
    return noisy.astype(np.float32)


def compute():
    base, mask = make_em_volume()
    return {
        "em_3d_snr_low": add_noise_for_snr(base, 5.0, 21),
        "em_3d_snr_mid": add_noise_for_snr(base, 11.0, 22),
        "em_3d_snr_high": add_noise_for_snr(base, 18.0, 23),
        "em_3d_mask": mask,
    }


def apply_result(payload):
    viewer.add_image(payload["em_3d_snr_low"], name="em_3d_snr_low", colormap="gray")
    viewer.add_image(payload["em_3d_snr_mid"], name="em_3d_snr_mid", colormap="gray")
    viewer.add_image(payload["em_3d_snr_high"], name="em_3d_snr_high", colormap="gray")
    viewer.add_labels(payload["em_3d_mask"], name="em_3d_mask")
    print("Added layers: em_3d_snr_low, em_3d_snr_mid, em_3d_snr_high, em_3d_mask")


run_in_background(compute, apply_result, label="Generate EM 3D SNR sweep")
""".strip(),
        "tags": ["demo", "em", "grayscale", "3d", "snr", "labels"],
    },
    {
        "title": "Demo Pack: RGB Cells 2D SNR Sweep",
        "code": """
import numpy as np
from scipy.ndimage import gaussian_filter


def make_cells_rgb_2d(y=512, x=512, n_cells=20, seed=7):
    rng = np.random.default_rng(seed)
    rgb = np.zeros((y, x, 3), dtype=np.float32)
    labels = np.zeros((y, x), dtype=np.int32)
    yy, xx = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")

    for cell_id in range(1, n_cells + 1):
        cy = rng.integers(36, y - 36)
        cx = rng.integers(36, x - 36)
        ry = rng.uniform(18.0, 34.0)
        rx = rng.uniform(18.0, 34.0)
        dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        cell = dist <= 1.0
        membrane = np.exp(-((np.sqrt(np.maximum(dist, 1e-8)) - 1.0) ** 2) / 0.01).astype(np.float32)
        cytoplasm = np.exp(-dist * 2.0).astype(np.float32)

        ncy = cy + rng.uniform(-4.0, 4.0)
        ncx = cx + rng.uniform(-4.0, 4.0)
        nry = ry * rng.uniform(0.35, 0.55)
        nrx = rx * rng.uniform(0.35, 0.55)
        nuc_dist = ((yy - ncy) / nry) ** 2 + ((xx - ncx) / nrx) ** 2
        nucleus = np.exp(-nuc_dist * 2.6).astype(np.float32)

        puncta = np.zeros((y, x), dtype=np.float32)
        for _ in range(rng.integers(12, 24)):
            py = cy + rng.uniform(-ry * 0.7, ry * 0.7)
            px = cx + rng.uniform(-rx * 0.7, rx * 0.7)
            pry = rng.uniform(0.8, 2.0)
            prx = rng.uniform(0.8, 2.0)
            pd = ((yy - py) / pry) ** 2 + ((xx - px) / prx) ** 2
            puncta += np.exp(-pd * rng.uniform(3.0, 7.0)).astype(np.float32)

        rgb[..., 0] += 0.75 * membrane
        rgb[..., 1] += 0.22 * cytoplasm + 0.18 * puncta * cell.astype(np.float32)
        rgb[..., 2] += 0.88 * nucleus
        labels[cell] = cell_id

    for c in range(3):
        rgb[..., c] = gaussian_filter(rgb[..., c], sigma=1.0)
        rgb[..., c] -= rgb[..., c].min()
        rgb[..., c] /= rgb[..., c].max() + 1e-8
    return rgb.astype(np.float32), labels


def add_noise_for_snr(rgb, snr_db, seed):
    rng = np.random.default_rng(seed)
    out = rgb.copy()
    for c in range(3):
        signal_std = float(np.std(out[..., c]))
        noise_std = signal_std / (10 ** (float(snr_db) / 20.0) + 1e-8)
        out[..., c] += rng.normal(0.0, noise_std, out[..., c].shape).astype(np.float32)
        out[..., c] = np.clip(out[..., c], 0.0, None)
        out[..., c] -= out[..., c].min()
        out[..., c] /= out[..., c].max() + 1e-8
    return out.astype(np.float32)


def compute():
    base, labels = make_cells_rgb_2d()
    return {
        "rgb_cells_2d_snr_low": add_noise_for_snr(base, 5.0, 31),
        "rgb_cells_2d_snr_mid": add_noise_for_snr(base, 11.0, 32),
        "rgb_cells_2d_snr_high": add_noise_for_snr(base, 18.0, 33),
        "rgb_cells_2d_labels": labels,
    }


def apply_result(payload):
    viewer.add_image(payload["rgb_cells_2d_snr_low"], name="rgb_cells_2d_snr_low", rgb=True)
    viewer.add_image(payload["rgb_cells_2d_snr_mid"], name="rgb_cells_2d_snr_mid", rgb=True)
    viewer.add_image(payload["rgb_cells_2d_snr_high"], name="rgb_cells_2d_snr_high", rgb=True)
    viewer.add_labels(payload["rgb_cells_2d_labels"], name="rgb_cells_2d_labels")
    print("Added layers: rgb_cells_2d_snr_low, rgb_cells_2d_snr_mid, rgb_cells_2d_snr_high, rgb_cells_2d_labels")


run_in_background(compute, apply_result, label="Generate RGB cells 2D SNR sweep")
""".strip(),
        "tags": ["demo", "rgb", "fluorescent", "2d", "snr", "labels"],
    },
    {
        "title": "Demo Pack: RGB Cells 3D SNR Sweep",
        "code": """
import numpy as np
from scipy.ndimage import gaussian_filter


def make_cells_rgb_volume(z=30, y=256, x=256, n_cells=18, seed=7):
    rng = np.random.default_rng(seed)
    vol = np.zeros((z, y, x, 3), dtype=np.float32)
    labels = np.zeros((z, y, x), dtype=np.int32)
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")

    for cell_id in range(1, n_cells + 1):
        cz = rng.integers(4, z - 4)
        cy = rng.integers(24, y - 24)
        cx = rng.integers(24, x - 24)
        rz = rng.uniform(2.0, 5.0)
        ry = rng.uniform(10.0, 20.0)
        rx = rng.uniform(10.0, 20.0)
        dist = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        cell = dist <= 1.0
        membrane = np.exp(-((np.sqrt(np.maximum(dist, 1e-8)) - 1.0) ** 2) / 0.008).astype(np.float32)
        cytoplasm = np.exp(-dist * 1.9).astype(np.float32)

        ncz = cz + rng.uniform(-1.0, 1.0)
        ncy = cy + rng.uniform(-3.0, 3.0)
        ncx = cx + rng.uniform(-3.0, 3.0)
        nrz = rz * rng.uniform(0.35, 0.55)
        nry = ry * rng.uniform(0.35, 0.55)
        nrx = rx * rng.uniform(0.35, 0.55)
        nuc_dist = ((zz - ncz) / nrz) ** 2 + ((yy - ncy) / nry) ** 2 + ((xx - ncx) / nrx) ** 2
        nucleus = np.exp(-nuc_dist * 2.7).astype(np.float32)

        puncta = np.zeros((z, y, x), dtype=np.float32)
        for _ in range(rng.integers(10, 20)):
            pz = cz + rng.uniform(-rz * 0.7, rz * 0.7)
            py = cy + rng.uniform(-ry * 0.7, ry * 0.7)
            px = cx + rng.uniform(-rx * 0.7, rx * 0.7)
            prz = rng.uniform(0.4, 1.0)
            pry = rng.uniform(0.8, 1.8)
            prx = rng.uniform(0.8, 1.8)
            pd = ((zz - pz) / prz) ** 2 + ((yy - py) / pry) ** 2 + ((xx - px) / prx) ** 2
            puncta += np.exp(-pd * rng.uniform(3.0, 6.5)).astype(np.float32)

        vol[..., 0] += 0.72 * membrane
        vol[..., 1] += 0.18 * cytoplasm + 0.16 * puncta * cell.astype(np.float32)
        vol[..., 2] += 0.86 * nucleus
        labels[cell] = cell_id

    for c in range(3):
        vol[..., c] = gaussian_filter(vol[..., c], sigma=(0.7, 0.9, 0.9))
        vol[..., c] -= vol[..., c].min()
        vol[..., c] /= vol[..., c].max() + 1e-8
    return vol.astype(np.float32), labels


def add_noise_for_snr(rgb, snr_db, seed):
    rng = np.random.default_rng(seed)
    out = rgb.copy()
    for c in range(3):
        signal_std = float(np.std(out[..., c]))
        noise_std = signal_std / (10 ** (float(snr_db) / 20.0) + 1e-8)
        out[..., c] += rng.normal(0.0, noise_std, out[..., c].shape).astype(np.float32)
        out[..., c] = np.clip(out[..., c], 0.0, None)
        out[..., c] -= out[..., c].min()
        out[..., c] /= out[..., c].max() + 1e-8
    return out.astype(np.float32)


def compute():
    base, labels = make_cells_rgb_volume()
    return {
        "rgb_cells_3d_snr_low": add_noise_for_snr(base, 5.0, 41),
        "rgb_cells_3d_snr_mid": add_noise_for_snr(base, 11.0, 42),
        "rgb_cells_3d_snr_high": add_noise_for_snr(base, 18.0, 43),
        "rgb_cells_3d_labels": labels,
    }


def apply_result(payload):
    viewer.add_image(payload["rgb_cells_3d_snr_low"], name="rgb_cells_3d_snr_low", rgb=True)
    viewer.add_image(payload["rgb_cells_3d_snr_mid"], name="rgb_cells_3d_snr_mid", rgb=True)
    viewer.add_image(payload["rgb_cells_3d_snr_high"], name="rgb_cells_3d_snr_high", rgb=True)
    viewer.add_labels(payload["rgb_cells_3d_labels"], name="rgb_cells_3d_labels")
    print("Added layers: rgb_cells_3d_snr_low, rgb_cells_3d_snr_mid, rgb_cells_3d_snr_high, rgb_cells_3d_labels")


run_in_background(compute, apply_result, label="Generate RGB cells 3D SNR sweep")
""".strip(),
        "tags": ["demo", "rgb", "fluorescent", "3d", "snr", "labels"],
    },
    {
        "title": "Demo Pack: Messy Masks 2D/3D",
        "code": """
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure


def make_clean_mask_2d(y=512, x=512, n_objects=10, seed=7):
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")
    mask = np.zeros((y, x), dtype=np.uint8)
    for _ in range(n_objects):
        cy = rng.integers(40, y - 40)
        cx = rng.integers(40, x - 40)
        ry = rng.uniform(18.0, 42.0)
        rx = rng.uniform(18.0, 42.0)
        dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        mask |= (dist <= 1.0).astype(np.uint8)
    return mask


def make_clean_mask_3d(z=30, y=256, x=256, n_objects=8, seed=11):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    mask = np.zeros((z, y, x), dtype=np.uint8)
    for _ in range(n_objects):
        cz = rng.integers(4, z - 4)
        cy = rng.integers(20, y - 20)
        cx = rng.integers(20, x - 20)
        rz = rng.uniform(2.0, 5.0)
        ry = rng.uniform(10.0, 20.0)
        rx = rng.uniform(10.0, 20.0)
        dist = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        mask |= (dist <= 1.0).astype(np.uint8)
    return mask


def make_messy_mask(mask, seed):
    rng = np.random.default_rng(seed)
    ndim = mask.ndim
    structure = generate_binary_structure(ndim, 1)
    messy = mask.astype(bool)

    messy = binary_dilation(messy, structure=structure, iterations=1)
    messy = binary_erosion(messy, structure=structure, iterations=1)

    foreground = np.argwhere(messy)
    if len(foreground):
        n_holes = max(4, len(foreground) // 3000)
        picks = foreground[rng.choice(len(foreground), size=min(n_holes, len(foreground)), replace=False)]
        for coord in picks:
            messy[tuple(coord)] = False

    noise = np.zeros_like(messy, dtype=bool)
    n_speckles = max(40, messy.size // 4000)
    coords = tuple(rng.integers(0, size, size=n_speckles) for size in messy.shape)
    noise[coords] = True
    messy |= noise

    return messy.astype(np.uint8)


def compute():
    clean_2d = make_clean_mask_2d()
    clean_3d = make_clean_mask_3d()
    messy_2d = make_messy_mask(clean_2d, seed=21)
    messy_3d = make_messy_mask(clean_3d, seed=22)
    filled_2d = binary_fill_holes(messy_2d > 0).astype(np.uint8)
    filled_3d = binary_fill_holes(messy_3d > 0).astype(np.uint8)
    return {
        "mask_clean_2d": clean_2d,
        "mask_messy_2d": messy_2d,
        "mask_filled_target_2d": filled_2d,
        "mask_clean_3d": clean_3d,
        "mask_messy_3d": messy_3d,
        "mask_filled_target_3d": filled_3d,
    }


def apply_result(payload):
    viewer.add_labels(payload["mask_clean_2d"], name="mask_clean_2d")
    viewer.add_labels(payload["mask_messy_2d"], name="mask_messy_2d")
    viewer.add_labels(payload["mask_filled_target_2d"], name="mask_filled_target_2d")
    viewer.add_labels(payload["mask_clean_3d"], name="mask_clean_3d")
    viewer.add_labels(payload["mask_messy_3d"], name="mask_messy_3d")
    viewer.add_labels(payload["mask_filled_target_3d"], name="mask_filled_target_3d")
    print("Added layers: mask_clean_2d, mask_messy_2d, mask_filled_target_2d, mask_clean_3d, mask_messy_3d, mask_filled_target_3d")


run_in_background(compute, apply_result, label="Generate messy masks demo pack")
""".strip(),
        "tags": ["demo", "labels", "cleanup", "2d", "3d", "mask"],
    },
]


def prompt_library_path() -> Path:
    return Path.home() / ".napari-chat-assistant" / "prompt_library.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def prompt_title(prompt_text: str, max_length: int = 64) -> str:
    text = " ".join(str(prompt_text or "").strip().split())
    if not text:
        return "Untitled Prompt"
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def stable_item_id(kind: str, content: str) -> str:
    digest = hashlib.sha1(str(content or "").strip().encode("utf-8")).hexdigest()[:12]
    return f"{kind}_{digest}"


def normalize_tags(values) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",")]
    for value in values or []:
        tag = " ".join(str(value or "").strip().split())
        if not tag:
            continue
        lowered = tag.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        tags.append(tag)
    return tags


def normalize_record(record: dict, *, source: str) -> dict:
    prompt = str(record.get("prompt", "")).strip()
    if not prompt:
        return {}
    return {
        "id": str(record.get("id") or stable_item_id("prompt", prompt)).strip(),
        "title": str(record.get("title") or prompt_title(prompt)).strip(),
        "prompt": prompt,
        "tags": normalize_tags(record.get("tags")),
        "source": source,
        "updated_at": str(record.get("updated_at") or utc_now_iso()),
    }


def normalize_code_record(record: dict, *, source: str) -> dict:
    code = str(record.get("code", "")).strip()
    if not code:
        return {}
    return {
        "id": str(record.get("id") or stable_item_id("code", code)).strip(),
        "title": str(record.get("title") or prompt_title(code)).strip(),
        "code": code,
        "tags": normalize_tags(record.get("tags")),
        "source": source,
        "updated_at": str(record.get("updated_at") or utc_now_iso()),
    }


def normalize_prompt_list(values: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        prompt = str(value or "").strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    return prompts


def default_prompt_records() -> list[dict]:
    return [
        {
            "id": stable_item_id("prompt", text),
            "title": prompt_title(text),
            "prompt": text,
            "tags": [],
            "source": "built_in",
            "updated_at": "",
        }
        for text in DEFAULT_PROMPTS
    ]


def default_code_records() -> list[dict]:
    return [
        {
            "id": stable_item_id("code", item["code"]),
            "title": str(item.get("title") or prompt_title(item["code"])).strip(),
            "code": str(item["code"]).strip(),
            "tags": normalize_tags(item.get("tags")),
            "source": "built_in",
            "updated_at": "",
        }
        for item in DEFAULT_CODE_SNIPPETS
        if str(item.get("code", "")).strip()
    ]


def load_prompt_library() -> dict:
    path = prompt_library_path()
    if not path.exists():
        return {
            "saved": [],
            "recent": [],
            "pinned_prompts": [],
            "hidden_built_in": [],
            "code_saved": [],
            "code_recent": [],
            "pinned_codes": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "saved": [],
            "recent": [],
            "pinned_prompts": [],
            "hidden_built_in": [],
            "code_saved": [],
            "code_recent": [],
            "pinned_codes": [],
        }
    legacy_saved = payload.get("saved", [])
    saved = [normalize_record(item, source="saved") for item in payload.get("saved", [])]
    recent = [normalize_record(item, source="recent") for item in payload.get("recent", [])]
    code_saved = [normalize_code_record(item, source="saved") for item in payload.get("code_saved", [])]
    code_recent = [normalize_code_record(item, source="recent") for item in payload.get("code_recent", [])]
    pinned_prompts = normalize_prompt_list(payload.get("pinned_prompts"))
    pinned_codes = normalize_prompt_list(payload.get("pinned_codes"))
    if not pinned_prompts:
        pinned_prompts = normalize_prompt_list(
            str(item.get("prompt", "")).strip() for item in legacy_saved if item.get("pinned", False)
        )
    return {
        "saved": [item for item in saved if item],
        "recent": [item for item in recent if item],
        "pinned_prompts": pinned_prompts,
        "hidden_built_in": normalize_prompt_list(payload.get("hidden_built_in")),
        "code_saved": [item for item in code_saved if item],
        "code_recent": [item for item in code_recent if item],
        "pinned_codes": pinned_codes,
    }


def save_prompt_library(data: dict) -> None:
    path = prompt_library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved": [
            {
                "title": item["title"],
                "id": item.get("id") or stable_item_id("prompt", item["prompt"]),
                "prompt": item["prompt"],
                "tags": normalize_tags(item.get("tags")),
                "updated_at": item.get("updated_at", utc_now_iso()),
            }
            for item in data.get("saved", [])
            if item.get("prompt")
        ],
        "recent": [
            {
                "title": item["title"],
                "id": item.get("id") or stable_item_id("prompt", item["prompt"]),
                "prompt": item["prompt"],
                "tags": normalize_tags(item.get("tags")),
                "updated_at": item.get("updated_at", utc_now_iso()),
            }
            for item in data.get("recent", [])
            if item.get("prompt")
        ],
        "pinned_prompts": normalize_prompt_list(data.get("pinned_prompts")),
        "hidden_built_in": normalize_prompt_list(data.get("hidden_built_in")),
        "code_saved": [
            {
                "title": item["title"],
                "id": item.get("id") or stable_item_id("code", item["code"]),
                "code": item["code"],
                "tags": normalize_tags(item.get("tags")),
                "updated_at": item.get("updated_at", utc_now_iso()),
            }
            for item in data.get("code_saved", [])
            if item.get("code")
        ],
        "code_recent": [
            {
                "title": item["title"],
                "id": item.get("id") or stable_item_id("code", item["code"]),
                "code": item["code"],
                "tags": normalize_tags(item.get("tags")),
                "updated_at": item.get("updated_at", utc_now_iso()),
            }
            for item in data.get("code_recent", [])
            if item.get("code")
        ],
        "pinned_codes": normalize_prompt_list(data.get("pinned_codes")),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_recent_prompt(data: dict, prompt_text: str, limit: int = 20) -> dict:
    prompt = str(prompt_text or "").strip()
    if not prompt:
        return data
    now = utc_now_iso()
    existing = next((item for item in data.get("recent", []) if item.get("prompt") == prompt), None)
    recent = [item for item in data.get("recent", []) if item.get("prompt") != prompt]
    recent.insert(
        0,
        {
            "id": stable_item_id("prompt", prompt),
            "title": str((existing or {}).get("title") or prompt_title(prompt)).strip(),
            "prompt": prompt,
            "tags": normalize_tags((existing or {}).get("tags")),
            "source": "recent",
            "updated_at": now,
        },
    )
    data["recent"] = recent[:limit]
    return data


def upsert_saved_prompt(data: dict, prompt_text: str, *, pin: bool | None = None) -> dict:
    prompt = str(prompt_text or "").strip()
    if not prompt:
        return data
    now = utc_now_iso()
    saved = list(data.get("saved", []))
    existing = next((item for item in saved if item.get("prompt") == prompt), None)
    remaining = []
    for item in saved:
        if item.get("prompt") == prompt:
            continue
        remaining.append(item)
    record = {
        "id": stable_item_id("prompt", prompt),
        "title": str((existing or {}).get("title") or prompt_title(prompt)).strip(),
        "prompt": prompt,
        "tags": normalize_tags((existing or {}).get("tags")),
        "source": "saved",
        "updated_at": now,
    }
    remaining.insert(0, record)
    data["saved"] = remaining
    if pin is not None:
        set_prompt_pinned(data, prompt, bool(pin))
    return data


def upsert_recent_code(data: dict, code_text: str, limit: int = 20) -> dict:
    code = str(code_text or "").strip()
    if not code:
        return data
    now = utc_now_iso()
    existing = next((item for item in data.get("code_recent", []) if item.get("code") == code), None)
    recent = [item for item in data.get("code_recent", []) if item.get("code") != code]
    recent.insert(
        0,
        {
            "id": stable_item_id("code", code),
            "title": str((existing or {}).get("title") or prompt_title(code)).strip(),
            "code": code,
            "tags": normalize_tags((existing or {}).get("tags")),
            "source": "recent",
            "updated_at": now,
        },
    )
    data["code_recent"] = recent[:limit]
    return data


def upsert_saved_code(data: dict, code_text: str, *, pin: bool | None = None) -> dict:
    code = str(code_text or "").strip()
    if not code:
        return data
    now = utc_now_iso()
    saved = list(data.get("code_saved", []))
    existing = next((item for item in saved if item.get("code") == code), None)
    remaining = [item for item in saved if item.get("code") != code]
    record = {
        "id": stable_item_id("code", code),
        "title": str((existing or {}).get("title") or prompt_title(code)).strip(),
        "code": code,
        "tags": normalize_tags((existing or {}).get("tags")),
        "source": "saved",
        "updated_at": now,
    }
    remaining.insert(0, record)
    data["code_saved"] = remaining
    if pin is not None:
        set_code_pinned(data, code, bool(pin))
    return data


def set_saved_prompt_pinned(data: dict, prompt_text: str, pinned: bool) -> dict:
    return set_prompt_pinned(data, prompt_text, pinned)


def set_prompt_pinned(data: dict, prompt_text: str, pinned: bool) -> dict:
    prompt = str(prompt_text or "").strip()
    pinned_prompts = normalize_prompt_list(data.get("pinned_prompts"))
    if pinned:
        if prompt and prompt not in pinned_prompts:
            pinned_prompts.insert(0, prompt)
    else:
        pinned_prompts = [item for item in pinned_prompts if item != prompt]
    data["pinned_prompts"] = pinned_prompts
    return data


def set_code_pinned(data: dict, code_text: str, pinned: bool) -> dict:
    code = str(code_text or "").strip()
    pinned_codes = normalize_prompt_list(data.get("pinned_codes"))
    if pinned:
        if code and code not in pinned_codes:
            pinned_codes.insert(0, code)
    else:
        pinned_codes = [item for item in pinned_codes if item != code]
    data["pinned_codes"] = pinned_codes
    return data


def remove_saved_prompt(data: dict, prompt_text: str) -> dict:
    return remove_prompt_record(data, prompt_text, source="saved")


def remove_recent_prompt(data: dict, prompt_text: str) -> dict:
    return remove_prompt_record(data, prompt_text, source="recent")


def remove_prompt_record(data: dict, prompt_text: str, *, source: str) -> dict:
    prompt = str(prompt_text or "").strip()
    if source == "saved":
        data["saved"] = [item for item in data.get("saved", []) if item.get("prompt") != prompt]
    elif source == "recent":
        data["recent"] = [item for item in data.get("recent", []) if item.get("prompt") != prompt]
    elif source == "built_in":
        hidden_built_in = normalize_prompt_list(data.get("hidden_built_in"))
        if prompt and prompt not in hidden_built_in:
            hidden_built_in.append(prompt)
        data["hidden_built_in"] = hidden_built_in
    return set_prompt_pinned(data, prompt, False)


def remove_code_record(data: dict, code_text: str, *, source: str) -> dict:
    code = str(code_text or "").strip()
    if source == "saved":
        data["code_saved"] = [item for item in data.get("code_saved", []) if item.get("code") != code]
    elif source == "recent":
        data["code_recent"] = [item for item in data.get("code_recent", []) if item.get("code") != code]
    return set_code_pinned(data, code, False)


def clear_prompt_library(data: dict, *, keep_saved: bool = True, keep_pinned: bool = True) -> dict:
    pinned_prompts = normalize_prompt_list(data.get("pinned_prompts")) if keep_pinned else []
    hidden_built_in = normalize_prompt_list(data.get("hidden_built_in"))
    hidden_built_in = list(hidden_built_in)
    for record in default_prompt_records():
        prompt = record["prompt"]
        if prompt in pinned_prompts:
            continue
        if prompt not in hidden_built_in:
            hidden_built_in.append(prompt)
    data["recent"] = []
    data["saved"] = list(data.get("saved", [])) if keep_saved else []
    data["pinned_prompts"] = pinned_prompts
    data["hidden_built_in"] = hidden_built_in
    return data


def clear_code_library(data: dict, *, keep_saved: bool = True, keep_pinned: bool = True) -> dict:
    pinned_codes = normalize_prompt_list(data.get("pinned_codes")) if keep_pinned else []
    data["code_recent"] = []
    data["code_saved"] = list(data.get("code_saved", [])) if keep_saved else []
    if keep_pinned:
        data["code_saved"] = [item for item in data["code_saved"] if item.get("code") in pinned_codes or keep_saved]
    data["pinned_codes"] = pinned_codes
    return data


def merged_prompt_records(data: dict) -> list[dict]:
    pinned_prompts = set(normalize_prompt_list(data.get("pinned_prompts")))
    hidden_built_in = set(normalize_prompt_list(data.get("hidden_built_in")))
    saved = sorted(data.get("saved", []), key=lambda item: item.get("updated_at", ""), reverse=True)
    recent = sorted(data.get("recent", []), key=lambda item: item.get("updated_at", ""), reverse=True)
    built_in = [item for item in default_prompt_records() if item.get("prompt") not in hidden_built_in]

    merged: list[dict] = []
    seen: set[str] = set()
    for record in [*saved, *recent, *built_in]:
        prompt = str(record.get("prompt", "")).strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        merged.append({**record, "pinned": prompt in pinned_prompts})

    pinned = [item for item in merged if item.get("pinned", False)]
    unpinned_saved = [item for item in merged if item.get("source") == "saved" and not item.get("pinned", False)]
    unpinned_recent = [item for item in merged if item.get("source") == "recent" and not item.get("pinned", False)]
    unpinned_built_in = [item for item in merged if item.get("source") == "built_in" and not item.get("pinned", False)]
    return pinned + unpinned_saved + unpinned_recent + unpinned_built_in


def merged_code_records(data: dict) -> list[dict]:
    pinned_codes = set(normalize_prompt_list(data.get("pinned_codes")))
    saved = sorted(data.get("code_saved", []), key=lambda item: item.get("updated_at", ""), reverse=True)
    recent = sorted(data.get("code_recent", []), key=lambda item: item.get("updated_at", ""), reverse=True)
    built_in = default_code_records()

    merged: list[dict] = []
    seen: set[str] = set()
    for record in [*saved, *recent]:
        code = str(record.get("code", "")).strip()
        if not code or code in seen:
            continue
        seen.add(code)
        merged.append({**record, "pinned": code in pinned_codes})

    # Keep built-in demo/readout entries visible even when the same code also
    # appears in recent or saved history, so the built-in catalog remains
    # stable after users load or run those snippets.
    for record in built_in:
        code = str(record.get("code", "")).strip()
        if not code:
            continue
        merged.append({**record, "pinned": code in pinned_codes})

    pinned = [item for item in merged if item.get("pinned", False)]
    unpinned_saved = [item for item in merged if item.get("source") == "saved" and not item.get("pinned", False)]
    unpinned_recent = [item for item in merged if item.get("source") == "recent" and not item.get("pinned", False)]
    unpinned_built_in = [item for item in merged if item.get("source") == "built_in" and not item.get("pinned", False)]
    return pinned + unpinned_saved + unpinned_recent + unpinned_built_in


def update_record_title(data: dict, *, kind: str, item_id: str, title: str) -> dict:
    clean_title = " ".join(str(title or "").strip().split())
    if not clean_title or not item_id:
        return data
    keys = ("saved", "recent") if kind == "prompt" else ("code_saved", "code_recent")
    for key in keys:
        for item in data.get(key, []):
            if str(item.get("id", "")).strip() == item_id:
                item["title"] = clean_title
                item["updated_at"] = utc_now_iso()
    return data


def update_record_tags(data: dict, *, kind: str, item_id: str, tags) -> dict:
    if not item_id:
        return data
    clean_tags = normalize_tags(tags)
    keys = ("saved", "recent") if kind == "prompt" else ("code_saved", "code_recent")
    for key in keys:
        for item in data.get(key, []):
            if str(item.get("id", "")).strip() == item_id:
                item["tags"] = clean_tags
                item["updated_at"] = utc_now_iso()
    return data
