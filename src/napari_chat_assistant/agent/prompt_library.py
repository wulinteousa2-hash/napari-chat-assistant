from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPTS = [
    "show me my layers",
    "inspect the selected layer",
    "apply CLAHE to the selected EM image with kernel_size 32, clip_limit 0.01, nbins 256",
    "apply CLAHE to all open EM images with kernel_size 64, clip_limit 0.02, nbins 512",
    "preview threshold for the selected image",
    "apply threshold for dim objects",
    "measure the current mask",
    "give me QtConsole code to print the selected layer name and shape",
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


def normalize_record(record: dict, *, source: str) -> dict:
    prompt = str(record.get("prompt", "")).strip()
    if not prompt:
        return {}
    return {
        "title": str(record.get("title") or prompt_title(prompt)).strip(),
        "prompt": prompt,
        "pinned": bool(record.get("pinned", False)),
        "source": source,
        "updated_at": str(record.get("updated_at") or utc_now_iso()),
    }


def default_prompt_records() -> list[dict]:
    return [
        {
            "title": prompt_title(text),
            "prompt": text,
            "pinned": False,
            "source": "built_in",
            "updated_at": "",
        }
        for text in DEFAULT_PROMPTS
    ]


def load_prompt_library() -> dict:
    path = prompt_library_path()
    if not path.exists():
        return {"saved": [], "recent": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"saved": [], "recent": []}
    saved = [normalize_record(item, source="saved") for item in payload.get("saved", [])]
    recent = [normalize_record(item, source="recent") for item in payload.get("recent", [])]
    return {
        "saved": [item for item in saved if item],
        "recent": [item for item in recent if item],
    }


def save_prompt_library(data: dict) -> None:
    path = prompt_library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved": [
            {
                "title": item["title"],
                "prompt": item["prompt"],
                "pinned": bool(item.get("pinned", False)),
                "updated_at": item.get("updated_at", utc_now_iso()),
            }
            for item in data.get("saved", [])
            if item.get("prompt")
        ],
        "recent": [
            {
                "title": item["title"],
                "prompt": item["prompt"],
                "updated_at": item.get("updated_at", utc_now_iso()),
            }
            for item in data.get("recent", [])
            if item.get("prompt")
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_recent_prompt(data: dict, prompt_text: str, limit: int = 20) -> dict:
    prompt = str(prompt_text or "").strip()
    if not prompt:
        return data
    now = utc_now_iso()
    recent = [item for item in data.get("recent", []) if item.get("prompt") != prompt]
    recent.insert(
        0,
        {
            "title": prompt_title(prompt),
            "prompt": prompt,
            "source": "recent",
            "pinned": False,
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
    existing = None
    remaining = []
    for item in saved:
        if item.get("prompt") == prompt and existing is None:
            existing = item
            continue
        remaining.append(item)
    record = {
        "title": prompt_title(prompt),
        "prompt": prompt,
        "source": "saved",
        "pinned": bool(existing and existing.get("pinned", False)) if pin is None else bool(pin),
        "updated_at": now,
    }
    remaining.insert(0, record)
    data["saved"] = remaining
    return data


def set_saved_prompt_pinned(data: dict, prompt_text: str, pinned: bool) -> dict:
    prompt = str(prompt_text or "").strip()
    saved = list(data.get("saved", []))
    for item in saved:
        if item.get("prompt") == prompt:
            item["pinned"] = bool(pinned)
            item["updated_at"] = utc_now_iso()
            break
    data["saved"] = saved
    return data


def remove_saved_prompt(data: dict, prompt_text: str) -> dict:
    prompt = str(prompt_text or "").strip()
    data["saved"] = [item for item in data.get("saved", []) if item.get("prompt") != prompt]
    return data


def merged_prompt_records(data: dict) -> list[dict]:
    built_in = default_prompt_records()
    saved = sorted(data.get("saved", []), key=lambda item: item.get("updated_at", ""), reverse=True)
    pinned = [item for item in saved if item.get("pinned", False)]
    unpinned = [item for item in saved if not item.get("pinned", False)]
    recent = data.get("recent", [])
    return pinned + unpinned + recent + built_in
