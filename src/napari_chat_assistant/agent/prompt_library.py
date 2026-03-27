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
            "title": prompt_title(text),
            "prompt": text,
            "source": "built_in",
            "updated_at": "",
        }
        for text in DEFAULT_PROMPTS
    ]


def load_prompt_library() -> dict:
    path = prompt_library_path()
    if not path.exists():
        return {"saved": [], "recent": [], "pinned_prompts": [], "hidden_built_in": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"saved": [], "recent": [], "pinned_prompts": [], "hidden_built_in": []}
    legacy_saved = payload.get("saved", [])
    saved = [normalize_record(item, source="saved") for item in payload.get("saved", [])]
    recent = [normalize_record(item, source="recent") for item in payload.get("recent", [])]
    pinned_prompts = normalize_prompt_list(payload.get("pinned_prompts"))
    if not pinned_prompts:
        pinned_prompts = normalize_prompt_list(
            str(item.get("prompt", "")).strip() for item in legacy_saved if item.get("pinned", False)
        )
    return {
        "saved": [item for item in saved if item],
        "recent": [item for item in recent if item],
        "pinned_prompts": pinned_prompts,
        "hidden_built_in": normalize_prompt_list(payload.get("hidden_built_in")),
    }


def save_prompt_library(data: dict) -> None:
    path = prompt_library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved": [
            {
                "title": item["title"],
                "prompt": item["prompt"],
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
        "pinned_prompts": normalize_prompt_list(data.get("pinned_prompts")),
        "hidden_built_in": normalize_prompt_list(data.get("hidden_built_in")),
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
        "updated_at": now,
    }
    remaining.insert(0, record)
    data["saved"] = remaining
    if pin is not None:
        set_prompt_pinned(data, prompt, bool(pin))
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
