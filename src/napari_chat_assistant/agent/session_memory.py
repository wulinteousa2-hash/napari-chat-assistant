from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


MEMORY_STATES = {"provisional", "approved", "rejected"}
MEMORY_TARGET_TYPES = {"classification", "recommendation", "tool_result", "code_result", "workflow_preference"}


def session_memory_path() -> Path:
    return Path.home() / ".napari-chat-assistant" / "session_memory.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_session_memory() -> dict:
    return {"session_goal": "", "active_dataset_focus": "", "items": []}


def load_session_memory() -> dict:
    path = session_memory_path()
    if not path.exists():
        return empty_session_memory()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return empty_session_memory()
    return normalize_session_memory(payload)


def save_session_memory(data: dict) -> None:
    path = session_memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prune_session_memory(normalize_session_memory(data)), indent=2), encoding="utf-8")


def normalize_session_memory(data: dict) -> dict:
    items = [normalize_memory_item(item) for item in data.get("items", [])]
    return {
        "session_goal": str(data.get("session_goal", "")).strip(),
        "active_dataset_focus": str(data.get("active_dataset_focus", "")).strip(),
        "items": [item for item in items if item],
    }


def normalize_memory_item(item: dict) -> dict:
    state = str(item.get("state", "provisional")).strip().lower()
    if state not in MEMORY_STATES:
        state = "provisional"
    target_type = str(item.get("target_type", "")).strip().lower()
    if target_type not in MEMORY_TARGET_TYPES:
        return {}
    summary = " ".join(str(item.get("summary", "")).split()).strip()
    if not summary:
        return {}
    return {
        "id": str(item.get("id") or uuid4()),
        "target_type": target_type,
        "target_layer": str(item.get("target_layer", "")).strip(),
        "viewer_fingerprint": normalize_viewer_fingerprint(item.get("viewer_fingerprint", {})),
        "summary": summary,
        "source": str(item.get("source", "assistant")).strip() or "assistant",
        "state": state,
        "created_at": str(item.get("created_at") or utc_now_iso()),
        "updated_at": str(item.get("updated_at") or utc_now_iso()),
    }


def normalize_viewer_fingerprint(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}
    return {
        "layer_class": str(data.get("layer_class", "")).strip(),
        "shape": list(data.get("shape")) if isinstance(data.get("shape"), list) else None,
        "dtype": str(data.get("dtype", "")).strip() or None,
        "semantic_type": str(data.get("semantic_type", "")).strip() or None,
    }


def make_viewer_fingerprint(profile: dict | None) -> dict:
    if not isinstance(profile, dict):
        return {}
    shape = profile.get("shape")
    return normalize_viewer_fingerprint(
        {
            "layer_class": profile.get("layer_class", ""),
            "shape": shape if isinstance(shape, list) else None,
            "dtype": profile.get("dtype"),
            "semantic_type": profile.get("semantic_type"),
        }
    )


def add_memory_item(
    data: dict,
    *,
    target_type: str,
    summary: str,
    target_layer: str = "",
    viewer_fingerprint: dict | None = None,
    source: str = "assistant",
    state: str = "provisional",
) -> tuple[dict, str | None]:
    if target_type not in MEMORY_TARGET_TYPES:
        return data, None
    item = normalize_memory_item(
        {
            "target_type": target_type,
            "summary": summary,
            "target_layer": target_layer,
            "viewer_fingerprint": viewer_fingerprint or {},
            "source": source,
            "state": state,
        }
    )
    if not item:
        return data, None
    items = list(data.get("items", []))
    items.insert(0, item)
    data["items"] = items
    return prune_session_memory(data), item["id"]


def update_session_goal(data: dict, text: str) -> dict:
    clean = " ".join(str(text or "").split()).strip()
    if clean:
        data["session_goal"] = clean[:160]
    return data


def set_active_dataset_focus(data: dict, layer_name: str) -> dict:
    data["active_dataset_focus"] = str(layer_name or "").strip()
    return data


def approve_memory_item(data: dict, item_id: str) -> dict:
    return _set_item_state(data, item_id, "approved")


def reject_memory_item(data: dict, item_id: str) -> dict:
    return _set_item_state(data, item_id, "rejected")


def reject_items(data: dict, item_ids: list[str]) -> dict:
    for item_id in item_ids:
        data = reject_memory_item(data, item_id)
    return data


def approve_items(data: dict, item_ids: list[str]) -> dict:
    for item_id in item_ids:
        data = approve_memory_item(data, item_id)
    return data


def promote_from_user_turn(data: dict, user_message: str, selected_profile: dict | None) -> tuple[dict, list[str]]:
    text = " ".join(str(user_message or "").strip().lower().split())
    if not text:
        return data, []
    approved: list[str] = []
    if _is_explicit_negative(text):
        return data, approved
    candidates = [
        item for item in data.get("items", [])
        if item.get("state") == "provisional" and _memory_item_matches_profile(item, selected_profile)
    ]
    if _is_explicit_confirmation(text):
        chosen = candidates[:2]
    elif _looks_like_followup_action(text):
        chosen = candidates[:1]
    else:
        chosen = []
    for item in chosen:
        data = approve_memory_item(data, item["id"])
        approved.append(item["id"])
    return data, approved


def build_session_memory_payload(data: dict, selected_profile: dict | None) -> dict:
    approved_items = []
    for item in data.get("items", []):
        if item.get("state") != "approved":
            continue
        if not _memory_item_matches_profile(item, selected_profile):
            continue
        approved_items.append(
            {
                "target_type": item["target_type"],
                "target_layer": item["target_layer"],
                "summary": item["summary"],
            }
        )
    return {
        "session_goal": str(data.get("session_goal", "")).strip(),
        "active_dataset_focus": str(data.get("active_dataset_focus", "")).strip(),
        "approved_items": approved_items[:5],
    }


def prune_session_memory(data: dict) -> dict:
    items = list(data.get("items", []))
    approved = [item for item in items if item.get("state") == "approved"][:5]
    provisional = [item for item in items if item.get("state") == "provisional"][:5]
    rejected = [item for item in items if item.get("state") == "rejected"][:5]
    data["items"] = approved + provisional + rejected
    return data


def _set_item_state(data: dict, item_id: str, state: str) -> dict:
    items = list(data.get("items", []))
    for item in items:
        if item.get("id") == item_id:
            item["state"] = state
            item["updated_at"] = utc_now_iso()
            break
    data["items"] = items
    return prune_session_memory(data)


def _memory_item_matches_profile(item: dict, selected_profile: dict | None) -> bool:
    if not isinstance(selected_profile, dict):
        return not item.get("target_layer")
    target_layer = str(item.get("target_layer", "")).strip()
    if target_layer and target_layer != str(selected_profile.get("layer_name", "")).strip():
        return False
    fingerprint = item.get("viewer_fingerprint", {})
    if not isinstance(fingerprint, dict):
        return True
    layer_class = str(fingerprint.get("layer_class", "")).strip()
    if layer_class and layer_class != str(selected_profile.get("layer_class", "")).strip():
        return False
    dtype = fingerprint.get("dtype")
    if dtype and dtype != selected_profile.get("dtype"):
        return False
    shape = fingerprint.get("shape")
    if isinstance(shape, list) and shape and shape != selected_profile.get("shape"):
        return False
    return True


def _is_explicit_confirmation(text: str) -> bool:
    phrases = ("yes", "correct", "that makes sense", "do that", "go ahead", "apply it", "exactly", "right")
    return any(phrase in text for phrase in phrases)


def _is_explicit_negative(text: str) -> bool:
    phrases = ("wrong", "incorrect", "not that", "no,", "no ", "don't", "do not")
    return any(phrase in text for phrase in phrases)


def _looks_like_followup_action(text: str) -> bool:
    triggers = ("measure it", "apply it", "segment it", "threshold it", "use that", "run it", "do it", "dilate it")
    return any(trigger in text for trigger in triggers)
