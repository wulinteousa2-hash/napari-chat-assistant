from __future__ import annotations

from napari_chat_assistant.agent.prompt_routing import (
    infer_tool_clarification_request,
    is_affirmative_followup,
    resolve_followup_choice_index,
    resolve_followup_layer_reference,
)


PENDING_ACTION_STATUSES = {"idle", "waiting", "resolved", "cancelled", "expired", "completed"}


def empty_pending_action() -> dict:
    return {
        "kind": "",
        "status": "idle",
        "tool": "",
        "operation_label": "",
        "arguments_collected": {},
        "missing_argument": "",
        "candidate_options": [],
        "selection_mode": "single",
        "default_resolution": "",
        "selected_layer_name": "",
        "created_turn_id": "",
        "turns_waited": 0,
        "expires_after_turns": 2,
    }


def normalize_pending_action(data: dict | None) -> dict:
    base = empty_pending_action()
    if not isinstance(data, dict):
        return base
    status = str(data.get("status", base["status"])).strip().lower()
    if status not in PENDING_ACTION_STATUSES:
        status = base["status"]
    arguments = data.get("arguments_collected", {})
    options = data.get("candidate_options", [])
    return {
        "kind": str(data.get("kind", "")).strip(),
        "status": status,
        "tool": str(data.get("tool", "")).strip(),
        "operation_label": str(data.get("operation_label", "")).strip(),
        "arguments_collected": dict(arguments) if isinstance(arguments, dict) else {},
        "missing_argument": str(data.get("missing_argument", "")).strip(),
        "candidate_options": [str(item).strip() for item in options if str(item).strip()],
        "selection_mode": str(data.get("selection_mode", "single")).strip() or "single",
        "default_resolution": str(data.get("default_resolution", "")).strip(),
        "selected_layer_name": str(data.get("selected_layer_name", "")).strip(),
        "created_turn_id": str(data.get("created_turn_id", "")).strip(),
        "turns_waited": max(0, int(data.get("turns_waited", 0) or 0)),
        "expires_after_turns": max(1, int(data.get("expires_after_turns", 2) or 2)),
    }


def is_pending_action_waiting(data: dict | None) -> bool:
    pending = normalize_pending_action(data)
    return bool(pending.get("tool")) and pending.get("status") == "waiting"


def build_pending_action_from_assistant_message(
    message: str,
    *,
    turn_id: str = "",
    selected_layer_name: str = "",
) -> dict:
    inferred = infer_tool_clarification_request(message)
    if not isinstance(inferred, dict):
        return empty_pending_action()
    return normalize_pending_action(
        {
            "kind": "tool_argument_request",
            "status": "waiting",
            "tool": inferred.get("tool", ""),
            "operation_label": inferred.get("operation_label", ""),
            "arguments_collected": inferred.get("arguments", {}),
            "missing_argument": inferred.get("layer_argument", "layer_name"),
            "candidate_options": inferred.get("options", []),
            "selection_mode": "single",
            "default_resolution": "selected_layer",
            "selected_layer_name": selected_layer_name,
            "created_turn_id": turn_id,
            "turns_waited": 0,
            "expires_after_turns": 2,
        }
    )


def resolve_pending_action(
    data: dict | None,
    *,
    user_text: str,
    selected_layer_name: str = "",
    available_layer_names: list[str] | tuple[str, ...] = (),
) -> dict | None:
    pending = normalize_pending_action(data)
    if not is_pending_action_waiting(pending):
        return None
    missing_argument = str(pending.get("missing_argument", "")).strip()
    if missing_argument != "layer_name":
        return None

    current_selected = str(selected_layer_name or pending.get("selected_layer_name", "")).strip()
    available_names = [str(name).strip() for name in available_layer_names if str(name).strip()]
    option_names = [name for name in pending.get("candidate_options", []) if name in available_names]

    matched_names = resolve_followup_layer_reference(
        user_text,
        selected_layer_name=current_selected,
        available_layer_names=available_names,
    )
    if not matched_names and option_names:
        indexed_choice = resolve_followup_choice_index(user_text, option_names)
        if indexed_choice:
            matched_names = [indexed_choice]
    if not matched_names and current_selected and pending.get("default_resolution") == "selected_layer":
        if is_affirmative_followup(user_text) and current_selected in available_names:
            matched_names = [current_selected]
    if not matched_names:
        return None

    arguments = dict(pending.get("arguments_collected", {}))
    arguments[missing_argument] = matched_names[0]
    return {
        "tool": str(pending.get("tool", "")).strip(),
        "arguments": arguments,
        "tool_message": f"Continuing with {str(pending.get('operation_label', 'the requested operation')).strip()} on [{matched_names[0]}].",
        "resolved_argument": missing_argument,
        "resolved_value": matched_names[0],
    }


def is_pending_action_cancel_message(text: str) -> bool:
    source = " ".join(str(text or "").strip().lower().split())
    if not source:
        return False
    return source in {"cancel", "never mind", "nevermind", "stop", "skip", "no"}


def cancel_pending_action(data: dict | None) -> dict:
    pending = normalize_pending_action(data)
    if not pending.get("tool"):
        return empty_pending_action()
    pending["status"] = "cancelled"
    return pending


def complete_pending_action(data: dict | None) -> dict:
    pending = normalize_pending_action(data)
    if not pending.get("tool"):
        return empty_pending_action()
    pending["status"] = "completed"
    return pending


def advance_pending_action_turn(data: dict | None) -> dict:
    pending = normalize_pending_action(data)
    if not is_pending_action_waiting(pending):
        return pending
    pending["turns_waited"] = int(pending.get("turns_waited", 0)) + 1
    if pending["turns_waited"] >= int(pending.get("expires_after_turns", 2)):
        pending["status"] = "expired"
    return pending
