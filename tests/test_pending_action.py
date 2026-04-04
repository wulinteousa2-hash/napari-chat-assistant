from __future__ import annotations

from napari_chat_assistant.agent.pending_action import (
    advance_pending_action_turn,
    build_pending_action_from_assistant_message,
    cancel_pending_action,
    complete_pending_action,
    empty_pending_action,
    is_pending_action_cancel_message,
    is_pending_action_waiting,
    normalize_pending_action,
    resolve_pending_action,
)


def test_build_pending_action_from_assistant_message_creates_waiting_state():
    pending = build_pending_action_from_assistant_message(
        "Which image layer would you like to apply Gaussian denoising to "
        "(e.g., em_2d_snr_low, em_2d_snr_mid, or em_2d_snr_high)?",
        turn_id="turn-1",
        selected_layer_name="em_2d_snr_mid",
    )

    assert pending["status"] == "waiting"
    assert pending["kind"] == "tool_argument_request"
    assert pending["tool"] == "gaussian_denoise"
    assert pending["missing_argument"] == "layer_name"
    assert pending["candidate_options"] == ["em_2d_snr_low", "em_2d_snr_mid", "em_2d_snr_high"]
    assert pending["selected_layer_name"] == "em_2d_snr_mid"
    assert pending["created_turn_id"] == "turn-1"


def test_resolve_pending_action_uses_selected_layer_alias():
    pending = build_pending_action_from_assistant_message(
        "Which image layer would you like to apply Gaussian denoising to "
        "(e.g., em_2d_snr_low, em_2d_snr_mid, or em_2d_snr_high)?",
        selected_layer_name="em_2d_snr_mid",
    )

    resolved = resolve_pending_action(
        pending,
        user_text="current selected one",
        selected_layer_name="em_2d_snr_mid",
        available_layer_names=["em_2d_snr_low", "em_2d_snr_mid", "em_2d_snr_high"],
    )

    assert resolved is not None
    assert resolved["tool"] == "gaussian_denoise"
    assert resolved["arguments"]["layer_name"] == "em_2d_snr_mid"


def test_resolve_pending_action_uses_numbered_option():
    pending = build_pending_action_from_assistant_message(
        "Which image layer would you like to apply Gaussian denoising to "
        "(e.g., em_2d_snr_low, em_2d_snr_mid, or em_2d_snr_high)?",
        selected_layer_name="em_2d_snr_low",
    )

    resolved = resolve_pending_action(
        pending,
        user_text="2",
        selected_layer_name="em_2d_snr_low",
        available_layer_names=["em_2d_snr_low", "em_2d_snr_mid", "em_2d_snr_high"],
    )

    assert resolved is not None
    assert resolved["arguments"]["layer_name"] == "em_2d_snr_mid"


def test_resolve_pending_action_uses_selected_layer_for_ok_when_default_is_selected_layer():
    pending = build_pending_action_from_assistant_message(
        "Which image layer would you like to apply Gaussian denoising to "
        "(e.g., em_2d_snr_low, em_2d_snr_mid, or em_2d_snr_high)?",
        selected_layer_name="em_2d_snr_high",
    )

    resolved = resolve_pending_action(
        pending,
        user_text="ok",
        selected_layer_name="em_2d_snr_high",
        available_layer_names=["em_2d_snr_low", "em_2d_snr_mid", "em_2d_snr_high"],
    )

    assert resolved is not None
    assert resolved["arguments"]["layer_name"] == "em_2d_snr_high"


def test_pending_action_can_be_cancelled_and_completed():
    pending = build_pending_action_from_assistant_message(
        "Which image layer would you like to apply Gaussian denoising to "
        "(e.g., em_2d_snr_low, em_2d_snr_mid, or em_2d_snr_high)?"
    )

    cancelled = cancel_pending_action(pending)
    completed = complete_pending_action(pending)

    assert cancelled["status"] == "cancelled"
    assert completed["status"] == "completed"
    assert is_pending_action_cancel_message("cancel") is True
    assert is_pending_action_cancel_message("never mind") is True


def test_pending_action_expires_after_two_unresolved_turns():
    pending = build_pending_action_from_assistant_message(
        "Which image layer would you like to apply Gaussian denoising to "
        "(e.g., em_2d_snr_low, em_2d_snr_mid, or em_2d_snr_high)?"
    )

    pending = advance_pending_action_turn(pending)
    assert pending["status"] == "waiting"
    assert pending["turns_waited"] == 1

    pending = advance_pending_action_turn(pending)
    assert pending["status"] == "expired"
    assert pending["turns_waited"] == 2


def test_empty_pending_action_is_not_waiting():
    pending = normalize_pending_action(empty_pending_action())

    assert is_pending_action_waiting(pending) is False
