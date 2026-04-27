from __future__ import annotations

import json
import socket
from urllib.error import HTTPError, URLError
from urllib import request


OLLAMA_TIMING_FIELDS = (
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
    "total_duration",
)


def _friendly_request_error(url: str, exc: Exception) -> RuntimeError:
    if isinstance(exc, HTTPError):
        return RuntimeError(f"Ollama request failed with HTTP {exc.code} for {url}.")
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", exc)
        return RuntimeError(
            f"Could not connect to Ollama at {url}. Start Ollama with 'ollama serve' or update the Base URL."
        )
    return RuntimeError(str(exc))


def http_json(
    url: str,
    payload: dict | None = None,
    *,
    timeout: int = 30,
    response_holder: dict | None = None,
    stop_event=None,
) -> dict:
    if payload is None:
        req = request.Request(url, method="GET")
    else:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            if response_holder is not None:
                response_holder["response"] = resp
            if stop_event is not None and stop_event.is_set():
                return {}
            body = resp.read().decode("utf-8")
    except Exception as exc:
        if stop_event is not None and stop_event.is_set():
            return {}
        if isinstance(exc, (HTTPError, URLError)):
            raise _friendly_request_error(url, exc) from exc
        raise
    finally:
        if response_holder is not None:
            response_holder.pop("response", None)
    return json.loads(body) if body else {}


def _estimate_tokens_from_chars(text: str) -> int:
    # A coarse provider-neutral estimate; Ollama metadata gives exact counts when available.
    return max(1, round(len(str(text or "")) / 4)) if text else 0


def _duration_ms(value) -> float | None:
    try:
        raw = float(value)
    except Exception:
        return None
    if raw <= 0:
        return None
    # Ollama durations are nanoseconds.
    return raw / 1_000_000.0


def _tokens_per_second(count, duration) -> float | None:
    try:
        token_count = float(count)
    except Exception:
        return None
    duration_ms = _duration_ms(duration)
    if not duration_ms:
        return None
    return token_count / (duration_ms / 1000.0)


def ollama_chat_request_stats(payload: dict) -> dict:
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    system_content = ""
    user_content = ""
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", ""))
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
    request_text = json.dumps(payload)
    input_text = system_content + "\n" + user_content
    return {
        "input_chars": len(input_text),
        "input_bytes": len(input_text.encode("utf-8")),
        "estimated_input_tokens": _estimate_tokens_from_chars(input_text),
        "system_prompt_chars": len(system_content),
        "system_prompt_bytes": len(system_content.encode("utf-8")),
        "estimated_system_prompt_tokens": _estimate_tokens_from_chars(system_content),
        "user_payload_chars": len(user_content),
        "user_payload_bytes": len(user_content.encode("utf-8")),
        "estimated_user_payload_tokens": _estimate_tokens_from_chars(user_content),
        "full_request_chars": len(request_text),
        "full_request_bytes": len(request_text.encode("utf-8")),
        "estimated_full_request_tokens": _estimate_tokens_from_chars(request_text),
    }


def ollama_response_metadata(result: dict) -> dict:
    metadata = {}
    for field in OLLAMA_TIMING_FIELDS:
        if field in result:
            metadata[field] = result.get(field)

    prompt_eval_duration_ms = _duration_ms(result.get("prompt_eval_duration"))
    eval_duration_ms = _duration_ms(result.get("eval_duration"))
    total_duration_ms = _duration_ms(result.get("total_duration"))
    if prompt_eval_duration_ms is not None:
        metadata["prompt_eval_duration_ms"] = prompt_eval_duration_ms
    if eval_duration_ms is not None:
        metadata["eval_duration_ms"] = eval_duration_ms
    if total_duration_ms is not None:
        metadata["total_duration_ms"] = total_duration_ms

    prompt_eval_tps = _tokens_per_second(result.get("prompt_eval_count"), result.get("prompt_eval_duration"))
    generation_tps = _tokens_per_second(result.get("eval_count"), result.get("eval_duration"))
    if prompt_eval_tps is not None:
        metadata["prompt_eval_tokens_per_second"] = prompt_eval_tps
    if generation_tps is not None:
        metadata["generation_tokens_per_second"] = generation_tps
    return metadata


def list_ollama_models(base_url: str) -> list[str]:
    tags = http_json(f"{base_url.rstrip('/')}/api/tags")
    return sorted({m.get("name", "") for m in tags.get("models", []) if m.get("name", "")})


def pull_ollama_model_events(base_url: str, model_name: str, *, stop_event=None, response_holder: dict | None = None):
    req = request.Request(
        f"{base_url.rstrip('/')}/api/pull",
        data=json.dumps({"model": model_name}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_event = {}
    try:
        with request.urlopen(req, timeout=3600) as resp:
            if response_holder is not None:
                response_holder["response"] = resp
            for raw_line in resp:
                if stop_event is not None and stop_event.is_set():
                    return
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    last_event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield last_event
    except socket.timeout:
        if stop_event is not None and stop_event.is_set():
            return
        raise RuntimeError(f"Ollama pull timed out for {base_url.rstrip('/')}/api/pull.")
    except (HTTPError, URLError) as exc:
        if stop_event is not None and stop_event.is_set():
            return
        raise _friendly_request_error(f"{base_url.rstrip('/')}/api/pull", exc) from exc
    finally:
        if response_holder is not None:
            response_holder.pop("response", None)


def unload_ollama_model(base_url: str, model_name: str) -> None:
    http_json(
        f"{base_url.rstrip('/')}/api/generate",
        {
            "model": model_name,
            "keep_alive": 0,
        },
    )


def load_ollama_model(base_url: str, model_name: str, *, keep_alive: str = "30m") -> None:
    http_json(
        f"{base_url.rstrip('/')}/api/generate",
        {
            "model": model_name,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        },
        timeout=1800,
    )


def chat_ollama(
    base_url: str,
    model_name: str,
    *,
    system_prompt: str,
    user_payload: dict,
    options: dict,
    timeout: int = 1800,
    response_holder: dict | None = None,
    request_metadata_holder: dict | None = None,
    response_metadata_holder: dict | None = None,
    stop_event=None,
) -> str:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        "stream": False,
        "options": options,
    }
    if request_metadata_holder is not None:
        request_metadata_holder.clear()
        request_metadata_holder.update(ollama_chat_request_stats(payload))
    result = http_json(
        f"{base_url.rstrip('/')}/api/chat",
        payload,
        timeout=timeout,
        response_holder=response_holder,
        stop_event=stop_event,
    )
    if stop_event is not None and stop_event.is_set():
        return ""
    if response_metadata_holder is not None:
        response_metadata_holder.clear()
        response_metadata_holder.update(ollama_response_metadata(result))
    message = result.get("message", {})
    return str(message.get("content", "")).strip() or '{"action":"reply","message":"[empty response]"}'
