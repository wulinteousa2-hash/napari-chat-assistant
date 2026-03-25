from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib import request


def _friendly_request_error(url: str, exc: Exception) -> RuntimeError:
    if isinstance(exc, HTTPError):
        return RuntimeError(f"Ollama request failed with HTTP {exc.code} for {url}.")
    if isinstance(exc, URLError):
        reason = getattr(exc, "reason", exc)
        return RuntimeError(
            f"Could not connect to Ollama at {url}. Start Ollama with 'ollama serve' or update the Base URL."
        )
    return RuntimeError(str(exc))


def http_json(url: str, payload: dict | None = None, *, timeout: int = 30) -> dict:
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
            body = resp.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise _friendly_request_error(url, exc) from exc
    return json.loads(body) if body else {}


def list_ollama_models(base_url: str) -> list[str]:
    tags = http_json(f"{base_url.rstrip('/')}/api/tags")
    return sorted({m.get("name", "") for m in tags.get("models", []) if m.get("name", "")})


def pull_ollama_model_events(base_url: str, model_name: str):
    req = request.Request(
        f"{base_url.rstrip('/')}/api/pull",
        data=json.dumps({"model": model_name}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_event = {}
    try:
        with request.urlopen(req, timeout=3600) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    last_event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield last_event
    except (HTTPError, URLError) as exc:
        raise _friendly_request_error(f"{base_url.rstrip('/')}/api/pull", exc) from exc


def unload_ollama_model(base_url: str, model_name: str) -> None:
    http_json(
        f"{base_url.rstrip('/')}/api/generate",
        {
            "model": model_name,
            "keep_alive": 0,
        },
    )


def chat_ollama(
    base_url: str,
    model_name: str,
    *,
    system_prompt: str,
    user_payload: dict,
    options: dict,
    timeout: int = 1800,
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
    result = http_json(f"{base_url.rstrip('/')}/api/chat", payload, timeout=timeout)
    message = result.get("message", {})
    return str(message.get("content", "")).strip() or '{"action":"reply","message":"[empty response]"}'
