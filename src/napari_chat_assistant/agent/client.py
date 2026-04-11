from __future__ import annotations

import json
import socket
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
    result = http_json(
        f"{base_url.rstrip('/')}/api/chat",
        payload,
        timeout=timeout,
        response_holder=response_holder,
        stop_event=stop_event,
    )
    if stop_event is not None and stop_event.is_set():
        return ""
    message = result.get("message", {})
    return str(message.get("content", "")).strip() or '{"action":"reply","message":"[empty response]"}'
