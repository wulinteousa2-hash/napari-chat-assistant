from __future__ import annotations

import json

from napari_chat_assistant.agent import prompt_library as lib


def test_load_prompt_library_migrates_legacy_saved_pins(tmp_path, monkeypatch):
    path = tmp_path / "prompt_library.json"
    path.write_text(
        json.dumps(
            {
                "saved": [
                    {"title": "Built-in clone", "prompt": "show me my layers", "pinned": True},
                    {"title": "Custom", "prompt": "custom prompt", "pinned": False},
                ],
                "recent": [{"title": "Recent", "prompt": "recent prompt"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lib, "prompt_library_path", lambda: path)

    state = lib.load_prompt_library()

    assert [item["prompt"] for item in state["saved"]] == ["show me my layers", "custom prompt"]
    assert [item["prompt"] for item in state["recent"]] == ["recent prompt"]
    assert state["pinned_prompts"] == ["show me my layers"]
    assert state["hidden_built_in"] == []


def test_remove_prompt_record_handles_saved_recent_and_built_in_sources():
    built_in_prompt = lib.DEFAULT_PROMPTS[0]
    state = {
        "saved": [{"title": "Saved", "prompt": "saved prompt", "source": "saved", "updated_at": "2026-01-01T00:00:00+00:00"}],
        "recent": [{"title": "Recent", "prompt": "recent prompt", "source": "recent", "updated_at": "2026-01-02T00:00:00+00:00"}],
        "pinned_prompts": ["saved prompt", "recent prompt", built_in_prompt],
        "hidden_built_in": [],
    }

    lib.remove_prompt_record(state, "saved prompt", source="saved")
    lib.remove_prompt_record(state, "recent prompt", source="recent")
    lib.remove_prompt_record(state, built_in_prompt, source="built_in")

    assert state["saved"] == []
    assert state["recent"] == []
    assert state["pinned_prompts"] == []
    assert state["hidden_built_in"] == [built_in_prompt]


def test_clear_prompt_library_keeps_saved_and_pinned_only():
    built_in_prompt = lib.DEFAULT_PROMPTS[0]
    state = {
        "saved": [{"title": "Saved", "prompt": "saved prompt", "source": "saved", "updated_at": "2026-01-01T00:00:00+00:00"}],
        "recent": [{"title": "Recent", "prompt": "recent prompt", "source": "recent", "updated_at": "2026-01-02T00:00:00+00:00"}],
        "pinned_prompts": [built_in_prompt],
        "hidden_built_in": [],
    }

    lib.clear_prompt_library(state, keep_saved=True, keep_pinned=True)
    merged = lib.merged_prompt_records(state)

    assert state["recent"] == []
    assert any(item["prompt"] == "saved prompt" and item["source"] == "saved" for item in merged)
    assert any(item["prompt"] == built_in_prompt and item["source"] == "built_in" and item["pinned"] for item in merged)
    assert all(item["prompt"] != "recent prompt" for item in merged)
    assert len(state["hidden_built_in"]) == len(lib.DEFAULT_PROMPTS) - 1


def test_merged_prompt_records_deduplicates_by_priority_and_applies_pins():
    built_in_prompt = lib.DEFAULT_PROMPTS[0]
    state = {
        "saved": [
            {"title": "Saved duplicate", "prompt": built_in_prompt, "source": "saved", "updated_at": "2026-01-03T00:00:00+00:00"},
            {"title": "Saved custom", "prompt": "saved prompt", "source": "saved", "updated_at": "2026-01-02T00:00:00+00:00"},
        ],
        "recent": [
            {"title": "Recent duplicate", "prompt": built_in_prompt, "source": "recent", "updated_at": "2026-01-04T00:00:00+00:00"},
            {"title": "Recent unique", "prompt": "recent prompt", "source": "recent", "updated_at": "2026-01-05T00:00:00+00:00"},
        ],
        "pinned_prompts": ["recent prompt", built_in_prompt],
        "hidden_built_in": [],
    }

    merged = lib.merged_prompt_records(state)

    assert merged[0]["prompt"] == built_in_prompt
    assert merged[0]["source"] == "saved"
    assert merged[0]["pinned"] is True
    assert merged[1]["prompt"] == "recent prompt"
    assert merged[1]["source"] == "recent"
    assert merged[1]["pinned"] is True
    assert sum(item["prompt"] == built_in_prompt for item in merged) == 1


def test_merged_code_records_keeps_built_in_demo_visible_when_recent_duplicate_exists():
    built_in = lib.DEFAULT_CODE_SNIPPETS[0]
    code = str(built_in["code"]).strip()
    state = {
        "code_saved": [],
        "code_recent": [
            {
                "title": "Recent duplicate",
                "code": code,
                "source": "recent",
                "updated_at": "2026-01-05T00:00:00+00:00",
            }
        ],
        "pinned_codes": [],
    }

    merged = lib.merged_code_records(state)

    assert any(item["source"] == "recent" and item["code"] == code for item in merged)
    assert any(item["source"] == "built_in" and item["code"] == code and item["title"] == built_in["title"] for item in merged)


def test_upsert_recent_code_preserves_code_formatting():
    state = {"code_recent": []}
    code = "if True:\n    print('x')\n"

    lib.upsert_recent_code(state, code)

    assert state["code_recent"][0]["code"] == code
