from __future__ import annotations

from typing import Any

import pytest
import routers.rag as rag_router


def test_get_ollama_chat_model_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_CHAT_MODEL", raising=False)

    value = rag_router._get_ollama_chat_model()
    assert value == "llama3.1"


def test_get_ollama_chat_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "custom-model")

    value = rag_router._get_ollama_chat_model()
    assert value == "custom-model"


def test_get_ollama_base_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    value = rag_router._get_ollama_base_url()
    assert value == "http://ollama:11434"


def test_get_ollama_base_url_env_override_trims_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.com:1234/")

    value = rag_router._get_ollama_base_url()
    assert value == "http://example.com:1234"


def test_call_ollama_chat_uses_session_and_parses_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class DummyResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return self._payload

    class DummySession:
        def post(self, url: str, json: dict[str, Any], timeout: int) -> DummyResponse:
            calls.append({"url": url, "json": json, "timeout": timeout})
            return DummyResponse({"message": {"content": "dummy answer"}})

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434/")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "llama3.1")

    monkeypatch.setattr(rag_router, "_session", DummySession())

    result = rag_router._call_ollama_chat(
        question="What is Miura Peninsula?",
        context="Some context about Miura Peninsula.",
    )

    assert result == "dummy answer"

    assert calls, "DummySession.post should have been called at least once."
    call = calls[0]

    assert call["url"].endswith("/api/chat")

    payload = call["json"]
    assert payload["model"] == "llama3.1"
    assert payload.get("stream") is False
    assert any(m["role"] == "user" for m in payload.get("messages", []))
