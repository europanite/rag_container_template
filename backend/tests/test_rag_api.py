from __future__ import annotations

from http import HTTPStatus

import pytest
import rag_store
import routers.rag as rag_router
from fastapi.testclient import TestClient
from rag_store import RAGChunk


def test_rag_ingest_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_add_document(text: str) -> None:
        calls.append(text)

    monkeypatch.setattr(rag_store, "add_document", fake_add_document)

    r = client.post("/rag/ingest", json={"documents": ["Doc1", "Doc2"]})

    assert r.status_code == HTTPStatus.OK, r.text
    assert r.json() == {"ingested": 2}
    assert calls == ["Doc1", "Doc2"]


def test_rag_ingest_empty_documents_returns_400(client: TestClient) -> None:
    r = client.post("/rag/ingest", json={"documents": []})
    assert r.status_code == HTTPStatus.BAD_REQUEST
    body = r.json()
    assert "documents" in body.get("detail", "").lower() or "empty" in body.get(
        "detail", ""
    ).lower()


def test_rag_ingest_all_fail_returns_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_add_document(_text: str) -> None:
        raise RuntimeError("test failure")

    monkeypatch.setattr(rag_store, "add_document", fake_add_document)

    r = client.post("/rag/ingest", json={"documents": ["x", "y"]})
    assert r.status_code == HTTPStatus.BAD_GATEWAY
    assert "failed" in r.json().get("detail", "").lower()


def test_rag_query_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_query_similar_chunks(question: str, top_k: int = 3) -> list[RAGChunk]:
        return [
            RAGChunk(
                text="Miura Peninsula is located in Kanagawa Prefecture.",
                distance=0.01,
                metadata={"source": "test"},
            )
        ]

    monkeypatch.setattr(rag_store, "query_similar_chunks", fake_query_similar_chunks)

    def fake_call_ollama_chat(*, question: str, context: str) -> str:
        assert "Miura Peninsula" in context
        return f"ANSWER to: {question}"

    monkeypatch.setattr(rag_router, "_call_ollama_chat", fake_call_ollama_chat)

    r = client.post("/rag/query", json={"question": "Where is Miura Peninsula?"})

    assert r.status_code == HTTPStatus.OK, r.text
    data = r.json()
    assert data["answer"].startswith("ANSWER to:")
    assert data["context"] == [
        "Miura Peninsula is located in Kanagawa Prefecture."
    ]


def test_rag_query_no_context_returns_404(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag_store, "query_similar_chunks", lambda q, top_k=3: [])

    r = client.post("/rag/query", json={"question": "hello?"})

    assert r.status_code == HTTPStatus.NOT_FOUND
    assert "No relevant context" in r.json().get("detail", "")


def test_rag_query_ollama_failure_returns_502(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_query_similar_chunks(question: str, top_k: int = 3) -> list[RAGChunk]:
        return [
            RAGChunk(
                text="Some stored chunk",
                distance=0.5,
                metadata={"source": "test"},
            )
        ]

    monkeypatch.setattr(rag_store, "query_similar_chunks", fake_query_similar_chunks)

    def fake_call_ollama_chat(*, question: str, context: str) -> str:
        raise RuntimeError("Ollama is down")

    monkeypatch.setattr(rag_router, "_call_ollama_chat", fake_call_ollama_chat)

    r = client.post("/rag/query", json={"question": "hello?"})

    assert r.status_code == HTTPStatus.BAD_GATEWAY
    assert "Ollama is down" in r.json().get("detail", "")
