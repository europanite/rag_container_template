from __future__ import annotations

from typing import Any

import pytest
import rag_store


class DummyCollection:
    """Very small in-memory stand-in for the Chroma collection."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def count(self) -> int:
        return len(self.records)

    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        for emb, doc, meta, _id in zip(
            embeddings, documents, metadatas, ids, strict=False
        ):
            self.records.append(
                {
                    "id": _id,
                    "document": doc,
                    "metadata": meta,
                    "embedding": emb,
                }
            )

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        docs = [r["document"] for r in self.records][:n_results]
        metas = [r["metadata"] for r in self.records][:n_results]
        dists = [0.1 for _ in docs]

        return {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }


def test_embed_texts_uses_ollama_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def fake_embed(text: str) -> list[float]:
        called.append(text)
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(rag_store, "_embed_with_ollama", fake_embed)

    vectors = rag_store.embed_texts(["hello", "world"])

    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert called == ["hello", "world"]


def test_add_document_stores_all_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_collection = DummyCollection()

    monkeypatch.setattr(rag_store, "_collection", dummy_collection)

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[float(i)] for i, _ in enumerate(texts)]

    monkeypatch.setattr(rag_store, "embed_texts", fake_embed_texts)

    text = "One sentence about Miura. Another one about Yokosuka."
    rag_store.add_document(text)

    assert dummy_collection.count() == len(dummy_collection.records) > 0

    first = dummy_collection.records[0]
    assert "document" in first
    assert "metadata" in first
    assert "chunk_index" in first["metadata"]


def test_query_similar_chunks_returns_rag_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_collection = DummyCollection()
    monkeypatch.setattr(rag_store, "_collection", dummy_collection)

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    monkeypatch.setattr(rag_store, "embed_texts", fake_embed_texts)

    rag_store.add_document("First document about Miura.")
    rag_store.add_document("Second document about Yokosuka.")

    monkeypatch.setattr(rag_store, "embed_texts", fake_embed_texts)

    EXPECTED_TOP_K = 2

    chunks = rag_store.query_similar_chunks("Tell me about Yokosuka", top_k=EXPECTED_TOP_K)

    assert len(chunks) == EXPECTED_TOP_K
    c0 = chunks[0]
    assert hasattr(c0, "text")
    assert hasattr(c0, "distance")
    assert hasattr(c0, "metadata")
    assert isinstance(c0.metadata, dict)
