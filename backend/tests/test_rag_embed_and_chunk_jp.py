from __future__ import annotations

from typing import Any

import pytest
import rag_store
from rag_store import chunk_text


# 1) JP_SENT_SPLIT
def test_chunk_text_splits_japanese_sentences_and_sets_metadata() -> None:
    text = "三浦半島は神奈川県にあります。海がきれいです。山もあります。"

    # ★ max_tokens を小さくして、文ごとに分割される状況を作る
    chunks = chunk_text(text, max_tokens=10)

    texts = [c.text.strip() for c in chunks]
    assert texts == [
        "三浦半島は神奈川県にあります。",
        "海がきれいです。",
        "山もあります。",
    ]

    num_chunks = 3

    # meta data
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]
    assert all(c.metadata["total_chunks"] == num_chunks for c in chunks)


# 2) embed_texts:  ->
def test_embed_texts_empty_list_returns_empty() -> None:
    assert rag_store.embed_texts([]) == []


# 3) embed_texts: -> cleaned_texts  return []
def test_embed_texts_only_blank_strings_returns_empty() -> None:
    vectors = rag_store.embed_texts(["   ", "", "\n\t"])
    assert vectors == []


# 4) embed_texts:  _embed_with_ollama
def test_embed_texts_raises_runtime_error_when_all_calls_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fail(_text: str) -> list[float]:
        raise RuntimeError("boom")

    monkeypatch.setattr(rag_store, "_embed_with_ollama", always_fail)

    with pytest.raises(RuntimeError) as excinfo:
        rag_store.embed_texts(["hello", "world"])

    msg = str(excinfo.value)
    assert "Failed to embed any of the 2 text chunks" in msg


# 5) _embed_with_ollama:
class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.mark.parametrize(
    "payload, expected",
    [
        # {"embeddings": [[...]]}
        ({"embeddings": [[0.1, 0.2, 0.3]]}, [0.1, 0.2, 0.3]),
        # {"embedding": [...]}
        ({"embedding": [1.0, 2.0]}, [1.0, 2.0]),
        # {"data": [{"embedding": [...]}]}
        ({"data": [{"embedding": [3.0]}]}, [3.0]),
    ],
)
def test__embed_with_ollama_accepts_multiple_response_shapes(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    expected: list[float],
) -> None:
    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _DummyResponse:
        return _DummyResponse(payload)

    monkeypatch.setattr(rag_store.requests, "post", fake_post)

    vec = rag_store._embed_with_ollama("dummy text")
    assert vec == expected


# 6) _embed_with_ollama:
def test__embed_with_ollama_raises_value_error_on_unexpected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _DummyResponse:
        # embedding
        return _DummyResponse({"unexpected": "shape"})

    monkeypatch.setattr(rag_store.requests, "post", fake_post)

    with pytest.raises(ValueError) as excinfo:
        rag_store._embed_with_ollama("dummy text")

    msg = str(excinfo.value)
    assert "Unexpected embedding response format" in msg
