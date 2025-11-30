from __future__ import annotations

from rag_store import DocumentChunk, chunk_text


def test_chunk_text_splits_long_text_and_sets_metadata() -> None:
    words = [f"word{i}" for i in range(25)]
    text = " ".join(words)

    chunks = chunk_text(text, max_tokens=10)

    assert len(chunks) == 3

    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, DocumentChunk)
        assert chunk.text.strip()
        assert isinstance(chunk.metadata, dict)
        assert chunk.metadata.get("chunk_index") == idx or chunk.metadata.get("index") == idx

    total = len(chunks)
    for chunk in chunks:
        assert chunk.metadata.get("total_chunks") == total
