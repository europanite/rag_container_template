from __future__ import annotations

from rag_store import DocumentChunk, chunk_text

EXPECTED_CHUNK_COUNT = 3

def test_chunk_text_splits_long_text_and_sets_metadata() -> None:
    words = [f"word{i}" for i in range(25)]
    text = " ".join(words)

    chunks = chunk_text(text, max_tokens=10)

    assert len(chunks) == EXPECTED_CHUNK_COUNT

    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, DocumentChunk)
        assert chunk.text.strip()
        assert isinstance(chunk.metadata, dict)
        assert chunk.metadata.get("chunk_index") == idx or chunk.metadata.get("index") == idx

    total = len(chunks)
    for chunk in chunks:
        assert chunk.metadata.get("total_chunks") == total
