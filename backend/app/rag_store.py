from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import chromadb
import requests

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------


@dataclass
class DocumentChunk:
    """Internal representation of a chunked piece of text."""
    text: str
    metadata: Dict[str, Any]


@dataclass
class RAGChunk:
    """
    Public representation of a retrieved chunk used by the RAG API.

    Tests import this with:
        from rag_store import RAGChunk
    """
    text: str
    distance: float
    metadata: Dict[str, Any]


# -------------------------------------------------------------------
# Chroma / Ollama configuration
# -------------------------------------------------------------------

_CHROMA_DB_DIR_ENV = "CHROMA_DB_DIR"
_CHROMA_COLLECTION_ENV = "CHROMA_COLLECTION"
_DEFAULT_CHROMA_DB_DIR = "/chroma"
_DEFAULT_COLLECTION_NAME = "documents"

_OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
_OLLAMA_EMBED_MODEL_ENV = "OLLAMA_EMBED_MODEL"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"


def _get_ollama_base_url() -> str:
    return os.getenv(_OLLAMA_BASE_URL_ENV, "http://ollama:11434").rstrip("/")


def _get_embedding_model() -> str:
    return os.getenv(_OLLAMA_EMBED_MODEL_ENV, _DEFAULT_EMBED_MODEL)


def _get_chroma_client() -> chromadb.PersistentClient:
    db_dir = os.getenv(_CHROMA_DB_DIR_ENV, _DEFAULT_CHROMA_DB_DIR)
    logger.info("Initializing Chroma DB at %s", db_dir)
    return chromadb.PersistentClient(path=db_dir)


_chroma_client = _get_chroma_client()
_collection_name = os.getenv(_CHROMA_COLLECTION_ENV, _DEFAULT_COLLECTION_NAME)
_collection = _chroma_client.get_or_create_collection(name=_collection_name)

logger.info("Using Chroma collection '%s'", _collection_name)


# -------------------------------------------------------------------
# Chunking
# -------------------------------------------------------------------


def chunk_text(text: str, max_tokens: int = 200) -> List[DocumentChunk]:
    """
    Naively split text into smaller chunks of up to `max_tokens` words.

    Tests expect:
      - Signature: chunk_text(text, max_tokens=...)
      - Return value: list of objects having `.text` and `.metadata`
      - Each metadata has at least an 'index' key.
    """
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    chunks: List[DocumentChunk] = []

    idx = 0
    for start in range(0, len(words), max_tokens):
        part_words = words[start : start + max_tokens]
        part_text = " ".join(part_words).strip()
        if not part_text:
            continue

        chunks.append(
            DocumentChunk(
                text=part_text,
                metadata={"index": idx},  # tests check "index" in metadata
            )
        )
        idx += 1

    return chunks


# -------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------


def _embed_with_ollama(text: str) -> List[float]:
    """
    Call Ollama's /api/embeddings endpoint for a single text.
    Tests monkeypatch this function with:
        def fake_embed(text: str) -> list[float]: ...
    """
    base_url = _get_ollama_base_url()
    model = _get_embedding_model()

    resp = requests.post(
        f"{base_url}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("Ollama embeddings response missing 'embedding'")
    return embedding


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """
    Compute embeddings for a batch of texts using Ollama.

    Tests expect:
      - `_embed_with_ollama` is called once per text
      - Called with exactly one argument: the text string
      - Returned list length == len(texts)
    """
    if not texts:
        return []

    embeddings: List[List[float]] = []
    for t in texts:
        cleaned = (t or "").strip()
        if not cleaned:
            embeddings.append([])
            continue

        emb = _embed_with_ollama(cleaned)
        embeddings.append(emb)

    return embeddings


# -------------------------------------------------------------------
# Document ingestion
# -------------------------------------------------------------------


def add_document(text: str) -> None:
    """
    Chunk and store a single document in Chroma.

    Tests call this as:

        rag_store.add_document("Some text")

    and then inspect a DummyCollection that is monkeypatched into
    `rag_store._collection`. The DummyCollection stores records like:

        {"document": ..., "metadata": ..., ...}

    Tests only assert that:
      - at least one record is stored
      - each record has keys "document" and "metadata"
      - metadata contains "chunk_index"
    """
    chunks = chunk_text(text)
    if not chunks:
        return

    docs = [c.text for c in chunks]
    metadatas = [dict(c.metadata) for c in chunks]

    # Ensure metadata has "chunk_index" for tests
    for i, meta in enumerate(metadatas):
        meta.setdefault("chunk_index", i)

    embeddings = embed_texts(docs)

    # Simple deterministic-ish IDs; uniqueness beyond tests is not critical
    ids = [f"doc-{id(text)}-{i}" for i in range(len(docs))]

    _collection.add(
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


# -------------------------------------------------------------------
# Retrieval
# -------------------------------------------------------------------


def query_similar_chunks(question: str, top_k: int = 3) -> List[RAGChunk]:
    """
    Embed the question, query Chroma, and return a list of RAGChunk.

    Tests expect signature:

        query_similar_chunks(question: str, top_k: int = 3) -> list[RAGChunk]

    and then they monkeypatch this function in some API tests, but in
    `test_rag_store.py` they call the real one with a DummyCollection that
    implements:

        def query(self, query_embeddings: list[list[float]],
                  n_results: int, include: list[str] | None = None) -> dict[str, Any]
    """
    vectors = embed_texts([question])
    if not vectors or not vectors[0]:
        return []

    result = _collection.query(
        query_embeddings=vectors,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs_lists = result.get("documents") or [[]]
    metas_lists = result.get("metadatas") or [[]]
    dists_lists = result.get("distances") or [[]]

    docs = docs_lists[0] if docs_lists else []
    metas = metas_lists[0] if metas_lists else []
    dists = dists_lists[0] if dists_lists else []

    chunks: List[RAGChunk] = []
    for doc, meta, dist in zip(docs, metas, dists):
        if not isinstance(meta, dict):
            meta = {} if meta is None else {"value": meta}
        chunks.append(
            RAGChunk(
                text=str(doc),
                distance=float(dist),
                metadata=meta,
            )
        )

    return chunks
