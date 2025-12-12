import logging
import os
import re
import uuid

import chromadb
import requests

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Simple data containers
# -------------------------------------------------------------------


class DocumentChunk:
    """Internal representation of a chunked piece of text."""

    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata or {}


class RAGChunk:
    """Chunk plus distance from query, used as RAG context."""

    def __init__(self, text, distance, metadata=None):
        self.text = text
        self.distance = distance
        self.metadata = metadata or {}


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

_CHROMA_DB_DIR_ENV = "CHROMA_DB_DIR"
_CHROMA_COLLECTION_ENV = "CHROMA_COLLECTION_NAME"
_DEFAULT_CHROMA_DB_DIR = "/chroma"
_DEFAULT_COLLECTION_NAME = "documents"

_OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
_OLLAMA_EMBED_MODEL_ENV = "EMBEDDING_MODEL"
_DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"

_client = None
_collection = None


# -------------------------------------------------------------------
# Helpers to read environment
# -------------------------------------------------------------------


def _get_chroma_db_dir():
    """Return the directory where Chroma will store data."""
    value = os.getenv(_CHROMA_DB_DIR_ENV)
    if value:
        return value
    return _DEFAULT_CHROMA_DB_DIR


def _get_chroma_collection_name():
    """Return the Chroma collection name."""
    value = os.getenv(_CHROMA_COLLECTION_ENV)
    if value:
        return value
    return _DEFAULT_COLLECTION_NAME


def _get_ollama_base_url():
    """Return base URL for Ollama HTTP API, e.g. http://ollama:11434"""
    value = os.getenv(_OLLAMA_BASE_URL_ENV)
    if value:
        return value
    return _DEFAULT_OLLAMA_BASE_URL


def _get_embedding_model():
    """Return embedding model name, e.g. `nomic-embed-text`."""
    value = os.getenv(_OLLAMA_EMBED_MODEL_ENV)
    if value:
        return value
    return _DEFAULT_EMBED_MODEL


# -------------------------------------------------------------------
# Chroma client / collection singletons
# -------------------------------------------------------------------


def _get_chroma_client():
    """
    Return a module-level singleton Chroma client.

    Tests expect:
      * The path is taken from CHROMA_DB_DIR (or default).
      * The same instance is returned when called twice.
    """

    if _client is not None:
        return _client

    db_dir = _get_chroma_db_dir()
    logger.info("Creating Chroma client at %s", db_dir)
    client = chromadb.PersistentClient(path=db_dir)
    # keep module-level cache without using `global`
    globals()["_client"] = client
    return client


def _get_collection():
    """
    Return a module-level singleton Chroma collection.

    Tests monkeypatch `_collection` directly, so if `_collection` is not
    None we just give it back and do NOT create a new one.
    """
    if _collection is not None:
        return _collection

    client = _get_chroma_client()
    name = _get_chroma_collection_name()
    logger.info("Getting/creating Chroma collection %s", name)
    collection = client.get_or_create_collection(name=name)
    # store on module without `global`
    globals()["_collection"] = collection
    return collection


# -------------------------------------------------------------------
# Chunking
# -------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE=256

JP_SENT_SPLIT = re.compile(r"(?<=[。！？])")  # noqa: RUF001
CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

def chunk_text(text: str, max_tokens: int = _DEFAULT_CHUNK_SIZE) -> list[DocumentChunk]:
    if not text:
        return []

    has_cjk = CJK_RE.search(text) is not None

    chunks: list[DocumentChunk] = []

    if has_cjk:
        sentences = [s for s in JP_SENT_SPLIT.split(text) if s.strip()]
        current: list[str] = []
        length = 0

        for s in sentences:
            s_len = len(s)
            if length + s_len > max_tokens and current:
                chunk_index = len(chunks)
                chunk_text_value = "".join(current)
                chunks.append(
                    DocumentChunk(
                        chunk_text_value,
                        {"chunk_index": chunk_index, "index": chunk_index},
                    )
                )
                current = []
                length = 0

            current.append(s)
            length += s_len

        if current:
            chunk_index = len(chunks)
            chunk_text_value = "".join(current)
            chunks.append(
                DocumentChunk(
                    chunk_text_value,
                    {"chunk_index": chunk_index, "index": chunk_index},
                )
            )

    else:
        words = text.split()
        current: list[str] = []
        count = 0

        for w in words:
            if count >= max_tokens and current:
                chunk_index = len(chunks)
                chunk_text_value = " ".join(current)
                chunks.append(
                    DocumentChunk(
                        chunk_text_value,
                        {"chunk_index": chunk_index, "index": chunk_index},
                    )
                )
                current = []
                count = 0

            current.append(w)
            count += 1

        if current:
            chunk_index = len(chunks)
            chunk_text_value = " ".join(current)
            chunks.append(
                DocumentChunk(
                    chunk_text_value,
                    {"chunk_index": chunk_index, "index": chunk_index},
                )
            )

    total = len(chunks)
    for c in chunks:
        c.metadata.setdefault("total_chunks", total)

    return chunks


# -------------------------------------------------------------------
# Embeddings (Ollama)
# -------------------------------------------------------------------

def _embed_with_ollama(text):
    """
    Call Ollama's /api/embed endpoint for a single text.

    It supports multiple possible response formats:

    New (Ollama /api/embed):

        {
          "model": "...",
          "embeddings": [[...], ...]
        }

    Older custom / other providers:

        {
          "embedding": [...]
        }

    Or:

        {
          "data": [{ "embedding": [...], ... }]
        }
    """
    base_url = _get_ollama_base_url().rstrip("/")
    model = _get_embedding_model()

    payload = {
        "model": model,
        "input": text,
    }

    url = base_url + "/api/embed"
    logger.debug("Requesting embedding from %s with model=%s", url, model)

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    embedding = None

    if isinstance(data, dict) and "embeddings" in data:
        embs = data["embeddings"]
        if isinstance(embs, list) and embs:
            embedding = embs[0]

    elif isinstance(data, dict) and "embedding" in data:
        embedding = data["embedding"]

    elif isinstance(data, dict) and "data" in data:
        first = data["data"][0]
        embedding = first["embedding"]

    if embedding is None:
        raise ValueError(f"Unexpected embedding response format: keys={list(data.keys())}")

    if not isinstance(embedding, list):
        raise ValueError("Embedding must be a list of floats")

    return embedding

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    cleaned_texts = []
    for t in texts:
        if not t:
            continue
        stripped = t.strip()
        if not stripped:
            continue
        cleaned_texts.append(stripped)

    if not cleaned_texts:
        return []

    embeddings: list[list[float]] = []
    errors: list[Exception] = []

    for t in cleaned_texts:
        try:
            emb = _embed_with_ollama(t)
            embeddings.append(emb)
        except Exception as exc:  # requests.HTTPError
            logger.exception("Embedding failed for text chunk", exc_info=exc)
            errors.append(exc)

    if not embeddings:
        raise RuntimeError(
            f"Failed to embed any of the {len(cleaned_texts)} text chunks; "
            f"last error: {errors[-1] if errors else 'unknown'}"
        )

    return embeddings


# -------------------------------------------------------------------
# Ingestion
# -------------------------------------------------------------------

def add_document(text: str) -> None:
    """Split text into chunks, embed them, and store in Chroma."""
    chunks = chunk_text(text, max_tokens=200)
    if not chunks:
        logger.warning("No chunks produced from text; nothing to add.")
        return

    documents = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    embeddings = embed_texts(documents)

    if not embeddings:
        raise RuntimeError("No embeddings produced; document not stored.")

    if len(embeddings) != len(documents):
        raise ValueError(
            f"Embedding count mismatch: {len(embeddings)} vs {len(documents)}"
        )

    collection = _get_collection()
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

# -------------------------------------------------------------------
# Retrieval
# -------------------------------------------------------------------


def query_similar_chunks(question, top_k=3):
    """
    Embed the question, query Chroma, and return a list of RAGChunk.

    Tests expect:

        query_similar_chunks(question: str, top_k: int = 3) -> list[RAGChunk]

    and then they monkeypatch this function in some API tests, but in
    `test_rag_store.py` they call the real one with a DummyCollection that
    implements:

        def query(self,
                  query_embeddings: list[list[float]],
                  n_results: int,
                  include: list[str] | None = None) -> dict[str, Any]

    The expected keys in the response are:

        {
          "documents": [[...]],
          "metadatas": [[...]],
          "distances": [[...]],
        }
    """
    cleaned = (question or "").strip()
    if not cleaned:
        return []

    query_embeddings = embed_texts([cleaned])
    if not query_embeddings:
        # Could happen if embedding fails; in that case, just return empty.
        return []

    collection = _get_collection()
    result = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs_lists = result.get("documents") or [[]]
    metas_lists = result.get("metadatas") or [[]]
    dists_lists = result.get("distances") or [[]]

    docs = docs_lists[0] if docs_lists else []
    metas = metas_lists[0] if metas_lists else []
    dists = dists_lists[0] if dists_lists else []

    chunks: list[RAGChunk] = []
    for doc, raw_meta, dist in zip(docs, metas, dists, strict=False):
        if isinstance(raw_meta, dict):
            normalized_meta = raw_meta
        elif raw_meta is None:
            normalized_meta = {}
        else:
            # Normalize strange metadata formats used in some backends.
            normalized_meta = {"value": raw_meta}

        chunks.append(
            RAGChunk(
                text=doc,
                distance=dist,
                metadata=normalized_meta,
            )
        )

    return chunks
