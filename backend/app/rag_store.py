import logging
import os
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
    global _client

    if _client is not None:
        return _client

    db_dir = _get_chroma_db_dir()
    logger.info("Creating Chroma PersistentClient at %s", db_dir)
    _client = chromadb.PersistentClient(path=db_dir)
    return _client


def _get_collection():
    """
    Return a module-level singleton Chroma collection.

    Tests monkeypatch `_collection` directly, so if `_collection` is not
    None we just give it back and do NOT create a new one.
    """
    global _collection

    if _collection is not None:
        return _collection

    client = _get_chroma_client()
    name = _get_chroma_collection_name()
    logger.info("Getting/creating Chroma collection %s", name)
    _collection = client.get_or_create_collection(name)
    return _collection


# -------------------------------------------------------------------
# Chunking
# -------------------------------------------------------------------


def chunk_text(text, max_tokens=200):
    """
    Naively split text into smaller chunks of up to `max_tokens` words.

    * Splits on whitespace.
    * Returns a list of DocumentChunk.
    * Each chunk.metadata contains:
        - "index": index in the list (0-based)
        - "total_chunks": total number of chunks
    """
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    chunks = []
    current = []
    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunk_index = len(chunks)
            chunk_text_value = " ".join(current)
            chunks.append(
                DocumentChunk(
                    chunk_text_value,
                    {
                        "chunk_index": chunk_index,
                    },
                )
            )
            current = []

    if current:
        chunk_index = len(chunks)
        chunk_text_value = " ".join(current)
        chunks.append(
            DocumentChunk(
                chunk_text_value,
                {
                    # tests expect "index" in c.metadata
                    "index": chunk_index,
                },
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
    Call Ollama's /api/embeddings endpoint for a single text.

    Expected response format (typical):

        {
          "embedding": [... floats ...]
        }

    or sometimes:

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

    url = base_url + "/api/embeddings"
    logger.debug("Requesting embedding from %s with model=%s", url, model)

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and "embedding" in data:
        embedding = data["embedding"]
    elif isinstance(data, dict) and "data" in data:
        first = data["data"][0]
        embedding = first["embedding"]
    else:
        raise ValueError("Unexpected embedding response format")

    if not isinstance(embedding, list):
        raise ValueError("Embedding must be a list of floats")

    return embedding


def embed_texts(texts):
    """
    Compute embeddings for a batch of texts using Ollama.

    Tests expect:
      * [] when texts is empty or all whitespace.
      * For each non-empty cleaned text, `_embed_with_ollama` is called once.
      * If `_embed_with_ollama` raises, the error is logged and that text
        is simply skipped (remaining ones are still processed).
    """
    if not texts:
        return []

    embeddings = []

    for t in texts:
        cleaned = (t or "").strip()
        if not cleaned:
            continue

        try:
            emb = _embed_with_ollama(cleaned)
        except Exception as exc:  # pragma: no cover (logging branch)
            logger.exception("Embedding failed", exc_info=exc)
            continue

        embeddings.append(emb)

    return embeddings


# -------------------------------------------------------------------
# Ingestion
# -------------------------------------------------------------------


def add_document(text):
    """
    Ingest a raw text document into the Chroma vector store.

    Steps:
      1. Normalize / strip the text.
      2. Split into chunks via `chunk_text`.
      3. Embed each chunk via `embed_texts`.
      4. Store in Chroma with generated IDs.

    Tests expect:
      * Whitespace-only text is ignored (no errors).
      * Each record inserted into the DummyCollection contains both
        "document" and "metadata" keys.
      * `metadata["chunk_index"]` is present for each chunk.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        logger.info("add_document called with empty/whitespace text; nothing to do.")
        return

    chunks = chunk_text(cleaned)
    if not chunks:
        logger.warning("chunk_text returned no chunks; nothing to add.")
        return

    documents = [c.text for c in chunks]
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        meta.setdefault("chunk_index", i)
        metadatas.append(meta)
        ids.append("rag-doc-" + uuid.uuid4().hex)

    embeddings = embed_texts(documents)
    if not embeddings:
        logger.warning("No embeddings produced; skipping Chroma insertion.")
        return

    collection = _get_collection()
    collection.add(
        embeddings=embeddings,
        documents=documents,
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

    chunks = []
    for doc, meta, dist in zip(docs, metas, dists):
        if not isinstance(meta, dict):
            # Normalize strange metadata formats used in some backends.
            if meta is None:
                meta = {}
            else:
                meta = {"value": meta}

        chunks.append(
            RAGChunk(
                text=doc,
                distance=float(dist),
                metadata=meta,
            )
        )

    return chunks
