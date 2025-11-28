from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import chromadb
import requests

logger = logging.getLogger(__name__)

# ---- Settings ----

# Embedding model name used by Ollama
EMBEDDING_MODEL_ENV = "EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"

# For documentation only: the actual dimension depends on the Ollama model.
# We do not rely on a fixed value anywhere in the code.
EMBEDDING_DIMENSION = 0

# Chroma settings
CHROMA_DIR_ENV = "CHROMA_DIR"
DEFAULT_CHROMA_DIR = "./chroma_db"

CHROMA_COLLECTION_ENV = "CHROMA_COLLECTION_ENV"
DEFAULT_COLLECTION_NAME = "CHROMA_COLLECTION"

# Ollama settings
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
# For Docker-compose you might set this to "http://ollama:11434"
# For a host-local Docker container, "http://host.docker.internal:11434" may be needed.
DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"


def _get_embedding_model() -> str:
    """Return the Ollama model name to use for embeddings."""
    return os.getenv(EMBEDDING_MODEL_ENV, DEFAULT_EMBEDDING_MODEL)


def _get_ollama_base_url() -> str:
    """Return the base URL for the Ollama HTTP API."""
    return os.getenv(OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_BASE_URL).rstrip("/")


# ---- Ollama & Chroma clients ----

# Reuse a single HTTP session for all embedding calls
_session = requests.Session()

_chroma_dir = os.getenv(CHROMA_DIR_ENV, DEFAULT_CHROMA_DIR)

# PersistentClient stores vectors on disk so the index survives restarts
_chroma_client = chromadb.PersistentClient(path=_chroma_dir)

_collection_name = os.getenv(CHROMA_COLLECTION_ENV, DEFAULT_COLLECTION_NAME)
_collection = _chroma_client.get_or_create_collection(name=_collection_name)

logger.info(
    "Initialized Chroma collection '%s' in '%s'", _collection_name, _chroma_dir
)
logger.info(
    "Using Ollama embeddings model='%s' base_url='%s'",
    _get_embedding_model(),
    _get_ollama_base_url(),
)


@dataclass
class DocumentChunk:
    """
    A single chunk of a larger document.
    """

    id: str
    text: str
    source: str
    doc_id: str
    chunk_index: int

# ---- Chunking ----

def chunk_text(text: str, max_chars: int = 800, overlap: int = 200) -> List[str]:
    """
    Very simple character-based chunking with overlap.
    """
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + max_chars, length)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        # Move start forward with overlap
        start = max(0, end - overlap)

    return chunks

# ---- Embeddings via Ollama ----

def _embed_with_ollama(text: str, model_name: str) -> List[float]:
    """
    Call Ollama's /api/embeddings endpoint for a single text and return a vector.
    """
    base_url = _get_ollama_base_url()
    url = f"{base_url}/api/embeddings"
    payload = {"model": model_name, "prompt": text}

    try:
        response = _session.post(url, json=payload, timeout=60)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while calling Ollama embeddings: %s", exc)
        raise RuntimeError(f"Failed to get embeddings from Ollama at {url}") from exc

    data = response.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        logger.error("Unexpected Ollama embeddings response: %r", data)
        raise RuntimeError("Unexpected response format from Ollama embeddings API")

    return embedding


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """
    Compute embeddings for a batch of texts using Ollama.

    Ollama's /api/embeddings endpoint currently accepts a single prompt,
    so we call it once per text.
    """
    if not texts:
        return []

    model_name = _get_embedding_model()
    embeddings: List[List[float]] = []

    for text in texts:
        cleaned = text.strip()
        if not cleaned:
            # Keep alignment between input texts and embeddings
            embeddings.append([])
            continue

        emb = _embed_with_ollama(cleaned, model_name)
        embeddings.append(emb)

    return embeddings

# ---- Add documents to Chroma ----

def add_document(doc_id: str, text: str, source: str) -> int:
    """
    Split a document into chunks, embed them, and insert into Chroma.

    Args:
        doc_id: Logical ID of the document (e.g., filename or URL).
        text: Raw text of the document.
        source: Human-readable source label (e.g., "guidebook", "blog", etc.).

    Returns:
        Number of chunks that were stored.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_{idx}"
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(
            {
                "source": source,
                "doc_id": doc_id,
                "chunk_index": idx,
            }
        )

    embeddings = embed_texts(documents)

    _collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # PersistentClient automatically writes to disk when the process exits.
    # For notebooks you might want _chroma_client.persist(), but not needed here.
    return len(chunks)

# ---- Similarity search ----

def query_similar_chunks(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Embed the question and retrieve top_k similar document chunks from Chroma.
    """
    question = question.strip()
    if not question:
        return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    [query_embedding] = embed_texts([question])

    result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # Chroma returns lists for each query (here we only issue one query)
    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    return {
        "ids": ids,
        "documents": docs,
        "metadatas": metas,
        "distances": dists,
    }
