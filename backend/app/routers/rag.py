import logging
import os
from typing import List

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import rag_store

router = APIRouter(prefix="/rag", tags=["rag"])

logger = logging.getLogger(__name__)

# ---- Ollama config ----

OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
DEFAULT_OLLAMA_BASE_URL = "http://ollama:11434"
RAG_ANSWER_MODEL_ENV = "RAG_ANSWER_MODEL"
DEFAULT_RAG_ANSWER_MODEL = "llama3"

_session = requests.Session()


def _get_ollama_base_url() -> str:
    return os.getenv(OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_BASE_URL).rstrip("/")


def _get_answer_model() -> str:
    return os.getenv(RAG_ANSWER_MODEL_ENV, DEFAULT_RAG_ANSWER_MODEL)


# ---- Schemas ----


class IngestDocument(BaseModel):
    """
    A single raw document to be ingested into the RAG system.
    """

    id: str = Field(..., description="Logical document id")
    text: str = Field(..., description="Full text content of the document")
    source: str = Field(..., description="Human-readable source (URL, file name, etc.)")


class IngestRequest(BaseModel):
    documents: List[IngestDocument]


class IngestResponse(BaseModel):
    total_chunks: int = Field(
        ..., description="Total number of chunks stored in the vector database"
    )


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve from the vector store",
    )


class RetrievedChunk(BaseModel):
    id: str
    text: str
    source: str
    chunk_index: int
    distance: float


class QueryResponse(BaseModel):
    answer: str
    chunks: List[RetrievedChunk]


SYSTEM_PROMPT = (
    "Use only the provided context from local documents"
    "to answer user questions. If the answer is not clearly in the context, say that you do not "
    "know based on the current documents, and suggest that the user look for more up-to-date "
    "information."
)


# ---- Ollama chat helper ----


def _call_ollama_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Call Ollama's /api/chat endpoint and return the assistant message text.
    """
    base_url = _get_ollama_base_url()
    model = _get_answer_model()
    url = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    try:
        resp = _session.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while calling Ollama chat: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Failed to call Ollama chat backend.",
        ) from exc

    data = resp.json()
    message = data.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        logger.error("Unexpected response from Ollama chat: %r", data)
        raise HTTPException(
            status_code=502,
            detail="Unexpected response from Ollama chat backend.",
        )

    return content

# ---- Endpoints ----

@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    """
    Ingest one or more documents into the vector database.

    This endpoint:
    - splits each document into chunks
    - embeds them with Ollama embeddings
    - stores them in Chroma with metadata (source, doc_id, chunk_index)
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    total_chunks = 0
    for doc in request.documents:
        total_chunks += rag_store.add_document(
            doc_id=doc.id,
            text=doc.text,
            source=doc.source,
        )

    return IngestResponse(total_chunks=total_chunks)


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Run a full RAG cycle:

    1. Embed the question (rag_store -> Ollama embeddings)
    2. Retrieve top_k similar chunks from Chroma
    3. Call Ollama chat with those chunks as context
    4. Return the model answer plus the raw chunks (for frontend citation display)
    """
    retrieval = rag_store.query_similar_chunks(
        question=request.question,
        top_k=request.top_k,
    )

    documents = retrieval["documents"]
    metadatas = retrieval["metadatas"]
    distances = retrieval["distances"]

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No relevant context found. Please ingest documents first.",
        )

    # Build a context block for the LLM and a structured list for the API response
    context_parts: List[str] = []
    chunk_objects: List[RetrievedChunk] = []

    for doc_text, meta, dist in zip(documents, metadatas, distances):
        source = meta.get("source", "")
        doc_id = meta.get("doc_id", "")
        chunk_index = int(meta.get("chunk_index", -1))

        chunk_id = f"{doc_id}_{chunk_index}"

        context_parts.append(
            f"[source={source} doc_id={doc_id} chunk={chunk_index} "
            f"distance={float(dist):.4f}]\n"
            f"{doc_text}"
        )

        chunk_objects.append(
            RetrievedChunk(
                id=chunk_id,
                text=doc_text,
                source=source,
                chunk_index=chunk_index,
                distance=float(dist),
            )
        )

    context_str = "\n\n---\n\n".join(context_parts)

    user_prompt = (
        "Answer the following question based ONLY on the context.\n\n"
        f"Question:\n{request.question}\n\n"
        "Context:\n"
        f"{context_str}\n\n"
        'If the answer is not clearly contained in the context, say '
        '"I do not know based on the current documents."'
    )

    answer_text = _call_ollama_chat(SYSTEM_PROMPT, user_prompt)

    return QueryResponse(
        answer=answer_text,
        chunks=chunk_objects,
    )
