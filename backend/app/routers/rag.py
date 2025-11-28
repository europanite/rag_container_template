# backend/app/routers/rag.py

from __future__ import annotations

import logging
import os
from http import HTTPStatus
from typing import List

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import rag_store
from rag_store import RAGChunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------


class IngestRequest(BaseModel):
    documents: List[str] = Field(..., description="List of raw document texts.")


class IngestResponse(BaseModel):
    ingested: int = Field(..., description="Number of successfully ingested documents.")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question.")
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of similar chunks to retrieve from the vector store.",
    )


class QueryResponse(BaseModel):
    answer: str
    # テスト側で data["context"] == [ ... ] として比較している
    context: List[str]


# -------------------------------------------------------------------
# Ollama chat wrapper (テストから monkeypatch される)
# -------------------------------------------------------------------


def _get_ollama_chat_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")


def _get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")


def _call_ollama_chat(*, question: str, context: str) -> str:
    """
    Call Ollama's /api/chat endpoint with a simple RAG-style prompt.

    テストではこの関数を monkeypatch して使うので、
    ここでは素直な実装にしておく。
    """
    base_url = _get_ollama_base_url()
    model = _get_ollama_chat_model()

    prompt = (
        "You are a helpful assistant for questions about various topics.\n\n"
        "Use ONLY the information in the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in English."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You answer using the given context."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Ollama の標準レスポンスに合わせた取り出し
    message = data.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("Ollama chat response missing 'message.content'")

    return content


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    """
    Ingest a list of documents into the vector store.

    テスト仕様:
      - POST /rag/ingest {"documents": ["Doc1", "Doc2"]}
      - 成功: 200, {"ingested": 2}
      - 空リスト: 400
      - 全件失敗: 502, detail に "failed" を含む
    """
    docs = request.documents or []

    if not docs:
        # test_rag_ingest_empty_documents_returns_400
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="documents list must not be empty",
        )

    successes = 0
    last_error: Exception | None = None

    for text in docs:
        try:
            rag_store.add_document(text)
            successes += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to ingest document", exc_info=exc)
            last_error = exc

    if successes == 0 and last_error is not None:
        # test_rag_ingest_all_fail_returns_502: detail に "failed" を含める
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=f"Document ingestion failed: {last_error}",
        )

    return IngestResponse(ingested=successes)


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Run a full RAG cycle:

      1. Embed the question
      2. Retrieve top_k similar chunks from Chroma
      3. Call Ollama chat with those chunks as context
      4. Return the model answer plus the raw context texts (for frontend display)
    """
    # --- Retrieve from vector store ---------------------------------
    try:
        # テストの fake は def fake_query_similar_chunks(question, top_k=3) なので
        # 「question」は **位置引数** で渡す（キーワードにすると TypeError）。
        chunks: List[RAGChunk] = rag_store.query_similar_chunks(
            request.question,
            top_k=request.top_k,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vector store query failed", exc_info=exc)
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=str(exc),
        )

    if not chunks:
        # test_rag_query_no_context_returns_404
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="No relevant context found for the given question.",
        )

    context_texts = [c.text for c in chunks]
    context_block = "\n\n".join(context_texts)

    # --- Call Ollama chat ------------------------------------------
    try:
        answer = _call_ollama_chat(question=request.question, context=context_block)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ollama chat failed", exc_info=exc)
        # test_rag_query_ollama_failure_returns_502 で
        # detail に "Ollama is down" を含めているので、そのまま str(exc) を返す
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=str(exc),
        )

    # test_rag_query_success expectations:
    #   - status_code == 200
    #   - data["answer"].startswith("ANSWER to:")
    #   - data["context"] == ["Miura Peninsula is located in Kanagawa Prefecture."]
    return QueryResponse(answer=answer, context=context_texts)
