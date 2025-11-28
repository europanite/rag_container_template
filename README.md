# [RAG Container Template](https://github.com/europanite/rag_container_template "RAG Container Template")

!["UI"](./assets/images/frontdnd.png)

This repository is a full-stack sandbox for building a **local Retrieval-Augmented Generation (RAG)** system.  
The backend is a FastAPI service with authentication and a RAG API, using **ChromaDB** as a persistent vector store and **Ollama** for both embeddings and chat. The frontend is an Expo / React Native app that talks to the backend.

---

## Features

- **Backend**
  - FastAPI application (`backend/app`) with:
    - `/health` endpoint for basic health checks
    - `/auth` endpoints (`/signup`, `/signin`) using JWT-based auth
    - `/rag` endpoints for document ingestion and question answering :contentReference[oaicite:1]{index=1}
  - PostgreSQL for application data (`db/init/*.sql` for initial schema)
  - High test coverage with `pytest` and `coverage.xml` reports in `backend/reports/ci/` :contentReference[oaicite:2]{index=2}

- **RAG (Retrieval-Augmented Generation)**
  - **Embeddings** via Ollama’s `/api/embeddings` endpoint  
    - Default model: `mxbai-embed-large`
  - **Vector store** using ChromaDB `PersistentClient` stored under `backend/app/chroma_db/`
  - **Chat / Answer generation** via Ollama’s `/api/chat` endpoint  
    - Default model: `llama3`
  - Simple character-based **chunking with overlap** and metadata (source, doc_id, chunk_index) in `rag_store.py` :contentReference[oaicite:3]{index=3}

- **Frontend**
  - Expo / React Native app under `frontend/app`
  - Screens:
    - `HomeScreen.tsx`: calls the RAG backend and displays answers/chunks
    - `SignInScreen.tsx` / `SignUpScreen.tsx`: simple auth flow
  - Jest tests for screens, components, and context (Auth, SettingsBar, etc.) :contentReference[oaicite:4]{index=4}

- **DevOps**
  - Dockerfiles for backend and frontend
  - `docker-compose.yml` to bring up db, backend, and frontend together
  - GitHub Actions workflows

---

## Architecture

```text
+-----------------------------+
|        Frontend (Expo)      |
|  - React Native app         |
|  - Calls backend /auth,     |
|    /items, /rag endpoints   |
+--------------+--------------+
               |
               v
+-----------------------------+
|       Backend (FastAPI)     |
|  - Auth & Items routers     |
|  - RAG router (/rag/...)    |
|  - SQLAlchemy + Postgres    |
+--------------+--------------+
               |
       +-------+----------+
       |                  |
       v                  v
+-------------+   +------------------+
|  ChromaDB   |   |   Ollama (LLM)   |
|  Vector DB  |   |  /api/chat       |
|  /chroma_db |   |  /api/embeddings |
+-------------+   +------------------+
```

---

## 🚀 Getting Started

### 1. Prerequisites
- [Docker Compose](https://docs.docker.com/compose/)
- [Expo Go](https://expo.dev/go) (for Android/iOS testing)

### 2. Build and start all services:

```bash
# set environment variables:
export REACT_NATIVE_PACKAGER_HOSTNAME=192.168.3.6

# Build the image
docker compose build

# Run the container
docker compose up
```
---

## Visit the services:

- Backend API: http://localhost:8000/docs
!["backend"](./assets/images/backend.png)

- Frontend UI (WEB): http://localhost:8081
- Frontend UI (mobile): exp://${YOUR_HOST}:8081: access it with the QR provided by Expo.
!["expo"](./assets/images/expo.png)


---

## RAG API

### Ingest documents

#### POST /rag/ingest

```bash
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "miura_intro_001",
        "text": "Miura Peninsula is located in Kanagawa, south of Yokohama. It is famous for its coastline, fresh seafood, and views of Mount Fuji on clear days.",
        "source": "local-notes"
      }
    ]
  }'
```

#### Response:
```bash
{
  "total_chunks": 1
}
```

### Ask a question

#### POST /rag/query

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Where is the Miura Peninsula and what is it famous for?",
    "top_k": 5
  }'
```

#### Example response:

```bash
{
  "answer": "The Miura Peninsula is in Kanagawa, south of Yokohama. It is known for its coastline, fresh seafood, and views of Mount Fuji on clear days.",
  "chunks": [
    {
      "id": "miura_intro_001_0",
      "text": "...",
      "source": "local-notes",
      "chunk_index": 0,
      "distance": 0.01
    }
  ]
}
```

---

# License
- Apache License 2.0