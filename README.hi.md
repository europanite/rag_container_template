---
layout: page
title: "🇮🇳 हिंदी"
permalink: /hi/
lang: hi
---

<p align="center">
  <a href="https://europanite.github.io/rag_container_template/">🇺🇸 English</a> |
  <a href="https://europanite.github.io/rag_container_template/hi/">🇮🇳 हिंदी</a> |
  <a href="https://europanite.github.io/rag_container_template/ja/">🇯🇵 日本語</a> |
  <a href="https://europanite.github.io/rag_container_template/zh-CN/">🇨🇳 简体中文</a> |
  <a href="https://europanite.github.io/rag_container_template/es/">🇪🇸 Español</a> |
  <a href="https://europanite.github.io/rag_container_template/pt-BR/">🇧🇷 Português (Brasil)</a> |
  <a href="https://europanite.github.io/rag_container_template/ko/">🇰🇷 한국어</a> |
  <a href="https://europanite.github.io/rag_container_template/de/">🇩🇪 Deutsch</a> |
  <a href="https://europanite.github.io/rag_container_template/fr/">🇫🇷 Français</a>
</p>

> **Note:** यह अनुवादित संस्करण है। अंग्रेज़ी `README.md` ही आधिकारिक स्रोत है।

# [RAG Container Template](https://github.com/europanite/rag_container_template "RAG Container Template")

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![OS](https://img.shields.io/badge/OS-Linux%20%7C%20macOS%20%7C%20Windows-blue)
[![Python](https://img.shields.io/badge/python-3.9|%203.10%20|%203.11|%203.12|%203.13-blue)](https://www.python.org/)
[![CI](https://github.com/europanite/rag_container_template/actions/workflows/ci.yml/badge.svg)](https://github.com/europanite/rag_container_template/actions/workflows/ci.yml)
[![Python Lint](https://github.com/europanite/rag_container_template/actions/workflows/lint.yml/badge.svg)](https://github.com/europanite/rag_container_template/actions/workflows/lint.yml)
[![CodeQL Advanced](https://github.com/europanite/rag_container_template/actions/workflows/codeql.yml/badge.svg)](https://github.com/europanite/rag_container_template/actions/workflows/codeql.yml)
[![pages-build-deployment](https://github.com/europanite/rag_container_template/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/europanite/rag_container_template/actions/workflows/pages/pages-build-deployment)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pytest](https://img.shields.io/badge/pytest-%23ffffff.svg?style=for-the-badge&logo=pytest&logoColor=2f9fe3)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![React Native](https://img.shields.io/badge/react_native-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)
![Jest](https://img.shields.io/badge/-jest-%23C21325?style=for-the-badge&logo=jest&logoColor=white)
![Expo](https://img.shields.io/badge/expo-1C1E24?style=for-the-badge&logo=expo&logoColor=#D04A37)

!["UI"](./assets/images/frontend.png)

यह रिपॉज़िटरी **local Retrieval-Augmented Generation (RAG)** सिस्टम बनाने के लिए एक full-stack sandbox है।  
Backend एक FastAPI service है जिसमें authentication और RAG API शामिल हैं। यह persistent vector store के रूप में **ChromaDB** और embeddings तथा chat दोनों के लिए **Ollama** का उपयोग करता है। Frontend एक Expo / React Native app है जो backend से संवाद करता है।

---

## Features

- **Backend**
  - FastAPI

- **Frontend**
  - Expo / React-Native

- **DataBase**
  - PostgreSQL

- **RAG (Retrieval-Augmented Generation)**
  - Ollama के साथ **Embeddings**
  - ChromaDB के साथ **Vector store**
  - **Chat / Answer generation** 

- **DevOps**
  - **Docker Compose**
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
- [Expo Go](https://expo.dev/go)

### 2. Build and start all services:

```bash
# set environment variables:
export REACT_NATIVE_PACKAGER_HOSTNAME=${YOUR_HOST}

# Build the image
docker compose build

# Run the container
docker compose up
```
---

### 3. Test:

```bash
# Backend pytest
docker compose \
  -f docker-compose.test.yml run \
  --rm \
  --entrypoint /bin/sh backend_test \
  -lc 'pytest -q'

# Backend Lint
docker compose \
  -f docker-compose.test.yml run \
  --rm \
  --entrypoint /bin/sh backend_test \
  -lc 'ruff check /app /tests'

# Frontend Test
docker compose \
  -f docker-compose.test.yml run \
  --rm frontend_test
```

## Visit the services:

- Backend API: http://localhost:8000/docs
!["backend"](./assets/images/backend.png)

- Frontend UI (WEB): http://localhost:8081
- Frontend UI (mobile): exp://${YOUR_HOST}:8081: Expo द्वारा दिए गए QR से इसे खोलें।
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
