# RAG & Tabular Learning Platform (Agentic PEV)

This repository contains the architecture for a unified Data Asset Platform capable of generic document ingestion, dynamic DuckDB schema materialization, and intelligent Plan-Execute-Verify (PEV) chat routing.

## 🌟 Core Features

- **Unified Assets API**: Upload any form of data (PDF, TXT, CSV, Excel) via a single unified endpoint (`/v1/assets`).
- **Hybrid Processing**:
  - Text-based documents are chunked and inserted into `pgvector` for semantic RAG search.
  - Tabular datasets are pushed into local `DuckDB` persistence for dynamic analytical SQL querying.
- **Agent PEV (Plan-Execute-Verify)**: The Chat system natively supports function calling. The Gemini LLM acts as an Agent that looks at user intent, probes Dataset schemas (`get_dataset_schema`), runs internal calculations (`run_duckdb_sql`), and searches knowledge bases (`search_knowledge`) simultaneously before returning a verified answer.

## 🏗️ Architecture

- `web/`: Next.js dashboard with Supabase-authenticated login and consolidated Asset/Chat views.
- `api/`: FastAPI service managing Workspaces, Assets, and hosting the Agentic LLM loop.
- `worker/`: A resilient background processor handling Document parsers (Chunking/Embeddings) and Tabular parsers (Pandas to DuckDB serialization).
- `infra/`: PostgreSQL+pgvector, Redis, and DuckDB volume mounts.

## 🚀 Local Deployment Prerequisites

- Node.js 20+
- Python 3.12+
- Docker Desktop
- A Supabase project for Auth & Storage (`STORAGE_BACKEND=supabase` or `local`).
- Gemini API Key (`GEMINI_API_KEY`) for Embeddings and PEV Agent.

## 🛠️ Environment Configuration

Copy these files and fill them in with appropriate keys:

- `web/.env.example` -> `web/.env.local`
- `api/.env.example` -> `api/.env`
- `worker/.env.example` -> `worker/.env`

_Ensure `STORAGE_BACKEND` matches across Web, API, and Worker._

## 🐳 Infrastructure Startup

Start PostgreSQL, Redis, and initialize volume boundaries:

```powershell
docker compose -f infra/docker-compose.yml up -d
```

## 🌐 Web Dashboard

```powershell
npm install
npm run web:dev
```

## ⚙️ Backend API & Worker

It is highly recommended to run the API and Worker in distinct terminal tabs within their virtual environments:

**Start API:**

```powershell
C:\Users\jvb\AppData\Local\Python\bin\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -e .\api
.venv\Scripts\python.exe -m uvicorn api.app.main:app --reload --app-dir .
```

**Start Worker:**

```powershell
.venv\Scripts\python.exe -m pip install -e .\worker
.venv\Scripts\python.exe -m worker.main
```

## 🧪 Tests

Comprehensive unit tests including Agent Tool Simulation, SQL Injection guards, and generic Asset workflows are provided.

Run the full suite (including Mock E2E Smoke testing):

```powershell
python -m pytest api/tests/
```
