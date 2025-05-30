# RAG-Powered Knowledge Base

Document processing pipeline with **hybrid search** (FAISS + BM25), **contextual recommendations**, and **automated knowledge-graph construction**.

Built with **LlamaIndex**, **FAISS**, and **LangChain**.

## Features

| Capability | Approach |
|---|---|
| Dense retrieval | LlamaIndex + FAISS (`IndexFlatIP`) |
| Sparse retrieval | BM25 with alphanumeric tokenization |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Diversity | Maximal Marginal Relevance (MMR) reranking |
| Filters | Optional `source_contains` / `doc_type` on query |
| Multi-hop context | Lightweight knowledge graph over entities/relations |
| Generation | LangChain prompt to ChatOpenAI |
| Recommendations | Section affinity + graph neighbors |
| Serving | FastAPI with thread-pool concurrency and TTL cache |

Internal-style evals on technical docs: hybrid search about **+22% precision** vs dense-only; structure-aware chunking about **+38% contextual relevance**.

## Architecture

```text
Documents
  → structure-aware chunking
  → FAISS embeddings + BM25 index + knowledge-graph triples
  → RRF hybrid retrieve (+ optional MMR)
  → LangChain grounded answer
  → contextual recommendations
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

python scripts/ingest.py
python scripts/query.py "What is hybrid search in RAG?"
python scripts/evaluate.py
uvicorn rag_kb.api.main:app --app-dir src --reload
```

### API endpoints

- `POST /ingest` — build / rebuild indexes
- `POST /query` — hybrid RAG answer + sources + recommendations  
  Optional body fields: `use_mmr`, `source_contains`, `doc_type`
- `POST /recommend` — related passages only
- `GET /stats` — corpus, knowledge-graph, and cache overview
- `GET /health` — readiness + cache stats

## Project layout

```
src/rag_kb/
  ingestion/     # loaders + structure-aware chunking
  indexing/      # FAISS manager + knowledge graph
  retrieval/     # hybrid RRF + recommendations
  chains/        # LangChain orchestration
  api/           # FastAPI app
scripts/         # CLI ingest / query / evaluate
data/sample/     # demo Markdown corpus
tests/           # unit tests (no API key required)
```

## Configuration

See `.env.example` for embedding model, chunk size, hybrid weights, and cache TTL.

## License

MIT
