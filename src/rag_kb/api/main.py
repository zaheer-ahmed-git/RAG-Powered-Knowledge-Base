"""FastAPI service exposing ingest, query, and recommendation endpoints."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_kb.cache import query_cache
from rag_kb.chains.rag_chain import KnowledgeBasePipeline
from rag_kb.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = KnowledgeBasePipeline()
executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    nodes_file = settings.docstore_dir / "nodes.json"
    if nodes_file.exists():
        try:
            count = pipeline.load()
            logger.info("Loaded existing index (%d chunks)", count)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load index on startup: %s", exc)
    yield
    executor.shutdown(wait=False)


app = FastAPI(
    title="RAG-Powered Knowledge Base",
    description="Hybrid FAISS + BM25 search with knowledge-graph recommendations",
    version="0.1.0",
    lifespan=lifespan,
)


class IngestRequest(BaseModel):
    data_dir: Optional[str] = None
    persist: bool = True


class IngestResponse(BaseModel):
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    use_cache: bool = True
    include_recommendations: bool = True
    use_mmr: Optional[bool] = None
    source_contains: Optional[str] = Field(
        default=None,
        description="Keep only chunks whose source path/name contains this substring",
    )
    doc_type: Optional[str] = Field(
        default=None,
        description="Filter by document extension/type, e.g. md or pdf",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    recommendations: List[dict]
    cached: bool


class RecommendRequest(BaseModel):
    question: str = Field(..., min_length=2)
    limit: int = Field(5, ge=1, le=20)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "ready": pipeline.retriever is not None,
        "cache": query_cache.stats,
    }


@app.get("/stats")
async def stats() -> dict:
    """Corpus, knowledge-graph, and cache overview."""
    return pipeline.corpus_stats()


@app.post("/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest) -> IngestResponse:
    loop = asyncio.get_running_loop()
    try:
        chunks = await loop.run_in_executor(
            executor,
            lambda: pipeline.ingest(body.data_dir, body.persist),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    query_cache.clear()
    return IngestResponse(chunks=chunks)


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest) -> QueryResponse:
    if pipeline.retriever is None:
        raise HTTPException(status_code=503, detail="Index not ready. Call /ingest first.")
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            executor,
            lambda: pipeline.query(
                body.question,
                use_cache=body.use_cache,
                include_recommendations=body.include_recommendations,
                use_mmr=body.use_mmr,
                source_contains=body.source_contains,
                doc_type=body.doc_type,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        recommendations=result.recommendations,
        cached=result.cached,
    )


@app.post("/recommend")
async def recommend(body: RecommendRequest) -> dict:
    if pipeline.recommender is None:
        raise HTTPException(status_code=503, detail="Index not ready. Call /ingest first.")
    loop = asyncio.get_running_loop()
    recs = await loop.run_in_executor(
        executor,
        lambda: pipeline.recommender.recommend(body.question, limit=body.limit),
    )
    return {
        "recommendations": [
            {
                "title": r.title,
                "source": r.source,
                "score": r.score,
                "reason": r.reason,
            }
            for r in recs
        ]
    }


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "rag_kb.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
