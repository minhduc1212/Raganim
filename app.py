"""
app.py  —  FastAPI web server cho Anime RAG
==========================================
Chạy: uvicorn app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core import RagEngine, SearchRequest
from src.config import (
    CHROMA_PATH, COLLECTION, EMBED_MODEL, DIMENSIONS,
    REWRITE_MODEL, LLM_MODEL, OPENAI_API_KEY, GEMINI_API_KEY, TOP_K
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Engine singleton ──────────────────────────────────────────────────────────

engine: RagEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo engine 1 lần khi startup, cleanup khi shutdown."""
    global engine
    log.info("Starting up RAG engine...")
    engine = RagEngine(
        chroma_path     = CHROMA_PATH,
        collection_name = COLLECTION,
        embed_model     = EMBED_MODEL,
        dimensions      = DIMENSIONS,
        rewrite_model   = REWRITE_MODEL,
        llm_model       = LLM_MODEL,
        openai_api_key  = OPENAI_API_KEY,
        gemini_api_key  = GEMINI_API_KEY,
        top_k           = TOP_K,
    )
    count = await engine.init()
    log.info("Engine ready: %d docs", count)
    yield
    log.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Anime RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # đổi thành domain cụ thể khi deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas (FastAPI cần Pydantic, không dùng dataclass) ─────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K

    model_config = {"json_schema_extra": {"example": {"query": "dark mecha like Evangelion", "top_k": 20}}}


class AnimeItem(BaseModel):
    rank: int
    title: str
    url: str
    mal_score: float
    why: str


class RetrievedItem(BaseModel):
    title: str
    url: str
    relevance: float


class QueryResponse(BaseModel):
    query: str
    rewritten_query: str
    excluded_titles: list[str]
    message: str
    recommendations: list[AnimeItem]
    all_retrieved: list[RetrievedItem]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check."""
    if not engine or not engine.collection:
        raise HTTPException(503, "Engine not ready")
    return {
        "status": "ok",
        "docs":   engine.collection.count(),
        "model":  LLM_MODEL,
    }


@app.post("/search", response_model=QueryResponse)
async def search(request: QueryRequest):
    """
    Main search endpoint.

    - Rewrites query semantically
    - Extracts titles to exclude (e.g. "anime similar to X" → exclude X)
    - Vector search ChromaDB
    - LLM ranks and explains results
    """
    if not engine:
        raise HTTPException(503, "Engine not ready")

    try:
        result = await engine.search(
            SearchRequest(query=request.query, top_k=request.top_k)
        )
        # Convert dataclasses → dict → Pydantic
        return QueryResponse(
            query           = result.query,
            rewritten_query = result.rewritten_query,
            excluded_titles = result.excluded_titles,
            message         = result.message,
            recommendations = [AnimeItem(**asdict(r)) for r in result.recommendations],
            all_retrieved   = [RetrievedItem(**asdict(r)) for r in result.all_retrieved],
        )
    except Exception as e:
        log.error("Search error: %s", e, exc_info=True)
        raise HTTPException(500, str(e))