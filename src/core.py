"""
rag_engine.py  —  Async RAG engine module cho FastAPI
=====================================================
Import vào app.py:
    from rag_engine import RagEngine, SearchRequest, SearchResponse
"""

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from openai import AsyncOpenAI
from google import genai
from google.genai import types

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Pydantic-style dataclasses (dùng được cả với/không có Pydantic)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SearchRequest:
    query: str
    top_k: int = 20


@dataclass
class AnimeResult:
    rank: int
    title: str
    url: str
    mal_score: float
    why: str


@dataclass
class RetrievedItem:
    title: str
    url: str
    relevance: float


@dataclass
class SearchResponse:
    query: str
    rewritten_query: str
    excluded_titles: list[str]
    message: str
    recommendations: list[AnimeResult]
    all_retrieved: list[RetrievedItem]


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_text(response) -> str | None:
    """
    Extract text từ Gemini response an toàn.
    response.text trả None khi bị safety filter / MAX_TOKENS.
    """
    try:
        if response.text is not None:
            return response.text
    except Exception:
        pass
    try:
        for candidate in (response.candidates or []):
            for part in (candidate.content.parts or []):
                if hasattr(part, "text") and part.text:
                    return part.text
    except Exception:
        pass
    return None


def _parse_json_response(raw: str | None, fallback: dict) -> dict:
    """Parse JSON response, strip markdown fences, fallback nếu fail."""
    if not raw:
        return fallback
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("JSON decode failed, using fallback")
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
#  RagEngine
# ══════════════════════════════════════════════════════════════════════════════

class RagEngine:
    """
    Async RAG engine. Khởi tạo 1 lần, dùng nhiều lần.

    Usage:
        engine = RagEngine(config)
        await engine.init()
        result = await engine.search(SearchRequest(query="..."))
    """

    def __init__(
        self,
        chroma_path: str,
        collection_name: str,
        embed_model: str,
        dimensions: int,
        rewrite_model: str,
        llm_model: str,
        openai_api_key: str,
        gemini_api_key: str,
        top_k: int = 20,
    ):
        self.chroma_path      = chroma_path
        self.collection_name  = collection_name
        self.embed_model      = embed_model
        self.dimensions       = dimensions
        self.rewrite_model    = rewrite_model
        self.llm_model        = llm_model
        self.top_k            = top_k

        self.openai   = AsyncOpenAI(api_key=openai_api_key)
        self.genai    = genai.Client(api_key=gemini_api_key)
        self.collection = None  # set sau khi init()

    async def init(self):
        """Kết nối ChromaDB — gọi 1 lần khi startup."""
        chroma = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = chroma.get_collection(name=self.collection_name)
        count = self.collection.count()
        log.info("ChromaDB connected: %d docs in '%s'", count, self.collection_name)
        return count

    # ── Embed ──────────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        response = await self.openai.embeddings.create(
            model=self.embed_model,
            input=text,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding

    # ── Query rewrite ──────────────────────────────────────────────────────

    async def _rewrite_query(self, query: str) -> dict:
        """
        Trả về:
          rewritten_query — semantic search string
          excluded_titles — anime cần loại khỏi kết quả
        """
        prompt = f"""You are an anime search assistant.
User query: "{query}"

Tasks:
1. Extract any specific anime titles the user wants to find SIMILAR anime to.
2. Rewrite the query into a rich semantic search string describing genres, themes,
   plot elements, mood, tone. Do NOT include the extracted titles in the rewrite.

Respond ONLY with JSON (no markdown):
{{
  "rewritten_query": "...",
  "excluded_titles": ["title1"]
}}"""

        try:
            response = self.genai.models.generate_content(
                model=self.rewrite_model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            )
            raw  = _safe_text(response)
            data = _parse_json_response(raw, {})
            return {
                "rewritten_query": data.get("rewritten_query") or query,
                "excluded_titles": data.get("excluded_titles") or [],
            }
        except Exception as e:
            log.warning("Query rewrite failed: %s", e)
            return {"rewritten_query": query, "excluded_titles": []}

    # ── Vector search ──────────────────────────────────────────────────────

    def _vector_search(
        self,
        embedding: list[float],
        top_k: int,
        excluded_titles: list[str],
    ) -> list[dict]:
        # Lấy dư 3× để bù cho những kết quả bị filter
        fetch_k = top_k * 3 if excluded_titles else top_k

        results   = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(fetch_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        excluded_lower = [t.lower() for t in excluded_titles]

        combined = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            if excluded_lower:
                title = str(meta.get("title", "")).lower()
                if any(ex in title for ex in excluded_lower):
                    continue
            combined.append({
                "doc":       doc,
                "meta":      meta,
                "relevance": round(1 - dist, 4),
            })

        combined.sort(
            key=lambda x: (x["relevance"], float(x["meta"].get("score", 0))),
            reverse=True,
        )
        return combined[:top_k]

    # ── LLM answer ────────────────────────────────────────────────────────

    def _build_context(self, results: list[dict]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            m = r["meta"]
            lines.append(
                f"[{i}] {m.get('title', '?')} "
                f"(MAL score: {m.get('score', '?')} | relevance: {r['relevance']})\n"
                f"{r['doc']}\n"
                f"URL: {m.get('url', '')}\n"
            )
        return "\n".join(lines)

    SYSTEM_PROMPT = (
        "You are an expert anime recommender. "
        "Analyze the retrieved list and answer accurately. "
        "Base answers ONLY on context. Rank by intent match, not score. "
        "Always respond in the EXACT JSON format. No text outside JSON."
    )

    async def _ask_llm(self, query: str, results: list[dict]) -> dict:
        context = self._build_context(results)
        n       = len(results)

        prompt = f"""Context ({n} anime retrieved, sorted by relevance):
{context}

User query: {query}

Respond ONLY with JSON:
{{
  "message": "Detailed answer explaining recommendations.",
  "recommendations": [
    {{
      "rank": 1,
      "title": "Anime Title",
      "url": "https://myanimelist.net/...",
      "mal_score": 8.5,
      "why": "One sentence why this matches."
    }}
  ],
  "all_retrieved": [
    {{"title": "Title", "url": "https://...", "relevance": 0.95}}
  ]
}}

recommendations: top picks (max 10). all_retrieved: ALL {n} anime in order."""

        fallback = {
            "message": "Could not generate answer.",
            "recommendations": [],
            "all_retrieved": [
                {"title": r["meta"].get("title", ""), "url": r["meta"].get("url", ""), "relevance": r["relevance"]}
                for r in results
            ],
        }

        try:
            response = self.genai.models.generate_content(
                model=self.llm_model,
                contents=[types.Content(
                    role="user",
                    parts=[types.Part(text=self.SYSTEM_PROMPT + "\n\n" + prompt)]
                )],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    max_output_tokens=8192,
                    temperature=0.1,
                ),
            )
            raw = _safe_text(response)
            if not raw:
                try:
                    reason = response.candidates[0].finish_reason
                    log.warning("LLM blocked: finish_reason=%s", reason)
                except Exception:
                    pass
                return fallback

            return _parse_json_response(raw, fallback)

        except Exception as e:
            log.error("LLM error: %s", e)
            return fallback

    # ── Public API ─────────────────────────────────────────────────────────

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Main entry point — gọi từ FastAPI endpoint."""
        if not self.collection:
            raise RuntimeError("Engine not initialized. Call await engine.init() first.")

        top_k = request.top_k or self.top_k

        # 1. Rewrite query
        rewrite      = await self._rewrite_query(request.query)
        rewritten_q  = rewrite["rewritten_query"]
        excluded     = rewrite["excluded_titles"]
        log.info("Query rewritten: %s | excluded: %s", rewritten_q, excluded)

        # 2. Embed + search
        embedding = await self._embed(rewritten_q)
        results   = self._vector_search(embedding, top_k, excluded)
        log.info("Vector search: %d results", len(results))

        # 3. LLM answer
        answer = await self._ask_llm(request.query, results)

        # 4. Build response
        recs = [
            AnimeResult(
                rank=r.get("rank", i + 1),
                title=r.get("title", ""),
                url=r.get("url", ""),
                mal_score=float(r.get("mal_score", 0)),
                why=r.get("why", ""),
            )
            for i, r in enumerate(answer.get("recommendations", []))
        ]
        retrieved = [
            RetrievedItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                relevance=float(r.get("relevance", 0)),
            )
            for r in answer.get("all_retrieved", [])
        ]

        return SearchResponse(
            query=request.query,
            rewritten_query=rewritten_q,
            excluded_titles=excluded,
            message=answer.get("message", ""),
            recommendations=recs,
            all_retrieved=retrieved,
        )