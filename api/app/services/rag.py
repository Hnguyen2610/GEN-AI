import json
import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator

import httpx
from google import genai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.app.core.config import Settings
from api.app.models.entities import KnowledgeAsset, KnowledgeChunk, KnowledgeVersion
from api.app.services.embedder import QueryEmbedder
from api.app.services.model_aliases import normalize_model_name

logger = logging.getLogger(__name__)

NO_ANSWER_MESSAGE = "Tôi không tìm thấy thông tin phù hợp trong tài liệu của bạn để trả lời câu hỏi này."


@dataclass
class RetrievedContext:
    chunk_id: uuid.UUID
    content: str
    asset_id: uuid.UUID
    asset_title: str
    original_filename: str
    distance: float
    source_page: int | None
    section_title: str | None


class RAGService:
    def __init__(self, settings: Settings, embedder: QueryEmbedder, session: AsyncSession):
        self._settings = settings
        self._embedder = embedder
        self._session = session

    def _normalize_for_matching(self, value: str) -> str:
        normalized = unicodedata.normalize("NFD", value.casefold())
        normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        return normalized.replace("\u0111", "d")

    async def retrieve_context(self, workspace_id: uuid.UUID, query: str) -> list[RetrievedContext]:
        query_embedding = await self._embedder.embed_query(query)

        statement = (
            select(
                KnowledgeChunk,
                KnowledgeChunk.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .join(KnowledgeChunk.knowledge_version)
            .join(KnowledgeVersion.knowledge_asset)
            .where(KnowledgeAsset.workspace_id == workspace_id)
            .where(KnowledgeChunk.embedding.isnot(None))
            .options(
                selectinload(KnowledgeChunk.knowledge_version).selectinload(
                    KnowledgeVersion.knowledge_asset
                )
            )
            .order_by(KnowledgeChunk.embedding.cosine_distance(query_embedding))
            .limit(self._settings.rag_candidate_k)
        )

        results = await self._session.execute(statement)

        candidates: list[RetrievedContext] = []
        for chunk, distance in results:
            if distance > self._settings.rag_max_distance:
                continue

            asset = chunk.knowledge_version.knowledge_asset
            metadata = chunk.metadata_json or {}
            source_page = metadata.get("source_page") or metadata.get("page")
            section_title = metadata.get("section_title")

            candidates.append(
                RetrievedContext(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    asset_id=asset.id,
                    asset_title=asset.title,
                    original_filename=asset.original_filename,
                    distance=distance,
                    source_page=source_page if isinstance(source_page, int) else None,
                    section_title=section_title if isinstance(section_title, str) else None,
                )
            )

        return self._select_relevant_contexts(query, candidates)

    def _query_terms(self, query: str) -> set[str]:
        normalized = self._normalize_for_matching(query)
        return {
            term
            for term in re.findall(r"[\w]+", normalized, flags=re.UNICODE)
            if len(term) >= 4
        }

    def _lexical_score(self, query: str, context: RetrievedContext) -> float:
        terms = self._query_terms(query)
        if not terms:
            return 0.0

        haystack = " ".join(
            [
                context.content,
                context.asset_title,
                context.original_filename,
                context.section_title or "",
            ]
        )
        haystack = self._normalize_for_matching(haystack)
        matches = sum(1 for term in terms if term in haystack)
        return matches / len(terms)

    def _select_relevant_contexts(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        if not contexts:
            return []

        scored = [
            (self._lexical_score(query, context), context)
            for context in sorted(contexts, key=lambda item: item.distance)
        ]
        lexical_matches = [(score, context) for score, context in scored if score > 0]

        if lexical_matches:
            lexical_matches.sort(key=lambda item: (-item[0], item[1].distance))
            pool = [context for _, context in lexical_matches]
        else:
            best_distance = min(context.distance for context in contexts)
            max_distance = min(
                self._settings.rag_max_distance,
                best_distance + self._settings.rag_distance_margin,
            )
            pool = [context for context in contexts if context.distance <= max_distance]
            pool.sort(key=lambda item: item.distance)

        selected: list[RetrievedContext] = []
        selected_assets: set[uuid.UUID] = set()

        for context in pool:
            if (
                context.asset_id not in selected_assets
                and len(selected_assets) >= self._settings.rag_max_citation_assets
            ):
                continue

            selected.append(context)
            selected_assets.add(context.asset_id)

            if len(selected) >= self._settings.rag_top_k:
                break

        return selected

    def _build_citations(self, contexts: list[RetrievedContext]) -> list[dict]:
        citations: list[dict] = []
        seen_assets: set[uuid.UUID] = set()

        for context in contexts:
            if context.asset_id in seen_assets:
                continue

            seen_assets.add(context.asset_id)
            citations.append(
                {
                    "asset_id": str(context.asset_id),
                    "original_filename": context.original_filename,
                    "chunk_id": str(context.chunk_id),
                    "source_page": context.source_page,
                    "quote": context.content[:200] + "...",
                    "distance": context.distance,
                }
            )

        return citations

    def build_grounded_prompt(self, query: str, contexts: list[RetrievedContext], model_name: str = "") -> str:
        is_small_model = not model_name.startswith("gemini")
        max_chunks = 3 if is_small_model else 5
        max_content_len = 600 if is_small_model else 1200

        context_parts = []
        for index, context in enumerate(contexts[:max_chunks], start=1):
            page_info = f" | Page {context.source_page}" if context.source_page else ""
            section_info = f" | Section: {context.section_title}" if context.section_title else ""
            content = context.content.strip()
            if len(content) > max_content_len:
                content = content[:max_content_len].rstrip() + "..."
            context_parts.append(
                f"[{index}] Document: {context.original_filename}{page_info}{section_info}\n{content}"
            )

        context_block = "\n\n".join(context_parts)

        return f"""You are a retrieval QA assistant.
Answer in Vietnamese.

Rules:
1. Use only the context below.
2. If the answer is explicitly present in the context, answer directly and concisely.
3. If the context includes times, dates, numbers, policies, or steps related to the question, include them exactly.
4. If the answer is not in the context, reply with exactly:
{NO_ANSWER_MESSAGE}
5. Do not say the context is unrelated if it clearly contains the answer.
6. Do not invent missing details.

Context:
{context_block}

Question: {query}

Answer:"""

    def _history_window(self, model_name: str, history: list[dict]) -> list[dict]:
        normalized_model = normalize_model_name(model_name)
        if normalized_model.startswith("gemini"):
            max_messages = 6
            max_chars = 1600
        else:
            max_messages = 2
            max_chars = 450

        selected: list[dict] = []
        total_chars = 0

        for item in reversed(history):
            role = item.get("role")
            content = " ".join(str(item.get("content", "")).split())
            if role not in {"user", "assistant"} or not content:
                continue

            if len(content) > 240:
                content = content[:240].rstrip() + "..."

            if total_chars + len(content) > max_chars:
                break

            selected.append({"role": role, "content": content})
            total_chars += len(content)

            if len(selected) >= max_messages:
                break

        selected.reverse()
        return selected

    async def stream_generation(
        self,
        model_name: str,
        query: str,
        contexts: list[RetrievedContext],
        history: list[dict],
    ) -> AsyncGenerator[tuple[str, dict | None], None]:
        model_name = normalize_model_name(model_name)

        if not contexts:
            yield NO_ANSWER_MESSAGE, None
            yield "", {
                "citations": [],
                "retrieval": {
                    "top_k": self._settings.rag_top_k,
                    "distance_threshold": self._settings.rag_max_distance,
                },
                "provider": {"name": "guardrail", "model": "none"},
                "error": None,
            }
            return

        prompt = self.build_grounded_prompt(query, contexts, model_name=model_name)
        citations = self._build_citations(contexts)
        message_history = self._history_window(model_name, history)
        messages = message_history + [{"role": "user", "content": prompt}]
        emitted_token = False

        try:
            if model_name.startswith("gemini"):
                async for token in self._stream_gemini(model_name, messages):
                    emitted_token = True
                    yield token, None
                provider = {"name": "gemini", "model": model_name}
            elif model_name.startswith(("llama", "gemma")):
                async for token in self._stream_ollama(model_name, messages):
                    emitted_token = True
                    yield token, None
                provider = {"name": "ollama", "model": model_name}
            else:
                raise ValueError(f"Unknown model identifier: {model_name}")

            yield "", {
                "citations": citations,
                "retrieval": {
                    "top_k": self._settings.rag_top_k,
                    "distance_threshold": self._settings.rag_max_distance,
                },
                "provider": provider,
                "error": None,
            }
        except Exception as exc:
            logger.exception("Generation provider failed for model %s", model_name)
            if not emitted_token:
                yield f"Loi khi goi model {model_name}: {exc}", None

            yield "", {
                "citations": citations,
                "retrieval": {
                    "top_k": self._settings.rag_top_k,
                    "distance_threshold": self._settings.rag_max_distance,
                },
                "provider": {"name": "unknown", "model": model_name},
                "error": str(exc),
            }

    async def _stream_gemini(self, model_name: str, messages: list[dict]) -> AsyncGenerator[str, None]:
        client = genai.Client(api_key=self._settings.gemini_api_key)

        gemini_messages = []
        for message in messages:
            role = "user" if message["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [{"text": message["content"]}]})

        response = await client.aio.models.generate_content_stream(
            model=model_name,
            contents=gemini_messages,
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    async def _stream_ollama(
        self, model_name: str, messages: list[dict]
    ) -> AsyncGenerator[str, None]:
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": self._settings.ollama_num_ctx},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{self._settings.ollama_url}/api/chat", json=payload) as response:
                await _raise_ollama_error(response, model_name)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]


async def _raise_ollama_error(response: httpx.Response, model_name: str) -> None:
    if response.status_code < 400:
        return

    body = await response.aread()
    detail = body.decode("utf-8", errors="replace")
    try:
        detail = json.loads(detail).get("error", detail)
    except json.JSONDecodeError:
        pass
    raise RuntimeError(f"Ollama model {model_name} failed: {detail}")
