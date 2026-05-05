import json
import logging
import re
import unicodedata
from typing import Literal
from uuid import UUID

import httpx
from google import genai
from pydantic import BaseModel
from sqlalchemy import String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.app.core.config import Settings
from api.app.models.entities import Asset, AssetKind, DatasetStatus
from api.app.services.model_aliases import normalize_model_name

logger = logging.getLogger(__name__)


RouteName = Literal["rag", "sql", "hybrid", "clarification", "agent", "chat"]


class RouteResult(BaseModel):
    route: RouteName
    reason: str
    confidence: float


class RouterService:
    def __init__(self, settings: Settings, session: AsyncSession):
        self._settings = settings
        self._session = session

    async def _get_workspace_inventory(self, workspace_id: UUID) -> tuple[str, list[str]]:
        """Return (inventory_text, list_of_asset_titles)."""
        statement = (
            select(Asset)
            .where(
                Asset.workspace_id == workspace_id,
                cast(Asset.status, String) == DatasetStatus.ready.value,
            )
            .order_by(Asset.created_at.desc())
        )
        assets = (await self._session.scalars(statement)).all()

        counts: dict[str, int] = {}
        asset_titles: list[str] = []
        for asset in assets:
            kind_val = getattr(asset.kind, "value", asset.kind)
            counts[kind_val] = counts.get(kind_val, 0) + 1
            asset_titles.append(f"{asset.title} [{kind_val}]")

        lines = [
            f"- Knowledge Assets (PDFs, TXTs, Documents): {counts.get(AssetKind.knowledge.value, 0)}",
            f"- Dataset Assets (CSV, Excel Tables): {counts.get(AssetKind.dataset.value, 0)}",
        ]
        if asset_titles:
            lines.append("\nAsset names:")
            for title in asset_titles:
                lines.append(f"  • {title}")

        return "\n".join(lines), asset_titles

    async def decide_route(
        self,
        workspace_id: UUID,
        query: str,
        model_name: str = "gemini-2.5-flash",
    ) -> RouteResult:
        inventory_context, asset_titles = await self._get_workspace_inventory(workspace_id)

        fast_path = self._fast_path_route(query, inventory_context)
        if fast_path is not None:
            return fast_path

        model_name = normalize_model_name(model_name)
        is_gemini = model_name.startswith("gemini")
        is_groq = model_name.startswith(("llama", "qwen", "openai", "groq", "deepseek"))
        
        if is_gemini and not self._settings.gemini_api_key:
            return self._heuristic_route(query, inventory_context, "GEMINI_API_KEY not configured.")
        
        if is_groq and not self._settings.groq_api_key:
            return self._heuristic_route(query, inventory_context, "GROQ_API_KEY not configured.")

        prompt = f"""You are an AI router for an internal asset Q&A system.
Task: choose the best processing lane for the user question.

Available assets in this workspace:
{inventory_context}

Route rules (pick the FIRST that fits):
1. "chat"  – greetings, thanks, casual conversation, or general world-knowledge not tied to any uploaded asset.
2. "hybrid" – the user wants to read, summarize, or query BOTH documents and datasets, or asks a vague question about the content of a file that might be an Excel dataset (e.g. "trong file X có gì").
3. "rag"   – reading, summarizing, or understanding the *content* of an uploaded document (PDF, DOCX, TXT, MD). This includes company policies, rules, penalty calculations ("cách tính tiền phạt", "quy định", "chính sách"), meaning the AI just needs to read the text rules.
4. "sql"   – ONLY for actual numerical data analysis over CSV/Excel databases (aggregations, counting, sum, average, doing math over rows). Do NOT use this for reading rule documents.
5. "clarification" – the request is too vague to execute safely.

Return JSON only in this exact shape:
{{
  "route": "rag" | "sql" | "hybrid" | "clarification" | "chat",
  "reason": "Short explanation under 20 words.",
  "confidence": 0.95
}}

User question: "{query}"
"""
        try:
            if is_gemini:
                data = await self._classify_gemini(model_name, prompt)
            elif is_groq:
                data = await self._classify_groq(model_name, prompt)
            else:
                data = await self._classify_ollama(model_name, prompt)

            route = data.get("route", "rag")
            if route not in {"rag", "sql", "hybrid", "clarification", "chat"}:
                raise ValueError(f"Unsupported route: {route}")

            return RouteResult(
                route=route,
                reason=data.get("reason", "Router selected route."),
                confidence=float(data.get("confidence", 0.5)),
            )
        except Exception as exc:
            logger.warning("Router LLM failure (%s), falling back to heuristic: %s", model_name, exc)
            return self._heuristic_route(query, inventory_context, str(exc))

    async def _classify_gemini(self, model_name: str, prompt: str) -> dict:
        client = genai.Client(api_key=self._settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return json.loads(_strip_json_markdown(response.text or "{}"))

    async def _classify_ollama(self, model_name: str, prompt: str) -> dict:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {
                "num_ctx": min(self._settings.ollama_num_ctx, 256),
                "temperature": 0,
            },
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(f"{self._settings.ollama_url}/api/chat", json=payload)
            res.raise_for_status()
            content = res.json()["message"]["content"]
            return json.loads(_strip_json_markdown(content))

    async def _classify_groq(self, model_name: str, prompt: str) -> dict:
        api_model_name = model_name.split("/")[-1] if "/" in model_name else model_name
        payload = {
            "model": api_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.groq_api_key}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            if res.status_code >= 400:
                body = await res.aread()
                raise RuntimeError(f"Groq classification failed: {res.status_code} - {body.decode()}")
            
            content = res.json()["choices"][0]["message"]["content"]
            return json.loads(_strip_json_markdown(content))

    def _fast_path_route(self, query: str, inventory_context: str) -> RouteResult | None:
        normalized_query = _normalize_text(query)
        has_dataset = "Dataset Assets (CSV, Excel Tables): 0" not in inventory_context
        has_knowledge = "Knowledge Assets (PDFs, TXTs, Documents): 0" not in inventory_context

        wants_sql = _contains_any(normalized_query, SQL_KEYWORDS)
        wants_rag = _contains_any(normalized_query, RAG_KEYWORDS)
        wants_content_reading = _contains_any(normalized_query, CONTENT_READING_KEYWORDS)
        wants_general_chat = _contains_any(normalized_query, CHAT_KEYWORDS)
        wants_general_knowledge = _looks_like_general_knowledge(normalized_query)

        # Content-reading intent logic
        if wants_content_reading:
            if has_dataset and not has_knowledge:
                return RouteResult(
                    route="hybrid",
                    reason="Read intent on dataset only, using hybrid/agent.",
                    confidence=0.90,
                )
            if has_dataset and has_knowledge:
                return RouteResult(
                    route="hybrid",
                    reason="Read intent on mixed assets, using hybrid/agent.",
                    confidence=0.90,
                )
            if has_knowledge:
                return RouteResult(
                    route="rag",
                    reason="Read intent on knowledge assets only.",
                    confidence=0.95,
                )

        if wants_sql and wants_rag and has_dataset and has_knowledge:
            return RouteResult(
                route="hybrid",
                reason="Fast-path keyword match for hybrid request.",
                confidence=0.92,
            )

        if wants_sql and not wants_rag and has_dataset:
            return RouteResult(
                route="sql",
                reason="Fast-path keyword match for dataset analysis.",
                confidence=0.92,
            )

        if wants_rag and not wants_sql and has_knowledge:
            return RouteResult(
                route="rag",
                reason="Fast-path keyword match for document question.",
                confidence=0.92,
            )

        if (wants_general_chat or wants_general_knowledge or len(normalized_query) < 5) and not wants_sql and not wants_rag:
            return RouteResult(
                route="chat",
                reason="Fast-path match for greeting or general chat.",
                confidence=0.97,
            )

        return None

    def _heuristic_route(self, query: str, inventory_context: str, reason: str) -> RouteResult:
        normalized_query = _normalize_text(query)
        has_dataset = "Dataset Assets (CSV, Excel Tables): 0" not in inventory_context
        has_knowledge = "Knowledge Assets (PDFs, TXTs, Documents): 0" not in inventory_context

        wants_sql = _contains_any(normalized_query, SQL_KEYWORDS)
        wants_rag = _contains_any(normalized_query, RAG_KEYWORDS)
        wants_general_chat = _contains_any(normalized_query, CHAT_KEYWORDS)

        if wants_sql and wants_rag and has_dataset and has_knowledge:
            route: RouteName = "hybrid"
        elif wants_sql and has_dataset:
            route = "sql"
        elif wants_rag and has_knowledge:
            route = "rag"
        elif wants_general_chat or (len(normalized_query) < 5 and not wants_sql and not wants_rag):
            route = "chat"
        elif _looks_like_general_knowledge(normalized_query):
            route = "chat"
        elif len(normalized_query) < 4:
            route = "clarification"
        else:
            route = "rag" if has_knowledge else "chat"

        return RouteResult(
            route=route,
            reason=f"Fallback heuristic route triggered ({reason})",
            confidence=0.5,
        )


CHAT_KEYWORDS = {
    "hello",
    "hi",
    "hey",
    "xin chao",
    "chao",
    "cam on",
    "thanks",
    "thank you",
    "tam biet",
    "bye",
    "ban la ai",
    "ten gi",
    "lam duoc gi",
}

GENERAL_KNOWLEDGE_KEYWORDS = {
    "capital",
    "thu do",
    "country",
    "quoc gia",
    "world",
    "the gioi",
    "la ai",
    "la gi",
}

RAG_KEYWORDS = {
    "tai lieu",
    "document",
    "file",
    "uploaded",
    "upload",
    "pdf",
    "docx",
    "noi quy",
    "quy dinh",
    "chinh sach",
    "huong dan",
    "quy trinh",
    "trong tai lieu",
    "trong file",
    "my document",
    "cv",
    "resume",
    "ho so",
    "ly lich",
}

# Keywords indicating the user wants to READ/UNDERSTAND content,
# not perform numerical analysis. These override SQL routing.
CONTENT_READING_KEYWORDS = {
    "co gi",
    "noi dung",
    "tom tat",
    "liet ke",
    "giai thich",
    "trong",
    "noi gi",
    "viet gi",
    "cho biet",
    "doc",
    "xem",
    "thong tin",
    "chi tiet",
    "mieu ta",
    "mo ta",
    "summarize",
    "describe",
    "explain",
    "what is in",
    "contents",
    "cach tinh",
    "quy dinh",
    "chinh sach",
    "luat",
    "phat",
    "tien phat",
    "ky luat",
}

SQL_KEYWORDS = {
    "sql",
    "bang",
    "du lieu",
    "table",
    "sum",
    "tong",
    "dem",
    "count",
    "dataset",
    "so lieu",
    "trung binh",
    "average",
    "max",
    "min",
    "bao nhieu",
    "thong ke",
    "doanh thu",
    "so luong",
}


def _strip_json_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        return text.removeprefix("```json").removesuffix("```").strip()
    if text.startswith("```"):
        return text.removeprefix("```").removesuffix("```").strip()
    return text


def _contains_any(value: str, keywords: set[str]) -> bool:
    for keyword in keywords:
        if " " in keyword:
            if keyword in value:
                return True
            continue

        if re.search(rf"(?<![a-z0-9_]){re.escape(keyword)}(?![a-z0-9_])", value):
            return True
    return False


def _looks_like_general_knowledge(value: str) -> bool:
    if _contains_any(value, GENERAL_KNOWLEDGE_KEYWORDS):
        return True
    return value.startswith(("what is ", "who is ", "where is ", "when is ", "why is ", "how is "))


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return normalized.replace("\u0111", "d").strip()
