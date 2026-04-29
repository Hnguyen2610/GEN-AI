import json
import logging
import unicodedata
from typing import AsyncGenerator

import httpx
from google import genai

from api.app.core.config import Settings
from api.app.services.model_aliases import normalize_model_name

logger = logging.getLogger(__name__)


class GeneralChatService:
    def __init__(self, settings: Settings):
        self._settings = settings

    async def stream_generation(
        self,
        model_name: str,
        query: str,
        history: list[dict],
    ) -> AsyncGenerator[tuple[str, dict | None], None]:
        model_name = normalize_model_name(model_name)
        quick_response = _build_quick_response(query)
        if quick_response is not None:
            yield quick_response, None
            yield "", {
                "route": "chat",
                "provider": {"name": "builtin", "model": "greeting-fast-path"},
                "retrieval_used": False,
                "citations": [],
                "error": None,
            }
            return

        messages = [{"role": "system", "content": GENERAL_CHAT_PROMPT}] + history + [
            {"role": "user", "content": query}
        ]
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
                raise ValueError(f"Unknown model identifier for general chat: {model_name}")

            yield "", {
                "route": "chat",
                "provider": provider,
                "retrieval_used": False,
                "citations": [],
                "error": None,
            }
        except Exception as exc:
            logger.exception("General chat generation failed for model %s", model_name)
            if not emitted_token:
                yield f"Loi khi thuc hien chat voi model {model_name}: {exc}", None

            yield "", {
                "route": "chat",
                "provider": {"name": "unknown", "model": model_name},
                "retrieval_used": False,
                "citations": [],
                "error": str(exc),
            }

    async def _stream_gemini(self, model_name: str, messages: list[dict]) -> AsyncGenerator[str, None]:
        if not self._settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        client = genai.Client(api_key=self._settings.gemini_api_key)
        gemini_messages = [
            {
                "role": "user" if message["role"] in {"user", "system"} else "model",
                "parts": [{"text": message["content"]}],
            }
            for message in messages
        ]

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


GENERAL_CHAT_PROMPT = (
    "You are a friendly general-purpose AI assistant. "
    "Answer naturally using general knowledge. "
    "Do not pretend you have read uploaded documents or datasets unless context is actually provided. "
    "If the user asks about uploaded documents or dataset analysis, briefly tell them to ask specifically about that document or dataset."
)


def _build_quick_response(query: str) -> str | None:
    normalized = _normalize_text(query)

    if normalized in {"hello", "hi", "hey", "xin chao", "chao"}:
        return "Xin chao. Toi co the giup gi cho ban?"

    if normalized in {"cam on", "thanks", "thank you"}:
        return "Khong co gi. Ban can toi ho tro them dieu gi?"

    if normalized in {"tam biet", "bye", "goodbye"}:
        return "Tam biet. Khi can ban cu nhan tin tiep."

    return None


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.casefold())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return normalized.replace("\u0111", "d").strip()


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
