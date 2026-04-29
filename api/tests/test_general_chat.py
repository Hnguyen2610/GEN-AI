from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.app.services.general_chat import GeneralChatService


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.gemini_api_key = "test-key"
    settings.ollama_num_ctx = 1024
    settings.ollama_url = "http://localhost:11434"
    return settings


async def _collect(stream):
    tokens = []
    metas = []
    async for token, meta in stream:
        if token:
            tokens.append(token)
        if meta:
            metas.append(meta)
    return tokens, metas


@pytest.mark.anyio
async def test_general_chat_returns_builtin_greeting_without_model_call(mock_settings, monkeypatch):
    service = GeneralChatService(settings=mock_settings)

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("Model call should be skipped for greeting fast-path")
        yield  # pragma: no cover

    monkeypatch.setattr(service, "_stream_ollama", fail_if_called)

    tokens, metas = await _collect(service.stream_generation("llama3.2:1b", "hello", []))

    assert tokens == ["Xin chao. Toi co the giup gi cho ban?"]
    assert metas[0]["provider"]["name"] == "builtin"
    assert metas[0]["route"] == "chat"


@pytest.mark.anyio
@patch("api.app.services.general_chat.genai.Client")
async def test_general_chat_normalizes_legacy_gemini_model_name(mock_client_cls, mock_settings):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    captured = {}

    class FakeStream:
        def __aiter__(self):
            async def iterator():
                yield MagicMock(text="Hello there")

            return iterator()

    async def fake_stream(*, model, contents, config=None):
        captured["model"] = model
        return FakeStream()

    mock_client.aio.models.generate_content_stream = fake_stream
    service = GeneralChatService(settings=mock_settings)

    tokens, metas = await _collect(service.stream_generation("gemini-1.5-flash", "How are you?", []))

    assert captured["model"] == "gemini-2.5-flash"
    assert tokens == ["Hello there"]
    assert metas[0]["provider"]["model"] == "gemini-2.5-flash"
