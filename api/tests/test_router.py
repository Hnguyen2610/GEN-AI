from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from api.app.services.router import RouterService


@pytest.fixture
def mock_session():
    session = AsyncMock()
    result = MagicMock()
    result.all.return_value = [("knowledge", 5), ("dataset", 2)]
    session.execute.return_value = result
    return session


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.gemini_api_key = "test_key"
    settings.ollama_num_ctx = 1024
    settings.ollama_url = "http://localhost:11434"
    return settings


@pytest.fixture
def heuristic_settings():
    settings = MagicMock()
    settings.gemini_api_key = ""
    settings.ollama_num_ctx = 1024
    settings.ollama_url = "http://localhost:11434"
    return settings


@pytest.mark.anyio
@patch("api.app.services.router.genai.Client")
async def test_router_service_uses_llm_for_ambiguous_rag_query(mock_client_cls, mock_settings, mock_session):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    response = MagicMock()
    response.text = '{"route": "rag", "reason": "Ask about onboarding policy", "confidence": 0.95}'
    mock_aio = AsyncMock()
    mock_aio.models.generate_content.return_value = response
    mock_client.aio = mock_aio

    service = RouterService(settings=mock_settings, session=mock_session)
    result = await service.decide_route(uuid4(), "Can you summarize the onboarding policy?", "gemini-2.5-flash")

    assert result.route == "rag"
    assert "onboarding" in result.reason
    assert result.confidence == 0.95


@pytest.mark.anyio
@patch("api.app.services.router.genai.Client")
async def test_router_service_uses_llm_for_ambiguous_chat_query(mock_client_cls, mock_settings, mock_session):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    response = MagicMock()
    response.text = '{"route": "chat", "reason": "Casual conversation", "confidence": 0.99}'
    mock_aio = AsyncMock()
    mock_aio.models.generate_content.return_value = response
    mock_client.aio = mock_aio

    service = RouterService(settings=mock_settings, session=mock_session)
    result = await service.decide_route(uuid4(), "Tell me a short joke about databases.", "gemini-2.5-flash")

    assert result.route == "chat"
    assert "Casual" in result.reason
    assert result.confidence == 0.99


@pytest.mark.anyio
@patch("api.app.services.router.genai.Client")
async def test_router_service_fallback_on_exception(mock_client_cls, mock_settings, mock_session):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_aio = AsyncMock()
    mock_aio.models.generate_content.side_effect = Exception("API Down")
    mock_client.aio = mock_aio

    service = RouterService(settings=mock_settings, session=mock_session)
    result = await service.decide_route(uuid4(), "Can you help me inspect this?", "gemini-2.5-flash")

    assert result.route == "rag"
    assert "Fallback" in result.reason


@pytest.mark.anyio
async def test_router_heuristic_chat_detection(heuristic_settings, mock_session):
    service = RouterService(settings=heuristic_settings, session=mock_session)

    assert (await service.decide_route(uuid4(), "hi", "gemini-2.5-flash")).route == "chat"
    assert (await service.decide_route(uuid4(), "cam on ban", "gemini-2.5-flash")).route == "chat"
    assert (await service.decide_route(uuid4(), "What is the capital of Vietnam?", "gemini-2.5-flash")).route == "chat"


@pytest.mark.anyio
async def test_router_heuristic_keeps_document_questions_in_rag(heuristic_settings, mock_session):
    service = RouterService(settings=heuristic_settings, session=mock_session)

    result = await service.decide_route(uuid4(), "What's in my document?", "gemini-2.5-flash")

    assert result.route == "rag"


@pytest.mark.anyio
@patch.object(RouterService, "_classify_ollama", new_callable=AsyncMock)
async def test_router_fast_path_skips_ollama_for_greeting(mock_classify_ollama, mock_settings, mock_session):
    service = RouterService(settings=mock_settings, session=mock_session)

    result = await service.decide_route(uuid4(), "hello", "llama3.2:1b")

    assert result.route == "chat"
    mock_classify_ollama.assert_not_called()
